#!/usr/bin/env python3
"""Validation harness comparing scene alignment against the Catalyst baseline.

Runs both the legacy Catalyst (map-dot cross-correlation) sync and the new
scene-based sync on the same pair of real Catalyst clips, then reports:

- Wall-clock runtime of each pipeline
- Per-sync-point |dt| statistics (max / mean / p95) between the two methods'
  t_A -> t_B mappings, evaluated on a common time grid
- |dt| statistics of each method against the OCR lap-clock ground truth
  (segment-transition correspondence pairs), when OCR data is available

Also supports a self-alignment check (--self VIDEO): scene-aligns a video
against itself and reports the maximum deviation from the identity mapping.

Each run appends a markdown row to docs/scene_alignment_validation.md.

Usage:
    .venv/bin/python scripts/validate_scene_alignment.py videoA.mp4 videoB.mp4
    .venv/bin/python scripts/validate_scene_alignment.py --self video.mp4

Task reference: docs/scene_alignment_tasks.md T13
Design reference: docs/scene_alignment_design.md SS8
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from tracksync.cross_correlation import (
    compute_cross_correlations_from_features,
    generate_sync_points,
)
from tracksync.embedding import make_embedder
from tracksync.feature_extraction import extract_video_features
from tracksync.fine_align import make_matcher
from tracksync.frame_analysis import FrameOCRData
from tracksync.models import SyncResult
from tracksync.scene_align import (
    coarse_align,
    extract_scene_features,
    generate_scene_sync_points,
)
from tracksync.scene_deps import MissingSceneDependenciesError, require_scene_deps

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "scene_alignment_validation.md"

LOG_TEMPLATE = """\
# Scene Alignment Validation Results

Operator log for T13 validation runs comparing scene-based alignment against
the legacy Catalyst pipeline (design doc SS8; acceptance gate for T14 is
p95 |dt| <= 1 native frame at sync points).

How to run (from the repo root, on the reference machine):

```bash
# Pair validation: Catalyst vs scene on a real Catalyst clip pair
.venv/bin/python scripts/validate_scene_alignment.py videoA.mp4 videoB.mp4

# Self-alignment check: scene-align a video against itself
.venv/bin/python scripts/validate_scene_alignment.py --self video.mp4
```

Columns: |dt| statistics are in seconds, formatted max / mean / p95.
"n/a" means the comparison was unavailable (e.g., no OCR data, self runs).

| Date | Kind | Inputs | Catalyst runtime (s) | Scene runtime (s) | Scene vs Catalyst \\|dt\\| | Catalyst vs OCR \\|dt\\| | Scene vs OCR \\|dt\\| | Notes |
|------|------|--------|----------------------|-------------------|------------------------|------------------------|--------------------|-------|
"""


@dataclass
class DeltaStats:
    """|dt| statistics between two time mappings, in seconds."""

    max_dt: float
    mean_dt: float
    p95_dt: float
    n: int

    @classmethod
    def from_deltas(cls, deltas: np.ndarray) -> "DeltaStats":
        deltas = np.abs(np.asarray(deltas, dtype=np.float64))
        return cls(
            max_dt=float(deltas.max()),
            mean_dt=float(deltas.mean()),
            p95_dt=float(np.percentile(deltas, 95)),
            n=int(deltas.size),
        )


def format_stats(stats: DeltaStats | None) -> str:
    """Format DeltaStats as 'max / mean / p95' or 'n/a'."""
    if stats is None or stats.n == 0:
        return "n/a"
    return f"{stats.max_dt:.3f} / {stats.mean_dt:.3f} / {stats.p95_dt:.3f}"


def mapping_from_sync_result(sync_result: SyncResult):
    """Build a piecewise-linear t_A -> t_B mapping from sync points.

    Returns:
        Tuple of (f, t_min, t_max) where f maps time_a to time_b and
        [t_min, t_max] is the supported domain in video A.

    Raises:
        ValueError: If the result has fewer than 2 sync points.
    """
    points = sorted(sync_result.sync_points, key=lambda p: p.time_a)
    if len(points) < 2:
        raise ValueError(
            f"Need at least 2 sync points to build a mapping, got {len(points)}"
        )
    times_a = np.array([p.time_a for p in points])
    times_b = np.array([p.time_b for p in points])

    def f(t_a):
        return np.interp(t_a, times_a, times_b)

    return f, float(times_a[0]), float(times_a[-1])


def compare_mappings(
    result_x: SyncResult,
    result_y: SyncResult,
    grid_hz: float = 10.0,
) -> DeltaStats | None:
    """Compare two sync results' t_A -> t_B mappings on a common grid.

    Evaluates both piecewise-linear mappings on a dense time grid over the
    overlapping video-A domain and returns |dt| statistics.

    Returns:
        DeltaStats, or None if the domains do not overlap or either result
        has fewer than 2 sync points.
    """
    try:
        f_x, x_min, x_max = mapping_from_sync_result(result_x)
        f_y, y_min, y_max = mapping_from_sync_result(result_y)
    except ValueError:
        return None

    t_min = max(x_min, y_min)
    t_max = min(x_max, y_max)
    if t_max <= t_min:
        return None

    n_samples = max(2, int(round((t_max - t_min) * grid_hz)) + 1)
    grid = np.linspace(t_min, t_max, n_samples)
    return DeltaStats.from_deltas(f_x(grid) - f_y(grid))


def ocr_transition_pairs(
    ocr_a: list[FrameOCRData],
    times_a: list[float],
    ocr_b: list[FrameOCRData],
    times_b: list[float],
) -> list[tuple[float, float]]:
    """Extract paired segment-transition times from two videos' OCR data.

    Segment transitions in the Catalyst overlay occur at fixed track
    positions, so a transition into segment N in video A corresponds to the
    transition into segment N in video B. Returns (t_a, t_b) correspondence
    pairs for segment numbers whose transition is observed exactly once in
    each video (repeated transitions would be lap-ambiguous).
    """

    def transitions(ocr_list, frame_times):
        seen: dict[int, list[float]] = {}
        prev = None
        for ocr, t in zip(ocr_list, frame_times):
            seg = ocr.segment_number if ocr is not None else None
            if seg is not None and seg != prev and prev is not None:
                seen.setdefault(seg, []).append(t)
            if seg is not None:
                prev = seg
        return seen

    trans_a = transitions(ocr_a, times_a)
    trans_b = transitions(ocr_b, times_b)

    pairs = []
    for seg in sorted(trans_a.keys() & trans_b.keys()):
        if len(trans_a[seg]) == 1 and len(trans_b[seg]) == 1:
            pairs.append((trans_a[seg][0], trans_b[seg][0]))
    return pairs


def ocr_delta_stats(
    sync_result: SyncResult,
    pairs: list[tuple[float, float]],
) -> DeltaStats | None:
    """Evaluate a sync result's mapping against OCR correspondence pairs.

    Computes |f(t_a) - t_b| for each OCR-derived (t_a, t_b) pair that falls
    inside the mapping's domain.

    Returns:
        DeltaStats, or None if there are no usable pairs.
    """
    if not pairs:
        return None
    try:
        f, t_min, t_max = mapping_from_sync_result(sync_result)
    except ValueError:
        return None

    deltas = [
        f(t_a) - t_b for t_a, t_b in pairs if t_min <= t_a <= t_max
    ]
    if not deltas:
        return None
    return DeltaStats.from_deltas(np.array(deltas))


def run_catalyst_sync(
    video_a: str,
    video_b: str,
    max_sync_interval: float = 3.0,
):
    """Run the legacy Catalyst sync pipeline, timed.

    Returns:
        Tuple of (SyncResult, features_a, features_b, runtime_seconds)
    """
    start = time.perf_counter()
    features_a = extract_video_features(video_a, progress_prefix="  [A] ")
    features_b = extract_video_features(video_b, progress_prefix="  [B] ")
    results = compute_cross_correlations_from_features(features_a, features_b)
    sync_result = generate_sync_points(
        results, features_a.crossings, features_b.crossings,
        max_interval=max_sync_interval,
    )
    runtime = time.perf_counter() - start
    return sync_result, features_a, features_b, runtime


def run_scene_sync(
    video_a: str,
    video_b: str,
    embedder_name: str = "dinov2-vitb14",
    matcher_name: str = "aliked-lightglue",
    sample_hz: float = 10.0,
    band_pct: float = 0.10,
    min_inliers: int = 30,
    fov_deg: float = 90.0,
    max_sync_interval: float = 3.0,
):
    """Run the scene-based sync pipeline, timed.

    Returns:
        Tuple of (SyncResult, runtime_seconds)
    """
    start = time.perf_counter()
    embedder = make_embedder(embedder_name)
    matcher = make_matcher(matcher_name)
    feat_a = extract_scene_features(video_a, embedder, sample_hz=sample_hz)
    feat_b = extract_scene_features(video_b, embedder, sample_hz=sample_hz)
    coarse = coarse_align(feat_a, feat_b, band_pct=band_pct)
    sync_result = generate_scene_sync_points(
        coarse, video_a, video_b, matcher,
        max_sync_interval=max_sync_interval,
        min_inliers=min_inliers,
        fov_deg=fov_deg,
    )
    runtime = time.perf_counter() - start
    return sync_result, runtime


def self_alignment_check(
    video: str,
    embedder_name: str = "dinov2-vitb14",
    sample_hz: float = 10.0,
    band_pct: float = 0.10,
):
    """Scene-align a video against itself and measure identity deviation.

    Returns:
        Tuple of (max |f(t) - t| in seconds, runtime_seconds)
    """
    start = time.perf_counter()
    embedder = make_embedder(embedder_name)
    feat = extract_scene_features(video, embedder, sample_hz=sample_hz)
    coarse = coarse_align(feat, feat, band_pct=band_pct)

    margin = 1.0 / sample_hz
    t_lo = coarse.trim_start_a + margin
    t_hi = coarse.trim_end_a - margin
    if t_hi <= t_lo:
        t_lo, t_hi = coarse.trim_start_a, coarse.trim_end_a
    grid = np.linspace(t_lo, t_hi, 100)
    mapped = np.array([coarse.f(float(t)) for t in grid])
    max_dev = float(np.max(np.abs(mapped - grid)))
    runtime = time.perf_counter() - start
    return max_dev, runtime


def append_markdown_row(log_path: str | Path, row: str) -> None:
    """Append a table row to the validation log, creating it from the template.

    Args:
        log_path: Path to the markdown log file
        row: Complete markdown table row (with leading/trailing pipes)
    """
    log_path = Path(log_path)
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(LOG_TEMPLATE)
    with open(log_path, "a") as fh:
        fh.write(row.rstrip("\n") + "\n")


def check_scene_deps(embedder_name: str, matcher_name: str | None) -> None:
    """Exit with code 2 if torch-based backends are requested but missing."""
    needs_torch = (
        embedder_name.startswith("dinov2")
        or "lightglue" in (matcher_name or "")
    )
    if not needs_torch:
        return
    try:
        require_scene_deps()
    except MissingSceneDependenciesError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


def run_pair_validation(args: argparse.Namespace) -> None:
    """Run Catalyst and scene sync on a clip pair and report/log results."""
    video_a, video_b = args.videos

    print("=" * 60)
    print("PAIR VALIDATION: Catalyst vs scene alignment")
    print(f"  Video A: {video_a}")
    print(f"  Video B: {video_b}")
    print("=" * 60)

    print("\nRunning legacy Catalyst sync...")
    catalyst_result, features_a, features_b, catalyst_s = run_catalyst_sync(
        video_a, video_b, max_sync_interval=args.max_sync_interval
    )
    print(f"  {len(catalyst_result.sync_points)} sync points "
          f"in {catalyst_s:.1f} s")

    print("\nRunning scene sync...")
    scene_result, scene_s = run_scene_sync(
        video_a, video_b,
        embedder_name=args.embedder,
        matcher_name=args.matcher,
        sample_hz=args.sample_hz,
        band_pct=args.band_pct,
        min_inliers=args.min_inliers,
        fov_deg=args.fov_deg,
        max_sync_interval=args.max_sync_interval,
    )
    print(f"  {len(scene_result.sync_points)} sync points in {scene_s:.1f} s")

    method_stats = compare_mappings(scene_result, catalyst_result)

    pairs = ocr_transition_pairs(
        features_a.interpolated_ocr, features_a.frame_times,
        features_b.interpolated_ocr, features_b.frame_times,
    )
    catalyst_ocr_stats = ocr_delta_stats(catalyst_result, pairs)
    scene_ocr_stats = ocr_delta_stats(scene_result, pairs)

    print("\nRESULTS (|dt| seconds, max / mean / p95)")
    print("-" * 60)
    print(f"  Catalyst runtime:   {catalyst_s:.1f} s")
    print(f"  Scene runtime:      {scene_s:.1f} s")
    print(f"  Scene vs Catalyst:  {format_stats(method_stats)}")
    print(f"  OCR pairs:          {len(pairs)}")
    print(f"  Catalyst vs OCR:    {format_stats(catalyst_ocr_stats)}")
    print(f"  Scene vs OCR:       {format_stats(scene_ocr_stats)}")

    row = (
        f"| {date.today().isoformat()} | pair "
        f"| {Path(video_a).name} vs {Path(video_b).name} "
        f"| {catalyst_s:.1f} | {scene_s:.1f} "
        f"| {format_stats(method_stats)} "
        f"| {format_stats(catalyst_ocr_stats)} "
        f"| {format_stats(scene_ocr_stats)} "
        f"| embedder={args.embedder}, matcher={args.matcher}, "
        f"ocr_pairs={len(pairs)} |"
    )
    append_markdown_row(args.log, row)
    print(f"\nAppended results row to {args.log}")


def run_self_validation(args: argparse.Namespace) -> None:
    """Run the self-alignment identity check and report/log results."""
    video = args.self

    print("=" * 60)
    print("SELF-ALIGNMENT CHECK")
    print(f"  Video: {video}")
    print("=" * 60)

    max_dev, runtime = self_alignment_check(
        video,
        embedder_name=args.embedder,
        sample_hz=args.sample_hz,
        band_pct=args.band_pct,
    )

    print(f"\n  Max |f(t) - t|: {max_dev:.3f} s "
          f"(1 sample = {1.0 / args.sample_hz:.3f} s)")
    print(f"  Runtime: {runtime:.1f} s")

    row = (
        f"| {date.today().isoformat()} | self "
        f"| {Path(video).name} "
        f"| n/a | {runtime:.1f} | n/a | n/a | n/a "
        f"| max identity deviation = {max_dev:.3f} s, embedder={args.embedder} |"
    )
    append_markdown_row(args.log, row)
    print(f"\nAppended results row to {args.log}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate scene alignment against the Catalyst baseline"
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Two Catalyst video files to compare (videoA videoB)",
    )
    parser.add_argument(
        "--self", metavar="VIDEO",
        help="Run a self-alignment check on a single video instead",
    )
    parser.add_argument(
        "--embedder", default="dinov2-vitb14",
        choices=["gist", "dinov2-vits14", "dinov2-vitb14"],
        help="Scene embedder backbone (default: dinov2-vitb14)",
    )
    parser.add_argument(
        "--matcher", default="aliked-lightglue",
        help="Fine-alignment feature matcher (default: aliked-lightglue)",
    )
    parser.add_argument(
        "--sample-hz", type=float, default=10.0,
        help="Scene-mode frame sampling rate (default: 10)",
    )
    parser.add_argument(
        "--band-pct", type=float, default=0.10,
        help="DTW band width as fraction of clip length (default: 0.10)",
    )
    parser.add_argument(
        "--min-inliers", type=int, default=30,
        help="Minimum RANSAC inliers to accept a fine sync point (default: 30)",
    )
    parser.add_argument(
        "--fov-deg", type=float, default=90.0,
        help="Assumed horizontal field of view in degrees (default: 90)",
    )
    parser.add_argument(
        "--max-sync-interval", type=float, default=3.0,
        help="Maximum seconds between sync points (default: 3.0)",
    )
    parser.add_argument(
        "--log", default=str(DEFAULT_LOG_PATH),
        help="Markdown results log to append to "
             "(default: docs/scene_alignment_validation.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.self:
        if args.videos:
            parser.error("--self takes no positional videos")
        check_scene_deps(args.embedder, None)
        run_self_validation(args)
    else:
        if len(args.videos) != 2:
            parser.error("expected exactly two videos (videoA videoB), "
                         "or use --self VIDEO")
        check_scene_deps(args.embedder, args.matcher)
        run_pair_validation(args)


if __name__ == "__main__":
    main()
