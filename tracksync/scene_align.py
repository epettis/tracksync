"""Scene-based video alignment pipeline.

This module orchestrates the complete scene-based video alignment process,
composing the coarse and fine alignment stages into a unified pipeline.

The pipeline:
1. Extracts frame embeddings at fixed sampling rate (embedding.py)
2. Applies static car-body masking (masking.py)
3. Computes cosine cost matrix and runs banded DTW (dtw.py)
4. Refines sync points via local feature matching (fine_align.py)
5. Generates SyncResult output compatible with existing CSV/video generation

This module provides the top-level entry points for scene mode, callable
from the CLI (tracksync sync --mode scene).

Design reference: docs/scene_alignment_design.md §3, §6
"""

# Pure composition layer; torch dependencies are isolated in embedding.py
# and fine_align.py submodules.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .dtw import DtwResult, SmoothPath, dtw_align, smooth_path
from .embedding import FrameEmbedder, build_cache_key, embed_video_cached
from .feature_extraction import sample_frames
from .fine_align import (
    FeatureMatcher,
    clamp_monotonic,
    intrinsics_from_fov,
    refine_sync_point,
)
from .frame_data import SceneFeatures
from .masking import compute_static_mask
from .models import SyncPoint, SyncResult

# Decoder callable: (video_path, center_s, half_window_s) -> (frames, times).
# Frames are RGB uint8 at native fps; times are float seconds. Injected into
# generate_scene_sync_points so tests can bypass real video decoding.
FrameWindowDecoder = Callable[[str, float, float], tuple[list[np.ndarray], np.ndarray]]


def extract_scene_features(
    video_path: str,
    embedder: FrameEmbedder,
    sample_hz: float = 10.0,
    cache_dir: Optional[str | Path] = None,
) -> SceneFeatures:
    """Extract scene features from a video for coarse alignment.

    Composes frame sampling, static masking, and cached embedding extraction
    into a single pipeline step. Features are cached on disk keyed by video
    path, mtime, embedder name, sample_hz, and mask hash.

    Args:
        video_path: Path to video file
        embedder: FrameEmbedder instance (e.g., GistEmbedder, DinoV2Embedder)
        sample_hz: Sampling rate in Hz (default 10.0)
        cache_dir: Directory for embedding cache (default: ~/.cache/tracksync/embeddings)

    Returns:
        SceneFeatures dataclass with embeddings, timestamps, mask, and metadata

    Design reference: docs/scene_alignment_design.md §4.1-4.3, task T7
    """
    # Step 1: Sample frames at fixed rate (T6)
    frames, frame_times = sample_frames(video_path, sample_hz=sample_hz, max_dim=518)

    # Step 2: Compute static car-body mask (T4)
    static_mask = compute_static_mask(frames)

    # Step 3: Extract embeddings with caching (T2)
    cache_key = build_cache_key(video_path, embedder.name, sample_hz, static_mask)
    emb_array = embed_video_cached(embedder, frames, static_mask, cache_key, cache_dir)

    return SceneFeatures(
        video_path=video_path,
        frame_times=frame_times,
        emb_array=emb_array,
        static_mask=static_mask,
        sample_hz=sample_hz,
    )


@dataclass
class CoarseAlignment:
    """Result of coarse scene alignment via DTW.

    Exposes the smoothed time-mapping function, DTW path, confidence margins,
    and trim boundaries derived from the path endpoints.

    Design reference: docs/scene_alignment_design.md §4.4, task T7
    """

    f: SmoothPath  # Smoothed t_A -> t_B mapping (callable)
    path: np.ndarray  # DTW path [K, 2] (index pairs)
    margins: np.ndarray  # Per-frame confidence [N_A]
    trim_start_a: float  # Start time in video A (from path start)
    trim_end_a: float  # End time in video A (from path end)
    trim_start_b: float  # Start time in video B (from path start)
    trim_end_b: float  # End time in video B (from path end)
    _frame_times_a: np.ndarray = None  # Private: for confidence interpolation

    def confidence(self, t_a: float) -> float:
        """Get confidence (margin) at time t_a in video A.

        Linearly interpolates the per-frame margins array to provide a
        continuous confidence accessor.

        Args:
            t_a: Time in video A (seconds)

        Returns:
            Confidence value (higher = more confident alignment)
        """
        if self._frame_times_a is None:
            raise ValueError("CoarseAlignment not properly initialized with frame times")
        return float(np.interp(t_a, self._frame_times_a, self.margins))


def compute_scene_cost_matrix(
    feat_a: SceneFeatures,
    feat_b: SceneFeatures,
    softmax_temp: float = 0.1,
) -> np.ndarray:
    """Compute a contrast-conditioned cost matrix between two videos' embeddings.

    Raw cosine costs between forward-facing track frames are nearly constant
    (embeddings of different track positions are ~99% similar), which leaves
    the open-end DTW initialization underdetermined: with almost no contrast,
    the path can enter at a wrong boundary correspondence and then ride the
    slope constraint for many seconds to reach the true diagonal.

    Two composed, complementary steps restore contrast (validated in
    docs/scene_alignment_dtw_contrast_experiments.md, which cut scene-vs-Catalyst
    p95 from 0.388 s to 0.230 s and removed the boundary-speed pathology):

    1. **Per-pair centering.** Subtract the shared mean descriptor of both
       videos and re-normalize. This removes the dominant "track appearance"
       component that inflates every cosine toward 1.0 (the similarity matrix is
       otherwise near rank-1), the root cause of the flat rows.
    2. **Two-way (dual-softmax) contrast.** cost = -log(softmax_row .
       softmax_col) of the similarity at temperature ``softmax_temp``. A cell is
       cheap only when its frames are mutual best matches, so a B-frame that is
       "everyone's 99% match" no longer attracts paths (the column axis, which a
       one-way per-row z-score left unnormalized, is now balanced).

    Both steps preserve the invariant that a self-alignment's diagonal is
    exactly zero cost (identity stays optimal), and the row-min-zero shift keeps
    genuinely ambiguous rows low-contrast so continuity can govern them.

    Earlier revisions used a per-row z-score of (1 - cosine); it was superseded
    because it left the column axis unnormalized and did not remove the shared
    background component (on the study harness it was ~inert). ``softmax_temp``
    is tuned for DINOv2-class cosine ranges; larger values soften contrast.

    Args:
        feat_a: Scene features for video A
        feat_b: Scene features for video B
        softmax_temp: Dual-softmax temperature (default 0.1)

    Returns:
        Cost matrix [N_A, N_B], non-negative, per row shifted to a zero minimum

    Design reference: docs/scene_alignment_design.md §4.4;
    docs/scene_alignment_dtw_contrast_experiments.md
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        emb_a_64 = feat_a.emb_array.astype(np.float64)
        emb_b_64 = feat_b.emb_array.astype(np.float64)

        # Step 1: per-pair centering, then re-normalize to unit length.
        mu = np.concatenate([emb_a_64, emb_b_64], axis=0).mean(axis=0, keepdims=True)
        emb_a_64 = emb_a_64 - mu
        emb_b_64 = emb_b_64 - mu
        emb_a_64 /= np.maximum(np.linalg.norm(emb_a_64, axis=1, keepdims=True), 1e-12)
        emb_b_64 /= np.maximum(np.linalg.norm(emb_b_64, axis=1, keepdims=True), 1e-12)

        similarity = np.clip(emb_a_64 @ emb_b_64.T, -1.0, 1.0)  # [N_A, N_B]

        # Step 2: two-way (dual-softmax) contrast at the given temperature.
        z = similarity / softmax_temp
        pr = np.exp(z - z.max(axis=1, keepdims=True))
        pr /= pr.sum(axis=1, keepdims=True)
        pc = np.exp(z - z.max(axis=0, keepdims=True))
        pc /= pc.sum(axis=0, keepdims=True)
        cost = -np.log(pr + 1e-30) - np.log(pc + 1e-30)

    # Guard against any non-finite values before the row-min shift.
    if not np.all(np.isfinite(cost)):
        finite_max = np.max(cost[np.isfinite(cost)]) if np.any(np.isfinite(cost)) else 1.0
        cost = np.nan_to_num(cost, nan=finite_max, posinf=finite_max, neginf=0.0)

    # Shift each row so its minimum cost is zero (keeps identity at zero cost).
    cost = cost - cost.min(axis=1, keepdims=True)

    return cost


def coarse_align(
    feat_a: SceneFeatures,
    feat_b: SceneFeatures,
    band_pct: float = 0.10,
    open_end_s: float = 5.0,
) -> CoarseAlignment:
    """Perform coarse alignment between two videos via DTW on embeddings.

    Computes a cosine cost matrix from the embeddings, runs banded open-end DTW,
    smooths the path into a continuous time-mapping function, and extracts trim
    boundaries from the path endpoints.

    ``open_end_s`` is effectively the maximum footage trimmed from each clip
    end: skipping low-contrast boundary frames is nearly free, so the open-end
    DTW consumes close to the full slack. It was lowered from 10 s to 5 s
    because 10 s over-trimmed ~4 s of real lap past the true start/finish
    crossings on the reference pair (measured against Catalyst), while 5 s lands
    on the crossings and recovers the boundary content; below ~3 s it begins to
    include genuine non-lap pre/post-roll. See
    docs/scene_alignment_dtw_contrast_experiments.md.

    Args:
        feat_a: Scene features for video A
        feat_b: Scene features for video B
        band_pct: Sakoe-Chiba band as fraction of sequence length (default 0.10)
        open_end_s: Open-end slack in seconds at start/end (default 5.0)

    Returns:
        CoarseAlignment with smoothed mapping, path, margins, and trim times

    Design reference: docs/scene_alignment_design.md §4.4, task T7
    """
    # Step 1: Compute cosine cost matrix (1 - similarity)
    cost = compute_scene_cost_matrix(feat_a, feat_b)

    # Step 2: Convert open_end_s to frames. Clamp the slack to a quarter of
    # the shorter sequence so it can never swallow the whole alignment
    # problem (large slack on short clips would let DTW skip most frames).
    n_min = min(len(feat_a.frame_times), len(feat_b.frame_times))
    max_slack = max(1, n_min // 4)
    open_end_frames_a = min(round(open_end_s * feat_a.sample_hz), max_slack)
    open_end_frames_b = min(round(open_end_s * feat_b.sample_hz), max_slack)
    open_end_frames = (open_end_frames_a, open_end_frames_b)

    # Step 3: Run DTW (T3)
    dtw_result: DtwResult = dtw_align(
        cost=cost,
        band_pct=band_pct,
        open_end_frames=open_end_frames,
        slope_max=2.0,
    )

    # Step 4: Smooth the path (T3)
    smoothed = smooth_path(
        path=dtw_result.path,
        times_a=feat_a.frame_times,
        times_b=feat_b.frame_times,
    )

    # Step 5: Extract trim times from path endpoints
    path_start_idx_a, path_start_idx_b = dtw_result.path[0]
    path_end_idx_a, path_end_idx_b = dtw_result.path[-1]

    trim_start_a = float(feat_a.frame_times[path_start_idx_a])
    trim_end_a = float(feat_a.frame_times[path_end_idx_a])
    trim_start_b = float(feat_b.frame_times[path_start_idx_b])
    trim_end_b = float(feat_b.frame_times[path_end_idx_b])

    return CoarseAlignment(
        f=smoothed,
        path=dtw_result.path,
        margins=dtw_result.margins,
        trim_start_a=trim_start_a,
        trim_end_a=trim_end_a,
        trim_start_b=trim_start_b,
        trim_end_b=trim_end_b,
        _frame_times_a=feat_a.frame_times,
    )


def decode_native_window(
    video_path: str,
    center_s: float,
    half_window_s: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Decode a window of frames around a center time at native fps.

    Args:
        video_path: Path to video file
        center_s: Center time of the window in seconds
        half_window_s: Half-width of the window in seconds (0.0 = single frame)

    Returns:
        Tuple of (frames, timestamps):
        - frames: List of RGB uint8 frames at native fps
        - timestamps: Float array of timestamps (frame index / fps convention)
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        start_idx = max(0, int(round((center_s - half_window_s) * fps)))
        end_idx = int(round((center_s + half_window_s) * fps))
        if frame_count > 0:
            end_idx = min(end_idx, frame_count - 1)

        video.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        for idx in range(start_idx, end_idx + 1):
            ret, frame = video.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(idx / fps)

        return frames, np.array(timestamps, dtype=np.float64)

    finally:
        video.release()


def _candidate_sync_times(
    coarse: CoarseAlignment,
    max_sync_interval: float,
) -> list[float]:
    """Generate candidate sync times in A, snapped to local margin maxima.

    Candidates are spaced so no gap exceeds max_sync_interval, even after
    snapping: interior candidates may shift by at most half the slack between
    the nominal spacing and max_sync_interval (design §5.1, §4.5).

    Args:
        coarse: Coarse alignment providing trim range and per-frame margins
        max_sync_interval: Maximum allowed gap between candidates in seconds

    Returns:
        Strictly increasing candidate times, starting at trim_start_a and
        ending at trim_end_a
    """
    t_start = coarse.trim_start_a
    t_end = coarse.trim_end_a
    duration = t_end - t_start
    if duration <= 0:
        return [t_start]

    n_intervals = max(1, int(np.ceil(duration / max_sync_interval)))
    spacing = duration / n_intervals

    # Snap radius: bounded so that worst-case gap (spacing + 2r) stays within
    # max_sync_interval, and never more than a quarter of the spacing.
    snap_radius = min(0.5 * (max_sync_interval - spacing), 0.25 * spacing)
    snap_radius = max(0.0, snap_radius)

    frame_times = coarse._frame_times_a
    margins = coarse.margins

    times = [t_start]
    for k in range(1, n_intervals):
        nominal = t_start + k * spacing
        snapped = nominal
        if snap_radius > 0 and frame_times is not None:
            in_window = (frame_times >= nominal - snap_radius) & \
                        (frame_times <= nominal + snap_radius)
            idxs = np.where(in_window)[0]
            if len(idxs) > 0:
                best = idxs[np.argmax(margins[idxs])]
                snapped = float(frame_times[best])
        # Keep candidates strictly increasing and strictly interior
        if not (times[-1] < snapped < t_end):
            snapped = nominal
        if times[-1] < snapped < t_end:
            times.append(snapped)
    times.append(t_end)

    return times


def generate_scene_sync_points(
    coarse: CoarseAlignment,
    video_a: str,
    video_b: str,
    matcher: FeatureMatcher,
    max_sync_interval: float = 3.0,
    min_inliers: int = 30,
    fov_deg: float = 90.0,
    decode_window: Optional[FrameWindowDecoder] = None,
) -> SyncResult:
    """Assemble scene-mode sync points into a SyncResult.

    For each candidate time in A (cadence <= max_sync_interval, snapped to
    high-confidence frames), decodes a +/-0.5 s native-fps window from video B
    around the coarse estimate f(t_A) and refines the correspondence via
    zero-crossing analysis (T8). Unverified candidates fall back to the
    smoothed coarse estimate. Timestamps are clamped monotonic and labeled
    start, sync_1..sync_N, end with per-segment speed ratios computed exactly
    as the legacy Catalyst assembly does.

    Args:
        coarse: Coarse alignment from coarse_align()
        video_a: Path to video A
        video_b: Path to video B
        matcher: FeatureMatcher for fine refinement
        max_sync_interval: Maximum gap between sync points in A (seconds)
        min_inliers: Minimum inlier count for fine verification
        fov_deg: Assumed horizontal FOV for intrinsics
        decode_window: Injected frame-window decoder (defaults to
            decode_native_window); tests can bypass video decoding

    Returns:
        SyncResult with labeled sync points, speed ratios, and trim fields
        populated from the coarse alignment

    Design reference: docs/scene_alignment_design.md §5.1-5.4, §11.2, task T10
    """
    if decode_window is None:
        decode_window = decode_native_window

    candidate_times = _candidate_sync_times(coarse, max_sync_interval)

    pairs: list[tuple[float, float]] = []
    for t_a in candidate_times:
        t_b_coarse = coarse.f(t_a)
        t_b = float(t_b_coarse)

        frames_a, _ = decode_window(video_a, t_a, 0.0)
        frames_b, times_b = decode_window(video_b, t_b_coarse, 0.5)

        if frames_a and len(frames_b) > 0:
            frame_a = frames_a[0]
            h_a, w_a = frame_a.shape[:2]
            h_b, w_b = frames_b[0].shape[:2]
            K_a = intrinsics_from_fov(w_a, h_a, fov_deg)
            K_b = intrinsics_from_fov(w_b, h_b, fov_deg)

            result = refine_sync_point(
                frame_a, frames_b, times_b, matcher, K_a, K_b,
                min_inliers=min_inliers,
            )
            if result.verified and result.t_b is not None:
                t_b = float(result.t_b)

        pairs.append((float(t_a), t_b))

    pairs = clamp_monotonic(pairs)

    # Label sync points: start, sync_1..sync_N, end (design §11.2)
    sync_points: list[SyncPoint] = []
    for i, (t_a, t_b) in enumerate(pairs):
        if i == 0:
            label = "start"
        elif i == len(pairs) - 1:
            label = "end"
        else:
            label = f"sync_{i}"
        sync_points.append(SyncPoint(time_a=t_a, time_b=t_b, label=label))

    # Per-segment speed ratios, exactly as the legacy assembly
    # (cross_correlation.py generate_sync_points)
    for i in range(len(sync_points) - 1):
        current = sync_points[i]
        next_point = sync_points[i + 1]

        delta_a = next_point.time_a - current.time_a
        delta_b = next_point.time_b - current.time_b

        if delta_a > 0:
            speed = delta_b / delta_a
        else:
            speed = 1.0

        sync_points[i] = SyncPoint(
            time_a=current.time_a,
            time_b=current.time_b,
            label=current.label,
            speed=speed,
        )

    return SyncResult(
        sync_points=sync_points,
        trim_start_a=coarse.trim_start_a,
        trim_end_a=coarse.trim_end_a,
        trim_start_b=coarse.trim_start_b,
        trim_end_b=coarse.trim_end_b,
        crossings_a=[],
        crossings_b=[],
    )
