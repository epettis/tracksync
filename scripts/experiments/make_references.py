#!/usr/bin/env python3
"""Generate and cache authoritative reference mappings for a clip pair.

Runs the legacy Catalyst pipeline once (the slow OCR path) to produce, for a
given pair, both:
  - the Catalyst t_A->t_B sync mapping, and
  - OCR segment-transition (t_a, t_b) ground-truth pairs.

These are cached to experiments/data/<pair>__refs.npz so the Phase-1
conditioning sweep can score scene alignments against a fixed reference
without re-running the ~25 min Catalyst pipeline each time.

Usage:
    .venv/bin/python scripts/experiments/make_references.py \
        --pair buttonwillow \
        samples/buttonwillow/eddie_pb.mp4 samples/buttonwillow/kurt_optimal.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validate_scene_alignment import (
    mapping_from_sync_result,
    ocr_transition_pairs,
    run_catalyst_sync,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="pair label, e.g. buttonwillow")
    ap.add_argument("videos", nargs=2)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    video_a, video_b = args.videos

    print(f"Running Catalyst pipeline on {video_a} vs {video_b} ...")
    cat_result, feat_a, feat_b, runtime = run_catalyst_sync(video_a, video_b)
    print(f"Catalyst: {len(cat_result.sync_points)} sync points in {runtime:.1f}s")

    # Catalyst mapping sampled as (t_a, t_b) sync points.
    cat_pairs = np.array(
        [(p.time_a, p.time_b) for p in sorted(cat_result.sync_points, key=lambda p: p.time_a)],
        dtype=np.float64,
    )

    ocr_pairs = ocr_transition_pairs(
        feat_a.interpolated_ocr, feat_a.frame_times,
        feat_b.interpolated_ocr, feat_b.frame_times,
    )
    ocr_pairs = np.array(ocr_pairs, dtype=np.float64) if ocr_pairs else np.zeros((0, 2))
    print(f"OCR transition pairs: {len(ocr_pairs)}")

    out = DATA / f"{args.pair}__refs.npz"
    np.savez(
        out,
        catalyst_pairs=cat_pairs,
        ocr_pairs=ocr_pairs,
        video_a=video_a,
        video_b=video_b,
        catalyst_runtime=np.float64(runtime),
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
