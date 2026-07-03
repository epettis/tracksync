#!/usr/bin/env python3
"""Extract and cache DINOv2 scene embeddings for the experiment pairs.

Warms the on-disk embedding cache and dumps a compact .npz per video
(emb_array, frame_times, sample_hz) into experiments/data/ so the Phase-1
conditioning sweep can run fully offline (pure NumPy on cached embeddings,
no GPU / no re-embedding).

Usage:
    .venv/bin/python scripts/experiments/extract_embeddings.py \
        --embedder dinov2-vitb14
"""

import argparse
import time
from pathlib import Path

import numpy as np

from tracksync.embedding import make_embedder
from tracksync.scene_align import extract_scene_features

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data"

# (label, relative video path). Labels are stable keys used by later phases.
VIDEOS = [
    ("buttonwillow_eddie_pb", "samples/buttonwillow/eddie_pb.mp4"),
    ("buttonwillow_kurt_optimal", "samples/buttonwillow/kurt_optimal.mp4"),
    ("laguna_eddie", "samples/laguna/eddie.mp4"),
    ("laguna_kurt", "samples/laguna/kurt.mp4"),
]


def flatness_report(emb: np.ndarray) -> str:
    """Quick self-similarity flatness diagnostics for a single video."""
    sim = emb @ emb.T
    n = sim.shape[0]
    off = sim[~np.eye(n, dtype=bool)]
    # Effective rank via singular values of the (centered) embedding matrix.
    s = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    p = s / s.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    top1_share = float((s[0] ** 2) / (s ** 2).sum())
    return (
        f"n={n} off-diag cos mean={off.mean():.4f} p95={np.percentile(off, 95):.4f} "
        f"max={off.max():.4f} | eff_rank={eff_rank:.1f}/{emb.shape[1]} "
        f"top1_var_share={top1_share:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="dinov2-vitb14")
    ap.add_argument("--sample-hz", type=float, default=10.0)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    embedder = make_embedder(args.embedder)
    tag = embedder.name

    for label, rel in VIDEOS:
        path = REPO / rel
        if not path.exists():
            print(f"SKIP {label}: {path} missing")
            continue
        t0 = time.perf_counter()
        feat = extract_scene_features(str(path), embedder, sample_hz=args.sample_hz)
        dt = time.perf_counter() - t0
        out = DATA / f"{label}__{tag}.npz"
        np.savez(
            out,
            emb_array=feat.emb_array,
            frame_times=feat.frame_times,
            sample_hz=np.float64(args.sample_hz),
            video_path=str(path),
        )
        print(f"[{label}] {feat.emb_array.shape} in {dt:.1f}s -> {out.name}")
        print(f"    {flatness_report(feat.emb_array)}")


if __name__ == "__main__":
    main()
