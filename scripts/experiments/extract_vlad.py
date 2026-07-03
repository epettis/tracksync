#!/usr/bin/env python3
"""Phase 2 — VLAD aggregation over DINOv2 patch tokens (AnyLoc recipe).

Replaces the pipeline's GeM pooling with unsupervised VLAD aggregation using a
per-pair vocabulary (k-means fit on BOTH videos' masked patch tokens). AnyLoc's
finding is that on unstructured/self-similar scenes VLAD is far more
discriminative than GeM (which is ~CLS-level). Uses the SAME DINOv2 backbone as
the GeM baseline so the A/B isolates the aggregation, not the backbone.

Saves per-video npz in the harness format (emb_array [N, clusters*D] L2-norm,
frame_times, sample_hz, video_path) as <label>__vlad<k>-<modeltag>.npz.

Approximation: patch tokens are randomly subsampled to --tokens-per-frame per
frame (both for vocabulary fitting and per-frame residual aggregation) to bound
memory. No sklearn in the venv -> small NumPy k-means++ (fixed seed).

Usage:
    .venv/bin/python scripts/experiments/extract_vlad.py --pair buttonwillow \
        samples/buttonwillow/eddie_pb.mp4 samples/buttonwillow/kurt_optimal.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tracksync.embedding import DinoV2Embedder
from tracksync.feature_extraction import sample_frames
from tracksync.masking import compute_static_mask

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data"

PAIRS = {
    "buttonwillow": [("buttonwillow_eddie_pb", "samples/buttonwillow/eddie_pb.mp4"),
                     ("buttonwillow_kurt_optimal", "samples/buttonwillow/kurt_optimal.mp4")],
    "laguna": [("laguna_eddie", "samples/laguna/eddie.mp4"),
               ("laguna_kurt", "samples/laguna/kurt.mp4")],
}


def masked_tokens_per_frame(emb: DinoV2Embedder, frames, mask, tokens_per_frame, rng):
    """Return a list of [<=tokens_per_frame, D] arrays of masked patch tokens."""
    import torch
    emb._ensure_model_loaded()
    out = []
    bs = emb.batch_size
    for start in range(0, len(frames), bs):
        batch = [emb._resize_to_multiple(f) for f in frames[start:start + bs]]
        tens = torch.stack([torch.from_numpy(f).float().permute(2, 0, 1) / 255.0 for f in batch]).to(emb._device)
        with torch.no_grad():
            tok = emb._model.forward_features(tens)["x_norm_patchtokens"].cpu().numpy()
        for i, f in enumerate(batch):
            h, w = f.shape[:2]
            hp, wp = h // emb._patch_size, w // emb._patch_size
            keep = ~emb._downsample_mask_to_patches(mask, hp, wp).flatten()
            t = tok[i][keep]
            if len(t) == 0:
                t = tok[i]
            if len(t) > tokens_per_frame:
                t = t[rng.choice(len(t), tokens_per_frame, replace=False)]
            out.append(t.astype(np.float32))
    return out


def _assign(x, C):
    """Nearest-center index for each row of x, via ||x-C||^2 = |x|^2 -2xC^T +|C|^2."""
    d2 = (x * x).sum(1)[:, None] - 2 * (x @ C.T) + (C * C).sum(1)[None, :]
    return np.argmin(d2, axis=1)


def kmeans(x, k, iters=25, seed=0):
    """Minimal k-means++ (NumPy, memory-safe distances). x: [M, D] -> [k, D]."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum()
        centers.append(x[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    C = np.array(centers)
    for _ in range(iters):
        assign = _assign(x, C)
        newC = np.array([x[assign == j].mean(0) if (assign == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def vlad_encode(tokens_list, centers):
    """VLAD-encode each frame's tokens against `centers` [k, D] -> [N, k*D]."""
    k, D = centers.shape
    out = np.zeros((len(tokens_list), k * D), dtype=np.float32)
    for i, t in enumerate(tokens_list):
        assign = _assign(t, centers)
        v = np.zeros((k, D), dtype=np.float32)
        for j in range(k):
            m = assign == j
            if m.any():
                res = t[m] - centers[j]
                v[j] = res.sum(0)
                nrm = np.linalg.norm(v[j])          # intra-normalization
                if nrm > 0:
                    v[j] /= nrm
        vf = v.flatten()
        nrm = np.linalg.norm(vf)                     # global L2
        out[i] = vf / nrm if nrm > 0 else vf
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIRS))
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--tokens-per-frame", type=int, default=128)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: cap frames")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    modeltag = args.model.replace("_", "-").replace("dinov2-", "")
    emb = DinoV2Embedder(args.model)

    vids = PAIRS[args.pair]
    per_video = []
    pooled = []
    for label, rel in vids:
        frames, times = sample_frames(str(REPO / rel), sample_hz=args.sample_hz, max_dim=518)
        if args.max_frames:
            frames, times = frames[:args.max_frames], times[:args.max_frames]
        mask = compute_static_mask(frames)
        toks = masked_tokens_per_frame(emb, frames, mask, args.tokens_per_frame, rng)
        per_video.append((label, rel, times, toks))
        pooled.append(np.concatenate(toks, 0))
        print(f"[{label}] {len(frames)} frames, tokens/frame~{args.tokens_per_frame}")

    # Per-pair vocabulary: k-means over a subsample of both videos' tokens.
    allt = np.concatenate(pooled, 0)
    sub = allt[rng.choice(len(allt), min(len(allt), 40000), replace=False)]
    print(f"Fitting k={args.clusters} vocabulary on {len(sub)} tokens (dim {sub.shape[1]}) ...")
    centers = kmeans(sub, args.clusters, seed=0)

    for label, rel, times, toks in per_video:
        desc = vlad_encode(toks, centers)
        out = DATA / f"{label}__vlad{args.clusters}-{modeltag}.npz"
        np.savez(out, emb_array=desc, frame_times=np.asarray(times),
                 sample_hz=np.float64(args.sample_hz), video_path=str(REPO / rel))
        norms = np.linalg.norm(desc, axis=1)
        print(f"[{label}] VLAD {desc.shape} norm[min,max]=({norms.min():.3f},{norms.max():.3f}) -> {out.name}")


if __name__ == "__main__":
    main()
