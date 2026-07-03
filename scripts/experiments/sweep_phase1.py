#!/usr/bin/env python3
"""Phase 1 — cheap cost-matrix conditioning sweep on cached embeddings.

Runs a grid of feature/matrix/DTW configurations against the Buttonwillow
(low-contrast, primary target) and Laguna (higher-contrast control) pairs,
scoring each against cached Catalyst+OCR references, the self-alignment
identity guard, boundary segment speeds, and cost-matrix contrast metrics.

Writes a JSON results blob to experiments/results/phase1.json for evaluation.

Usage:
    .venv/bin/python scripts/experiments/sweep_phase1.py [--tag dinov2-vitb14]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments import exputil as X

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results"

PAIRS = [
    ("buttonwillow", "buttonwillow_eddie_pb", "buttonwillow_kurt_optimal"),
    ("laguna", "laguna_eddie", "laguna_kurt"),
]


def build_configs() -> list[X.Config]:
    P = X.DtwParams
    cfgs: list[X.Config] = []

    # --- baselines ---
    cfgs.append(X.Config("baseline_main", X.feat_identity, X.mat_baseline, P()))
    cfgs.append(X.Config("plain_cost", X.feat_identity, X.mat_plain, P()))

    # --- feature-side conditioning (with the current baseline matrix norm) ---
    cfgs.append(X.Config("center", X.feat_center, X.mat_baseline, P()))
    cfgs.append(X.Config("whiten", X.feat_whiten, X.mat_baseline, P()))
    for k in (5, 10):
        cfgs.append(X.Config(f"window{k}", X.make_feat_window(k), X.mat_baseline, P()))

    # --- matrix-side conditioning (identity features) ---
    for tau in (0.05, 0.1, 0.2):
        cfgs.append(X.Config(f"rowsoftmax_t{tau}", X.feat_identity, X.make_row_softmax(tau), P()))
        cfgs.append(X.Config(f"dualsoftmax_t{tau}", X.feat_identity, X.make_dual_softmax(tau), P()))
    for R in (5, 10, 20):
        cfgs.append(X.Config(f"seqslam_R{R}_axisB", X.feat_identity, X.make_seqslam_local(R, axis=0), P()))
    cfgs.append(X.Config("seqslam_R10_axisA", X.feat_identity, X.make_seqslam_local(10, axis=1), P()))
    cfgs.append(X.Config("diagsmooth", X.feat_identity, X.make_diag_smooth(), P()))

    # --- DTW-side conditioning (baseline matrix) ---
    for lam in (0.05, 0.1, 0.2):
        cfgs.append(X.Config(f"steppen{lam}", X.feat_identity, X.mat_baseline, P(step_penalty=lam)))
    for sp in (0.5, 1.0):
        cfgs.append(X.Config(f"slackpen{sp}", X.feat_identity, X.mat_baseline, P(slack_penalty=sp)))

    # --- promising compositions ---
    cfgs.append(X.Config("center+dualsoftmax_t0.1", X.feat_center, X.make_dual_softmax(0.1), P()))
    cfgs.append(X.Config("center+steppen0.1", X.feat_center, X.mat_baseline, P(step_penalty=0.1)))
    cfgs.append(X.Config("center+dualsoftmax+steppen0.1", X.feat_center, X.make_dual_softmax(0.1), P(step_penalty=0.1)))
    cfgs.append(X.Config("center+window5+dualsoftmax", X.compose_feat(X.feat_center, X.make_feat_window(5)), X.make_dual_softmax(0.1), P()))
    cfgs.append(X.Config("center+seqslamR10+steppen0.1", X.feat_center, X.make_seqslam_local(10, 0), P(step_penalty=0.1)))
    return cfgs


def load_refs(pair: str) -> dict | None:
    f = X.DATA / f"{pair}__refs.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {"catalyst_pairs": d["catalyst_pairs"], "ocr_pairs": d["ocr_pairs"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dinov2-vitb14")
    ap.add_argument("--out", default=str(RESULTS / "phase1.json"))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    results = []
    for pair_label, la, lb in PAIRS:
        try:
            a = X.load_video(la, args.tag)
            b = X.load_video(lb, args.tag)
        except FileNotFoundError as e:
            print(f"SKIP pair {pair_label}: {e}")
            continue
        refs = load_refs(pair_label)
        print(f"\n=== pair {pair_label} ({len(a.times)}x{len(b.times)} frames, refs={'yes' if refs else 'no'}) ===")
        for cfg in configs:
            try:
                m = X.evaluate(a, b, cfg, refs)
            except Exception as e:
                print(f"  {cfg.name}: ERROR {type(e).__name__}: {e}")
                continue
            m["pair_label"] = pair_label
            results.append(m)
            vc = m.get("vs_catalyst", {})
            vo = m.get("vs_ocr", {})
            print(f"  {cfg.name:34s} id={m['identity_dev']:.3f} "
                  f"bsp=({m['boundary_speed_start']:.2f},{m['boundary_speed_end']:.2f}) "
                  f"catP95={vc.get('p95', float('nan')):.3f} ocrP95={vo.get('p95', float('nan')):.3f} "
                  f"ent={m['contrast_row_entropy_norm']:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
