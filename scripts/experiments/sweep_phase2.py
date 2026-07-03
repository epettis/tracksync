#!/usr/bin/env python3
"""Phase 2 — VLAD vs GeM aggregation A/B, scored on Buttonwillow references.

Compares the current GeM-pooled DINOv2 descriptors against per-pair VLAD
descriptors (same vitb14 backbone), each under the production row-z-score and
the Phase-1 winning conditioning (whiten + dual-softmax). All scored on the
common Catalyst point set for apples-to-apples comparison.

Usage: .venv/bin/python scripts/experiments/sweep_phase2.py
       [--vlad-tag vlad32-vitb14]
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


def configs():
    P = X.DtwParams
    ds = X.make_dual_softmax
    return [
        ("baseline_rowz", X.Config("baseline_rowz", X.feat_identity, X.mat_baseline, P())),
        ("whiten+dualsoftmax", X.Config("w+ds", X.feat_whiten, ds(0.1), P())),
        ("center+dualsoftmax", X.Config("c+ds", X.feat_center, ds(0.1), P())),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlad-tag", default="vlad32-vitb14")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    variants = {
        "GeM": ("dinov2-vitb14", "dinov2-vitb14"),
        "VLAD": (args.vlad_tag, args.vlad_tag),
    }
    d = np.load(X.DATA / "buttonwillow__refs.npz", allow_pickle=True)
    cat = d["catalyst_pairs"]

    # gather all (variant, cfg) alignments, find common domain
    loaded = {}
    for vname, (tag_a, tag_b) in variants.items():
        try:
            a = X.load_video("buttonwillow_eddie_pb", tag_a)
            b = X.load_video("buttonwillow_kurt_optimal", tag_b)
        except FileNotFoundError as e:
            print(f"SKIP {vname}: {e}")
            continue
        loaded[vname] = (a, b)

    aligned = []
    for vname, (a, b) in loaded.items():
        for cname, cfg in configs():
            f, path, C, dom = X.align(a, b, cfg)
            aligned.append((vname, cname, cfg, a, b, f, C, dom))
    common_lo = max(x[7][0] for x in aligned)
    common_hi = min(x[7][1] for x in aligned)
    common = cat[(cat[:, 0] >= common_lo) & (cat[:, 0] <= common_hi)]
    print(f"Common domain [{common_lo:.1f},{common_hi:.1f}], {len(common)}/{len(cat)} Catalyst points\n")

    rows = []
    print(f"{'variant':7s} {'config':20s} {'dim':>6s} {'catP95':>7s} {'catMax':>7s} {'catMean':>7s} {'id':>6s} {'bsp_s':>6s} {'bsp_e':>6s} {'ent':>5s}")
    for vname, cname, cfg, a, b, f, C, dom in aligned:
        st = X.vs_pairs(f, (common_lo, common_hi), common)
        bs = X.boundary_speeds(f, dom)
        idd = X.identity_dev(a, cfg)
        cm = X.contrast_metrics(C)
        rows.append({"variant": vname, "config": cname, "dim": int(a.emb.shape[1]),
                     "vs_catalyst_common": st, "identity_dev": idd,
                     "boundary_speed_start": bs[0], "boundary_speed_end": bs[1],
                     "contrast": cm})
        print(f"{vname:7s} {cname:20s} {a.emb.shape[1]:6d} {st['p95']:7.3f} {st['max']:7.3f} "
              f"{st['mean']:7.3f} {idd:6.3f} {bs[0]:6.2f} {bs[1]:6.2f} {cm['row_entropy_norm']:5.2f}")

    (RESULTS / "phase2.json").write_text(json.dumps(
        {"common_domain": [common_lo, common_hi], "n_common_catalyst": len(common), "rows": rows}, indent=2))
    print(f"\nWrote -> {RESULTS/'phase2.json'}")


if __name__ == "__main__":
    main()
