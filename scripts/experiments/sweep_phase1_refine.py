#!/usr/bin/env python3
"""Phase 1 refinement arms (Fable's item-6 list) + harness fixes.

- Scores every config on a COMMON set of Catalyst points (the intersection of
  all configs' mapped domains) so the ranking is apples-to-apples.
- Audits the 5 OCR pairs' lap positions vs Catalyst coverage.
- Adds the refinement configs: whiten+dualsoftmax, center+dualsoftmax at other
  tau, center+dualsoftmax+seqslam, delta descriptors, slope-1 diagonal average,
  small bounded slack on the winner.

Usage: .venv/bin/python scripts/experiments/sweep_phase1_refine.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments import exputil as X

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results"


def build_configs() -> list[X.Config]:
    P = X.DtwParams
    C = X.Config
    ds = X.make_dual_softmax
    cfgs = [
        # anchors from the first sweep
        C("baseline_main", X.feat_identity, X.mat_baseline, P()),
        C("WINNER center+dualsoftmax_t0.1", X.feat_center, ds(0.1), P()),
        # refinement 1: whiten + dualsoftmax
        C("whiten+dualsoftmax_t0.1", X.feat_whiten, ds(0.1), P()),
        C("whiten+dualsoftmax_t0.05", X.feat_whiten, ds(0.05), P()),
        # refinement 2: center+dualsoftmax at other tau
        C("center+dualsoftmax_t0.05", X.feat_center, ds(0.05), P()),
        C("center+dualsoftmax_t0.2", X.feat_center, ds(0.2), P()),
        # refinement 3: compose #1 and #2 mechanisms (center+ds + seqslam)
        C("center+ds+seqslamR10", X.feat_center, X.make_local_z(ds(0.1), 10, 0), P()),
        # refinement 4: delta descriptors; slope-1 diagonal average (concat-window)
        C("delta5", X.make_feat_delta(5), X.mat_baseline, P()),
        C("center+delta5+dualsoftmax", X.compose_feat(X.feat_center, X.make_feat_delta(5)), ds(0.1), P()),
        C("diagavg5", X.feat_identity, X.make_diag_avg(5), P()),
        C("center+diagavg5", X.feat_center, X.make_diag_avg(5), P()),
        # refinement 5: small bounded slack on the winner
        C("winner+slack0.25", X.feat_center, ds(0.1), P(slack_penalty=0.25)),
        C("winner+openend5", X.feat_center, ds(0.1), P(open_end_s=5.0)),
    ]
    return cfgs


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    a = X.load_video("buttonwillow_eddie_pb")
    b = X.load_video("buttonwillow_kurt_optimal")
    d = np.load(X.DATA / "buttonwillow__refs.npz", allow_pickle=True)
    cat = d["catalyst_pairs"]; ocr = d["ocr_pairs"]

    print(f"Catalyst pairs: n={len(cat)} t_a range [{cat[:,0].min():.1f}, {cat[:,0].max():.1f}]")
    print(f"OCR pairs t_a: {np.round(ocr[:,0],1).tolist()} (audit: within Catalyst coverage?)")
    cov_lo, cov_hi = cat[:, 0].min(), cat[:, 0].max()
    for ta, tb in ocr:
        inside = cov_lo <= ta <= cov_hi
        print(f"   OCR t_a={ta:.1f} -> t_b={tb:.1f}  {'inside' if inside else 'OUTSIDE'} Catalyst span")

    configs = build_configs()

    # First pass: compute each config's mapped domain to find the common window.
    aligned = []
    for cfg in configs:
        f, path, Cm, dom = X.align(a, b, cfg)
        aligned.append((cfg, f, Cm, dom))
    common_lo = max(dom[0] for _, _, _, dom in aligned)
    common_hi = min(dom[1] for _, _, _, dom in aligned)
    common_cat = cat[(cat[:, 0] >= common_lo) & (cat[:, 0] <= common_hi)]
    print(f"\nCommon domain [{common_lo:.1f}, {common_hi:.1f}], "
          f"scoring on {len(common_cat)}/{len(cat)} Catalyst points\n")

    rows = []
    print(f"{'config':34s} {'catP95':>7s} {'catMax':>7s} {'catMean':>7s} {'id':>6s} {'bsp_s':>6s} {'bsp_e':>6s} {'ent':>5s}")
    for cfg, f, Cm, dom in aligned:
        st = X.vs_pairs(f, (common_lo, common_hi), common_cat)
        bs = X.boundary_speeds(f, dom)
        idd = X.identity_dev(a, cfg)
        cm = X.contrast_metrics(Cm)
        row = {"config": cfg.name, "vs_catalyst_common": st, "identity_dev": idd,
               "boundary_speed_start": bs[0], "boundary_speed_end": bs[1],
               "dom": dom, "contrast": cm}
        rows.append(row)
        print(f"{cfg.name:34s} {st['p95']:7.3f} {st['max']:7.3f} {st['mean']:7.3f} "
              f"{idd:6.3f} {bs[0]:6.2f} {bs[1]:6.2f} {cm['row_entropy_norm']:5.2f}")

    (RESULTS / "phase1_refine.json").write_text(json.dumps(
        {"common_domain": [common_lo, common_hi], "n_common_catalyst": len(common_cat),
         "ocr_pairs": ocr.tolist(), "rows": rows}, indent=2))
    print(f"\nWrote -> {RESULTS/'phase1_refine.json'}")


if __name__ == "__main__":
    main()
