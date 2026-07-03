#!/usr/bin/env python3
"""Phase 3 — fuse the appearance cost with the yaw-rate (and speed) motion
cost matrices and sweep fusion weights, scored on the Buttonwillow references.

Appearance base = the Phase-1 winner (whiten + dual-softmax). Motion costs are
absolute differences of per-lap z-normalized traces. All costs are unit-mean
normalized before the weighted sum so the weights are comparable.

Usage: .venv/bin/python scripts/experiments/sweep_phase3.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments import exputil as X
from scripts.experiments import motion as Mot

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results"

APP = X.Config("whiten+dualsoftmax_t0.1", X.feat_whiten, X.make_dual_softmax(0.1), X.DtwParams())


def fused_eval(a, b, ma, mb, w_yaw, w_speed, common, common_lo, common_hi):
    Capp = Mot.normalize_cost(X.cost_matrix(a, b, APP))
    C = Capp.copy()
    if w_yaw:
        C = C + w_yaw * Mot.normalize_cost(Mot.trace_cost(ma.yaw, mb.yaw))
    if w_speed:
        C = C + w_speed * Mot.normalize_cost(Mot.trace_cost(ma.speed, mb.speed))
    f, path, dom = X.align_with_cost(a, b, C, APP.dtw)
    st = X.vs_pairs(f, (common_lo, common_hi), common)
    bs = X.boundary_speeds(f, dom)
    # identity via self-fusion (yaw_a vs yaw_a etc.)
    Cid = Mot.normalize_cost(X.cost_matrix(a, a, APP))
    if w_yaw:
        Cid = Cid + w_yaw * Mot.normalize_cost(Mot.trace_cost(ma.yaw, ma.yaw))
    if w_speed:
        Cid = Cid + w_speed * Mot.normalize_cost(Mot.trace_cost(ma.speed, ma.speed))
    fid, _, (lo, hi) = X.align_with_cost(a, a, Cid, APP.dtw)
    m = 1.0 / a.hz
    grid = np.linspace(lo + m, hi - m, 100)
    idd = float(np.max(np.abs(fid(grid) - grid)))
    return st, bs, idd, dom


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    a = X.load_video("buttonwillow_eddie_pb")
    b = X.load_video("buttonwillow_kurt_optimal")
    ma = Mot.load_traces("buttonwillow_eddie_pb")
    mb = Mot.load_traces("buttonwillow_kurt_optimal")
    assert len(ma.yaw) == len(a.times) and len(mb.yaw) == len(b.times), "grid mismatch"

    d = np.load(X.DATA / "buttonwillow__refs.npz", allow_pickle=True)
    cat = d["catalyst_pairs"]
    # common domain from appearance-only alignment coverage
    _, _, dom0 = X.align_with_cost(a, b, X.cost_matrix(a, b, APP), APP.dtw)
    common_lo, common_hi = dom0
    common = cat[(cat[:, 0] >= common_lo) & (cat[:, 0] <= common_hi)]
    print(f"Appearance base = {APP.name}; scoring on {len(common)}/{len(cat)} Catalyst points "
          f"[{common_lo:.1f},{common_hi:.1f}]\n")

    grid = [(0, 0), (0.25, 0), (0.5, 0), (1.0, 0), (2.0, 0), (4.0, 0),
            (0.5, 0.5), (1.0, 0.5)]
    # yaw-only (appearance off) as a reference: temporarily zero appearance
    rows = []
    print(f"{'w_yaw':>6s} {'w_speed':>7s} {'catP95':>7s} {'catMax':>7s} {'catMean':>7s} {'id':>6s} {'bsp_s':>6s} {'bsp_e':>6s}")
    for wy, ws in grid:
        st, bs, idd, dom = fused_eval(a, b, ma, mb, wy, ws, common, common_lo, common_hi)
        rows.append({"w_yaw": wy, "w_speed": ws, "vs_catalyst_common": st,
                     "identity_dev": idd, "boundary_speed_start": bs[0],
                     "boundary_speed_end": bs[1], "dom": dom})
        print(f"{wy:6.2f} {ws:7.2f} {st['p95']:7.3f} {st['max']:7.3f} {st['mean']:7.3f} "
              f"{idd:6.3f} {bs[0]:6.2f} {bs[1]:6.2f}")

    # yaw-only alignment (no appearance): build cost purely from yaw
    Cyaw = Mot.normalize_cost(Mot.trace_cost(ma.yaw, mb.yaw))
    f, _, dom = X.align_with_cost(a, b, Cyaw, APP.dtw)
    st = X.vs_pairs(f, (common_lo, common_hi), common)
    print(f"\nyaw-ONLY (no appearance): catP95={st['p95']:.3f} max={st['max']:.3f} "
          f"mean={st['mean']:.3f} dom=[{dom[0]:.1f},{dom[1]:.1f}]")
    rows.append({"w_yaw": "only", "w_speed": 0, "vs_catalyst_common": st, "dom": dom})

    (RESULTS / "phase3.json").write_text(json.dumps(
        {"appearance": APP.name, "n_common_catalyst": len(common), "rows": rows}, indent=2))
    print(f"\nWrote -> {RESULTS/'phase3.json'}")


if __name__ == "__main__":
    main()
