"""Offline experiment harness for the DTW cost-matrix contrast study.

Everything here operates on *cached* embeddings (experiments/data/*.npz), so a
full conditioning sweep runs in pure NumPy with no GPU and no re-embedding.

Three composable pieces per experiment configuration:
  - a feature transform   emb_a, emb_b        -> emb_a', emb_b'   (§3.1, §3.3)
  - a matrix transform     similarity S        -> cost C >= 0      (§3.2)
  - DTW parameters         (band, slope, slack, penalties)        (§3.4)

Design references are to docs/scene_alignment_dtw_contrast_research.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@dataclass
class Video:
    label: str
    emb: np.ndarray        # [N, D] L2-normalized
    times: np.ndarray      # [N] seconds
    hz: float


def load_video(label: str, tag: str = "dinov2-vitb14") -> Video:
    d = np.load(DATA / f"{label}__{tag}.npz", allow_pickle=True)
    return Video(label, d["emb_array"].astype(np.float64), d["frame_times"], float(d["sample_hz"]))


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


# --------------------------------------------------------------------------
# Feature transforms (operate on both videos' descriptors jointly)
# --------------------------------------------------------------------------

FeatTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def feat_identity(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return a, b


def feat_center(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the shared mean descriptor of the pair, then renormalize (§3.1)."""
    mu = np.concatenate([a, b], axis=0).mean(0, keepdims=True)
    return l2norm(a - mu), l2norm(b - mu)


def feat_whiten(a: np.ndarray, b: np.ndarray, eps: float = 1e-3, keep: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """PCA-whiten descriptors using pair statistics, then renormalize (§3.1)."""
    both = np.concatenate([a, b], axis=0)
    mu = both.mean(0, keepdims=True)
    both_c = both - mu
    # SVD of centered stack: columns of Vt are principal directions.
    _, s, vt = np.linalg.svd(both_c, full_matrices=False)
    if keep is not None:
        s, vt = s[:keep], vt[:keep]
    scale = 1.0 / np.sqrt(s ** 2 / both_c.shape[0] + eps)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings
        W = (vt.T * scale) @ vt  # symmetric whitening back into original space
        return l2norm((a - mu) @ W), l2norm((b - mu) @ W)


def _window_mean(emb: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return emb
    n = len(emb)
    out = np.empty_like(emb)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        out[i] = emb[lo:hi].mean(0)
    return out


def make_feat_window(k: int) -> FeatTransform:
    """shapeDTW-lite: average each descriptor over a +/-k frame window (§3.3)."""
    def f(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return l2norm(_window_mean(a, k)), l2norm(_window_mean(b, k))
    return f


def make_feat_delta(k: int = 5) -> FeatTransform:
    """Delta descriptors (Garg 2020, §2.3): each descriptor minus its local
    temporal mean, renormalized. In self-similar scenes the *change* carries
    the discriminative signal even when absolute descriptors saturate."""
    def f(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return l2norm(a - _window_mean(a, k)), l2norm(b - _window_mean(b, k))
    return f


def compose_feat(*transforms: FeatTransform) -> FeatTransform:
    def f(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for t in transforms:
            a, b = t(a, b)
        return a, b
    return f


# --------------------------------------------------------------------------
# Matrix transforms (similarity S -> non-negative cost C)
# --------------------------------------------------------------------------

MatTransform = Callable[[np.ndarray], np.ndarray]


def _row_min_zero(c: np.ndarray) -> np.ndarray:
    return c - c.min(axis=1, keepdims=True)


def mat_plain(S: np.ndarray) -> np.ndarray:
    """cost = 1 - cos, no normalization (sanity baseline)."""
    return _row_min_zero(1.0 - S)


def mat_baseline(S: np.ndarray) -> np.ndarray:
    """Current main: per-row z-score of (1-cos), shifted to row-min 0 (§1)."""
    c = 1.0 - S
    mu = c.mean(axis=1, keepdims=True)
    sd = c.std(axis=1, keepdims=True) + 1e-12
    return _row_min_zero((c - mu) / sd)


def make_row_softmax(tau: float = 0.1) -> MatTransform:
    def f(S: np.ndarray) -> np.ndarray:
        z = S / tau
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return _row_min_zero(-np.log(p + 1e-30))
    return f


def make_dual_softmax(tau: float = 0.1) -> MatTransform:
    """LoFTR-style two-way contrast: -log(softmax_row . softmax_col) (§3.2)."""
    def f(S: np.ndarray) -> np.ndarray:
        zr = S / tau
        zr = zr - zr.max(axis=1, keepdims=True)
        pr = np.exp(zr); pr /= pr.sum(axis=1, keepdims=True)
        zc = S / tau
        zc = zc - zc.max(axis=0, keepdims=True)
        pc = np.exp(zc); pc /= pc.sum(axis=0, keepdims=True)
        return _row_min_zero(-np.log(pr + 1e-30) - np.log(pc + 1e-30))
    return f


def make_seqslam_local(R: int = 10, axis: int = 0) -> MatTransform:
    """SeqSLAM local contrast: z-score each cost cell within a +/-R//2 window
    along `axis` (0 = reference/B index, SeqSLAM's choice) (§3.2)."""
    half = max(1, R // 2)

    def f(S: np.ndarray) -> np.ndarray:
        c = 1.0 - S
        # rolling mean/std along axis via cumulative sums
        c_ax = np.moveaxis(c, axis, 0)
        n = c_ax.shape[0]
        out = np.empty_like(c_ax)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            win = c_ax[lo:hi]
            mu = win.mean(0)
            sd = win.std(0) + 1e-12
            out[i] = (c_ax[i] - mu) / sd
        out = np.moveaxis(out, 0, axis)
        return _row_min_zero(out)
    return f


def make_local_z(base: "MatTransform", R: int = 10, axis: int = 0) -> "MatTransform":
    """Apply SeqSLAM-style local-window z-score to the COST produced by `base`
    (composes two-way contrast with local contrast, §3.2)."""
    half = max(1, R // 2)

    def f(S: np.ndarray) -> np.ndarray:
        C = base(S)
        cax = np.moveaxis(C, axis, 0)
        out = np.empty_like(cax)
        for i in range(cax.shape[0]):
            lo, hi = max(0, i - half), min(cax.shape[0], i + half + 1)
            win = cax[lo:hi]
            out[i] = (cax[i] - win.mean(0)) / (win.std(0) + 1e-12)
        return _row_min_zero(np.moveaxis(out, 0, axis))
    return f


def make_diag_smooth(L: int = 7, slopes: tuple[float, ...] = (0.5, 0.67, 1.0, 1.5, 2.0)) -> MatTransform:
    """Multi-slope diagonal smoothing of similarity, max over slopes (§3.3)."""
    def f(S: np.ndarray) -> np.ndarray:
        na, nb = S.shape
        half = L // 2
        best = np.full_like(S, -np.inf)
        ii, jj = np.arange(na)[:, None], np.arange(nb)[None, :]
        for sl in slopes:
            acc = np.zeros_like(S)
            cnt = np.zeros_like(S)
            for d in range(-half, half + 1):
                si = ii + d
                sj = np.round(jj + d * sl).astype(int)
                valid = (si >= 0) & (si < na) & (sj >= 0) & (sj < nb)
                si_c = np.clip(si, 0, na - 1)
                sj_c = np.clip(sj, 0, nb - 1)
                acc += np.where(valid, S[si_c, sj_c], 0.0)
                cnt += valid
            sm = acc / np.maximum(cnt, 1)
            best = np.maximum(best, sm)
        return _row_min_zero(1.0 - best)
    return f


def make_diag_avg(L: int = 5) -> MatTransform:
    """Slope-1 diagonal averaging of similarity — the shapeDTW-concatenation
    equivalent (cos of window-concatenated descriptors ~ mean of S along the
    main diagonal). Distinct from make_diag_smooth (which maxes over slopes)."""
    half = L // 2

    def f(S: np.ndarray) -> np.ndarray:
        na, nb = S.shape
        acc = np.zeros_like(S)
        cnt = np.zeros_like(S)
        for d in range(-half, half + 1):
            # cell (i,j) accumulates S[i+d, j+d] where in range
            i0, i1 = max(0, -d), min(na, na - d)
            j0, j1 = max(0, -d), min(nb, nb - d)
            n = min(i1 - i0, j1 - j0)
            if n <= 0:
                continue
            acc[i0:i0 + n, j0:j0 + n] += S[i0 + d:i0 + d + n, j0 + d:j0 + d + n]
            cnt[i0:i0 + n, j0:j0 + n] += 1
        sm = acc / np.maximum(cnt, 1)
        return _row_min_zero(1.0 - sm)
    return f


# --------------------------------------------------------------------------
# Extended DTW: production algorithm + optional step / slack penalties (§3.4)
# --------------------------------------------------------------------------

@dataclass
class DtwParams:
    band_pct: float = 0.10
    slope_max: float = 2.0
    open_end_s: float = 10.0
    step_penalty: float = 0.0      # added per unit of |di-dj| on each step
    slack_penalty: float = 0.0     # added per boundary frame skipped at ends


def dtw_ext(cost: np.ndarray, open_end_frames: tuple[int, int], p: DtwParams) -> np.ndarray:
    """Banded, slope-constrained, open-end DTW with optional penalties.

    With step_penalty=slack_penalty=0 this reproduces tracksync.dtw.dtw_align's
    path selection. Returns the path as an [K, 2] int array.
    """
    N_A, N_B = cost.shape
    band = p.band_pct
    smax = int(p.slope_max)
    dp = np.full((N_A, N_B), np.inf)
    bp = np.zeros((N_A, N_B), dtype=np.int32)
    steps = np.zeros((N_A, N_B), dtype=np.int32)

    start_slack = open_end_frames[0]
    for i in range(min(start_slack + 1, N_A)):
        for j in range(min(start_slack + 1, N_B)):
            if abs(i / max(N_A - 1, 1) - j / max(N_B - 1, 1)) <= band:
                # entry slack: penalize the max(i,j) skipped boundary frames
                dp[i, j] = cost[i, j] + p.slack_penalty * max(i, j)
                steps[i, j] = 1

    bands = []
    for i in range(N_A):
        inorm = i / max(N_A - 1, 1)
        jlo = max(0, int(np.floor((inorm - band) * (N_B - 1))))
        jhi = min(N_B - 1, int(np.ceil((inorm + band) * (N_B - 1))))
        bands.append((jlo, jhi))

    # candidate steps: (di, dj, penalty_units)
    cand = [(1, 1, 0)]
    for k in range(2, smax + 1):
        cand.append((1, k, k - 1))
        cand.append((k, 1, k - 1))

    for i in range(N_A):
        jlo, jhi = bands[i]
        if i == 0 and jlo == 0:
            jlo = 1
        if i <= start_slack:
            jlo = max(jlo, start_slack + 1)
        if jlo > jhi:
            continue
        for j in range(jlo, jhi + 1):
            best = np.inf; bstep = 0; bsteps = 0
            for (di, dj, pen) in cand:
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0:
                    continue
                v = dp[pi, pj]
                if v == np.inf:
                    continue
                v = v + p.step_penalty * pen
                if v < best:
                    best = v; bstep = di * 1000 + dj; bsteps = steps[pi, pj]
            if best < np.inf:
                dp[i, j] = best + cost[i, j]
                bp[i, j] = bstep
                steps[i, j] = bsteps + 1

    end_slack = open_end_frames[1]
    best_avg = np.inf; ei, ej = N_A - 1, N_B - 1
    for i in range(max(0, N_A - end_slack - 1), N_A):
        for j in range(max(0, N_B - end_slack - 1), N_B):
            if dp[i, j] < np.inf and steps[i, j] > 0:
                skipped = max(N_A - 1 - i, N_B - 1 - j)
                avg = (dp[i, j] + p.slack_penalty * skipped) / steps[i, j]
                if avg < best_avg:
                    best_avg = avg; ei, ej = i, j

    path = []
    i, j = ei, ej
    while True:
        path.append((i, j))
        if (i <= start_slack and j <= start_slack and (i == 0 or j == 0)) or (i == 0 and j == 0):
            break
        step = bp[i, j]
        if step == 0:
            break
        di, dj = step // 1000, step % 1000
        if i >= di and j >= dj:
            i, j = i - di, j - dj
        else:
            break
        if len(path) > N_A + N_B:
            break
    path.reverse()
    return np.array(path, dtype=np.int32)


# --------------------------------------------------------------------------
# Coarse alignment driver + mapping
# --------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    feat: FeatTransform = feat_identity
    mat: MatTransform = mat_baseline
    dtw: DtwParams = field(default_factory=DtwParams)


def _open_end_frames(a: Video, b: Video, open_end_s: float) -> tuple[int, int]:
    n_min = min(len(a.times), len(b.times))
    max_slack = max(1, n_min // 4)
    fa = min(round(open_end_s * a.hz), max_slack)
    fb = min(round(open_end_s * b.hz), max_slack)
    return (fa, fb)


def cost_matrix(a: Video, b: Video, cfg: Config) -> np.ndarray:
    ea, eb = cfg.feat(a.emb, b.emb)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings
        S = ea @ eb.T
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0)
    S = np.clip(S, -1.0, 1.0)
    C = cfg.mat(S)
    return np.nan_to_num(C, nan=0.0, posinf=float(np.nanmax(C[np.isfinite(C)]) if np.isfinite(C).any() else 1.0))


def align_with_cost(a: Video, b: Video, C: np.ndarray, p: DtwParams):
    """Align on a precomputed cost matrix (e.g. a fused appearance+motion cost)."""
    oef = _open_end_frames(a, b, p.open_end_s)
    path = dtw_ext(C, oef, p)
    ta = a.times[path[:, 0]]
    tb = b.times[path[:, 1]]
    keep = np.concatenate([[True], np.diff(ta) > 0])
    tak, tbk = ta[keep], tb[keep]

    def f(t):
        return np.interp(t, tak, tbk)

    return f, path, (float(tak[0]), float(tak[-1]))


def align(a: Video, b: Video, cfg: Config):
    """Return (f, path, cost) where f maps t_A -> t_B (np.interp closure)."""
    C = cost_matrix(a, b, cfg)
    oef = _open_end_frames(a, b, cfg.dtw.open_end_s)
    path = dtw_ext(C, oef, cfg.dtw)
    ta = a.times[path[:, 0]]
    tb = b.times[path[:, 1]]
    keep = np.concatenate([[True], np.diff(ta) > 0])
    tak, tbk = ta[keep], tb[keep]

    def f(t):
        return np.interp(t, tak, tbk)

    return f, path, C, (float(tak[0]), float(tak[-1]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def identity_dev(a: Video, cfg: Config) -> float:
    """Max |f(t)-t| for self-alignment; guard should stay <= 1 sample (§8.1)."""
    f, _, _, (lo, hi) = align(a, a, cfg)
    m = 1.0 / a.hz
    lo2, hi2 = (lo + m, hi - m) if hi - m > lo + m else (lo, hi)
    grid = np.linspace(lo2, hi2, 100)
    return float(np.max(np.abs(f(grid) - grid)))


def vs_pairs(f, dom: tuple[float, float], pairs: np.ndarray) -> Optional[dict]:
    if pairs is None or len(pairs) == 0:
        return None
    lo, hi = dom
    m = [(f(ta) - tb) for ta, tb in pairs if lo <= ta <= hi]
    if not m:
        return None
    d = np.abs(np.array(m))
    return {"max": float(d.max()), "mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "n": len(d)}


def boundary_speeds(f, dom: tuple[float, float], w: float = 4.0) -> tuple[float, float]:
    """Local slope df/dt over the first and last `w` seconds of the domain."""
    lo, hi = dom
    w = min(w, (hi - lo) / 2)
    start = (f(lo + w) - f(lo)) / w
    end = (f(hi) - f(hi - w)) / w
    return float(start), float(end)


def contrast_metrics(C: np.ndarray) -> dict:
    """Flatness diagnostics of a conditioned cost matrix (lower cost = better)."""
    # Convert per-row cost to a similarity-like score for a softmax distribution.
    z = -C
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    row_entropy = -(p * np.log(p + 1e-30)).sum(1)
    # per-row margin: gap between best and second-best cost
    part = np.partition(C, 1, axis=1)
    margin = part[:, 1] - part[:, 0]
    s = np.linalg.svd(C - C.mean(), compute_uv=False)
    ps = s / s.sum()
    eff_rank = float(np.exp(-(ps * np.log(ps + 1e-12)).sum()))
    return {
        "row_entropy_mean": float(row_entropy.mean()),
        "row_entropy_norm": float(row_entropy.mean() / np.log(C.shape[1])),
        "margin_mean": float(margin.mean()),
        "eff_rank": eff_rank,
        "top1_share": float(s[0] ** 2 / (s ** 2).sum()),
    }


def evaluate(a: Video, b: Video, cfg: Config, refs: Optional[dict] = None) -> dict:
    """Full metric bundle for one config on one pair."""
    f, path, C, dom = align(a, b, cfg)
    bs_start, bs_end = boundary_speeds(f, dom)
    out = {
        "config": cfg.name,
        "pair": f"{a.label}|{b.label}",
        "dom": dom,
        "boundary_speed_start": bs_start,
        "boundary_speed_end": bs_end,
        "path_len": int(len(path)),
        **{f"contrast_{k}": v for k, v in contrast_metrics(C).items()},
    }
    out["identity_dev"] = identity_dev(a, cfg)
    if refs is not None:
        for key in ("catalyst_pairs", "ocr_pairs"):
            st = vs_pairs(f, dom, refs.get(key))
            if st:
                out[f"vs_{key.split('_')[0]}"] = st
    return out
