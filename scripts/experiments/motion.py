"""Phase 3 — non-appearance motion signals for temporal lap alignment.

Extracts two appearance-invariant 1-D traces per video from sparse optical
flow, aligned to the same 10 Hz grid as the embeddings:

  - yaw_rate proxy : signed median horizontal flow of FAR-FIELD points
    (upper image, translation-induced flow falls off as 1/depth so distant
    flow is rotation-dominated). Large at corners, ~0 on straights.
  - speed proxy    : median flow magnitude of GROUND points (lower image),
    proportional to v / camera_height up to a per-lap constant.

Car-body pixels are excluded via the existing static mask. Because both
traces are z-normalized per lap before use, absolute scale (focal length,
mount height) is irrelevant — only the shape matters to DTW, so no
intrinsics are needed. Research reference: dtw_contrast_research.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tracksync.feature_extraction import sample_frames
from tracksync.masking import compute_static_mask

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "data"


@dataclass
class MotionTraces:
    label: str
    times: np.ndarray       # [N]
    yaw: np.ndarray         # [N] signed horizontal far-field flow (px)
    speed: np.ndarray       # [N] ground flow magnitude (px)
    hz: float


def _flow_stats(prev_gray, gray, mask_keep, far_row, ground_row):
    """Median signed-dx (far field) and flow magnitude (ground) between frames.

    mask_keep: HxW bool, True where pixels are usable (NOT car body).
    far_row / ground_row: row index splitting far-field (above) vs ground (below).
    """
    h, w = prev_gray.shape
    # Seed features only on usable, non-body pixels.
    feat_mask = (mask_keep * 255).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=600, qualityLevel=0.01,
                                  minDistance=7, mask=feat_mask)
    if pts is None or len(pts) < 8:
        return np.nan, np.nan
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                          winSize=(21, 21), maxLevel=3)
    st = st.ravel().astype(bool)
    p0 = pts.reshape(-1, 2)[st]
    p1 = nxt.reshape(-1, 2)[st]
    if len(p0) < 8:
        return np.nan, np.nan
    d = p1 - p0
    y0 = p0[:, 1]
    far = y0 < far_row
    ground = y0 > ground_row
    yaw = float(np.median(d[far, 0])) if far.sum() >= 4 else np.nan
    speed = float(np.median(np.hypot(d[ground, 0], d[ground, 1]))) if ground.sum() >= 4 else np.nan
    return yaw, speed


def extract_motion(video_path: str, label: str, sample_hz: float = 10.0,
                   max_dim: int = 518) -> MotionTraces:
    frames, times = sample_frames(video_path, sample_hz=sample_hz, max_dim=max_dim)
    static = compute_static_mask(frames)          # HxW bool, True = static/body
    keep = ~static
    h, w = frames[0].shape[:2]
    far_row = int(0.45 * h)       # above this = far field / horizon
    ground_row = int(0.55 * h)    # below this = road surface
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    n = len(frames)
    yaw = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    for i in range(n - 1):
        yaw[i], speed[i] = _flow_stats(grays[i], grays[i + 1], keep, far_row, ground_row)
    # last frame copies its predecessor; fill any NaN by interpolation
    yaw[-1] = yaw[-2]
    speed[-1] = speed[-2]
    yaw = _fill_nan(yaw)
    speed = _fill_nan(speed)
    return MotionTraces(label, np.asarray(times), yaw, speed, sample_hz)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def save_traces(t: MotionTraces) -> Path:
    out = DATA / f"{t.label}__motion.npz"
    np.savez(out, times=t.times, yaw=t.yaw, speed=t.speed, hz=np.float64(t.hz), label=t.label)
    return out


def load_traces(label: str) -> MotionTraces:
    d = np.load(DATA / f"{label}__motion.npz", allow_pickle=True)
    return MotionTraces(str(d["label"]), d["times"], d["yaw"], d["speed"], float(d["hz"]))


# --------------------------------------------------------------------------
# Motion cost matrices + fusion
# --------------------------------------------------------------------------

def _zn(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def trace_cost(trace_a: np.ndarray, trace_b: np.ndarray) -> np.ndarray:
    """Absolute-difference cost between two z-normalized 1-D traces (row-min 0)."""
    za, zb = _zn(trace_a), _zn(trace_b)
    C = np.abs(za[:, None] - zb[None, :])
    return C - C.min(axis=1, keepdims=True)


def normalize_cost(C: np.ndarray) -> np.ndarray:
    """Scale a cost matrix to unit mean so heterogeneous costs are fusable."""
    m = C[np.isfinite(C)].mean()
    return C / (m + 1e-12)
