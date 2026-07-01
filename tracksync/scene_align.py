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

import numpy as np

from .dtw import DtwResult, SmoothPath, dtw_align, smooth_path
from .embedding import FrameEmbedder, build_cache_key, embed_video_cached
from .feature_extraction import sample_frames
from .frame_data import SceneFeatures
from .masking import compute_static_mask


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


def coarse_align(
    feat_a: SceneFeatures,
    feat_b: SceneFeatures,
    band_pct: float = 0.10,
    open_end_s: float = 10.0,
) -> CoarseAlignment:
    """Perform coarse alignment between two videos via DTW on embeddings.

    Computes a cosine cost matrix from the embeddings, runs banded open-end DTW,
    smooths the path into a continuous time-mapping function, and extracts trim
    boundaries from the path endpoints.

    Args:
        feat_a: Scene features for video A
        feat_b: Scene features for video B
        band_pct: Sakoe-Chiba band as fraction of sequence length (default 0.10)
        open_end_s: Open-end slack in seconds at start/end (default 10.0)

    Returns:
        CoarseAlignment with smoothed mapping, path, margins, and trim times

    Design reference: docs/scene_alignment_design.md §4.4, task T7
    """
    # Step 1: Compute cosine cost matrix (1 - similarity)
    # Embeddings are L2-normalized, so cosine similarity = dot product
    # Use float64 for numerical stability and suppress overflow warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        emb_a_64 = feat_a.emb_array.astype(np.float64)
        emb_b_64 = feat_b.emb_array.astype(np.float64)
        similarity = emb_a_64 @ emb_b_64.T  # [N_A, N_B]

    # Clip to valid range for cosine similarity [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)
    cost = 1.0 - similarity

    # Ensure cost matrix is valid (no NaN/Inf)
    if not np.all(np.isfinite(cost)):
        # Replace invalid values with a high cost
        cost = np.nan_to_num(cost, nan=2.0, posinf=2.0, neginf=0.0)

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
