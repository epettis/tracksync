"""Frame embedding extraction for scene-based video alignment.

This module provides interfaces and implementations for extracting global
descriptors from video frames. Embeddings are used in the coarse alignment
stage to match frames across videos with different camera poses and mounting
positions.

The module supports pluggable embedders via the FrameEmbedder protocol,
with implementations ranging from simple deterministic baselines (GistEmbedder)
to pretrained vision transformers (DINOv2).

Design reference: docs/scene_alignment_design.md §4.3
"""

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

# Lazy torch imports: torch is only imported when actually creating a
# torch-based embedder, not at module import time. This allows the module
# to be imported in environments where torch is not installed.


class FrameEmbedder(Protocol):
    """Protocol for frame embedders.

    Embedders extract global descriptors from video frames, producing
    L2-normalized feature vectors suitable for cosine-distance matching.

    Attributes:
        name: Embedder identifier used in cache keys (e.g., "dinov2-vitb14")
    """

    name: str

    def embed(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Extract embeddings from a list of frames.

        Args:
            frames: List of RGB uint8 frames, each HxWx3
            mask: Optional HxW bool array, True = static/exclude pixels

        Returns:
            float32 array of shape [N, D], rows L2-normalized
        """
        ...


class GistEmbedder:
    """Deterministic baseline embedder using downsampled GIST-like features.

    Pure NumPy implementation for testing and debugging. Converts frames to
    grayscale, sets masked (static) pixels to the frame mean, downsamples to
    16x16, subtracts mean, L2-normalizes, and flattens to D=256.

    Deterministic and fast, but not robust to significant viewpoint changes.
    """

    name = "gist"

    def embed(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Extract GIST-like embeddings from frames.

        Args:
            frames: List of RGB uint8 frames, each HxWx3
            mask: Optional HxW bool array, True = static/exclude pixels

        Returns:
            float32 array of shape [N, 256], rows L2-normalized
        """
        embeddings = []

        for frame in frames:
            # Convert to grayscale
            if frame.ndim == 3:
                gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = frame.astype(np.float32)

            # Apply mask: set static pixels to frame mean
            if mask is not None:
                frame_mean = gray.mean()
                gray = gray.copy()
                gray[mask] = frame_mean

            # Downsample to 16x16 using simple area averaging (pure NumPy)
            h, w = gray.shape
            # Use simple block averaging for downsampling
            h_step = h / 16
            w_step = w / 16
            downsample = np.zeros((16, 16), dtype=np.float32)
            for i in range(16):
                for j in range(16):
                    y_start = int(i * h_step)
                    y_end = int((i + 1) * h_step)
                    x_start = int(j * w_step)
                    x_end = int((j + 1) * w_step)
                    downsample[i, j] = gray[y_start:y_end, x_start:x_end].mean()

            # Subtract mean
            downsample = downsample - downsample.mean()

            # L2-normalize
            norm = np.linalg.norm(downsample)
            if norm > 0:
                downsample = downsample / norm

            # Flatten
            embedding = downsample.flatten()

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


def _compute_cache_key(
    video_path: str,
    embedder_name: str,
    sample_hz: float,
    mask_hash: str
) -> str:
    """Compute cache key from video path, mtime, embedder name, sample_hz, and mask hash.

    Args:
        video_path: Path to video file
        embedder_name: Name of the embedder
        sample_hz: Sampling rate in Hz
        mask_hash: Hash of the mask array

    Returns:
        SHA256 hex digest string
    """
    mtime = os.path.getmtime(video_path)
    key_parts = [
        video_path,
        str(mtime),
        embedder_name,
        str(sample_hz),
        mask_hash
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _hash_mask(mask: np.ndarray | None) -> str:
    """Hash a mask array for cache key computation.

    Args:
        mask: Optional HxW bool array

    Returns:
        SHA256 hex digest of the mask bytes, or "none" if mask is None
    """
    if mask is None:
        return "none"
    return hashlib.sha256(mask.tobytes()).hexdigest()


def build_cache_key(
    video_path: str,
    embedder_name: str,
    sample_hz: float,
    mask: np.ndarray | None = None
) -> str:
    """Build a cache key from video path, mtime, embedder name, sample_hz, and mask.

    Helper function that computes a unique cache key based on all relevant
    parameters that affect the embedding output. Cache hits skip recomputation.

    Args:
        video_path: Path to video file
        embedder_name: Name of the embedder
        sample_hz: Sampling rate in Hz
        mask: Optional HxW bool array, True = static/exclude pixels

    Returns:
        Cache key string suitable for use with embed_video_cached
    """
    mask_hash = _hash_mask(mask)
    return _compute_cache_key(video_path, embedder_name, sample_hz, mask_hash)


def embed_video_cached(
    embedder: FrameEmbedder,
    frames: list[np.ndarray],
    mask: np.ndarray | None,
    cache_key: str,
    cache_dir: str | Path | None = None
) -> np.ndarray:
    """Embed frames with disk caching.

    Memoizes the embedding array as a .npy file under cache_dir. On cache hit,
    skips recomputation and loads from disk. On cache miss, computes embeddings
    and saves to disk.

    Args:
        embedder: FrameEmbedder instance
        frames: List of RGB uint8 frames, each HxWx3
        mask: Optional HxW bool array, True = static/exclude pixels
        cache_key: Unique key for this embedding computation (use build_cache_key)
        cache_dir: Directory for cache files (default: ~/.cache/tracksync/embeddings)

    Returns:
        float32 array of shape [N, D], rows L2-normalized
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "tracksync" / "embeddings"
    else:
        cache_dir = Path(cache_dir)

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Construct cache file path
    cache_file = cache_dir / f"{cache_key}.npy"

    # Check for cache hit
    if cache_file.exists():
        return np.load(cache_file)

    # Cache miss: compute embeddings
    embeddings = embedder.embed(frames, mask)

    # Save to cache
    np.save(cache_file, embeddings)

    return embeddings
