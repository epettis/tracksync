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


class DinoV2Embedder:
    """DINOv2 vision transformer embedder for robust visual place recognition.

    Uses pretrained DINOv2 models from Facebook Research via torch.hub. Extracts
    patch tokens, applies spatial masking to exclude static car-body regions,
    and aggregates tokens via GeM pooling (generalized mean, p=3).

    Design reference: docs/scene_alignment_design.md §4.3, §11.1

    Attributes:
        name: Embedder identifier for cache keys
        model_name: DINOv2 model name (e.g., "dinov2_vitb14")
        device: Compute device (mps/cuda/cpu), auto-selected if None
        batch_size: Number of frames to process in one forward pass
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | None = None,
        batch_size: int = 16
    ):
        """Initialize DinoV2Embedder.

        Args:
            model_name: DINOv2 model name to load via torch.hub
            device: Compute device (mps/cuda/cpu), auto-selected if None
            batch_size: Number of frames to process per batch
        """
        # Check scene dependencies before any torch imports
        from tracksync.scene_deps import require_scene_deps
        require_scene_deps()

        self.model_name = model_name
        self.name = model_name.replace("_", "-")  # e.g., "dinov2-vitb14"
        self.device_str = device
        self.batch_size = batch_size
        self._model = None
        self._device = None
        self._patch_size = 14  # DINOv2 patch size

    def _ensure_model_loaded(self):
        """Lazy model loading on first embed() call."""
        if self._model is not None:
            return

        # Import torch only when actually needed
        import torch

        # Auto-select device
        if self.device_str is None:
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.device_str)

        # Load model via torch.hub
        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def _resize_to_multiple(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so H and W are multiples of patch_size (14).

        Args:
            frame: RGB uint8 frame HxWx3

        Returns:
            Resized frame with H and W as multiples of 14
        """
        import cv2
        h, w = frame.shape[:2]

        # Round up to next multiple of patch_size
        h_new = ((h + self._patch_size - 1) // self._patch_size) * self._patch_size
        w_new = ((w + self._patch_size - 1) // self._patch_size) * self._patch_size

        if h_new != h or w_new != w:
            frame = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        return frame

    def _downsample_mask_to_patches(
        self,
        mask: np.ndarray,
        h_patches: int,
        w_patches: int
    ) -> np.ndarray:
        """Downsample boolean mask to patch grid.

        A patch is marked True (masked) if > 50% of its pixels are True in the
        original mask.

        Args:
            mask: HxW boolean mask, True = static/masked
            h_patches: Number of patch rows
            w_patches: Number of patch columns

        Returns:
            h_patches x w_patches boolean mask
        """
        import cv2
        h, w = mask.shape

        # Resize mask to patch grid, counting True pixels in each patch
        # Use area interpolation which averages the values
        mask_float = mask.astype(np.float32)
        mask_downsampled = cv2.resize(
            mask_float,
            (w_patches, h_patches),
            interpolation=cv2.INTER_AREA
        )

        # A patch is masked if > 50% of its pixels were True
        return mask_downsampled > 0.5

    def _gem_pool(self, tokens: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
        """Generalized mean pooling over tokens.

        GeM pooling: (mean(clamp(x, min=eps) ** p)) ** (1/p)

        Args:
            tokens: [N_tokens, D] float array
            p: GeM power parameter
            eps: Clamp minimum to avoid zero

        Returns:
            [D] pooled descriptor
        """
        # Clamp to avoid zeros
        tokens_clamped = np.maximum(tokens, eps)
        # Raise to power p
        tokens_pow = tokens_clamped ** p
        # Mean over tokens
        mean_pow = tokens_pow.mean(axis=0)
        # Take p-th root
        pooled = mean_pow ** (1.0 / p)
        return pooled

    def embed(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Extract DINOv2 embeddings from frames.

        Args:
            frames: List of RGB uint8 frames, each HxWx3
            mask: Optional HxW bool array, True = static/exclude pixels

        Returns:
            float32 array of shape [N, D], rows L2-normalized
        """
        if not frames:
            return np.array([], dtype=np.float32).reshape(0, 0)

        # Ensure model is loaded
        self._ensure_model_loaded()

        import torch

        # Process frames in batches
        all_embeddings = []

        for batch_start in range(0, len(frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            # Resize frames to multiples of patch_size
            resized_frames = [self._resize_to_multiple(f) for f in batch_frames]

            # Convert to torch tensors [B, 3, H, W], normalized to [0, 1]
            batch_tensors = []
            for frame in resized_frames:
                # Convert to CHW format and normalize
                tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(self._device)

            # Extract patch tokens
            with torch.no_grad():
                features = self._model.forward_features(batch_tensor)
                patch_tokens = features["x_norm_patchtokens"]  # [B, N_patches, D]

            # Convert to numpy
            patch_tokens_np = patch_tokens.cpu().numpy()  # [B, N_patches, D]

            # Process each frame in the batch
            for i, frame in enumerate(resized_frames):
                tokens = patch_tokens_np[i]  # [N_patches, D]

                # Apply mask if provided
                if mask is not None:
                    h, w = frame.shape[:2]
                    h_patches = h // self._patch_size
                    w_patches = w // self._patch_size

                    # Downsample mask to patch grid
                    patch_mask = self._downsample_mask_to_patches(mask, h_patches, w_patches)

                    # Flatten patch mask to match tokens
                    patch_mask_flat = patch_mask.flatten()  # [N_patches]

                    # Keep only tokens where mask is False (not static)
                    keep_indices = ~patch_mask_flat
                    tokens = tokens[keep_indices]

                # GeM pooling over remaining tokens
                if len(tokens) > 0:
                    pooled = self._gem_pool(tokens, p=3.0)
                else:
                    # If all tokens are masked, use zero vector
                    pooled = np.zeros(tokens.shape[1], dtype=np.float32)

                # L2 normalize
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm

                all_embeddings.append(pooled)

        return np.array(all_embeddings, dtype=np.float32)


def make_embedder(name: str) -> FrameEmbedder:
    """Factory function for creating embedders by name.

    Args:
        name: Embedder name, one of:
            - "gist": GistEmbedder (deterministic baseline)
            - "dinov2-vits14": DINOv2 ViT-S/14
            - "dinov2-vitb14": DINOv2 ViT-B/14 (default backbone)

    Returns:
        FrameEmbedder instance

    Raises:
        ValueError: If name is not recognized
    """
    if name == "gist":
        return GistEmbedder()
    elif name == "dinov2-vits14":
        return DinoV2Embedder(model_name="dinov2_vits14")
    elif name == "dinov2-vitb14":
        return DinoV2Embedder(model_name="dinov2_vitb14")
    else:
        valid_names = ["gist", "dinov2-vits14", "dinov2-vitb14"]
        raise ValueError(
            f"Unknown embedder name: {name}. Valid options: {', '.join(valid_names)}"
        )
