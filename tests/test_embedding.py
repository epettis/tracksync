"""Tests for frame embedding extraction.

Tests the FrameEmbedder protocol, GistEmbedder implementation, and caching
functionality.
"""

import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from tracksync.embedding import (
    FrameEmbedder,
    GistEmbedder,
    build_cache_key,
    embed_video_cached,
)


class TestGistEmbedder:
    """Tests for the GistEmbedder implementation."""

    def test_output_shape_and_dtype(self):
        """Test that output has correct shape and dtype."""
        embedder = GistEmbedder()
        frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8) for _ in range(5)]

        embeddings = embedder.embed(frames)

        assert embeddings.shape == (5, 256), "Expected shape [N, 256]"
        assert embeddings.dtype == np.float32, "Expected float32 dtype"

    def test_l2_normalization(self):
        """Test that output rows are L2-normalized."""
        embedder = GistEmbedder()
        frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8) for _ in range(3)]

        embeddings = embedder.embed(frames)

        # Compute L2 norms of each row
        norms = np.linalg.norm(embeddings, axis=1)

        # Should be normalized to 1.0 (within floating-point tolerance)
        assert np.allclose(norms, 1.0, atol=1e-6), "Rows should be L2-normalized"

    def test_determinism(self):
        """Test that same input produces identical output."""
        embedder = GistEmbedder()
        frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8) for _ in range(3)]

        embeddings1 = embedder.embed(frames)
        embeddings2 = embedder.embed(frames)

        assert np.array_equal(embeddings1, embeddings2), "Same input should produce identical output"

    def test_mask_changes_output(self):
        """Test that mask affects the output."""
        embedder = GistEmbedder()
        frame = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

        # Create a mask that covers the left half
        mask = np.zeros((100, 150), dtype=bool)
        mask[:, :75] = True

        # Embed without mask
        emb_no_mask = embedder.embed([frame], mask=None)

        # Embed with mask
        emb_with_mask = embedder.embed([frame], mask=mask)

        # The embeddings should be different
        assert not np.allclose(emb_no_mask, emb_with_mask, atol=1e-6), \
            "Mask should change the embedding"

    def test_grayscale_input(self):
        """Test that grayscale input is handled correctly."""
        embedder = GistEmbedder()
        frames_gray = [np.random.randint(0, 256, (100, 150), dtype=np.uint8) for _ in range(2)]

        embeddings = embedder.embed(frames_gray)

        assert embeddings.shape == (2, 256), "Grayscale input should work"
        assert embeddings.dtype == np.float32

    def test_embedder_name(self):
        """Test that embedder has correct name attribute."""
        embedder = GistEmbedder()
        assert embedder.name == "gist", "Embedder name should be 'gist'"

    def test_zero_norm_handling(self):
        """Test handling of frames that would produce zero norm after mean subtraction."""
        embedder = GistEmbedder()
        # Create a constant frame
        constant_frame = np.full((100, 150, 3), 128, dtype=np.uint8)

        embeddings = embedder.embed([constant_frame])

        # Should still produce valid output (zero vector is acceptable)
        assert embeddings.shape == (1, 256)
        assert embeddings.dtype == np.float32


class CountingEmbedder:
    """Test embedder that counts embed() calls."""

    def __init__(self):
        self.name = "counting"
        self.call_count = 0

    def embed(self, frames: list[np.ndarray], mask: np.ndarray | None = None) -> np.ndarray:
        """Embed frames and increment call count."""
        self.call_count += 1
        n = len(frames)
        # Return dummy embeddings
        return np.random.randn(n, 128).astype(np.float32)


class TestEmbedVideoCached:
    """Tests for cached embedding functionality."""

    def test_cache_hit_avoids_recompute(self):
        """Test that cache hit skips recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = CountingEmbedder()
            frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8) for _ in range(3)]
            mask = None
            cache_key = "test_key_1"

            # First call should compute
            result1 = embed_video_cached(embedder, frames, mask, cache_key, cache_dir=tmpdir)
            assert embedder.call_count == 1, "First call should compute embeddings"

            # Second call with same key should hit cache
            result2 = embed_video_cached(embedder, frames, mask, cache_key, cache_dir=tmpdir)
            assert embedder.call_count == 1, "Second call should hit cache, not recompute"

            # Results should be identical
            assert np.array_equal(result1, result2), "Cached result should match original"

    def test_cache_invalidates_on_key_change(self):
        """Test that different cache key causes recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = CountingEmbedder()
            frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8) for _ in range(3)]
            mask = None

            # First call with key1
            result1 = embed_video_cached(embedder, frames, mask, "key1", cache_dir=tmpdir)
            assert embedder.call_count == 1

            # Second call with different key
            result2 = embed_video_cached(embedder, frames, mask, "key2", cache_dir=tmpdir)
            assert embedder.call_count == 2, "Different key should cause recomputation"

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a subdirectory that doesn't exist yet
            cache_dir = Path(tmpdir) / "nested" / "cache"
            assert not cache_dir.exists()

            embedder = GistEmbedder()
            frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)]

            embed_video_cached(embedder, frames, None, "test_key", cache_dir=cache_dir)

            assert cache_dir.exists(), "Cache directory should be created"

    def test_cache_file_saved(self):
        """Test that cache file is actually saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = GistEmbedder()
            frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)]
            cache_key = "test_key_save"

            embed_video_cached(embedder, frames, None, cache_key, cache_dir=tmpdir)

            cache_file = Path(tmpdir) / f"{cache_key}.npy"
            assert cache_file.exists(), "Cache file should be saved"

    def test_default_cache_directory(self):
        """Test that default cache directory is used when not specified."""
        embedder = GistEmbedder()
        frames = [np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)]
        cache_key = "test_default_dir"

        result = embed_video_cached(embedder, frames, None, cache_key)

        # Should create default cache directory
        default_cache = Path.home() / ".cache" / "tracksync" / "embeddings"
        assert default_cache.exists(), "Default cache directory should be created"

        # Clean up
        cache_file = default_cache / f"{cache_key}.npy"
        if cache_file.exists():
            cache_file.unlink()


class TestBuildCacheKey:
    """Tests for cache key building."""

    def test_key_includes_all_parameters(self):
        """Test that cache key changes with any parameter."""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            f.write(b"dummy")

        try:
            mask1 = np.zeros((100, 150), dtype=bool)
            mask2 = np.ones((100, 150), dtype=bool)

            # Same parameters should produce same key
            key1a = build_cache_key(video_path, "embedder1", 10.0, mask1)
            key1b = build_cache_key(video_path, "embedder1", 10.0, mask1)
            assert key1a == key1b, "Same parameters should produce same key"

            # Different embedder name
            key2 = build_cache_key(video_path, "embedder2", 10.0, mask1)
            assert key1a != key2, "Different embedder name should change key"

            # Different sample_hz
            key3 = build_cache_key(video_path, "embedder1", 5.0, mask1)
            assert key1a != key3, "Different sample_hz should change key"

            # Different mask
            key4 = build_cache_key(video_path, "embedder1", 10.0, mask2)
            assert key1a != key4, "Different mask should change key"

            # None mask vs actual mask
            key5 = build_cache_key(video_path, "embedder1", 10.0, None)
            assert key1a != key5, "None mask should produce different key"

        finally:
            # Clean up
            Path(video_path).unlink()

    def test_key_changes_with_mtime(self):
        """Test that cache key changes when file is modified."""
        import time

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            f.write(b"dummy")

        try:
            key1 = build_cache_key(video_path, "embedder", 10.0, None)

            # Wait a bit and modify the file
            time.sleep(0.1)
            with open(video_path, "ab") as f:
                f.write(b"more")

            key2 = build_cache_key(video_path, "embedder", 10.0, None)

            assert key1 != key2, "Modified file should produce different key"

        finally:
            Path(video_path).unlink()


class TestFrameEmbedderProtocol:
    """Tests for the FrameEmbedder protocol."""

    def test_gist_embedder_satisfies_protocol(self):
        """Test that GistEmbedder satisfies the FrameEmbedder protocol."""
        embedder: FrameEmbedder = GistEmbedder()

        # Should have name attribute
        assert hasattr(embedder, "name")
        assert isinstance(embedder.name, str)

        # Should have embed method
        assert hasattr(embedder, "embed")
        assert callable(embedder.embed)

    def test_counting_embedder_satisfies_protocol(self):
        """Test that CountingEmbedder satisfies the FrameEmbedder protocol."""
        embedder: FrameEmbedder = CountingEmbedder()

        assert hasattr(embedder, "name")
        assert isinstance(embedder.name, str)
        assert hasattr(embedder, "embed")
        assert callable(embedder.embed)
