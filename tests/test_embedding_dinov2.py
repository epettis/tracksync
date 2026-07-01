"""Tests for DINOv2 embedder.

Tests the DinoV2Embedder implementation and make_embedder factory function.
Includes both fast tests (no torch/model download) and slow tests that use
the actual model.
"""

import numpy as np
import pytest

from tracksync.scene_deps import MissingSceneDependenciesError


class TestMakeEmbedderFactory:
    """Tests for the make_embedder factory function."""

    def test_factory_returns_gist(self):
        """Test that factory returns GistEmbedder for 'gist' name."""
        from tracksync.embedding import GistEmbedder, make_embedder

        embedder = make_embedder("gist")
        assert isinstance(embedder, GistEmbedder)
        assert embedder.name == "gist"

    def test_factory_unknown_name_raises(self):
        """Test that factory raises ValueError for unknown name."""
        from tracksync.embedding import make_embedder

        with pytest.raises(ValueError, match="Unknown embedder name"):
            make_embedder("nonexistent")

        # Check that error message lists valid options
        with pytest.raises(ValueError, match="gist"):
            make_embedder("nonexistent")


class TestDinoV2EmbedderWithoutTorch:
    """Tests for DinoV2Embedder that don't require torch."""

    def test_construction_requires_scene_deps(self, monkeypatch):
        """Test that constructing DinoV2Embedder raises error without torch."""
        # Mock require_scene_deps to raise
        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr("tracksync.scene_deps.require_scene_deps", mock_require)

        from tracksync.embedding import DinoV2Embedder

        with pytest.raises(MissingSceneDependenciesError):
            DinoV2Embedder()

    def test_factory_dinov2_requires_scene_deps(self, monkeypatch):
        """Test that factory raises error for dinov2 names without torch."""
        # Mock require_scene_deps to raise
        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr("tracksync.scene_deps.require_scene_deps", mock_require)

        from tracksync.embedding import make_embedder

        with pytest.raises(MissingSceneDependenciesError):
            make_embedder("dinov2-vits14")

        with pytest.raises(MissingSceneDependenciesError):
            make_embedder("dinov2-vitb14")


# The following tests require torch and should only run when marked as slow
# and when torch is actually available


def _torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="torch not available")
class TestDinoV2EmbedderSlow:
    """Slow tests that use the actual DINOv2 model."""

    def test_factory_returns_correct_types(self):
        """Test that factory returns DinoV2Embedder for dinov2 names."""
        from tracksync.embedding import DinoV2Embedder, make_embedder

        embedder_s = make_embedder("dinov2-vits14")
        assert isinstance(embedder_s, DinoV2Embedder)
        assert embedder_s.name == "dinov2-vits14"

        embedder_b = make_embedder("dinov2-vitb14")
        assert isinstance(embedder_b, DinoV2Embedder)
        assert embedder_b.name == "dinov2-vitb14"

    def test_name_attribute_correctness(self):
        """Test that DinoV2Embedder has correct name attribute."""
        from tracksync.embedding import DinoV2Embedder

        embedder_s = DinoV2Embedder("dinov2_vits14")
        assert embedder_s.name == "dinov2-vits14"

        embedder_b = DinoV2Embedder("dinov2_vitb14")
        assert embedder_b.name == "dinov2-vitb14"

    def test_real_model_duplicate_detection(self):
        """Test that real model can distinguish duplicates from unrelated frames.

        Uses small frames (126x126) for speed and dinov2_vits14 to minimize download.
        Creates 4 frames:
        - 2 near-duplicates (same noise image with tiny brightness change)
        - 2 unrelated noise images

        Asserts that cosine similarity of duplicate pair > unrelated pair.
        """
        from tracksync.embedding import DinoV2Embedder

        try:
            # Use vits14 for smaller download
            embedder = DinoV2Embedder("dinov2_vits14")

            # Create test frames (126x126 = 9x14, multiple of patch size)
            np.random.seed(42)

            # Frame 0: random noise
            frame0 = np.random.randint(0, 256, (126, 126, 3), dtype=np.uint8)

            # Frame 1: same as frame0 but with tiny brightness change (near-duplicate)
            frame1 = np.clip(frame0.astype(np.int16) + 5, 0, 255).astype(np.uint8)

            # Frame 2: different random noise
            frame2 = np.random.randint(0, 256, (126, 126, 3), dtype=np.uint8)

            # Frame 3: yet another different random noise
            frame3 = np.random.randint(100, 200, (126, 126, 3), dtype=np.uint8)

            frames = [frame0, frame1, frame2, frame3]

            # Embed all frames
            embeddings = embedder.embed(frames)

            # Check output shape and dtype
            assert embeddings.shape == (4, embeddings.shape[1]), "Should have 4 embeddings"
            assert embeddings.dtype == np.float32, "Should be float32"

            # Check L2 normalization
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), "Rows should be L2-normalized"

            # Compute cosine similarities
            # Duplicate pair: frame0 vs frame1
            sim_duplicate = embeddings[0] @ embeddings[1]

            # Unrelated pairs: frame0 vs frame2, frame0 vs frame3
            sim_unrelated_1 = embeddings[0] @ embeddings[2]
            sim_unrelated_2 = embeddings[0] @ embeddings[3]
            sim_unrelated_avg = (sim_unrelated_1 + sim_unrelated_2) / 2

            # Duplicate similarity should exceed unrelated similarity
            assert sim_duplicate > sim_unrelated_avg, (
                f"Duplicate pair similarity ({sim_duplicate:.4f}) should exceed "
                f"unrelated pair similarity ({sim_unrelated_avg:.4f})"
            )

            # Print measured similarities for reporting
            print(f"\nMeasured cosine similarities:")
            print(f"  Duplicate pair (frame0 vs frame1): {sim_duplicate:.4f}")
            print(f"  Unrelated pair 1 (frame0 vs frame2): {sim_unrelated_1:.4f}")
            print(f"  Unrelated pair 2 (frame0 vs frame3): {sim_unrelated_2:.4f}")
            print(f"  Unrelated average: {sim_unrelated_avg:.4f}")

        except Exception as e:
            # If we can't fetch the model (network issue), skip the test
            if "HTTP" in str(e) or "timeout" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Could not fetch model from torch.hub: {e}")
            raise

    def test_masking_excludes_patches(self):
        """Test that static mask correctly excludes patches."""
        from tracksync.embedding import DinoV2Embedder

        try:
            embedder = DinoV2Embedder("dinov2_vits14")

            # Create a frame (140x140 = 10x14 patches)
            np.random.seed(123)
            frame = np.random.randint(0, 256, (140, 140, 3), dtype=np.uint8)

            # Create a mask that covers the left half
            mask = np.zeros((140, 140), dtype=bool)
            mask[:, :70] = True

            # Embed without mask
            emb_no_mask = embedder.embed([frame], mask=None)

            # Embed with mask
            emb_with_mask = embedder.embed([frame], mask=mask)

            # Embeddings should be different
            assert not np.allclose(emb_no_mask, emb_with_mask, atol=1e-4), (
                "Mask should change the embedding"
            )

        except Exception as e:
            if "HTTP" in str(e) or "timeout" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Could not fetch model from torch.hub: {e}")
            raise

    def test_empty_frames_list(self):
        """Test handling of empty frames list."""
        from tracksync.embedding import DinoV2Embedder

        embedder = DinoV2Embedder("dinov2_vits14")

        # Should return empty array with correct shape
        embeddings = embedder.embed([])
        assert embeddings.shape[0] == 0
        assert embeddings.dtype == np.float32
