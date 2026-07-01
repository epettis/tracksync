"""Tests for the LightGlue feature matcher.

Tests the LightGlueMatcher implementation and make_matcher factory function.
Fast tests exercise the factory, dependency guard, and license warning without
importing torch. Slow tests run the real ALIKED+LightGlue models against
synthetic images related by a known homography.

Task reference: docs/scene_alignment_tasks.md T9
Design reference: docs/scene_alignment_design.md §5.2, §9.4
"""

import numpy as np
import pytest

from tracksync.scene_deps import MissingSceneDependenciesError


class TestMakeMatcherFactory:
    """Tests for the make_matcher factory function."""

    def test_factory_unknown_name_raises(self):
        """Test that factory raises ValueError for unknown name."""
        from tracksync.fine_align import make_matcher

        with pytest.raises(ValueError, match="Unknown matcher name"):
            make_matcher("nonexistent")

        # Error message should list the valid options
        with pytest.raises(ValueError, match="aliked-lightglue"):
            make_matcher("nonexistent")

    def test_factory_synthetic_reserved(self):
        """Test that 'synthetic' is reserved for tests and rejected."""
        from tracksync.fine_align import make_matcher

        with pytest.raises(ValueError, match="reserved for tests"):
            make_matcher("synthetic")

    def test_factory_requires_scene_deps(self, monkeypatch):
        """Test that factory raises error for lightglue names without torch."""
        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", mock_require
        )

        from tracksync.fine_align import make_matcher

        with pytest.raises(MissingSceneDependenciesError):
            make_matcher("aliked-lightglue")

        with pytest.raises(MissingSceneDependenciesError):
            make_matcher("superpoint-lightglue")


class TestLightGlueMatcherWithoutTorch:
    """Tests for LightGlueMatcher that don't require torch.

    Construction is lazy (models load on first match() call), so these tests
    only need the dependency guard monkeypatched to a no-op.
    """

    def test_construction_requires_scene_deps(self, monkeypatch):
        """Test that constructing LightGlueMatcher raises error without torch."""
        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", mock_require
        )

        from tracksync.fine_align import LightGlueMatcher

        with pytest.raises(MissingSceneDependenciesError):
            LightGlueMatcher()

    def test_invalid_features_raises(self, monkeypatch):
        """Test that an unknown features type raises ValueError."""
        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", lambda: None
        )

        from tracksync.fine_align import LightGlueMatcher

        with pytest.raises(ValueError, match="Unknown features type"):
            LightGlueMatcher(features="orb")

    def test_name_attribute(self, monkeypatch):
        """Test that the matcher name reflects the feature extractor."""
        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", lambda: None
        )

        from tracksync.fine_align import LightGlueMatcher

        matcher = LightGlueMatcher(features="aliked")
        assert matcher.name == "aliked-lightglue"

    def test_superpoint_license_warning(self, monkeypatch):
        """Test that selecting SuperPoint emits the license caveat warning."""
        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", lambda: None
        )

        from tracksync.fine_align import LightGlueMatcher

        with pytest.warns(UserWarning, match="research-only license"):
            matcher = LightGlueMatcher(features="superpoint")
        assert matcher.name == "superpoint-lightglue"

    def test_aliked_no_license_warning(self, monkeypatch, recwarn):
        """Test that the default ALIKED extractor emits no warning."""
        monkeypatch.setattr(
            "tracksync.scene_deps.require_scene_deps", lambda: None
        )

        from tracksync.fine_align import LightGlueMatcher

        LightGlueMatcher(features="aliked")
        assert len(recwarn) == 0


# The following tests require torch/lightglue and should only run when marked
# as slow and when the scene dependencies are actually available


def _lightglue_available():
    """Check if torch and lightglue are available."""
    try:
        import lightglue  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_textured_image(rng: np.random.Generator, size: int = 480) -> np.ndarray:
    """Create a synthetic textured image with strong local structure.

    Blends smooth low-frequency blobs with a jittered checkerboard and random
    high-contrast rectangles so that feature extractors find plenty of
    distinctive keypoints.
    """
    import cv2

    # Low-frequency background: upsampled smooth noise
    coarse = rng.integers(0, 256, (size // 16, size // 16), dtype=np.uint8)
    background = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)

    # High-frequency detail: fine noise
    detail = rng.integers(0, 256, (size, size), dtype=np.uint8)

    gray = cv2.addWeighted(background, 0.6, detail, 0.4, 0)

    # Add random high-contrast rectangles for corner-like structure
    for _ in range(40):
        x0, y0 = rng.integers(0, size - 40, size=2)
        w, h = rng.integers(10, 40, size=2)
        val = int(rng.integers(0, 256))
        cv2.rectangle(gray, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)),
                      val, thickness=-1)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@pytest.mark.slow
@pytest.mark.skipif(not _lightglue_available(),
                    reason="torch/lightglue not available")
class TestLightGlueMatcherSlow:
    """Slow tests that run the real ALIKED+LightGlue models."""

    def test_factory_returns_matcher(self):
        """Test that factory returns LightGlueMatcher instances."""
        from tracksync.fine_align import LightGlueMatcher, make_matcher

        matcher = make_matcher("aliked-lightglue")
        assert isinstance(matcher, LightGlueMatcher)
        assert matcher.name == "aliked-lightglue"

    def test_matches_satisfy_known_homography(self):
        """Matched points must satisfy a known homography within 2 px.

        Renders a textured image, warps it with a known homography, matches
        the pair, and asserts that > 70% of matches transfer within 2 px.
        """
        import cv2

        from tracksync.fine_align import make_matcher

        rng = np.random.default_rng(42)
        img_a = _make_textured_image(rng)
        h, w = img_a.shape[:2]

        # Known homography: mild rotation + scale + translation + perspective
        angle_rad = np.deg2rad(5.0)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        scale = 0.95
        H = np.array([
            [scale * c, -scale * s, 20.0],
            [scale * s, scale * c, -10.0],
            [1e-5, 2e-5, 1.0],
        ], dtype=np.float64)

        img_b = cv2.warpPerspective(img_a, H, (w, h))

        matcher = make_matcher("aliked-lightglue")
        pts_a, pts_b = matcher.match(img_a, img_b)

        assert pts_a.shape == pts_b.shape
        assert pts_a.ndim == 2 and pts_a.shape[1] == 2
        assert len(pts_a) >= 50, f"Too few matches: {len(pts_a)}"

        # Transfer pts_a through H and compare with pts_b
        pts_a_h = cv2.perspectiveTransform(
            pts_a.reshape(-1, 1, 2).astype(np.float64), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(pts_a_h - pts_b.astype(np.float64), axis=1)

        inlier_frac = float(np.mean(errors < 2.0))
        assert inlier_frac > 0.70, (
            f"Only {inlier_frac:.1%} of {len(pts_a)} matches satisfy the "
            f"homography within 2 px (median error {np.median(errors):.2f} px)"
        )

    def test_masked_region_yields_no_keypoints(self):
        """Keypoints inside the static mask must be excluded from matches."""
        from tracksync.fine_align import make_matcher

        rng = np.random.default_rng(7)
        img = _make_textured_image(rng)
        h, w = img.shape[:2]

        # Mask the bottom third of both images (simulated hood region)
        mask = np.zeros((h, w), dtype=bool)
        mask[2 * h // 3:, :] = True

        matcher = make_matcher("aliked-lightglue")
        # Self-match with the mask applied to both sides
        pts_a, pts_b = matcher.match(img, img, mask_a=mask, mask_b=mask)

        assert len(pts_a) > 0, "Expected matches in the unmasked region"

        # No matched keypoint may fall inside the masked region
        assert np.all(pts_a[:, 1] < 2 * h / 3), (
            "Matched keypoints found inside masked region of image A"
        )
        assert np.all(pts_b[:, 1] < 2 * h / 3), (
            "Matched keypoints found inside masked region of image B"
        )

    def test_no_matches_returns_empty_arrays(self):
        """Fully masked images must return empty [0, 2] arrays."""
        from tracksync.fine_align import make_matcher

        rng = np.random.default_rng(3)
        img = _make_textured_image(rng, size=256)
        h, w = img.shape[:2]
        full_mask = np.ones((h, w), dtype=bool)

        matcher = make_matcher("aliked-lightglue")
        pts_a, pts_b = matcher.match(img, img, mask_a=full_mask, mask_b=full_mask)

        assert pts_a.shape == (0, 2)
        assert pts_b.shape == (0, 2)
