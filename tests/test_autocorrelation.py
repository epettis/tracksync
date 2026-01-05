"""Tests for the autocorrelation module."""

import numpy as np
import pytest
from tracksync.autocorrelation import (
    binarize_frame,
    compute_correlation,
    CorrelationResult,
    AlignmentPoint,
)


class TestBinarizeFrame:
    """Tests for the binarize_frame function."""

    def test_all_white_frame(self):
        """All-white frame should become all 1s."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        binary = binarize_frame(frame)
        assert np.all(binary == 1)

    def test_all_black_frame(self):
        """All-black frame should become all 0s."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        binary = binarize_frame(frame)
        assert np.all(binary == 0)

    def test_white_threshold_boundary(self):
        """Pixels at exactly threshold should become 1."""
        frame = np.full((100, 100, 3), 240, dtype=np.uint8)
        binary = binarize_frame(frame, white_threshold=240)
        assert np.all(binary == 1)

    def test_white_below_threshold(self):
        """Pixels just below threshold should become 0."""
        frame = np.full((100, 100, 3), 239, dtype=np.uint8)
        binary = binarize_frame(frame, white_threshold=240)
        assert np.all(binary == 0)

    def test_red_detection(self):
        """Red pixels (R=220, G=50, B=50) should become 1."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # R
        frame[:, :, 1] = 50   # G
        frame[:, :, 2] = 50   # B
        binary = binarize_frame(frame)
        assert np.all(binary == 1)

    def test_red_boundary_r_min(self):
        """Red at exactly red_min threshold should be detected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # R at threshold
        frame[:, :, 1] = 50   # G
        frame[:, :, 2] = 50   # B
        binary = binarize_frame(frame, red_min=200)
        assert np.all(binary == 1)

    def test_red_below_r_min(self):
        """Red below red_min threshold should not be detected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 199  # R below threshold
        frame[:, :, 1] = 50   # G
        frame[:, :, 2] = 50   # B
        binary = binarize_frame(frame, red_min=200)
        assert np.all(binary == 0)

    def test_orange_not_detected(self):
        """Orange pixels (R=220, G=150, B=50) should NOT be detected (G too high)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # R
        frame[:, :, 1] = 150  # G too high
        frame[:, :, 2] = 50   # B
        binary = binarize_frame(frame)
        assert np.all(binary == 0)

    def test_pink_not_detected(self):
        """Pink pixels (R=220, G=100, B=100) should NOT be detected (G and B too high)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # R
        frame[:, :, 1] = 100  # G too high
        frame[:, :, 2] = 100  # B too high
        binary = binarize_frame(frame)
        assert np.all(binary == 0)

    def test_red_boundary_gb_max(self):
        """Red at exactly red_max_gb threshold should be detected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # R
        frame[:, :, 1] = 80   # G at threshold
        frame[:, :, 2] = 80   # B at threshold
        binary = binarize_frame(frame, red_max_gb=80)
        assert np.all(binary == 1)

    def test_red_above_gb_max(self):
        """Red with G or B above red_max_gb should not be detected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # R
        frame[:, :, 1] = 81   # G above threshold
        frame[:, :, 2] = 50   # B
        binary = binarize_frame(frame, red_max_gb=80)
        assert np.all(binary == 0)

    def test_mixed_colors(self):
        """Mixed frame should correctly detect white and red regions."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Top-left quadrant: white
        frame[:50, :50, :] = 255

        # Top-right quadrant: red
        frame[:50, 50:, 0] = 220
        frame[:50, 50:, 1] = 50
        frame[:50, 50:, 2] = 50

        # Bottom-left quadrant: blue (should not be detected)
        frame[50:, :50, 0] = 50
        frame[50:, :50, 1] = 50
        frame[50:, :50, 2] = 220

        # Bottom-right quadrant: black
        # Already zeros

        binary = binarize_frame(frame)

        assert np.all(binary[:50, :50] == 1)   # White detected
        assert np.all(binary[:50, 50:] == 1)   # Red detected
        assert np.all(binary[50:, :50] == 0)   # Blue not detected
        assert np.all(binary[50:, 50:] == 0)   # Black not detected

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, :] = 200  # Gray

        # With default threshold (240), this should be 0
        binary_default = binarize_frame(frame, white_threshold=240)
        assert np.all(binary_default == 0)

        # With lower threshold (200), this should be 1
        binary_custom = binarize_frame(frame, white_threshold=200)
        assert np.all(binary_custom == 1)


class TestComputeCorrelation:
    """Tests for the compute_correlation function."""

    def test_identical_frames(self):
        """Identical frames should have maximum correlation."""
        frame = np.ones((100, 100), dtype=np.uint8)
        score = compute_correlation(frame, frame)
        assert score == 100 * 100  # All pixels match

    def test_inverted_frames(self):
        """Completely different frames should have zero correlation."""
        frame_a = np.ones((100, 100), dtype=np.uint8)
        frame_b = np.zeros((100, 100), dtype=np.uint8)
        score = compute_correlation(frame_a, frame_b)
        assert score == 0

    def test_partial_overlap(self):
        """Partial overlap should give proportional correlation."""
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.zeros((100, 100), dtype=np.uint8)

        # Set top half of both frames to 1
        frame_a[:50, :] = 1
        frame_b[:50, :] = 1

        score = compute_correlation(frame_a, frame_b)
        assert score == 50 * 100  # Only top half matches

    def test_no_overlap(self):
        """Non-overlapping regions should give zero correlation."""
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.zeros((100, 100), dtype=np.uint8)

        # Set top half of A to 1, bottom half of B to 1
        frame_a[:50, :] = 1
        frame_b[50:, :] = 1

        score = compute_correlation(frame_a, frame_b)
        assert score == 0

    def test_single_pixel_match(self):
        """Single matching pixel should give correlation of 1."""
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.zeros((100, 100), dtype=np.uint8)

        frame_a[50, 50] = 1
        frame_b[50, 50] = 1

        score = compute_correlation(frame_a, frame_b)
        assert score == 1

    def test_returns_integer(self):
        """Correlation should return an integer."""
        frame = np.ones((100, 100), dtype=np.uint8)
        score = compute_correlation(frame, frame)
        assert isinstance(score, int)


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_correlation_result(self):
        """CorrelationResult should store all fields."""
        result = CorrelationResult(
            best_time=1.5,
            best_score=1000,
            all_scores=[(1.0, 800), (1.5, 1000), (2.0, 900)]
        )
        assert result.best_time == 1.5
        assert result.best_score == 1000
        assert len(result.all_scores) == 3

    def test_alignment_point(self):
        """AlignmentPoint should store all fields."""
        point = AlignmentPoint(
            time_a=10.0,
            time_b=12.5,
            correlation=5000
        )
        assert point.time_a == 10.0
        assert point.time_b == 12.5
        assert point.correlation == 5000
