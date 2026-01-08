"""Tests for the autocorrelation module."""

import numpy as np
import pytest
from tracksync.autocorrelation import (
    binarize_frame,
    FrameOCRData,
    StartFinishCrossing,
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


class TestFrameOCRData:
    """Tests for FrameOCRData dataclass."""

    def test_default_values(self):
        """FrameOCRData should have correct default values."""
        ocr = FrameOCRData()
        assert ocr.lap_number is None
        assert ocr.is_optimal_lap is False
        assert ocr.segment_number is None
        assert ocr.lap_time_seconds is None

    def test_with_values(self):
        """FrameOCRData should store all fields."""
        ocr = FrameOCRData(
            lap_number=3,
            is_optimal_lap=False,
            segment_number=5,
            lap_time_seconds=65.5
        )
        assert ocr.lap_number == 3
        assert ocr.is_optimal_lap is False
        assert ocr.segment_number == 5
        assert ocr.lap_time_seconds == 65.5

    def test_optimal_lap(self):
        """FrameOCRData should handle optimal lap flag."""
        ocr = FrameOCRData(is_optimal_lap=True)
        assert ocr.is_optimal_lap is True
        assert ocr.lap_number is None


class TestStartFinishCrossing:
    """Tests for StartFinishCrossing dataclass."""

    def test_crossing(self):
        """StartFinishCrossing should store all fields."""
        crossing = StartFinishCrossing(
            frame_index=100,
            video_time=3.33,
            lap_before=1,
            lap_after=2
        )
        assert crossing.frame_index == 100
        assert crossing.video_time == 3.33
        assert crossing.lap_before == 1
        assert crossing.lap_after == 2
