"""Tests for speed ratio calculations."""

import pytest

from tracksync.models import Segment
from tracksync.speed_calculator import build_processed_segments, calculate_speed_ratios


class TestCalculateSpeedRatios:
    def test_equal_timing_gives_ratio_one(self):
        """When both videos have same segment durations, ratio is 1.0."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 10.0, 80.0),
            Segment("Finish", 20.0, 100.0),
        ]
        ratios = calculate_speed_ratios(segments, segments)
        assert ratios == [1.0, 1.0]

    def test_slower_target_gives_ratio_greater_than_one(self):
        """When target is slower, ratio > 1 to speed it up."""
        target = [
            Segment("Start", 0.0, 100.0),
            Segment("Finish", 10.0, 100.0),  # 10s duration
        ]
        reference = [
            Segment("Start", 0.0, 100.0),
            Segment("Finish", 15.0, 100.0),  # 15s duration
        ]
        ratios = calculate_speed_ratios(target, reference)
        assert ratios == [1.5]  # 15/10 = 1.5

    def test_faster_target_gives_ratio_less_than_one(self):
        """When target is faster, ratio < 1 to slow it down."""
        target = [
            Segment("Start", 0.0, 100.0),
            Segment("Finish", 20.0, 100.0),  # 20s duration
        ]
        reference = [
            Segment("Start", 0.0, 100.0),
            Segment("Finish", 10.0, 100.0),  # 10s duration
        ]
        ratios = calculate_speed_ratios(target, reference)
        assert ratios == [0.5]  # 10/20 = 0.5

    def test_multiple_segments(self, sample_segments, reference_segments):
        """Test with multiple segments."""
        ratios = calculate_speed_ratios(sample_segments, reference_segments)
        assert len(ratios) == 3
        assert ratios[0] == pytest.approx(1.5)  # 15/10
        assert ratios[1] == pytest.approx(1.0)  # 10/10
        assert ratios[2] == pytest.approx(1.5)  # 15/10

    def test_segment_count_mismatch_raises(self):
        """Mismatched segment counts should raise ValueError."""
        target = [Segment("Start", 0.0, 100.0), Segment("T1", 10.0, 80.0)]
        reference = [Segment("Start", 0.0, 100.0)]

        with pytest.raises(ValueError, match="Segment count mismatch"):
            calculate_speed_ratios(target, reference)

    def test_zero_duration_raises(self):
        """Zero or negative duration should raise ValueError."""
        target = [
            Segment("Start", 10.0, 100.0),
            Segment("T1", 10.0, 80.0),  # Same timestamp = 0 duration
        ]
        reference = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 10.0, 80.0),
        ]

        with pytest.raises(ValueError, match="Invalid target duration"):
            calculate_speed_ratios(target, reference)

    def test_negative_duration_raises(self):
        """Negative duration (timestamps out of order) should raise ValueError."""
        target = [
            Segment("Start", 20.0, 100.0),
            Segment("T1", 10.0, 80.0),  # Earlier timestamp = negative duration
        ]
        reference = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 10.0, 80.0),
        ]

        with pytest.raises(ValueError, match="Invalid target duration"):
            calculate_speed_ratios(target, reference)

    def test_empty_segments(self):
        """Empty segment lists should return empty ratios."""
        ratios = calculate_speed_ratios([], [])
        assert ratios == []

    def test_single_segment(self):
        """Single segment should return empty ratios (no intervals)."""
        target = [Segment("Start", 0.0, 100.0)]
        reference = [Segment("Start", 0.0, 100.0)]
        ratios = calculate_speed_ratios(target, reference)
        assert ratios == []


class TestBuildProcessedSegments:
    def test_builds_correct_segments(self):
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 10.0, 80.0),
            Segment("Finish", 20.0, 100.0),
        ]
        ratios = [1.5, 0.8]

        processed = build_processed_segments(segments, ratios)

        assert len(processed) == 2
        assert processed[0].start_time == 0.0
        assert processed[0].end_time == 10.0
        assert processed[0].speed_ratio == 1.5
        assert processed[1].start_time == 10.0
        assert processed[1].end_time == 20.0
        assert processed[1].speed_ratio == 0.8

    def test_empty_segments(self):
        """Empty segments should return empty processed list."""
        processed = build_processed_segments([], [])
        assert processed == []

    def test_single_segment(self):
        """Single segment returns empty processed list (no intervals)."""
        segments = [Segment("Start", 0.0, 100.0)]
        processed = build_processed_segments(segments, [])
        assert processed == []
