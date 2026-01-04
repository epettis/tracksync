"""Tests for segment validation and clamping."""

import pytest

from tracksync.models import Segment
from tracksync.segment_validator import (
    ClampedSegment,
    clamp_segments,
    build_processed_segments_with_clamping,
)


class TestClampSegments:
    def test_no_clamping_needed(self):
        """Segments within video duration are unchanged."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 10.0, 80.0),
            Segment("Finish", 20.0, 100.0),
        ]
        clamped, warnings = clamp_segments(segments, video_duration=60.0)

        assert len(clamped) == 3
        assert len(warnings) == 0
        assert all(not c.was_clamped for c in clamped)
        assert all(not c.is_beyond_video for c in clamped)
        assert clamped[0].clamped_timestamp == 0.0
        assert clamped[1].clamped_timestamp == 10.0
        assert clamped[2].clamped_timestamp == 20.0

    def test_clamp_end_time_to_duration(self):
        """Timestamps exceeding duration are clamped."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 50.0, 80.0),
            Segment("Finish", 70.0, 100.0),  # Exceeds 60s
        ]
        clamped, warnings = clamp_segments(segments, video_duration=60.0)

        assert len(clamped) == 3
        assert len(warnings) == 1
        assert clamped[0].clamped_timestamp == 0.0
        assert clamped[1].clamped_timestamp == 50.0
        assert clamped[2].clamped_timestamp == 60.0
        assert clamped[2].was_clamped is True

    def test_segment_entirely_beyond_video(self):
        """Segment with start >= duration is marked as beyond video."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("T1", 30.0, 80.0),
            Segment("T2", 65.0, 90.0),  # Beyond 60s
            Segment("Finish", 75.0, 100.0),  # Beyond 60s
        ]
        clamped, warnings = clamp_segments(segments, video_duration=60.0)

        assert len(warnings) == 2
        assert clamped[2].is_beyond_video is True
        assert clamped[3].is_beyond_video is True
        assert clamped[2].clamped_timestamp == 60.0
        assert clamped[3].clamped_timestamp == 60.0

    def test_clamping_produces_warnings(self):
        """Clamping generates appropriate warning messages."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("Late", 100.0, 80.0),
        ]
        _, warnings = clamp_segments(segments, video_duration=60.0)

        assert len(warnings) == 1
        assert "Late" in warnings[0]
        assert "100.00s" in warnings[0]
        assert "60.00s" in warnings[0]

    def test_exact_duration_is_clamped(self):
        """Timestamp exactly at duration is clamped (>= comparison)."""
        segments = [
            Segment("Start", 0.0, 100.0),
            Segment("End", 60.0, 100.0),  # Exactly at duration
        ]
        clamped, warnings = clamp_segments(segments, video_duration=60.0)

        assert clamped[1].was_clamped is True
        assert clamped[1].is_beyond_video is True


class TestBuildProcessedSegmentsWithClamping:
    def test_normal_segments_no_freeze_frame(self):
        """Valid segments get is_freeze_frame=False."""
        target_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 10.0, 80.0), 10.0, False, False),
            ClampedSegment(Segment("End", 20.0, 100.0), 20.0, False, False),
        ]
        ref_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 15.0, 80.0), 15.0, False, False),
            ClampedSegment(Segment("End", 30.0, 100.0), 30.0, False, False),
        ]

        processed = build_processed_segments_with_clamping(
            target_segments, ref_segments, 60.0, 60.0
        )

        assert len(processed) == 2
        assert all(not p.is_freeze_frame for p in processed)
        assert processed[0].speed_ratio == pytest.approx(10.0 / 15.0)
        assert processed[1].speed_ratio == pytest.approx(10.0 / 15.0)

    def test_beyond_video_segment_creates_freeze_frame(self):
        """Segment beyond video creates freeze frame."""
        target_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 50.0, 80.0), 50.0, False, False),
            ClampedSegment(Segment("End", 70.0, 100.0), 60.0, True, True),  # Clamped
        ]
        ref_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 50.0, 80.0), 50.0, False, False),
            ClampedSegment(Segment("End", 70.0, 100.0), 70.0, False, False),
        ]

        processed = build_processed_segments_with_clamping(
            target_segments, ref_segments, 60.0, 70.0
        )

        assert len(processed) == 2
        # First segment is normal
        assert processed[0].is_freeze_frame is False
        # Second segment: target 50-60 (10s), ref 50-70 (20s)
        # Target has 10s of content, so it's a normal segment
        assert processed[1].is_freeze_frame is False

    def test_zero_duration_segment_creates_freeze_frame(self):
        """Zero-duration clamped segment creates freeze frame."""
        target_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 60.0, 80.0), 60.0, True, True),  # At limit
            ClampedSegment(Segment("End", 70.0, 100.0), 60.0, True, True),  # Clamped
        ]
        ref_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("T1", 60.0, 80.0), 60.0, False, False),
            ClampedSegment(Segment("End", 80.0, 100.0), 80.0, False, False),
        ]

        processed = build_processed_segments_with_clamping(
            target_segments, ref_segments, 60.0, 80.0
        )

        assert len(processed) == 2
        # Second segment has zero target duration (60-60)
        assert processed[1].is_freeze_frame is True
        assert processed[1].freeze_frame_duration == 20.0  # ref 60-80

    def test_freeze_frame_duration_matches_reference(self):
        """Freeze frame duration matches reference segment timing."""
        target_segments = [
            ClampedSegment(Segment("Start", 55.0, 100.0), 55.0, False, False),
            ClampedSegment(Segment("End", 70.0, 100.0), 60.0, True, True),  # Clamped
        ]
        ref_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("End", 25.0, 100.0), 25.0, False, False),
        ]

        processed = build_processed_segments_with_clamping(
            target_segments, ref_segments, 60.0, 60.0
        )

        # Target: 55-60 = 5s, Reference: 0-25 = 25s
        # Not a freeze frame because target has positive duration
        assert processed[0].is_freeze_frame is False

    def test_segment_count_mismatch_raises(self):
        """Mismatched segment counts raise ValueError."""
        target = [ClampedSegment(Segment("A", 0.0, 100.0), 0.0, False, False)]
        ref = [
            ClampedSegment(Segment("A", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("B", 10.0, 100.0), 10.0, False, False),
        ]

        with pytest.raises(ValueError, match="Segment count mismatch"):
            build_processed_segments_with_clamping(target, ref, 60.0, 60.0)

    def test_reference_zero_duration_uses_ratio_one(self):
        """Zero reference duration uses ratio 1.0."""
        target_segments = [
            ClampedSegment(Segment("Start", 0.0, 100.0), 0.0, False, False),
            ClampedSegment(Segment("End", 10.0, 100.0), 10.0, False, False),
        ]
        ref_segments = [
            ClampedSegment(Segment("Start", 50.0, 100.0), 50.0, False, False),
            ClampedSegment(Segment("End", 60.0, 100.0), 50.0, True, True),  # Zero dur
        ]

        processed = build_processed_segments_with_clamping(
            target_segments, ref_segments, 60.0, 50.0
        )

        # Reference has zero duration, so use ratio 1.0
        assert processed[0].speed_ratio == 1.0
        assert processed[0].is_freeze_frame is False
