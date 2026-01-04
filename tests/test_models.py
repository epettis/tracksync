"""Tests for data models."""

import pytest

from tracksync.models import ProcessedSegment, Segment, VideoMetadata


class TestSegment:
    def test_segment_creation(self):
        seg = Segment(title="T1", timestamp=10.5, speed=85.0)
        assert seg.title == "T1"
        assert seg.timestamp == 10.5
        assert seg.speed == 85.0

    def test_segment_immutable(self):
        seg = Segment(title="T1", timestamp=10.5, speed=85.0)
        with pytest.raises(AttributeError):
            seg.timestamp = 20.0

    def test_segment_equality(self):
        seg1 = Segment(title="T1", timestamp=10.0, speed=80.0)
        seg2 = Segment(title="T1", timestamp=10.0, speed=80.0)
        assert seg1 == seg2


class TestVideoMetadata:
    def test_video_creation(self):
        video = VideoMetadata(driver="Eddie", segments=[])
        assert video.driver == "Eddie"
        assert video.segments == []

    def test_video_creation_default_segments(self):
        video = VideoMetadata(driver="Eddie")
        assert video.segments == []

    def test_add_segment(self):
        video = VideoMetadata(driver="Eddie")
        seg = Segment("T1", 10.0, 80.0)
        video.add_segment(seg)
        assert len(video.segments) == 1
        assert video.segments[0] == seg

    def test_add_multiple_segments(self):
        video = VideoMetadata(driver="Eddie")
        video.add_segment(Segment("Start", 0.0, 100.0))
        video.add_segment(Segment("T1", 10.0, 80.0))
        video.add_segment(Segment("Finish", 20.0, 100.0))
        assert len(video.segments) == 3


class TestProcessedSegment:
    def test_processed_segment_creation(self):
        ps = ProcessedSegment(start_time=0.0, end_time=10.0, speed_ratio=1.5)
        assert ps.start_time == 0.0
        assert ps.end_time == 10.0
        assert ps.speed_ratio == 1.5

    def test_processed_segment_immutable(self):
        ps = ProcessedSegment(start_time=0.0, end_time=10.0, speed_ratio=1.5)
        with pytest.raises(AttributeError):
            ps.speed_ratio = 2.0
