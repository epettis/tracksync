"""Tests for video processor using synthetic clips."""

import pytest
import numpy as np

from tracksync.models import ProcessedSegment
from tracksync.video_processor import SyntheticClipFactory, VideoProcessor


class TestVideoProcessor:
    def test_extract_segment(self, synthetic_clip):
        """Test that segment extraction works correctly."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        extracted = processor.extract_segment(clip, 2.0, 5.0)

        assert extracted.duration == pytest.approx(3.0, abs=0.1)

    def test_extract_segment_from_start(self, synthetic_clip):
        """Test extracting segment from the beginning."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        extracted = processor.extract_segment(clip, 0.0, 3.0)

        assert extracted.duration == pytest.approx(3.0, abs=0.1)

    def test_apply_speed_doubles_speed(self, synthetic_clip):
        """Test that speedx(2) halves the duration."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        sped_up = processor.apply_speed(clip, 2.0)

        assert sped_up.duration == pytest.approx(5.0, abs=0.1)

    def test_apply_speed_halves_speed(self, synthetic_clip):
        """Test that speedx(0.5) doubles the duration."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        slowed = processor.apply_speed(clip, 0.5)

        assert slowed.duration == pytest.approx(20.0, abs=0.1)

    def test_apply_speed_no_change(self, synthetic_clip):
        """Test that speedx(1.0) keeps duration unchanged."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        unchanged = processor.apply_speed(clip, 1.0)

        assert unchanged.duration == pytest.approx(10.0, abs=0.1)

    def test_process_segments(self, synthetic_clip):
        """Test processing multiple segments."""
        clip = synthetic_clip(duration=30.0)
        processor = VideoProcessor()

        segments = [
            ProcessedSegment(start_time=0.0, end_time=10.0, speed_ratio=1.5),
            ProcessedSegment(start_time=10.0, end_time=20.0, speed_ratio=1.0),
            ProcessedSegment(start_time=20.0, end_time=30.0, speed_ratio=0.5),
        ]

        result = processor.process_segments(clip, segments)

        # Expected durations: 10/1.5 + 10/1.0 + 10/0.5 = 6.67 + 10 + 20 = 36.67
        expected_duration = (10 / 1.5) + (10 / 1.0) + (10 / 0.5)
        assert result.duration == pytest.approx(expected_duration, abs=0.5)

    def test_process_single_segment(self, synthetic_clip):
        """Test processing a single segment."""
        clip = synthetic_clip(duration=20.0)
        processor = VideoProcessor()

        segments = [
            ProcessedSegment(start_time=5.0, end_time=15.0, speed_ratio=2.0),
        ]

        result = processor.process_segments(clip, segments)

        # 10 second segment at 2x speed = 5 seconds
        assert result.duration == pytest.approx(5.0, abs=0.5)

    def test_stack_vertically(self, color_clip):
        """Test vertical stacking produces correct dimensions."""
        top = color_clip(duration=5.0, size=(320, 240), color=(255, 0, 0))
        bottom = color_clip(duration=5.0, size=(320, 240), color=(0, 0, 255))
        processor = VideoProcessor()

        stacked = processor.stack_vertically(top, bottom)

        assert stacked.size == (320, 480)  # Same width, double height

    def test_stack_vertically_preserves_duration(self, color_clip):
        """Test that stacking preserves the clip duration."""
        top = color_clip(duration=5.0, size=(320, 240))
        bottom = color_clip(duration=5.0, size=(320, 240))
        processor = VideoProcessor()

        stacked = processor.stack_vertically(top, bottom)

        assert stacked.duration == pytest.approx(5.0, abs=0.1)

    def test_dependency_injection(self):
        """Test that custom clip loader can be injected."""
        mock_clips = {}

        def mock_loader(filepath):
            if filepath not in mock_clips:
                mock_clips[filepath] = SyntheticClipFactory.create_color_clip(5.0)
            return mock_clips[filepath]

        processor = VideoProcessor(clip_loader=mock_loader)
        clip = processor.load_video("fake/path.mp4")

        assert clip.duration == 5.0
        assert "fake/path.mp4" in mock_clips

    def test_extract_reference_segment(self, synthetic_clip):
        """Test extracting reference segment."""
        clip = synthetic_clip(duration=60.0)
        processor = VideoProcessor()

        reference = processor.extract_reference_segment(clip, 10.0, 50.0)

        assert reference.duration == pytest.approx(40.0, abs=0.1)


class TestSyntheticClipFactory:
    def test_create_color_clip(self):
        """Test creating a color clip."""
        clip = SyntheticClipFactory.create_color_clip(
            duration=5.0, size=(640, 480), color=(0, 255, 0)
        )

        assert clip.duration == 5.0
        assert clip.size == (640, 480)

    def test_create_color_clip_default_values(self):
        """Test creating a color clip with default values."""
        clip = SyntheticClipFactory.create_color_clip(duration=3.0)

        assert clip.duration == 3.0
        assert clip.size == (320, 240)

    def test_create_gradient_clip(self):
        """Test creating a gradient clip."""
        clip = SyntheticClipFactory.create_gradient_clip(
            duration=3.0, size=(320, 240), fps=30
        )

        assert clip.duration == 3.0
        assert clip.fps == 30

    def test_gradient_clip_changes_over_time(self):
        """Test that gradient clip frames are different at different times."""
        clip = SyntheticClipFactory.create_gradient_clip(duration=3.0)

        frame_start = clip.get_frame(0)
        frame_end = clip.get_frame(2.9)

        # Red channel should be different (0 at start, ~255 at end)
        assert not np.array_equal(frame_start, frame_end)
        assert frame_start[0, 0, 0] < frame_end[0, 0, 0]  # Red increases


class TestFreezeFrame:
    def test_create_freeze_frame_basic(self, synthetic_clip):
        """Test creating a basic freeze frame."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        freeze = processor.create_freeze_frame(clip, frame_time=5.0, duration=3.0)

        assert freeze.duration == pytest.approx(3.0, abs=0.1)

    def test_create_freeze_frame_at_start(self, synthetic_clip):
        """Test freeze frame from beginning of clip."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        freeze = processor.create_freeze_frame(clip, frame_time=0.0, duration=2.0)

        assert freeze.duration == pytest.approx(2.0, abs=0.1)

    def test_create_freeze_frame_near_end(self, synthetic_clip):
        """Test freeze frame from near end of clip is clamped safely."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        # Request frame beyond clip end - should be clamped
        freeze = processor.create_freeze_frame(clip, frame_time=15.0, duration=5.0)

        assert freeze.duration == pytest.approx(5.0, abs=0.1)

    def test_create_freeze_frame_preserves_frame_content(self):
        """Test that freeze frame captures the correct frame."""
        clip = SyntheticClipFactory.create_gradient_clip(duration=10.0)
        processor = VideoProcessor()

        # Get frame at t=5 (halfway, red should be ~128)
        freeze = processor.create_freeze_frame(clip, frame_time=5.0, duration=2.0)

        # Check that the frozen frame has the expected color value
        frame = freeze.get_frame(0)
        # At t=5 of 10s, red should be around 127-128
        assert 100 < frame[0, 0, 0] < 160

    def test_process_segments_with_freeze_frame(self, synthetic_clip):
        """Test processing a mix of normal and freeze frame segments."""
        clip = synthetic_clip(duration=30.0)
        processor = VideoProcessor()

        segments = [
            ProcessedSegment(
                start_time=0.0, end_time=10.0, speed_ratio=1.0, is_freeze_frame=False
            ),
            ProcessedSegment(
                start_time=10.0,
                end_time=10.0,  # Zero duration
                speed_ratio=0.0,
                is_freeze_frame=True,
                freeze_frame_time=9.99,
                freeze_frame_duration=5.0,
            ),
            ProcessedSegment(
                start_time=10.0, end_time=20.0, speed_ratio=1.0, is_freeze_frame=False
            ),
        ]

        result = processor.process_segments(clip, segments)

        # Expected: 10s normal + 5s freeze + 10s normal = 25s
        assert result.duration == pytest.approx(25.0, abs=0.5)

    def test_process_segments_all_freeze_frames(self, synthetic_clip):
        """Test processing only freeze frames."""
        clip = synthetic_clip(duration=10.0)
        processor = VideoProcessor()

        segments = [
            ProcessedSegment(
                start_time=0.0,
                end_time=0.0,
                speed_ratio=0.0,
                is_freeze_frame=True,
                freeze_frame_time=0.0,
                freeze_frame_duration=3.0,
            ),
            ProcessedSegment(
                start_time=0.0,
                end_time=0.0,
                speed_ratio=0.0,
                is_freeze_frame=True,
                freeze_frame_time=5.0,
                freeze_frame_duration=4.0,
            ),
        ]

        result = processor.process_segments(clip, segments)

        # Expected: 3s + 4s = 7s of freeze frames
        assert result.duration == pytest.approx(7.0, abs=0.5)

    def test_freeze_frame_has_fps(self, synthetic_clip):
        """Test that freeze frame inherits fps from source clip."""
        clip = SyntheticClipFactory.create_color_clip(duration=5.0, fps=30)
        processor = VideoProcessor()

        freeze = processor.create_freeze_frame(clip, frame_time=2.0, duration=3.0)

        assert freeze.fps == 30
