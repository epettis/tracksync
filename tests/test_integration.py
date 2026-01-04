"""Integration tests using synthetic clips."""

import pytest

from tracksync.models import Segment, VideoMetadata
from tracksync.speed_calculator import build_processed_segments, calculate_speed_ratios
from tracksync.video_processor import SyntheticClipFactory, VideoProcessor


class TestEndToEndWithSyntheticClips:
    def test_full_pipeline_with_synthetic_videos(self):
        """
        Test the complete pipeline from segments to output video.
        Uses synthetic clips instead of real video files.
        """
        # Create test data
        target = VideoMetadata(
            driver="Target",
            segments=[
                Segment("Start", 0.0, 100.0),
                Segment("T1", 10.0, 80.0),
                Segment("Finish", 20.0, 100.0),
            ],
        )

        reference = VideoMetadata(
            driver="Reference",
            segments=[
                Segment("Start", 0.0, 95.0),
                Segment("T1", 15.0, 75.0),  # ratio = 1.5
                Segment("Finish", 30.0, 95.0),  # ratio = 1.5
            ],
        )

        # Calculate speed ratios
        ratios = calculate_speed_ratios(target.segments, reference.segments)
        assert ratios == [pytest.approx(1.5), pytest.approx(1.5)]

        # Build processed segments
        processed = build_processed_segments(target.segments, ratios)
        assert len(processed) == 2
        assert processed[0].start_time == 0.0
        assert processed[0].end_time == 10.0
        assert processed[0].speed_ratio == pytest.approx(1.5)

        # Create mock video loader
        def mock_loader(filepath):
            return SyntheticClipFactory.create_color_clip(duration=60.0, size=(320, 240))

        processor = VideoProcessor(clip_loader=mock_loader)

        # Process video segments
        clip = mock_loader("target.mp4")
        result = processor.process_segments(clip, processed)

        # Original segments: 10s + 10s = 20s
        # After speedup: 10/1.5 + 10/1.5 = 6.67 + 6.67 = 13.33s
        expected = (10 / 1.5) + (10 / 1.5)
        assert result.duration == pytest.approx(expected, abs=0.5)

    def test_stacked_output_dimensions(self):
        """Test that final output has correct dimensions."""
        processor = VideoProcessor()

        top = SyntheticClipFactory.create_color_clip(5.0, (640, 480))
        bottom = SyntheticClipFactory.create_color_clip(5.0, (640, 480))

        stacked = processor.stack_vertically(top, bottom)

        assert stacked.size == (640, 960)
        assert stacked.duration == 5.0

    def test_complete_comparison_workflow(self):
        """Test the complete workflow of generating a comparison."""
        # Setup test data with known timing differences
        target = VideoMetadata(
            driver="FastDriver",
            segments=[
                Segment("Start", 0.0, 100.0),
                Segment("Turn1", 8.0, 85.0),  # 8 seconds
                Segment("Turn2", 18.0, 90.0),  # 10 seconds
                Segment("Finish", 28.0, 100.0),  # 10 seconds
            ],
        )

        reference = VideoMetadata(
            driver="SlowDriver",
            segments=[
                Segment("Start", 0.0, 95.0),
                Segment("Turn1", 12.0, 80.0),  # 12 seconds (ratio = 1.5)
                Segment("Turn2", 22.0, 85.0),  # 10 seconds (ratio = 1.0)
                Segment("Finish", 37.0, 95.0),  # 15 seconds (ratio = 1.5)
            ],
        )

        # Calculate ratios
        ratios = calculate_speed_ratios(target.segments, reference.segments)

        # Verify ratios
        assert ratios[0] == pytest.approx(12.0 / 8.0)  # 1.5
        assert ratios[1] == pytest.approx(10.0 / 10.0)  # 1.0
        assert ratios[2] == pytest.approx(15.0 / 10.0)  # 1.5

        # Build processed segments
        processed = build_processed_segments(target.segments, ratios)

        # Create processor with mock loader
        def mock_loader(filepath):
            return SyntheticClipFactory.create_color_clip(duration=60.0, size=(640, 480))

        processor = VideoProcessor(clip_loader=mock_loader)

        # Load and process clips
        target_clip = processor.load_video("FastDriver.mp4")
        reference_clip = processor.load_video("SlowDriver.mp4")

        processed_target = processor.process_segments(target_clip, processed)
        processed_reference = processor.extract_reference_segment(
            reference_clip,
            reference.segments[0].timestamp,
            reference.segments[-1].timestamp,
        )

        # Stack them
        stacked = processor.stack_vertically(processed_target, processed_reference)

        # Verify dimensions
        assert stacked.size == (640, 960)

        # Verify target processing duration
        # Original: 8 + 10 + 10 = 28s
        # After scaling: 8/1.5 + 10/1.0 + 10/1.5 = 5.33 + 10 + 6.67 = 22s
        expected_target_duration = (8 / 1.5) + (10 / 1.0) + (10 / 1.5)
        assert processed_target.duration == pytest.approx(expected_target_duration, abs=0.5)

        # Verify reference duration (37 - 0 = 37s)
        assert processed_reference.duration == pytest.approx(37.0, abs=0.5)

    def test_symmetric_comparison(self):
        """Test that A vs B and B vs A produce different but valid results."""
        driver_a = VideoMetadata(
            driver="A",
            segments=[
                Segment("Start", 0.0, 100.0),
                Segment("End", 10.0, 100.0),
            ],
        )

        driver_b = VideoMetadata(
            driver="B",
            segments=[
                Segment("Start", 0.0, 100.0),
                Segment("End", 15.0, 100.0),
            ],
        )

        # A as target, B as reference
        ratios_a_vs_b = calculate_speed_ratios(driver_a.segments, driver_b.segments)
        # B as target, A as reference
        ratios_b_vs_a = calculate_speed_ratios(driver_b.segments, driver_a.segments)

        # Ratios should be reciprocals
        assert ratios_a_vs_b[0] == pytest.approx(1.5)  # 15/10
        assert ratios_b_vs_a[0] == pytest.approx(1 / 1.5)  # 10/15
