"""Tests for csv_writer module and CSV transitivity with csv_reader."""

import pytest
import tempfile
import os

from tracksync.models import VideoMetadata, Segment, SyncPoint
from tracksync.csv_writer import (
    format_csv,
    write_csv,
    write_sync_csv,
)
from tracksync.csv_reader import parse_csv_content, read_csv


class TestFormatCSV:
    """Tests for format_csv function."""

    def test_empty_list(self):
        """Empty list produces empty string."""
        result = format_csv([])
        assert result == ""

    def test_single_driver_simple_format(self):
        """Single driver produces valid CSV."""
        video = VideoMetadata(driver="Driver1", segments=[
            Segment(title="Start", timestamp=0.0, speed=1.0),
            Segment(title="End", timestamp=10.0, speed=1.0),
        ])
        result = format_csv([video])
        lines = result.strip().split("\n")
        assert lines[0] == "milestone,Driver1"
        assert lines[1] == "Start,0.000"
        assert lines[2] == "End,10.000"

    def test_multiple_drivers_simple_format(self):
        """Multiple drivers produce valid CSV."""
        video1 = VideoMetadata(driver="Alice", segments=[
            Segment(title="Start", timestamp=0.0, speed=1.0),
            Segment(title="Middle", timestamp=5.0, speed=1.0),
            Segment(title="End", timestamp=10.0, speed=1.0),
        ])
        video2 = VideoMetadata(driver="Bob", segments=[
            Segment(title="Start", timestamp=0.5, speed=1.0),
            Segment(title="Middle", timestamp=6.0, speed=1.0),
            Segment(title="End", timestamp=11.0, speed=1.0),
        ])
        result = format_csv([video1, video2])
        lines = result.strip().split("\n")
        assert lines[0] == "milestone,Alice,Bob"
        assert lines[1] == "Start,0.000,0.500"
        assert lines[2] == "Middle,5.000,6.000"
        assert lines[3] == "End,10.000,11.000"

    def test_segment_count_mismatch_raises(self):
        """Mismatched segment counts raise ValueError."""
        video1 = VideoMetadata(driver="A", segments=[
            Segment(title="Start", timestamp=0.0, speed=1.0),
        ])
        video2 = VideoMetadata(driver="B", segments=[
            Segment(title="Start", timestamp=0.0, speed=1.0),
            Segment(title="End", timestamp=10.0, speed=1.0),
        ])
        with pytest.raises(ValueError, match="same number of segments"):
            format_csv([video1, video2])


class TestCSVTransitivity:
    """Tests for write/read transitivity - data survives round-trip."""

    def test_single_driver_round_trip(self):
        """Single driver data survives write/read cycle."""
        original = [VideoMetadata(driver="Solo", segments=[
            Segment(title="Start", timestamp=0.0, speed=1.0),
            Segment(title="Middle", timestamp=5.5, speed=1.0),
            Segment(title="End", timestamp=10.123, speed=1.0),
        ])]

        csv_content = format_csv(original)
        restored = parse_csv_content(csv_content)

        assert len(restored) == 1
        assert restored[0].driver == "Solo"
        assert len(restored[0].segments) == 3
        assert restored[0].segments[0].title == "Start"
        assert restored[0].segments[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert restored[0].segments[1].title == "Middle"
        assert restored[0].segments[1].timestamp == pytest.approx(5.5, abs=0.001)
        assert restored[0].segments[2].title == "End"
        assert restored[0].segments[2].timestamp == pytest.approx(10.123, abs=0.001)

    def test_multiple_drivers_round_trip(self):
        """Multiple drivers data survives write/read cycle."""
        original = [
            VideoMetadata(driver="Alice", segments=[
                Segment(title="Start", timestamp=0.0, speed=1.0),
                Segment(title="T1", timestamp=10.5, speed=1.0),
                Segment(title="T2", timestamp=20.75, speed=1.0),
                Segment(title="Finish", timestamp=30.0, speed=1.0),
            ]),
            VideoMetadata(driver="Bob", segments=[
                Segment(title="Start", timestamp=0.5, speed=1.0),
                Segment(title="T1", timestamp=11.0, speed=1.0),
                Segment(title="T2", timestamp=21.25, speed=1.0),
                Segment(title="Finish", timestamp=31.5, speed=1.0),
            ]),
        ]

        csv_content = format_csv(original)
        restored = parse_csv_content(csv_content)

        assert len(restored) == 2
        assert restored[0].driver == "Alice"
        assert restored[1].driver == "Bob"

        # Verify Alice's segments
        assert len(restored[0].segments) == 4
        assert restored[0].segments[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert restored[0].segments[1].timestamp == pytest.approx(10.5, abs=0.001)
        assert restored[0].segments[2].timestamp == pytest.approx(20.75, abs=0.001)
        assert restored[0].segments[3].timestamp == pytest.approx(30.0, abs=0.001)

        # Verify Bob's segments
        assert len(restored[1].segments) == 4
        assert restored[1].segments[0].timestamp == pytest.approx(0.5, abs=0.001)
        assert restored[1].segments[1].timestamp == pytest.approx(11.0, abs=0.001)
        assert restored[1].segments[2].timestamp == pytest.approx(21.25, abs=0.001)
        assert restored[1].segments[3].timestamp == pytest.approx(31.5, abs=0.001)

    def test_three_drivers_round_trip(self):
        """Three drivers data survives write/read cycle."""
        original = [
            VideoMetadata(driver="A", segments=[
                Segment(title="Start", timestamp=0.0, speed=1.0),
                Segment(title="End", timestamp=10.0, speed=1.0),
            ]),
            VideoMetadata(driver="B", segments=[
                Segment(title="Start", timestamp=0.1, speed=1.0),
                Segment(title="End", timestamp=11.0, speed=1.0),
            ]),
            VideoMetadata(driver="C", segments=[
                Segment(title="Start", timestamp=0.2, speed=1.0),
                Segment(title="End", timestamp=12.0, speed=1.0),
            ]),
        ]

        csv_content = format_csv(original)
        restored = parse_csv_content(csv_content)

        assert len(restored) == 3
        assert [v.driver for v in restored] == ["A", "B", "C"]

    def test_file_round_trip(self):
        """Data survives write to file and read back."""
        original = [
            VideoMetadata(driver="FileTest1", segments=[
                Segment(title="Segment1", timestamp=1.234, speed=1.0),
                Segment(title="Segment2", timestamp=5.678, speed=1.0),
            ]),
            VideoMetadata(driver="FileTest2", segments=[
                Segment(title="Segment1", timestamp=2.345, speed=1.0),
                Segment(title="Segment2", timestamp=6.789, speed=1.0),
            ]),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            write_csv(original, temp_path)
            restored = read_csv(temp_path)

            assert len(restored) == 2
            assert restored[0].driver == "FileTest1"
            assert restored[1].driver == "FileTest2"
            assert restored[0].segments[0].timestamp == pytest.approx(1.234, abs=0.001)
            assert restored[1].segments[1].timestamp == pytest.approx(6.789, abs=0.001)
        finally:
            os.unlink(temp_path)

    def test_segment_titles_preserved(self):
        """Segment titles are preserved through round-trip."""
        original = [VideoMetadata(driver="Test", segments=[
            Segment(title="Start Line", timestamp=0.0, speed=1.0),
            Segment(title="Turn 1 Apex", timestamp=5.0, speed=1.0),
            Segment(title="Back Straight", timestamp=10.0, speed=1.0),
            Segment(title="Finish", timestamp=15.0, speed=1.0),
        ])]

        csv_content = format_csv(original)
        restored = parse_csv_content(csv_content)

        titles = [seg.title for seg in restored[0].segments]
        assert titles == ["Start Line", "Turn 1 Apex", "Back Straight", "Finish"]

    def test_precision_preserved(self):
        """Timestamp precision is preserved to 3 decimal places."""
        original = [VideoMetadata(driver="Precision", segments=[
            Segment(title="A", timestamp=0.001, speed=1.0),
            Segment(title="B", timestamp=99.999, speed=1.0),
            Segment(title="C", timestamp=123.456, speed=1.0),
        ])]

        csv_content = format_csv(original)
        restored = parse_csv_content(csv_content)

        assert restored[0].segments[0].timestamp == pytest.approx(0.001, abs=0.0001)
        assert restored[0].segments[1].timestamp == pytest.approx(99.999, abs=0.0001)
        assert restored[0].segments[2].timestamp == pytest.approx(123.456, abs=0.0001)


class TestWriteSyncCSV:
    """Tests for write_sync_csv function."""

    def test_basic_sync_points(self):
        """Basic sync points are written correctly."""
        sync_points = [
            SyncPoint(time_a=0.0, time_b=0.5, label="start", speed=1.0),
            SyncPoint(time_a=10.0, time_b=10.5, label="segment_1", speed=1.0),
            SyncPoint(time_a=20.0, time_b=21.0, label="end", speed=1.0),
        ]

        csv_content = write_sync_csv(sync_points, "Driver1", "Driver2")
        lines = csv_content.strip().split("\n")

        assert lines[0] == "milestone,Driver1,Driver2"
        assert lines[1] == "start,0.000,0.500"
        assert lines[2] == "segment_1,10.000,10.500"
        assert lines[3] == "end,20.000,21.000"

    def test_sync_csv_round_trip(self):
        """Sync CSV can be read back with csv_reader."""
        sync_points = [
            SyncPoint(time_a=0.0, time_b=1.0, label="start", speed=1.0),
            SyncPoint(time_a=30.0, time_b=31.5, label="finish", speed=1.0),
        ]

        csv_content = write_sync_csv(sync_points, "Alice", "Bob")
        restored = parse_csv_content(csv_content)

        assert len(restored) == 2
        assert restored[0].driver == "Alice"
        assert restored[1].driver == "Bob"
        assert restored[0].segments[0].title == "start"
        assert restored[0].segments[0].timestamp == pytest.approx(0.0, abs=0.001)
        assert restored[1].segments[0].timestamp == pytest.approx(1.0, abs=0.001)
