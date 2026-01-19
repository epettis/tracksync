"""Tests for CSV parsing."""

import pytest

from tracksync.csv_reader import parse_csv_content, read_csv


class TestParseCSVContent:
    def test_parses_header_correctly(self, sample_csv_content):
        """Test that driver names are extracted from header."""
        videos = parse_csv_content(sample_csv_content)

        assert len(videos) == 2
        assert videos[0].driver == "Driver1"
        assert videos[1].driver == "Driver2"

    def test_parses_segments_correctly(self, sample_csv_content):
        """Test that segments are parsed correctly."""
        videos = parse_csv_content(sample_csv_content)

        assert len(videos[0].segments) == 4
        assert videos[0].segments[0].title == "Start"
        assert videos[0].segments[0].timestamp == 0.0

    def test_parses_all_segment_values(self, sample_csv_content):
        """Test all segment values are parsed correctly."""
        videos = parse_csv_content(sample_csv_content)

        # Check Driver1 segments
        d1 = videos[0]
        assert d1.segments[0].title == "Start"
        assert d1.segments[0].timestamp == 0.0
        assert d1.segments[1].title == "T1"
        assert d1.segments[1].timestamp == 10.0
        assert d1.segments[2].title == "T2"
        assert d1.segments[2].timestamp == 20.0
        assert d1.segments[3].title == "Finish"
        assert d1.segments[3].timestamp == 30.0

        # Check Driver2 segments
        d2 = videos[1]
        assert d2.segments[0].timestamp == 0.0
        assert d2.segments[1].timestamp == 15.0
        assert d2.segments[2].timestamp == 25.0
        assert d2.segments[3].timestamp == 40.0

    def test_multiple_drivers(self):
        """Test parsing with multiple drivers."""
        content = """Ref,A,B,C
Start,0.0,0.1,0.2
End,10.0,11.0,12.0
"""
        videos = parse_csv_content(content)

        assert len(videos) == 3
        assert [v.driver for v in videos] == ["A", "B", "C"]

    def test_single_driver(self):
        """Test parsing with a single driver."""
        content = """Ref,Solo
Start,0.0
Middle,5.0
End,10.0
"""
        videos = parse_csv_content(content)

        assert len(videos) == 1
        assert videos[0].driver == "Solo"
        assert len(videos[0].segments) == 3


class TestReadCSV:
    def test_read_actual_file(self, tmp_path):
        """Test reading from an actual file."""
        csv_content = """Ref,Driver1
Start,0.0
End,10.0
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        videos = read_csv(str(csv_file))

        assert len(videos) == 1
        assert videos[0].driver == "Driver1"
        assert len(videos[0].segments) == 2

    def test_read_file_with_multiple_drivers(self, tmp_path):
        """Test reading a file with multiple drivers."""
        csv_content = """Ref,A,B
Start,0.0,0.5
Mid,5.0,6.0
End,10.0,11.0
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        videos = read_csv(str(csv_file))

        assert len(videos) == 2
        assert videos[0].driver == "A"
        assert videos[1].driver == "B"
        assert len(videos[0].segments) == 3
        assert len(videos[1].segments) == 3
