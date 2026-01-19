"""CSV parsing for tracksync timestamp files."""

import csv
import io
from typing import List

from .models import Segment, VideoMetadata


def read_csv(csv_filename: str) -> List[VideoMetadata]:
    """
    Read a tracksync CSV file and return VideoMetadata objects.

    CSV format:
    - Header row: milestone_name, driver1,, driver2,, ...
    - Data rows: milestone, timestamp1, speed1, timestamp2, speed2, ...

    Args:
        csv_filename: Path to the CSV file

    Returns:
        List of VideoMetadata objects, one per driver
    """
    with open(csv_filename, "r") as csvfile:
        return _parse_csv_reader(csv.reader(csvfile))


def parse_csv_content(content: str) -> List[VideoMetadata]:
    """
    Parse CSV content from a string (for testing).

    Args:
        content: CSV content as a string

    Returns:
        List of VideoMetadata objects
    """
    return _parse_csv_reader(csv.reader(io.StringIO(content)))


def _parse_csv_reader(reader) -> List[VideoMetadata]:
    """Parse CSV from a csv.reader object.

    Supports two formats:
    1. Simple format: milestone,driver1,driver2 (no speed columns)
    2. Legacy format: milestone,driver1,,driver2,, (with speed columns)
    """
    videos: List[VideoMetadata] = []
    num_drivers = 0
    has_speed_columns = False

    for row in reader:
        if len(videos) == 0:
            # Detect format by checking for empty columns (legacy format)
            # Legacy: milestone,driver1,,driver2,, has empty strings at even indices > 0
            # Simple: milestone,driver1,driver2 has no empty strings
            has_speed_columns = any(row[i] == '' for i in range(2, len(row), 2) if i < len(row))

            if has_speed_columns:
                # Legacy format: driver names at odd indices (1, 3, 5...)
                drivers = [str(row[i]) for i in range(len(row)) if i % 2 == 1]
            else:
                # Simple format: driver names at indices 1, 2, 3...
                drivers = [str(row[i]) for i in range(1, len(row))]

            num_drivers = len(drivers)
            for d in drivers:
                videos.append(VideoMetadata(driver=d, segments=[]))
            continue

        # Parse data rows
        segment_title = str(row[0])
        for i in range(num_drivers):
            if has_speed_columns:
                # Legacy format: timestamp at 2*i+1, speed at 2*i+2
                timestamp = float(row[2 * i + 1])
                speed = float(row[2 * i + 2]) if row[2 * i + 2] else 1.0
            else:
                # Simple format: timestamp at i+1, no speed (default to 1.0)
                timestamp = float(row[i + 1])
                speed = 1.0

            segment = Segment(
                title=segment_title,
                timestamp=timestamp,
                speed=speed,
            )
            videos[i].add_segment(segment)

    return videos
