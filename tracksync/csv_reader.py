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
    """Parse CSV from a csv.reader object."""
    videos: List[VideoMetadata] = []
    num_drivers = 0

    for row in reader:
        if len(videos) == 0:
            # Parse header row for driver names (odd-indexed columns)
            drivers = [str(row[i]) for i in range(len(row)) if i % 2 == 1]
            num_drivers = len(drivers)
            for d in drivers:
                videos.append(VideoMetadata(driver=d, segments=[]))
            continue

        # Parse data rows
        segment_title = str(row[0])
        for i in range(num_drivers):
            segment = Segment(
                title=segment_title,
                timestamp=float(row[2 * i + 1]),
                speed=float(row[2 * i + 2]),
            )
            videos[i].add_segment(segment)

    return videos
