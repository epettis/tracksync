"""CSV writing for tracksync timestamp files."""

import csv
import io
import sys
from typing import List, Optional

from .models import VideoMetadata, Segment


def write_csv(videos: List[VideoMetadata], output_path: str) -> None:
    """
    Write VideoMetadata objects to a tracksync CSV file.

    CSV format (simple format):
    - Header row: milestone, driver1, driver2, ...
    - Data rows: milestone, timestamp1, timestamp2, ...

    Args:
        videos: List of VideoMetadata objects to write
        output_path: Path to the output CSV file
    """
    content = format_csv(videos)
    with open(output_path, "w") as f:
        f.write(content)


def format_csv(videos: List[VideoMetadata], include_speed: bool = False) -> str:
    """
    Format VideoMetadata objects as CSV content.

    Args:
        videos: List of VideoMetadata objects to format
        include_speed: If True, include speed columns (legacy format)

    Returns:
        CSV content as a string
    """
    if not videos:
        return ""

    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')

    # Build header row
    if include_speed:
        # Legacy format: milestone, driver1, , driver2, , ...
        header = ["milestone"]
        for video in videos:
            header.append(video.driver)
            header.append("")  # Empty column for speed
    else:
        # Simple format: milestone, driver1, driver2, ...
        header = ["milestone"] + [video.driver for video in videos]

    writer.writerow(header)

    # Verify all videos have the same number of segments
    if videos:
        num_segments = len(videos[0].segments)
        for video in videos[1:]:
            if len(video.segments) != num_segments:
                raise ValueError(
                    f"All videos must have the same number of segments. "
                    f"{videos[0].driver} has {num_segments}, "
                    f"{video.driver} has {len(video.segments)}"
                )

    # Build data rows
    if videos and videos[0].segments:
        for seg_idx in range(len(videos[0].segments)):
            segment_title = videos[0].segments[seg_idx].title

            if include_speed:
                # Legacy format: milestone, timestamp1, speed1, timestamp2, speed2, ...
                row = [segment_title]
                for video in videos:
                    seg = video.segments[seg_idx]
                    row.append(f"{seg.timestamp:.3f}")
                    row.append(f"{seg.speed:.1f}" if seg.speed != 1.0 else "")
            else:
                # Simple format: milestone, timestamp1, timestamp2, ...
                row = [segment_title]
                for video in videos:
                    row.append(f"{video.segments[seg_idx].timestamp:.3f}")

            writer.writerow(row)

    return output.getvalue()


def write_sync_csv(
    sync_points: list,
    driver_a: str,
    driver_b: str,
    output_path: Optional[str] = None
) -> str:
    """
    Output synchronization results in tracksync CSV format.

    The tracksync CSV format:
    - Header: milestone, driver1, driver2
    - Data: milestone, timestamp1, timestamp2

    Args:
        sync_points: List of SyncPoint objects
        driver_a: Name of driver A
        driver_b: Name of driver B
        output_path: Optional output file path

    Returns:
        CSV content as string
    """
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')

    # Header row
    writer.writerow(["milestone", driver_a, driver_b])

    # Data rows
    for point in sync_points:
        writer.writerow([point.label, f"{point.time_a:.3f}", f"{point.time_b:.3f}"])

    csv_content = output.getvalue()

    if output_path:
        with open(output_path, "w") as f:
            f.write(csv_content)
        print(f"\nTracksync CSV saved to: {output_path}", file=sys.stderr)

    return csv_content
