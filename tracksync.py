"""Generates pairwise video comparisons between drivers.

This script takes in a csv containing timestamps and speeds for
various drivers' Catalyst videos. Then, it uses one video as a
reference and modulates the speed of the other to create a
comparison video where the cars move at approximately the same
speed.

This module provides backwards compatibility. For new code,
import directly from the tracksync package.
"""

from tracksync.cli import main
from tracksync.csv_reader import read_csv
from tracksync.models import ProcessedSegment, Segment, VideoMetadata
from tracksync.speed_calculator import build_processed_segments, calculate_speed_ratios
from tracksync.video_processor import SyntheticClipFactory, VideoProcessor

# Backwards compatibility aliases
Video = VideoMetadata

__all__ = [
    "Segment",
    "Video",
    "VideoMetadata",
    "ProcessedSegment",
    "read_csv",
    "calculate_speed_ratios",
    "build_processed_segments",
    "VideoProcessor",
    "SyntheticClipFactory",
    "main",
]

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
