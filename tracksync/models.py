"""Data models for tracksync."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Segment:
    """A milestone segment in a video."""

    title: str
    timestamp: float
    speed: float


@dataclass
class VideoMetadata:
    """Metadata about a driver's video without actual video data."""

    driver: str
    segments: List[Segment] = field(default_factory=list)

    def add_segment(self, segment: Segment) -> None:
        """Add a segment to this video."""
        self.segments.append(segment)


@dataclass(frozen=True)
class ProcessedSegment:
    """A segment with its calculated speed ratio applied."""

    start_time: float
    end_time: float
    speed_ratio: float
