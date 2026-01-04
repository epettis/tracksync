"""Data models for tracksync."""

from dataclasses import dataclass, field
from typing import List, Optional


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
    """A segment with its calculated speed ratio applied.

    For normal segments, start_time/end_time define the video region to extract
    and speed_ratio defines the playback speed adjustment.

    For freeze frames (when is_freeze_frame=True), freeze_frame_time specifies
    which frame to extract and freeze_frame_duration specifies how long to
    display it. The start_time/end_time/speed_ratio are ignored.
    """

    start_time: float
    end_time: float
    speed_ratio: float
    is_freeze_frame: bool = False
    freeze_frame_time: Optional[float] = None
    freeze_frame_duration: Optional[float] = None
