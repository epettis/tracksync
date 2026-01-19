"""Data models for tracksync."""

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_analysis import StartFinishCrossing


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

    start_time/end_time define the video region to extract and
    speed_ratio defines the playback speed adjustment.
    """

    start_time: float
    end_time: float
    speed_ratio: float


@dataclass
class SyncPoint:
    """A synchronization point between two videos."""
    time_a: float  # Timestamp in video A
    time_b: float  # Corresponding timestamp in video B
    label: str  # Description (e.g., "start", "segment_5", "periodic")
    speed: float = 1.0  # Speed ratio (time_b progression / time_a progression)


@dataclass
class SyncResult:
    """Result of synchronization analysis."""
    sync_points: list[SyncPoint]
    trim_start_a: float  # Start time to trim video A
    trim_end_a: float  # End time to trim video A
    trim_start_b: float  # Start time to trim video B
    trim_end_b: float  # End time to trim video B
    crossings_a: list["StartFinishCrossing"]
    crossings_b: list["StartFinishCrossing"]


@dataclass
class TurnApex:
    """A detected turn apex (local maximum in sharpness)."""
    time: float  # Video timestamp
    angle: float  # Interior angle in degrees (smaller = sharper turn)
    sharpness: float  # 180 - angle (higher = sharper turn)
    prominence: float  # Peak prominence (how significant this apex is)


@dataclass
class TurnAnalysis:
    """Turn detection analysis for a video."""
    times: list[float]  # Timestamps for each angle measurement
    angles: list[float]  # Interior angles in degrees
    sharpness: list[float]  # 180 - angle for each measurement
    apexes: list[TurnApex]  # Detected turn apexes (local maxima in sharpness)
    window_seconds: float  # Window size used for calculation
