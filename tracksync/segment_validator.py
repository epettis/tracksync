"""Segment validation and clamping for out-of-bounds timestamps."""

from dataclasses import dataclass
from typing import List, Tuple

from .models import Segment, ProcessedSegment


@dataclass
class ClampedSegment:
    """A segment with clamping information."""

    original: Segment
    clamped_timestamp: float
    was_clamped: bool
    is_beyond_video: bool  # True if original timestamp >= video duration


def clamp_segments(
    segments: List[Segment],
    video_duration: float,
) -> Tuple[List[ClampedSegment], List[str]]:
    """
    Clamp segment timestamps to video duration.

    Args:
        segments: Original segments from CSV
        video_duration: Actual duration of the video in seconds

    Returns:
        Tuple of (clamped segments, warning messages)
    """
    clamped = []
    warnings = []

    for seg in segments:
        if seg.timestamp >= video_duration:
            # Timestamp is at or beyond video end
            clamped.append(
                ClampedSegment(
                    original=seg,
                    clamped_timestamp=video_duration,
                    was_clamped=True,
                    is_beyond_video=True,
                )
            )
            warnings.append(
                f"Segment '{seg.title}' timestamp {seg.timestamp:.2f}s "
                f"exceeds video duration {video_duration:.2f}s"
            )
        elif seg.timestamp > video_duration:
            # This case is covered above, but kept for clarity
            clamped.append(
                ClampedSegment(
                    original=seg,
                    clamped_timestamp=video_duration,
                    was_clamped=True,
                    is_beyond_video=False,
                )
            )
            warnings.append(
                f"Segment '{seg.title}' timestamp {seg.timestamp:.2f}s "
                f"clamped to video duration {video_duration:.2f}s"
            )
        else:
            # Timestamp is within video duration
            clamped.append(
                ClampedSegment(
                    original=seg,
                    clamped_timestamp=seg.timestamp,
                    was_clamped=False,
                    is_beyond_video=False,
                )
            )

    return clamped, warnings


def build_processed_segments_with_clamping(
    target_segments: List[ClampedSegment],
    reference_segments: List[ClampedSegment],
    target_duration: float,
    reference_duration: float,
) -> List[ProcessedSegment]:
    """
    Build processed segments with clamping.

    Handles edge cases:
    - Segment end_time > video duration: clamp to duration
    - Segment becomes zero duration after clamping: skip it (end video there)
    - Reference zero duration: use 1.0 speed ratio

    When CSV timestamps exceed video duration, the video ends at the actual
    video duration - no freeze frames or audio continuation.

    Args:
        target_segments: Clamped segments from target video
        reference_segments: Clamped segments from reference video
        target_duration: Actual duration of target video
        reference_duration: Actual duration of reference video

    Returns:
        List of ProcessedSegment objects (only segments with positive duration)
    """
    if len(target_segments) != len(reference_segments):
        raise ValueError(
            f"Segment count mismatch: {len(target_segments)} vs {len(reference_segments)}"
        )

    processed = []

    for i in range(1, len(target_segments)):
        target_start = target_segments[i - 1].clamped_timestamp
        target_end = min(target_segments[i].clamped_timestamp, target_duration)

        ref_start = reference_segments[i - 1].clamped_timestamp
        ref_end = min(reference_segments[i].clamped_timestamp, reference_duration)

        target_seg_duration = target_end - target_start
        ref_seg_duration = ref_end - ref_start

        # Skip segments with zero or negative duration (beyond video end)
        if target_seg_duration <= 0:
            continue

        # Calculate speed ratio
        if ref_seg_duration > 0:
            speed_ratio = target_seg_duration / ref_seg_duration
        else:
            # Reference has no duration, use 1.0 (no speed change)
            speed_ratio = 1.0

        processed.append(
            ProcessedSegment(
                start_time=target_start,
                end_time=target_end,
                speed_ratio=speed_ratio,
            )
        )

    return processed
