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
    Build processed segments with clamping and freeze frame detection.

    Handles edge cases:
    - Segment end_time > video duration: clamp to duration
    - Segment becomes zero duration after clamping: use freeze frame
    - Both target and reference are zero: use freeze frame

    Args:
        target_segments: Clamped segments from target video
        reference_segments: Clamped segments from reference video
        target_duration: Actual duration of target video
        reference_duration: Actual duration of reference video

    Returns:
        List of ProcessedSegment objects, some may be freeze frames
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

        # Determine if we need a freeze frame
        needs_freeze_frame = False
        freeze_frame_time = None
        freeze_frame_duration = None

        if target_seg_duration <= 0:
            # Target segment has no content, use freeze frame
            needs_freeze_frame = True
            # Use the last valid frame before this segment
            freeze_frame_time = max(0, target_start - 0.01)
            freeze_frame_time = min(freeze_frame_time, target_duration - 0.01)
            # Duration should match reference timing
            freeze_frame_duration = ref_seg_duration if ref_seg_duration > 0 else 1.0

        elif ref_seg_duration <= 0:
            # Reference has no duration, can't calculate meaningful ratio
            # Play target at 1x speed for a reasonable duration
            needs_freeze_frame = False
            # Use ratio of 1.0 (no speed change)
            speed_ratio = 1.0

        if needs_freeze_frame:
            processed.append(
                ProcessedSegment(
                    start_time=target_start,
                    end_time=target_end,
                    speed_ratio=0.0,  # Not used for freeze frames
                    is_freeze_frame=True,
                    freeze_frame_time=freeze_frame_time,
                    freeze_frame_duration=freeze_frame_duration,
                )
            )
        else:
            # Normal segment processing
            if ref_seg_duration > 0:
                speed_ratio = target_seg_duration / ref_seg_duration
            else:
                speed_ratio = 1.0

            processed.append(
                ProcessedSegment(
                    start_time=target_start,
                    end_time=target_end,
                    speed_ratio=speed_ratio,
                    is_freeze_frame=False,
                )
            )

    return processed
