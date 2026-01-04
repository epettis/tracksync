"""Speed ratio calculations for syncing videos."""

from typing import List

from .models import Segment, ProcessedSegment


def calculate_speed_ratios(
    target_segments: List[Segment],
    reference_segments: List[Segment],
) -> List[float]:
    """
    Calculate speed ratios needed to sync target video to reference.

    The ratio represents how much to scale the target video's playback speed
    to match the reference video's timing at each segment:
    - ratio > 1.0: Target is faster, needs to slow down (longer duration)
    - ratio < 1.0: Target is slower, needs to speed up (shorter duration)
    - ratio = 1.0: Same speed, no adjustment needed

    Example: Target takes 10s, reference takes 15s for a segment.
    - ratio = 10/15 = 0.67
    - with_speed_scaled(0.67) slows target so 10s becomes 15s

    Args:
        target_segments: Segments from the video to be adjusted
        reference_segments: Segments from the reference video

    Returns:
        List of speed ratios for each segment (between consecutive timestamps)

    Raises:
        ValueError: If segment counts don't match or timestamps are invalid
    """
    if len(target_segments) != len(reference_segments):
        raise ValueError(
            f"Segment count mismatch: {len(target_segments)} vs {len(reference_segments)}"
        )

    ratios = []
    for i in range(1, len(target_segments)):
        target_duration = (
            target_segments[i].timestamp - target_segments[i - 1].timestamp
        )
        ref_duration = (
            reference_segments[i].timestamp - reference_segments[i - 1].timestamp
        )

        if target_duration <= 0:
            raise ValueError(
                f"Invalid target duration at segment {i}: {target_duration}"
            )

        ratio = target_duration / ref_duration
        ratios.append(ratio)

    return ratios


def build_processed_segments(
    segments: List[Segment],
    speed_ratios: List[float],
) -> List[ProcessedSegment]:
    """
    Convert segments and ratios into ProcessedSegment objects.

    Args:
        segments: Original segments with timestamps
        speed_ratios: Calculated speed ratios for each segment

    Returns:
        List of ProcessedSegment objects ready for video processing
    """
    processed = []
    for i in range(1, len(segments)):
        processed.append(
            ProcessedSegment(
                start_time=segments[i - 1].timestamp,
                end_time=segments[i].timestamp,
                speed_ratio=speed_ratios[i - 1],
            )
        )
    return processed
