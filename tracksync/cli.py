"""Command-line interface for tracksync."""

import sys
from pathlib import Path
from typing import Optional

from .csv_reader import read_csv
from .models import VideoMetadata
from .segment_validator import (
    build_processed_segments_with_clamping,
    clamp_segments,
)
from .video_processor import VideoProcessor


def generate_comparison(
    target: VideoMetadata,
    reference: VideoMetadata,
    video_dir: Path,
    output_dir: Path,
    processor: Optional[VideoProcessor] = None,
) -> str:
    """
    Generate a comparison video between target and reference.

    Handles CSV timestamps that exceed video duration by clamping them
    and using freeze frames when necessary.

    Args:
        target: Video to be speed-adjusted
        reference: Reference video (unchanged)
        video_dir: Directory containing source videos
        output_dir: Directory for output videos
        processor: VideoProcessor instance (for testing injection)

    Returns:
        Path to the output video file
    """
    processor = processor or VideoProcessor()

    # Load source videos FIRST (need duration for clamping)
    target_clip = processor.load_video(str(video_dir / f"{target.driver}.mp4"))
    reference_clip = processor.load_video(str(video_dir / f"{reference.driver}.mp4"))

    # Clamp segments to video duration
    clamped_target, target_warnings = clamp_segments(
        target.segments, target_clip.duration
    )
    clamped_reference, ref_warnings = clamp_segments(
        reference.segments, reference_clip.duration
    )

    # Log warnings
    for warning in target_warnings:
        print(f"Warning ({target.driver}): {warning}")
    for warning in ref_warnings:
        print(f"Warning ({reference.driver}): {warning}")

    # Calculate speed ratios with freeze frame detection
    processed_segments = build_processed_segments_with_clamping(
        clamped_target,
        clamped_reference,
        target_clip.duration,
        reference_clip.duration,
    )

    # Process target video (trim, speed adjust, concatenate, or freeze frames)
    processed_target = processor.process_segments(target_clip, processed_segments)

    # Extract reference segment (first to last timestamp, no speed change)
    # Clamp reference timestamps as well
    ref_start = min(clamped_reference[0].clamped_timestamp, reference_clip.duration)
    ref_end = min(clamped_reference[-1].clamped_timestamp, reference_clip.duration)
    processed_reference = processor.extract_reference_segment(
        reference_clip, ref_start, ref_end
    )

    # Stack videos vertically
    stacked = processor.stack_vertically(processed_target, processed_reference)

    # Export with audio from reference
    output_filename = f"{target.driver}_v_{reference.driver}.mp4"
    output_path = str(output_dir / output_filename)
    processor.export(stacked, output_path, audio_from=reference_clip)

    # Clean up
    target_clip.close()
    reference_clip.close()

    return output_path


def main(argv: list = None) -> None:
    """Main entry point."""
    argv = argv or sys.argv

    if len(argv) < 2:
        print("Usage: tracksync <timestamps.csv> [video_dir] [output_dir]")
        sys.exit(1)

    csv_filename = Path(argv[1])
    video_dir = Path(argv[2]) if len(argv) > 2 else Path(".")
    output_dir = Path(argv[3]) if len(argv) > 3 else Path(".")

    videos = read_csv(str(csv_filename))

    for target in videos:
        for reference in videos:
            if target.driver != reference.driver:
                output = generate_comparison(target, reference, video_dir, output_dir)
                print(f"Generated: {output}")


if __name__ == "__main__":
    main()
