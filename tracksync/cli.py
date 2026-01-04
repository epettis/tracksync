"""Command-line interface for tracksync."""

import sys
from pathlib import Path
from typing import Optional

from .csv_reader import read_csv
from .models import VideoMetadata
from .speed_calculator import build_processed_segments, calculate_speed_ratios
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

    # Calculate speed ratios
    ratios = calculate_speed_ratios(target.segments, reference.segments)
    processed_segments = build_processed_segments(target.segments, ratios)

    # Load source videos
    target_clip = processor.load_video(str(video_dir / f"{target.driver}.mp4"))
    reference_clip = processor.load_video(str(video_dir / f"{reference.driver}.mp4"))

    # Process target video (trim, speed adjust, concatenate)
    processed_target = processor.process_segments(target_clip, processed_segments)

    # Extract reference segment (first to last timestamp, no speed change)
    ref_start = reference.segments[0].timestamp
    ref_end = reference.segments[-1].timestamp
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
