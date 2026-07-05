"""Unified command-line interface for tracksync.

Usage:
    tracksync sync <video_a> <video_b> [options]     # Auto-sync two videos
    tracksync sync --all <video1> <video2> ...       # Batch sync multiple videos
    tracksync generate <timestamps.csv> [options]    # Generate video from CSV
    tracksync debug <video_a> <video_b> [options]    # Interactive debug mode
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2

from .csv_reader import read_csv, parse_csv_content
from .models import VideoMetadata, SyncResult
from .frame_data import VideoFeatures
from .segment_validator import (
    build_processed_segments_with_clamping,
    clamp_segments,
)
from .video_processor import VideoProcessor
from .feature_extraction import (
    extract_video_features,
    extract_frame_data,
    interpolate_missing_circles,
)
from .cross_correlation import (
    compute_cross_correlations,
    compute_cross_correlations_from_features,
    generate_sync_points,
    generate_pairwise_sync_from_features,
    output_tracksync_csv,
)
from .scene_align import (
    _candidate_sync_times,
    coarse_align,
    compute_scene_cost_matrix,
    decode_native_window,
    extract_scene_features,
    generate_scene_sync_points,
)
from .embedding import make_embedder
from .fine_align import make_matcher
from .scene_deps import MissingSceneDependenciesError, require_scene_deps
from .turn_analysis import compute_turn_analysis
from .frame_analysis import interpolate_ocr_data, get_frame_at_time
from .visualization import (
    create_debug_display_v2,
    create_debug_display_from_diagnostic,
    create_scene_debug_display,
)
from .diagnostic_io import (
    export_diagnostic,
    import_diagnostic,
    get_results_from_diagnostic,
    get_turn_analysis_from_diagnostic,
)


def generate_comparison(
    target: VideoMetadata,
    reference: VideoMetadata,
    video_dir: Path,
    output_dir: Path,
    processor: Optional[VideoProcessor] = None,
    relative_timer: bool = True,
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

    # Burn in the iRacing-style relative timer on the seam. Sync times are the
    # clamped milestone timestamps, index-paired between target (top) and
    # reference (bottom); the overlay derives the per-frame gap from them.
    if relative_timer:
        top_times = [seg.clamped_timestamp for seg in clamped_target]
        bottom_times = [seg.clamped_timestamp for seg in clamped_reference]
        stacked = processor.overlay_relative_timer(stacked, top_times, bottom_times)

    # Export with audio from the reference SEGMENT (unwarped), not the full
    # reference clip: processed_reference is trimmed to [ref_start, ref_end],
    # so its audio matches the output span. Passing the full clip would stretch
    # the output to the untrimmed audio length and offset audio from video.
    output_filename = f"{target.driver}_v_{reference.driver}.mp4"
    output_path = str(output_dir / output_filename)
    processor.export(stacked, output_path, audio_from=processed_reference)

    # Clean up
    target_clip.close()
    reference_clip.close()

    return output_path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='tracksync',
        description='Synchronize and compare racing dashcam videos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-sync two videos
    tracksync sync video_a.mp4 video_b.mp4 -o sync.csv

    # Batch sync multiple videos
    tracksync sync --all video1.mp4 video2.mp4 video3.mp4 --output-dir ./out

    # Generate comparison video from CSV
    tracksync generate timestamps.csv --video-dir ./videos

    # Interactive debug mode
    tracksync debug video_a.mp4 video_b.mp4 --short
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # 'sync' subcommand
    sync_parser = subparsers.add_parser(
        'sync',
        help='Auto-synchronize videos using cross-correlation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sync two videos and output CSV
    tracksync sync video_a.mp4 video_b.mp4 -o sync.csv

    # Sync and generate comparison video
    tracksync sync video_a.mp4 video_b.mp4 --generate-video --output-dir ./out

    # Batch sync multiple videos efficiently
    tracksync sync --all v1.mp4 v2.mp4 v3.mp4 --output-dir ./sync_output
        """
    )
    sync_parser.add_argument('video_a', nargs='?', help='First video file')
    sync_parser.add_argument('video_b', nargs='?', help='Second video file')
    sync_parser.add_argument(
        '--all', nargs='+', metavar='VIDEO',
        help='Process multiple videos with O(n) feature extraction'
    )
    sync_parser.add_argument('-o', '--output', help='Output CSV file')
    sync_parser.add_argument('--output-dir', default='.', help='Output directory')
    sync_parser.add_argument(
        '--generate-video', action='store_true',
        help='Generate comparison video after sync'
    )
    sync_parser.add_argument('--video-dir', help='Source video directory')
    sync_parser.add_argument(
        '--overwrite', action='store_true',
        help='[--all] Recompute pairs even if their sync CSV already exists. '
             'By default an existing CSV is treated as a completed checkpoint '
             'and the pair is skipped, so a crashed run can be resumed by '
             'simply re-running the same command.'
    )
    sync_parser.add_argument(
        '--max-sync-interval', type=float, default=3.0,
        help='Maximum interval between sync points (default: 3.0)'
    )
    sync_parser.add_argument(
        '--short', '-s', action='store_true',
        help='Analyze only first 20 seconds'
    )
    sync_parser.add_argument(
        '--red-min', type=int, default=200,
        help='Minimum R value for red detection (default: 200)'
    )
    sync_parser.add_argument(
        '--red-max-gb', type=int, default=80,
        help='Maximum G/B value for red detection (default: 80)'
    )
    sync_parser.add_argument(
        '--mode', choices=['catalyst', 'scene'], default='scene',
        help='Alignment mode: scene (camera-agnostic, requires scene extras) '
             'or catalyst (legacy Garmin map dot) (default: scene)'
    )
    sync_parser.add_argument(
        '--sample-hz', type=float, default=None,
        help='[scene mode] Frame sampling rate in Hz (default: 10)'
    )
    sync_parser.add_argument(
        '--band-pct', type=float, default=None,
        help='[scene mode] DTW band as fraction of sequence length '
             '(default: 0.10)'
    )
    sync_parser.add_argument(
        '--min-inliers', type=int, default=None,
        help='[scene mode] Minimum RANSAC inliers to verify a sync point '
             '(default: 30)'
    )
    sync_parser.add_argument(
        '--fov-deg', type=float, default=None,
        help='[scene mode] Horizontal camera field of view in degrees '
             '(default: 90)'
    )
    sync_parser.add_argument(
        '--matcher', default=None,
        help='[scene mode] Feature matcher: aliked-lightglue or '
             'superpoint-lightglue (default: aliked-lightglue)'
    )
    sync_parser.add_argument(
        '--embedder', default=None,
        help='[scene mode] Frame embedder: dinov2-vitb14, dinov2-vits14, '
             'or gist (default: dinov2-vitb14)'
    )

    # 'generate' subcommand
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate comparison videos from CSV timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate videos from CSV
    tracksync generate timestamps.csv --video-dir ./videos --output-dir ./out

    # Using current directory for videos
    tracksync generate sync.csv
        """
    )
    gen_parser.add_argument('csv_file', help='Timestamps CSV file')
    gen_parser.add_argument(
        '--video-dir', default='.',
        help='Source video directory (default: current)'
    )
    gen_parser.add_argument(
        '--output-dir', default='.',
        help='Output video directory (default: current)'
    )
    gen_parser.add_argument(
        '--no-relative-timer', action='store_true',
        help='Disable the iRacing-style relative timer overlay'
    )

    # 'debug' subcommand
    debug_parser = subparsers.add_parser(
        'debug',
        help='Interactive debugging visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive debug mode
    tracksync debug video_a.mp4 video_b.mp4

    # Quick test with first 20 seconds
    tracksync debug video_a.mp4 video_b.mp4 --short

Controls:
    Arrow keys: Navigate frames
    Up/Down: Jump 1 second
    ESC: Exit
        """
    )
    debug_parser.add_argument('video_a', help='First video file')
    debug_parser.add_argument('video_b', help='Second video file')
    debug_parser.add_argument(
        '--short', '-s', action='store_true',
        help='Analyze only first 20 seconds'
    )
    debug_parser.add_argument(
        '--red-min', type=int, default=200,
        help='Minimum R value for red detection (default: 200)'
    )
    debug_parser.add_argument(
        '--red-max-gb', type=int, default=80,
        help='Maximum G/B value for red detection (default: 80)'
    )
    debug_parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Sample interval in seconds (default: 1.0)'
    )
    debug_parser.add_argument(
        '--diagnose-output', metavar='DIR',
        help='Export diagnostic data to DIR (protobuf text format)'
    )
    debug_parser.add_argument(
        '--mode', choices=['catalyst', 'scene'], default='scene',
        help='Alignment mode: scene (camera-agnostic, requires scene extras) '
             'or catalyst (legacy Garmin map dot) (default: scene)'
    )
    debug_parser.add_argument(
        '--sample-hz', type=float, default=None,
        help='[scene mode] Frame sampling rate in Hz (default: 10)'
    )
    debug_parser.add_argument(
        '--band-pct', type=float, default=None,
        help='[scene mode] DTW band as fraction of sequence length '
             '(default: 0.10)'
    )
    debug_parser.add_argument(
        '--embedder', default=None,
        help='[scene mode] Frame embedder: dinov2-vitb14, dinov2-vits14, '
             'or gist (default: dinov2-vitb14)'
    )
    debug_parser.add_argument(
        '--matcher', default=None,
        help='[scene mode] Optional feature matcher for drawing '
             'correspondences at the cursor (default: off)'
    )

    # 'diagnose' subcommand
    diagnose_parser = subparsers.add_parser(
        'diagnose',
        help='Load diagnostic data and run debug visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load diagnostic data and run debug UI
    tracksync diagnose ./diagnostic_output

    # Diagnose a specific video pair
    tracksync diagnose ./diagnostic_output/driver_a_v_driver_b.pb

Controls:
    Arrow keys: Navigate frames
    Up/Down: Jump 1 second
    ESC: Exit
        """
    )
    diagnose_parser.add_argument(
        'diagnostic_path',
        help='Directory containing .pb files or path to a specific .pb file'
    )

    return parser


# Scene-only sync flags: (attribute name, CLI flag, scene-mode default).
# All use default=None so catalyst mode can detect and reject explicit use.
_SCENE_ONLY_FLAGS = [
    ('sample_hz', '--sample-hz', 10.0),
    ('band_pct', '--band-pct', 0.10),
    ('min_inliers', '--min-inliers', 30),
    ('fov_deg', '--fov-deg', 90.0),
    ('matcher', '--matcher', 'aliked-lightglue'),
    ('embedder', '--embedder', 'dinov2-vitb14'),
]


def validate_sync_args(args: argparse.Namespace) -> None:
    """Validate sync arguments across modes.

    In catalyst mode, scene-only flags are rejected with a clear error
    (exit code 2). In scene mode, unset scene-only flags receive their
    defaults.
    """
    if args.mode == 'catalyst':
        offending = [
            flag for attr, flag, _ in _SCENE_ONLY_FLAGS
            if getattr(args, attr) is not None
        ]
        if offending:
            print(
                f"Error: {', '.join(offending)} "
                f"require{'s' if len(offending) == 1 else ''} --mode scene",
                file=sys.stderr
            )
            sys.exit(2)
    else:
        for attr, _, default in _SCENE_ONLY_FLAGS:
            if getattr(args, attr) is None:
                setattr(args, attr, default)


def check_scene_deps(args: argparse.Namespace) -> None:
    """Exit with code 2 if required scene dependencies are missing.

    Only the DINOv2 embedders and LightGlue matchers need torch; the gist
    embedder and test-only matchers do not, so they skip the check.
    """
    needs_torch = (
        args.embedder.startswith('dinov2')
        or 'lightglue' in (args.matcher or '')
    )
    if not needs_torch:
        return
    try:
        require_scene_deps()
    except MissingSceneDependenciesError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


# Scene-only debug flags: (attribute name, CLI flag, scene-mode default).
# --matcher stays None in scene mode: correspondence drawing is opt-in.
_SCENE_ONLY_DEBUG_FLAGS = [
    ('sample_hz', '--sample-hz', 10.0),
    ('band_pct', '--band-pct', 0.10),
    ('embedder', '--embedder', 'dinov2-vitb14'),
    ('matcher', '--matcher', None),
]


def validate_debug_args(args: argparse.Namespace) -> None:
    """Validate debug arguments across modes.

    In catalyst mode, scene-only flags are rejected with a clear error
    (exit code 2). In scene mode, unset scene-only flags receive their
    defaults.
    """
    if args.mode == 'catalyst':
        offending = [
            flag for attr, flag, _ in _SCENE_ONLY_DEBUG_FLAGS
            if getattr(args, attr) is not None
        ]
        if offending:
            print(
                f"Error: {', '.join(offending)} "
                f"require{'s' if len(offending) == 1 else ''} --mode scene",
                file=sys.stderr
            )
            sys.exit(2)
    else:
        for attr, _, default in _SCENE_ONLY_DEBUG_FLAGS:
            if getattr(args, attr) is None and default is not None:
                setattr(args, attr, default)


def run_sync_mode(args: argparse.Namespace) -> SyncResult:
    """
    Run synchronization mode to generate timestamps for video scaling.

    This mode:
    1. Extracts frame data at native FPS using VideoFeatures
    2. Computes cross-correlations using cached features
    3. Generates sync points with trim information
    4. Outputs tracksync-compatible CSV
    5. Optionally generates comparison video
    """
    print("=" * 60, file=sys.stderr)
    print("VIDEO SYNCHRONIZATION MODE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Check if both videos are the same file
    same_file = os.path.realpath(args.video_a) == os.path.realpath(args.video_b)

    # Determine max duration based on short mode
    max_duration = 20.0 if args.short else None

    print("\nPHASE 1: Extracting video features", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    # Extract features from both videos
    print(f"\nVideo A: {args.video_a}", file=sys.stderr)
    features_a = extract_video_features(
        args.video_a, args.red_min, args.red_max_gb,
        max_duration, "  [A] "
    )

    if same_file:
        print(f"\nVideo B: same as Video A (reusing extracted features)", file=sys.stderr)
        features_b = features_a
    else:
        print(f"\nVideo B: {args.video_b}", file=sys.stderr)
        features_b = extract_video_features(
            args.video_b, args.red_min, args.red_max_gb,
            max_duration, "  [B] "
        )

    print(f"\nExtracted {len(features_a.frames)} frames from A ({features_a.fps:.1f} fps), "
          f"{len(features_b.frames)} frames from B ({features_b.fps:.1f} fps)", file=sys.stderr)

    print("\nPHASE 2: Computing cross-correlations", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    # Compute cross-correlations using pre-extracted features
    results = compute_cross_correlations_from_features(features_a, features_b)

    print(f"\nComputed {len(results)} cross-correlation results", file=sys.stderr)

    print("\nPHASE 3: Generating synchronization points", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    # Generate sync points
    sync_result = generate_sync_points(
        results, features_a.crossings, features_b.crossings,
        max_interval=args.max_sync_interval
    )

    # Output tracksync CSV
    csv_output_path = args.output
    if not csv_output_path:
        # Default output path based on video names
        base_a = os.path.splitext(os.path.basename(args.video_a))[0]
        base_b = os.path.splitext(os.path.basename(args.video_b))[0]
        csv_output_path = f"{base_a}_v_{base_b}_sync.csv"

    csv_content = output_tracksync_csv(sync_result, args.video_a, args.video_b, csv_output_path)

    # Optionally generate comparison video
    if args.generate_video:
        print("\nPHASE 4: Generating synchronized comparison video", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        try:
            # Determine video directory
            video_dir = Path(args.video_dir) if args.video_dir else Path(args.video_a).parent
            output_dir = Path(args.output_dir)

            # Parse the CSV we just generated
            videos = parse_csv_content(csv_content)

            if len(videos) >= 2:
                target = videos[0]
                reference = videos[1]

                output_path = generate_comparison(
                    target, reference,
                    video_dir, output_dir
                )
                print(f"\nGenerated comparison video: {output_path}", file=sys.stderr)
            else:
                print("\nError: Need at least 2 drivers in sync data", file=sys.stderr)

        except Exception as e:
            print(f"\nError generating video: {e}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("SYNCHRONIZATION COMPLETE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return sync_result


def run_sync_all_mode(args: argparse.Namespace) -> dict[tuple[str, str], SyncResult]:
    """
    Run synchronization for all video pairs with O(n) feature extraction.

    This mode extracts features from each video exactly once, then generates
    pairwise sync results for all combinations.
    """
    video_paths = args.all
    output_dir = args.output_dir
    max_duration = 20.0 if args.short else None

    print("=" * 60, file=sys.stderr)
    print("MULTI-VIDEO SYNCHRONIZATION MODE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    n_videos = len(video_paths)
    print(f"\nProcessing {n_videos} videos", file=sys.stderr)
    print(f"This will generate {n_videos * (n_videos - 1)} pairwise comparisons", file=sys.stderr)
    print(f"Output directory: {os.path.abspath(output_dir)}", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Extract features from all videos (O(n))
    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 1: Extracting features from all videos", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_features: dict[str, VideoFeatures] = {}
    for i, video_path in enumerate(video_paths):
        print(f"\n[{i+1}/{n_videos}] ", end="", file=sys.stderr)
        features = extract_video_features(
            video_path, args.red_min, args.red_max_gb, max_duration, "  "
        )
        all_features[features.driver_name] = features

    # Phase 2: Generate pairwise sync results (O(n^2) but using cached features)
    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 2: Computing pairwise synchronizations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results: dict[tuple[str, str], SyncResult] = {}
    driver_names = list(all_features.keys())
    pair_count = 0
    total_pairs = n_videos * (n_videos - 1)

    for driver_a in driver_names:
        for driver_b in driver_names:
            if driver_a == driver_b:
                continue

            pair_count += 1
            print(f"\n[{pair_count}/{total_pairs}] {driver_a} vs {driver_b}", file=sys.stderr)

            features_a = all_features[driver_a]
            features_b = all_features[driver_b]

            sync_result = generate_pairwise_sync_from_features(
                features_a, features_b, args.max_sync_interval
            )
            results[(driver_a, driver_b)] = sync_result

            # Output CSV for this pair
            csv_output_path = os.path.join(output_dir, f"{driver_a}_v_{driver_b}_sync.csv")
            output_tracksync_csv(
                sync_result,
                features_a.video_path,
                features_b.video_path,
                csv_output_path
            )

    # Phase 3: Optionally generate videos
    if args.generate_video:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 3: Generating comparison videos", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        video_dir_path = Path(args.video_dir) if args.video_dir else Path(video_paths[0]).parent
        output_dir_path = Path(output_dir)

        print(f"  Video source directory: {video_dir_path}", file=sys.stderr)
        print(f"  Video output directory: {output_dir_path}", file=sys.stderr)

        for (driver_a, driver_b), sync_result in results.items():
            print(f"\n  Generating video: {driver_a} vs {driver_b}...", file=sys.stderr)

            try:
                # Read the CSV file that was already created in Phase 2
                csv_path = os.path.join(output_dir, f"{driver_a}_v_{driver_b}_sync.csv")
                videos = read_csv(csv_path)

                if len(videos) >= 2:
                    output_path = generate_comparison(
                        videos[0], videos[1],
                        video_dir_path, output_dir_path
                    )
                    print(f"    Generated: {output_path}", file=sys.stderr)
                else:
                    print(f"    Error: CSV has fewer than 2 videos", file=sys.stderr)
            except Exception as e:
                import traceback
                print(f"    Error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("MULTI-VIDEO SYNCHRONIZATION COMPLETE", file=sys.stderr)
    print(f"Generated {len(results)} pairwise sync results", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return results


def run_scene_sync_mode(args: argparse.Namespace) -> SyncResult:
    """
    Run scene-mode synchronization for a single video pair.

    This mode:
    1. Extracts scene features (sampled frames -> static mask -> embeddings)
    2. Coarse-aligns via banded open-end DTW on the embeddings
    3. Refines sync points via feature matching and epipolar geometry
    4. Outputs tracksync-compatible CSV
    5. Optionally generates comparison video
    """
    check_scene_deps(args)

    print("=" * 60, file=sys.stderr)
    print("VIDEO SYNCHRONIZATION MODE (SCENE)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    embedder = make_embedder(args.embedder)
    matcher = make_matcher(args.matcher)

    # Check if both videos are the same file
    same_file = os.path.realpath(args.video_a) == os.path.realpath(args.video_b)

    print("\nPHASE 1: Extracting scene features", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    print(f"\nVideo A: {args.video_a}", file=sys.stderr)
    feat_a = extract_scene_features(
        args.video_a, embedder, sample_hz=args.sample_hz
    )

    if same_file:
        print("\nVideo B: same as Video A (reusing extracted features)",
              file=sys.stderr)
        feat_b = feat_a
    else:
        print(f"\nVideo B: {args.video_b}", file=sys.stderr)
        feat_b = extract_scene_features(
            args.video_b, embedder, sample_hz=args.sample_hz
        )

    print(f"\nExtracted {len(feat_a.frame_times)} samples from A, "
          f"{len(feat_b.frame_times)} samples from B", file=sys.stderr)

    print("\nPHASE 2: Coarse alignment (DTW)", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    coarse = coarse_align(feat_a, feat_b, band_pct=args.band_pct)

    print(f"\nCoarse alignment: A [{coarse.trim_start_a:.2f}s, "
          f"{coarse.trim_end_a:.2f}s] -> B [{coarse.trim_start_b:.2f}s, "
          f"{coarse.trim_end_b:.2f}s]", file=sys.stderr)

    print("\nPHASE 3: Fine refinement and sync point generation",
          file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    sync_result = generate_scene_sync_points(
        coarse, args.video_a, args.video_b, matcher,
        max_sync_interval=args.max_sync_interval,
        min_inliers=args.min_inliers,
        fov_deg=args.fov_deg,
    )

    # Output tracksync CSV
    csv_output_path = args.output
    if not csv_output_path:
        base_a = os.path.splitext(os.path.basename(args.video_a))[0]
        base_b = os.path.splitext(os.path.basename(args.video_b))[0]
        csv_output_path = f"{base_a}_v_{base_b}_sync.csv"

    csv_content = output_tracksync_csv(
        sync_result, args.video_a, args.video_b, csv_output_path
    )

    # Optionally generate comparison video
    if args.generate_video:
        print("\nPHASE 4: Generating synchronized comparison video",
              file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        try:
            video_dir = (Path(args.video_dir) if args.video_dir
                         else Path(args.video_a).parent)
            output_dir = Path(args.output_dir)

            videos = parse_csv_content(csv_content)

            if len(videos) >= 2:
                output_path = generate_comparison(
                    videos[0], videos[1], video_dir, output_dir
                )
                print(f"\nGenerated comparison video: {output_path}",
                      file=sys.stderr)
            else:
                print("\nError: Need at least 2 drivers in sync data",
                      file=sys.stderr)
        except Exception as e:
            print(f"\nError generating video: {e}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("SYNCHRONIZATION COMPLETE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return sync_result


def run_scene_sync_all_mode(
    args: argparse.Namespace,
) -> dict[tuple[str, str], SyncResult]:
    """
    Run scene-mode synchronization for all video pairs.

    Scene features are extracted once per video (and cached on disk), then
    each ordered pair is coarse-aligned and refined.
    """
    check_scene_deps(args)

    video_paths = args.all
    output_dir = args.output_dir

    print("=" * 60, file=sys.stderr)
    print("MULTI-VIDEO SYNCHRONIZATION MODE (SCENE)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    n_videos = len(video_paths)
    print(f"\nProcessing {n_videos} videos", file=sys.stderr)
    print(f"This will generate {n_videos * (n_videos - 1)} pairwise "
          f"comparisons", file=sys.stderr)
    print(f"Output directory: {os.path.abspath(output_dir)}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    embedder = make_embedder(args.embedder)
    matcher = make_matcher(args.matcher)

    # Phase 1: Extract scene features from all videos (O(n), disk-cached)
    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 1: Extracting scene features from all videos",
          file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_features = {}
    driver_paths: dict[str, str] = {}
    for i, video_path in enumerate(video_paths):
        driver = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{i+1}/{n_videos}] {video_path}", file=sys.stderr)
        all_features[driver] = extract_scene_features(
            video_path, embedder, sample_hz=args.sample_hz
        )
        driver_paths[driver] = video_path

    # Phase 2: Pairwise coarse + fine alignment
    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 2: Computing pairwise synchronizations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results: dict[tuple[str, str], SyncResult] = {}
    driver_names = list(all_features.keys())
    pair_count = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []
    total_pairs = n_videos * (n_videos - 1)

    for driver_a in driver_names:
        for driver_b in driver_names:
            if driver_a == driver_b:
                continue

            pair_count += 1
            csv_output_path = os.path.join(
                output_dir, f"{driver_a}_v_{driver_b}_sync.csv"
            )

            # Resume checkpoint: an existing CSV means this pair already
            # completed on a prior run, so skip it unless --overwrite is set.
            if not args.overwrite and os.path.exists(csv_output_path) \
                    and os.path.getsize(csv_output_path) > 0:
                skipped += 1
                print(f"\n[{pair_count}/{total_pairs}] {driver_a} vs "
                      f"{driver_b} -- skipping (CSV exists, resuming)",
                      file=sys.stderr)
                continue

            print(f"\n[{pair_count}/{total_pairs}] {driver_a} vs {driver_b}",
                  file=sys.stderr)

            # Isolate per-pair failures: log and continue so one bad pair
            # (or a transient error) does not throw away the whole run.
            try:
                coarse = coarse_align(
                    all_features[driver_a], all_features[driver_b],
                    band_pct=args.band_pct
                )
                sync_result = generate_scene_sync_points(
                    coarse, driver_paths[driver_a], driver_paths[driver_b],
                    matcher,
                    max_sync_interval=args.max_sync_interval,
                    min_inliers=args.min_inliers,
                    fov_deg=args.fov_deg,
                )
            except Exception as e:  # noqa: BLE001 - keep batch alive
                import traceback
                failed.append((driver_a, driver_b, str(e)))
                print(f"    ERROR: {driver_a} vs {driver_b} failed: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue

            results[(driver_a, driver_b)] = sync_result

            # Write to a temp file then rename so an interrupted write never
            # leaves a truncated CSV that a later resume would trust.
            tmp_path = csv_output_path + ".tmp"
            output_tracksync_csv(
                sync_result,
                driver_paths[driver_a],
                driver_paths[driver_b],
                tmp_path
            )
            os.replace(tmp_path, csv_output_path)

    # Phase 3: Optionally generate videos
    if args.generate_video:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 3: Generating comparison videos", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        video_dir_path = (Path(args.video_dir) if args.video_dir
                          else Path(video_paths[0]).parent)
        output_dir_path = Path(output_dir)

        print(f"  Video source directory: {video_dir_path}", file=sys.stderr)
        print(f"  Video output directory: {output_dir_path}", file=sys.stderr)

        # Iterate every ordered pair whose CSV exists on disk (not just the
        # pairs computed this run) so --generate-video works on a resume too.
        for driver_a in driver_names:
          for driver_b in driver_names:
            if driver_a == driver_b:
                continue
            csv_path = os.path.join(
                output_dir, f"{driver_a}_v_{driver_b}_sync.csv"
            )
            if not os.path.exists(csv_path):
                continue
            print(f"\n  Generating video: {driver_a} vs {driver_b}...",
                  file=sys.stderr)

            try:
                videos = read_csv(csv_path)

                if len(videos) >= 2:
                    output_path = generate_comparison(
                        videos[0], videos[1],
                        video_dir_path, output_dir_path
                    )
                    print(f"    Generated: {output_path}", file=sys.stderr)
                else:
                    print("    Error: CSV has fewer than 2 videos",
                          file=sys.stderr)
            except Exception as e:
                import traceback
                print(f"    Error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("MULTI-VIDEO SYNCHRONIZATION COMPLETE", file=sys.stderr)
    print(f"Computed {len(results)} pair(s) this run; "
          f"skipped {skipped} already-done pair(s).", file=sys.stderr)
    if failed:
        print(f"{len(failed)} pair(s) FAILED (not written; re-run to retry):",
              file=sys.stderr)
        for driver_a, driver_b, err in failed:
            print(f"  - {driver_a} vs {driver_b}: {err}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return results


def run_debug_mode(args: argparse.Namespace) -> None:
    """Run interactive debug mode with pre-computed cross-correlations."""
    video_a_path = args.video_a
    video_b_path = args.video_b
    red_min = args.red_min
    red_max_gb = args.red_max_gb
    interval = args.interval
    short_mode = args.short

    # Check if both videos are the same file
    same_file = os.path.realpath(video_a_path) == os.path.realpath(video_b_path)

    print("=" * 60, file=sys.stderr)
    print("PHASE 1: Extracting frame data from videos (native FPS)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Determine max duration based on short mode
    max_duration = 20.0 if short_mode else None

    # Extract all frame data from both videos at native FPS
    print(f"\nVideo A: {video_a_path}", file=sys.stderr)
    frames_a, fps_a = extract_frame_data(
        video_a_path, interval, red_min, red_max_gb,
        "  [A] ", use_native_fps=True, max_duration=max_duration
    )

    if same_file:
        print(f"\nVideo B: same as Video A (reusing extracted data)", file=sys.stderr)
        frames_b = frames_a
        fps_b = fps_a
    else:
        print(f"\nVideo B: {video_b_path}", file=sys.stderr)
        frames_b, fps_b = extract_frame_data(
            video_b_path, interval, red_min, red_max_gb,
            "  [B] ", use_native_fps=True, max_duration=max_duration
        )

    print(f"\nExtracted {len(frames_a)} frames from A ({fps_a:.1f} fps), "
          f"{len(frames_b)} frames from B ({fps_b:.1f} fps)", file=sys.stderr)
    if same_file:
        print("(Same file optimization: frames extracted only once)", file=sys.stderr)

    # Count frames with detected circles
    circles_a = sum(1 for f in frames_a if f.circle is not None)
    circles_b = sum(1 for f in frames_b if f.circle is not None)
    print(f"Red circles detected: A={circles_a}/{len(frames_a)}, B={circles_b}/{len(frames_b)}", file=sys.stderr)

    # Interpolate OCR data and detect start-finish crossings
    print("\nInterpolating OCR data...", file=sys.stderr)
    frame_times_a = [f.time for f in frames_a]
    frame_times_b = [f.time for f in frames_b]
    ocr_list_a = [f.ocr_data for f in frames_a]
    ocr_list_b = [f.ocr_data for f in frames_b]

    interpolated_ocr_a, crossings_a = interpolate_ocr_data(ocr_list_a, frame_times_a, fps_a)
    if same_file:
        interpolated_ocr_b = interpolated_ocr_a
        crossings_b = crossings_a
    else:
        interpolated_ocr_b, crossings_b = interpolate_ocr_data(ocr_list_b, frame_times_b, fps_b)

    # Update frame data with interpolated OCR
    for i, ocr in enumerate(interpolated_ocr_a):
        frames_a[i].ocr_data = ocr
    if not same_file:
        for i, ocr in enumerate(interpolated_ocr_b):
            frames_b[i].ocr_data = ocr

    # Report OCR statistics
    ocr_laps_a = sum(1 for ocr in interpolated_ocr_a if ocr.lap_number is not None)
    ocr_segs_a = sum(1 for ocr in interpolated_ocr_a if ocr.segment_number is not None)
    ocr_times_a = sum(1 for ocr in interpolated_ocr_a if ocr.lap_time_seconds is not None)
    print(f"  Video A OCR: {ocr_laps_a} laps, {ocr_segs_a} segments, {ocr_times_a} lap times filled", file=sys.stderr)
    print(f"  Video A crossings: {len(crossings_a)} start-finish line crossings detected", file=sys.stderr)

    if not same_file:
        ocr_laps_b = sum(1 for ocr in interpolated_ocr_b if ocr.lap_number is not None)
        ocr_segs_b = sum(1 for ocr in interpolated_ocr_b if ocr.segment_number is not None)
        ocr_times_b = sum(1 for ocr in interpolated_ocr_b if ocr.lap_time_seconds is not None)
        print(f"  Video B OCR: {ocr_laps_b} laps, {ocr_segs_b} segments, {ocr_times_b} lap times filled", file=sys.stderr)
        print(f"  Video B crossings: {len(crossings_b)} start-finish line crossings detected", file=sys.stderr)

    # Report crossing times
    if crossings_a:
        print("\n  Video A start-finish crossings:", file=sys.stderr)
        for c in crossings_a:
            lap_info = f"Lap {c.lap_before} -> {c.lap_after}" if c.lap_before and c.lap_after else "?"
            print(f"    {c.video_time:.2f}s ({lap_info})", file=sys.stderr)

    if not same_file and crossings_b:
        print("\n  Video B start-finish crossings:", file=sys.stderr)
        for c in crossings_b:
            lap_info = f"Lap {c.lap_before} -> {c.lap_after}" if c.lap_before and c.lap_after else "?"
            print(f"    {c.video_time:.2f}s ({lap_info})", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 2: Computing cross-correlations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Pre-compute all cross-correlations with crossing constraints
    results = compute_cross_correlations(
        frames_a, frames_b,
        crossings_a=crossings_a,
        crossings_b=crossings_b
    )

    print(f"\nComputed {len(results)} cross-correlation results", file=sys.stderr)

    # Compute turn analysis for both videos
    print("\nComputing turn analysis...", file=sys.stderr)
    positions_a = interpolate_missing_circles(frames_a)
    if same_file:
        positions_b = positions_a
    else:
        positions_b = interpolate_missing_circles(frames_b)

    turn_analysis_a = compute_turn_analysis(positions_a, frame_times_a, fps_a, window_seconds=3.0)
    if same_file:
        turn_analysis_b = turn_analysis_a
    else:
        turn_analysis_b = compute_turn_analysis(positions_b, frame_times_b, fps_b, window_seconds=3.0)

    print(f"  Video A: {len(turn_analysis_a.apexes)} turn apexes detected", file=sys.stderr)
    if not same_file:
        print(f"  Video B: {len(turn_analysis_b.apexes)} turn apexes detected", file=sys.stderr)

    # Export diagnostic data if requested
    diagnose_output = getattr(args, 'diagnose_output', None)
    if diagnose_output:
        print("\n" + "=" * 60, file=sys.stderr)
        print("EXPORTING DIAGNOSTIC DATA", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Get video durations
        cap_a = cv2.VideoCapture(video_a_path)
        duration_a = cap_a.get(cv2.CAP_PROP_FRAME_COUNT) / cap_a.get(cv2.CAP_PROP_FPS)
        cap_a.release()

        if same_file:
            duration_b = duration_a
        else:
            cap_b = cv2.VideoCapture(video_b_path)
            duration_b = cap_b.get(cv2.CAP_PROP_FRAME_COUNT) / cap_b.get(cv2.CAP_PROP_FPS)
            cap_b.release()

        # Build output path
        driver_a = os.path.splitext(os.path.basename(video_a_path))[0]
        driver_b = os.path.splitext(os.path.basename(video_b_path))[0]
        output_path = os.path.join(diagnose_output, f"{driver_a}_v_{driver_b}.pb")

        export_diagnostic(
            results=results,
            turn_analysis_a=turn_analysis_a,
            turn_analysis_b=turn_analysis_b,
            video_a_path=video_a_path,
            video_b_path=video_b_path,
            fps_a=fps_a,
            fps_b=fps_b,
            duration_a=duration_a,
            duration_b=duration_b,
            crossings_a=crossings_a,
            crossings_b=crossings_b,
            output_path=output_path,
        )

        print(f"\nExported diagnostic data to: {output_path}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 3: Interactive visualization", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Use actual FPS for jumping by 1 second
    frames_per_second = int(fps_a)

    print("\nControls:", file=sys.stderr)
    print("  → (Right Arrow): Next frame", file=sys.stderr)
    print("  ← (Left Arrow): Previous frame", file=sys.stderr)
    print(f"  ↑ (Up Arrow): Jump forward 1 second ({frames_per_second} frames)", file=sys.stderr)
    print(f"  ↓ (Down Arrow): Jump back 1 second ({frames_per_second} frames)", file=sys.stderr)
    print("  ESC: Exit", file=sys.stderr)
    print(file=sys.stderr)

    # Interactive display loop
    current_idx = 0

    while True:
        result = results[current_idx]

        # Create visualization from pre-computed data
        display = create_debug_display_v2(result, len(results), current_idx, turn_analysis_a, turn_analysis_b)

        cv2.imshow("Cross-Correlation Debug", display)

        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 83 or key == 3:  # Right arrow - next frame
            current_idx = min(current_idx + 1, len(results) - 1)
        elif key == 81 or key == 2:  # Left arrow - previous frame
            current_idx = max(current_idx - 1, 0)
        elif key == 82 or key == 0:  # Up arrow - jump forward 1 second
            current_idx = min(current_idx + frames_per_second, len(results) - 1)
        elif key == 84 or key == 1:  # Down arrow - jump back 1 second
            current_idx = max(current_idx - frames_per_second, 0)

    cv2.destroyAllWindows()


def run_scene_debug_mode(args: argparse.Namespace) -> None:
    """Run interactive scene-mode debug visualization.

    Shows the DTW cost-matrix heatmap with path overlay, the frame pair at
    the cursor (with feature correspondences when a matcher is selected),
    and the per-frame confidence trace with sync points marked. Panel
    rendering is done by pure functions in visualization.py; this loop only
    computes alignment data and displays the composed image.
    """
    check_scene_deps(args)

    print("=" * 60, file=sys.stderr)
    print("SCENE-MODE DEBUG VISUALIZATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    embedder = make_embedder(args.embedder)
    matcher = make_matcher(args.matcher) if args.matcher else None

    same_file = os.path.realpath(args.video_a) == os.path.realpath(args.video_b)

    print("\nPHASE 1: Extracting scene features", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    print(f"\nVideo A: {args.video_a}", file=sys.stderr)
    feat_a = extract_scene_features(
        args.video_a, embedder, sample_hz=args.sample_hz
    )
    if same_file:
        print("\nVideo B: same as Video A (reusing extracted features)",
              file=sys.stderr)
        feat_b = feat_a
    else:
        print(f"\nVideo B: {args.video_b}", file=sys.stderr)
        feat_b = extract_scene_features(
            args.video_b, embedder, sample_hz=args.sample_hz
        )

    print("\nPHASE 2: Coarse alignment (DTW)", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    coarse = coarse_align(feat_a, feat_b, band_pct=args.band_pct)
    cost = compute_scene_cost_matrix(feat_a, feat_b)
    sync_times = _candidate_sync_times(coarse, max_sync_interval=3.0)

    print(f"\nCoarse alignment: A [{coarse.trim_start_a:.2f}s, "
          f"{coarse.trim_end_a:.2f}s] -> B [{coarse.trim_start_b:.2f}s, "
          f"{coarse.trim_end_b:.2f}s]", file=sys.stderr)

    print("\nPHASE 3: Interactive visualization", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    frame_times = feat_a.frame_times
    total = len(frame_times)
    samples_per_second = max(1, int(round(args.sample_hz)))

    print("\nControls:", file=sys.stderr)
    print("  Left/Right Arrow: Previous/next sample", file=sys.stderr)
    print("  Up/Down Arrow: Jump 1 second", file=sys.stderr)
    print("  ESC: Exit", file=sys.stderr)
    print(file=sys.stderr)

    current_idx = 0
    while True:
        t_a = float(frame_times[current_idx])
        t_b = float(coarse.f(t_a))

        frames_a, _ = decode_native_window(args.video_a, t_a, 0.0)
        frames_b, _ = decode_native_window(args.video_b, t_b, 0.0)
        frame_a = frames_a[0] if frames_a else None
        frame_b = frames_b[0] if frames_b else None

        pts_a = pts_b = None
        if matcher is not None and frame_a is not None and frame_b is not None:
            pts_a, pts_b = matcher.match(frame_a, frame_b)

        display = create_scene_debug_display(
            cost=cost,
            path=coarse.path,
            frame_times=frame_times,
            margins=coarse.margins,
            frame_a=frame_a,
            frame_b=frame_b,
            time_a=t_a,
            time_b=t_b,
            cursor_idx=current_idx,
            total=total,
            sync_times=sync_times,
            pts_a=pts_a,
            pts_b=pts_b,
        )

        cv2.imshow("Scene Alignment Debug", display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 83 or key == 3:  # Right arrow - next sample
            current_idx = min(current_idx + 1, total - 1)
        elif key == 81 or key == 2:  # Left arrow - previous sample
            current_idx = max(current_idx - 1, 0)
        elif key == 82 or key == 0:  # Up arrow - jump forward 1 second
            current_idx = min(current_idx + samples_per_second, total - 1)
        elif key == 84 or key == 1:  # Down arrow - jump back 1 second
            current_idx = max(current_idx - samples_per_second, 0)

    cv2.destroyAllWindows()


def run_generate_mode(args: argparse.Namespace) -> None:
    """Generate comparison videos from CSV timestamps."""
    csv_filename = Path(args.csv_file)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    videos = read_csv(str(csv_filename))
    relative_timer = not getattr(args, 'no_relative_timer', False)

    for target in videos:
        for reference in videos:
            if target.driver != reference.driver:
                output = generate_comparison(
                    target, reference, video_dir, output_dir,
                    relative_timer=relative_timer,
                )
                print(f"Generated: {output}")


def run_diagnose_mode(args: argparse.Namespace) -> None:
    """Run debug visualization from pre-computed diagnostic data."""
    diagnostic_path = args.diagnostic_path

    # Find .pb files
    pb_files = []
    if os.path.isfile(diagnostic_path) and diagnostic_path.endswith('.pb'):
        pb_files = [diagnostic_path]
    elif os.path.isdir(diagnostic_path):
        pb_files = sorted(Path(diagnostic_path).glob('*.pb'))
        if not pb_files:
            print(f"Error: No .pb files found in {diagnostic_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: {diagnostic_path} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)

    # For now, process the first .pb file (could add selection UI later)
    pb_file = str(pb_files[0])

    print("=" * 60, file=sys.stderr)
    print("LOADING DIAGNOSTIC DATA", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"\nLoading: {pb_file}", file=sys.stderr)

    # Load diagnostic data
    diagnostic = import_diagnostic(pb_file)

    print(f"  Version: {diagnostic.tracksync_version}", file=sys.stderr)
    print(f"  Video A: {diagnostic.video_a.driver_name} ({diagnostic.video_a.video_path})", file=sys.stderr)
    print(f"  Video B: {diagnostic.video_b.driver_name} ({diagnostic.video_b.video_path})", file=sys.stderr)
    print(f"  Results: {len(diagnostic.results)} frames", file=sys.stderr)

    # Extract data from diagnostic
    results = get_results_from_diagnostic(diagnostic)
    turn_analysis_a, turn_analysis_b = get_turn_analysis_from_diagnostic(diagnostic)

    # Open video files for on-demand frame loading
    video_a_path = diagnostic.video_a.video_path
    video_b_path = diagnostic.video_b.video_path

    if not os.path.exists(video_a_path):
        print(f"Warning: Video A not found at {video_a_path}", file=sys.stderr)
        print("  Frames will not be displayed.", file=sys.stderr)
        cap_a = None
    else:
        cap_a = cv2.VideoCapture(video_a_path)

    if not os.path.exists(video_b_path):
        print(f"Warning: Video B not found at {video_b_path}", file=sys.stderr)
        print("  Frames will not be displayed.", file=sys.stderr)
        cap_b = None
    else:
        cap_b = cv2.VideoCapture(video_b_path)

    fps_a = diagnostic.video_a.fps
    frames_per_second = int(fps_a) if fps_a > 0 else 30

    print("\n" + "=" * 60, file=sys.stderr)
    print("INTERACTIVE VISUALIZATION (from diagnostic data)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print("\nControls:", file=sys.stderr)
    print("  → (Right Arrow): Next frame", file=sys.stderr)
    print("  ← (Left Arrow): Previous frame", file=sys.stderr)
    print(f"  ↑ (Up Arrow): Jump forward 1 second ({frames_per_second} frames)", file=sys.stderr)
    print(f"  ↓ (Down Arrow): Jump back 1 second ({frames_per_second} frames)", file=sys.stderr)
    print("  ESC: Exit", file=sys.stderr)
    print(file=sys.stderr)

    # Interactive display loop
    current_idx = 0

    while True:
        result = results[current_idx]

        # Load frames on-demand from video files
        frame_a = None
        frame_b = None

        if cap_a is not None:
            frame_a = get_frame_at_time(cap_a, result.time_a)

        if cap_b is not None and result.best_time_b is not None:
            frame_b = get_frame_at_time(cap_b, result.best_time_b)

        # Create visualization with on-demand frames
        display = create_debug_display_from_diagnostic(
            result, frame_a, frame_b,
            len(results), current_idx,
            turn_analysis_a, turn_analysis_b
        )

        cv2.imshow("Cross-Correlation Debug (Diagnostic Mode)", display)

        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 83 or key == 3:  # Right arrow - next frame
            current_idx = min(current_idx + 1, len(results) - 1)
        elif key == 81 or key == 2:  # Left arrow - previous frame
            current_idx = max(current_idx - 1, 0)
        elif key == 82 or key == 0:  # Up arrow - jump forward 1 second
            current_idx = min(current_idx + frames_per_second, len(results) - 1)
        elif key == 84 or key == 1:  # Down arrow - jump back 1 second
            current_idx = max(current_idx - frames_per_second, 0)

    cv2.destroyAllWindows()

    # Clean up
    if cap_a is not None:
        cap_a.release()
    if cap_b is not None:
        cap_b.release()


def main(argv: list = None) -> None:
    """Main entry point."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'sync':
        validate_sync_args(args)
        if args.all:
            if args.mode == 'scene':
                run_scene_sync_all_mode(args)
            else:
                run_sync_all_mode(args)
        else:
            if not args.video_a or not args.video_b:
                print("Error: sync requires video_a and video_b (or --all)", file=sys.stderr)
                sys.exit(1)
            if args.mode == 'scene':
                run_scene_sync_mode(args)
            else:
                run_sync_mode(args)
    elif args.command == 'generate':
        run_generate_mode(args)
    elif args.command == 'debug':
        validate_debug_args(args)
        if args.mode == 'scene':
            run_scene_debug_mode(args)
        else:
            run_debug_mode(args)
    elif args.command == 'diagnose':
        run_diagnose_mode(args)


if __name__ == "__main__":
    main()
