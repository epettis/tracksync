#!/usr/bin/env python3
"""
Video alignment utility for automatically syncing two racing videos.

This tool uses cross-correlation of red dot positions on the track overlay
to find matching timestamps between videos.

Usage:
    python video_align.py video_a.mp4 video_b.mp4 --output timestamps.csv
    python video_align.py video_a.mp4 video_b.mp4 --debug
"""

import argparse
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from tracksync.autocorrelation import (
    binarize_frame,
    compute_correlation,
    get_frame_at_time,
    get_video_info,
    find_best_match,
    find_red_circle,
    detect_segment_number,
    extract_frame_ocr,
    interpolate_ocr_data,
    find_start_finish_crossings,
    align_videos,
    AlignmentPoint,
    CorrelationResult,
    FrameOCRData,
    StartFinishCrossing,
)


@dataclass
class FrameData:
    """Pre-computed data for a single frame."""
    time: float
    frame: np.ndarray  # Original RGB frame
    circle: Optional[tuple[int, int, int]]  # (x, y, radius) or None
    segment: Optional[int]  # Segment number or None
    red_mask: Optional[np.ndarray] = None  # Binary mask of red pixels in search region
    ocr_data: Optional[FrameOCRData] = None  # OCR-extracted lap/segment/time info


@dataclass
class CrossCorrelationResult:
    """Result of cross-correlating one frame against all frames in another video."""
    time_a: float
    frame_a: np.ndarray
    circle_a: Optional[tuple[int, int, int]]
    segment_a: Optional[int]
    best_time_b: Optional[float]  # None if no match found
    best_frame_b: Optional[np.ndarray]  # None if no match found
    best_circle_b: Optional[tuple[int, int, int]]
    best_segment_b: Optional[int]
    best_distance: Optional[float]  # None if no match found
    all_distances: list[tuple[float, float]]  # (time_b, distance) for all frames in B
    red_mask_a: Optional[np.ndarray] = None  # Binary mask of red pixels for frame A
    red_mask_b: Optional[np.ndarray] = None  # Binary mask of red pixels for frame B
    circle_a_interpolated: bool = False  # True if circle_a was interpolated
    circle_b_interpolated: bool = False  # True if circle_b was interpolated
    no_match: bool = False  # True if circle in A could not be detected/interpolated
    # For debugging: global minimum (unconstrained by window)
    global_min_time_b: Optional[float] = None
    global_min_distance: Optional[float] = None
    # OCR data for both frames
    ocr_a: Optional[FrameOCRData] = None
    ocr_b: Optional[FrameOCRData] = None


def compute_red_mask(
    frame: np.ndarray,
    red_min: int = 200,
    red_max_gb: int = 80
) -> np.ndarray:
    """
    Compute a binary mask showing which pixels are considered "red" for matching.

    Args:
        frame: RGB frame as numpy array
        red_min: Minimum R value for red detection
        red_max_gb: Maximum G and B values for red detection

    Returns:
        Binary mask (H, W) where 1 indicates red pixels
    """
    h, w = frame.shape[:2]

    # Create full-frame mask initialized to 0
    mask = np.zeros((h, w), dtype=np.uint8)

    # Define search region (top-right quadrant)
    y1, y2, x1, x2 = 0, h // 2, w // 2, w

    # Extract the search region
    region = frame[y1:y2, x1:x2]

    # Create red mask for search region
    r = region[:, :, 0]
    g = region[:, :, 1]
    b = region[:, :, 2]
    red_mask = ((r >= red_min) & (g <= red_max_gb) & (b <= red_max_gb)).astype(np.uint8)

    # Place the red mask into the full-frame mask
    mask[y1:y2, x1:x2] = red_mask

    return mask


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatically align two racing videos using frame correlation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic alignment (output to stdout)
    python video_align.py video_a.mp4 video_b.mp4

    # Output to CSV file
    python video_align.py video_a.mp4 video_b.mp4 --output timestamps.csv

    # Interactive debug mode
    python video_align.py video_a.mp4 video_b.mp4 --debug
        """
    )

    parser.add_argument("video_a", help="Path to reference video (video A)")
    parser.add_argument("video_b", help="Path to target video (video B)")

    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path (default: stdout)"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable interactive debug mode with visualization"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=240,
        help="White detection threshold (default: 240)"
    )

    parser.add_argument(
        "--red-min",
        type=int,
        default=200,
        help="Minimum R value for red detection (default: 200)"
    )

    parser.add_argument(
        "--red-max-gb",
        type=int,
        default=80,
        help="Maximum G/B value for red detection (default: 80)"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Search window in frames (default: 100)"
    )

    parser.add_argument(
        "--initial-search",
        type=float,
        default=20.0,
        help="Initial search duration in seconds (default: 20.0)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample interval in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable advanced analysis (static mask, segment detection). Faster but less accurate."
    )

    parser.add_argument(
        "--short", "-s",
        action="store_true",
        help="Analyze only the first 20 seconds of video (for faster debugging)"
    )

    return parser.parse_args(argv)


def output_csv(
    alignments: list[AlignmentPoint],
    video_a_path: str,
    video_b_path: str,
    output_path: Optional[str] = None
) -> None:
    """Output alignment results as CSV."""
    # Check if any alignments have segment info
    has_segments = any(point.segment is not None for point in alignments)

    lines = [
        f"# Auto-generated alignment timestamps",
        f"# Video A: {video_a_path}",
        f"# Video B: {video_b_path}",
    ]

    if has_segments:
        lines.append("time_a,time_b,correlation,segment")
    else:
        lines.append("time_a,time_b,correlation")

    for point in alignments:
        if has_segments:
            seg = point.segment if point.segment is not None else ""
            lines.append(f"{point.time_a:.3f},{point.time_b:.3f},{point.correlation},{seg}")
        else:
            lines.append(f"{point.time_a:.3f},{point.time_b:.3f},{point.correlation}")

    output = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Alignment saved to: {output_path}", file=sys.stderr)
    else:
        print(output)


def extract_frame_data(
    video_path: str,
    interval: float,
    red_min: int,
    red_max_gb: int,
    progress_prefix: str = "",
    use_native_fps: bool = False,
    max_duration: Optional[float] = None
) -> tuple[list[FrameData], float]:
    """
    Extract frame data (frame, red circle, segment) from a video.

    Args:
        video_path: Path to video file
        interval: Sampling interval in seconds (ignored if use_native_fps=True)
        red_min: Minimum R value for red detection
        red_max_gb: Maximum G/B values for red detection
        progress_prefix: Prefix for progress output
        use_native_fps: If True, extract every frame at native video FPS
        max_duration: If set, only extract frames up to this duration in seconds

    Returns:
        Tuple of (list of FrameData, actual fps used)
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        duration, fps, frame_count = get_video_info(video)
        frames: list[FrameData] = []

        # Apply max_duration limit if specified
        effective_duration = duration
        if max_duration is not None:
            effective_duration = min(duration, max_duration)
            if effective_duration < duration:
                print(f"{progress_prefix}Limiting to first {effective_duration:.1f}s (--short mode)", file=sys.stderr)

        if use_native_fps:
            # Extract every frame at native FPS
            actual_interval = 1.0 / fps
            total_frames = int(effective_duration * fps)
        else:
            actual_interval = interval
            total_frames = int(effective_duration / interval)

        time = 0.0
        frame_num = 0

        while time < effective_duration:
            frame_num += 1
            pct = (frame_num / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r{progress_prefix}Extracting frames: {pct:5.1f}% ({frame_num}/{total_frames})   ", end="", file=sys.stderr)

            frame = get_frame_at_time(video, time)
            if frame is not None:
                circle = find_red_circle(frame, red_min, red_max_gb)
                segment = detect_segment_number(frame)
                red_mask = compute_red_mask(frame, red_min, red_max_gb)
                ocr_data = extract_frame_ocr(frame)
                frames.append(FrameData(
                    time=time,
                    frame=frame,
                    circle=circle,
                    segment=segment,
                    red_mask=red_mask,
                    ocr_data=ocr_data
                ))

            time += actual_interval

        print(file=sys.stderr)  # Newline after progress
        return frames, fps

    finally:
        video.release()


def interpolate_missing_circles(frames: list[FrameData]) -> list[Optional[tuple[int, int]]]:
    """
    Interpolate missing circle positions using linear interpolation.

    For frames where circle detection failed, estimate the position by
    linearly interpolating between the nearest frames before and after
    where the circle was successfully detected.

    Only interpolates between two known positions. Does NOT extrapolate
    at the beginning or end - those remain None (no match possible).

    Args:
        frames: List of FrameData with some circles possibly None

    Returns:
        List of (x, y) positions for each frame, or None if cannot be determined
    """
    n = len(frames)
    positions: list[Optional[tuple[int, int]]] = []

    # Extract known positions
    for f in frames:
        if f.circle is not None:
            positions.append((f.circle[0], f.circle[1]))
        else:
            positions.append(None)

    # Find indices where we have valid positions
    known_indices = [i for i, p in enumerate(positions) if p is not None]

    if not known_indices:
        # No circles detected at all - return None for all
        return [None] * n

    # Interpolate missing positions (only between known positions, not extrapolate)
    result: list[Optional[tuple[int, int]]] = []

    for i in range(n):
        if positions[i] is not None:
            result.append(positions[i])
        else:
            # Find nearest known position before and after
            prev_idx = None
            next_idx = None

            for ki in known_indices:
                if ki < i:
                    prev_idx = ki
                elif ki > i and next_idx is None:
                    next_idx = ki
                    break

            if prev_idx is not None and next_idx is not None:
                # Interpolate between prev and next
                prev_pos = positions[prev_idx]
                next_pos = positions[next_idx]
                t = (i - prev_idx) / (next_idx - prev_idx)
                x = int(prev_pos[0] + t * (next_pos[0] - prev_pos[0]))
                y = int(prev_pos[1] + t * (next_pos[1] - prev_pos[1]))
                result.append((x, y))
            else:
                # Cannot interpolate - at start or end without bracketing positions
                # Return None to indicate no match possible
                result.append(None)

    return result


def compute_cross_correlations(
    frames_a: list[FrameData],
    frames_b: list[FrameData],
    window_seconds: float = 20.0,
    crossings_a: Optional[list[StartFinishCrossing]] = None,
    crossings_b: Optional[list[StartFinishCrossing]] = None
) -> list[CrossCorrelationResult]:
    """
    Compute cross-correlation for each frame in A against frames in B.

    Uses Euclidean distance between circle centers as the correlation metric.
    Lower distance = better match.

    Only matches against frames within +/- window_seconds of the current
    time in video A to avoid matching with later laps that have similar positions.

    Start-finish crossings provide hard synchronization constraints: when both
    videos have a crossing for the same lap transition (e.g., lap 1->2), those
    crossings must be aligned. This provides anchor points for matching.

    Missing circle positions are interpolated from neighboring frames.
    If a circle position cannot be determined (even with interpolation),
    that frame is marked as no_match.

    The best match time never goes backwards - if multiple frames have
    the same distance, we pick the one that maintains forward progress.

    Args:
        frames_a: Frames from video A
        frames_b: Frames from video B
        window_seconds: Match against frames within +/- this many seconds
        crossings_a: Start-finish crossings detected in video A
        crossings_b: Start-finish crossings detected in video B
    """
    # Build crossing offset map: find matching lap transitions
    # If video A crosses lap 1->2 at time T_a and video B at T_b,
    # then the time offset is (T_b - T_a) for frames in that lap
    crossing_offsets: list[tuple[float, float, float]] = []  # (time_a_start, time_a_end, offset)

    if crossings_a and crossings_b:
        # Match crossings by lap number
        for ca in crossings_a:
            if ca.lap_before is None or ca.lap_after is None:
                continue
            for cb in crossings_b:
                if cb.lap_before == ca.lap_before and cb.lap_after == ca.lap_after:
                    # Matching crossing found!
                    # offset = time_b - time_a (add to time_a to get time_b)
                    offset = cb.video_time - ca.video_time
                    # This offset applies from this crossing until the next
                    crossing_offsets.append((ca.video_time, float('inf'), offset))
                    print(f"\n  Crossing sync: Lap {ca.lap_before}->{ca.lap_after} "
                          f"A={ca.video_time:.2f}s B={cb.video_time:.2f}s offset={offset:+.2f}s",
                          file=sys.stderr)
                    break

        # Update end times for each offset region
        crossing_offsets.sort(key=lambda x: x[0])
        for i in range(len(crossing_offsets) - 1):
            start, _, offset = crossing_offsets[i]
            next_start = crossing_offsets[i + 1][0]
            crossing_offsets[i] = (start, next_start, offset)
    print("\nInterpolating missing circle positions...", file=sys.stderr)

    # Interpolate missing circles for both videos
    positions_a = interpolate_missing_circles(frames_a)
    positions_b = interpolate_missing_circles(frames_b)

    # Count statistics
    detected_a = sum(1 for f in frames_a if f.circle is not None)
    detected_b = sum(1 for f in frames_b if f.circle is not None)
    interp_a = sum(1 for i, p in enumerate(positions_a) if p is not None and frames_a[i].circle is None)
    interp_b = sum(1 for i, p in enumerate(positions_b) if p is not None and frames_b[i].circle is None)
    unmatchable_a = sum(1 for p in positions_a if p is None)
    unmatchable_b = sum(1 for p in positions_b if p is None)

    print(f"  Video A: {detected_a} detected, {interp_a} interpolated, {unmatchable_a} unmatchable", file=sys.stderr)
    print(f"  Video B: {detected_b} detected, {interp_b} interpolated, {unmatchable_b} unmatchable", file=sys.stderr)
    print(f"  Matching window: +/- {window_seconds:.0f} seconds", file=sys.stderr)

    # === VECTORIZED DISTANCE COMPUTATION ===
    # Pre-compute all pairwise distances using NumPy broadcasting
    print("  Computing distance matrix...", file=sys.stderr)

    n_a = len(frames_a)
    n_b = len(frames_b)

    # Create position arrays with NaN for missing positions
    # Shape: (n_a, 2) and (n_b, 2)
    pos_array_a = np.full((n_a, 2), np.nan, dtype=np.float64)
    pos_array_b = np.full((n_b, 2), np.nan, dtype=np.float64)

    for i, pos in enumerate(positions_a):
        if pos is not None:
            pos_array_a[i] = pos

    for j, pos in enumerate(positions_b):
        if pos is not None:
            pos_array_b[j] = pos

    # Compute all pairwise distances using broadcasting
    # pos_array_a[:, None, :] has shape (n_a, 1, 2)
    # pos_array_b[None, :, :] has shape (1, n_b, 2)
    # diff has shape (n_a, n_b, 2)
    diff = pos_array_a[:, None, :] - pos_array_b[None, :, :]
    # distance_matrix has shape (n_a, n_b)
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    # Replace NaN distances with sentinel value (10000.0)
    distance_matrix = np.where(np.isnan(distance_matrix), 10000.0, distance_matrix)

    # Create masks for valid positions
    valid_a = ~np.isnan(pos_array_a[:, 0])  # Shape: (n_a,)
    valid_b = ~np.isnan(pos_array_b[:, 0])  # Shape: (n_b,)

    # Pre-compute time arrays for vectorized window operations
    times_a = np.array([f.time for f in frames_a])
    times_b = np.array([f.time for f in frames_b])

    # Pre-compute crossing offsets array for vectorized lookup
    # For each frame in A, determine the window center offset
    window_offsets = np.zeros(n_a, dtype=np.float64)
    if crossing_offsets:
        for i, time_a in enumerate(times_a):
            for start_a, end_a, offset in crossing_offsets:
                if start_a <= time_a < end_a:
                    window_offsets[i] = offset
                    break

    print("  Finding best matches...", file=sys.stderr)

    results: list[CrossCorrelationResult] = []
    prev_best_time: Optional[float] = None  # Track previous best time to prevent backwards matching

    for i, fa in enumerate(frames_a):
        pct = ((i + 1) / n_a) * 100
        print(f"\rCross-correlating: {pct:5.1f}% ({i + 1}/{n_a})   ", end="", file=sys.stderr)

        time_a = fa.time

        # If we can't determine position for frame A, mark as no match
        if not valid_a[i]:
            results.append(CrossCorrelationResult(
                time_a=fa.time,
                frame_a=fa.frame,
                circle_a=None,
                segment_a=fa.segment,
                best_time_b=None,
                best_frame_b=None,
                best_circle_b=None,
                best_segment_b=None,
                best_distance=None,
                all_distances=[],
                red_mask_a=fa.red_mask,
                red_mask_b=None,
                circle_a_interpolated=False,
                circle_b_interpolated=False,
                no_match=True,
                ocr_a=fa.ocr_data,
                ocr_b=None
            ))
            continue

        # Get pre-computed distances for this frame
        distances = distance_matrix[i]  # Shape: (n_b,)

        # Build all_distances list for result (needed for visualization)
        all_distances = list(zip(times_b.tolist(), distances.tolist()))

        # Find global minimum (unconstrained)
        global_min_idx = int(np.argmin(distances))
        global_min_distance = distances[global_min_idx]

        # Determine window center based on crossing offsets
        window_center = time_a + window_offsets[i]

        # Create window mask (vectorized)
        window_mask = (times_b >= window_center - window_seconds) & (times_b <= window_center + window_seconds)

        # Find best match within window
        if not np.any(window_mask):
            # No frames in window
            pos_a = positions_a[i]
            results.append(CrossCorrelationResult(
                time_a=fa.time,
                frame_a=fa.frame,
                circle_a=fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5),
                segment_a=fa.segment,
                best_time_b=None,
                best_frame_b=None,
                best_circle_b=None,
                best_segment_b=None,
                best_distance=None,
                all_distances=all_distances,
                red_mask_a=fa.red_mask,
                red_mask_b=None,
                circle_a_interpolated=fa.circle is None,
                circle_b_interpolated=False,
                no_match=True,
                global_min_time_b=frames_b[global_min_idx].time if global_min_distance < 10000 else None,
                global_min_distance=global_min_distance if global_min_distance < 10000 else None,
                ocr_a=fa.ocr_data,
                ocr_b=None
            ))
            continue

        # Apply window mask and find minimum
        windowed_distances = np.where(window_mask, distances, np.inf)
        best_idx = int(np.argmin(windowed_distances))
        best_distance = windowed_distances[best_idx]

        # If no valid match found in window
        if best_distance == np.inf:
            pos_a = positions_a[i]
            results.append(CrossCorrelationResult(
                time_a=fa.time,
                frame_a=fa.frame,
                circle_a=fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5),
                segment_a=fa.segment,
                best_time_b=None,
                best_frame_b=None,
                best_circle_b=None,
                best_segment_b=None,
                best_distance=None,
                all_distances=all_distances,
                red_mask_a=fa.red_mask,
                red_mask_b=None,
                circle_a_interpolated=fa.circle is None,
                circle_b_interpolated=False,
                no_match=True,
                global_min_time_b=frames_b[global_min_idx].time if global_min_distance < 10000 else None,
                global_min_distance=global_min_distance if global_min_distance < 10000 else None,
                ocr_a=fa.ocr_data,
                ocr_b=None
            ))
            continue

        # Forward-progress constraint: if best match would go backwards, find best forward match
        if prev_best_time is not None and times_b[best_idx] < prev_best_time:
            # Find best match that doesn't go backwards (within window)
            forward_mask = window_mask & (times_b >= prev_best_time)
            if np.any(forward_mask):
                forward_distances = np.where(forward_mask, distances, np.inf)
                best_idx = int(np.argmin(forward_distances))
                best_distance = forward_distances[best_idx]

        best_fb = frames_b[best_idx]
        prev_best_time = best_fb.time  # Update for next iteration

        # Track whether positions are interpolated or detected
        circle_a_interpolated = fa.circle is None
        circle_b_interpolated = best_fb.circle is None

        # Store the interpolated position if original was None
        pos_a = positions_a[i]
        circle_a = fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5)
        pos_b = positions_b[best_idx]
        circle_b = best_fb.circle if best_fb.circle is not None else (pos_b[0], pos_b[1], 5) if pos_b is not None else None

        # Determine global min info for debugging
        global_min_time = frames_b[global_min_idx].time if global_min_distance < 10000 else None
        global_min_dist = global_min_distance if global_min_distance < 10000 else None

        results.append(CrossCorrelationResult(
            time_a=fa.time,
            frame_a=fa.frame,
            circle_a=circle_a,
            segment_a=fa.segment,
            best_time_b=best_fb.time,
            best_frame_b=best_fb.frame,
            best_circle_b=circle_b,
            best_segment_b=best_fb.segment,
            best_distance=best_distance,
            all_distances=all_distances,
            red_mask_a=fa.red_mask,
            red_mask_b=best_fb.red_mask,
            circle_a_interpolated=circle_a_interpolated,
            circle_b_interpolated=circle_b_interpolated,
            no_match=False,
            global_min_time_b=global_min_time,
            global_min_distance=global_min_dist,
            ocr_a=fa.ocr_data,
            ocr_b=best_fb.ocr_data
        ))

    print(file=sys.stderr)  # Newline after progress
    return results


def run_debug_mode(
    video_a_path: str,
    video_b_path: str,
    threshold: int,
    red_min: int,
    red_max_gb: int,
    window_frames: int,
    initial_search_seconds: float,
    interval: float,
    short_mode: bool = False
) -> None:
    """Run interactive debug mode with pre-computed cross-correlations."""
    import os

    # Check if both videos are the same file
    same_file = os.path.realpath(video_a_path) == os.path.realpath(video_b_path)

    print("=" * 60, file=sys.stderr)
    print("PHASE 1: Extracting frame data from videos (native FPS)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Determine max duration based on short mode
    max_duration = 20.0 if short_mode else None

    # Extract all frame data from both videos at native FPS
    print(f"\nVideo A: {video_a_path}", file=sys.stderr)
    frames_a, fps_a = extract_frame_data(video_a_path, interval, red_min, red_max_gb, "  [A] ", use_native_fps=True, max_duration=max_duration)

    if same_file:
        print(f"\nVideo B: same as Video A (reusing extracted data)", file=sys.stderr)
        # Reuse frames_a for frames_b - they're the same video
        frames_b = frames_a
        fps_b = fps_a
    else:
        print(f"\nVideo B: {video_b_path}", file=sys.stderr)
        frames_b, fps_b = extract_frame_data(video_b_path, interval, red_min, red_max_gb, "  [B] ", use_native_fps=True, max_duration=max_duration)

    print(f"\nExtracted {len(frames_a)} frames from A ({fps_a:.1f} fps), {len(frames_b)} frames from B ({fps_b:.1f} fps)", file=sys.stderr)
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
        display = create_debug_display_v2(result, len(results), current_idx)

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


def apply_mask_overlay(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    considered_opacity: float = 1.0,
    masked_opacity: float = 0.25
) -> np.ndarray:
    """
    Apply a visual overlay to show which pixels are being considered.

    Pixels where mask=1 keep full opacity (considered for matching).
    Pixels where mask=0 are dimmed to masked_opacity (not considered).

    Args:
        frame: BGR frame as numpy array
        mask: Binary mask (H, W) where 1 = considered, 0 = not considered
        considered_opacity: Opacity for considered pixels (default: 1.0)
        masked_opacity: Opacity for masked-out pixels (default: 0.25)

    Returns:
        Frame with mask overlay applied
    """
    if mask is None:
        return frame

    # Resize mask to match frame if needed
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create output frame
    result = frame.copy().astype(np.float32)

    # Apply opacity based on mask
    # Where mask=0, dim the pixels to masked_opacity
    # Where mask=1, keep at considered_opacity
    mask_3ch = np.stack([mask, mask, mask], axis=2).astype(np.float32)

    # Blend: result = frame * (mask * considered_opacity + (1-mask) * masked_opacity)
    opacity = mask_3ch * considered_opacity + (1 - mask_3ch) * masked_opacity
    result = result * opacity

    return result.astype(np.uint8)


def create_debug_display_v2(
    result: CrossCorrelationResult,
    total_results: int,
    current_idx: int
) -> np.ndarray:
    """Create debug visualization from pre-computed cross-correlation result."""
    frame_a = result.frame_a

    # Get frame dimensions
    h, w = frame_a.shape[:2]

    # Handle no-match case - show black frame for video B
    if result.no_match or result.best_frame_b is None:
        frame_b = np.zeros_like(frame_a)  # Black frame
    else:
        frame_b = result.best_frame_b

    # Scale frames to fit display (max 640 width each)
    max_frame_width = 640
    scale = min(1.0, max_frame_width / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frames
    frame_a_resized = cv2.resize(frame_a, (new_w, new_h))
    frame_b_resized = cv2.resize(frame_b, (new_w, new_h))

    # Convert RGB to BGR for OpenCV display
    frame_a_bgr = cv2.cvtColor(frame_a_resized, cv2.COLOR_RGB2BGR)
    frame_b_bgr = cv2.cvtColor(frame_b_resized, cv2.COLOR_RGB2BGR)

    # Apply mask overlay to show which pixels are being considered
    # Resize masks to match resized frames
    if result.red_mask_a is not None:
        mask_a_resized = cv2.resize(result.red_mask_a, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_a_bgr = apply_mask_overlay(frame_a_bgr, mask_a_resized)
    if result.red_mask_b is not None:
        mask_b_resized = cv2.resize(result.red_mask_b, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_b_bgr = apply_mask_overlay(frame_b_bgr, mask_b_resized)

    # Draw detected red circles (scaled to resized frame)
    # Green = detected, Yellow = interpolated
    if result.circle_a is not None:
        cx, cy, r = result.circle_a
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Use yellow for interpolated, green for detected
        color_a = (0, 255, 255) if result.circle_a_interpolated else (0, 255, 0)
        cv2.circle(frame_a_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_a, 2)

    if result.best_circle_b is not None:
        cx, cy, r = result.best_circle_b
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Use yellow for interpolated, green for detected
        color_b = (0, 255, 255) if result.circle_b_interpolated else (0, 255, 0)
        cv2.circle(frame_b_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_b, 2)

    # Add labels to frames
    cv2.putText(frame_a_bgr, f"Video A: {result.time_a:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if result.no_match or result.best_time_b is None:
        cv2.putText(frame_b_bgr, "Video B: NO MATCH", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text
    else:
        cv2.putText(frame_b_bgr, f"Video B: {result.best_time_b:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add segment labels if detected
    segment_a_str = f"Seg: {result.segment_a}" if result.segment_a is not None else "Seg: ?"
    segment_b_str = f"Seg: {result.best_segment_b}" if result.best_segment_b is not None else "Seg: ?"
    cv2.putText(frame_a_bgr, segment_a_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_b_bgr, segment_b_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add circle position info (with interpolation indicator)
    if result.circle_a is not None:
        interp_a = " [INTERP]" if result.circle_a_interpolated else ""
        circle_a_str = f"Circle: ({result.circle_a[0]}, {result.circle_a[1]}){interp_a}"
    else:
        circle_a_str = "Circle: Not found"
    if result.best_circle_b is not None:
        interp_b = " [INTERP]" if result.circle_b_interpolated else ""
        circle_b_str = f"Circle: ({result.best_circle_b[0]}, {result.best_circle_b[1]}){interp_b}"
    else:
        circle_b_str = "Circle: Not found"
    # Use yellow text for interpolated, cyan for detected
    color_text_a = (0, 255, 255) if result.circle_a_interpolated else (0, 200, 200)
    color_text_b = (0, 255, 255) if result.circle_b_interpolated else (0, 200, 200)
    cv2.putText(frame_a_bgr, circle_a_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_a, 1)
    cv2.putText(frame_b_bgr, circle_b_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_b, 1)

    # Stack frames horizontally
    frames_row = np.hstack([frame_a_bgr, frame_b_bgr])

    # Create OCR info bar beneath frames
    ocr_bar_height = 50
    ocr_bar_width = new_w * 2
    ocr_bar = np.zeros((ocr_bar_height, ocr_bar_width, 3), dtype=np.uint8)
    ocr_bar[:] = (50, 50, 50)  # Slightly lighter gray

    # Format OCR data for frame A
    ocr_a = result.ocr_a
    if ocr_a is not None:
        # Line 1: Lap info
        if ocr_a.is_optimal_lap:
            lap_str_a = "OPTIMAL LAP"
        elif ocr_a.lap_number is not None:
            lap_str_a = f"LAP {ocr_a.lap_number}"
        else:
            lap_str_a = "LAP ?"
        seg_str_a = f"SEG {ocr_a.segment_number}" if ocr_a.segment_number is not None else "SEG ?"

        # Line 2: Lap time
        if ocr_a.lap_time_seconds is not None:
            lap_time_a = format_lap_time(ocr_a.lap_time_seconds)
        else:
            lap_time_a = "--:--.--"

        cv2.putText(ocr_bar, f"{lap_str_a} | {seg_str_a}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(ocr_bar, f"Lap Time: {lap_time_a}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(ocr_bar, "OCR: No data", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Format OCR data for frame B
    ocr_b = result.ocr_b
    if ocr_b is not None:
        # Line 1: Lap info
        if ocr_b.is_optimal_lap:
            lap_str_b = "OPTIMAL LAP"
        elif ocr_b.lap_number is not None:
            lap_str_b = f"LAP {ocr_b.lap_number}"
        else:
            lap_str_b = "LAP ?"
        seg_str_b = f"SEG {ocr_b.segment_number}" if ocr_b.segment_number is not None else "SEG ?"

        # Line 2: Lap time
        if ocr_b.lap_time_seconds is not None:
            lap_time_b = format_lap_time(ocr_b.lap_time_seconds)
        else:
            lap_time_b = "--:--.--"

        cv2.putText(ocr_bar, f"{lap_str_b} | {seg_str_b}", (new_w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(ocr_bar, f"Lap Time: {lap_time_b}", (new_w + 10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    elif not result.no_match:
        cv2.putText(ocr_bar, "OCR: No data", (new_w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Create distance graph (shows distance to all frames in B)
    graph_height = 200
    graph_width = new_w * 2
    graph = create_distance_graph(
        result.all_distances,
        result.best_time_b,
        result.best_distance,
        result.global_min_time_b,
        result.global_min_distance,
        graph_width,
        graph_height
    )

    # Create info bar with more details
    info_height = 60
    info_bar = np.zeros((info_height, graph_width, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)

    # First line: times and distance
    if result.no_match or result.best_time_b is None:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | NO MATCH (circle not detected)"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)  # Red text
    else:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | Time B: {result.best_time_b:.2f}s | Distance: {result.best_distance:.1f}px"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Second line: segment match and controls
    if result.no_match:
        seg_match = "Segment: N/A"
    else:
        seg_match = "Segment: MATCH" if result.segment_a == result.best_segment_b and result.segment_a is not None else "Segment: mismatch"
    info_text2 = f"{seg_match} | Controls: ←→ (1 frame), ↑↓ (1 sec), ESC (exit)"
    cv2.putText(info_bar, info_text2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Stack everything vertically
    display = np.vstack([frames_row, ocr_bar, graph, info_bar])

    return display


def format_lap_time(seconds: float) -> str:
    """Format lap time in seconds to mm:ss.ss format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def format_exp(value: float) -> str:
    """Format a float value in exponential notation with 3 significant figures."""
    if value == 0:
        return "0.00e+0"
    return f"{value:.2e}"


def create_distance_graph(
    all_distances: list[tuple[float, float]],
    best_time: Optional[float],
    best_distance: Optional[float],
    global_min_time: Optional[float],
    global_min_distance: Optional[float],
    width: int,
    height: int
) -> np.ndarray:
    """
    Create a graph showing distance to all frames in video B.

    Shows the constrained best match (green dot) and the global minimum (yellow dot)
    if they differ. Labels each with exponential notation values.
    """
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)  # Dark gray background

    if not all_distances:
        return graph

    times = [d[0] for d in all_distances]
    distances = [d[1] for d in all_distances]

    # Cap distances for display (ignore 10000 sentinel values)
    display_distances = [min(d, 500) for d in distances]

    min_dist = 0
    max_dist = max(d for d in display_distances if d < 10000) if any(d < 10000 for d in display_distances) else 500
    if max_dist == min_dist:
        max_dist = min_dist + 1

    # Calculate graph margins
    margin_left = 60
    margin_right = 20
    margin_top = 20
    margin_bottom = 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Draw axis lines
    cv2.line(graph, (margin_left, margin_top),
             (margin_left, height - margin_bottom), (100, 100, 100), 1)
    cv2.line(graph, (margin_left, height - margin_bottom),
             (width - margin_right, height - margin_bottom), (100, 100, 100), 1)

    # Build mapping from time to point coordinates
    time_to_point: dict[float, tuple[int, int]] = {}
    points = []

    for i, (t, dist) in enumerate(zip(times, display_distances)):
        # X position based on time (linear mapping from times[0] to times[-1])
        if len(times) > 1:
            t_normalized = (t - times[0]) / (times[-1] - times[0])
        else:
            t_normalized = 0.5
        x = margin_left + int(t_normalized * plot_width)
        # Y position based on distance (inverted - lower distance = higher on graph)
        normalized = (dist - min_dist) / (max_dist - min_dist)
        y = margin_top + int(normalized * plot_height)
        points.append((x, y))
        time_to_point[t] = (x, y)

    # Draw line connecting points
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (100, 100, 255), 1)

    # Find constrained best point (by best_time)
    constrained_point = None
    constrained_idx = None
    if best_time is not None:
        for i, t in enumerate(times):
            if abs(t - best_time) < 0.001:  # float comparison
                constrained_point = points[i]
                constrained_idx = i
                break

    # Find global minimum point (by global_min_time)
    global_min_point = None
    global_min_idx = None
    if global_min_time is not None:
        for i, t in enumerate(times):
            if abs(t - global_min_time) < 0.001:
                global_min_point = points[i]
                global_min_idx = i
                break

    # Determine if global min differs from constrained best
    show_global_min = (
        global_min_point is not None and
        constrained_point is not None and
        global_min_idx != constrained_idx
    )

    # Draw global minimum marker (yellow) if different from constrained
    if show_global_min and global_min_point is not None and global_min_distance is not None:
        cv2.circle(graph, global_min_point, 6, (0, 255, 255), -1)  # Yellow dot
        label = f"Global: {format_exp(global_min_distance)}"
        # Position label above the point
        label_x = max(margin_left, min(global_min_point[0] - 40, width - margin_right - 100))
        label_y = max(margin_top + 15, global_min_point[1] - 10)
        cv2.putText(graph, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Draw constrained best marker (green)
    if constrained_point is not None and best_distance is not None:
        cv2.circle(graph, constrained_point, 6, (0, 255, 0), -1)  # Green dot
        label = f"Best: {format_exp(best_distance)}"
        # Position label below or to the side
        label_x = max(margin_left, min(constrained_point[0] - 30, width - margin_right - 80))
        label_y = min(height - margin_bottom - 5, constrained_point[1] + 20)
        cv2.putText(graph, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Draw Y-axis labels (distance)
    cv2.putText(graph, f"{format_exp(min_dist)}", (2, margin_top + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(graph, f"{format_exp(max_dist)}", (2, height - margin_bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Time labels
    if times:
        cv2.putText(graph, f"{times[0]:.0f}s", (margin_left, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(graph, f"{times[-1]:.0f}s", (width - margin_right - 40, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Title with legend
    title = "Distance (lower=better) | "
    cv2.putText(graph, title, (margin_left + 10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Green legend
    legend_x = margin_left + 10 + int(len(title) * 7)
    cv2.circle(graph, (legend_x, 12), 4, (0, 255, 0), -1)
    cv2.putText(graph, "Constrained", (legend_x + 8, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    # Yellow legend
    legend_x2 = legend_x + 80
    cv2.circle(graph, (legend_x2, 12), 4, (0, 255, 255), -1)
    cv2.putText(graph, "Global Min", (legend_x2 + 8, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return graph


def create_debug_display(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    binary_a: np.ndarray,
    time_a: float,
    result: CorrelationResult,
    center_time: Optional[float],
    fps_b: float,
    circle_a: Optional[tuple[int, int, int]] = None,
    circle_b: Optional[tuple[int, int, int]] = None,
    segment_a: Optional[int] = None,
    segment_b: Optional[int] = None
) -> np.ndarray:
    """Create the debug visualization display with red circle markers and segment info."""
    # Get frame dimensions
    h, w = frame_a.shape[:2]

    # Scale frames to fit display (max 640 width each)
    max_frame_width = 640
    scale = min(1.0, max_frame_width / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frames
    frame_a_resized = cv2.resize(frame_a, (new_w, new_h))
    frame_b_resized = cv2.resize(frame_b, (new_w, new_h))

    # Convert RGB to BGR for OpenCV display
    frame_a_bgr = cv2.cvtColor(frame_a_resized, cv2.COLOR_RGB2BGR)
    frame_b_bgr = cv2.cvtColor(frame_b_resized, cv2.COLOR_RGB2BGR)

    # Draw detected red circles (scaled to resized frame)
    if circle_a is not None:
        cx, cy, r = circle_a
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Draw crosshair at detected circle position
        cv2.circle(frame_a_bgr, (cx_scaled, cy_scaled), r_scaled + 5, (0, 255, 0), 2)
        cv2.line(frame_a_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), (0, 255, 0), 2)
        cv2.line(frame_a_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), (0, 255, 0), 2)

    if circle_b is not None:
        cx, cy, r = circle_b
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Draw crosshair at detected circle position
        cv2.circle(frame_b_bgr, (cx_scaled, cy_scaled), r_scaled + 5, (0, 255, 0), 2)
        cv2.line(frame_b_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), (0, 255, 0), 2)
        cv2.line(frame_b_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), (0, 255, 0), 2)

    # Add labels to frames
    cv2.putText(frame_a_bgr, f"Video A: {time_a:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_b_bgr, f"Video B: {result.best_time:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add segment labels if detected
    segment_a_str = f"Seg: {segment_a}" if segment_a is not None else "Seg: ?"
    segment_b_str = f"Seg: {segment_b}" if segment_b is not None else "Seg: ?"
    cv2.putText(frame_a_bgr, segment_a_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_b_bgr, segment_b_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add circle position info
    if circle_a is not None:
        circle_a_str = f"Circle: ({circle_a[0]}, {circle_a[1]})"
    else:
        circle_a_str = "Circle: Not found"
    if circle_b is not None:
        circle_b_str = f"Circle: ({circle_b[0]}, {circle_b[1]})"
    else:
        circle_b_str = "Circle: Not found"
    cv2.putText(frame_a_bgr, circle_a_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    cv2.putText(frame_b_bgr, circle_b_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    # Stack frames horizontally
    frames_row = np.hstack([frame_a_bgr, frame_b_bgr])

    # Create correlation graph
    graph_height = 200
    graph_width = new_w * 2
    graph = create_correlation_graph(result, graph_width, graph_height, center_time, fps_b)

    # Create info bar with more details
    info_height = 60
    info_bar = np.zeros((info_height, graph_width, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)

    # First line: times and correlation
    info_text = f"Time A: {time_a:.2f}s | Time B: {result.best_time:.2f}s | Correlation: {result.best_score}"
    cv2.putText(info_bar, info_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Second line: circle distance and segment match
    if circle_a is not None and circle_b is not None:
        dist = np.sqrt((circle_a[0] - circle_b[0])**2 + (circle_a[1] - circle_b[1])**2)
        dist_text = f"Circle dist: {dist:.1f}px"
    else:
        dist_text = "Circle dist: N/A"

    seg_match = "Segment: Match" if segment_a == segment_b and segment_a is not None else "Segment: Mismatch"
    info_text2 = f"{dist_text} | {seg_match} | Controls: ←→ (1s), ↑↓ (10s), ESC (exit)"
    cv2.putText(info_bar, info_text2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Stack everything vertically
    display = np.vstack([frames_row, graph, info_bar])

    return display


def create_correlation_graph(
    result: CorrelationResult,
    width: int,
    height: int,
    center_time: Optional[float],
    fps: float
) -> np.ndarray:
    """Create a correlation graph visualization."""
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)  # Dark gray background

    if not result.all_scores:
        return graph

    # Extract scores
    times = [s[0] for s in result.all_scores]
    scores = [s[1] for s in result.all_scores]

    if not scores:
        return graph

    # Normalize scores for display
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score > min_score else 1

    # Calculate graph margins
    margin_left = 60
    margin_right = 20
    margin_top = 20
    margin_bottom = 30
    graph_width = width - margin_left - margin_right
    graph_height = height - margin_top - margin_bottom

    # Draw axis lines
    cv2.line(graph, (margin_left, margin_top),
             (margin_left, height - margin_bottom), (100, 100, 100), 1)
    cv2.line(graph, (margin_left, height - margin_bottom),
             (width - margin_right, height - margin_bottom), (100, 100, 100), 1)

    # Draw correlation line
    points = []
    best_point_idx = 0
    best_score_val = -1

    for i, (t, score) in enumerate(zip(times, scores)):
        # X position based on index
        x = margin_left + int((i / max(len(times) - 1, 1)) * graph_width)
        # Y position based on score (inverted, higher score = higher on graph)
        y = height - margin_bottom - int(((score - min_score) / score_range) * graph_height)
        points.append((x, y))

        if score > best_score_val:
            best_score_val = score
            best_point_idx = i

    # Draw line connecting points
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (0, 200, 0), 1)

    # Draw best match marker
    if points:
        best_point = points[best_point_idx]
        cv2.circle(graph, best_point, 6, (0, 0, 255), -1)  # Red dot

    # Draw labels
    cv2.putText(graph, f"{max_score}", (5, margin_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(graph, f"{min_score}", (5, height - margin_bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Time labels
    if times:
        cv2.putText(graph, f"{times[0]:.1f}s", (margin_left, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(graph, f"{times[-1]:.1f}s", (width - margin_right - 40, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Title
    cv2.putText(graph, "Correlation (frames in Video B)", (width // 2 - 100, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return graph


def run_basic_mode(args: argparse.Namespace) -> None:
    """Run basic alignment mode (non-interactive)."""
    last_message = [""]

    def progress_callback(current: float, total: float, message: str = "Processing"):
        pct = (current / total) * 100 if total > 0 else 0
        if message != last_message[0]:
            print(file=sys.stderr)  # Newline when message changes
            last_message[0] = message
        print(f"\r{message}: {pct:5.1f}% ({current:.1f}s / {total:.1f}s)   ", end="", file=sys.stderr)

    use_analysis = not args.no_analysis
    if use_analysis:
        print("Analyzing videos (this may take a while)...", file=sys.stderr)
    else:
        print("Aligning videos (basic mode)...", file=sys.stderr)

    alignments = align_videos(
        video_a_path=args.video_a,
        video_b_path=args.video_b,
        sample_interval=args.interval,
        window_frames=args.window,
        initial_search_seconds=args.initial_search,
        white_threshold=args.threshold,
        red_min=args.red_min,
        red_max_gb=args.red_max_gb,
        progress_callback=progress_callback,
        use_analysis=use_analysis
    )

    print(file=sys.stderr)  # Newline after progress

    output_csv(alignments, args.video_a, args.video_b, args.output)


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    if args.debug:
        run_debug_mode(
            video_a_path=args.video_a,
            video_b_path=args.video_b,
            threshold=args.threshold,
            red_min=args.red_min,
            red_max_gb=args.red_max_gb,
            window_frames=args.window,
            initial_search_seconds=args.initial_search,
            interval=args.interval,
            short_mode=args.short
        )
    else:
        run_basic_mode(args)


if __name__ == "__main__":
    main()
