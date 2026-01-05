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
    align_videos,
    AlignmentPoint,
    CorrelationResult,
)


@dataclass
class FrameData:
    """Pre-computed data for a single frame."""
    time: float
    frame: np.ndarray  # Original RGB frame
    circle: Optional[tuple[int, int, int]]  # (x, y, radius) or None
    segment: Optional[int]  # Segment number or None
    red_mask: Optional[np.ndarray] = None  # Binary mask of red pixels in search region


@dataclass
class CrossCorrelationResult:
    """Result of cross-correlating one frame against all frames in another video."""
    time_a: float
    frame_a: np.ndarray
    circle_a: Optional[tuple[int, int, int]]
    segment_a: Optional[int]
    best_time_b: float
    best_frame_b: np.ndarray
    best_circle_b: Optional[tuple[int, int, int]]
    best_segment_b: Optional[int]
    best_distance: float
    all_distances: list[tuple[float, float]]  # (time_b, distance) for all frames in B
    red_mask_a: Optional[np.ndarray] = None  # Binary mask of red pixels for frame A
    red_mask_b: Optional[np.ndarray] = None  # Binary mask of red pixels for frame B


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
    use_native_fps: bool = False
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

    Returns:
        Tuple of (list of FrameData, actual fps used)
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        duration, fps, frame_count = get_video_info(video)
        frames: list[FrameData] = []

        if use_native_fps:
            # Extract every frame at native FPS
            actual_interval = 1.0 / fps
            total_frames = frame_count
        else:
            actual_interval = interval
            total_frames = int(duration / interval)

        time = 0.0
        frame_num = 0

        while time < duration:
            frame_num += 1
            pct = (frame_num / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r{progress_prefix}Extracting frames: {pct:5.1f}% ({frame_num}/{total_frames})   ", end="", file=sys.stderr)

            frame = get_frame_at_time(video, time)
            if frame is not None:
                circle = find_red_circle(frame, red_min, red_max_gb)
                segment = detect_segment_number(frame)
                red_mask = compute_red_mask(frame, red_min, red_max_gb)
                frames.append(FrameData(
                    time=time,
                    frame=frame,
                    circle=circle,
                    segment=segment,
                    red_mask=red_mask
                ))

            time += actual_interval

        print(file=sys.stderr)  # Newline after progress
        return frames, fps

    finally:
        video.release()


def compute_cross_correlations(
    frames_a: list[FrameData],
    frames_b: list[FrameData]
) -> list[CrossCorrelationResult]:
    """
    Compute cross-correlation for each frame in A against ALL frames in B.

    Uses red circle position distance as the correlation metric.
    Lower distance = better match.
    """
    results: list[CrossCorrelationResult] = []

    for i, fa in enumerate(frames_a):
        pct = ((i + 1) / len(frames_a)) * 100
        print(f"\rCross-correlating: {pct:5.1f}% ({i + 1}/{len(frames_a)})   ", end="", file=sys.stderr)

        all_distances: list[tuple[float, float]] = []
        best_distance = float('inf')
        best_idx = 0

        for j, fb in enumerate(frames_b):
            # Compute distance between red circles
            if fa.circle is not None and fb.circle is not None:
                dx = fa.circle[0] - fb.circle[0]
                dy = fa.circle[1] - fb.circle[1]
                distance = np.sqrt(dx * dx + dy * dy)
            else:
                # If either circle is missing, use a large distance
                distance = 10000.0

            all_distances.append((fb.time, distance))

            if distance < best_distance:
                best_distance = distance
                best_idx = j

        best_fb = frames_b[best_idx]
        results.append(CrossCorrelationResult(
            time_a=fa.time,
            frame_a=fa.frame,
            circle_a=fa.circle,
            segment_a=fa.segment,
            best_time_b=best_fb.time,
            best_frame_b=best_fb.frame,
            best_circle_b=best_fb.circle,
            best_segment_b=best_fb.segment,
            best_distance=best_distance,
            all_distances=all_distances,
            red_mask_a=fa.red_mask,
            red_mask_b=best_fb.red_mask
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
    interval: float
) -> None:
    """Run interactive debug mode with pre-computed cross-correlations."""
    import os

    # Check if both videos are the same file
    same_file = os.path.realpath(video_a_path) == os.path.realpath(video_b_path)

    print("=" * 60, file=sys.stderr)
    print("PHASE 1: Extracting frame data from videos (native FPS)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Extract all frame data from both videos at native FPS
    print(f"\nVideo A: {video_a_path}", file=sys.stderr)
    frames_a, fps_a = extract_frame_data(video_a_path, interval, red_min, red_max_gb, "  [A] ", use_native_fps=True)

    if same_file:
        print(f"\nVideo B: same as Video A (reusing extracted data)", file=sys.stderr)
        # Reuse frames_a for frames_b - they're the same video
        frames_b = frames_a
        fps_b = fps_a
    else:
        print(f"\nVideo B: {video_b_path}", file=sys.stderr)
        frames_b, fps_b = extract_frame_data(video_b_path, interval, red_min, red_max_gb, "  [B] ", use_native_fps=True)

    print(f"\nExtracted {len(frames_a)} frames from A ({fps_a:.1f} fps), {len(frames_b)} frames from B ({fps_b:.1f} fps)", file=sys.stderr)
    if same_file:
        print("(Same file optimization: frames extracted only once)", file=sys.stderr)

    # Count frames with detected circles
    circles_a = sum(1 for f in frames_a if f.circle is not None)
    circles_b = sum(1 for f in frames_b if f.circle is not None)
    print(f"Red circles detected: A={circles_a}/{len(frames_a)}, B={circles_b}/{len(frames_b)}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("PHASE 2: Computing cross-correlations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Pre-compute all cross-correlations
    results = compute_cross_correlations(frames_a, frames_b)

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
    frame_b = result.best_frame_b

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

    # Apply mask overlay to show which pixels are being considered
    # Resize masks to match resized frames
    if result.red_mask_a is not None:
        mask_a_resized = cv2.resize(result.red_mask_a, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_a_bgr = apply_mask_overlay(frame_a_bgr, mask_a_resized)
    if result.red_mask_b is not None:
        mask_b_resized = cv2.resize(result.red_mask_b, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_b_bgr = apply_mask_overlay(frame_b_bgr, mask_b_resized)

    # Draw detected red circles (scaled to resized frame)
    if result.circle_a is not None:
        cx, cy, r = result.circle_a
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Draw crosshair at detected circle position
        cv2.circle(frame_a_bgr, (cx_scaled, cy_scaled), r_scaled + 5, (0, 255, 0), 2)
        cv2.line(frame_a_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), (0, 255, 0), 2)
        cv2.line(frame_a_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), (0, 255, 0), 2)

    if result.best_circle_b is not None:
        cx, cy, r = result.best_circle_b
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        # Draw crosshair at detected circle position
        cv2.circle(frame_b_bgr, (cx_scaled, cy_scaled), r_scaled + 5, (0, 255, 0), 2)
        cv2.line(frame_b_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), (0, 255, 0), 2)
        cv2.line(frame_b_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), (0, 255, 0), 2)

    # Add labels to frames
    cv2.putText(frame_a_bgr, f"Video A: {result.time_a:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_b_bgr, f"Video B: {result.best_time_b:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add segment labels if detected
    segment_a_str = f"Seg: {result.segment_a}" if result.segment_a is not None else "Seg: ?"
    segment_b_str = f"Seg: {result.best_segment_b}" if result.best_segment_b is not None else "Seg: ?"
    cv2.putText(frame_a_bgr, segment_a_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_b_bgr, segment_b_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add circle position info
    if result.circle_a is not None:
        circle_a_str = f"Circle: ({result.circle_a[0]}, {result.circle_a[1]})"
    else:
        circle_a_str = "Circle: Not found"
    if result.best_circle_b is not None:
        circle_b_str = f"Circle: ({result.best_circle_b[0]}, {result.best_circle_b[1]})"
    else:
        circle_b_str = "Circle: Not found"
    cv2.putText(frame_a_bgr, circle_a_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    cv2.putText(frame_b_bgr, circle_b_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    # Stack frames horizontally
    frames_row = np.hstack([frame_a_bgr, frame_b_bgr])

    # Create distance graph (shows distance to all frames in B)
    graph_height = 200
    graph_width = new_w * 2
    graph = create_distance_graph(result.all_distances, result.best_time_b, graph_width, graph_height)

    # Create info bar with more details
    info_height = 60
    info_bar = np.zeros((info_height, graph_width, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)

    # First line: times and distance
    info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | Time B: {result.best_time_b:.2f}s | Distance: {result.best_distance:.1f}px"
    cv2.putText(info_bar, info_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Second line: segment match and controls
    seg_match = "Segment: MATCH" if result.segment_a == result.best_segment_b and result.segment_a is not None else "Segment: mismatch"
    info_text2 = f"{seg_match} | Controls: ←→ (1 frame), ↑↓ (1 sec), ESC (exit)"
    cv2.putText(info_bar, info_text2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Stack everything vertically
    display = np.vstack([frames_row, graph, info_bar])

    return display


def create_distance_graph(
    all_distances: list[tuple[float, float]],
    best_time: float,
    width: int,
    height: int
) -> np.ndarray:
    """Create a graph showing distance to all frames in video B."""
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
    graph_width = width - margin_left - margin_right
    graph_height = height - margin_top - margin_bottom

    # Draw axis lines
    cv2.line(graph, (margin_left, margin_top),
             (margin_left, height - margin_bottom), (100, 100, 100), 1)
    cv2.line(graph, (margin_left, height - margin_bottom),
             (width - margin_right, height - margin_bottom), (100, 100, 100), 1)

    # Draw distance line
    points = []
    best_point_idx = 0
    best_dist_val = float('inf')

    for i, (t, dist) in enumerate(zip(times, display_distances)):
        # X position based on index
        x = margin_left + int((i / max(len(times) - 1, 1)) * graph_width)
        # Y position based on distance (inverted - lower distance = higher on graph)
        normalized = (dist - min_dist) / (max_dist - min_dist)
        y = margin_top + int(normalized * graph_height)
        points.append((x, y))

        if distances[i] < best_dist_val:
            best_dist_val = distances[i]
            best_point_idx = i

    # Draw line connecting points
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (100, 100, 255), 1)

    # Draw best match marker (lowest distance)
    if points:
        best_point = points[best_point_idx]
        cv2.circle(graph, best_point, 6, (0, 255, 0), -1)  # Green dot for best match

    # Draw labels
    cv2.putText(graph, f"{int(min_dist)}px", (5, margin_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(graph, f"{int(max_dist)}px", (5, height - margin_bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Time labels
    if times:
        cv2.putText(graph, f"{times[0]:.0f}s", (margin_left, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(graph, f"{times[-1]:.0f}s", (width - margin_right - 40, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Title
    cv2.putText(graph, "Distance to frames in Video B (lower = better match)", (width // 2 - 150, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
            interval=args.interval
        )
    else:
        run_basic_mode(args)


if __name__ == "__main__":
    main()
