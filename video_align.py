#!/usr/bin/env python3
"""
Video alignment utility for automatically syncing two racing videos.

This tool uses autocorrelation of binarized frames to find matching
timestamps between videos based on track overlay and car position markers.

Usage:
    python video_align.py video_a.mp4 video_b.mp4 --output timestamps.csv
    python video_align.py video_a.mp4 video_b.mp4 --debug
"""

import argparse
import sys
import cv2
import numpy as np
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
    """Run interactive debug mode with visualization."""
    video_a = cv2.VideoCapture(video_a_path)
    video_b = cv2.VideoCapture(video_b_path)

    if not video_a.isOpened():
        print(f"Error: Could not open video A: {video_a_path}", file=sys.stderr)
        sys.exit(1)
    if not video_b.isOpened():
        print(f"Error: Could not open video B: {video_b_path}", file=sys.stderr)
        sys.exit(1)

    try:
        duration_a, fps_a, _ = get_video_info(video_a)
        duration_b, fps_b, _ = get_video_info(video_b)

        print(f"Video A: {duration_a:.1f}s @ {fps_a:.1f} fps")
        print(f"Video B: {duration_b:.1f}s @ {fps_b:.1f} fps")
        print()
        print("Controls:")
        print("  → (Right Arrow): Advance 1 second")
        print("  ← (Left Arrow): Go back 1 second")
        print("  ↑ (Up Arrow): Advance 10 seconds")
        print("  ↓ (Down Arrow): Go back 10 seconds")
        print("  ESC: Exit")
        print()

        current_time_a = 0.0
        previous_best_time_b: Optional[float] = None

        while True:
            # Get frame from video A
            frame_a = get_frame_at_time(video_a, current_time_a)
            if frame_a is None:
                print(f"Warning: Could not read frame at {current_time_a:.1f}s")
                current_time_a = min(current_time_a + interval, duration_a - 0.1)
                continue

            # Detect red circle in frame A
            circle_a = find_red_circle(frame_a, red_min, red_max_gb)

            # Detect segment number in frame A
            segment_a = detect_segment_number(frame_a)

            # Binarize frame A
            binary_a = binarize_frame(frame_a, threshold, red_min, red_max_gb)

            # Find best match in video B
            result = find_best_match(
                binary_a,
                video_b,
                center_time=previous_best_time_b,
                window_frames=window_frames,
                initial_search_seconds=initial_search_seconds,
                white_threshold=threshold,
                red_min=red_min,
                red_max_gb=red_max_gb
            )

            # Get best matching frame from video B
            frame_b = get_frame_at_time(video_b, result.best_time)
            if frame_b is None:
                frame_b = np.zeros_like(frame_a)

            # Detect red circle and segment in frame B
            circle_b = find_red_circle(frame_b, red_min, red_max_gb)
            segment_b = detect_segment_number(frame_b)

            # Create visualization
            display = create_debug_display(
                frame_a, frame_b, binary_a,
                current_time_a, result,
                previous_best_time_b, fps_b,
                circle_a, circle_b,
                segment_a, segment_b
            )

            # Show display
            cv2.imshow("Video Alignment Debug", display)

            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 83 or key == 3:  # Right arrow
                current_time_a = min(current_time_a + interval, duration_a - 0.1)
                previous_best_time_b = result.best_time
            elif key == 81 or key == 2:  # Left arrow
                current_time_a = max(current_time_a - interval, 0.0)
                # When going back, we need to reset tracking
                if current_time_a == 0.0:
                    previous_best_time_b = None
            elif key == 82 or key == 0:  # Up arrow - advance 10 seconds
                current_time_a = min(current_time_a + 10.0, duration_a - 0.1)
                previous_best_time_b = result.best_time
            elif key == 84 or key == 1:  # Down arrow - go back 10 seconds
                current_time_a = max(current_time_a - 10.0, 0.0)
                if current_time_a == 0.0:
                    previous_best_time_b = None

    finally:
        video_a.release()
        video_b.release()
        cv2.destroyAllWindows()


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
