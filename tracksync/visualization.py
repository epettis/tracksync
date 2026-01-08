"""Visualization utilities for debugging and display.

This module provides functions to create debug displays, graphs, and
visual overlays for analyzing video alignment results.

Note: This module requires OpenCV (cv2) which is an optional dependency.
"""

from typing import Optional

import cv2
import numpy as np

from .frame_data import CrossCorrelationResult
from .models import TurnAnalysis


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
    mask_3ch = np.stack([mask, mask, mask], axis=2).astype(np.float32)
    opacity = mask_3ch * considered_opacity + (1 - mask_3ch) * masked_opacity
    result = result * opacity

    return result.astype(np.uint8)


def create_turn_angle_graph(
    turn_analysis: TurnAnalysis,
    current_time: float,
    width: int,
    height: int,
    label: str = "A"
) -> np.ndarray:
    """
    Create a graph showing the turn angle/sharpness time series.

    Shows:
    - Sharpness values as a line graph
    - Turn apexes as highlighted points
    - Current time position as a vertical line

    Args:
        turn_analysis: TurnAnalysis containing angles and apexes
        current_time: Current video timestamp to highlight
        width: Graph width in pixels
        height: Graph height in pixels
        label: Video label (A or B)

    Returns:
        BGR image of the graph
    """
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)  # Dark gray background

    if not turn_analysis.times:
        cv2.putText(graph, f"Turn Angles ({label}): No data", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        return graph

    times = turn_analysis.times
    sharpness = turn_analysis.sharpness
    apexes = turn_analysis.apexes

    min_time = min(times)
    max_time = max(times)
    time_range = max_time - min_time
    if time_range == 0:
        time_range = 1.0

    # Sharpness ranges from 0 (straight) to ~180 (U-turn)
    min_sharp = 0
    max_sharp = max(sharpness) if sharpness else 100
    if max_sharp == min_sharp:
        max_sharp = min_sharp + 1

    # Margins
    left_margin = 50
    right_margin = 10
    top_margin = 25
    bottom_margin = 25
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    def time_to_x(t):
        return int(left_margin + ((t - min_time) / time_range) * plot_width)

    def sharp_to_y(s):
        return int(top_margin + plot_height - ((s - min_sharp) / (max_sharp - min_sharp)) * plot_height)

    # Draw grid lines
    for i in range(5):
        y = top_margin + int(i * plot_height / 4)
        cv2.line(graph, (left_margin, y), (width - right_margin, y), (50, 50, 50), 1)

    # Draw sharpness line
    points = []
    for t, s in zip(times, sharpness):
        x = time_to_x(t)
        y = sharp_to_y(s)
        points.append((x, y))

    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (100, 180, 100), 1)  # Green line

    # Draw turn apexes as highlighted circles
    for apex in apexes:
        x = time_to_x(apex.time)
        y = sharp_to_y(apex.sharpness)
        # Size based on prominence
        radius = max(3, min(8, int(apex.prominence / 10)))
        cv2.circle(graph, (x, y), radius, (0, 255, 255), -1)  # Yellow filled
        cv2.circle(graph, (x, y), radius, (0, 200, 200), 1)  # Darker outline

    # Draw current time indicator
    if min_time <= current_time <= max_time:
        x_current = time_to_x(current_time)
        cv2.line(graph, (x_current, top_margin), (x_current, height - bottom_margin),
                 (0, 100, 255), 2)  # Orange vertical line

        # Find sharpness at current time (nearest)
        nearest_idx = min(range(len(times)), key=lambda i: abs(times[i] - current_time))
        if abs(times[nearest_idx] - current_time) < 1.0:  # Within 1 second
            current_sharpness = sharpness[nearest_idx]
            # Draw current point
            y_current = sharp_to_y(current_sharpness)
            cv2.circle(graph, (x_current, y_current), 5, (0, 100, 255), -1)

    # Labels
    cv2.putText(graph, f"Turn Sharpness ({label})", (left_margin, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Y-axis labels
    cv2.putText(graph, f"{max_sharp:.0f}", (5, top_margin + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(graph, "0", (5, height - bottom_margin),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Apex count
    cv2.putText(graph, f"Apexes: {len(apexes)}", (width - 80, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return graph


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
    points = []

    for i, (t, dist) in enumerate(zip(times, display_distances)):
        if len(times) > 1:
            t_normalized = (t - times[0]) / (times[-1] - times[0])
        else:
            t_normalized = 0.5
        x = margin_left + int(t_normalized * plot_width)
        normalized = (dist - min_dist) / (max_dist - min_dist)
        y = margin_top + int(normalized * plot_height)
        points.append((x, y))

    # Draw line connecting points
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], (100, 100, 255), 1)

    # Find constrained best point
    constrained_point = None
    constrained_idx = None
    if best_time is not None:
        for i, t in enumerate(times):
            if abs(t - best_time) < 0.001:
                constrained_point = points[i]
                constrained_idx = i
                break

    # Find global minimum point
    global_min_point = None
    global_min_idx = None
    if global_min_time is not None:
        for i, t in enumerate(times):
            if abs(t - global_min_time) < 0.001:
                global_min_point = points[i]
                global_min_idx = i
                break

    show_global_min = (
        global_min_point is not None and
        constrained_point is not None and
        global_min_idx != constrained_idx
    )

    # Draw global minimum marker (yellow) if different from constrained
    if show_global_min and global_min_point is not None and global_min_distance is not None:
        cv2.circle(graph, global_min_point, 6, (0, 255, 255), -1)
        label = f"Global: {format_exp(global_min_distance)}"
        label_x = max(margin_left, min(global_min_point[0] - 40, width - margin_right - 100))
        label_y = max(margin_top + 15, global_min_point[1] - 10)
        cv2.putText(graph, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Draw constrained best marker (green)
    if constrained_point is not None and best_distance is not None:
        cv2.circle(graph, constrained_point, 6, (0, 255, 0), -1)
        label = f"Best: {format_exp(best_distance)}"
        label_x = max(margin_left, min(constrained_point[0] - 30, width - margin_right - 80))
        label_y = min(height - margin_bottom - 5, constrained_point[1] + 20)
        cv2.putText(graph, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Draw Y-axis labels
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


def create_debug_display_v2(
    result: CrossCorrelationResult,
    total_results: int,
    current_idx: int,
    turn_analysis_a: Optional[TurnAnalysis] = None,
    turn_analysis_b: Optional[TurnAnalysis] = None
) -> np.ndarray:
    """Create debug visualization from pre-computed cross-correlation result."""
    frame_a = result.frame_a

    # Get frame dimensions
    h, w = frame_a.shape[:2]

    # Handle no-match case - show black frame for video B
    if result.no_match or result.best_frame_b is None:
        frame_b = np.zeros_like(frame_a)
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

    # Apply mask overlay
    if result.red_mask_a is not None:
        mask_a_resized = cv2.resize(result.red_mask_a, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_a_bgr = apply_mask_overlay(frame_a_bgr, mask_a_resized)
    if result.red_mask_b is not None:
        mask_b_resized = cv2.resize(result.red_mask_b, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        frame_b_bgr = apply_mask_overlay(frame_b_bgr, mask_b_resized)

    # Draw detected red circles
    if result.circle_a is not None:
        cx, cy, r = result.circle_a
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        color_a = (0, 255, 255) if result.circle_a_interpolated else (0, 255, 0)
        cv2.circle(frame_a_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_a, 2)

    if result.best_circle_b is not None:
        cx, cy, r = result.best_circle_b
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        color_b = (0, 255, 255) if result.circle_b_interpolated else (0, 255, 0)
        cv2.circle(frame_b_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_b, 2)

    # Add labels to frames
    cv2.putText(frame_a_bgr, f"Video A: {result.time_a:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if result.no_match or result.best_time_b is None:
        cv2.putText(frame_b_bgr, "Video B: NO MATCH", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame_b_bgr, f"Video B: {result.best_time_b:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add segment labels
    segment_a_str = f"Seg: {result.segment_a}" if result.segment_a is not None else "Seg: ?"
    segment_b_str = f"Seg: {result.best_segment_b}" if result.best_segment_b is not None else "Seg: ?"
    cv2.putText(frame_a_bgr, segment_a_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_b_bgr, segment_b_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add circle position info
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
    color_text_a = (0, 255, 255) if result.circle_a_interpolated else (0, 200, 200)
    color_text_b = (0, 255, 255) if result.circle_b_interpolated else (0, 200, 200)
    cv2.putText(frame_a_bgr, circle_a_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_a, 1)
    cv2.putText(frame_b_bgr, circle_b_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_b, 1)

    # Stack frames horizontally
    frames_row = np.hstack([frame_a_bgr, frame_b_bgr])

    # Create OCR info bar
    ocr_bar_height = 50
    ocr_bar_width = new_w * 2
    ocr_bar = np.zeros((ocr_bar_height, ocr_bar_width, 3), dtype=np.uint8)
    ocr_bar[:] = (50, 50, 50)

    # Format OCR data for frame A
    ocr_a = result.ocr_a
    if ocr_a is not None:
        if ocr_a.is_optimal_lap:
            lap_str_a = "OPTIMAL LAP"
        elif ocr_a.lap_number is not None:
            lap_str_a = f"LAP {ocr_a.lap_number}"
        else:
            lap_str_a = "LAP ?"
        seg_str_a = f"SEG {ocr_a.segment_number}" if ocr_a.segment_number is not None else "SEG ?"

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
        if ocr_b.is_optimal_lap:
            lap_str_b = "OPTIMAL LAP"
        elif ocr_b.lap_number is not None:
            lap_str_b = f"LAP {ocr_b.lap_number}"
        else:
            lap_str_b = "LAP ?"
        seg_str_b = f"SEG {ocr_b.segment_number}" if ocr_b.segment_number is not None else "SEG ?"

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

    # Create distance graph
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

    # Create info bar
    info_height = 60
    info_bar = np.zeros((info_height, graph_width, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)

    if result.no_match or result.best_time_b is None:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | NO MATCH (circle not detected)"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    else:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | Time B: {result.best_time_b:.2f}s | Distance: {result.best_distance:.1f}px"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if result.no_match:
        seg_match = "Segment: N/A"
    else:
        seg_match = "Segment: MATCH" if result.segment_a == result.best_segment_b and result.segment_a is not None else "Segment: mismatch"
    info_text2 = f"{seg_match} | Controls: ←→ (1 frame), ↑↓ (1 sec), ESC (exit)"
    cv2.putText(info_bar, info_text2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Create turn angle graphs if provided
    turn_graphs = None
    if turn_analysis_a is not None or turn_analysis_b is not None:
        turn_graph_height = 100
        turn_graph_width = new_w

        if turn_analysis_a is not None:
            turn_graph_a = create_turn_angle_graph(
                turn_analysis_a, result.time_a, turn_graph_width, turn_graph_height, "A"
            )
        else:
            turn_graph_a = np.zeros((turn_graph_height, turn_graph_width, 3), dtype=np.uint8)
            turn_graph_a[:] = (30, 30, 30)

        if turn_analysis_b is not None and result.best_time_b is not None:
            turn_graph_b = create_turn_angle_graph(
                turn_analysis_b, result.best_time_b, turn_graph_width, turn_graph_height, "B"
            )
        else:
            turn_graph_b = np.zeros((turn_graph_height, turn_graph_width, 3), dtype=np.uint8)
            turn_graph_b[:] = (30, 30, 30)

        turn_graphs = np.hstack([turn_graph_a, turn_graph_b])

    # Stack everything vertically
    if turn_graphs is not None:
        display = np.vstack([frames_row, ocr_bar, graph, turn_graphs, info_bar])
    else:
        display = np.vstack([frames_row, ocr_bar, graph, info_bar])

    return display


def create_debug_display_from_diagnostic(
    result: CrossCorrelationResult,
    frame_a: Optional[np.ndarray],
    frame_b: Optional[np.ndarray],
    total_results: int,
    current_idx: int,
    turn_analysis_a: Optional[TurnAnalysis] = None,
    turn_analysis_b: Optional[TurnAnalysis] = None
) -> np.ndarray:
    """Create debug visualization from diagnostic data with on-demand frame loading.

    This function is similar to create_debug_display_v2 but takes frames as
    separate parameters instead of getting them from the result object.
    This allows frames to be loaded on-demand from video files.

    Args:
        result: CrossCorrelationResult from diagnostic data (without frame data)
        frame_a: RGB frame for video A (loaded on-demand), or None if unavailable
        frame_b: RGB frame for video B (loaded on-demand), or None if unavailable
        total_results: Total number of results for progress display
        current_idx: Current result index for progress display
        turn_analysis_a: Turn analysis for video A
        turn_analysis_b: Turn analysis for video B

    Returns:
        BGR image for OpenCV display
    """
    # Default frame size if no frames available
    default_h, default_w = 480, 640

    # Handle missing frame A
    if frame_a is None:
        frame_a = np.zeros((default_h, default_w, 3), dtype=np.uint8)
        frame_a[:] = (40, 40, 40)  # Dark gray
        cv2.putText(frame_a, "Video A: Frame not available", (50, default_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

    # Get frame dimensions from frame A
    h, w = frame_a.shape[:2]

    # Handle missing frame B or no-match case
    if frame_b is None or result.no_match or result.best_time_b is None:
        frame_b = np.zeros((h, w, 3), dtype=np.uint8)
        if result.no_match:
            frame_b[:] = (20, 20, 40)  # Dark red tint
        else:
            frame_b[:] = (40, 40, 40)  # Dark gray

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

    # Draw detected red circles (no mask overlay in diagnostic mode)
    if result.circle_a is not None:
        cx, cy, r = result.circle_a
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        color_a = (0, 255, 255) if result.circle_a_interpolated else (0, 255, 0)
        cv2.circle(frame_a_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_a, 2)
        cv2.line(frame_a_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_a, 2)

    if result.best_circle_b is not None:
        cx, cy, r = result.best_circle_b
        cx_scaled = int(cx * scale)
        cy_scaled = int(cy * scale)
        r_scaled = max(int(r * scale), 3)
        color_b = (0, 255, 255) if result.circle_b_interpolated else (0, 255, 0)
        cv2.circle(frame_b_bgr, (cx_scaled, cy_scaled), r_scaled + 5, color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled - 15, cy_scaled), (cx_scaled + 15, cy_scaled), color_b, 2)
        cv2.line(frame_b_bgr, (cx_scaled, cy_scaled - 15), (cx_scaled, cy_scaled + 15), color_b, 2)

    # Add labels to frames
    cv2.putText(frame_a_bgr, f"Video A: {result.time_a:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if result.no_match or result.best_time_b is None:
        cv2.putText(frame_b_bgr, "Video B: NO MATCH", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame_b_bgr, f"Video B: {result.best_time_b:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add segment labels
    segment_a_str = f"Seg: {result.segment_a}" if result.segment_a is not None else "Seg: ?"
    segment_b_str = f"Seg: {result.best_segment_b}" if result.best_segment_b is not None else "Seg: ?"
    cv2.putText(frame_a_bgr, segment_a_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_b_bgr, segment_b_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add circle position info
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
    color_text_a = (0, 255, 255) if result.circle_a_interpolated else (0, 200, 200)
    color_text_b = (0, 255, 255) if result.circle_b_interpolated else (0, 200, 200)
    cv2.putText(frame_a_bgr, circle_a_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_a, 1)
    cv2.putText(frame_b_bgr, circle_b_str, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_b, 1)

    # Stack frames horizontally
    frames_row = np.hstack([frame_a_bgr, frame_b_bgr])

    # Create OCR info bar
    ocr_bar_height = 50
    ocr_bar_width = new_w * 2
    ocr_bar = np.zeros((ocr_bar_height, ocr_bar_width, 3), dtype=np.uint8)
    ocr_bar[:] = (50, 50, 50)

    # Format OCR data for frame A
    ocr_a = result.ocr_a
    if ocr_a is not None:
        if ocr_a.is_optimal_lap:
            lap_str_a = "OPTIMAL LAP"
        elif ocr_a.lap_number is not None:
            lap_str_a = f"LAP {ocr_a.lap_number}"
        else:
            lap_str_a = "LAP ?"
        seg_str_a = f"SEG {ocr_a.segment_number}" if ocr_a.segment_number is not None else "SEG ?"

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
        if ocr_b.is_optimal_lap:
            lap_str_b = "OPTIMAL LAP"
        elif ocr_b.lap_number is not None:
            lap_str_b = f"LAP {ocr_b.lap_number}"
        else:
            lap_str_b = "LAP ?"
        seg_str_b = f"SEG {ocr_b.segment_number}" if ocr_b.segment_number is not None else "SEG ?"

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

    # Create distance graph
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

    # Create info bar
    info_height = 60
    info_bar = np.zeros((info_height, graph_width, 3), dtype=np.uint8)
    info_bar[:] = (40, 40, 40)

    if result.no_match or result.best_time_b is None:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | NO MATCH (circle not detected)"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    else:
        info_text = f"Frame {current_idx + 1}/{total_results} | Time A: {result.time_a:.2f}s | Time B: {result.best_time_b:.2f}s | Distance: {result.best_distance:.1f}px"
        cv2.putText(info_bar, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if result.no_match:
        seg_match = "Segment: N/A"
    else:
        seg_match = "Segment: MATCH" if result.segment_a == result.best_segment_b and result.segment_a is not None else "Segment: mismatch"
    info_text2 = f"{seg_match} | Controls: ←→ (1 frame), ↑↓ (1 sec), ESC (exit) | [DIAGNOSTIC MODE]"
    cv2.putText(info_bar, info_text2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Create turn angle graphs if provided
    turn_graphs = None
    if turn_analysis_a is not None or turn_analysis_b is not None:
        turn_graph_height = 100
        turn_graph_width = new_w

        if turn_analysis_a is not None:
            turn_graph_a = create_turn_angle_graph(
                turn_analysis_a, result.time_a, turn_graph_width, turn_graph_height, "A"
            )
        else:
            turn_graph_a = np.zeros((turn_graph_height, turn_graph_width, 3), dtype=np.uint8)
            turn_graph_a[:] = (30, 30, 30)

        if turn_analysis_b is not None and result.best_time_b is not None:
            turn_graph_b = create_turn_angle_graph(
                turn_analysis_b, result.best_time_b, turn_graph_width, turn_graph_height, "B"
            )
        else:
            turn_graph_b = np.zeros((turn_graph_height, turn_graph_width, 3), dtype=np.uint8)
            turn_graph_b[:] = (30, 30, 30)

        turn_graphs = np.hstack([turn_graph_a, turn_graph_b])

    # Stack everything vertically
    if turn_graphs is not None:
        display = np.vstack([frames_row, ocr_bar, graph, turn_graphs, info_bar])
    else:
        display = np.vstack([frames_row, ocr_bar, graph, info_bar])

    return display
