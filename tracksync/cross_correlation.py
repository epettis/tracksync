"""Cross-correlation engine for video alignment.

This module provides functions to compute cross-correlations between
racing videos and generate synchronization points.
"""

import os
import sys
from typing import Optional

import numpy as np

from .autocorrelation import StartFinishCrossing
from .feature_extraction import interpolate_missing_circles
from .frame_data import CrossCorrelationResult, FrameData, VideoFeatures
from .models import SyncPoint, SyncResult


def compute_cross_correlations_from_features(
    features_a: VideoFeatures,
    features_b: VideoFeatures,
    window_seconds: float = 20.0
) -> list[CrossCorrelationResult]:
    """
    Compute cross-correlation using pre-extracted VideoFeatures.

    This is an optimized version of compute_cross_correlations that uses
    pre-computed position arrays from VideoFeatures, avoiding redundant
    interpolation and array construction.

    Args:
        features_a: Pre-extracted features from video A
        features_b: Pre-extracted features from video B
        window_seconds: Match against frames within +/- this many seconds

    Returns:
        List of CrossCorrelationResult for each frame in A
    """
    frames_a = features_a.frames
    frames_b = features_b.frames
    crossings_a = features_a.crossings
    crossings_b = features_b.crossings
    positions_a = features_a.positions
    positions_b = features_b.positions
    pos_array_a = features_a.pos_array
    pos_array_b = features_b.pos_array

    # Build crossing offset map
    crossing_offsets: list[tuple[float, float, float]] = []

    if crossings_a and crossings_b:
        for ca in crossings_a:
            if ca.lap_before is None or ca.lap_after is None:
                continue
            for cb in crossings_b:
                if cb.lap_before == ca.lap_before and cb.lap_after == ca.lap_after:
                    offset = cb.video_time - ca.video_time
                    crossing_offsets.append((ca.video_time, float('inf'), offset))
                    print(f"\n  Crossing sync: Lap {ca.lap_before}->{ca.lap_after} "
                          f"A={ca.video_time:.2f}s B={cb.video_time:.2f}s offset={offset:+.2f}s",
                          file=sys.stderr)
                    break

        crossing_offsets.sort(key=lambda x: x[0])
        for i in range(len(crossing_offsets) - 1):
            start, _, offset = crossing_offsets[i]
            next_start = crossing_offsets[i + 1][0]
            crossing_offsets[i] = (start, next_start, offset)

    n_a = len(frames_a)
    n_b = len(frames_b)

    # Compute distance matrix using pre-computed position arrays
    print("  Computing distance matrix...", file=sys.stderr)
    diff = pos_array_a[:, None, :] - pos_array_b[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    distance_matrix = np.where(np.isnan(distance_matrix), 10000.0, distance_matrix)

    valid_a = ~np.isnan(pos_array_a[:, 0])
    valid_b = ~np.isnan(pos_array_b[:, 0])

    times_a = np.array(features_a.frame_times)
    times_b = np.array(features_b.frame_times)

    window_offsets = np.zeros(n_a, dtype=np.float64)
    if crossing_offsets:
        for i, time_a in enumerate(times_a):
            for start_a, end_a, offset in crossing_offsets:
                if start_a <= time_a < end_a:
                    window_offsets[i] = offset
                    break

    print("  Finding best matches...", file=sys.stderr)

    results: list[CrossCorrelationResult] = []
    prev_best_time: Optional[float] = None

    for i, fa in enumerate(frames_a):
        pct = ((i + 1) / n_a) * 100
        print(f"\rCross-correlating: {pct:5.1f}% ({i + 1}/{n_a})   ", end="", file=sys.stderr)

        time_a = fa.time

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

        distances = distance_matrix[i]
        all_distances = list(zip(times_b.tolist(), distances.tolist()))

        global_min_idx = int(np.argmin(distances))
        global_min_distance = distances[global_min_idx]

        window_center = time_a + window_offsets[i]
        window_mask = (times_b >= window_center - window_seconds) & (times_b <= window_center + window_seconds)

        if not np.any(window_mask):
            pos_a = positions_a[i]
            results.append(CrossCorrelationResult(
                time_a=fa.time,
                frame_a=fa.frame,
                circle_a=fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5) if pos_a else None,
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

        windowed_distances = np.where(window_mask, distances, np.inf)
        best_idx = int(np.argmin(windowed_distances))
        best_distance = windowed_distances[best_idx]

        if best_distance == np.inf:
            pos_a = positions_a[i]
            results.append(CrossCorrelationResult(
                time_a=fa.time,
                frame_a=fa.frame,
                circle_a=fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5) if pos_a else None,
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

        if prev_best_time is not None and times_b[best_idx] < prev_best_time:
            forward_mask = window_mask & (times_b >= prev_best_time)
            if np.any(forward_mask):
                forward_distances = np.where(forward_mask, distances, np.inf)
                best_idx = int(np.argmin(forward_distances))
                best_distance = forward_distances[best_idx]

        best_fb = frames_b[best_idx]
        prev_best_time = best_fb.time

        circle_a_interpolated = fa.circle is None
        circle_b_interpolated = best_fb.circle is None

        pos_a = positions_a[i]
        circle_a = fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5) if pos_a else None
        pos_b = positions_b[best_idx]
        circle_b = best_fb.circle if best_fb.circle is not None else (pos_b[0], pos_b[1], 5) if pos_b is not None else None

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

    print(file=sys.stderr)
    return results


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
    crossing_offsets: list[tuple[float, float, float]] = []

    if crossings_a and crossings_b:
        for ca in crossings_a:
            if ca.lap_before is None or ca.lap_after is None:
                continue
            for cb in crossings_b:
                if cb.lap_before == ca.lap_before and cb.lap_after == ca.lap_after:
                    offset = cb.video_time - ca.video_time
                    crossing_offsets.append((ca.video_time, float('inf'), offset))
                    print(f"\n  Crossing sync: Lap {ca.lap_before}->{ca.lap_after} "
                          f"A={ca.video_time:.2f}s B={cb.video_time:.2f}s offset={offset:+.2f}s",
                          file=sys.stderr)
                    break

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

    # Vectorized distance computation
    print("  Computing distance matrix...", file=sys.stderr)

    n_a = len(frames_a)
    n_b = len(frames_b)

    # Create position arrays with NaN for missing positions
    pos_array_a = np.full((n_a, 2), np.nan, dtype=np.float64)
    pos_array_b = np.full((n_b, 2), np.nan, dtype=np.float64)

    for i, pos in enumerate(positions_a):
        if pos is not None:
            pos_array_a[i] = pos

    for j, pos in enumerate(positions_b):
        if pos is not None:
            pos_array_b[j] = pos

    # Compute all pairwise distances using broadcasting
    diff = pos_array_a[:, None, :] - pos_array_b[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    distance_matrix = np.where(np.isnan(distance_matrix), 10000.0, distance_matrix)

    valid_a = ~np.isnan(pos_array_a[:, 0])
    valid_b = ~np.isnan(pos_array_b[:, 0])

    times_a = np.array([f.time for f in frames_a])
    times_b = np.array([f.time for f in frames_b])

    window_offsets = np.zeros(n_a, dtype=np.float64)
    if crossing_offsets:
        for i, time_a in enumerate(times_a):
            for start_a, end_a, offset in crossing_offsets:
                if start_a <= time_a < end_a:
                    window_offsets[i] = offset
                    break

    print("  Finding best matches...", file=sys.stderr)

    results: list[CrossCorrelationResult] = []
    prev_best_time: Optional[float] = None

    for i, fa in enumerate(frames_a):
        pct = ((i + 1) / n_a) * 100
        print(f"\rCross-correlating: {pct:5.1f}% ({i + 1}/{n_a})   ", end="", file=sys.stderr)

        time_a = fa.time

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

        distances = distance_matrix[i]
        all_distances = list(zip(times_b.tolist(), distances.tolist()))

        global_min_idx = int(np.argmin(distances))
        global_min_distance = distances[global_min_idx]

        window_center = time_a + window_offsets[i]
        window_mask = (times_b >= window_center - window_seconds) & (times_b <= window_center + window_seconds)

        if not np.any(window_mask):
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

        windowed_distances = np.where(window_mask, distances, np.inf)
        best_idx = int(np.argmin(windowed_distances))
        best_distance = windowed_distances[best_idx]

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

        # Forward-progress constraint
        if prev_best_time is not None and times_b[best_idx] < prev_best_time:
            forward_mask = window_mask & (times_b >= prev_best_time)
            if np.any(forward_mask):
                forward_distances = np.where(forward_mask, distances, np.inf)
                best_idx = int(np.argmin(forward_distances))
                best_distance = forward_distances[best_idx]

        best_fb = frames_b[best_idx]
        prev_best_time = best_fb.time

        circle_a_interpolated = fa.circle is None
        circle_b_interpolated = best_fb.circle is None

        pos_a = positions_a[i]
        circle_a = fa.circle if fa.circle is not None else (pos_a[0], pos_a[1], 5)
        pos_b = positions_b[best_idx]
        circle_b = best_fb.circle if best_fb.circle is not None else (pos_b[0], pos_b[1], 5) if pos_b is not None else None

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

    print(file=sys.stderr)
    return results


def generate_sync_points(
    results: list[CrossCorrelationResult],
    crossings_a: list[StartFinishCrossing],
    crossings_b: list[StartFinishCrossing],
    max_interval: float = 3.0
) -> SyncResult:
    """
    Generate synchronization points for video scaling.

    The synchronization strategy:
    1. First timestamp: when video A crosses start-finish (first crossing)
       - All frames/audio before this are dropped from both videos
    2. Last timestamp: when video A crosses start-finish for second time
       - All frames/audio after this are dropped from both videos
    3. Segment changes: synchronize at every segment transition
    4. Periodic: ensure at least one sync point every max_interval seconds

    Args:
        results: Cross-correlation results from compute_cross_correlations
        crossings_a: Start-finish crossings in video A
        crossings_b: Start-finish crossings in video B
        max_interval: Maximum time between sync points in video A

    Returns:
        SyncResult with sync points and trim information
    """
    if not results:
        return SyncResult(
            sync_points=[],
            trim_start_a=0.0,
            trim_end_a=0.0,
            trim_start_b=0.0,
            trim_end_b=0.0,
            crossings_a=crossings_a,
            crossings_b=crossings_b
        )

    # Step 1: Determine trim points from start-finish crossings
    if len(crossings_a) >= 2:
        trim_start_a = crossings_a[0].video_time
        trim_end_a = crossings_a[1].video_time
    elif len(crossings_a) == 1:
        trim_start_a = crossings_a[0].video_time
        trim_end_a = results[-1].time_a
    else:
        trim_start_a = results[0].time_a
        trim_end_a = results[-1].time_a

    # Find corresponding trim points in video B
    trim_start_b = None
    trim_end_b = None

    for r in results:
        if r.time_a >= trim_start_a and r.best_time_b is not None:
            trim_start_b = r.best_time_b
            break

    for r in reversed(results):
        if r.time_a <= trim_end_a and r.best_time_b is not None:
            trim_end_b = r.best_time_b
            break

    # Fallback if no matches found
    if trim_start_b is None:
        for r in results:
            if r.best_time_b is not None:
                trim_start_b = r.best_time_b
                break
    if trim_end_b is None:
        for r in reversed(results):
            if r.best_time_b is not None:
                trim_end_b = r.best_time_b
                break

    trim_start_b = trim_start_b or 0.0
    trim_end_b = trim_end_b or 0.0

    print(f"\nTrim points:", file=sys.stderr)
    print(f"  Video A: {trim_start_a:.2f}s to {trim_end_a:.2f}s", file=sys.stderr)
    print(f"  Video B: {trim_start_b:.2f}s to {trim_end_b:.2f}s", file=sys.stderr)

    # Step 2: Collect sync points
    sync_points: list[SyncPoint] = []

    # Add start point
    sync_points.append(SyncPoint(
        time_a=trim_start_a,
        time_b=trim_start_b,
        label="start"
    ))

    # Step 3: Find segment changes and add sync points
    prev_segment = None
    for r in results:
        if r.time_a < trim_start_a or r.time_a > trim_end_a:
            continue
        if r.no_match or r.best_time_b is None:
            continue

        current_segment = r.ocr_a.segment_number if r.ocr_a else None
        if current_segment is not None and current_segment != prev_segment and prev_segment is not None:
            sync_points.append(SyncPoint(
                time_a=r.time_a,
                time_b=r.best_time_b,
                label=f"segment_{current_segment}"
            ))
        prev_segment = current_segment

    # Add end point
    sync_points.append(SyncPoint(
        time_a=trim_end_a,
        time_b=trim_end_b,
        label="end"
    ))

    # Sort sync points by time_a
    sync_points.sort(key=lambda p: p.time_a)

    # Use sync points directly without periodic interpolation
    filled_points = sync_points

    # Step 5: Calculate speed ratios for each segment
    for i in range(len(filled_points) - 1):
        current = filled_points[i]
        next_point = filled_points[i + 1]

        delta_a = next_point.time_a - current.time_a
        delta_b = next_point.time_b - current.time_b

        if delta_a > 0:
            speed = delta_b / delta_a
        else:
            speed = 1.0

        filled_points[i] = SyncPoint(
            time_a=current.time_a,
            time_b=current.time_b,
            label=current.label,
            speed=speed
        )

    print(f"\nGenerated {len(filled_points)} sync points:", file=sys.stderr)
    for p in filled_points[:5]:
        print(f"  {p.label}: A={p.time_a:.2f}s -> B={p.time_b:.2f}s (speed={p.speed:.3f})", file=sys.stderr)
    if len(filled_points) > 5:
        print(f"  ... and {len(filled_points) - 5} more", file=sys.stderr)

    return SyncResult(
        sync_points=filled_points,
        trim_start_a=trim_start_a,
        trim_end_a=trim_end_a,
        trim_start_b=trim_start_b,
        trim_end_b=trim_end_b,
        crossings_a=crossings_a,
        crossings_b=crossings_b
    )


def output_tracksync_csv(
    sync_result: SyncResult,
    video_a_path: str,
    video_b_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Output synchronization results in tracksync CSV format.

    The tracksync CSV format:
    - Header: milestone_name, driver1, driver2
    - Data: milestone, timestamp1, timestamp2

    Args:
        sync_result: Synchronization result from generate_sync_points
        video_a_path: Path to video A
        video_b_path: Path to video B
        output_path: Optional output file path

    Returns:
        CSV content as string
    """
    # Extract driver names from filenames
    driver_a = os.path.splitext(os.path.basename(video_a_path))[0]
    driver_b = os.path.splitext(os.path.basename(video_b_path))[0]

    lines = []

    # Header row
    lines.append(f"milestone,{driver_a},{driver_b}")

    # Data rows
    for point in sync_result.sync_points:
        lines.append(f"{point.label},{point.time_a:.3f},{point.time_b:.3f}")

    csv_content = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(csv_content)
        print(f"\nTracksync CSV saved to: {output_path}", file=sys.stderr)

    return csv_content


def generate_pairwise_sync_from_features(
    features_a: VideoFeatures,
    features_b: VideoFeatures,
    max_sync_interval: float = 3.0,
    window_seconds: float = 20.0
) -> SyncResult:
    """
    Generate synchronization result for a pair of videos using cached features.

    This is the main entry point for pairwise sync when features have already
    been extracted. It computes cross-correlations and generates sync points.

    Args:
        features_a: Pre-extracted features for video A (target)
        features_b: Pre-extracted features for video B (reference)
        max_sync_interval: Maximum interval between sync points in seconds
        window_seconds: Correlation window size in seconds

    Returns:
        SyncResult with sync points and trim information
    """
    print(f"\n  Correlating {features_a.driver_name} vs {features_b.driver_name}...", file=sys.stderr)

    # Compute cross-correlations using pre-extracted features
    results = compute_cross_correlations_from_features(
        features_a, features_b, window_seconds
    )

    # Generate sync points
    sync_result = generate_sync_points(
        results,
        features_a.crossings,
        features_b.crossings,
        max_interval=max_sync_interval
    )

    return sync_result
