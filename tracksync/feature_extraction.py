"""Feature extraction from video frames.

This module provides functions to extract features from racing videos,
including circle detection, OCR data, and position interpolation.
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

from .frame_analysis import (
    detect_segment_number,
    extract_frame_ocr,
    find_red_circle,
    get_frame_at_time,
    get_video_info,
    interpolate_ocr_data,
)
from .frame_data import FrameData, VideoFeatures


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


def extract_frame_data(
    video_path: str,
    interval: float,
    red_min: int,
    red_max_gb: int,
    progress_prefix: str = "",
    use_native_fps: bool = False,
    max_duration: Optional[float] = None,
    store_frames: bool = True
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
        store_frames: If True, store RGB frames in FrameData (uses more memory).
                     Set to False for sync mode to reduce memory usage.

    Returns:
        Tuple of (list of FrameData, actual fps used)
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        duration, fps, frame_count = get_video_info(video)

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

        # Preallocate list for better performance
        frames: list[FrameData] = [None] * total_frames  # type: ignore
        actual_count = 0

        time = 0.0
        frame_idx = 0

        while time < effective_duration and frame_idx < total_frames:
            pct = ((frame_idx + 1) / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r{progress_prefix}Extracting frames: {pct:5.1f}% ({frame_idx + 1}/{total_frames})   ", end="", file=sys.stderr)

            frame = get_frame_at_time(video, time)
            if frame is not None:
                circle = find_red_circle(frame, red_min, red_max_gb)
                segment = detect_segment_number(frame)
                ocr_data = extract_frame_ocr(frame)

                # Only store frame and red_mask if requested (debug mode)
                if store_frames:
                    red_mask = compute_red_mask(frame, red_min, red_max_gb)
                    frames[frame_idx] = FrameData(
                        time=time,
                        circle=circle,
                        segment=segment,
                        frame=frame,
                        red_mask=red_mask,
                        ocr_data=ocr_data
                    )
                else:
                    frames[frame_idx] = FrameData(
                        time=time,
                        circle=circle,
                        segment=segment,
                        ocr_data=ocr_data
                    )
                actual_count += 1

            frame_idx += 1
            time += actual_interval

        print(file=sys.stderr)  # Newline after progress

        # Trim list to actual count (in case some frames were skipped)
        frames = [f for f in frames if f is not None]

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


def extract_video_features(
    video_path: str,
    red_min: int = 200,
    red_max_gb: int = 80,
    max_duration: Optional[float] = None,
    progress_prefix: str = ""
) -> VideoFeatures:
    """
    Extract all features from a video for later pairwise comparison.

    This function performs all expensive operations (frame extraction, OCR,
    circle detection) once per video, allowing O(n) extraction instead of
    O(n^2) when comparing multiple videos.

    Args:
        video_path: Path to video file
        red_min: Minimum R value for red detection
        red_max_gb: Maximum G/B values for red detection
        max_duration: If set, only extract frames up to this duration
        progress_prefix: Prefix for progress output

    Returns:
        VideoFeatures with all extracted and interpolated data
    """
    driver_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"{progress_prefix}Extracting features from: {video_path}", file=sys.stderr)

    # Extract frame data at native FPS (don't store frames to save memory)
    frames, fps = extract_frame_data(
        video_path, 1.0, red_min, red_max_gb,
        f"{progress_prefix}  ", use_native_fps=True, max_duration=max_duration,
        store_frames=False
    )

    # Get frame times and OCR data
    frame_times = [f.time for f in frames]
    ocr_list = [f.ocr_data for f in frames]

    # Interpolate OCR and detect crossings
    print(f"{progress_prefix}  Interpolating OCR data...", file=sys.stderr)
    interpolated_ocr, crossings = interpolate_ocr_data(ocr_list, frame_times, fps)

    # Update frame data with interpolated OCR
    for i, ocr in enumerate(interpolated_ocr):
        frames[i].ocr_data = ocr

    # Interpolate missing circle positions
    print(f"{progress_prefix}  Interpolating circle positions...", file=sys.stderr)
    positions = interpolate_missing_circles(frames)

    # Build position array for vectorized correlation
    n = len(frames)
    pos_array = np.full((n, 2), np.nan, dtype=np.float64)
    for i, pos in enumerate(positions):
        if pos is not None:
            pos_array[i] = pos

    # Statistics
    detected = sum(1 for f in frames if f.circle is not None)
    interpolated = sum(1 for i, p in enumerate(positions) if p is not None and frames[i].circle is None)
    unmatchable = sum(1 for p in positions if p is None)
    print(f"{progress_prefix}  Circles: {detected} detected, {interpolated} interpolated, {unmatchable} unmatchable", file=sys.stderr)
    print(f"{progress_prefix}  Crossings: {len(crossings)} start-finish crossings detected", file=sys.stderr)
    for c in crossings:
        lap_info = f"Lap {c.lap_before} -> {c.lap_after}" if c.lap_before and c.lap_after else "?"
        print(f"{progress_prefix}    {c.video_time:.2f}s ({lap_info})", file=sys.stderr)

    return VideoFeatures(
        video_path=video_path,
        driver_name=driver_name,
        fps=fps,
        frames=frames,
        frame_times=frame_times,
        interpolated_ocr=interpolated_ocr,
        crossings=crossings,
        positions=positions,
        pos_array=pos_array
    )
