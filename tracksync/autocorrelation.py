"""
Video autocorrelation module for automatic video alignment.

This module provides functions for OCR-based video analysis, including:
- Frame OCR data extraction (lap numbers, segments, lap times)
- Start-finish line crossing detection
- Red circle (car marker) detection
- Frame binarization for visual analysis
"""

import numpy as np
import cv2
import re
from dataclasses import dataclass
from typing import Optional

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def binarize_frame(
    frame: np.ndarray,
    white_threshold: int = 240,
    red_min: int = 200,
    red_max_gb: int = 80
) -> np.ndarray:
    """
    Convert RGB frame to binary (0/1) array.

    Two types of pixels become 1:
    1. White pixels: R, G, B all >= white_threshold
    2. Red pixels: R >= red_min AND G <= red_max_gb AND B <= red_max_gb

    All other pixels become 0.

    Args:
        frame: RGB frame as numpy array (H, W, 3)
        white_threshold: Minimum value for R, G, B to be considered "white" (default: 240)
        red_min: Minimum R value for red detection (default: 200)
        red_max_gb: Maximum G and B values for red detection (default: 80)

    Returns:
        Binary array (H, W) with dtype uint8
    """
    # Extract channels
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]

    # Detect white pixels
    white_mask = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)

    # Detect red pixels
    red_mask = (r >= red_min) & (g <= red_max_gb) & (b <= red_max_gb)

    # Combine masks
    binary = (white_mask | red_mask).astype(np.uint8)

    return binary


def get_frame_at_time(video: cv2.VideoCapture, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extract a frame from a video at a specific time.

    Args:
        video: OpenCV VideoCapture object
        time_seconds: Time in seconds

    Returns:
        RGB frame as numpy array, or None if seek failed
    """
    video.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    ret, frame = video.read()
    if not ret:
        return None
    # OpenCV reads as BGR, convert to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_video_info(video: cv2.VideoCapture) -> tuple[float, float, int]:
    """
    Get video metadata.

    Args:
        video: OpenCV VideoCapture object

    Returns:
        (duration_seconds, fps, frame_count)
    """
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    return duration, fps, frame_count


def detect_segment_number(
    frame: np.ndarray,
    ocr_region: tuple[int, int, int, int] = None
) -> Optional[int]:
    """
    Detect segment number from a frame using OCR.

    Looks for text matching "SEGMENT N" pattern in the top portion of the frame.

    Args:
        frame: RGB frame as numpy array
        ocr_region: Optional (y1, y2, x1, x2) region to search. Defaults to top 200 pixels.

    Returns:
        Segment number as int, or None if not detected
    """
    if not HAS_TESSERACT:
        return None

    h, w = frame.shape[:2]

    if ocr_region is None:
        # Default: top 200 pixels, full width
        y1, y2, x1, x2 = 0, 200, 0, w
    else:
        y1, y2, x1, x2 = ocr_region

    # Extract region and convert to grayscale
    region = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Threshold for white text
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    try:
        text = pytesseract.image_to_string(binary, config='--psm 6')
        match = re.search(r'SEGMENT\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except Exception:
        pass

    return None


@dataclass
class FrameOCRData:
    """OCR-extracted data from a single frame."""
    lap_number: Optional[int] = None  # Lap number, or None if optimal lap
    is_optimal_lap: bool = False  # True if "OPTIMAL LAP" detected
    segment_number: Optional[int] = None  # Segment number
    lap_time_seconds: Optional[float] = None  # Lap time in seconds


def _parse_lap_info(text: str) -> tuple[Optional[int], bool]:
    """
    Parse lap information from OCR text.

    Args:
        text: OCR output text

    Returns:
        Tuple of (lap_number, is_optimal_lap)
    """
    # Check for OPTIMAL LAP first
    if re.search(r'OPTIMAL\s*LAP', text, re.IGNORECASE):
        return None, True

    # Check for LAP N
    match = re.search(r'LAP\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), False

    return None, False


def _parse_segment_number(text: str) -> Optional[int]:
    """
    Parse segment number from OCR text.

    Args:
        text: OCR output text

    Returns:
        Segment number as int, or None if not detected
    """
    match = re.search(r'SEGMENT\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_lap_time(text: str) -> Optional[float]:
    """
    Parse lap time from OCR text.

    Args:
        text: OCR output text

    Returns:
        Lap time in seconds as float, or None if not detected
    """
    # Look for time pattern: M:SS.ss or MM:SS.ss
    # Also handle 0:00.00 format
    match = re.search(r'(\d{1,2}):(\d{2})\.(\d{2})', text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        total_seconds = minutes * 60 + seconds + centiseconds / 100.0
        return total_seconds
    return None


def extract_frame_ocr(frame: np.ndarray) -> FrameOCRData:
    """
    Extract all OCR data from a single frame using a single OCR call.

    Combines multiple regions of interest into one image, runs OCR once,
    then parses all patterns from the output.

    The regions are:
    - Top region (0-200px): Contains LAP/OPTIMAL LAP and SEGMENT info
    - Bottom-left region (bottom 200px, left half): Contains LAP TIME

    Args:
        frame: RGB frame as numpy array

    Returns:
        FrameOCRData with lap number, segment, and lap time
    """
    if not HAS_TESSERACT:
        return FrameOCRData()

    h, w = frame.shape[:2]

    # Define regions to extract
    # Top region: contains LAP N, OPTIMAL LAP, and SEGMENT N
    top_y1, top_y2 = 0, min(200, h)
    top_x1, top_x2 = 0, w

    # Bottom-left region: contains LAP TIME
    bottom_y1, bottom_y2 = max(0, h - 200), h
    bottom_x1, bottom_x2 = 0, w // 2

    # Extract and process top region
    top_region = frame[top_y1:top_y2, top_x1:top_x2]
    top_gray = cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY)
    _, top_binary = cv2.threshold(top_gray, 200, 255, cv2.THRESH_BINARY)

    # Extract and process bottom region
    bottom_region = frame[bottom_y1:bottom_y2, bottom_x1:bottom_x2]
    bottom_gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
    _, bottom_binary = cv2.threshold(bottom_gray, 200, 255, cv2.THRESH_BINARY)

    # Combine regions vertically with a separator line
    # This allows single OCR call while keeping regions visually separate
    separator_height = 10
    separator = np.zeros((separator_height, max(top_binary.shape[1], bottom_binary.shape[1])), dtype=np.uint8)

    # Pad bottom region to match top width if needed
    if bottom_binary.shape[1] < top_binary.shape[1]:
        pad_width = top_binary.shape[1] - bottom_binary.shape[1]
        bottom_binary = np.pad(bottom_binary, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # Stack: top region + separator + bottom region
    combined = np.vstack([top_binary, separator, bottom_binary])

    try:
        # Single OCR call on combined image
        text = pytesseract.image_to_string(combined, config='--psm 6')

        # Parse all patterns from the combined output
        lap_number, is_optimal = _parse_lap_info(text)
        segment = _parse_segment_number(text)
        lap_time = _parse_lap_time(text)

        return FrameOCRData(
            lap_number=lap_number,
            is_optimal_lap=is_optimal,
            segment_number=segment,
            lap_time_seconds=lap_time
        )
    except Exception:
        return FrameOCRData()


@dataclass
class StartFinishCrossing:
    """Represents a start-finish line crossing detected from lap time rollover."""
    frame_index: int  # Index in the frames list
    video_time: float  # Time in the video (seconds)
    lap_before: Optional[int]  # Lap number before crossing
    lap_after: Optional[int]  # Lap number after crossing (should be lap_before + 1)


def interpolate_ocr_data(
    ocr_data_list: list[Optional[FrameOCRData]],
    frame_times: list[float],
    fps: float
) -> tuple[list[FrameOCRData], list[StartFinishCrossing]]:
    """
    Interpolate missing OCR data using continuity constraints.

    Laps and segments are continuous - if frames A and C have the same value,
    frame B between them must also have that value.

    Lap times can be interpolated between known values, accounting for rollover
    at the start-finish line (when lap time goes from high to ~0).

    Args:
        ocr_data_list: List of FrameOCRData (may have None values or None fields)
        frame_times: Video timestamps for each frame
        fps: Video frame rate (for estimating times near rollover)

    Returns:
        Tuple of:
        - List of FrameOCRData with interpolated values
        - List of StartFinishCrossing events detected from lap time rollovers
    """
    n = len(ocr_data_list)
    if n == 0:
        return [], []

    # Initialize with copies of existing data (or empty FrameOCRData for None)
    result: list[FrameOCRData] = []
    for ocr in ocr_data_list:
        if ocr is None:
            result.append(FrameOCRData())
        else:
            result.append(FrameOCRData(
                lap_number=ocr.lap_number,
                is_optimal_lap=ocr.is_optimal_lap,
                segment_number=ocr.segment_number,
                lap_time_seconds=ocr.lap_time_seconds
            ))

    # Track start-finish crossings
    crossings: list[StartFinishCrossing] = []

    # Step 1: Interpolate lap numbers using continuity
    # If frames at indices i and j have the same lap number, all frames between must too
    _interpolate_discrete_values(
        result,
        lambda x: x.lap_number,
        lambda x, v: setattr(x, 'lap_number', v)
    )

    # Step 2: Interpolate segment numbers using continuity
    _interpolate_discrete_values(
        result,
        lambda x: x.segment_number,
        lambda x, v: setattr(x, 'segment_number', v)
    )

    # Step 3: Detect lap time rollovers and interpolate lap times
    crossings = _interpolate_lap_times(result, frame_times, fps)

    return result, crossings


def _interpolate_discrete_values(
    data: list[FrameOCRData],
    getter: callable,
    setter: callable
) -> None:
    """
    Interpolate discrete values (lap number, segment) using continuity.

    If frames at i and j have the same value, all frames between them
    must have that same value.

    Modifies data in place.
    """
    n = len(data)
    if n == 0:
        return

    # Find all indices with known values
    known_indices = [(i, getter(data[i])) for i in range(n) if getter(data[i]) is not None]

    if not known_indices:
        return

    # Fill gaps where both endpoints have the same value
    for idx in range(len(known_indices) - 1):
        i, val_i = known_indices[idx]
        j, val_j = known_indices[idx + 1]

        if val_i == val_j:
            # Fill all frames between i and j with this value
            for k in range(i + 1, j):
                setter(data[k], val_i)


def _interpolate_lap_times(
    data: list[FrameOCRData],
    frame_times: list[float],
    fps: float
) -> list[StartFinishCrossing]:
    """
    Interpolate lap times, detecting rollovers at start-finish line.

    A rollover is detected when lap time decreases significantly
    (e.g., 1:45.00 -> 0:00.25).

    For frames between two known lap times:
    - If no rollover, linearly interpolate
    - If rollover detected, estimate crossing point and interpolate from there

    Args:
        data: List of FrameOCRData to modify in place
        frame_times: Video timestamps for each frame
        fps: Frame rate for time calculations

    Returns:
        List of detected start-finish crossings
    """
    n = len(data)
    if n == 0:
        return []

    crossings: list[StartFinishCrossing] = []

    # Find indices with known lap times
    known_indices = [
        (i, data[i].lap_time_seconds)
        for i in range(n)
        if data[i].lap_time_seconds is not None
    ]

    if len(known_indices) < 2:
        return crossings

    # Process each pair of known lap times
    for idx in range(len(known_indices) - 1):
        i, time_i = known_indices[idx]
        j, time_j = known_indices[idx + 1]

        if j == i + 1:
            # Adjacent frames, just check for rollover
            if time_j < time_i - 5.0:  # Rollover threshold: 5 seconds decrease
                # Validate this is a real crossing, not an OCR error:
                # 1. The new lap time should be small (near start of lap), OR
                # 2. The lap number should have incremented
                lap_before = data[i].lap_number
                lap_after = data[j].lap_number
                lap_incremented = (
                    lap_after is not None and
                    lap_before is not None and
                    lap_after > lap_before
                )
                new_lap_time_small = time_j < 10.0  # Within 10 seconds of lap start

                if lap_incremented or new_lap_time_small:
                    # Rollover detected between i and j
                    # Estimate crossing at midpoint
                    crossing_time = (frame_times[i] + frame_times[j]) / 2
                    crossings.append(StartFinishCrossing(
                        frame_index=i,
                        video_time=crossing_time,
                        lap_before=lap_before,
                        lap_after=lap_after
                    ))
            continue

        # Multiple frames between i and j
        video_time_i = frame_times[i]
        video_time_j = frame_times[j]
        video_duration = video_time_j - video_time_i

        # Check for rollover
        if time_j < time_i - 5.0:
            # Validate this is a real crossing, not an OCR error:
            # 1. The new lap time should be small (near start of lap), OR
            # 2. The lap number should have incremented
            lap_before = data[i].lap_number
            lap_after = data[j].lap_number
            lap_incremented = (
                lap_after is not None and
                lap_before is not None and
                lap_after > lap_before
            )
            new_lap_time_small = time_j < 10.0  # Within 10 seconds of lap start

            if lap_incremented or new_lap_time_small:
                # Rollover occurred somewhere between i and j
                # Estimate where: lap_time reaches ~0 and restarts

                # Method: Find where the time would have been ~0
                # If lap_time_i is X seconds, car crosses start-finish X seconds later
                # (assuming constant speed, lap_time increases at ~1 second per second)
                estimated_crossing_offset = time_i  # seconds after frame i

                # Find the frame closest to the crossing
                crossing_video_time = video_time_i + estimated_crossing_offset
                crossing_frame_idx = i

                for k in range(i + 1, j + 1):
                    if frame_times[k] >= crossing_video_time:
                        # Crossing happened between k-1 and k
                        crossing_frame_idx = k - 1 if k > i else i
                        crossing_video_time = min(crossing_video_time, frame_times[j])
                        break
                else:
                    crossing_frame_idx = j - 1
                    crossing_video_time = frame_times[j]

                crossings.append(StartFinishCrossing(
                    frame_index=crossing_frame_idx,
                    video_time=crossing_video_time,
                    lap_before=lap_before,
                    lap_after=lap_after
                ))

                # Interpolate before the crossing
                for k in range(i + 1, crossing_frame_idx + 1):
                    elapsed = frame_times[k] - video_time_i
                    # Lap time increases roughly 1:1 with real time
                    data[k].lap_time_seconds = time_i + elapsed

                # Interpolate after the crossing
                for k in range(crossing_frame_idx + 1, j):
                    elapsed = frame_times[k] - crossing_video_time
                    # After crossing, lap time starts from ~0
                    data[k].lap_time_seconds = elapsed
            else:
                # OCR error - treat as no rollover, just interpolate
                for k in range(i + 1, j):
                    t = (frame_times[k] - video_time_i) / video_duration
                    data[k].lap_time_seconds = time_i + t * (time_j - time_i)

        else:
            # No rollover - simple linear interpolation
            # But we need to account for real-time progression
            # Lap time should increase at roughly 1 second per second

            for k in range(i + 1, j):
                # Linear interpolation based on video time
                t = (frame_times[k] - video_time_i) / video_duration
                data[k].lap_time_seconds = time_i + t * (time_j - time_i)

    return crossings


def find_start_finish_crossings(
    ocr_data_list: list[Optional[FrameOCRData]],
    frame_times: list[float],
    fps: float
) -> list[StartFinishCrossing]:
    """
    Find all start-finish line crossings in the OCR data.

    Crossings are detected by:
    1. Lap time rollover (time drops from high value to near 0)
    2. Lap number increment

    Args:
        ocr_data_list: List of OCR data for each frame
        frame_times: Video timestamp for each frame
        fps: Video frame rate

    Returns:
        List of StartFinishCrossing events
    """
    crossings: list[StartFinishCrossing] = []
    n = len(ocr_data_list)

    if n < 2:
        return crossings

    prev_lap_time: Optional[float] = None
    prev_lap_number: Optional[int] = None
    prev_idx: int = 0

    for i, ocr in enumerate(ocr_data_list):
        if ocr is None:
            continue

        # Check for lap time rollover
        if ocr.lap_time_seconds is not None and prev_lap_time is not None:
            # Rollover: time drops by more than 5 seconds
            if prev_lap_time - ocr.lap_time_seconds > 5.0:
                # Validate this is a real crossing, not an OCR error:
                # 1. The new lap time should be small (near start of lap), OR
                # 2. The lap number should have incremented
                lap_incremented = (
                    ocr.lap_number is not None and
                    prev_lap_number is not None and
                    ocr.lap_number > prev_lap_number
                )
                new_lap_time_small = ocr.lap_time_seconds < 10.0  # Within 10 seconds of lap start

                if lap_incremented or new_lap_time_small:
                    # Estimate crossing time between prev and current frame
                    # The crossing happens when lap_time would have been 0
                    # If prev_lap_time is X, crossing is X seconds after prev frame
                    estimated_offset = prev_lap_time
                    crossing_video_time = frame_times[prev_idx] + estimated_offset

                    # Clamp to between the two frames
                    crossing_video_time = min(crossing_video_time, frame_times[i])

                    crossings.append(StartFinishCrossing(
                        frame_index=prev_idx,
                        video_time=crossing_video_time,
                        lap_before=prev_lap_number,
                        lap_after=ocr.lap_number
                    ))

        if ocr.lap_time_seconds is not None:
            prev_lap_time = ocr.lap_time_seconds
            prev_idx = i

        if ocr.lap_number is not None:
            prev_lap_number = ocr.lap_number

    return crossings


def find_red_circle(
    frame: np.ndarray,
    red_min: int = 200,
    red_max_gb: int = 80,
    min_radius: int = 3,
    max_radius: int = 30,
    search_region: Optional[tuple[int, int, int, int]] = None
) -> Optional[tuple[int, int, int]]:
    """
    Find the red circle (car marker) in a frame by detecting the red-filled region.

    This function finds the centroid of red pixels, not edges/outlines.
    It works by:
    1. Creating a mask of pixels that match the red color criteria
    2. Finding contours in that mask
    3. Returning the centroid of the largest red region

    Args:
        frame: RGB frame as numpy array
        red_min: Minimum R value for red detection (default: 200)
        red_max_gb: Maximum G and B values for red detection (default: 80)
        min_radius: Minimum circle radius to detect (default: 3)
        max_radius: Maximum circle radius to detect (default: 30)
        search_region: Optional (y1, y2, x1, x2) to restrict search area.
                      If None, defaults to top-right quadrant where track overlay typically is.

    Returns:
        (x, y, radius) of the detected circle in full frame coordinates, or None if not found
    """
    h, w = frame.shape[:2]

    # Default to top-right quadrant where track overlay typically is
    if search_region is None:
        y1, y2, x1, x2 = 0, h // 2, w // 2, w
    else:
        y1, y2, x1, x2 = search_region

    # Extract the search region
    region = frame[y1:y2, x1:x2]

    # Create red mask - detect the RED FILL, not the black outline
    # This finds uniformly colored red pixels
    r = region[:, :, 0]
    g = region[:, :, 1]
    b = region[:, :, 2]
    red_mask = ((r >= red_min) & (g <= red_max_gb) & (b <= red_max_gb)).astype(np.uint8) * 255

    # Apply morphological operations to clean up the mask
    # Use a slightly larger kernel to fill in small gaps
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of red regions (the filled red circle)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour that fits our size criteria
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:  # Skip tiny noise
            continue
        # Estimate radius from area (area = pi * r^2)
        estimated_radius = np.sqrt(area / np.pi)
        if min_radius <= estimated_radius <= max_radius:
            valid_contours.append((contour, area, estimated_radius))

    if not valid_contours:
        return None

    # Use the largest valid contour
    largest_contour, area, radius = max(valid_contours, key=lambda x: x[1])

    # Calculate centroid of the red region
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Adjust to full frame coordinates
        return (cx + x1, cy + y1, int(radius))

    return None
