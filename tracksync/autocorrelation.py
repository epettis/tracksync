"""
Video autocorrelation module for automatic video alignment.

This module provides functions to align two racing videos by detecting
the position of a car marker on a track overlay using binary frame correlation.

The algorithm uses:
1. Median frame to detect static template regions (track overlay, text)
2. Template matching to find the red car marker position
3. OCR to detect segment numbers for filtering matches
4. Masked correlation focusing on relevant regions
"""

import numpy as np
import cv2
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


@dataclass
class CorrelationResult:
    """Result of a correlation search."""
    best_time: float
    best_score: int
    all_scores: list[tuple[float, int]]  # (time, score) pairs
    segment_a: Optional[int] = None
    segment_b: Optional[int] = None


@dataclass
class AlignmentPoint:
    """A single alignment point between two videos."""
    time_a: float
    time_b: float
    correlation: int
    segment: Optional[int] = None


@dataclass
class VideoAnalysis:
    """Pre-computed analysis of a video for alignment."""
    static_mask: np.ndarray  # Binary mask of static template regions
    red_circle_positions: list[tuple[float, int, int]]  # (time, x, y) of red circle centers
    segment_times: dict[int, list[float]]  # segment_number -> list of times
    duration: float
    fps: float
    frame_count: int


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


def compute_correlation(frame_a: np.ndarray, frame_b: np.ndarray) -> int:
    """
    Compute correlation between two binary frames.

    Uses element-wise AND and counts matching 1s.
    This is equivalent to dot product for binary arrays.

    Args:
        frame_a: Binary frame from video A
        frame_b: Binary frame from video B

    Returns:
        Integer count of matching white/red pixels
    """
    return int(np.sum(frame_a & frame_b))


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


def compute_static_mask(
    video: cv2.VideoCapture,
    num_samples: int = 100,
    threshold: int = 240
) -> np.ndarray:
    """
    Compute a mask of static template regions by taking the median of sampled frames.

    Static regions (track overlay, text templates) will have consistent pixel values
    across frames, while dynamic regions (scenery, car marker) will vary.

    Args:
        video: OpenCV VideoCapture object
        num_samples: Number of frames to sample for median (default: 100)
        threshold: Minimum value to consider a pixel as "white" (default: 240)

    Returns:
        Binary mask (H, W) where 1 indicates static white regions
    """
    duration, fps, frame_count = get_video_info(video)

    # Sample frames evenly across the video
    sample_times = np.linspace(0, duration * 0.95, num_samples)

    frames = []
    for t in sample_times:
        frame = get_frame_at_time(video, t)
        if frame is not None:
            # Convert to grayscale for simpler median computation
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frames.append(gray)

    if not frames:
        raise ValueError("Could not read any frames from video")

    # Stack frames and compute median
    frame_stack = np.stack(frames, axis=0)
    median_frame = np.median(frame_stack, axis=0).astype(np.uint8)

    # Threshold to get static white regions
    static_mask = (median_frame >= threshold).astype(np.uint8)

    return static_mask


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


def analyze_video(
    video_path: str,
    sample_interval: float = 1.0,
    progress_callback: Optional[callable] = None
) -> VideoAnalysis:
    """
    Pre-analyze a video to extract static mask, red circle positions, and segment times.

    Args:
        video_path: Path to video file
        sample_interval: How often to sample for red circle and segment detection
        progress_callback: Optional callback(current_time, total_duration, message)

    Returns:
        VideoAnalysis object with all pre-computed data
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        duration, fps, frame_count = get_video_info(video)

        # Step 1: Compute static mask
        if progress_callback:
            progress_callback(0, duration, "Computing static mask...")
        static_mask = compute_static_mask(video)

        # Step 2: Detect red circles and segments throughout video
        red_positions: list[tuple[float, int, int]] = []
        segment_times: dict[int, list[float]] = {}

        time = 0.0
        while time < duration:
            if progress_callback:
                progress_callback(time, duration, "Analyzing frames...")

            frame = get_frame_at_time(video, time)
            if frame is not None:
                # Detect red circle
                circle = find_red_circle(frame)
                if circle:
                    x, y, _ = circle
                    red_positions.append((time, x, y))

                # Detect segment number
                segment = detect_segment_number(frame)
                if segment is not None:
                    if segment not in segment_times:
                        segment_times[segment] = []
                    segment_times[segment].append(time)

            time += sample_interval

        return VideoAnalysis(
            static_mask=static_mask,
            red_circle_positions=red_positions,
            segment_times=segment_times,
            duration=duration,
            fps=fps,
            frame_count=frame_count
        )

    finally:
        video.release()


def get_segment_at_time(analysis: VideoAnalysis, time: float) -> Optional[int]:
    """
    Get the segment number at a specific time based on pre-analyzed data.

    Args:
        analysis: VideoAnalysis object
        time: Time in seconds

    Returns:
        Segment number, or None if unknown
    """
    best_segment = None
    best_distance = float('inf')

    for segment, times in analysis.segment_times.items():
        for t in times:
            distance = abs(t - time)
            if distance < best_distance and distance < 2.0:  # Within 2 seconds
                best_distance = distance
                best_segment = segment

    return best_segment


def get_red_circle_mask(
    analysis: VideoAnalysis,
    time: float,
    frame_shape: tuple[int, int],
    radius: int = 20
) -> np.ndarray:
    """
    Create a mask for the red circle region at a specific time.

    Args:
        analysis: VideoAnalysis object
        time: Time in seconds
        frame_shape: (height, width) of the frame
        radius: Radius of the mask circle

    Returns:
        Binary mask with 1s where the red circle is expected
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)

    # Find the closest red circle position
    best_pos = None
    best_distance = float('inf')

    for t, x, y in analysis.red_circle_positions:
        distance = abs(t - time)
        if distance < best_distance:
            best_distance = distance
            best_pos = (x, y)

    if best_pos and best_distance < 2.0:  # Within 2 seconds
        cv2.circle(mask, best_pos, radius, 1, -1)

    return mask


def find_best_match_by_position(
    circle_a: tuple[int, int, int],
    video_b: cv2.VideoCapture,
    center_time: Optional[float],
    window_seconds: float = 10.0,
    initial_search_seconds: float = 20.0,
    segment_filter: Optional[int] = None,
    analysis_b: Optional[VideoAnalysis] = None
) -> CorrelationResult:
    """
    Find the frame in video B with the closest red circle position to frame A.

    This method matches frames by finding where the car marker (red circle)
    is in the same position on the track overlay.

    Args:
        circle_a: (x, y, radius) of the red circle in frame A
        video_b: OpenCV VideoCapture for video B
        center_time: Time in video B to center the search window (None for first frame)
        window_seconds: Seconds before/after center_time to search (default: 10.0)
        initial_search_seconds: Seconds to search for first frame (default: 20.0)
        segment_filter: Optional segment number to prefer matching frames
        analysis_b: Optional VideoAnalysis for segment info

    Returns:
        CorrelationResult with best match time, score (negative distance), and all scores
    """
    duration, fps, frame_count = get_video_info(video_b)
    x_a, y_a, _ = circle_a

    if center_time is None:
        start_time = 0.0
        end_time = min(initial_search_seconds, duration)
    else:
        start_time = max(0.0, center_time - window_seconds)
        end_time = min(duration, center_time + window_seconds)

    all_scores: list[tuple[float, int]] = []
    best_time = start_time
    best_distance = float('inf')

    # Sample at ~1 second intervals for speed
    sample_interval = 1.0
    time_b = start_time

    while time_b <= end_time:
        frame_b = get_frame_at_time(video_b, time_b)
        if frame_b is None:
            time_b += sample_interval
            continue

        circle_b = find_red_circle(frame_b)
        if circle_b is None:
            time_b += sample_interval
            continue

        x_b, y_b, _ = circle_b
        distance = np.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

        # Apply segment bonus/penalty
        if segment_filter is not None and analysis_b is not None:
            segment_b = get_segment_at_time(analysis_b, time_b)
            if segment_b is not None:
                if segment_b == segment_filter:
                    distance *= 0.5  # Bonus for matching segment
                else:
                    distance *= 2.0  # Penalty for mismatched segment

        # Store as negative distance so higher = better (for consistency)
        score = int(-distance)
        all_scores.append((time_b, score))

        if distance < best_distance:
            best_distance = distance
            best_time = time_b

        time_b += sample_interval

    return CorrelationResult(
        best_time=best_time,
        best_score=int(-best_distance) if best_distance < float('inf') else 0,
        all_scores=all_scores,
        segment_a=segment_filter,
        segment_b=get_segment_at_time(analysis_b, best_time) if analysis_b else None
    )


def find_best_match(
    binary_frame_a: np.ndarray,
    video_b: cv2.VideoCapture,
    center_time: Optional[float],
    window_frames: int = 100,
    initial_search_seconds: float = 20.0,
    white_threshold: int = 240,
    red_min: int = 200,
    red_max_gb: int = 80,
    mask: Optional[np.ndarray] = None,
    segment_filter: Optional[int] = None,
    analysis_b: Optional[VideoAnalysis] = None
) -> CorrelationResult:
    """
    Find the frame in video B that best matches video A frame.

    Search strategy:
    - If center_time is None (first frame): search first initial_search_seconds of video B
    - Otherwise: search ±window_frames around center_time

    Args:
        binary_frame_a: Binarized frame from video A
        video_b: OpenCV VideoCapture for video B
        center_time: Time in video B to center the search window (None for first frame)
        window_frames: Number of frames before/after center_time to search (default: 100)
        initial_search_seconds: Seconds to search for first frame (default: 20.0)
        white_threshold: Threshold for white detection
        red_min: Minimum R value for red detection
        red_max_gb: Maximum G/B values for red detection
        mask: Optional binary mask to restrict correlation to specific regions
        segment_filter: Optional segment number to filter candidate frames
        analysis_b: Optional VideoAnalysis for segment filtering

    Returns:
        CorrelationResult with best match time, score, and all scores
    """
    duration, fps, frame_count = get_video_info(video_b)

    if center_time is None:
        # First frame: search first N seconds
        start_time = 0.0
        end_time = min(initial_search_seconds, duration)
    else:
        # Subsequent frames: search ±window_frames around center_time
        window_seconds = window_frames / fps
        start_time = max(0.0, center_time - window_seconds)
        end_time = min(duration, center_time + window_seconds)

    # Calculate frame times to check
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Apply mask to frame A if provided
    if mask is not None:
        masked_frame_a = binary_frame_a & mask
    else:
        masked_frame_a = binary_frame_a

    all_scores: list[tuple[float, int]] = []
    best_time = start_time
    best_score = -1

    for frame_idx in range(start_frame, end_frame + 1):
        time_b = frame_idx / fps

        # Filter by segment if specified
        if segment_filter is not None and analysis_b is not None:
            segment_b = get_segment_at_time(analysis_b, time_b)
            if segment_b is not None and segment_b != segment_filter:
                continue

        frame_b = get_frame_at_time(video_b, time_b)

        if frame_b is None:
            continue

        binary_b = binarize_frame(frame_b, white_threshold, red_min, red_max_gb)

        # Apply mask to frame B if provided
        if mask is not None:
            binary_b = binary_b & mask

        score = compute_correlation(masked_frame_a, binary_b)

        all_scores.append((time_b, score))

        if score > best_score:
            best_score = score
            best_time = time_b

    return CorrelationResult(
        best_time=best_time,
        best_score=best_score,
        all_scores=all_scores,
        segment_a=segment_filter,
        segment_b=get_segment_at_time(analysis_b, best_time) if analysis_b else None
    )


def align_videos(
    video_a_path: str,
    video_b_path: str,
    sample_interval: float = 1.0,
    window_frames: int = 100,
    initial_search_seconds: float = 20.0,
    white_threshold: int = 240,
    red_min: int = 200,
    red_max_gb: int = 80,
    progress_callback: Optional[callable] = None,
    use_analysis: bool = True
) -> list[AlignmentPoint]:
    """
    Generate alignment timestamps between two videos.

    When use_analysis=True (default), this function:
    1. Pre-analyzes both videos to find static template regions
    2. Detects red circle positions throughout both videos
    3. Extracts segment numbers via OCR
    4. Uses masked correlation focusing only on template + red circle regions
    5. Filters matches to only consider frames with matching segment numbers

    Args:
        video_a_path: Path to reference video (video A)
        video_b_path: Path to target video (video B)
        sample_interval: How often to sample in seconds (default: 1.0)
        window_frames: Search window in frames for subsequent frames (default: 100)
        initial_search_seconds: Search duration for first frame (default: 20.0)
        white_threshold: Threshold for white detection (default: 240)
        red_min: Minimum R value for red detection (default: 200)
        red_max_gb: Maximum G/B values for red detection (default: 80)
        progress_callback: Optional callback(current_time, total_duration, message) for progress
        use_analysis: Whether to use advanced analysis (static mask, segments). Default True.

    Returns:
        List of AlignmentPoint objects representing matched timestamps
    """
    video_a = cv2.VideoCapture(video_a_path)
    video_b = cv2.VideoCapture(video_b_path)

    if not video_a.isOpened():
        raise ValueError(f"Could not open video A: {video_a_path}")
    if not video_b.isOpened():
        raise ValueError(f"Could not open video B: {video_b_path}")

    try:
        duration_a, fps_a, _ = get_video_info(video_a)
        duration_b, fps_b, _ = get_video_info(video_b)

        # Pre-analyze videos if enabled
        analysis_a: Optional[VideoAnalysis] = None
        analysis_b: Optional[VideoAnalysis] = None
        combined_mask: Optional[np.ndarray] = None

        if use_analysis:
            # Analyze video A
            if progress_callback:
                progress_callback(0, duration_a, "Analyzing video A...")
            video_a.release()
            analysis_a = analyze_video(video_a_path, sample_interval)
            video_a = cv2.VideoCapture(video_a_path)

            # Analyze video B
            if progress_callback:
                progress_callback(0, duration_b, "Analyzing video B...")
            video_b.release()
            analysis_b = analyze_video(video_b_path, sample_interval)
            video_b = cv2.VideoCapture(video_b_path)

            # Create combined mask from static regions of both videos
            # The mask includes areas that are consistently white in BOTH videos
            combined_mask = analysis_a.static_mask & analysis_b.static_mask

            if progress_callback:
                progress_callback(0, duration_a, "Aligning videos...")

        alignments: list[AlignmentPoint] = []
        previous_best_time: Optional[float] = None

        time_a = 0.0
        while time_a < duration_a:
            if progress_callback:
                progress_callback(time_a, duration_a, "Aligning...")

            # Get frame from video A
            frame_a = get_frame_at_time(video_a, time_a)
            if frame_a is None:
                time_a += sample_interval
                continue

            # Get segment number for filtering
            segment_a = None
            if analysis_a is not None:
                segment_a = get_segment_at_time(analysis_a, time_a)

            # Try position-based matching first (uses red circle position)
            circle_a = find_red_circle(frame_a, red_min, red_max_gb)

            if circle_a is not None and use_analysis:
                # Use position-based matching (more accurate)
                # The window should be based on how much time could pass between samples
                # At 5s intervals, allow ±10s search (sample_interval * 2)
                search_window = max(sample_interval * 2, 5.0)
                result = find_best_match_by_position(
                    circle_a,
                    video_b,
                    center_time=previous_best_time,
                    window_seconds=search_window,
                    initial_search_seconds=initial_search_seconds,
                    segment_filter=segment_a,
                    analysis_b=analysis_b
                )
            else:
                # Fallback to correlation-based matching
                binary_a = binarize_frame(frame_a, white_threshold, red_min, red_max_gb)

                # Build the correlation mask
                mask = combined_mask.copy() if combined_mask is not None else None

                # Add red circle regions from video A to the mask
                if mask is not None and analysis_a is not None:
                    h, w = mask.shape
                    red_mask_a = get_red_circle_mask(analysis_a, time_a, (h, w), radius=25)
                    mask = mask | red_mask_a

                result = find_best_match(
                    binary_a,
                    video_b,
                    center_time=previous_best_time,
                    window_frames=window_frames,
                    initial_search_seconds=initial_search_seconds,
                    white_threshold=white_threshold,
                    red_min=red_min,
                    red_max_gb=red_max_gb,
                    mask=mask,
                    segment_filter=segment_a,
                    analysis_b=analysis_b
                )

            alignments.append(AlignmentPoint(
                time_a=time_a,
                time_b=result.best_time,
                correlation=result.best_score,
                segment=segment_a
            ))

            # Update tracking position for next iteration
            previous_best_time = result.best_time

            time_a += sample_interval

        return alignments

    finally:
        video_a.release()
        video_b.release()
