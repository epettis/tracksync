"""Data structures for video frame analysis and cross-correlation.

This module contains the core dataclasses used for storing frame-level
data, video features, and cross-correlation results.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .autocorrelation import FrameOCRData, StartFinishCrossing


@dataclass
class FrameData:
    """Pre-computed data for a single frame.

    The frame and red_mask fields are optional to save memory during sync mode.
    They are only populated in debug mode where visualization is needed.
    """
    time: float
    circle: Optional[tuple[int, int, int]]  # (x, y, radius) or None
    segment: Optional[int]  # Segment number or None
    frame: Optional[np.ndarray] = None  # Original RGB frame (only in debug mode)
    red_mask: Optional[np.ndarray] = None  # Binary mask of red pixels (only in debug mode)
    ocr_data: Optional[FrameOCRData] = None  # OCR-extracted lap/segment/time info


@dataclass
class VideoFeatures:
    """Cached features extracted from a single video.

    This dataclass stores all the expensive-to-compute features for a video,
    allowing O(n) extraction instead of O(n^2) when comparing multiple videos.
    """
    video_path: str
    driver_name: str  # Extracted from filename
    fps: float
    frames: list[FrameData]
    frame_times: list[float]
    interpolated_ocr: list[FrameOCRData]
    crossings: list[StartFinishCrossing]
    # Pre-computed position arrays for vectorized correlation
    positions: list[Optional[tuple[int, int]]]  # Interpolated circle positions
    pos_array: np.ndarray  # Shape (n_frames, 2), NaN for missing


@dataclass
class CrossCorrelationResult:
    """Result of cross-correlating one frame against all frames in another video.

    The frame_a and best_frame_b fields are optional to save memory during sync mode.
    They are only populated in debug mode where visualization is needed.
    """
    time_a: float
    circle_a: Optional[tuple[int, int, int]]
    segment_a: Optional[int]
    best_time_b: Optional[float]  # None if no match found
    best_circle_b: Optional[tuple[int, int, int]]
    best_segment_b: Optional[int]
    best_distance: Optional[float]  # None if no match found
    all_distances: list[tuple[float, float]]  # (time_b, distance) for all frames in B
    frame_a: Optional[np.ndarray] = None  # Original frame (only in debug mode)
    best_frame_b: Optional[np.ndarray] = None  # Best matching frame (only in debug mode)
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
