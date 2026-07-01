"""Static car-body masking for scene-based video alignment.

This module provides automatic detection and masking of static regions in
onboard racing videos. Static regions include the car hood, dashboard,
roll cage, wheels, and picture-in-picture overlays that are fixed per-video
and differ between cars.

By masking these regions, we ensure that embedding extraction and feature
matching focus only on the actual track scene visible through the windshield,
improving alignment robustness across different camera mounting positions.

Design reference: docs/scene_alignment_design.md §4.2
"""

# Pure NumPy/OpenCV implementation, no torch dependency.

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_static_mask(
    frames: list[np.ndarray],
    var_threshold: float = 0.03,
    downsample: int = 4,
) -> np.ndarray:
    """Compute a mask identifying static regions (car body, overlays) in video frames.

    Static regions are detected by analyzing temporal variance of pixel intensities.
    Areas with low variance (relative to the global variance) are marked as static.
    This is robust to global brightness changes through per-frame normalization.

    Algorithm:
    1. Convert frames to grayscale
    2. Normalize each frame by its mean (removes global brightness flicker)
    3. Downsample for efficiency
    4. Compute per-pixel temporal variance
    5. Threshold variance (relative to global frame variance)
    6. Apply morphological operations to clean up the mask
    7. Upsample back to original resolution

    Args:
        frames: List of RGB uint8 frames (HxWx3), all same size
        var_threshold: Variance threshold relative to global variance (scale-invariant).
                      Lower values = more aggressive masking. Default 0.03.
        downsample: Downsampling factor for efficiency. Default 4.

    Returns:
        Boolean mask (HxW) where True = static region

    Warns:
        If the resulting mask covers > 60% of the frame
    """
    if not frames:
        raise ValueError("frames list cannot be empty")

    # Get original dimensions
    h_orig, w_orig = frames[0].shape[:2]
    h_down = h_orig // downsample
    w_down = w_orig // downsample

    # Convert to grayscale, downsample, then normalize per-frame
    # Normalization: subtract mean to center each frame, making variance insensitive
    # to global brightness shifts
    gray_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Downsample first
        gray_small = cv2.resize(gray, (w_down, h_down), interpolation=cv2.INTER_AREA)

        # Mean-center to remove global brightness offset
        gray_small = gray_small - gray_small.mean()

        gray_frames.append(gray_small)

    # Stack frames and compute per-pixel temporal variance
    frame_stack = np.stack(gray_frames, axis=0)  # Shape: (N, H_down, W_down)
    pixel_variance = np.var(frame_stack, axis=0)  # Shape: (H_down, W_down)

    # Compute global variance for scale-invariant thresholding
    global_variance = pixel_variance.mean()

    # Threshold: pixels with variance below threshold are static
    absolute_threshold = var_threshold * global_variance
    static_mask_small = (pixel_variance < absolute_threshold).astype(np.uint8) * 255

    # Morphological operations to clean up the mask (mirroring frame_analysis.py:674-676)
    # Close fills small holes, then a modest dilation connects nearby regions
    kernel = np.ones((3, 3), np.uint8)
    static_mask_small = cv2.morphologyEx(static_mask_small, cv2.MORPH_CLOSE, kernel)
    # Use a smaller dilation to avoid excessive expansion
    static_mask_small = cv2.dilate(static_mask_small, kernel, iterations=1)

    # Upsample back to original size
    static_mask = cv2.resize(
        static_mask_small,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert to boolean
    static_mask_bool = static_mask > 127

    # Check coverage and warn if > 60%
    coverage = static_mask_bool.sum() / static_mask_bool.size
    if coverage > 0.6:
        logger.warning(
            f"Static mask covers {coverage*100:.1f}% of frame (> 60% threshold). "
            "This may indicate limited camera motion or mostly static scene."
        )

    return static_mask_bool
