"""Turn analysis and apex detection for racing videos.

This module provides functions to detect turn apexes from car position
data by analyzing the interior angles formed by the car's trajectory.
"""

from typing import Optional

import numpy as np

from .models import TurnAnalysis, TurnApex


def calculate_interior_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculate the interior angle at p2 formed by points p1 -> p2 -> p3.

    Args:
        p1: First point (x, y)
        p2: Middle point (x, y) - the vertex
        p3: Third point (x, y)

    Returns:
        Interior angle in degrees (0-180). Smaller angle = sharper turn.
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)

    # Calculate angle using dot product
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    if mag1 == 0 or mag2 == 0:
        return 180.0  # No movement, straight line

    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(np.degrees(angle))


def compute_turn_analysis(
    positions: list[Optional[tuple[int, int]]],
    frame_times: list[float],
    fps: float,
    window_seconds: float = 3.0,
    min_prominence: float = 5.0,
    min_sharpness: float = 45.0
) -> TurnAnalysis:
    """
    Compute turn analysis from circle positions.

    For each point at time t, we look at points at t-window and t+window
    and calculate the interior angle. Local maxima in sharpness (180 - angle)
    indicate turn apexes.

    Args:
        positions: List of (x, y) positions (or None for missing)
        frame_times: List of timestamps
        fps: Frames per second
        window_seconds: Time offset for before/after points
        min_prominence: Minimum prominence for peak detection
        min_sharpness: Minimum sharpness (180 - angle) to qualify as a turn apex.
                       Default 45.0 means interior angle must be < 135 degrees.

    Returns:
        TurnAnalysis with angles, sharpness values, and detected apexes
    """
    from scipy.signal import find_peaks

    n = len(positions)
    window_frames = int(window_seconds * fps)

    angle_times = []
    angles = []

    for i in range(window_frames, n - window_frames):
        p1 = positions[i - window_frames]
        p2 = positions[i]
        p3 = positions[i + window_frames]

        # Skip if any position is missing
        if p1 is None or p2 is None or p3 is None:
            continue

        angle = calculate_interior_angle(p1, p2, p3)
        angle_times.append(frame_times[i])
        angles.append(angle)

    # Calculate sharpness (higher = sharper turn)
    sharpness = [180.0 - a for a in angles]

    # Find turn apexes (local maxima in sharpness)
    apexes = []
    if len(sharpness) >= 3:
        peaks, properties = find_peaks(sharpness, prominence=min_prominence, distance=3)

        for idx, peak_idx in enumerate(peaks):
            peak_sharpness = sharpness[peak_idx]
            # Only include apexes where sharpness exceeds threshold
            if peak_sharpness >= min_sharpness:
                apexes.append(TurnApex(
                    time=angle_times[peak_idx],
                    angle=angles[peak_idx],
                    sharpness=peak_sharpness,
                    prominence=float(properties['prominences'][idx])
                ))

    return TurnAnalysis(
        times=angle_times,
        angles=angles,
        sharpness=sharpness,
        apexes=apexes,
        window_seconds=window_seconds
    )
