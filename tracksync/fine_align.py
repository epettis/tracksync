"""Fine alignment via local feature matching and relative pose estimation.

This module implements the fine alignment stage (Stage B) of scene-based video
alignment. Given coarse frame-level correspondences from DTW, it refines sync
points to sub-frame precision using:

1. Local feature matching (SuperPoint/ALIKED + LightGlue) between candidate
   frame pairs
2. Essential matrix estimation and inlier verification via RANSAC
3. Longitudinal motion analysis via relative pose to find zero-crossing points

The approach is robust to camera mounting differences (6-DoF pose variations)
by explicitly factoring out the fixed mounting offset and tracking only the
longitudinal progress component.

Design reference: docs/scene_alignment_design.md §5
"""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class FeatureMatcher(Protocol):
    """Protocol for feature matching between image pairs.

    Implementations can use any local feature detector/matcher (LightGlue,
    LoFTR, etc.) or synthetic correspondences for testing.
    """

    name: str

    def match(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        mask_a: np.ndarray | None = None,
        mask_b: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match features between two images.

        Args:
            img_a: First image (RGB uint8 HxWx3)
            img_b: Second image (RGB uint8 HxWx3)
            mask_a: Optional static mask for image A (HxW bool, True = exclude)
            mask_b: Optional static mask for image B (HxW bool, True = exclude)

        Returns:
            pts_a: Matched keypoints in image A, shape [M, 2] (x, y)
            pts_b: Corresponding keypoints in image B, shape [M, 2] (x, y)
        """
        ...


@dataclass
class PairGeometry:
    """Geometric relationship between a pair of frames.

    Computed from matched feature points via essential matrix estimation
    and pose recovery.
    """

    n_inliers: int
    R: np.ndarray  # 3x3 rotation matrix (or None if failed)
    t_unit: np.ndarray  # 3x1 unit translation vector (or None if failed)
    inlier_mask: np.ndarray  # [M] bool mask of inliers


@dataclass
class FineResult:
    """Result of fine alignment refinement for a single sync point.

    Contains the refined sub-frame timestamp, verification status, and
    diagnostic data about the longitudinal sweep.
    """

    t_b: float | None  # Refined timestamp in video B (None if failed)
    verified: bool  # True if geometry verification passed
    n_inliers: int  # Best inlier count in the refinement window
    longitudinal: np.ndarray  # Per-frame longitudinal components


def intrinsics_from_fov(width: int, height: int, fov_deg: float = 90.0) -> np.ndarray:
    """Construct a pinhole camera intrinsic matrix from horizontal FOV.

    Assumes principal point at image center and square pixels (fx = fy).

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Horizontal field of view in degrees

    Returns:
        K: 3x3 intrinsic matrix
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx  # Square pixels
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K


def score_pair(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    K_a: np.ndarray,
    K_b: np.ndarray,
) -> PairGeometry:
    """Estimate essential matrix and recover relative pose between two frames.

    Uses RANSAC to robustly estimate the essential matrix from point
    correspondences, then recovers the rotation and translation (up to scale).

    Args:
        pts_a: Keypoints in image A, shape [M, 2]
        pts_b: Keypoints in image B, shape [M, 2]
        K_a: Intrinsic matrix for camera A, 3x3
        K_b: Intrinsic matrix for camera B, 3x3

    Returns:
        PairGeometry with inlier count, R, t_unit, and inlier mask.
        Returns n_inliers=0 with None R/t if estimation fails.
    """
    # Handle degenerate cases
    if len(pts_a) < 5 or len(pts_b) < 5:
        # Need at least 5 points for essential matrix
        return PairGeometry(
            n_inliers=0,
            R=None,
            t_unit=None,
            inlier_mask=np.zeros(len(pts_a), dtype=bool) if len(pts_a) > 0 else np.array([], dtype=bool),
        )

    # Normalize points using intrinsics
    # Convert to homogeneous coordinates and apply K^-1
    K_a_inv = np.linalg.inv(K_a)
    K_b_inv = np.linalg.inv(K_b)

    # Convert to homogeneous
    pts_a_h = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pts_b_h = np.hstack([pts_b, np.ones((len(pts_b), 1))])

    # Normalize
    pts_a_norm = (K_a_inv @ pts_a_h.T).T[:, :2]
    pts_b_norm = (K_b_inv @ pts_b_h.T).T[:, :2]

    # Estimate essential matrix with normalized coordinates
    # Using identity K since points are already normalized
    E, mask = cv2.findEssentialMat(
        pts_a_norm,
        pts_b_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.001,  # Normalized coordinates, small threshold
    )

    # Handle failure or degenerate cases
    if E is None or mask is None:
        return PairGeometry(
            n_inliers=0,
            R=None,
            t_unit=None,
            inlier_mask=np.zeros(len(pts_a), dtype=bool),
        )

    # OpenCV sometimes returns multiple E matrices stacked
    # Take the first one
    if E.shape[0] > 3:
        E = E[:3, :]

    mask = mask.ravel().astype(bool)
    n_inliers = int(np.sum(mask))

    if n_inliers < 5:
        return PairGeometry(
            n_inliers=n_inliers,
            R=None,
            t_unit=None,
            inlier_mask=mask,
        )

    # Recover pose from essential matrix
    # recoverPose needs normalized points
    _, R, t, pose_mask = cv2.recoverPose(
        E,
        pts_a_norm,
        pts_b_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        mask=mask.astype(np.uint8),
    )

    # t is unit vector from camera A to camera B in A's coordinate frame
    t_unit = t.reshape(3)

    return PairGeometry(
        n_inliers=n_inliers,
        R=R,
        t_unit=t_unit,
        inlier_mask=mask,
    )


def longitudinal_component(t_unit: np.ndarray, forward_axis: np.ndarray) -> float:
    """Project translation vector onto track-forward direction.

    Args:
        t_unit: Unit translation vector (3D)
        forward_axis: Track-forward direction (3D, unit vector)

    Returns:
        Signed scalar projection of t_unit onto forward_axis
    """
    return float(np.dot(t_unit, forward_axis))


def estimate_forward_axis(prev_to_next_geometry: list[PairGeometry]) -> np.ndarray:
    """Estimate track-forward direction from consecutive frame translations.

    Uses the dominant direction of motion from vehicle B's own consecutive
    frame pairs to establish its track-forward axis.

    Args:
        prev_to_next_geometry: List of PairGeometry for consecutive frames
            in video B (frame i to frame i+1)

    Returns:
        forward_axis: Unit 3D vector representing track-forward direction
    """
    # Accumulate translation vectors from consecutive frames
    translations = []
    for geom in prev_to_next_geometry:
        if geom.n_inliers >= 5 and geom.t_unit is not None:
            translations.append(geom.t_unit)

    if len(translations) == 0:
        # Fallback: assume camera looks along -Z axis (forward)
        return np.array([0.0, 0.0, -1.0])

    # Average the translation vectors (they should be roughly aligned
    # with the forward motion direction)
    mean_translation = np.mean(translations, axis=0)

    # Normalize to unit vector
    norm = np.linalg.norm(mean_translation)
    if norm < 1e-6:
        # Degenerate case: no motion detected
        return np.array([0.0, 0.0, -1.0])

    forward_axis = mean_translation / norm
    return forward_axis


def refine_sync_point(
    frame_a: np.ndarray,
    frames_b: list[np.ndarray],
    times_b: np.ndarray,
    matcher: FeatureMatcher,
    K_a: np.ndarray,
    K_b: np.ndarray,
    min_inliers: int = 30,
) -> FineResult:
    """Refine a sync point to sub-frame precision via zero-crossing analysis.

    Matches frame A against each frame in the B window, computes the
    longitudinal component of the relative translation, finds the zero
    crossing (where vehicle B passes vehicle A's position), and linearly
    interpolates to get sub-frame timing.

    Args:
        frame_a: Reference frame from video A (RGB uint8)
        frames_b: List of frames from video B window (RGB uint8)
        times_b: Timestamps of frames_b, shape [N]
        matcher: Feature matcher implementation
        K_a: Intrinsic matrix for camera A
        K_b: Intrinsic matrix for camera B
        min_inliers: Minimum inlier count for verification

    Returns:
        FineResult with refined timestamp, verification status, and diagnostics
    """
    if len(frames_b) == 0:
        return FineResult(
            t_b=None,
            verified=False,
            n_inliers=0,
            longitudinal=np.array([]),
        )

    # Score frame_a against every frame in frames_b
    geometries = []
    for frame_b in frames_b:
        pts_a, pts_b = matcher.match(frame_a, frame_b)
        geom = score_pair(pts_a, pts_b, K_a, K_b)
        geometries.append(geom)

    # Find best inlier count
    inlier_counts = [g.n_inliers for g in geometries]
    best_inliers = max(inlier_counts) if inlier_counts else 0

    # Check verification threshold
    if best_inliers < min_inliers:
        return FineResult(
            t_b=None,
            verified=False,
            n_inliers=best_inliers,
            longitudinal=np.array(inlier_counts, dtype=np.float32),
        )

    # Estimate forward axis from consecutive B frames
    # Match consecutive frames in frames_b to get their relative motion
    prev_to_next = []
    for i in range(len(frames_b) - 1):
        pts_i, pts_j = matcher.match(frames_b[i], frames_b[i + 1])
        geom = score_pair(pts_i, pts_j, K_b, K_b)
        prev_to_next.append(geom)

    forward_axis = estimate_forward_axis(prev_to_next)

    # Compute longitudinal components for A-to-B geometries
    longitudinal_vals = []
    for geom in geometries:
        if geom.t_unit is not None and geom.n_inliers >= 5:
            long_comp = longitudinal_component(geom.t_unit, forward_axis)
        else:
            long_comp = np.nan
        longitudinal_vals.append(long_comp)

    longitudinal_arr = np.array(longitudinal_vals, dtype=np.float32)

    # Find sign change (zero crossing)
    # Valid indices are those with non-NaN values
    valid_mask = ~np.isnan(longitudinal_arr)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return FineResult(
            t_b=None,
            verified=False,
            n_inliers=best_inliers,
            longitudinal=longitudinal_arr,
        )

    # Check for sign change
    valid_vals = longitudinal_arr[valid_indices]
    sign_changes = []
    for i in range(len(valid_vals) - 1):
        if valid_vals[i] * valid_vals[i + 1] < 0:
            # Sign change between valid_indices[i] and valid_indices[i+1]
            idx_before = valid_indices[i]
            idx_after = valid_indices[i + 1]
            sign_changes.append((idx_before, idx_after))

    if len(sign_changes) == 0:
        # No zero crossing found
        return FineResult(
            t_b=None,
            verified=False,
            n_inliers=best_inliers,
            longitudinal=longitudinal_arr,
        )

    # Take the first sign change (should be the main crossing)
    idx_before, idx_after = sign_changes[0]

    # Linear interpolation to find zero crossing
    val_before = longitudinal_arr[idx_before]
    val_after = longitudinal_arr[idx_after]
    t_before = times_b[idx_before]
    t_after = times_b[idx_after]

    # Linear fit: val = a * t + b
    # At zero crossing: 0 = a * t_cross + b
    # t_cross = -b / a
    # where a = (val_after - val_before) / (t_after - t_before)
    #       b = val_before - a * t_before

    if abs(t_after - t_before) < 1e-9:
        # Degenerate case: same timestamp
        t_cross = t_before
    else:
        # Interpolate
        alpha = -val_before / (val_after - val_before)
        t_cross = t_before + alpha * (t_after - t_before)

    return FineResult(
        t_b=t_cross,
        verified=True,
        n_inliers=best_inliers,
        longitudinal=longitudinal_arr,
    )


def clamp_monotonic(sync_times: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Enforce forward-progress invariant on sync point timestamps.

    Each successive t_b must be strictly greater than the previous one.
    Violators are clamped to maintain monotonicity.

    Args:
        sync_times: List of (t_a, t_b) timestamp pairs

    Returns:
        Clamped list maintaining strict monotonicity in both t_a and t_b
    """
    if len(sync_times) <= 1:
        return sync_times.copy()

    clamped = [sync_times[0]]

    for i in range(1, len(sync_times)):
        t_a, t_b = sync_times[i]
        prev_t_a, prev_t_b = clamped[-1]

        # Enforce strict monotonicity
        # t_a should already be monotonic from input construction,
        # but we clamp t_b if needed
        if t_b <= prev_t_b:
            # Clamp to just above previous t_b
            # Use a small epsilon based on typical frame time (1/30 fps ~ 0.033s)
            epsilon = 0.001  # 1 ms
            t_b = prev_t_b + epsilon

        clamped.append((t_a, t_b))

    return clamped
