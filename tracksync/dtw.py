"""Dynamic Time Warping for video alignment.

This module implements banded, slope-constrained, open-end Dynamic Time Warping
(DTW) for aligning video frame sequences based on embedding cost matrices.

The DTW implementation supports:
- Sakoe-Chiba band constraints for computational efficiency and robustness
- Slope constraints to prevent unrealistic local speed differences
- Open-ended alignment (subsequence DTW) to handle slack around lap boundaries
- Path smoothing via monotone interpolation for stable frame mapping

Design reference: docs/scene_alignment_design.md §4.4
"""

# Pure NumPy/SciPy implementation, no torch dependency.

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class DtwResult:
    """Result of DTW alignment.

    Attributes:
        path: Monotonic index pairs [K, 2] mapping frames from sequence A to B.
        path_cost: Total alignment cost along the optimal path.
        margins: Per-frame confidence for sequence A [N_A]. Margin is the
            difference between the best off-path cost within the band and the
            on-path cost. Higher margins indicate more confident alignment.
    """
    path: np.ndarray  # [K, 2] int
    path_cost: float
    margins: np.ndarray  # [N_A] float


class SmoothPath:
    """Smoothed monotonic mapping from sequence A time to sequence B time.

    Uses PCHIP interpolation to provide a smooth, strictly monotonically
    increasing function t_B = f(t_A).

    Attributes:
        t_a_start: Minimum valid input time (start of trimmed domain).
        t_a_end: Maximum valid input time (end of trimmed domain).
    """

    def __init__(self, interpolator: PchipInterpolator, t_a_start: float, t_a_end: float):
        self._interpolator = interpolator
        self.t_a_start = t_a_start
        self.t_a_end = t_a_end

    def __call__(self, t_a: float) -> float:
        """Evaluate the smooth mapping at time t_a.

        Args:
            t_a: Time in sequence A (should be within [t_a_start, t_a_end]).

        Returns:
            Corresponding time in sequence B.
        """
        return float(self._interpolator(t_a))


def dtw_align(
    cost: np.ndarray,
    band_pct: float = 0.10,
    open_end_frames: tuple[int, int] = (0, 0),
    slope_max: float = 2.0,
) -> DtwResult:
    """Perform banded, slope-constrained, open-end DTW alignment.

    Args:
        cost: Cost matrix [N_A, N_B] where cost[i, j] is the dissimilarity
            between frame i of sequence A and frame j of sequence B.
        band_pct: Normalized band width as a fraction of sequence length.
            Constrains |i/N_A - j/N_B| <= band_pct.
        open_end_frames: Number of frames at start and end where the path
            may begin/terminate freely. Tuple of (start_slack, end_slack).
        slope_max: Maximum allowed local slope. Steps are constrained to
            [1/slope_max, slope_max].

    Returns:
        DtwResult containing the optimal path, its cost, and per-frame margins.
    """
    N_A, N_B = cost.shape

    # Initialize DP table with infinity
    dp = np.full((N_A, N_B), np.inf, dtype=np.float64)

    # Backpointer stores step info: (di, dj) encoded as single int
    # We use a simple encoding: value = di * 1000 + dj
    # Using int16 to handle large indices (max ~32000)
    backpointer = np.zeros((N_A, N_B), dtype=np.int16)

    # Path length (in cells) accompanying dp, used to normalize open-end
    # selection so shorter paths are not spuriously preferred.
    steps = np.zeros((N_A, N_B), dtype=np.int32)

    # Initialize open start positions
    start_slack = open_end_frames[0]
    for i in range(min(start_slack + 1, N_A)):
        for j in range(min(start_slack + 1, N_B)):
            # Check if (i, j) is within the band
            if abs(i / max(N_A - 1, 1) - j / max(N_B - 1, 1)) <= band_pct:
                dp[i, j] = cost[i, j]
                steps[i, j] = 1

    # Pre-compute valid bands for each row
    bands = []
    for i in range(N_A):
        i_norm = i / max(N_A - 1, 1)
        j_min = max(0, int(np.floor((i_norm - band_pct) * (N_B - 1))))
        j_max = min(N_B - 1, int(np.ceil((i_norm + band_pct) * (N_B - 1))))
        bands.append((j_min, j_max))

    # Fill DP table with vectorized inner loop
    for i in range(N_A):
        j_min, j_max = bands[i]

        if i == 0 and j_min == 0:
            j_min = 1  # Skip (0,0) which is already initialized

        # Skip positions in the open start region
        if i <= start_slack:
            j_start = max(j_min, start_slack + 1)
        else:
            j_start = j_min

        if j_start > j_max:
            continue

        # Prepare candidate arrays for vectorized comparison
        j_range = np.arange(j_start, j_max + 1)
        n_j = len(j_range)

        # Initialize with inf
        best_costs = np.full(n_j, np.inf, dtype=np.float64)
        best_steps = np.zeros(n_j, dtype=np.int16)

        # Diagonal step (1, 1)
        if i > 0:
            mask = j_range > 0
            diag_costs = np.where(mask, dp[i-1, j_range - 1], np.inf)
            better = diag_costs < best_costs
            best_costs = np.where(better, diag_costs, best_costs)
            best_steps = np.where(better, 1001, best_steps)  # encode (1,1)

        # Horizontal steps: (1, k) for k in [1, slope_max]
        if i > 0:
            for k in range(1, int(slope_max) + 1):
                mask = j_range >= k
                horiz_costs = np.where(mask, dp[i-1, j_range - k], np.inf)
                better = horiz_costs < best_costs
                best_costs = np.where(better, horiz_costs, best_costs)
                best_steps = np.where(better, 1000 + k, best_steps)  # encode (1,k)

        # Vertical steps: (k, 1) for k in [1, slope_max]
        for k in range(1, int(slope_max) + 1):
            if i >= k:
                mask = j_range > 0
                vert_costs = np.where(mask, dp[i-k, j_range - 1], np.inf)
                better = vert_costs < best_costs
                best_costs = np.where(better, vert_costs, best_costs)
                best_steps = np.where(better, k * 1000 + 1, best_steps)  # encode (k,1)

        # Update DP table and backpointers
        valid = best_costs < np.inf
        j_valid = j_range[valid]
        dp[i, j_valid] = best_costs[valid] + cost[i, j_valid]
        backpointer[i, j_valid] = best_steps[valid]
        di_arr = (best_steps[valid] // 1000).astype(np.int32)
        dj_arr = (best_steps[valid] % 1000).astype(np.int32)
        steps[i, j_valid] = steps[i - di_arr, j_valid - dj_arr] + 1

    # Find the best end position within open-end slack. Compare average
    # per-cell cost (dp / steps) rather than raw accumulated cost: raw
    # comparison would always favor the shortest path in the slack region,
    # degenerating to a single cell when the slack is large.
    end_slack = open_end_frames[1]
    best_end_avg = np.inf
    best_end_cost = np.inf
    best_end_i, best_end_j = N_A - 1, N_B - 1

    for i in range(max(0, N_A - end_slack - 1), N_A):
        for j in range(max(0, N_B - end_slack - 1), N_B):
            if dp[i, j] < np.inf and steps[i, j] > 0:
                avg = dp[i, j] / steps[i, j]
                if avg < best_end_avg:
                    best_end_avg = avg
                    best_end_cost = dp[i, j]
                    best_end_i, best_end_j = i, j

    # Backtrack to recover the path
    path = []
    i, j = best_end_i, best_end_j

    while True:
        path.append([i, j])

        # Check if we've reached a start position
        if i <= start_slack and j <= start_slack and (i == 0 or j == 0):
            break

        if i == 0 and j == 0:
            break

        # Decode backpointer
        step = backpointer[i, j]
        if step == 0:
            break  # No valid predecessor

        di = step // 1000
        dj = step % 1000

        if i >= di and j >= dj:
            i, j = i - di, j - dj
        else:
            break

        # Safety check to avoid infinite loops
        if len(path) > N_A + N_B:
            break

    path.reverse()
    path = np.array(path, dtype=np.int32)

    # Compute margins: best off-path cost within band minus on-path cost
    margins = np.zeros(N_A, dtype=np.float64)

    # Build a set for fast on-path lookup
    path_set = set((i, j) for i, j in path)

    # Build a lookup for on-path costs
    on_path_cost = np.full(N_A, np.inf, dtype=np.float64)
    for i, j in path:
        on_path_cost[i] = min(on_path_cost[i], cost[i, j])

    # For each A frame, find the best off-path cost within the band (vectorized)
    for i in range(N_A):
        j_min, j_max = bands[i]

        # Get all costs in the band for this row
        band_costs = cost[i, j_min:j_max+1]
        j_indices = np.arange(j_min, j_max + 1)

        # Create mask for off-path positions
        off_path_mask = np.array([(i, j) not in path_set for j in j_indices])

        if np.any(off_path_mask):
            best_off_path = np.min(band_costs[off_path_mask])
        else:
            best_off_path = np.inf

        # Margin is off-path minus on-path
        if on_path_cost[i] < np.inf and best_off_path < np.inf:
            margins[i] = best_off_path - on_path_cost[i]
        else:
            margins[i] = 0.0

    return DtwResult(path=path, path_cost=best_end_cost, margins=margins)


def smooth_path(
    path: np.ndarray,
    times_a: np.ndarray,
    times_b: np.ndarray,
) -> SmoothPath:
    """Smooth the DTW path into a monotone interpolant.

    Fits a strictly monotonically increasing PCHIP interpolator to the
    discrete DTW path, mapping times from sequence A to sequence B.

    Args:
        path: DTW path [K, 2] with index pairs into sequences A and B.
        times_a: Timestamps for sequence A [N_A].
        times_b: Timestamps for sequence B [N_B].

    Returns:
        SmoothPath callable that maps t_A -> t_B with trimmed domain endpoints.
    """
    # Extract times along the path
    t_a_path = times_a[path[:, 0]]
    t_b_path = times_b[path[:, 1]]

    # PCHIP requires strictly increasing x values
    # Remove duplicate t_a values, keeping the first occurrence
    unique_mask = np.concatenate([[True], t_a_path[1:] > t_a_path[:-1]])
    t_a_unique = t_a_path[unique_mask]
    t_b_unique = t_b_path[unique_mask]

    # Ensure we have at least 2 distinct points for interpolation
    if len(t_a_unique) < 2:
        if t_a_path[-1] > t_a_path[0]:
            # Fallback: use path endpoints
            t_a_unique = np.array([t_a_path[0], t_a_path[-1]])
            t_b_unique = np.array([t_b_path[0], t_b_path[-1]])
        else:
            raise ValueError(
                "DTW path is degenerate (fewer than 2 distinct A-times); "
                "cannot build a smooth time mapping. This usually indicates "
                "an alignment failure (e.g., excessive open-end slack)."
            )

    # Ensure strict monotonicity in y values by adding tiny increments where needed
    # This handles cases where the path has horizontal segments
    for i in range(1, len(t_b_unique)):
        if t_b_unique[i] <= t_b_unique[i-1]:
            # Add a minimal increment to maintain strict monotonicity
            t_b_unique[i] = t_b_unique[i-1] + 1e-9

    # Create PCHIP interpolator (monotone by construction)
    interpolator = PchipInterpolator(t_a_unique, t_b_unique)

    return SmoothPath(
        interpolator=interpolator,
        t_a_start=float(t_a_unique[0]),
        t_a_end=float(t_a_unique[-1]),
    )
