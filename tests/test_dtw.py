"""Tests for DTW alignment module.

Tests use synthetic cost matrices to verify:
- Path recovery for identity and offset alignments
- Band and slope constraint enforcement
- smooth_path monotonicity and PCHIP interpolation
- Margin computation for confidence
- Performance on large matrices
"""

import time

import numpy as np
import pytest

from tracksync.dtw import DtwResult, SmoothPath, dtw_align, smooth_path


def test_identity_cost_diagonal_path():
    """Identity cost (zeros on diagonal) should recover diagonal path."""
    N = 100
    # Create cost matrix: zeros on diagonal, ones elsewhere
    cost = np.ones((N, N), dtype=np.float32)
    np.fill_diagonal(cost, 0.0)

    result = dtw_align(cost, band_pct=0.2)

    # Path should be diagonal or very close to it
    assert len(result.path) > 0
    assert result.path[0, 0] == 0  # Start at (0, 0)
    assert result.path[-1, 0] == N - 1  # End at (N-1, N-1)
    assert result.path[-1, 1] == N - 1

    # Check that path is mostly diagonal (i ≈ j)
    for i, j in result.path:
        assert abs(i - j) <= 2, f"Path deviates too far from diagonal at ({i}, {j})"

    # Cost should be close to zero (sum of diagonal elements)
    assert result.path_cost < N * 0.1


def test_constant_offset_within_slack():
    """Constant offset within open-end slack should be recovered exactly."""
    N = 120
    offset = 10  # 10 frame offset

    # Create cost matrix: zeros on offset diagonal
    cost = np.ones((N, N), dtype=np.float32)
    for i in range(N - offset):
        cost[i, i + offset] = 0.0

    result = dtw_align(cost, band_pct=0.2, open_end_frames=(15, 15))

    # Path should recover the offset
    # Check that j ≈ i + offset for most of the path
    offsets = result.path[:, 1] - result.path[:, 0]
    median_offset = int(np.median(offsets))

    assert abs(median_offset - offset) <= 1, f"Expected offset {offset}, got {median_offset}"


def test_smooth_synthetic_warp():
    """Smooth sinusoidal warp with slope <= 2 should be recovered within 1 index."""
    N_A = 300
    N_B = 300
    amplitude = 8

    # Create a smooth warp: index_b = index_a + amplitude * sin(2π * index_a / N_A)
    cost = np.ones((N_A, N_B), dtype=np.float32)

    ground_truth_j = []
    for i in range(N_A):
        j_true = i + amplitude * np.sin(2 * np.pi * i / N_A)
        j_true = int(np.round(np.clip(j_true, 0, N_B - 1)))
        ground_truth_j.append(j_true)
        # Set low cost near the true mapping
        for dj in range(-2, 3):
            j = j_true + dj
            if 0 <= j < N_B:
                cost[i, j] = abs(dj) * 0.1

    # Verify maximum local slope is <= 2 (allowing exactly 2)
    max_slope = 0
    for i in range(1, N_A):
        slope = abs(ground_truth_j[i] - ground_truth_j[i-1])
        max_slope = max(max_slope, slope)
    assert max_slope <= 2, f"Test precondition failed: max slope {max_slope} > 2"

    result = dtw_align(cost, band_pct=0.3, slope_max=2.0)

    # Build path lookup
    path_dict = {i: j for i, j in result.path}

    # Check deviation from ground truth
    max_deviation = 0
    for i in range(N_A):
        if i in path_dict:
            deviation = abs(path_dict[i] - ground_truth_j[i])
            max_deviation = max(max_deviation, deviation)

    assert max_deviation <= 1, f"Max path deviation {max_deviation} > 1 index"


def test_band_constraint_respected():
    """DTW path should respect Sakoe-Chiba band constraint."""
    N_A = 150
    N_B = 150
    band_pct = 0.1

    # Random cost matrix
    cost = np.random.rand(N_A, N_B).astype(np.float32)

    result = dtw_align(cost, band_pct=band_pct)

    # Check each point on the path
    for i, j in result.path:
        i_norm = i / max(N_A - 1, 1)
        j_norm = j / max(N_B - 1, 1)
        assert abs(i_norm - j_norm) <= band_pct + 0.01, \
            f"Path point ({i}, {j}) violates band constraint: |{i_norm} - {j_norm}| > {band_pct}"


def test_slope_constraint_respected():
    """DTW path should respect slope constraints."""
    N = 100
    slope_max = 2.0

    # Random cost matrix with identity bias
    cost = np.random.rand(N, N).astype(np.float32)
    np.fill_diagonal(cost, 0.0)

    result = dtw_align(cost, band_pct=0.2, slope_max=slope_max)

    # Check local slopes
    for k in range(1, len(result.path)):
        di = result.path[k, 0] - result.path[k-1, 0]
        dj = result.path[k, 1] - result.path[k-1, 1]

        assert di > 0 or dj > 0, "Path must be monotonic"

        # Compute slope (handle zero denominators)
        if di > 0 and dj > 0:
            slope = max(di / dj, dj / di)
            assert slope <= slope_max + 0.1, \
                f"Local slope {slope} exceeds slope_max {slope_max} at step {k}"


def test_smooth_path_strictly_monotonic():
    """smooth_path should produce strictly monotonic output."""
    # Create a random valid path
    N_A = 100
    N_B = 110

    # Generate a monotonic path with some variation
    path = []
    i, j = 0, 0
    while i < N_A - 1 or j < N_B - 1:
        path.append([i, j])
        # Randomly choose step direction
        step = np.random.choice(['diag', 'right', 'down'])
        if step == 'diag' and i < N_A - 1 and j < N_B - 1:
            i += 1
            j += 1
        elif step == 'right' and j < N_B - 1:
            j += 1
        elif i < N_A - 1:
            i += 1
    path.append([N_A - 1, N_B - 1])
    path = np.array(path, dtype=np.int32)

    # Create time arrays
    times_a = np.linspace(0, 100, N_A)
    times_b = np.linspace(0, 110, N_B)

    smoothed = smooth_path(path, times_a, times_b)

    # Sample the smoothed function
    t_samples = np.linspace(smoothed.t_a_start, smoothed.t_a_end, 200)
    t_b_values = [smoothed(t) for t in t_samples]

    # Check strict monotonicity
    for k in range(1, len(t_b_values)):
        assert t_b_values[k] > t_b_values[k-1], \
            f"smooth_path not strictly monotonic at sample {k}: {t_b_values[k-1]} >= {t_b_values[k]}"


def test_margins_sharp_valley():
    """Sharp low-cost valley should produce large margins."""
    N = 80
    valley_width = 5

    # Create cost with a sharp valley along the diagonal
    cost = np.ones((N, N), dtype=np.float32) * 10.0
    for i in range(N):
        for dj in range(-valley_width, valley_width + 1):
            j = i + dj
            if 0 <= j < N:
                if dj == 0:
                    cost[i, j] = 0.0  # On-path cost
                else:
                    cost[i, j] = 5.0  # Off-path but in band

    result = dtw_align(cost, band_pct=0.3)

    # Margins should be positive (off-path cost > on-path cost)
    # For frames on the diagonal, margin should be ~5.0
    positive_margins = result.margins > 0
    assert np.sum(positive_margins) > N * 0.5, \
        "Expected majority of frames to have positive margins"

    # Check that some margins are substantial
    assert np.max(result.margins) > 3.0, \
        f"Expected large margins in sharp valley, got max {np.max(result.margins)}"


def test_margins_flat_cost():
    """Flat cost rows should produce near-zero margins."""
    N = 60

    # Create uniform cost matrix
    cost = np.ones((N, N), dtype=np.float32) * 5.0

    result = dtw_align(cost, band_pct=0.3)

    # All costs are equal, so margins should be near zero
    assert np.max(np.abs(result.margins)) < 0.5, \
        f"Expected near-zero margins for flat cost, got max {np.max(np.abs(result.margins))}"


def test_open_end_frames():
    """Open-end frames should allow flexible start/end positions."""
    N = 100
    offset_start = 8
    offset_end = 12

    # Create cost with offset alignment
    cost = np.ones((N, N), dtype=np.float32) * 10.0

    # Good alignment starts at (offset_start, 0) and ends at (N-1, N-1-offset_end)
    for i in range(offset_start, N):
        j = i - offset_start
        if 0 <= j < N - offset_end:
            cost[i, j] = 0.0

    result = dtw_align(cost, band_pct=0.3, open_end_frames=(10, 15))

    # Path should start near (offset_start, 0)
    assert result.path[0, 0] <= offset_start + 2, \
        f"Path start {result.path[0, 0]} doesn't use open-end flexibility"

    # Path should end near (N-1, N-1-offset_end)
    expected_end_j = N - 1 - offset_end
    assert abs(result.path[-1, 1] - expected_end_j) <= 15, \
        f"Path end {result.path[-1, 1]} doesn't use open-end flexibility"


def test_dtw_result_attributes():
    """DtwResult should have correct attributes and types."""
    N = 50
    cost = np.random.rand(N, N).astype(np.float32)

    result = dtw_align(cost)

    # Check types
    assert isinstance(result, DtwResult)
    assert isinstance(result.path, np.ndarray)
    assert isinstance(result.path_cost, float)
    assert isinstance(result.margins, np.ndarray)

    # Check shapes
    assert result.path.ndim == 2
    assert result.path.shape[1] == 2
    assert result.margins.shape == (N,)

    # Check path monotonicity
    for k in range(1, len(result.path)):
        assert result.path[k, 0] >= result.path[k-1, 0], "Path must be monotonic in A"
        assert result.path[k, 1] >= result.path[k-1, 1], "Path must be monotonic in B"


def test_smooth_path_attributes():
    """SmoothPath should expose trimmed domain endpoints."""
    path = np.array([[0, 0], [10, 12], [20, 24], [30, 36]], dtype=np.int32)
    times_a = np.linspace(0, 90, 31)
    times_b = np.linspace(0, 108, 37)

    smoothed = smooth_path(path, times_a, times_b)

    assert isinstance(smoothed, SmoothPath)
    assert hasattr(smoothed, 't_a_start')
    assert hasattr(smoothed, 't_a_end')

    # Domain endpoints should match path endpoints
    assert abs(smoothed.t_a_start - times_a[0]) < 1e-6
    assert abs(smoothed.t_a_end - times_a[30]) < 1e-6

    # Should be callable
    t_mid = (smoothed.t_a_start + smoothed.t_a_end) / 2
    t_b = smoothed(t_mid)
    assert isinstance(t_b, float)


def test_performance_1200x1200():
    """1200x1200 random matrix should align in < 5 seconds."""
    N = 1200
    cost = np.random.rand(N, N).astype(np.float32)

    start_time = time.time()
    result = dtw_align(cost, band_pct=0.10)
    elapsed = time.time() - start_time

    assert elapsed < 5.0, f"DTW took {elapsed:.2f}s, expected < 5s"
    assert len(result.path) > 0, "Path should not be empty"
    assert result.path.shape[1] == 2, "Path should have 2 columns"


def test_different_sequence_lengths():
    """DTW should handle sequences of different lengths."""
    N_A = 80
    N_B = 120

    # Create cost favoring a smooth diagonal-ish path
    cost = np.ones((N_A, N_B), dtype=np.float32)
    for i in range(N_A):
        j = int(i * N_B / N_A)
        for dj in range(-5, 6):
            if 0 <= j + dj < N_B:
                cost[i, j + dj] = abs(dj) * 0.1

    result = dtw_align(cost, band_pct=0.2)

    # Path should span both sequences
    assert result.path[0, 0] <= 5
    assert result.path[-1, 0] >= N_A - 6
    assert result.path[0, 1] <= 5
    assert result.path[-1, 1] >= N_B - 6


def test_smooth_path_minimal_path():
    """smooth_path should handle minimal paths (2 points)."""
    path = np.array([[0, 0], [10, 10]], dtype=np.int32)
    times_a = np.linspace(0, 30, 11)
    times_b = np.linspace(0, 30, 11)

    smoothed = smooth_path(path, times_a, times_b)

    # Should still work
    t_mid = 15.0
    t_b = smoothed(t_mid)
    assert isinstance(t_b, float)
    assert smoothed.t_a_start == times_a[0]
    assert smoothed.t_a_end == times_a[10]


def test_path_uniqueness_in_smooth_path():
    """smooth_path should handle paths with duplicate t_a values."""
    # Create a path where multiple steps have the same i (vertical steps)
    path = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2, 3],  # Vertical step: same i
        [2, 4],  # Vertical step: same i
        [3, 5],
        [4, 6],
    ], dtype=np.int32)

    times_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    times_b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    smoothed = smooth_path(path, times_a, times_b)

    # Should produce a valid monotonic function
    t_samples = np.linspace(smoothed.t_a_start, smoothed.t_a_end, 10)
    t_b_values = [smoothed(t) for t in t_samples]

    for k in range(1, len(t_b_values)):
        assert t_b_values[k] >= t_b_values[k-1], "Smoothed path not monotonic"
