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
