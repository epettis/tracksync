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

# Lazy torch imports for LightGlue matcher; geometry utilities use only
# OpenCV and NumPy.
