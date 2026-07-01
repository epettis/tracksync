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
