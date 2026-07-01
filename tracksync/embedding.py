"""Frame embedding extraction for scene-based video alignment.

This module provides interfaces and implementations for extracting global
descriptors from video frames. Embeddings are used in the coarse alignment
stage to match frames across videos with different camera poses and mounting
positions.

The module supports pluggable embedders via the FrameEmbedder protocol,
with implementations ranging from simple deterministic baselines (GistEmbedder)
to pretrained vision transformers (DINOv2).

Design reference: docs/scene_alignment_design.md §4.3
"""

# Lazy torch imports: torch is only imported when actually creating a
# torch-based embedder, not at module import time. This allows the module
# to be imported in environments where torch is not installed.
