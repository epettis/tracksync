"""Scene-based video alignment pipeline.

This module orchestrates the complete scene-based video alignment process,
composing the coarse and fine alignment stages into a unified pipeline.

The pipeline:
1. Extracts frame embeddings at fixed sampling rate (embedding.py)
2. Applies static car-body masking (masking.py)
3. Computes cosine cost matrix and runs banded DTW (dtw.py)
4. Refines sync points via local feature matching (fine_align.py)
5. Generates SyncResult output compatible with existing CSV/video generation

This module provides the top-level entry points for scene mode, callable
from the CLI (tracksync sync --mode scene).

Design reference: docs/scene_alignment_design.md §3, §6
"""

# Pure composition layer; torch dependencies are isolated in embedding.py
# and fine_align.py submodules.
