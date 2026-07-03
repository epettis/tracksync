# Open research note: DTW cost-matrix contrast at low-feature tracks

Status: OPEN — the fix currently on `main` is a stopgap; a dedicated study is
planned. See §4.4 of `scene_alignment_design.md` for the coarse stage and
`scene_alignment_validation.md` for the results referenced below.

A literature survey of the research directions below, with ranked
recommendations and a phased experiment plan, is in
`scene_alignment_dtw_contrast_research.md`.

## Problem

The scene coarse-alignment stage builds a cost matrix of `1 - cosine`
similarity between per-frame DINOv2 global descriptors
(`compute_scene_cost_matrix`, `tracksync/scene_align.py`) and runs open-ended
DTW over it.

At visually sparse tracks, every forward-facing frame around the lap embeds to
nearly the same vector (~99% similar), so each **row** of the cost matrix has
almost no contrast — there is no clear best-match column for a given moment.
With rows that flat, the open-ended DTW latches onto a wrong correspondence at
the clip boundaries and then rides its slope limit for ~10 s before converging
to the true diagonal. This surfaced as implausible ~0.5x segment speeds at the
start and end of the clip.

## Domain hypothesis

This is expected to be worst at **Buttonwillow Raceway Park** — a desert track
that is visually sparse (corner stations, tarmac, kerbs, little signage or
scenery), so global appearance barely changes with track position, which is
precisely what flattens per-row contrast. Other low-feature/desert tracks
likely behave the same; feature-rich circuits should be less affected.
Validating this across tracks is a good first step.

## Current stopgap (on `main`)

Per-row z-score normalization of the cost matrix, shifted so each row's minimum
is zero. This restores contrast while preserving relative magnitudes, and keeps
a self-alignment's diagonal at exactly zero cost (identity stays optimal).

Results on the `eddie_pb` / `kurt_optimal` pair:

| Scene-vs-Catalyst \|Δt\| | Before | After |
|---|---|---|
| p95 | 2.176 s | 0.795 s |
| max | 5.127 s | 1.611 s |
| boundary segment speed | ~0.50 | ~0.95 |

**Residual:** p95 0.795 s is still ~24x the 1-frame parity target. Row
normalization fixed the boundary drift but mid-lap disagreement remains — this
is a conditioning trick on a fundamentally low-information signal, not a cure.
Scene mode is accepted on visual review rather than frame-parity (design doc
§11.3).

## Directions to research

Surveyed in `scene_alignment_dtw_contrast_research.md` (§ references below
point there); none implemented yet.

- **Better descriptors for low-texture scenes** (§2): VPR-specialized
  backbones (MixVPR, CosPlace, AnyLoc recipes) or finer patch-token
  aggregation instead of a single global vector per frame.
- **Complementary non-appearance signals** (§4): optical flow / motion, a
  yaw-rate or curvature profile, horizon geometry.
- **Cost-matrix conditioning** beyond per-row z-score (§3).
- **Lean harder on the fine stage** where coarse contrast is inherently low
  (note the LightGlue/MPS cost if the fine stage is expanded) (§5).
