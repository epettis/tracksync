# Open research note: DTW cost-matrix contrast at low-feature tracks

Status: NARROWED — the study is done (one reference pair). Cost-matrix
conditioning (§3) is **resolved and promoted to production**; the two
information-adding directions (descriptors §2, motion §4) were tested and gave
no benefit on the available pair; the open item is now frame-parity via the
fine stage (§5) vs. acceptance on visual review (design §11.3). See §4.4 of
`scene_alignment_design.md` for the coarse stage.

The literature survey and phased plan are in
`scene_alignment_dtw_contrast_research.md`; the full results (all three phases,
Fable-evaluated) are in `scene_alignment_dtw_contrast_experiments.md`.

**Single-pair caveat:** every quantitative result below is from ONE track pair
with references (`eddie_pb`/`kurt_optimal`, Buttonwillow, 24 Catalyst points).
The Phase-0 feature-rich-pair-with-references prerequisite was never met (Laguna
has no Catalyst overlay), so nothing here should be generalized across tracks
until at least one more low-contrast reference pair exists.

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

## Domain hypothesis — NOT supported at the global-descriptor level

The original hypothesis was that Buttonwillow (visually sparse desert track) is
uniquely flat and feature-rich circuits are less affected. Measured, this does
not hold: Buttonwillow and Laguna both embed to ~0.99 off-diagonal cosine, and
Laguna is if anything slightly flatter by effective rank. **Descriptor collapse
is a pipeline-wide property of GeM-pooled DINOv2 on forward-facing track video,
not Buttonwillow-specific** (cf. AnyLoc's "GeM ≈ CLS out of domain"). What
differs between tracks is not global flatness but *where the failure bites*:
Buttonwillow's baseline latches at the clip end (boundary speed 0.74) while
Laguna's baseline boundaries are near-healthy — i.e. the discriminating variable
is *local* contrast near the path, which global metrics can't see. Any future
cross-track predictor should be a local diagnostic (cost margin along the path),
not a global flatness metric.

## Resolution (promoted to production)

`compute_scene_cost_matrix` now applies **per-pair centering + two-way
(dual-softmax) contrast** (τ=0.1): subtract the shared mean descriptor of both
videos (removing the dominant "track appearance" component that inflates every
cosine toward 1.0 — the near-rank-1 root cause), then
`cost = -log(softmax_row · softmax_col)` so a cell is cheap only when its frames
are mutual best matches. This keeps a self-alignment's diagonal at exactly zero
cost (identity optimal). The prior per-row z-score stopgap is superseded — on
the study harness it was ~inert (it left the column axis unnormalized and did
not remove the shared background).

Results on `eddie_pb` / `kurt_optimal` through the **real pipeline**
(`compute_scene_cost_matrix`→`dtw_align`→PCHIP `smooth_path`), common 24 Catalyst
points (a *different* evaluation set than the 0.795 s figure previously recorded
here, so the two are not directly comparable):

| Scene-vs-Catalyst \|Δt\| | baseline row-z | center+dualsoftmax |
|---|---|---|
| p95 | 0.403 s | 0.336 s |
| max | 0.425 s | 0.426 s |
| boundary speed (start, end) | 1.02, 0.74 | 0.98, 0.95 |

The **boundary-speed pathology fix** (end 0.74→0.95) is the robust win; the p95
gain is more modest through PCHIP than on the config-ranking harness (which uses
linear interpolation and shows 0.388→0.230, and 0.197 for a fragile
`whiten+dualsoftmax` variant not shipped). The ~0.1 s gap between PCHIP and
linear is a separate path-smoothing lever, out of scope here. See the
experiments doc for the full comparison.

**Residual:** p95 0.336 s is still ~10x the 1-frame parity target, and a hard
error floor (max ≈ 0.42 s here, ≈ 0.29 s on the linear harness) recurs across
essentially every conditioned config —
it looks descriptor-limited, not conditioning-limited. Conditioning removed the
rank-1 background and the boundary pathology but cannot manufacture positional
information the global descriptors already discarded. Scene mode remains accepted
on visual review rather than frame-parity (design §11.3). Notably, that 0.29 s
max lands *inside* the fine stage's ±0.5 s refinement window with ~40% margin, so
the fine stage is the right owner of the remaining gap.

## Directions — outcomes

Surveyed in `scene_alignment_dtw_contrast_research.md`; full results in
`scene_alignment_dtw_contrast_experiments.md`.

- **Cost-matrix conditioning** (§3): **RESOLVED / promoted.** Centering +
  dual-softmax; DP step/slack penalties, window-averaging, and diagonal
  smoothing were dead ends.
- **Better descriptors** (§2): **tested, no benefit.** Per-pair VLAD-over-patch-
  tokens (AnyLoc recipe, k=32, ViT-B, subsampled tokens) lost to conditioned GeM
  (0.284 vs 0.197); it helps the *unconditioned* matrix but conditioning already
  captures that gain more cheaply. Test weakened by an underdetermined
  24,576-dim whitening and token subsampling — inconclusive in principle,
  unnecessary in practice.
- **Complementary non-appearance signals** (§4): **tested, no benefit (yaw);
  speed untested.** The yaw-rate trace is well-measured (cross-lap corr 0.71) but
  globally aliased (yaw-only DTW p95 7.5 s); fixed-weight and entropy-gated
  fusion both tie-or-degrade conditioned appearance, because corners are
  redundant across modalities and straights are flat in both. The speed proxy
  was a broken *measurement* (ground ROI occluded by dash/overlay), so
  speed-profile fusion remains untested as a concept.
- **Lean harder on the fine stage** (§5): **the live direction.** With the coarse
  max error ~0.29 s inside the ±0.5 s fine window, invest in confidence-gated
  fine-stage windows (soft-DTW posterior / cycle-back error) rather than more
  coarse-stage information sources.
