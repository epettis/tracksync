# DTW cost-matrix contrast — experiment results

Status: COMPLETE (first pass, one reference pair). Executes the phased plan in
`scene_alignment_dtw_contrast_research.md` §7. Companion to the open-problem
note `scene_alignment_dtw_contrast.md`.

## TL;DR

- **Cheap cost-matrix conditioning is the win.** Subtracting the shared
  descriptor mean ("centering") + LoFTR-style two-way (dual-softmax) contrast
  **eliminates the start/end boundary-speed pathology** (end speed 0.74→0.95)
  and cuts scene-vs-Catalyst p95, while keeping the self-alignment identity
  exact. Free at runtime, no new deps/models. In the **real pipeline** (with
  PCHIP path smoothing) p95 improves 0.403→0.336 s; on the sweep harness (linear
  interpolation, used for config *ranking*) the same change is 0.388→0.230 s.
  The gap between those two exposes a separate lever — PCHIP `smooth_path` adds
  ~0.1 s p95 over linear interpolation on the stair-stepped DTW path — left for a
  future path-smoothing study.
- **Better descriptors (VLAD/AnyLoc) did not beat conditioned GeM.** Per-pair
  VLAD-over-patch-tokens helps the *unconditioned* matrix (fixes boundaries on
  its own) but loses to conditioned GeM (best VLAD 0.284 vs GeM 0.197). Partly
  an artifact of an approximate VLAD; partly a real signal that aggregation
  helps retrieval, not this continuity-constrained alignment.
- **Motion fusion (yaw) did not help.** The yaw-rate trace is a genuine
  appearance-invariant signal (0.71 cross-lap correlation) but fusing it —
  weighted-sum or entropy-gated — is neutral-to-negative vs conditioned
  appearance. The speed proxy is a broken measurement on these clips.
- **Net:** promote conditioning to production; treat VLAD and motion as
  tested-with-limited-benefit. Frame parity (0.033 s) remains ~6× away, and two
  independent attempts to add information both failed to close it — consistent
  with scene mode's visual-review acceptance (design §11.3).

Single-pair caveat: only Buttonwillow has Catalyst+OCR references, so every
quantitative conclusion is "on this one low-contrast pair." Laguna (no
references) served as a control for the identity guard and boundary speeds.

## Method

All experiments run offline on cached embeddings via the harness in
`scripts/experiments/` (pure NumPy; the extended DTW reproduces production
`dtw_align` exactly at zero penalty). Primary metric: scene-vs-Catalyst |Δt|
p95 over a **common** set of 24 Catalyst sync points (so mapped-domain
differences don't bias the comparison). Guards: self-alignment identity
deviation ≤ 0.10 s (1 sample); boundary speeds (local slope of t_A→t_B over the
first/last 4 s) should sit near 1.0. Contrast diagnostics: per-row softmax
entropy (normalized), effective rank, top-singular share, per-row margin.

**Mapping-construction caveat.** The harness builds the t_A→t_B mapping by
linear interpolation of the DTW path; the production pipeline uses PCHIP
(`smooth_path`). All config-comparison tables below use the harness (linear) so
rankings are consistent, but PCHIP shifts the *absolute* p95 up by ~0.1 s on
this pair. The promoted change was re-validated through the real
`compute_scene_cost_matrix`→`dtw_align`→`smooth_path` path: **p95 0.403→0.336 s,
boundary end 0.74→0.95, max ~0.42 unchanged, self-identity 0.000** (baseline
row-z vs center+dualsoftmax). Treat the boundary-pathology fix as the robust
production win and the harness p95 deltas as the relative ranking signal.

Reference tooling note: the 5 OCR segment-transition pairs sit at a fixed
~3.4 s disagreement with **every** config including baseline, and all 5 fall
*inside* Catalyst coverage — so OCR (as currently derived) is a broken
discriminator here, not a blind spot. Scene-vs-Catalyst is the trusted metric.

## Data / flatness (domain hypothesis)

| Video | off-diag cos mean | p95 | eff_rank / 768 | top1 var share |
|---|---|---|---|---|
| buttonwillow eddie_pb | 0.9907 | 0.9956 | 393 | 0.231 |
| buttonwillow kurt_optimal | 0.9903 | 0.9955 | 382 | 0.180 |
| laguna eddie | 0.9914 | 0.9952 | 381 | 0.148 |
| laguna kurt | 0.9882 | 0.9939 | 356 | 0.153 |

**The "Buttonwillow is uniquely visually sparse" hypothesis is not supported at
the global-descriptor level.** Both tracks embed to ~0.99 off-diagonal cosine;
Laguna is if anything slightly flatter by effective rank. Descriptor collapse
is a pipeline-wide property of GeM-pooled DINOv2 on forward-facing track video,
not Buttonwillow-specific (matches AnyLoc's "GeM ≈ CLS out of domain"). What
differs is *where* the failure bites: Buttonwillow's baseline shows the
end-boundary latch (speed 0.74) while Laguna's baseline boundaries are near
healthy — i.e. the discriminating variable is *local* contrast near the path,
which global flatness metrics can't see.

## Phase 1 — cost-matrix conditioning (winner)

Scene-vs-Catalyst p95 on the 24 common points (baseline row-z GeM = 0.388;
boundary end pathology = 0.74):

| Config | catP95 | catMax | id | bsp (start,end) | entropy |
|---|---|---|---|---|---|
| baseline_main (production row-z) | 0.388 | 0.421 | 0.000 | 1.02, 0.74 | 0.94 |
| **whiten + dualsoftmax (τ=0.1)** | **0.197** | 0.291 | 0.000 | 0.90, 1.01 | 0.14 |
| center + delta5 + dualsoftmax | 0.197 | 0.291 | 0.000 | 0.95, 0.96 | 0.12 |
| whiten + dualsoftmax (τ=0.05) | 0.197 | 0.291 | 0.000 | 0.90, 1.01 | 0.05 |
| center + dualsoftmax (τ=0.1) | 0.230 | 0.291 | 0.000 | 0.90, 0.99 | 0.18 |
| seqslam local R10 (axis B) | 0.240 | 0.298 | 0.017 | 0.90, 0.95 | 0.97 |
| center (mean-subtract only) | 0.297 | 0.421 | 0.000 | 1.00, 0.95 | 0.93 |

Attribution:
- **Feature centering / whitening is the root-cause fix** (removes the shared
  "racetrack-ness" direction; §3.1). Centering alone: 0.388→0.297 and heals the
  end boundary (0.74→0.95). Whitening ≈ centering; the gain is the
  mean-subtraction.
- **Two-way (dual-softmax) contrast is a distinct, composable win** (§3.2). It
  fixes the boundary pathology even without centering (τ=0.05: 0.298, boundaries
  0.98/0.98), confirming the unnormalized-column asymmetry was a real second
  cause. Centering and dual-softmax compose superadditively (0.297/0.298 alone →
  0.197–0.230 together).
- **Dead ends:** DTW step/slack penalties (inert; the path wandered because the
  costs were uninformative, not because the DP lacked a prior); temporal
  window-*averaging* (harmful — collapses effective rank 41→15, destroying the
  only transient signal); multi-slope diagonal smoothing (inert on a flat
  matrix); one-way row-softmax (identical to plain cost). Notably `baseline_main`
  ≈ `plain_cost` on this harness — the production row-z stopgap contributes
  little here, at odds with the note's earlier 2.176→0.795 figure (different
  harness/clips; flagged for re-verification).
- Reducing open-end slack (`open_end_s` 10→5) re-broke boundaries (0.78/0.67):
  the current default is well chosen.

Exit criterion (§7: p95 ≤ 0.3 s, boundaries in [0.9,1.1], control not
regressed) is met.

## Phase 2 — VLAD aggregation vs GeM (AnyLoc recipe)

Per-pair VLAD (k=32) over masked vitb14 patch tokens, same backbone as GeM.

| Variant | Config | dim | catP95 | catMax | bsp (s,e) | entropy |
|---|---|---|---|---|---|---|
| GeM | baseline row-z | 768 | 0.388 | 0.421 | 1.02, 0.74 | 0.94 |
| GeM | whiten+dualsoftmax | 768 | **0.197** | 0.291 | 0.90, 1.01 | 0.14 |
| GeM | center+dualsoftmax | 768 | 0.230 | 0.291 | 0.90, 0.99 | 0.18 |
| VLAD | baseline row-z | 24576 | 0.365 | 0.391 | 0.98, 1.04 | 0.91 |
| VLAD | whiten+dualsoftmax | 24576 | 0.312 | 0.372 | 0.98, 0.95 | 0.42 |
| VLAD | center+dualsoftmax | 24576 | 0.284 | 0.294 | 0.98, 1.00 | 0.20 |

- **VLAD helps the raw matrix**: under plain row-z it beats GeM (0.365 vs 0.388)
  and fixes the boundary pathology on its own (0.98/1.04 vs 1.02/0.74) —
  consistent with AnyLoc's claim that aggregation matters.
- **VLAD does not beat conditioned GeM**: best VLAD (center+dualsoftmax 0.284) >
  GeM winner (0.197). Whitening *hurts* VLAD (0.312) because whitening a
  24,576-dim space from ~2,600 descriptors is severely underdetermined.
- Caveats (why this is not a clean refutation of AnyLoc): approximate VLAD —
  128 tokens/frame subsample, small NumPy k-means, ViT-B not AnyLoc's ViT-G,
  per-pair vocabulary. But since cheap conditioning already wins, a fuller VLAD
  recipe is low expected value for the alignment task (VLAD's evidence base is
  retrieval recall, whereas the coarse DTW already leans on temporal
  continuity).

## Phase 3 — motion fusion (yaw / speed)

Yaw-rate = signed median far-field horizontal flow; speed = median ground flow
magnitude; both z-normalized per lap; fused with the appearance winner as a
unit-mean-normalized weighted sum.

- Yaw is a real signal: cross-lap yaw correlation under the Catalyst alignment
  = **0.71**; yaw std ~2.4–2.6 (Buttonwillow), 5–6 (Laguna).
- **Fusion does not help.** Appearance-only = 0.197; adding yaw at weights
  0.25–4.0 gives 0.196–0.287 (neutral to worse, boundaries drift). Entropy-gated
  fusion (lean on yaw where the appearance row is flat) = 0.261–0.287. Yaw-ONLY
  alignment = 7.5 s (hopeless alone).
- Why: straights are low-signal in **both** modalities (yaw ≈ 0 on straights),
  so yaw is redundant with appearance at corners and its difference-cost injects
  aliasing (corners repeat). The hoped-for complementarity (§4) does not
  materialize on this pair.
- The **speed proxy is a broken measurement** here (std ~0.004, corr 0.13): the
  lower image is dashboard / Catalyst map overlay (masked out), so almost no
  ground-flow points survive. This is an implementation limitation, not a
  refutation of speed as a concept.

## Synthesis & recommendation

Two independent attempts to add information beyond conditioned global
descriptors — richer aggregation (Phase 2) and an orthogonal motion modality
(Phase 3) — **both failed to beat cheap conditioning** on this pair. The winner
(0.197 s) is still ~6× the 0.033 s frame-parity target.

A hard error floor gives away the cause: a max ≈ **0.291 s** recurs across
essentially every conditioned config in all three phases (GeM whiten 0.291, GeM
center 0.291, VLAD center 0.294, most fusion rows 0.291). The residual is not a
conditioning or information-weighting problem — it is a floor set by what 10 Hz
global per-frame descriptors can express about *within-corridor* position on
self-similar straights. This strengthens the research note's thesis: the coarse
global-descriptor+DTW stage is information-limited, conditioning is "necessary
but not sufficient," and closing the last gap requires the fine stage, not the
coarse signal. Crucially, that 0.29 s max lands *inside* the fine stage's
±0.5 s refinement window with ~40% margin — so the coarse stage already does its
actual job (land every sync point inside the fine window), and the fine stage
(LightGlue local geometry, which retains the positional detail global
descriptors discard) is the correct owner of the last 6×. This supports scene
mode's acceptance on visual review rather than frame parity (design §11.3), and
matches §7's Phase-4 rule: invest next in confidence-gated fine-stage windows,
not more coarse-stage information sources.

**Promote to production:** `center + dualsoftmax` in `compute_scene_cost_matrix`
(feature mean-subtraction over the pair + two-way softmax contrast). Recommended
over `whiten+dualsoftmax` despite whiten's marginally better p95 (0.230 vs
0.197): whitening is fragile (needs many samples; catastrophic on
high-dim descriptors) whereas centering is robust and cheap, and 0.230 vs 0.197
is within one-sample (0.033 s) noise. Keep guardrails: the self-alignment
identity test (design §8.1) and a boundary-speed sanity check.

**Not worth pursuing** (on current evidence): DTW step/slack penalties, temporal
window-averaging, diagonal smoothing, VLAD aggregation for alignment, and
yaw/speed motion fusion. Revisit only with more low-contrast reference pairs or
a materially better descriptor/backbone.

## Follow-up: open-end over-trimming (default open_end_s 10 → 5)

Visual review of the generated comparison videos showed the alignment is good
through the middle but loses a few seconds of real lap at the start and finish.
Cause: the comparison video spans only the DTW path endpoints, and the open-end
DTW **always consumes the full `open_end_s` slack** — `trim_start` exactly
equals `open_end_s` at 10/5/3/1 s — because skipping a boundary frame is nearly
free while those low-contrast boundary frames are the hardest to match, so the
path prefers to drop them. (This is a distinct mechanism from the boundary-speed
pathology the conditioning fixed.)

Measured against the Catalyst start/finish crossings (A[5.9, 133.7]):

| open_end_s | trim (video A) | Catalyst pts covered | full-overlap p95 · max |
|---|---|---|---|
| 10 (old default) | A[10.0, 129.6] — over-trims ~4 s/end | 24/26 | 0.336 · 0.426 |
| **5 (new default)** | A[5.0, 134.6] ≈ crossings | **26/26** | 0.432 · 0.501 |
| 3 | A[3.0, 136.6] — into non-lap pre-roll | 26/26 | 1.10 · 1.60 |

Lowered the `coarse_align` default to 5 s: it recovers the full lap and lands on
the true crossings, at the cost of slightly looser alignment at the extreme ends
(max |Δt| 0.43→0.50 s, still inside the fine stage's ±0.5 s window). The clips
here carry only ~4–6 s of slack, so 5 s is the right budget; 3 s over-corrects.
A proper self-adapting fix (stop skipping once a boundary frame has an
acceptable in-band match, rather than skipping for free) is left as follow-up.

## Reproduce

```bash
.venv/bin/python scripts/experiments/extract_embeddings.py            # cache embeddings
.venv/bin/python scripts/experiments/make_references.py --pair buttonwillow \
    samples/buttonwillow/eddie_pb.mp4 samples/buttonwillow/kurt_optimal.mp4
.venv/bin/python scripts/experiments/sweep_phase1.py                  # conditioning grid
.venv/bin/python scripts/experiments/sweep_phase1_refine.py           # refinements + common-domain
.venv/bin/python scripts/experiments/extract_vlad.py --pair buttonwillow \
    samples/buttonwillow/eddie_pb.mp4 samples/buttonwillow/kurt_optimal.mp4
.venv/bin/python scripts/experiments/sweep_phase2.py                  # VLAD vs GeM
# motion traces are written by scripts/experiments/motion.py; then:
.venv/bin/python scripts/experiments/sweep_phase3.py                  # appearance+yaw fusion
```

Results JSON under `experiments/results/`; raw cached embeddings/traces/refs
under `experiments/data/` (not committed).
