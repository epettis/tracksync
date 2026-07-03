# Research survey: approaches to the DTW cost-matrix contrast problem

Status: SURVEY COMPLETE — literature reviewed, approaches ranked, experiment
plan proposed. No implementation yet. Companion to
`scene_alignment_dtw_contrast.md` (the open problem note); design context in
`scene_alignment_design.md` §4.

## 1. Problem recap and framing

At visually sparse tracks, per-frame DINOv2 global descriptors are ~99%
cosine-similar across the whole lap, so rows of the DTW cost matrix
(`compute_scene_cost_matrix`, `tracksync/scene_align.py:126`) are nearly
flat. Open-ended DTW then latches onto wrong boundary correspondences and
rides the slope limit; the per-row z-score stopgap on `main` fixed boundary
drift (p95 2.176 s → 0.795 s) but mid-lap disagreement remains.

The literature has a name for this: **perceptual aliasing**, the central
failure mode of visual place recognition (VPR) in self-similar environments
(deserts, rail corridors, tunnels, highways). Framing the coarse stage as
"VPR along a route + sequence matching" unlocks a large body of directly
applicable work. Three root causes stack up in our pipeline:

1. **Descriptor collapse.** All frames share a dominant "desert + tarmac +
   sky" component, so descriptors concentrate in a small cone of feature
   space and the similarity matrix is approximately rank-1:
   `sim(i, j) ≈ g(i)·g(j)`. AnyLoc observed exactly this collapse (via PCA
   analysis) for urban-trained VPR models evaluated out-of-domain.
2. **Aggregation washout.** GeM pooling (`DinoV2Embedder._gem_pool`,
   `tracksync/embedding.py:366`) averages patch tokens into one vector,
   discarding the *spatial arrangement* of what little discriminative
   structure exists. AnyLoc's headline result is that this matters a lot in
   unstructured scenes (§2.1).
3. **Per-frame matching.** Each cost cell compares single frames. A frame
   has almost no unique content, but a ±1 s *window* of frames (how the
   scene evolves) is far more distinctive. Sequence-based VPR and shapeDTW
   both exploit this (§2.3, §3.3).

The directions below are ordered as in the problem note: descriptors (§2),
cost-matrix/DTW conditioning (§3), non-appearance signals (§4), fine-stage
reliance (§5). §6 ranks everything; §7 is the proposed experiment plan.

Verification note: findings below were gathered via web research
(2026-07). Claims verified against papers/repos/source code are stated
plainly; items resting on secondary sources are flagged *(unverified)*.

## 2. Direction A — better descriptors for low-texture scenes

### 2.1 VLAD aggregation over patch tokens (AnyLoc recipe) — strongest evidence

AnyLoc (Keetha et al., RA-L 2023, arXiv:2308.00688, BSD-3) is the direct
answer to "is GeM the weak link?" Its findings, on zero-training DINOv2
features across structured and *unstructured* (subterranean, underwater,
aerial, degraded) domains:

- Relative to the DINOv2 CLS token, **GeM gains only ~+2% R@1 in
  unstructured environments, while unsupervised VLAD aggregation gains
  ~+18%** — i.e. our GeM pooling is barely better than the CLS token in
  exactly the environments where we fail, and VLAD is ~16 points ahead
  *(numbers via paper summaries; direction consistent across sources)*.
- AnyLoc beats trained VPR models (MixVPR, CosPlace, NetVLAD) by up to 4×
  R@1 in unstructured domains; urban-trained models' features collapse
  out-of-domain — the same pathology we see.
- Recommended config (verified in their repo): DINOv2 patch features
  (they use ViT-G/14 layer 31, `value` facet), **VLAD with 32 clusters**,
  hard assignment, per-cluster residual L2-normalization.

Why VLAD restores contrast: instead of one pooled vector, the descriptor
stores *residuals from a vocabulary of local visual elements* (kerb, cone,
tarmac crack, horizon ridge), preserving which elements are present and how
they deviate — precisely the information GeM averages away. Critically, the
vocabulary is **unsupervised k-means fit on the target data itself** — we
can fit it per video pair from the masked patch tokens of both laps. No
training, no new weights, no new backbone: this is an aggregation change on
top of the tokens the pipeline already computes. With ViT-B/14 tokens and
32 clusters the descriptor becomes 32×768 = 24,576-D; cosine matching still
applies (cost matrix stays 1200×1200; the dot product just gets wider).

Practical notes: requires caching patch tokens (or re-running aggregation
inside `embed()`), a k-means (~10⁵–10⁶ token vectors — seconds with
minibatch k-means), and invalidating the embedding cache key on aggregation
params. ViT-G is not required; the recipe applies to our ViT-B tokens at
near-zero extra inference cost.

### 2.2 Off-the-shelf VPR model swaps (one-line benchmarks, domain caveat)

All modern trained VPR descriptors are trained on urban street-view data
(GSV-Cities, SF-XL, MSLS); desert racetracks are out-of-distribution, which
is AnyLoc's argument for why they may stay flat at Buttonwillow. Still,
several are one-line `torch.hub` trials behind our `FrameEmbedder` protocol
and worth benchmarking:

| Model | Venue | Notes | License |
|---|---|---|---|
| **MegaLoc** (arXiv:2502.17237) | CVPR-W 2025 | DINOv2-B + SALAD-style OT aggregation, 8448-D; multi-dataset trained; the 2025–26 consensus "just use this" model; `torch.hub.load("gmberton/MegaLoc", ...)` | MIT |
| **BoQ** (arXiv:2405.07364) | CVPR 2024 | DINOv2-B + learnable-query aggregation, 12288-D; MSLS-val R@1 93.8 | MIT |
| **SALAD** (arXiv:2311.15937) | CVPR 2024 | DINOv2-B + Sinkhorn OT assignment with a dustbin that discards uninformative patches | **GPL-3.0** — license caution |
| **SALAD + CliqueMining** (arXiv:2407.02422) | ECCV 2024 | Trains for *geographic distance sensitivity* — fixes "visually similar nearby frames get near-identical scores", our flat-row problem in training-side form; Nordland (self-similar rail) R@1 76.0 → 90.7 | GPL-3.0 |

CliqueMining's diagnosis is the most on-point published statement of our
problem; its checkpoint is arguably the best off-the-shelf single-frame
model for self-similar scenery, but GPL-3.0 and urban training temper it.
All are one DINOv2-B-class forward per frame — same cost class as today,
fine on MPS.

### 2.3 Sequence-aware descriptors (cheap, big literature payoff)

Sequence information reliably rescues single-frame aliasing:

- **SeqSLAM** (Milford & Wyeth, ICRA 2012): with heavily downsampled images
  under extreme appearance change, raw matching put ~0.55% of correct
  matches at top rank; patch normalization + difference-matrix local
  normalization + *sequence* search recovered 74% within the top 10%
  *(numbers via OpenSeqSLAM2.0 analysis, arXiv:1804.02156)*.
- **SeqNet** (RA-L 2021, arXiv:2102.11603), **SeqVLAD** (RA-L 2022,
  arXiv:2207.03868, MIT), **JIST** (RA-L 2024, arXiv:2403.19787, MIT):
  learned sequence descriptors add ~10–30 R@1 points over their own
  single-frame baselines on Nordland/Oxford-type self-similar routes.
- **Delta Descriptors** (Garg et al., RA-L 2020, arXiv:2006.05700):
  **unsupervised** — replace each descriptor with its smoothed temporal
  difference along the route; explicitly motivated by aliasing ("changes
  are unique and consistent across traverses even when raw descriptors are
  nearly identical"). One line of math on top of our existing embeddings.

We have ordered video, so the cheapest variants apply directly: concatenate
or pool descriptors over ±k frames (the shapeDTW construction, §3.3), or
delta descriptors. Both compose with everything else in this survey.

### 2.4 Backbone upgrade: DINOv3

DINOv3 (Meta, Aug 2025, arXiv:2508.10104) targets exactly dense-feature
quality (Gram anchoring fixes DINOv2's dense-feature degradation); the
paper reports +4.3% recall on geometric correspondences vs DINOv2
*(via paper summary)*. No published DINOv2-vs-DINOv3 VPR A/B exists as of
mid-2026, but recent VPR systems are adopting DINOv3-L as the extractor.
ViT-B/16 is a near drop-in for ViT-B/14 on MPS (patch size 14→16 changes
mask/grid geometry). Caveats: gated weights, custom non-OSI license
(attribution + pass-through) — fine for a personal tool. Treat as an
experiment, not a cited claim.

## 3. Direction B — cost-matrix conditioning and DTW variants

These compose with the existing banded, slope-constrained NumPy DP
(`dtw_align`, `tracksync/dtw.py:68`) unless noted. A useful lens: the
current per-row z-score is one point in a family; the literature offers
strictly stronger members of the same family, plus DP-side fixes that
attack the boundary pathology directly.

### 3.1 Feature-side: per-lap-pair centering and whitening (near-free, attacks root cause)

The ~0.99-everywhere similarity means all descriptors share a dominant mean
component. Subtracting the mean descriptor (computed over both videos'
embeddings for the pair), optionally followed by whitening/PCA, removes the
shared "racetrack-ness" direction *before* cosine — the retrieval-community
result that centering + whitening dramatically improves cosine
discrimination (Jégou & Chum, ECCV 2012). Schubert, Neubert & Protzel
(ICRA 2020, arXiv:2001.08960) show unsupervised per-deployment-set
standardization "considerably improves" VPR matching when conditions make
different places look alike. Matrix-side equivalent: subtract the top
singular component of the similarity matrix (rank-1 background removal) —
but feature-side centering is cheaper and better understood. Cost: a few
lines in `compute_scene_cost_matrix`; O(N·D).

Caution: centering makes the self-alignment diagonal no longer exactly
zero-cost *before* row normalization; the existing min-shift restores the
per-row zero. The self-alignment identity test (design §8.1) is the
regression guard.

### 3.2 Matrix-side: stronger normalization than per-row z-score

- **SeqSLAM local contrast enhancement** (the canonical fix for exactly
  this): z-score each cost cell within a *local temporal window* along the
  reference axis (window R≈10 frames), not the whole row — "is this B-frame
  a better match for this A-moment than its temporal neighbors?"
  Mechanism verified in pySeqSLAM source. In the OpenSeqSLAM2.0 ablation,
  this difference-matrix normalization contributed a ~4× improvement on its
  own (§2.3 numbers). Two axes to try: normalize along B (SeqSLAM's
  choice), along A (our current row direction), and both.
- **Dual-softmax** (LoFTR, CVPR 2021, arXiv:2104.00680):
  `P = softmax_row(S/τ) ⊙ softmax_col(S/τ)`; use `C' = −log P`. A soft
  mutual-nearest-neighbor score — a cell scores high only if it wins in
  *both* directions. Directly fixes the asymmetry our per-row scheme leaves:
  a B-frame that is "everyone's 99% match" still attracts paths today,
  because columns are never normalized. −log of a softmax is non-negative
  and additive — the natural DTW cost. Temperature τ is the sharpening
  knob (equivalent to power-sharpening). ~5 lines of NumPy.
- **Sinkhorn / optimal transport** (SuperGlue, CVPR 2020; VAVA, CVPR 2022,
  arXiv:2111.09301): iterated row+column normalization to a doubly
  stochastic plan, with dustbins absorbing unmatchable frames. Composes as
  preprocessing (`−log` plan), but the 1-to-1 mass constraint mildly fights
  our deliberate 2:1 slope allowance, and open ends need dustbin care.
  Rank below dual-softmax here.
- **Rank/quantile normalization**: maximal contrast equalization, but
  destroys magnitude information — genuinely ambiguous rows get artificial
  contrast (the reason the current fix deliberately preserved relative
  magnitudes). Keep as an ablation arm only.

### 3.3 Temporal context: shapeDTW windows and diagonal smoothing

- **shapeDTW** (Zhao & Itti, Pattern Recognition 2018, arXiv:1606.01601):
  replace each element's descriptor with a descriptor of its temporal
  *neighborhood* (window length 5–30), then run vanilla DTW. Beat DTW on
  64/84 UCR datasets. For us: stack/average DINOv2 descriptors over
  ±0.5–1.5 s before the cost matrix — frames match only if their contexts
  agree. Feature-side; DP untouched.
- **Diagonal path enhancement** (Müller & Kurth, ICASSP 2006; SM Toolbox,
  AES 2014 — standard in music structure analysis): smooth the similarity
  matrix along a small bank of diagonal slopes (use {1/2, 2/3, 1, 3/2, 2}
  to match `slope_max=2.0`), take the max across slopes. Coherent temporal
  runs reinforce; isolated aliased spikes wash out. The matrix-side twin of
  shapeDTW; O(L·S·N²) — trivial at 1200².

### 3.4 DP-side: slope prior, bounded slack, and richer recursions

- **Transition penalty toward slope 1** (cf. weighted DTW, Jeong et al.,
  Pattern Recognition 2011): add a small constant λ to non-diagonal steps
  in the recursion of `dtw_align`. In flat regions the path currently
  wanders to the band edge for free; a tie-breaking prior pins it to
  slope ~1 unless the data pays for deviation. **Probably the single
  highest value-per-line change for the mid-lap residual** — a ~2-line
  edit at `tracksync/dtw.py:144-168`. A soft variant of the same idea:
  weight cost by |i−j−δ̂| around a global offset estimate δ̂ obtained from
  the best-sum diagonal of the cost matrix (aggregated over ~1200 cells, so
  far better conditioned than any single row); this "anchor first, then
  locally warp" pattern also stabilizes the open ends.
- **Bounded, penalized slack (ψ-DTW)** (Silva, Batista & Keogh, KDD MiLeTS
  2016): endpoint mishandling dominates DTW error (in their study a 6%
  prefix caused 70.5% of error). Instead of fully open ends, allow at most
  ψ free boundary cells and charge each skipped frame a calibrated cost
  (e.g. a percentile of the normalized cost distribution, the Drop-DTW
  recipe). Note the existing average-per-cell endpoint selection
  (`tracksync/dtw.py:182-195`) is the sanctioned length-bias fix, but with
  near-flat costs the average is itself nearly constant across candidate
  endpoints — selection is decided by noise. A slack penalty re-poses the
  choice.
- **Drop-DTW** (Dvornik et al., NeurIPS 2021, arXiv:2108.11996): a second
  DP layer allowing per-element drops at a specified cost — principled
  skipping of junk segments (traffic, dust, glare) and an alternative
  formulation of open ends. ~80-line NumPy port from their `exact_dp.py`;
  still banded/monotonic. Drops need interpolation to keep a continuous
  time mapping.
- **soft-DTW posterior as a confidence signal** (Cuturi & Blondel, ICML
  2017, arXiv:1703.01541): the forward-backward gradient of soft-DTW gives
  a per-cell alignment posterior — a principled *measure* of mid-lap
  underdetermination, better than the current best-vs-second-best margin,
  and usable to gate fine-stage effort and to report uncertainty bands.
  Doesn't add contrast by itself; a diagnostic/robustness upgrade.

### 3.5 Video-alignment literature cross-check

TCC (CVPR 2019), LAV (CVPR 2021), VAVA (CVPR 2022), and GTA/Smooth-DTW
(CVPR 2021, arXiv:2105.05217) all align videos with learned features, but
their *alignment layers* transfer to frozen features: GTA explicitly
softmax-normalizes distances into contrastive probabilities before DTW
(their stated motivation matches our problem: raw distances carry no
calibrated meaning, only relative comparisons do); VAVA blends a Gaussian
diagonal prior with a content prior inside Sinkhorn (the OT version of
§3.4's slope prior); LAV documents that alignment losses degenerate on
weakly discriminative features without an explicit contrast term. TCC's
cycle-back error (align A→B→A, measure return distance) is computable with
frozen features and makes a cheap per-frame reliability diagnostic.

## 4. Direction C — complementary non-appearance signals

The motion signature of a lap (yaw rate through corners, speed profile
through braking zones) is highly structured, nearly appearance-invariant,
and cheap to extract — the natural complement to appearance exactly where
appearance is flat.

### 4.1 Yaw-rate profile from the existing toolbox

The standard cheap method: for a forward-facing camera, flow of *distant*
points (upper image, near horizon) is rotation-dominated
(translation-induced flow falls off as 1/depth), so
`yaw_rate ≈ (median horizontal KLT flow of far-field points / fx) × fps`.
Ingredients already exist: OpenCV KLT, the static car-body mask
(`tracksync/masking.py`), and `intrinsics_from_fov`
(`tracksync/fine_align.py:85`). More principled variants when needed:
infinite-homography decomposition (`K⁻¹HK` on far-field inliers), or the
essential-matrix chain already implemented in `score_pair`
(`tracksync/fine_align.py:113`) — Scaramuzza's 1-point RANSAC (IJCV 2011)
shows vehicle frame-to-frame motion reduces to a single angle under
nonholonomic constraints.

Signal quality: race-car yaw rates reach ~50°/s at corner entry — large,
easily measurable flow; near zero on straights. So the yaw cost matrix has
strong contrast at corners and a flat corridor on straights; DTW
monotonicity + the appearance/speed channels carry the straights.
Robustness: a constant camera-mount yaw offset shifts image content but
leaves the yaw-*rate* profile shape intact (mount pitch/roll mixes in
second-order terms); vibration is zero-mean (average into the 10 Hz bins);
rolling shutter contributes a smooth gain error that per-lap z-normalization
absorbs. Runtime: seconds for 1200 frames, CPU.

Evidence this signal is discriminative for *position along a route*:
- Elastic Pathing (UbiComp 2014, arXiv:1401.0052): the speed profile
  *alone*, elastically warped, recovers driven routes.
- DP-matching of yaw-rate time series for vehicle localization (SICE JCMSI
  2021); heading-sequence map matching (arXiv:2005.13704); curvature
  arc-length map matching (arXiv:2410.12208); OpenStreetSLAM (ICRA 2013).
- Race-engineering practice: MoTeC i2 aligns laps in distance domain with
  "distance stretching" (a crude uniform DTW); Racelogic VBOX moved to
  GPS-position-based alignment because distance-based drifts >10 m (~1 s)
  by lap end — practitioner confirmation that *track position* (which
  integrated yaw approximates) is the right alignment domain.

### 4.2 Relative-speed profile from ground-plane flow

For ground-plane points at fixed image coordinates, flow magnitude ∝ v/h
(camera height h constant within a lap), so robust-averaged flow magnitude
in a lower-image ROI (masked by the car-body mask) is proportional to speed
up to one per-lap gain — removed by per-lap z-normalization, exactly what
DTW on shape needs. High signal at braking and corner exit, usefully
bracketing the corners that yaw covers. Note the essential matrix cannot
supply this (unit-norm translation per pair); it must come from flow
magnitude. TTC/divergence proxies exist but conflate speed with scene
depth — rank lower. Learned VO (TartanVO, CoRL 2020 — up-to-scale by
construction) is a heavier fallback; DPVO/DROID-SLAM are CUDA-only and
unusable on Apple Silicon (verified); pySLAM is GPLv3. The OpenCV-only
path is the right one.

### 4.3 Fusing heterogeneous cost matrices

Standard recipe (multi-dimensional DTW practice): normalize each cost
matrix (z-score or rank), then weighted sum — the DP is untouched.
Precedents: Multi-Process Fusion (Hausler et al., RA-L 2019,
arXiv:1903.03305) fuses difference matrices from multiple methods inside a
sequence-matching framework on the rationale "at least one method works in
any environment"; Caspi & Irani (IJCV 2002) legitimized aligning sequences
on "coherent temporal behavior" instead of coherent appearance.

Adaptive weighting: no canonical per-row-gated DTW paper exists (flagged as
engineering synthesis) — the transferable scheme is entropy gating: softmax
each row into a distribution, weight each modality's row by its contrast
(low entropy → high weight). The modalities are naturally complementary
around a lap: appearance where scenery changes, yaw at corners, speed at
braking zones. A fixed-weight sum is the right first experiment; entropy
gating second.

Audio (engine note) is *not* an alignment signal across different laps/cars
(different acoustic events, gearing, shift points); an RPM-vs-time trace
via harmonic analysis is an opportunistic same-car-only third modality at
best.

### 4.4 A note on the domain hypothesis

The motion channel also directly tests the Buttonwillow hypothesis in the
problem note: if boundary/mid-lap error correlates with appearance-row
entropy across tracks, and the yaw-fused run removes the correlation, the
"visual sparsity flattens rows" causal story is confirmed.

## 5. Direction D — leaning harder on the fine stage

The research above mostly *reduces the need* for this. Two observations:

- The fine stage can only refine what the coarse stage puts within its
  ±0.5 s window (`generate_scene_sync_points`,
  `tracksync/scene_align.py:407`). With p95 coarse error at 0.795 s, a
  meaningful fraction of candidates start with the true match *outside*
  the refinement window — fine-stage expansion means widening windows
  (more LightGlue pairs, linear cost growth) while still trusting a wrong
  coarse prior. Fixing coarse contrast is upstream of, and cheaper than,
  fine-stage expansion.
- The cheap fine-stage lever that *is* worth taking: use a proper
  confidence signal (soft-DTW posterior §3.4, or TCC cycle-back error
  §3.5) to widen the refinement window only on low-confidence spans,
  rather than expanding uniformly.

## 6. Ranked recommendations

Ordered by expected value per unit effort for this codebase. The top three
are independent and compose; each is guarded by the self-alignment identity
test (design §8.1).

| # | Change | Where | Effort | Rationale |
|---|---|---|---|---|
| 1 | Per-lap-pair descriptor centering (+ optional whitening) | `compute_scene_cost_matrix` | hours | Attacks the root cause of 0.99-everywhere cosines; retrieval-standard (Jégou & Chum 2012; Schubert 2020); free at runtime |
| 2 | Slope-1 transition penalty λ + bounded/penalized end slack | `dtw_align` recursion + endpoint selection | hours | Directly targets both boundary latching and mid-lap wander; ψ-DTW evidence that endpoints dominate DTW error; ~2-line core change |
| 3 | Two-way contrast normalization: dual-softmax −log P (τ knob), and/or SeqSLAM local-window z-score | `compute_scene_cost_matrix` | hours | Strict upgrade of the current one-way row z-score; SeqSLAM ablation shows difference-matrix local normalization alone is worth ~4×; fixes the unnormalized-column asymmetry |
| 4 | Temporal context: windowed (shapeDTW-style) descriptors or multi-slope diagonal smoothing of C; delta descriptors as a variant | embedding post-proc or cost post-proc | ~1 day | Sequence info demonstrably rescues single-frame aliasing (SeqSLAM/SeqNet/SeqVLAD); we have ordered video for free |
| 5 | VLAD aggregation over existing DINOv2 patch tokens, per-pair unsupervised vocabulary (AnyLoc recipe) | `DinoV2Embedder` + cache key | days | Strongest published evidence for exactly this failure mode (GeM ≈ CLS in unstructured scenes; VLAD +16–18 R@1); no new model |
| 6 | Yaw-rate cost channel (far-field KLT) + speed channel (ground-ROI flow), fixed-weight fusion | new small module + `coarse_align` | days | Appearance-invariant signal with strong route-position evidence; complements appearance exactly where it is flat; OpenCV-only, seconds of runtime |
| 7 | Off-the-shelf embedder benchmarks: MegaLoc, BoQ (MIT, torch.hub one-liners); SALAD+CliqueMining (GPL caution) | new `FrameEmbedder` impls | ~1 day each | Cheap A/Bs; urban-trained so may still collapse at Buttonwillow — measure, don't assume |
| 8 | DINOv3-B backbone A/B | new embedder | ~1 day | Plausible dense-feature upgrade; gated custom license; no published VPR A/B yet |
| 9 | Drop-DTW layer; soft-DTW posterior confidence | replace/extend DP | days | Bigger-ticket DP work; posterior confidence also improves fine-stage gating and debug output |

Items 1–4 require no model changes and can be ablated offline against
*cached* embeddings (`~/.cache/tracksync/embeddings`) in minutes per
configuration — no re-embedding, no GPU. That makes a broad sweep cheap.

## 7. Proposed experiment plan

### Phase 0 — instrumentation and data (prerequisite)

1. Add contrast diagnostics to the validation script
   (`scripts/validate_scene_alignment.py`) and debug panels: per-row
   margin distribution, per-row softmax entropy, similarity-matrix
   effective rank / top-1 singular-value share, boundary segment speeds.
   These are the metrics every later ablation is judged on, and they
   quantify "flatness" per track.
2. Collect at least one Buttonwillow pair and one feature-rich-track pair
   with Catalyst + OCR ground truth. This validates the domain hypothesis
   (flatness metric should separate the tracks) and prevents overfitting
   conditioning choices to the single `eddie_pb`/`kurt_optimal` pair.

### Phase 1 — cheap conditioning sweep (items 1–4), offline on cached embeddings

Grid over: {current row z-score} × {± centering} × {± whitening} ×
{dual-softmax τ ∈ {0.05, 0.1, 0.2}} × {SeqSLAM window R ∈ {5, 10, 20},
axis A/B/both} × {window/context length ∈ {0, ±5, ±10} frames} ×
{λ ∈ {0, small, medium}} × {open-end: current vs ψ-penalized}.
Metrics: |Δt| vs Catalyst and vs OCR (max/mean/p95), boundary segment
speed, self-alignment identity error (must stay ≤ 1 sample), plus Phase-0
contrast metrics. Promote the Pareto-best combination.

Exit criterion worth aiming for: p95 |Δt| vs Catalyst ≤ 0.3 s on the
existing pair without regressing the feature-rich pair, boundary speeds
in [0.9, 1.1].

### Phase 2 — aggregation and embedder A/Bs (items 5, 7, 8)

VLAD-over-patch-tokens vs GeM under the Phase-1 winning conditioning;
then MegaLoc/BoQ (and optionally DINOv3-B) as embedder swaps. Requires
re-embedding (cache-key changes); each A/B is one validation-script run
per pair.

### Phase 3 — motion channel and fusion (item 6)

Implement yaw + speed traces (10 Hz, masked KLT); validate each trace
against the Catalyst pair's known geometry (corners visible as yaw pulses);
then fixed-weight fusion sweep (appearance weight ∈ {1.0, 0.7, 0.5}),
then entropy-gated per-row weighting. Success criterion: mid-lap p95
improves on the sparse track *and* the boundary pathology stays fixed with
appearance weight reduced — evidence the pipeline no longer depends on a
low-information signal.

### Phase 4 — decision point

If Phases 1–3 reach frame-parity-class accuracy (p95 ≤ ~0.2 s), revisit
design §11.3's relaxed acceptance and consider re-tightening toward the
original 1-frame parity gate. If not, invest in item 9 (posterior
confidence + adaptive fine-stage windows) rather than uniformly expanding
the LightGlue budget (§5).

## 8. References

VPR / descriptors
- AnyLoc: Keetha et al., RA-L 2023, arXiv:2308.00688 (BSD-3)
- MegaLoc: Berton & Masone, CVPR-W 2025, arXiv:2502.17237 (MIT)
- BoQ: Ali-bey et al., CVPR 2024, arXiv:2405.07364 (MIT)
- SALAD: Izquierdo & Civera, CVPR 2024, arXiv:2311.15937 (GPL-3.0)
- CliqueMining: Izquierdo & Civera, ECCV 2024, arXiv:2407.02422
- CricaVPR: Lu et al., CVPR 2024, arXiv:2402.19231; SelaVPR: ICLR 2024,
  arXiv:2402.14505; SuperVLAD: NeurIPS 2024; EffoVPR: ICLR 2025,
  arXiv:2405.18065; MixVPR: WACV 2023, arXiv:2303.02190; CosPlace: CVPR
  2022, arXiv:2204.02287; EigenPlaces: ICCV 2023, arXiv:2308.10832
- DINOv3: Siméoni et al., arXiv:2508.10104 (gated, custom license)
- Delta Descriptors: Garg et al., RA-L 2020, arXiv:2006.05700
- SeqSLAM: Milford & Wyeth, ICRA 2012; OpenSeqSLAM2.0: Talbot et al.,
  IROS 2018, arXiv:1804.02156; SeqNet: arXiv:2102.11603; SeqVLAD:
  arXiv:2207.03868; JIST: arXiv:2403.19787
- Descriptor standardization: Schubert, Neubert & Protzel, ICRA 2020,
  arXiv:2001.08960; Jégou & Chum, ECCV 2012 (PCA/whitening)
- FAB-MAP (aliasing model): Cummins & Newman, IJRR 2008

Cost conditioning / DTW
- LoFTR (dual-softmax): Sun et al., CVPR 2021, arXiv:2104.00680
- SuperGlue (Sinkhorn/dustbin): Sarlin et al., CVPR 2020, arXiv:1911.11763
- shapeDTW: Zhao & Itti, Pattern Recognition 2018, arXiv:1606.01601
- Diagonal smoothing: Müller & Kurth, ICASSP 2006; SM Toolbox, AES 2014
- Weighted DTW: Jeong et al., Pattern Recognition 2011
- ψ-DTW endpoints: Silva, Batista & Keogh, KDD MiLeTS 2016
- Open-end DTW: Tormene et al., Artif. Intell. Med. 2009; Giorgino, JSS 2009
- Drop-DTW: Dvornik et al., NeurIPS 2021, arXiv:2108.11996
- soft-DTW: Cuturi & Blondel, ICML 2017, arXiv:1703.01541
- Derivative DTW: Keogh & Pazzani, SDM 2001
- TCC: Dwibedi et al., CVPR 2019, arXiv:1904.07846; LAV: Haresh et al.,
  CVPR 2021, arXiv:2103.17260; VAVA: Liu et al., CVPR 2022,
  arXiv:2111.09301; GTA: Hadji et al., CVPR 2021, arXiv:2105.05217

Motion signals / fusion
- Elastic Pathing: Gao et al., UbiComp 2014, arXiv:1401.0052
- Yaw-rate DP localization: SICE JCMSI 2021 (Tandfonline
  10.1080/18824889.2021.1906018); heading-sequence matching:
  arXiv:2005.13704; curvature map matching: arXiv:2410.12208;
  OpenStreetSLAM: Floros et al., ICRA 2013
- 1-point RANSAC vehicle VO: Scaramuzza, IJCV 2011
- Non-overlapping sequence alignment: Caspi & Irani, IJCV 2002
- Multi-Process Fusion: Hausler et al., RA-L 2019, arXiv:1903.03305
- TartanVO: Wang et al., CoRL 2020, arXiv:2011.00359
- Rolling-shutter epipolar geometry: Dai et al., CVPR 2016
- Racelogic VBOX Circuit Tools alignment docs; MoTeC i2 distance
  stretching (practitioner sources)
