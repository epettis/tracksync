# Design: Scene-Based Video Alignment

Status: APPROVED — open questions resolved (see §11)
Replaces: map-dot (red circle) cross-correlation as the alignment signal

## 1. Problem statement

tracksync currently aligns two onboard videos by tracking the red car marker
on the Garmin Catalyst's white track-map overlay (`find_red_circle()`,
`tracksync/frame_analysis.py:625`) and cross-correlating the (x, y) dot
positions (`tracksync/cross_correlation.py:19-233`). This only works for
Catalyst exports with the overlay enabled and at the expected screen layout.

This design replaces the map-dot signal with the **actual camera feed**:
aligning videos by matching physical track features visible through the
windshield — curbs, advertising signs, corner stations, brake markers,
pavement cracks, horizon geometry.

### Key difficulty

Different cars mount cameras at different heights, lateral positions, and
orientations (full 6-DoF pose differences, plus different lenses/intrinsics).
The same curb appears at different pixel locations, scales, and perspectives
in the two videos. Naive pixel or template comparison fails; the method must
be viewpoint-tolerant.

### Simplifying assumptions (per product decision)

Drivers overwhelmingly compare **best laps**. Therefore:

- **Input is a single lap per video**, ± a few seconds of slack before/after
  the start/finish crossing. Clip length ≈ 1–3 minutes.
- **No lap aliasing.** With one lap per clip, a monotonic alignment over the
  whole clip cannot confuse Turn 3 on lap 2 with Turn 3 on lap 5. (The old
  pipeline needed OCR lap anchors and a ±20 s window for this;
  `tracksync/cross_correlation.py:132-133`.)
- **Low traffic occlusion.** A best lap is usually a clear lap; we do not
  need robustness to a car filling the frame for many consecutive seconds.
- Both laps are on the same track, same configuration, driven in the same
  direction. Sessions may differ in time of day, weather, and season.

## 2. Goals and non-goals

### Goals

1. Camera-agnostic sync: works for any forward-facing onboard camera
   (GoPro, Catalyst raw feed, AiM SmartyCam, phone mounts), no overlay
   required.
2. Robust to camera pose differences between cars (height, lateral offset,
   pitch/yaw/roll) and moderate appearance change (lighting, weather).
3. Accuracy: ≤ 1 frame error at sync points on the coarse pass; sub-frame
   accuracy after the fine pass.
4. Preserve downstream pipeline unchanged: sync points → CSV
   (`tracksync/csv_writer.py`) → comparison video
   (`tracksync/video_processor.py`), and the `sync` / `generate` / `debug`
   CLI structure (`tracksync/cli.py`).
5. Runs locally on Apple Silicon (MPS) or CPU in reasonable time. Budget is
   liberal: the current map-dot pipeline takes 10+ minutes on the reference
   machine, so anything at or under that is acceptable. This allows
   defaulting to quality over speed (larger backbone, denser sampling,
   dense matcher) where it helps accuracy.

### Non-goals

- Full-session, multi-lap alignment (future work; requires lap-aliasing
  disambiguation — see §10).
- 3D track reconstruction, racing-line comparison, distance-domain analysis,
  ghost-car overlays (Option C in the methodology review; future work).
- Real-time operation.
- Removing the existing Catalyst OCR path entirely: lap-time OCR remains
  useful as optional metadata where available, but is no longer required for
  alignment.

## 3. Approach overview

Two-stage, coarse-to-fine:

```
Video A ─┬─> frame sampling ──> car-body masking ──> per-frame embeddings ─┐
Video B ─┘                                                                 │
                                                                           v
                                              cosine cost matrix (N_A × N_B)
                                                                           │
                                                                           v
                                          banded monotonic DTW  (COARSE)
                                                                           │
                                                    candidate sync points  │
                                                                           v
                              local feature matching + relative pose (FINE)
                                                                           │
                                                                           v
                                     sub-frame sync points ──> existing CSV
                                                       ──> comparison video
```

- **Coarse (Stage A):** per-frame global descriptors from a pretrained
  vision foundation model, matched by dynamic time warping. Answers "which
  ~200 ms window of B corresponds to each moment of A."
- **Fine (Stage B):** learned local-feature matching between candidate frame
  pairs; epipolar geometry yields inlier verification and sub-frame timing.
  Answers "at exactly which instant did car B pass the point car A occupies
  in this frame," while explicitly factoring out camera mounting pose.

## 4. Stage A — coarse alignment via global descriptors + DTW

### 4.1 Frame sampling

- Sample both videos at a fixed rate, default **10 Hz** (configurable
  `--sample-hz`). At 100 km/h this is one sample per ~2.8 m of track —
  dense enough for DTW; the fine stage recovers precision. The liberal
  runtime budget (§7) permits this density; drop to 5 Hz only if profiling
  shows a problem.
- A 2-minute clip → ~1200 samples/video. Cost matrix 1200×1200 is trivial.
- Decode with existing OpenCV ingestion (`tracksync/feature_extraction.py`);
  resize frames to the embedding model's input resolution at decode time.

### 4.2 Car-body masking

The hood, dash, roll cage, wheel, and any picture-in-picture overlay are
static per-video and differ between cars; they are pure noise for matching.

- Compute per-pixel temporal variance over the sampled frames (grayscale,
  downsampled). Pixels with variance below a threshold are "static."
- Morphologically close/dilate to form a static-region mask; exclude those
  patch tokens from descriptor aggregation (or matte to gray before
  embedding — implementation detail, benchmark both).
- This is fully automatic, per-video, no calibration or user input.
- Reuse pattern: same NumPy/OpenCV idioms as the existing red-mask
  morphology (`tracksync/frame_analysis.py:674-676`).

### 4.3 Per-frame embedding

- Default backbone: **DINOv2 ViT-B/14** patch features aggregated to a
  single global descriptor, following the AnyLoc recipe (unsupervised
  aggregation, e.g., GeM pooling over foreground patches). No training or
  fine-tuning required. ViT-B chosen over ViT-S because the runtime budget
  is liberal (§7); `--embedder` can select ViT-S for speed.
- Alternative backbones behind the same interface (pluggable
  `FrameEmbedder` protocol): MixVPR / CosPlace weights if we want a
  VPR-specialized model; CLIP ViT as a fallback.
- Output: `emb_array: float32 [N, D]`, L2-normalized, stored on
  `VideoFeatures` alongside the existing arrays
  (`tracksync/frame_data.py:31-46`).
- Runtime estimate: 1200 frames through ViT-B on MPS ≈ 1–2 min/video.

### 4.4 Cost matrix and DTW

- Cost matrix `C = 1 − emb_A @ emb_B.T` (cosine distance), vectorized
  exactly like the current Euclidean distance matrix
  (`tracksync/cross_correlation.py:74-76`).
- **Monotonic DTW** over `C` with:
  - **Open ends** (subsequence DTW): the ± few seconds of slack around
    start/finish means the path should be free to start/end anywhere within
    the first/last ~10 s band of either clip, rather than pinned to the
    corners.
  - **Sakoe–Chiba band**: since both clips are one lap of comparable
    duration, constrain |t_A/T_A − t_B/T_B| to a normalized band
    (default ±10% of lap time, configurable). Replaces the current ±20 s
    window logic. Keeps the path sane through any brief low-texture stretch.
  - **Slope constraint** (e.g., steps limited to 1:2 / 2:1): two clean laps
    of the same track can't differ in local speed by more than ~2×.
- Path smoothing: fit a monotone piecewise-linear or PCHIP function
  t_B = f(t_A) to the DTW path to suppress stair-stepping before Stage B.
- The DTW path endpoints define `trim_start_*` / `trim_end_*`, replacing
  start-finish crossing detection from OCR (`SyncResult` fields,
  `tracksync/models.py:54-62`). When Catalyst OCR crossings are available
  they may be passed as optional anchor constraints, but are not required.

### 4.5 Confidence signal

For each frame of A, record the margin between the best and
second-best-off-path costs in its band. Low-margin spans (long straights
with featureless walls, fog) are flagged; sync points are preferentially
placed at high-margin frames. Exposed in `debug` visualization.

## 5. Stage B — fine alignment via local features + relative pose

Stage A gives a frame-level mapping. Stage B refines each intended sync
point to sub-frame precision and validates it geometrically.

### 5.1 Sync point candidates

- Generate candidate sync times in A on the existing cadence
  (`--max-sync-interval`, default 3 s; `tracksync/cross_correlation.py:491`),
  snapped to high-confidence frames per §4.5.
- For each candidate `t_A`, take the Stage-A estimate `t_B = f(t_A)` and a
  refinement window of ±0.5 s in B at **native fps** (decode ~30 frames).

### 5.2 Local feature matching

- Default matcher: **SuperPoint + LightGlue** (LightGlue is Apache-2.0;
  note SuperPoint weights carry a restrictive research license — see Risks
  §9.4 for the ALIKED/DISK alternative). Optional heavy mode: dense matcher
  (RoMa/LoFTR) via the same pluggable interface for hard pairs
  (rain vs. sun, large pose differences).
- Match frame A(t_A) against each frame in the B window, using the §4.2
  static masks to drop keypoints on car bodywork.
- Estimate the essential matrix per pair with RANSAC (OpenCV
  `findEssentialMat`; assume a generic pinhole intrinsic per video —
  a rough FOV guess is sufficient for inlier counting; expose
  `--fov-deg` override).
- **Verification:** inlier count is the match score. A candidate whose best
  pair in the window has < `min_inliers` (default ~30) is rejected and the
  sync point falls back to the smoothed Stage-A estimate, flagged
  low-confidence.

### 5.3 Sub-frame timing via relative pose

For the same physical track location, the *relative pose* between camera A
at `t_A` and camera B at time `t` decomposes into (a) the fixed mounting
difference between the two cars and (b) car B's longitudinal progress along
track. As `t` sweeps through the window:

- Decompose E → (R, t_unit) for each B frame; project the translation onto
  the track-forward direction (dominant motion axis of B, obtained from
  B's own frame-to-frame flow/epipole).
- The longitudinal component changes sign at the instant car B passes car
  A's position; the mounting offset contributes only a constant lateral/
  vertical bias, which is exactly what this formulation cancels.
- Fit the signed longitudinal component vs. `t` near the zero crossing
  (it is locally near-linear at speed) and interpolate → **sub-frame
  `t_B`**. Monocular scale ambiguity is irrelevant because we only need
  the sign and zero crossing, not metric distance.
- Sanity clamp: refined `t_B` must lie within the Stage-A band and preserve
  monotonicity across consecutive sync points (reuse forward-progress
  invariant from `tracksync/cross_correlation.py:189-194`).

### 5.4 Output

Refined sync points feed the existing `generate_sync_points()`-equivalent
assembly → `SyncPoint` list with per-segment speed ratios
(`tracksync/models.py:45-50`) → CSV → video generation. **No changes** to
`csv_writer.py`, `csv_reader.py`, `video_processor.py`,
`segment_validator.py`.

## 6. Integration with existing code

| Component | Disposition |
| --- | --- |
| `frame_analysis.py` red-circle + OCR | Retained but demoted to optional Catalyst metadata path; no longer the alignment signal |
| `feature_extraction.py` | Extended: sampling at `--sample-hz`, static masking, embedding extraction; circle interpolation path bypassed in scene mode |
| `frame_data.py` `VideoFeatures` | Gains `emb_array`, `static_mask`, confidence margins |
| `cross_correlation.py` | Distance matrix → cosine on embeddings; window/forward-progress logic → banded open-end DTW; crossing anchors become optional |
| New `tracksync/embedding.py` | `FrameEmbedder` protocol + DINOv2 implementation |
| New `tracksync/dtw.py` | Banded, slope-constrained, open-end DTW + path smoothing (NumPy; no new heavy deps) |
| New `tracksync/fine_align.py` | LightGlue matching, essential-matrix scoring, zero-crossing refinement |
| `turn_analysis.py` | Trajectory-based apex detection loses its (x, y) input in scene mode; scene mode emits generic `sync_N` labels (decision §11.2). A yaw-rate proxy from epipolar rotation may restore named apex labels later (future work §10) |
| `visualization.py` / `debug` CLI | New panels: cost matrix + DTW path heatmap; side-by-side matched frames with drawn correspondences; per-frame confidence trace |
| `cli.py` | `sync --mode scene` vs. `--mode catalyst` (legacy). Catalyst remains the default until the regression harness (§8.3) demonstrates parity, then scene becomes default (decision §11.3). New flags: `--sample-hz`, `--band-pct`, `--min-inliers`, `--fov-deg`, `--matcher`, `--embedder` |
| Protobuf diagnostics (`diagnose`) | Extend schema with embeddings/path for offline debugging |

New dependencies: `torch` (MPS/CPU), `dinov2` weights via `torch.hub` or
`timm`, `lightglue` (pip). All model weights cached locally after first run.

## 7. Accuracy & performance budget (2-minute clips, M-series Mac)

Budget context: the current map-dot pipeline takes 10+ minutes on the
reference machine; that is the ceiling, not the target.

| Step | Est. cost |
| --- | --- |
| Decode + mask, 10 Hz, both videos | ~20 s |
| Embeddings (2 × 1200 frames, ViT-B, MPS) | ~2–4 min |
| Cost matrix + DTW + smoothing | < 1 s |
| Fine stage: ~40 sync points × ~30 native frames decoded + ~30 LightGlue pairs | ~30–60 s |
| **Total** | **~3–5 min/pair** (well under the 10+ min ceiling) |

Batch mode (`sync --all`) retains O(n) embedding extraction, O(n²) DTW —
same shape as today (`tracksync/cli.py:395-504`); embeddings are cached
per video on disk (hash of path + mtime + params).

Accuracy targets: coarse ≤ ±1 sampled frame (±200 ms) everywhere on-path;
fine ≤ ±0.25 native frame (~8 ms) at verified sync points. Validation
method in §8.

## 8. Validation plan

1. **Self-alignment test:** align a lap against itself → DTW path must be
   the identity; fine stage zero crossings at Δt = 0. Catches sign errors.
2. **Synthetic viewpoint perturbation:** align a lap against a cropped/
   scaled/rotated (±5°) copy of itself with a simulated hood mask → known
   ground truth of identity timing under viewpoint change.
3. **Catalyst ground truth:** for Catalyst clip pairs, run legacy map-dot
   sync and scene sync on the same inputs; compare sync-point timings.
   The existing OCR lap-time readout gives an independent per-frame clock
   for quantifying error. This is our primary regression harness.
4. **Cross-camera acceptance test:** Catalyst vs. GoPro footage of the same
   session (needs collection); verify visually with the `debug` overlay and
   the generated side-by-side video (start/finish and 2–3 apexes frame-
   stepped by eye).
5. Unit tests: DTW monotonicity/band/open-end properties on synthetic cost
   matrices; masking on synthetic static regions; zero-crossing
   interpolation on synthetic pose sweeps. Follow existing pytest layout.

## 9. Risks and mitigations

1. **Low-texture spans** (long straights, featureless runoff): DTW band +
   slope constraints carry the path through; confidence gating (§4.5)
   avoids placing sync points there. Between sync points, playback speed is
   piecewise-constant anyway (existing design), so mid-straight precision
   matters little.
2. **Severe appearance gap** (wet vs. dry, dawn vs. dusk): DINOv2-class
   descriptors are the SOTA answer, but degradation is possible. Mitigation:
   pluggable dense matcher (RoMa) in fine stage; document limits; the
   Catalyst legacy mode remains available.
3. **Extreme camera pose difference** (e.g., roof cam vs. dash cam):
   embeddings degrade gracefully; the epipolar formulation in Stage B is
   pose-difference-tolerant by construction. Acceptance test §8.4 bounds
   what we claim to support.
4. **SuperPoint license** is research-only. Mitigation: default to
   **ALIKED + LightGlue** (both permissive) — benchmark shows comparable or
   better quality; keep SuperPoint as an opt-in.
5. **Dependency weight** (Torch ≈ 2 GB): accepted — this is a personal
   project and no Torch-free scene mode is needed (decision §11.4). Torch
   becomes an unconditional dependency of scene mode; document install.
6. **Rolling shutter / vibration** blurs keypoints at speed: LightGlue-class
   matchers tolerate this in practice; fine-stage RANSAC rejects bad
   geometry; falls back to coarse estimate when inliers are insufficient.
7. **Different lap counts in clip** (user trims sloppily, clip contains
   1.5 laps): out of scope per assumptions; detect via DTW path slope
   anomalies and warn the user to trim.

## 10. Future work (explicitly deferred)

- Multi-lap sessions: lap segmentation from embedding self-similarity
  (the cost matrix of a video against itself shows lap periodicity), then
  per-lap application of this design.
- Distance-domain alignment / racing-line & ghost overlays via SfM
  (Option C from methodology review).
- Named semantic sync points ("turn_3_apex") via landmark detection or a
  yaw-rate proxy from epipolar rotation (restores the apex labels dropped
  from scene-mode v1 per §11.2).
- Fine-tuning a small embedder on track footage if pretrained descriptors
  prove insufficient in some conditions.

## 11. Resolved decisions

1. **Runtime budget: liberal.** The current pipeline takes 10+ minutes on
   the reference machine; anything at or under that is fine. Consequence:
   defaults favor quality — DINOv2 ViT-B backbone, 10 Hz sampling (§4.1,
   §4.3).
2. **Named apex labels: dropped for scene-mode v1.** Generic `sync_N`
   labels are used; a yaw-rate proxy may restore named apexes as future
   work (§10).
3. **Default mode: Catalyst until efficacy is demonstrated.** Scene mode
   ships behind `--mode scene`; it becomes the default once the regression
   harness (§8.3) shows parity with the map-dot pipeline on Catalyst
   footage.
4. **No Torch-free scene mode.** Personal-project scope; Torch is an
   unconditional dependency of scene mode. Legacy Catalyst mode continues
   to work as-is.

## 12. References

- STA-VPR: Spatio-temporal Alignment for Visual Place Recognition —
  https://arxiv.org/pdf/2103.13580
- AnyLoc: Towards Universal Visual Place Recognition —
  https://arxiv.org/abs/2308.00688
- Temporal Cycle-Consistency Learning — https://arxiv.org/abs/1904.07846
- LightGlue: Local Feature Matching at Light Speed —
  https://github.com/cvg/LightGlue
- LoMa: Local Feature Matching Revisited —
  https://arxiv.org/html/2604.04931v1
- RoMa vs. sparse matcher comparison (moving-car inspection) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10891783/
- VROOM: Visual Reconstruction over Onboard Multiview (F1 onboards) —
  https://arxiv.org/html/2508.17172v1
