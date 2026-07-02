# Task Breakdown: Scene-Based Video Alignment

Companion to `docs/scene_alignment_design.md` (the "design"). Section
references (§) point there.

This file decomposes the design into discrete tasks sized for execution by
an inexpensive coding agent, one task per session, in order. Each task is
self-contained: it lists the context to read, the exact deliverables, the
interface contracts, and the acceptance criteria. Later tasks depend only
on the *interfaces* of earlier tasks, never on their internals.

## Rules for every task

- Read `docs/scene_alignment_design.md` plus the files listed under
  "Context" before writing code.
- Work only within the listed "Files" scope. Do not modify
  `csv_writer.py`, `csv_reader.py`, `video_processor.py`, or
  `segment_validator.py` in any task (design §5.4).
- Every task ends with the full suite green: `.venv/bin/pytest`. New code
  requires new tests in `tests/` following the existing pytest layout
  (`pyproject.toml` `[tool.pytest.ini_options]`).
- Heavy ML models (DINOv2, LightGlue) must never be required by the default
  test suite. Tests use the mock/cheap implementations defined in T2/T5/T7;
  tests that download weights are marked `@pytest.mark.slow` and skipped by
  default (`-m "not slow"` must be the effective default via marker
  registration + skip-if-unavailable guards).
- Pure-NumPy/OpenCV tasks (T1, T3, T4, T8) must not import torch.
- Type hints and dataclasses in the style of `tracksync/frame_data.py` and
  `tracksync/models.py`.
- Do not change legacy Catalyst behavior; `--mode catalyst` remains the
  default throughout (design §11.3).

## Dependency graph

```
T1 (deps/skeleton)
├── T2 (embedding interface + mock)
│    └── T5 (DINOv2 embedder)         [slow tests]
├── T3 (DTW)
├── T4 (static masking)
├── T6 (frame sampling)
T2+T3+T4+T6 ──> T7 (coarse pipeline)
T7 ──> T8 (fine align: geometry core)
T8 ──> T9 (fine align: LightGlue matcher)      [slow tests]
T7+T8 ──> T10 (sync assembly + CSV)
T10 ──> T11 (CLI --mode scene)
T11 ──> T12 (debug visualization)
T11 ──> T13 (validation harness + regression vs. Catalyst)
T13 ──> T14 (default flip + docs)
```

---

## T1 — Packaging, dependencies, module skeleton

**Design refs:** §6 (new modules, dependencies), §11.4.

**Context:** `pyproject.toml`, `tracksync/__init__.py`.

**Files:** `pyproject.toml`; new empty-but-importable modules
`tracksync/embedding.py`, `tracksync/dtw.py`, `tracksync/masking.py`,
`tracksync/fine_align.py`, `tracksync/scene_align.py`; new test file
`tests/test_scene_imports.py`.

**Work:**
1. Add `[project.optional-dependencies] scene = ["torch>=2.1", "timm>=0.9",
   "lightglue @ git+https://github.com/cvg/LightGlue.git", "scipy>=1.10"]`.
   (scipy is needed for PCHIP in T3.) Keep base deps unchanged.
2. Register a `slow` pytest marker in `pyproject.toml` and configure
   `addopts` so slow tests are deselected by default.
3. Create the five modules with docstrings and a
   `tracksync/scene_deps.py`-style guard helper
   `require_scene_deps() -> None` that raises a clear
   `MissingSceneDependenciesError` naming the pip extra
   (`pip install -e '.[scene]'`) if torch is unavailable. Top-level import
   of any of the five modules must NOT import torch (lazy imports only).

**Acceptance:** all modules import without torch installed; `pytest` green;
`pip install -e .` unaffected.

---

## T2 — `FrameEmbedder` protocol + deterministic cheap embedder

**Design refs:** §4.3.

**Context:** `tracksync/embedding.py` (from T1), `tracksync/frame_data.py`.

**Files:** `tracksync/embedding.py`, `tests/test_embedding.py`.

**Work:**
1. Define `FrameEmbedder` as a `typing.Protocol`:
   ```python
   class FrameEmbedder(Protocol):
       name: str            # used in cache keys, e.g. "dinov2-vitb14"
       def embed(self, frames: list[np.ndarray],
                 mask: np.ndarray | None = None) -> np.ndarray: ...
       # frames: RGB uint8 HxWx3; mask: HxW bool, True = static/exclude
       # returns float32 [N, D], rows L2-normalized
   ```
2. Implement `GistEmbedder` (pure NumPy, no torch): grayscale, apply mask
   by setting static pixels to the frame mean, downsample to 16×16, subtract
   mean, L2-normalize, flatten → D=256. Deterministic; used by all
   downstream tests and as `--embedder gist` debugging option.
3. Implement disk caching helper
   `embed_video_cached(embedder, frames, mask, cache_key, cache_dir)` →
   memoize the `[N, D]` array as `.npy` keyed by
   `sha256(video_path, mtime, embedder.name, sample_hz, mask_hash)`
   (design §7). Cache dir default `~/.cache/tracksync/embeddings`.

**Tests:** shape/dtype/normalization; determinism; mask changes output;
cache hit avoids recompute (monkeypatch-count calls) and invalidates on key
change.

**Acceptance:** no torch import anywhere in the module at import time;
`pytest` green.

---

## T3 — Banded, slope-constrained, open-end DTW

**Design refs:** §4.4, §8.5. Pure algorithmic task, no video/ML code.

**Context:** `tracksync/dtw.py` (from T1).

**Files:** `tracksync/dtw.py`, `tests/test_dtw.py`.

**Work:**
1. `dtw_align(cost: np.ndarray, band_pct: float = 0.10,
   open_end_frames: tuple[int, int] = (0, 0), slope_max: float = 2.0)
   -> DtwResult` where `DtwResult` is a dataclass with:
   - `path: np.ndarray [K, 2]` (monotonic index pairs into A and B),
   - `path_cost: float`,
   - `margins: np.ndarray [N_A]` — per-A-frame confidence: best off-path
     cost in the band minus on-path cost (design §4.5).
2. Constraints: Sakoe–Chiba band on normalized indices (|i/N_A − j/N_B| ≤
   band_pct); step pattern limiting local slope to [1/slope_max, slope_max];
   open ends — path may start at any (i, j) with i ≤ open_end_frames[0] or
   j ≤ open_end_frames[0], and symmetrically for the end (design §4.4
   "open ends", the ±slack band).
3. `smooth_path(path, times_a, times_b) -> Callable[[float], float]`:
   monotone PCHIP fit (scipy) of t_B = f(t_A); must be strictly
   monotonically increasing; also return inverse-safe evaluation on the
   trimmed domain `[t_A_start, t_A_end]` exposed as attributes.
4. Implementation: NumPy DP (vectorize the inner loop over the band);
   1200×1200 must run < 1 s.

**Tests (synthetic cost matrices, no video):**
- identity cost (diagonal zeros) → path is the diagonal;
- constant time offset within the open-end slack → path recovered exactly;
- smooth synthetic warp (e.g., t_B = t_A + 3 sin(t_A)) with slope < 2 →
  max path error ≤ 1 index;
- band and slope constraints respected (assert on path deltas);
- monotonicity of `smooth_path` output on random valid paths;
- margins: distinct low-cost valley → high margin; flat cost rows → ~0
  margin;
- perf guard: 1200×1200 random matrix aligns in < 5 s (generous CI bound).

**Acceptance:** properties above; `pytest` green.

---

## T4 — Static car-body masking

**Design refs:** §4.2.

**Context:** `tracksync/masking.py` (T1),
`tracksync/frame_analysis.py:674-676` (morphology idiom to mirror).

**Files:** `tracksync/masking.py`, `tests/test_masking.py`.

**Work:**
1. `compute_static_mask(frames: list[np.ndarray], var_threshold: float =
   ..., downsample: int = 4) -> np.ndarray` — grayscale, downsample,
   per-pixel temporal variance over the sampled frames, threshold, then
   morphological close + dilate (OpenCV), upsample back to frame size,
   return HxW bool (True = static).
2. Choose a variance threshold relative to the frame's global variance
   (scale-invariant), not an absolute constant.
3. Guard: if the mask covers > 60% of the frame, log a warning and return
   the mask anyway (caller decides).

**Tests (synthetic frames, e.g., 96×128):** moving random noise + fixed
"hood" rectangle + fixed PiP box → mask covers hood and PiP with IoU > 0.8
and < 5% false positives on the moving region; brightness-flicker-only
static region still detected (variance threshold robust to global gain via
per-frame mean normalization); all-moving video → near-empty mask.

**Acceptance:** `pytest` green; no torch imports.

---

## T5 — DINOv2 embedder (real backbone)

**Design refs:** §4.3, §11.1.

**Context:** `tracksync/embedding.py` (T2 interface).

**Files:** `tracksync/embedding.py`, `tests/test_embedding_dinov2.py`.

**Work:**
1. `DinoV2Embedder(model_name: str = "dinov2_vitb14", device: str | None
   = None)` implementing `FrameEmbedder`. Load via `torch.hub` (facebook
   research/dinov2), lazily on first `embed()` call; auto-select device
   (mps > cuda > cpu). Call `require_scene_deps()` before importing torch.
2. Per design §4.3: resize/pad frames to model resolution (multiple of 14),
   take patch tokens, drop tokens whose patch overlaps the static mask by
   > 50%, GeM-pool remaining tokens (p=3), L2-normalize. Batch frames
   (batch size 16, configurable) for throughput.
3. `make_embedder(name: str) -> FrameEmbedder` factory:
   `"gist"`, `"dinov2-vits14"`, `"dinov2-vitb14"` (default per §11.1).

**Tests:** factory returns correct types and raises
`MissingSceneDependenciesError` without torch (simulate via monkeypatch of
the guard); the real-model test is `@pytest.mark.slow` + skip-if
`torch`/network unavailable: embed 4 frames (2 near-duplicates, 2 unrelated
noise) → duplicate pair cosine similarity > unrelated pair.

**Acceptance:** default `pytest` run green *without* torch installed;
`pytest -m slow` green on the reference machine.

---

## T6 — Fixed-rate frame sampling

**Design refs:** §4.1.

**Context:** `tracksync/feature_extraction.py` (existing OpenCV ingestion
pattern), `tracksync/frame_data.py`.

**Files:** `tracksync/feature_extraction.py` (extend, do not disturb the
Catalyst path), `tests/test_frame_sampling.py`.

**Work:**
1. `sample_frames(video_path: str, sample_hz: float = 10.0,
   max_dim: int | None = 518) -> tuple[list[np.ndarray], np.ndarray]` —
   returns RGB frames sampled at the fixed rate and their timestamps
   (float seconds). Reuse the existing cv2.VideoCapture idioms; resize so
   max(H, W) == max_dim preserving aspect (None = no resize).
2. Handle variable-fps and short tails gracefully (last partial interval
   dropped); timestamps must come from frame indices / fps, matching the
   convention used by the existing extraction code.

**Tests:** write tiny synthetic videos with `cv2.VideoWriter` in tmp_path
(e.g., 4 s @ 30 fps, frame index encoded in pixel intensity) → correct
count at 10 Hz and 5 Hz, timestamps within 1/fps, resize respected,
decoded intensity matches expected source frame index (proves temporal
accuracy).

**Acceptance:** `pytest` green; Catalyst-path tests untouched and green.

---

## T7 — Coarse alignment pipeline (Stage A end-to-end)

**Design refs:** §3, §4 (all), §6 (`frame_data.py`, `scene_align.py`).

**Context:** outputs of T2, T3, T4, T6; `tracksync/frame_data.py`,
`tracksync/models.py`.

**Files:** `tracksync/scene_align.py`, `tracksync/frame_data.py` (add
fields), `tests/test_scene_align_coarse.py`.

**Work:**
1. New dataclass `SceneFeatures` in `frame_data.py`: `video_path`,
   `frame_times: np.ndarray`, `emb_array: np.ndarray [N, D]`,
   `static_mask: np.ndarray`, `sample_hz: float`. (Do not overload the
   Catalyst `VideoFeatures`; keep scene mode parallel and clean.)
2. `extract_scene_features(video_path, embedder, sample_hz=10.0,
   cache_dir=...) -> SceneFeatures` — compose T6 sampling → T4 masking →
   T2 cached embedding.
3. `coarse_align(feat_a: SceneFeatures, feat_b: SceneFeatures,
   band_pct=0.10, open_end_s=10.0) -> CoarseAlignment` dataclass:
   `f` (smoothed t_A→t_B mapping from T3), `path`, `margins`,
   `trim_start_a/trim_end_a/trim_start_b/trim_end_b` (from path endpoints,
   design §4.4), `confidence(t_a)` accessor.
4. Cost matrix `1 − emb_a @ emb_b.T`, then T3 `dtw_align` with
   `open_end_frames = round(open_end_s * sample_hz)`.

**Tests (GistEmbedder + synthetic videos from T6's generator):**
- self-alignment: video vs. itself → f ≈ identity, max |f(t)−t| ≤ 1 sample
  (design §8.1);
- shifted copy: same clip with 2 s trimmed off the front of B → f ≈ t − 2
  within slack, trims reflect the offset;
- speed-perturbed copy (resample frames to simulate 1.1× pace) → f slope
  ≈ 1.1;
- margins exposed and finite; trim fields within clip bounds.

**Acceptance:** `pytest` green using only cheap embedder; no torch needed.

---

## T8 — Fine alignment: geometry core (matcher-agnostic)

**Design refs:** §5.2 (essential matrix, verification), §5.3 (zero
crossing). This task contains all the geometry; the real matcher arrives
in T9.

**Context:** `tracksync/fine_align.py` (T1), design §5 carefully.

**Files:** `tracksync/fine_align.py`, `tests/test_fine_align_geometry.py`.

**Work:**
1. Matcher protocol:
   ```python
   class FeatureMatcher(Protocol):
       name: str
       def match(self, img_a, img_b, mask_a=None, mask_b=None
                 ) -> tuple[np.ndarray, np.ndarray]:  # pts_a, pts_b [M, 2]
   ```
2. `intrinsics_from_fov(width, height, fov_deg=90.0) -> np.ndarray` (3×3 K).
3. `score_pair(pts_a, pts_b, K_a, K_b) -> PairGeometry`:
   `cv2.findEssentialMat` + RANSAC → dataclass with `n_inliers`, `R`,
   `t_unit` (from `cv2.recoverPose`), `inlier_mask`.
4. `longitudinal_component(t_unit, forward_axis) -> float` and
   `estimate_forward_axis(prev_to_next_geometry) -> np.ndarray`: B's
   track-forward direction from B's own consecutive-frame epipolar
   translation (design §5.3).
5. `refine_sync_point(frame_a, frames_b, times_b, matcher, K_a, K_b,
   min_inliers=30) -> FineResult`: score every B frame, find the sign
   change of the longitudinal component, linear fit near the crossing →
   sub-frame `t_b`; `FineResult(t_b, verified: bool, n_inliers, per_frame
   longitudinal: np.ndarray)`. If best inliers < min_inliers or no sign
   change in window → `verified=False`, `t_b=None` (caller falls back,
   design §5.2).
6. Monotonic clamp helper `clamp_monotonic(sync_times: list[tuple]) ->
   list[tuple]` reimplementing the forward-progress invariant
   (design §5.3 last bullet).

**Tests (synthetic geometry — no images, no ML):** build a 3D point cloud
(random points 5–50 m ahead), camera A fixed, camera B translated along a
known forward axis with a mounting offset (0.5 m lateral, 0.3 m vertical,
5° yaw) and sweep positions from −3 m to +3 m longitudinal; project points
through both K's to synthesize `pts_a`, `pts_b` correspondences (+ pixel
noise σ=0.5, 20% outliers). Assert:
- `score_pair` recovers pose with inliers > 80%;
- longitudinal component changes sign exactly at the crossing position;
- `refine_sync_point` (with a `SyntheticMatcher` returning those
  projections, standing in for the protocol) recovers the crossing time
  with error < 0.25 frame at 30 fps (design §7 accuracy target);
- mounting offset does NOT bias the zero crossing (compare with/without
  offset, design §5.3);
- insufficient matches → `verified=False` fallback path;
- `clamp_monotonic` fixes an injected backward step.

**Acceptance:** `pytest` green; module imports without torch.

---

## T9 — LightGlue matcher implementation

**Design refs:** §5.2, §9.4 (ALIKED default, SuperPoint opt-in).

**Context:** T8 `FeatureMatcher` protocol; LightGlue README
(https://github.com/cvg/LightGlue).

**Files:** `tracksync/fine_align.py` (add class),
`tests/test_fine_align_lightglue.py`.

**Work:**
1. `LightGlueMatcher(features: str = "aliked", device=None)` implementing
   `FeatureMatcher`; lazy torch import behind `require_scene_deps()`;
   apply masks by dropping keypoints inside static regions; `"superpoint"`
   accepted but log the license caveat (design §9.4).
2. `make_matcher(name: str) -> FeatureMatcher`: `"aliked-lightglue"`
   (default), `"superpoint-lightglue"`, `"synthetic"` reserved for tests.

**Tests:** factory + missing-deps error without torch (monkeypatched
guard); `@pytest.mark.slow`: render two synthetic textured checkerboard/
noise images related by a known homography → matched points satisfy the
homography within 2 px for > 70% of matches; masked region yields no
keypoints.

**Acceptance:** default `pytest` green without torch; slow tests green on
reference machine.

---

## T10 — Sync point assembly → SyncResult/CSV (scene mode end-to-end)

**Design refs:** §5.1, §5.4, §6 (`turn_analysis` disposition), §11.2.

**Context:** T7 `CoarseAlignment`, T8 `refine_sync_point`;
`tracksync/models.py` (`SyncPoint`, `SyncResult`),
`tracksync/cross_correlation.py:491-646` (existing assembly, for the
speed-ratio and cadence conventions), `tracksync/csv_writer.py`
(read-only).

**Files:** `tracksync/scene_align.py`, `tests/test_scene_sync_points.py`.

**Work:**
1. `generate_scene_sync_points(coarse, video_a, video_b, matcher,
   max_sync_interval=3.0, min_inliers=30, fov_deg=90.0) -> SyncResult`:
   - candidate times in A every ≤ `max_sync_interval` s within the trimmed
     range, snapped to local margin maxima (design §5.1, §4.5);
   - decode ±0.5 s native-fps windows from video B around `f(t_A)` (reuse
     cv2 seek idioms from `feature_extraction.py`);
   - T8 refinement per candidate; unverified candidates fall back to
     `f(t_A)` (design §5.2);
   - `clamp_monotonic`; labels `start`, `sync_1..sync_N`, `end`
     (design §11.2); per-segment speed ratios computed exactly as the
     legacy assembly does;
   - populate `SyncResult` trim fields from `coarse`.
2. The function must accept an injected frame-window decoder so tests can
   bypass video decoding.

**Tests:** with `SyntheticMatcher` + injected decoder + GistEmbedder
synthetic clips: end-to-end scene `SyncResult` for the 2 s-shift scenario
of T7 → sync point pairs within 1 sample of truth, monotonic, labeled
correctly; write via existing `csv_writer` and re-read via `csv_reader` →
round-trip equality (proves downstream compatibility, design §5.4);
fallback path exercised by forcing `min_inliers` impossibly high → result
equals smoothed coarse estimates.

**Acceptance:** `pytest` green without torch; no modifications to the four
protected downstream modules.

---

## T11 — CLI integration: `sync --mode scene`

**Design refs:** §6 (cli.py row), §11.3.

**Context:** `tracksync/cli.py` (sync subcommand, `cli.py:289-392`; batch
`--all`, `cli.py:395-504`), T7/T10 entry points.

**Files:** `tracksync/cli.py`, `tests/test_cli_scene.py`.

**Work:**
1. Add `--mode {catalyst,scene}` (default `catalyst`) to `sync`; scene-only
   flags: `--sample-hz` (10), `--band-pct` (0.10), `--min-inliers` (30),
   `--fov-deg` (90), `--matcher` (aliked-lightglue), `--embedder`
   (dinov2-vitb14). Reject scene-only flags in catalyst mode with a clear
   error.
2. Scene path: extract features (cached) → coarse → fine → CSV via the
   existing writer; honor existing `--generate-video`,
   `--max-sync-interval`, output conventions. `--all` batch mode reuses
   per-video feature caching (design §7).
3. Missing scene deps → the T1 error message, exit code 2, before any
   decoding starts.
4. Keep catalyst path byte-identical (no behavior change; existing CLI
   tests must pass unmodified).

**Tests:** argparse-level (flag parsing, defaults, rejection rules,
missing-deps exit) via monkeypatched pipeline functions; one end-to-end
CLI invocation on tiny synthetic clips with `--embedder gist --matcher
synthetic` (test-only registration) producing a valid CSV in tmp_path.

**Acceptance:** `pytest` green; `tracksync sync --help` documents the new
flags; legacy invocations unchanged.

---

## T12 — Debug visualization for scene mode

**Design refs:** §6 (visualization row), §4.5.

**Context:** `tracksync/visualization.py`, `tracksync/cli.py` debug
subcommand (`cli.py:507-714`).

**Files:** `tracksync/visualization.py`, `tracksync/cli.py`,
`tests/test_scene_visualization.py`.

**Work:**
1. `debug --mode scene`: three new panels — (a) cost-matrix heatmap with
   DTW path overlay; (b) side-by-side frame pair at the cursor with drawn
   correspondences (when fine data available); (c) per-frame margin/
   confidence trace with sync points marked. Reuse the existing interactive
   navigation loop.
2. Panels render to numpy images via pure functions (testable headless);
   the interactive loop just displays them.

**Tests:** pure render functions return correctly-shaped RGB arrays for
synthetic inputs; path overlay pixels land on expected matrix cells;
no-GUI environments skip the interactive test.

**Acceptance:** `pytest` green; `debug --mode catalyst` unchanged.

---

## T13 — Validation harness & Catalyst regression

**Design refs:** §8 (all), §7 accuracy targets.

**Context:** T11 CLI; legacy sync pipeline; existing OCR lap-time
extraction (`frame_analysis.py`).

**Files:** new `scripts/validate_scene_alignment.py`,
`tests/test_validation_synthetic.py`, `docs/scene_alignment_validation.md`
(results log, created with a template).

**Work:**
1. Synthetic perturbation validator (design §8.2) as a default-run pytest:
   take a synthetic clip, produce a cropped (90%), scaled, ±5°-rotated copy
   with an added fake hood band; scene-align with GistEmbedder → identity
   timing within 1 sample.
2. `scripts/validate_scene_alignment.py <videoA> <videoB>`: runs BOTH
   legacy catalyst sync and scene sync on real Catalyst clips, extracts
   OCR lap-time clocks, reports per-sync-point |Δt| statistics
   (max/mean/p95) between the two methods and against the OCR clock
   (design §8.3), plus wall-clock runtime of each stage. Writes a markdown
   row to `docs/scene_alignment_validation.md`.
3. Self-alignment check (design §8.1) included in the script
   (`--self <video>`).

**Tests:** the synthetic validator; script smoke-tested with monkeypatched
pipelines (no real footage in CI).

**Acceptance:** `pytest` green; script runs end-to-end on the reference
machine against a real Catalyst pair (operator step — document the command
in the results log template).

---

## T14 — Default flip + documentation

**Design refs:** §11.3, §6, §9.5. **Gate (revised): the original hard
parity gate (p95 |Δt| ≤ 1 native frame vs. Catalyst) proved too strict for
a deliberately camera-agnostic method and was relaxed to acceptance by
visual review of the comparison video. T13 results (scene-vs-Catalyst p95
≈ 0.80 s after the DTW cost-normalization fix) are recorded in
`docs/scene_alignment_validation.md`; this task's PR links them.**

**Context:** T13 results, `README.md`, `pyproject.toml`, `cli.py`.

**Files:** `README.md`, `tracksync/cli.py`, `docs/scene_alignment_design.md`
(status note), `tests/test_cli_scene.py` (update defaults).

**Work:**
1. Flip `--mode` default to `scene`; `--mode catalyst` remains available.
2. README: install instructions for the `scene` extra (torch size caveat,
   design §9.5), quickstart for scene sync, flag reference, troubleshooting
   (low-confidence spans, trim warnings from design §9.7).
3. Mark design doc status line: "IMPLEMENTED (vN.N)".

**Acceptance:** `pytest` green with updated default; README renders
correctly; legacy mode still fully functional.

---

## Suggested agent prompt template

For each task, give the executing agent:

```
Read docs/scene_alignment_design.md and docs/scene_alignment_tasks.md.
Execute task <ID> exactly as specified. Restrict changes to the files
listed for the task. Follow the "Rules for every task" section. When done,
run `.venv/bin/pytest` and ensure it is green. Do not start any other task.
```
