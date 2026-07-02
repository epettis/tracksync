# Scene Alignment Validation Results

Operator log for T13 validation runs comparing scene-based alignment against
the legacy Catalyst pipeline (design doc SS8; acceptance gate for T14 is
p95 |dt| <= 1 native frame at sync points).

How to run (from the repo root, on the reference machine):

```bash
# Pair validation: Catalyst vs scene on a real Catalyst clip pair
.venv/bin/python scripts/validate_scene_alignment.py videoA.mp4 videoB.mp4

# Self-alignment check: scene-align a video against itself
.venv/bin/python scripts/validate_scene_alignment.py --self video.mp4
```

Columns: |dt| statistics are in seconds, formatted max / mean / p95.
"n/a" means the comparison was unavailable (e.g., no OCR data, self runs).

| Date | Kind | Inputs | Catalyst runtime (s) | Scene runtime (s) | Scene vs Catalyst \|dt\| | Catalyst vs OCR \|dt\| | Scene vs OCR \|dt\| | Notes |
|------|------|--------|----------------------|-------------------|------------------------|------------------------|--------------------|-------|
| 2026-07-01 | pair | eddie_pb.mp4 vs kurt_optimal.mp4 | 1540.5 | 5426.4 | 5.127 / 0.602 / 2.176 | 3.537 / 1.542 / 3.464 | 4.083 / 1.638 / 3.929 | embedder=dinov2-vitb14, matcher=aliked-lightglue, ocr_pairs=5 |
| 2026-07-02 | pair | eddie_pb.mp4 vs kurt_optimal.mp4 | 1541.6 | 5187.0 | 1.611 / 0.385 / 0.795 | 3.537 / 1.542 / 3.464 | 3.983 / 1.655 / 3.809 | embedder=dinov2-vitb14, matcher=aliked-lightglue, ocr_pairs=5 |
