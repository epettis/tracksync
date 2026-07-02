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
