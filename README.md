# tracksync: Comparing your dashcam to their dashcam

Tracksync synchronizes and compares racing dashcam videos. It automatically
detects synchronization points — by default from the **camera feed itself**
(scene mode: curbs, signage, pavement, corner stations), or from Garmin
Catalyst track-overlay positions (legacy `catalyst` mode) — or you can supply
timestamps manually from a CSV file. Videos are scaled to match at milestones
(e.g., turn apexes) and vertically stacked for comparison.

Scene mode is camera-agnostic: it compares any dashcam against any other,
without requiring a Catalyst overlay. It is the default as of the scene
alignment work (see `docs/scene_alignment_design.md`).

## Installation

### Set up virtual environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install the package (base: catalyst mode, generate, debug)
pip install -e .

# For scene mode (the default sync mode) — adds torch, timm, LightGlue, scipy
pip install -e '.[scene]'

# For OCR-based features (optional)
pip install pytesseract

# For development (includes testing tools)
pip install -e '.[dev]'
```

> **Scene mode requires the `scene` extra.** It pulls in PyTorch (a large,
> multi-hundred-MB download; GPU/MPS is used automatically when available but
> not required). If you only need legacy `catalyst` mode, `pip install -e .`
> is enough. Running scene mode without the extra exits with a clear
> `pip install -e '.[scene]'` message.

### Activate/Deactivate

```bash
# Activate the virtual environment
source .venv/bin/activate

# Deactivate when done
deactivate
```

## Usage

Tracksync provides three subcommands:

### Auto-sync videos (`sync`)

Automatically synchronize two videos. **The default is scene mode** (`--mode
scene`), which aligns the videos from their camera feeds and works on any
dashcam footage. Legacy `--mode catalyst` uses cross-correlation of the red
car marker on the Garmin Catalyst track overlay.

```bash
# Sync two videos (scene mode, default) and output timestamps to CSV
tracksync sync video_a.mp4 video_b.mp4 -o sync.csv

# Sync and also generate a comparison video
tracksync sync video_a.mp4 video_b.mp4 --generate-video --output-dir ./out

# Batch sync multiple videos efficiently (O(n) feature extraction)
tracksync sync --all video1.mp4 video2.mp4 video3.mp4 --output-dir ./sync_output

# Use the legacy Catalyst overlay pipeline instead
tracksync sync video_a.mp4 video_b.mp4 --mode catalyst -o sync.csv
```

Common options:
- `--mode {scene,catalyst}` - Alignment method (default: `scene`)
- `-o, --output FILE` - Output CSV file for sync timestamps
- `--output-dir DIR` - Output directory for generated files
- `--generate-video` - Generate comparison video after sync
- `--video-dir DIR` - Source video directory
- `--max-sync-interval SECS` - Maximum interval between sync points (default: 3.0)

Scene-mode options (only valid with `--mode scene`; each has a sensible default):
- `--sample-hz HZ` - Frame sampling rate for embeddings (default: 10)
- `--band-pct FRAC` - DTW band width as a fraction of clip length (default: 0.10)
- `--min-inliers N` - Minimum RANSAC inliers to accept a fine sync point (default: 30)
- `--fov-deg DEG` - Assumed horizontal camera field of view (default: 90)
- `--embedder NAME` - `dinov2-vitb14` (default), `dinov2-vits14`, or `gist`
- `--matcher NAME` - `aliked-lightglue` (default) or `superpoint-lightglue`

Catalyst-mode options (only valid with `--mode catalyst`):
- `--short, -s` - Analyze only first 20 seconds
- `--red-min VALUE` - Minimum R value for red detection (default: 200)
- `--red-max-gb VALUE` - Maximum G/B value for red detection (default: 80)

### Generate comparison videos (`generate`)

Generate comparison videos from an existing CSV file with timestamps.

```bash
# Generate videos from CSV
tracksync generate timestamps.csv --video-dir ./videos --output-dir ./out

# Using current directory for videos
tracksync generate sync.csv
```

Options:
- `--video-dir DIR` - Source video directory (default: current)
- `--output-dir DIR` - Output video directory (default: current)

Audio: the comparison video's audio comes from the *reference* (unwarped,
second) video, trimmed to the synced span. The speed-warped target audio is
never used, since time-scaling it would shift its pitch. If the reference
video has no audio track, the output is intentionally silent (a warning is
printed) — put the clip with usable audio second.

### Interactive debug mode (`debug`)

Visualize the cross-correlation process for debugging alignment issues.

```bash
# Interactive debug mode
tracksync debug video_a.mp4 video_b.mp4

# Quick test with first 20 seconds
tracksync debug video_a.mp4 video_b.mp4 --short
```

Controls:
- Arrow keys: Navigate frames
- Up/Down: Jump 1 second
- ESC: Exit

Options:
- `--short, -s` - Analyze only first 20 seconds
- `--red-min VALUE` - Minimum R value for red detection (default: 200)
- `--red-max-gb VALUE` - Maximum G/B value for red detection (default: 80)
- `--interval SECS` - Sample interval in seconds (default: 1.0)

## CSV Format

Tracksync uses a simple CSV format for timestamps:

```csv
milestone,driver1,driver2
start,0.000,0.500
turn_1,10.500,11.000
turn_2,20.750,21.250
finish,30.000,31.500
```

The first column is the milestone name. Subsequent columns contain the timestamp
(in seconds) when each driver reaches that milestone. Driver names are taken from
the header row.

## Troubleshooting

- **`pip install -e '.[scene]'` message on exit.** Scene mode needs the
  `scene` extra (torch/LightGlue). Install it, or run `--mode catalyst`.
- **Low-confidence spans / few sync points.** Scene mode verifies fine sync
  points geometrically and falls back to the coarse estimate when a match is
  weak. Very different lighting, weather, or camera mounting between the two
  clips reduces confidence. Try lowering `--min-inliers`, or set `--fov-deg`
  closer to your camera's true field of view.
- **Trim warnings / short overlap.** Scene mode aligns the overlapping span
  of the two clips and trims the rest; a short overlap yields a short output.
  Ensure both clips cover the same lap.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## How It Works

### Scene Alignment (default)

1. **Embedding**: Sample frames and embed each with a vision backbone
   (DINOv2 by default), masking static regions (car body, overlays).
2. **Coarse alignment**: Match the two embedding sequences with banded,
   open-ended dynamic time warping to get a frame-level time mapping.
3. **Fine refinement**: For each sync point, match local features
   (ALIKED + LightGlue) and use relative-pose geometry to refine timing to
   sub-frame precision, falling back to the coarse estimate when confidence
   is low.
4. **Video generation**: Same as below — scale and stack the two videos.

See `docs/scene_alignment_design.md` for the full design.

### Catalyst Auto-sync Algorithm (`--mode catalyst`)

1. **Feature Extraction**: Extract the position of the red car marker on the
   track overlay for each frame
2. **Cross-Correlation**: Match positions between videos to find corresponding
   frames where cars are at the same track position
3. **Sync Point Generation**: Generate milestone timestamps at regular intervals
   along the track
4. **Video Generation**: Scale and combine videos so they reach each milestone
   simultaneously

### Batch Processing

When using `--all` mode with multiple videos, tracksync extracts features from
each video only once (O(n) complexity) and then computes pairwise correlations.
This is much faster than processing each pair independently.
