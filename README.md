# tracksync: Comparing your dashcam to their dashcam

Tracksync synchronizes and compares racing dashcam videos. It can automatically
detect synchronization points using cross-correlation of track overlay positions,
or use manually provided timestamps from a CSV file. Videos are scaled to match
at milestones (e.g., turn apexes) and vertically stacked for comparison.

## Installation

### Set up virtual environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install moviepy opencv-python numpy

# For OCR-based features (optional)
pip install pytesseract

# For development (includes testing tools)
pip install pytest pytest-cov
```

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

Automatically synchronize videos using cross-correlation of the red car marker
position on the track overlay.

```bash
# Sync two videos and output timestamps to CSV
tracksync sync video_a.mp4 video_b.mp4 -o sync.csv

# Sync and also generate comparison video
tracksync sync video_a.mp4 video_b.mp4 --generate-video --output-dir ./out

# Quick test with first 20 seconds only
tracksync sync video_a.mp4 video_b.mp4 -o sync.csv --short

# Batch sync multiple videos efficiently (O(n) feature extraction)
tracksync sync --all video1.mp4 video2.mp4 video3.mp4 --output-dir ./sync_output
```

Options:
- `-o, --output FILE` - Output CSV file for sync timestamps
- `--output-dir DIR` - Output directory for generated files
- `--generate-video` - Generate comparison video after sync
- `--video-dir DIR` - Source video directory
- `--max-sync-interval SECS` - Maximum interval between sync points (default: 3.0)
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

### Legacy format with speed

For compatibility, tracksync also supports a legacy format with speed columns:

```csv
milestone,driver1,,driver2,,
start,0.0,100.0,0.1,100.2
m1,12.3,78.9,11.2,77.8
```

Empty header columns indicate speed values follow the timestamps.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## How It Works

### Auto-sync Algorithm

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
