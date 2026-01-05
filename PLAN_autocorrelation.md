# Video Autocorrelation Utility Plan

## Overview

Create a standalone command-line utility that automatically aligns two racing videos by detecting the position of a car marker on an overlaid track map. The utility uses autocorrelation of binarized frames to find matching timestamps between videos.

## Key Concepts

### Frame Processing Pipeline
1. **Extract frame** from video at time `t`
2. **Convert to monochrome binary**: pixels close to white → 1, all others → 0
3. **Autocorrelate** the binary frame from Video A against frames from Video B
4. **Find maximum correlation** to identify the best-matching frame

### Why Binary/Monochrome?
- The track overlay is white with black edges
- The car position marker is red with black edges
- The segment number text is white with black edges
- Binary comparison uses integer math (faster than floating point)
- Eliminates noise from differing video content (different drivers, cameras, etc.)

### Color Detection
Two colors are detected and set to 1 in the binary frame:
1. **White pixels**: R ≥ 240, G ≥ 240, B ≥ 240 (configurable threshold, default 240)
2. **Red pixels**: R ≥ 200, G ≤ 80, B ≤ 80 (captures the red car marker)

### Search Window Strategy
- **First frame**: Search the first 20 seconds of video B
- **Subsequent frames**: Search ±100 frames around the *previous frame's best match position*

This "tracking" approach works because:
- Cars start together (both videos begin near race start)
- Within 1 second, relative positions won't change dramatically
- Much more efficient than searching the entire video each time

## Architecture

### New Files to Create

```
tracksync/
├── tracksync/
│   └── autocorrelation.py    # Core autocorrelation logic
└── video_align.py            # Standalone CLI utility
```

### Dependencies to Add
- `opencv-python` - For efficient video frame extraction and image processing
- `numpy` - Already available via moviepy, used for array operations

Note: Using OpenCV instead of moviepy for frame extraction because:
- Much faster for random frame access
- Better suited for frame-by-frame analysis
- Native support for image processing operations

## Implementation Details

### 1. Frame Binarization (`autocorrelation.py`)

```python
def binarize_frame(
    frame: np.ndarray,
    white_threshold: int = 240,
    red_min: int = 200,
    red_max_gb: int = 80
) -> np.ndarray:
    """
    Convert RGB frame to binary (0/1) array.

    Two types of pixels become 1:
    1. White pixels: R, G, B all >= white_threshold
    2. Red pixels: R >= red_min AND G <= red_max_gb AND B <= red_max_gb

    All other pixels become 0.

    Args:
        frame: RGB frame as numpy array (H, W, 3)
        white_threshold: Minimum value for R, G, B to be considered "white" (default: 240)
        red_min: Minimum R value for red detection (default: 200)
        red_max_gb: Maximum G and B values for red detection (default: 80)

    Returns:
        Binary array (H, W) with dtype uint8
    """
```

### 2. Autocorrelation Function

```python
def compute_correlation(frame_a: np.ndarray, frame_b: np.ndarray) -> int:
    """
    Compute correlation between two binary frames.

    Uses element-wise AND and counts matching 1s.
    This is equivalent to dot product for binary arrays.

    Args:
        frame_a: Binary frame from video A
        frame_b: Binary frame from video B

    Returns:
        Integer count of matching white pixels
    """
    return np.sum(frame_a & frame_b)
```

### 3. Frame Search Window

```python
def find_best_match(
    video_a_frame: np.ndarray,
    video_b: cv2.VideoCapture,
    center_time: float | None,
    window_frames: int = 100,
    initial_search_seconds: float = 20.0
) -> tuple[float, int, list[tuple[float, int]]]:
    """
    Find the frame in video B that best matches video A frame.

    Search strategy:
    - If center_time is None (first frame): search first initial_search_seconds of video B
    - Otherwise: search ±window_frames around center_time

    Args:
        video_a_frame: Binarized frame from video A
        video_b: OpenCV VideoCapture for video B
        center_time: Time in video B to center the search window (None for first frame)
        window_frames: Number of frames before/after center_time to search (default: 100)
        initial_search_seconds: Seconds to search for first frame (default: 20.0)

    Returns:
        (best_match_time, best_correlation_score, all_correlation_values)
    """
```

### 4. Video Alignment Pipeline

```python
def align_videos(
    video_a_path: str,
    video_b_path: str,
    sample_interval: float = 1.0  # Sample once per second
) -> list[tuple[float, float]]:
    """
    Generate alignment timestamps between two videos.

    Args:
        video_a_path: Path to reference video (video A)
        video_b_path: Path to target video (video B)
        sample_interval: How often to sample (seconds)

    Returns:
        List of (time_a, time_b) tuples representing matched timestamps
    """
```

## CLI Utility (`video_align.py`)

### Usage

```bash
# Basic alignment (output timestamps to stdout or file)
python video_align.py video_a.mp4 video_b.mp4 --output timestamps.csv

# Debug mode with interactive visualization
python video_align.py video_a.mp4 video_b.mp4 --debug
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `video_a` | Path to reference video |
| `video_b` | Path to target video |
| `--output` | Output CSV file path (default: stdout) |
| `--debug` | Enable interactive debug mode |
| `--threshold` | White detection threshold (default: 240) |
| `--window` | Search window in frames (default: 100) |
| `--initial-search` | Initial search duration in seconds (default: 20.0) |
| `--red-min` | Minimum R value for red detection (default: 200) |
| `--red-max-gb` | Maximum G/B value for red detection (default: 80) |
| `--interval` | Sample interval in seconds (default: 1.0) |

### Debug Mode Visualization

The debug mode creates a window with three panels:

```
+-------------------+-------------------+
|                   |                   |
|   Video A Frame   |   Video B Frame   |
|   (current)       |   (best match)    |
|                   |                   |
+-------------------+-------------------+
|                                       |
|   Correlation Graph                   |
|   (x-axis: frame offset from center)  |
|   (y-axis: correlation value)         |
|   (vertical line at best match)       |
|                                       |
+---------------------------------------+
|  Time A: 12.5s  |  Time B: 13.2s     |
|  Correlation: 4523  |  Frame: 396    |
+---------------------------------------+
```

### Debug Mode Controls

| Key | Action |
|-----|--------|
| `→` (Right Arrow) | Advance 1 second in video A |
| `←` (Left Arrow) | Go back 1 second in video A |
| `ESC` | Exit debug mode |

## Testing Strategy

### Unit Tests (`tests/test_autocorrelation.py`)

1. **Binarization tests**
   - All-white frame → all 1s
   - All-black frame → all 0s
   - Mixed frame → correct threshold behavior
   - Near-white (RGB 239,239,239) → 0 with threshold 240
   - Red pixels (RGB 220,50,50) → 1
   - Orange pixels (RGB 220,150,50) → 0 (G too high)
   - Pink pixels (RGB 220,100,100) → 0 (G and B too high)

2. **Correlation tests**
   - Identical frames → maximum correlation
   - Inverted frames → zero correlation
   - Partial overlap → proportional correlation

3. **Search window tests**
   - Best match at center
   - Best match at edge of window
   - Consistent results across multiple runs

### Integration Test

Create synthetic test videos with:
- Known track overlay positions
- Predictable car marker movement
- Verify alignment accuracy within 1 frame

## Implementation Order

1. **Phase 1: Core Logic**
   - [ ] Create `autocorrelation.py` with binarization and correlation functions
   - [ ] Write unit tests for core functions
   - [ ] Verify with simple test cases

2. **Phase 2: Video Processing**
   - [ ] Add OpenCV-based frame extraction
   - [ ] Implement search window logic
   - [ ] Add alignment pipeline

3. **Phase 3: CLI Utility**
   - [ ] Create `video_align.py` with argument parsing
   - [ ] Implement basic mode (timestamp output)
   - [ ] Add CSV output formatting

4. **Phase 4: Debug Visualization**
   - [ ] Create visualization window layout
   - [ ] Implement correlation graph
   - [ ] Add keyboard navigation
   - [ ] Polish display formatting

5. **Phase 5: Testing & Refinement**
   - [ ] Test with real racing videos
   - [ ] Tune threshold and window parameters
   - [ ] Handle edge cases (video start/end, missing markers)

## Output Format

The utility outputs timestamps compatible with the existing `timestamps.csv` format:

```csv
# Auto-generated alignment timestamps
# Video A: eddie.mp4
# Video B: dan.mp4
time_a,time_b,correlation
0.0,0.5,4523
1.0,1.6,4812
2.0,2.7,4756
...
```

## Performance Considerations

- **Frame extraction**: Use OpenCV's `CAP_PROP_POS_MSEC` for efficient seeking
- **Binarization**: Vectorized numpy operations (no Python loops)
- **Correlation**: Use `np.sum(a & b)` for fast binary correlation
- **Memory**: Process frames one at a time, don't load entire videos
- **Caching**: Cache binarized frames from video B within search window

## Edge Cases to Handle

1. **Videos of different lengths**: Clamp search window to video B duration
2. **No white pixels in frame**: Return 0 correlation, warn user
3. **Multiple local maxima**: Report the global maximum, show all in debug
4. **Video codec issues**: Graceful error with helpful message
5. **Frame rate differences**: Normalize to time-based comparison
