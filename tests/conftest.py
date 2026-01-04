"""Pytest fixtures for tracksync tests."""

import pytest
import numpy as np
from moviepy import VideoClip, ColorClip

from tracksync.models import Segment


@pytest.fixture
def sample_segments():
    """Create sample segments for testing."""
    return [
        Segment("Start", 0.0, 100.0),
        Segment("T1", 10.0, 80.0),
        Segment("T2", 20.0, 90.0),
        Segment("Finish", 30.0, 100.0),
    ]


@pytest.fixture
def reference_segments():
    """Create reference segments with different timing."""
    return [
        Segment("Start", 0.0, 95.0),
        Segment("T1", 15.0, 75.0),  # 15s vs 10s = ratio 1.5
        Segment("T2", 25.0, 85.0),  # 10s vs 10s = ratio 1.0
        Segment("Finish", 40.0, 95.0),  # 15s vs 10s = ratio 1.5
    ]


@pytest.fixture
def synthetic_clip():
    """Create a synthetic video clip for testing."""

    def _create(duration=5.0, size=(320, 240), fps=24):
        def make_frame(t):
            # Create frame with time-based color gradient
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            # Red channel increases with time
            frame[:, :, 0] = int((t / duration) * 255) if duration > 0 else 0
            return frame

        return VideoClip(make_frame, duration=duration).with_fps(fps)

    return _create


@pytest.fixture
def color_clip():
    """Create a solid color clip."""

    def _create(duration=5.0, size=(320, 240), color=(255, 0, 0)):
        return ColorClip(size=size, color=color, duration=duration).with_fps(24)

    return _create


@pytest.fixture
def mock_video_loader(synthetic_clip):
    """Create a mock video loader that returns synthetic clips."""
    clips = {}

    def loader(filepath):
        if filepath not in clips:
            # Create a new synthetic clip for this filepath
            clips[filepath] = synthetic_clip(duration=60.0)
        return clips[filepath]

    return loader


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing.

    Format: milestone_name, driver1_timestamp, driver1_speed, driver2_timestamp, driver2_speed, ...
    """
    return """Reference,Driver1,speed1,Driver2,speed2
Start,0.00,100.00,0.00,95.00
T1,10.00,80.00,15.00,75.00
T2,20.00,90.00,25.00,85.00
Finish,30.00,100.00,40.00,95.00
"""
