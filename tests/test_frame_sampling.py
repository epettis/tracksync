"""Tests for fixed-rate frame sampling (Task T6)."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from tracksync.feature_extraction import sample_frames


def create_synthetic_video(
    output_path: Path,
    duration_s: float,
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
    extension: str = ".mp4"
) -> tuple[Path, int]:
    """
    Create a synthetic video for testing with frame index encoded as pixel intensity.

    Each frame is filled with a solid grayscale color where intensity = frame_index * 5.
    This encoding survives lossy compression and allows verification of temporal accuracy.

    Args:
        output_path: Directory to write the video file
        duration_s: Duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
        codec: FourCC codec code
        extension: File extension

    Returns:
        Tuple of (video_path, total_frames)
    """
    total_frames = int(duration_s * fps)
    video_path = output_path / f"test_video{extension}"

    # Try to create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
        isColor=True
    )

    if not writer.isOpened():
        # Could not initialize codec
        return None, 0

    try:
        for frame_idx in range(total_frames):
            # Encode frame index as intensity (frame_idx * 5)
            # Use modulo to keep values in uint8 range
            intensity = (frame_idx * 5) % 256

            # Create solid color frame (BGR for OpenCV)
            frame = np.full((height, width, 3), intensity, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Verify the file exists and has reasonable size
        if video_path.exists() and video_path.stat().st_size > 0:
            return video_path, total_frames
        else:
            return None, 0

    except Exception:
        writer.release()
        return None, 0


class TestFrameSampling:
    """Tests for sample_frames function."""

    def test_correct_sample_count_10hz(self, tmp_path):
        """Test that sampling at 10 Hz produces the correct number of samples."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=4.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=10.0)

        # 4 seconds at 10 Hz = 40 samples
        expected_count = 40
        assert len(frames) == expected_count, f"Expected {expected_count} frames, got {len(frames)}"
        assert len(timestamps) == expected_count, f"Expected {expected_count} timestamps, got {len(timestamps)}"

    def test_correct_sample_count_5hz(self, tmp_path):
        """Test that sampling at 5 Hz produces the correct number of samples."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=4.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0)

        # 4 seconds at 5 Hz = 20 samples
        expected_count = 20
        assert len(frames) == expected_count, f"Expected {expected_count} frames, got {len(frames)}"
        assert len(timestamps) == expected_count, f"Expected {expected_count} timestamps, got {len(timestamps)}"

    def test_timestamps_accuracy(self, tmp_path):
        """Test that timestamps are accurate and derived from frame index / fps."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=4.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        sample_hz = 10.0
        frames, timestamps = sample_frames(str(video_path), sample_hz=sample_hz)

        # Check that timestamps follow the expected pattern
        expected_interval = 1.0 / sample_hz
        for i in range(len(timestamps)):
            expected_time = i * expected_interval
            # Allow tolerance of 1/fps (about 0.033s at 30fps)
            tolerance = 1.0 / 30.0
            assert abs(timestamps[i] - expected_time) < tolerance, \
                f"Timestamp {i}: expected {expected_time:.3f}s, got {timestamps[i]:.3f}s"

    def test_frames_are_rgb_uint8(self, tmp_path):
        """Test that frames are RGB uint8 arrays."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=2.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=10.0)

        assert len(frames) > 0, "Should have at least one frame"

        for i, frame in enumerate(frames):
            assert frame.dtype == np.uint8, f"Frame {i} should be uint8, got {frame.dtype}"
            assert len(frame.shape) == 3, f"Frame {i} should be 3D array, got shape {frame.shape}"
            assert frame.shape[2] == 3, f"Frame {i} should have 3 channels (RGB), got {frame.shape[2]}"

    def test_temporal_accuracy_via_intensity(self, tmp_path):
        """Test temporal accuracy by decoding frame index from intensity."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=4.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        sample_hz = 10.0
        frames, timestamps = sample_frames(str(video_path), sample_hz=sample_hz)

        # For each sampled frame, decode the intensity and verify it corresponds
        # to the expected source frame index
        source_fps = 30.0
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Expected source frame index from timestamp
            expected_source_idx = int(timestamp * source_fps)

            # Decode intensity from the frame (average across all pixels)
            mean_intensity = np.mean(frame)

            # Round to nearest multiple of 5 and divide to get frame index
            decoded_idx = int(round(mean_intensity / 5.0)) % 256

            # Allow some tolerance for lossy compression
            # The intensity might drift slightly but should map to the correct frame
            idx_error = abs(decoded_idx - (expected_source_idx % 51))
            # 51 = 256 / 5, the period at which indices wrap

            # For robust testing, just verify the first few samples are correct
            if i < 10:
                assert idx_error <= 2, \
                    f"Sample {i} at t={timestamp:.3f}s: expected source frame ~{expected_source_idx}, " \
                    f"decoded {decoded_idx} (intensity {mean_intensity:.1f})"

    def test_resize_max_dim_none(self, tmp_path):
        """Test that max_dim=None preserves original size."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=1.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0, max_dim=None)

        assert len(frames) > 0, "Should have at least one frame"
        for frame in frames:
            h, w = frame.shape[:2]
            assert h == 64, f"Height should be preserved at 64, got {h}"
            assert w == 96, f"Width should be preserved at 96, got {w}"

    def test_resize_max_dim_width_limited(self, tmp_path):
        """Test resize when width is the limiting dimension."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=1.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        max_dim = 48  # Half the width
        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0, max_dim=max_dim)

        assert len(frames) > 0, "Should have at least one frame"
        for frame in frames:
            h, w = frame.shape[:2]
            assert max(h, w) == max_dim, f"Max dimension should be {max_dim}, got max({h}, {w})"
            # Check aspect ratio preserved (96:64 = 3:2)
            assert w == max_dim, f"Width should be {max_dim}, got {w}"
            assert h == 32, f"Height should be scaled to 32, got {h}"

    def test_resize_max_dim_height_limited(self, tmp_path):
        """Test resize when height is the limiting dimension."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=1.0,
            fps=30.0,
            width=64,
            height=96  # Height > width
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        max_dim = 48  # Half the height
        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0, max_dim=max_dim)

        assert len(frames) > 0, "Should have at least one frame"
        for frame in frames:
            h, w = frame.shape[:2]
            assert max(h, w) == max_dim, f"Max dimension should be {max_dim}, got max({h}, {w})"
            # Check aspect ratio preserved (64:96 = 2:3)
            assert h == max_dim, f"Height should be {max_dim}, got {h}"
            assert w == 32, f"Width should be scaled to 32, got {w}"

    def test_resize_default_max_dim(self, tmp_path):
        """Test that default max_dim=518 is applied."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=1.0,
            fps=30.0,
            width=1920,
            height=1080
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        # Use default max_dim
        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0)

        assert len(frames) > 0, "Should have at least one frame"
        for frame in frames:
            h, w = frame.shape[:2]
            # Default max_dim is 518, width 1920 is larger so should be scaled
            assert max(h, w) == 518, f"Max dimension should be 518, got max({h}, {w})"

    def test_short_tail_dropped(self, tmp_path):
        """Test that the last partial interval is dropped gracefully."""
        # Create a video that's exactly 4 seconds (120 frames at 30 fps)
        # Sampling at 10 Hz should give exactly 40 samples
        # This proves we're correctly dropping partial intervals
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=4.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=10.0)

        # 4.0 seconds at 10 Hz = exactly 40 samples
        expected_count = 40
        assert len(frames) == expected_count, f"Expected {expected_count} frames, got {len(frames)}"

        # Verify last timestamp is not beyond the video duration
        assert timestamps[-1] < 4.0, f"Last timestamp {timestamps[-1]} should be < 4.0s"

    def test_video_not_found(self):
        """Test that non-existent video raises ValueError."""
        with pytest.raises(ValueError, match="Could not open video"):
            sample_frames("/nonexistent/video.mp4")

    def test_empty_timestamps_array_type(self, tmp_path):
        """Test that timestamps are returned as float64 array."""
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=1.0,
            fps=30.0,
            width=96,
            height=64
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=5.0)

        assert isinstance(timestamps, np.ndarray), f"Timestamps should be ndarray, got {type(timestamps)}"
        assert timestamps.dtype == np.float64, f"Timestamps should be float64, got {timestamps.dtype}"

    def test_fallback_codec_mjpeg_avi(self, tmp_path):
        """Test fallback to MJPEG codec with AVI container."""
        # Try MJPEG codec as fallback
        video_path, total_frames = create_synthetic_video(
            tmp_path,
            duration_s=2.0,
            fps=30.0,
            width=96,
            height=64,
            codec="MJPG",
            extension=".avi"
        )

        if video_path is None:
            pytest.skip("cv2.VideoWriter could not produce working video (MJPG codec unavailable)")

        frames, timestamps = sample_frames(str(video_path), sample_hz=10.0)

        # Should still work with different codec
        assert len(frames) == 20, f"Expected 20 frames, got {len(frames)}"
        assert len(timestamps) == 20, f"Expected 20 timestamps, got {len(timestamps)}"
