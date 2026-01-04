"""Video processing operations using MoviePy."""

from typing import Callable, List, Optional

import numpy as np
from moviepy import (
    ColorClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)

from .models import ProcessedSegment


class VideoProcessor:
    """Handles all MoviePy video operations."""

    def __init__(
        self, clip_loader: Optional[Callable[[str], VideoFileClip]] = None
    ) -> None:
        """
        Initialize processor with optional custom clip loader.

        Args:
            clip_loader: Function to load video clips (for testing injection)
        """
        self._load_clip = clip_loader or VideoFileClip

    def load_video(self, filepath: str) -> VideoFileClip:
        """Load a video file."""
        return self._load_clip(filepath)

    def extract_segment(
        self,
        clip: VideoFileClip,
        start: float,
        end: float,
    ) -> VideoFileClip:
        """Extract a time segment from a clip."""
        return clip.subclipped(start, end)

    def apply_speed(
        self,
        clip: VideoFileClip,
        speed_ratio: float,
    ) -> VideoFileClip:
        """
        Apply speed adjustment to a clip.

        MoviePy's with_speed_scaled() multiplies playback speed:
        - with_speed_scaled(2) makes video play 2x faster (shorter duration)
        - with_speed_scaled(0.5) makes video play at half speed (longer duration)

        Our ratio is: target_duration / ref_duration
        - ratio > 1: Target is faster, slow it down to match reference
        - ratio < 1: Target is slower, speed it up to match reference

        Example: Target 10s, reference 15s → ratio = 10/15 = 0.67
        with_speed_scaled(0.67) slows target from 10s to 15s.
        """
        return clip.with_speed_scaled(speed_ratio)

    def create_freeze_frame(
        self,
        clip: VideoFileClip,
        frame_time: float,
        duration: float,
    ) -> VideoClip:
        """
        Create a static frame clip from the video.

        Used when a segment extends beyond the video duration or when
        a segment would have zero duration after clamping.

        Args:
            clip: Source video clip
            frame_time: Time in the video to extract the frame from
            duration: How long to display the freeze frame

        Returns:
            A VideoClip showing a single frame for the specified duration
        """
        # Ensure frame_time is within valid bounds
        safe_time = max(0, min(frame_time, clip.duration - 0.01))
        frame_array = clip.get_frame(safe_time)
        fps = clip.fps if clip.fps else 24
        return ImageClip(frame_array).with_duration(duration).with_fps(fps)

    def process_segments(
        self,
        clip: VideoFileClip,
        processed_segments: List[ProcessedSegment],
    ) -> VideoFileClip:
        """
        Process all segments: trim, speed adjust, and concatenate.

        Handles both normal segments (extracted and speed-adjusted) and
        freeze frame segments (static image for specified duration).

        Args:
            clip: Source video clip
            processed_segments: List of segments with speed ratios or freeze frame info

        Returns:
            Concatenated video with all segments processed
        """
        segment_clips = []
        for seg in processed_segments:
            if seg.is_freeze_frame:
                # Create a static frame for this segment
                freeze = self.create_freeze_frame(
                    clip,
                    seg.freeze_frame_time,
                    seg.freeze_frame_duration,
                )
                segment_clips.append(freeze)
            else:
                # Normal segment processing
                trimmed = self.extract_segment(clip, seg.start_time, seg.end_time)
                adjusted = self.apply_speed(trimmed, seg.speed_ratio)
                segment_clips.append(adjusted)

        return concatenate_videoclips(segment_clips, method="compose")

    def extract_reference_segment(
        self,
        clip: VideoFileClip,
        start_time: float,
        end_time: float,
    ) -> VideoFileClip:
        """Extract the reference video segment (no speed adjustment)."""
        return clip.subclipped(start_time, end_time)

    def stack_vertically(
        self,
        top_clip: VideoFileClip,
        bottom_clip: VideoFileClip,
    ) -> VideoFileClip:
        """Stack two clips vertically (top over bottom)."""
        return clips_array([[top_clip], [bottom_clip]])

    def export(
        self,
        clip: VideoFileClip,
        output_path: str,
        audio_from: Optional[VideoFileClip] = None,
        fps: int = 30,
        codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> None:
        """
        Export the final video to file.

        Args:
            clip: Video clip to export
            output_path: Output file path
            audio_from: Optional clip to use audio from (typically reference)
            fps: Frames per second
            codec: Video codec
            audio_codec: Audio codec
        """
        if audio_from is not None and audio_from.audio is not None:
            clip = clip.with_audio(audio_from.audio)

        clip.write_videofile(
            output_path,
            fps=fps,
            codec=codec,
            audio_codec=audio_codec,
        )


class SyntheticClipFactory:
    """Factory for creating synthetic clips for testing."""

    @staticmethod
    def create_color_clip(
        duration: float,
        size: tuple = (320, 240),
        color: tuple = (255, 0, 0),
        fps: int = 24,
    ) -> VideoClip:
        """Create a solid color clip for testing."""
        return ColorClip(size=size, color=color, duration=duration).with_fps(fps)

    @staticmethod
    def create_gradient_clip(
        duration: float,
        size: tuple = (320, 240),
        fps: int = 24,
    ) -> VideoClip:
        """
        Create a clip with time-based color gradient for verification.

        The red channel increases with time, making it possible to verify
        speed changes by checking frame colors.
        """

        def make_frame(t):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            # Red channel increases with time
            frame[:, :, 0] = int((t / duration) * 255) if duration > 0 else 0
            return frame

        return VideoClip(make_frame, duration=duration).with_fps(fps)
