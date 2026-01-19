"""Tracksync - Compare racing dashcam videos by syncing timestamps."""

from .models import (
    Segment,
    VideoMetadata,
    ProcessedSegment,
    SyncPoint,
    SyncResult,
    TurnApex,
    TurnAnalysis,
)
from .speed_calculator import calculate_speed_ratios, build_processed_segments
from .csv_reader import read_csv, parse_csv_content
from .csv_writer import format_csv, write_csv, write_sync_csv
from .video_processor import VideoProcessor
from .segment_validator import (
    ClampedSegment,
    clamp_segments,
    build_processed_segments_with_clamping,
)
from .frame_data import FrameData, VideoFeatures, CrossCorrelationResult
from .feature_extraction import (
    extract_video_features,
    extract_frame_data,
    interpolate_missing_circles,
    compute_red_mask,
)
from .cross_correlation import (
    compute_cross_correlations,
    compute_cross_correlations_from_features,
    generate_sync_points,
    generate_pairwise_sync_from_features,
    output_tracksync_csv,
)
from .turn_analysis import calculate_interior_angle, compute_turn_analysis
from .frame_analysis import (
    FrameOCRData,
    StartFinishCrossing,
    extract_frame_ocr,
    interpolate_ocr_data,
    find_start_finish_crossings,
    find_red_circle,
    detect_segment_number,
    get_frame_at_time,
    get_video_info,
    binarize_frame,
)

__all__ = [
    # Models
    "Segment",
    "VideoMetadata",
    "ProcessedSegment",
    "SyncPoint",
    "SyncResult",
    "TurnApex",
    "TurnAnalysis",
    # Speed calculation
    "calculate_speed_ratios",
    "build_processed_segments",
    # CSV
    "read_csv",
    "parse_csv_content",
    "format_csv",
    "write_csv",
    "write_sync_csv",
    # Video processing
    "VideoProcessor",
    # Segment validation
    "ClampedSegment",
    "clamp_segments",
    "build_processed_segments_with_clamping",
    # Frame data
    "FrameData",
    "VideoFeatures",
    "CrossCorrelationResult",
    # Feature extraction
    "extract_video_features",
    "extract_frame_data",
    "interpolate_missing_circles",
    "compute_red_mask",
    # Cross-correlation
    "compute_cross_correlations",
    "compute_cross_correlations_from_features",
    "generate_sync_points",
    "generate_pairwise_sync_from_features",
    "output_tracksync_csv",
    # Turn analysis
    "calculate_interior_angle",
    "compute_turn_analysis",
    # OCR
    "FrameOCRData",
    "StartFinishCrossing",
    "extract_frame_ocr",
    "interpolate_ocr_data",
    "find_start_finish_crossings",
    "find_red_circle",
    "detect_segment_number",
    "get_frame_at_time",
    "get_video_info",
    "binarize_frame",
]
