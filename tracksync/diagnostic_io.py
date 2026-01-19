"""Diagnostic data export and import for tracksync.

This module provides functions to serialize and deserialize diagnostic data
to/from protocol buffer format, enabling offline analysis of cross-correlation
results without re-running the expensive video processing pipeline.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from google.protobuf import text_format

from .proto import diagnostic_pb2 as pb
from .frame_data import CrossCorrelationResult as CCResult
from .autocorrelation import FrameOCRData, StartFinishCrossing
from .models import TurnAnalysis, TurnApex


# Version string for diagnostic files
TRACKSYNC_VERSION = "2.0.0"


def _circle_to_proto(circle: Optional[tuple]) -> Optional[pb.Circle]:
    """Convert a circle tuple (x, y, radius) to protobuf Circle."""
    if circle is None:
        return None
    return pb.Circle(x=circle[0], y=circle[1], radius=circle[2])


def _proto_to_circle(circle_pb: pb.Circle) -> Optional[tuple]:
    """Convert a protobuf Circle to tuple (x, y, radius)."""
    if not circle_pb.ByteSize():
        return None
    return (circle_pb.x, circle_pb.y, circle_pb.radius)


def _ocr_to_proto(ocr: Optional[FrameOCRData]) -> Optional[pb.FrameOCRData]:
    """Convert FrameOCRData to protobuf."""
    if ocr is None:
        return None
    result = pb.FrameOCRData(is_optimal_lap=ocr.is_optimal_lap)
    if ocr.lap_number is not None:
        result.lap_number = ocr.lap_number
    if ocr.segment_number is not None:
        result.segment_number = ocr.segment_number
    if ocr.lap_time_seconds is not None:
        result.lap_time_seconds = ocr.lap_time_seconds
    return result


def _proto_to_ocr(ocr_pb: pb.FrameOCRData) -> Optional[FrameOCRData]:
    """Convert protobuf to FrameOCRData."""
    if not ocr_pb.ByteSize():
        return None
    return FrameOCRData(
        lap_number=ocr_pb.lap_number if ocr_pb.HasField('lap_number') else None,
        is_optimal_lap=ocr_pb.is_optimal_lap,
        segment_number=ocr_pb.segment_number if ocr_pb.HasField('segment_number') else None,
        lap_time_seconds=ocr_pb.lap_time_seconds if ocr_pb.HasField('lap_time_seconds') else None,
    )


def _crossing_to_proto(crossing: StartFinishCrossing) -> pb.StartFinishCrossing:
    """Convert StartFinishCrossing to protobuf."""
    result = pb.StartFinishCrossing(
        frame_index=crossing.frame_index,
        video_time=crossing.video_time,
    )
    if crossing.lap_before is not None:
        result.lap_before = crossing.lap_before
    if crossing.lap_after is not None:
        result.lap_after = crossing.lap_after
    return result


def _proto_to_crossing(crossing_pb: pb.StartFinishCrossing) -> StartFinishCrossing:
    """Convert protobuf to StartFinishCrossing."""
    return StartFinishCrossing(
        frame_index=crossing_pb.frame_index,
        video_time=crossing_pb.video_time,
        lap_before=crossing_pb.lap_before if crossing_pb.HasField('lap_before') else None,
        lap_after=crossing_pb.lap_after if crossing_pb.HasField('lap_after') else None,
    )


def _turn_apex_to_proto(apex: TurnApex) -> pb.TurnApex:
    """Convert TurnApex to protobuf."""
    return pb.TurnApex(
        time=apex.time,
        angle=apex.angle,
        sharpness=apex.sharpness,
        prominence=apex.prominence,
    )


def _proto_to_turn_apex(apex_pb: pb.TurnApex) -> TurnApex:
    """Convert protobuf to TurnApex."""
    return TurnApex(
        time=apex_pb.time,
        angle=apex_pb.angle,
        sharpness=apex_pb.sharpness,
        prominence=apex_pb.prominence,
    )


def _turn_analysis_to_proto(analysis: TurnAnalysis) -> pb.TurnAnalysis:
    """Convert TurnAnalysis to protobuf."""
    result = pb.TurnAnalysis(window_seconds=analysis.window_seconds)
    result.times.extend(analysis.times)
    result.angles.extend(analysis.angles)
    result.sharpness.extend(analysis.sharpness)
    result.apexes.extend(_turn_apex_to_proto(a) for a in analysis.apexes)
    return result


def _proto_to_turn_analysis(analysis_pb: pb.TurnAnalysis) -> TurnAnalysis:
    """Convert protobuf to TurnAnalysis."""
    return TurnAnalysis(
        times=list(analysis_pb.times),
        angles=list(analysis_pb.angles),
        sharpness=list(analysis_pb.sharpness),
        apexes=[_proto_to_turn_apex(a) for a in analysis_pb.apexes],
        window_seconds=analysis_pb.window_seconds,
    )


def _cc_result_to_proto(result: CCResult) -> pb.CrossCorrelationResult:
    """Convert CrossCorrelationResult to protobuf."""
    pb_result = pb.CrossCorrelationResult(
        time_a=result.time_a,
        circle_a_interpolated=result.circle_a_interpolated,
        circle_b_interpolated=result.circle_b_interpolated,
        no_match=result.no_match,
    )

    # Optional circle_a
    if result.circle_a is not None:
        pb_result.circle_a.CopyFrom(_circle_to_proto(result.circle_a))

    # Optional segment_a
    if result.segment_a is not None:
        pb_result.segment_a = result.segment_a

    # Optional best match fields
    if result.best_time_b is not None:
        pb_result.best_time_b = result.best_time_b
    if result.best_circle_b is not None:
        pb_result.best_circle_b.CopyFrom(_circle_to_proto(result.best_circle_b))
    if result.best_segment_b is not None:
        pb_result.best_segment_b = result.best_segment_b
    if result.best_distance is not None:
        pb_result.best_distance = result.best_distance

    # All distances
    for time_b, distance in result.all_distances:
        pb_result.all_distances.append(
            pb.DistancePoint(time_b=time_b, distance=distance)
        )

    # Global min fields
    if result.global_min_time_b is not None:
        pb_result.global_min_time_b = result.global_min_time_b
    if result.global_min_distance is not None:
        pb_result.global_min_distance = result.global_min_distance

    # OCR data
    if result.ocr_a is not None:
        pb_result.ocr_a.CopyFrom(_ocr_to_proto(result.ocr_a))
    if result.ocr_b is not None:
        pb_result.ocr_b.CopyFrom(_ocr_to_proto(result.ocr_b))

    return pb_result


def _proto_to_cc_result(pb_result: pb.CrossCorrelationResult) -> CCResult:
    """Convert protobuf to CrossCorrelationResult.

    Note: frame_a, best_frame_b, red_mask_a, red_mask_b are not stored
    in the protobuf and will be None. These are loaded on-demand from
    the video files during visualization.
    """
    return CCResult(
        time_a=pb_result.time_a,
        circle_a=_proto_to_circle(pb_result.circle_a) if pb_result.HasField('circle_a') else None,
        segment_a=pb_result.segment_a if pb_result.HasField('segment_a') else None,
        best_time_b=pb_result.best_time_b if pb_result.HasField('best_time_b') else None,
        best_circle_b=_proto_to_circle(pb_result.best_circle_b) if pb_result.HasField('best_circle_b') else None,
        best_segment_b=pb_result.best_segment_b if pb_result.HasField('best_segment_b') else None,
        best_distance=pb_result.best_distance if pb_result.HasField('best_distance') else None,
        all_distances=[(dp.time_b, dp.distance) for dp in pb_result.all_distances],
        frame_a=None,  # Not stored in protobuf
        best_frame_b=None,  # Not stored in protobuf
        red_mask_a=None,  # Not stored in protobuf
        red_mask_b=None,  # Not stored in protobuf
        circle_a_interpolated=pb_result.circle_a_interpolated,
        circle_b_interpolated=pb_result.circle_b_interpolated,
        no_match=pb_result.no_match,
        global_min_time_b=pb_result.global_min_time_b if pb_result.HasField('global_min_time_b') else None,
        global_min_distance=pb_result.global_min_distance if pb_result.HasField('global_min_distance') else None,
        ocr_a=_proto_to_ocr(pb_result.ocr_a) if pb_result.HasField('ocr_a') else None,
        ocr_b=_proto_to_ocr(pb_result.ocr_b) if pb_result.HasField('ocr_b') else None,
    )


def export_diagnostic(
    results: List[CCResult],
    turn_analysis_a: TurnAnalysis,
    turn_analysis_b: TurnAnalysis,
    video_a_path: str,
    video_b_path: str,
    fps_a: float,
    fps_b: float,
    duration_a: float,
    duration_b: float,
    crossings_a: List[StartFinishCrossing],
    crossings_b: List[StartFinishCrossing],
    output_path: str,
) -> None:
    """
    Export diagnostic data to a protobuf text file.

    Args:
        results: List of CrossCorrelationResult objects
        turn_analysis_a: TurnAnalysis for video A
        turn_analysis_b: TurnAnalysis for video B
        video_a_path: Path to video A
        video_b_path: Path to video B
        fps_a: Frame rate of video A
        fps_b: Frame rate of video B
        duration_a: Duration of video A in seconds
        duration_b: Duration of video B in seconds
        crossings_a: Start-finish crossings for video A
        crossings_b: Start-finish crossings for video B
        output_path: Path to write the protobuf text file
    """
    # Build the protobuf message
    diagnostic = pb.DiagnosticData(
        tracksync_version=TRACKSYNC_VERSION,
        created_timestamp=int(time.time()),
    )

    # Video A info
    driver_a = os.path.splitext(os.path.basename(video_a_path))[0]
    diagnostic.video_a.CopyFrom(pb.VideoInfo(
        video_path=video_a_path,
        driver_name=driver_a,
        fps=fps_a,
        duration=duration_a,
    ))
    diagnostic.video_a.crossings.extend(
        _crossing_to_proto(c) for c in crossings_a
    )

    # Video B info
    driver_b = os.path.splitext(os.path.basename(video_b_path))[0]
    diagnostic.video_b.CopyFrom(pb.VideoInfo(
        video_path=video_b_path,
        driver_name=driver_b,
        fps=fps_b,
        duration=duration_b,
    ))
    diagnostic.video_b.crossings.extend(
        _crossing_to_proto(c) for c in crossings_b
    )

    # Cross-correlation results
    diagnostic.results.extend(_cc_result_to_proto(r) for r in results)

    # Turn analysis
    diagnostic.turn_analysis_a.CopyFrom(_turn_analysis_to_proto(turn_analysis_a))
    diagnostic.turn_analysis_b.CopyFrom(_turn_analysis_to_proto(turn_analysis_b))

    # Write as text format
    text_content = text_format.MessageToString(diagnostic)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(text_content)


def import_diagnostic(input_path: str) -> pb.DiagnosticData:
    """
    Import diagnostic data from a protobuf text file.

    Args:
        input_path: Path to the protobuf text file

    Returns:
        DiagnosticData protobuf message
    """
    with open(input_path, 'r') as f:
        text_content = f.read()

    diagnostic = pb.DiagnosticData()
    text_format.Parse(text_content, diagnostic)
    return diagnostic


def get_results_from_diagnostic(diagnostic: pb.DiagnosticData) -> List[CCResult]:
    """
    Extract CrossCorrelationResult objects from diagnostic data.

    Args:
        diagnostic: DiagnosticData protobuf message

    Returns:
        List of CrossCorrelationResult objects (without frame data)
    """
    return [_proto_to_cc_result(r) for r in diagnostic.results]


def get_turn_analysis_from_diagnostic(
    diagnostic: pb.DiagnosticData
) -> tuple[TurnAnalysis, TurnAnalysis]:
    """
    Extract TurnAnalysis objects from diagnostic data.

    Args:
        diagnostic: DiagnosticData protobuf message

    Returns:
        Tuple of (turn_analysis_a, turn_analysis_b)
    """
    return (
        _proto_to_turn_analysis(diagnostic.turn_analysis_a),
        _proto_to_turn_analysis(diagnostic.turn_analysis_b),
    )


def get_crossings_from_diagnostic(
    diagnostic: pb.DiagnosticData
) -> tuple[List[StartFinishCrossing], List[StartFinishCrossing]]:
    """
    Extract StartFinishCrossing objects from diagnostic data.

    Args:
        diagnostic: DiagnosticData protobuf message

    Returns:
        Tuple of (crossings_a, crossings_b)
    """
    crossings_a = [_proto_to_crossing(c) for c in diagnostic.video_a.crossings]
    crossings_b = [_proto_to_crossing(c) for c in diagnostic.video_b.crossings]
    return crossings_a, crossings_b
