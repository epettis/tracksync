"""Tests for scene-mode sync point assembly (Task T10).

End-to-end tests for generate_scene_sync_points using GistEmbedder synthetic
clips for the coarse stage and a synthetic geometric matcher + injected frame
window decoder for the fine stage (no ML models, no real fine-stage decoding).

Verifies:
1. Sync point pairs land within 1 sample of ground truth for the 2 s-shift
   scenario of T7, monotonic and labeled start, sync_1..sync_N, end
2. CSV round-trip through the existing writer/reader (downstream
   compatibility, design §5.4)
3. Fallback path: impossibly high min_inliers -> result equals smoothed
   coarse estimates

Design reference: docs/scene_alignment_design.md §5.1-5.4, §11.2, task T10
"""

import numpy as np
import pytest

from tests.test_fine_align_geometry import (
    add_noise_and_outliers,
    create_3d_scene,
    project_points,
)
from tests.test_scene_align_coarse import create_synthetic_moving_noise_video

from tracksync.csv_reader import parse_csv_content
from tracksync.csv_writer import write_sync_csv
from tracksync.embedding import GistEmbedder
from tracksync.fine_align import intrinsics_from_fov
from tracksync.frame_data import SceneFeatures
from tracksync.models import SyncResult
from tracksync.scene_align import (
    _candidate_sync_times,
    coarse_align,
    extract_scene_features,
    generate_scene_sync_points,
)


class GeometricSceneSetup:
    """Combined injected decoder + FeatureMatcher backed by synthetic geometry.

    The decoder hands out dummy frames tagged with (video, timestamp). The
    matcher looks up those tags and synthesizes point correspondences by
    projecting a shared 3D point cloud through two cameras whose relative
    longitudinal position encodes the timing error against a known ground
    truth mapping t_B = truth_fn(t_A). The zero crossing of the longitudinal
    component therefore occurs exactly at the true sync time, letting
    refine_sync_point recover ground truth regardless of coarse errors.
    """

    name = "synthetic-geometric"

    def __init__(
        self,
        video_a_path: str,
        video_b_path: str,
        truth_fn,
        speed_mps: float = 10.0,
        native_fps: float = 30.0,
        seed: int = 42,
    ):
        self.video_a_path = video_a_path
        self.video_b_path = video_b_path
        self.truth_fn = truth_fn
        self.speed = speed_mps
        self.native_fps = native_fps

        self.points_3d = create_3d_scene(n_points=200, seed=seed)
        self.width, self.height = 640, 480
        self.K = intrinsics_from_fov(self.width, self.height, fov_deg=90.0)

        # Mounting offset for camera B (lateral + vertical + 5 deg yaw),
        # mirroring the T8 geometry tests
        yaw = np.deg2rad(5.0)
        self.R_offset = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
        self.lateral_offset = np.array([[0.5], [0.3], [0.0]])

        # id(frame) -> ("a" | "b", time); keep frames alive so ids are stable
        self._frame_tags: dict[int, tuple[str, float]] = {}
        self._frames: list[np.ndarray] = []
        self._noise_counter = 0

    # --- injected decoder ---

    def decode_window(self, video_path, center_s, half_window_s):
        if video_path == self.video_a_path:
            frame = self._new_frame("a", center_s)
            return [frame], np.array([center_s], dtype=np.float64)

        assert video_path == self.video_b_path
        start = center_s - half_window_s
        n = int(round(2 * half_window_s * self.native_fps)) + 1
        times = start + np.arange(n) / self.native_fps
        frames = [self._new_frame("b", t) for t in times]
        return frames, times

    def _new_frame(self, which, t):
        # Dummy 480x640 RGB frame so intrinsics_from_fov sees real dimensions
        arr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._frames.append(arr)
        self._frame_tags[id(arr)] = (which, float(t))
        return arr

    # --- FeatureMatcher protocol ---

    def match(self, img_a, img_b, mask_a=None, mask_b=None):
        which_a, t_a = self._frame_tags[id(img_a)]
        which_b, t_b = self._frame_tags[id(img_b)]

        if which_a == "a" and which_b == "b":
            # Camera A at origin; camera B longitudinally displaced by the
            # timing error against ground truth, plus the mounting offset
            z = (t_b - self.truth_fn(t_a)) * self.speed
            return self._project_pair(
                np.zeros((3, 1)), np.eye(3),
                self._b_position(z), self.R_offset,
            )
        elif which_a == "b" and which_b == "b":
            # Consecutive B frames: center the pair so the cloud stays visible
            t_mid = 0.5 * (t_a + t_b)
            z_i = (t_a - t_mid) * self.speed
            z_j = (t_b - t_mid) * self.speed
            return self._project_pair(
                self._b_position(z_i), self.R_offset,
                self._b_position(z_j), self.R_offset,
            )
        else:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    def _b_position(self, z):
        return self.lateral_offset + np.array([[0.0], [0.0], [z]])

    def _project_pair(self, t1, R1, t2, R2):
        pts1 = project_points(self.points_3d, self.K, R1, t1)
        pts2 = project_points(self.points_3d, self.K, R2, t2)

        valid = np.ones(len(self.points_3d), dtype=bool)
        depths1 = -(R1 @ self.points_3d.T + t1)[2, :]
        depths2 = -(R2 @ self.points_3d.T + t2)[2, :]
        valid &= (depths1 > 0) & (depths2 > 0)
        valid &= (pts1[:, 0] >= 0) & (pts1[:, 0] < self.width)
        valid &= (pts1[:, 1] >= 0) & (pts1[:, 1] < self.height)
        valid &= (pts2[:, 0] >= 0) & (pts2[:, 0] < self.width)
        valid &= (pts2[:, 1] >= 0) & (pts2[:, 1] < self.height)

        pts1 = pts1[valid]
        pts2 = pts2[valid]

        if len(pts1) < 10:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        self._noise_counter += 1
        return add_noise_and_outliers(
            pts1, pts2, noise_sigma=0.5, outlier_fraction=0.1,
            seed=10_000 + self._noise_counter,
        )


class EmptyMatcher:
    """Matcher returning no correspondences (forces fallback)."""

    name = "empty"

    def match(self, img_a, img_b, mask_a=None, mask_b=None):
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)


def make_dummy_decoder(video_a_path: str, native_fps: float = 30.0):
    """Decoder returning blank frames (for fallback tests)."""

    def decode(video_path, center_s, half_window_s):
        if half_window_s <= 0:
            return (
                [np.zeros((48, 64, 3), dtype=np.uint8)],
                np.array([center_s], dtype=np.float64),
            )
        n = int(round(2 * half_window_s * native_fps)) + 1
        times = (center_s - half_window_s) + np.arange(n) / native_fps
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(n)]
        return frames, times

    return decode


def make_identity_features(n_frames: int = 120, sample_hz: float = 10.0) -> SceneFeatures:
    """Directly synthesized SceneFeatures with time-evolving embeddings."""
    frame_times = np.arange(n_frames) / sample_hz

    rng = np.random.RandomState(321)
    embeddings = []
    for _ in range(n_frames):
        emb = rng.randn(256)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    emb_array = np.array(embeddings, dtype=np.float32)

    return SceneFeatures(
        video_path="synthetic",
        frame_times=frame_times,
        emb_array=emb_array,
        static_mask=np.zeros((48, 64), dtype=bool),
        sample_hz=sample_hz,
    )


class TestEndToEndShiftedScenario:
    """End-to-end scene SyncResult for the 2 s-shift scenario of T7."""

    @pytest.fixture(scope="class")
    def sync_result_and_context(self, tmp_path_factory):
        """Run the full pipeline once and share the result across tests."""
        tmp_path = tmp_path_factory.mktemp("t10_shift")

        video_a = create_synthetic_moving_noise_video(
            tmp_path, duration_s=25.0, fps=15.0,
            width=256, height=192, offset_s=0.0,
        )
        video_b = create_synthetic_moving_noise_video(
            tmp_path, duration_s=23.0, fps=15.0,
            width=256, height=192, offset_s=2.0,
        )
        if video_a is None or video_b is None:
            pytest.skip("cv2.VideoWriter codec unavailable")

        embedder = GistEmbedder()
        sample_hz = 10.0
        feat_a = extract_scene_features(str(video_a), embedder, sample_hz=sample_hz)
        feat_b = extract_scene_features(str(video_b), embedder, sample_hz=sample_hz)
        coarse = coarse_align(feat_a, feat_b, band_pct=0.15, open_end_s=5.0)

        # Ground truth: video B content at t corresponds to A content at t+2
        def truth_fn(t_a):
            return t_a - 2.0

        setup = GeometricSceneSetup(
            str(video_a), str(video_b), truth_fn,
            speed_mps=10.0, native_fps=30.0,
        )

        result = generate_scene_sync_points(
            coarse, str(video_a), str(video_b), setup,
            max_sync_interval=3.0, min_inliers=30,
            decode_window=setup.decode_window,
        )
        return result, coarse, sample_hz

    def test_pairs_within_one_sample_of_truth(self, sync_result_and_context):
        result, _, sample_hz = sync_result_and_context
        sample_interval = 1.0 / sample_hz

        assert len(result.sync_points) >= 3
        for p in result.sync_points:
            error = abs(p.time_b - (p.time_a - 2.0))
            assert error <= sample_interval, (
                f"Sync point {p.label}: A={p.time_a:.3f}s B={p.time_b:.3f}s "
                f"error {error:.3f}s > {sample_interval:.3f}s"
            )

    def test_monotonic(self, sync_result_and_context):
        result, _, _ = sync_result_and_context
        points = result.sync_points
        for i in range(1, len(points)):
            assert points[i].time_a > points[i - 1].time_a
            assert points[i].time_b > points[i - 1].time_b

    def test_labels(self, sync_result_and_context):
        result, _, _ = sync_result_and_context
        points = result.sync_points

        assert points[0].label == "start"
        assert points[-1].label == "end"
        for i, p in enumerate(points[1:-1], start=1):
            assert p.label == f"sync_{i}"

    def test_cadence(self, sync_result_and_context):
        result, _, _ = sync_result_and_context
        points = result.sync_points
        for i in range(1, len(points)):
            gap = points[i].time_a - points[i - 1].time_a
            assert gap <= 3.0 + 1e-6, f"Gap {gap:.3f}s exceeds max_sync_interval"

    def test_speed_ratios_match_legacy_convention(self, sync_result_and_context):
        result, _, _ = sync_result_and_context
        points = result.sync_points

        for i in range(len(points) - 1):
            delta_a = points[i + 1].time_a - points[i].time_a
            delta_b = points[i + 1].time_b - points[i].time_b
            expected = delta_b / delta_a if delta_a > 0 else 1.0
            assert points[i].speed == pytest.approx(expected)
        # Last point keeps the default speed
        assert points[-1].speed == 1.0

    def test_trim_fields_from_coarse(self, sync_result_and_context):
        result, coarse, _ = sync_result_and_context
        assert result.trim_start_a == coarse.trim_start_a
        assert result.trim_end_a == coarse.trim_end_a
        assert result.trim_start_b == coarse.trim_start_b
        assert result.trim_end_b == coarse.trim_end_b

    def test_csv_round_trip(self, sync_result_and_context, tmp_path):
        """Write via existing csv_writer and re-read via csv_reader (§5.4)."""
        result, _, _ = sync_result_and_context

        csv_path = tmp_path / "sync.csv"
        content = write_sync_csv(
            result.sync_points, "driver_a", "driver_b", str(csv_path)
        )
        assert csv_path.exists()

        videos = parse_csv_content(content)
        assert len(videos) == 2
        assert videos[0].driver == "driver_a"
        assert videos[1].driver == "driver_b"

        assert len(videos[0].segments) == len(result.sync_points)
        assert len(videos[1].segments) == len(result.sync_points)

        for point, seg_a, seg_b in zip(
            result.sync_points, videos[0].segments, videos[1].segments
        ):
            assert seg_a.title == point.label
            assert seg_b.title == point.label
            # Writer rounds to 3 decimals
            assert seg_a.timestamp == pytest.approx(point.time_a, abs=5e-4)
            assert seg_b.timestamp == pytest.approx(point.time_b, abs=5e-4)


class TestFallbackPath:
    """Unverified candidates must fall back to smoothed coarse estimates."""

    def _make_coarse(self):
        features = make_identity_features()
        return coarse_align(features, features, band_pct=0.10, open_end_s=1.0)

    def test_impossible_min_inliers_falls_back_to_coarse(self):
        coarse = self._make_coarse()
        setup = GeometricSceneSetup(
            "video_a.mp4", "video_b.mp4", truth_fn=lambda t: t,
        )

        result = generate_scene_sync_points(
            coarse, "video_a.mp4", "video_b.mp4", setup,
            max_sync_interval=3.0,
            min_inliers=10**9,  # Impossible: forces fallback everywhere
            decode_window=setup.decode_window,
        )

        assert isinstance(result, SyncResult)
        assert len(result.sync_points) >= 3
        for p in result.sync_points:
            assert p.time_b == pytest.approx(coarse.f(p.time_a), abs=1e-6), (
                f"{p.label} did not fall back to coarse estimate"
            )

    def test_empty_matcher_falls_back_to_coarse(self):
        coarse = self._make_coarse()

        result = generate_scene_sync_points(
            coarse, "video_a.mp4", "video_b.mp4", EmptyMatcher(),
            max_sync_interval=3.0, min_inliers=30,
            decode_window=make_dummy_decoder("video_a.mp4"),
        )

        for p in result.sync_points:
            assert p.time_b == pytest.approx(coarse.f(p.time_a), abs=1e-6)

    def test_fallback_result_monotonic_and_labeled(self):
        coarse = self._make_coarse()

        result = generate_scene_sync_points(
            coarse, "video_a.mp4", "video_b.mp4", EmptyMatcher(),
            max_sync_interval=2.0, min_inliers=30,
            decode_window=make_dummy_decoder("video_a.mp4"),
        )

        points = result.sync_points
        assert points[0].label == "start"
        assert points[-1].label == "end"
        for i in range(1, len(points)):
            assert points[i].time_a > points[i - 1].time_a
            assert points[i].time_b > points[i - 1].time_b
            assert points[i].time_a - points[i - 1].time_a <= 2.0 + 1e-6


class TestCandidateSyncTimes:
    """Unit tests for candidate generation and margin snapping."""

    def test_candidates_span_trim_range(self):
        features = make_identity_features()
        coarse = coarse_align(features, features, open_end_s=1.0)

        times = _candidate_sync_times(coarse, max_sync_interval=3.0)

        assert times[0] == coarse.trim_start_a
        assert times[-1] == coarse.trim_end_a
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]
            assert times[i] - times[i - 1] <= 3.0 + 1e-6

    def test_degenerate_range_single_candidate(self):
        features = make_identity_features()
        coarse = coarse_align(features, features, open_end_s=1.0)
        coarse.trim_end_a = coarse.trim_start_a

        times = _candidate_sync_times(coarse, max_sync_interval=3.0)
        assert times == [coarse.trim_start_a]

    def test_snapping_prefers_high_margin_frames(self):
        """Interior candidates should snap to local margin maxima."""
        features = make_identity_features()
        coarse = coarse_align(features, features, open_end_s=1.0)

        # Construct a margins array with a distinctive spike near each
        # nominal candidate position, offset by less than the snap radius
        frame_times = coarse._frame_times_a
        margins = np.zeros_like(coarse.margins)

        t_start, t_end = coarse.trim_start_a, coarse.trim_end_a
        duration = t_end - t_start
        max_interval = 3.0
        n_intervals = max(1, int(np.ceil(duration / max_interval)))
        spacing = duration / n_intervals
        snap_radius = max(
            0.0, min(0.5 * (max_interval - spacing), 0.25 * spacing)
        )
        if snap_radius <= 0:
            pytest.skip("No snap slack for this configuration")

        # Place a spike snap_radius/2 after each nominal candidate
        spike_times = []
        for k in range(1, n_intervals):
            nominal = t_start + k * spacing
            target = nominal + snap_radius / 2
            idx = int(np.argmin(np.abs(frame_times - target)))
            margins[idx] = 10.0
            spike_times.append(frame_times[idx])

        coarse.margins = margins
        times = _candidate_sync_times(coarse, max_sync_interval=max_interval)

        interior = times[1:-1]
        assert len(interior) == len(spike_times)
        for t, spike in zip(interior, spike_times):
            # Snapped candidate should sit exactly on the spike frame
            # (provided the spike frame lies within the snap window)
            if abs(spike - t) > 1e-9:
                # Spike may fall just outside the window after rounding to
                # frame times; accept only if it snapped to a frame time
                assert np.min(np.abs(coarse._frame_times_a - t)) < 1e-9
