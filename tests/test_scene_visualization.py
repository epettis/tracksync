"""Tests for scene-mode debug visualization (T12).

Covers the pure panel render functions (cost-matrix heatmap with DTW path
overlay, frame-pair panel with correspondences, confidence trace), the
composite display, debug CLI flag validation/dispatch, and a headless run
of the interactive loop with display calls monkeypatched.

Task reference: docs/scene_alignment_tasks.md T12
Design reference: docs/scene_alignment_design.md §4.5, §6
"""

import argparse

import cv2
import numpy as np
import pytest

from tracksync import cli
from tracksync.cli import create_parser, main, validate_debug_args
from tracksync.frame_data import SceneFeatures
from tracksync.visualization import (
    cost_cell_to_pixel,
    create_scene_debug_display,
    render_cost_matrix_panel,
    render_frame_pair_panel,
    render_margin_panel,
)

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 100, 255)


def _has_color(img: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(img == np.array(color, dtype=np.uint8), axis=-1)))


class TestCostMatrixPanel:
    """Tests for render_cost_matrix_panel."""

    def test_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        cost = rng.uniform(0, 2, (20, 30))
        path = np.array([[i, int(i * 29 / 19)] for i in range(20)])

        panel = render_cost_matrix_panel(cost, path, width=400, height=300)

        assert panel.shape == (300, 400, 3)
        assert panel.dtype == np.uint8

    def test_path_pixels_land_on_expected_cells(self):
        """Path overlay must be green at the mapped cell centers."""
        n = 8
        cost = np.ones((n, n))  # Constant cost: uniform heatmap background
        path = np.array([[k, k] for k in range(n)])
        width, height = 400, 300

        panel = render_cost_matrix_panel(cost, path, width, height)

        for i, j in path:
            x, y = cost_cell_to_pixel(int(i), int(j), n, n, width, height)
            assert tuple(panel[y, x]) == GREEN, (
                f"Cell ({i}, {j}) at pixel ({x}, {y}) is {tuple(panel[y, x])},"
                f" expected green"
            )

    def test_empty_path_renders(self):
        cost = np.random.default_rng(1).uniform(0, 2, (10, 10))
        panel = render_cost_matrix_panel(
            cost, np.zeros((0, 2), dtype=int), width=300, height=200
        )
        assert panel.shape == (200, 300, 3)
        assert not _has_color(panel, GREEN)

    def test_cursor_column_drawn(self):
        cost = np.ones((10, 10))
        path = np.zeros((0, 2), dtype=int)

        without = render_cost_matrix_panel(cost, path, 300, 200)
        with_cursor = render_cost_matrix_panel(
            cost, path, 300, 200, cursor_idx_a=5
        )

        assert not np.array_equal(without, with_cursor)
        assert _has_color(with_cursor, ORANGE)


class TestFramePairPanel:
    """Tests for render_frame_pair_panel."""

    def test_shape_with_frames(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        panel = render_frame_pair_panel(frame, frame, 1.0, 2.0)
        assert panel.shape == (240, 640, 3)
        assert panel.dtype == np.uint8

    def test_placeholder_when_frames_missing(self):
        panel = render_frame_pair_panel(None, None, 0.0, 0.0)
        assert panel.shape == (480, 1280, 3)
        assert panel.dtype == np.uint8

    def test_downscales_large_frames(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        panel = render_frame_pair_panel(frame, frame, 0.0, 0.0)
        assert panel.shape == (360, 1280, 3)  # Scaled to 640 per frame

    def test_correspondences_drawn(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        pts_a = np.array([[100.0, 200.0]])
        pts_b = np.array([[50.0, 100.0]])

        panel = render_frame_pair_panel(
            frame, frame, 0.0, 0.0, pts_a=pts_a, pts_b=pts_b
        )

        # Green keypoint markers at the A point and the offset B point
        assert tuple(panel[200, 100]) == GREEN
        assert tuple(panel[100, 320 + 50]) == GREEN
        # Connecting line color present somewhere
        assert _has_color(panel, (255, 255, 0))

    def test_no_correspondences_no_markers(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        panel = render_frame_pair_panel(frame, frame, 0.0, 0.0)
        assert not _has_color(panel, GREEN)


class TestMarginPanel:
    """Tests for render_margin_panel."""

    def test_shape_and_dtype(self):
        times = np.linspace(0, 10, 50)
        margins = np.random.default_rng(2).uniform(0, 1, 50)
        panel = render_margin_panel(times, margins, width=400, height=120)
        assert panel.shape == (120, 400, 3)
        assert panel.dtype == np.uint8

    def test_empty_data_renders_placeholder(self):
        panel = render_margin_panel(
            np.array([]), np.array([]), width=300, height=100
        )
        assert panel.shape == (100, 300, 3)

    def test_sync_markers_drawn(self):
        times = np.linspace(0, 10, 50)
        margins = np.full(50, 0.5)
        panel = render_margin_panel(
            times, margins, width=400, height=120, sync_times=[2.0, 8.0]
        )
        assert _has_color(panel, YELLOW)

    def test_cursor_drawn(self):
        times = np.linspace(0, 10, 50)
        margins = np.full(50, 0.5)
        panel = render_margin_panel(
            times, margins, width=400, height=120, cursor_time=5.0
        )
        assert _has_color(panel, ORANGE)

    def test_flat_margins_no_crash(self):
        """Constant margins (zero range) must not divide by zero."""
        times = np.linspace(0, 5, 10)
        margins = np.zeros(10)
        panel = render_margin_panel(times, margins, width=300, height=100)
        assert panel.shape == (100, 300, 3)


class TestSceneDebugDisplay:
    """Tests for the composite create_scene_debug_display."""

    def test_composition_shape(self):
        n = 12
        cost = np.random.default_rng(3).uniform(0, 2, (n, n))
        path = np.array([[k, k] for k in range(n)])
        times = np.arange(n) / 2.0
        margins = np.random.default_rng(4).uniform(0, 1, n)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        display = create_scene_debug_display(
            cost=cost, path=path, frame_times=times, margins=margins,
            frame_a=frame, frame_b=frame, time_a=1.0, time_b=1.5,
            cursor_idx=2, total=n, sync_times=[0.0, 5.5],
        )

        assert display.dtype == np.uint8
        assert display.ndim == 3 and display.shape[2] == 3
        # Pair panel (240) + cost (300) + margins (120) + info bar (40)
        assert display.shape == (240 + 300 + 120 + 40, 640, 3)

    def test_composition_with_missing_frames(self):
        n = 5
        cost = np.ones((n, n))
        path = np.array([[k, k] for k in range(n)])
        times = np.arange(n, dtype=float)
        margins = np.ones(n)

        display = create_scene_debug_display(
            cost=cost, path=path, frame_times=times, margins=margins,
            frame_a=None, frame_b=None, time_a=0.0, time_b=0.0,
            cursor_idx=0, total=n,
        )
        assert display.shape == (480 + 300 + 120 + 40, 1280, 3)


class TestDebugArgsValidation:
    """Tests for debug-subcommand flag parsing and validation."""

    def _parse(self, extra):
        parser = create_parser()
        return parser.parse_args(['debug', 'a.mp4', 'b.mp4'] + extra)

    def test_mode_defaults_to_scene(self):
        args = self._parse([])
        assert args.mode == 'scene'
        assert args.sample_hz is None
        assert args.band_pct is None
        assert args.embedder is None
        assert args.matcher is None

    @pytest.mark.parametrize('flag_args', [
        ['--sample-hz', '10'],
        ['--band-pct', '0.1'],
        ['--embedder', 'gist'],
        ['--matcher', 'aliked-lightglue'],
    ])
    def test_catalyst_rejects_scene_only_flags(self, flag_args, capsys):
        args = self._parse(['--mode', 'catalyst'] + flag_args)
        with pytest.raises(SystemExit) as exc_info:
            validate_debug_args(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert flag_args[0] in err
        assert '--mode scene' in err

    def test_scene_mode_fills_defaults_matcher_stays_off(self):
        args = self._parse(['--mode', 'scene'])
        validate_debug_args(args)
        assert args.sample_hz == 10.0
        assert args.band_pct == 0.10
        assert args.embedder == 'dinov2-vitb14'
        assert args.matcher is None  # Correspondences are opt-in

    def test_catalyst_without_scene_flags_is_noop(self):
        args = self._parse(['--mode', 'catalyst'])
        validate_debug_args(args)  # Must not raise
        assert args.mode == 'catalyst'


class TestDebugDispatch:
    """Tests for main() routing of the debug subcommand."""

    def test_catalyst_dispatches_run_debug_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, 'run_debug_mode', lambda args: calls.append(args))
        monkeypatch.setattr(
            cli, 'run_scene_debug_mode',
            lambda args: pytest.fail("scene debug taken in catalyst mode"),
        )

        main(['debug', 'a.mp4', 'b.mp4', '--mode', 'catalyst'])

        assert len(calls) == 1
        assert calls[0].mode == 'catalyst'

    def test_scene_dispatches_run_scene_debug_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli, 'run_debug_mode',
            lambda args: pytest.fail("catalyst debug taken in scene mode"),
        )
        monkeypatch.setattr(
            cli, 'run_scene_debug_mode', lambda args: calls.append(args)
        )

        main(['debug', 'a.mp4', 'b.mp4', '--mode', 'scene'])

        assert len(calls) == 1
        assert calls[0].embedder == 'dinov2-vitb14'


class TestSceneDebugLoopHeadless:
    """Headless run of the interactive loop with display calls mocked."""

    def test_loop_renders_and_exits_on_esc(self, monkeypatch):
        n = 10
        sample_hz = 2.0
        rng = np.random.default_rng(7)
        emb = rng.normal(size=(n, 16))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        features = SceneFeatures(
            video_path='a.mp4',
            frame_times=np.arange(n) / sample_hz,
            emb_array=emb.astype(np.float32),
            static_mask=np.zeros((24, 32), dtype=bool),
            sample_hz=sample_hz,
        )

        monkeypatch.setattr(
            cli, 'extract_scene_features',
            lambda video_path, embedder, sample_hz=10.0: features,
        )

        dummy_frame = np.zeros((48, 64, 3), dtype=np.uint8)
        monkeypatch.setattr(
            cli, 'decode_native_window',
            lambda path, t, half: ([dummy_frame], np.array([t])),
        )

        shown = []
        monkeypatch.setattr(
            cv2, 'imshow', lambda name, img: shown.append((name, img))
        )
        monkeypatch.setattr(cv2, 'waitKey', lambda ms: 27)  # ESC immediately
        monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)

        args = argparse.Namespace(
            video_a='a.mp4', video_b='b.mp4', mode='scene',
            sample_hz=sample_hz, band_pct=0.10,
            embedder='gist', matcher=None,
        )

        cli.run_scene_debug_mode(args)

        assert len(shown) == 1
        name, img = shown[0]
        assert name == "Scene Alignment Debug"
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3
