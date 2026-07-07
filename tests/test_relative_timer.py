"""Tests for the iRacing-style relative timer overlay."""

import numpy as np
import pytest

from tracksync.relative_timer import (
    draw_relative_timer,
    delta_series_from_sync_points,
)


# Top lap runs 0..12 (start=0); bottom lap reaches matched positions faster
# early (ahead) then falls behind by the end.
TOP = [0.0, 3.0, 6.0, 9.0, 12.0]
BOTTOM = [0.0, 2.5, 5.0, 8.5, 12.5]


class TestDeltaSeries:
    def test_zero_at_start(self):
        deltas, _ = delta_series_from_sync_points(TOP, BOTTOM, [0.0])
        assert deltas[0] == pytest.approx(0.0)

    def test_sign_convention_negative_is_ahead(self):
        # Output t=2.5 -> bottom elapsed 2.5, top elapsed interp(2.5) over
        # (bottom->top) = 3.0, so delta = 2.5 - 3.0 = -0.5 (bottom ahead).
        deltas, _ = delta_series_from_sync_points(TOP, BOTTOM, [2.5])
        assert deltas[0] == pytest.approx(-0.5)

    def test_positive_is_behind(self):
        # Near the end the bottom lap is slower: at output t=12.0 the top lap's
        # elapsed time is < 12.0, so delta is positive (behind).
        deltas, _ = delta_series_from_sync_points(TOP, BOTTOM, [12.0])
        assert deltas[0] > 0.0

    def test_linear_interpolation_matches_constant_speed_warp(self):
        # Midway through the first segment (bottom 1.25) maps linearly to top
        # 1.5, so elapsed_top=1.5, delta = 1.25 - 1.5 = -0.25.
        deltas, _ = delta_series_from_sync_points(TOP, BOTTOM, [1.25])
        assert deltas[0] == pytest.approx(-0.25)

    def test_clamp_is_max_abs_delta(self):
        times = np.linspace(0, 12.5, 200)
        deltas, clamp = delta_series_from_sync_points(TOP, BOTTOM, times)
        assert clamp == pytest.approx(max(abs(min(deltas)), abs(max(deltas))))

    def test_clamp_has_floor_for_flat_lap(self):
        # Identical laps -> all-zero deltas, but clamp stays positive so the
        # bar still renders without a divide-by-zero.
        _, clamp = delta_series_from_sync_points(TOP, TOP, [0.0, 3.0, 6.0])
        assert clamp > 0.0


class TestDrawRelativeTimer:
    def _frame(self, w=640, h=480):
        return np.full((h, w, 3), 40, dtype=np.uint8)

    def test_returns_same_shape_and_dtype(self):
        frame = self._frame()
        out = draw_relative_timer(frame, -0.3, 1.0)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_does_not_mutate_input(self):
        frame = self._frame()
        before = frame.copy()
        draw_relative_timer(frame, 0.5, 1.0)
        assert np.array_equal(frame, before)

    def test_overlay_changes_pixels_near_seam(self):
        frame = self._frame()
        out = draw_relative_timer(frame, 0.5, 1.0)
        seam = frame.shape[0] // 2
        band = out[seam - 60:seam + 60]
        assert not np.array_equal(band, frame[seam - 60:seam + 60])

    def test_ahead_paints_green_side_rgb(self):
        # Negative delta -> green bar to the RIGHT of center. In an RGB frame
        # the green channel should dominate on the right of the seam.
        frame = self._frame()
        out = draw_relative_timer(frame, -0.9, 1.0, frame_is_bgr=False)
        h, w = frame.shape[:2]
        right = out[h // 2 - 10:h // 2 + 30, w // 2 + 40:w // 2 + 120]
        r, g, b = right[..., 0].mean(), right[..., 1].mean(), right[..., 2].mean()
        assert g > r and g > b

    def test_bgr_flag_swaps_channels(self):
        # The same delta rendered as BGR vs RGB must differ (channel order).
        frame = self._frame()
        rgb = draw_relative_timer(frame, -0.9, 1.0, frame_is_bgr=False)
        bgr = draw_relative_timer(frame, -0.9, 1.0, frame_is_bgr=True)
        assert not np.array_equal(rgb, bgr)

    def test_handles_zero_clamp_without_error(self):
        frame = self._frame()
        out = draw_relative_timer(frame, 0.0, 0.0)
        assert out.shape == frame.shape

    def test_number_renders_when_no_system_font_resolves(self):
        # Regression: when none of the candidate font paths exist, the fallback
        # must still draw a readable (scaled) number, not a ~10px speck. We
        # detect the yellow number glyphs directly in the widget bitmap.
        import tracksync.relative_timer as rt

        def _yellow_px(delta):
            rt._font_cache.clear()
            widget = np.asarray(rt._make_widget(delta, 1.0))
            m = (
                (widget[..., 0] > 200)
                & (widget[..., 1] > 150)
                & (widget[..., 2] < 120)
                & (widget[..., 3] > 128)
            )
            return int(m.sum())

        saved = (rt._FONT_CANDIDATES_BOLD, rt._FONT_CANDIDATES_REGULAR)
        try:
            rt._FONT_CANDIDATES_BOLD = ()
            rt._FONT_CANDIDATES_REGULAR = ()
            fallback_px = _yellow_px(0.34)
        finally:
            rt._FONT_CANDIDATES_BOLD, rt._FONT_CANDIDATES_REGULAR = saved
            rt._font_cache.clear()
        # A real, readable number is hundreds+ of pixels; the old tiny default
        # produced only a couple dozen.
        assert fallback_px > 500


class TestFontResolution:
    def test_segoe_ui_present_for_both_weights(self):
        import tracksync.relative_timer as rt

        bold = [p for p, _ in rt._FONT_CANDIDATES_BOLD]
        regular = [p for p, _ in rt._FONT_CANDIDATES_REGULAR]
        assert any(p.endswith("segoeuib.ttf") for p in bold)
        assert any(p.endswith("segoeui.ttf") for p in regular)
        assert any(p.endswith("arialbd.ttf") for p in bold)

    def test_windows_font_dir_respects_windir_env(self, monkeypatch):
        import importlib
        import tracksync.relative_timer as rt

        monkeypatch.setenv("WINDIR", r"D:\CustomWin")
        importlib.reload(rt)
        try:
            assert rt._WIN_FONTS.startswith(r"D:\CustomWin")
            assert any(r"D:\CustomWin" in p for p, _ in rt._FONT_CANDIDATES_BOLD)
        finally:
            monkeypatch.delenv("WINDIR", raising=False)
            importlib.reload(rt)


class TestVideoProcessorWiring:
    def test_overlay_applies_to_clip(self):
        from tracksync.video_processor import (
            VideoProcessor,
            SyntheticClipFactory,
        )

        clip = SyntheticClipFactory.create_color_clip(
            duration=2.0, size=(640, 480), color=(40, 40, 40), fps=30
        )
        out = VideoProcessor().overlay_relative_timer(
            clip, [0.0, 1.0, 2.0], [0.0, 0.8, 2.0], fps=30
        )
        seam = 240
        base = clip.get_frame(1.0)
        overlaid = out.get_frame(1.0)
        assert overlaid.shape == (480, 640, 3)
        # A non-zero delta at t=1.0 must have painted the seam region.
        assert not np.array_equal(
            overlaid[seam - 40:seam + 40], base[seam - 40:seam + 40]
        )
        assert out.duration == pytest.approx(clip.duration)
