"""iRacing-style relative timer overlay for stacked comparison videos.

The comparison video stacks two dashcam laps vertically (top over bottom) that
have been synchronized so both are at the same track position at every output
frame. This overlay draws a horizontal bar, centered on the seam between the
two videos, showing the *relative lap-time gap* at the current track position:

    GREEN bar growing RIGHT  -> the compared car is AHEAD of the reference lap
    RED   bar growing LEFT   -> the compared car is BEHIND the reference lap
    yellow number above       -> the gap in seconds (2 decimals)

The zero line is a bright center tick. The bar is clamped to +/- ``clamp``
seconds; pass the lap's maximum absolute delta as ``clamp`` so the bar uses its
full travel across the lap.

The rendering core, :func:`draw_relative_timer`, takes a single BGR frame and a
signed delta (in seconds, negative = ahead) and returns the annotated frame.
:func:`delta_series_from_sync_points` derives the per-output-frame delta curve
and the recommended clamp from a synchronized sync-point table.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Layout constants (tuned against 1920-wide frames; scaled by frame width).
# ---------------------------------------------------------------------------
_REF_W = 1920.0
_PANEL_W, _PANEL_H = 660, 172
_PAD = 30
_BAR_HALF = 250
_BAR_H = 44
_BAR_TOP = 94          # y of bar within the panel content
_SCALE_TICKS = (0.25, 0.5, 0.75)  # fractions of clamp to mark faintly

_GREEN = (60, 210, 90)
_RED = (255, 96, 88)
_YELLOW = (255, 216, 24)

# Windows keeps its fonts under %WINDIR%\Fonts (normally C:\Windows\Fonts, but
# resolved from the env var so a non-standard install root still works).
_WIN_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# Candidate faces per weight, each as (path, face_index). Spans macOS (incl. the
# Supplemental/ location fonts moved to on newer versions), Windows (Segoe UI --
# the Windows 11 default -- then Arial), and Linux, so the timer text renders
# wherever the pipeline runs. Fonts are resolved lazily and cached by (size, bold).
_FONT_CANDIDATES_BOLD = (
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),   # .ttc index 1 = bold face
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    (os.path.join(_WIN_FONTS, "segoeuib.ttf"), 0),
    (os.path.join(_WIN_FONTS, "arialbd.ttf"), 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
)
_FONT_CANDIDATES_REGULAR = (
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    (os.path.join(_WIN_FONTS, "segoeui.ttf"), 0),
    (os.path.join(_WIN_FONTS, "arial.ttf"), 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
)
_font_cache: dict = {}


def _font(size: int, bold: bool):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path, index in candidates:
        try:
            f = ImageFont.truetype(path, size, index=index)
            _font_cache[key] = f
            return f
        except (OSError, IOError):
            continue
    # Last resort: Pillow's bundled font, shipped inside the wheel on every OS.
    # Pass the size so it stays readable -- a bare load_default() returns a
    # fixed ~10px bitmap, which made the relative-timer number invisible when no
    # system font resolved (as happened on Windows before Windows paths existed).
    try:
        f = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 has no size argument
        f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def _fill_bar(canvas: Image.Image, cx: int, top: int, length: int,
              color: Tuple[int, int, int], to_right: bool) -> None:
    if length < 2:
        return
    x0 = int(cx if to_right else cx - length)
    strip = Image.new("RGBA", (int(length), _BAR_H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(strip)
    sd.rounded_rectangle([0, 0, length - 1, _BAR_H - 1], radius=_BAR_H // 2,
                         fill=color + (255,))
    for y in range(_BAR_H // 2):                       # subtle top highlight
        a = int(75 * (1 - y / (_BAR_H / 2)))
        sd.line([(0, y), (length, y)], fill=(255, 255, 255, a))
    canvas.paste(strip, (x0, top), strip)


def _make_widget(delta: float, clamp: float) -> Image.Image:
    """Render the timer panel (with drop shadow) as an RGBA image."""
    ox = oy = _PAD
    im = Image.new("RGBA", (_PANEL_W + 2 * _PAD, _PANEL_H + 2 * _PAD), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        [ox, oy + 7, ox + _PANEL_W - 1, oy + _PANEL_H - 1 + 7], radius=22,
        fill=(0, 0, 0, 160))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(13)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([ox, oy, ox + _PANEL_W - 1, oy + _PANEL_H - 1], radius=20,
                        fill=(10, 12, 16, 247), outline=(255, 255, 255, 85), width=2)

    cx = ox + _PANEL_W // 2
    bar_top = oy + _BAR_TOP
    d.rounded_rectangle([cx - _BAR_HALF, bar_top, cx + _BAR_HALF, bar_top + _BAR_H - 1],
                        radius=_BAR_H // 2, fill=(48, 52, 60, 255))
    for frac in _SCALE_TICKS:
        off = int(_BAR_HALF * frac)
        for sx in (cx - off, cx + off):
            d.line([(sx, bar_top + 7), (sx, bar_top + _BAR_H - 8)],
                   fill=(255, 255, 255, 70), width=2)

    mag = min(abs(delta) / clamp, 1.0) if clamp > 1e-6 else 0.0
    length = int(_BAR_HALF * mag)
    if delta > 0.005:
        _fill_bar(im, cx, bar_top, length, _RED, to_right=False)
    elif delta < -0.005:
        _fill_bar(im, cx, bar_top, length, _GREEN, to_right=True)
    d = ImageDraw.Draw(im)
    d.line([(cx, bar_top - 7), (cx, bar_top + _BAR_H + 6)], fill=(255, 255, 255, 245), width=4)

    f_num = _font(80, bold=True)
    # Signed gap: '-' = ahead (less elapsed time), '+' = behind. Snap a hair
    # either side of zero to a clean +0.00 so it doesn't flicker "-0.00".
    disp = 0.0 if abs(delta) < 0.005 else delta
    txt = f"{disp:+.2f}"
    tb = d.textbbox((0, 0), txt, font=f_num)
    d.text((cx - (tb[2] - tb[0]) / 2, oy + 6 - tb[1]), txt, font=f_num, fill=_YELLOW + (255,))
    f_cap = _font(26, bold=False)
    for side, lab, col, anc in ((cx - _BAR_HALF, "BEHIND", _RED, "ls"),
                                (cx + _BAR_HALF, "AHEAD", _GREEN, "rs")):
        d.text((side, bar_top - 8), lab, font=f_cap, fill=col + (175,), anchor=anc)
    return im


def draw_relative_timer(frame_bgr: np.ndarray, delta: float, clamp: float,
                        seam_y: int | None = None,
                        frame_is_bgr: bool = True) -> np.ndarray:
    """Composite the relative timer onto a stacked frame (in place-safe).

    Args:
        frame_bgr: HxWx3 uint8 frame of the stacked comparison video.
        delta: signed gap in seconds. Negative = compared car ahead (green,
            right); positive = behind (red, left).
        clamp: bar reaches full travel at this absolute delta (seconds).
            Typically the lap's maximum absolute delta.
        seam_y: y-coordinate of the seam between the two videos. Defaults to
            the vertical midpoint of the frame.
        frame_is_bgr: True if the frame is BGR (OpenCV); False if RGB (MoviePy).
            The widget is authored in RGB and its channels are swapped to match.

    Returns:
        A new frame (same channel order as the input) with the overlay burned in.
    """
    h, w = frame_bgr.shape[:2]
    if seam_y is None:
        seam_y = h // 2
    scale = w / _REF_W

    widget = _make_widget(delta, clamp)
    if scale != 1.0:
        widget = widget.resize((max(1, round(widget.width * scale)),
                                max(1, round(widget.height * scale))),
                               Image.LANCZOS)
    ww, wh = widget.size
    x0 = w // 2 - ww // 2
    y0 = int(seam_y) - wh // 2

    # Clip the paste region to the frame bounds.
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(w, x0 + ww), min(h, y0 + wh)
    if fx0 >= fx1 or fy0 >= fy1:
        return frame_bgr
    wx0, wy0 = fx0 - x0, fy0 - y0
    wa = np.asarray(widget)[wy0:wy0 + (fy1 - fy0), wx0:wx0 + (fx1 - fx0)]

    rgb = wa[..., :3].astype(np.float32)
    a = (wa[..., 3:4].astype(np.float32)) / 255.0
    region = frame_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
    # Widget is authored in RGB; swap to BGR only when the frame is BGR.
    widget_px = rgb[..., ::-1] if frame_is_bgr else rgb
    blended = region * (1.0 - a) + widget_px * a

    out = frame_bgr.copy()
    out[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)
    return out


def delta_series_from_sync_points(
    sync_times_top: Sequence[float],
    sync_times_bottom: Sequence[float],
    output_times: Sequence[float],
) -> Tuple[List[float], float]:
    """Per-output-frame relative delta and the recommended clamp.

    The bottom (reference) clip plays unwarped, so its elapsed lap time equals
    the output time. The top clip is warped; its elapsed time at each output
    frame comes from interpolating the sync-point mapping. The signed delta is
    ``elapsed_bottom - elapsed_top`` (negative => bottom car ahead).

    Args:
        sync_times_top: sync-point timestamps in the TOP video (reference lap).
        sync_times_bottom: matching timestamps in the BOTTOM video.
        output_times: output-frame times (seconds from output start).

    Returns:
        (deltas, clamp) where clamp is the maximum absolute delta across the
        series (>= a small floor so an all-zero lap still renders).

    The bottom-video time -> top-video time mapping is piecewise-linear between
    sync points. This matches the pipeline's warp, where each segment plays at a
    constant speed, so the overlay agrees with the actual on-screen alignment.
    """
    top = np.asarray(sync_times_top, dtype=float)
    bot = np.asarray(sync_times_bottom, dtype=float)
    start_top, start_bot = top[0], bot[0]

    ot = np.asarray(output_times, dtype=float)
    elapsed_bottom = ot                                  # bottom elapsed == output time
    elapsed_top = np.interp(start_bot + ot, bot, top) - start_top
    deltas = elapsed_bottom - elapsed_top
    clamp = float(max(np.abs(deltas).max(), 1e-3))
    return deltas.tolist(), clamp
