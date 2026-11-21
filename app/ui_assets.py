"""
Fast Procedural High-DPI Visual & Audio Assets for Gardner's 5x5 Minichess Studio.

Provides:
  - Crisp 3x supersampled antialiased Staunton chess piece sprites (Pillow + LANCZOS)
  - Pre-baked full-board background images (frame + 25 squares with fast wood grain)
  - Crisp circular Chess.com-style move-quality badges (Great '!', Good '✓', Mistake '?')
  - Soft procedural wooden-click / capture / check / promotion / game-over WAV cues
  - Graceful fallback when Pillow or winsound is unavailable
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import math
import struct
import sys
import wave
from typing import Dict, List, Optional, Tuple

from engine.game_state import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import winsound

    WINSOUND_AVAILABLE = sys.platform == "win32"
except ImportError:
    WINSOUND_AVAILABLE = False


# ---------------------------------------------------------------------------
# BOARD THEMES & GRADE COLORS
# ---------------------------------------------------------------------------

GRADE_COLORS: Dict[str, Tuple[str, str, str]] = {
    # grade -> (circle_hex, symbol, short_tag)
    "Great": ("#26a9e0", "!", "! GREAT"),
    "Good": ("#7cb342", "✓", "✓ GOOD"),
    "Mistake": ("#e57342", "?", "? MISTAKE"),
}


@dataclass(frozen=True)
class BoardTheme:
    name: str
    light_sq: str
    dark_sq: str
    frame_outer: str
    frame_inner: str
    frame_accent: str
    last_move_light: str
    last_move_dark: str
    selected_sq: str
    hover_sq: str
    check_sq: str
    legal_dot: str
    legal_ring: str
    move_arrow: str
    hint_arrow: str
    pv_arrow: str
    coord_on_light: str
    coord_on_dark: str
    use_wood_grain: bool = False


BOARD_THEMES: Dict[str, BoardTheme] = {
    "Classic Wood": BoardTheme(
        name="Classic Wood",
        light_sq="#ecd9bc",
        dark_sq="#a67751",
        frame_outer="#3b2618",
        frame_inner="#28180e",
        frame_accent="#c9a56b",
        last_move_light="#f3e27c",
        last_move_dark="#cbb048",
        selected_sq="#7ab8e6",
        hover_sq="#8fd3a8",
        check_sq="#e64e58",
        legal_dot="#2b6b42",
        legal_ring="#cf3a44",
        move_arrow="#e67e22",
        hint_arrow="#2ecc71",
        pv_arrow="#5dade2",
        coord_on_light="#7d5637",
        coord_on_dark="#f3e5cf",
        use_wood_grain=True,
    ),
    "Tournament Green": BoardTheme(
        name="Tournament Green",
        light_sq="#eeeed2",
        dark_sq="#769656",
        frame_outer="#2a3822",
        frame_inner="#1d2717",
        frame_accent="#9fc47b",
        last_move_light="#f7f779",
        last_move_dark="#baca44",
        selected_sq="#75b9e7",
        hover_sq="#9be0b0",
        check_sq="#e64e58",
        legal_dot="#365928",
        legal_ring="#d63c46",
        move_arrow="#f39c12",
        hint_arrow="#27ae60",
        pv_arrow="#3498db",
        coord_on_light="#5d7a40",
        coord_on_dark="#f4f4df",
        use_wood_grain=False,
    ),
    "Slate Blue": BoardTheme(
        name="Slate Blue",
        light_sq="#dee3e6",
        dark_sq="#8ca2ad",
        frame_outer="#27323c",
        frame_inner="#1b232b",
        frame_accent="#8ab4cf",
        last_move_light="#b9d6f2",
        last_move_dark="#6d9dc5",
        selected_sq="#82c9e8",
        hover_sq="#9be0b0",
        check_sq="#e64e58",
        legal_dot="#2f5266",
        legal_ring="#d63c46",
        move_arrow="#e67e22",
        hint_arrow="#2ecc71",
        pv_arrow="#5dade2",
        coord_on_light="#5b7280",
        coord_on_dark="#eef2f5",
        use_wood_grain=False,
    ),
    "Obsidian Gold": BoardTheme(
        name="Obsidian Gold",
        light_sq="#d6dbe6",
        dark_sq="#5a657d",
        frame_outer="#191c28",
        frame_inner="#11131c",
        frame_accent="#d9b468",
        last_move_light="#e8d88a",
        last_move_dark="#a69546",
        selected_sq="#74b0e3",
        hover_sq="#80d4a4",
        check_sq="#e64e58",
        legal_dot="#2d4c6b",
        legal_ring="#e04852",
        move_arrow="#f0932b",
        hint_arrow="#38d97c",
        pv_arrow="#68b5f0",
        coord_on_light="#485269",
        coord_on_dark="#e6eaf2",
        use_wood_grain=False,
    ),
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# FAST PROCEDURAL 3x SUPERSAMPLED STAUNTON CHESS PIECES
# ---------------------------------------------------------------------------

def _pt(x: float, y: float, s: int) -> Tuple[float, float]:
    return x * s, y * s


def _poly(pts: List[Tuple[float, float]], s: int) -> List[Tuple[float, float]]:
    return [(x * s, y * s) for x, y in pts]


def _box(x0: float, y0: float, x1: float, y1: float, s: int) -> List[float]:
    return [x0 * s, y0 * s, x1 * s, y1 * s]


_GRADIENT_CACHE: Dict[Tuple[int, bool], "Image.Image"] = {}


def _get_shading_gradient(size: int, is_white: bool) -> "Image.Image":
    key = (size, is_white)
    cached = _GRADIENT_CACHE.get(key)
    if cached is not None:
        return cached

    if is_white:
        tl, mid, br = (255, 253, 248), (242, 235, 222), (206, 195, 176)
    else:
        tl, mid, br = (74, 80, 102), (38, 41, 56), (17, 19, 28)

    g_sz = 32
    buf = bytearray(g_sz * g_sz * 4)
    idx = 0
    for y in range(g_sz):
        ty = y / (g_sz - 1)
        for x in range(g_sz):
            tx = x / (g_sz - 1)
            t = 0.35 * tx + 0.65 * ty
            if t < 0.5:
                u = t * 2.0
                r = int(tl[0] * (1 - u) + mid[0] * u)
                g = int(tl[1] * (1 - u) + mid[1] * u)
                b = int(tl[2] * (1 - u) + mid[2] * u)
            else:
                u = (t - 0.5) * 2.0
                r = int(mid[0] * (1 - u) + br[0] * u)
                g = int(mid[1] * (1 - u) + br[1] * u)
                b = int(mid[2] * (1 - u) + br[2] * u)
            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            buf[idx + 3] = 255
            idx += 4

    grad = Image.frombytes("RGBA", (g_sz, g_sz), bytes(buf)).resize((size, size), Image.Resampling.BILINEAR)
    _GRADIENT_CACHE[key] = grad
    return grad


def _draw_base_pedestal(draw: "ImageDraw.ImageDraw", s: int, wide: bool, fill: int) -> None:
    if wide:
        draw.rounded_rectangle(_box(0.18, 0.80, 0.82, 0.89, s), radius=0.04 * s, fill=fill)
        draw.rounded_rectangle(_box(0.24, 0.73, 0.76, 0.82, s), radius=0.03 * s, fill=fill)
    else:
        draw.rounded_rectangle(_box(0.22, 0.80, 0.78, 0.89, s), radius=0.04 * s, fill=fill)
        draw.rounded_rectangle(_box(0.28, 0.73, 0.72, 0.82, s), radius=0.03 * s, fill=fill)


def _draw_piece_mask_and_details(
    piece_type: int,
    s: int,
    mask_draw: "ImageDraw.ImageDraw",
    detail_draw: "ImageDraw.ImageDraw",
    accent_rgba: Tuple[int, int, int, int],
    highlight_rgba: Tuple[int, int, int, int],
) -> None:
    lw = max(2, int(round(s * 0.022)))
    thin = max(2, int(round(s * 0.014)))

    if piece_type == PAWN:
        _draw_base_pedestal(mask_draw, s, wide=False, fill=255)
        stem = [
            (0.30, 0.76), (0.37, 0.62), (0.41, 0.49),
            (0.59, 0.49), (0.63, 0.62), (0.70, 0.76),
        ]
        mask_draw.polygon(_poly(stem, s), fill=255)
        mask_draw.rounded_rectangle(_box(0.34, 0.45, 0.66, 0.52, s), radius=0.03 * s, fill=255)
        mask_draw.ellipse(_box(0.35, 0.17, 0.65, 0.47, s), fill=255)

        detail_draw.line([_pt(0.27, 0.80, s), _pt(0.73, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.35, 0.73, s), _pt(0.65, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.38, 0.49, s), _pt(0.62, 0.49, s)], fill=accent_rgba, width=thin)
        detail_draw.arc(_box(0.39, 0.21, 0.56, 0.37, s), start=195, end=275, fill=highlight_rgba, width=lw)

    elif piece_type == ROOK:
        _draw_base_pedestal(mask_draw, s, wide=True, fill=255)
        shaft = [(0.28, 0.75), (0.32, 0.36), (0.68, 0.36), (0.72, 0.75)]
        mask_draw.polygon(_poly(shaft, s), fill=255)
        crown_band = [(0.23, 0.31), (0.77, 0.31), (0.71, 0.40), (0.29, 0.40)]
        mask_draw.polygon(_poly(crown_band, s), fill=255)
        mask_draw.rectangle(_box(0.23, 0.24, 0.77, 0.32, s), fill=255)
        mask_draw.rounded_rectangle(_box(0.23, 0.14, 0.37, 0.28, s), radius=0.015 * s, fill=255)
        mask_draw.rounded_rectangle(_box(0.43, 0.14, 0.57, 0.28, s), radius=0.015 * s, fill=255)
        mask_draw.rounded_rectangle(_box(0.63, 0.14, 0.77, 0.28, s), radius=0.015 * s, fill=255)

        detail_draw.line([_pt(0.23, 0.80, s), _pt(0.77, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.28, 0.73, s), _pt(0.72, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.31, 0.40, s), _pt(0.69, 0.40, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.25, 0.31, s), _pt(0.75, 0.31, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.36, 0.44, s), _pt(0.34, 0.69, s)], fill=highlight_rgba, width=lw)

    elif piece_type == KNIGHT:
        _draw_base_pedestal(mask_draw, s, wide=True, fill=255)
        horse = [
            (0.27, 0.75), (0.31, 0.63), (0.39, 0.53), (0.44, 0.45),
            (0.38, 0.44), (0.31, 0.47), (0.24, 0.48), (0.18, 0.43),
            (0.19, 0.36), (0.28, 0.27), (0.36, 0.20), (0.42, 0.11),
            (0.49, 0.17), (0.58, 0.18), (0.68, 0.24), (0.75, 0.36),
            (0.77, 0.51), (0.74, 0.66), (0.71, 0.75),
        ]
        mask_draw.polygon(_poly(horse, s), fill=255)

        detail_draw.line([_pt(0.23, 0.80, s), _pt(0.77, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.28, 0.73, s), _pt(0.72, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.ellipse(_box(0.34, 0.27, 0.40, 0.32, s), fill=accent_rgba)
        detail_draw.ellipse(_box(0.21, 0.39, 0.25, 0.43, s), fill=accent_rgba)
        detail_draw.line([_pt(0.25, 0.45, s), _pt(0.34, 0.43, s)], fill=accent_rgba, width=thin)
        mane = [(0.55, 0.23), (0.65, 0.31), (0.69, 0.44), (0.68, 0.59), (0.64, 0.70)]
        detail_draw.line(_poly(mane, s), fill=highlight_rgba, width=lw)
        jaw = [(0.45, 0.44), (0.52, 0.38), (0.51, 0.52)]
        detail_draw.line(_poly(jaw, s), fill=accent_rgba, width=thin)

    elif piece_type == BISHOP:
        _draw_base_pedestal(mask_draw, s, wide=True, fill=255)
        stem = [(0.31, 0.75), (0.38, 0.54), (0.62, 0.54), (0.69, 0.75)]
        mask_draw.polygon(_poly(stem, s), fill=255)
        mask_draw.rounded_rectangle(_box(0.32, 0.51, 0.68, 0.58, s), radius=0.03 * s, fill=255)
        miter = [
            (0.36, 0.52), (0.30, 0.43), (0.31, 0.32), (0.38, 0.22),
            (0.50, 0.16), (0.62, 0.22), (0.69, 0.32), (0.70, 0.43), (0.64, 0.52),
        ]
        mask_draw.polygon(_poly(miter, s), fill=255)
        mask_draw.ellipse(_box(0.45, 0.09, 0.55, 0.18, s), fill=255)

        detail_draw.line([_pt(0.23, 0.80, s), _pt(0.77, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.28, 0.73, s), _pt(0.72, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.35, 0.54, s), _pt(0.65, 0.54, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.55, 0.35, s), _pt(0.67, 0.23, s)], fill=accent_rgba, width=int(lw * 1.35))
        detail_draw.line([_pt(0.50, 0.36, s), _pt(0.50, 0.48, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.44, 0.42, s), _pt(0.56, 0.42, s)], fill=accent_rgba, width=thin)
        detail_draw.arc(_box(0.35, 0.22, 0.55, 0.46, s), start=175, end=250, fill=highlight_rgba, width=lw)

    elif piece_type == QUEEN:
        _draw_base_pedestal(mask_draw, s, wide=True, fill=255)
        body = [
            (0.27, 0.75), (0.31, 0.58), (0.16, 0.27), (0.32, 0.47),
            (0.32, 0.19), (0.43, 0.44), (0.50, 0.15), (0.57, 0.44),
            (0.68, 0.19), (0.68, 0.47), (0.84, 0.27), (0.69, 0.58), (0.73, 0.75),
        ]
        mask_draw.polygon(_poly(body, s), fill=255)
        pr = 0.044
        for px_c, py_c in [(0.16, 0.25), (0.32, 0.17), (0.50, 0.13), (0.68, 0.17), (0.84, 0.25)]:
            mask_draw.ellipse(_box(px_c - pr, py_c - pr, px_c + pr, py_c + pr, s), fill=255)

        detail_draw.line([_pt(0.23, 0.80, s), _pt(0.77, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.27, 0.73, s), _pt(0.73, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.31, 0.63, s), _pt(0.69, 0.63, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.32, 0.58, s), _pt(0.68, 0.58, s)], fill=highlight_rgba, width=thin)
        for jx in (0.39, 0.50, 0.61):
            detail_draw.ellipse(_box(jx - 0.022, 0.66, jx + 0.022, 0.70, s), fill=accent_rgba)

    elif piece_type == KING:
        _draw_base_pedestal(mask_draw, s, wide=True, fill=255)
        mask_draw.rounded_rectangle(_box(0.455, 0.07, 0.545, 0.29, s), radius=0.015 * s, fill=255)
        mask_draw.rounded_rectangle(_box(0.37, 0.125, 0.63, 0.20, s), radius=0.015 * s, fill=255)
        crown = [
            (0.26, 0.75), (0.28, 0.64), (0.20, 0.50), (0.19, 0.37),
            (0.26, 0.29), (0.38, 0.28), (0.46, 0.33), (0.50, 0.27),
            (0.54, 0.33), (0.62, 0.28), (0.74, 0.29), (0.81, 0.37),
            (0.80, 0.50), (0.72, 0.64), (0.74, 0.75),
        ]
        mask_draw.polygon(_poly(crown, s), fill=255)

        detail_draw.line([_pt(0.23, 0.80, s), _pt(0.77, 0.80, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.27, 0.73, s), _pt(0.73, 0.73, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.29, 0.65, s), _pt(0.71, 0.65, s)], fill=accent_rgba, width=thin)
        detail_draw.line([_pt(0.29, 0.61, s), _pt(0.71, 0.61, s)], fill=highlight_rgba, width=thin)
        detail_draw.line([_pt(0.50, 0.31, s), _pt(0.50, 0.60, s)], fill=accent_rgba, width=thin)
        detail_draw.arc(_box(0.24, 0.33, 0.45, 0.58, s), start=150, end=250, fill=highlight_rgba, width=lw)


def render_staunton_piece_image(piece_code: int, px_size: int) -> "Image.Image":
    """
    Fast 3x supersampled Staunton chess piece RGBA image with crisp outline & drop shadow.
    Uses separable multi-pass MaxFilter(5) for ~5x faster startup than large single kernels.
    """
    ss = 3
    s = max(60, px_size * ss)
    piece_type = abs(piece_code)
    is_white = piece_code > 0

    grad = _get_shading_gradient(s, is_white)
    if is_white:
        outline_rgba = (36, 30, 26, 255)
        accent_rgba = (74, 63, 54, 215)
        highlight_rgba = (255, 255, 255, 210)
    else:
        outline_rgba = (10, 11, 16, 255)
        accent_rgba = (176, 184, 210, 190)
        highlight_rgba = (145, 155, 185, 165)

    mask = Image.new("L", (s, s), 0)
    details = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _draw_piece_mask_and_details(
        piece_type, s, ImageDraw.Draw(mask), ImageDraw.Draw(details), accent_rgba, highlight_rgba
    )

    # Fast crisp dilation via two small 5x5 passes (~radius 4px at 3x -> ~1.5px stroke at 1x)
    dilated_mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MaxFilter(5))

    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    shadow_alpha = dilated_mask.filter(ImageFilter.GaussianBlur(radius=s * 0.022))
    shadow_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    shadow_layer.paste((12, 10, 14, 115), (int(s * 0.022), int(s * 0.030)), shadow_alpha)
    canvas = Image.alpha_composite(canvas, shadow_layer)

    stroke_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    stroke_layer.paste(outline_rgba, (0, 0), dilated_mask)
    canvas = Image.alpha_composite(canvas, stroke_layer)

    body_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    body_layer.paste(grad, (0, 0), mask)
    canvas = Image.alpha_composite(canvas, body_layer)

    clipped_details = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    clipped_details.paste(details, (0, 0), mask)
    canvas = Image.alpha_composite(canvas, clipped_details)

    return canvas.resize((px_size, px_size), Image.Resampling.LANCZOS)


def render_wood_square_image(base_hex: str, px_size: int, seed_offset: int) -> "Image.Image":
    """Fast organic horizontal wood-grain square tile (generated at 48px and scaled smoothly)."""
    r0, g0, b0 = _hex_to_rgb(base_hex)
    g_sz = 48
    buf = bytearray(g_sz * g_sz * 3)
    idx = 0
    for y in range(g_sz):
        w1 = math.sin((y + seed_offset * 17) * 0.56)
        w2 = math.sin((y + seed_offset * 31) * 1.42)
        base_d = w1 * 4.5 + w2 * 2.5
        for x in range(g_sz):
            delta = int(base_d + math.sin((x + seed_offset * 13) * 0.12 + w1 * 1.4) * 3.0)
            buf[idx] = max(0, min(255, r0 + delta))
            buf[idx + 1] = max(0, min(255, g0 + int(delta * 0.85)))
            buf[idx + 2] = max(0, min(255, b0 + int(delta * 0.65)))
            idx += 3
    return Image.frombytes("RGB", (g_sz, g_sz), bytes(buf)).resize((px_size, px_size), Image.Resampling.BILINEAR)


def render_board_background_image(theme: BoardTheme, sq_px: int, frame_pad: int) -> "Image.Image":
    """Pre-bake beveled frame + all 25 squares into a single flat image for 1-item canvas blit."""
    board_px = sq_px * 5 + 2 * frame_pad
    img = Image.new("RGB", (board_px, board_px), _hex_to_rgb(theme.frame_outer))
    draw = ImageDraw.Draw(img)
    accent = _hex_to_rgb(theme.frame_accent)
    inner = _hex_to_rgb(theme.frame_inner)

    draw.rectangle([2, 2, board_px - 3, board_px - 3], outline=accent, width=1)
    draw.rectangle(
        [frame_pad - 2, frame_pad - 2, board_px - frame_pad + 1, board_px - frame_pad + 1],
        fill=inner,
        outline=accent,
        width=1,
    )

    wood_light = [render_wood_square_image(theme.light_sq, sq_px, v) for v in range(4)] if theme.use_wood_grain else None
    wood_dark = [render_wood_square_image(theme.dark_sq, sq_px, v) for v in range(4)] if theme.use_wood_grain else None
    light_rgb = _hex_to_rgb(theme.light_sq)
    dark_rgb = _hex_to_rgb(theme.dark_sq)

    for vis_row in range(5):
        for vis_col in range(5):
            # Parity of a 5x5 board is invariant under 180-degree flip: (4-r)+(4-f) == (r+f) mod 2
            is_light = ((4 - vis_row) + vis_col) % 2 == 1
            x0 = frame_pad + vis_col * sq_px
            y0 = frame_pad + vis_row * sq_px
            if theme.use_wood_grain and wood_light is not None and wood_dark is not None:
                tile = wood_light[(vis_row * 5 + vis_col) % 4] if is_light else wood_dark[(vis_row * 5 + vis_col) % 4]
                img.paste(tile, (x0, y0))
            else:
                draw.rectangle(
                    [x0, y0, x0 + sq_px - 1, y0 + sq_px - 1],
                    fill=light_rgb if is_light else dark_rgb,
                )
    return img


def render_check_glow_image(px_size: int) -> "Image.Image":
    """Fast smooth radial crimson halo tile (rendered at 48px and bilinear-upscaled)."""
    g_sz = 48
    buf = bytearray(g_sz * g_sz * 4)
    half = g_sz / 2.0
    idx = 0
    for y in range(g_sz):
        dy = (y - half) / half
        for x in range(g_sz):
            dx = (x - half) / half
            dist = math.hypot(dx, dy)
            alpha = int(215 * ((1.0 - dist) ** 1.35)) if dist < 1.0 else 0
            buf[idx] = 235
            buf[idx + 1] = 58
            buf[idx + 2] = 70
            buf[idx + 3] = alpha
            idx += 4
    return Image.frombytes("RGBA", (g_sz, g_sz), bytes(buf)).resize((px_size, px_size), Image.Resampling.BILINEAR)


def render_grade_badge_image(grade: str, diameter: int) -> "Image.Image":
    """Render a crisp 3x supersampled Chess.com-style circular badge ('!', '✓', '?')."""
    ss = 3
    s = max(36, diameter * ss)
    color_hex = GRADE_COLORS.get(grade, GRADE_COLORS["Good"])[0]
    rgb = _hex_to_rgb(color_hex)

    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = int(s * 0.08)
    # Subtle drop shadow
    d.ellipse([pad + 2, pad + 3, s - pad + 2, s - pad + 3], fill=(10, 12, 18, 120))
    # Colored circle with crisp white rim
    rim_w = max(2, int(s * 0.07))
    d.ellipse([pad, pad, s - pad, s - pad], fill=rgb + (255,), outline=(255, 255, 255, 235), width=rim_w)

    white = (255, 255, 255, 255)
    sw = max(3, int(s * 0.115))

    if grade == "Great":
        # Bold exclamation mark '!'
        d.rounded_rectangle([s * 0.43, s * 0.23, s * 0.57, s * 0.60], radius=s * 0.05, fill=white)
        d.ellipse([s * 0.425, s * 0.66, s * 0.575, s * 0.80], fill=white)
    elif grade == "Good":
        # Crisp checkmark '✓'
        pts = [(s * 0.28, s * 0.52), (s * 0.44, s * 0.69), (s * 0.74, s * 0.34)]
        d.line(pts, fill=white, width=sw, joint="curve")
    else:
        # Clean question mark '?'
        d.arc([s * 0.34, s * 0.23, s * 0.67, s * 0.54], start=185, end=35, fill=white, width=sw)
        d.line([(s * 0.60, s * 0.48), (s * 0.50, s * 0.58), (s * 0.50, s * 0.62)], fill=white, width=sw)
        d.ellipse([s * 0.43, s * 0.68, s * 0.57, s * 0.81], fill=white)

    return img.resize((diameter, diameter), Image.Resampling.LANCZOS)


class PieceSpriteCache:
    """Caches Tkinter PhotoImages for pieces, baked boards, badges, and glows."""

    def __init__(self) -> None:
        self._pieces: Dict[Tuple[int, int], "ImageTk.PhotoImage"] = {}
        self._boards: Dict[Tuple[str, int, int], "ImageTk.PhotoImage"] = {}
        self._check_glows: Dict[int, "ImageTk.PhotoImage"] = {}
        self._badges: Dict[Tuple[str, int], "ImageTk.PhotoImage"] = {}

    def prewarm(self, theme: BoardTheme, sq_px: int, frame_pad: int) -> None:
        """Pre-generate all active sprites once so drags and animations have zero stutter."""
        if not PIL_AVAILABLE:
            return
        piece_px = max(24, int(round(sq_px * 0.90)))
        for p in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
            self.get_piece(p, piece_px)
            self.get_piece(-p, piece_px)
        self.get_board_bg(theme, sq_px, frame_pad)
        self.get_check_glow(sq_px)
        badge_d = max(18, int(round(sq_px * 0.30)))
        for g in ("Great", "Good", "Mistake"):
            self.get_grade_badge(g, badge_d)

    def get_piece(self, piece_code: int, px_size: int) -> Optional["ImageTk.PhotoImage"]:
        if not PIL_AVAILABLE or px_size < 16:
            return None
        key = (piece_code, px_size)
        img = self._pieces.get(key)
        if img is None:
            img = ImageTk.PhotoImage(render_staunton_piece_image(piece_code, px_size))
            self._pieces[key] = img
        return img

    def get_board_bg(self, theme: BoardTheme, sq_px: int, frame_pad: int) -> Optional["ImageTk.PhotoImage"]:
        if not PIL_AVAILABLE or sq_px < 16:
            return None
        key = (theme.name, sq_px, frame_pad)
        img = self._boards.get(key)
        if img is None:
            img = ImageTk.PhotoImage(render_board_background_image(theme, sq_px, frame_pad))
            self._boards[key] = img
        return img

    def get_check_glow(self, px_size: int) -> Optional["ImageTk.PhotoImage"]:
        if not PIL_AVAILABLE or px_size < 16:
            return None
        img = self._check_glows.get(px_size)
        if img is None:
            img = ImageTk.PhotoImage(render_check_glow_image(px_size))
            self._check_glows[px_size] = img
        return img

    def get_grade_badge(self, grade: str, diameter: int) -> Optional["ImageTk.PhotoImage"]:
        if not PIL_AVAILABLE or diameter < 12:
            return None
        key = (grade, diameter)
        img = self._badges.get(key)
        if img is None:
            img = ImageTk.PhotoImage(render_grade_badge_image(grade, diameter))
            self._badges[key] = img
        return img


# ---------------------------------------------------------------------------
# PROCEDURAL IN-MEMORY SOUND CUES (NO EXTERNAL AUDIO FILES REQUIRED)
# ---------------------------------------------------------------------------

def _samples_to_wav_bytes(samples: List[float], sample_rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        packed = bytearray()
        for s in samples:
            val = max(-1.0, min(1.0, s))
            packed.extend(struct.pack("<h", int(val * 32767)))
        wf.writeframes(bytes(packed))
    return buf.getvalue()


def _synth_wood_click(freq: float, duration_s: float, volume: float = 0.45, double_tap: bool = False) -> bytes:
    sr = 22050
    n = int(sr * duration_s)
    samples = [0.0] * n
    for i in range(n):
        t = i / sr
        env = math.exp(-t * 48.0)
        tone = math.sin(2.0 * math.pi * (freq * (1.0 - 0.35 * min(1.0, t * 25.0))) * t)
        overtone = 0.35 * math.sin(2.0 * math.pi * freq * 2.4 * t) * math.exp(-t * 90.0)
        samples[i] = volume * env * (tone + overtone)
    if double_tap:
        offset = int(sr * 0.022)
        for i in range(offset, n):
            t = (i - offset) / sr
            env = math.exp(-t * 55.0)
            samples[i] += volume * 0.75 * env * math.sin(2.0 * math.pi * (freq * 1.45) * t)
    return _samples_to_wav_bytes(samples, sr)


def _synth_chime(freqs: List[float], step_s: float, total_s: float, volume: float = 0.30) -> bytes:
    sr = 22050
    n = int(sr * total_s)
    samples = [0.0] * n
    for idx, freq in enumerate(freqs):
        start = int(idx * step_s * sr)
        for i in range(start, n):
            t = (i - start) / sr
            attack = min(1.0, t / 0.008)
            decay = math.exp(-t * 10.5)
            samples[i] += volume * attack * decay * (
                0.75 * math.sin(2.0 * math.pi * freq * t)
                + 0.25 * math.sin(2.0 * math.pi * freq * 2.0 * t)
            )
    return _samples_to_wav_bytes(samples, sr)


class SoundEffects:
    """Lightweight non-blocking synthesized sound manager with mute control."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and WINSOUND_AVAILABLE
        self._clips: Dict[str, bytes] = {}
        if WINSOUND_AVAILABLE:
            try:
                self._clips = {
                    "move": _synth_wood_click(210.0, 0.065, volume=0.45, double_tap=False),
                    "capture": _synth_wood_click(285.0, 0.085, volume=0.52, double_tap=True),
                    "check": _synth_chime([659.25, 880.0], step_s=0.055, total_s=0.19, volume=0.28),
                    "promote": _synth_chime([523.25, 659.25, 783.99], step_s=0.045, total_s=0.22, volume=0.26),
                    "gameover": _synth_chime([392.0, 493.88, 587.33, 783.99], step_s=0.06, total_s=0.38, volume=0.25),
                }
            except Exception:
                self._clips = {}

    def play(self, name: str) -> None:
        if not self.enabled or not WINSOUND_AVAILABLE:
            return
        clip = self._clips.get(name)
        if clip is None:
            return
        try:
            winsound.PlaySound(clip, winsound.SND_MEMORY | winsound.SND_ASYNC)
        except Exception:
            pass
