"""Shared look: palette with fixed meanings, colormaps and image helpers.

Colour meanings are kept the same in every scene, so a colour can carry an
idea across the whole video:

  orientation  -> hue (a 180-degree colour wheel; 16 sheets, 16 hues)
  excitation   -> EXC (blue)        inhibition -> INH (red)
  feedforward  -> FF  (blue)        feedback / recurrent -> FB (gold)
  line ends    -> END (orange)      junctions -> JUNC (pink)
  continuation -> CONT (green)      attention label -> ATTN (yellow)
  OD / blood   -> BLOOD (deep red)
"""
from __future__ import annotations

import colorsys

import numpy as np
from manim import (BLUE_C, GOLD, GREEN_C, GREY_B, ORANGE, PINK, RED_C, YELLOW,
                   ImageMobject, ManimColor)
from manim.constants import RESAMPLING_ALGORITHMS

EXC = BLUE_C
INH = RED_C
FF = BLUE_C
FB = GOLD
END = ORANGE
JUNC = PINK
CONT = GREEN_C
ATTN = YELLOW
BLOOD = ManimColor("#C0392B")
MUTED = GREY_B


def orientation_color(theta: float) -> ManimColor:
    """Hue for an orientation in radians (period pi)."""
    h = (theta % np.pi) / np.pi
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 1.0)
    return ManimColor.from_rgb((r, g, b))


# ------------------------------------------------------------------ colormaps

_MAGMA = np.array([  # a short magma-like ramp: black -> purple -> orange -> pale yellow
    [0.00, 0.00, 0.02],
    [0.23, 0.06, 0.44],
    [0.55, 0.16, 0.51],
    [0.87, 0.32, 0.38],
    [0.99, 0.64, 0.36],
    [0.99, 0.99, 0.75],
])
_GREY = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
_DIVERGING = np.array([[0.36, 0.62, 0.95], [0.0, 0.0, 0.0], [0.95, 0.40, 0.36]])


def colormap(values: np.ndarray, ramp: str | np.ndarray = "magma", vmin=None, vmax=None) -> np.ndarray:
    """Map a 2-D array to uint8 RGBA with a piecewise-linear ramp."""
    ramp = {"magma": _MAGMA, "grey": _GREY, "diverging": _DIVERGING}.get(ramp, ramp) if isinstance(ramp, str) else ramp
    v = np.asarray(values, float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(ramp))
    rgb = np.stack([np.interp(t, xs, ramp[:, c]) for c in range(3)], axis=-1)
    rgba = np.concatenate([rgb, np.ones(v.shape + (1,))], axis=-1)
    return (rgba * 255).astype(np.uint8)


def image(values: np.ndarray, height: float, ramp="grey", vmin=None, vmax=None) -> ImageMobject:
    """An ImageMobject of a 2-D array, shown pixelated (nearest neighbour)."""
    im = ImageMobject(colormap(values, ramp, vmin, vmax))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


class PixelFrame:
    """Maps image pixel coordinates (x right, y down) to scene points for an
    image of shape (h, w) shown with the given height and centre."""

    def __init__(self, shape, height: float, centre):
        self.h, self.w = shape
        self.height = height
        self.width = height * self.w / self.h
        self.centre = np.asarray(centre, float)
        self.unit = height / self.h  # scene units per pixel

    def point(self, x, y):
        return self.centre + np.array([(x + 0.5) / self.w - 0.5, 0.5 - (y + 0.5) / self.h, 0.0]) * [self.width, self.height, 0]

    def pixel(self, point):
        """Inverse of point(): scene point -> (x, y) in pixels."""
        d = (np.asarray(point, float) - self.centre)[:2] / [self.width, self.height]
        return (d[0] + 0.5) * self.w - 0.5, (0.5 - d[1]) * self.h - 0.5

    def screen_angle(self, theta: float) -> float:
        """Image orientation theta (y down) as a scene angle (y up)."""
        return -theta


def bilinear(stack: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample a (k, h, w) stack at sub-pixel (x, y)."""
    h, w = stack.shape[-2:]
    x = float(np.clip(x, 0, w - 1.001))
    y = float(np.clip(y, 0, h - 1.001))
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    s = stack[..., y0:y0 + 2, x0:x0 + 2]
    return (s[..., 0, 0] * (1 - fx) * (1 - fy) + s[..., 0, 1] * fx * (1 - fy)
            + s[..., 1, 0] * (1 - fx) * fy + s[..., 1, 1] * fx * fy)
