"""Drawing helpers for chapters 8 and 9 (end-stopped and junction cells).

Colours keep their meanings: line ends END (orange), passing lines CONT (green,
a line that continues through the point), junctions JUNC (pink) and the
vesselscene junction-type colours (vsdata.JUNCTION_COLORS).
"""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, WHITE, Arrow, Circle, DashedLine, DecimalNumber, Dot,
                   Line, MathTex, Rectangle, Tex, VGroup)

from .style import CONT, END, JUNC, MUTED
from .v2cells import scene_vec
from .vsdata import JUNCTION_COLORS

# black -> ember -> orange -> pale: the end map
END_RAMP = np.array([[0.0, 0.0, 0.0], [0.30, 0.09, 0.0], [0.80, 0.36, 0.03],
                     [1.0, 0.60, 0.15], [1.0, 0.88, 0.62]])

TYPE_COLOR = {"T": JUNCTION_COLORS["pseudo-T"], "Y": JUNCTION_COLORS["bifurcation"],
              "crossing": JUNCTION_COLORS["crossing"], "crossing (broken)": JUNCTION_COLORS["crossing"],
              "compound": JUNCTION_COLORS["compound"]}
TYPE_LABEL = {"T": r"3-way (T)", "Y": r"3-way (Y)", "crossing": r"crossing",
              "crossing (broken)": r"crossing", "compound": r"compound"}


def end_arrow(pf, x, y, phi, length=0.5, color=END, width=5, back=0.0):
    """An arrow at the end (x, y) pointing along its free direction phi (image angle)."""
    p = pf.point(x, y)
    v = scene_vec(phi)
    return Arrow(p - v * back, p + v * length, buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.38, max_stroke_width_to_length_ratio=12)


def junction_ring(pf, x, y, kind, radius_px=9.0, width=3.5):
    col = TYPE_COLOR.get(kind, JUNC)
    return Circle(radius=radius_px * pf.unit, stroke_color=col, stroke_width=width).move_to(pf.point(x, y))


def arm_glyph(kinds, size=0.42, phis=None):
    """A small icon of a junction's arms around a centre: 'pass' arms as green rays from the
    centre, 'end' arms as orange arrows pointing into it.  kinds: list of 'pass' / 'end';
    phis: their outward scene angles (default: spread evenly)."""
    n = len(kinds)
    if phis is None:
        phis = [np.pi / 2 + 2 * np.pi * i / n for i in range(n)]
    g = VGroup()
    for k, a in zip(kinds, phis):
        d = np.array([np.cos(a), np.sin(a), 0.0])
        if k == "pass":
            g.add(Line(0.0 * d, d * size, stroke_color=CONT, stroke_width=4))
        else:
            g.add(Arrow(d * size * 1.12, d * size * 0.12, buff=0, color=END, stroke_width=4,
                        max_tip_length_to_length_ratio=0.4, max_stroke_width_to_length_ratio=14))
    g.add(Dot(radius=0.035, color=WHITE))
    return g


class Meters(VGroup):
    """Vertical bars with coloured MathTex labels below and live numbers above."""

    def __init__(self, names, colors, vmax, height=2.4, spacing=1.0, bar_width=0.5, **kw):
        super().__init__(**kw)
        self.vmax, self.h, self.bw, self.colors = vmax, height, bar_width, colors
        n = len(names)
        self.xs = [(i - (n - 1) / 2) * spacing for i in range(n)]
        self.base = Line(LEFT * (n * spacing / 2), RIGHT * (n * spacing / 2), stroke_color=MUTED, stroke_width=2)
        self.labels = VGroup(*[MathTex(nm, color=c).scale(0.8).move_to([x, -0.38, 0])
                               for nm, c, x in zip(names, colors, self.xs)])
        self.add(self.base, self.labels)

    def bar(self, i, v):
        h = max(self.h * np.clip(v, 0, self.vmax) / self.vmax, 1e-3)
        b = Rectangle(width=self.bw, height=h, stroke_width=0, fill_color=self.colors[i], fill_opacity=0.9)
        b.move_to(self.base.get_center() + RIGHT * self.xs[i] + UP * h / 2)
        return b

    def bars(self, values, which=None):
        idx = range(len(values)) if which is None else which
        return VGroup(*[self.bar(i, values[i]) for i in idx])

    def number(self, i, getter, places=0):
        d = DecimalNumber(getter(), num_decimal_places=places, color=self.colors[i]).scale(0.55)

        def upd(m):
            m.set_value(getter())
            top = self.base.get_center()[1] + max(self.h * np.clip(getter(), 0, self.vmax) / self.vmax, 0)
            m.move_to([self.base.get_center()[0] + self.xs[i], top + 0.2, 0])
        d.add_updater(upd)
        upd(d)
        return d

    def level_line(self, i, v, color=MUTED):
        y = self.base.get_center()[1] + self.h * v / self.vmax
        x = self.base.get_center()[0] + self.xs[i]
        return DashedLine([x - self.bw * 0.8, y, 0], [x + self.bw * 0.8, y, 0], color=color, stroke_width=2)


def overlay_image(values, under, height, ramp="magma", vmin=0.0, vmax=1.0, under_max=1.0, under_level=0.32):
    """A map in a colour ramp over a dim grey picture of what it was computed from
    (so a map that is zero along the lines still shows where the lines are)."""
    from manim import ImageMobject
    from manim.constants import RESAMPLING_ALGORITHMS
    from .style import colormap
    top = colormap(values, ramp, vmin, vmax).astype(float)
    g = np.clip(np.asarray(under, float) / under_max, 0, 1) * under_level * 255
    top[..., :3] = np.maximum(top[..., :3], g[..., None])
    im = ImageMobject(top.astype(np.uint8))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def tiny(text, color=MUTED, scale=0.45):
    return Tex(text, color=color).scale(scale)
