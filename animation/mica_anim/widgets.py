"""Reusable mobjects: polar tuning plots, receptive-field glyphs, bar charts."""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, WHITE, Circle, DashedVMobject,
                   Ellipse, Line, MathTex, Rectangle, VGroup, VMobject)

from .neuro import N_ORIENT, THETAS
from .style import EXC, INH, MUTED, orientation_color


class PolarTuning(VGroup):
    """A hypercolumn's tuning curve in polar form, as in the spec's step-4
    figure: each orientation theta_k is drawn at theta_k and theta_k + 180
    degrees, so a line gives two opposite lobes and a crossing four.

    Scene angles are minus image angles (image y points down), so a lobe
    points along the line it responds to in the image beside it."""

    def __init__(self, radius=2.2, vmax=16.0, rings=(5, 10, 15), **kw):
        super().__init__(**kw)
        self.radius = radius
        self.vmax = vmax
        self.axes = VGroup()
        for v in rings:
            if v <= vmax:
                self.axes.add(Circle(radius=self.r(v), stroke_color=MUTED, stroke_width=1, stroke_opacity=0.35))
        for k in range(N_ORIENT):
            a = -THETAS[k]
            d = np.array([np.cos(a), np.sin(a), 0.0])
            self.axes.add(Line(-d * radius, d * radius, stroke_color=MUTED, stroke_width=0.6, stroke_opacity=0.18))
        self.add(self.axes)

    @property
    def centre(self):
        return self.axes.get_center()

    def r(self, v):
        return self.radius * np.clip(v, 0.0, None) / self.vmax

    def direction(self, k, flip=False):
        a = -THETAS[k] + (np.pi if flip else 0.0)
        return np.array([np.cos(a), np.sin(a), 0.0])

    def points_for(self, values):
        vals = np.clip(np.asarray(values, float), 0.0, None)
        pts = [self.centre + self.direction(k) * self.r(v) for k, v in enumerate(vals)]
        pts += [self.centre + self.direction(k, True) * self.r(v) for k, v in enumerate(vals)]
        return pts

    def curve(self, values, color=WHITE, width=3.0, fill_opacity=0.0):
        pts = self.points_for(values)
        c = VMobject(stroke_color=color, stroke_width=width, fill_color=color, fill_opacity=fill_opacity)
        c.set_points_as_corners(pts + [pts[0]])
        return c

    def spokes(self, values, k_range=None, width=4.0):
        """One pair of coloured spokes per orientation (k, and k + 180 deg)."""
        g = VGroup()
        for k in (range(N_ORIENT) if k_range is None else k_range):
            g.add(self.spoke(k, values[k], width))
        return g

    def spoke(self, k, value, width=4.0):
        c = self.centre
        col = orientation_color(THETAS[k])
        r = self.r(value)
        return VGroup(Line(c, c + self.direction(k) * max(r, 1e-3), stroke_color=col, stroke_width=width),
                      Line(c, c + self.direction(k, True) * max(r, 1e-3), stroke_color=col, stroke_width=width))

    def ring(self, value, color=WHITE, dashed=True, width=2.0):
        c = Circle(radius=max(self.r(value), 1e-3), stroke_color=color, stroke_width=width).move_to(self.centre)
        return DashedVMobject(c, num_dashes=40) if dashed else c


def receptive_field(length=1.6, sigma=0.12, theta_scene=0.0):
    """Glyph of an even-symmetric simple cell: an excitatory stripe (+)
    between two inhibitory flanks (-), inside its outline."""
    outline = Ellipse(width=length, height=6.5 * sigma, stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.8)
    centre = Ellipse(width=length * 0.92, height=2.0 * sigma, stroke_width=0, fill_color=EXC, fill_opacity=0.55)
    flank_l = Ellipse(width=length * 0.8, height=1.5 * sigma, stroke_width=0, fill_color=INH, fill_opacity=0.45).shift(UP * 2.0 * sigma)
    flank_r = flank_l.copy().shift(DOWN * 4.0 * sigma)
    plus = MathTex("+", color=WHITE).scale(0.45)
    g = VGroup(outline, flank_l, flank_r, centre, plus)
    g.rotate(theta_scene, about_point=ORIGIN)
    return g


def bar_chart(values, width=4.0, height=2.0, vmax=None, colors=None, gap=0.15):
    """Plain vertical bars for a few values (left to right)."""
    vals = np.asarray(values, float)
    vmax = vals.max() if vmax is None else vmax
    n = len(vals)
    bw = width / n
    bars = VGroup()
    for i, v in enumerate(vals):
        h = max(height * max(v, 0.0) / vmax, 1e-3)
        b = Rectangle(width=bw * (1 - gap), height=h, stroke_width=0,
                      fill_color=(colors[i] if colors is not None else EXC), fill_opacity=0.9)
        b.move_to(LEFT * width / 2 + RIGHT * (i + 0.5) * bw + UP * h / 2, aligned_edge=ORIGIN)
        bars.add(b)
    base = Line(LEFT * width / 2, RIGHT * width / 2, stroke_color=MUTED, stroke_width=1.5)
    return VGroup(base, bars)
