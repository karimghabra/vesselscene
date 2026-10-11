"""Helpers for the chapters that frame the video: the cold open (0), the two
modes of vision (2) and the whole loop (18).

  vesselscene imagery for display: the burst's frames and the stabilised
  average on one shared grey ramp (`average_rgba`, `frame_rgba`, `photo`)
  the observable truth as overlays (`truth_runs`, `truth_rings`)
  the architecture map placed above the captions (`fitted_map`), and
  explicit lighting of its boxes (`light`, `restore`)

Everything here reads vesselscene's cache through mica_anim.vsdata.  The
burst file is rewritten while frames are still being rendered, so loads retry.
"""
from __future__ import annotations

import time
import zipfile
from functools import lru_cache

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, WHITE, AnimationGroup, Circle, Dot,
                   ImageMobject, ManimColor, Transform, VGroup, VMobject)
from manim.constants import RESAMPLING_ALGORITHMS
from scipy.ndimage import distance_transform_edt, gaussian_filter

from . import vsdata as D
from .archmap import BOXES, FEEDFORWARD, ArchitectureMap
from .style import FB, FF, PixelFrame

_LOAD_ERRORS = (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile)


# ------------------------------------------------------------------ data

def scene():
    """vesselscene's default scene (average, OD, truth), retrying while the
    burst file is being rewritten."""
    for _ in range(30):
        try:
            return D.load()
        except _LOAD_ERRORS:
            time.sleep(2.0)
    return D.load()


@lru_cache(maxsize=None)
def burst() -> np.ndarray:
    """The burst's frames, (n, h, w); however many have been rendered so far."""
    path = D.CACHE / f"{D.DEFAULT}_burst.npz"
    for _ in range(30):
        try:
            with np.load(path) as z:
                return np.asarray(z["frames"], np.float32)
        except _LOAD_ERRORS:
            time.sleep(2.0)
    S = scene()
    return np.asarray(S.burst if S.burst is not None else S.average[None], np.float32)


# ------------------------------------------------------- display (one ramp)

# warm paper-white for the sclera, near-black for blood: a grey ramp with a
# faint tint, the same for every frame and for the average
_RAMP = np.array([[0.035, 0.030, 0.040],
                  [0.420, 0.405, 0.410],
                  [0.965, 0.950, 0.925]])


@lru_cache(maxsize=None)
def _display_params():
    """The average's valid mask, smooth log background and stretch.  Frames
    are shown through the same mapping, so a frame and the average differ on
    screen only where the images differ."""
    A = np.asarray(scene().average, float)
    ok = np.isfinite(A) & (A > 0)
    L = np.log(np.where(ok, A, 1.0))
    if (~ok).any():                                   # fill the cropped border by nearest value
        idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        L = L[tuple(idx)]
    bg = gaussian_filter(L, 40)
    F = L - bg
    lo, hi = np.percentile(F[ok], [0.8, 99.7])
    return ok, bg, lo, hi


def _ramp(t: np.ndarray) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, len(_RAMP))
    return np.stack([np.interp(t, xs, _RAMP[:, c]) for c in range(3)], axis=-1)


def display_rgba(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ok, bg, lo, hi = _display_params()
    I = np.asarray(img, float)
    good = np.isfinite(I) & (I > 0)
    L = np.log(np.where(good, I, np.exp(bg)))
    t = np.clip((L - bg - lo) / (hi - lo), 0.0, 1.0)
    rgb = _ramp(t)
    a = np.ones(t.shape) if mask is None else mask.astype(float)
    return (np.concatenate([rgb, a[..., None]], axis=-1) * 255).astype(np.uint8)


@lru_cache(maxsize=None)
def average_rgba() -> np.ndarray:
    ok = _display_params()[0]
    return display_rgba(scene().average, ok)


@lru_cache(maxsize=None)
def frame_rgba(i: int) -> np.ndarray:
    return display_rgba(burst()[i])


@lru_cache(maxsize=None)
def running_mean_rgba(k: int) -> np.ndarray:
    """The mean of the burst's first k frames, as displayed."""
    return display_rgba(burst()[:k].mean(axis=0))


@lru_cache(maxsize=None)
def half_mean(parity: int) -> np.ndarray:
    """The average of the odd (1) or even (0) frames (split-half check)."""
    fr = burst()
    return fr[parity::2].mean(axis=0)


def photo(rgba: np.ndarray, height: float) -> ImageMobject:
    """A photograph-like ImageMobject (smooth resampling)."""
    im = ImageMobject(rgba)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
    im.height = height
    return im


def scene_frame(height: float = 6.3, centre=UP * 0.3) -> PixelFrame:
    return PixelFrame(scene().shape, height, centre)


# --------------------------------------------------------------- the truth

FORK_TYPES = ("bifurcation", "confluence", "pseudo-T")
GROUPS = ("fork", "crossing", "compound")


def group_of(t: str) -> str:
    return "fork" if t in FORK_TYPES else t


def group_color(g: str) -> ManimColor:
    return {"fork": D.JUNCTION_COLORS["bifurcation"], "crossing": D.JUNCTION_COLORS["crossing"],
            "compound": D.JUNCTION_COLORS["compound"]}[g]


def run_mobject(run, pf: PixelFrame, step: int = 2) -> VMobject:
    xy = np.asarray(run["xy"], float)
    if len(xy) > 2 * step:
        xy = np.concatenate([xy[:-1:step], xy[-1:]])
    m = VMobject()
    pts = [pf.point(x, y) for x, y in xy]
    if len(pts) > 2:
        m.set_points_smoothly(pts)
    else:
        m.set_points_as_corners(pts)
    return m


def truth_runs(pf: PixelFrame, width: float = 2.6, min_len_px: int = 12, order: str = "x") -> VGroup:
    """Every observable run in its vessel's identity colour, ordered left to
    right by its centre (so a lagged Create sweeps across the image)."""
    runs = [r for r in scene().truth["runs"] if len(r["xy"]) >= min_len_px]
    if order == "x":
        runs.sort(key=lambda r: float(np.mean(np.asarray(r["xy"])[:, 0])))
    g = VGroup()
    for r in runs:
        g.add(run_mobject(r, pf).set_stroke(D.identity_color(r["vid"]), width=width, opacity=0.95))
    return g


def truth_rings(pf: PixelFrame, width: float = 2.6, rmin: float = 6.0, rmax: float = 13.0) -> dict:
    """The observable junctions as rings, by group: fork (3-way), crossing,
    compound.  Radii are the truth's, clipped for legibility."""
    out = {g: VGroup() for g in GROUPS}
    for j in scene().truth["junctions"]:
        g = group_of(j["type_observable"])
        r = float(np.clip(j["radius"], rmin, rmax)) * pf.unit
        c = Circle(radius=r, stroke_color=group_color(g), stroke_width=width).move_to(pf.point(j["x"], j["y"]))
        c.jid = j["id"]
        out[g].add(c)
    return out


def junction(jid: int) -> dict:
    return next(j for j in scene().truth["junctions"] if j["id"] == jid)


def runs_near(x: float, y: float, radius: float) -> list:
    """The truth runs with any point within radius of (x, y)."""
    out = []
    for r in scene().truth["runs"]:
        xy = np.asarray(r["xy"], float)
        if (np.hypot(xy[:, 0] - x, xy[:, 1] - y) < radius).any():
            out.append(r)
    return out


# ------------------------------------------------------------------ the map

def fitted_map(scale: float = 0.86, bottom: float = -2.98, x: float = 0.0) -> ArchitectureMap:
    """The architecture map, scaled to sit above the caption line."""
    m = ArchitectureMap()
    m.scale(scale)
    m.shift(UP * (bottom - m.get_bottom()[1]) + RIGHT * (x - m.get_center()[0]))
    return m


def _base_color(key: str):
    return FF if key in FEEDFORWARD else WHITE


def light(m: ArchitectureMap, keys, color=FB, dim: float = 0.22, fill: float = 0.24,
          arrows: float | None = None, feedback: bool | None = None) -> AnimationGroup:
    """Light the given boxes and dim the others, from whatever state the map
    is in (unlike ArchitectureMap.highlight, this can be chained).  A key may
    be 'feedback' to keep the gold loops lit."""
    keys = set(keys)
    anims = []
    for k, box in m.boxes.items():
        if k in keys:
            anims.append(box[0].animate.set_stroke(color, width=3.2, opacity=1.0).set_fill(color, opacity=fill))
            anims.append(box[1].animate.set_opacity(1.0))
        else:
            anims.append(box[0].animate.set_stroke(_base_color(k), width=1.6, opacity=dim).set_fill(_base_color(k), opacity=0.02))
            anims.append(box[1].animate.set_opacity(dim + 0.1))
    anims.append(m.arrows.animate.set_opacity(dim if arrows is None else arrows))
    fb_on = ("feedback" in keys) if feedback is None else feedback
    anims += [fade_arrow(a, 1.0 if fb_on else dim) for a in m.feedback]
    anims.append(m.labels.animate.set_opacity(0.6))
    return AnimationGroup(*anims)


def fade_arrow(arrow, opacity: float) -> Transform:
    """Set a (curved) arrow's opacity without filling its arc."""
    t = arrow.copy()
    t.set_stroke(opacity=opacity)
    t.tip.set_fill(opacity=opacity)
    return Transform(arrow, t)


def restore(m: ArchitectureMap) -> AnimationGroup:
    """Back to the map's own look."""
    anims = []
    for k, box in m.boxes.items():
        c = _base_color(k)
        anims.append(box[0].animate.set_stroke(c, width=1.6, opacity=1.0).set_fill(c, opacity=0.06))
        anims.append(box[1].animate.set_opacity(1.0))
    anims.append(m.arrows.animate.set_opacity(1.0))
    anims += [fade_arrow(a, 1.0) for a in m.feedback]
    anims.append(m.labels.animate.set_opacity(1.0))
    return AnimationGroup(*anims)


def edge_arrow(m: ArchitectureMap, a: str, b: str):
    from .archmap import EDGES
    return m.arrows[EDGES.index((a, b))]


# ------------------------------------------------------------------ glyphs

def glow_dot(color=WHITE, radius: float = 0.07) -> VGroup:
    """A small glowing pulse: a bright core inside two soft halos."""
    return VGroup(Dot(radius=radius * 3.2, color=color, fill_opacity=0.12),
                  Dot(radius=radius * 1.9, color=color, fill_opacity=0.30),
                  Dot(radius=radius, color=WHITE, fill_opacity=1.0))


def clipped_runs(x0: float, y0: float, w: int, h: int, pf: PixelFrame, vids=None, width: float = 4.0,
                 step: int = 2) -> VGroup:
    """The truth runs inside a window (optionally only the given vessels),
    each cut to the window, in identity colours.  pf is the window's frame."""
    g = VGroup()
    for r in scene().truth["runs"]:
        if vids is not None and r["vid"] not in vids:
            continue
        xy = np.asarray(r["xy"], float) - [x0, y0]
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
        if inside.sum() < 4:
            continue
        idx = np.flatnonzero(inside)
        for piece in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(piece) < 4:
                continue
            m = VMobject()
            m.set_points_smoothly([pf.point(x, y) for x, y in xy[piece][::step]])
            g.add(m.set_stroke(D.identity_color(r["vid"]), width=width, opacity=0.95))
    return g


def rings_in(x0: float, y0: float, w: int, h: int, pf: PixelFrame, width: float = 3.0,
             rmin: float = 5.0, rmax: float = 12.0) -> VGroup:
    """The observable junctions inside a window whose rings fit inside it,
    as rings by type."""
    g = VGroup()
    for j in scene().truth["junctions"]:
        x, y = j["x"] - x0, j["y"] - y0
        rp = float(np.clip(j["radius"], rmin, rmax))
        if rp - 1 <= x <= w - rp and rp - 1 <= y <= h - rp:
            grp = group_of(j["type_observable"])
            r = rp * pf.unit
            g.add(Circle(radius=r, stroke_color=group_color(grp), stroke_width=width).move_to(pf.point(x, y)))
    return g


# ------------------------------------------------------------------ pacing

class PacedCaptions:
    """Mixin for a ChapterScene: before a caption is replaced or removed, wait
    out whatever is left of its reading time (chapter.reading_time), so every
    caption is on screen at least that long, however fast the animations
    under it run.  Use as `class X(PacedCaptions, ChapterScene)`."""

    def _settle(self):
        need = getattr(self, "_cap_need", 0.0)
        if need and getattr(self, "_cap", None) is not None:
            left = need - (self.renderer.time - self._cap_t0)
            if left > 0.05:
                self.wait(left)
        self._cap_need = 0.0

    def say(self, text, wait=0.0):
        from .chapter import reading_time
        self._settle()
        t0 = self.renderer.time
        super().say(text, wait)
        self._cap_t0, self._cap_need = t0, reading_time(text)

    def unsay(self):
        self._settle()
        super().unsay()

    def outro(self):
        self._settle()
        super().outro()
