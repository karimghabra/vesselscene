"""vesselscene imagery for the animation (made by tools/make_vs_data.py).

    S = load()                     # the default scene
    S.average, S.od, S.burst       # float arrays; burst is (n, h, w)
    S.view(S.average)              # display array: flattened ln I, vessels dark
    S.truth["runs"], S.truth["junctions"]

Overlay helpers draw the observable truth on a PixelFrame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import BLUE_B, PURPLE_B, VGroup, VMobject, Circle, ManimColor

from .style import JUNC

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
DEFAULT = "healthy_s000_960x540"

# junction type colours: forks pink, crossings blue, compounds purple
JUNCTION_COLORS = {"bifurcation": JUNC, "confluence": JUNC, "pseudo-T": JUNC, "crossing": BLUE_B, "compound": PURPLE_B}


def identity_color(vid: int) -> ManimColor:
    """A distinct colour per vessel identity (golden-ratio hues)."""
    import colorsys
    h = (vid * 0.6180339887) % 1.0
    return ManimColor.from_rgb(colorsys.hsv_to_rgb(h, 0.6, 1.0))


@dataclass
class VSScene:
    tag: str
    average: np.ndarray
    od: np.ndarray
    centreline: np.ndarray
    burst: np.ndarray | None
    truth: dict

    @property
    def shape(self):
        return self.average.shape

    @staticmethod
    def view(img, lo_pct=1.0, hi_pct=99.7):
        """Flattened log image for display: ln I minus its smooth background,
        stretched between percentiles (NaN -> white, like a stabilised mean)."""
        from scipy.ndimage import gaussian_filter
        I = np.asarray(img, float)
        ok = np.isfinite(I) & (I > 0)
        L = np.where(ok, np.log(np.where(ok, I, 1.0)), 0.0)
        w = gaussian_filter(ok.astype(float), 40) + 1e-9
        bg = gaussian_filter(np.where(ok, L, 0.0), 40) / w
        F = L - bg
        lo, hi = np.percentile(F[ok], [lo_pct, hi_pct])
        out = np.clip((F - lo) / (hi - lo), 0, 1)
        out[~ok] = 1.0
        return out


@lru_cache(maxsize=None)
def load(tag: str = DEFAULT) -> VSScene:
    a = np.load(CACHE / f"{tag}_average.npz")
    bp = CACHE / f"{tag}_burst.npz"
    burst = np.load(bp)["frames"] if bp.exists() else None
    truth = json.loads((CACHE / f"{tag}_truth.json").read_text())
    return VSScene(tag, a["image"], a["od"], a["centreline"], burst, truth)


def available(tag: str = DEFAULT) -> bool:
    return (CACHE / f"{tag}_average.npz").exists() and (CACHE / f"{tag}_truth.json").exists()


def run_path(run, pf, step=3):
    """A run's polyline as a smooth VMobject on the PixelFrame pf."""
    xy = np.asarray(run["xy"], float)[::step]
    if len(xy) < 2:
        xy = np.asarray(run["xy"], float)
    pts = [pf.point(x, y) for x, y in xy]
    m = VMobject()
    m.set_points_smoothly(pts) if len(pts) > 2 else m.set_points_as_corners(pts)
    return m


def runs_overlay(truth, pf, min_len_px=25, width=2.5, by="identity"):
    """Every observable run, coloured by vessel identity."""
    g = VGroup()
    for r in truth["runs"]:
        if len(r["xy"]) < min_len_px:
            continue
        col = identity_color(r["vid"])
        g.add(run_path(r, pf).set_stroke(col, width=width, opacity=0.95))
    return g


def junctions_overlay(truth, pf, width=2.5, min_radius_px=6.0):
    """A ring per observable junction, coloured by its observable type."""
    g = VGroup()
    for j in truth["junctions"]:
        col = JUNCTION_COLORS.get(j["type_observable"], JUNC)
        r = max(j["radius"], min_radius_px) * pf.unit
        g.add(Circle(radius=r, stroke_color=col, stroke_width=width).move_to(pf.point(j["x"], j["y"])))
    return g
