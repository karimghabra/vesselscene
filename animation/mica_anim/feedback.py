"""Top-down feedback (spec 4.11) on a toy fork: the computations behind chapter 15.

A fork whose third arm is faint, just under the seed threshold; dark texture and
white noise elsewhere.  The graph found so far (the parent and the strong
daughter) is rendered; the residual err = max(OD - R, 0) / noise shows what the
graph does not explain: the missing arm, and texture.  The gain
U0' = U0 (1 + gamma_fb blur(err, 1) near) is applied only near the junction
cell (within 2 R_j) and ahead of free ends (3 w), or, for the contrast,
everywhere (the earlier whole-image re-proposal).  Then line evidence is
thresholded with hysteresis (seeds at t_high, growth down to t_low).

Everything is the toy model's, not MICA's layers.  Coordinates are image
pixels (x right, y down).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from . import neuro as N

SHAPE = (120, 200)
J = np.array([92.0, 64.0])                  # the fork
PARENT = (np.array([-20.0, 76.0]), J, 3.2, 0.13)
STRONG = (J, np.array([220.0, 18.0]), 2.6, 0.11)
FAINT_END = np.array([196.0, 112.0])
FAINT = (J, FAINT_END, 2.0, 0.028)
NOISE = 0.01
BLOBS = [((40, 61), 3.5, 0.032), ((150, 34.5), 3.2, 0.032), ((10, 82), 3.5, 0.03), ((40, 20), 4.0, 0.035),
         ((30, 106), 5.0, 0.03), ((176, 64), 3.5, 0.03)]      # dark texture blobs (centre, sigma, OD)
GAMMA_FB = 0.5
R_J = 9.0                                   # the junction cell's radius (px)
T_HIGH, T_LOW = 16.0, 8.0                   # seed (the spec's tracking start, 2.4) and growth thresholds, noise units


@lru_cache(maxsize=None)
def scene():
    """OD (noisy), the clean vessel OD, the texture, and the render of the graph so far."""
    shape = SHAPE
    vessels = N.vessel_od(shape, [PARENT, STRONG, FAINT], blur=1.0, combine="max")
    rng = np.random.default_rng(21)
    # dark texture: smooth random field, positive part, plus a few blobs (vessel-like widths)
    f = ndi.gaussian_filter(rng.normal(0, 1, shape), 3.0)
    f = np.maximum(f / f.std() - 1.2, 0) * 0.016
    blobs = sum(N.blob_od(shape, c, s, a) for c, s, a in BLOBS)
    texture = f + blobs
    od = N.add_noise(vessels + texture, NOISE, seed=4)
    R = N.vessel_od(shape, [PARENT, STRONG], blur=1.0, combine="max")
    return dict(od=od, vessels=vessels, texture=texture, R=R)


@lru_cache(maxsize=None)
def layers():
    """U0: the 16 orientation layers' rates in noise units (the toy's step-5 input)."""
    s = scene()
    nz = N.simple_cells(N.add_noise(np.zeros((160, 160)), NOISE, seed=99), 2.0, 3.0)
    sd = np.array([u[30:-30, 30:-30].std() for u in nz])
    U = N.simple_cells(s["od"], 2.0, 3.0) / sd[:, None, None]
    return np.maximum(U, 0.0)


def residual():
    s = scene()
    return np.maximum(s["od"] - s["R"], 0.0) / NOISE


def near_mask(free_ends=()):
    """Within 2 R_j of the junction cell, or within 3 w ahead of a free end ((x, y, dx, dy, w))."""
    x, y = N.grid(SHAPE)
    m = np.hypot(x - J[0], y - J[1]) <= 2 * R_J
    for ex, ey, dx, dy, w in free_ends:
        t = (x - ex) * dx + (y - ey) * dy
        n = np.abs(-(x - ex) * dy + (y - ey) * dx)
        m |= (t >= 0) & (t <= 3 * w) & (n <= 1.5 * w)
    return m


def gained(near):
    """U0' = U0 (1 + gamma_fb blur(err, 1) near)."""
    g = 1.0 + GAMMA_FB * ndi.gaussian_filter(residual(), 1.0) * near
    return layers() * g[None], g


def evidence(U):
    return U.max(axis=0)


def hysteresis(E, seeds_allowed=None, exclude=None):
    """Components of E > T_LOW containing a seed (E > T_HIGH, where seeds are allowed)."""
    lo = E > T_LOW
    hi = E > T_HIGH
    if seeds_allowed is not None:
        hi &= seeds_allowed
    lab, n = ndi.label(lo)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[hi])] = True
    keep[0] = False
    m = keep[lab]
    return m


def free(E):
    """Evidence outside what the graph already holds (its render's support, dilated):
    a trace follows one path, so a missing arm needs a seed of its own."""
    known = ndi.binary_dilation(scene()["R"] > 0.02, iterations=3)
    return np.where(known, 0.0, E), known


@lru_cache(maxsize=None)
def runs():
    """The three outcomes: no feedback, the local gain (near only; no new seed
    outside it), and the global re-proposal (gain and seeds everywhere)."""
    s = scene()
    U0 = layers()
    near = near_mask()
    E0, known = free(evidence(U0))
    Ul, gl = gained(near)
    El, _ = free(evidence(Ul))
    Ug, gg = gained(np.ones(SHAPE, bool))
    Eg, _ = free(evidence(Ug))
    out = dict(E={"none": E0, "local": El, "global": Eg}, known=known, near=near, gain_local=gl, gain_global=gg)
    for name, E, allowed in (("none", E0, None), ("local", El, near), ("global", Eg, None)):
        m = hysteresis(E, seeds_allowed=allowed)
        lab, n = ndi.label(m)
        pieces = []
        for i in range(1, n + 1):
            pm = lab == i
            if pm.sum() < 6:
                continue
            on_faint = (s["vessels"][pm] > 0.01).mean() > 0.5
            touches = (ndi.binary_dilation(pm, iterations=2) & known).any()
            ys, xs = np.nonzero(pm)
            pieces.append(dict(mask=pm, true=bool(on_faint), touches=bool(touches),
                               centre=(float(xs.mean()), float(ys.mean())), size=int(pm.sum())))
        out[name] = dict(mask=m, pieces=pieces)
    return out


def branch_profile(step=1.0):
    """Line evidence along the faint arm (distance from the fork, px): without
    feedback, with the local gain, with the global gain (the graph's own support masked)."""
    r = runs()
    L = float(np.hypot(*(FAINT_END - J)))
    d = np.arange(0.0, L - 6, step)
    u = (FAINT_END - J) / L
    xs, ys = J[0] + d * u[0], J[1] + d * u[1]
    out = {k: ndi.map_coordinates(r["E"][k], [ys, xs], order=1) for k in ("none", "local", "global")}
    out["d"] = d
    return out


def mask_rgba(mask, color, alpha=0.9):
    """An RGBA overlay of a boolean mask (colour where True, transparent elsewhere)."""
    from manim import ManimColor
    rgb = np.array(ManimColor(color).to_rgb())
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[..., :3] = (rgb * 255).astype(np.uint8)
    out[..., 3] = (mask * alpha * 255).astype(np.uint8)
    return out
