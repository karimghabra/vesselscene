"""Good continuation (spec 4.4, 4.10b): the straight turn rule, co-circularity,
and the toy vessel ends they are tested on.

Geometry is done in a "math" frame: pixels, x right, y UP, origin wherever the
caller puts it.  Images are rendered by mapping that frame to image pixels
(y down) with `Toy.img(p)`, and to the scene with `Toy.scene(pf, p)`, so every
angle drawn on screen has the sign a viewer reads (counter-clockwise positive).
The co-circular sum is unchanged by that reflection (both angles flip).

    angle(u, v)            atan2(u x v, u . v), in radians
    cocircular(a, ha, b, hb)  phi_a, phi_b, their sum, turn, radius r, chord d
    straight_score(...)    exp(-turn^2 / 2 tol^2) * width term   (first rule)
    cocircular_score(...)  exp(-(phi_a+phi_b)^2 / 2 tol^2) * radius penalty * width term
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from . import neuro as N

TOL_DEG = 20.0            # spec 4.4: 20 degrees (on the turn; now on the asymmetry phi_a + phi_b)
R_MIN = 15.0              # spec 4.4: r_min ~ 15 px, the tightest arc cell (4.8)
SIGMA_K = 1.0 / 30.0      # toy's choice: width of the soft penalty on curvature beyond 1 / r_min (px^-1)
LN15 = np.log(1.5)        # spec 4.4: width tolerance ln 1.5


# ------------------------------------------------------------------ geometry

def unit(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


def rot(v, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def angle(u, v) -> float:
    """Signed angle from u to v: atan2(u x v, u . v) (spec 4.4)."""
    return float(np.arctan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))


def cocircular(a, ha, b, hb) -> dict:
    """The spec's co-circularity geometry for ends a, b with outgoing headings
    ha, hb (out of each vessel, into the gap)."""
    a, b, ha, hb = (np.asarray(v, float) for v in (a, b, ha, hb))
    c = b - a
    d = float(np.hypot(*c))
    pa, pb = angle(c, ha), angle(-c, hb)
    turn = abs(pa - pb)
    r = d / (2 * np.sin(turn / 2)) if turn > 1e-9 else np.inf
    return dict(phi_a=pa, phi_b=pb, sum=pa + pb, turn=turn, r=r, d=d)


def straight_turn(ha, hb) -> float:
    """First rule: the angle between u_a and -u_b (0 for a straight continuation)."""
    return abs(angle(np.asarray(ha, float), -np.asarray(hb, float)))


def width_term(wa=1.0, wb=1.0) -> float:
    return float(np.exp(-np.log(wa / wb) ** 2 / (2 * LN15 ** 2)))


def turn_term(theta_rad, tol_deg=TOL_DEG) -> float:
    return float(np.exp(-np.rad2deg(theta_rad) ** 2 / (2 * tol_deg ** 2)))


def radius_penalty(r, r_min=R_MIN, sigma_k=SIGMA_K) -> float:
    """Soft penalty for r < r_min (the toy's form: a Gaussian in curvature
    beyond 1 / r_min; 1 for any r >= r_min)."""
    k = 0.0 if not np.isfinite(r) or r <= 0 else 1.0 / r
    ex = max(0.0, k - 1.0 / r_min)
    return float(np.exp(-ex ** 2 / (2 * sigma_k ** 2)))


def straight_score(ha, hb, wa=1.0, wb=1.0, tol_deg=TOL_DEG) -> float:
    return turn_term(straight_turn(ha, hb), tol_deg) * width_term(wa, wb)


def cocircular_score(a, ha, b, hb, wa=1.0, wb=1.0, tol_deg=TOL_DEG) -> float:
    g = cocircular(a, ha, b, hb)
    return turn_term(g["sum"], tol_deg) * radius_penalty(g["r"]) * width_term(wa, wb)


def arc(p, t, kappa, s):
    """Points along the circle through p with unit tangent t and signed
    curvature kappa (counter-clockwise positive), at arc lengths s:
    p + (sin(kappa s) / kappa) t + ((1 - cos(kappa s)) / kappa) n   (spec 4.8)."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    n = np.array([-t[1], t[0]])
    s = np.atleast_1d(np.asarray(s, float))
    if abs(kappa) < 1e-9:
        return p[None] + s[:, None] * t[None]
    return p[None] + (np.sin(kappa * s) / kappa)[:, None] * t[None] + ((1 - np.cos(kappa * s)) / kappa)[:, None] * n[None]


def circle_through(a, ha, b):
    """The circle tangent to ha at a that passes through b: its signed
    curvature, the arc length from a to b, and its tangent at b (pointing
    back into the gap, i.e. the heading b would need to be co-circular)."""
    a, ha, b = (np.asarray(v, float) for v in (a, ha, b))
    c = b - a
    d = float(np.hypot(*c))
    psi = angle(ha, c)                      # from the heading to the chord
    kappa = 2 * np.sin(psi) / d
    s_b = d if abs(psi) < 1e-9 else psi * d / np.sin(psi)
    tb_forward = rot(ha, np.rad2deg(kappa * s_b))
    return kappa, s_b, -tb_forward


def fit_circle(pts):
    """Least-squares (Kasa) circle through points: centre, radius."""
    pts = np.asarray(pts, float)
    A = np.c_[2 * pts, np.ones(len(pts))]
    y = (pts ** 2).sum(1)
    cx, cy, k = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([cx, cy]), float(np.sqrt(k + cx ** 2 + cy ** 2))


# ------------------------------------------------------------------ toy images

def _dist_to_polyline(x, y, pts):
    d = np.full(x.shape, np.inf)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, N.dist_to_segment(x, y, p0, p1)[0])
    return d


@dataclass
class Toy:
    """A toy stimulus in the math frame (px, y up) and its OD image."""
    shape: tuple                      # (h, w) of the image
    origin: np.ndarray                # image pixel of the math frame's origin
    pieces: list = field(default_factory=list)   # [(polyline points (n, 2) in math frame, width px)]
    amp: float = 0.15
    blur: float = 1.0
    noise: float = 0.008
    seed: int = 3
    combine: str = "max"

    def img(self, p):
        p = np.asarray(p, float)
        return np.array([self.origin[0] + p[..., 0], self.origin[1] - p[..., 1]]).T if p.ndim > 1 else \
            np.array([self.origin[0] + p[0], self.origin[1] - p[1]])

    def scene(self, pf, p):
        q = self.img(p)
        return pf.point(q[0], q[1])

    def od(self):
        x, y = N.grid(self.shape)
        layers = []
        for pts, w in self.pieces:
            q = self.img(np.asarray(pts, float))
            layers.append(N.chord(_dist_to_polyline(x, y, q), w / 2, self.amp))
        od = np.max(layers, axis=0) if self.combine == "max" else np.sum(layers, axis=0)
        od = gaussian_filter(od, self.blur)
        return N.add_noise(od, self.noise, self.seed) if self.noise > 0 else od


def gap_pair(kind: str, d=12.0, deg=None):
    """Ends a, b (gap midpoint at the origin, chord horizontal) and their
    outgoing headings for the three test cases:

      'bend'  both ends cut from one circle (spec's check: r 30 px, d 12 px)
      's'     an S-bend: headings parallel, both tilted the same way from the chord
      'kink'  a points straight at b, b leaves at an angle
    """
    a, b = np.array([-d / 2, 0.0]), np.array([d / 2, 0.0])
    if kind == "bend":
        r = 30.0 if deg is None else deg
        half = np.rad2deg(np.arcsin(d / (2 * r)))
        return a, unit(half), b, unit(180 - half)
    if kind == "s":
        phi = 20.0 if deg is None else deg
        return a, unit(phi), b, unit(180 + phi)
    if kind == "kink":
        phi = 40.0 if deg is None else deg
        return a, unit(0.0), b, unit(180 - phi)
    raise ValueError(kind)


def end_piece(p, h, length, kappa=0.0, step=0.5):
    """A vessel piece ending at p with outgoing heading h: it comes from
    behind along a circle of curvature kappa (0 = straight) and stops at p.
    Returned from its far end to p."""
    s = np.arange(0.0, length + 1e-9, step)
    pts = arc(p, -np.asarray(h, float), -kappa, s)      # walk backwards from the end
    return pts[::-1]


def gap_pieces(kind: str, L=22.0):
    """The two vessel pieces behind the ends of gap_pair(kind), in the math
    frame.  The bend is mirror-symmetric (one circle of radius 30 px); the
    S-bend is point-symmetric about the gap's midpoint (lobes of radius
    20 px outside the gap, so the curve inflects inside it); the kink is two
    straight pieces."""
    a, ha, b, hb = gap_pair(kind)
    if kind == "bend":
        pa = end_piece(a, ha, L, -1 / 30.0)                    # approaching a, the circle turns clockwise
        return pa, pa * np.array([-1.0, 1.0])                  # b's piece: the mirror image
    if kind == "s":
        pa = end_piece(a, ha, L, -1 / 20.0)
        return pa, -pa                                         # b's piece: a rotated by 180 degrees
    return end_piece(a, ha, L), end_piece(b, hb, L)


@lru_cache(maxsize=None)
def case_toy(kind: str):
    """Small OD images for the bend / S-bend / kink comparison."""
    a, ha, b, hb = gap_pair(kind)
    pa, pb = gap_pieces(kind, 17.0 if kind == "s" else 24.0)
    toy = Toy(shape=(36, 48), origin=np.array([23.5, 17.5]), pieces=[(pa, 5.0), (pb, 5.0)], noise=0.008, seed=7)
    return toy, toy.od(), dict(a=a, ha=ha, b=b, hb=hb, pa=pa, pb=pb)


@lru_cache(maxsize=None)
def bend_toy():
    """The level-1 stimulus: a vessel along a circle of radius 30 px with a
    12 px gap at its apex.  Math frame: gap midpoint at the origin."""
    r, d = 30.0, 12.0
    half = np.arcsin(d / (2 * r))
    centre = np.array([0.0, -r * np.cos(half)])
    th_a, th_b = np.pi / 2 + half, np.pi / 2 - half      # the arc from the lower left up to a, and from b down
    ta = np.linspace(np.pi + 0.35, th_a, 260)
    tb = np.linspace(th_b, -0.35, 260)
    A = centre[None] + r * np.c_[np.cos(ta), np.sin(ta)]
    B = centre[None] + r * np.c_[np.cos(tb), np.sin(tb)]
    toy = Toy(shape=(40, 66), origin=np.array([32.5, 8.5]), pieces=[(A, 6.0), (B, 6.0)], noise=0.008, seed=5)
    a, ha, b, hb = gap_pair("bend")
    return toy, toy.od(), dict(a=a, ha=ha, b=b, hb=hb, centre=centre, r=r)


@lru_cache(maxsize=None)
def bend_crop(rows=20, c0=15, cols=36):
    """The same image, cropped around the gap (the zoomed main view); the
    pixels are the full image's, so the two views agree exactly."""
    toy, od, g = bend_toy()
    crop = Toy(shape=(rows, cols), origin=toy.origin - np.array([c0, 0.0]), pieces=toy.pieces, noise=toy.noise,
               seed=toy.seed)
    return crop, od[:rows, c0:c0 + cols], g


def case_scores(kind: str) -> dict:
    a, ha, b, hb = gap_pair(kind)
    g = cocircular(a, ha, b, hb)
    return dict(turn=np.rad2deg(straight_turn(ha, hb)), sum=np.rad2deg(g["sum"]),
                straight20=straight_score(ha, hb), straight40=straight_score(ha, hb, tol_deg=40.0),
                cocirc=cocircular_score(a, ha, b, hb), r=g["r"])


def piece_rgba(toy: Toy, k: int, vmin=-0.02, vmax=0.2, alpha_lo=0.008, alpha_hi=0.05):
    """One vessel piece of a toy alone, noise-free, as an RGBA image on a
    transparent background (magma, like the OD views), so it can be moved."""
    from .style import colormap
    x, y = N.grid(toy.shape)
    pts, w = toy.pieces[k]
    od = gaussian_filter(N.chord(_dist_to_polyline(x, y, toy.img(np.asarray(pts, float))), w / 2, toy.amp), toy.blur)
    rgba = colormap(od, "magma", vmin, vmax)
    rgba[..., 3] = (np.clip((od - alpha_lo) / (alpha_hi - alpha_lo), 0, 1) * 255).astype(np.uint8)
    return rgba


# ------------------------------------------------------------------ junction arms

def junction_pairs(arm_deg, widths, R):
    """Arms of one junction as ends on its disc (radius R), headings pointing
    in.  Returns {(i, j): (score, r, sum_deg)} for every pair."""
    out = {}
    n = len(arm_deg)
    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = unit(arm_deg[i]), unit(arm_deg[j])
            a, b = R * ui, R * uj
            g = cocircular(a, -ui, b, -uj)
            s = cocircular_score(a, -ui, b, -uj, widths[i], widths[j])
            out[(i, j)] = (s, g["r"], np.rad2deg(g["sum"]))
    return out


def pairings4(pairs):
    """The three ways to pair four arms into two lines, with the product of cont."""
    ps = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    return [(p, pairs[p[0]][0] * pairs[p[1]][0]) for p in ps]


@lru_cache(maxsize=None)
def junction_toy(kind: str):
    """A crossing (4 arms, two depths: ODs add) or a fork (3 arms, one lumen:
    union), as half-lines from the centre.  Math frame: centre at the origin."""
    if kind == "crossing":
        arms = (20.0, 95.0, 200.0, 275.0)            # two straight lines crossing at 75 degrees
        widths = (6.0, 4.5, 6.0, 4.5)
        amps = (0.15, 0.11, 0.15, 0.11)
        combine = "sum"
    else:
        arms = (195.0, 20.0, 290.0)                  # parent 195 -> 20 (a 5 degree bend), branch at 290
        widths = (7.0, 6.5, 4.6)
        amps = (0.16, 0.155, 0.11)
        combine = "max"
    L = 30.0
    x, y = N.grid((56, 56))
    origin = np.array([27.5, 27.5])
    layers = []
    for deg, w, amp in zip(arms, widths, amps):
        u = unit(deg)
        p1 = origin + np.array([u[0], -u[1]]) * L
        layers.append(N.chord(N.dist_to_segment(x, y, origin, p1)[0], w / 2, amp))
    if combine == "sum":
        od = np.maximum(layers[0], layers[2]) + np.maximum(layers[1], layers[3]) * 0.87
    else:
        od = np.max(layers, axis=0)
    od = N.add_noise(gaussian_filter(od, 1.0), 0.008, 11)
    return dict(arms=arms, widths=widths, od=od, origin=origin, shape=(56, 56))


def half_height_width(profile, xs):
    """Width at half height of a 1-D OD profile against its baseline (the
    median of the outer quarter on each side)."""
    n = len(profile)
    base = np.median(np.r_[profile[: n // 8], profile[-n // 8:]])
    p = profile - base
    k = int(np.argmax(p))
    half = p[k] / 2
    i = k
    while i > 0 and p[i] > half:
        i -= 1
    j = k
    while j < n - 1 and p[j] > half:
        j += 1
    xl = xs[i] + (half - p[i]) / (p[i + 1] - p[i]) * (xs[i + 1] - xs[i])
    xr = xs[j - 1] + (half - p[j - 1]) / (p[j] - p[j - 1]) * (xs[j] - xs[j - 1])
    return xr - xl, base + half, xl, xr


@lru_cache(maxsize=None)
def width_profiles(blur=1.0):
    """Chord-law profiles of vessels 21 and 5 px wide, blurred, and their
    half-height widths (toy)."""
    xs = np.arange(-40, 40.001, 0.25)
    out = {}
    for w in (21.0, 5.0):
        prof = N.chord(np.abs(xs), w / 2, 0.15)
        prof = gaussian_filter(prof, blur / 0.25)
        out[w] = (xs, prof, half_height_width(prof, xs))
    return out
