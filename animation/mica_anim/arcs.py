"""Arc cells (spec 4.8), on toy rings.

Short simple cells (elongation 1.2) are pooled along circles: the arc cell
(k, kappa) at p is the Gaussian-weighted mean of the short cells' drive along
the circle through p tangent to orientation k with curvature kappa,

    sample at arc length s:  p + (sin(kappa s)/kappa) t_k + ((1 - cos(kappa s))/kappa) n_k
    each sample reads the sheet of the local tangent theta_k + kappa s (interpolated)
    weights exp(-s^2 / 2 sigma_s^2),  sigma_s = sigma sqrt(e_long^2 - 1.2^2)

built, as the spec says, as small per-sheet kernels (one correlation per sheet
a path touches).  Each cell type is divided by its own background spread.
Image coordinates throughout: x right, y down; t_k = (cos, sin), n_k = (-sin, cos).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from . import neuro as N

SIGMA = 2.1                                   # fine channel: sigma_s = 2.1 sqrt(9 - 1.44) = 5.8 px (spec 6.3)
E_LONG, E_SHORT = 3.0, 1.2
SIGMA_S = SIGMA * np.sqrt(E_LONG ** 2 - E_SHORT ** 2)
KAPPAS = np.array([0.0, 1 / 60, -1 / 60, 1 / 30, -1 / 30, 1 / 15, -1 / 15])
RADII = (60, 30, 15)
STEP = np.pi / N.N_ORIENT

# the reference test, laid out in one image: three rings and a straight vessel (6 px wide, peak OD 0.15)
SHAPE = (170, 372)
RINGS = {60: (86.0, 85.0), 30: (210.0, 85.0), 15: (290.0, 85.0)}
STRAIGHT = ((345.0, 12.0), (345.0, 158.0))
R_VES, A_VES = 3.0, 0.15


def path(p, k_theta, kappa, s):
    """Sample points (x, y) along the arc through p, tangent to k_theta, curvature kappa, at arc lengths s."""
    t = np.array([np.cos(k_theta), np.sin(k_theta)])
    n = np.array([-np.sin(k_theta), np.cos(k_theta)])
    s = np.asarray(s, float)
    if abs(kappa) < 1e-9:
        a, b = s, np.zeros_like(s)
    else:
        a, b = np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa
    return np.asarray(p, float)[None, :] + a[:, None] * t[None, :] + b[:, None] * n[None, :]


def samples(smax_sd=3.0, ds=0.5):
    s = np.arange(-smax_sd * SIGMA_S, smax_sd * SIGMA_S + 1e-9, ds)
    return s, np.exp(-s ** 2 / (2 * SIGMA_S ** 2))


def arc_kernels(k, kappa):
    """Per-sheet kernels {sheet j: 2-D weights} for arc cell (k, kappa); correlate U[j] with them and sum."""
    s, w = samples()
    w = w / w.sum()
    th = N.THETAS[k]
    pts = path((0.0, 0.0), th, kappa, s)
    R = int(np.ceil(np.abs(pts).max())) + 2
    jf = (th + kappa * s) / STEP
    j0 = np.floor(jf).astype(int)
    fj = jf - j0
    ks = {}
    for (x, y), wi, a, f in zip(pts, w, j0, fj):
        for j, wj in ((a % N.N_ORIENT, wi * (1 - f)), ((a + 1) % N.N_ORIENT, wi * f)):
            if wj < 1e-6:
                continue
            K = ks.setdefault(j, np.zeros((2 * R + 1, 2 * R + 1)))
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - x0, y - y0
            for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                K[R + y0 + dy, R + x0 + dx] += wj * ww
    return ks


def arc_cell(U, k, kappa):
    out = np.zeros(U.shape[1:])
    for j, K in arc_kernels(k, kappa).items():
        out += fftconvolve(U[j], K[::-1, ::-1], mode="same")
    return out


def _bg_sd(stack, bg):
    return np.array([u[bg].std() for u in stack])


@lru_cache(maxsize=None)
def reference(seed: int = 5):
    """The reference test on the toy rings: every cell type's maps, in its own noise units."""
    x, y = N.grid(SHAPE)
    od = N.vessel_od(SHAPE, [(STRAIGHT[0], STRAIGHT[1], R_VES, A_VES)], blur=1.0)
    for r, c in RINGS.items():
        od = od + N.ring_od(SHAPE, c, r, R_VES, A_VES, blur=1.0)
    clean = od
    od = N.add_noise(clean, 0.01, seed)
    near = np.zeros(SHAPE, bool)
    for r, c in RINGS.items():
        near |= np.abs(np.hypot(x - c[0], y - c[1]) - r) < 26
    near |= np.abs(x - STRAIGHT[0][0]) < 26
    bg = ~near
    bg[:12] = bg[-12:] = False
    bg[:, :12] = bg[:, -12:] = False

    long_ = N.simple_cells(od, SIGMA, E_LONG)
    long_ = long_ / _bg_sd(long_, bg)[:, None, None]
    short_drive = N.simple_cells(od, SIGMA, E_SHORT)
    short = short_drive / _bg_sd(short_drive, bg)[:, None, None]
    # arc cells from the short cells' drive (noise units), then renormalised per (k, kappa)
    arcs = np.zeros((len(KAPPAS), N.N_ORIENT) + SHAPE)
    for i, kap in enumerate(KAPPAS):
        for k in range(N.N_ORIENT):
            a = arc_cell(short, k, kap)
            arcs[i, k] = a / a[bg].std()
    best = arcs.max(axis=0)                       # per orientation, the best curvature wins
    K = KAPPAS[arcs.argmax(axis=0)]               # ... and its value is kept in a map K
    return dict(od=od, clean=clean, bg=bg, long=long_, short=short, straight_pool=arcs[0], arcs=arcs, best=best, K=K)


def centreline(name, n=360):
    """Centreline sample points of a ring (by radius) or of the straight vessel ('straight')."""
    if name == "straight":
        ys = np.linspace(STRAIGHT[0][1] + 25, STRAIGHT[1][1] - 25, n)
        return np.stack([np.full(n, STRAIGHT[0][0]), ys], axis=1)
    c = RINGS[name]
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + name * np.cos(ph), c[1] + name * np.sin(ph)], axis=1)


def sample(stack, pts):
    """Bilinear samples of every sheet of a (k, h, w) stack at points (x, y)."""
    return np.stack([map_coordinates(u, [pts[:, 1], pts[:, 0]], order=1) for u in stack])


@lru_cache(maxsize=None)
def table():
    """Median along the centreline of the best orientation's response (noise units) and,
    for the rings, as % of the same cells' straight response; the background's 99th percentile."""
    ref = reference()
    rows = {}
    for key, stack in (("long", ref["long"]), ("short", ref["short"]), ("pooled", ref["straight_pool"]), ("arcs", ref["best"])):
        st = float(np.median(sample(stack, centreline("straight")).max(axis=0)))
        rings = [100 * float(np.median(sample(stack, centreline(r)).max(axis=0))) / st for r in RADII]
        p99 = float(np.percentile(stack.max(axis=0)[ref["bg"]], 99))
        rows[key] = (st, *rings, p99)
    return rows


def winning_kappa(r, n=360):
    """The winning |kappa| along a ring's centreline (at the best orientation): its median."""
    ref = reference()
    pts = centreline(r, n)
    best_k = sample(ref["best"], pts).argmax(axis=0)
    Ks = sample(ref["K"], pts)                    # interpolated, so take the nearest pixel instead
    xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    kk = ref["K"][best_k, yi, xi]
    return float(np.median(np.abs(kk)))


def response_vs_kappa(U, p, k_theta, kappas):
    """An arc cell's (unnormalised) mean drive at one point, for a continuous range of curvatures,
    sampling the stack trilinearly (x, y and orientation)."""
    s, w = samples()
    w = w / w.sum()
    out = []
    for kap in kappas:
        pts = path(p, k_theta, kap, s)
        jf = (k_theta + kap * s) / STEP
        j0 = np.floor(jf).astype(int)
        f = jf - j0
        v0 = np.array([map_coordinates(U[j % N.N_ORIENT], [[y], [x]], order=1)[0] for (x, y), j in zip(pts, j0)])
        v1 = np.array([map_coordinates(U[(j + 1) % N.N_ORIENT], [[y], [x]], order=1)[0] for (x, y), j in zip(pts, j0)])
        out.append(float(np.sum(w * ((1 - f) * v0 + f * v1))))
    return np.array(out)
