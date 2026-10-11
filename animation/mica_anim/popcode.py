"""Population codes (spec 4.12) on the toy hypercolumn: the computations
behind chapter 16.

Angles here are SCREEN angles (counter-clockwise, y up), so the complex plane
drawn on screen and the vessel drawn beside it agree.  An image orientation
theta (y down) is the screen angle -theta; the 16 cells are re-indexed so that
cell j prefers the screen angle j * 11.25 degrees (image cell k = -j mod 16).

Everything is the toy model's (neuro.simple_cells: one size channel, white
noise in OD), not MICA's layers.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

from . import neuro as N

SHAPE = (96, 96)
CENTRE = np.array([48.0, 48.0])
PHI = np.arange(N.N_ORIENT) * 180.0 / N.N_ORIENT       # screen angle of cell j (degrees)
J2K = (-np.arange(N.N_ORIENT)) % N.N_ORIENT           # cell j -> neuro's image index k
SIGMA, ELONG = 2.0, 3.0
R_VES, A_VES, NOISE = 3.0, 0.15, 0.01


def image_theta(j):
    """neuro's image orientation (radians) of screen cell j, for orientation_color."""
    return N.THETAS[J2K[j]]


def wrap180(d):
    return (np.asarray(d) + 90.0) % 180.0 - 90.0


@lru_cache(maxsize=None)
def layer_sd(sigma=SIGMA, elong=ELONG):
    """Each orientation layer's spread over pure background noise (noise units, spec 4.0)."""
    nz = N.add_noise(np.zeros((160, 160)), NOISE, seed=99)
    U = N.simple_cells(nz, sigma, elong)
    return np.array([u[30:-30, 30:-30].std() for u in U])


def _dir(phi_deg):
    a = np.deg2rad(phi_deg)
    return np.array([np.cos(a), -np.sin(a)])          # image direction of a screen angle


def patch(phi_deg: float, noise: float = NOISE, seed: int = 3, offset: float = 0.0, shape=SHAPE):
    """A straight vessel at screen angle phi through the patch centre (shifted
    `offset` px along its normal), its noisy OD, and the 16 layers in noise units
    (re-indexed by screen cell j)."""
    c = np.array([shape[1] / 2.0, shape[0] / 2.0])
    u = _dir(phi_deg)
    n = np.array([u[1], -u[0]])                       # the screen-left normal (screen up for phi = 0)
    c = c + offset * n
    clean = N.vessel_od(shape, [(c - 200 * u, c + 200 * u, R_VES, A_VES)], blur=1.0)
    od = N.add_noise(clean, noise, seed) if noise > 0 else clean
    U = N.simple_cells(od, SIGMA, ELONG) / layer_sd()[:, None, None]
    return od, U[J2K]


def rates(phi_deg: float, **kw):
    """C_j at the patch centre: the 16 cells' rates max(0, U), noise units."""
    od, U = patch(float(phi_deg), **kw)
    h, w = od.shape
    return np.maximum(U[:, h // 2, w // 2], 0.0)


def pop_vector(C, mult=2.0):
    """sum_j C_j exp(i mult phi_j) as a complex number."""
    return (np.asarray(C) * np.exp(1j * mult * np.deg2rad(PHI))).sum()


def decode(C):
    """theta = 1/2 arg sum C_j e^{2 i phi_j} (degrees, 0..180)."""
    return float(np.rad2deg(np.angle(pop_vector(C))) / 2.0 % 180.0)


def winner(C):
    return float(PHI[int(np.argmax(C))])


@lru_cache(maxsize=None)
def sweep(a0=0.0, a1=45.0, step=0.25):
    """True angle against the winner's and the population vector's error, with
    each angle's OD patch (float32) and rates."""
    angles = np.arange(a0, a1 + 1e-9, step)
    ods, Cs = [], []
    for a in angles:
        od, U = patch(float(a))
        ods.append(od.astype(np.float32))
        Cs.append(np.maximum(U[:, SHAPE[0] // 2, SHAPE[1] // 2], 0.0))
    Cs = np.array(Cs)
    win = np.array([winner(C) for C in Cs])
    pop = np.array([decode(C) for C in Cs])
    return dict(angles=angles, C=Cs, od=np.array(ods), err_win=wrap180(win - angles),
                err_pop=wrap180(pop - angles), win=win, pop=pop)


# ------------------------------------------------------------------ position: a parabola through 3 samples

def parabola_vertex(cm, c0, cp):
    """Offset of the vertex of the parabola through (-1, cm), (0, c0), (1, cp)."""
    return 0.5 * (cm - cp) / (cm - 2.0 * c0 + cp)


@lru_cache(maxsize=None)
def ridge_profile(offset=0.3, half=4):
    """A horizontal vessel whose centre lies `offset` px above a pixel centre
    (screen up = image -y); the 0-degree cell's response sampled down one column.
    Returns positions (screen-up px, integers), C there, the vertex and the truth."""
    od, U = patch(0.0, offset=offset, shape=(64, 64))
    col = 32
    ys = np.arange(32 - half, 32 + half + 1)
    C = U[0, ys, col]
    pos = 32 - ys                                   # screen-up offset of each sample from row 32
    i0 = int(np.argmax(C))
    v = parabola_vertex(C[i0 - 1], C[i0], C[i0 + 1]) if 0 < i0 < len(C) - 1 else 0.0
    # vertex in screen-up px: samples ordered top (pos +half) .. bottom; index step +1 = pos -1
    vertex = pos[i0] - v
    return dict(od=od, U0=U[0], pos=pos, C=C, i0=i0, vertex=float(vertex), truth=float(offset), col=col)


# ------------------------------------------------------------------ width, blur, contrast: other populations

SCALES = np.geomspace(0.8, 6.0, 9)            # the 9 size channels' sigma (px)


def _profile_od(x, r, s, a=1.0):
    """The blurred-cylinder model across a vessel: a * chord(x; r) * Gaussian(s)."""
    xf = np.arange(-40.0, 40.0 + 1e-9, 0.05)
    ch = a * np.sqrt(np.clip(1.0 - (xf / r) ** 2, 0.0, None))
    if s > 0:
        k = np.exp(-0.5 * (np.arange(-8 * s, 8 * s + 1e-9, 0.05) / s) ** 2)
        ch = np.convolve(ch, k / k.sum(), mode="same")
    return np.interp(x, xf, ch)


def _line_resp(prof, x, sigma):
    """Scale-normalised line cell (negative second derivative of a Gaussian,
    sigma^2-normalised) at the centre of a 1-D profile sampled at x (step dx)."""
    dx = x[1] - x[0]
    g = (1 - x ** 2 / sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return float((prof * g).sum() * dx)


def _edge_resp(prof, x, sigma=0.8):
    """Fine edge cell (first derivative of a Gaussian, sigma-normalised): its peak along the profile."""
    d = np.gradient(gaussian_filter(prof, sigma / (x[1] - x[0])), x) * sigma
    return float(np.abs(d).max())


@lru_cache(maxsize=None)
def scale_calibration(s=1.0):
    """Each size channel's preferred half width: the r for which it is the best
    scale (noise-free blurred-cylinder model, blur s)."""
    x = np.arange(-40.0, 40.0 + 1e-9, 0.05)
    sg = np.geomspace(0.5, 10, 200)
    rr = np.geomspace(0.8, 9, 60)
    best = np.array([sg[np.argmax([_line_resp(_profile_od(x, r, s), x, q) for q in sg])] for r in rr])
    return np.interp(SCALES, best, rr)


def scale_population_from_profile(prof, x):
    """The 9 size channels at a vessel's centre (1-D profile across it).  The
    winner's preferred half width, and the population's peak scale: the vertex of
    the parabola through the top 3 responses, in log scale (as for position)."""
    resp = np.array([_line_resp(prof, x, sg) for sg in SCALES])
    pref = scale_calibration()
    i = int(np.clip(np.argmax(resp), 1, len(SCALES) - 2))
    v = parabola_vertex(resp[i - 1], resp[i], resp[i + 1])
    ls = np.log(SCALES)
    return dict(resp=resp, pref=pref, win=int(np.argmax(resp)), top=i, vertex=v,
                log_sigma=float(ls[i] + v * (ls[1] - ls[0])), r_win=float(pref[np.argmax(resp)]))


@lru_cache(maxsize=None)
def scale_population(r=3.5, s=1.2, a=0.14):
    x = np.arange(-14.0, 14.0 + 1e-9, 0.05)
    return scale_population_from_profile(_profile_od(x, r, s, a), x)


def _edge_line(prof, x):
    sp = scale_population_from_profile(prof, x)
    return sp, _edge_resp(prof, x) / sp["resp"].max()


@lru_cache(maxsize=None)
def rs_table():
    """The blurred-cylinder model's lookup table: for each (r, s), the
    population's peak scale and the edge / line ratio."""
    x = np.arange(-14.0, 14.0 + 1e-9, 0.05)
    rr = np.linspace(1.0, 7.0, 49)
    ss = np.linspace(0.4, 2.6, 34)
    F = np.zeros((len(rr), len(ss), 2))
    for i, r in enumerate(rr):
        for j, s in enumerate(ss):
            sp, ratio = _edge_line(_profile_od(x, r, s), x)
            F[i, j] = sp["log_sigma"], ratio
    return rr, ss, F


def lookup_rs(log_sigma, ratio):
    rr, ss, F = rs_table()
    D = ((F[..., 0] - log_sigma) / 0.05) ** 2 + ((F[..., 1] - ratio) / 0.02) ** 2
    i, j = np.unravel_index(np.argmin(D), D.shape)
    return float(rr[i]), float(ss[j])


def blur_curve(r):
    """The table's slice at half width r: edge / line ratio against s."""
    rr, ss, F = rs_table()
    i = int(np.argmin(np.abs(rr - r)))
    return ss, F[i, :, 1]


def centre_gain(r, s):
    """The model's OD at the centre for unit contrast: a = OD_centre / this."""
    return float(_profile_od(np.array([0.0]), r, s, 1.0)[0])


# ------------------------------------------------------------------ the fit: a toy spline network on the OD

FIT_SHAPE = (100, 170)
TRUE_CTRL = np.array([[-12, 72], [30, 76], [62, 30], [100, 20], [140, 64], [182, 70]], float)
TRUE_RSA = (3.5, 1.2, 0.14)          # half width r (px), blur s (px), contrast a (OD)
N_NODES = 9


def _bspline_basis(n_ctrl, t):
    """Clamped cubic B-spline basis (len(t), n_ctrl)."""
    from scipy.interpolate import BSpline
    k = 3
    knots = np.concatenate([[0] * k, np.linspace(0, 1, n_ctrl - k + 1), [1] * k])
    B = np.zeros((len(t), n_ctrl))
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        B[:, i] = BSpline(knots, c, k, extrapolate=False)(np.clip(t, 0, 1 - 1e-12))
    return B


@lru_cache(maxsize=None)
def true_curve(n=1500):
    return _bspline_basis(len(TRUE_CTRL), np.linspace(0, 1, n)) @ TRUE_CTRL


def node_curve(nodes, n=700, ext=24.0):
    """The model's centreline: a cubic through the nodes (chord-length parameter),
    extended straight past the end nodes (beyond the patch)."""
    from scipy.interpolate import CubicSpline
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    cs = CubicSpline(d, nodes, bc_type="natural")
    mid = cs(np.linspace(0, d[-1], n))
    t0 = cs(0, 1); t0 /= np.linalg.norm(t0)
    t1 = cs(d[-1], 1); t1 /= np.linalg.norm(t1)
    e = np.linspace(0, ext, 40)[:, None]
    return np.vstack([nodes[0] - e[::-1] * t0, mid, nodes[-1] + e * t1])


def _render_curve(pts, r, s, a, shape=FIT_SHAPE):
    from scipy.spatial import cKDTree
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    d, _ = cKDTree(pts).query(np.c_[x.ravel(), y.ravel()])
    od = a * np.sqrt(np.clip(1 - (d.reshape(shape) / max(r, 0.3)) ** 2, 0, None))
    return gaussian_filter(od, max(s, 0.05), mode="nearest")


@lru_cache(maxsize=None)
def fit_scene():
    clean = _render_curve(true_curve(), *TRUE_RSA)
    return N.add_noise(clean, NOISE, seed=5)


@lru_cache(maxsize=None)
def fit_cells():
    return N.simple_cells(fit_scene(), SIGMA, ELONG) / layer_sd()[:, None, None]


def true_points(n=N_NODES):
    """Where the grouped vessel is sampled: n points spread over the patch (pixel nearest the truth)."""
    pts = true_curve()
    ok = np.where((pts[:, 0] > 3) & (pts[:, 0] < FIT_SHAPE[1] - 4))[0]
    return pts[np.linspace(ok[0], ok[-1], n).astype(int)]


def measure_point(p, mode="population"):
    """Decode one point of the toy vessel from the noisy OD, starting at the pixel
    nearest it: tangent from the 16 cells (winner or population vector), position
    across the ridge (whole-pixel top or parabola vertex), and the profile across
    it (for r, s and a)."""
    od = fit_scene()
    U = fit_cells()
    base = np.round(p)
    tc = np.maximum(U[:, int(base[1]), int(base[0])], 0.0)[J2K]
    phi = winner(tc) if mode == "winner" else decode(tc)
    t = _dir(phi)
    nrm = np.array([-t[1], t[0]])
    offs = np.arange(-4, 5)
    sm = gaussian_filter(od, 1.0)
    smp = map_coordinates(sm, [base[1] + offs * nrm[1], base[0] + offs * nrm[0]], order=1)
    i0 = int(np.argmax(smp))
    v = parabola_vertex(smp[i0 - 1], smp[i0], smp[i0 + 1]) if (mode == "population" and 0 < i0 < len(offs) - 1) else 0.0
    q = base + (offs[i0] + v) * nrm
    x = np.arange(-14.0, 14.0 + 1e-9, 0.05)
    prof = np.mean([map_coordinates(od, [q[1] + x * nrm[1] + a * t[1], q[0] + x * nrm[0] + a * t[0]], order=1, mode="nearest")
                    for a in range(-3, 4)], axis=0)
    return dict(phi=phi, pos=q, tangent=t, normal=nrm, prof=prof, x=x, tc=tc)


def decode_rsa(prof, x):
    """r and s from the scale population's peak and the edge / line ratio, through
    the model's lookup table; a from the centre OD (the unnormalised pathway)."""
    sp, ratio = _edge_line(prof, x)
    r, s = lookup_rs(sp["log_sigma"], ratio)
    od_c = float(np.mean(prof[np.abs(x) <= 0.5]))
    return dict(sp=sp, ratio=ratio, r=r, s=s, od_c=od_c, g=centre_gain(r, s), a=od_c / centre_gain(r, s))


def geometry_error(nodes):
    """Mean distance (px) from the true centreline, inside the patch, to the model's."""
    from scipy.spatial import cKDTree
    tru = true_curve()
    inside = (tru[:, 0] > 3) & (tru[:, 0] < FIT_SHAPE[1] - 4)
    d, _ = cKDTree(node_curve(nodes)).query(tru[inside])
    return float(d.mean())


def _lm(od, base, nrm, rsa0, n_iter=10, lam=1e-2):
    """Levenberg-Marquardt on the OD residual over the nodes' offsets along their
    normals and r, s, a.  Returns the iterates (nodes, r, s, a)."""
    n = len(base)

    def nodes_of(x):
        return base + x[:n, None] * nrm

    def res(x):
        return (_render_curve(node_curve(nodes_of(x)), x[n], x[n + 1], x[n + 2]) - od).ravel()

    steps = np.r_[np.full(n, 0.05), 0.02, 0.02, 0.002]
    x = np.r_[np.zeros(n), rsa0]
    hist = [(nodes_of(x), *x[n:])]
    f = res(x)
    cost = f @ f
    for _ in range(n_iter):
        Jm = np.empty((f.size, x.size))
        for i in range(x.size):
            xp = x.copy()
            xp[i] += steps[i]
            Jm[:, i] = (res(xp) - f) / steps[i]
        A = Jm.T @ Jm
        g = Jm.T @ f
        while True:
            xn = x - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn[n:] = np.maximum(xn[n:], [0.5, 0.2, 0.01])
            fn = res(xn)
            if fn @ fn < cost:
                x, f, cost = xn, fn, fn @ fn
                lam = max(lam / 3, 1e-6)
                break
            lam *= 4
            if lam > 1e6:
                break
        hist.append((nodes_of(x), *x[n:]))
    return hist


@lru_cache(maxsize=None)
def fit_runs():
    """Two starts for the same toy fit: the winners only (whole-pixel ridge top,
    best of 16 tangents, best of 9 widths, blur unknown (1 px), contrast = the
    raw centre OD) and the population codes; then Levenberg-Marquardt on the OD."""
    od = fit_scene()
    out = {}
    for mode in ("winner", "population"):
        ms = [measure_point(p, mode) for p in true_points()]
        base = np.array([m["pos"] for m in ms])
        nrm = np.array([m["normal"] for m in ms])
        dec = [decode_rsa(m["prof"], m["x"]) for m in ms]
        if mode == "winner":
            rsa = (float(np.median([d["sp"]["r_win"] for d in dec])), 1.0, float(np.median([d["od_c"] for d in dec])))
        else:
            rsa = tuple(float(np.median([d[k] for d in dec])) for k in ("r", "s", "a"))
        hist = _lm(od, base, nrm, np.array(rsa))
        out[mode] = dict(measures=ms, hist=hist, geo=[geometry_error(h[0]) for h in hist],
                         rsa=[np.array(h[1:]) for h in hist])
    return out
