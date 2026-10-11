"""Horizontal connections (spec 4.7), on toy stimuli.

Line evidence U0 (16 sheets, noise units): line cells minus edge cells,
z = z_even - 0.7 |z_odd| (spec 4.6), which narrows the crest along a vessel
to 1-2 px, then step 4's surround (minus the mean of the lower 8 of 16),
rectified.

The recurrent field, as in the spec:

    Pm[k]       = max(C[k], 0.8 C[k-1], 0.8 C[k+1])
    excite[k]   = sqrt(ahead[k](Pm) * behind[k](Pm))          (bipole)
    parallel[k] = min(F_left[k](C), F_right[k](C))            (the weaker flank)
    rival[k]    = max(C[k-2], C[k-1], C[k+1], C[k+2])
    C[k]       <- max(0, U0 + g_E excite - g_P parallel - g_R relu(rival - C)) / (1 + g_E)

and today's step 5, C <- (U0 + bipole(C)) / 2 for 3 iterations.  The lobes are
1-px rays along the cell's axis (as step 5's), sampled bilinearly, or, for the
fix the spec proposes, rays with a Gaussian width across.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from . import neuro as N

SIGMA, ELONG = 1.5, 3.0          # the fine channel's line and edge cells
EDGE_W = 0.7
LOBE = (2.0, 12.0, 6.0)          # lobe from d = 2 to 12 px along the axis, Gaussian weight sd 6 px
FLANK = (5.0, 5.0)               # flank strips: 5 px to either side, +-5 px along
T_ITER, G_E, G_P, G_R = 8, 1.0, 0.3, 0.5     # spec 6.3


def edge_kernel(theta, sigma=SIGMA, elong=ELONG):
    half = int(np.ceil(3.5 * sigma * max(elong, 1.0)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    t = x * np.cos(theta) + y * np.sin(theta)
    n = -x * np.sin(theta) + y * np.cos(theta)
    k = (n / sigma) * np.exp(-n ** 2 / (2 * sigma ** 2)) * np.exp(-t ** 2 / (2 * (sigma * elong) ** 2))
    return k / k[k > 0].sum()


@lru_cache(maxsize=None)
def _noise_sd():
    z = N.add_noise(np.zeros((400, 400)), 0.01, 123)
    ev = np.array([fftconvolve(z, N.line_kernel(t, SIGMA, ELONG), mode="valid").std() for t in N.THETAS])
    od = np.array([fftconvolve(z, edge_kernel(t), mode="valid").std() for t in N.THETAS])
    return ev, od


def line_evidence(od):
    """U0: line minus edge cells in noise units, step 4's surround, rectified. Shape (16, h, w)."""
    sd_e, sd_o = _noise_sd()
    ev = np.stack([fftconvolve(od, N.line_kernel(t, SIGMA, ELONG), mode="same") for t in N.THETAS]) / sd_e[:, None, None]
    odd = np.stack([fftconvolve(od, edge_kernel(t), mode="same") for t in N.THETAS]) / sd_o[:, None, None]
    z = np.maximum(ev - EDGE_W * np.abs(odd), 0.0)        # rates
    floor = np.sort(z, axis=0)[:8].mean(axis=0)            # step 4: the orientation-blind part
    return np.maximum(z - floor[None], 0.0)


# ------------------------------------------------------------------ lobe and flank kernels

def _splat(points, weights, R):
    K = np.zeros((2 * R + 1, 2 * R + 1))
    for (x, y), w in zip(points, weights):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            K[R + y0 + dy, R + x0 + dx] += w * ww
    return K / K.sum()


@lru_cache(maxsize=None)
def lobe_kernel(k, sign, width=0.0):
    """Weights of the lobe ahead (sign +1) or behind (-1) of orientation k: a 1-px ray along the
    axis (width 0), or a ray with a Gaussian profile of that sd across it."""
    th = N.THETAS[k]
    t = np.array([np.cos(th), np.sin(th)])
    n = np.array([-np.sin(th), np.cos(th)])
    d0, d1, sd = LOBE
    ds = np.arange(d0, d1 + 1e-9, 0.5)
    across = [0.0] if width <= 0 else np.arange(-3 * width, 3 * width + 1e-9, 0.5)
    pts, ws = [], []
    for d in ds:
        for a in across:
            pts.append(sign * d * t + a * n)
            ws.append(np.exp(-d ** 2 / (2 * sd ** 2)) * (1.0 if width <= 0 else np.exp(-a ** 2 / (2 * width ** 2))))
    return _splat(pts, ws, int(np.ceil(d1 + 3 * width)) + 2)


@lru_cache(maxsize=None)
def flank_kernel(k, side):
    th = N.THETAS[k]
    t = np.array([np.cos(th), np.sin(th)])
    n = np.array([-np.sin(th), np.cos(th)])
    h, half = FLANK
    pts = [side * h * n + s * t for s in np.arange(-half, half + 1e-9, 0.5)]
    return _splat(pts, np.ones(len(pts)), int(np.ceil(h + half)) + 2)


def _corr(a, K):
    return fftconvolve(a, K[::-1, ::-1], mode="same")


def lobes(P, width=0.0):
    ahead = np.stack([_corr(P[k], lobe_kernel(k, +1, width)) for k in range(N.N_ORIENT)])
    behind = np.stack([_corr(P[k], lobe_kernel(k, -1, width)) for k in range(N.N_ORIENT)])
    return np.maximum(ahead, 0), np.maximum(behind, 0)


def pooled(C):
    return np.maximum(C, np.maximum(0.8 * np.roll(C, 1, axis=0), 0.8 * np.roll(C, -1, axis=0)))


def terms(C, U0, width=0.0):
    """The four terms of one iteration, from the current C."""
    ahead, behind = lobes(pooled(C), width)
    excite = np.sqrt(ahead * behind)
    Fl = np.stack([_corr(C[k], flank_kernel(k, +1)) for k in range(N.N_ORIENT)])
    Fr = np.stack([_corr(C[k], flank_kernel(k, -1)) for k in range(N.N_ORIENT)])
    parallel = np.minimum(Fl, Fr)
    rival = np.max(np.stack([np.roll(C, s, axis=0) for s in (-2, -1, 1, 2)]), axis=0)
    return dict(ahead=ahead, behind=behind, excite=excite, parallel=parallel, rival=rival)


def recurrent(U0, T=T_ITER, gE=G_E, gP=G_P, gR=G_R, width=0.0):
    """The spec's recurrent field; returns the list C_0 = U0, C_1, ..., C_T."""
    Cs = [U0.copy()]
    C = U0.copy()
    for _ in range(T):
        tm = terms(C, U0, width)
        C = np.maximum(0.0, U0 + gE * tm["excite"] - gP * tm["parallel"] - gR * np.maximum(tm["rival"] - C, 0.0)) / (1 + gE)
        Cs.append(C)
    return Cs


def today(U0, T=3, width=0.0):
    """Today's step 5: excitation only, C <- (U0 + bipole(C)) / 2."""
    Cs = [U0.copy()]
    C = U0.copy()
    for _ in range(T):
        a, b = lobes(pooled(C), width)
        C = (U0 + np.sqrt(a * b)) / 2
        Cs.append(C)
    return Cs


def change_per_iteration(Cs, mask=None):
    """Relative change of C per iteration: sum |C_t - C_{t-1}| / sum C_t (over mask)."""
    out = []
    for a, b in zip(Cs[:-1], Cs[1:]):
        d = np.abs(b - a)
        if mask is not None:
            d, bb = d[:, mask], b[:, mask]
        else:
            bb = b
        out.append(float(d.sum() / max(bb.sum(), 1e-9)))
    return np.array(out)


# ------------------------------------------------------------------ toy scene for the live run

SCENE_SHAPE = (120, 200)


@lru_cache(maxsize=None)
def toy_scene(seed=11):
    """Two vessels along cell axes (0 and 90 deg) crossing, a texture 'web' of short parallel
    pieces, and scattered short dashes.  Returns OD, masks per class and the runs."""
    h, w = SCENE_SHAPE
    rng = np.random.default_rng(seed)
    segs = [((-50.0, 34.0), (260.0, 34.0), 3.0, 0.15), ((150.0, -50.0), (150.0, 180.0), 3.0, 0.15)]
    web = []
    th = np.deg2rad(135.0)
    u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    c0 = np.array([52.0, 84.0])
    for i in range(-3, 4):
        c = c0 + i * 5.0 * n
        web.append((tuple(c - 8 * u), tuple(c + 8 * u), 1.6, 0.15))
    dashes = []
    tries = 0
    while len(dashes) < 9 and tries < 500:
        tries += 1
        c = np.array([rng.uniform(95, 192), rng.uniform(58, 112)]) if len(dashes) < 6 else np.array([rng.uniform(10, 130), rng.uniform(52, 60)])
        if len(dashes) >= 6:
            c = np.array([rng.uniform(8, 120), rng.uniform(6, 20)])
        if abs(c[0] - 150) < 14 or abs(c[1] - 34) < 16 or np.hypot(*(c - c0)) < 34:
            continue
        if any(np.hypot(*(c - np.array(d[2]))) < 20 for d in dashes):
            continue
        a = rng.uniform(0, np.pi)
        L = rng.uniform(10, 14)
        e = np.array([np.cos(a), np.sin(a)]) * L / 2
        dashes.append((tuple(c - e), tuple(c + e), tuple(c)))
    dsegs = [(d[0], d[1], 1.6, 0.15) for d in dashes]
    clean = N.vessel_od(SCENE_SHAPE, segs + web + dsegs, blur=1.0)
    od = N.add_noise(clean, 0.01, seed)
    x, y = N.grid(SCENE_SHAPE)
    def near(seglist, r=2.0):
        m = np.zeros(SCENE_SHAPE, bool)
        for p0, p1, *_ in seglist:
            m |= N.dist_to_segment(x, y, p0, p1)[0] <= r
        return m
    masks = dict(vessel=near(segs[:2], 0.5) & (np.hypot(x - 150, y - 34) > 10), web=near(web, 0.75), dashes=near(dsegs, 0.75))
    bg = ~near(segs + web + dsegs, 14.0)
    bg[:10] = bg[-10:] = False
    bg[:, :10] = bg[:, -10:] = False
    masks["background"] = bg
    return dict(od=od, clean=clean, masks=masks, dashes=dashes, web=web, vessels=segs)


@lru_cache(maxsize=None)
def toy_run():
    sc = toy_scene()
    U0 = line_evidence(sc["od"])
    Cs = recurrent(U0)
    Ct = today(U0)
    return dict(U0=U0, Cs=Cs, today=Ct)


def class_medians(Cs, masks, keys=("vessel", "dashes", "web")):
    """Median of the best orientation's C over each class's pixels, per iteration, relative to U0."""
    out = {}
    for k in keys:
        m = masks[k]
        base = np.median(Cs[0].max(axis=0)[m])
        out[k] = np.array([np.median(C.max(axis=0)[m]) / base for C in Cs])
    return out


def background_p99(Cs, mask):
    return np.array([np.percentile(C.max(axis=0)[mask], 99) for C in Cs])


# ------------------------------------------------------------------ Field, Hayes and Hess (1993): a path among distractors

@lru_cache(maxsize=None)
def fhh(seed=9, cols=21, rows=11, spacing=40.0, path_n=11, turn_deg=16.0):
    """Elements on a jittered grid with random orientations; a smooth path of path_n elements
    aligned with its tangent.  Returns positions (x, y) in px (y down), orientations, path flags
    and, per element, the association field's support ahead (A+) and behind (A-)."""
    rng = np.random.default_rng(seed)
    W, H = cols * spacing, rows * spacing
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * spacing, (np.arange(rows) + 0.5) * spacing)
    pos = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-0.22, 0.22, (cols * rows, 2)) * spacing
    ori = rng.uniform(0, np.pi, len(pos))
    # the path: start left of centre, heading roughly right, turning by +-turn_deg each step
    p = np.array([W * 0.24, H * 0.62])
    head = np.deg2rad(-18.0)
    pts, heads = [], []
    for i in range(path_n):
        pts.append(p.copy()); heads.append(head)
        head += np.deg2rad(turn_deg) * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
        p = p + spacing * np.array([np.cos(head), np.sin(head)])
    pts = np.array(pts)
    keep = np.array([np.min(np.hypot(*(pts - q).T)) > 0.62 * spacing for q in pos])
    pos, ori = pos[keep], ori[keep]
    tang = []
    for i in range(path_n):                      # element orientation = local path tangent
        a = pts[max(i - 1, 0)]; b = pts[min(i + 1, path_n - 1)]
        tang.append(np.arctan2(b[1] - a[1], b[0] - a[0]) % np.pi)
    pos = np.concatenate([pos, pts]); ori = np.concatenate([ori, np.array(tang) + rng.normal(0, np.deg2rad(4), path_n)])
    on_path = np.concatenate([np.zeros(keep.sum(), bool), np.ones(path_n, bool)])
    Ap, Am = association(pos, ori, spacing)
    return dict(pos=pos, ori=ori % np.pi, path=on_path, W=W, H=H, spacing=spacing, Ap=Ap, Am=Am, path_pts=pts)


def association(pos, ori, spacing, sd_d=1.0, sd_phi=22.0, sd_delta=12.0):
    """Support each element gets from its neighbours, ahead and behind: a co-circular
    association field (Field, Hayes and Hess 1993)."""
    n = len(pos)
    Ap, Am = np.zeros(n), np.zeros(n)
    for i in range(n):
        t = np.array([np.cos(ori[i]), np.sin(ori[i])])
        d = pos - pos[i]
        r = np.hypot(d[:, 0], d[:, 1])
        r[i] = np.inf
        along = d @ t
        phi = np.arctan2(d @ np.array([-t[1], t[0]]), np.abs(along))        # angle off the axis, either way
        co = ori[i] + np.sign(along) * 2 * phi                               # co-circular orientation there
        delta = (ori - co + np.pi / 2) % np.pi - np.pi / 2
        w = (np.exp(-(r / spacing) ** 2 / (2 * sd_d ** 2)) * np.exp(-np.rad2deg(phi) ** 2 / (2 * sd_phi ** 2))
             * np.exp(-np.rad2deg(delta) ** 2 / (2 * sd_delta ** 2)))
        w[r > 2.6 * spacing] = 0.0
        Ap[i] = w[along > 0].sum()
        Am[i] = w[along < 0].sum()
    return Ap, Am


def gabor_image(F, px_per=1.0, sigma=0.17, wavelength=0.32, contrast=0.42):
    """The field as Gabor patches on grey (stripes along each element's orientation)."""
    sp = F["spacing"]
    W, H = int(F["W"] * px_per), int(F["H"] * px_per)
    img = np.zeros((H, W))
    s, lam = sigma * sp * px_per, wavelength * sp * px_per
    R = int(3 * s) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(float)
    for (x, y), th in zip(F["pos"] * px_per, F["ori"]):
        cx, cy = int(round(x)), int(round(y))
        fx, fy = xx - (x - cx), yy - (y - cy)
        n = -fx * np.sin(th) + fy * np.cos(th)
        g = np.cos(2 * np.pi * n / lam) * np.exp(-(fx ** 2 + fy ** 2) / (2 * s ** 2))
        y0, y1, x0, x1 = cy - R, cy + R + 1, cx - R, cx + R + 1
        sy0, sx0 = max(0, -y0), max(0, -x0)
        sy1, sx1 = g.shape[0] - max(0, y1 - H), g.shape[1] - max(0, x1 - W)
        img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] += g[sy0:sy1, sx0:sx1]
    return np.clip(0.5 + contrast * img, 0, 1)


# ------------------------------------------------------------------ the finding: thin lobes slide off the crest

def vessel_at(angle_deg, shape=(128, 160), seed=1):
    c = np.array([shape[1] / 2, shape[0] / 2])
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)])
    clean = N.vessel_od(shape, [(c - 300 * u, c + 300 * u, 3.0, 0.15)], blur=1.0)
    return N.add_noise(clean, 0.01, seed), c, u


def _read(stack, pts):
    from scipy.ndimage import map_coordinates
    return np.stack([map_coordinates(s, [pts[:, 1], pts[:, 0]], order=1) for s in stack])


@lru_cache(maxsize=None)
def support_vs_angle(angles=tuple(np.round(np.arange(0, 22.51, 1.40625), 4)), widths=(0.0, 1.0)):
    """Feedforward reading of the lobes along a straight vessel: median over the centreline of
    excite / Pm at the best orientation, for each vessel angle and lobe width, normalised to 0 deg."""
    out = {w: [] for w in widths}
    for ang in angles:
        od, c, u = vessel_at(ang)
        U0 = line_evidence(od)
        P = pooled(U0)
        pts = c[None] + np.arange(-30, 31)[:, None] * u[None]
        k = _read(U0, pts).argmax(axis=0)
        p_here = _read(P, pts)[k, np.arange(len(pts))]
        for w in widths:
            a, b = lobes(P, w)
            e = _read(np.sqrt(a * b), pts)[k, np.arange(len(pts))]
            out[w].append(float(np.median(e / np.maximum(p_here, 1e-6))))
    return np.array(angles), {w: np.array(v) / v[0] for w, v in out.items()}


@lru_cache(maxsize=None)
def change_along(angle_deg, T=T_ITER):
    """The recurrent field's median change along a straight vessel at this angle (the spec's measure)."""
    od, c, u = vessel_at(angle_deg)
    U0 = line_evidence(od)
    CT = recurrent(U0, T)[-1]
    pts = c[None] + np.arange(-40, 41)[:, None] * u[None]
    return float(np.median(_read(CT, pts).max(axis=0)) / np.median(_read(U0, pts).max(axis=0)) - 1)


@lru_cache(maxsize=None)
def other_measures():
    """Isolated 14 px dash, gaps of 8 and 18 px, the last 10 px before an end (all at 0 deg)."""
    S = (128, 160)
    out = {}
    clean = N.vessel_od(S, [((73, 64), (87, 64), 3.0, 0.15)], blur=1.0)
    U0 = line_evidence(N.add_noise(clean, 0.01, 2))
    CT = recurrent(U0)[-1]
    pts = np.array([[x, 64.0] for x in range(74, 87)])
    out["dash"] = float(np.median(_read(CT, pts).max(axis=0)) / np.median(_read(U0, pts).max(axis=0)) - 1)
    for g in (8, 18):
        clean = N.vessel_od(S, [((-50, 64), (80 - g / 2, 64), 3.0, 0.15), ((80 + g / 2, 64), (250, 64), 3.0, 0.15)], blur=1.0)
        U0 = line_evidence(N.add_noise(clean, 0.01, 3))
        CT = recurrent(U0)[-1]
        out[f"gap{g}"] = float(CT[:, 64, 80].max() / U0[:, 64, 80].max() - 1)
    clean = N.vessel_od(S, [((-50, 64), (100, 64), 3.0, 0.15)], blur=1.0)
    U0 = line_evidence(N.add_noise(clean, 0.01, 4))
    CT = recurrent(U0)[-1]
    out["end"] = float(np.median(CT[:, 64, 90:100].max(axis=0)) / np.median(U0[:, 64, 90:100].max(axis=0)) - 1)
    return out
