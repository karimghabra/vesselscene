"""Teaching versions of spec 4.9 (border ownership, grouping cells), 4.9b
(evidence fusion) and 4.9c (the readout per orientation), for chapters 12-13.

Everything here runs on toy stimuli built with neuro.py's chord-law vessels,
one size channel and 16 orientations, as the rest of the animation does.  The
parameters are chosen to make the effects legible at animation scale; they
are not MICA's.

Conventions: images are (h, w) arrays, x right and y down.  For orientation
theta_k the line direction is t_k = (cos, sin) and its normal is
n_k = (-sin, cos), as in neuro.line_kernel.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, map_coordinates
from scipy.signal import fftconvolve

from . import neuro as N

THETAS = N.THETAS
K = N.N_ORIENT


def normal(theta):
    return np.array([-np.sin(theta), np.cos(theta)])


def tangent(theta):
    return np.array([np.cos(theta), np.sin(theta)])


# =========================================================== 1-D: one profile
DX = 0.1                                  # sample spacing of the 1-D profiles (px)
X1 = np.arange(-30.0, 30.0 + DX / 2, DX)


def _blur1(f, s):
    return gaussian_filter1d(f, s / DX)


def vessel_profile(r=5.0, a=0.12, blur=1.0, x=X1):
    """OD across a vessel: the chord law, blurred."""
    return _blur1(N.chord(np.abs(x), r, a), blur)


def step_profile(a=0.08, width=1.0, x=X1):
    """OD across a one-sided edge (an illumination step, a glare edge): dark on the right."""
    return a / (1.0 + np.exp(-x / width))


def odd_profile(od, sigma=1.5):
    """The odd (edge) cell across the profile: -d/dx of the blurred OD.
    Positive where the OD falls to the right: an edge darkening toward the left."""
    return -np.gradient(_blur1(od, sigma), DX)


def shifted(odd, h, x=X1):
    """left[h](x) = max(0, odd(x + h)) and right[h](x) = max(0, -odd(x - h))."""
    L = np.maximum(0.0, np.interp(x + h, x, odd, left=0.0, right=0.0))
    R = np.maximum(0.0, -np.interp(x - h, x, odd, left=0.0, right=0.0))
    return L, R


def grouping_profile(odd, h, x=X1):
    L, R = shifted(odd, h, x)
    return np.sqrt(L * R)


def half_height_halfwidth(od, x=X1):
    above = x[od >= od.max() / 2]
    return (above.max() - above.min()) / 2


@lru_cache(maxsize=None)
def width_bias(sigma=1.5, radii=(1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0)):
    """h* against the OD's half-height half width, for vessels of several radii:
    (true half widths, h*)."""
    hs = np.arange(0.4, 16.0, 0.05)
    true, hstar = [], []
    for r in radii:
        od = vessel_profile(r)
        o = odd_profile(od, sigma)
        g0 = [np.sqrt(max(0.0, np.interp(h, X1, o)) * max(0.0, -np.interp(-h, X1, o))) for h in hs]
        true.append(half_height_halfwidth(od))
        hstar.append(hs[int(np.argmax(g0))])
    return np.array(true), np.array(hstar)


# ======================================================= 2-D: grouping cells
def odd_kernel(theta, sigma=1.5, elong=3.0):
    """Odd-symmetric (edge) cell for orientation theta: -d/dn of a Gaussian
    (sigma across, sigma*elong along).  Its response is -dOD/dn: positive
    where the OD falls along +n_k."""
    half = int(np.ceil(3.5 * sigma * max(elong, 1.0)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    t = x * np.cos(theta) + y * np.sin(theta)
    n = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-n ** 2 / (2 * sigma ** 2) - t ** 2 / (2 * (sigma * elong) ** 2))
    k = n / sigma ** 2 * g                      # convolution with this = -(d/dn)(G * OD)
    return k / np.abs(k).sum()


def _filter(img, kern):
    """Convolution with reflected borders (no false edges at the frame)."""
    p = kern.shape[0] // 2 + 1
    big = np.pad(img, p, mode="reflect")
    return fftconvolve(big, kern, mode="same")[p:-p, p:-p]


def line_cells(od, sigma=2.0, elong=3.0):
    """neuro.simple_cells with reflected borders."""
    return np.stack([_filter(od, N.line_kernel(th, sigma, elong)) for th in THETAS])


def odd_cells(od, sigma=1.5, elong=3.0):
    return np.stack([_filter(od, odd_kernel(th, sigma, elong)) for th in THETAS])


def sample_shift(img, dx, dy):
    """out(p) = img(p + (dx, dy)), bilinear, zero outside."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    return map_coordinates(img, [yy + dy, xx + dx], order=1, mode="nearest")


def grouping_cells(odd, hs):
    """G[k, h](p) = sqrt(left * right), left = max(0, odd[k](p + h n_k)),
    right = max(0, -odd[k](p - h n_k)).  Returns G of shape (K, H, h, w)."""
    out = np.zeros((len(THETAS), len(hs)) + odd.shape[1:])
    for k, th in enumerate(THETAS):
        nx, ny = normal(th)
        for j, h in enumerate(hs):
            L = np.maximum(0.0, sample_shift(odd[k], h * nx, h * ny))
            R = np.maximum(0.0, -sample_shift(odd[k], -h * nx, -h * ny))
            out[k, j] = np.sqrt(L * R)
    return out


def _bg_units(stack, bg):
    sd = np.array([s[bg].std() for s in stack])
    return stack / np.maximum(sd, 1e-12)[:, None, None]


# The chapter-12 toy: a thin vessel, a wide flat-floored vessel, a faint vessel
# and a one-sided edge (an illumination step), in white noise.
BO_SHAPE = (120, 200)
BO_HS = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])


def _flat_profile(d, half, a, soft=1.2):
    """A wide vessel with a flat floor: a smoothed box of half width `half`."""
    return a / (1.0 + np.exp((d - half) / soft))


@lru_cache(maxsize=None)
def border_scene(noise=0.006, seed=3, edge_od=0.10, faint_od=0.01, thin_od=0.05):
    h, w = BO_SHAPE
    x, y = N.grid(BO_SHAPE)
    # thin vessel: a gentle diagonal, 7 px wide
    thin_p0, thin_p1 = np.array([-10.0, 104.0]), np.array([212.0, 20.0])
    d_thin, _ = N.dist_to_segment(x, y, thin_p0, thin_p1)
    thin = N.chord(d_thin, 3.5, thin_od)
    # wide flat-floored vessel: nearly vertical, 22 px wide, shallow
    wide_p0, wide_p1 = np.array([142.0, -10.0]), np.array([128.0, 130.0])
    d_wide, _ = N.dist_to_segment(x, y, wide_p0, wide_p1)
    wide = _flat_profile(d_wide, 11.0, 0.045)
    # faint vessel: 5 px wide, OD 0.035
    faint_p0, faint_p1 = np.array([-10.0, 22.0]), np.array([110.0, 128.0])
    d_faint, _ = N.dist_to_segment(x, y, faint_p0, faint_p1)
    faint = N.chord(d_faint, 2.5, faint_od)
    # a one-sided edge: the aperture's border, darker to its right
    e0, e1 = np.array([62.0, -10.0]), np.array([48.0, 130.0])
    tvec = (e1 - e0) / np.linalg.norm(e1 - e0)
    nvec = np.array([-tvec[1], tvec[0]])               # points to the left of e0->e1 in image coords
    s = (x - e0[0]) * nvec[0] + (y - e0[1]) * nvec[1]   # signed distance from the edge line
    edge = edge_od / (1.0 + np.exp(s / 1.0))                # dark on the side where s < 0
    clean = gaussian_filter(thin + wide + faint, 1.0) + edge
    od = N.add_noise(clean, noise, seed)
    masks = dict(thin=d_thin < 2.0, wide=d_wide < 4.0, faint=d_faint < 1.5,
                 edge=(np.abs(s + 2.0) < 2.0) & (d_thin > 8) & (d_faint > 8) & (d_wide > 16),
                 bg=(d_thin > 10) & (d_wide > 20) & (d_faint > 9) & (np.abs(s) > 9))
    # display geometry (image coordinates) for overlays
    geo = dict(thin=(thin_p0, thin_p1, 3.5), wide=(wide_p0, wide_p1, 11.0), faint=(faint_p0, faint_p1, 2.5),
               edge=(e0, e1), edge_dark=-nvec)
    return dict(od=od, clean=clean, masks=masks, geo=geo)


@lru_cache(maxsize=None)
def border_maps(noise=0.006, seed=3, edge_od=0.10, faint_od=0.01, thin_od=0.05):
    """Line evidence C[k] and two-border evidence G[k] (best h) in noise units,
    h*[k] in px, and the fused evidence F[k] = C[k] + G[k]."""
    sc = border_scene(noise, seed, edge_od, faint_od, thin_od)
    od, bg = sc["od"], sc["masks"]["bg"]
    C = N.noise_units(line_cells(od, 2.0, 3.0), bg)
    odd = N.noise_units(odd_cells(od, 1.5, 3.0), bg)
    Gk = grouping_cells(odd, BO_HS)
    j = Gk.argmax(axis=1)
    G = np.take_along_axis(Gk, j[:, None], axis=1)[:, 0]
    hstar = BO_HS[j]
    G = _bg_units(G, bg)
    Cp = np.maximum(C, 0.0)
    F = Cp + G
    return dict(C=Cp, G=G, F=F, hstar=hstar, odd=odd)


def best(stack):
    """Max over orientation and its index."""
    k = stack.argmax(axis=0)
    return np.take_along_axis(stack, k[None], axis=0)[0], k


# ========================================================= 4.9b evidence fusion
@lru_cache(maxsize=None)
def cue_plane(false_alarm=0.003, n_bg=700, seed=0):
    """Each pixel's line and two-border evidence at its best fused orientation,
    and the two decision rules set to the same false-alarm rate on background:
    union (C > t_u or G > t_u) and sum (C + G > t_s)."""
    sc, M = border_scene(), border_maps()
    k = M["F"].argmax(axis=0)
    C = np.take_along_axis(M["C"], k[None], 0)[0]
    G = np.take_along_axis(M["G"], k[None], 0)[0]
    bg, faint = sc["masks"]["bg"], sc["masks"]["faint"]
    ts = np.linspace(0.0, 40.0, 8001)
    cb, gb = C[bg], G[bg]
    t_u = ts[np.argmax([np.mean((cb > t) | (gb > t)) <= false_alarm for t in ts])]
    t_s = ts[np.argmax([np.mean(cb + gb > t) <= false_alarm for t in ts])]
    cf, gf = C[faint], G[faint]
    rng = np.random.default_rng(seed)
    pick = rng.choice(bg.sum(), size=min(n_bg, int(bg.sum())), replace=False)
    return dict(t_u=float(t_u), t_s=float(t_s),
                bg=np.stack([cb[pick], gb[pick]], 1), faint=np.stack([cf, gf], 1),
                hit_union=float(np.mean((cf > t_u) | (gf > t_u))), hit_sum=float(np.mean(cf + gf > t_s)),
                fa_union=float(np.mean((cb > t_u) | (gb > t_u))), fa_sum=float(np.mean(cb + gb > t_s)))


def medial_axis(min_g=9.0, spacing=7.0, margin=4):
    """Grouping-cell ridge points (local maxima of G across the best
    orientation's normal) with their half width h*: [(x, y, h*, G), ...],
    thinned greedily to the given spacing, strongest first."""
    M = border_maps()
    G, k = best(M["G"])
    hs = np.take_along_axis(M["hstar"], k[None], 0)[0]
    h, w = G.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = -np.sin(THETAS[k]), np.cos(THETAS[k])
    ridge = G > min_g
    for s in np.arange(1.0, BO_HS.max() + 0.5, 1.0):     # the maximum across its own strip (|s| <= h*)
        for sg in (1.0, -1.0):
            o = map_coordinates(G, [yy + sg * s * ny, xx + sg * s * nx], order=1, mode="nearest")
            ridge &= (G >= o) | (s > hs)
    ridge[:margin] = ridge[-margin:] = False
    ridge[:, :margin] = ridge[:, -margin:] = False
    ys, xs = np.nonzero(ridge)
    order = np.argsort(-G[ys, xs])
    kept = []
    for i in order:
        p = np.array([xs[i], ys[i]], float)
        if all(np.hypot(*(p - q[:2])) >= spacing for q in kept):
            kept.append((p[0], p[1], float(hs[ys[i], xs[i]]), float(G[ys[i], xs[i]])))
    return kept


# ================================================== 4.9c readout per orientation
RO_SHAPE = (120, 120)
RO_CENTRE = np.array([60.0, 60.0])
RO_REF = 0.01            # noise units of this white noise (OD); images are noise-free


def _units(U, sigma=2.0, elong=3.0, ref=RO_REF):
    sd = np.array([np.sqrt((N.line_kernel(th, sigma, elong) ** 2).sum()) for th in THETAS]) * ref
    return U / sd[:, None, None]


@lru_cache(maxsize=None)
def crossing_stack(angle=30.0, a_strong=0.15, a_faint=0.03, base=160.0, r=3.0):
    """Two straight vessels crossing at the centre; the faint one at `angle`
    degrees from the strong one.  Returns C[k] >= 0 in noise units."""
    c = RO_CENTRE
    us, uf = tangent(np.radians(base)), tangent(np.radians(base - angle))
    od = N.vessel_od(RO_SHAPE, [(c - 90 * us, c + 90 * us, r, a_strong), (c - 90 * uf, c + 90 * uf, r, a_faint)], blur=1.0)
    C = np.maximum(_units(line_cells(od, 2.0, 3.0)), 0.0)
    return dict(od=od, C=C, theta_s=np.radians(base) % np.pi, theta_f=np.radians(base - angle) % np.pi, us=us, uf=uf)


def _ridge_across(V, theta_map):
    h, w = V.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = -np.sin(theta_map), np.cos(theta_map)
    a = map_coordinates(V, [yy + ny, xx + nx], order=1, mode="nearest")
    b = map_coordinates(V, [yy - ny, xx - nx], order=1, mode="nearest")
    return (V >= a) & (V >= b)


def collapsed_readout(C, t_low=2.0):
    """The old step 6: best orientation per pixel, then ridges of that map."""
    k = C.argmax(axis=0)
    M = C.max(axis=0)
    R = _ridge_across(M, THETAS[k]) & (M > t_low)
    return M, k, R


def sheet_ridges(C, t_low=2.0):
    """Layer 06: ridge tops in each orientation sheet, across that sheet's own normal."""
    return np.stack([_ridge_across(C[k], np.full(C.shape[1:], THETAS[k])) & (C[k] > t_low) for k in range(len(THETAS))])


def _kdist(k, k0):
    d = np.abs(k - k0) % len(THETAS)
    return np.minimum(d, len(THETAS) - d)


def missing_stretch(st, reach=30.0, tol_px=1.5):
    """Length (px) of the faint vessel, within `reach` of the crossing, with no
    track of its own: (collapsed readout, per-orientation readout), and the
    uncovered sample positions of each (t along the faint vessel)."""
    C = st["C"]
    k_f = int(np.argmin(_kdist(np.arange(len(THETAS)), 0) * 0 + np.abs(((THETAS - st["theta_f"]) + np.pi / 2) % np.pi - np.pi / 2)))
    M, kb, Rc = collapsed_readout(C)
    Rs = sheet_ridges(C)
    own_c = Rc & (_kdist(kb, k_f) <= 1)
    own_s = Rs[[(k_f - 1) % 16, k_f, (k_f + 1) % 16]].any(axis=0)
    ts = np.arange(-reach, reach + 0.01, 0.5)
    out, gaps = [], []
    for own in (own_c, own_s):
        ys, xs = np.nonzero(own)
        pts = np.stack([xs, ys], 1).astype(float)
        miss = []
        for t in ts:
            p = RO_CENTRE + t * st["uf"]
            ok = len(pts) and np.min(np.hypot(*(pts - p).T)) <= tol_px
            miss.append(not ok)
        miss = np.array(miss)
        out.append(float(miss.sum() * 0.5))
        gaps.append(ts[miss])
    return dict(collapsed=out[0], sheets=out[1], gap_c=gaps[0], gap_s=gaps[1], k_f=k_f, Rc=Rc, Rs=Rs, M=M, kb=kb)


@lru_cache(maxsize=None)
def missing_table():
    """The spec's four synthetic crossings (4.9c), on this toy."""
    rows = []
    for name, ang, a_f in (("equal vessels, 70°", 70.0, 0.15), ("equal, 30°", 30.0, 0.15),
                           ("0.03 OD across 0.15 OD, 70°", 70.0, 0.03), ("0.03 across 0.15, 30°", 30.0, 0.03)):
        m = missing_stretch(crossing_stack(ang, 0.15, a_f))
        rows.append((name, m["collapsed"], m["sheets"]))
    return rows


SINE_SHAPE = (110, 240)


@lru_cache(maxsize=None)
def sine_trace(amp=22.0, period=150.0):
    """A sine-shaped vessel traced by the readout per orientation (start above
    t_high, continue above t_low, a step may move one sheet).  Returns the
    traced points with their sheet, the decoded orientation (parabola through
    the three sheets around the best) and the true tangent there."""
    from .attention import Grouper
    xs = np.linspace(-10, 250, 400)
    ys = 55 + amp * np.sin(2 * np.pi * xs / period)
    pts = np.stack([xs, ys], 1)
    x, y = N.grid(SINE_SHAPE)
    best = np.full(SINE_SHAPE, np.inf)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d, _ = N.dist_to_segment(x, y, p0, p1)
        best = np.minimum(best, d)
    od = gaussian_filter(N.chord(best, 3.0, 0.12), 1.0)
    g = Grouper(od, (), gamma=0.0, gaps=False, compete=False, min_len=40.0, max_vessels=3)
    g.run()
    v = max(g.vessels, key=lambda v: v.length)
    xy, _ = v.polyline()
    ks = np.array([p[2] for p in v.paths[0][::-1]] + [p[2] for p in v.paths[1][1:]])
    dec, true = [], []
    for (px, py), k in zip(xy, ks):
        c = np.array([g.C[(k + d) % K][int(round(py)), int(round(px))] for d in (-1, 0, 1)])
        den = c[0] - 2 * c[1] + c[2]
        off = 0.5 * (c[0] - c[2]) / den if den < 0 else 0.0
        dec.append((THETAS[k] + np.clip(off, -1, 1) * np.pi / K) % np.pi)
        slope = amp * 2 * np.pi / period * np.cos(2 * np.pi * px / period)
        true.append(np.arctan(slope) % np.pi)
    dec, true = np.array(dec), np.array(true)
    err = np.degrees(np.abs((dec - true + np.pi / 2) % np.pi - np.pi / 2))
    sheet_err = np.degrees(np.abs((THETAS[ks] - true + np.pi / 2) % np.pi - np.pi / 2))
    return dict(od=od, xy=xy, k=ks, decoded=dec, true=true, err=err, sheet_err=sheet_err, length=v.length)


@lru_cache(maxsize=None)
def crossing_tracks(angle=30.0, a_faint=0.03):
    """The readout per orientation traced on the unequal crossing (no gain,
    no gaps): its two strongest tracks, as (points, sheets)."""
    from .attention import Grouper
    st = crossing_stack(angle, 0.15, a_faint)
    g = Grouper(st["od"], (), gamma=0.0, gaps=False, compete=False, min_len=20.0, max_vessels=2, ref=RO_REF)
    g.run()
    out = []
    for v in g.vessels:
        xy, _ = v.polyline()
        ks = [p[2] for p in v.paths[0][::-1]] + [p[2] for p in v.paths[1][1:]]
        out.append((xy, np.array(ks)))
    return out
