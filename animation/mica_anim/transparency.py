"""Fork or crossing (spec 4.5, 4.5b): transparency on the OD, and the local
model comparison for the junctions it leaves ambiguous.

Toy junctions are two straight vessels through one centre, chord-law
profiles, blurred, with white noise in OD:

  crossing (two depths): OD = max(L1, L2) + kappa * min(L1, L2)   ODs add (kappa ~ 0.87)
  fork (one lumen):      OD = max(L1, L2)                          the union

Measurement, as the spec writes it (on the OD, never on a render):

  a1, a2 = median OD along each line's two arms, from R to R + 2w out of the centre
  ac     = max OD within 2 px of the centre
  rho    = ac / (a1 + a2)

Everything here is the toy model's; nothing is a measurement on real data.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import neuro as N

KAPPA = 0.87              # vesselscene's reference crossing additivity (spec 4.5: 0.87-0.94)
R0 = 4.0                  # junction disc: R = r0 + w / 2 (spec 4.3)
NOISE = 0.006             # toy white noise in OD
SIZE = 64


def unit_img(deg):
    """Unit vector at a screen angle (degrees, counter-clockwise), in image
    coordinates (y down)."""
    a = np.deg2rad(deg)
    return np.array([np.cos(a), -np.sin(a)])


def _layers(shape, c, lines):
    x, y = N.grid(shape)
    out = []
    for u, a, w in lines:
        d = N.dist_to_segment(x, y, c - 80 * u, c + 80 * u)[0]
        out.append(N.chord(d, w / 2, a))
    return out


def junction(kind, a1, a2, w1, w2, deg1, deg2, size=SIZE, blur=1.0, noise=NOISE, seed=0, kappa=KAPPA):
    """OD image of a toy crossing or fork; returns od and the geometry."""
    c = np.array([(size - 1) / 2, (size - 1) / 2])
    u1, u2 = unit_img(deg1), unit_img(deg2)
    L1, L2 = _layers((size, size), c, [(u1, a1, w1), (u2, a2, w2)])
    od = np.maximum(L1, L2) + (kappa * np.minimum(L1, L2) if kind == "crossing" else 0.0)
    od = gaussian_filter(od, blur) if blur > 0 else od
    if noise > 0:
        od = od + np.random.default_rng(seed).normal(0.0, noise, od.shape)
    return od, dict(c=c, u1=u1, u2=u2, w1=w1, w2=w2, deg1=deg1, deg2=deg2)


def arm_samples(geo, k, w_max=None):
    """Sample points of line k's two arms, from R to R + 2w out of the centre."""
    c, u, w = geo["c"], geo[f"u{k}"], geo[f"w{k}"]
    R = R0 + (max(geo["w1"], geo["w2"]) if w_max is None else w_max) / 2
    s = np.arange(R, R + 2 * w + 1e-9, 0.5)
    return np.concatenate([c[None] + s[:, None] * u[None], c[None] - s[:, None] * u[None]]), R


def measure(od, geo):
    """The spec's a1, a2, ac and rho."""
    out = {}
    for k in (1, 2):
        pts, R = arm_samples(geo, k)
        out[f"a{k}"] = float(np.median(map_coordinates(od, [pts[:, 1], pts[:, 0]], order=1)))
        out["R"] = R
    x, y = N.grid(od.shape)
    disc = np.hypot(x - geo["c"][0], y - geo["c"][1]) <= 2.0
    out["ac"] = float(od[disc].max())
    out["rho"] = out["ac"] / (out["a1"] + out["a2"])
    mx, mn = max(out["a1"], out["a2"]), min(out["a1"], out["a2"])
    out["m"] = mn / mx
    out["pred_cross"] = (mx + KAPPA * mn) / (out["a1"] + out["a2"])
    out["pred_fork"] = mx / (out["a1"] + out["a2"])
    out["separation"] = KAPPA * mn / (out["a1"] + out["a2"])
    return out


def profile(od, geo, k=1, half=26.0, step=0.25):
    """OD along line k through the centre."""
    s = np.arange(-half, half + 1e-9, step)
    p = geo["c"][None] + s[:, None] * geo[f"u{k}"][None]
    return s, map_coordinates(od, [p[:, 1], p[:, 0]], order=1)


def curve_cross(m, kappa=KAPPA):
    return (1 + kappa * m) / (1 + m)


def curve_fork(m):
    return 1 / (1 + m)


# ------------------------------------------------------------------ the toy population

A_STRONG = 0.15


@lru_cache(maxsize=None)
def monte_carlo(n_per=10, ms=(0.04, 0.08, 0.12, 0.17, 0.23, 0.3, 0.4, 0.5, 0.62, 0.75, 0.88, 1.0), seed=1):
    """Toy crossings and forks at several contrast ratios: the strong vessel
    0.15 OD (5-8 px wide), the other m times that (3.5-7 px), crossing angles 55-90
    degrees, noise NOISE.  Returns a structured dict of arrays."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for m in ms:
        for kind in ("crossing", "fork"):
            for _ in range(n_per):
                w1 = rng.uniform(5.0, 8.0)
                w2 = rng.uniform(3.5, 7.0)
                d0 = rng.uniform(0, 180)
                ang = rng.uniform(55, 90)
                od, geo = junction(kind, A_STRONG, A_STRONG * m, w1, w2, d0, d0 + ang, seed=1000 + i)
                r = measure(od, geo)
                rows.append((kind == "crossing", m, r["m"], r["rho"], r["a1"], r["a2"], r["ac"], w1, w2, d0, ang, 1000 + i))
                i += 1
    a = np.array(rows, dtype=float)
    keys = ["is_cross", "m_true", "m", "rho", "a1", "a2", "ac", "w1", "w2", "d0", "ang", "seed"]
    return {k: a[:, j] for j, k in enumerate(keys)}


@lru_cache(maxsize=None)
def bands(deg=2):
    """Each hypothesis's toy distribution of rho against the measured m: the
    offset from its noise-free prediction (the noise lifts the 2 px maximum)
    and the spread, as smooth functions of m (polynomial fits)."""
    mc = monte_carlo()
    out = {}
    for kind, pred in (("crossing", curve_cross), ("fork", curve_fork)):
        s = mc["is_cross"] == (kind == "crossing")
        m, rho = mc["m"][s], mc["rho"][s]
        off = rho - pred(m)
        cm = np.polyfit(m, off, deg)
        resid = off - np.polyval(cm, m)
        cs = np.polyfit(m, np.abs(resid) * np.sqrt(np.pi / 2), 1)       # mean |dev| -> SD
        out[kind] = (cm, cs)
    return out


def band_mean_sd(kind, m):
    cm, cs = bands()[kind]
    pred = curve_cross(m) if kind == "crossing" else curve_fork(m)
    return pred + np.polyval(cm, m), np.maximum(np.polyval(cs, m), 0.006)


def log_lr(rho, m):
    """ln of the likelihood ratio crossing : fork (Gaussians from the bands)."""
    mc, sc = band_mean_sd("crossing", m)
    mf, sf = band_mean_sd("fork", m)
    return (-(rho - mc) ** 2 / (2 * sc ** 2) - np.log(sc)) - (-(rho - mf) ** 2 / (2 * sf ** 2) - np.log(sf))


def ambiguous(rho, m, small_sep=2.0):
    """The spec's rule: ambiguous when the likelihood ratio is between 1/3 and 3,
    or when the separation is small (toy: under 2 band SDs)."""
    mc, sc = band_mean_sd("crossing", m)
    mf, sf = band_mean_sd("fork", m)
    small = (mc - mf) < small_sep * np.maximum(sc, sf)
    return (np.abs(log_lr(rho, m)) < np.log(3.0)) | small


def ambiguous_zone(m_lo=0.005, n_m=300, n_rho=4000, rho_range=(0.45, 1.15)):
    """The ambiguous region of the (m, rho) plane, as polygons' outlines:
    a block at small m where the separation is small, then the band around
    the curve where the likelihood ratio is 1, between 1/3 and 3."""
    mg = np.linspace(m_lo, 1.0, n_m)
    mcr, scr = band_mean_sd("crossing", mg)
    mfk, sfk = band_mean_sd("fork", mg)
    small = (mcr - mfk) < 2.0 * np.maximum(scr, sfk)
    m_star = float(mg[small].max()) if small.any() else 0.0
    ys = np.linspace(*rho_range, n_rho)
    band = []
    for j, m in enumerate(mg):
        if m <= m_star:
            continue
        llr = log_lr(ys, m)
        inside = (ys > mfk[j]) & (ys < mcr[j])
        i = int(np.argmin(np.where(inside, np.abs(llr), np.inf)))
        if abs(llr[i]) >= np.log(3.0):
            continue
        i0 = i
        while i0 > 0 and abs(llr[i0 - 1]) < np.log(3.0):
            i0 -= 1
        i1 = i
        while i1 < n_rho - 1 and abs(llr[i1 + 1]) < np.log(3.0):
            i1 += 1
        band.append((m, ys[i0], ys[i1]))
    return m_star, np.array(band)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


# ------------------------------------------------------------------ V2e: local model comparison

def _edge_dist(shape, c, u, half):
    """Distance to a full line through c (half=False) or a half line from c along u."""
    x, y = N.grid(shape)
    p1 = c + 80 * u
    p0 = c if half else c - 80 * u
    return N.dist_to_segment(x, y, p0, p1)[0]


def hypotheses(geo, shape):
    """The hypotheses for a four-arm candidate (spec 4.5b): the crossing (the
    4.4 pairing), and a fork with each pair as the parent.  Each: a list of
    edges (distance map) and how they combine."""
    c, u1, u2 = geo["c"], geo["u1"], geo["u2"]
    L1, L2 = _edge_dist(shape, c, u1, False), _edge_dist(shape, c, u2, False)
    return {
        "crossing": dict(edges=[L1, L2], combine="kappa", groups=[1, 2]),
        "fork, parent 1": dict(edges=[L1, _edge_dist(shape, c, u2, True), _edge_dist(shape, c, -u2, True)],
                               combine="union", groups=[1, 2, 2]),
        "fork, parent 2": dict(edges=[L2, _edge_dist(shape, c, u1, True), _edge_dist(shape, c, -u1, True)],
                               combine="union", groups=[2, 1, 1]),
    }


def _layers_and_grads(h, params):
    """Each edge's chord profile and its derivatives in (amplitude, radius)."""
    L, dA, dR = [], [], []
    for d, (a, r) in zip(h["edges"], params.reshape(-1, 2)):
        q = np.clip(1.0 - (d / r) ** 2, 0.0, None)
        sq = np.sqrt(q)
        L.append(a * sq)
        dA.append(sq)
        dR.append(np.where(q > 1e-3, a * (d ** 2 / r ** 3) / np.maximum(sq, 1e-3), 0.0))
    return np.array(L), np.array(dA), np.array(dR)


def render(h, params, blur=1.0, grads=False):
    """A hypothesis's OD: union (max) at nodes, kappa at a crossing; blurred.
    With grads, also d(OD)/d(params), one map per parameter."""
    L, dA, dR = _layers_and_grads(h, np.asarray(params, float))
    n = len(L)
    if h["combine"] == "kappa":
        hi = L[0] >= L[1]
        od = np.where(hi, L[0], L[1]) + KAPPA * np.where(hi, L[1], L[0])
        w = [np.where(hi, 1.0, KAPPA), np.where(hi, KAPPA, 1.0)]
    else:
        k = np.argmax(L, axis=0)
        od = np.take_along_axis(L, k[None], 0)[0]
        w = [(k == i).astype(float) for i in range(n)]
    od = gaussian_filter(od, blur)
    if not grads:
        return od
    J = []
    for i in range(n):
        J.append(gaussian_filter(w[i] * dA[i], blur))
        J.append(gaussian_filter(w[i] * dR[i], blur))
    return od, np.array(J)


def fit(od, geo, h, sigma, meas, iters=20):
    """Fit each edge's profile (amplitude, radius) for `iters` Levenberg-
    Marquardt iterations, positions fixed, initialised from the arms' OD;
    return the weighted residual 0.5 sum((OD - render) / sigma)^2 over the
    window of radius R + 3w, the description cost (the fit's prune cost,
    0.25 * 0.5 * n_params * ln(max(pixels / 6, 2)) per edge) and the render."""
    x, y = N.grid(od.shape)
    R = meas["R"]
    win = np.hypot(x - geo["c"][0], y - geo["c"][1]) <= R + 3 * max(geo["w1"], geo["w2"])
    p = []
    for g in h["groups"]:
        p += [max(meas[f"a{g}"], 0.01) * 1.15, geo[f"w{g}"] / 2 * 0.9]
    p = np.array(p)
    lo = np.tile([0.0, 0.8], len(h["edges"]))
    hi = np.tile([0.6, 8.0], len(h["edges"]))
    target = od[win]

    def cost(q):
        return 0.5 * float(np.sum(((target - render(h, q)[win]) / sigma) ** 2))

    c0, lam = cost(p), 1e-2
    for _ in range(iters):
        model, J = render(h, p, grads=True)
        r = (target - model[win]) / sigma
        Jw = J[:, win].T / sigma
        A = Jw.T @ Jw
        g = Jw.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        q = np.clip(p + step, lo, hi)
        c1 = cost(q)
        if c1 < c0:
            p, c0, lam = q, c1, lam / 3
        else:
            lam *= 4
    model = render(h, p)
    desc = 0.0
    for d, (a, r) in zip(h["edges"], p.reshape(-1, 2)):
        pix = int(((d < r + 2) & win).sum())
        desc += 0.25 * 0.5 * 2 * np.log(max(pix / 6.0, 2.0))
    return dict(params=p, model=model, resid=(od - model), chi=c0, desc=desc, total=c0 + desc, window=win)


def compare(od, geo, sigma=NOISE):
    meas = measure(od, geo)
    hs = hypotheses(geo, od.shape)
    return {k: fit(od, geo, h, sigma, meas) for k, h in hs.items()}, meas


@lru_cache(maxsize=None)
def compare_population():
    """Model comparison on every ambiguous junction of the toy population:
    (truth is crossing, chosen is crossing) per junction."""
    mc = monte_carlo()
    amb = ambiguous(mc["rho"], mc["m"])
    out = []
    for i in np.where(amb)[0]:
        kind = "crossing" if mc["is_cross"][i] else "fork"
        od, geo = junction(kind, A_STRONG, A_STRONG * mc["m_true"][i], mc["w1"][i], mc["w2"][i], mc["d0"][i],
                           mc["d0"][i] + mc["ang"][i], seed=int(mc["seed"][i]))
        fits, _ = compare(od, geo)
        best = min(fits, key=lambda k: fits[k]["total"])
        out.append((kind == "crossing", best == "crossing"))
    return np.array(out, dtype=bool)
