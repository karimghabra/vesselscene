"""Motion (spec 4.14) on vesselscene's burst: the computations behind chapter 17.

Flicker, on the registered frames F_t of the burst (vesselscene renders each
frame at the registration reference, so no registration is needed here):

    L_t  = ln F_t                      (each frame's global level removed: the lamp flickers too)
    D_t  = L_t - median_t(L_t)
    Phi  = 1.4826 median_t |D_t|       (robust temporal spread, the MAD)
    Phi2 = max(Phi^2 - noise_var(I), 0)
    flow = sqrt(Phi2) / local_rms(sqrt(Phi2), background, 48)

The camera's noise variance at each brightness, noise_var(I), is estimated from
the burst itself: the median Phi^2 over background pixels, binned by
brightness (a smooth curve, close to shot noise's 1/I).  The background is
the truth's (pixels at least 6 px from any vessel run), as a stand-in for
step 1's mask.

Direction, along one vessel: the OD sampled along the truth run's centreline
(xy in increasing arc length s, from the upstream node u to the downstream v)
in every frame gives D(s, t); motion energy in its 2-D spectrum
(Adelson-Bergen: quadrature pairs tuned to +v and -v) gives the sign of the
flow and its speed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from . import vsdata

FPS = 32.2                       # vesselscene's frame rate (imaging.ImagingParams.fps)
UM_PER_PX = 1.5                  # 6 um per d_c, 4 px per d_c (healthy preset)


@lru_cache(maxsize=None)
def frames():
    S = vsdata.load()
    if S.burst is None:
        raise RuntimeError("no burst in data_cache")
    F = np.asarray(S.burst, np.float64)
    L = np.log(np.maximum(F, 1.0))
    L -= np.median(L, axis=(1, 2), keepdims=True)      # each frame's overall level (lamp flicker)
    return F, L


@lru_cache(maxsize=None)
def background_mask(dist_px=6.0):
    S = vsdata.load()
    h, w = S.shape
    m = np.zeros((h, w), bool)
    for r in S.truth["runs"]:
        xy = np.round(np.asarray(r["xy"])).astype(int)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        m[xy[ok, 1], xy[ok, 0]] = True
    d = ndi.distance_transform_edt(~m)
    wid = max(dist_px, 1.0)
    return (d > wid) & np.isfinite(S.average)


@lru_cache(maxsize=None)
def flicker():
    F, L = frames()
    n = len(F)
    D = L - np.median(L, axis=0)
    Phi = 1.4826 * np.median(np.abs(D), axis=0)
    I = np.mean(F, axis=0)
    bg = background_mask()
    # noise_var(I): median Phi^2 over background pixels in brightness bins, interpolated
    edges = np.quantile(I[bg], np.linspace(0.0, 1.0, 25))
    centres, nv = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = bg & (I >= a) & (I < b)
        if sel.sum() > 200:
            centres.append(float(np.median(I[sel])))
            nv.append(float(np.median(Phi[sel] ** 2)))
    centres, nv = np.array(centres), np.array(nv)
    # a smooth model: var = c0 + c1 / I (read noise in log units, shot noise), least squares on the bins
    A = np.c_[np.ones_like(centres), 1.0 / centres]
    coef, *_ = np.linalg.lstsq(A, nv, rcond=None)
    noise_var = coef[0] + coef[1] / np.maximum(I, 1.0)
    Phi2 = np.maximum(Phi ** 2 - noise_var, 0.0)
    rt = np.sqrt(Phi2)
    win = 48
    num = ndi.uniform_filter(np.where(bg, Phi2, 0.0), win)
    den = ndi.uniform_filter(bg.astype(float), win)
    local_rms = np.sqrt(num / np.maximum(den, 1e-3))
    local_rms = np.maximum(local_rms, np.median(local_rms[bg]) * 0.25)
    flow = rt / local_rms
    return dict(n=n, D=D, Phi=Phi, I=I, noise_var=noise_var, bins=(centres, nv), coef=coef, Phi2=Phi2, flow=flow, bg=bg)


def hysteresis(flow, hi=3.0, lo=1.5):
    lab, n = ndi.label(flow > lo)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[flow > hi])] = True
    keep[0] = False
    return keep[lab]


def run_by_vid(vid, longest=True):
    S = vsdata.load()
    rs = [r for r in S.truth["runs"] if r["vid"] == vid]
    return max(rs, key=lambda r: len(r["xy"])) if longest else rs[0]


def resample(xy, step=1.0):
    xy = np.asarray(xy, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    s = np.arange(0, d[-1], step)
    return s, np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])


def spacetime(xy, step=1.0, across=(-1.0, 0.0, 1.0)):
    """D(s, t): each frame's OD along the centreline (averaged over a few
    offsets across the lumen) minus its temporal median (positive: more blood)."""
    _, L = frames()
    s, x, y = resample(xy, step)
    tx, ty = np.gradient(x), np.gradient(y)
    nn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / nn, tx / nn
    od = np.array([-np.mean([ndi.map_coordinates(Lt, [y + o * ny, x + o * nx], order=1) for o in across], axis=0)
                   for Lt in L])
    return s, od - np.median(od, axis=0)


def gabor_pair(v, sign, T, lam_scale=2.6, sigma_t=2.0):
    """A quadrature pair of space-time Gabors tuned to speed sign*v (px per frame)."""
    lam = max(lam_scale * v, 12.0)
    sigma_s = max(0.5 * lam, 6.0)
    k = 2 * np.pi / lam
    hs = int(2.5 * sigma_s)
    ss = np.arange(-hs, hs + 1)
    tt = np.arange(-(T // 2), T // 2 + 1)
    S_, T_ = np.meshgrid(ss, tt)
    env = np.exp(-S_ ** 2 / (2 * sigma_s ** 2) - T_ ** 2 / (2 * sigma_t ** 2))
    ph = k * (S_ - sign * v * T_)
    return env * np.cos(ph), env * np.sin(ph)


def motion_energy(D, speeds=np.arange(3, 25, 1.0)):
    """Opponent motion energy of D(s, t) (rows t, columns s; Adelson and Bergen
    1985): for each speed v, quadrature pairs tuned to +v (downstream, increasing
    s) and to -v.  The wavelength grows with v (2.6 v, at least 12 px), so each
    filter stays under the temporal Nyquist limit.  Returns E+(v), E-(v)."""
    T = D.shape[0]
    Dw = D - D.mean()
    Ep, Em = [], []
    for v in speeds:
        e = []
        for sgn in (+1, -1):
            ev, od = gabor_pair(v, sgn, T)
            a = ndi.convolve(Dw, ev, mode="constant")
            b = ndi.convolve(Dw, od, mode="constant")
            e.append(float((a ** 2 + b ** 2).sum()))
        Ep.append(e[0])
        Em.append(e[1])
    return np.asarray(speeds, float), np.array(Ep), np.array(Em)


def xcorr_shift(D, max_shift=30):
    """Frame-to-frame cross-correlation along s against shift (px per frame)."""
    sh = np.arange(-max_shift, max_shift + 1)
    n = D.shape[1]
    cc = []
    for k in sh:
        a = D[:-1, max(0, -k):n - max(0, k)]
        c = D[1:, max(0, k):n + min(0, k)]
        cc.append(float(np.mean(a * c) / np.sqrt(np.mean(a * a) * np.mean(c * c) + 1e-12)))
    return sh, np.array(cc)


def run_direction(r, s0=None, s1=None):
    """Measured flow sign and speed along a truth run (or a stretch of it)."""
    xy = np.asarray(r["xy"], float)
    s, D = spacetime(xy)
    if s0 is not None:
        sel = (s >= s0) & (s <= s1)
        D = D[:, sel]
    sh, cc = xcorr_shift(D)
    v = float(sh[np.argmax(cc)])
    sp, Ep, Em = motion_energy(D)
    opp = (Ep.max() - Em.max()) / (Ep.max() + Em.max() + 1e-12)
    return dict(v_xcorr=v, cc=float(cc.max()), speeds=sp, Ep=Ep, Em=Em, opponent=float(opp),
                v_energy=float(sp[np.argmax(np.maximum(Ep, Em))]) * (1 if Ep.max() >= Em.max() else -1))


@lru_cache(maxsize=None)
def views():
    """Display arrays (flattened ln I) for the average and every frame, on the average's stretch."""
    S = vsdata.load()
    F, _ = frames()
    I = np.asarray(S.average, float)
    ok = np.isfinite(I) & (I > 0)
    L = np.where(ok, np.log(np.where(ok, I, 1.0)), 0.0)
    w = ndi.gaussian_filter(ok.astype(float), 40) + 1e-9
    bg = ndi.gaussian_filter(np.where(ok, L, 0.0), 40) / w
    lo, hi = np.percentile((L - bg)[ok], [1.0, 99.7])
    out = [np.where(ok, np.clip((L - bg - lo) / (hi - lo), 0, 1), 1.0)]
    for f in F:
        Lf = np.log(np.maximum(f, 1.0))
        Lf = Lf - np.median(Lf) + np.median(L[ok])
        out.append(np.where(ok, np.clip((Lf - bg - lo) / (hi - lo), 0, 1), 1.0))
    return np.array(out, np.float32)


def run_pixels(r, w, h):
    xy = np.round(np.asarray(r["xy"])).astype(int)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return xy[ok]


@lru_cache(maxsize=None)
def coverage():
    """Share of truth centreline pixels inside the flicker mask, by vessel class and
    width; the share of background pixels inside it."""
    S = vsdata.load()
    h, w = S.shape
    f = flicker()
    m = hysteresis(f["flow"])
    md = ndi.binary_dilation(m, iterations=1)
    groups = {"capillaries": [], "arterioles and venules": [], "episcleral (wide)": []}
    for r in S.truth["runs"]:
        p = run_pixels(r, w, h)
        if len(p) == 0:
            continue
        g = ("capillaries" if r["cls"] == "capillary" else
             "episcleral (wide)" if r["cls"].startswith("episcleral") else "arterioles and venules")
        groups[g].append(md[p[:, 1], p[:, 0]])
    cov = {k: float(np.concatenate(v).mean()) if v else float("nan") for k, v in groups.items()}
    cov["background"] = float(m[f["bg"]].mean())
    return cov


def junction_arms(jid, reach=90.0):
    """For a truth junction: each arm's run stretch next to it, the measured
    flow along it (sign of motion relative to the run's s), and whether that
    means into or out of the junction; with the truth's in / out."""
    S = vsdata.load()
    T = S.truth
    j = next(j for j in T["junctions"] if j["id"] == jid)
    arms = []
    for ln in j["lines"]:
        best = None
        for r in T["runs"]:
            if r["vid"] != ln["vid"]:
                continue
            xy = np.asarray(r["xy"], float)
            d = np.hypot(xy[:, 0] - j["x"], xy[:, 1] - j["y"])
            if best is None or d.min() < best[0]:
                best = (d.min(), r, int(np.argmin(d)))
        d0, r, k = best
        xy = np.asarray(r["xy"], float)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        sj = s[k]
        if ln["flow"] == "in":
            a, b, toward = max(0.0, sj - reach), sj - 4.0, +1        # +s points toward the junction
        else:
            a, b, toward = sj + 4.0, min(s[-1], sj + reach), -1       # +s points away from it
        if b - a < 24:
            arms.append(dict(vid=ln["vid"], xy=xy[(s >= a - 20) & (s <= b + 20)], truth=ln["flow"], measured=None, v=0.0,
                             v_energy=0.0, opponent=0.0, width=ln["width_px"]))
            continue
        try:
            dd = run_direction(r, a, b)
        except ValueError:
            dd = None
        if dd is None:
            arms.append(dict(vid=ln["vid"], xy=xy[(s >= a) & (s <= b)], truth=ln["flow"], measured=None, v=0.0,
                             v_energy=0.0, opponent=0.0, width=ln["width_px"]))
            continue
        sign = np.sign(dd["opponent"])
        measured = "in" if sign * toward > 0 else "out"
        sel = (s >= a) & (s <= b)
        arms.append(dict(vid=ln["vid"], xy=xy[sel], truth=ln["flow"], measured=measured, v=dd["v_xcorr"],
                         v_energy=dd["v_energy"], opponent=dd["opponent"], width=ln["width_px"]))
    return j, arms


def best_window(w=320, h=180):
    """The window (x0, y0) whose truth vessels flicker most (for the close-up)."""
    S = vsdata.load()
    H, W = S.shape
    f = flicker()
    vm = np.zeros((H, W), bool)
    for r in S.truth["runs"]:
        p = run_pixels(r, W, H)
        vm[p[:, 1], p[:, 0]] = True
    score = ndi.uniform_filter(np.where(vm, f["flow"], 0.0), (h, w))
    score[: h // 2 + 2, :] = 0
    score[H - h // 2 - 2:, :] = 0
    score[:, : w // 2 + 2] = 0
    score[:, W - w // 2 - 2:] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return int(cx - w // 2), int(cy - h // 2)


def edge_vs_centre(min_width=10.0):
    """Mean flicker (flow) across wide vessels: at the centre against at the walls."""
    S = vsdata.load()
    H, W = S.shape
    fl = flicker()["flow"]
    prof = []
    for r in S.truth["runs"]:
        if not (r["width_px"] and r["width_px"] >= min_width):
            continue
        s, x, y = resample(r["xy"], 2.0)
        tx, ty = np.gradient(x), np.gradient(y)
        nn = np.hypot(tx, ty) + 1e-9
        nx, ny = -ty / nn, tx / nn
        half = r["width_px"] / 2
        offs = np.linspace(-1.6, 1.6, 33) * half
        ok = (x > 5) & (x < W - 5) & (y > 5) & (y < H - 5)
        if ok.sum() < 10:
            continue
        p = np.array([ndi.map_coordinates(fl, [y[ok] + o * ny[ok], x[ok] + o * nx[ok]], order=1).mean() for o in offs])
        prof.append(p)
    return np.linspace(-1.6, 1.6, 33), np.mean(prof, axis=0), len(prof)
