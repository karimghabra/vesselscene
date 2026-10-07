r"""Mechanism figures of experiments/PIPELINE.md: HOW each step works, on the same development image as
pipeline_figures.py (cross-sections, tuning curves, lobes, tracing candidates, junction anatomy, prune gains,
render corrections, fit progress).

    python -m experiments.splinefit.pipeline_mechanisms --out DIR [--scene healthy_s007_480x768] [--mask-scene pathologic_s000_480x768]

Every quantity comes from the pipeline's own functions; where a figure needs a quantity the pipeline does not
return (one scale's even and odd responses), the function is called with a Config that isolates it (one scale,
odd_alpha 0 or 1), never re-implemented. The truth is read to place the cross-sections and is drawn only where
a caption says so. Dev scenes only.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace

import numpy as np

from experiments.neuromimetic import neuromimetic as N

from . import score as S
from .pipeline_figures import DPI, Z, _norm, _plt, _zoom_window

C_L, C_B0, C_B, C_OD, C_T = "#444444", "#9467bd", "#1f77b4", "#d62728", "#2ca02c"


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=DPI, pil_kwargs=dict(quality=90))
    _plt().close(fig)


def _section(xy0, nrm, half, step=0.5):
    u = np.arange(-half, half + step / 2, step)
    return u, xy0[0] + u * nrm[0], xy0[1] + u * nrm[1]


def _sample(X, xs, ys, nearest=False):
    if nearest:
        H, W = X.shape
        return X[np.clip(np.round(ys).astype(int), 0, H - 1), np.clip(np.round(xs).astype(int), 0, W - 1)]
    return N._bilinear(np.asarray(X, np.float32), xs, ys)


def _spans(ax, u, m, colour, alpha=0.12, label=None):
    """Shade the runs of u where the bool profile m is true."""
    m = np.asarray(m, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(u[a], u[b - 1], color=colour, alpha=alpha, lw=0, label=label)
        label = None


def _img_with_line(ax, gray, xs, ys, win, title, extra=None):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
    ax.plot(xs, ys, "-", color="#ffd400", lw=2)
    ax.plot(xs[0], ys[0], "o", color="#ffd400", ms=6)
    ax.annotate("start", (xs[0], ys[0]), color="#ffd400", fontsize=8, xytext=(4, 4), textcoords="offset points")
    if extra:
        extra(ax)
    x0, y0, w = win
    ax.set_xlim(x0 - 0.5, x0 + w - 0.5)
    ax.set_ylim(y0 + w - 0.5, y0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, loc="left")


# ---------------------------------------------------------------------------------------------- context
def context(scene_dir: str) -> dict:
    """Stages 1-6 of the example image, every intermediate kept (full grid)."""
    cfg = N.DEFAULT
    case = S.load_truth(scene_dir, "average", reference_target=False)
    img, valid = case["image"], case["valid"]
    s1 = N.photoreceptors(img, valid, cfg)
    s1["B0"] = N._upper_envelope(s1["L"], cfg.env_radius)
    Zc, rms = N._band_cnr(s1["OD"], s1["bg"], cfg)
    chans = N.simple_cells(s1["OD"], s1["bg"], cfg)
    U3, U4, C5, Sx, traces = [], [], [], [], []
    for ci, ch in enumerate(chans):
        f = ch["f"]
        u4 = N.surround(ch["U"], ch["sigma"], cfg, f)
        c5 = N.association_field(u4, ch["sigma"], cfg, f)
        U3.append(N._up(ch["U"], f, ch["shape"]))
        U4.append(N._up(u4, f, ch["shape"]))
        C5.append(N._up(c5, f, ch["shape"]))
        Sx.append(N._up(ch["S"], f, ch["shape"], nearest=True))
        traces.append(N.readout(C5[-1], Sx[-1], cfg, chan=ci, valid=s1["ok"]))
    merged = N.merge_channels([t for tr in traces for t in tr], cfg, shape=s1["OD"].shape)
    return dict(cfg=cfg, case=case, img=img, valid=valid, s1=s1, Z=Zc, rms=rms, chans=chans, U3=U3, U4=U4, C5=C5,
                Sx=Sx, traces=traces, merged=merged, win=_zoom_window(case), gray=_norm(img, valid))


def _wide_section(ctx, half=60.0):
    """The cross-section of the figures of steps 1-3: normal to the widest merged trace crossing the zoom, at
    its point nearest the zoom centre."""
    x0, y0 = ctx["win"]
    cx, cy = x0 + Z / 2, y0 + Z / 2
    best = None
    for t in ctx["merged"]:
        xy = t["xy"]
        inside = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + Z) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + Z)
        if inside.sum() < 20:
            continue
        w = float(np.median(t["w"]))
        if best is None or w > best[0]:
            best = (w, t)
    t = best[1]
    i = int(np.argmin(np.hypot(t["xy"][:, 0] - cx, t["xy"][:, 1] - cy)))
    tg = N._tangents(t["xy"], 2)[i]
    nrm = np.array([-tg[1], tg[0]])
    u, xs, ys = _section(t["xy"][i], nrm, half)
    theta = math.atan2(tg[1], tg[0]) % math.pi
    return dict(u=u, xs=xs, ys=ys, theta=theta, k=int(round(theta / (math.pi / ctx["cfg"].n_orient))) % ctx["cfg"].n_orient,
                p=t["xy"][i], nrm=nrm)


# ---------------------------------------------------------------------------------------------- step 1
def fig_step1(ctx, out):
    plt = _plt()
    s1, sec = ctx["s1"], _wide_section(ctx)
    u, xs, ys = sec["u"], sec["xs"], sec["ys"]
    L, B0, B, OD = (_sample(s1[k], xs, ys) for k in ("L", "B0", "B", "OD"))
    M = _sample(s1["mask"].astype(np.float32), xs, ys, nearest=True) > 0.5
    T = _sample(ctx["case"]["od_img"], xs, ys)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3), layout="constrained", gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
    x0, y0 = ctx["win"]
    _img_with_line(axs[0], ctx["gray"], xs, ys, (x0 - 20, y0 - 20, Z + 40), "where: a line across the vessels (yellow)")
    ax = axs[1]
    _spans(ax, u, M, "#ff7f0e", 0.15, "vessel mask M (excluded from the background fit)")
    ax.plot(u, L, color=C_L, lw=2, label="L = ln I  (log of the image)")
    ax.plot(u, B0, color=C_B0, lw=1.5, ls="--", label="B0: grey closing (first guess of the background)")
    ax.plot(u, B, color=C_B, lw=2, label="B: background refitted outside M")
    ax.set_xlabel("px along the line")
    ax.set_ylabel("log intensity")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("1. vessels are dips in L; B bridges each dip from the pixels around it", fontsize=9, loc="left")
    ax = axs[2]
    _spans(ax, u, M, "#ff7f0e", 0.15)
    ax.plot(u, OD, color=C_OD, lw=2, label="OD = B - L (the target)")
    ax.plot(u, T, color=C_T, lw=1.2, ls=":", label="vesselscene's clean vessel OD (truth, for reference)")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("px along the line")
    ax.set_ylabel("optical density (nepers)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("2. the depth of each dip below B is the vessel's optical density", fontsize=9, loc="left")
    fig.suptitle("Step 1, how it works: one line across the example image", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech1_background.jpg")


def fig_leak(mask_scene_dir, out):
    """The background leak on the pathologic dev scene, and what an OD-derived mask changes (a sketch of
    improvement 1: the background is refitted on the larger mask's negative; this is not the fit)."""
    from experiments.splinefit import continuity as CT
    plt = _plt()
    cfg = N.DEFAULT
    case = S.load_truth(mask_scene_dir, "average", reference_target=False)
    s1 = N.photoreceptors(case["image"], case["valid"], cfg)
    with np.load(os.path.join(mask_scene_dir, "labels.npz")) as z:
        lum = z["observable_lumen_average"].astype(bool)
    Mv = np.asarray(s1["mask"], bool)
    Mod = CT.od_mask(s1["OD"], case["valid"], s1["bg"]) | CT.od_mask(s1["OD"], case["valid"], s1["bg"], smooth=3.0)
    Bn = N._masked_mean(s1["L"], s1["ok"] & ~(Mv | Mod), cfg.bg_sigma)
    # the line: across the wide observable vessel sample whose lumen the vesselness mask misses most
    P = case["prof"]
    keep = P["in_frame"] & (P["merged_into"] < 0) & P["observable"] & (P["radius_px"] >= 8)
    idx = np.flatnonzero(keep)
    H, W = lum.shape
    best = None
    for i in idx[::25]:
        x, y, r = P["x_px"][i], P["y_px"][i], P["radius_px"][i]
        if not (40 <= x < W - 40 and 40 <= y < H - 40):
            continue
        yy, xx = np.ogrid[:H, :W]
        d = (xx - x) ** 2 + (yy - y) ** 2 <= (0.8 * r) ** 2
        miss = float((d & lum & ~Mv).sum() / max((d & lum).sum(), 1))
        if best is None or miss > best[0]:
            best = (miss, i)
    i = best[1]
    vid = np.searchsorted(P["offsets"], i, side="right") - 1
    a, b = max(i - 3, P["offsets"][vid]), min(i + 3, P["offsets"][vid + 1] - 1)
    tg = np.array([P["x_px"][b] - P["x_px"][a], P["y_px"][b] - P["y_px"][a]], float)
    tg /= np.hypot(*tg)
    nrm = np.array([-tg[1], tg[0]])
    half = float(P["radius_px"][i]) + 35
    u, xs, ys = _section(np.array([P["x_px"][i], P["y_px"][i]]), nrm, half)
    L, B, Bn_, OD = (_sample(X, xs, ys) for X in (s1["L"], s1["B"], Bn, s1["OD"]))
    T = _sample(case["od_img"], xs, ys)
    lm = _sample(lum.astype(np.float32), xs, ys, nearest=True) > 0.5
    mv = _sample(Mv.astype(np.float32), xs, ys, nearest=True) > 0.5
    mo = _sample((Mv | Mod).astype(np.float32), xs, ys, nearest=True) > 0.5
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3), layout="constrained", gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
    c = (int(P["x_px"][i]), int(P["y_px"][i]))
    w = int(2 * half + 40)
    _img_with_line(axs[0], _norm(case["image"], case["valid"]), xs, ys, (c[0] - w // 2, c[1] - w // 2, w),
                   "pathologic dev scene: a line across a wide, faint vessel")
    ax = axs[1]
    _spans(ax, u, lm, C_T, 0.10, "true lumen (truth)")
    _spans(ax, u, mv, "#ff7f0e", 0.20, "step 1's vessel mask")
    ax.plot(u, L, color=C_L, lw=2, label="L = ln I")
    ax.plot(u, B, color=C_B, lw=2, label="B fitted outside step 1's mask")
    ax.plot(u, Bn_, color="#17becf", lw=2, ls="--", label="B fitted outside step 1's mask OR the OD mask")
    ax.set_xlabel("px along the line")
    ax.set_ylabel("log intensity")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("the mask misses most of the lumen, so B is fitted ON the vessel and sags into it",
                 fontsize=9, loc="left")
    ax = axs[2]
    _spans(ax, u, lm, C_T, 0.10)
    ax.plot(u, T, color=C_T, lw=1.5, ls=":", label="vesselscene's clean vessel OD (truth)")
    ax.plot(u, OD, color=C_OD, lw=2, label="OD from step 1 (the fit's target today)")
    ax.plot(u, Bn_ - L, color="#17becf", lw=2, ls="--", label="OD with the larger mask")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("px along the line")
    ax.set_ylabel("optical density")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("the sag is subtracted from the vessel: the target loses most of its OD", fontsize=9, loc="left")
    fig.suptitle(f"The background leak, how it happens (mask covers {mv[lm].mean():.0%} of this line's lumen; with "
                 f"the OD mask {mo[lm].mean():.0%})", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech_leak.jpg")


# ---------------------------------------------------------------------------------------------- step 2
def fig_step2(ctx, out):
    plt = _plt()
    s1, sec, cfg = ctx["s1"], _wide_section(ctx), ctx["cfg"]
    u, xs, ys = sec["u"], sec["xs"], sec["ys"]
    OD = s1["OD"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 7.2), layout="constrained")
    for r, s in enumerate((2.0, 8.0)):
        g1, g2 = N._gblur(OD, s), N._gblur(OD, 2 * s)
        D = g1 - g2
        rm = ctx["rms"][s]
        ax = axs[r, 0]
        ax.plot(u, _sample(OD, xs, ys), color=C_OD, lw=1.2, label="OD")
        ax.plot(u, _sample(g1, xs, ys), color="#1f77b4", lw=2, label=f"blurred, sigma {s:g} px (centre)")
        ax.plot(u, _sample(g2, xs, ys), color="#ff7f0e", lw=2, label=f"blurred, sigma {2 * s:g} px (surround)")
        ax.set_ylabel("OD")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"band s = {s:g}: a centre blur and a surround blur twice as wide", fontsize=9, loc="left")
        ax = axs[r, 1]
        d, q = _sample(D, xs, ys), _sample(rm, xs, ys)
        ax.fill_between(u, -q, q, color="#bbbbbb", alpha=0.5, label="+- the local noise level RMS_s")
        ax.plot(u, d, color="#2ca02c", lw=2, label="D_s = centre - surround")
        ax.plot(u, cfg.mask_z * q, color="#555555", lw=1, ls="--", label=f"{cfg.mask_z:g} x RMS_s (step 1's mask threshold)")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"D_s / RMS_s is the contrast-to-noise ratio of band s (here peak {np.max(d / q):.0f})",
                     fontsize=9, loc="left")
    for ax in axs[1]:
        ax.set_xlabel("px along the same line as step 1")
    fig.suptitle("Step 2, how it works: a centre-surround filter (difference of two blurs) measured against the "
                 "local noise", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech2_cnr.jpg")


# ---------------------------------------------------------------------------------------------- step 3
def _one_scale(ctx, si, odd_alpha):
    """Stage 3 with one scale only (the pipeline's simple_cells, isolated by its Config): U (16, H, W) at the
    full grid. odd_alpha 0 gives z_even; z_even - odd_alpha |z_odd| otherwise."""
    cfg = ctx["cfg"]
    ci = next(c for c, ch in enumerate(cfg.channels) if si in ch)
    c1 = replace(cfg, channels=((si,),), chan_factor=(cfg.chan_factor[ci],), elong=(cfg.elong[ci],), odd_alpha=odd_alpha)
    ch = N.simple_cells(ctx["s1"]["OD"], ctx["s1"]["bg"], c1)[0]
    return N._up(ch["U"], ch["f"], ch["shape"])


def _peaks(v, sep=3):
    """The two largest circular local maxima of a tuning curve at least sep layers apart: (k1, v1, k2, v2)."""
    n = len(v)
    loc = [k for k in range(n) if v[k] >= v[(k - 1) % n] and v[k] >= v[(k + 1) % n]]
    loc.sort(key=lambda k: -v[k])
    k1 = loc[0]
    rest = [k for k in loc[1:] if min(abs(k - k1), n - abs(k - k1)) >= sep]
    return (k1, v[k1], rest[0], v[rest[0]]) if rest else (k1, v[k1], None, 0.0)


def _pixels(ctx):
    """Three pixels for the tuning curves (the truth places them): on a vessel away from junctions; at the truth
    crossing whose two vessels respond most evenly (fine channel, within 2 px of it); on background texture
    (no vessel OD in the truth, >= 8 px from the vessel mask; the pixel at the 98th percentile of the fine
    channel's response there: texture that looks somewhat like a line)."""
    from scipy.spatial import cKDTree
    case, U = ctx["case"], ctx["U3"][0]
    H, W = U.shape[1:]
    x0, y0 = ctx["win"]
    js = case["obs"]["junctions"]
    tree = cKDTree(np.array([[j["x"], j["y"]] for j in js]))
    vessel = None
    for t in sorted(ctx["traces"][0], key=lambda t: -float(np.median(t["c"]))):
        for p in t["xy"][len(t["xy"]) // 4: 3 * len(t["xy"]) // 4]:
            if tree.query(p)[0] > 20 and x0 <= p[0] < x0 + Z and y0 <= p[1] < y0 + Z:
                vessel = (int(round(p[0])), int(round(p[1])))
                break
        if vessel is not None:
            break
    best = None
    for j in js:
        if j["type_observable"] != "crossing" or not (8 <= j["x"] < W - 8 and 8 <= j["y"] < H - 8):
            continue
        cx, cy = int(round(j["x"])), int(round(j["y"]))
        for y in range(cy - 2, cy + 3):
            for x in range(cx - 2, cx + 3):
                k1, v1, k2, v2 = _peaks(np.clip(U[:, y, x], 0, None))
                if k2 is not None and (best is None or v2 > best[0]):
                    best = (v2, (x, y))
    from scipy import ndimage as ndi
    far = ndi.distance_transform_edt(~ctx["s1"]["mask"]) >= 8
    bgm = far & (ctx["case"]["od_img"] < 0.003) & ctx["s1"]["ok"]
    bgm[:24], bgm[-24:], bgm[:, :24], bgm[:, -24:] = False, False, False, False
    m = U.max(0)
    q = float(np.percentile(m[bgm], 98))
    cand = np.argwhere(bgm & (np.abs(m - q) < 0.05 * q))
    ty, tx = cand[len(cand) // 2]
    return dict(vessel=vessel, crossing=best[1], texture=(int(tx), int(ty)))


def fig_step3(ctx, out):
    plt = _plt()
    cfg, sec = ctx["cfg"], _wide_section(ctx)
    u, xs, ys, k = sec["u"], sec["xs"], sec["ys"], sec["k"]
    th = N.orientations(cfg)
    fig = plt.figure(figsize=(17, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])
    for r, si in enumerate((1, 4)):
        ze = _one_scale(ctx, si, 0.0)[k]
        zg = _one_scale(ctx, si, 1.0)[k]
        zo = ze - zg                                                          # |z_odd|
        z = ze - cfg.odd_alpha * zo
        ax = fig.add_subplot(gs[r, 0:2])
        od = _sample(ctx["s1"]["OD"], xs, ys)
        ax2 = ax.twinx()
        ax2.fill_between(u, 0, od, color="#dddddd", lw=0)
        ax2.set_ylabel("OD (grey)", color="#888888")
        ax2.set_ylim(0, 1.6 * od.max())
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(u, _sample(ze, xs, ys), color="#1f77b4", lw=1.5, label="even (line) cell, z_even")
        ax.plot(u, _sample(zo, xs, ys), color="#ff7f0e", lw=1.5, label="odd (edge) cell, |z_odd|")
        ax.plot(u, _sample(z, xs, ys), color="k", lw=2.2, label=f"line evidence z = z_even - {cfg.odd_alpha:g} |z_odd|")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_ylabel("response (contrast-to-noise units)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"scale sigma = {cfg.scales[si]:g} px, orientation {math.degrees(th[k]):.0f} deg (the wide vessel's): "
                     f"responses along the line of step 1", fontsize=9, loc="left")
    ax.set_xlabel("px along the line")
    # tuning curves at three pixels
    px = _pixels(ctx)
    names = dict(vessel="on a vessel", crossing="at a crossing", texture="on textured background")
    cols = ("#ff4d4d", "#4da6ff", "#2ca02c")
    axp = [fig.add_subplot(gs[0, 2], projection="polar"), fig.add_subplot(gs[0, 3], projection="polar"),
           fig.add_subplot(gs[1, 2], projection="polar")]
    tt = np.r_[th, th + math.pi, th[:1] + 2 * math.pi]
    for ax, key in zip(axp, ("vessel", "crossing", "texture")):
        x, y = px[key]
        for ci in range(3):
            v = np.clip(ctx["U3"][ci][:, y, x], 0, None)
            ax.plot(tt, np.r_[v, v, v[:1]], color=cols[ci], lw=2, label=("fine", "medium", "coarse")[ci])
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)                                            # y down, as in the image
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.tick_params(labelsize=7)
        ax.set_title(f"{names[key]} ({x}, {y})", fontsize=9)
    axp[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(-0.35, 1.1))
    axi = fig.add_subplot(gs[1, 3])
    axi.imshow(ctx["gray"], cmap="gray", vmin=0, vmax=1)
    for key, c in zip(("vessel", "crossing", "texture"), ("#ff2b2b", "#4da6ff", "#2ca02c")):
        axi.plot(*px[key], "o", ms=9, mfc="none", mec=c, mew=2)
        axi.annotate(names[key], px[key], color=c, fontsize=8, xytext=(6, 6), textcoords="offset points")
    axi.set_xlim(-0.5, ctx["gray"].shape[1] - 0.5)
    axi.set_ylim(ctx["gray"].shape[0] - 0.5, -0.5)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title("where the three pixels are", fontsize=9, loc="left")
    fig.suptitle("Step 3, how it works. Left: across a vessel the line cell peaks at the centre, the edge cell at the "
                 "walls.\nRight: each pixel's response at every orientation (its tuning curve; radius = response, "
                 "the lobes point along the vessel)", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech3_simple_cells.jpg")
    return px


# ---------------------------------------------------------------------------------------------- steps 4-5
def _polar(ax, th, curves, title):
    tt = np.r_[th, th + math.pi, th[:1] + 2 * math.pi]
    for v, c, lab, ls in curves:
        v = np.clip(v, 0, None)
        ax.plot(tt, np.r_[v, v, v[:1]], color=c, lw=2, ls=ls, label=lab)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def _blob_pixel(ctx):
    """A background pixel (no vessel OD in the truth, >= 8 px from the vessel mask) whose fine-channel response
    is the most orientation-blind: the largest isotropic part (mean of the lower half over orientation) among
    those that reach the tracing threshold t_low at their best orientation."""
    from scipy import ndimage as ndi
    U = np.clip(ctx["U3"][0], 0, None)
    far = ndi.distance_transform_edt(~ctx["s1"]["mask"]) >= 8
    bgm = far & (ctx["case"]["od_img"] < 0.003) & ctx["s1"]["ok"]
    bgm[:24], bgm[-24:], bgm[:, :24], bgm[:, -24:] = False, False, False, False
    iso = np.sort(U, 0)[: U.shape[0] // 2].mean(0)
    score = np.where(bgm & (U.max(0) > ctx["cfg"].t_low), iso, -1)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return int(x), int(y)


def fig_step4(ctx, out, px):
    plt = _plt()
    th = N.orientations(ctx["cfg"])
    px = dict(px, blob=_blob_pixel(ctx))
    names = dict(blob="blob-like background texture", texture="line-like background texture", vessel="on a vessel",
                 crossing="at a crossing")
    fig = plt.figure(figsize=(19, 5.2), layout="constrained")
    gs = fig.add_gridspec(1, 4)
    for c, key in enumerate(("blob", "texture", "vessel", "crossing")):
        x, y = px[key]
        ax = fig.add_subplot(gs[0, c], projection="polar")
        v3, v4 = ctx["U3"][0][:, y, x], ctx["U4"][0][:, y, x]
        _polar(ax, th, [(v3, "#888888", "step 3: simple cells", "-"), (v4, "#d62728", "step 4: after the surround", "-")],
               f"{names[key]} ({x}, {y})\nbest orientation {np.max(v3):.1f} -> {np.max(v4):.1f}")
        if c == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(-0.3, 1.18))
    fig.suptitle("Step 4, how it works: tuning curves (fine channel) before and after the surround. The part of the "
                 "response that is the same at every orientation is subtracted:\nit removes most of a blob's response "
                 "and little of a line's, but a crossing (two orientations) also loses part of its response",
                 fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech4_surround.jpg")
    return px


def _along(ctx, ci, t):
    """U4 and C5 of channel ci sampled along a trace, in the orientation layer of the trace's tangent."""
    n = ctx["cfg"].n_orient
    tg = N._tangents(t["xy"], 2)
    k = np.round((np.arctan2(tg[:, 1], tg[:, 0]) % math.pi) / (math.pi / n)).astype(int) % n
    xs, ys = t["xy"][:, 0], t["xy"][:, 1]
    u4 = np.array([N._bilinear(ctx["U4"][ci][kk], np.array([x]), np.array([y]))[0] for kk, x, y in zip(k, xs, ys)])
    c5 = np.array([N._bilinear(ctx["C5"][ci][kk], np.array([x]), np.array([y]))[0] for kk, x, y in zip(k, xs, ys)])
    return k, u4, c5


def fig_step5(ctx, out):
    plt = _plt()
    cfg = ctx["cfg"]
    # the trace (fine channel) where the association field lifts the most samples from below the seed threshold
    best = None
    for t in ctx["traces"][0]:
        if len(t["xy"]) < 40:
            continue
        k, u4, c5 = _along(ctx, 0, t)
        gain = int(((u4 < cfg.t_high) & (c5 >= cfg.t_high)).sum())
        if best is None or gain > best[0]:
            best = (gain, t, k, u4, c5)
    _, t, k, u4, c5 = best
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(t["xy"], axis=0).T))]
    fig = plt.figure(figsize=(17, 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 1.6])
    # the bipole lobes of one orientation layer (fine channel): the weights w(d) at p +- d t
    sig = ctx["chans"][0]["sigma"]
    ell = cfg.assoc_len * math.sqrt(max(1.0, sig / 2))
    ax = fig.add_subplot(gs[0, 0])
    tk = math.pi / 6
    d = np.arange(1, int(3 * ell) + 1)
    wd = np.exp(-d ** 2 / (2 * ell ** 2))
    for sgn, c in ((1, "#1f77b4"), (-1, "#ff7f0e")):
        ax.scatter(sgn * d * math.cos(tk), sgn * d * math.sin(tk), s=60 * wd / wd.max() + 4, color=c,
                   label=("lobe ahead, A+" if sgn > 0 else "lobe behind, A-"))
    ax.plot(0, 0, "k+", ms=14, mew=2)
    ax.annotate("p", (0, 0), xytext=(-12, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"the bipole of pixel p in one orientation layer\n(fine channel: l = {ell:.1f} px, dots out to 3 l; "
                 f"size = weight)", fontsize=9, loc="left")
    ax.set_xlabel("px")
    # where the trace is
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(ctx["gray"], cmap="gray", vmin=0, vmax=1)
    ax.plot(t["xy"][:, 0], t["xy"][:, 1], "-", color="#ffd400", lw=2)
    ax.plot(*t["xy"][0], "o", color="#ffd400", ms=6)
    pad = 30
    ax.set_xlim(t["xy"][:, 0].min() - pad, t["xy"][:, 0].max() + pad)
    ax.set_ylim(t["xy"][:, 1].max() + pad, t["xy"][:, 1].min() - pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("a traced vessel (yellow; dot = start)", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(s, u4, color="#888888", lw=2, label="before: step 4's response in the vessel's orientation layer")
    ax.plot(s, c5, color="#d62728", lw=2, label="after 3 steps of the association field")
    ax.axhline(cfg.t_high, color="k", lw=1, ls="--", label=f"seed threshold {cfg.t_high:g}")
    ax.axhline(cfg.t_low, color="k", lw=1, ls=":", label=f"continue threshold {cfg.t_low:g}")
    ax.fill_between(s, 0, 1, where=(u4 < cfg.t_high) & (c5 >= cfg.t_high), color="#2ca02c", alpha=0.15,
                    transform=ax.get_xaxis_transform(), label="lifted above the seed threshold")
    ax.set_xlabel("px along the vessel")
    ax.set_ylabel("response (contrast-to-noise units)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("along the vessel: each pixel is averaged with its collinear neighbours: dips between strong "
                 "stretches rise, peaks fall", fontsize=9, loc="left")
    fig.suptitle("Step 5, how it works: a pixel is supported only when there is line evidence both ahead of it and "
                 "behind it in its own orientation (bipole = sqrt(A+ x A-)); three steps of C = (U + bipole(C)) / 2",
                 fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech5_association.jpg")


# ---------------------------------------------------------------------------------------------- step 6
def _candidates(C, cfg, valid):
    """Stage 6's candidate rule (readout, re-stated for display): a local maximum across the line (+-1 px along
    the normal), at least its two neighbouring orientation layers, above t_low, border_px inside valid data."""
    from scipy import ndimage as ndi
    n, H, W = C.shape
    th = N.orientations(cfg)
    cand = np.zeros(C.shape, bool)
    for k, t in enumerate(th):
        nx, ny = -math.sin(t), math.cos(t)
        cand[k] = (C[k] >= N._shift(C[k], nx, ny)) & (C[k] > N._shift(C[k], -nx, -ny)) & (C[k] > cfg.t_low)
    cand &= (C >= np.roll(C, 1, 0)) & (C > np.roll(C, -1, 0))
    b = cfg.border_px
    inner = ndi.binary_erosion(valid, iterations=b)
    inner[:b], inner[-b:], inner[:, :b], inner[:, -b:] = False, False, False, False
    return cand & inner[None]


def _patch(ax, gray, cx, cy, half, title=None):
    from matplotlib.patches import Rectangle
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    sub = gray[max(y0, 0):y0 + 2 * half + 1, max(x0, 0):x0 + 2 * half + 1]
    lo, hi = np.percentile(sub, [1, 99]) if sub.size else (0, 1)
    ax.imshow(np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlim(x0 - 0.5, x0 + 2 * half + 0.5)
    ax.set_ylim(y0 + 2 * half + 0.5, y0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    return Rectangle


def fig_step6(ctx, out, px):
    plt = _plt()
    cfg = ctx["cfg"]
    C = ctx["C5"][0]
    cand = _candidates(C, cfg, ctx["s1"]["ok"])
    x, y = px["crossing"]
    k1, _, k2, _ = _peaks(np.clip(C[:, y, x], 0, None))
    n = cfg.n_orient
    deg = lambda k: math.degrees(k * math.pi / n)                                  # noqa: E731
    fig = plt.figure(figsize=(18, 6.0), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
    # (a) candidates in the two crossing vessels' layers
    ax = fig.add_subplot(gs[0, 0])
    half = 22
    _patch(ax, ctx["img"].astype(float), x, y, half,
           f"(a) candidates (ridge tops above {cfg.t_low:g}) in the two vessels' orientation layers")
    for kk, col in ((k1, "#ff3b3b"), (k2, "#3b8bff")):
        ys, xs = np.nonzero(cand[[(kk - 1) % n, kk, (kk + 1) % n]].any(0))
        m = (np.abs(xs - x) <= half) & (np.abs(ys - y) <= half)
        ax.plot(xs[m], ys[m], "s", ms=4, color=col, alpha=0.85,
                label=f"layers {deg(kk):.0f} deg +- one ({'first' if kk == k1 else 'second'} vessel)")
    ax.legend(fontsize=8, frameon=True, loc="lower left")
    # (b) one tracing step, re-stated from readout's walk: from a candidate on the first vessel, 8 px before the
    # crossing, the 3 lateral positions x 3 layers one pixel ahead, their scores, the winner
    t1 = np.array([math.cos(k1 * math.pi / n), math.sin(k1 * math.pi / n)])
    best0 = None
    for dd in range(6, 14):
        for sg in (1, -1):
            qx, qy = int(round(x - sg * dd * t1[0])), int(round(y - sg * dd * t1[1]))
            if cand[k1, qy, qx] and (best0 is None or dd < best0[0]):
                best0 = (dd, qx, qy, sg)
    _, cx0, cy0, sg = best0
    dx, dy = sg * t1
    opts = []
    for dk in (0, -1, 1):
        kk = (k1 + dk) % n
        tx, ty = math.cos(kk * math.pi / n), math.sin(kk * math.pi / n)
        if tx * dx + ty * dy < 0:
            tx, ty = -tx, -ty
        nx, ny = -ty, tx
        for o in (0, -1, 1):
            qy, qx = int(round(cy0 + ty + o * ny)), int(round(cx0 + tx + o * nx))
            ok = bool(cand[kk, qy, qx])
            sc = float(C[kk, qy, qx]) * (1 - 0.15 * abs(o)) * (1 - 0.1 * abs(dk))
            opts.append((sc if ok else -1.0, dk, o, qx, qy, ok, float(C[kk, qy, qx])))
    win = max(opts)
    ax = fig.add_subplot(gs[0, 1])
    grid = np.full((3, 3), np.nan)
    for sc, dk, o, qx, qy, ok, c in opts:
        grid[dk + 1, o + 1] = sc if ok else np.nan
    ax.imshow(np.nan_to_num(grid, nan=0.0), cmap="Greens", vmin=0, vmax=max(1.0, np.nanmax(grid) * 1.2))
    for sc, dk, o, qx, qy, ok, c in opts:
        txt = (f"C = {c:.1f}\nx {1 - 0.15 * abs(o):.2f} x {1 - 0.1 * abs(dk):.2f}\n= {sc:.1f}" if ok else
               f"C = {c:.1f}\nnot a ridge top\n(not a candidate)")
        ax.text(o + 1, dk + 1, txt, ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((win[2] + 1 - 0.5, win[1] + 1 - 0.5), 1, 1, fill=False, ec="#d62728", lw=3))
    ax.set_xticks([0, 1, 2], ["1 px to one side", "straight ahead", "1 px to the other side"], fontsize=8)
    ax.set_yticks([0, 1, 2], [f"layer - 1\n({deg((k1 - 1) % n):.0f} deg)", f"same layer\n({deg(k1):.0f} deg)",
                              f"layer + 1\n({deg((k1 + 1) % n):.0f} deg)"], fontsize=8)
    ax.set_title(f"(b) one step from ({cx0}, {cy0}): 9 positions 1 px ahead (red = chosen)", fontsize=9, loc="left")
    # (c) the orientation of the fine trace through the crossing, +-40 px around it
    ax = fig.add_subplot(gs[0, 2])
    tr = min(ctx["traces"][0], key=lambda t: float(np.min(np.hypot(t["xy"][:, 0] - x, t["xy"][:, 1] - y))))
    tg = N._tangents(tr["xy"], 2)
    ang = np.degrees(np.arctan2(tg[:, 1], tg[:, 0]) % math.pi)
    sarc = np.r_[0, np.cumsum(np.hypot(*np.diff(tr["xy"], axis=0).T))]
    i0 = int(np.argmin(np.hypot(tr["xy"][:, 0] - x, tr["xy"][:, 1] - y)))
    m = np.abs(sarc - sarc[i0]) <= 40
    ref = ang[i0]
    ang = ref + (ang - ref + 90) % 180 - 90                                      # no jumps at 0 / 180
    ax.plot(sarc[m] - sarc[i0], ang[m], "-", color="#ff3b3b", lw=2.5, label="the traced vessel's orientation")
    other = ref + (deg(k2) - ref + 90) % 180 - 90
    ax.axhline(other, color="#3b8bff", lw=2, ls=":", label="the crossing vessel's orientation")
    ax.axvline(0, color="#888888", lw=1.5, ls="--", label="the crossing")
    ax.set_xlabel("px along the trace from the crossing")
    ax.set_ylabel("orientation (deg)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_title("(c) the trace keeps its own orientation straight through the crossing", fontsize=9, loc="left")
    fig.suptitle("Step 6, how it works: each vessel is a ridge in its own orientation layer, so two vessels can cross "
                 "without their ridges touching; a trace steps along its ridge", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech6_tracing.jpg")
    return dict(layers=(deg(k1), deg(k2)), step=[(round(o[0], 2), o[1], o[2], o[5]) for o in opts])


# ---------------------------------------------------------------------------------------------- steps 7-8
def stage78(ctx):
    """The pipeline's stage 7 on the merged traces (_assemble_profiled), its events and clusters re-derived the
    way end_stopping does (checked against its junctions), and stage 8's first prune round."""
    cfg, OD, s1 = ctx["cfg"], ctx["s1"]["OD"], ctx["s1"]
    edges, juncs, traces = N._assemble_profiled([dict(t) for t in ctx["merged"]], cfg, OD)
    tr = [dict(t) for t in ctx["merged"] if N._plen(t["xy"]) >= cfg.min_len]
    tr = N._join_gaps(tr, cfg)
    tr.sort(key=lambda t: (-len(t["xy"]), float(t["xy"][0, 0]), float(t["xy"][0, 1])))
    for t in tr:
        t["w_ro"] = np.asarray(t["w"], float).copy()
        t["w_od"], t["a_od"] = N.od_widths(OD, t["xy"], t["w"])
        t["w"] = np.clip(t["w_od"], 1.5, None)
    ev = N._events(tr, cfg)
    groups = N._cluster_events(ev, cfg)
    cl = []
    for g in groups:
        E = [ev[k] for k in g]
        c = np.mean([e["p"] for e in E], 0)
        R = max(math.hypot(*(e["p"] - c)) + cfg.r0 + 0.5 * e["wide"] for e in E)
        cl.append((c, R, E))
    for J in juncs:                                    # every junction is one of the re-derived clusters
        assert min(math.hypot(J["x"] - c[0], J["y"] - c[1]) for c, _, _ in cl) < 1e-6
    hp = OD - N._gblur(OD, 8.0)
    sig2 = np.where(s1["ok"], np.maximum(N._local_rms(hp, s1["bg"], 32.0), 1e-4) ** 2, 1e12)
    jm = N._junction_mask(OD.shape, juncs, cfg)
    sw = np.where(jm, 1e12, sig2)
    rd0 = N._Render(OD, sw, edges, cfg)
    gains = np.array([rd0.gain(k) for k in range(len(edges))])
    keep, rd = N._mdl_prune(OD, sw, edges, cfg, ctx["rms"])
    cost = np.array([cfg.mdl_k * (1 + N._plen(e["xy"]) / cfg.mdl_len) for e in edges])
    cnr = np.array([e.get("cnr", np.inf) for e in edges])
    free = np.array([e["j"][0] is None or e["j"][1] is None for e in edges])
    return dict(edges=edges, juncs=juncs, traces=tr, events=ev, clusters=cl, keep=np.array(keep), gains=gains,
                cost=cost, cnr=cnr, free=free, render=OD - rd.residual())


def _decision(J, cfg):
    arms = J["arms"]
    n = len(arms)
    if n == 3:
        s = "3 arms -> 3-way"
    elif n == 4:
        best = None
        for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            ua, ub, uc, ud = (arms[k][2] for k in (a, b, c, d))
            t1, t2 = N._angle(ua, -ub), N._angle(uc, -ud)
            same = (arms[a][0] == arms[b][0]) + (arms[c][0] == arms[d][0])
            key = (-same, max(t1, t2))
            if best is None or key < best[0]:
                best = (key, ua - ub, uc - ud, max(t1, t2))
        _, ax1, ax2, turn = best
        ax1, ax2 = ax1 / np.linalg.norm(ax1), ax2 / np.linalg.norm(ax2)
        sep = min(N._angle(ax1, ax2), N._angle(ax1, -ax2))
        ok = turn <= cfg.cross_turn_deg and sep >= 15.0
        s = (f"4 arms pair into 2 lines: worst turn {turn:.0f} deg ({'<=' if turn <= cfg.cross_turn_deg else '>'} "
             f"{cfg.cross_turn_deg:g}), lines {sep:.0f} deg apart ({'>=' if sep >= 15 else '<'} 15) -> "
             f"{'crossing' if ok else 'compound'}")
    else:
        s = f"{n} arms -> compound"
    if J["r"] >= cfg.compound_r and J["geom"] != "compound":
        s += f"; radius {J['r']:.0f} >= {cfg.compound_r:g} px -> compound"
    return s


def fig_step7(ctx, out, st):
    plt = _plt()
    cfg = ctx["cfg"]
    x0, y0 = ctx["win"]
    cx, cy = x0 + Z / 2, y0 + Z / 2
    picks = []
    for typ in ("pseudo-T", "crossing", "compound"):
        js = [J for J in st["juncs"] if J["geom"] == typ and J["type"] == typ and J["r"] < 14]
        if typ == "compound":
            js = [J for J in st["juncs"] if J["type"] == "compound" and len(J["arms"]) >= 5] or js
        picks.append(min(js, key=lambda J: math.hypot(J["x"] - cx, J["y"] - cy)))
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), layout="constrained")
    cm = plt.get_cmap("tab10")
    for ax, J in zip(axs, picks):
        c = np.array([J["x"], J["y"]])
        half = int(max(22, J["r"] + 16))
        _patch(ax, ctx["img"].astype(float), c[0], c[1], half)
        for t in st["traces"]:
            ax.plot(t["xy"][:, 0], t["xy"][:, 1], "-", color="#ffffff", lw=0.8, alpha=0.5)
        col = {tid: cm(i % 10) for i, tid in enumerate(J["tids"])}
        for tid in J["tids"]:
            P = st["traces"][tid]["xy"]
            ax.plot(P[:, 0], P[:, 1], "-", color=col[tid], lw=2.2)
        E = next(E for cc, R, E in st["clusters"] if math.hypot(cc[0] - c[0], cc[1] - c[1]) < 1e-6)
        for e in E:
            mk, mc = ("^", "#ff9f1c") if e["kind"] == "T" else ("X", "#ff2bd6")
            ax.plot(*e["p"], mk, ms=11, mfc=mc, mec="k", mew=0.8)
        ax.add_patch(plt.Circle(c, J["r"], fill=False, ec="w", ls="--", lw=1.5))
        for tid, side, u, L, _ in J["arms"]:
            ax.annotate("", xy=c + u * (J["r"] + 9), xytext=c, arrowprops=dict(arrowstyle="->", color=col[tid], lw=2.2))
            ax.annotate(f"{L:.0f} px", c + u * (J["r"] + 12), color="w", fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", alpha=0.6, lw=0))
        ax.plot(*c, "+", color="w", ms=12, mew=2)
        n_t = sum(e["kind"] == "T" for e in E)
        ax.set_title(f"{J['type']} at ({c[0]:.0f}, {c[1]:.0f}): {n_t} T and {len(E) - n_t} X events, radius "
                     f"{J['r']:.1f} px\n{_decision(J, cfg)}", fontsize=9, loc="left")
    fig.suptitle("Step 7, how it works: events (orange triangle = a trace ends on another, a 'T'; magenta cross = two "
                 "traces cross, an 'X') are grouped into one junction (white disc).\nThe traces leaving the disc are "
                 "its arms (arrows; the number is the arm's length), and the number of arms and how they pair decide "
                 "the type", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech7_junctions.jpg")
    return [(J["type"], round(J["x"]), round(J["y"]), len(J["arms"])) for J in picks]


def fig_step8(ctx, out, st):
    plt = _plt()
    cfg = ctx["cfg"]
    g = st["gains"] / st["cost"]
    v = st["cnr"] / cfg.cnr_min
    fr, kp = st["free"], st["keep"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    clip = lambda a: np.clip(a, 1e-3, 1e4)                                          # noqa: E731
    ax.scatter(clip(g[~fr]), clip(v[~fr]), s=14, color="#bbbbbb", label="edge between two junctions (never removed)")
    ax.scatter(clip(g[fr & kp]), clip(v[fr & kp]), s=22, color="#1f77b4", label="free-end edge, kept")
    ax.scatter(clip(g[fr & ~kp]), clip(v[fr & ~kp]), s=40, color="#d62728", marker="x", label="free-end edge, removed")
    ax.axvline(1, color="k", lw=1, ls="--")
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test 1, pays for itself: gain / cost  (gain = weighted squared error it removes; cost = "
                  f"{cfg.mdl_k:g} (1 + length / {cfg.mdl_len:g}))", fontsize=9)
    ax.set_ylabel(f"test 2, visible: contrast-to-noise / {cfg.cnr_min:g}", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"every edge of the first round ({len(g)}): a free-end edge must pass both tests (upper right); "
                 f"{int((fr & ~kp).sum())} removed", fontsize=9, loc="left")
    # the removed edges on the image, with their reason
    ax = axs[1]
    ax.imshow(ctx["gray"], cmap="gray", vmin=0, vmax=1)
    for e, k in zip(st["edges"], kp):
        ax.plot(e["xy"][:, 0], e["xy"][:, 1], "-", color="#1f77b4" if k else "#d62728", lw=1.0 if k else 2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("kept (blue) and removed (red) edges", fontsize=9, loc="left")
    fig.suptitle("Step 8, how it works: render all edges, then remove, one at a time, the free-end edge that fails "
                 "a test worst, re-scoring its neighbours (values shown are before any removal)", fontsize=12,
                 x=0.01, ha="left")
    _save(fig, out, "mech8_prune.jpg")
    return dict(n=len(g), free=int(fr.sum()), removed=int((~kp).sum()),
                fail_gain=int((fr & (g < 1)).sum()), fail_cnr=int((fr & (v < 1)).sum()))


# ---------------------------------------------------------------------------------------------- steps 9-12
def fit_run(ctx):
    """The pipeline's spline fit on the example image, with every optimisation stage's loss history recorded
    (fit.optimize wrapped; nothing else changes)."""
    from experiments.splinefit import fit as F
    from experiments.splinefit import pipeline as P
    hists = []
    orig = F.optimize

    def rec(*a, **k):
        h = orig(*a, **k)
        hists.append(dict(iters=a[1] if len(a) > 1 else k.get("iters"), fit_pos=k.get("fit_pos", True), hist=h))
        return h
    F.optimize = rec
    try:
        dbg = {}
        outp = P.annotate_cfg(ctx["img"].copy(), ctx["valid"].copy(), P.DEFAULT, debug=dbg)
    finally:
        F.optimize = orig
    return dict(out=outp, dbg=dbg, hists=hists)


def fig_step9(ctx, out, fr):
    from experiments.splinefit.proposals import build_network
    plt = _plt()
    dbg = fr["dbg"]
    prop = dbg["proposal"]
    net0, info = build_network(prop, ctx["s1"]["OD"].shape)
    x0, y0 = ctx["win"]
    cx, cy = x0 + Z / 2, y0 + Z / 2
    picks = []
    for want in ("crossing", "node"):
        best = None
        for ji, J in enumerate(prop["junctions"]):
            ends = sum((e["j"][0] == ji) + (e["j"][1] == ji) for e in prop["edges"])
            if want == "crossing" and not (J["type"] == "crossing" and ends == 4):
                continue
            if want == "node" and not (J["type"] != "crossing" and ends >= 3):
                continue
            d = math.hypot(J["x"] - cx, J["y"] - cy)
            if best is None or d < best[0]:
                best = (d, J)
        picks.append(best[1])
    fig, axs = plt.subplots(2, 2, figsize=(12, 12.4), layout="constrained")
    cm = plt.get_cmap("tab10")
    for r, J in enumerate(picks):
        half = int(max(26, J["r"] + 18))
        for c, what in enumerate(("proposal", "network")):
            ax = axs[r, c]
            _patch(ax, ctx["img"].astype(float), J["x"], J["y"], half)
            if what == "proposal":
                k = 0
                for e in prop["edges"]:
                    P_ = np.asarray(e["xy"])
                    if np.min(np.hypot(P_[:, 0] - J["x"], P_[:, 1] - J["y"])) > half * 1.4:
                        continue
                    ax.plot(P_[:, 0], P_[:, 1], "-", color=cm(k % 10), lw=2.2)
                    ax.plot(*P_[0], "o", color=cm(k % 10), ms=5, mec="k")
                    ax.plot(*P_[-1], "o", color=cm(k % 10), ms=5, mec="k")
                    k += 1
                ax.add_patch(plt.Circle((J["x"], J["y"]), J["r"], fill=False, ec="w", ls="--", lw=1.2))
                ax.set_title(f"proposal: a {J['type']} junction; edges stop at it (dots = ends)", fontsize=9, loc="left")
            else:
                k = 0
                for eid in net0.edges:
                    smp = net0.sample(eid, 1.0)
                    P_ = smp["xy"]
                    if np.min(np.hypot(P_[:, 0] - J["x"], P_[:, 1] - J["y"])) > half * 1.4:
                        continue
                    ax.plot(P_[:, 0], P_[:, 1], "-", color=cm(k % 10), lw=2.2)
                    ctrl = net0.edges[eid].ctrl
                    ax.plot(ctrl[:, 0], ctrl[:, 1], "o", color=cm(k % 10), ms=4, mec="k", mew=0.4)
                    k += 1
                for nid, nd in net0.nodes.items():
                    if abs(nd.x - J["x"]) <= half and abs(nd.y - J["y"]) <= half:
                        ax.plot(nd.x, nd.y, "o", ms=13, mfc="none", mec="w", mew=2)
                ax.set_title("spline network (dots = control points, ring = node)", fontsize=9, loc="left")
    fig.suptitle("Step 9, how it works: at a crossing the four ends pair into two continuous curves and no node is "
                 "made;\nat a fork the parent vessel continues through the node and the branch ends on it",
                 fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech9_network.jpg")
    return [(J["type"], round(J["x"]), round(J["y"])) for J in picks]


def fig_step10(ctx, out, fr):
    import torch
    from experiments.splinefit import fit as F
    from experiments.splinefit import pipeline as P
    plt = _plt()
    dbg, outp = fr["dbg"], fr["out"]
    model, net = dbg["model"], dbg["net"]
    T = np.asarray(outp["od_target"], np.float32)
    with torch.no_grad():
        Rj = model.optical_density().numpy()
        kap = float(model.kappa())
        madd = F.make_model(net, T, dbg["weight"], replace(P.DEFAULT.fit, junctions=False), kap)
        Ra = madd.optical_density().numpy()
    picks = []
    for kind in ("node", "cross"):
        H, W = T.shape
        si = [q for q in model.site_info if q["kind"] == kind and len(q["edges"]) == (3 if kind == "node" else 2)
              and len(q["centres"]) == 1 and 30 <= q["x"] < W - 30 and 30 <= q["y"] < H - 30]
        if kind == "node":                             # the fork where the union changes the render most
            best = max(si, key=lambda q: float(np.abs(Rj - Ra)[int(q["y"]) - 2:int(q["y"]) + 3,
                                                                int(q["x"]) - 2:int(q["x"]) + 3].max()))
        else:                                          # the most visible crossing (target OD at its centre)
            best = max(si, key=lambda q: float(T[int(round(q["y"])), int(round(q["x"]))]))
        picks.append(best)
        print(kind, "sites", len(si))
    v = float(np.percentile(T[ctx["valid"]], 99.5))
    fig, axs = plt.subplots(2, 4, figsize=(18, 9.6), layout="constrained")
    for r, q in enumerate(picks):
        half = int(min(max(18, q["reach"] + 8), 36))
        for c, (img, cmap, lo, hi, ttl) in enumerate((
                (T, "gray_r", 0, v, "the target OD"),
                (Ra, "gray_r", 0, v, "plain sum of the edges"),
                (Rj, "gray_r", 0, v, "the junction-aware render (what is fitted)"),
                (Rj - Ra, "RdBu_r", -v / 3, v / 3, "junction-aware minus plain sum"))):
            ax = axs[r, c]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xlim(q["x"] - half, q["x"] + half)
            ax.set_ylim(q["y"] + half, q["y"] - half)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{'fork (node)' if q['kind'] == 'node' else 'crossing'}: {ttl}", fontsize=9, loc="left")
        yc, xc = int(round(q["y"])), int(round(q["x"]))
        axs[r, 3].set_xlabel(f"at the centre: target {T[yc, xc]:.3f}, plain sum {Ra[yc, xc]:.3f}, junction-aware "
                             f"{Rj[yc, xc]:.3f} OD", fontsize=9)
        print(q["kind"], "centre T, Ra, Rj", round(float(T[yc, xc]), 3), round(float(Ra[yc, xc]), 3),
              round(float(Rj[yc, xc]), 3))
    fig.suptitle(f"Step 10, how it works: at a fork (top) the plain sum double-counts where the lumens overlap and the "
                 f"union removes it (blue);\nat a crossing (bottom) the overlap is reduced by (1 - kappa), kappa = "
                 f"{kap:.2f} here", fontsize=12, x=0.01, ha="left")
    _save(fig, out, "mech10_junction_render.jpg")
    return dict(kappa=kap, sites=[(q["kind"], round(q["x"]), round(q["y"])) for q in picks])


def fig_step12(ctx, out, fr):
    plt = _plt()
    hs, log = fr["hists"], fr["dbg"]["log"]
    names = ("1. profiles and optics only (geometry fixed)", "2. everything, positions anchored", "4. profiles again, "
             "on the retargeted target")
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    off = 0
    cols = ("#1f77b4", "#d62728", "#2ca02c")
    for i, h in enumerate(hs):
        arr = np.array(h["hist"])
        it = off + np.arange(len(arr))
        ax.plot(it, arr[:, 1], "-", color=cols[i % 3], lw=2, label=(names[i] + ": data error") if i < 3 else None)
        ax.plot(it, arr[:, 0], ":", color=cols[i % 3], lw=1.2, label="... plus the priors" if i == 0 else None)
        off += len(arr)
        if i == 1:
            pr = next((q for q in log if "prune" in q), None)
            ax.axvline(off - 0.5, color="k", lw=1)
            ax.annotate(f"3. prune: {pr['prune']} edges removed, then retarget", (off, float(arr[:, 0].max())),
                        fontsize=8, rotation=90, va="top", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("data error = 0.5 sum w (OD - R)^2. Small bumps every 25 iterations: pixels re-assigned to edges. "
                 "Stage 4 fits a new target", fontsize=9, loc="left")
    ax = axs[1]
    lr = [0.5 * (1 + math.cos(math.pi * i / 100)) * 0.9 + 0.1 for i in range(100)]
    ax.plot(np.arange(100), lr, color="k", lw=2)
    ax.set_xlabel("iteration within a stage (here 100)")
    ax.set_ylabel("learning-rate factor")
    ax.set_title("each stage's step size decays from 1 to 0.1 on a cosine", fontsize=9, loc="left")
    fig.suptitle("Steps 11-12, how it works: gradient descent (Adam) on the loss, in stages", fontsize=12, x=0.01,
                 ha="left")
    _save(fig, out, "mech12_fit.jpg")
    return [(h["iters"], round(h["hist"][0][1]), round(h["hist"][-1][1])) for h in hs]


def run(scene_dir: str, mask_scene_dir: str, out: str, only=None):
    from experiments.splinefit import fastset as FS
    FS.check_not_heldout(scene_dir)
    FS.check_not_heldout(mask_scene_dir)
    os.makedirs(out, exist_ok=True)
    ctx = context(scene_dir)
    info = {}
    figs = dict(step1=lambda: fig_step1(ctx, out), leak=lambda: fig_leak(mask_scene_dir, out),
                step2=lambda: fig_step2(ctx, out), step3=lambda: fig_step3(ctx, out),
                step4=lambda: fig_step4(ctx, out, _pixels(ctx)), step5=lambda: fig_step5(ctx, out),
                step6=lambda: fig_step6(ctx, out, _pixels(ctx)))
    for name, f in figs.items():
        if only and name not in only:
            continue
        info[name] = f()
    if not only or {"step7", "step8"} & set(only):
        st = stage78(ctx)
        info["step7"] = fig_step7(ctx, out, st)
        info["step8"] = fig_step8(ctx, out, st)
    if not only or {"step9", "step10", "step12"} & set(only):
        fr = fit_run(ctx)
        info["step9"] = fig_step9(ctx, out, fr)
        info["step10"] = fig_step10(ctx, out, fr)
        info["step12"] = fig_step12(ctx, out, fr)
    return info


def main(argv=None):
    from experiments.splinefit import fastset as FS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="healthy_s007_480x768")
    ap.add_argument("--mask-scene", default="pathologic_s000_480x768")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    info = run(os.path.join(FS.DEV_DIR, a.scene), os.path.join(FS.DEV_DIR, a.mask_scene), a.out, a.only)
    for k, v in info.items():
        print(k, v)


if __name__ == "__main__":
    main()
