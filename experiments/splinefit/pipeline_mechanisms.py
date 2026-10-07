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


def run(scene_dir: str, mask_scene_dir: str, out: str, only=None):
    from experiments.splinefit import fastset as FS
    FS.check_not_heldout(scene_dir)
    FS.check_not_heldout(mask_scene_dir)
    os.makedirs(out, exist_ok=True)
    ctx = context(scene_dir)
    info = {}
    figs = dict(step1=lambda: fig_step1(ctx, out), leak=lambda: fig_leak(mask_scene_dir, out),
                step2=lambda: fig_step2(ctx, out), step3=lambda: fig_step3(ctx, out),
                step4=lambda: fig_step4(ctx, out, _pixels(ctx)), step5=lambda: fig_step5(ctx, out))
    for name, f in figs.items():
        if only and name not in only:
            continue
        info[name] = f()
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
