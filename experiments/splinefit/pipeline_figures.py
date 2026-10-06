r"""Figures of experiments/PIPELINE.md: one development image followed through every step from image to graph.

    python -m experiments.splinefit.pipeline_figures --out DIR [--scene healthy_s007_480x768] [--mask-scene pathologic_s000_480x768]

Runs the neuromimetic stages one by one (their own functions, so every intermediate is the pipeline's own) and
the spline fit with a debug dict on a dev averaged still, and draws per step the full frame with a zoom on the
192 px window holding the most observable truth junctions. Dev scenes only (fastset.check_not_heldout); the
truth is read only to choose the zoom and to draw the last figures.
"""
from __future__ import annotations

import argparse
import math
import os
import textwrap

import numpy as np
from scipy.spatial import cKDTree

from experiments.neuromimetic import neuromimetic as N

from . import score as S
from .report_figures import COARSE, JSTYLE, TRUTH

Z = 192          # zoom window, px
DPI = 90


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _norm(X, valid=None, lo=0.5, hi=99.5):
    v = X[valid] if valid is not None and valid.any() else X.ravel()
    a, b = np.percentile(v[np.isfinite(v)], [lo, hi])
    return np.clip((X - a) / max(b - a, 1e-9), 0, 1)


def _zoom_window(case):
    """The 192 px window (16 px inside the border) with the most kinds of truth junction, then the most junctions."""
    js = [(j["x"], j["y"], COARSE.get(j["type_observable"], "3-way")) for j in case["obs"]["junctions"]]
    H, W = case["image"].shape
    best = None
    for y0 in range(16, H - Z - 16 + 1, 16):
        for x0 in range(16, W - Z - 16 + 1, 16):
            inn = [t for x, y, t in js if x0 + 12 <= x < x0 + Z - 12 and y0 + 12 <= y < y0 + Z - 12]
            key = (len(set(inn)), len(inn), -y0, -x0)
            if best is None or key > best[0]:
                best = (key, x0, y0)
    return best[1], best[2]


def _show(ax, img, cmap="gray", vmin=None, vmax=None, title=None, win=None, interp="antialiased"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    if win is not None:
        x0, y0 = win
        ax.set_xlim(x0 - 0.5, x0 + Z - 0.5)
        ax.set_ylim(y0 + Z - 0.5, y0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, loc="left")


def _box(ax, win):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((win[0], win[1]), Z, Z, fill=False, ec="#ffd400", lw=1.2))


def _pair_fig(panels, win, path, suptitle, ncols=None, extra=None):
    """panels: [(title, draw(ax, win_or_None))]; each as the full frame above its zoom, ncols panels per row (one
    panel: the frame beside its zoom). extra(fig), if given, draws on the figure before it is saved."""
    plt = _plt()
    n = len(panels)
    asp = 480 / 768
    if n == 1:
        fig, axs = plt.subplots(1, 2, figsize=(4.6 / asp + 4.6 + 0.6, 4.6 + 1.1), layout="constrained",
                                gridspec_kw=dict(width_ratios=[1 / asp, 1.0]))
        pairs, nc = [(axs[0], axs[1])], 2
    else:
        nc = ncols or n
        nb = -(-n // nc)
        fig, axs = plt.subplots(2 * nb, nc, figsize=(4.6 * nc, nb * 4.6 * (1 + asp) + 0.9), layout="constrained",
                                gridspec_kw=dict(height_ratios=[asp, 1.0] * nb))
        axs = np.asarray(axs).reshape(2 * nb, nc)
        for ax in axs.ravel():
            ax.set_axis_off()
        pairs = [(axs[2 * (k // nc), k % nc], axs[2 * (k // nc) + 1, k % nc]) for k in range(n)]
    for (title, draw), (top, bot) in zip(panels, pairs):
        top.set_axis_on()
        bot.set_axis_on()
        draw(top, None)
        top.set_title(textwrap.fill(title, 62 if n > 1 else 110), fontsize=10, loc="left")
        _box(top, win)
        draw(bot, win)
        if n == 1:
            bot.set_title(f"zoom: the yellow box ({Z} px)", fontsize=10, loc="left")
    if extra is not None:
        extra(fig)
    fig.suptitle(textwrap.fill(suptitle, int(10.5 * fig.get_size_inches()[0])), fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=DPI, pil_kwargs=dict(quality=88))
    plt.close(fig)


def _orient_key(host):
    """A key of the orientation colours, inset in the lower left corner of the axes host: a line at each of the 16
    tangent angles in its hue (x right, y down)."""
    import matplotlib.colors as mc
    ax = host.inset_axes([0.0, 0.0, 0.26, 0.26])
    ax.set_facecolor("k")
    for k in range(16):
        t = k * math.pi / 16
        c = mc.hsv_to_rgb((k / 16, 0.85, 1.0))
        ax.plot([-math.cos(t), math.cos(t)], [-math.sin(t), math.sin(t)], "-", color=c, lw=2.2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(1.15, -1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#ffffff")


def _orient_rgb(U, thr=6.0):
    """Orientation score as colour: hue = the best orientation, brightness = the score (CNR units)."""
    import matplotlib.colors as mc
    k = np.argmax(U, 0)
    m = np.max(U, 0)
    hsv = np.stack([k / U.shape[0], np.full(k.shape, 0.85), np.clip(m / thr, 0, 1)], -1)
    return mc.hsv_to_rgb(hsv)


def _lines(ax, polys, colour=None, lw=1.2, cmap="tab20"):
    plt = _plt()
    cm = plt.get_cmap(cmap)
    for i, P in enumerate(polys):
        P = np.asarray(P, float).reshape(-1, 2)
        ax.plot(P[:, 0], P[:, 1], "-", color=colour or cm(i % 20), lw=lw, solid_capstyle="round")


def _juncs(ax, js, ms=8, ring=False):
    for x, y, t in js:
        st = JSTYLE.get(COARSE.get(t, t), JSTYLE["3-way"])
        if ring:
            ax.plot(x, y, linestyle="none", marker=st["marker"], ms=ms * 1.4, mfc="none", mec=st["color"], mew=1.2)
        else:
            ax.plot(x, y, linestyle="none", marker=st["marker"], ms=ms, mfc=st["color"], mec="k", mew=0.5)


def _clip_axes(ax, win, H, W):
    if win is None:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)


def run(scene_dir: str, mask_scene_dir: str, out: str):
    from experiments.splinefit import fastset as FS
    from experiments.splinefit import pipeline as P
    from experiments.splinefit import continuity as CT
    from experiments.splinefit.proposals import build_network
    FS.check_not_heldout(scene_dir)
    FS.check_not_heldout(mask_scene_dir)
    os.makedirs(out, exist_ok=True)
    plt = _plt()
    cfg = N.DEFAULT
    case = S.load_truth(scene_dir, "average", reference_target=False)
    img, valid = case["image"], case["valid"]
    H, W = img.shape
    win = _zoom_window(case)
    name = os.path.basename(os.path.normpath(scene_dir)).split("_480")[0].replace("_s00", " ")
    gray = _norm(img, valid)
    zv = np.zeros_like(valid)
    zv[win[1]:win[1] + Z, win[0]:win[0] + Z] = True
    gray_z = _norm(img, valid & zv, 1.0, 99.0)
    G = lambda w: gray if w is None else gray_z                                  # noqa: E731

    # ------------------------------------------------------------ step 1
    s1 = N.photoreceptors(img, valid, cfg)
    L, B, OD, M = s1["L"], s1["B"], s1["OD"], s1["mask"]
    B0 = N._upper_envelope(L, cfg.env_radius)
    vod = float(np.percentile(OD[valid], 99.5))
    _pair_fig([
        ("the image I (averaged still)", lambda ax, w: _show(ax, G(w), win=w)),
        ("L = ln I", lambda ax, w: _show(ax, _norm(L, valid), win=w)),
        ("B0: grey closing of L, 60 px disc", lambda ax, w: _show(ax, _norm(B0, valid), win=w)),
        ("vessel mask M (CNR > 2 or OD > 3 RMS, dilated)", lambda ax, w: _show(ax, M, vmin=0, vmax=1, win=w)),
        ("B: refitted outside M only", lambda ax, w: _show(ax, _norm(B, valid), win=w)),
        ("OD = B - L (the target)", lambda ax, w: _show(ax, OD, "gray_r", 0, vod, win=w)),
    ], win, os.path.join(out, "step1_background.jpg"), ncols=3,
        suptitle=f"Step 1 (dev {name}): optical density and a background fitted only outside the vessel mask")

    # ------------------------------------------------------------ step 2
    Zc, rms = N._band_cnr(OD, s1["bg"], cfg)
    D2 = N._gblur(OD, 2.0) - N._gblur(OD, 4.0)
    D8 = N._gblur(OD, 8.0) - N._gblur(OD, 16.0)
    _pair_fig([
        ("band s = 2: G2*OD - G4*OD", lambda ax, w: _show(ax, D2, "gray_r", 0, float(np.percentile(D2, 99.5)), win=w)),
        ("its local RMS (background pixels)", lambda ax, w: _show(ax, rms[2.0], "magma", win=w)),
        ("band s = 8: G8*OD - G16*OD", lambda ax, w: _show(ax, D8, "gray_r", 0, float(np.percentile(D8, 99.5)), win=w)),
        ("Z = max over bands of D_s / RMS_s", lambda ax, w: _show(ax, np.clip(Zc, 0, 8), "viridis", 0, 8, win=w)),
    ], win, os.path.join(out, "step2_cnr.jpg"), ncols=2, suptitle="Step 2: contrast-to-noise per band (feeds step 1's mask and step 8)")

    # ------------------------------------------------------------ steps 3-5 per channel
    chans = N.simple_cells(OD, s1["bg"], cfg)
    st3, st4, st5 = [], [], []
    for ch in chans:
        f = ch["f"]
        U4 = N.surround(ch["U"], ch["sigma"], cfg, f)
        C5 = N.association_field(U4, ch["sigma"], cfg, f)
        st3.append(N._up(ch["U"], f, ch["shape"]))
        st4.append(N._up(U4, f, ch["shape"]))
        st5.append(N._up(C5, f, ch["shape"]))
    # the filter bank (even / odd kernels) at 4 of the 16 orientations, one scale per channel
    fig, axs = plt.subplots(2, 8, figsize=(16, 4.6), layout="constrained")
    for c, (si, el) in enumerate(((1, 3.0), (4, 2.5))):
        s = cfg.scales[si]
        n = int(3 * el * s + 2)
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(float)
        for kk, k in enumerate((0, 2, 4, 6)):
            t = N.orientations(cfg)[k]
            tn = -xx * math.sin(t) + yy * math.cos(t)
            tt = xx * math.cos(t) + yy * math.sin(t)
            Gk = np.exp(-0.5 * (tn ** 2 / s ** 2 + tt ** 2 / (el * s) ** 2))
            even = (1 - tn ** 2 / s ** 2) * Gk            # -s^2 d2/dn2 G, up to a constant
            odd = -(tn / s) * Gk                          # s d/dn G
            for r, K in enumerate((even, odd)):
                ax = axs[r, 4 * c + kk]
                v = float(np.abs(K).max())
                ax.imshow(K, cmap="RdBu_r", vmin=-v, vmax=v)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{'even (line)' if r == 0 else 'odd (edge)'}  sigma {s:g}, theta {math.degrees(t):.0f} deg",
                             fontsize=8)
    fig.suptitle("Step 3: the oriented filter pair (16 orientations, 9 scales in 3 channels); two scales, 4 orientations "
                 "shown", fontsize=11, x=0.01, ha="left")
    fig.savefig(os.path.join(out, "step3_filters.jpg"), dpi=DPI, pil_kwargs=dict(quality=88))
    plt.close(fig)
    names = ("fine (1.5-3 px)", "medium (4.2-6 px)", "coarse (8.5-24 px)")
    _pair_fig([(f"{names[c]}: max over theta, hue = best theta",
                (lambda c: lambda ax, w: _show(ax, _orient_rgb(st3[c]), win=w))(c)) for c in range(3)],
              win, os.path.join(out, "step3_orientation.jpg"),
              "Step 3: orientation score per channel, max over the 16 orientations (brightness = line CNR, full at 6; "
              "colour = the best orientation; key lower left)", extra=lambda fig: _orient_key(fig.axes[3]))
    c_show = 0
    tl, th = cfg.t_low, cfg.t_high
    above = lambda X: np.where(np.max(X, 0) > th, 1.0, np.where(np.max(X, 0) > tl, 0.45, 0.0))   # noqa: E731
    _pair_fig([
        ("step 3: simple cells (fine channel)", lambda ax, w: _show(ax, _orient_rgb(st3[c_show], th), win=w)),
        ("step 4: minus the isotropic part and the weaker flank", lambda ax, w: _show(ax, _orient_rgb(st4[c_show], th),
                                                                                       win=w)),
        ("step 5: association field (3 bipole steps)", lambda ax, w: _show(ax, _orient_rgb(st5[c_show], th), win=w)),
        (f"step 3 thresholded: white > {th:g} (seeds), grey > {tl:g} (tracing continues)",
         lambda ax, w: _show(ax, above(st3[c_show]), vmin=0, vmax=1, win=w, interp="nearest")),
        ("step 4 thresholded: texture specks gone", lambda ax, w: _show(ax, above(st4[c_show]), vmin=0, vmax=1, win=w,
                                                                       interp="nearest")),
        ("step 5 thresholded: collinear gaps bridged", lambda ax, w: _show(ax, above(st5[c_show]), vmin=0, vmax=1,
                                                                          win=w, interp="nearest")),
    ], win, os.path.join(out, "step45_context.jpg"), ncols=3,
        suptitle=f"Steps 4 and 5 on the fine channel (brightness full at the seed threshold {th:g}; bottom: what the "
                 "readout of step 6 sees)", extra=lambda fig: _orient_key(fig.axes[3]))

    # ------------------------------------------------------------ step 6
    traces_ch = []
    for ci, (ch, C) in enumerate(zip(chans, st5)):
        Sx = N._up(ch["S"], ch["f"], ch["shape"], nearest=True)
        traces_ch.append(N.readout(C, Sx, cfg, chan=ci, valid=s1["ok"]))
    merged = N.merge_channels([t for tr in traces_ch for t in tr], cfg, shape=OD.shape)
    ccol = ("#ff4d4d", "#4da6ff", "#7CFC00")

    def draw_traces(ax, w, per_channel=True):
        _show(ax, G(w), win=w)
        if per_channel:
            for ci, tr in enumerate(traces_ch):
                _lines(ax, [t["xy"] for t in tr], colour=ccol[ci], lw=1.0)
        else:
            _lines(ax, [t["xy"] for t in merged], lw=1.2)
        _clip_axes(ax, w, H, W)
    _pair_fig([
        ("traces per channel: fine red, medium blue, coarse green", lambda ax, w: draw_traces(ax, w, True)),
        ("merged (duplicates and wall echoes dropped)", lambda ax, w: draw_traces(ax, w, False)),
    ], win, os.path.join(out, "step6_traces.jpg"),
        "Step 6: non-maximum suppression and hysteresis tracing in (x, y, theta), per channel, then merged")
    # a cross-section and its full width at half maximum
    t = max(merged, key=lambda t: float(np.median(t["w"])) if len(t["xy"]) > 40 else 0)
    i = len(t["xy"]) // 2
    Tg = N._tangents(t["xy"], 2)[i]
    nrm = np.array([-Tg[1], Tg[0]])
    U = min(max(6.0, 1.5 * float(np.median(t["w"]))), 60.0)
    us = np.arange(-U, U + 0.01, 0.5)
    prof = N._bilinear(OD, t["xy"][i, 0] + us * nrm[0], t["xy"][i, 1] + us * nrm[1])
    k = max(2, len(us) // 5)
    base = 0.5 * (np.median(prof[:k]) + np.median(prof[-k:]))
    pk = prof[len(us) // 2 - 2:len(us) // 2 + 3].max() - base
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    _show(axs[0], gray, title="where: the widest long trace, at its middle")
    axs[0].plot(t["xy"][:, 0], t["xy"][:, 1], "-", color="#ff4d4d", lw=1)
    axs[0].plot(t["xy"][i, 0] + us * nrm[0], t["xy"][i, 1] + us * nrm[1], "-", color="#ffd400", lw=2)
    axs[1].plot(us, prof, "-", color="#1f77b4", lw=2, label="OD along the normal")
    axs[1].axhline(base, color="#888888", lw=1, ls=":", label="baseline: median of the outer fifths")
    axs[1].axhline(base + pk / 2, color="#d62728", lw=1, ls="--", label="half maximum")
    c0 = len(us) // 2
    below = prof < base + pk / 2
    rgt = int(np.argmax(below[c0:])) if below[c0:].any() else len(us) - 1 - c0
    lft = int(np.argmax(below[c0::-1])) if below[c0::-1].any() else c0
    for u0 in (us[c0 - lft], us[c0 + rgt]):
        axs[1].axvline(u0, color="#d62728", lw=1, ls=":")
    axs[1].annotate("", xy=(us[c0 + rgt], base + pk / 2), xytext=(us[c0 - lft], base + pk / 2),
                    arrowprops=dict(arrowstyle="<->", color="#d62728"))
    axs[1].text(0.5 * (us[c0 - lft] + us[c0 + rgt]), base + 0.55 * pk, f"FWHM {0.5 * (lft + rgt):.1f} px",
                ha="center", color="#d62728", fontsize=9)
    axs[1].set_xlabel("px across the vessel")
    axs[1].set_ylabel("OD")
    axs[1].legend(fontsize=8, frameon=False)
    axs[1].set_title("width = full width at half maximum of the OD cross-section (not a filter response)",
                     fontsize=9, loc="left")
    fig.savefig(os.path.join(out, "step6_width.jpg"), dpi=DPI, pil_kwargs=dict(quality=88))
    plt.close(fig)

    # ------------------------------------------------------------ steps 7 and 8
    edges7, junc7, traces7 = N._assemble_profiled([dict(t) for t in merged], cfg, OD)
    jm = N._junction_mask(OD.shape, junc7, cfg)
    hp = OD - N._gblur(OD, 8.0)
    sig2 = np.maximum(N._local_rms(hp, s1["bg"], 32.0), 1e-4) ** 2
    sig2 = np.where(s1["ok"], sig2, 1e12)
    s1["rms"] = rms
    keep, rd = N._mdl_prune(OD, np.where(jm, 1e12, sig2), edges7, cfg, rms)
    R8 = OD - rd.residual()
    j7 = [(j["x"], j["y"], j["type"]) for j in junc7]

    def draw_edges(ax, w, keepflags=None):
        _show(ax, G(w), win=w)
        if keepflags is None:
            _lines(ax, [e["xy"] for e in edges7], lw=1.3)
            _juncs(ax, j7, ms=7 if w is None else 10)
            from matplotlib.patches import Circle
            for j in junc7:
                ax.add_patch(Circle((j["x"], j["y"]), j["r"], fill=False, ec="#ffffff", lw=0.6, alpha=0.7))
        else:
            _lines(ax, [e["xy"] for e, k in zip(edges7, keepflags) if k], colour="#4da6ff", lw=1.3)
            _lines(ax, [e["xy"] for e, k in zip(edges7, keepflags) if not k], colour="#ff2b2b", lw=1.8)
        _clip_axes(ax, w, H, W)
    _pair_fig([
        ("edges cut at junctions; markers by type; discs = junction regions", lambda ax, w: draw_edges(ax, w)),
    ], win, os.path.join(out, "step7_junctions.jpg"),
        "Step 7: T events (an end on a line) and X events (lines through each other), clustered and typed by their arms "
        "(cyan 3-way, magenta crossing, yellow compound)")
    _pair_fig([
        ("additive render of the profiled edges", lambda ax, w: _show(ax, R8, "gray_r", 0, vod, win=w)),
        ("residual OD - render", lambda ax, w: _show(ax, OD - R8, "RdBu_r", -vod / 2, vod / 2, win=w)),
        ("MDL prune: kept blue, removed red", lambda ax, w: draw_edges(ax, w, keep)),
    ], win, os.path.join(out, "step8_prune.jpg"),
        "Step 8 (first round): render, compare with the OD target, remove free-end edges that do not pay for themselves")

    # the pipeline's own stages 1-6 must give the same merged traces
    pr = N.propose(img, valid, cfg)
    assert len(pr["traces"]) == len(merged), (len(pr["traces"]), len(merged))

    # ------------------------------------------------------------ steps 9-12: the spline fit
    dbg = {}
    outp = P.annotate_cfg(img.copy(), valid.copy(), P.DEFAULT, debug=dbg)
    net0, _ = build_network(dbg["proposal"], OD.shape)
    netf = dbg["net"]

    def draw_net(ax, w, net, ctrl=True):
        _show(ax, G(w), win=w)
        cm = plt.get_cmap("tab20")
        for kk, eid in enumerate(net.edges):
            smp = net.sample(eid, 1.0)
            ax.plot(smp["xy"][:, 0], smp["xy"][:, 1], "-", color=cm(kk % 20), lw=1.3)
            if ctrl and w is not None:
                c = net.edges[eid].ctrl
                ax.plot(c[:, 0], c[:, 1], "o", color=cm(kk % 20), ms=3, mec="k", mew=0.3)
        deg = net.degrees()
        for nid, nd in net.nodes.items():
            if deg.get(nid, 0) >= 3:
                ax.plot(nd.x, nd.y, "o", ms=6 if w is None else 9, mfc="none", mec="#ffffff", mew=1.2)
        _clip_axes(ax, w, H, W)
    _pair_fig([
        ("initial spline network (dots: control points in the zoom; rings: nodes)", lambda ax, w: draw_net(ax, w, net0)),
        ("after the joint fit and the MDL prune", lambda ax, w: draw_net(ax, w, netf)),
    ], win, os.path.join(out, "step9_network.jpg"),
        "Step 9 and 12: the proposal as clamped cubic B-splines (one per chain through crossings and through-nodes), before "
        "and after the fit")
    # the render model, 1D: a vessel's chord profile blurred; union at a node and kappa at a crossing (both taken on
    # the sharp lumens, then blurred, as render.JunctionModel does); the image-wide halo
    du = 0.05
    u = np.arange(-40, 40 + du / 2, du)              # wider than the widest kernel (5 s_h)

    def sharp(c, r, a=1.0):                   # centre OD a, chord profile of a cylinder of radius r centred at c
        return a * np.sqrt(np.maximum(1 - ((u - c) / r) ** 2, 0))

    def blur(X, s):
        k = np.arange(-int(5 * s / du), int(5 * s / du) + 1) * du
        g = np.exp(-0.5 * (k / s) ** 2)
        return np.convolve(X, g / g.sum(), mode="same")

    fig, axs = plt.subplots(1, 4, figsize=(20, 4.0), layout="constrained")
    axs[0].plot(u, sharp(0, 4.0), "k:", lw=1, label="no blur: a sqrt(1 - (d / r)^2)")
    for sg in (0.5, 1.5, 3.0):
        axs[0].plot(u, blur(sharp(0, 4.0), sg), lw=2, label=f"blur s = {sg:g} px")
    axs[0].set_title("one vessel (r = 4 px, a = 1): the chord of a cylinder,\nblurred by a Gaussian (closed form, erf)",
                     fontsize=9, loc="left")
    c1, c2 = sharp(0, 4.0), sharp(3.0, 3.0, 0.8)
    axs[1].plot(u, blur(c1, 1.5), lw=1.2, label="vessel 1")
    axs[1].plot(u, blur(c2, 1.5), lw=1.2, label="vessel 2 (3 px off, r 3)")
    axs[1].plot(u, blur(c1 + c2, 1.5), "k-", lw=2, label="sum (additive render)")
    axs[1].plot(u, blur(np.maximum(c1, c2), 1.5), "r--", lw=2, label="union: blur(max of the sharp lumens)")
    axs[1].set_title("at a node (fork) the lumens are one blood volume:\nunion, not sum", fontsize=9, loc="left")
    kap = 0.87
    axs[2].plot(u, blur(c1 + c2, 1.5), "k-", lw=2, label="sum")
    axs[2].plot(u, blur(c1 + c2 - (1 - kap) * np.minimum(c1, c2), 1.5), "m--", lw=2,
                label="crossing: blur(sum - (1 - kappa) min)")
    axs[2].set_title("at a crossing the vessels lie at different depths:\nnearly additive (kappa fitted, starts at 0.87)",
                     fontsize=9, loc="left")
    hw, hs = 0.25, 6.0
    b1 = blur(c1, 1.5)
    axs[3].plot(u, b1, "k:", lw=1, label="no halo")
    axs[3].plot(u, (1 - hw) * b1 + hw * blur(b1, hs), "c-", lw=2, label="halo: (1 - h) X + h G(s_h) * X")
    axs[3].set_title(f"the whole render passes an image-wide halo\n(h, s_h fitted; start {hw:g}, {hs:g} px)",
                     fontsize=9, loc="left")
    for ax in axs:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_xlabel("px across")
        ax.set_xlim(-12, 14)
        ax.set_ylim(-0.05, 2.1 if ax is not axs[0] else 1.25)
    axs[0].set_ylabel("OD")
    fig.suptitle("Step 10: the render model, one cross-section (every edge: centreline spline + r, s, a profiles)",
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(os.path.join(out, "step10_render_model.jpg"), dpi=DPI, pil_kwargs=dict(quality=88))
    plt.close(fig)
    Rf = np.asarray(outp["od_render"], np.float32)
    Tf = np.asarray(outp["od_target"], np.float32)
    OD1 = np.asarray(dbg["proposal"]["s1"]["OD"], np.float32)      # (debug's od_stage1 is the retargeted OD)
    wgt = np.asarray(dbg["weight"], np.float32)
    lw_ = np.log10(np.maximum(wgt, 1e-12))
    lw_lo, lw_hi = (float(v) for v in np.percentile(lw_[wgt > 0], [1, 99]))
    _pair_fig([
        ("stage-1 target OD (what the fit starts on)", lambda ax, w: _show(ax, OD1, "gray_r", 0, vod, win=w)),
        ("precision weight 1 / sigma^2 (log; dark = textured, trusted less)",
         lambda ax, w: _show(ax, lw_, "magma", lw_lo, lw_hi, win=w)),
        ("fitted render", lambda ax, w: _show(ax, Rf, "gray_r", 0, vod, win=w)),
        ("target after the retarget (B refitted outside mask OR render support)",
         lambda ax, w: _show(ax, Tf, "gray_r", 0, vod, win=w)),
        ("what the retarget changed: new target - stage-1 target",
         lambda ax, w: _show(ax, Tf - OD1, "RdBu_r", -vod / 4, vod / 4, win=w)),
        ("residual: target - render (same scale as step 8)",
         lambda ax, w: _show(ax, Tf - Rf, "RdBu_r", -vod / 2, vod / 2, win=w)),
    ], win, os.path.join(out, "step12_fit.jpg"), ncols=3,
        suptitle="Steps 11-12: the whole-image render fitted to the target (30 iterations profiles, 100 everything, "
                 "MDL prune, retarget, 60 profiles)")

    # ------------------------------------------------------------ step 13: the output graph against the truth
    polys, juncs = S.network_graph(outp["network"])
    tl = [np.asarray(e["xy"], float).reshape(-1, 2) for e in case["obs"]["graph"]["edges"]]
    tj = [(float(j["x"]), float(j["y"]), COARSE.get(j["type_observable"], "3-way")) for j in case["obs"]["junctions"]]

    def draw_out(ax, w, truth):
        _show(ax, G(w), win=w)
        if truth:
            _lines(ax, tl, colour=TRUTH, lw=1.4)
            _juncs(ax, tj, ms=6 if w is None else 9)
        else:
            _lines(ax, [e["xy"] for e in outp["network"]["edges"]], lw=1.3)
            _juncs(ax, juncs, ms=6 if w is None else 9)
            _juncs(ax, tj, ms=6 if w is None else 9, ring=True)
        _clip_axes(ax, w, H, W)
    sc = S.score(case, outp)
    _pair_fig([
        ("the truth (observable graph)", lambda ax, w: draw_out(ax, w, True)),
        (f"the output graph (rings: truth junctions)  line F1 {sc['centreline_f1']:.2f}, junction F1 "
         f"{sc['junction_f1_strict']:.2f}, type {sc['junction_type_balanced_coarse']:.2f}",
         lambda ax, w: draw_out(ax, w, False)),
    ], win, os.path.join(out, "step13_output.jpg"), "Step 13: the exported graph (edges cut at nodes, typed junctions)")

    # ------------------------------------------------------------ the mask improvement (pathologic dev)
    cm2 = S.load_truth(mask_scene_dir, "average", reference_target=False)
    s1m = N.photoreceptors(cm2["image"], cm2["valid"], cfg)
    with np.load(os.path.join(mask_scene_dir, "labels.npz")) as z:
        lum = z["observable_lumen_average"].astype(bool)
    Mv = np.asarray(s1m["mask"], bool) & cm2["valid"]
    Mf = CT.od_mask(s1m["OD"], cm2["valid"], s1m["bg"])
    Mc = CT.od_mask(s1m["OD"], cm2["valid"], s1m["bg"], smooth=3.0)
    Pq = cm2["prof"]                                   # wide lumen: nearest truth centreline sample has r >= 8 px
    kq = Pq["in_frame"] & (Pq["merged_into"] < 0)
    Yg, Xg = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
    _, iq = cKDTree(np.stack([Pq["x_px"], Pq["y_px"]], 1)[kq]).query(np.stack([Xg[lum], Yg[lum]], 1))
    wide = np.zeros_like(lum)
    wide[lum] = Pq["radius_px"][kq][iq] >= 8.0

    def overlay(M):
        rgb = np.zeros(M.shape + (3,), np.float32)
        rgb[M & lum] = (0.30, 0.85, 0.35)        # lumen covered
        rgb[lum & ~M] = (0.95, 0.20, 0.20)       # lumen missed: the background is fitted on it
        rgb[M & ~lum] = (0.45, 0.45, 0.55)       # mask outside the lumen (blur flanks)
        return rgb
    rec = lambda M, R: float((M & R).sum() / max(R.sum(), 1))                    # noqa: E731
    cover = {k: (rec(M, lum), rec(M, wide)) for k, M in (("vesselness", Mv), ("od_fine", Mf), ("od_coarse", Mc))}
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.2), layout="constrained")
    _show(axs[0], _norm(cm2["image"], cm2["valid"]), title="pathologic dev scene (averaged still)")
    for ax, (k, M, t) in zip(axs[1:], (("vesselness", Mv, "step 1's vesselness mask (CNR > 2 or OD > 3 RMS)"),
                                        ("od_fine", Mf, "hysteresis on the OD, 1 px (3 / 1.5 x noise)"),
                                        ("od_coarse", Mc, "the same on the OD smoothed 3 px"))):
        _show(ax, overlay(M), title=f"{t}\nlumen covered {cover[k][0]:.0%}, wide-vessel lumen {cover[k][1]:.0%}")
    fig.suptitle("Improvement 1: the background is fitted outside the mask. Red = true lumen the mask misses (the "
                 "background absorbs it), green = covered, grey = mask on the blur flanks", fontsize=11, x=0.01, ha="left")
    fig.savefig(os.path.join(out, "improve_mask.jpg"), dpi=DPI, pil_kwargs=dict(quality=88))
    plt.close(fig)
    return dict(win=win, score={k: sc[k] for k in ("composite", "graph_score", "centreline_f1", "junction_f1_strict",
                                                   "junction_type_balanced_coarse", "edge_cover", "explained",
                                                   "explained_true")},
                n_edges7=len(edges7), n_removed=int(len(keep) - sum(keep)), n_junc7=len(junc7),
                n_traces=[len(t) for t in traces_ch], n_merged=len(merged), mask_cover=cover,
                log=dbg.get("log"))


def main(argv=None):
    from experiments.splinefit import fastset as FS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="healthy_s007_480x768")
    ap.add_argument("--mask-scene", default="pathologic_s000_480x768")
    a = ap.parse_args(argv)
    info = run(os.path.join(FS.DEV_DIR, a.scene), os.path.join(FS.DEV_DIR, a.mask_scene), a.out)
    for k, v in info.items():
        print(k, v)


if __name__ == "__main__":
    main()
