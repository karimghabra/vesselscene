r"""Annotated-scene figures of the held-out report: the observable truth, the neuromimetic proposals, the fitted
spline network and LIMBUS vesselmap drawn over the image, the renders and residuals, and junction close-ups.

    python -m experiments.splinefit.report_figures run  SCENE_DIR KIND [SCENE_DIR KIND ...] [--cache DIR]
    python -m experiments.splinefit.report_figures plot SCENE_DIR KIND [SCENE_DIR KIND ...] [--cache DIR] --out DIR
    python -m experiments.splinefit.report_figures viewer SCENE_DIR KIND [...] [--cache DIR] --out FILE.html

``run`` annotates each image three ways and keeps the outputs in --cache (one pickle per image and row):
``fit`` = pipeline.annotate, ``proposals`` = evaluate_fit.proposals_annotate (rendered by the scorer, as in
its row), ``vesselmap`` = the cached LIMBUS network (evaluate_fit.vesselmap_output).  Each output is scored
by score.score and its digest stored, so a figure can be checked against the held-out JSONs (the same digest
means the same network and render).  ``plot`` writes per image:

    <scene>_<kind>_annotated.jpg   2 x 2: truth / proposals / fit / vesselmap over the image.  The truth's
                                   observable centrelines are green; an annotator's edges are coloured one by
                                   one (the fit's with its fitted lumen, centreline +- r, shaded); junctions
                                   by coarse type: cyan circle 3-way, magenta X crossing, yellow square
                                   compound (the truth's as rings of the same colours);
    <scene>_<kind>_render.jpg      the image, the pipeline's own target (background fitted on the vesselness
                                   mask's negative), the fitted render; the residual OD_obs - render of the
                                   fit, the proposals and vesselmap (OD_obs: the scorer's oracle-background
                                   target, score.py);
    <scene>_<kind>_junctions.jpg   close-ups of truth junctions: per coarse type the junction at the median
                                   and the one at the maximum of the fit's summed |residual| in its disc (not
                                   hand-picked), each as image + truth, proposals, fit, vesselmap, target,
                                   fit render, residual;
    <scene>_<kind>_thumb.jpg       the fit over the image with the truth underneath (the gallery);
    <scene>_<kind>_vs_truth.jpg    vesselscene's own render of the scene (OD_img: the clean vessel OD through
                                   the image formation's blur, no camera noise, no background) next to the
                                   fitted render on the same OD scale, the image, and OD_img - render;
    <scene>_<kind>_vs_truth_junctions.jpg   the junction windows above, as OD_img | render | difference.

``viewer`` writes one self-contained HTML page (--out FILE) that compares those layers by a wipe, at 1x to 4x.

Held-out scenes need SPLINEFIT_ALLOW_HELDOUT=1 (fastset.check_not_heldout).  Reads the truth: a report tool,
never a pipeline path.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import json                                         # noqa: E402
import pickle                                       # noqa: E402
import time                                         # noqa: E402

import numpy as np                                  # noqa: E402

from . import set_threads                           # noqa: E402
from . import score as S                            # noqa: E402

COARSE = {"bifurcation": "3-way", "confluence": "3-way", "pseudo-T": "3-way", "branch": "3-way",
          "crossing": "crossing", "compound": "compound"}
JSTYLE = {"3-way": dict(marker="o", color="#00e5ff"), "crossing": dict(marker="X", color="#ff2bd6"),
          "compound": dict(marker="s", color="#ffd400")}
TRUTH = "#39d353"
ROWS = ("fit", "proposals", "vesselmap")
LABEL = {"truth": "observable truth", "fit": "fitted spline network (this work)",
         "proposals": "neuromimetic proposals", "vesselmap": "LIMBUS vesselmap build_map"}
HALF = 36                     # px, half size of a junction close-up


def _name(folder, kind):
    return f"{os.path.basename(os.path.normpath(folder))}_{kind}"


def _strip(out: dict) -> dict:
    keep = ("polylines", "junctions", "network", "od_render", "od_target", "halo", "kappa", "build_seconds")
    return {k: out[k] for k in keep if k in out}


def run(folder, kind, cache):
    from experiments.splinefit import evaluate_fit as EF
    from experiments.splinefit import fastset as FS
    from experiments.splinefit import pipeline as P
    FS.check_not_heldout(folder)
    set_threads()
    case = S.load_truth(folder, kind)
    img, val = case["image"], case["valid"]
    for row in ("proposals", "vesselmap", "fit"):          # the quick ones first
        path = os.path.join(cache, f"{_name(folder, kind)}_{row}.pkl")
        if os.path.exists(path):
            continue
        t0 = time.time()
        if row == "fit":
            out = P.annotate(img.copy(), val.copy())
        elif row == "proposals":
            out = EF.proposals_annotate(img.copy(), val.copy())
            out["od_render"] = S.render_network(S.to_vesselmap(out["network"], img.shape), img.shape)
        else:
            out = EF.vesselmap_output(img)
            if out is None:
                print(f"{_name(folder, kind)}: no cached vesselmap network, skipped")
                continue
        secs = time.time() - t0
        s = S.score(case, {k: v for k, v in out.items() if not (row == "proposals" and k == "od_render")})
        out = _strip(out)
        out.update(score={k: v for k, v in s.items() if isinstance(v, (int, float, str)) or v is None},
                   digest=S.digest(out if row != "proposals" else {k: v for k, v in out.items()
                                                                     if k != "od_render"}),
                   seconds=secs)
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print(f"{_name(folder, kind)} {row}: composite {s['composite']:.3f} explained {s['explained']:.3f} "
              f"digest {out['digest']} ({secs:.0f} s)", flush=True)


# ======================================================================== drawing
def _load(folder, kind, cache):
    res = {}
    for row in ROWS:
        path = os.path.join(cache, f"{_name(folder, kind)}_{row}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                res[row] = pickle.load(fh)
    return res


def _gray(img, valid):
    v = img[valid] if valid.any() else img.ravel()
    lo, hi = np.percentile(v, [0.5, 99.5])
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.where(valid, g, 0.0)


def _truth_lines(case):
    return [np.asarray(e["xy"], float).reshape(-1, 2) for e in case["obs"]["graph"]["edges"]]


def _truth_junctions(case):
    return [(float(j["x"]), float(j["y"]), COARSE.get(j["type_observable"], "3-way")) for j in case["obs"]["junctions"]]


def _out_junctions(out):
    _, js = S.network_graph(out["network"])
    return [(x, y, COARSE.get(t, "3-way")) for x, y, t in js]


def _draw_junctions(ax, js, ring=False, ms=7.0, lw=1.1):
    for typ, st in JSTYLE.items():
        pts = np.array([(x, y) for x, y, t in js if t == typ], float).reshape(-1, 2)
        if not len(pts):
            continue
        if ring:
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker=st["marker"], ms=ms * 1.5, mfc="none",
                    mec=st["color"], mew=lw)
        else:
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker=st["marker"], ms=ms, mfc=st["color"],
                    mec="k", mew=0.5)


def _edge_colours(n):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    order = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 9, 11, 13, 17, 19]      # skip greys (14, 15)
    return [cm(order[i % len(order)]) for i in range(n)]


def _draw_network(ax, out, widths=False, lw=1.2):
    from matplotlib.patches import Polygon
    edges = out["network"]["edges"]
    cols = _edge_colours(len(edges))
    for e, c in zip(edges, cols):
        xy = np.asarray(e["xy"], float).reshape(-1, 2)
        if len(xy) < 2:
            continue
        if widths:
            r = np.asarray(e["r"], float).reshape(-1)
            if len(r) == len(xy):
                t = np.gradient(xy, axis=0)
                t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
                nrm = np.stack([-t[:, 1], t[:, 0]], 1)
                poly = np.concatenate([xy + r[:, None] * nrm, (xy - r[:, None] * nrm)[::-1]])
                ax.add_patch(Polygon(poly, closed=True, fc=c, ec="none", alpha=0.22))
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=lw, solid_capstyle="round")


def _panel(ax, case, gray, row=None, out=None, truth_under=True, title=None, xlim=None, ylim=None, ms=7.0):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1.25, interpolation="lanczos" if xlim else "antialiased")
    if row is None:
        for xy in _truth_lines(case):
            ax.plot(xy[:, 0], xy[:, 1], "-", color=TRUTH, lw=1.3)
        _draw_junctions(ax, _truth_junctions(case), ms=ms)
    else:
        if truth_under:
            for xy in _truth_lines(case):
                ax.plot(xy[:, 0], xy[:, 1], "-", color=TRUTH, lw=3.2 if xlim else 2.4, alpha=0.35)
            _draw_junctions(ax, _truth_junctions(case), ring=True, ms=ms * 0.9, lw=0.9)
        if out is not None:
            _draw_network(ax, out, widths=(row == "fit"), lw=1.6 if xlim else 1.0)
            _draw_junctions(ax, _out_junctions(out), ms=ms)
    H, W = gray.shape
    ax.set_xlim(*(xlim or (-0.5, W - 0.5)))
    ax.set_ylim(*(ylim or (H - 0.5, -0.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, loc="left")


def _metrics(sc):
    if not sc:
        return ""
    return (f"line F1 {sc['centreline_f1']:.2f} · junc F1 strict {sc['junction_f1_strict']:.2f} · "
            f"crossing R {sc['crossing_recall']:.2f} · explained {sc['explained']:.2f} · "
            f"composite {sc['composite']:.3f}")


def _legend(fig, loc="outside lower center"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=TRUTH, lw=2, label="truth centreline"),
         Line2D([], [], color="#1f77b4", lw=1.5, label="annotated edge (one colour each)")]
    h += [Line2D([], [], linestyle="none", marker=st["marker"], mfc=st["color"], mec="k", ms=7,
                 label=f"{typ} junction") for typ, st in JSTYLE.items()]
    h += [Line2D([], [], linestyle="none", marker="o", mfc="none", mec="#aaaaaa", ms=9,
                 label="truth junction (ring, under an annotator)")]
    fig.legend(handles=h, loc=loc, ncol=6, fontsize=8, frameon=False)


def plot_annotated(case, res, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gray = _gray(case["image"], case["valid"])
    H, W = gray.shape
    fig, axs = plt.subplots(2, 2, figsize=(15.4, 15.4 * H / W + 1.6), layout="constrained")
    nT = len(case["obs"]["junctions"])
    _panel(axs[0, 0], case, gray, title=f"{LABEL['truth']}: {len(_truth_lines(case))} edges, {nT} junctions")
    for ax, row in ((axs[0, 1], "proposals"), (axs[1, 0], "fit"), (axs[1, 1], "vesselmap")):
        out = res.get(row)
        _panel(ax, case, gray, row=row, out=out,
               title=f"{LABEL[row]}\n{_metrics(out['score']) if out else 'not available'}")
    fig.suptitle(f"{name}: annotated (truth in green; the fit's shading is its fitted lumen, centreline ± r)",
                 fontsize=11, x=0.01, ha="left")
    _legend(fig, loc="outside lower center")
    fig.savefig(path, dpi=100, pil_kwargs=dict(quality=88))
    plt.close(fig)


def plot_thumb(case, res, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gray = _gray(case["image"], case["valid"])
    H, W = gray.shape
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 9.6 * H / W + 0.5))
    out = res.get("fit")
    _panel(ax, case, gray, row="fit", out=out, ms=6,
           title=f"{name} · fit · {_metrics(out['score']) if out else ''}")
    fig.tight_layout()
    fig.savefig(path, dpi=80, pil_kwargs=dict(quality=82))
    plt.close(fig)


def plot_render(case, res, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = case["od_obs"]
    band = case["band"]
    v = float(np.nanpercentile(np.abs(t[band]), 99)) if band.any() else 1.0
    fit = res.get("fit")
    H, W = t.shape
    fig, axs = plt.subplots(2, 3, figsize=(17, 2 * 17 / 3 * H / W + 1.4), layout="constrained")
    gray = _gray(case["image"], case["valid"])
    panels = [(gray, "gray", 0, 1.25, "the image (single frame or averaged still)")]
    if fit is not None:
        panels += [(fit["od_target"], "gray_r", 0, v, "the fit's target OD: background fitted on the vesselness "
                                                      "mask's negative"),
                   (fit["od_render"], "gray_r", 0, v, "the fitted render (junction-aware spline network)")]
    for row in ROWS:
        out = res.get(row)
        if out is None:
            continue
        R = np.asarray(out["od_render"], np.float32)
        sc = out["score"]
        panels.append((np.where(np.isfinite(t), t - R, 0.0), "RdBu_r", -v / 2, v / 2,
                       f"residual OD_obs - render, {row}: explained {sc['explained']:.3f}, "
                       f"at junctions {sc['explained_junction']:.3f}"))
    for ax, (img, cm, lo, hi, ttl) in zip(axs.ravel(), panels):
        ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="antialiased")
        ax.set_title(ttl, fontsize=8.5, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{name}: target, render and residual (red = render under-predicts, blue = over-predicts; "
                 f"residual scale ±{v / 2:.2f} OD)", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=90, pil_kwargs=dict(quality=88))
    plt.close(fig)


def pick_junctions(case, res):
    """Per coarse truth type, the junction at the median and at the maximum of the fit's summed |residual| in
    its disc (radius >= 4 px), away from the image border: a rule, not a choice."""
    fit = res.get("fit")
    if fit is None:
        return []
    t = case["od_obs"]
    R = np.asarray(fit["od_render"], np.float32)
    r_ = np.where(np.isfinite(t) & np.isfinite(R), t - R, 0.0)
    H, W = t.shape
    rows = []
    for j, (x, y, rad) in zip(case["obs"]["junctions"], case["discs"]):
        if not (HALF <= x < W - HALF and HALF <= y < H - HALF):
            continue
        m = S.disc_mask(t.shape, np.array([[x, y, max(rad, 4.0)]])) & case["usable"]
        rows.append((COARSE.get(j["type_observable"], "3-way"), float(np.abs(r_[m]).sum()), float(x), float(y),
                     j["type_observable"]))
    pick = []
    for typ in JSTYLE:
        rr = sorted([r for r in rows if r[0] == typ], key=lambda r: r[1])
        if not rr:
            continue
        sel = [("median", rr[len(rr) // 2]), ("worst", rr[-1])] if len(rr) > 1 else [("only", rr[0])]
        pick += [(lab, r) for lab, r in sel]
    return pick


def plot_junctions(case, res, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = pick_junctions(case, res)
    if not pick:
        return
    gray = _gray(case["image"], case["valid"])
    t = case["od_obs"]
    v = float(np.nanpercentile(np.abs(t[case["band"]]), 99)) if case["band"].any() else 1.0
    fit = res["fit"]
    R = np.asarray(fit["od_render"], np.float32)
    cols = ["truth", "proposals", "fit", "vesselmap", "target", "render", "residual"]
    fig, axs = plt.subplots(len(pick), len(cols), figsize=(2.15 * len(cols), 2.25 * len(pick) + 0.6))
    axs = np.atleast_2d(axs)
    for i, (lab, (typ, absr, x, y, fine)) in enumerate(pick):
        xl, yl = (x - HALF, x + HALF), (y + HALF, y - HALF)
        for c, col in enumerate(cols):
            ax = axs[i, c]
            if col == "truth":
                _panel(ax, case, gray, xlim=xl, ylim=yl, ms=9)
            elif col in ROWS:
                _panel(ax, case, gray, row=col, out=res.get(col), xlim=xl, ylim=yl, ms=9)
            else:
                img, cm, lo, hi = {"target": (fit["od_target"], "gray_r", 0, v), "render": (R, "gray_r", 0, v),
                                   "residual": (np.where(np.isfinite(t), t - R, 0.0), "RdBu_r", -v / 2, v / 2)}[col]
                ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_xlim(*xl)
                ax.set_ylim(*yl)
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 0:
                ax.set_title({"truth": "truth", "proposals": "proposals", "fit": "fit (± r shaded)",
                              "vesselmap": "vesselmap", "target": "fit's target OD", "render": "fit render",
                              "residual": "residual (OD_obs - R)"}[col], fontsize=9)
        axs[i, 0].set_ylabel(f"{typ} ({fine})\n{lab} |res| · ({x:.0f}, {y:.0f})", fontsize=8)
    fig.suptitle(f"{name}: truth junctions close up ({2 * HALF} px windows); per coarse type the median and the "
                 f"worst by the fit's residual", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.4, w_pad=0.3)
    fig.savefig(path, dpi=95, pil_kwargs=dict(quality=88))
    plt.close(fig)


def _truth_layers(case, res):
    """The clean OD OD_img (vesselscene's own render of the scene's vessels through the image formation's blur,
    no noise, no background: score.formed_od), the fit's render and OD_img - render, 0 outside the valid
    region; and the shared OD scale (99th percentile of OD_img over the band)."""
    val = case["valid"]
    T = np.where(val, case["od_img"], 0.0).astype(np.float32)
    R = np.where(val, np.asarray(res["fit"]["od_render"], np.float32), 0.0).astype(np.float32)
    v = float(np.percentile(case["od_img"][case["band"]], 99)) if case["band"].any() else 1.0
    return T, R, T - R, v


def _truth_title(sc):
    return (f"explains {sc['explained_true']:.3f} of the clean OD "
            f"({sc['explained_true_junction']:.3f} at junctions; its own target's fidelity "
            f"{sc['target_fidelity']:.3f})")


def plot_vs_truth(case, res, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if res.get("fit") is None:
        return
    T, R, D, v = _truth_layers(case, res)
    H, W = T.shape
    fig, axs = plt.subplots(2, 2, figsize=(17, 17 * H / W + 1.3), layout="constrained")
    gray = _gray(case["image"], case["valid"])
    panels = [(axs[0, 0], T, "gray_r", 0, v, "vesselscene's render: the clean vessel OD (no camera noise, no background)"),
              (axs[0, 1], R, "gray_r", 0, v, "the fitted render · " + _truth_title(res["fit"]["score"])),
              (axs[1, 0], gray, "gray", 0, 1.25, "the image the annotator sees"),
              (axs[1, 1], D, "RdBu_r", -v / 2, v / 2, "vesselscene's render - fitted render (red: OD the fit "
                                                      "misses, blue: OD the fit adds)")]
    ims = {}
    for ax, img, cm, lo, hi, ttl in panels:
        ims[cm] = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="antialiased")
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims["gray_r"], ax=axs[0, :], shrink=0.8, pad=0.01, label="OD")
    fig.colorbar(ims["RdBu_r"], ax=axs[1, 1], shrink=0.8, pad=0.02, label="OD difference")
    fig.suptitle(f"{name}: vesselscene's render vs the fitted render (same OD scale, 0 to {v:.2f})",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=90, pil_kwargs=dict(quality=88))
    plt.close(fig)


def plot_vs_truth_junctions(case, res, name, path):
    """The junction windows of plot_junctions (same rule), as vesselscene's render | fitted render | difference:
    per coarse type the median on the left, the worst on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = pick_junctions(case, res)
    if not pick:
        return
    T, R, D, v = _truth_layers(case, res)
    types = [t for t in JSTYLE if any(r[0] == t for _, r in pick)]
    fig, axs = plt.subplots(len(types), 6, figsize=(2.3 * 6, 2.45 * len(types) + 0.7))
    axs = np.atleast_2d(axs)
    trio = [(T, "gray_r", 0, v, "vesselscene"), (R, "gray_r", 0, v, "fit"), (D, "RdBu_r", -v / 2, v / 2, "difference")]
    for i, typ in enumerate(types):
        sel = [(lab, r) for lab, r in pick if r[0] == typ]
        for k in range(2):
            for c in range(3):
                ax = axs[i, 3 * k + c]
                ax.set_xticks([])
                ax.set_yticks([])
                if k >= len(sel):
                    ax.axis("off")
                    continue
                lab, (_, _, x, y, fine) = sel[k]
                img, cm, lo, hi, ttl = trio[c]
                ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_xlim(x - HALF, x + HALF)
                ax.set_ylim(y + HALF, y - HALF)
                if c == 0:
                    ax.set_ylabel(f"{typ} ({fine})\n{lab} · ({x:.0f}, {y:.0f})", fontsize=8)
                if i == 0:
                    ax.set_title(ttl, fontsize=9)
    fig.suptitle(f"{name}: junctions close up ({2 * HALF} px), vesselscene's render vs the fitted render; per coarse "
                 f"type the median (left) and the worst (right) by the fit's residual", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.4, w_pad=0.3)
    fig.savefig(path, dpi=95, pil_kwargs=dict(quality=88))
    plt.close(fig)


def plot(folder, kind, cache, out_dir):
    from experiments.splinefit import fastset as FS
    FS.check_not_heldout(folder)
    case = S.load_truth(folder, kind, reference_target=False)
    res = _load(folder, kind, cache)
    name = _name(folder, kind)
    os.makedirs(out_dir, exist_ok=True)
    plot_annotated(case, res, name, os.path.join(out_dir, f"{name}_annotated.jpg"))
    plot_render(case, res, name, os.path.join(out_dir, f"{name}_render.jpg"))
    plot_junctions(case, res, name, os.path.join(out_dir, f"{name}_junctions.jpg"))
    plot_thumb(case, res, name, os.path.join(out_dir, f"{name}_thumb.jpg"))
    plot_vs_truth(case, res, name, os.path.join(out_dir, f"{name}_vs_truth.jpg"))
    plot_vs_truth_junctions(case, res, name, os.path.join(out_dir, f"{name}_vs_truth_junctions.jpg"))
    summary = {row: dict(digest=o["digest"], **{k: o["score"].get(k) for k in
                         ("composite", "graph_score", "pos", "width", "explained", "explained_junction",
                          "explained_true", "explained_true_junction", "target_fidelity",
                          "centreline_f1", "junction_f1_strict", "crossing_recall", "junction_type_balanced_coarse")})
               for row, o in res.items()}
    summary["truth"] = dict(n_junctions=len(case["obs"]["junctions"]), n_edges=len(_truth_lines(case)),
                            types={t: sum(COARSE.get(j["type_observable"]) == t for j in case["obs"]["junctions"])
                                   for t in JSTYLE})
    return name, summary


# ======================================================================== viewer
VIEWER_CSS = """
:root { --bg: #ffffff; --fg: #1d2127; --muted: #5b6573; --rule: #d9dee5; --panel: #f3f5f8; --accent: #1f6feb; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1216; --fg: #e3e7ec; --muted: #9aa5b1; --rule: #2c333c; --panel: #181d23; --accent: #6aa8ff; } }
:root[data-theme="dark"] { --bg: #0f1216; --fg: #e3e7ec; --muted: #9aa5b1; --rule: #2c333c; --panel: #181d23;
  --accent: #6aa8ff; }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--fg); }
body { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 1.4rem; margin: 0 0 0.3em; }
p { margin: 0.3em 0 0.8em; max-width: 90ch; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 12px 0 8px; }
label { display: inline-flex; gap: 6px; align-items: center; }
select, button { font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--rule);
  border-radius: 6px; padding: 4px 8px; }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.metrics { font-variant-numeric: tabular-nums; margin: 4px 0 10px; }
.metrics b { font-weight: 600; }
.wrap { overflow: auto; border: 1px solid var(--rule); border-radius: 6px; background: #fff; max-height: 82vh;
  touch-action: pan-x pan-y; }
.stage { position: relative; width: 100%; user-select: none; cursor: ew-resize; }
.stage img { display: block; width: 100%; height: auto; -webkit-user-drag: none; pointer-events: none; }
.stage img.top { position: absolute; inset: 0; }
.zoomed img { image-rendering: pixelated; }
.divider { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; background: #ffd400;
  box-shadow: 0 0 0 1px rgba(0,0,0,.45); pointer-events: none; }
.view { position: relative; }
.tag { position: absolute; top: 6px; z-index: 1; padding: 2px 6px; border-radius: 4px; font-size: 12px;
  background: rgba(15,18,22,.72); color: #fff; pointer-events: none; }
.tag.l { left: 6px; } .tag.r { right: 6px; }
input[type=range] { width: 100%; margin: 10px 0 4px; accent-color: var(--accent); }
.legend { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 12.5px; flex-wrap: wrap; }
.nowrap { display: inline-flex; gap: 8px; align-items: center; white-space: nowrap; }
.bar { width: 180px; height: 10px; border-radius: 3px; border: 1px solid var(--rule); }
"""

VIEWER_JS = """
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const LAYERS = D.layers;
let zoom = 1, wipe = 50;
for (const s of D.scenes) $('scene').add(new Option(s.name, s.name));
for (const sel of ['left', 'right']) for (const [k, v] of Object.entries(LAYERS)) $(sel).add(new Option(v.short, k));
$('scene').value = D.start; $('left').value = 'truth'; $('right').value = 'fit';
function scene() { return D.scenes.find(s => s.name === $('scene').value); }
function draw() {
  const s = scene(), L = $('left').value, R = $('right').value;
  $('imgL').src = s.png[L]; $('imgR').src = s.png[R];
  $('tagL').textContent = '\\u25C0 ' + LAYERS[L].short; $('tagR').textContent = LAYERS[R].short + ' \\u25B6';
  $('imgL').alt = LAYERS[L].long; $('imgR').alt = LAYERS[R].long;
  $('desc').textContent = 'Left: ' + LAYERS[L].long + '. Right: ' + LAYERS[R].long + '.';
  $('metrics').innerHTML = s.metrics;
  $('lo').textContent = '-' + s.half; $('hi').textContent = '+' + s.half;
  setWipe(wipe);
}
function setWipe(p) {
  wipe = Math.max(0, Math.min(100, p));
  $('imgR').style.clipPath = 'inset(0 0 0 ' + wipe + '%)';
  $('div').style.left = wipe + '%'; $('wipe').value = wipe;
}
function setZoom(z) {
  const w = $('wrap'), cy = (w.scrollTop + w.clientHeight / 2) / Math.max(w.scrollHeight, 1);
  zoom = z; $('stage').style.width = (100 * z) + '%';
  $('stage').classList.toggle('zoomed', z > 1);
  w.scrollLeft = $('stage').offsetWidth * wipe / 100 - w.clientWidth / 2;      // keep the wipe line in view
  w.scrollTop = cy * w.scrollHeight - w.clientHeight / 2;
  for (const b of document.querySelectorAll('[data-z]')) b.setAttribute('aria-pressed', String(+b.dataset.z === z));
}
let drag = false;
const at = e => { const r = $('stage').getBoundingClientRect(); setWipe(100 * (e.clientX - r.left) / r.width); };
$('stage').addEventListener('pointerdown', e => { e.preventDefault(); drag = true;
  $('stage').setPointerCapture(e.pointerId); at(e); });
$('stage').addEventListener('pointermove', e => { if (drag) at(e); });
for (const ev of ['pointerup', 'pointercancel']) $('stage').addEventListener(ev, () => { drag = false; });
$('wipe').addEventListener('input', e => setWipe(+e.target.value));
for (const id of ['scene', 'left', 'right']) $(id).addEventListener('change', draw);
for (const b of document.querySelectorAll('[data-z]')) b.addEventListener('click', () => setZoom(+b.dataset.z));
$('swap').addEventListener('click', () => { const l = $('left').value; $('left').value = $('right').value;
  $('right').value = l; draw(); });
setZoom(1); draw();
"""


def _b64(arr, mode, fmt="PNG", palette=None) -> str:
    import base64
    import io

    from PIL import Image
    arr = np.ascontiguousarray(arr)
    im = Image.frombytes(mode, (arr.shape[1], arr.shape[0]), arr.tobytes())      # "P": the bytes are the indices
    if palette is not None:
        im.putpalette(palette)
    buf = io.BytesIO()
    im.save(buf, format=fmt, **(dict(optimize=True) if fmt == "PNG" else dict(quality=90)))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _viewer_scene(case, res, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, R, D, v = _truth_layers(case, res)

    def od8(a):                                                                          # gray_r, 0 to v
        return np.round(255 * (1 - np.clip(a / v, 0, 1))).astype(np.uint8)

    idx = np.round(255 * np.clip((D / (v / 2) + 1) / 2, 0, 1)).astype(np.uint8)          # RdBu_r, -v/2 to v/2
    lut = (np.asarray(plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(np.uint8)
    gray = np.round(255 * np.clip(_gray(case["image"], case["valid"]) / 1.25, 0, 1)).astype(np.uint8)
    sc = res["fit"]["score"]
    other = " · ".join(f"{row} {res[row]['score']['explained_true']:.3f}" for row in ("proposals", "vesselmap")
                       if row in res)
    metrics = (f"<b>Fit</b> explains <b>{sc['explained_true']:.3f}</b> of vesselscene's render "
               f"({sc['explained_true_junction']:.3f} at junctions; {sc['explained']:.3f} of the noisy OD the "
               f"scorer uses). Its own target's fidelity to the render: {sc['target_fidelity']:.3f}. "
               f"Same measure for the other rows: {other}. Composite {sc['composite']:.3f}.")
    return dict(name=name, half=f"{v / 2:.2f} OD", metrics=metrics,
                png=dict(truth=_b64(od8(T), "L"), fit=_b64(od8(R), "L"), image=_b64(gray, "L", fmt="JPEG"),
                         diff=_b64(idx, "P", palette=lut.ravel().tolist())))


def viewer(pairs, cache, out_html, start=None):
    """One self-contained page: per image vesselscene's render, the fitted render, the image and their
    difference, compared by a wipe (drag on the image or the slider) at 1x / 2x / 4x."""
    from experiments.splinefit import fastset as FS
    scenes = []
    for folder, kind in pairs:
        FS.check_not_heldout(folder)
        res = _load(folder, kind, cache)
        if res.get("fit") is None:
            continue
        case = S.load_truth(folder, kind, reference_target=False)
        scenes.append(_viewer_scene(case, res, _name(folder, kind)))
        print("viewer:", scenes[-1]["name"], flush=True)
    layers = {"truth": dict(short="vesselscene render", long="vesselscene's render of the scene: the clean vessel "
                            "OD through the image formation's blur, no camera noise, no background (dark = more OD)"),
              "fit": dict(short="fitted render", long="the fitted render of the spline network, junctions and all, "
                          "on the same OD scale"),
              "image": dict(short="image", long="the image the annotator sees"),
              "diff": dict(short="difference", long="vesselscene's render minus the fitted render: red is OD the "
                           "fit misses, blue is OD the fit adds, white is a match")}
    data = dict(scenes=scenes, layers=layers,
                start=start if start in [s["name"] for s in scenes] else scenes[0]["name"])
    blob = json.dumps(data).replace("</", "<\\/")          # nothing in the JSON may close the script
    import matplotlib.pyplot as plt
    rdbu = ", ".join(f"rgb({r},{g},{b})" for r, g, b in
                     (np.asarray(plt.get_cmap("RdBu_r")(np.linspace(0, 1, 9)))[:, :3] * 255).round().astype(int))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Render vs fit</title><style>{VIEWER_CSS}</style></head><body><main>
<h1>vesselscene's render vs the fitted render</h1>
<p>The held-out images of experiments/REPORT.md. Drag across the image (or use the slider) to wipe between the two
layers; zoom to see junctions pixel by pixel. Both renders share one OD scale per image.</p>
<div class="controls">
<label>Image <select id="scene"></select></label>
<label>Left <select id="left"></select></label>
<label>Right <select id="right"></select></label>
<button id="swap" type="button">Swap</button>
<span><button type="button" data-z="1">1&times;</button> <button type="button" data-z="2">2&times;</button>
<button type="button" data-z="4">4&times;</button></span>
</div>
<div class="metrics" id="metrics"></div>
<div class="view"><span class="tag l" id="tagL"></span><span class="tag r" id="tagR"></span>
<div class="wrap" id="wrap"><div class="stage" id="stage"><img id="imgL" alt="" draggable="false">
<img id="imgR" class="top" alt="" draggable="false"><div class="divider" id="div"></div></div></div></div>
<input type="range" id="wipe" min="0" max="100" step="0.1" aria-label="wipe position">
<div class="legend"><span>difference:</span><span class="nowrap"><span id="lo"></span>
<span class="bar" style="background: linear-gradient(90deg, {rdbu})"></span><span id="hi"></span></span>
<span>(blue: the fit adds OD · red: the fit misses OD)</span></div>
<p id="desc"></p>
<script type="application/json" id="data">{blob}</script>
<script>{VIEWER_JS}</script>
</main></body></html>
"""
    with open(out_html, "w") as fh:
        fh.write(page)
    print(f"wrote {out_html} ({os.path.getsize(out_html) / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("run", "plot", "viewer"))
    ap.add_argument("images", nargs="+", help="SCENE_DIR KIND pairs")
    ap.add_argument("--cache", default=os.path.join(os.environ.get(
        "SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/scratchpad/"
        "work/splinefit"), "report_cache"))
    ap.add_argument("--out", default=None, help="plot: the figure folder; viewer: the HTML file")
    ap.add_argument("--start", default=None, help="viewer: the image shown first")
    a = ap.parse_args(argv)
    if len(a.images) % 2:
        ap.error("images are SCENE_DIR KIND pairs")
    pairs = list(zip(a.images[::2], a.images[1::2]))
    os.makedirs(a.cache, exist_ok=True)
    if a.mode == "run":
        for folder, kind in pairs:
            run(folder, kind, a.cache)
        return
    if not a.out:
        ap.error(f"{a.mode} needs --out")
    if a.mode == "viewer":
        viewer(pairs, a.cache, a.out, start=a.start)
        return
    summ = {}
    sp = os.path.join(a.out, "summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            summ = json.load(fh)
    for folder, kind in pairs:
        name, s = plot(folder, kind, a.cache, a.out)
        summ[name] = s
        print("plotted", name, flush=True)
    with open(sp, "w") as fh:
        json.dump(summ, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
