r"""Annotated-scene figures of the held-out report: the observable truth, the neuromimetic proposals, the fitted
spline network and LIMBUS vesselmap drawn over the image, the renders and residuals, and junction close-ups.

    python -m experiments.splinefit.report_figures run  SCENE_DIR KIND [SCENE_DIR KIND ...] [--cache DIR]
    python -m experiments.splinefit.report_figures plot SCENE_DIR KIND [SCENE_DIR KIND ...] [--cache DIR] --out DIR

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
    <scene>_<kind>_thumb.jpg       the fit over the image with the truth underneath (the gallery).

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
    summary = {row: dict(digest=o["digest"], **{k: o["score"].get(k) for k in
                         ("composite", "graph_score", "pos", "width", "explained", "explained_junction",
                          "centreline_f1", "junction_f1_strict", "crossing_recall", "junction_type_balanced_coarse")})
               for row, o in res.items()}
    summary["truth"] = dict(n_junctions=len(case["obs"]["junctions"]), n_edges=len(_truth_lines(case)),
                            types={t: sum(COARSE.get(j["type_observable"]) == t for j in case["obs"]["junctions"])
                                   for t in JSTYLE})
    return name, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("run", "plot"))
    ap.add_argument("images", nargs="+", help="SCENE_DIR KIND pairs")
    ap.add_argument("--cache", default=os.path.join(os.environ.get(
        "SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/scratchpad/"
        "work/splinefit"), "report_cache"))
    ap.add_argument("--out", default=None)
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
        ap.error("plot needs --out")
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
