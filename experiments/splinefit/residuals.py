r"""Recording: the prediction error of a fitted render, looked at where it matters (junctions and crossings).

In the predictive-coding frame the render is the top-down prediction and the residual OD_obs - R is the
prediction error; a whole-image score averages it away.  As an electrophysiologist records from the cells a
manipulation should affect, this module records the residual of one pipeline at the observable truth's
junctions and crossings (DIAGNOSTIC: it reads the truth to know where to look and to type the junctions,
never in a pipeline path).

Per case (a fast-tier crop, a dev image or a probe):

    overview    OD_obs (the scorer's oracle-background target, score.py), the render R, the residual
                OD_obs - R (diverging, symmetric scale = the 99th percentile of |OD_obs| in the band), with the
                fitted edges (thin lines) and the truth junctions (circles by type) overlaid;
    gallery     the K junctions with the largest summed |residual| in their disc (the scorer's junction disc,
                radius at least 4 px), each a 2 * HALF px window of target / render / residual;
    stats       per truth junction type: n, the signed residual sum over the disc / the target sum (the
                junction bias, > 0 = the render over-predicts) and the share of the case's squared residual
                inside the discs.

    python -m experiments.splinefit.residuals [--pipeline mod:fn] [--fast [I ...]] [--probe NAME ...]
        [--scene FOLDER KIND [x0 y0 w h]] [--k 6] [--out DIR]

writes <out>/<case>.png and <out>/residuals.json (the stats) and prints the stats table.  Default out:
$SPLINEFIT_WORK/runs/residuals.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import json                                         # noqa: E402

import numpy as np                                  # noqa: E402

WORK = os.environ.get("SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/"
                                        "scratchpad/work/splinefit")
HALF = 16                     # px, half size of a gallery window
TYPE_COLOUR = {"bifurcation": "#1b9e77", "confluence": "#7570b3", "pseudo-T": "#e6ab02", "crossing": "#d95f02",
               "compound": "#e7298a", "branch": "#66a61e"}


def _jtype(j: dict) -> str:
    return j.get("type_observable", j.get("type", "?"))


def junction_stats(case: dict, R: np.ndarray) -> dict:
    """Per truth junction: its type, centre, disc residual sum / target sum and squared residual."""
    from experiments.splinefit import score as S
    t = case["od_obs"]
    res = np.where(np.isfinite(t) & np.isfinite(R), t - R, 0.0)
    tot = float((res[case["band"]] ** 2).sum()) or 1.0
    rows = []
    for j, (x, y, r) in zip(case["obs"]["junctions"], case["discs"]):
        m = S.disc_mask(t.shape, np.array([[x, y, max(r, 4.0)]])) & case["usable"]
        den = float(np.abs(t[m]).sum())
        rows.append(dict(type=_jtype(j), x=float(x), y=float(y), abs_res=float(np.abs(res[m]).sum()),
                         bias=float(-res[m].sum() / den) if den > 0 else float("nan"),
                         sq_share=float((res[m] ** 2).sum()) / tot))
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r)
    summ = {ty: dict(n=len(rs), bias=float(np.nanmean([r["bias"] for r in rs])),
                     sq_share=float(sum(r["sq_share"] for r in rs))) for ty, rs in sorted(by.items())}
    return dict(junctions=rows, by_type=summ,
                sq_share_discs=float(sum(r["sq_share"] for r in rows)))


def plot_case(case: dict, out: dict, path: str, title: str, k: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = case["od_obs"]
    R = np.asarray(out.get("od_render", np.zeros_like(t)), np.float32)
    res = t - R
    band = case["band"]
    v = float(np.nanpercentile(np.abs(t[band]), 99)) if band.any() else float(np.nanmax(np.abs(t)) or 1.0)
    st = junction_stats(case, R)
    top = sorted(range(len(st["junctions"])), key=lambda i: -st["junctions"][i]["abs_res"])[:k]
    ncol = 3
    fig = plt.figure(figsize=(11, 3.9 + 1.25 * len(top)))
    gs = fig.add_gridspec(1 + len(top), 6, height_ratios=[3.4] + [1.1] * len(top))
    edges = (out.get("network") or {}).get("edges", [])
    for i, (img, nm, cm, lo, hi) in enumerate(((t, "OD_obs (target)", "gray_r", 0, v), (R, "render", "gray_r", 0, v),
                                                (res, "residual OD_obs - R", "RdBu_r", -v / 2, v / 2))):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        if i == 2:
            for e in edges:
                xy = np.asarray(e["xy"], float)
                ax.plot(xy[:, 0], xy[:, 1], "-", color="k", lw=0.4, alpha=0.6)
        for n, jr in enumerate(st["junctions"]):
            ax.add_patch(plt.Circle((jr["x"], jr["y"]), 5, fill=False, lw=0.8,
                                    color=TYPE_COLOUR.get(jr["type"], "k")))
            if i == 2 and n in top:
                ax.text(jr["x"] + 6, jr["y"] - 6, str(top.index(n) + 1), fontsize=6)
        ax.set_xlim(-0.5, t.shape[1] - 0.5)
        ax.set_ylim(t.shape[0] - 0.5, -0.5)
        ax.set_title(nm, fontsize=8)
        ax.axis("off")
    for row, n in enumerate(top, start=1):
        jr = st["junctions"][n]
        cx, cy = int(round(jr["x"])), int(round(jr["y"]))
        y0, x0 = max(0, cy - HALF), max(0, cx - HALF)
        sl = (slice(y0, cy + HALF), slice(x0, cx + HALF))
        for c, (img, cm, lo, hi) in enumerate(((t, "gray_r", 0, v), (R, "gray_r", 0, v),
                                               (res, "RdBu_r", -v / 2, v / 2))):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(img[sl], cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.axis("off")
            if c == 0:
                ax.set_title(f"#{row} {jr['type']} bias {jr['bias']:+.2f}", fontsize=7, loc="left")
        ax = fig.add_subplot(gs[row, ncol:])
        ax.axis("off")
        ax.text(0, 0.5, f"({jr['x']:.0f}, {jr['y']:.0f})  sum|res| {jr['abs_res']:.2f}  "
                        f"share of sq. residual {jr['sq_share']:.3f}", fontsize=7, va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return st


def cases(fast=None, probes=None, scene=None):
    """(name, folder, kind, crop) of the requested cases."""
    from experiments.splinefit import fastset as FS
    from experiments.splinefit import probes as P
    out = []
    if fast is not None:
        crops = FS.load()
        for i in (fast or range(len(crops))):
            c = crops[i]
            out.append((f"fast{i}_{c['scene']}_{c['kind']}", os.path.join(FS.DEV_DIR, c["scene"]), c["kind"],
                        c["crop"]))
    if probes:
        bat = {st["name"]: st for st in P.load_battery()}
        for nm in probes:
            out.append((f"probe_{nm}", bat[nm]["path"], bat[nm]["kind"], None))
    if scene:
        folder, kind, *crop = scene
        crop = [int(v) for v in crop] or None
        out.append((f"{os.path.basename(folder)}_{kind}", folder, kind, crop))
    for _, folder, _, _ in out:
        FS.check_not_heldout(folder)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipeline", default="experiments.splinefit.pipeline:annotate")
    ap.add_argument("--fast", nargs="*", type=int, help="fast-tier crop indices (none given: all)")
    ap.add_argument("--probe", nargs="*", help="probe names")
    ap.add_argument("--scene", nargs="+", help="FOLDER KIND [x0 y0 w h]")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(WORK, "runs", "residuals"))
    a = ap.parse_args(argv)
    import importlib
    from experiments.splinefit import score as S
    from experiments.splinefit import set_threads
    set_threads(2)
    mod, f = a.pipeline.split(":")
    fn = getattr(importlib.import_module(mod), f)
    os.makedirs(a.out, exist_ok=True)
    allst = {}
    for name, folder, kind, crop in cases(a.fast, a.probe, a.scene):
        case = S.load_truth(folder, kind, crop)
        out = fn(case["image"].copy(), case["valid"].copy())
        if out.get("od_render") is None and out.get("network") is not None:
            out["od_render"] = S.render_network(S.to_vesselmap(out["network"], case["image"].shape,
                                                               out.get("halo")), case["image"].shape)
        sc = S.score(case, out)
        st = plot_case(case, out, os.path.join(a.out, f"{name}.png"),
                       f"{name}  composite {sc['composite']:.3f}  explained {sc['explained']:.3f}  "
                       f"explained_junction {sc['explained_junction']:.3f}  junction_bias {sc['junction_bias']:+.3f}",
                       a.k)
        st["scores"] = {k: sc[k] for k in ("composite", "graph_score", "pos", "width", "explained",
                                           "explained_junction", "junction_bias")}
        allst[name] = st
        bt = "  ".join(f"{ty} n{v['n']} bias {v['bias']:+.2f} sq {v['sq_share']:.2f}" for ty, v in st["by_type"].items())
        print(f"{name:48s} junction discs hold {st['sq_share_discs']:.2f} of the squared residual | {bt}", flush=True)
    with open(os.path.join(a.out, "residuals.json"), "w", encoding="utf-8") as fh:
        json.dump(allst, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
