r"""Every truth junction of the held-out averages, judged one by one, and a random gallery of them.

    python -m experiments.splinefit.junction_gallery SCENE_DIR [...] --out DIR [--cache DIR] [--per-type 8] [--seed 0]

Per observable truth junction (score.py's strict matching, neuromimetic.harness.match_junctions with the truth
radius capped at STRICT_RADIUS_PX), for the fit, the proposals and vesselmap:
  found      a junction of the annotator's graph is matched to it;
  typed      ... and has the truth's coarse type (3-way, crossing, compound);
  render     the render's share of vesselscene's clean OD in the junction's disc (within the band):
             1 - sum (OD_img - R)^2 / sum OD_img^2; 'render ok' when >= 0.95;
  all        found, typed and render ok.
Writes scorecard.json / scorecard.md and junctions_<type>.jpg: PER_TYPE junctions per coarse type drawn at random
(seeded) from all held-out averages, each row the image, the truth graph, the fitted graph, vesselscene's render,
the fitted render and their difference, 64 px around the junction.  Reads the truth: a report tool.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from experiments.neuromimetic import harness as HN

from . import score as S
from .report_figures import COARSE, JSTYLE, TRUTH, _gray, _load

HALF = 32
ROWS = ("fit", "proposals", "vesselmap")


def judge(case, res) -> list:
    """Per truth junction: dict(x, y, type, image row results)."""
    T = case["od_img"]
    js = case["obs"]["junctions"]
    out = [dict(x=float(j["x"]), y=float(j["y"]), type=COARSE.get(j["type_observable"], "3-way"),
                fine=j["type_observable"], id=j["id"]) for j in js]
    by_id = {j["id"]: k for k, j in enumerate(js)}
    for row in ROWS:
        o = res.get(row)
        if o is None:
            continue
        _, juncs = S.network_graph(o["network"])
        pairs = HN.match_junctions(case["obs"], juncs, cap=HN.STRICT_RADIUS_PX)
        got = {by_id[j["id"]]: t for t, j in pairs}
        R = np.asarray(o["od_render"], np.float32)
        for k, (j, d) in enumerate(zip(js, case["discs"])):
            m = S.disc_mask(T.shape, np.array([[j["x"], j["y"], max(d[2], 4.0)]])) & case["band"]
            ex = 1.0 - float(((T - R)[m] ** 2).sum() / max((T[m] ** 2).sum(), 1e-12)) if m.any() else float("nan")
            t = got.get(k)
            out[k][row] = dict(found=t is not None, typed=t is not None and COARSE.get(t[2], "?") == out[k]["type"],
                               as_type=None if t is None else COARSE.get(t[2], t[2]), render=ex)
    return out


def scorecard(alljs) -> dict:
    card = {}
    for row in ROWS:
        card[row] = {}
        for typ in list(JSTYLE) + ["all"]:
            rr = [j[row] for j in alljs if row in j and (typ == "all" or j["type"] == typ)]
            if not rr:
                continue
            ok = [r["render"] >= 0.95 for r in rr if np.isfinite(r["render"])]
            card[row][typ] = dict(n=len(rr), found=float(np.mean([r["found"] for r in rr])),
                                  typed=float(np.mean([r["typed"] for r in rr])),
                                  render_ok=float(np.mean(ok)) if ok else float("nan"),
                                  render_median=float(np.nanmedian([r["render"] for r in rr])),
                                  all=float(np.mean([r["typed"] and np.isfinite(r["render"]) and r["render"] >= 0.95
                                                     for r in rr])))
    return card


def _graph(ax, polys, juncs, colour=None):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    for i, P in enumerate(polys):
        P = np.asarray(P, float).reshape(-1, 2)
        ax.plot(P[:, 0], P[:, 1], "-", color=colour or cm(i % 20), lw=1.6)
    for x, y, t in juncs:
        st = JSTYLE.get(COARSE.get(t, t), JSTYLE["3-way"])
        ax.plot(x, y, linestyle="none", marker=st["marker"], ms=10, mfc=st["color"], mec="k", mew=0.6)


def gallery(items, path, typ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ("image", "truth graph", "fitted graph", "vesselscene render", "fitted render", "difference")
    fig, axs = plt.subplots(len(items), len(cols), figsize=(2.3 * len(cols), 2.45 * len(items) + 0.8))
    axs = np.atleast_2d(axs)
    for i, (j, case, res, name) in enumerate(items):
        x, y = j["x"], j["y"]
        xl, yl = (x - HALF, x + HALF), (y + HALF, y - HALF)
        T = case["od_img"]
        R = np.asarray(res["fit"]["od_render"], np.float32)
        v = float(np.percentile(T[case["band"]], 99))
        gray = _gray(case["image"], case["valid"])
        tp = [np.asarray(e["xy"], float).reshape(-1, 2) for e in case["obs"]["graph"]["edges"]]
        tj = [(float(q["x"]), float(q["y"]), COARSE.get(q["type_observable"], "3-way")) for q in case["obs"]["junctions"]]
        fp, fj = S.network_graph(res["fit"]["network"])
        for c, col in enumerate(cols):
            ax = axs[i, c]
            if col in ("image", "truth graph", "fitted graph"):
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1.25, interpolation="nearest")
                if col == "truth graph":
                    _graph(ax, tp, tj, colour=TRUTH)
                elif col == "fitted graph":
                    _graph(ax, fp, fj)
                else:
                    ax.plot(x, y, "+", color="#ffd400", ms=12, mew=1.5)
            else:
                img, cm_, lo, hi = {"vesselscene render": (T, "gray_r", 0, v), "fitted render": (R, "gray_r", 0, v),
                                    "difference": (T - R, "RdBu_r", -v / 2, v / 2)}[col]
                ax.imshow(img, cmap=cm_, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col, fontsize=9)
        f = j["fit"]
        verdict = (f"fit: {f['as_type']}" if f["found"] else "fit: MISSED") + (" (right type)" if f["typed"] else
                                                                                "" if not f["found"] else " (WRONG type)")
        axs[i, 0].set_ylabel(f"{name}\n{j['fine']} ({x:.0f}, {y:.0f})\n{verdict}\nrender {f['render']:.2f}", fontsize=7.5)
    fig.suptitle(f"Held-out averages: {len(items)} {typ} junctions drawn at random (64 px windows). Truth graph green; "
                 f"fitted graph one colour per edge; markers: cyan 3-way, magenta crossing, yellow compound",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.3, w_pad=0.2)
    fig.savefig(path, dpi=85, pil_kwargs=dict(quality=88))
    plt.close(fig)


def main(argv=None):
    from experiments.splinefit import fastset as FS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--kind", default="average")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.join(os.environ.get(
        "SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/scratchpad/"
        "work/splinefit"), "report_cache"))
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    alljs, pool = [], []
    for folder in a.scenes:
        FS.check_not_heldout(folder)
        case = S.load_truth(folder, a.kind, reference_target=False)
        res = _load(folder, a.kind, a.cache)
        name = os.path.basename(os.path.normpath(folder)).split("_480")[0].replace("_s00", " ")
        js = judge(case, res)
        H, W = case["od_img"].shape
        for j in js:
            j["image"] = name
            alljs.append(j)
            if HALF <= j["x"] < W - HALF and HALF <= j["y"] < H - HALF:
                pool.append((j, case, res, name))
    card = scorecard(alljs)
    with open(os.path.join(a.out, "scorecard.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(card=card, junctions=alljs), fh, indent=1, default=float)
    lines = ["| annotator | junction type | n | found | right type | render >= 0.95 | found, right type and render >= 0.95 |",
             "|---|---|---|---|---|---|---|"]
    for row in ROWS:
        for typ, c in card[row].items():
            lines.append(f"| {row} | {typ} | {c['n']} | {c['found']:.0%} | {c['typed']:.0%} | {c['render_ok']:.0%} | "
                         f"{c['all']:.0%} |")
    with open(os.path.join(a.out, "scorecard.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    rng = random.Random(a.seed)
    for typ in JSTYLE:
        cand = [p for p in pool if p[0]["type"] == typ]
        items = rng.sample(cand, min(a.per_type, len(cand)))
        gallery(items, os.path.join(a.out, f"junctions_{typ}.jpg"), typ)
        print("wrote", typ, len(items))


if __name__ == "__main__":
    main()
