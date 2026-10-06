"""Held-out comparison tables of the final spline fit against its references (proposals alone, LIMBUS
vesselmap, the fit started from the truth, the true network).

    python -m experiments.splinefit.heldout_report --fit FIT.json --reference DIR --out DIR

FIT.json is run_experiment's tier-2 output on the held-out scenes; DIR holds evaluate_fit's row JSONs
(proposals.json, vesselmap.json, oracle_init.json, true_complete.json).  Writes table.md (means per image kind
over all held-out images and over the scenes no development agent ever read), paired.md (the fit minus each
reference, per image: the mean difference and the number of images where the fit is higher) and fit.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .fastset import KINDS

PARTS = [("composite", "composite"), ("graph_score", "graph"), ("pos", "pos"), ("width", "width"),
         ("explained", "explained"), ("explained_junction", "expl. junction"), ("centreline_f1", "line F1"),
         ("junction_f1_strict", "junc F1 strict"), ("junction_type_balanced_coarse", "coarse bal."),
         ("edge_cover", "edge cover"), ("crossing_recall", "X R"), ("offset_px", "offset px"),
         ("width_rel_err", "width rel err"), ("blur_err_px", "blur err px"), ("contrast_rel_err", "contrast rel err"),
         ("seconds", "s / image")]
PAIRED = [("composite", "composite"), ("graph_score", "graph"), ("pos", "pos"), ("width", "width"),
          ("explained", "explained"), ("explained_junction", "expl. junction"), ("centreline_f1", "line F1"),
          ("junction_f1_strict", "junc F1 strict"), ("crossing_recall", "X R")]
# healthy seeds 1 and 2 had their junction-type counts read by a reviewer agent during the neuromimetic study
UNTOUCHED = {"healthy_s003_480x768", "healthy_s005_480x768", "healthy_s006_480x768", "pathologic_s001_480x768"}
ORDER = ["fit", "proposals", "vesselmap", "oracle_init", "true_complete"]


def _load(fit_json, ref_dir) -> dict:
    with open(fit_json) as fh:
        res = {"fit": json.load(fh)["rows"]}
    for f in sorted(os.listdir(ref_dir)):
        if f.endswith(".json"):
            with open(os.path.join(ref_dir, f)) as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "rows" in d and "row" in d:
                res[d["row"]] = d["rows"]
    return {n: res[n] for n in ORDER if n in res}


def _fmt(v, key):
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.0f}" if key == "seconds" else f"{v:.3f}"


def table(res: dict, scenes=None) -> str:
    out = []
    for kind in KINDS:
        out.append(f"\n**{kind}**\n")
        out.append("| row | n | " + " | ".join(p[1] for p in PARTS) + " |")
        out.append("|---" * (len(PARTS) + 2) + "|")
        for name, rows in res.items():
            rr = [r for r in rows if r["kind"] == kind and (scenes is None or r["scene"] in scenes)]
            if not rr:
                continue
            cells = [_fmt(float(np.nanmean([r.get(k, np.nan) for r in rr])), k) for k, _ in PARTS]
            out.append(f"| {name} | {len(rr)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def paired(res: dict, scenes=None) -> str:
    fit = {(r["scene"], r["kind"]): r for r in res["fit"]}
    out = []
    for kinds, label in [((k,), k) for k in KINDS]:
        out.append(f"\n**{label}: fit minus each row** (mean per-image difference; images where the fit is "
                   f"higher / tied / all)\n")
        out.append("| minus | " + " | ".join(p[1] for p in PAIRED) + " |")
        out.append("|---" * (len(PAIRED) + 1) + "|")
        for name, rows in res.items():
            if name == "fit":
                continue
            b = {(r["scene"], r["kind"]): r for r in rows}
            ids = [i for i in sorted(fit) if i in b and i[1] in kinds and (scenes is None or i[0] in scenes)]
            cells = []
            for k, _ in PAIRED:
                d = [fit[i][k] - b[i][k] for i in ids if np.isfinite(fit[i].get(k, np.nan))
                     and np.isfinite(b[i].get(k, np.nan))]
                cells.append(f"{np.mean(d):+.3f} ({sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{len(d)})"
                             if d else "-")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = _load(a.fit, a.reference)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write("## All 6 held-out scenes\n" + table(res)
                 + "\n## The 4 scenes no development agent ever read (healthy 3, 5, 6; pathologic 1)\n"
                 + table(res, UNTOUCHED))
    with open(os.path.join(a.out, "paired.md"), "w") as fh:
        fh.write("## All 6 held-out scenes\n" + paired(res)
                 + "\n## The 4 untouched scenes\n" + paired(res, UNTOUCHED))
    for name, rows in res.items():
        with open(os.path.join(a.out, f"{name}.json"), "w") as fh:
            json.dump(dict(row=name, rows=rows), fh, indent=1, default=float)
    print(open(os.path.join(a.out, "table.md")).read())
    print(open(os.path.join(a.out, "paired.md")).read())


if __name__ == "__main__":
    main()
