r"""Score several annotators on the same scenes and write one comparison table.

    python -m experiments.neuromimetic.evaluate --scenes DIR [DIR ...] --out results/test \
        --annotators experiments.neuromimetic.neuromimetic:annotate experiments.neuromimetic.baseline_hessian:annotate

Writes <out>/<module>.<function>.json per annotator (harness.run rows and summary) and <out>/table.md: the
mean and range over the scenes of each score, per image kind, in two tables (lines and edges; junctions and
their types, with the majority-label and balanced-accuracy references next to the type accuracies).

Existing results are reused: an annotator whose <out>/<name>.json already exists is not run again, its rows
are read from that file (delete the file to re-run it).  table.md covers only the annotators named on this
command line.  The tables of a whole results folder -- table_all.md over every JSON in it, paired.md (the full
annotator minus each other one, per image) and ceiling.json (the truth's own graph scored by the harness) --
are made by report.py:

    python -m experiments.neuromimetic.report results/test [--scenes DIR [DIR ...] --ceiling]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import harness as H

LINES = [("centreline_recall", "line R"), ("centreline_precision", "line P"), ("centreline_f1", "line F1"),
         ("edge_cover", "edge cover"), ("pieces_per_edge", "pieces/edge"), ("polyline_purity", "purity"),
         ("cost_seconds", "s / image")]
JUNCS = [("junction_recall", "junc R"), ("junction_precision", "junc P"), ("junction_f1", "junc F1"),
         ("junction_f1_strict", "junc F1 strict"), ("junction_type_accuracy_coarse", "type coarse"),
         ("junction_type_majority_coarse", "coarse majority ref"), ("junction_type_balanced_coarse", "coarse balanced"),
         ("junction_type_accuracy", "type exact"), ("junction_type_majority", "exact majority ref"),
         ("crossing_recall", "X R"), ("crossing_precision", "X P")]


def _cell(vals, key):
    v = np.asarray([x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)], float)
    if not len(v):
        return "-"
    fmt = "{:.0f}" if key == "cost_seconds" else "{:.2f}"
    if len(v) == 1 or np.ptp(v) == 0:
        return fmt.format(v.mean())
    return f"{fmt.format(v.mean())} ({fmt.format(v.min())}-{fmt.format(v.max())})"


def _det(rr) -> str:
    d = [r.get("deterministic") for r in rr]
    if any(x is None for x in d):
        return "not tested"
    return "yes" if all(d) else "no"


def table(results: dict, kinds) -> str:
    """Two markdown tables per image kind: lines and edges, then junctions and their types.  's / image' is
    the build time for a cached annotator (vesselmap's build_seconds), else the harness's run time."""
    out = []
    for kind in kinds:
        for title, cols, det in (("lines and edges", LINES, True), ("junctions and types", JUNCS, False)):
            out.append(f"\n**{kind}: {title}** (mean over scenes, range in brackets)\n")
            out.append("| annotator | " + " | ".join(c[1] for c in cols) + (" | deterministic |" if det else " |"))
            out.append("|---" * (len(cols) + 1 + det) + "|")
            for name, rows in results.items():
                rr = [dict(r, cost_seconds=r.get("build_seconds", r.get("seconds"))) for r in rows if r["kind"] == kind]
                if not rr:
                    continue
                cells = " | ".join(_cell([r.get(k) for r in rr], k) for k, _ in cols)
                out.append(f"| {name} | {cells}" + (f" | {_det(rr)} |" if det else " |"))
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["average", "frame"])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--no-repeat", nargs="*", default=[], help="annotators to run once (known not bit-reproducible)")
    ap.add_argument("--out", required=True,
                    help="results folder; an existing <out>/<name>.json is reused, not re-run")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for ann in a.annotators:
        name = ann.replace("experiments.neuromimetic.", "").replace(":", ".")
        path = os.path.join(a.out, name + ".json")
        if os.path.exists(path):
            with open(path) as fh:
                results[name] = json.load(fh)["rows"]
            continue
        rep = 1 if ann in a.no_repeat else a.repeat
        rows = H.run(ann, a.scenes, a.kinds, rep)
        for r in rows:
            r["repeat"] = rep
        with open(path, "w") as fh:
            json.dump(dict(annotator=ann, rows=rows, summary=H.summarise(rows)), fh, indent=1, default=float)
        results[name] = rows
    md = table(results, a.kinds)
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
