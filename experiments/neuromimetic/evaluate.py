"""Score several annotators on the same scenes and write one comparison table.

    python -m experiments.neuromimetic.evaluate --scenes DIR [DIR ...] --out results/test \
        --annotators experiments.neuromimetic.neuromimetic:annotate experiments.neuromimetic.baseline_hessian:annotate

Writes <out>/<module>.<function>.json per annotator (harness.run rows and summary) and <out>/table.md: the
mean and range over the scenes of each score, per image kind.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import harness as H

COLS = [("centreline_recall", "line R"), ("centreline_precision", "line P"), ("centreline_f1", "line F1"),
        ("junction_recall", "junc R"), ("junction_precision", "junc P"), ("junction_f1", "junc F1"),
        ("junction_type_accuracy_coarse", "type (coarse)"), ("junction_type_accuracy", "type (exact)"),
        ("crossing_recall", "X R"), ("crossing_precision", "X P"), ("edge_cover", "edge cover"),
        ("pieces_per_edge", "pieces/edge"), ("seconds", "s / image")]


def _cell(vals, key):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "-"
    fmt = "{:.0f}" if key == "seconds" else "{:.2f}"
    if len(v) == 1 or np.ptp(v) == 0:
        return fmt.format(v.mean())
    return f"{fmt.format(v.mean())} ({fmt.format(v.min())}-{fmt.format(v.max())})"


def table(results: dict, kinds) -> str:
    out = []
    for kind in kinds:
        out.append(f"\n**{kind}** (mean over scenes, range in brackets)\n")
        out.append("| annotator | " + " | ".join(c[1] for c in COLS) + " | deterministic |")
        out.append("|---" * (len(COLS) + 2) + "|")
        for name, rows in results.items():
            rr = [r for r in rows if r["kind"] == kind]
            if not rr:
                continue
            det = "yes" if all(r["deterministic"] for r in rr) else "no"
            if all(r.get("repeat", 2) == 1 for r in rr):
                det = "not tested"
            out.append(f"| {name} | " + " | ".join(_cell([r.get(k) for r in rr], k) for k, _ in COLS) + f" | {det} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["average", "frame"])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--no-repeat", nargs="*", default=[], help="annotators to run once (known not bit-reproducible)")
    ap.add_argument("--out", required=True)
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
