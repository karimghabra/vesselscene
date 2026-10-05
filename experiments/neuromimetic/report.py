"""Tables from the per-annotator JSONs that evaluate.py writes, and the truth's own score (the ceiling).

    python -m experiments.neuromimetic.report RESULTS_DIR [--scenes DIR [DIR ...] --ceiling] [--kinds ...]

Reads every <RESULTS_DIR>/<name>.json that holds harness rows (``evaluate.py`` writes them as
<module>.<function>.json, e.g. neuromimetic.annotate.json) and writes, in RESULTS_DIR:

  * table_all.md: ``evaluate.table`` over every annotator present, in a fixed order: the ceiling,
    neuromimetic.annotate, the baselines, then the neuromimetic variants (``ORDER`` for the known names,
    any other name alphabetically within its group);
  * paired.md: per image kind, the full annotator (neuromimetic.annotate) minus each other annotator on the
    same images: the mean per-image difference and on how many images the full annotator is higher (images
    where either score is NaN are left out, so the count can fall below the number of images);
  * with --ceiling (needs --scenes, the scene folders the JSONs were scored on): first ceiling.json, the
    truth's own graph scored by the harness -- polylines = the observable graph's edges, junctions = the
    observable junctions with their observable type -- in harness.run's row format (seconds 0, repeat 1,
    determinism not tested), so it enters table_all.md as the first row.  This reads the truth files, which
    is what a scorer does; it is not an annotator.

Nothing is re-run: the annotator rows come from the JSONs as they are.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import evaluate as E
from . import harness as H

FULL = "neuromimetic.annotate"
CEILING = "ceiling"
CEILING_LABEL = "ceiling (truth graph edges, typed truth junctions)"
# the order of the held-out tables; a name not listed goes after the listed ones of its group, alphabetically
ORDER = [CEILING, FULL, "baseline_hessian.annotate", "baseline_vesselmap.annotate",
         "neuromimetic.annotate_no_verify", "neuromimetic.annotate_no_association", "neuromimetic.annotate_no_surround",
         "neuromimetic.annotate_v1_only", "neuromimetic.annotate_repropose", "neuromimetic.annotate_bg_traced",
         "neuromimetic.annotate_no_compound_prior"]
KIND_ORDER = ["average", "frame"]
PAIRED = [("centreline_f1", "line F1"), ("centreline_precision", "line P"), ("centreline_recall", "line R"),
          ("junction_f1", "junc F1"), ("junction_precision", "junc P"), ("crossing_recall", "X R"),
          ("crossing_precision", "X P"), ("junction_type_accuracy_coarse", "type coarse"),
          ("junction_type_balanced_coarse", "coarse bal"), ("edge_cover", "cover")]


def _group(name: str) -> int:
    if name == CEILING:
        return 0
    if name == FULL:
        return 1
    if name.startswith("baseline_"):
        return 2
    if name.startswith("neuromimetic."):
        return 3
    return 4


def sort_names(names) -> list:
    """The fixed table order (module docstring)."""
    return sorted(names, key=lambda n: (_group(n), ORDER.index(n) if n in ORDER else len(ORDER), n))


def short(name: str) -> str:
    """A row label for paired.md: neuromimetic.annotate_v1_only -> annotate_v1_only,
    baseline_hessian.annotate -> baseline_hessian."""
    if name.startswith("neuromimetic."):
        return name[len("neuromimetic."):]
    return name[:-len(".annotate")] if name.endswith(".annotate") else name


def load_results(results_dir: str) -> dict:
    """{name: rows} for every JSON in results_dir with harness rows, in the fixed order."""
    res = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as fh:
            d = json.load(fh)
        if isinstance(d, dict) and isinstance(d.get("rows"), list):
            res[os.path.splitext(os.path.basename(path))[0]] = d["rows"]
    return {n: res[n] for n in sort_names(res)}


def kinds_of(results: dict) -> list:
    present = {r["kind"] for rows in results.values() for r in rows}
    return [k for k in KIND_ORDER if k in present] + sorted(present - set(KIND_ORDER))


def ceiling_rows(scenes, kinds) -> list:
    """The truth's own graph scored by the harness, one harness.run-style row per scene and kind."""
    rows = []
    for folder in scenes:
        for kind in kinds:
            case = H.load_case(folder, kind)
            obs = case["obs"]
            out = dict(polylines=[e["xy"] for e in obs["graph"]["edges"]],
                       junctions=[(j["x"], j["y"], j["type_observable"]) for j in obs["junctions"]])
            s = H.score(case, out)
            s.update(scene=os.path.basename(os.path.normpath(folder)), kind=kind, seconds=0.0, repeat=1,
                     digest=H.digest(out), deterministic=None)
            rows.append(s)
    return rows


def paired(results: dict, kinds, full: str = FULL) -> str:
    """Per kind, ``full`` minus each other annotator (not the ceiling) on the same (scene, kind) images."""
    a_all = {(r["scene"], r["kind"]): r for r in results[full]}
    out = []
    for kind in kinds:
        a = {i: r for i, r in a_all.items() if i[1] == kind}
        out.append(f"\n**{kind}: full annotator minus each (mean per-image difference, images where the full "
                   f"annotator is higher / all)**\n")
        out.append("| minus | " + " | ".join(c[1] for c in PAIRED) + " |")
        out.append("|---" * (len(PAIRED) + 1) + "|")
        for name, rows in results.items():
            if name in (full, CEILING):
                continue
            b = {(r["scene"], r["kind"]): r for r in rows if r["kind"] == kind}
            if not b:
                continue
            cells = []
            for k, _ in PAIRED:
                d = [a[i][k] - b[i][k] for i in sorted(a) if i in b
                     and np.isfinite(a[i].get(k, np.nan)) and np.isfinite(b[i].get(k, np.nan))]
                cells.append(f"{np.mean(d):+.2f} ({sum(x > 0 for x in d)}/{len(d)})" if d else "-")
            out.append(f"| {short(name)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("results_dir", help="folder of <module>.<function>.json files written by evaluate.py")
    ap.add_argument("--scenes", nargs="+", default=[], help="scene folders (only needed for --ceiling)")
    ap.add_argument("--ceiling", action="store_true", help="score the truth's own graph into ceiling.json first")
    ap.add_argument("--kinds", nargs="+", help="image kinds (default: those in the JSONs, average and frame first)")
    a = ap.parse_args(argv)
    if a.ceiling:
        if not a.scenes:
            ap.error("--ceiling needs --scenes")
        rows = ceiling_rows(a.scenes, a.kinds or KIND_ORDER)
        with open(os.path.join(a.results_dir, CEILING + ".json"), "w") as fh:
            json.dump(dict(annotator=CEILING_LABEL, rows=rows, summary=H.summarise(rows)), fh, indent=1,
                      default=float)
    results = load_results(a.results_dir)
    if not results:
        ap.error(f"no harness JSONs in {a.results_dir}")
    kinds = a.kinds or kinds_of(results)
    with open(os.path.join(a.results_dir, "table_all.md"), "w") as fh:
        fh.write(E.table(results, kinds))
    written = ["table_all.md"]
    if FULL in results:
        with open(os.path.join(a.results_dir, "paired.md"), "w") as fh:
            fh.write(paired(results, kinds))
        written.append("paired.md")
    print("wrote", ", ".join(os.path.join(a.results_dir, w) for w in ([CEILING + ".json"] if a.ceiling else [])
                             + written))


if __name__ == "__main__":
    main()
