"""Re-judge the overnight loop's decisions on the averaged stills alone (re-baseline R2; a diagnostic, outside
the frozen metric).  Reads the run JSONs the loop kept under $SPLINEFIT_WORK (never committed) and replays each
decision on its own A/B pair twice: on both kinds and on the average still only.  Every run's summary composite
must equal its logged one (results/results.tsv, LOGGED below), so the replay reads the runs the loop judged.

    python -m experiments.splinefit.rejudge_average [--work DIR]

Tier 1: composite = 0.5 mean composite over the crops + 0.5 mean probe_composite over the probes, of the
kinds in question (the night's fast set: one average and one frame crop per dev scene, so the replay has 5
average crops, not R2's 10); the rule keeps a change that improves on the parent by >= 0.002, or ties within
0.002 and is simpler (program.md 5.6).  A decision FLIPS when the rule's verdict on the averages differs from
its verdict on both kinds.  The logged status can differ from the rule for reasons that have nothing to do
with the image kind (the controls veto, a correctness fix kept within 0.002); those are listed, not flipped.
E12 (ba7302d) kept no run JSON (only b4/e12.log) and is not replayed.  Tier 2: paired.paired (per scene, guards) on the rows of each kind set.  Controls: the
multi-seed controls score over the average-still stimuli only.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np

from . import paired as PD
from .run_experiment import WORK

BOTH, AVG = ("average", "frame"), ("average",)
# label, parent run, run, logged status, simpler (folders relative to WORK)
TIER1 = (("E1 retarget early", "runs/baseline/tier1.json", "runs/e1_tier1.json", "discard", False),
         ("E2 anchor_px 1.5", "runs/baseline/tier1.json", "runs/e2_tier1.json", "discard", False),
         ("E3 final fit, geometry frozen", "runs/baseline/tier1.json", "runs/e3_tier1.json", "keep", False),
         ("E4 support 6 px", "runs/e3_tier1.json", "runs/e4_tier1.json", "discard", False),
         ("E5 wide edge flank test", "runs/e3_tier1.json", "runs/e5_tier1.json", "keep", False),
         ("E6 flank test faint only", "runs/e5_tier1.json", "runs/e6_tier1.json", "keep", False),
         ("E7 stage-7 proposals", "runs/e6_tier1.json", "runs/e7_tier1.json", "discard", False),
         ("E8 E7 + mdl_scale 0.5", "runs/e6_tier1.json", "runs/e8_tier1.json", "discard", False),
         ("E9 prune simplification", "runs/b3r_tier1.json", "runs/e9_tier1.json", "keep", True),
         ("E10 stage-7 + soft wide test", "runs/e9_tier1.json", "runs/e10_tier1.json", "keep", False),
         ("E11 pruned retarget", "runs/e10_tier1.json", "runs/e11_tier1.json", "keep", False),
         ("E13 re-propose arms, 20x", "runs/e9_tier1.json", "b4/e13_tier1.json", "discard", False),
         ("E14 re-propose arms", "runs/e9_tier1.json", "runs/e14_tier1.json", "keep", False),
         ("E15 type nodes from arms", "runs/e14_tier1.json", "runs/e15_tier1.json", "discard", False),
         ("E16 CFAR arms", "runs/e14_tier1.json", "runs/e16_tier1.json", "discard", False),
         ("E17 two alternations", "runs/e14_tier1.json", "runs/e17_tier1.json", "discard", False),
         ("E18 coarse channel", "b6/r0_tier1.json", "b6/e18_tier1.json", "discard", False),
         ("E19 + deep proposals", "b6/r0_tier1.json", "b6/e19_tier1.json", "discard", False),
         ("E20 novel coarse ridges", "b6/r0_tier1.json", "b6/e20_tier1.json", "keep", False),
         ("E20a deep edges off", "b6/r0_tier1.json", "b6/e20a_tier1.json", "keep", False),
         ("E21 ridges out of data term", "b7/r1_tier1.json", "b7/e21b_tier1.json", "discard", False))
TIER2 = (("E3", "runs/baseline/tier2.json", "runs/e3_tier2.json", "confirm"),
         ("E6", "runs/e3_tier2.json", "runs/e6_tier2.json", "confirm"),
         ("E9 (B3 re-baseline)", "runs/e6_tier2.json", "runs/e9_tier2.json", "confirm"),
         ("E10+E11", "runs/e9_tier2.json", "runs/e11_tier2.json", "regress"),
         ("E14", "runs/e9_tier2.json", "runs/e14_tier2.json", "confirm, later provisional"),
         ("E20 (deep on)", "runs/e9_tier2.json", "b6/e20_tier2.json", "regress"),
         ("E20a (deep off)", "runs/e9_tier2.json", "b6/e20_nodeep_tier2.json", "regress (re-judged)"),
         ("R1 final", "runs/e9_tier2.json", "final/dev_tier2_R1.json", "confirm"))
LOGGED = {
    'runs/baseline/tier1.json': 0.759238, 'runs/e1_tier1.json': 0.752183, 'runs/e2_tier1.json': 0.759045, 'runs/e3_tier1.json': 0.768662, 'runs/e4_tier1.json': 0.768156, 'runs/e5_tier1.json': 0.781836, 'runs/e6_tier1.json': 0.785308, 'runs/e7_tier1.json': 0.786362, 'runs/e8_tier1.json': 0.779753, 'runs/b3r_tier1.json': 0.785309, 'runs/e9_tier1.json': 0.785309, 'runs/e10_tier1.json': 0.787348, 'runs/e11_tier1.json': 0.788777, 'b4/e13_tier1.json': 0.786965, 'runs/e14_tier1.json': 0.788423, 'runs/e15_tier1.json': 0.788423, 'runs/e16_tier1.json': 0.787676, 'runs/e17_tier1.json': 0.78964, 'b6/r0_tier1.json': 0.785309, 'b6/e18_tier1.json': 0.793358, 'b6/e19_tier1.json': 0.791564, 'b6/e20_tier1.json': 0.788501, 'b6/e20a_tier1.json': 0.787735, 'b7/r1_tier1.json': 0.785309, 'b7/e21b_tier1.json': 0.78441}
CONTROLS = (("E9", "b4/controls_e9.json"), ("E14", "b4/controls_e14.json"), ("E18", "b6/controls_e18.json"),
            ("E19", "b6/controls_e19.json"), ("E20", "b6/controls_e20.json"))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def tier1(d: dict, kinds) -> tuple:
    """(composite, fast_composite, probe_score) of a tier-1 run over the rows of the given kinds."""
    crops = [r["composite"] for r in d["rows"] if r.get("kind") in kinds and "composite" in r]
    probes = [r["probe_composite"] for r in d["probe"]["rows"] if r.get("kind") in kinds]
    f, p = float(np.mean(crops)), float(np.mean(probes))
    return 0.5 * f + 0.5 * p, f, p


def _kind_only(path: str, kind: str, tmp: str) -> str:
    d = _load(path)
    d["rows"] = [r for r in d["rows"] if r.get("kind") == kind]
    out = os.path.join(tmp, path.replace(os.sep, "_"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    return out


def controls_score(path: str, kinds) -> float:
    return float(np.mean([r["probe_composite"] for r in _load(path)["rows"] if r.get("kind") in kinds]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=WORK)
    a = ap.parse_args(argv)
    W = lambda p: os.path.join(a.work, p)                                        # noqa: E731
    print("TIER 1 (rule: keep if composite >= parent + 0.002, or within 0.002 and simpler)")
    print(f"{'decision':30s} {'logged':8s} | {'both':>8s} {'rule':8s} {'check':>5s} | {'average':>8s} "
          f"{'crops':>8s} {'probes':>8s} {'rule':8s}")
    flips, other = [], []
    rule = lambda d, simpler: "keep" if d >= 0.002 or (abs(d) < 0.002 and simpler) else "discard"   # noqa: E731
    for lab, pa, pb, st, simpler in TIER1:
        if not (os.path.exists(W(pa)) and os.path.exists(W(pb))):
            print(f"{lab:30s} missing run")
            continue
        A, B = _load(W(pa)), _load(W(pb))
        ok = all(abs(X["summary"]["composite"] - LOGGED[q]) < 5e-7 for X, q in ((A, pa), (B, pb)))
        ok &= all(abs(tier1(X, BOTH)[0] - X["summary"]["composite"]) < 1e-6 for X in (A, B))
        both = tier1(B, BOTH)[0] - tier1(A, BOTH)[0]
        (ca, fa, qa), (cb, fb, qb) = tier1(A, AVG), tier1(B, AVG)
        d = cb - ca
        vb, va = rule(both, simpler), rule(d, simpler)
        mark = "  <-- flips" if va != vb else ("  (logged differs from the rule: not the image kind)"
                                                if vb != st.split()[0] else "")
        flips += [lab] if va != vb else []
        other += [lab] if va == vb and vb != st.split()[0] else []
        print(f"{lab:30s} {st:8s} | {both:+8.4f} {vb:8s} {'ok' if ok else 'XX':>5s} | {d:+8.4f} {fb - fa:+8.4f} "
              f"{qb - qa:+8.4f} {va:8s}{mark}")
    print("\nTIER 2 (paired per scene: confirmed if mean > 2 SE, no scene below -5e-4, no guard veto)")
    tmp = tempfile.mkdtemp()
    for lab, pa, pb, st in TIER2:
        if not (os.path.exists(W(pa)) and os.path.exists(W(pb))):
            print(f"{lab:22s} missing run")
            continue
        pboth = PD.paired(W(pa), W(pb))
        pavg = PD.paired(_kind_only(W(pa), "average", tmp), _kind_only(W(pb), "average", tmp))
        veto = [g["key"] for g in pavg["guards"] if g["veto"]]
        print(f"{lab:22s} logged {st:26s} | both {pboth['mean']:+.5f} t {pboth['t']:+6.2f} {pboth['verdict']:11s} | "
              f"average {pavg['mean']:+.5f} t {pavg['t']:+6.2f} up {pavg['up']} down {pavg['down']} "
              f"{pavg['verdict']}{'  veto ' + ','.join(veto) if veto else ''}")
    print("\nCONTROLS (mean probe_composite over the multi-seed controls)")
    for lab, p in CONTROLS:
        if os.path.exists(W(p)):
            print(f"{lab:5s} both {controls_score(W(p), BOTH):.4f}  average {controls_score(W(p), AVG):.4f}")
    print("\ntier-1 decisions that flip on the averages:", ", ".join(flips) or "none")
    print("logged differently from the rule on both kinds and on the averages (not the image kind):",
          ", ".join(other) or "none")


if __name__ == "__main__":
    main()
