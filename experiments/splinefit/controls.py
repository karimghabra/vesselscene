"""Multi-seed controls (a diagnostic; NOT part of the frozen tier-1 metric).

The frozen probe battery (probes.py) renders every stimulus with ONE imaging draw (SEED 11, common random
numbers), so one background feature (e.g. the illumination lump at the bottom-right frame edge) appears in
every probe, and a change that fixes that one feature is counted about six times (batch-4 review, finding 1).
This re-renders a few battery stimuli with OTHER imaging seeds (the negative control, a thin and a faint
line, a wide fork and a 45 deg crossing; the same geometry, optics and scoring as the battery) and reports
the pipeline's mean probe_composite per stimulus over the seeds, and the false length fitted on the empty
frames. A probe-only tier-1 gain counts as real only if it also shows here.

    python -m experiments.splinefit.controls [--pipeline MOD:FN] [--seeds 1 2 ...] [--only NAME ...]

The stimuli are cached under $SPLINEFIT_WORK/controls/<geometry>_s<seed>/ (generated on first use, never
committed). Truth is read only by the scorer (probes.score_probe).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from . import set_threads
from . import probes as PR
from .run_experiment import WORK, load_pipeline

CONTROL_DIR = os.path.join(WORK, "controls")
STIMULI = ("empty_average", "empty_frame", "line_d3", "line_d6_h0.25", "fork_wide", "cross_a45")
SEEDS = (1, 2, 3, 4, 5, 6, 7)


def _stimulus(name: str) -> dict:
    for st in PR.BATTERY:
        if st["name"] == name:
            return st
    raise KeyError(name)


def case_for(st: dict, seed: int):
    """The stimulus rendered with imaging seed `seed` (cached); None when vesselscene cannot export it."""
    from vesselscene import scene as SC
    from . import score as S
    folder = os.path.join(CONTROL_DIR, f"{st['geom']}_{'_'.join(f'{k}{v}' for k, v in sorted(st['args'].items()))}"
                          f"_f{st['focus']:g}_s{seed}")
    if not os.path.exists(os.path.join(folder, "scene.json")):
        g = PR.BUILDERS[st["geom"]](**st["args"])
        sc = SC.make_scene(seed, "healthy", shape=PR.SHAPE, graph=g, optics=PR.optics(st["focus"]), device="cpu",
                           frame_focus_offset=0.0)
        try:
            SC.save_scene(sc, folder, overlay=False, vesselmap_export=False)
        except ValueError:                      # a NaN in the truth export (seen once on an empty draw)
            return None
    return S.load_truth(folder, st["kind"])


def run(pipeline: str, seeds=SEEDS, only=STIMULI, verbose=True) -> dict:
    set_threads()
    fn = load_pipeline(pipeline)
    from . import score as S
    rows, h = [], hashlib.sha256()
    for name in only:
        st = _stimulus(name)
        for seed in seeds:
            case = case_for(st, seed)
            if case is None:
                continue
            t0 = time.perf_counter()
            out = fn(case["image"].copy(), case["valid"].copy())
            row = PR.score_probe(dict(st, name=f"{name}_s{seed}"), case, out)
            row.update(stimulus=name, seed=seed, seconds=time.perf_counter() - t0, digest=S.digest(out))
            h.update(row["digest"].encode())
            rows.append(row)
            if verbose:
                print(f"{name:16s} seed {seed}  composite {row['probe_composite']:.3f}  "
                      f"len {row['fitted_len_px']:6.1f}  junctions {row['junction_types_fit']}", flush=True)
    summ = {}
    for name in only:
        rr = [r for r in rows if r["stimulus"] == name]
        if rr:
            summ[name] = dict(mean=float(np.mean([r["probe_composite"] for r in rr])),
                              false_len=float(np.mean([r["fitted_len_px"] for r in rr])) if "empty" in name else None,
                              n=len(rr))
    score = float(np.mean([v["mean"] for v in summ.values()])) if summ else float("nan")
    return dict(controls_score=score, per_stimulus=summ, rows=rows, digest=h.hexdigest()[:16])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", default="experiments.splinefit.pipeline:annotate")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--only", nargs="*", default=list(STIMULI))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run(a.pipeline, a.seeds, a.only)
    for name, v in res["per_stimulus"].items():
        extra = f"  false_len {v['false_len']:.1f}" if v["false_len"] is not None else ""
        print(f"control_{name}: {v['mean']:.4f}  (n {v['n']}){extra}")
    print(f"controls_score: {res['controls_score']:.6f}")
    print(f"controls_digest: {res['digest']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
