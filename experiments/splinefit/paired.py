"""Paired comparison of two run_experiment outputs (diagnostic; outside the frozen metric).

The tier-2 confirm rule of program.md. The unit of replication is the SCENE, not the image: the average and
the frame of one scene share every vessel, so their differences are correlated, and counting them as two
images overstated n (batch-7 review: E20a's t 2.53 over 10 images is t 1.83 over 5 scenes). Every row's
difference B - A is therefore averaged over the scene's kinds (and crops) first, and the test runs on the
per-scene means (n = number of scenes):

- CONFIRMED: mean > 2 SE, no scene below -MATERIAL, and no guard vetoed;
- NON-INFERIOR (mean within +-2 SE): keeps a simplification, not added code (PROVISIONAL);
- REGRESSED: mean < -2 SE.

'up' / 'down' count only scenes whose difference exceeds the materiality threshold (MATERIAL = 5e-4 of the
composite; the pipeline is deterministic, so any non-zero change counts as numerically up or down, even a
negligible one). The leave-one-out range (one scene left out) shows whether a gain rests on one scene.

GUARDS (batch-7 review): errors the composite does not see. A change can gain explained OD by letting a
neighbouring vessel absorb OD that nothing in the model causes; that shows up as contrast and blur errors.
Each guard is lower-is-better (a bias as its absolute value); a guard whose mean over scenes worsens by more
than its threshold (GUARDS), or that worsens on any one scene by more than twice it, VETOES a confirmation,
whatever the composite does (the hard scenes are few, so a cost on one of them must not average away).
The thresholds were set in batch 7 with E14, E20 and E20a in view (calibrated on them, not tested by them).

    python -m experiments.splinefit.paired A.json B.json [--key composite] [--per] [--by image]   (B - A)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

MATERIAL = 5e-4
# guard: (row key, absolute value?, veto threshold of the per-scene mean worsening)
GUARDS = (("blur_err_px", False, 0.02), ("contrast_rel_err", False, 0.005), ("contrast_bias", True, 0.02),
          ("blur_bias_px", True, 0.05), ("width_rel_err", False, 0.005), ("width_bias", True, 0.02))


def _rows(path):
    d = json.load(open(path))
    return {(r.get("scene"), r.get("kind"), str(r.get("crop"))): r for r in d["rows"] if "composite" in r}


def _diffs(ra, rb, key, by="scene", absval=False):
    """Per-unit mean differences B - A of key (unit: the scene, or every row with by='image')."""
    f = abs if absval else (lambda v: v)
    units = {}
    for k in sorted(set(ra) & set(rb)):
        a, b = ra[k].get(key), rb[k].get(key)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        units.setdefault(k[0] if by == "scene" else k, []).append(f(b) - f(a))
    ks = sorted(units, key=str)
    return ks, np.array([np.mean(units[u]) for u in ks], float)


def _stats(d):
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    loo = [float(np.delete(d, i).mean()) for i in range(n)] if n > 1 else []
    return dict(n=n, mean=float(d.mean()) if n else float("nan"), se=se,
                t=float(d.mean() / se) if n > 1 and se > 0 else float("nan"),
                loo_min=min(loo, default=np.nan), loo_max=max(loo, default=np.nan))


def paired(a: str, b: str, key: str = "composite", by: str = "scene", material: float = MATERIAL) -> dict:
    ra, rb = _rows(a), _rows(b)
    ks, d = _diffs(ra, rb, key, by)
    st = _stats(d)
    guards = []
    for g, absval, thr in GUARDS:
        _, dg = _diffs(ra, rb, g, by, absval)
        if len(dg):
            sg = _stats(dg)
            guards.append(dict(key=f"|{g}|" if absval else g, thr=thr, worst=float(dg.max()),
                               veto=bool(sg["mean"] > thr or dg.max() > 2 * thr), **sg))
    veto = any(g["veto"] for g in guards)
    big = st["mean"] > 2 * st["se"]
    verdict = ("confirmed" if big and not veto and d.min() >= -material else
               "regressed" if st["mean"] < -2 * st["se"] else "provisional")
    return dict(st, up=int((d > material).sum()), down=int((d < -material).sum()), material=material,
                by=by, guards=guards, veto=veto, confirmed=verdict == "confirmed", verdict=verdict,
                per=list(zip(ks, d.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", default="composite")
    ap.add_argument("--by", default="scene", choices=("scene", "image"))
    ap.add_argument("--per", action="store_true")
    args = ap.parse_args()
    p = paired(args.a, args.b, args.key, args.by)
    if args.per:
        for k, x in p["per"]:
            print(f"{str(k):60s} {x:+.5f}")
    print(f"paired {args.key} (B - A) by {p['by']}: n {p['n']}  mean {p['mean']:+.5f}  SE {p['se']:.5f}  "
          f"t {p['t']:+.2f}  up {p['up']} down {p['down']} (|d| > {p['material']:g})  leave-one-out "
          f"[{p['loo_min']:+.5f}, {p['loo_max']:+.5f}]")
    for g in p["guards"]:
        print(f"  guard {g['key']:18s} mean {g['mean']:+.5f}  SE {g['se']:.5f}  t {g['t']:+.2f}  "
              f"worst scene {g['worst']:+.4f}  (veto: mean > +{g['thr']:g} or a scene > +{2 * g['thr']:g})"
              f"{'  VETO' if g['veto'] else ''}")
    print(f"verdict: {p['verdict']}  (confirmed: mean > 2 SE, no scene below -{p['material']:g}, no guard veto)")


if __name__ == "__main__":
    main()
