"""Paired comparison of two run_experiment outputs (diagnostic; outside the frozen metric).

The tier-2 confirm rule of program.md (batch 6, the review's finding): a change is CONFIRMED only when the
paired per-image composite difference against the last confirmed run is positive and larger than twice its
standard error (mean +- SE, t = mean / SE, sign count); a difference inside that band is NON-INFERIOR
(not regressed), which keeps a simplification but not added code. For tier 1 it also prints the leave-one-out
range of the fast-crop mean, which shows whether a gain rests on one crop.

    python -m experiments.splinefit.paired A.json B.json [--key composite]   (B - A)
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _rows(path):
    d = json.load(open(path))
    return {(r.get("scene"), r.get("kind"), str(r.get("crop"))): r for r in d["rows"] if "composite" in r}


def paired(a: str, b: str, key: str = "composite") -> dict:
    ra, rb = _rows(a), _rows(b)
    ks = sorted(set(ra) & set(rb))
    d = np.array([rb[k][key] - ra[k][key] for k in ks], float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    loo = [float(np.delete(d, i).mean()) for i in range(n)] if n > 1 else []
    return dict(n=n, mean=float(d.mean()), se=se, t=float(d.mean() / se) if se > 0 else float("nan"),
                up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), loo_min=min(loo, default=np.nan),
                loo_max=max(loo, default=np.nan), confirmed=bool(d.mean() > 2 * se),
                per=[(k, float(x)) for k, x in zip(ks, d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", default="composite")
    ap.add_argument("--per", action="store_true")
    args = ap.parse_args()
    p = paired(args.a, args.b, args.key)
    if args.per:
        for k, x in p["per"]:
            print(f"{k[0]:28s} {k[1]:8s} {k[2]:22s} {x:+.4f}")
    print(f"paired {args.key} (B - A): n {p['n']}  mean {p['mean']:+.5f}  SE {p['se']:.5f}  t {p['t']:+.2f}  "
          f"up {p['up']} down {p['down']}  leave-one-out [{p['loo_min']:+.5f}, {p['loo_max']:+.5f}]  "
          f"confirmed(mean > 2 SE): {p['confirmed']}")


if __name__ == "__main__":
    main()
