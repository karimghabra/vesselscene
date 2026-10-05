"""Fresh-process determinism check (a diagnostic; not frozen).

run_experiment's tier-2 'deterministic' flag compares two back-to-back runs inside one process, which cannot
see a dependence on the run's history (e.g. a JIT cache filled by earlier images). This re-runs chosen rows
of a saved run_experiment JSON, each in its OWN fresh Python process, and compares their digests with the
in-run digests.

    python -m experiments.splinefit.determinism [--json PATH] [--rows 0 3 ...]

Prints one line per row ('same' / 'DIFFERENT') and exits 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .run_experiment import WORK

_CHILD = """
import json, os, sys
from experiments.splinefit import set_threads, fastset as FS, score as S
from experiments.splinefit.run_experiment import load_pipeline
set_threads(2)
scene, kind, crop, pipe = json.loads(sys.argv[1])
case = S.load_truth(os.path.join(FS.DEV_DIR, scene), kind, crop)
out = load_pipeline(pipe)(case["image"].copy(), case["valid"].copy())
print("DIGEST", S.digest(out))
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=os.path.join(WORK, "runs", "last_tier1.json"))
    ap.add_argument("--rows", type=int, nargs="*", help="row indices (default: all)")
    a = ap.parse_args(argv)
    res = json.load(open(a.json, encoding="utf-8"))
    rows = res["rows"]
    idx = a.rows if a.rows else range(len(rows))
    env = dict(os.environ, OMP_NUM_THREADS="2")
    bad = 0
    for i in idx:
        r = rows[i]
        arg = json.dumps([r["scene"], r["kind"], r["crop"], res["pipeline"]])
        p = subprocess.run([sys.executable, "-c", _CHILD, arg], capture_output=True, text=True, env=env,
                           cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        got = [ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("DIGEST")]
        got = got[0] if got else "crash:" + p.stderr[-200:]
        same = got == r["digest"]
        bad += not same
        print(f"{i} {r['scene']} {r['kind']} {r['crop']}: in-run {r['digest']} fresh {got} "
              f"{'same' if same else 'DIFFERENT'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
