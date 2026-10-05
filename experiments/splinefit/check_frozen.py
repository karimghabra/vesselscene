"""Check that the frozen evaluator is unchanged: every FROZEN file's git blob hash and the dev set's digest
against frozen.json (written once at the freeze, program.md section 3).

    python -m experiments.splinefit.check_frozen            # exit 1 on any mismatch
    python -m experiments.splinefit.check_frozen --write    # only at the freeze or a logged re-baseline

The dev-set digest is the sha256 over the sorted (folder, file name, file sha256) of every file in the dev
scene folders (fastset.DEV_DIR).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = ("score.py", "run_experiment.py", "probes.py", "probe_battery.json", "fastset.py", "fastset.json",
          "evaluate_fit.py", "oracle.py", "check_frozen.py", "tests/test_score.py", "tests/test_probes.py",
          "tests/test_loop.py")
FROZEN_JSON = os.path.join(HERE, "frozen.json")


def blob_hash(path: str) -> str:
    return subprocess.run(["git", "hash-object", path], capture_output=True, text=True, check=True).stdout.strip()


def dev_digest() -> str:
    from experiments.splinefit import fastset as FS
    h = hashlib.sha256()
    for folder in FS.dev_scenes():
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    h.update(f"{os.path.basename(folder)}/{fn}:{hashlib.sha256(fh.read()).hexdigest()}".encode())
    return h.hexdigest()[:16]


def current() -> dict:
    return dict(files={f: blob_hash(os.path.join(HERE, f)) for f in FROZEN}, dev_set=dev_digest())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cur = current()
    if a.write:
        with open(FROZEN_JSON, "w", encoding="utf-8") as fh:
            json.dump(cur, fh, indent=1)
        print(json.dumps(cur, indent=1))
        return 0
    with open(FROZEN_JSON, encoding="utf-8") as fh:
        ref = json.load(fh)
    bad = [f for f in FROZEN if cur["files"][f] != ref["files"].get(f)]
    if cur["dev_set"] != ref["dev_set"]:
        bad.append("dev set")
    for f in FROZEN:
        print(f"{'OK ' if f not in bad else 'BAD'} {cur['files'][f]}  {f}")
    print(f"{'OK ' if 'dev set' not in bad else 'BAD'} {cur['dev_set']}  dev set")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
