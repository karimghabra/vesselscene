r"""One experiment: run a pipeline on a frozen tier and print its score (autoresearch's 'train + evaluate').

Frozen after the Instrument phase together with score.py and fastset.json: an experiment changes the
pipeline, never this file.

    python -m experiments.splinefit.run_experiment [--tier 1|2] [--pipeline module:function] [--repeat N]
        [--out PATH]

Tier 1 (the loop's tier; budget TIER1_BUDGET_S = 6 min, hard limit TIER1_LIMIT_S = 10 min of wall clock,
everything included, past which the run is a crash): the probe battery (``experiments.splinefit.probes.
run_probes(annotate, deadline) -> dict(probe_score, rows, tuning, digest)``, deadline a time.monotonic()
value), then the fast crops of fastset.json, each run once.  The tier-1 score is

    composite = 0.5 * fast_composite (mean composite over the crops) + 0.5 * probe_score.

Tier 2 (the confirmation tier, limit --limit, default TIER2_LIMIT_S): every dev image (fastset.dev_scenes,
both kinds), each run --repeat times (default 2); the digests must agree (deterministic).  Its composite is
the mean composite over the images (no probes).

Each run calls ``annotate(image, valid)`` (score.py's pipeline contract) with copies of the image (or crop)
and its valid mask only, then scores it with score.score(case, out, audit=True).  The pipeline's wall clock
counts against the limit; when it is exceeded the run is a crash.  Output, grep-able, at the end:

    status: ok | crash
    composite: X           the tier's score (higher is better): tier 1 0.5 * fast_composite + 0.5 * probe_score,
                           tier 2 the mean composite over the images
    fast_composite: X      tier 1: mean composite over the crops
    graph: X               mean graph_score
    pos: X, width: X       mean geometry scores
    explained: X           mean explained (raw, against OD_obs), explained_junction: X likewise
    probe_score: X         tier 1: the probe battery's mean probe_composite
    probe_<sweep>: X       tier 1: each sweep's threshold (probes.SWEEPS), probe_mean_<sweep>: its mean
    seconds: X             wall clock of the whole run
    digest: XXXXXXXX       sha256 of the per-run digests (score.digest) in order: a sham change must reproduce it
    deterministic: True|False   (tier 2)

and a JSON of every row (all scalar scores, scene, kind, crop, seconds, digest; probe rows) to --out (default
$SPLINEFIT_WORK/runs/last_tier<N>.json).  On a crash: 'status: crash', 'error: ...', the partial JSON, and
exit status 1.  Threads: 2 for every numeric library, torch deterministic (the machine is shared).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import hashlib                                      # noqa: E402
import importlib                                    # noqa: E402
import json                                         # noqa: E402
import signal                                       # noqa: E402
import sys                                          # noqa: E402
import time                                         # noqa: E402
import traceback                                    # noqa: E402

import numpy as np                                  # noqa: E402

TIER1_BUDGET_S = 360.0          # the loop's time budget (the fast set and battery are sized to it)
TIER1_LIMIT_S = 600.0           # past this a tier-1 run is a crash
TIER2_LIMIT_S = 5400.0
DEFAULT_PIPELINE = "experiments.splinefit.pipeline:annotate"
WORK = os.environ.get("SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/"
                                        "scratchpad/work/splinefit")
SUMMARY = ("composite", "graph", "pos", "width", "explained", "explained_junction")
ROW_KEY = dict(graph="graph_score")


class Timeout(BaseException):
    """Not an Exception, so a pipeline's `except Exception` cannot swallow it (and it re-fires every second)."""


def _alarm(signum, frame):
    raise Timeout("wall-clock limit exceeded")


def _scalars(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, dict) and k in ("type_confusion",):
            out[k] = v
    return out


def load_pipeline(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def _run_one(fn, case, repeat: int):
    """The pipeline on one case `repeat` times: (last output, digests, min seconds)."""
    from .score import digest
    digests, times, out = [], [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(case["image"].copy(), case["valid"].copy())
        times.append(time.perf_counter() - t0)
        digests.append(digest(out))
    return out, digests, min(times)


def tier_cases(tier: int):
    """(folder, kind, crop) of every run of a tier."""
    from . import fastset as FS
    if tier == 1:
        return [(os.path.join(FS.DEV_DIR, c["scene"]), c["kind"], c["crop"]) for c in FS.load()]
    return [(f, k, None) for f in FS.dev_scenes() for k in FS.KINDS]


def summarise(rows: list, probe: dict | None) -> dict:
    s = {}
    for k in SUMMARY:
        v = [r.get(ROW_KEY.get(k, k), np.nan) for r in rows]
        v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
        s[k] = float(np.mean(v)) if v else float("nan")
    if probe is not None:
        s["probe_score"] = float(probe.get("probe_score", float("nan")))
        s["fast_composite"] = s["composite"]
        s["composite"] = 0.5 * s["fast_composite"] + 0.5 * s["probe_score"]
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r.get("digest")).encode())
    if probe is not None and probe.get("digest"):
        h.update(str(probe["digest"]).encode())
    s["digest"] = h.hexdigest()[:16]
    return s


def run(tier: int, pipeline: str, limit: float, verbose: bool = True, repeat: int | None = None) -> dict:
    from . import set_threads
    from . import score as S
    set_threads(2)
    t_start = time.monotonic()
    deadline = t_start + limit
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit, 1.0)
    res = dict(tier=tier, pipeline=pipeline, limit_s=limit, rows=[], probe=None, status="ok", error=None)
    try:
        fn = load_pipeline(pipeline)
        if tier == 1:
            try:
                probes = importlib.import_module("experiments.splinefit.probes")
            except ModuleNotFoundError as exc:
                if exc.name != "experiments.splinefit.probes":
                    raise
                probes = None
            if probes is not None and hasattr(probes, "run_probes"):
                res["probe"] = probes.run_probes(fn, deadline)
        repeat = repeat or (1 if tier == 1 else 2)
        for folder, kind, crop in tier_cases(tier):
            case = S.load_truth(folder, kind, crop)
            out, digests, secs = _run_one(fn, case, repeat)
            row = _scalars(S.score(case, out, audit=True))
            row.update(scene=os.path.basename(folder), kind=kind, crop=crop, seconds=secs, digest=digests[0],
                       deterministic=(len(set(digests)) == 1) if repeat > 1 else None,
                       **{k: v for k, v in _scalars(out).items() if k not in row})
            res["rows"].append(row)
            if verbose:
                print("row: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                            if k in ("scene", "kind", "crop", "composite", "graph_score", "pos",
                                                     "width", "explained", "explained_junction", "seconds",
                                                     "digest", "deterministic")}), flush=True)
    except BaseException as exc:                      # a crash (incl. the timeout) is a result, not an abort
        if isinstance(exc, KeyboardInterrupt):
            raise
        res["status"] = "crash"
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    res["seconds"] = time.monotonic() - t_start
    res["summary"] = summarise(res["rows"], res["probe"])
    if tier == 2:
        det = [r["deterministic"] for r in res["rows"] if r["deterministic"] is not None]
        res["summary"]["deterministic"] = bool(det) and all(det)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", type=int, choices=(1, 2), default=1)
    ap.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    ap.add_argument("--limit", type=float, default=TIER2_LIMIT_S, help="tier 2 only (tier 1 is fixed at 600 s)")
    ap.add_argument("--repeat", type=int, help="runs per image (default: tier 1 once, tier 2 twice)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    limit = TIER1_LIMIT_S if a.tier == 1 else a.limit
    res = run(a.tier, a.pipeline, limit, repeat=a.repeat)
    out = a.out or os.path.join(WORK, "runs", f"last_tier{a.tier}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    s = res["summary"]
    print(f"status: {res['status']}")
    if res["status"] != "ok":
        print(f"error: {res['error']}")
        print(res.get("traceback", ""), file=sys.stderr)
    for k in SUMMARY + ("fast_composite", "probe_score"):
        if k in s:
            print(f"{k}: {s[k]:.6f}")
    if res.get("probe"):
        tu = res["probe"].get("tuning", {})
        for sw, v in tu.get("thresholds", {}).items():
            print(f"probe_{sw}: {v}")
        for sw, v in tu.get("sweep_mean", {}).items():
            print(f"probe_mean_{sw}: {v:.4f}")
    print(f"seconds: {res['seconds']:.1f}")
    print(f"digest: {s['digest']}")
    if "deterministic" in s:
        print(f"deterministic: {s['deterministic']}")
    print(f"rows: {len(res['rows'])}  json: {out}")
    if res["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
