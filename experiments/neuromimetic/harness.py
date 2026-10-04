"""Shared evaluation harness for vessel annotators on vesselscene scenes.

Every annotator under test implements one function

    annotate(image, valid) -> dict(
        polylines = [ (n, 2) float arrays, x then y, px ],       # one per traced edge (junction to junction)
        junctions = [ (x, y, type) ],                            # type in TYPES or 'branch' (3-way, direction unknown)
    )

image is the still as saved (float32 DN, NaN = no data for the average; uint16 DN for the frame, cast to
float32), valid the bool mask of finite pixels.  Nothing else from the scene folder may be read by an
annotator: the truth files are for scoring only.

Scores (per image):
  * truth.score_tracing at tol 3 px: centreline recall / precision / F1, junction recall / precision / F1 and
    exact type accuracy (bifurcation / confluence / pseudo-T / crossing / compound);
  * coarse type accuracy over the same matched pairs: '3-way' (bifurcation, confluence, pseudo-T, branch),
    'crossing', 'compound' -- bifurcation vs confluence needs the flow direction, which a still does not show;
  * crossing recall / precision: the observable junctions typed crossing against the traced ones typed crossing;
  * edge cover: for every observable image-graph edge at least one lambda long, the share of its length that
    the single best traced polyline covers (length-weighted mean; 1 = each truth edge is one traced edge), and
    pieces per edge (traced polylines covering at least
    20 % of it and at least lambda px; a neighbour touching it at a junction does not count);
  * runtime and a determinism digest of the output (two runs must give the same digest).

Usage:
    python -m experiments.neuromimetic.harness --annotator experiments.neuromimetic.baseline_hessian:annotate \
        --scenes DIR [DIR ...] --kinds average frame --out results.json [--repeat 2]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from vesselscene import truth as T

LAMBDA_PX = T.LAMBDA_PX
TOL_PX = 3.0
COARSE = {"bifurcation": "3-way", "confluence": "3-way", "pseudo-T": "3-way", "branch": "3-way",
          "crossing": "crossing", "compound": "compound"}


def load_case(folder: str, kind: str) -> dict:
    import tifffile
    img = tifffile.imread(os.path.join(folder, f"still_{kind}.tif")).astype(np.float32)
    valid = np.isfinite(img)
    obs = T.load_observable(os.path.join(folder, f"observable_{kind}.json"))
    prof = T.load_profiles(os.path.join(folder, f"complete_visibility_{kind}.npz"))
    return dict(folder=folder, kind=kind, image=img, valid=valid, obs=obs, prof=prof)


def _resample(P, step=1.0):
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) < 2:
        return P
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if d[-1] <= 0:
        return P[:1]
    s = np.arange(0, d[-1] + 1e-9, step)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1)


def match_junctions(obs, traced_junctions, jt=LAMBDA_PX):
    """score_tracing's greedy matching, returning the matched (traced, truth) pairs."""
    js = obs["junctions"]
    used, pairs = set(), []
    for t in traced_junctions:
        x, y = float(t[0]), float(t[1])
        best, bd = None, None
        for j in js:
            if j["id"] in used:
                continue
            dd = math.hypot(j["x"] - x, j["y"] - y)
            if dd <= max(j["radius"], jt) and (bd is None or dd < bd):
                best, bd = j, dd
        if best is not None:
            used.add(best["id"])
            pairs.append((t, best))
    return pairs


def edge_cover(obs, polylines, tol=TOL_PX):
    pts, ids = [], []
    for i, P in enumerate(polylines):
        Q = _resample(P, 0.5)
        if len(Q):
            pts.append(Q)
            ids.append(np.full(len(Q), i))
    if not pts:
        return dict(edge_cover=0.0, pieces_per_edge=float("nan"), n_truth_edges=0)
    pts = np.concatenate(pts)
    ids = np.concatenate(ids)
    tree = cKDTree(pts)
    num = den = 0.0
    pieces = []
    n = 0
    for e in obs["graph"]["edges"]:
        Q = _resample(e["xy"], 1.0)
        if len(Q) < LAMBDA_PX:
            continue
        n += 1
        near = tree.query_ball_point(Q, tol)
        cnt = {}
        for k, lst in enumerate(near):
            for pid in set(ids[lst].tolist()):
                cnt[pid] = cnt.get(pid, 0) + 1
        best = max(cnt.values()) if cnt else 0
        num += best
        den += len(Q)
        pieces.append(max(1, sum(1 for v in cnt.values() if v >= max(0.2 * len(Q), LAMBDA_PX))))
    return dict(edge_cover=num / den if den else float("nan"),
                pieces_per_edge=float(np.mean(pieces)) if pieces else float("nan"), n_truth_edges=n)


def score(case, out) -> dict:
    obs, prof = case["obs"], case["prof"]
    polys = [np.asarray(P, float).reshape(-1, 2) for P in out.get("polylines", []) if len(P)]
    juncs = [tuple(j) for j in out.get("junctions", [])]
    s = T.score_tracing(obs, polys, juncs, prof=prof, tol_px=TOL_PX)
    pairs = match_junctions(obs, juncs)
    typed = [(t, j) for t, j in pairs if len(t) > 2]
    s["junction_type_accuracy_coarse"] = (float(np.mean([COARSE.get(t[2], "?") == COARSE[j["type_observable"]]
                                                         for t, j in typed])) if typed else float("nan"))
    tc = [j for j in juncs if len(j) > 2 and j[2] == "crossing"]
    oc = [j for j in obs["junctions"] if j["type_observable"] == "crossing"]
    mc = sum(1 for t, j in typed if t[2] == "crossing" and j["type_observable"] == "crossing")
    s["crossing_recall"] = mc / len(oc) if oc else float("nan")
    s["crossing_precision"] = mc / len(tc) if tc else float("nan")
    s.update(edge_cover(obs, polys))
    s["n_polylines"] = len(polys)
    return s


def digest(out) -> str:
    h = hashlib.sha256()
    for P in out.get("polylines", []):
        h.update(np.round(np.asarray(P, np.float64), 3).tobytes())
    for j in out.get("junctions", []):
        h.update(repr((round(float(j[0]), 3), round(float(j[1]), 3), j[2] if len(j) > 2 else "")).encode())
    return h.hexdigest()[:16]


def run(annotator: str, scenes, kinds, repeat=1) -> list:
    mod, fn = annotator.split(":")
    f = getattr(importlib.import_module(mod), fn)
    rows = []
    for folder in scenes:
        for kind in kinds:
            case = load_case(folder, kind)
            digests, times, out = [], [], None
            for _ in range(repeat):
                t0 = time.perf_counter()
                out = f(case["image"].copy(), case["valid"].copy())
                times.append(time.perf_counter() - t0)
                digests.append(digest(out))
            s = score(case, out)
            s.update(scene=os.path.basename(os.path.normpath(folder)), kind=kind, seconds=min(times),
                     digest=digests[0], deterministic=len(set(digests)) == 1)
            rows.append(s)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return rows


KEYS = ["centreline_recall", "centreline_precision", "centreline_f1", "junction_recall", "junction_precision",
        "junction_f1", "junction_type_accuracy", "junction_type_accuracy_coarse", "crossing_recall",
        "crossing_precision", "edge_cover", "pieces_per_edge", "seconds"]


def summarise(rows) -> dict:
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == kind]
        out[kind] = {k: float(np.nanmean([r.get(k, np.nan) for r in rr])) for k in KEYS}
        out[kind]["n_images"] = len(rr)
        out[kind]["all_deterministic"] = all(r["deterministic"] for r in rr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["average", "frame"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = run(a.annotator, a.scenes, a.kinds, a.repeat)
    summ = summarise(rows)
    print(json.dumps(summ, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(annotator=a.annotator, rows=rows, summary=summ), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
