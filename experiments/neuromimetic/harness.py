"""Shared evaluation harness for vessel annotators on vesselscene scenes.

Every annotator under test implements one function

    annotate(image, valid) -> dict(
        polylines = [ (n, 2) float arrays, x then y, px ],       # one per traced edge (junction to junction)
        junctions = [ (x, y, type) ],                            # type in TYPES or 'branch' (3-way, direction unknown)
        ...                                                      # optional extra scalars (build_seconds, config, ...)
    )

image is the still as saved (float32 DN, NaN = no data for the average; uint16 DN for the frame, cast to
float32), valid the bool mask of finite pixels.  Nothing else from the scene folder may be read by an
annotator: the truth files are for scoring only.  A junction given as (x, y) without a type counts as matched
but wrongly typed.  Polylines are cut at every traced junction they pass within TOL_PX before the edge scores,
so running a polyline on through a junction or cutting it there is the same tracing (the truth's edges are
cut at its junctions).  Extra scalar or dict entries of the output are copied into the result row.

Scores (per image):
  * truth.score_tracing at tol 3 px: centreline recall / precision / F1, junction recall / precision / F1 and
    exact type accuracy (bifurcation / confluence / pseudo-T / crossing / compound).  A traced junction matches
    within max(radius, lambda), and the pathologic radii reach 60-160 px, so junction recall is cheap: quote it
    only next to precision, and lead with F1;
  * junction recall / precision / F1 'strict': the same matching with the radius capped at STRICT_RADIUS_PX;
  * centreline precision 'dedup': over the distinct 1 px cells the tracing touches, so duplicated polylines
    do not raise it;
  * coarse type accuracy over the same matched pairs: '3-way' (bifurcation, confluence, pseudo-T, branch),
    'crossing', 'compound' -- bifurcation vs confluence needs the flow direction, which a still does not show;
  * typing references: the accuracy of a constant label, the image's majority observable type, on the same
    matched pairs (exact and coarse), the balanced accuracy (mean per-truth-class recall over the matched pairs,
    exact and coarse; a constant label scores 1 / number of classes) and the confusion counts.  The truth is
    about half 'compound', so a typing rule is competent only where it beats the majority reference;
  * crossing recall / precision: the observable junctions typed crossing against the traced ones typed
    crossing (traced crossings on a don't-care junction are left out, as for junction precision);
  * edge cover: for every observable image-graph edge at least one lambda long, the share of its length that
    the single best traced polyline covers (length-weighted mean; 1 = each truth edge is one traced edge), and
    pieces per edge (traced polylines covering at least 20 % of it and at least lambda px; a neighbour touching
    it at a junction does not count), which catches over-segmentation;
  * polyline purity, which catches under-segmentation: the share of the traced points near a truth edge that
    lie on their polyline's best truth edge (length-weighted; points within 2 TOL_PX of a truth edge end are
    left out, being shared by the edges meeting there);
  * runtime and a determinism digest of the output (with --repeat 2 or more, every run must give the same
    digest; with one run determinism is not tested and is reported as None).

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
STRICT_RADIUS_PX = 2 * LAMBDA_PX
TYPES = ("bifurcation", "confluence", "pseudo-T", "crossing", "compound")
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


def match_junctions(obs, traced_junctions, jt=LAMBDA_PX, cap=None, full=False):
    """score_tracing's greedy matching, returning the matched (traced, truth) pairs; with full=True also the
    unmatched traced junctions on a don't-care junction (excused) and the rest (false positives).  cap limits
    the truth radius (the strict score)."""
    js = obs["junctions"]
    dcj = obs.get("dont_care_junctions", [])
    used, pairs, excused, fp = set(), [], [], []
    for t in traced_junctions:
        x, y = float(t[0]), float(t[1])
        best, bd = None, None
        for j in js:
            if j["id"] in used:
                continue
            dd = math.hypot(j["x"] - x, j["y"] - y)
            r = j["radius"] if cap is None else min(j["radius"], cap)
            if dd <= max(r, jt) and (bd is None or dd < bd):
                best, bd = j, dd
        if best is not None:
            used.add(best["id"])
            pairs.append((t, best))
        elif any(math.hypot(d["x"] - x, d["y"] - y) <= max(d["radius"], jt) for d in dcj):
            excused.append(t)
        else:
            fp.append(t)
    return (pairs, excused, fp) if full else pairs


def split_at_junctions(polylines, junctions, tol=TOL_PX):
    """Cut every polyline where it passes within tol of a traced junction (at its closest point there), so
    that a polyline running on through a junction counts like one cut there."""
    J = np.array([[float(j[0]), float(j[1])] for j in junctions], float).reshape(-1, 2)
    if not len(J):
        return list(polylines)
    jt = cKDTree(J)
    out = []
    for P in polylines:
        Q = _resample(P, 0.5)
        if len(Q) < 3:
            out.append(Q)
            continue
        d, _ = jt.query(Q)
        near = d <= tol
        cuts = []
        for run in np.split(np.arange(len(Q)), np.flatnonzero(np.diff(near.astype(np.int8))) + 1):
            if near[run[0]]:
                k = int(run[np.argmin(d[run])])
                if 0 < k < len(Q) - 1:
                    cuts.append(k)
        for a, b in zip([0] + cuts, cuts + [len(Q) - 1]):
            out.append(Q[a:b + 1])
    return out


def edge_cover(obs, polylines, tol=TOL_PX):
    pts, ids = [], []
    for i, P in enumerate(polylines):
        Q = _resample(P, 0.5)
        if len(Q):
            pts.append(Q)
            ids.append(np.full(len(Q), i))
    if not pts:
        return dict(edge_cover=0.0, pieces_per_edge=float("nan"), polyline_purity=float("nan"), n_truth_edges=0)
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
    # purity: each traced point near a truth edge (away from the edge ends) votes for its nearest edge
    E = [_resample(e["xy"], 1.0) for e in obs["graph"]["edges"]]
    E = [(k, Q) for k, Q in enumerate(E) if len(Q)]
    pure = float("nan")
    if E:
        ep = np.concatenate([Q for _, Q in E])
        eid = np.concatenate([np.full(len(Q), k) for k, Q in E])
        ends = np.concatenate([Q[[0, -1]] for _, Q in E])
        d, nn = cKDTree(ep).query(pts)
        keep = (d <= tol) & (cKDTree(ends).query(pts)[0] > 2 * tol)
        hit = hits = 0
        for pid in np.unique(ids[keep]):
            m = keep & (ids == pid)
            hit += np.bincount(eid[nn[m]]).max()
            hits += m.sum()
        pure = hit / hits if hits else float("nan")
    return dict(edge_cover=num / den if den else float("nan"),
                pieces_per_edge=float(np.mean(pieces)) if pieces else float("nan"),
                polyline_purity=float(pure), n_truth_edges=n)


def _typing(pairs, obs) -> dict:
    """Coarse type accuracy over the matched pairs (an untyped traced junction is wrong, as in score_tracing's
    exact accuracy), the majority-label and balanced-accuracy references (exact and coarse) and the confusion
    counts 'truth>traced'."""
    out = {}
    tr = [j["type_observable"] for _, j in pairs]
    pr = [t[2] if len(t) > 2 else "?" for t, _ in pairs]
    allt = [j["type_observable"] for j in obs["junctions"]]
    for sfx, f, order in (("", lambda x: x, TYPES), ("_coarse", lambda x: COARSE.get(x, "?"),
                                                      ("3-way", "crossing", "compound"))):
        a, b = [f(x) for x in tr], [f(x) for x in pr]
        if sfx:
            out["junction_type_accuracy_coarse"] = float(np.mean([x == y for x, y in zip(a, b)])) if a else float("nan")
        if allt:      # the image's most frequent observable type (ties: first in order) as a constant label
            maj = max(order, key=[f(x) for x in allt].count)
            out[f"junction_type_majority{sfx}"] = float(np.mean([x == maj for x in a])) if a else float("nan")
            out[f"junction_type_majority_label{sfx}"] = maj
        rec = [np.mean([y == c for x, y in zip(a, b) if x == c]) for c in sorted(set(a))]
        out[f"junction_type_balanced{sfx}"] = float(np.mean(rec)) if rec else float("nan")
    conf = {}
    for x, y in zip(tr, pr):
        conf[f"{x}>{y}"] = conf.get(f"{x}>{y}", 0) + 1
    out["type_confusion"] = dict(sorted(conf.items()))
    return out


def score(case, out) -> dict:
    obs, prof = case["obs"], case["prof"]
    polys = [np.asarray(P, float).reshape(-1, 2) for P in out.get("polylines", []) if len(P)]
    juncs = [tuple(j) for j in out.get("junctions", [])]
    s = T.score_tracing(obs, polys, juncs, prof=prof, tol_px=TOL_PX)
    # precision over the distinct 1 px cells traced (duplicates once)
    cells = np.unique(np.round(T._pts_of_polylines(polys)), axis=0) if polys else np.zeros((0, 2))
    s["centreline_precision_dedup"] = T.score_tracing(obs, [c[None] for c in cells], prof=prof,
                                                      tol_px=TOL_PX)["centreline_precision"]
    pairs, excused, fp = match_junctions(obs, juncs, full=True)
    s.update(_typing(pairs, obs))
    tc = sum(1 for j in [t for t, _ in pairs] + fp if len(j) > 2 and j[2] == "crossing")
    oc = [j for j in obs["junctions"] if j["type_observable"] == "crossing"]
    mc = sum(1 for t, j in pairs if len(t) > 2 and t[2] == "crossing" and j["type_observable"] == "crossing")
    s["crossing_recall"] = mc / len(oc) if oc else float("nan")
    s["crossing_precision"] = mc / tc if tc else float("nan")
    sp, _, sfp = match_junctions(obs, juncs, cap=STRICT_RADIUS_PX, full=True)
    nj = len(obs["junctions"])
    jr = len(sp) / nj if nj else float("nan")
    jp = len(sp) / (len(sp) + len(sfp)) if sp or sfp else float("nan")
    s.update(junction_recall_strict=jr, junction_precision_strict=jp,
             junction_f1_strict=2 * jr * jp / (jr + jp) if nj and (sp or sfp) and jr + jp > 0 else 0.0)
    s.update(edge_cover(obs, split_at_junctions(polys, juncs)))
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
            for k, v in out.items():                  # the annotator's own scalars (build_seconds, config, ...)
                if k not in ("polylines", "junctions") and k not in s and isinstance(v, (int, float, str, bool, dict)):
                    s[k] = v
            s.update(scene=os.path.basename(os.path.normpath(folder)), kind=kind, seconds=min(times), repeat=repeat,
                     digest=digests[0], deterministic=(len(set(digests)) == 1) if repeat >= 2 else None)
            rows.append(s)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return rows


KEYS = ["centreline_recall", "centreline_precision", "centreline_f1", "centreline_precision_dedup",
        "junction_recall", "junction_precision", "junction_f1", "junction_recall_strict",
        "junction_precision_strict", "junction_f1_strict", "junction_type_accuracy", "junction_type_majority",
        "junction_type_balanced", "junction_type_accuracy_coarse", "junction_type_majority_coarse",
        "junction_type_balanced_coarse", "crossing_recall", "crossing_precision", "edge_cover", "pieces_per_edge",
        "polyline_purity", "seconds", "build_seconds"]


def summarise(rows) -> dict:
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == kind]
        out[kind] = {k: float(np.nanmean([r.get(k, np.nan) for r in rr])) for k in KEYS
                     if any(np.isfinite(r.get(k, np.nan)) for r in rr)}
        out[kind]["n_images"] = len(rr)
        det = [r["deterministic"] for r in rr if r.get("deterministic") is not None]
        out[kind]["all_deterministic"] = all(det) if det else None       # None: not tested (one run each)
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
