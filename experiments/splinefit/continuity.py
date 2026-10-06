r"""Continuity and mask diagnostics on the development images (batch 8; a diagnostic, outside the frozen metric).

    python -m experiments.splinefit.continuity [--pipeline MOD:FN] [--kind average] [--out JSON] [SCENE_DIR ...]

Runs the pipeline (annotate_cfg with a debug dict, or a variant module exposing `annotate_debug(image, valid,
debug)`) on each dev image and measures, on the initial network (the proposal, build_network) and on the
fitted one:
  strokes     the truth's long vessels: observable edges chained at observable nodes through the pair of ends
              that continue each other best (turn < TRUTH_TURN_DEG), as vesselscene builds them (a long vessel
              is tangent-continuous through its junctions); per stroke of at least MIN_STROKE_PX, the number of
              network edges that cover at least COVER_PX of it (a sample is covered by the nearest edge within
              TOL_PX); reported as the length-weighted mean and the share of strokes covered by one edge;
  false_ends  edge ends at degree-1 nodes that lie in the truth's observable lumen, farther than END_TOL_PX from
              every truth end (a degree-1 truth node) and from the border: a vessel broken where it continues;
  broken      nodes with three or more incident ends at which two ends of different edges continue each other
              (turn < CONT_DEG): a continuation the network does not make; with the relative radius jump
              |r1 - r2| / max(r1, r2) at those ends;
  steps       neighbouring-pixel steps of the fitted render that exceed vesselscene's clean OD's by > 0.03 OD
              (as report_diagnostics), more than 24 px from the border.
and the masks a mask term could fit, against the truth's observable lumen (recall overall, inside the
junction discs and on wide vessels (nearest truth sample r >= 8 px), and precision):
  od_fine     hysteresis on stage 1's cleaned OD / noise (seeds > 3, grown > 1.5);
  od_coarse   the same on the OD smoothed by a 3 px Gaussian (noise re-estimated), for wide faint vessels;
  vesselness  stage 1's own vessel mask (the one whose negative the background is fitted on).
Reads the truth: a diagnostic, never a pipeline path.  Dev scenes only (fastset.check_not_heldout).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import score as S

TRUTH_TURN_DEG = 25.0
MIN_STROKE_PX = 40.0
COVER_PX = 8.0
TOL_PX = 3.0
END_TOL_PX = 8.0
CONT_DEG = 30.0
BORDER = 24


# ======================================================================== truth strokes
def _out(xy, end, look=8.0):
    P = xy if end == 1 else xy[::-1]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    j = int(np.searchsorted(s, s[-1] - min(look, 0.6 * s[-1])))
    v = P[-1] - P[min(j, len(P) - 1)]
    n = math.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _turn(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(-np.dot(u, v))))))


def truth_strokes(case) -> tuple[list, np.ndarray]:
    """(strokes as polylines, the truth's degree-1 node positions)."""
    g = case["obs"]["graph"]
    E = [(int(e["source"]), int(e["target"]), np.asarray(e["xy"], float).reshape(-1, 2)) for e in g["edges"]]
    E = [e for e in E if len(e[2]) >= 2]
    inc = {}
    for k, (u, v, _) in enumerate(E):
        inc.setdefault(u, []).append((k, 0))
        inc.setdefault(v, []).append((k, 1))
    link = {}
    for n, ends in inc.items():
        cand = sorted((_turn(_out(E[a][2], ea), _out(E[b][2], eb)), i, j)
                      for i, (a, ea) in enumerate(ends) for j, (b, eb) in enumerate(ends) if i < j and a != b)
        used = set()
        for t, i, j in cand:
            if t > TRUTH_TURN_DEG or i in used or j in used:
                continue
            used |= {i, j}
            link[ends[i]], link[ends[j]] = ends[j], ends[i]
    seen, strokes = set(), []
    for k in range(len(E)):
        for end in (0, 1):
            if k in seen or (k, end) in link:
                continue
            parts, kk, ee = [], k, end
            while True:
                seen.add(kk)
                xy = E[kk][2] if ee == 0 else E[kk][2][::-1]
                parts.append(xy)
                nxt = (kk, 1 - ee)
                if nxt not in link or link[nxt][0] in seen:
                    break
                kk, ee = link[nxt]
            strokes.append(np.concatenate(parts))
    for k in range(len(E)):                             # closed loops
        if k not in seen:
            seen.add(k)
            strokes.append(E[k][2])
    deg = {n: len(v) for n, v in inc.items()}
    ends = np.array([[nd["x"], nd["y"]] for nd in g["nodes"] if deg.get(int(nd["id"]), 0) == 1], float).reshape(-1, 2)
    return strokes, ends


def _resample(xy, step=2.0):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if s[-1] <= 0:
        return xy[:1]
    q = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)


# ======================================================================== network metrics
def network_metrics(net, case, strokes, tends) -> dict:
    eids = list(net.edges)
    smp = {e: net.sample(e, 1.0) for e in eids}
    if not eids:
        return dict(pieces_per_stroke=float("nan"), one_piece=0.0, false_ends=0, broken=0, jump=float("nan"))
    P = np.concatenate([smp[e]["xy"] for e in eids])
    lab = np.concatenate([np.full(len(smp[e]["xy"]), i) for i, e in enumerate(eids)])
    tree = cKDTree(P)
    pieces, weights, one = [], [], 0
    for st in strokes:
        L = float(np.hypot(*np.diff(st, axis=0).T).sum())
        if L < MIN_STROKE_PX:
            continue
        q = _resample(st, 2.0)
        d, i = tree.query(q)
        ok = d <= TOL_PX
        if not ok.any():
            continue
        cnt = np.bincount(lab[i[ok]], minlength=len(eids)) * 2.0
        k = int((cnt >= COVER_PX).sum())
        if k == 0:
            continue
        pieces.append(k)
        weights.append(L)
        one += k == 1
    deg = net.degrees()
    lumen = case["lumen"]
    H, W = lumen.shape
    tt = cKDTree(tends) if len(tends) else None
    false_ends = 0
    for n, dg in deg.items():
        if dg != 1:
            continue
        x, y = net.nodes[n].x, net.nodes[n].y
        if x < BORDER or y < BORDER or x > W - 1 - BORDER or y > H - 1 - BORDER:
            continue
        if not lumen[int(round(min(max(y, 0), H - 1))), int(round(min(max(x, 0), W - 1)))]:
            continue
        if tt is not None and tt.query([x, y])[0] <= END_TOL_PX:
            continue
        false_ends += 1
    broken, jumps = 0, []
    for n in net.nodes:
        inc = net.incident(n)
        if len(inc) + 2 * len(net.passing(n)) < 3 or len(inc) < 2:
            continue
        dirs = [(e, end, net.end_tangent(e, end)) for e, end in inc]
        best = None
        for a in range(len(dirs)):
            for b in range(a + 1, len(dirs)):
                if dirs[a][0] == dirs[b][0]:
                    continue
                t = _turn(dirs[a][2], dirs[b][2])
                if t < CONT_DEG and (best is None or t < best[0]):
                    best = (t, a, b)
        if best is not None:
            broken += 1
            ra = net.edges[dirs[best[1]][0]].r[0 if dirs[best[1]][1] == 0 else -1]
            rb = net.edges[dirs[best[2]][0]].r[0 if dirs[best[2]][1] == 0 else -1]
            jumps.append(abs(ra - rb) / max(ra, rb, 1e-6))
    return dict(edges=len(eids), through=int(sum(len(net.edges[e].through) for e in eids)),
                pieces_per_stroke=float(np.average(pieces, weights=weights)) if pieces else float("nan"),
                one_piece=float(one / len(pieces)) if pieces else float("nan"), n_strokes=len(pieces),
                false_ends=false_ends, broken=broken, jump=float(np.median(jumps)) if jumps else float("nan"))


def steps(R, T) -> int:
    n = 0
    for ax in (0, 1):
        g, gT = np.abs(np.diff(R, axis=ax)), np.abs(np.diff(T, axis=ax))
        m = g - gT > 0.03
        m[:BORDER] = m[-BORDER:] = False
        m[:, :BORDER] = m[:, -BORDER:] = False
        n += int(m.sum())
    return n


# ======================================================================== masks
def _noise(OD, bgmask):
    hp = OD - ndi.gaussian_filter(OD, 3.0)
    v = hp[bgmask] if bgmask.any() else hp.ravel()
    return float(1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)


def od_mask(OD, valid, bgmask, smooth=0.0, hi=3.0, lo=1.5):
    X = ndi.gaussian_filter(OD, smooth) if smooth > 0 else OD
    z = X / _noise(X, bgmask & valid)
    seeds, grow = (z > hi) & valid, (z > lo) & valid
    lab, _ = ndi.label(grow)
    keep = np.unique(lab[seeds])
    return np.isin(lab, keep[keep > 0])


def mask_metrics(M, case, wide) -> dict:
    L = case["lumen"]
    jd = S.disc_mask(L.shape, case["discs"]) & L
    return dict(recall=float((M & L).sum() / max(L.sum(), 1)), recall_junction=float((M & jd).sum() / max(jd.sum(), 1)),
                recall_wide=float((M & wide).sum() / max(wide.sum(), 1)), precision=float((M & L).sum() / max(M.sum(), 1)))


# ======================================================================== driver
def _annotate_debug(pipeline):
    if pipeline:
        mod, fn = pipeline.split(":")
        return getattr(importlib.import_module(mod), fn)
    from experiments.splinefit import pipeline as P

    def run(image, valid, debug):
        return P.annotate_cfg(image, valid, P.DEFAULT, debug=debug)
    return run


def diagnose(folder, kind, annotate_debug) -> dict:
    from experiments.splinefit import fastset as FS
    from experiments.splinefit.proposals import build_network
    FS.check_not_heldout(folder)
    case = S.load_truth(folder, kind, reference_target=False)
    with np.load(os.path.join(folder, "labels.npz")) as z:
        case["lumen"] = z[f"observable_lumen_{kind}"].astype(bool)
    dbg = {}
    out = annotate_debug(case["image"].copy(), case["valid"].copy(), dbg)
    strokes, tends = truth_strokes(case)
    net0, _ = build_network(dbg["proposal"], case["image"].shape)
    res = dict(initial=network_metrics(net0, case, strokes, tends),
               fitted=network_metrics(dbg["net"], case, strokes, tends),
               steps=steps(np.asarray(out["od_render"], np.float32), case["od_img"]))
    s1 = dbg["proposal"]["s1"]
    OD, valid = np.asarray(s1["OD"], np.float32), case["valid"]
    bg = np.asarray(s1["bg"], bool)
    P = case["prof"]
    keep = P["in_frame"] & (P["merged_into"] < 0)
    xy, r = np.stack([P["x_px"], P["y_px"]], 1)[keep], P["radius_px"][keep]
    H, W = OD.shape
    Y, X = np.mgrid[0:H, 0:W]
    L = case["lumen"]
    _, i = cKDTree(xy).query(np.stack([X[L], Y[L]], 1))
    wide = np.zeros_like(L)
    wide[L] = r[i] >= 8.0
    masks = dict(od_fine=od_mask(OD, valid, bg), od_coarse=od_mask(OD, valid, bg, smooth=3.0),
                 vesselness=np.asarray(s1["mask"], bool) & valid)
    res["masks"] = {k: mask_metrics(M, case, wide) for k, M in masks.items()}
    return res


def main(argv=None):
    from experiments.splinefit import fastset as FS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--kind", default="average")
    ap.add_argument("--pipeline", default=None, help="MOD:FN with signature (image, valid, debug) -> output")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    scenes = a.scenes or FS.dev_scenes()
    fn = _annotate_debug(a.pipeline)
    allres = {}
    for folder in scenes:
        name = os.path.basename(os.path.normpath(folder))
        r = allres[name] = diagnose(folder, a.kind, fn)
        for st in ("initial", "fitted"):
            m = r[st]
            print(f"{name} {st:8s} edges {m['edges']:4d} through {m['through']:3d}  pieces/stroke "
                  f"{m['pieces_per_stroke']:.2f}  one-piece {m['one_piece']:.2f} (n {m['n_strokes']})  "
                  f"false ends {m['false_ends']:3d}  broken continuations {m['broken']:3d} (median r jump "
                  f"{m['jump']:.2f})", flush=True)
        print(f"{name} steps {r['steps']}")
        for k, m in r["masks"].items():
            print(f"{name} mask {k:10s} recall {m['recall']:.3f}  at junctions {m['recall_junction']:.3f}  "
                  f"wide {m['recall_wide']:.3f}  precision {m['precision']:.3f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(allres, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
