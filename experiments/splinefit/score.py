r"""The fit scorer (the frozen evaluator): graph, geometry and render scores of a fitted spline network on one
vesselscene image or on one crop of it.

Frozen after the Instrument phase, like autoresearch's prepare.py: a later change only to fix a confirmed bug,
followed by a logged re-baseline (NOTES / results.tsv).  The truth is read here, in the probe generator and in
clearly labelled oracle diagnostics only, never in a pipeline.

The pipeline contract
---------------------
A pipeline is ``annotate(image, valid) -> dict`` (image: the still or crop as saved, float32 DN, NaN = no
data; valid: its finite pixels; nothing else) with

    polylines   [(n, 2) x, y px]              the harness contract (optional when network is given)
    junctions   [(x, y, type)]                 idem
    network     dict(nodes=[{id, x, y, kind}],
                     edges=[{u, v, xy (n, 2) px, r (n,) px half width, s (n,) px blur, a (n,) OD contrast}],
                     crossings=[{x, y, e1, e2}])
    od_render   (H, W) the fitted render in OD (vessels only, positive), optional
    od_target   (H, W) the pipeline's own cleaned target OD = B - ln I, optional (diagnostic only)
    halo        (weight, sigma px), optional: the halo of the pipeline's renderer (for the render audit)

r is the lumen half width, s the Gaussian blur of the profile (all optics and defocus in one std; a halo, if
the renderer has one, is not part of s), a the centre OD of the UNBLURRED profile: vesselmap's convention, the
peak of the chord (or box) profile before the PSF.  Coordinates: pixel (0, 0) is the centre of the first pixel,
x to the right, y down.

Graph (``score_graph``)
-----------------------
With a network, the graph is the network's (``network_graph``): one polyline per edge (its xy), and as
junctions every node whose kind is a harness junction type (bifurcation, confluence, pseudo-T, crossing,
compound, branch) or whose degree (edge ends at it) is at least 3, typed by its kind when that is a harness
type, else 'branch' (degree 3) / 'compound' (4 or more); every listed crossing is a junction typed
'crossing'.  The harness cuts the polylines where they pass within 3 px of a junction, so a through vessel is
cut at its crossing point.  Without a network, the output's polylines and junctions are scored as they are.
The scores are experiments.neuromimetic.harness.score's (centreline recall / precision / F1, junction F1
strict, coarse balanced type accuracy, crossing recall / precision, edge cover, purity, ...) and

    graph_score = mean(centreline_f1, junction_f1_strict, junction_type_balanced_coarse, edge_cover)

(a NaN term, e.g. no matched junction to type, counts 0).

Geometry vs the true network (``true_samples``, ``score_geometry``)
-------------------------------------------------------------------
The TRUE samples are the vessels' 1 px samples of complete_visibility_<kind>.npz that lie on the observable
truth runs: in the frame, observable, on the vessel's own line, not merged into another line, not an absorbed
stub (the runs' observed pieces, harness centreline recall's truth).  Per sample: centre (x_px, y_px), unit
tangent, r = radius_px, s = blur_px (the vessel's depth blur combined with the image's formation blur,
scene.json formed[kind].blur_px: what scene.to_vesselnetwork exports as s), and two contrasts: a_closed = the
linear-law centre OD whose nested boxes best match the chord-law staircase (exactly scene.to_vesselnetwork's
a, least squares over the boxes' areas, evaluated per sample, not through its 30 px spline refit) and a =
a_closed x od_ratio (the ratio of the contrast the rendered image shows to the closed form's: a frame's
red-cell gaps, a fading capillary; 1 for most samples).  The scene's halo (optics halo_h, halo_px) is a
separate PSF component, as in vesselmap.

The fit's edges are resampled every 1 px of arclength with r, s, a interpolated linearly, as segments.  Each
true sample is matched to the nearest fitted segment point within its gate max(6 px, r_true) whose segment
direction is within 60 deg of the true tangent (|cos| >= 0.5; an arm of another vessel at a junction is not a
match).  Unmatched samples (a missed vessel: graph recall's business) are left out of the medians and counted
in geom_matched (length-weighted share matched).  Over the matched samples, length-weighted medians (weights:
the samples' arclength spacing) of

    offset_px      the distance to the fitted centreline (point to segment, exact)
    width_rel_err  |r_fit - r_true| / r_true
    blur_err_px    |s_fit - s_true|
    contrast_rel_err         |a_fit - a_true| / a_true,  contrast_closed_rel_err with a_closed
    *_bias         the signed medians (r_fit / r_true - 1, s_fit - s_true, a_fit / a_true - 1)

each also '_nojunc', over the samples outside every observable junction disc (radius min(radius, 2 lambda)).
Then

    pos = 1 - min(1, offset_px / 3),   width = 1 - min(1, width_rel_err)   (0 when nothing matched).

Render (``score_render``)
-------------------------
The target of the composite's 'explained' is the image's OWN optical density with an ORACLE background,

    OD_obs = B* - ln I,    B* = masked normalised Gaussian (8 px) of ln I + OD_img over the usable pixels,

OD_img = -ln G_form * exp(-OD_true) the scene's clean vessel OD (labels.npz od_<kind>, halo included) through
the image formation's blur (scene.json formed[kind].blur_cov: a Gaussian for an average, the line of the
frame's motion for a frame; applied in intensity, as imaging.form_image does).  OD_obs therefore holds the
vessels, the image's noise and fine texture, and every model mismatch (chord law, red cells, glare bloom),
but no background-estimation error: a pipeline is not rewarded for its own target (which it could make
trivially easy) nor tied to one background estimator's faults (the reference target, stage 1, attenuates the
junctions, whose wide lumens leak into its background).  Usable pixels: valid, finite, 0 < I < 4095 (glare).

Masks: the band = usable pixels within 3 px of the observable truth lumens (labels.npz
observable_lumen_<kind>, dilated by a 7 px disc); the junction discs = the observable junctions' discs of
radius min(radius, 2 lambda), inside the band.  With R the render:

    explained          = 1 - sum_band (OD_obs - R)^2 / sum_band OD_obs^2
    explained_junction = the same over the junction discs
    junction_bias      = sum_disc (R - OD_obs) / sum_disc OD_obs  (> 0: the render over-predicts the junctions,
                         the additive render's fault at forks)
    chi2               = mean over usable pixels of ((OD_obs - R) / sigma)^2, sigma the image's noise in OD:
                         the robust std (1.4826 MAD) of ln I + OD_img - B* (high-pass noise and texture);
                         the clean OD_img as the render gives 1.0-1.7 on dev (the noise's heavier tails)
    explained_self, explained_self_junction   the same against the pipeline's own od_target (diagnostic)
    explained_ref, explained_ref_junction     the same against the reference target, neuromimetic stage 1
                         (photoreceptors: background fitted on the vesselness mask's negative) of the image the
                         pipeline saw (diagnostic)
    explained_true, explained_true_junction   the same against OD_img (no noise, no texture; oracle diagnostic)
    target_fidelity    = 1 - sum_band (od_target - OD_img)^2 / sum_band OD_img^2 (the pipeline's cleaning;
                         ref_fidelity the same for the reference target)

Without od_render, a network is rendered with vesselmap's renderer (``render_network``: halo from out['halo']
= (weight, sigma) or vesselmap's default; render_source 'vesselmap').  Render audit (``audit=True``): with
both od_render and a network, render_agreement = explained of the pipeline's od_render by vesselmap's
additive render of the same network, over the band outside the junction discs and crossings (where the two
renderers should agree); a low value flags an od_render that is not the network's render.

Crops (``load_truth(..., crop=(x0, y0, w, h))``, the fast tier)
---------------------------------------------------------------
The pipeline sees image[y0:y0+h, x0:x0+w] and its valid mask; every truth is cut to the window and shifted by
(-x0, -y0): the observable runs and image-graph edges are clipped to their pieces inside (a piece's end at the
window border becomes a 'border' node), the true samples (and the profiles' in_frame) keep those inside;
OD_obs, sigma and the band are computed on the whole image and cut; the reference target is stage 1 of the
crop.  Observable junctions closer than CROP_BORDER_PX to the window border are moved to the don't-care
junctions (a junction detected there is neither rewarded nor penalised) and get no disc.

Composite
---------
    composite = 0.5 graph_score + 0.5 mean(pos, width, clip(explained, 0, 1))

(explained NaN, e.g. no render, counts 0).  The truth-reading functions here are for scoring and for clearly
labelled oracle / self-check rows only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from experiments.neuromimetic import harness as H
from vesselscene import render as VR
from vesselscene import truth as T
from vesselscene.graph import VesselGraph

LAMBDA_PX = T.LAMBDA_PX
POS_SCALE_PX = 3.0                    # pos = 1 - offset / this
GATE_MIN_PX = 6.0                     # a true sample matches a fitted centreline within max(this, r_true)
COS_MIN = 0.5                         # ... running within 60 deg of its tangent
BAND_PX = 3                           # render band: within this of the observable lumens
DISC_CAP_PX = 2.0 * LAMBDA_PX         # junction discs: radius min(radius, this) (harness STRICT_RADIUS_PX)
BG_SIGMA = 8.0                        # px, the oracle background's masked Gaussian (and the noise high-pass)
SATURATION = 4095.0                   # 12-bit full scale: glare is not usable
CROP_BORDER_PX = 16.0                 # crop: junctions nearer the window border are don't-care
JTYPES = ("bifurcation", "confluence", "pseudo-T", "crossing", "compound", "branch")
GRAPH_KEYS = ("centreline_f1", "junction_f1_strict", "junction_type_balanced_coarse", "edge_cover")


# ======================================================================== truth
def _scene_json(folder: str) -> dict:
    with open(os.path.join(folder, "scene.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_truth(folder: str, kind: str, crop=None, reference_target: bool = True) -> dict:
    """Everything the scorer needs for one image (or one crop (x0, y0, w, h) of it): harness.load_case
    (image, valid, observable truth, profiles), the true samples, the junction discs, the render masks, the
    clean OD, the oracle-background target OD_obs and the noise sigma, and (reference_target) the reference
    target OD of neuromimetic stage 1 of the image the pipeline sees."""
    case = H.load_case(folder, kind)
    sj = _scene_json(folder)
    case["scene"] = sj
    case["crop"] = None
    case["samples"] = true_samples(folder, kind, case["prof"], sj)
    with np.load(os.path.join(folder, "labels.npz")) as z:
        lumen = z[f"observable_lumen_{kind}"].astype(bool)
        od_true = z[f"od_{kind}"].astype(np.float64)
    case["od_img"] = formed_od(od_true, sj, kind).astype(np.float32)
    case["od_obs"], case["sigma"] = oracle_target(case["image"], case["valid"], case["od_img"])
    case["usable"] = np.isfinite(case["od_obs"])
    yy, xx = np.mgrid[-BAND_PX:BAND_PX + 1, -BAND_PX:BAND_PX + 1]
    case["band_all"] = ndi.binary_dilation(lumen, structure=(xx * xx + yy * yy) <= BAND_PX * BAND_PX)
    if crop is not None:
        case = crop_case(case, crop)
    js = case["obs"]["junctions"]
    case["discs"] = np.array([[j["x"], j["y"], min(float(j["radius"]), DISC_CAP_PX)] for j in js],
                             float).reshape(-1, 3)
    case["samples"]["near_junction"] = _in_discs(case["samples"]["xy"], case["discs"])
    case["band"] = case.pop("band_all") & case["usable"]
    case["jdisc"] = disc_mask(case["image"].shape, case["discs"]) & case["band"]
    if reference_target:
        case["od_ref"] = reference_od(case["image"], case["valid"])
    return case


def disc_mask(shape, discs: np.ndarray) -> np.ndarray:
    Hh, Ww = int(shape[0]), int(shape[1])
    jd = np.zeros((Hh, Ww), bool)
    Y, X = np.mgrid[0:Hh, 0:Ww]
    for x, y, r in discs:
        x0, x1 = max(0, int(x - r - 1)), min(Ww, int(x + r + 2))
        y0, y1 = max(0, int(y - r - 1)), min(Hh, int(y + r + 2))
        if x1 > x0 and y1 > y0:
            jd[y0:y1, x0:x1] |= (X[y0:y1, x0:x1] - x) ** 2 + (Y[y0:y1, x0:x1] - y) ** 2 <= r * r
    return jd


def _in_discs(xy: np.ndarray, discs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(xy), bool)
    if not len(discs) or not len(xy):
        return out
    tree = cKDTree(xy)
    for x, y, r in discs:
        out[tree.query_ball_point([x, y], r)] = True
    return out


def true_samples(folder: str, kind: str, prof: dict | None = None, sj: dict | None = None) -> dict:
    """The true per-sample centre, tangent, r, s and contrasts on the observable runs (module docstring)."""
    prof = prof if prof is not None else T.load_profiles(os.path.join(folder, f"complete_visibility_{kind}.npz"))
    sj = sj if sj is not None else _scene_json(folder)
    graph = VesselGraph.load(os.path.join(folder, "truth.json"))
    S = T._samples_from_profiles(graph, sj, kind, prof)          # exact, raises if the sampling differs
    area = np.diff(np.r_[0.0, VR.BOX_C])
    Hk = VR.BOX_H
    tan, a_cl, ds = [], [], []
    for v in prof["vids"].tolist():
        d = S.v[int(v)]
        Lv = VR.chord_levels(d["a"], d["f"])
        a_cl.append((Lv * Hk * area).sum(1) / (Hk * Hk * area).sum())     # scene.to_vesselnetwork's a
        tan.append(d["T"])
        ds.append(np.full(d["n"], d["ds"]))
    tan, a_cl, ds = np.concatenate(tan), np.concatenate(a_cl), np.concatenate(ds)
    xy = np.stack([prof["x_px"], prof["y_px"]], 1).astype(float)
    m = prof["in_frame"] & prof["observable"] & prof["line"] & (prof["merged_into"] < 0) & ~prof["absorbed"]
    vid = np.repeat(prof["vids"], np.diff(prof["offsets"]))
    return dict(xy=xy[m], tan=tan[m], r=prof["radius_px"][m].astype(float), s=prof["blur_px"][m].astype(float),
                a=(a_cl * prof["od_ratio"])[m].astype(float), a_closed=a_cl[m].astype(float), w=ds[m],
                vid=vid[m].astype(int))


def formed_od(od_true: np.ndarray, sj: dict, kind: str) -> np.ndarray:
    """OD_img = -ln G * exp(-OD_true): the clean vessel OD through the image formation's blur (module
    docstring), applied in intensity as imaging.form_image does."""
    from vesselscene import imaging as IM
    cov = np.asarray(sj["formed"][kind].get("blur_cov", np.zeros((2, 2))), float)
    E = np.exp(-np.asarray(od_true, np.float64))
    if kind == "frame":                                # the frame's motion: a line of length |v|, cov = v v^T / 12
        w, V = np.linalg.eigh(cov)
        v = V[:, -1] * math.sqrt(max(12.0 * float(w[-1]), 0.0))
        G = IM._blur_line(E, v) if np.hypot(*v) > 1e-6 else E
    else:
        G = IM._blur_gauss(E, cov)
    return -np.log(np.clip(G, 1e-12, None))


def oracle_target(image: np.ndarray, valid: np.ndarray, od_img: np.ndarray):
    """OD_obs = B* - ln I with the oracle background B* (module docstring; NaN where not usable) and the
    noise sigma in OD."""
    from experiments.neuromimetic.neuromimetic import _masked_mean
    I = np.asarray(image, np.float32)
    ok = np.asarray(valid, bool) & np.isfinite(I) & (I > 0) & (I < SATURATION)
    L = np.zeros(I.shape, np.float32)
    L[ok] = np.log(I[ok])
    X = np.where(ok, L + od_img, 0.0).astype(np.float32)
    B = _masked_mean(X, ok, BG_SIGMA)
    od = np.where(ok, B - L, np.nan).astype(np.float32)
    hp = (X - B)[ok]
    sigma = float(1.4826 * np.median(np.abs(hp - np.median(hp)))) if ok.any() else float("nan")
    return od, sigma


def reference_od(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The reference target: neuromimetic stage 1's OD = B - ln I, B fitted outside the vesselness mask."""
    from experiments.neuromimetic import neuromimetic as NM
    return NM.photoreceptors(image, valid)["OD"].astype(np.float32)


def _clip_pieces(P: np.ndarray, w: int, h: int):
    """Index runs of the points of P inside the window [-0.5, w - 0.5] x [-0.5, h - 0.5]."""
    ins = (P[:, 0] >= -0.5) & (P[:, 0] <= w - 0.5) & (P[:, 1] >= -0.5) & (P[:, 1] <= h - 0.5)
    runs, start = [], None
    for i, f in enumerate(ins.tolist() + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def crop_case(case: dict, crop) -> dict:
    """The case cut to the window crop = (x0, y0, w, h) and shifted (module docstring, 'Crops')."""
    x0, y0, w, h = (int(c) for c in crop)
    sh = np.array([x0, y0], float)
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    out = dict(case)
    out["crop"] = [x0, y0, w, h]
    for key in ("image", "valid", "od_img", "od_obs", "usable", "band_all"):
        out[key] = np.ascontiguousarray(case[key][sl])
    assert out["image"].shape == (h, w), ("crop outside the image", crop, case["image"].shape)
    # profiles: shifted, in_frame restricted to the window (the observable and don't-care samples outside
    # stay: a traced point at the border near one is judged as on the whole image)
    prof = dict(case["prof"])
    prof["x_px"] = (prof["x_px"] - x0).astype(prof["x_px"].dtype)
    prof["y_px"] = (prof["y_px"] - y0).astype(prof["y_px"].dtype)
    inside = (prof["x_px"] >= -0.5) & (prof["x_px"] <= w - 0.5) & (prof["y_px"] >= -0.5) & (prof["y_px"] <= h - 0.5)
    prof["in_frame"] = prof["in_frame"] & inside
    out["prof"] = prof
    ts = case["samples"]
    xy = ts["xy"] - sh
    m = (xy[:, 0] >= -0.5) & (xy[:, 0] <= w - 0.5) & (xy[:, 1] >= -0.5) & (xy[:, 1] <= h - 0.5)
    out["samples"] = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in ts.items()}
    out["samples"]["xy"] = xy[m]
    # observable truth
    obs = case["obs"]
    o = {k: v for k, v in obs.items() if k not in ("runs", "junctions", "dont_care_junctions", "graph")}
    runs = []
    for r in obs["runs"]:
        P = np.asarray(r["xy"], float).reshape(-1, 2) - sh
        bad = np.zeros(len(P), bool)
        for a, b in r.get("unobserved") or ():
            bad[a:b + 1] = True
        for a, b in _clip_pieces(P, w, h):
            seg = bad[a:b + 1]
            un, k = [], 0
            while k < len(seg):                       # unobserved index ranges re-based on the piece
                if seg[k]:
                    k2 = k
                    while k2 + 1 < len(seg) and seg[k2 + 1]:
                        k2 += 1
                    un.append([k, k2])
                    k = k2 + 1
                else:
                    k += 1
            runs.append(dict(id=len(runs), vid=r.get("vid"), xy=P[a:b + 1].tolist(), unobserved=un))
    o["runs"] = runs
    js, dcj = [], []
    for j in obs["junctions"]:
        x, y = float(j["x"]) - x0, float(j["y"]) - y0
        d = min(x + 0.5, w - 0.5 - x, y + 0.5, h - 0.5 - y)
        if d >= CROP_BORDER_PX:
            js.append(dict(j, x=x, y=y))
        elif d >= -float(j["radius"]):                # near or across the border: don't care
            dcj.append(dict(x=x, y=y, radius=float(j["radius"]), source="crop_border", ref=j["id"]))
    for d_ in obs.get("dont_care_junctions", []):
        x, y = float(d_["x"]) - x0, float(d_["y"]) - y0
        if -d_["radius"] <= x <= w + d_["radius"] and -d_["radius"] <= y <= h + d_["radius"]:
            dcj.append(dict(d_, x=x, y=y))
    o["junctions"], o["dont_care_junctions"] = js, dcj
    g = obs["graph"]
    gn = {n["id"]: n for n in g["nodes"]}
    nodes, edges = {}, []
    nid = max(gn) + 1 if gn else 0
    for e in g["edges"]:
        P = np.asarray(e["xy"], float).reshape(-1, 2) - sh
        for a, b in _clip_pieces(P, w, h):
            ends = []
            for idx, orig, here in ((a, e["source"], a == 0), (b, e["target"], b == len(P) - 1)):
                if here:                              # the edge's own end node, inside the window
                    n = gn[orig]
                    nodes[orig] = dict(n, x=float(n["x"]) - x0, y=float(n["y"]) - y0)
                    ends.append(orig)
                else:                                 # cut by the window border
                    nodes[nid] = dict(id=nid, kind="border", x=float(P[idx, 0]), y=float(P[idx, 1]))
                    ends.append(nid)
                    nid += 1
            edges.append(dict(e, id=len(edges), source=ends[0], target=ends[1], xy=P[a:b + 1].tolist(),
                              length_px=float(np.hypot(*np.diff(P[a:b + 1], axis=0).T).sum())))
    o["graph"] = dict(g, nodes=[nodes[k] for k in sorted(nodes)], edges=edges)
    out["obs"] = o
    return out
# ======================================================================== networks
def _arr(v, shape_tail=()):
    return np.asarray(v, float).reshape((-1,) + tuple(shape_tail))


def network_graph(network: dict):
    """The harness polylines and typed junctions of a contract network (module docstring, 'Graph')."""
    edges = network.get("edges", [])
    deg = Counter()
    for e in edges:
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    junctions = []
    for n in sorted(network.get("nodes", []), key=lambda n: n["id"]):
        kind, d = n.get("kind"), deg[n["id"]]
        if kind in JTYPES or d >= 3:
            junctions.append((float(n["x"]), float(n["y"]), kind if kind in JTYPES else
                              ("branch" if d == 3 else "compound")))
    for c in network.get("crossings", []):
        junctions.append((float(c["x"]), float(c["y"]), "crossing"))
    polylines = [_arr(e["xy"], (2,)) for e in edges if len(e["xy"])]
    return polylines, junctions


def network_segments(network: dict, step: float = 1.0) -> dict:
    """Every edge resampled every `step` px of arclength (r, s, a linear), as segments p0 -> p1."""
    P0, P1, Q0, Q1 = [], [], [], []
    for e in network.get("edges", []):
        xy = _arr(e["xy"], (2,))
        if len(xy) < 2:
            continue
        prm = np.stack([_arr(e[k]) for k in ("r", "s", "a")], 1)
        t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        if t[-1] <= 0:
            continue
        q = np.linspace(0.0, t[-1], max(2, int(math.ceil(t[-1] / step)) + 1))
        P = np.stack([np.interp(q, t, xy[:, 0]), np.interp(q, t, xy[:, 1])], 1)
        Q = np.stack([np.interp(q, t, prm[:, j]) for j in range(3)], 1)
        P0.append(P[:-1])
        P1.append(P[1:])
        Q0.append(Q[:-1])
        Q1.append(Q[1:])
    if not P0:
        z = np.zeros((0, 2))
        return dict(p0=z, p1=z, q0=np.zeros((0, 3)), q1=np.zeros((0, 3)))
    return dict(p0=np.concatenate(P0), p1=np.concatenate(P1), q0=np.concatenate(Q0), q1=np.concatenate(Q1))


def match_samples(truth_s: dict, seg: dict, k: int = 64) -> dict:
    """Each true sample's matched fitted point (module docstring): distance, interpolated r, s, a, or NaN."""
    n = len(truth_s["xy"])
    out = dict(d=np.full(n, np.nan), r=np.full(n, np.nan), s=np.full(n, np.nan), a=np.full(n, np.nan))
    if not len(seg["p0"]) or not n:
        return out
    p0, p1 = seg["p0"], seg["p1"]
    mid = 0.5 * (p0 + p1)
    half = 0.5 * float(np.hypot(*(p1 - p0).T).max())
    gate = np.maximum(GATE_MIN_PX, truth_s["r"])
    k = min(k, len(mid))
    _, idx = cKDTree(mid).query(truth_s["xy"], k=k, distance_upper_bound=float(gate.max()) + half)
    idx = idx.reshape(n, k)
    ok = idx < len(mid)
    ii = np.where(ok, idx, 0)
    A, B = p0[ii], p1[ii]                                   # (n, k, 2)
    D = B - A
    L2 = np.maximum((D * D).sum(-1), 1e-12)
    X = truth_s["xy"][:, None, :]
    u = np.clip(((X - A) * D).sum(-1) / L2, 0.0, 1.0)
    dist = np.hypot(*(A + u[..., None] * D - X).transpose(2, 0, 1))
    cos = np.abs((D * truth_s["tan"][:, None, :]).sum(-1)) / np.sqrt(L2)
    ok &= (dist <= gate[:, None]) & (cos >= COS_MIN)
    dist = np.where(ok, dist, np.inf)
    j = np.argmin(dist, 1)
    hit = np.isfinite(dist[np.arange(n), j])
    sel = ii[np.arange(n), j]
    uu = u[np.arange(n), j]
    q = seg["q0"][sel] + uu[:, None] * (seg["q1"][sel] - seg["q0"][sel])
    out["d"][hit] = dist[np.arange(n), j][hit]
    for c, key in enumerate(("r", "s", "a")):
        out[key][hit] = q[hit, c]
    return out


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x)
    if not m.any():
        return float("nan")
    x, w = x[m], w[m]
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def score_geometry(case: dict, network: dict | None) -> dict:
    """Offset, width, blur and contrast errors of the fit vs the true observable samples, pos and width."""
    ts = case["samples"]
    m = match_samples(ts, network_segments(network or {}))
    hit = np.isfinite(m["d"])
    w = ts["w"]
    out = dict(geom_matched=float(w[hit].sum() / w.sum()) if len(w) else float("nan"), geom_n_true=int(len(w)))
    terms = dict(offset_px=m["d"], width_rel_err=np.abs(m["r"] - ts["r"]) / ts["r"],
                 blur_err_px=np.abs(m["s"] - ts["s"]),
                 contrast_rel_err=np.abs(m["a"] - ts["a"]) / np.maximum(ts["a"], 1e-6),
                 contrast_closed_rel_err=np.abs(m["a"] - ts["a_closed"]) / np.maximum(ts["a_closed"], 1e-6),
                 width_bias=m["r"] / ts["r"] - 1.0, blur_bias_px=m["s"] - ts["s"],
                 contrast_bias=m["a"] / np.maximum(ts["a"], 1e-6) - 1.0)
    far = ~ts["near_junction"]
    for key, v in terms.items():
        out[key] = _wmedian(v, w)
        out[key + "_nojunc"] = _wmedian(np.where(far, v, np.nan), w)
    out["pos"] = 1.0 - min(1.0, out["offset_px"] / POS_SCALE_PX) if hit.any() else 0.0
    out["width"] = 1.0 - min(1.0, out["width_rel_err"]) if hit.any() else 0.0
    return out


# ======================================================================== render
def _explained(t, R, m) -> float:
    m = m & np.isfinite(t) & np.isfinite(R)
    den = float((t[m].astype(np.float64) ** 2).sum())
    return 1.0 - float(((t[m] - R[m]).astype(np.float64) ** 2).sum()) / den if den > 0 and m.any() else float("nan")


RENDER_KEYS = ("explained", "explained_junction", "junction_bias", "chi2", "explained_self", "explained_self_junction",
               "explained_ref", "explained_ref_junction", "explained_true", "explained_true_junction",
               "target_fidelity", "ref_fidelity")


def score_render(case: dict, od_render: np.ndarray | None, od_target: np.ndarray | None) -> dict:
    """explained / explained_junction / junction_bias / chi2 against the oracle-background target OD_obs, and
    the diagnostics against the pipeline's target, the reference target and the clean OD (module docstring)."""
    out = {k: float("nan") for k in RENDER_KEYS}
    band, jd, use = case["band"], case["jdisc"], case["usable"]
    ref = case.get("od_ref")
    if od_target is not None:
        od_target = np.asarray(od_target, np.float32)
        out["target_fidelity"] = _explained(case["od_img"], od_target, band)
    if ref is not None:
        out["ref_fidelity"] = _explained(case["od_img"], ref, band)
    if od_render is None:
        return out
    R = np.asarray(od_render, np.float32)
    assert R.shape == case["image"].shape, ("od_render shape", R.shape, case["image"].shape)
    t = case["od_obs"]
    out["explained"] = _explained(t, R, band)
    out["explained_junction"] = _explained(t, R, jd)
    m = jd & np.isfinite(R)
    den = float(t[m].astype(np.float64).sum())
    out["junction_bias"] = float((R[m] - t[m]).astype(np.float64).sum()) / den if m.any() and den > 0 \
        else float("nan")
    ok = use & np.isfinite(R)
    sg = max(case["sigma"], 1e-9)
    out["chi2"] = float((((t[ok] - R[ok]) / sg).astype(np.float64) ** 2).mean()) if ok.any() else float("nan")
    if od_target is not None:
        out["explained_self"] = _explained(od_target, R, band)
        out["explained_self_junction"] = _explained(od_target, R, jd)
    if ref is not None:
        out["explained_ref"] = _explained(ref, R, band)
        out["explained_ref_junction"] = _explained(ref, R, jd)
    out["explained_true"] = _explained(case["od_img"], R, band)
    out["explained_true_junction"] = _explained(case["od_img"], R, jd)
    return out


def to_vesselmap(network: dict, shape, halo=None):
    """A contract network as a vesselmap VesselNetwork (edges refitted faithfully on its spline bases)."""
    from experiments.splinefit import use_vesselmap
    use_vesselmap()
    from vesselmap.network import VesselNetwork
    net = VesselNetwork(shape)
    ids = {}
    for n in sorted(network.get("nodes", []), key=lambda n: n["id"]):
        ids[n["id"]] = net.add_node(float(n["x"]), float(n["y"]))
    for e in network.get("edges", []):
        xy = _arr(e["xy"], (2,))
        if len(xy) < 2:
            continue
        net.add_edge_dense(xy, _arr(e["r"]), _arr(e["s"]), _arr(e["a"]), u=ids.get(e["u"]), v=ids.get(e["v"]),
                           faithful=True)
    if halo is not None:
        net.meta["optics"] = dict(halo_weight=float(halo[0]), halo_sigma=float(halo[1]))
    return net


def render_network(net, shape) -> np.ndarray:
    """The OD render of a network (a contract dict or a vesselmap VesselNetwork) by vesselmap's renderer,
    without its width / blur caps (a scorer must not clip a wide true vessel).  Halo: net.meta['optics']."""
    from experiments.splinefit import use_vesselmap
    use_vesselmap()
    import torch
    from vesselmap import render as VMR
    from vesselmap.network import R_MIN, S_MIN
    if isinstance(net, dict):
        net = to_vesselmap(net, shape)
    Hh, Ww = int(shape[0]), int(shape[1])
    if not net.edges:
        return np.zeros((Hh, Ww), np.float32)
    net = net.copy()
    gh, gw = int(math.ceil((Hh - 1) / 64.0)) + 1, int(math.ceil((Ww - 1) / 64.0)) + 1
    net.background, net.bg_spacing = np.zeros((gh, gw), np.float32), 64.0
    with torch.no_grad():
        model = VMR.NetworkModel(net, np.zeros((Hh, Ww), np.float32), np.ones((Hh, Ww), np.float32),
                                 fit_background=False)
        cat = lambda key: np.concatenate([getattr(net.edges[e], key) for e in model.eids])
        model.raw_r_max = torch.full_like(model.raw_r_max, float("inf"))
        model.raw_s_max = torch.full_like(model.raw_s_max, float("inf"))
        model.raw_r.data = torch.tensor(VMR.inv_softplus(np.maximum(cat("r"), R_MIN + 1e-3) - R_MIN), dtype=torch.float32)
        model.raw_s.data = torch.tensor(VMR.inv_softplus(np.maximum(cat("s"), S_MIN + 1e-3) - S_MIN), dtype=torch.float32)
        model.rebuild()
        return model.optical_density().numpy().astype(np.float32)


# ======================================================================== the whole score
def score_graph(case: dict, out: dict) -> dict:
    if out.get("network") is not None:
        polys, juncs = network_graph(out["network"])
        src = "network"
    else:
        polys, juncs = out.get("polylines", []), out.get("junctions", [])
        src = "polylines"
    s = H.score(case, dict(polylines=polys, junctions=juncs))
    s["graph_score"] = float(np.mean([s.get(k) if np.isfinite(s.get(k, np.nan)) else 0.0 for k in GRAPH_KEYS]))
    s["graph_source"] = src
    s["n_junctions_fit"] = len(juncs)
    s["n_junctions_truth"] = len(case["obs"]["junctions"])
    return s


def render_agreement(case: dict, out: dict, R: np.ndarray) -> float:
    """Render audit (module docstring): explained of the pipeline's od_render by vesselmap's additive render
    of its network, over the band outside the junction discs and the network's crossings."""
    net = out["network"]
    V = render_network(to_vesselmap(net, case["image"].shape, out.get("halo")), case["image"].shape)
    cr = np.array([[c["x"], c["y"], DISC_CAP_PX] for c in net.get("crossings", [])], float).reshape(-1, 3)
    m = case["band"] & ~disc_mask(R.shape, np.concatenate([case["discs"], cr]))
    return _explained(np.asarray(R, np.float32), V, m)


def score(case: dict, out: dict, audit: bool = False) -> dict:
    """Every score of one pipeline output on one image or crop (module docstring), composite included."""
    s = score_graph(case, out)
    s.update(score_geometry(case, out.get("network")))
    R, src = out.get("od_render"), "pipeline"
    if R is None and out.get("network") is not None:
        net = to_vesselmap(out["network"], case["image"].shape, out.get("halo"))
        R, src = render_network(net, case["image"].shape), "vesselmap"
    s.update(score_render(case, R, out.get("od_target")))
    s["render_source"] = src if R is not None else None
    s["render_agreement"] = render_agreement(case, out, R) if audit and src == "pipeline" and R is not None \
        and out.get("network") is not None else float("nan")
    s["noise_sigma"] = case["sigma"]
    s["composite"] = composite(s)
    return s


def composite(s: dict) -> float:
    """0.5 graph_score + 0.5 mean(pos, width, clip(explained, 0, 1)); a NaN explained counts 0."""
    e = s.get("explained", float("nan"))
    e = min(1.0, max(0.0, e)) if np.isfinite(e) else 0.0
    return 0.5 * s["graph_score"] + 0.5 * float(np.mean([s["pos"], s["width"], e]))


def digest(out: dict) -> str:
    """harness.digest of the polylines and junctions, plus the network's nodes, edges (xy, r, s, a) and
    crossings, and od_render, rounded (1e-3 px / 1e-4 OD)."""
    h = hashlib.sha256(H.digest(out).encode())
    net = out.get("network")
    if isinstance(net, dict):
        for n in sorted(net.get("nodes", []), key=lambda n: n["id"]):
            h.update(repr((n["id"], round(float(n["x"]), 3), round(float(n["y"]), 3), n.get("kind"))).encode())
        for e in net.get("edges", []):
            h.update(repr((e["u"], e["v"])).encode())
            h.update(np.round(_arr(e["xy"]), 3).tobytes())
            for k in ("r", "s", "a"):
                h.update(np.round(_arr(e[k]), 4).tobytes())
        for c in net.get("crossings", []):
            h.update(repr((round(float(c["x"]), 3), round(float(c["y"]), 3), c.get("e1"), c.get("e2"))).encode())
    if out.get("od_render") is not None:
        h.update(np.round(np.asarray(out["od_render"], np.float64), 4).tobytes())
    return h.hexdigest()[:16]


# ======================================================================== true networks (self-check, oracle)
def true_network_observable(case: dict) -> dict:
    """SELF-CHECK: the observable image graph as a contract network: its edges (1 px run samples) with the
    true r, s, a of their vessel's nearest profile sample, its nodes (junctions typed type_observable)."""
    obs, prof = case["obs"], case["prof"]
    pxy = np.stack([prof["x_px"], prof["y_px"]], 1).astype(float)
    vid = np.repeat(prof["vids"], np.diff(prof["offsets"]))
    sm = case["samples"]
    a_tree = cKDTree(sm["xy"])
    trees = {}
    nodes = [dict(id=int(n["id"]), x=float(n["x"]), y=float(n["y"]),
                  kind=n.get("type_observable") if n["kind"] == "junction" else n["kind"]) for n in obs["graph"]["nodes"]]
    edges = []
    for e in obs["graph"]["edges"]:
        xy = np.asarray(e["xy"], float).reshape(-1, 2)
        v = int(e["vid"])
        if v not in trees:
            ii = np.flatnonzero(vid == v)
            trees[v] = (cKDTree(pxy[ii]), ii)
        tr, ii = trees[v]
        k = ii[tr.query(xy)[1]]
        _, ka = a_tree.query(xy)                       # a: the nearest observable true sample's (od_ratio applied)
        edges.append(dict(u=int(e["source"]), v=int(e["target"]), xy=xy, r=prof["radius_px"][k].astype(float),
                          s=prof["blur_px"][k].astype(float), a=sm["a"][ka]))
    return dict(nodes=nodes, edges=edges, crossings=[])


def true_vesselnetwork(folder: str, kind: str, crop=None):
    """ORACLE: the complete truth as a vesselmap VesselNetwork (scene.to_vesselnetwork with the scene's own
    optics and formation blur; halo in meta['optics']), in the coordinates of crop (x0, y0, w, h) if given."""
    from experiments.splinefit import use_vesselmap
    use_vesselmap()
    from vesselscene.scene import to_vesselnetwork
    sj = _scene_json(folder)
    key = "optics_frame" if kind == "frame" and "optics_frame" in sj else "optics"
    g = VesselGraph.load(os.path.join(folder, "truth.json"))
    k = float(sj["k"])
    origin = np.asarray(sj.get("origin", (0.0, 0.0)), float)
    shape = tuple(sj["shape"])
    if crop is not None:
        origin = origin + np.array(crop[:2], float) / k
        shape = (int(crop[3]), int(crop[2]))
    return to_vesselnetwork(g, k, tuple(origin), shape, optics=VR.Optics(**sj[key]),
                            blur_px=float(sj["formed"][kind]["blur_px"]))


def vesselmap_to_contract(net, node_kinds: dict | None = None, crossings=None, step: float = 0.5) -> dict:
    """A vesselmap VesselNetwork as a contract network: edges sampled every `step` px; node kinds from
    node_kinds (id -> kind) else from the degree (1 'end', 2 'joint', 3 'branch', 4+ 'compound')."""
    deg = net.degrees()
    nodes = []
    for nid in sorted(net.nodes):
        n = net.nodes[nid]
        d = deg[nid]
        kind = (node_kinds or {}).get(nid) or {0: "end", 1: "end", 2: "joint", 3: "branch"}.get(d, "compound")
        nodes.append(dict(id=int(nid), x=float(n.x), y=float(n.y), kind=kind))
    edges = []
    for eid in sorted(net.edges):
        smp = net.sample(eid, step)
        e = net.edges[eid]
        edges.append(dict(id=int(eid), u=int(e.u), v=int(e.v), xy=smp["xy"], r=smp["r"], s=smp["s"], a=smp["a"]))
    return dict(nodes=nodes, edges=edges, crossings=list(crossings or []))


def true_network_complete(folder: str, kind: str, shape, crop=None) -> dict:
    """ORACLE: the complete true network as a contract (the oracle-init start): one edge per vessel near the
    frame, nodes typed by flow (1 in 2 out 'bifurcation', 2 in 1 out 'confluence', else by degree), the
    in-frame crossings of crossings.json between exported vessels; plus 'vesselnetwork' (the VesselNetwork).
    shape: the image's (or the crop's) (H, W); crop (x0, y0, w, h) shifts everything into the crop."""
    net = true_vesselnetwork(folder, kind, crop)
    vid2e = {int(e.info["true_vessel"]): eid for eid, e in net.edges.items()}
    kinds = {}
    for nid in net.nodes:
        inc = net.incident(nid)
        n_in = sum(1 for _, end in inc if end == 1)
        if len(inc) == 3:
            kinds[nid] = "bifurcation" if n_in == 1 else "confluence" if n_in == 2 else "branch"
    with open(os.path.join(folder, "crossings.json"), encoding="utf-8") as fh:
        crs = json.load(fh)
    Hh, Ww = int(shape[0]), int(shape[1])
    x0, y0 = (float(crop[0]), float(crop[1])) if crop is not None else (0.0, 0.0)
    cr = [dict(x=float(c["x_px"]) - x0, y=float(c["y_px"]) - y0, e1=int(vid2e[c["vessels"][0]]),
               e2=int(vid2e[c["vessels"][1]]))
          for c in crs if all(int(v) in vid2e for v in c["vessels"])
          and -0.5 <= c["x_px"] - x0 <= Ww - 0.5 and -0.5 <= c["y_px"] - y0 <= Hh - 0.5]
    out = vesselmap_to_contract(net, kinds, cr)
    out["vesselnetwork"] = net
    return out
