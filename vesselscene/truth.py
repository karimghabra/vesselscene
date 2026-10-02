"""Ground truth per image: what a careful person could observe (OBSERVABLE) and everything (COMPLETE).

The user's decision (2026-10-01): annotate only what one could realistically see and measure in the image;
optimise a spline-network fitting routine against everything.  So each image of a scene (the average still and
the single frame) has two ground truths:

    observable_truth(src, kind)          the annotation target of THIS image (dict, JSON-able): observable
                                         runs per vessel, observable junctions, the observable image graph
    complete_truth(src, kinds)           the full VesselGraph, all junctions (hidden ones too) and per-vessel
                                         per-sample visibility profiles, to score a fit against everything
    observable_rasters(obs, profiles)    centreline / edge ids, lumen, don't-care, junction heatmaps by type
    draw_observable / draw_complete      overlays;  figure_truth: image | observable | complete, full + zooms
    save_observable / load_observable / save_graphml / to_networkx / save_profiles / load_profiles
    check_observable(obs, prof)          consistency problems (none expected)
    score_tracing(...), score_fit(...), traced_by_cnr(...)   how an annotation / a fit is scored, and the
                                         data to recalibrate CNR_OBSERVE on human tracings

src is a make_scene() Scene or a save_scene() folder.  Units: image px (pixel (0, 0) at the d_c position
`origin`, k px per d_c), arclength along a vessel in px (s_px) and d_c (s_dc), from its upstream end.

Definitions
-----------
Samples and CNR.  Every vessel within junctions.PAD_PX of the frame is sampled every px of arclength
(junctions._Samples).  A sample's CNR is the image-junction truth's: scene.visibility's band-matched CNR (the
vessel's own DoG-band response over the background's RMS in that band) times the ratio of the contrast the
rendered OD shows to the closed form's, measured at the vessel's isolated samples (a frame's red-cell gaps, an
unperfused capillary, a vessel lost under a black vein).

OBSERVABLE samples: CNR >= cnr_observe (CNR_OBSERVE = 3) inside the frame, in contiguous in-frame runs of at
least min_run_px (MIN_RUN_PX = 1 lambda = 11.9 px), gaps shorter than bridge_px (BRIDGE_PX = lambda / 2)
bridged.  DON'T CARE samples: seen under junctions.py's permissive rule (CNR >= CNR_VISIBLE = 1.5, gaps < 1
lambda bridged, runs >= lambda / 2) but not observable: an annotator is neither rewarded nor penalised there.

Merged (one line in the image).  Two observable vessels are unresolved where their walls are closer than the
resolution gap (junctions.g_res: a 25 % dip between them).  Outside the junctions' claims, and not where a
parent runs into its own child at their node, the less prominent of the two (lower median own contrast over
the stretch; then the thinner) is MERGED into the other's line there: its samples belong to the dominant line
(chains followed), the edge of that line carries both vessel ids.  Only between vessels of similar scale (total
blurs within MERGE_BLUR_RATIO = 2): a sharp thin vessel on a blurred wide one is seen as its own line (the
dip rule, made for similar profiles, called a thin vessel 'unresolved' up to ~2.4 x the blurred one's sigma
from its wall, and cut one visible junction on a blurred vessel into merges spread 20-50 px along it).  A merge
junction sits on the dominant line's axis at the sample nearest to where the weaker line becomes unresolved.

Observable junctions.  The candidates are the image junctions of junctions.py computed with the observable
rules (seen = observable, keep_hidden): node, crossing, touching and fading-end (pseudo-T) events joined into
junctions, each holding a claim (a stretch) on each of its vessels (one junction's claims on a vessel closer
than ARM_MIN_PX are one claim; where junctions.py's 1-D split cut a claim A in two around a claim B with no gap,
A B A, the piece of A farther from A's centre goes to B).  Each vessel's LINE is its observable, unmerged
samples, inside the claims too, plus the claims its line reaches (within bridge_px): through a claim when the
line goes on beyond it (observable inside or not: a thin vessel under a black vein), else up to its last
observable sample in it (or the claim's first sample: it ends at the junction).  A line piece lying wholly
inside one junction's claim is INTERNAL to it (internal_vessels: part of the junction, no line leaves it).  A
candidate is an observable junction when at least 3 lines leave it (its degree in the image graph; a line piece
shorter than ARM_MIN_PX = lambda / 2 ending in a fade or a merge is a stub: absorbed, not a line; nothing longer
is ever absorbed).  Candidates with <= 2 lines release their claims and the lines pass through (repeated until
stable).  Lines that become unresolved from a stronger line and part from it again away from any junction make
a MERGE junction where they meet / part.  type_observable, from the lines (v2 junction audit: from the
observable arm pattern, every observable vessel through the region with its lines, no node-without-arm
shortcut): 3 lines -> 'bifurcation' / 'confluence' when they are 3 vessels of one node of the junction (by how
many flow in), otherwise 'pseudo-T' (a merge of two siblings / co-parents: bifurcation / confluence); 4 lines
-> 'crossing' when they pair up into two through lines (each 'in' line with an 'out' line of the same vessel,
or of a child of it at a node of the junction: a parent running on into its only observable daughter; a line
stands for every vessel id it carries), otherwise 'compound'; 5 or more -> 'compound'.  A junction's
hidden_vessels are the vessels of its events not observable in it.

Observable runs.  Per vessel, the stretches of its line (claims of observable junctions included: a run passes
through a junction; 'unobserved': its samples that are not observable themselves, only inside junctions), all
inside the frame, each end labelled by the kind of the image-graph node it ends at: 'junction' (an observable
junction), 'fade' (its CNR falls below the threshold; also a vessel ending at a node where nothing observable
continues), 'border' (it leaves the frame), 'merged' (it becomes unresolved into a stronger line, away from any
junction), 'hidden_junction' (the line continues in another vessel there: at a true node that is no observable
junction, a parent into its only observable daughter, or where a merged vessel takes the line over);
merged_with names the line a vessel merges into whatever its end's label.

Observable image graph (the annotation target).  Nodes: the observable junctions and the run ends ('fade',
'border', 'merged') and 'hidden_junction' nodes (degree 2: the line changes vessel at a true node that is not
observable; also where a merged vessel takes over a line; splitting a traced line there is optional: don't
care).  Edges: the runs split at the observable junctions, each with its vessel id, every vessel id its line
carries ('vids'), class and its px polyline (1 px samples) from where it leaves one node to where it reaches
the next: at a junction, the run's sample nearest the junction's centre (its attachment), so the edges cover
the runs; 'linked' where the vessel is not resolved between two junctions (their claims touch).  A crossing
whose other vessel is not observable makes no node: the line passes through.  A junction's radius is its
region's (junctions.py: up to the widest lumen + 2 sigma around each event): ~2 lambda where thin vessels cross
a wide vein, larger among the pathologic preset's 50-120 px vessels.

COMPLETE truth: the VesselGraph (truth.json), all junctions under the permissive rules including hidden ones
(junctions_all_<kind>.json: junctions.image_junctions(keep_hidden=True)) and per-vessel per-sample visibility
profiles (complete_visibility_<kind>.npz: arclength, position, CNR, observable / don't-care / in-frame flags,
radius, blur, contrast).
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from . import junctions as J
from . import render as R
from .graph import VesselGraph

LAMBDA_PX = J.LAMBDA_PX            # 11.9 px: the real stills' median vessel FWHM
CNR_OBSERVE = 3.0                  # a stretch is observable where its CNR reaches this ...
MIN_RUN_PX = LAMBDA_PX             # ... in a contiguous in-frame run at least this long (px)
BRIDGE_PX = 0.5 * LAMBDA_PX        # gaps shorter than this (px) are bridged
ARM_MIN_PX = 0.5 * LAMBDA_PX       # a line piece shorter than this beyond a junction, ending in a fade: a stub
CNR_DONT_CARE = J.CNR_VISIBLE      # 1.5: the permissive 'seen' level; seen but not observable = don't care
DONT_CARE_MARGIN_PX = 2.0          # the don't-care mask: the lumen of the don't-care samples + this
MERGE_BLUR_RATIO = 2.0             # two unresolved vessels are one line only if their total blurs are within this
                                   # ratio: a sharp thin vessel on a blurred wide one is seen as its own line
OBS_TYPES = ("bifurcation", "confluence", "pseudo-T", "crossing", "compound")
END_LABELS = ("junction", "fade", "border", "merged", "hidden_junction")
NODE_KINDS = ("junction", "fade", "border", "merged", "hidden_junction")
FORMAT = "vesselscene-observable"
VERSION = 2                        # 2: the truth review's fixes (lines inside claims, unobserved, internal_vessels,
                                   # edges from attachment to attachment, labels from node kinds)


def rules(cnr_observe=CNR_OBSERVE, min_run_px=MIN_RUN_PX, bridge_px=BRIDGE_PX, arm_min_px=ARM_MIN_PX) -> dict:
    return dict(cnr_observe=float(cnr_observe), min_run_px=float(min_run_px), bridge_px=float(bridge_px),
                arm_min_px=float(arm_min_px), cnr_dont_care=float(CNR_DONT_CARE), lambda_px=LAMBDA_PX,
                sample_px=J.SAMPLE_PX, dont_care_margin_px=DONT_CARE_MARGIN_PX, obs_types=list(OBS_TYPES),
                end_labels=list(END_LABELS), version=VERSION)


# ======================================================================== inputs
class _Ctx:
    """What one image's truth is built from: the graph, the samples (with CNR), crossings, the image, the
    complete junctions (if at hand) and where the noise came from."""

    def __init__(self, graph, S, crossings=None, image=None, junctions_all=None, noise_source="exact",
                 folder=None, cnr_from_profiles=False):
        self.graph, self.S, self.crossings, self.image = graph, S, crossings, image
        self.junctions_all, self.noise_source, self.folder = junctions_all, noise_source, folder
        self.cnr_from_profiles = cnr_from_profiles


def _is_scene(src) -> bool:
    return hasattr(src, "graph") and hasattr(src, "params") and hasattr(src, "images")


def scene_context(scene, kind: str, samples=None) -> _Ctx:
    S = samples if samples is not None else J.scene_samples(scene, kind)
    ja = getattr(scene, "junctions_all", {}) or {}
    return _Ctx(scene.graph, S, scene.crossings or None, scene.images.get(kind), ja.get(kind), "exact")


def _samples_from_profiles(graph, sj, kind, prof):
    """junctions._Samples rebuilt from the saved geometry (truth.json, scene.json) with the CNR and contrast of
    complete_visibility_<kind>.npz: exact, without the render or the noise estimate.  Raises ValueError if they
    do not match (another graph or sampling)."""
    key = "optics_frame" if kind == "frame" and "optics_frame" in sj else "optics"
    optics = R.Optics(**sj[key])
    k = float(sj["k"])
    origin = np.asarray(sj.get("origin", [0.0, 0.0]), float)
    shape = tuple(int(x) for x in sj["shape"])
    blur_px = float(sj["formed"][kind]["blur_px"])
    meta = prof["meta"]
    S = J._Samples(graph, k, origin, shape, optics, blur_px, None, float(meta.get("pad_px", J.PAD_PX)))
    vids, offs = prof["vids"], prof["offsets"]
    pos = {int(v): i for i, v in enumerate(vids)}
    if set(pos) != set(S.v):
        raise ValueError("the saved profiles cover other vessels")
    for v, d in S.v.items():
        i = pos[v]
        a, b = int(offs[i]), int(offs[i + 1])
        if b - a != d["n"]:
            raise ValueError(f"vessel {v}: {b - a} saved samples, {d['n']} now")
        d["cnr"] = prof["cnr"][a:b].astype(np.float64)
        d["cnr_cf"] = prof["cnr_closed_form"][a:b].astype(np.float64)
        d["ratio"] = prof["od_ratio"][a:b].astype(np.float64)
        d["peak"] = prof["peak_od"][a:b].astype(np.float64)
        d["contrast_own"] = prof["contrast"][a:b].astype(np.float64)
        d["contrast"] = prof["contrast_image"][a:b].astype(np.float64)
        d["dfl"] = d["r"] + 2.5 * d["s"] + 1.0
    S.set_seen()
    S.k, S.origin, S.kind, S.optics, S.blur_px = k, origin, kind, optics, blur_px
    S.band_rms = {float(b_): float(v_) for b_, v_ in (meta.get("band_rms") or {}).items()}
    S.has_od = bool(meta.get("has_od", True))
    return S


def folder_context(folder: str, kind: str, noise=None, use_profiles: bool = True) -> _Ctx:
    """The inputs of a save_scene() folder.  The per-sample CNR comes from complete_visibility_<kind>.npz when
    it is there (exactly what save_scene computed: thresholds can change without the render); otherwise from
    junctions.inputs_from_saved (the rendered OD of labels.npz; the background noise estimated from the still,
    0.84-1.09x the exact value; no red-cell filling, the OD ratio still catches a frame's gaps)."""
    import tifffile
    g = VesselGraph.load(os.path.join(folder, "truth.json"))
    with open(os.path.join(folder, "scene.json"), encoding="utf-8") as fh:
        sj = json.load(fh)
    crs = None
    pc = os.path.join(folder, "crossings.json")
    if os.path.exists(pc):
        with open(pc, encoding="utf-8") as fh:
            crs = json.load(fh)
    pi = os.path.join(folder, f"still_{kind}.tif")
    image = tifffile.imread(pi).astype(np.float32) if os.path.exists(pi) else None
    ja = None
    pj = os.path.join(folder, f"junctions_all_{kind}.json")
    if os.path.exists(pj):
        ja = J.load_junctions(pj)
    S, src, src_prof = None, None, False
    pp = os.path.join(folder, f"complete_visibility_{kind}.npz")
    if use_profiles and noise is None and os.path.exists(pp):
        try:
            prof = load_profiles(pp)
            S = _samples_from_profiles(g, sj, kind, prof)
            src = prof["meta"].get("noise_source", "exact")       # (the CNR read from the profiles is exact)
            src_prof = True
        except ValueError:
            S = None
    if S is None:
        kw, _ = J.inputs_from_saved(folder, kind, noise=noise)
        kw.pop("crossings", None)
        S = J.junction_samples(**kw)
        src = "given" if noise is not None else "estimated"
    return _Ctx(g, S, crs, image, ja, src, folder, cnr_from_profiles=S is not None and src_prof)


def _context(src, kind, samples=None, noise=None) -> _Ctx:
    if isinstance(src, _Ctx):
        return src
    if _is_scene(src):
        return scene_context(src, kind, samples)
    return folder_context(str(src), kind, noise=noise)


# ======================================================================== helpers
def _seen_mask(d, thr, bridge_px, min_px, in_frame_only) -> np.ndarray:
    """junctions._Samples.set_seen's rule for one vessel, without changing the samples."""
    raw = d["cnr"] >= thr
    if in_frame_only:
        raw = raw & d["inf"]
    ok = J._bridge(raw, max(1, int(round(bridge_px / d["ds"]))))
    if in_frame_only:
        ok &= d["inf"]
    seen = np.zeros(d["n"], bool)
    for a, b in J._segments(ok):
        if (b - a) * d["ds"] >= min_px:
            seen[a:b + 1] = True
    return seen


def _r(x, nd=2):
    return [round(float(v), nd) for v in np.asarray(x, float)]


def _rs(x, sig=4):
    """Rounded to `sig` significant digits (JSON size)."""
    out = []
    for v in np.asarray(x, float):
        if not np.isfinite(v):
            out.append(None)
        elif v == 0:
            out.append(0.0)
        else:
            out.append(float(f"{v:.{sig}g}"))
    return out


class _G:
    """Global (concatenated) sample arrays of the vessels of S."""

    def __init__(self, S):
        self.vids = list(S.v)
        self.n = np.array([S.v[v]["n"] for v in self.vids], np.int64)
        self.off = np.r_[0, np.cumsum(self.n)].astype(np.int64)
        self.N = int(self.off[-1])
        mx = max(self.vids, default=0) + 1
        self.vix = np.full(mx, -1, np.int64)
        self.vix[self.vids] = np.arange(len(self.vids))
        self.gvid = np.repeat(np.array(self.vids, np.int64), self.n)

    def cat(self, S, key):
        return np.concatenate([np.asarray(S.v[v][key]) for v in self.vids]) if self.vids else np.zeros(0)

    def o(self, v) -> int:
        return int(self.off[self.vix[v]])


# ======================================================================== the observable truth
def observable_truth(src, kind: str = "average", cnr_observe: float = CNR_OBSERVE, min_run_px: float = MIN_RUN_PX,
                     *, bridge_px: float = BRIDGE_PX, arm_min_px: float = ARM_MIN_PX, samples=None, noise=None,
                     return_profiles: bool = False):
    """The observable truth of one image (module docstring).  src: a make_scene() Scene, a save_scene() folder
    or a context (scene_context / folder_context).  samples: junctions.scene_samples(scene, kind) to reuse.
    Returns a JSON-able dict:
      format, version, kind, shape, k, origin, rules (+ noise_source, band_rms),
      runs: [id, vid, cls, s0_px, s1_px, s0_dc, s1_dc, length_px, start / end (label, node: image-graph node,
        junction: observable junction id or None, merged_with: vid or None, vessel_node: graph node or None,
        xy), junctions (observable junction ids passed, in order), merged (other vessels merged into this
        line: vid, s0_px, s1_px on this vessel), unobserved ([k0, k1] index ranges into xy: on the line
        through a junction, not observable itself), xy (1 px polyline), width_px, blur_px, contrast, cnr],
      junctions: [id, node, x, y, radius, source ('event' / 'merge'), type (the truth: junctions.TYPES or
        'merge'), type_observable, n_lines, lines (edge, vid, vids, flow 'in' / 'out', xy, dir, angle_deg,
        width_px, blur_px, cnr, contrast), members (the truth events, as junctions.py), vessels, hidden_vessels
        (vessels of its events not observable in it), internal_vessels (observable only inside it),
        ambiguous_reasons, complete_ids],
      hidden_junctions: [node, x, y, vids, vessel_node, truth_type ('bifurcation', 'confluence',
        'anastomosis' for a graph node, 'merge'), degree],
      dont_care_junctions: [x, y, radius, source ('hidden_junction' / 'complete'), ref],
      merged: [vid, into, s0_px, s1_px, s0_dc, s1_dc]: where a vessel is unresolved into a stronger line,
      graph: node-link data (networkx node_link_data, edges="edges"): nodes (id, kind, x, y, degree, junction,
        type_observable, vids, vessel_node, truth_type), edges (source, target, key, id, vid, vids, cls, run,
        s0_px, s1_px, s0_dc, s1_dc, length_px, linked, xy: from attachment / run end to attachment / run end),
      summary (summary(obs)).
    return_profiles: also the per-sample profiles (visibility_profiles) -> (obs, profiles)."""
    ctx = _context(src, kind, samples, noise)
    obs, prof = _build(ctx, kind, float(cnr_observe), float(min_run_px), float(bridge_px), float(arm_min_px))
    return (obs, prof) if return_profiles else obs


_F32_KEYS = ("cnr", "cnr_cf", "ratio", "peak", "contrast_own", "contrast")


def _quantize(S):
    """The per-sample visibility rounded to float32, as complete_visibility_<kind>.npz stores it (in place,
    idempotent): the truth of a scene and its recomputation from the saved profiles (the CLI) start from the same
    numbers and are the same, byte for byte (review: 423 junction-line values and the noise source differed)."""
    for d in S.v.values():
        for key in _F32_KEYS:
            if key in d:
                d[key] = np.asarray(d[key], np.float32).astype(np.float64)


def _build(ctx: _Ctx, kind, cnr_observe, min_run_px, bridge_px, arm_min_px):
    S, graph = ctx.S, ctx.graph
    _quantize(S)
    H, W = S.shape
    k = float(S.k)
    # ---- candidates: the image junctions under the observable rules (seen = observable)
    internals = {}
    cand = J.junctions_from_samples(S, graph, crossings=ctx.crossings, keep_hidden=True, cnr_seen=cnr_observe,
                                    cnr_visible=cnr_observe, seen_bridge_px=bridge_px, seen_min_px=min_run_px,
                                    in_frame_only=True, internals=internals)
    G = _G(S)
    N = G.N
    obs = G.cat(S, "seen").astype(bool) if N else np.zeros(0, bool)
    inf = G.cat(S, "inf").astype(bool) if N else np.zeros(0, bool)
    prom = np.nan_to_num(G.cat(S, "contrast_own").astype(np.float64), nan=0.0) if N else np.zeros(0)
    rad = G.cat(S, "r").astype(np.float64) if N else np.zeros(0)
    blur = G.cat(S, "st").astype(np.float64) if N else np.zeros(0)
    Bk = max(1, int(round(bridge_px / J.SAMPLE_PX)))
    cj = {j["id"]: j for j in cand}
    cen = {j["id"]: np.array([j["x"], j["y"]], float) for j in cand}
    node_cand = {}                                                   # graph node -> the candidate holding it
    for j in cand:
        for m in j["members"]:
            if "node" in m:
                node_cand[int(m["node"])] = j["id"]
    # claims (global sample ranges) of every candidate
    claims = []                                                      # (jid, vid, g0, g1)
    for j in cand:
        for vs, lst in (j.get("claim_px") or {}).items():
            v = int(vs)
            if v not in S.v:
                continue
            ds, o_ = S.v[v]["ds"], G.o(v)
            for lo, hi in lst:
                claims.append((j["id"], v, o_ + int(round(lo / ds)), o_ + int(round(hi / ds))))
    # ---- unresolved records of the observable samples, minus a parent running into its own child at their node
    rec = np.asarray(internals.get("rec_res", np.zeros((0, 4), np.int64)), np.int64)
    if len(rec):
        excl = np.zeros(len(rec), bool)
        runs_ = J._Runs(rec)
        M = int(max(graph.vessels, default=0)) + 1
        code = rec[:, 0] * M + rec[:, 1]
        o = np.argsort(code, kind="stable")
        cs = code[o]
        for nid in graph.nodes:
            par = [v for v in graph.in_vessels(nid) if v in S.v]
            chi = [v for v in graph.out_vessels(nid) if v in S.v]
            for p in par:
                n_p = S.v[p]["n"]
                for c in chi:
                    if not (runs_.by_pair.get((p, c)) or runs_.by_pair.get((c, p))):
                        continue
                    lo_p, _ = runs_.from_end(p, c, 1, n_p, max(S.margin(p, n_p - 1), Bk))
                    _, hi_c = runs_.from_end(c, p, 0, S.v[c]["n"], max(S.margin(c, 0), Bk))
                    # unresolved is measured across each vessel: the two directions' extents differ by a sample
                    # or two; take both, and Bk samples more (the line runs on through its own node)
                    for (x, y), (ix, iy) in (((p, c), (2, 3)), ((c, p), (3, 2))):
                        a0, a1 = np.searchsorted(cs, x * M + y), np.searchsorted(cs, x * M + y, side="right")
                        if a1 > a0:
                            sel = o[a0:a1]
                            near_p = rec[sel, ix] >= lo_p
                            near_c = rec[sel, iy] <= hi_c
                            if (near_p).any():
                                hi_c = max(hi_c, int(rec[sel[near_p], iy].max()))
                            if (near_c).any():
                                lo_p = min(lo_p, int(rec[sel[near_c], ix].min()))
                    lo_p, hi_c = lo_p - Bk, hi_c + Bk
                    for (x, y), (ix, iy) in (((p, c), (2, 3)), ((c, p), (3, 2))):
                        a0, a1 = np.searchsorted(cs, x * M + y), np.searchsorted(cs, x * M + y, side="right")
                        if a1 > a0:
                            sel = o[a0:a1]
                            excl[sel] |= (rec[sel, ix] >= lo_p) & (rec[sel, iy] <= hi_c)
        rec = rec[~excl]
        a_, b_ = rec[:, 0], rec[:, 1]
        sw = a_ > b_
        rec = np.where(sw[:, None], rec[:, [1, 0, 3, 2]], rec)
        ga = G.off[G.vix[rec[:, 0]]] + rec[:, 2]
        gb = G.off[G.vix[rec[:, 1]]] + rec[:, 3]
    else:
        ga = gb = np.zeros(0, np.int64)
    # ---- iterate: lines, edges, degrees; candidates with <= 2 lines release their claims (and the lines pass
    # through); short pieces dangling into a fade are absorbed (stubs, < arm_min_px: never a longer stretch)
    active = {j["id"] for j in cand if j.get("claim_px")}
    absorbed = np.zeros(N, bool)
    for _it in range(50):
        st = _state(S, G, obs, inf, prom, rad, blur, claims, active, ga, gb, rec, absorbed, Bk, arm_min_px, cen)
        changed = False
        new_abs = st["stub_samples"] & ~absorbed
        if new_abs.any():
            absorbed |= new_abs
            changed = True
        drop = {jid for jid in active if st["degree"].get(jid, 0) <= 2}
        if drop:
            active -= drop
            changed = True
        if not changed:
            break
    else:                                              # not converged (never seen): the state of the last rules
        st = _state(S, G, obs, inf, prom, rad, blur, claims, active, ga, gb, rec, absorbed, Bk, arm_min_px, cen)
    obs_d = _assemble(ctx, kind, st, cand, cj, node_cand, S, G, graph, obs, inf, rad, absorbed, Bk, k, (H, W),
                      cnr_observe, min_run_px, bridge_px, arm_min_px)
    prof = visibility_profiles(S, G, obs, st, absorbed, kind, k, cnr_observe, min_run_px, bridge_px,
                               ctx.noise_source)
    ds = np.concatenate([np.full(S.v[v]["n"], S.v[v]["ds"]) for v in G.vids]) if N else np.zeros(0)
    dc = prof["dont_care"]
    obs_d["totals"] = dict(observable_px=round(float(ds[obs].sum()), 1),
                           n_vessels_observable=int(len(set(G.gvid[obs].tolist()))),
                           dont_care_px=round(float(ds[dc].sum()), 1),
                           in_frame_px=round(float(ds[inf].sum()), 1),
                           line_px=round(float(ds[st["line"]].sum()), 1),
                           merged_px=round(float(ds[(st["mdom"] >= 0) & obs & ~absorbed].sum()), 1),
                           absorbed_px=round(float(ds[absorbed & obs].sum()), 1),
                           internal_px=round(float(ds[st["internal_s"] & obs].sum()), 1),
                           unobserved_line_px=round(float(ds[st["line"] & ~obs].sum()), 1))
    obs_d["summary"] = summary(obs_d, ctx.junctions_all)
    return obs_d, prof


def _owners(S, G, claims, active, arm_min_px, cen):
    """The active junction owning each sample (global; -1: none) and per vessel its claim stretches [(g0, g1,
    jid)] (global sample indices, in order)."""
    owner = np.full(G.N, -1, np.int64)
    for jid, v, g0, g1 in claims:
        if jid in active:
            owner[g0:g1 + 1] = jid
    # one junction's claims on a vessel closer than arm_min_px (a one-sample gap is common) are one claim: else
    # the gap is a junction-to-itself edge of a few px, counted twice in its degree (review D3: X's typed compound)
    segs_v = {}
    for v in G.vids:
        o_, n_ = G.o(v), S.v[v]["n"]
        ow = owner[o_:o_ + n_]
        if not (ow >= 0).any():
            continue
        sg = _owner_segments(ow)
        gap_max = arm_min_px / S.v[v]["ds"]
        for (a0, a1, j0), (b0, b1, j1) in zip(sg, sg[1:]):
            if j0 == j1 and 0 < b0 - a1 - 1 < gap_max:
                ow[a1 + 1:b0] = j0
        # A B A with no gap: junctions.py's 1-D split of overlapping claims cut A's claim in two around B's; the
        # vessel never leaves the pair, so it would go A -> B -> A, two lines of one vessel at A for nothing.  The
        # piece of A farther from A's centre goes to B (the vessel leaves the pair from B there)
        C = S.v[v]["C"]
        for _ in range(len(sg)):
            sg = _owner_segments(ow)
            hit = None
            for k_ in range(len(sg) - 2):
                (a0, a1, ja), (b0, b1, jb), (c0, c1, jc) = sg[k_:k_ + 3]
                if ja == jc != jb and a1 + 1 == b0 and b1 + 1 == c0:
                    hit = (a0, a1, c0, c1, ja, jb)
                    break
            if hit is None:
                break
            a0, a1, c0, c1, ja, jb = hit
            da = float(np.hypot(*(C[a0:a1 + 1] - cen[ja]).T).min())
            dc = float(np.hypot(*(C[c0:c1 + 1] - cen[ja]).T).min())
            if da > dc:
                ow[a0:a1 + 1] = jb
            else:
                ow[c0:c1 + 1] = jb
        segs_v[v] = [(o_ + a, o_ + b, jid) for a, b, jid in _owner_segments(ow)]
    return owner, segs_v


def _state(S, G, obs, inf, prom, rad, blur, claims, active, ga, gb, rec, absorbed, Bk, arm_min_px, cen) -> dict:
    """One pass of the line model for the current active junctions and absorbed samples."""
    N = G.N
    owner, segs_v = _owners(S, G, claims, active, arm_min_px, cen)
    # ---- merged: the weaker of two unresolved observable vessels outside the claims, per stretch of the pair
    mdom = np.full(N, -1, np.int64)
    mprom = np.full(N, -np.inf)
    if len(ga):
        keep = (owner[ga] < 0) & (owner[gb] < 0) & obs[ga] & obs[gb]
        A, B = rec[keep, 0], rec[keep, 1]
        I, Jx = rec[keep, 2], rec[keep, 3]
        gA, gB = ga[keep], gb[keep]
        if len(A):
            o = np.lexsort((I, B, A))
            A, B, I, Jx, gA, gB = A[o], B[o], I[o], Jx[o], gA[o], gB[o]
            brk = np.flatnonzero((np.diff(A) != 0) | (np.diff(B) != 0) | (np.diff(I) > Bk)) + 1
            for grp in np.split(np.arange(len(A)), brk):
                a, b = int(A[grp[0]]), int(B[grp[0]])
                ba, bb = float(np.median(blur[gA[grp]])), float(np.median(blur[gB[grp]]))
                if max(ba, bb) > MERGE_BLUR_RATIO * min(ba, bb):
                    continue                                # different scales: both lines are seen
                pa, pb = float(np.median(prom[gA[grp]])), float(np.median(prom[gB[grp]]))
                wa, wb = float(np.median(rad[gA[grp]])), float(np.median(rad[gB[grp]]))
                if (pa, wa, -a) < (pb, wb, -b):
                    sub_i, dom_i, sub, dom = I[grp], Jx[grp], a, b
                else:
                    sub_i, dom_i, sub, dom = Jx[grp], I[grp], b, a
                rng = np.arange(int(sub_i.min()), int(sub_i.max()) + 1)
                # each merged sample's partner: the nearest sample of the dominant line (where a merge junction
                # will sit), among the samples it is unresolved from (+- Bk)
                d0, d1 = max(int(dom_i.min()) - Bk, 0), min(int(dom_i.max()) + Bk, S.v[dom]["n"] - 1)
                Cs, Cd = S.v[sub]["C"][rng], S.v[dom]["C"][d0:d1 + 1]
                if len(Cs) * len(Cd) <= 4_000_000:
                    dd = (Cs[:, None, 0] - Cd[None, :, 0]) ** 2 + (Cs[:, None, 1] - Cd[None, :, 1]) ** 2
                    pt = d0 + np.argmin(dd, 1)
                else:
                    u, inv = np.unique(sub_i, return_inverse=True)
                    part = np.bincount(inv, weights=dom_i) / np.bincount(inv)
                    pt = np.rint(np.interp(rng, u, part)).astype(np.int64)
                gs = G.o(sub) + rng
                gp = G.o(dom) + np.clip(pt, 0, S.v[dom]["n"] - 1)
                ok = obs[gs] & (owner[gs] < 0)
                better = ok & (prom[gp] > mprom[gs])
                mdom[gs[better]] = gp[better]
                mprom[gs[better]] = prom[gp[better]]
    for _ in range(12):                                             # chains: the line the dominant belongs to
        m = mdom >= 0
        nxt = np.where(m, mdom, 0)
        go = m & (mdom[nxt] >= 0)
        if not go.any():
            break
        mdom = np.where(go, mdom[nxt], mdom)
    m = mdom >= 0
    cyc = m & (mdom[np.where(m, mdom, 0)] >= 0)                     # an unresolved cycle: keep these as own lines
    mdom[cyc] = -1
    # ---- lines: every observable, unmerged sample is on its vessel's line, inside the junctions' claims too (review
    # D1: a vessel lying wholly in a chain of claims, e.g. a black vein crossed every 30 px, had no line at all)
    base = obs & (mdom < 0) & ~absorbed
    line = base.copy()
    # a line reaching a claim (within Bk samples) enters it: through it when it goes on beyond (whether or not the
    # vessel is observable inside: a thin vessel crossing under a black vein), else up to its last observable sample
    # in it (or the claim's first sample: it ends at the junction, as junctions.py's pseudo-T)
    fill = np.zeros(N, bool)
    for v, lst in segs_v.items():
        o_, n_ = G.o(v), S.v[v]["n"]
        grown = True
        while grown:
            grown = False
            for g0, g1, jid in lst:
                if line[g0:g1 + 1].all():
                    continue
                lo_b, hi_b = max(g0 - Bk, o_), min(g1 + Bk, o_ + n_ - 1)
                left = np.flatnonzero(line[lo_b:g0])
                right = np.flatnonzero(line[g1 + 1:hi_b + 1])
                if not (len(left) or len(right)):
                    continue
                ins = np.flatnonzero(line[g0:g1 + 1])
                if len(left) and len(right):
                    a0, a1 = lo_b + int(left[-1]) + 1, g1 + int(right[0])
                elif len(left):
                    a0, a1 = lo_b + int(left[-1]) + 1, g0 + (int(ins[-1]) if len(ins) else 0)
                else:
                    a0, a1 = g0 + (int(ins[0]) if len(ins) else g1 - g0), g1 + int(right[0])
                seg = slice(a0, a1 + 1)
                ok = inf[seg] & (~absorbed[seg] | (owner[seg] >= 0)) & ~line[seg]
                if ok.any():
                    line[seg] |= ok
                    fill[seg] |= ok & ~base[seg]
                    grown = True
    # ---- runs and edges
    runs, edges = [], []
    internal = defaultdict(set)                                     # jid -> vessels observable only inside it
    internal_s = np.zeros(N, bool)
    for v in G.vids:
        o_ = G.o(v)
        d = S.v[v]
        lv = line[o_:o_ + d["n"]]
        for a, b in J._segments(lv):
            ow = owner[o_ + a:o_ + b + 1]
            if ow[0] >= 0 and (ow == ow[0]).all():
                # wholly inside one junction's claim: the vessel is part of that junction (no line leaves it)
                internal[int(ow[0])].add(int(v))
                internal_s[o_ + a:o_ + b + 1] = True
                continue
            rid = len(runs)
            st_c = _cause(S, G, v, a, -1, owner, mdom, inf, absorbed, Bk)
            en_c = _cause(S, G, v, b, +1, owner, mdom, inf, absorbed, Bk)
            runs.append(dict(vid=v, a=a, b=b, start=st_c, end=en_c, edges=[]))
            # each junction's claim the run passes: where its lines attach (the run's sample nearest its centre);
            # an edge runs from attachment to attachment, so the edges cover the whole run (review: the linked
            # edges had no polyline and the edges covered only 41-55 % of the lines)
            att = {}
            for s0_, s1_, jid in _owner_segments(ow):
                q = d["C"][a + s0_:a + s1_ + 1] - cen[jid]
                att[s0_] = a + s0_ + int(np.argmin(q[:, 0] ** 2 + q[:, 1] ** 2))
            cur, cur_p = (("J", int(ow[0])), att[0]) if ow[0] >= 0 else (("E", rid, 0), a)
            prev_claim = ow[0] >= 0
            i = 0
            L = b - a + 1
            while i < L:
                if ow[i] >= 0:
                    jid = int(ow[i])
                    j2 = i
                    while j2 + 1 < L and ow[j2 + 1] == jid:
                        j2 += 1
                    if cur[0] == "J" and cur[1] != jid and prev_claim:          # two claims touching: linked
                        edges.append(dict(run=rid, vid=v, i0=a + i, i1=a + i - 1, u=cur, w=("J", jid), linked=True,
                                          p0=cur_p, p1=att[i]))
                        runs[rid]["edges"].append(len(edges) - 1)
                    cur, cur_p = ("J", jid), att[i]
                    prev_claim = True
                    i = j2 + 1
                else:
                    j2 = i
                    while j2 + 1 < L and ow[j2 + 1] < 0:
                        j2 += 1
                    nxt, nxt_p = (("J", int(ow[j2 + 1])), att[j2 + 1]) if j2 + 1 < L else (("E", rid, 1), a + j2)
                    edges.append(dict(run=rid, vid=v, i0=a + i, i1=a + j2, u=cur, w=nxt, linked=False,
                                      p0=cur_p if cur[0] == "J" else a + i, p1=nxt_p))
                    runs[rid]["edges"].append(len(edges) - 1)
                    cur, cur_p = nxt, nxt_p
                    prev_claim = False
                    i = j2 + 1
    # ---- stubs: short pieces dangling from a junction into a fade or a merge; short loose pieces
    stub = np.zeros(len(edges), bool)
    stub_samples = np.zeros(N, bool)
    for ei, e in enumerate(edges):
        if e["linked"]:
            continue
        d = S.v[e["vid"]]
        o_ = G.o(e["vid"])
        lo, hi = o_ + e["i0"], o_ + e["i1"]           # with the samples next to it absorbed before: one stretch
        while lo > o_ and absorbed[lo - 1]:
            lo -= 1
        while hi < o_ + d["n"] - 1 and absorbed[hi + 1]:
            hi += 1
        if (hi - lo + 1) * d["ds"] >= arm_min_px:
            continue
        ends = [x for x in (e["u"], e["w"]) if x[0] == "E"]
        if not ends:
            continue
        labels = [runs[x[1]]["start" if x[2] == 0 else "end"][0] for x in ends]
        if any(lb == "node" for lb in labels):
            continue
        if len(ends) == 1 and labels[0] == "border":
            continue
        stub[ei] = True
        o_ = G.o(e["vid"])
        stub_samples[o_ + e["i0"]:o_ + e["i1"] + 1] = True
    # a piece shorter than arm_min_px between two stretches merged into lines of one junction: part of that
    # junction (else a few px junction-to-itself loop; not absorbed when absorbing it would join a longer stretch)
    drop = set()
    for rid, ru in enumerate(runs):
        if len(ru["edges"]) != 1 or stub[ru["edges"][0]]:
            continue
        cs, ce = ru["start"], ru["end"]
        if cs[0] != "merged" or ce[0] != "merged" or owner[cs[1]] < 0 or owner[cs[1]] != owner[ce[1]]:
            continue
        d = S.v[ru["vid"]]
        if (ru["b"] - ru["a"] + 1) * d["ds"] >= arm_min_px:
            continue
        o_ = G.o(ru["vid"])
        drop.add(rid)
        internal[int(owner[cs[1]])].add(int(ru["vid"]))
        internal_s[o_ + ru["a"]:o_ + ru["b"] + 1] = True
    if drop:
        keep_e = [ei for ei, e in enumerate(edges) if e["run"] not in drop]
        new_rid = {rid: n for n, rid in enumerate(r_ for r_ in range(len(runs)) if r_ not in drop)}
        new_eid = {ei: n for n, ei in enumerate(keep_e)}
        edges = [dict(edges[ei], run=new_rid[edges[ei]["run"]]) for ei in keep_e]
        for e in edges:
            e["u"] = ("E", new_rid[e["u"][1]], e["u"][2]) if e["u"][0] == "E" else e["u"]
            e["w"] = ("E", new_rid[e["w"][1]], e["w"][2]) if e["w"][0] == "E" else e["w"]
        stub = stub[keep_e]
        runs = [dict(ru, edges=[new_eid[ei] for ei in ru["edges"]]) for rid, ru in enumerate(runs) if rid not in drop]
    degree = Counter()
    has_line = set()
    for ei, e in enumerate(edges):
        if stub[ei]:
            continue
        for x in (e["u"], e["w"]):
            if x[0] == "J":
                degree[x[1]] += 1
                has_line.add((x[1], e["vid"]))
    line &= ~internal_s
    return dict(owner=owner, mdom=mdom, line=line, fill=fill & line, runs=runs, edges=edges, stub=stub,
                stub_samples=stub_samples, degree=degree, has_line=has_line, active=set(active), internal=internal,
                internal_s=internal_s)


def _owner_segments(ow) -> list:
    """(start, end, jid) of the stretches of equal owner >= 0 along one vessel (inclusive, local indices)."""
    out = []
    i, n = 0, len(ow)
    while i < n:
        if ow[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and ow[j + 1] == ow[i]:
            j += 1
        out.append((i, j, int(ow[i])))
        i = j + 1
    return out


def _cause(S, G, v, i, step, owner, mdom, inf, absorbed, Bk):
    """What ends a run of vessel v at its sample i on the side `step` (-1 before, +1 after): ('junction', jid),
    ('merged', partner global sample), ('border',), ('node',) (the vessel's own end) or ('fade',)."""
    d = S.v[v]
    o_ = G.o(v)
    if owner[o_ + i] >= 0:
        return ("junction", int(owner[o_ + i]))
    t = 0
    kk = i
    while t < Bk:
        kk += step
        if kk < 0 or kk >= d["n"]:
            return ("node",)
        g = o_ + kk
        if mdom[g] >= 0:
            return ("merged", int(mdom[g]))
        if not d["inf"][kk]:
            return ("border",)
        if absorbed[g]:
            continue
        t += 1
    return ("fade",)


def _assemble(ctx, kind, st, cand, cj, node_cand, S, G, graph, obs, inf, rad, absorbed, Bk, k, shape, cnr_observe,
              min_run_px, bridge_px, arm_min_px) -> dict:
    """The output dict from the converged state."""
    H, W = shape
    runs, edges_all, stub, owner, mdom, line = (st[x] for x in ("runs", "edges", "stub", "owner", "mdom", "line"))
    edges = [dict(e, eid=ei) for ei, e in enumerate(edges_all) if not stub[ei]]   # (none are stubs once converged)
    # ---- image-graph nodes
    nodes = {}                                       # key -> dict

    def node(key, **kw):
        if key not in nodes:
            nodes[key] = dict(kw)
        return key
    for jid in st["active"]:
        j = cj[jid]
        node(("J", jid), kind="junction", x=float(j["x"]), y=float(j["y"]), source="event", jid=jid)
    # run ends at a vessel's own node: hidden junctions, or the line ends there (fade)
    at_node = defaultdict(list)
    for rid, ru in enumerate(runs):
        for side in ("start", "end"):
            if ru[side][0] == "node":
                e = graph.vessels[ru["vid"]]
                at_node[e.u if side == "start" else e.v].append((rid, side))
    end_node = {}                                    # (rid, side) -> node key
    end_label = {}
    for nid, lst in at_node.items():
        p = k * (np.asarray(graph.nodes[nid].xy, float) - S.origin)
        if len(lst) >= 2:
            key = node(("H", nid), kind="hidden_junction", x=float(p[0]), y=float(p[1]), vessel_node=int(nid),
                       truth_type=_node_type(graph, nid))
            for rs in lst:
                end_node[rs], end_label[rs] = key, "hidden_junction"
        else:
            end_label[lst[0]] = "fade"
    # merged ends: attach to the line they merge into
    split_req = defaultdict(list)                    # edge eid -> [(sample, (rid, side))]
    edge_of = np.full(G.N, -1, np.int64)
    for e in edges:
        if not e["linked"]:
            o_ = G.o(e["vid"])
            edge_of[o_ + e["i0"]:o_ + e["i1"] + 1] = e["eid"]
    run_end_at = {}                                  # global sample -> (rid, side) of run ends
    for rid, ru in enumerate(runs):
        o_ = G.o(ru["vid"])
        run_end_at[o_ + ru["a"]] = (rid, "start")
        run_end_at[o_ + ru["b"]] = (rid, "end")
    merged_attach = {}
    for rid, ru in enumerate(runs):
        for side in ("start", "end"):
            c = ru[side]
            if c[0] != "merged":
                continue
            gp = c[1]
            end_label[(rid, side)] = "merged"
            if owner[gp] >= 0 and line[gp] and ("J", int(owner[gp])) in nodes:
                end_node[(rid, side)] = ("J", int(owner[gp]))
                merged_attach[(rid, side)] = "junction"
            elif edge_of[gp] >= 0:
                split_req[int(edge_of[gp])].append((gp, (rid, side)))
                merged_attach[(rid, side)] = "split"
            else:
                near = [run_end_at[g] for g in range(gp - Bk, gp + Bk + 1) if g in run_end_at
                        and G.gvid[min(max(g, 0), G.N - 1)] == G.gvid[gp]]
                if near:
                    merged_attach[(rid, side)] = ("end", near[0])
                else:
                    merged_attach[(rid, side)] = "alone"
    # split edges at merge points
    new_edges = []
    alias = {}                                       # run end -> the run end (of the dominant) it attaches to
    for e in edges:
        reqs = split_req.get(e["eid"])
        if not reqs:
            new_edges.append(e)
            continue
        o_ = G.o(e["vid"])
        pts = sorted(((gp - o_, rs) for gp, rs in reqs), key=lambda t: t[0])
        clusters = []
        for m_, rs in pts:
            if clusters and m_ - clusters[-1][0] <= 2:
                clusters[-1][1].append(rs)
            else:
                clusters.append([m_, [rs]])
        cur_u, cur_i0, cur_p0 = e["u"], e["i0"], e["p0"]
        for m_, rss in clusters:
            if m_ <= cur_i0 + 1:
                key = cur_u
            elif m_ >= e["i1"] - 1:
                key = e["w"]
            else:
                p = S.v[e["vid"]]["C"][m_]
                key = node(("M", e["vid"], int(m_)), kind="merge", x=float(p[0]), y=float(p[1]), source="merge",
                           dominant=int(e["vid"]))
                new_edges.append(dict(e, i0=cur_i0, i1=m_, u=cur_u, w=key, p0=cur_p0, p1=m_))
                cur_u, cur_i0, cur_p0 = key, m_, m_
            for rs in rss:
                if key[0] == "E":
                    alias[rs] = (key[1], "start" if key[2] == 0 else "end")
                else:
                    end_node[rs] = key
        new_edges.append(dict(e, i0=cur_i0, u=cur_u, p0=cur_p0))
    edges = new_edges
    # remaining run ends: fade / border / merged alone / attached to another run's end
    def own_end(rs):
        ru, c = runs[rs[0]], runs[rs[0]][rs[1]]
        p = S.v[ru["vid"]]["C"][ru["a"] if rs[1] == "start" else ru["b"]]
        lab = end_label.get(rs, c[0])
        end_label[rs] = lab
        end_node[rs] = node(("E", rs[0], rs[1]), kind=lab, x=float(p[0]), y=float(p[1]))
    for rid, ru in enumerate(runs):
        for side in ("start", "end"):
            rs = (rid, side)
            c = ru[side]
            if c[0] == "junction":
                end_node[rs], end_label[rs] = ("J", c[1]), "junction"
                continue
            if rs in end_node or rs in alias:
                continue
            own_end(rs)
    for rs in alias:                                 # a merged end at a split point that is the dominant's run end
        tgt, seen = alias[rs], {rs}
        while tgt in alias and tgt not in seen and tgt not in end_node:     # that end merged in turn: follow
            seen.add(tgt)
            tgt = alias[tgt]
        if tgt not in end_node:                      # merged ends merging into each other: one node of their own
            own_end(tgt)
        key = end_node[tgt]
        end_node[rs] = key
        if nodes[key]["kind"] in ("fade", "border", "merged"):
            nodes[key]["kind"] = "merge"
            nodes[key]["source"] = "merge"
            nodes[key].setdefault("dominant", int(runs[tgt[0]]["vid"]))
    for rs, how in merged_attach.items():            # a merged end at the end of the line it merges into
        if isinstance(how, tuple) and how[0] == "end":
            other = end_node.get(how[1])
            mine = end_node.get(rs)
            if other is not None and mine is not None and other != mine and nodes.get(other, {}).get("kind") in (
                    "fade", "border", "merged"):
                # the two ends meet: one node, the line changes vessel there
                for kk, vv in list(end_node.items()):
                    if vv == mine:
                        end_node[kk] = other
                nodes[other]["kind"] = "merge"
                nodes[other]["source"] = "merge"
                nodes[other].setdefault("dominant", int(runs[how[1][0]]["vid"]))
                nodes.pop(mine, None)
    # resolve edge endpoints ('E' keys are run ends)
    def resolve(x):
        if x[0] == "E":
            return end_node[(x[1], "start" if x[2] == 0 else "end")]
        return x
    for e in edges:
        e["u"], e["w"] = resolve(e["u"]), resolve(e["w"])
    # ---- degrees, kinds
    deg = Counter()
    inc = defaultdict(list)
    for n_e, e in enumerate(edges):
        for side, x in (("u", e["u"]), ("w", e["w"])):
            deg[x] += 1
            inc[x].append((n_e, side))
    for key, nd in nodes.items():
        nd["degree"] = int(deg.get(key, 0))
        if nd["kind"] == "merge":
            if nd["degree"] >= 3:
                nd["kind"] = "junction"
            elif nd["degree"] == 2:
                nd["kind"] = "hidden_junction"
                nd["truth_type"] = "merge"
            else:
                nd["kind"] = "merged"
        elif nd["kind"] == "hidden_junction" and nd["degree"] >= 3:
            # a true node released by the iteration (a line merged into another right at the node) that the
            # final lines still show with >= 3 lines (one merging in close by): an observable junction after all
            nd["kind"] = "junction"
            nd["source"] = "event"
            jn = node_cand.get(nd.get("vessel_node"))
            if jn is not None:
                nd["jid"] = jn
            else:
                nd["source"] = "node"
    # every run end is labelled by the final kind of its node (review D4: a promoted hidden junction kept its runs'
    # 'hidden_junction' labels, a fade / border end taken over by a merged line kept 'fade', merged ends attached
    # to a junction said 'merged'); merged_with still names the line a 'merged' cause merged into
    for rs, key in end_node.items():
        end_label[rs] = nodes[key]["kind"]
    # ---- merged vids carried by each (final) edge and run; merged stretches per subordinate vessel
    for e_i, e in enumerate(edges):
        e["id"] = e_i
    edge_of2 = np.full(G.N, -1, np.int64)
    run_of = np.full(G.N, -1, np.int64)
    for e in edges:                                  # interiors only: split pieces share their boundary sample
        if not e["linked"]:
            o_ = G.o(e["vid"])
            if e["i1"] - e["i0"] >= 2:
                edge_of2[o_ + e["i0"] + 1:o_ + e["i1"]] = e["id"]
            else:
                edge_of2[o_ + e["i0"]:o_ + e["i1"] + 1] = e["id"]
    for rid, ru in enumerate(runs):
        o_ = G.o(ru["vid"])
        run_of[o_ + ru["a"]:o_ + ru["b"] + 1] = rid
    sub = np.flatnonzero((mdom >= 0) & obs & ~absorbed)
    carry = defaultdict(set)
    run_merged = defaultdict(lambda: defaultdict(list))         # rid -> merged vid -> partner samples
    if len(sub):
        eo = edge_of2[mdom[sub]]
        ro = run_of[mdom[sub]]
        for gs, eid, rid_, gp in zip(sub.tolist(), eo.tolist(), ro.tolist(), mdom[sub].tolist()):
            if eid >= 0:
                carry[eid].add(int(G.gvid[gs]))
            if rid_ >= 0:
                run_merged[rid_][int(G.gvid[gs])].append(gp)
    merged_list = []
    for v in G.vids:
        o_ = G.o(v)
        d = S.v[v]
        mv = (mdom[o_:o_ + d["n"]] >= 0) & obs[o_:o_ + d["n"]] & ~absorbed[o_:o_ + d["n"]]
        for a, b in J._segments(mv):
            into = Counter(G.gvid[mdom[o_ + a:o_ + b + 1]].tolist()).most_common(1)[0][0]
            merged_list.append(dict(vid=int(v), into=int(into), s0_px=round(float(d["sig"][a]), 2),
                                    s1_px=round(float(d["sig"][b]), 2), s0_dc=round(float(d["sig"][a]) / k, 3),
                                    s1_dc=round(float(d["sig"][b]) / k, 3)))
    # ---- observable junctions
    jkeys = [key for key, nd in nodes.items() if nd["kind"] == "junction"]
    jkeys.sort(key=lambda kk: (round(nodes[kk]["y"], 3), round(nodes[kk]["x"], 3)))
    other = [key for key in nodes if key not in set(jkeys)]
    order = {"hidden_junction": 0, "merged": 1, "fade": 2, "border": 3}
    other.sort(key=lambda kk: (order.get(nodes[kk]["kind"], 9), round(nodes[kk]["y"], 3), round(nodes[kk]["x"], 3)))
    nid_of = {key: i for i, key in enumerate(jkeys + other)}
    junc_out = []
    for key in jkeys:
        nd = nodes[key]
        lines = [_line_info(S, edges[n_e], side, carry) for n_e, side in inc[key]]
        if nd.get("source") == "node":                 # a graph node no candidate holds
            nid = nd["vessel_node"]
            par, chi = graph.in_vessels(nid), graph.out_vessels(nid)
            members = [dict(type=_node_type(graph, nid), vessels=sorted(par + chi), node=int(nid), parents=par,
                            children=chi, xy=[nd["x"], nd["y"]])]
            ev_v = sorted(par + chi)
            typ = members[0]["type"]
            tobs = _obs_type(lines, members, graph)
            radius = float(max([0.5 * ln["width_px"] + 2 * ln["blur_px"] for ln in lines], default=3.0))
            reasons = []
        elif nd.get("source") == "event":
            j = cj[nd["jid"]]
            members = j["members"]
            ev_v = sorted({int(v) for m in members for v in m["vessels"]})
            arms = {(a["vid"], a["flow"]): a for a in j["arms"]}
            for ln in lines:
                a = arms.get((ln["vid"], ln["flow"]))
                if a is not None:
                    ln.update(xy=_r(a["xy"]), dir=_r(a["dir"], 4), angle_deg=round(float(a["angle_deg"]), 2),
                              cnr=None if a["cnr"] is None else round(float(a["cnr"]), 3),
                              contrast=None if a["contrast"] is None else round(float(a["contrast"]), 5))
            typ = j["type"]
            tobs = _obs_type(lines, members, graph)
            radius = float(j["radius"])
            reasons = [r_ for r_ in j["ambiguous_reasons"] if r_ in ("shallow_angle", "symmetric_fork")]
        else:
            dom = nd.get("dominant", -1)
            vs = sorted({ln["vid"] for ln in lines})
            members, ev_v = [], vs
            typ = "merge"
            tobs = _merge_type(lines, dom, graph)
            radius = float(max([0.5 * ln["width_px"] + 2 * ln["blur_px"] for ln in lines], default=3.0))
            reasons = ["merged_lines"]
        if any(len(ln["vids"]) > 1 for ln in lines) and "merged_lines" not in reasons:
            reasons.append("merged_lines")
        with_line = {ln["vid"] for ln in lines} | {v for ln in lines for v in ln["vids"]}
        inner = sorted(st["internal"].get(nd["jid"], set()) - with_line) if nd.get("source") == "event" else []
        junc_out.append(dict(id=nid_of[key], node=nid_of[key], x=nd["x"], y=nd["y"], radius=round(radius, 2),
                             source=nd.get("source", "event"), type=typ, type_observable=tobs, n_lines=len(lines),
                             lines=lines, members=members, vessels=ev_v,
                             hidden_vessels=[v for v in ev_v if v not in with_line and v not in inner],
                             internal_vessels=inner, ambiguous_reasons=sorted(reasons), complete_ids=[]))
    if ctx.junctions_all:
        _match_complete(junc_out, ctx.junctions_all)
    jid_of_key = {key: nid_of[key] for key in jkeys}
    # ---- hidden junction nodes
    hidden = []
    for key in other:
        nd = nodes[key]
        if nd["kind"] != "hidden_junction":
            continue
        vs = sorted({edges[n_e]["vid"] for n_e, _ in inc[key]})
        hidden.append(dict(node=nid_of[key], x=nd["x"], y=nd["y"], vids=vs, vessel_node=nd.get("vessel_node"),
                           truth_type=nd.get("truth_type"), degree=nd["degree"]))
    # ---- runs output
    runs_out = []
    run_junctions = defaultdict(list)
    for e in edges:
        for x in (e["u"], e["w"]):
            if x in jid_of_key and jid_of_key[x] not in run_junctions[e["run"]]:
                run_junctions[e["run"]].append(jid_of_key[x])
    for rid, ru in enumerate(runs):
        v = ru["vid"]
        d = S.v[v]
        a, b = ru["a"], ru["b"]
        sl = slice(a, b + 1)
        ends = {}
        for side in ("start", "end"):
            rs = (rid, side)
            c = ru[side]
            key = end_node[rs]
            ends[side] = dict(label=end_label[rs], node=nid_of[key],
                              junction=jid_of_key.get(key),
                              merged_with=int(G.gvid[c[1]]) if c[0] == "merged" else None,
                              vessel_node=(int(graph.vessels[v].u if side == "start" else graph.vessels[v].v)
                                           if c[0] == "node" else None),
                              xy=_r(d["C"][a if side == "start" else b]))
        o_ = G.o(v)
        mg_here = []
        for w_, gps in sorted(run_merged.get(rid, {}).items()):
            pts = np.asarray(gps, np.int64) - o_
            mg_here.append(dict(vid=int(w_), s0_px=round(float(d["sig"][pts.min()]), 2),
                                s1_px=round(float(d["sig"][pts.max()]), 2)))
        unobs = [[int(p_), int(q_)] for p_, q_ in J._segments(~obs[o_ + a:o_ + b + 1])]
        runs_out.append(dict(
            id=rid, vid=int(v), cls=d["cls"], s0_px=round(float(d["sig"][a]), 2), s1_px=round(float(d["sig"][b]), 2),
            s0_dc=round(float(d["sig"][a]) / k, 3), s1_dc=round(float(d["sig"][b]) / k, 3),
            length_px=round(float(d["sig"][b] - d["sig"][a]), 2), start=ends["start"], end=ends["end"],
            junctions=run_junctions.get(rid, []), merged=mg_here, unobserved=unobs,
            xy=[_r(p) for p in d["C"][sl]], width_px=_rs(2.0 * d["r"][sl], 3), blur_px=_rs(d["st"][sl], 3),
            contrast=_rs(d["contrast_own"][sl], 3), cnr=_rs(d["cnr"][sl], 3)))
    # ---- graph (node-link)
    g_nodes = []
    for key in jkeys + other:
        nd = nodes[key]
        rec_ = dict(id=nid_of[key], kind=nd["kind"], x=round(nd["x"], 2), y=round(nd["y"], 2), degree=nd["degree"],
                    junction=jid_of_key.get(key), vessel_node=nd.get("vessel_node"), truth_type=nd.get("truth_type"),
                    vids=sorted({edges[n_e]["vid"] for n_e, _ in inc[key]}))
        if key in jid_of_key:
            jo = junc_out[jid_of_key[key]]
            rec_["type_observable"] = jo["type_observable"]
            rec_["truth_type"] = jo["type"]
        g_nodes.append(rec_)
    g_edges = []
    for e in edges:
        d = S.v[e["vid"]]
        p0, p1 = int(e["p0"]), int(e["p1"])           # from attachment (or run end) to attachment (or run end)
        s0, s1 = float(d["sig"][p0]), float(d["sig"][p1])
        xy = [_r(p) for p in d["C"][p0:p1 + 1]]
        vids = sorted({int(e["vid"])} | carry.get(e["id"], set()))
        g_edges.append(dict(source=nid_of[e["u"]], target=nid_of[e["w"]], key=e["id"], id=e["id"], vid=int(e["vid"]),
                            vids=vids, cls=d["cls"], run=int(e["run"]), s0_px=round(s0, 2), s1_px=round(s1, 2),
                            s0_dc=round(s0 / k, 3), s1_dc=round(s1 / k, 3), length_px=round(s1 - s0, 2),
                            linked=bool(e["linked"]), xy=xy))
    # ---- don't-care junctions: hidden junction nodes, and complete junctions shown at the permissive level
    dcj = [dict(x=h["x"], y=h["y"], radius=0.5 * LAMBDA_PX, source="hidden_junction", ref=h["node"]) for h in hidden]
    if ctx.junctions_all:
        for jc in ctx.junctions_all:
            if jc.get("type_visible", "none") != "none" and not jc.get("_obs_matched") and jc.get("in_frame", True):
                dcj.append(dict(x=jc["x"], y=jc["y"], radius=float(max(jc["radius"], 0.5 * LAMBDA_PX)),
                                source="complete", ref=jc["id"]))
        for jc in ctx.junctions_all:
            jc.pop("_obs_matched", None)
    rl = rules(cnr_observe, min_run_px, bridge_px, arm_min_px)
    rl.update(noise_source=ctx.noise_source, band_rms={str(b_): v_ for b_, v_ in (S.band_rms or {}).items()},
              od=bool(getattr(S, "has_od", True)))
    return dict(format=FORMAT, version=VERSION, kind=kind, shape=[int(H), int(W)], k=k,
                origin=[float(S.origin[0]), float(S.origin[1])], rules=rl, runs=runs_out, junctions=junc_out,
                hidden_junctions=hidden, dont_care_junctions=dcj, merged=merged_list,
                graph=dict(directed=False, multigraph=True, graph=dict(kind=kind, format=FORMAT), nodes=g_nodes,
                           edges=g_edges))


def _node_type(graph, nid) -> str:
    i_, o_ = len(graph.in_vessels(nid)), len(graph.out_vessels(nid))
    if i_ == 1 and o_ >= 2:
        return "bifurcation"
    if o_ == 1 and i_ >= 2:
        return "confluence"
    if i_ >= 2 and o_ >= 2:
        return "anastomosis"
    return "node"


def _line_info(S, e, side, carry) -> dict:
    """A line leaving a junction node: the edge e at its end `side` ('u': the edge's upstream end is at the
    node, so blood flows out of the junction along it; 'w': into it)."""
    d = S.v[e["vid"]]
    if e["linked"]:
        i = min(max(e["i0"], 0), d["n"] - 1)
    else:
        i = e["i0"] if side == "u" else e["i1"]
    flow = "out" if side == "u" else "in"
    t = d["T"][i] * (1.0 if flow == "out" else -1.0)
    sgn = 1 if flow == "out" else -1
    win = np.arange(i, i + sgn * int(round(LAMBDA_PX / d["ds"])) + sgn, sgn)
    win = win[(win >= 0) & (win < d["n"])]
    cw = d["contrast_own"][win]
    return dict(edge=int(e["id"]), vid=int(e["vid"]), vids=sorted({int(e["vid"])} | carry.get(e["id"], set())),
                flow=flow, xy=_r(d["C"][i]), dir=_r(t, 4), angle_deg=round(math.degrees(math.atan2(t[1], t[0])), 2),
                width_px=round(float(2.0 * d["r"][i]), 3), blur_px=round(float(d["st"][i]), 3),
                cnr=round(float(np.median(d["cnr"][win])), 3) if len(win) else None,
                contrast=round(float(np.median(cw[np.isfinite(cw)])), 5) if np.isfinite(cw).any() else None,
                linked=bool(e["linked"]))


def _through_pairs(lines, members):
    """The lines leaving a junction paired into through lines, or None: every 'in' line with one 'out' line of
    the same vessel, or of a child of it at a node of the junction (a parent running on into a daughter: its
    other daughter not observable, or merged with it), a line standing for every vessel id it carries.  Returns
    [(i, j)] (indices into lines) or None if they do not pair up."""
    import itertools
    cont = set()
    for m in members or ():
        if "node" in m:
            for p_ in m.get("parents", []):
                for c_ in m.get("children", []):
                    cont.add((int(p_), int(c_)))
    ins = [i for i, ln in enumerate(lines) if ln["flow"] == "in"]
    outs = [i for i, ln in enumerate(lines) if ln["flow"] == "out"]
    if len(ins) != len(outs) or not ins or len(ins) > 6:
        return None

    def vs(ln):
        return set(ln.get("vids") or [ln["vid"]]) | {ln["vid"]}

    def ok(i, j):
        a, b = vs(lines[i]), vs(lines[j])
        return bool(a & b) or any((x, y) in cont for x in a for y in b)
    for perm in itertools.permutations(outs):
        if all(ok(i, j) for i, j in zip(ins, perm)):
            return list(zip(ins, perm))
    return None


def _obs_type(lines, members, graph) -> str:
    """type_observable from the lines leaving a junction and its truth events (module docstring): 3 lines ->
    bifurcation / confluence when they are 3 vessels of one node, else pseudo-T; 4 -> crossing when they pair
    up into two through lines (_through_pairs: the same vessel in and out, or a parent in and its only observable
    daughter out: review D2, a fork with one daughter hidden and a vessel crossing at the node is an X), else
    compound; 5 or more -> compound."""
    n = len(lines)
    if n <= 2:
        return "none"
    vs = [ln["vid"] for ln in lines]
    if n == 3:
        if len(set(vs)) == 3:
            for m in members:
                if "node" not in m:
                    continue
                par, chi = set(m.get("parents", [])), set(m.get("children", []))
                if set(vs) <= par | chi:
                    n_in = sum(v in par for v in vs)
                    return "bifurcation" if n_in <= 1 else "confluence"
        return "pseudo-T"
    if n == 4 and _through_pairs(lines, members) is not None:
        return "crossing"
    return "compound"


def _merge_type(lines, dom, graph) -> str:
    """type_observable of a merge junction: siblings of one fork part like a fork, co-tributaries of a
    confluence join like one; otherwise a T; 4 lines pairing up into two through lines (two lines touching and
    parting at one place): a crossing."""
    n = len(lines)
    if n <= 2:
        return "none"
    if n >= 4:
        return "crossing" if n == 4 and _through_pairs(lines, []) is not None else "compound"
    subs = {ln["vid"] for ln in lines if ln["vid"] != dom}
    if dom in graph.vessels and len(subs) == 1:
        s = next(iter(subs))
        if s in graph.vessels:
            ed, es = graph.vessels[dom], graph.vessels[s]
            if ed.u == es.u:
                return "bifurcation"
            if ed.v == es.v:
                return "confluence"
    return "pseudo-T"


def _events(members) -> list:
    """(kind, vessel ids (frozenset), node id or None, xy) of a junction's truth events."""
    out = []
    for m in members or ():
        if "node" in m:
            out.append(("node", frozenset(), int(m["node"]), None))
        else:
            out.append((m["type"], frozenset(int(v) for v in m["vessels"]), None, np.asarray(m["xy"], float)))
    return out


def _same_event(e1, e2) -> bool:
    """One truth event seen under two rules: the same graph node; or the same kind of event of the same two
    vessels within lambda (a crossing is fixed geometry; where a touching or a fading end sits depends on what
    the rule sees)."""
    if e1[0] == "node" or e2[0] == "node":
        return e1[0] == e2[0] and e1[2] == e2[2]
    return e1[0] == e2[0] and e1[1] == e2[1] and float(np.hypot(*(e1[3] - e2[3]))) <= LAMBDA_PX


def _match_complete(junc_out, junctions_all):
    """complete_ids: the complete junctions (permissive rules, hidden ones too) that hold one of the observable
    junction's truth events (_same_event; merge junctions have none); marks them _obs_matched.  (Was: centres
    within the larger radius + lambda / 2, which matched nearly everything near the big junctions of a wide
    vein.)"""
    if not junctions_all or not junc_out:
        return
    from scipy.spatial import cKDTree
    P = np.array([[j["x"], j["y"]] for j in junctions_all], float)
    tree = cKDTree(P)
    rmax = max(float(max(j["radius"] for j in junctions_all)), 0.0)
    ev_c = [_events(j["members"]) for j in junctions_all]
    for jo in junc_out:
        mine = _events(jo["members"])
        ids = []
        if mine:
            for c in tree.query_ball_point([jo["x"], jo["y"]], float(jo["radius"]) + rmax + 2 * LAMBDA_PX):
                if any(_same_event(e1, e2) for e1 in mine for e2 in ev_c[c]):
                    ids.append(int(junctions_all[c]["id"]))
                    junctions_all[c]["_obs_matched"] = True
        jo["complete_ids"] = sorted(ids)


# ======================================================================== profiles (the complete truth)
def visibility_profiles(S, G, obs, st, absorbed, kind, k, cnr_observe, min_run_px, bridge_px,
                        noise_source="exact") -> dict:
    """Per vessel near the frame, per sample (every px of arclength): flat arrays and offsets[i]:offsets[i + 1]
    per vids[i]: s_px, s_dc, x_px, y_px, cnr (with the OD ratio), cnr_closed_form, od_ratio, peak_od, contrast
    (the vessel's own), contrast_image (raw), radius_px, blur_px (total), depth_dc, in_frame, observable,
    dont_care (seen at CNR_DONT_CARE but not observable), line (on its own observable line; through a junction
    also where not observable itself), merged_into (the vid whose line it is merged into, -1), internal
    (observable only inside one observable junction: part of it, no line leaves it), absorbed (observable but
    part of no line: a stub shorter than arm_min_px); meta."""
    vids = G.vids
    cat = (lambda key, dt: np.concatenate([np.asarray(S.v[v][key], dt) for v in vids]) if vids else np.zeros(0, dt))
    seen_dc = np.concatenate([_seen_mask(S.v[v], CNR_DONT_CARE, LAMBDA_PX, 0.5 * LAMBDA_PX, False) for v in vids]) \
        if vids else np.zeros(0, bool)
    inf = cat("inf", bool)
    mdom = st["mdom"]
    merged_into = np.where((mdom >= 0) & obs, G.gvid[np.where(mdom >= 0, mdom, 0)], -1).astype(np.int32)
    C = cat("C", np.float64).reshape(-1, 2) if vids else np.zeros((0, 2))
    out = dict(
        vids=np.array(vids, np.int32), offsets=G.off.astype(np.int64),
        s_px=cat("sig", np.float32), s_dc=(cat("sig", np.float64) / k).astype(np.float32),
        x_px=C[:, 0].astype(np.float32), y_px=C[:, 1].astype(np.float32),
        cnr=cat("cnr", np.float32), cnr_closed_form=cat("cnr_cf", np.float32), od_ratio=cat("ratio", np.float32),
        peak_od=cat("peak", np.float32), contrast=cat("contrast_own", np.float32),
        contrast_image=cat("contrast", np.float32), radius_px=cat("r", np.float32), blur_px=cat("st", np.float32),
        depth_dc=cat("z", np.float32), in_frame=inf, observable=obs.copy(),
        dont_care=seen_dc & inf & ~obs, line=st["line"].copy(), merged_into=merged_into,
        internal=st["internal_s"] & obs, absorbed=absorbed & obs)
    out["meta"] = dict(kind=kind, k=float(k), origin=[float(S.origin[0]), float(S.origin[1])],
                       shape=[int(S.shape[0]), int(S.shape[1])], rules=rules(cnr_observe, min_run_px, bridge_px),
                       noise_source=noise_source, band_rms={str(b): v for b, v in (S.band_rms or {}).items()},
                       pad_px=J.PAD_PX, sample_px=J.SAMPLE_PX, has_od=bool(getattr(S, "has_od", True)),
                       blur_px=float(S.blur_px))
    return out


def profile_of(prof: dict, vid: int) -> dict:
    """One vessel's profile (views into the flat arrays)."""
    i = int(np.flatnonzero(prof["vids"] == vid)[0])
    a, b = int(prof["offsets"][i]), int(prof["offsets"][i + 1])
    return {key: v[a:b] for key, v in prof.items() if isinstance(v, np.ndarray) and v.shape[:1] == prof["s_px"].shape}


def save_profiles(prof: dict, path: str):
    arrs = {key: v for key, v in prof.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta=np.array(json.dumps(prof["meta"])), **arrs)


def load_profiles(path: str) -> dict:
    with np.load(path) as z:
        out = {key: z[key] for key in z.files if key != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def complete_truth(src, kinds=("average", "frame"), samples=None) -> dict:
    """The complete truth of a Scene or folder: graph (VesselGraph), junctions_all {kind: all junctions under the
    permissive rules, hidden ones too}, visibility {kind: visibility_profiles at the default observable rules}.
    From a folder: the saved junctions_all_<kind>.json / complete_visibility_<kind>.npz when present."""
    out = dict(graph=None, junctions_all={}, visibility={})
    for kind in kinds:
        if not _is_scene(src):
            folder = str(src)
            pj = os.path.join(folder, f"junctions_all_{kind}.json")
            pp = os.path.join(folder, f"complete_visibility_{kind}.npz")
            if out["graph"] is None:
                out["graph"] = VesselGraph.load(os.path.join(folder, "truth.json"))
            if os.path.exists(pj) and os.path.exists(pp):
                out["junctions_all"][kind] = J.load_junctions(pj)
                out["visibility"][kind] = load_profiles(pp)
                continue
            if not os.path.exists(os.path.join(folder, f"still_{kind}.tif")):
                continue
        ctx = _context(src, kind, (samples or {}).get(kind) if isinstance(samples, dict) else None)
        out["graph"] = ctx.graph
        ja = ctx.junctions_all
        if ja is None:
            ja = J.junctions_from_samples(ctx.S, ctx.graph, crossings=ctx.crossings, keep_hidden=True)
            ctx.junctions_all = ja
        out["junctions_all"][kind] = ja
        _, prof = _build(ctx, kind, CNR_OBSERVE, MIN_RUN_PX, BRIDGE_PX, ARM_MIN_PX)
        out["visibility"][kind] = prof
    return out


# ======================================================================== summary, check
def summary(obs: dict, junctions_all=None) -> dict:
    """Counts of one image's observable truth (and, with the complete junctions, how many of them are
    observable: a complete junction counts when an observable junction holds one of its truth events, the same
    graph node or the same crossing / touching / fading end of the same two vessels: _match_complete)."""
    js = obs["junctions"]
    runs = obs["runs"]
    ends = Counter(r_[side]["label"] for r_ in runs for side in ("start", "end"))
    kinds = Counter(n["kind"] for n in obs["graph"]["nodes"])
    out = dict(obs.get("totals") or {})
    out.update(n_runs=len(runs), n_vessels=len({r_["vid"] for r_ in runs}),
               run_length_px=round(float(sum(r_["length_px"] for r_ in runs)), 1),
               n_junctions=len(js), types_observable=dict(sorted(Counter(j["type_observable"] for j in js).items())),
               sources=dict(sorted(Counter(j["source"] for j in js).items())),
               n_hidden_junctions=len(obs["hidden_junctions"]), end_labels=dict(sorted(ends.items())),
               n_nodes=len(obs["graph"]["nodes"]), n_edges=len(obs["graph"]["edges"]),
               node_kinds=dict(sorted(kinds.items())), n_merged=len(obs["merged"]),
               merged_px=round(float(sum(m["s1_px"] - m["s0_px"] for m in obs["merged"])), 1),
               n_dont_care_junctions=len(obs.get("dont_care_junctions", [])))
    if junctions_all:
        matched = {c for j in js for c in j.get("complete_ids", [])}
        inf = [j for j in junctions_all if j.get("in_frame", True)]
        shown = [j for j in inf if j.get("type_visible", "none") != "none"]
        out.update(complete_n=len(inf), complete_shown_n=len(shown),
                   complete_observable=sum(j["id"] in matched for j in inf),
                   complete_shown_observable=sum(j["id"] in matched for j in shown))
        out["complete_observable_share"] = round(out["complete_observable"] / max(len(inf), 1), 4)
        out["complete_shown_observable_share"] = round(out["complete_shown_observable"] / max(len(shown), 1), 4)
    return out


def check_observable(obs: dict, prof: dict | None = None) -> list[str]:
    """Consistency problems of an observable truth (empty when sound): edges between existing nodes, with a
    polyline; node degrees as stored; observable junctions with >= 3 lines and a type that fits their lines (4
    lines pairing up into two through lines: a crossing, else compound); hidden junctions of degree 2; fade /
    border / merged nodes of degree 1; no junction-to-itself edge shorter than arm_min_px; run ends on existing
    nodes, labelled by their node's kind (and its junction id); run points inside the frame; a vessel's runs not
    overlapping; hidden vessels without a line at the junction.  With the image's profiles (prof): every vessel
    with min_run_px of observable samples is on a line, merged into one or internal to a junction (review D1),
    no absorbed stretch longer than arm_min_px, line samples in the frame."""
    out = []
    H, W = obs["shape"]
    rl = obs.get("rules", {})
    arm_min = float(rl.get("arm_min_px", ARM_MIN_PX))
    nodes = {n["id"]: n for n in obs["graph"]["nodes"]}
    deg = Counter()
    for e in obs["graph"]["edges"]:
        for x in (e["source"], e["target"]):
            if x not in nodes:
                out.append(f"edge {e['id']}: node {x} missing")
            deg[x] += 1
        if not e["xy"]:
            out.append(f"edge {e['id']}: no polyline")
        if e["vid"] not in e["vids"]:
            out.append(f"edge {e['id']}: vid not among its vids")
        if e["source"] == e["target"] and e["length_px"] < arm_min:
            out.append(f"edge {e['id']}: a {e['length_px']} px loop at node {e['source']}")
    for nid, n in nodes.items():
        if deg[nid] != n["degree"]:
            out.append(f"node {nid}: degree {n['degree']} stored, {deg[nid]} edges")
        if n["kind"] not in NODE_KINDS:
            out.append(f"node {nid}: kind {n['kind']}")
        if n["kind"] == "junction" and deg[nid] < 3:
            out.append(f"node {nid}: junction with {deg[nid]} lines")
        if n["kind"] == "hidden_junction" and deg[nid] != 2:
            out.append(f"node {nid}: hidden junction of degree {deg[nid]}")
        if n["kind"] in ("fade", "border", "merged") and deg[nid] != 1:
            out.append(f"node {nid}: {n['kind']} end of degree {deg[nid]}")
    for j in obs["junctions"]:
        n = j["n_lines"]
        ok = (n == 3 and j["type_observable"] in ("bifurcation", "confluence", "pseudo-T")) or              (n >= 4 and j["type_observable"] in ("crossing", "compound"))
        if not ok:
            out.append(f"junction {j['id']}: {n} lines typed {j['type_observable']}")
        if n >= 4:
            pairs = n == 4 and _through_pairs(j["lines"], j["members"]) is not None
            if pairs != (j["type_observable"] == "crossing"):
                out.append(f"junction {j['id']}: {n} lines {'pair' if pairs else 'do not pair'} up, typed "
                           f"{j['type_observable']}")
        if nodes.get(j["node"], {}).get("kind") != "junction":
            out.append(f"junction {j['id']}: its node {j['node']} is not a junction node")
        with_line = {v for ln in j["lines"] for v in ln["vids"]}
        if set(j.get("hidden_vessels", [])) & with_line:
            out.append(f"junction {j['id']}: hidden vessels with a line")
    by_v = defaultdict(list)
    for r_ in obs["runs"]:
        for side in ("start", "end"):
            e = r_[side]
            if e["label"] not in END_LABELS:
                out.append(f"run {r_['id']}: {side} label {e['label']}")
            nd = nodes.get(e["node"])
            if nd is None:
                out.append(f"run {r_['id']}: {side} node missing")
                continue
            if e["label"] != nd["kind"]:
                out.append(f"run {r_['id']}: {side} labelled {e['label']} at a {nd['kind']} node")
            if e["junction"] != nd.get("junction"):
                out.append(f"run {r_['id']}: {side} junction {e['junction']}, its node's {nd.get('junction')}")
        P = np.asarray(r_["xy"], float).reshape(-1, 2)
        if ((P[:, 0] < -0.5) | (P[:, 0] > W - 0.5) | (P[:, 1] < -0.5) | (P[:, 1] > H - 0.5)).any():
            out.append(f"run {r_['id']}: points outside the frame")
        by_v[r_["vid"]].append((r_["s0_px"], r_["s1_px"], r_["id"]))
    for v, lst in by_v.items():
        lst.sort()
        for (a0, a1, i0), (b0, b1, i1) in zip(lst, lst[1:]):
            if b0 <= a1:
                out.append(f"vessel {v}: runs {i0} and {i1} overlap")
    if prof is not None:
        min_run = float(rl.get("min_run_px", MIN_RUN_PX))
        internal = prof.get("internal", np.zeros(len(prof["observable"]), bool))
        for i, v in enumerate(prof["vids"].tolist()):
            a, b = int(prof["offsets"][i]), int(prof["offsets"][i + 1])
            ob = prof["observable"][a:b]
            if not ob.any():
                continue
            ds = float(np.median(np.diff(prof["s_px"][a:b]))) if b - a > 1 else 1.0
            if ob.sum() * ds >= min_run and not (prof["line"][a:b].any() or (prof["merged_into"][a:b] >= 0).any()
                                                 or internal[a:b].any()):
                out.append(f"vessel {v}: {ob.sum() * ds:.0f} px observable, on no line, merged nowhere")
            for p_, q_ in J._segments(prof["absorbed"][a:b]):
                if (q_ - p_) * ds >= arm_min:
                    out.append(f"vessel {v}: {(q_ - p_ + 1) * ds:.0f} px absorbed")
        if (prof["line"] & ~prof["in_frame"]).any():
            out.append("line samples outside the frame")
    return out


# ======================================================================== files
def save_observable(obs: dict, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obs, fh, separators=(",", ":"), allow_nan=False)


def load_observable(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def to_networkx(obs: dict):
    """The observable image graph as a networkx MultiGraph (node and edge attributes as in the JSON)."""
    import networkx as nx
    G = nx.MultiGraph(**obs["graph"].get("graph", {}))
    for n in obs["graph"]["nodes"]:
        G.add_node(n["id"], **{k_: v for k_, v in n.items() if k_ != "id"})
    for e in obs["graph"]["edges"]:
        G.add_edge(e["source"], e["target"], key=e["key"],
                   **{k_: v for k_, v in e.items() if k_ not in ("source", "target", "key")})
    return G


def save_graphml(obs: dict, path: str):
    """GraphML of the observable image graph: lists as comma-separated strings, the polyline as 'x,y;x,y;...',
    None as -1 for an integer attribute (junction, vessel_node), NaN for a float one, '' for a string one: one
    GraphML type per attribute."""
    import networkx as nx
    G = nx.MultiGraph()
    G.graph.update(kind=obs["kind"], format=FORMAT, k=float(obs["k"]),
                   **{f"rule_{k_}": v for k_, v in obs["rules"].items() if isinstance(v, (int, float, str, bool))})

    def flat(v):
        if isinstance(v, (list, tuple)):
            if v and isinstance(v[0], (list, tuple)):
                return ";".join(",".join(f"{c:g}" for c in p) for p in v)
            return ",".join(str(x) for x in v)
        return v

    def typed(items, skip):
        """Each attribute with one GraphML type: None as -1 where the others are int, NaN where float, '' else
        (review: 'junction' and 'vessel_node' mixed int and '', two keys of one name)."""
        recs = [{k_: flat(v) for k_, v in it.items() if k_ not in skip} for it in items]
        kinds = defaultdict(set)
        for r_ in recs:
            for k_, v in r_.items():
                kinds[k_]
                if v is not None:
                    kinds[k_].add(bool if isinstance(v, bool) else int if isinstance(v, int) else
                                  float if isinstance(v, float) else str)
        fill = {k_: (-1 if t == {int} else float("nan") if t and t <= {int, float} else False if t == {bool} else "")
                for k_, t in kinds.items()}
        for r_ in recs:
            for k_ in kinds:
                v = r_.get(k_)
                if v is None:
                    r_[k_] = fill[k_]
                elif float in kinds[k_] and isinstance(v, int) and not isinstance(v, bool):
                    r_[k_] = float(v)
                elif len(kinds[k_]) > 1 and not kinds[k_] <= {int, float}:
                    r_[k_] = str(v)
        return recs
    for n, r_ in zip(obs["graph"]["nodes"], typed(obs["graph"]["nodes"], ("id",))):
        G.add_node(n["id"], **r_)
    for e, r_ in zip(obs["graph"]["edges"], typed(obs["graph"]["edges"], ("source", "target", "key"))):
        G.add_edge(e["source"], e["target"], key=e["key"], **r_)
    nx.write_graphml(G, path)


# ======================================================================== rasters
def _heat(H, W, x, y, sg, out):
    R_ = int(math.ceil(3 * sg))
    x0, x1 = max(int(math.floor(x)) - R_, 0), min(int(math.ceil(x)) + R_ + 1, W)
    y0, y1 = max(int(math.floor(y)) - R_, 0), min(int(math.ceil(y)) + R_ + 1, H)
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gk = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sg ** 2).astype(np.float32)
        np.maximum(out[y0:y1, x0:x1], gk, out=out[y0:y1, x0:x1])


def _heat_key(t: str) -> str:
    return t.replace("-", "_")


def observed_pieces(run: dict) -> list:
    """The observable stretches of a run's polyline ((n, 2) arrays): the run minus its 'unobserved' samples (a
    line passing through a junction where the vessel itself is not observable, e.g. under a black vein)."""
    P = np.asarray(run["xy"], float).reshape(-1, 2)
    ok = np.ones(len(P), bool)
    for p_, q_ in run.get("unobserved") or ():
        ok[p_:q_ + 1] = False
    return [P[a:b + 1] for a, b in J._segments(ok)]


def observable_rasters(obs: dict, prof: dict | None = None, shape=None, sigma_min: float = 2.0) -> dict:
    """Rasters (H, W) of one image's observable truth (keys without the kind; save_scene stores them in
    labels.npz as observable_<key>_<kind>):
      centreline (int32): the vessel id on a 1 px raster of the observable lines (the runs' observable
        stretches: observed_pieces), -1 elsewhere;
      edge (int32): the image-graph edge id on the same raster (outside junctions), -1 elsewhere;
      lumen (bool): the projected lumen (|d| <= r) of every observable sample (merged ones included);
      dont_care (bool): the lumen + DONT_CARE_MARGIN_PX of the don't-care samples (seen at CNR_DONT_CARE, not
        observable) and of the line samples that are not observable themselves (a line through a junction),
        outside the observable lumen: neither reward nor penalty there;
      junction_heat_<type> (float16; pseudo_T) for each of OBS_TYPES and junction_heat_any: a Gaussian of
        sigma max(sigma_min, radius / 2) at each observable junction's centre (peak 1);
      junction_dont_care (bool): discs at the don't-care junctions (hidden junctions, junctions shown only at
        the permissive level).
    prof: the image's visibility_profiles (needed for lumen and dont_care)."""
    import cv2
    H, W = (int(x) for x in (shape or obs["shape"]))
    cl = np.full((H, W), -1, np.int32)
    ed = np.full((H, W), -1, np.int32)
    for r_ in obs["runs"]:
        for P in observed_pieces(r_):
            P = np.rint(P).astype(np.int32)
            if len(P) >= 2:
                cv2.polylines(cl, [P.reshape(-1, 1, 2)], False, int(r_["vid"]), 1)
            elif len(P) == 1 and 0 <= P[0, 1] < H and 0 <= P[0, 0] < W:
                cl[P[0, 1], P[0, 0]] = r_["vid"]
    for e in obs["graph"]["edges"]:
        P = np.rint(np.asarray(e["xy"], float)).astype(np.int32)
        if len(P) >= 2:
            cv2.polylines(ed, [P.reshape(-1, 1, 2)], False, int(e["id"]), 1)
        elif len(P) == 1 and 0 <= P[0, 1] < H and 0 <= P[0, 0] < W:
            ed[P[0, 1], P[0, 0]] = e["id"]
    out = dict(centreline=cl, edge=ed)
    if prof is not None:
        lum = np.zeros((H, W), np.uint8)
        dc = np.zeros((H, W), np.uint8)
        x, y, r = prof["x_px"], prof["y_px"], prof["radius_px"]
        dcm = prof["dont_care"] | (prof["line"] & ~prof["observable"])
        for mask, img, extra in ((prof["observable"], lum, 0.0), (dcm, dc, DONT_CARE_MARGIN_PX)):
            for i in np.flatnonzero(mask):
                cv2.circle(img, (int(round(float(x[i]))), int(round(float(y[i])))),
                           int(round(float(r[i]) + extra)), 1, -1)
        out["lumen"] = lum.astype(bool)
        out["dont_care"] = dc.astype(bool) & ~out["lumen"]
    heat = {_heat_key(t): np.zeros((H, W), np.float32) for t in OBS_TYPES}
    heat["any"] = np.zeros((H, W), np.float32)
    for j in obs["junctions"]:
        sg = max(sigma_min, 0.5 * float(j["radius"]))
        _heat(H, W, j["x"], j["y"], sg, heat[_heat_key(j["type_observable"])])
        _heat(H, W, j["x"], j["y"], sg, heat["any"])
    for key, v in heat.items():
        out[f"junction_heat_{key}"] = v.astype(np.float16)
    jdc = np.zeros((H, W), np.uint8)
    for d in obs.get("dont_care_junctions", []):
        cv2.circle(jdc, (int(round(d["x"])), int(round(d["y"]))), int(round(max(d["radius"], 3.0))), 1, -1)
    out["junction_dont_care"] = jdc.astype(bool)
    return out


def raster_keys(kind: str) -> list[str]:
    keys = ["centreline", "edge", "lumen", "dont_care"] + [f"junction_heat_{_heat_key(t)}" for t in OBS_TYPES] + \
        ["junction_heat_any", "junction_dont_care"]
    return [f"observable_{k_}_{kind}" for k_ in keys]


# ======================================================================== drawing
OBS_COLORS = {                                     # BGR
    "bifurcation": (40, 200, 40), "confluence": (230, 140, 30), "pseudo-T": (0, 130, 255),
    "crossing": (0, 215, 255), "compound": (40, 40, 240)}
END_STYLE = {"fade": (0, 230, 230), "border": (200, 200, 200), "merged": (255, 255, 0),
             "hidden_junction": (255, 255, 255)}


def _canvas(image, crop, scale):
    import cv2
    base = J._base_view(image)
    H, W = base.shape[:2]
    y0, x0, h, w = crop if crop is not None else (0, 0, H, W)
    base = base[y0:y0 + h, x0:x0 + w]
    f = float(scale)
    if f != 1.0:
        base = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    return base.copy(), np.array([x0, y0], float), f, (y0, x0, h, w)


def _P(xy, off, f, sh=4):
    return tuple(np.rint(((np.asarray(xy, float) - off) * f + (f - 1) / 2) * (1 << sh)).astype(np.int64).tolist())


def _text(out, lines, x=8, y=16):
    import cv2
    for t in lines:
        for cc, th in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.putText(out, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, th, cv2.LINE_AA)
        y += 16


def draw_observable(image, obs: dict, path: str | None = None, crop=None, scale: float = 1.0,
                    prof: dict | None = None, title: str | None = None, legend: bool = True,
                    label: bool | None = None) -> np.ndarray:
    """The observable truth over an image (flattened as scene.flat_view): each edge of the image graph as its
    polyline in its vessel's colour (2 px where it carries merged vessels), joined to its junction centres by a
    thin line; observable junctions as circles in their type_observable colour (squares: merge junctions);
    run ends: fade (yellow ring), border (grey square), merged (cyan diamond), hidden junction (white x);
    with prof, the don't-care samples as grey dots.  crop = (y0, x0, h, w), scale enlarges.  Returns BGR."""
    import cv2
    from .scene import _id_color
    out, off, f, (y0, x0, h, w) = _canvas(image, crop, scale)
    sh = 4
    label = (f >= 2) if label is None else label
    if prof is not None:
        m = prof["dont_care"] & (prof["x_px"] >= x0 - 2) & (prof["x_px"] < x0 + w + 2) & \
            (prof["y_px"] >= y0 - 2) & (prof["y_px"] < y0 + h + 2)
        for i in np.flatnonzero(m)[::3]:
            cv2.circle(out, _P((prof["x_px"][i], prof["y_px"][i]), off, f), int(0.8 * max(f, 1) * (1 << sh)),
                       (150, 150, 150), -1, cv2.LINE_AA, shift=sh)
    nodes = {n["id"]: n for n in obs["graph"]["nodes"]}
    th_e = 1 if f < 2 else 2
    for e in obs["graph"]["edges"]:
        col = _id_color(e["vid"])
        P = [np.asarray(p, float) for p in e["xy"]]
        nu, nw = nodes[e["source"]], nodes[e["target"]]
        for nd, end in ((nu, 0), (nw, -1)):
            if nd["kind"] in ("junction", "hidden_junction") and P:
                cv2.line(out, _P((nd["x"], nd["y"]), off, f), _P(P[end], off, f), col, 1, cv2.LINE_AA, shift=sh)
        if not P:
            cv2.line(out, _P((nu["x"], nu["y"]), off, f), _P((nw["x"], nw["y"]), off, f), col, 1, cv2.LINE_AA,
                     shift=sh)
            continue
        pts = np.array([_P(p, off, f) for p in P], np.int64).reshape(-1, 1, 2)
        th = th_e + (1 if len(e["vids"]) > 1 else 0)
        cv2.polylines(out, [pts], False, col, th, cv2.LINE_AA, shift=sh)
    s_ = max(f, 1.0)
    for n in obs["graph"]["nodes"]:
        c = _P((n["x"], n["y"]), off, f)
        if n["kind"] == "fade":
            cv2.circle(out, c, int(2.5 * s_ * (1 << sh)), END_STYLE["fade"], 1, cv2.LINE_AA, shift=sh)
        elif n["kind"] == "border":
            q = int(2 * s_ * (1 << sh))
            cv2.rectangle(out, (c[0] - q, c[1] - q), (c[0] + q, c[1] + q), END_STYLE["border"], 1, cv2.LINE_AA,
                          shift=sh)
        elif n["kind"] == "merged":
            q = int(3 * s_ * (1 << sh))
            pts = np.array([[c[0], c[1] - q], [c[0] + q, c[1]], [c[0], c[1] + q], [c[0] - q, c[1]]], np.int64)
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, END_STYLE["merged"], 1, cv2.LINE_AA, shift=sh)
        elif n["kind"] == "hidden_junction":
            q = int(2.5 * s_ * (1 << sh))
            cv2.line(out, (c[0] - q, c[1] - q), (c[0] + q, c[1] + q), END_STYLE["hidden_junction"], 1, cv2.LINE_AA,
                     shift=sh)
            cv2.line(out, (c[0] - q, c[1] + q), (c[0] + q, c[1] - q), END_STYLE["hidden_junction"], 1, cv2.LINE_AA,
                     shift=sh)
    for j in obs["junctions"]:
        col = OBS_COLORS.get(j["type_observable"], (200, 200, 200))
        c = _P((j["x"], j["y"]), off, f)
        rr = int(min(max(j["radius"], 4.0), 3 * LAMBDA_PX) * f * (1 << sh))
        if j["source"] == "merge":
            cv2.rectangle(out, (c[0] - rr, c[1] - rr), (c[0] + rr, c[1] + rr), col, 1, cv2.LINE_AA, shift=sh)
        else:
            cv2.circle(out, c, rr, col, 1 if f < 2 else 2, cv2.LINE_AA, shift=sh)
        cv2.circle(out, c, int(2.2 * s_ * (1 << sh)), (0, 0, 0), -1, cv2.LINE_AA, shift=sh)
        cv2.circle(out, c, int(1.6 * s_ * (1 << sh)), col, -1, cv2.LINE_AA, shift=sh)
        if label:
            q = ((np.array([j["x"], j["y"]]) - off) * f).astype(int)
            _text(out, [str(j["id"])], int(q[0] + 5), int(q[1] - 5))
    if legend:
        s = obs.get("summary") or summary(obs)
        rl = obs["rules"]
        lines = ([title] if title else []) + [
            f"observable: CNR >= {rl['cnr_observe']:g}, runs >= {rl['min_run_px']:g} px",
            f"{s['n_runs']} runs, {s['n_junctions']} junctions, {s['n_hidden_junctions']} hidden"]
        _text(out, lines)
        yy = 16 * (len(lines) + 1)
        for name, colr in OBS_COLORS.items():
            cv2.circle(out, (14, yy - 4), 5, colr, -1, cv2.LINE_AA)
            _text(out, [name], 26, yy)
            yy += 16
        _text(out, ["ring: fade  square: border  diamond: merged", "x: hidden junction  grey dots: don't care",
                    "lines: by vessel, thick = merged vessels"], 8, yy)
    if path:
        cv2.imwrite(path, out)
    return out


def draw_complete(image, prof: dict, junctions_all: list | None = None, path: str | None = None, crop=None,
                  scale: float = 1.0, title: str | None = None, legend: bool = True) -> np.ndarray:
    """The complete truth over an image: every vessel near the frame as its centreline in its own colour,
    solid where observable, dashed (every other 2 px) where don't care, dotted grey where invisible; all
    junctions (the permissive rules, hidden ones too) as circles in their truth type's colour
    (junctions.TYPE_COLORS), dashed where type_visible is 'none'."""
    import cv2
    from .scene import _id_color
    out, off, f, (y0, x0, h, w) = _canvas(image, crop, scale)
    sh = 4
    th_e = 1 if f < 2 else 2
    vids, offs = prof["vids"], prof["offsets"]
    for i, v in enumerate(vids.tolist()):
        a, b = int(offs[i]), int(offs[i + 1])
        x, y = prof["x_px"][a:b], prof["y_px"][a:b]
        sel = (x >= x0 - 4) & (x < x0 + w + 4) & (y >= y0 - 4) & (y < y0 + h + 4) & prof["in_frame"][a:b]
        if not sel.any():
            continue
        col = _id_color(int(v))
        st_ = np.where(prof["observable"][a:b], 2, np.where(prof["dont_care"][a:b], 1, 0))
        for q in range(a, b - 1):
            k_ = q - a
            if not (sel[k_] and sel[k_ + 1]):
                continue
            p0, p1 = _P((x[k_], y[k_]), off, f), _P((x[k_ + 1], y[k_ + 1]), off, f)
            if st_[k_] == 2:
                cv2.line(out, p0, p1, col, th_e, cv2.LINE_AA, shift=sh)
            elif st_[k_] == 1:
                if (k_ // 2) % 2 == 0:
                    cv2.line(out, p0, p1, col, 1, cv2.LINE_AA, shift=sh)
            elif k_ % 4 == 0:
                cv2.circle(out, p0, int(0.6 * max(f, 1) * (1 << sh)), (120, 120, 120), -1, cv2.LINE_AA, shift=sh)
    for j in junctions_all or []:
        col = J.TYPE_COLORS.get(j["type"], (200, 200, 200))
        c = _P((j["x"], j["y"]), off, f)
        rr = int(min(max(j["radius"], 4.0), 3 * LAMBDA_PX) * f * (1 << sh))
        if j.get("type_visible", "none") == "none":
            for a0 in range(0, 360, 40):
                cv2.ellipse(out, c, (rr, rr), 0, a0, a0 + 20, col, 1, cv2.LINE_AA, shift=sh)
        else:
            cv2.circle(out, c, rr, col, 1, cv2.LINE_AA, shift=sh)
        cv2.circle(out, c, int(1.5 * max(f, 1) * (1 << sh)), col, -1, cv2.LINE_AA, shift=sh)
    if legend:
        lines = ([title] if title else []) + ["complete: every vessel and junction",
                                              "solid observable, dashed don't care, grey dots invisible",
                                              "circles: all junctions by truth type, dashed: not shown (permissive)"]
        _text(out, lines)
    if path:
        cv2.imwrite(path, out)
    return out


def figure_truth(image, obs: dict, prof: dict, junctions_all, path: str, crops=(), zoom: float = 2.0,
                 full_scale: float = 0.5, title: str = "", crop_titles=None, crop_marks=None) -> str:
    """One figure: the full frame (image | observable | complete, at full_scale) and, per crop (y0, x0, h, w),
    the plain crop | observable | complete at `zoom` (nearest-neighbour: the crop's pixels as they are);
    crop_titles: a line written on each crop's plain panel; crop_marks: (x, y, r) px per crop (or None), a
    dashed white circle drawn on its three panels."""
    import cv2
    rows = []
    a = J._base_view(image)
    b = draw_observable(image, obs, prof=prof, title=f"{title} observable".strip())
    c = draw_complete(image, prof, junctions_all, title=f"{title} complete".strip())
    row = np.hstack([a, b, c])
    if full_scale != 1.0:
        row = cv2.resize(row, None, fx=full_scale, fy=full_scale, interpolation=cv2.INTER_AREA)
    rows.append(row)
    for n_, cr in enumerate(crops):
        y0, x0, h, w = cr
        pa, _, _, _ = _canvas(image, cr, zoom)
        _text(pa, ([crop_titles[n_]] if crop_titles else []) + [f"crop y {y0} x {x0} ({h}x{w}) at {zoom:g}x"])
        pb = draw_observable(image, obs, crop=cr, scale=zoom, prof=prof, legend=False)
        pc = draw_complete(image, prof, junctions_all, crop=cr, scale=zoom, legend=False)
        mk = crop_marks[n_] if crop_marks else None
        if mk is not None:
            c_ = _P((mk[0], mk[1]), np.array([x0, y0], float), zoom)
            rr = int(mk[2] * zoom * 16)
            for pnl in (pa, pb, pc):
                for a0 in range(0, 360, 30):
                    cv2.ellipse(pnl, c_, (rr, rr), 0, a0, a0 + 18, (255, 255, 255), 2, cv2.LINE_AA, shift=4)
        r_ = np.hstack([pa, pb, pc])
        if r_.shape[1] != rows[0].shape[1]:
            fx = rows[0].shape[1] / r_.shape[1]
            r_ = cv2.resize(r_, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA if fx < 1 else cv2.INTER_NEAREST)
        rows.append(r_)
    sep = [np.full((6, rows[0].shape[1], 3), 255, np.uint8)]
    fig = np.vstack([x for r_ in rows for x in (r_, sep[0])][:-1])
    cv2.imwrite(path, fig)
    return path


# ======================================================================== scoring and calibration
def _pts_of_polylines(polylines, step=0.5) -> np.ndarray:
    """Points every `step` px along polylines [(n, 2) arrays of x, y]."""
    out = []
    for P in polylines:
        P = np.asarray(P, float).reshape(-1, 2)
        if len(P) == 1:
            out.append(P)
            continue
        for p, q in zip(P[:-1], P[1:]):
            n = max(1, int(math.ceil(float(np.hypot(*(q - p))) / step)))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(p + t * (q - p))
        out.append(P[-1:])
    return np.concatenate(out) if out else np.zeros((0, 2))


def score_tracing(obs: dict, traced_polylines, traced_junctions=None, prof: dict | None = None,
                  tol_px: float = 3.0, junction_tol_px: float | None = None) -> dict:
    """Score a hand tracing of one image against its observable truth.
    traced_polylines: [(n, 2) x, y px]; traced_junctions: [(x, y)] or [(x, y, type)].
    Centrelines: recall = the share of the truth's observable line samples (the runs' observable stretches,
    1 px) within tol_px of the tracing; precision = the share of traced points within tol_px of the runs
    (through junctions too) or an observable sample (prof: any, merged ones included) or inside a don't-care
    stretch (prof: within tol_px of a don't-care sample): traced don't-care stretches are neither rewarded nor
    penalised.
    Junctions: a traced junction matches the nearest unmatched observable junction within max(radius,
    junction_tol_px (default lambda)); unmatched traced junctions near a don't-care junction (hidden junctions,
    junctions shown only at the permissive level) are ignored.  Returns recall, precision, F1 for both, the
    matched type accuracy, and the counts."""
    from scipy.spatial import cKDTree
    T = _pts_of_polylines(traced_polylines)
    pieces = [P for r_ in obs["runs"] for P in observed_pieces(r_)]
    truth = np.concatenate(pieces) if pieces else np.zeros((0, 2))
    out = {}
    if len(truth) and len(T):
        tt = cKDTree(T)
        rec = float(np.mean(tt.query(truth)[0] <= tol_px))
        good = np.concatenate([np.asarray(r_["xy"], float).reshape(-1, 2) for r_ in obs["runs"]])
        if prof is not None:                         # the lines, and every observable sample (merged ones too)
            good = np.concatenate([good, np.stack([prof["x_px"][prof["observable"]],
                                                    prof["y_px"][prof["observable"]]], 1)])
        d_good = cKDTree(good).query(T)[0]
        ok = d_good <= tol_px
        care = np.ones(len(T), bool)
        if prof is not None and prof["dont_care"].any():
            dc = np.stack([prof["x_px"][prof["dont_care"]], prof["y_px"][prof["dont_care"]]], 1)
            care = ~((cKDTree(dc).query(T)[0] <= tol_px) & ~ok)
        prec = float(np.mean(ok[care])) if care.any() else float("nan")
    else:
        rec, prec = (0.0 if len(truth) else float("nan")), (0.0 if len(T) else float("nan"))
    out.update(centreline_recall=rec, centreline_precision=prec,
               centreline_f1=(2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0),
               n_truth_px=int(len(truth)), n_traced_pts=int(len(T)))
    if traced_junctions is not None:
        tj = list(traced_junctions)
        js = obs["junctions"]
        jt = junction_tol_px if junction_tol_px is not None else LAMBDA_PX
        used, matched, type_ok = set(), 0, 0
        fp = 0
        dcj = obs.get("dont_care_junctions", [])
        for t in tj:
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
                matched += 1
                if len(t) > 2 and t[2] == best["type_observable"]:
                    type_ok += 1
                continue
            if any(math.hypot(d["x"] - x, d["y"] - y) <= max(d["radius"], jt) for d in dcj):
                continue
            fp += 1
        jr = matched / len(js) if js else float("nan")
        jp = matched / (matched + fp) if matched + fp else float("nan")
        out.update(junction_recall=jr, junction_precision=jp,
                   junction_f1=(2 * jr * jp / (jr + jp) if js and matched + fp and jr + jp > 0 else 0.0),
                   junction_type_accuracy=(type_ok / matched if matched else float("nan")), n_truth_junctions=len(js),
                   n_traced_junctions=len(tj), junction_false_positives=fp)
    return out


def score_fit(prof: dict, fitted_polylines, tol_px: float = 3.0, fitted_extra_ok: bool = True) -> dict:
    """Score a fitted spline network (its centrelines as px polylines) against the complete truth of one image:
    the share of truth samples (in the frame) within tol_px of the fit, per visibility level (observable,
    don't care, invisible) and per vessel; and the share of fitted points within tol_px of any truth sample
    (precision: what the fit invents).  A fit should find the observable vessels; the don't-care and invisible
    levels show how far it extrapolates correctly."""
    from scipy.spatial import cKDTree
    T = _pts_of_polylines(fitted_polylines)
    inf = prof["in_frame"]
    P = np.stack([prof["x_px"], prof["y_px"]], 1)
    out = {}
    if len(T):
        d = cKDTree(T).query(P)[0]
        hit = d <= tol_px
    else:
        hit = np.zeros(len(P), bool)
    inv = inf & ~prof["observable"] & ~prof["dont_care"]
    for name, m in (("observable", inf & prof["observable"]), ("dont_care", inf & prof["dont_care"]),
                    ("invisible", inv), ("all", inf)):
        out[f"recall_{name}"] = float(hit[m].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    if len(T):
        dd = cKDTree(P[inf]).query(T)[0] if inf.any() else np.full(len(T), np.inf)
        out["precision"] = float(np.mean(dd <= tol_px))
    else:
        out["precision"] = float("nan")
    per = {}
    for i, v in enumerate(prof["vids"].tolist()):
        a, b = int(prof["offsets"][i]), int(prof["offsets"][i + 1])
        m = inf[a:b]
        if m.any():
            per[int(v)] = float(hit[a:b][m].mean())
    out["per_vessel_recall"] = per
    return out


def traced_by_cnr(prof: dict, traced_polylines, tol_px: float = 3.0, bins=None) -> dict:
    """For recalibrating CNR_OBSERVE on a human tracing of the same image: per CNR bin of the truth's in-frame
    samples, the share a person traced (within tol_px).  The threshold where the share crosses 0.5 (logistic
    fit in log CNR: 'cnr50') is the calibrated CNR_OBSERVE; the shortest traced isolated runs bound
    MIN_RUN_PX."""
    from scipy.spatial import cKDTree
    bins = np.asarray(bins if bins is not None else [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 20, 50, 1e9], float)
    T = _pts_of_polylines(traced_polylines)
    inf = prof["in_frame"]
    cnr = prof["cnr"][inf].astype(np.float64)
    P = np.stack([prof["x_px"][inf], prof["y_px"][inf]], 1)
    hit = (cKDTree(T).query(P)[0] <= tol_px) if len(T) else np.zeros(len(P), bool)
    idx = np.digitize(cnr, bins) - 1
    share = [float(hit[idx == i].mean()) if (idx == i).any() else float("nan") for i in range(len(bins) - 1)]
    n = [int((idx == i).sum()) for i in range(len(bins) - 1)]
    cnr50 = float("nan")
    x = np.log(np.maximum(cnr, 1e-3))
    if hit.any() and (~hit).any():
        a, b = 0.0, 1.0
        for _ in range(50):                          # Newton steps of a logistic fit p = 1 / (1 + exp(-(a + b x)))
            z = a + b * x
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(p * (1 - p), 1e-9)
            g = np.array([np.sum(hit - p), np.sum((hit - p) * x)])
            Hm = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
            step = np.linalg.solve(Hm + 1e-9 * np.eye(2), g)
            a, b = a + step[0], b + step[1]
            if np.abs(step).max() < 1e-8:
                break
        if b > 0:
            cnr50 = float(np.exp(-a / b))
    return dict(bins=bins.tolist(), share_traced=share, n=n, cnr50=cnr50)
