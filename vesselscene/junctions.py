"""Image-junction ground truth: every place where vessels meet or overlap in the 2-D image.

Intersection identification on a real still finds the places where vessels
meet or overlap in the image, classifies them (a fork and a crossing look
alike in 2-D) and pairs their arms (which arms belong to the same vessel), so
that the vessel graph can be built by following arms between junctions.
This module makes the ground truth for that from a scene's VesselGraph
(graph.py) and, for visibility, the image's rendered optical density and
noise.  The junctions differ between a scene's average and its frame (the
frame's own focus, red-cell filling and noise): one list per image.

    image_junctions(graph, k, origin, shape, od, noise, kind)   the junctions (list of JSON-able dicts)
    junction_samples(...) + junctions_from_samples(S, graph)     the same in two steps: the sampling with its
                                                                 CNR (the costly half), then the rules (the seen
                                                                 thresholds are parameters: truth.py reuses one
                                                                 sampling for its observable truth)
    visible_subset(all)                                          the default list from a keep_hidden=True one
    scene_junctions(scene, kind)                                 the same for a make_scene() Scene
    inputs_from_saved(folder, kind)                              the arguments for a save_scene() folder
    junction_rasters(junctions, shape)                           heatmaps per type, arm-direction map
    draw_junctions(image, junctions, path)                       an overlay PNG

Units: image px (pixel (0, 0) at the d_c position `origin`, k px per d_c,
x right, y down; angles in degrees, atan2(dy, dx) in image coordinates, so
90 deg points down), depths in d_c, OD in Np, lambda = LAMBDA_PX = 11.9 px.

Definitions
-----------
Seen.  Every vessel near the frame is sampled every SAMPLE_PX px.  A sample's
CNR is scene.visibility's (the vessel's band-filtered closed-form response
over the background's RMS in the same DoG band), times, when the image's OD
is given, the ratio of the contrast the rendered OD shows (centreline minus
the lighter flank r + 2.5 s + 1 px away) to the closed form's, measured at
ISOLATED samples (no stronger-than-a-fifth neighbour on the centre or on
both flanks) and interpolated along the vessel between them: a frame's
red-cell gap, an unperfused capillary, a vessel vanishing under a black
vein is not seen, while one squeezed between two neighbours is not mistaken
for faint.  SEEN samples: CNR >= CNR_VISIBLE (1.5) with gaps shorter than
1 lambda bridged, in runs >= lambda / 2.  Only seen samples take part in
what follows: a vessel the image does not show neither hides another nor
links junctions.

Overlap.  Vessel a's cross-section at a sample (the segment of half-length
r_a across it) OVERLAPS vessel b if b's centreline comes within r_b of it
(the projected lumens overlap), and is UNRESOLVED from b if within r_b +
g_res: g_res, the resolution gap, is the wall gap at which two parallel
blurred vessels show a 25 % dip between them (about Rayleigh's 26 %); on the
renderer's staircase profiles it depends only on rho = r / sigma (the
thinner radius over the RMS of the two total blurs): 2.7 sigma at rho 0,
1.1 sigma at rho 1, 0 from rho 2.5 (a round lumen's chord falls to 0 at its
wall, so wide vessels show a dip as soon as their lumens part).  For a thin
vessel crossing a wide vein at 90 deg this gives the overlap rectangle (the
vein's cross-sections over +-(r_thin + g) along it, the thin vessel's over
+-(r_vein + g)), not a disc.  Contiguous stretches of a w.r.t. b (gaps up to
BRIDGE samples bridged) are runs.

Events (the truth, from the graph):
  * node: a graph node with >= 3 vessels inside the frame: 'fork' (1 in,
    >= 2 out), 'confluence' (>= 2 in, 1 out), 'anastomosis' (>= 2 in and
    >= 2 out).  Inlets and outlets (the padded domain's border) and nodes
    outside the frame make no event: a vessel leaving the frame is not a
    junction.
  * crossing: graph.crossings() (centrelines crossing at different depths,
    away from a shared node) inside the frame.
  * touching: two seen vessels unresolved over a stretch where they neither
    cross nor share a node (pairs, bundles, any two vessels that kiss); one
    event per connected group of runs of a pair, unless the stretch lies on
    both vessels within node / crossing cores (a thin vessel crossing a
    confluence overlaps the lumens of the vessels whose centreline it does
    not cross: that is part of the crossing).  lumen_overlap: whether their
    projected lumens overlap or only the blur merges them.  A stretch
    longer than TOUCH_SPLIT_PX (2 lambda) is two events, part 'start' and
    'end', where the vessels meet and where they part, each with
    TOUCH_END_PX of the merged stretch (the arms into it are merged_with
    each other); otherwise one, part 'whole'.
  * vis_end (truth type 'pseudo-T'): a seen stretch of A ending inside the
    frame, not at a node, with another seen vessel B within 1 lambda (wall
    to wall) ahead of it (within VIS_END_ANGLE of A's direction) or touching
    it: A dives or fades at B and the image shows a T where there is no fork.
Each event has two cores per vessel: the overlap core (where its lumen
overlaps the other's) and the unresolved core (where it is unresolved from
it; at a crossing at least the straight-line extent (r_b + g + r_a |cos t|)
/ sin t, sin t >= 0.1, when b is seen there).

Image junctions.  An event's region (footprint) is where its lumens overlap:
the union of the pairwise intersections of its vessels' overlap-core bands
(half-width r + MERGE_SIG sigma, extended by MERGE_SIG sigma along the
vessel; a thin vessel crossing a wide vein: its path across the vein, not
the vein's band), a node's disc (thinnest incident lumen + sigma) and a
vis_end's gap, clipped to CLIP_LAMBDA lambda (or the widest lumen + 2
sigma) around the event's centre: joining is local, an elongated overlap (a
shallow crossing, two vessels braided along each other) does not swallow
everything along it.  Events whose regions intersect (lumens within about 2
sigma) are joined, nearest first, while every two events of the junction
stay closer than the sum of their extents (complete linkage: a vessel
crossed every 30 px is not one giant junction), or when one group's event
centres all lie inside the other's region.  A vis_end whose pair also
crosses / touches in the same junction, or which lies at a node of both
vessels, is dropped (the other event and the arms' visibility describe it).
The junction's centre is its events' medoid, its radius the farthest point
of its region.

Junctions whose claims on a common vessel (below) overlap by at least
CORE_JOIN of the shorter one are joined too, while all their events stay
within JOIN_DIAM_PX (2 lambda) of each other (complete linkage): one lies
in the other's unresolved stretch (v2 fix pass).

Types: 'bifurcation' (a fork), 'confluence', 'anastomosis', 'crossing',
'touching', 'pseudo-T' (a vis_end), 'compound' (>= 2 events: a crossing next
to a fork, two crossings on one vessel, ...).  type_visible: what a perfect
observer of THIS image would call it, from its SEEN LINES (_type_visible):
the seen (visible or faint) arms, arms of different vessels still
unresolved where they leave ('merged_with') counting as one line.  At most
2 seen lines: 'none' (a line, a bend, an end).  An event shows by its own
vessels' lines (a node: >= 3 of its vessels with a line of their own; a
crossing / touching: both vessels and >= 3 lines; a vis_end: both).  One
such event explaining every seen line: its type, except a crossing or
touching with 3 lines ('pseudo-T': the image shows a T) and a vis_end with
>= 4 ('crossing': the fading vessel shows again beyond the other); lines
of other events too, or >= 2 events showing: 'compound'.  (v2 fix pass:
the visible type came from whichever events survived, so pseudo-Ts showed
4-5 arms, forks had fewer than 3, and 'none' had 3 or more.)

Claims and arms: each vessel's merged unresolved cores in the junction,
clipped to the stretch within the junction's radius of its centre
('clipped' arms where the vessel is still unresolved beyond, as in a
shallow crossing or a daughter running along its sibling), are its CLAIM
on the vessel; where claims of different junctions overlap the samples are
shared out by a 1-D Voronoi split along the vessel around each claim's
anchor (its events' samples), so no stretch of a vessel belongs to two
junctions (_claims; 'claim_px' per vessel in the output).  The arm: the
margin ARM_MARGIN_SIG sigma + 1 px beyond the claim, but at most halfway to
the next junction's claim along the vessel ('crowded' if that leaves less
than half the margin); where the next junction's claim starts at the next
sample, the arm sits at the meeting point and is 'linked' to that junction
(its id), which puts the vessel's opposite arm there too: the vessel is not
resolved between them, and the arms can still be followed from junction to
junction.  A side has no arm only where the vessel ends at a node event OF
THIS junction (v2 fix pass, correctness review defect 1: a core that ran
into a node held by another junction lost its arm).  At the arm point:
direction (outward), width 2r, blur (own and total with the image's), depth,
peak OD (closed form after the PSF), contrast (the vessel's own: the closed
form's centre-to-flank contrast times the ratio the rendered OD shows at its
isolated samples; 'contrast_image' the raw centre-to-flank difference in the
rendered OD, which a neighbour contaminates) and CNR (median over the next
lambda outward), visibility ('visible' CNR >= 3, 'faint' >= 1.5,
'invisible', 'out_of_frame'), the visible run beyond it, flow ('in': blood
flows along it into the junction).

Partition: arms of the same vessel are one group (a vessel through a
crossing: two arms in one group).  At a fork the parent and the daughters
are different vessels (singletons); the lineage (parents -> children per
node) is recorded for velocity work, and pairs_visual is the pairing a
straightness rule would make: 'same' vessel, 'lineage' (parent and child),
'sibling' (two daughters of a fork, two tributaries of a confluence),
'through' (two siblings in line whose trunk is narrower than THROUGH_W x
the narrower of them, or not seen: the image shows a through vessel with a
side branch) or 'unrelated' (a merge of two vessels waiting to happen).
check_junctions(junctions, graph) lists consistency problems (none
expected): arms of one vessel at one point, arms inside another junction's
claim, linked arms without their partner, crossing vessels missing an arm.

ambiguous (the image cannot decide the pairing): a crossing or touching
below SHALLOW_DEG ('shallow_angle'); a fork / confluence whose two branches
are nearly equal (r ratio > SYM_R) and symmetric (turn margin < SYM_TURN
deg; 'symmetric_fork', about which daughter continues the parent); an
in-frame arm invisible ('invisible_arm'); two visible arms of different
vessels still unresolved where they leave ('merged_arms'); a straightness
pairing of unrelated vessels ('misleading_geometry').  Listed in
ambiguous_reasons with 'at_border' (an arm out of the frame) and 'crowded'
(an arm cut short by a neighbouring junction), which do not count as
ambiguous.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr

from . import render as R
from .graph import VesselGraph

# Mirrors of scene.py's visibility constants (scene imports this module; a test checks they agree).
LAMBDA_PX = 11.9                  # the real stills' median vessel FWHM (px)
CNR_REQUIRED, CNR_VISIBLE = 3.0, 1.5

TYPES = ("bifurcation", "confluence", "anastomosis", "crossing", "touching", "pseudo-T", "compound")
VISIBLE_TYPES = ("none",) + TYPES
VIS_NAMES = ("visible", "faint", "invisible", "out_of_frame")
SAMPLE_PX = 1.0                   # px: sampling along every vessel
BRIDGE = 3                        # samples: a run's gaps of up to BRIDGE - 1 samples are bridged
PAD_PX = 48.0                     # px: vessels sampled within this margin around the frame
MERGE_SIG = 1.0                   # footprint: lumen + this many total blur sigmas (two events join ~2 sigma apart)
ARM_MARGIN_SIG = 2.0              # arms measured this many sigmas (+ 1 px) beyond the unresolved core
SHALLOW_DEG = 15.0                # crossings / touching below this angle: the pairing is ambiguous
SYM_R, SYM_TURN = 0.85, 10.0      # forks: branches nearly equal (r ratio) and symmetric (turn margin, deg)
VISUAL_TURN_MAX = 45.0            # deg: a straightness rule pairs two arms that turn by less than this
THROUGH_W = 0.75                  # two siblings in line, the trunk narrower than this x the narrower: 'through'
VIS_END_ANGLE = 60.0              # deg: B must lie ahead of A's fading end within this angle (or touch it)
RUN_CAP_PX = 4 * LAMBDA_PX        # an arm's visible run is followed this far
TOUCH_SPLIT_PX = 2 * LAMBDA_PX    # a touching stretch longer than this: two junctions, where they meet and part
TOUCH_END_PX = 0.5 * LAMBDA_PX    # ... each with this much of the merged stretch as its core
CLIP_LAMBDA = 2.0                 # an event's region for joining reaches at most this many lambda from its centre
FOOTPRINT_CORES = "lcores"        # the cores events are joined by: 'lcores' (lumen overlap) or 'cores' (unresolved)
DIP_MIN = 0.25                    # the dip that resolves two vessels (_GRES was computed with it)
# g_res / sigma against rho = r / sigma (equal parallel vessels, the renderer's staircase, 25 % dip; the table
# hardly depends on the contrast law: centre OD 0.1-1.0 Np, f 0.05-0.3 within 0.05; unequal pairs within ~25 %;
# _scratch/v2/junctions/dip_calib.py)
_RHO = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
_GRES = np.array([2.73, 2.57, 2.38, 2.2, 2.03, 1.86, 1.7, 1.4, 1.13, 0.84, 0.59, 0.39, 0.21, 0.0])
# The background's RMS per DoG band (Np), when no noise is given: medians over the 12 healthy v1 scenes
# (_scratch/v1/fin2) by band_rms_from_image (_scratch/v2/junctions/noise_table.py; that estimate is 0.84-1.09x,
# median 0.93x, the exact scene.background_band_rms on 2 fresh scenes: noise_check.py).
DEFAULT_BAND_RMS = {
    "average": {1: 0.0005, 2: 0.0010, 4: 0.0022, 8: 0.0041, 16: 0.0063, 32: 0.0084},
    "frame": {1: 0.0041, 2: 0.0023, 4: 0.0025, 8: 0.0041, 16: 0.0063, 32: 0.0084},
}
DEFAULT_IMAGE_BLUR = {"average": 0.85, "frame": 0.7}     # px: image formation's blur (scene.params['formed'])


def _scene():
    from . import scene as SC          # lazy: scene imports this module
    return SC


def g_res(r_a, r_b, s_a, s_b) -> np.ndarray:
    """The resolution gap (px): two vessels of radii r_a, r_b (px) and total
    blurs s_a, s_b (px) whose walls are closer than this look like one."""
    sb = np.sqrt(0.5 * (np.asarray(s_a, float) ** 2 + np.asarray(s_b, float) ** 2))
    rho = np.minimum(r_a, r_b) / np.maximum(sb, 1e-6)
    return sb * np.interp(rho, _RHO, _GRES, right=0.0)


def _bridge(ok: np.ndarray, gap: int) -> np.ndarray:
    """Fill runs of False shorter than `gap` samples between Trues."""
    ok = ok.copy()
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return ok
    d = np.diff(idx)
    for i in np.flatnonzero((d > 1) & (d <= gap)):
        ok[idx[i]:idx[i + 1]] = True
    return ok


def _segments(ok: np.ndarray):
    """(start, end) inclusive index pairs of the runs of True."""
    if not ok.any():
        return []
    e = np.diff(np.r_[0, ok.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1).tolist(), (np.flatnonzero(e == -1) - 1).tolist()))


def _in_frame(p, shape, margin=0.0) -> bool:
    H, W = shape
    return bool((-0.5 - margin <= p[0] <= W - 0.5 + margin) and (-0.5 - margin <= p[1] <= H - 0.5 + margin))


# ======================================================================== sampling
class _Samples:
    """Every vessel near the frame sampled every SAMPLE_PX px of arclength:
    centre C (px), unit tangent T (downstream) and normal N, radius r (px),
    depth z, own blur s and total blur st (with the image's), the chord
    law's a and f, in-frame and near-frame flags; visibility() adds the
    closed-form peak OD after the PSF, the CNR, the measured contrast and
    the seen flags, and the global arrays of the seen samples."""

    def __init__(self, graph: VesselGraph, k, origin, shape, optics, blur_px, fill=None, pad=PAD_PX):
        H, W = shape
        self.shape = (H, W)
        self.v = {}
        fill = fill or {}
        for vid, e in graph.vessels.items():
            c = k * (np.asarray(e.ctrl, float) - origin)
            if c[:, 0].max() < -pad or c[:, 0].min() > W - 1 + pad or c[:, 1].max() < -pad or \
                    c[:, 1].min() > H - 1 + pad:
                continue
            tr = R._Track(e, k, origin, optics, fill.get(vid))
            L = float(tr.Lpx)
            n = max(2, int(math.ceil(L / SAMPLE_PX)) + 1)
            sig = np.linspace(0.0, L, n)
            q = tr.at(sig)
            C = q["C"]
            near = (C[:, 0] >= -pad) & (C[:, 0] <= W - 1 + pad) & (C[:, 1] >= -pad) & (C[:, 1] <= H - 1 + pad)
            if not near.any():
                continue
            inf = (C[:, 0] >= -0.5) & (C[:, 0] <= W - 0.5) & (C[:, 1] >= -0.5) & (C[:, 1] <= H - 0.5)
            T = q["T"]
            st = np.sqrt(q["s"] ** 2 + blur_px ** 2)
            self.v[vid] = dict(vid=vid, sig=sig, ds=L / (n - 1), n=n, L=L, C=C, T=T,
                               N=np.stack([-T[:, 1], T[:, 0]], 1), r=q["r"], z=q["z"], s=q["s"], st=st,
                               a=q["a"], f=q["f"], near=near, inf=inf, cls=e.cls,
                               stagnant=bool(e.info.get("stagnant", False)) if isinstance(e.info, dict) else False)

    def visibility(self, od, rms: dict, optics, blur_px):
        SC = _scene()
        bands = sorted(rms)
        noise = np.maximum(np.array([rms[b] for b in bands], float), 1e-9)
        for d in self.v.values():
            tb = dict(s=d["s"], a=d["a"], f=d["f"], r=d["r"])
            pk, resp = SC.band_response(tb, optics, blur_px, bands)
            d["peak"] = pk
            d["cnr_cf"] = (resp / noise[None, :]).max(1)
            d["dfl"] = d["r"] + 2.5 * d["s"] + 1.0
        if od is not None:
            self._od_ratios(od, optics)
        for d in self.v.values():
            if od is None:
                d["ratio"], d["contrast"] = np.ones(d["n"]), d["peak"].copy()
                d["contrast_own"] = d["peak"].copy()
            else:                       # the vessel's own contrast as this image shows it (no neighbour's OD)
                d["contrast_own"] = np.where(d["inf"], d["c_cf"] * d["ratio"], np.nan)
            d["cnr"] = d["cnr_cf"] * d["ratio"]
        self.set_seen()

    def set_seen(self, cnr_seen: float | None = None, cnr_visible: float | None = None,
                 bridge_px: float | None = None, min_px: float | None = None, in_frame_only: bool = False):
        """The SEEN samples (module docstring) for the given rules, and the
        global arrays of them: CNR >= cnr_seen (CNR_VISIBLE), gaps shorter
        than bridge_px (1 lambda) bridged, in runs >= min_px (lambda / 2);
        in_frame_only: only in-frame samples (truth.py's observable truth:
        a vessel just outside the frame is not observable).  cnr_visible
        (CNR_REQUIRED): an arm's 'visible' level ('faint' from cnr_seen).
        The CNR per sample is visibility()'s; only the thresholds change, so
        one sampling serves several rules (truth.py)."""
        self.cnr_seen = float(CNR_VISIBLE if cnr_seen is None else cnr_seen)
        self.cnr_vis = float(CNR_REQUIRED if cnr_visible is None else cnr_visible)
        self.bridge_px = float(LAMBDA_PX if bridge_px is None else bridge_px)
        self.min_px = float(0.5 * LAMBDA_PX if min_px is None else min_px)
        self.in_frame_only = bool(in_frame_only)
        for d in self.v.values():
            raw = d["cnr"] >= self.cnr_seen
            if in_frame_only:
                raw &= d["inf"]
            ok = _bridge(raw, max(1, int(round(self.bridge_px / d["ds"]))))
            if in_frame_only:
                ok &= d["inf"]
            seen = np.zeros(d["n"], bool)
            for a, b in _segments(ok):
                if (b - a) * d["ds"] >= self.min_px:
                    seen[a:b + 1] = True
            d["seen"] = seen
        vids = list(self.v)
        sel = [np.flatnonzero(self.v[v]["near"] & self.v[v]["seen"]) for v in vids]
        if vids and sum(len(x) for x in sel):
            self.gvid = np.concatenate([np.full(len(x), v, np.int64) for v, x in zip(vids, sel)])
            self.gloc = np.concatenate(sel).astype(np.int64)
            for key in ("C", "N", "r", "st"):
                setattr(self, "g" + key, np.concatenate([self.v[v][key][x] for v, x in zip(vids, sel)]))
        else:
            self.gvid = self.gloc = np.zeros(0, np.int64)
            self.gC = self.gN = np.zeros((0, 2))
            self.gr = self.gst = np.zeros(0)

    def _od_ratios(self, od, optics):
        """Per sample, the ratio of the contrast the rendered OD shows to the
        closed form's (_od_contrast), measured where the sample is ISOLATED
        and interpolated along the vessel between isolated samples (the
        filling and fading are properties along a vessel; merging with a
        neighbour is geometry, left to the overlap tests).  Isolated: no
        other vessel of at least a fifth of its peak OD reaches (r + 2 sigma)
        its centre, nor both of its flanks (the measure takes the lighter
        flank).  A vessel without an isolated sample in the frame keeps the
        closed form (ratio 1)."""
        vids = list(self.v)
        sel = [np.flatnonzero(self.v[v]["near"]) for v in vids]
        gv = np.concatenate([np.full(len(x), v, np.int64) for v, x in zip(vids, sel)])
        C, N, r, st, pk, dfl = (np.concatenate([self.v[v][key][x] for v, x in zip(vids, sel)])
                                for key in ("C", "N", "r", "st", "peak", "dfl"))
        reach = r + 2.0 * st
        i, j = _candidate_pairs(C, np.maximum(dfl, reach), gv)
        cbad, lbad, rbad = (np.zeros(len(C), bool) for _ in range(3))
        for a, b in ((i, j), (j, i)):
            strong = pk[b] >= 0.2 * pk[a]
            a_, b_ = a[strong], b[strong]
            off = N[a_] * dfl[a_][:, None]
            for flag, P in ((cbad, C[a_]), (lbad, C[a_] + off), (rbad, C[a_] - off)):
                hit = np.hypot(*(P - C[b_]).T) < reach[b_]
                flag[a_[hit]] = True
        bad = cbad | (lbad & rbad)
        contaminated = dict(zip(vids, np.split(bad, np.cumsum([len(x) for x in sel])[:-1])))
        for v, x in zip(vids, sel):
            d = self.v[v]
            raw, c_img = _od_contrast(d, od, optics)
            iso = np.zeros(d["n"], bool)
            iso[x] = ~contaminated[v]
            iso &= d["inf"]
            k_ = np.flatnonzero(iso)
            if len(k_):
                rk = raw[k_]
                if len(rk) >= 5:
                    rk = ndi.median_filter(rk, size=5, mode="nearest")
                ratio = np.interp(np.arange(d["n"]), k_, rk)
            else:
                ratio = np.ones(d["n"])
            d["ratio"] = np.where(d["inf"], ratio, 1.0)
            d["contrast"] = c_img
            d["isolated"] = iso

    def margin(self, vid, i) -> int:
        """The arm margin (samples) at sample i of vessel vid: ARM_MARGIN_SIG
        total blurs + 1 px, at least BRIDGE."""
        d = self.v[vid]
        return max(BRIDGE, int(math.ceil((ARM_MARGIN_SIG * d["st"][min(max(i, 0), d["n"] - 1)] + 1.0) / d["ds"])))

    def sig_samples(self, vid, i) -> int:
        """MERGE_SIG total blurs at sample i, in samples."""
        d = self.v[vid]
        return int(math.ceil(MERGE_SIG * d["st"][min(max(i, 0), d["n"] - 1)] / d["ds"]))


def _profile(d, optics, blur_px, dist) -> np.ndarray:
    """The closed-form cross-section (Np) after the PSF at distance dist (px)
    from the centreline (the renderer's staircase, own blur + blur_px + halo)."""
    s = np.sqrt(d["s"] ** 2 + blur_px ** 2)
    Wk = R.box_weights(d["a"], d["f"])
    w = d["r"][:, None] * R.BOX_C
    dist = np.broadcast_to(np.asarray(dist, float), s.shape)[:, None]
    terms = R._psf_terms(optics)
    terms = [(1.0 - sum(w_ for w_, _ in terms), 0.0)] + terms
    out = np.zeros(len(s))
    for wt, st in terms:
        S = np.sqrt(s ** 2 + st ** 2)[:, None]
        out += wt * (Wk * (ndtr((w - dist) / S) + ndtr((w + dist) / S) - 1.0)).sum(1)
    return out


def _od_contrast(d, od, optics):
    """(ratio, contrast) along a vessel: the contrast the rendered OD shows
    (the OD on the centreline minus the lighter of the two flanks dfl = r +
    2.5 s + 1 px away; NaN outside the frame) and its ratio to the closed
    form's same difference (own blur: the OD raster is before image
    formation), clipped to [0, 1]."""
    dfl = d["dfl"]
    C, N = d["C"], d["N"]

    def at(P):
        return ndi.map_coordinates(od, [P[:, 1], P[:, 0]], order=1, mode="nearest")
    c_img = at(C) - np.minimum(at(C + N * dfl[:, None]), at(C - N * dfl[:, None]))
    c_cf = _profile(d, optics, 0.0, 0.0) - _profile(d, optics, 0.0, dfl)
    d["c_cf"] = c_cf
    ratio = np.clip(c_img / np.maximum(c_cf, 1e-6), 0.0, 1.0)
    return ratio, np.where(d["inf"], c_img, np.nan)


# ======================================================================== runs
def _candidate_pairs(C, h, vid):
    """Index pairs (i, j) of samples of different vessels closer than
    h_i + h_j, by KD-trees per bucket of h (a wide vein's reach does not
    inflate the thin vessels' queries)."""
    from scipy.spatial import cKDTree
    edges = np.array([6.0, 12.0, 24.0, 48.0, 96.0])
    bk = np.digitize(h, edges)
    idx = [np.flatnonzero(bk == i) for i in range(len(edges) + 1)]
    trees = [cKDTree(C[ix]) if len(ix) else None for ix in idx]
    hmax = [float(h[ix].max()) if len(ix) else 0.0 for ix in idx]
    out_i, out_j = [], []
    for A in range(len(idx)):
        if not len(idx[A]):
            continue
        for B in range(A, len(idx)):
            if not len(idx[B]):
                continue
            rad = hmax[A] + hmax[B]
            if A == B:
                p = trees[A].query_pairs(rad, output_type="ndarray")
                i, j = idx[A][p[:, 0]], idx[A][p[:, 1]]
            else:
                M = trees[A].sparse_distance_matrix(trees[B], rad, output_type="ndarray")
                i, j = idx[A][M["i"]], idx[B][M["j"]]
            keep = (vid[i] != vid[j]) & (np.hypot(*(C[i] - C[j]).T) < h[i] + h[j])
            out_i.append(i[keep])
            out_j.append(j[keep])
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64)


def _pair_records(S: _Samples):
    """(unresolved, overlap) records (a, b, i on a, j on b): sample i of a is
    unresolved from / overlaps sample j of b (module docstring)."""
    empty = np.zeros((0, 4), np.int64)
    if len(S.gvid) < 2:
        return empty, empty
    h = S.gr + 2.75 * S.gst + 0.5                    # h_a + h_b >= r_a + r_b + g_res (g_res <= 2.73 max sigma)
    gi, gj = _candidate_pairs(S.gC, h, S.gvid)
    gr = g_res(S.gr[gi], S.gr[gj], S.gst[gi], S.gst[gj])
    res, lum = [], []
    for p, q in ((gi, gj), (gj, gi)):                # a = the owner of p, b = the owner of q
        w = S.gC[q] - S.gC[p]
        Np_ = S.gN[p]
        t = np.clip((w * Np_).sum(1), -S.gr[p], S.gr[p])
        dist = np.hypot(*(w - t[:, None] * Np_).T)
        for out, lim in ((res, S.gr[q] + gr), (lum, S.gr[q])):
            f = dist < lim
            out.append(np.stack([S.gvid[p][f], S.gvid[q][f], S.gloc[p][f], S.gloc[q][f]], 1))
    return np.concatenate(res), np.concatenate(lum)


class _Runs:
    """For each ordered pair (a, b) the runs: contiguous stretches (sample
    indices i0..i1 on a) of records, with the range j0..j1 of b's samples
    that cause them."""

    def __init__(self, rec: np.ndarray):
        self.by_pair = defaultdict(list)
        self.used = set()
        if not len(rec):
            self.a = self.b = self.i0 = self.i1 = self.j0 = self.j1 = np.zeros(0, np.int64)
            return
        o = np.lexsort((rec[:, 2], rec[:, 1], rec[:, 0]))
        a, b, ia, jb = rec[o].T
        brk = (np.diff(a) != 0) | (np.diff(b) != 0) | (np.diff(ia) > BRIDGE)
        st = np.r_[0, np.flatnonzero(brk) + 1]
        self.a, self.b = a[st], b[st]
        self.i0 = ia[st]
        self.i1 = np.maximum.reduceat(ia, st)
        self.j0 = np.minimum.reduceat(jb, st)
        self.j1 = np.maximum.reduceat(jb, st)
        for rid, (x, y) in enumerate(zip(self.a.tolist(), self.b.tolist())):
            self.by_pair[(x, y)].append(rid)

    def from_end(self, a, b, end, n, chain):
        """The extent (lo, hi) on a of a's runs w.r.t. b chained from a's end
        (0: index 0, 1: the last index): runs starting within `chain` samples
        of the reach so far.  Marks them used."""
        rids = self.by_pair.get((a, b), [])
        if end == 0:
            reach = hi = 0
            for rid in sorted(rids, key=lambda r: self.i0[r]):
                if self.i0[rid] <= reach + chain:
                    reach = hi = max(reach, int(self.i1[rid]))
                    self.used.add(rid)
            return 0, hi
        reach = lo = n - 1
        for rid in sorted(rids, key=lambda r: -self.i1[r]):
            if self.i1[rid] >= reach - chain:
                reach = lo = min(reach, int(self.i0[rid]))
                self.used.add(rid)
        return lo, n - 1

    def around(self, a, b, i, chain):
        """The extent on a of a's runs w.r.t. b around index i (the run within
        `chain` samples of i, grown by the runs within `chain` of it).  Marks
        them used.  None if there is none."""
        rids = sorted(self.by_pair.get((a, b), []), key=lambda r: self.i0[r])
        lo = hi = None
        for rid in rids:
            if self.i0[rid] - chain <= i <= self.i1[rid] + chain:
                lo, hi = int(self.i0[rid]), int(self.i1[rid])
                self.used.add(rid)
                break
        if lo is None:
            return None
        grown = True
        while grown:
            grown = False
            for rid in rids:
                if rid not in self.used and self.i0[rid] <= hi + chain and self.i1[rid] >= lo - chain:
                    lo, hi = min(lo, int(self.i0[rid])), max(hi, int(self.i1[rid]))
                    self.used.add(rid)
                    grown = True
        return lo, hi

    def overlaps(self, a, b, lo, hi) -> bool:
        return any(self.i0[r] <= hi and self.i1[r] >= lo for r in self.by_pair.get((a, b), []))


# ======================================================================== events
def _node_events(graph, S, res, lum, k, origin, shape):
    out = []
    deg = graph.degrees()
    for nid, node in graph.nodes.items():
        i_, o_ = deg.get(nid, (0, 0))
        if i_ + o_ < 3:
            continue
        p = k * (np.asarray(node.xy, float) - origin)
        if not _in_frame(p, shape, margin=PAD_PX):
            continue
        ntype = "fork" if (i_ == 1 and o_ >= 2) else "confluence" if (o_ == 1 and i_ >= 2) else "anastomosis"
        inc = [(v, 0) for v in graph.out_vessels(nid)] + [(v, 1) for v in graph.in_vessels(nid)]
        inc = [(v, end) for v, end in inc if v in S.v]
        cores, lcores, ends = {}, {}, {}
        for v, end in inc:
            n = S.v[v]["n"]
            e0 = 0 if end == 0 else n - 1
            lo = hi = llo = lhi = e0
            for w, _ in inc:
                if w != v:
                    a, b = res.from_end(v, w, end, n, S.margin(v, e0))
                    lo, hi = min(lo, a), max(hi, b)
                    a, b = lum.from_end(v, w, end, n, BRIDGE)
                    llo, lhi = min(llo, a), max(lhi, b)
            if not S.v[v]["seen"][lo:hi + 1].any():          # not seen at the node: measure it outside the union
                e = int(math.ceil(R.junction_radius(graph, nid, k) / S.v[v]["ds"]))
                lo, hi = (0, min(max(hi, e), n - 1)) if end == 0 else (max(min(lo, n - 1 - e), 0), n - 1)
            cores[v], lcores[v], ends[v] = (lo, hi), (llo, lhi), end
        if not _in_frame(p, shape):                 # a node outside the frame: its runs are used up, no event
            continue
        rmin = min((float(S.v[v]["r"][0 if e == 0 else -1] + MERGE_SIG * S.v[v]["st"][0 if e == 0 else -1])
                    for v, e in inc), default=1.0)
        out.append(dict(kind="node", ntype=ntype, nid=int(nid), xy=p, vessels=[v for v, _ in inc], cores=cores,
                        lcores=lcores, ends=ends, at={v: (0 if e == 0 else S.v[v]["n"] - 1) for v, e in inc},
                        parents=[v for v, e in inc if e == 1],
                        children=[v for v, e in inc if e == 0], disc=(p, rmin)))
    return out


def _crossing_events(graph, S, res, lum, k, origin, shape, crossings):
    out = []
    crs = crossings if crossings is not None else graph.crossings()
    for c in crs:
        p = k * (np.array([c["x"], c["y"]], float) - origin)
        if not _in_frame(p, shape):
            continue
        a, b = (int(x) for x in c["vessels"])
        if a not in S.v or b not in S.v:
            continue
        idx = {v: int(np.clip(round(k * s_dc / S.v[v]["ds"]), 0, S.v[v]["n"] - 1))
               for v, s_dc in ((a, c["s"][0]), (b, c["s"][1]))}
        sin_t = max(math.sin(math.radians(float(c["angle_deg"]))), 0.1)
        cos_t = abs(math.cos(math.radians(float(c["angle_deg"]))))
        cores, lcores = {}, {}
        for v, w in ((a, b), (b, a)):
            d, dw = S.v[v], S.v[w]
            i, iw = idx[v], idx[w]
            ext = res.around(v, w, i, S.margin(v, i))
            lo, hi = ext if ext is not None else (i, i)
            ext = lum.around(v, w, i, BRIDGE)
            llo, lhi = ext if ext is not None else (i, i)
            # at least the straight-line extent (|s - s*| sin t < r_w + g + r_v |cos t|), when w is seen there:
            # an invisible vessel hides nothing
            if dw["seen"][max(iw - 3, 0):iw + 4].any():
                g = float(g_res(d["r"][i], dw["r"][iw], d["st"][i], dw["st"][iw]))
                for gg, key in ((g, "res"), (0.0, "lum")):
                    e = int(math.ceil((dw["r"][iw] + gg + d["r"][i] * cos_t) / sin_t / d["ds"]))
                    if key == "res":
                        lo, hi = min(lo, max(i - e, 0)), max(hi, min(i + e, d["n"] - 1))
                    else:
                        llo, lhi = min(llo, max(i - e, 0)), max(lhi, min(i + e, d["n"] - 1))
            cores[v], lcores[v] = (lo, hi), (llo, lhi)
        out.append(dict(kind="crossing", xy=p, vessels=[a, b], cores=cores, lcores=lcores, ends={}, at=dict(idx),
                        angle=float(c["angle_deg"]), above=int(c["above"]),
                        depth=[float(x) for x in c["depth"]]))
    return out


def _touching_events(S, res, lum, shape, events=()):
    """The unresolved runs no node or crossing used: one event per connected
    group of the runs of a pair (a's runs and b's linked where one's partner
    range overlaps the other's range).  A group whose stretches lie, on both
    vessels, within cores of the node and crossing events is not an event of
    its own (a thin vessel crossing a confluence overlaps the lumens of the
    vessels it does not cross the centreline of: that is one junction); nor
    is one whose centrelines cross (a crossing outside the frame)."""
    import shapely
    taken = defaultdict(list)
    for ev in events:
        for v, ext in ev["cores"].items():
            taken[v].append(ext)

    def inside(v, lo, hi):
        return any(l2 <= hi and h2 >= lo for l2, h2 in taken.get(v, ()))
    by_pair = defaultdict(list)
    for rid in range(len(res.a)):
        if rid not in res.used:
            a, b = int(res.a[rid]), int(res.b[rid])
            by_pair[(min(a, b), max(a, b))].append(rid)
    out = []
    for (a, b), rids in by_pair.items():
        par = {r: r for r in rids}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i, r1 in enumerate(rids):
            for r2 in rids[i + 1:]:
                if res.a[r1] == res.a[r2]:            # both on one vessel: linked if their partner ranges overlap
                    ov = res.j0[r1] <= res.j1[r2] + BRIDGE and res.j0[r2] <= res.j1[r1] + BRIDGE
                else:                                 # r1's partners vs r2's own range, and vice versa
                    ov = (res.j0[r1] <= res.i1[r2] + BRIDGE and res.i0[r2] <= res.j1[r1] + BRIDGE) or \
                         (res.j0[r2] <= res.i1[r1] + BRIDGE and res.i0[r1] <= res.j1[r2] + BRIDGE)
                if ov:
                    par[find(r1)] = find(r2)
        groups = defaultdict(list)
        for r in rids:
            groups[find(r)].append(r)
        for grp in groups.values():
            ext = {a: [], b: []}
            for r in grp:
                v, w = int(res.a[r]), int(res.b[r])
                ext[v].append((int(res.i0[r]), int(res.i1[r])))
                ext[w].append((int(res.j0[r]), int(res.j1[r])))
            cores = {v: (min(x for x, _ in e), max(y for _, y in e)) for v, e in ext.items()}
            if inside(a, *cores[a]) and inside(b, *cores[b]):
                continue
            da, db = S.v[a], S.v[b]
            # their centrelines cross in the stretch: a crossing not among the events (outside the frame, where
            # Scene.crossings / crossings.json leave it out), not a touching
            if cores[a][1] > cores[a][0] and cores[b][1] > cores[b][0] and shapely.intersects(
                    shapely.LineString(da["C"][cores[a][0]:cores[a][1] + 1]),
                    shapely.LineString(db["C"][cores[b][0]:cores[b][1] + 1])):
                continue
            ma, mb = (cores[a][0] + cores[a][1]) // 2, (cores[b][0] + cores[b][1]) // 2
            p = 0.5 * (da["C"][ma] + db["C"][mb])
            if not _in_frame(p, shape):
                continue
            if not (da["inf"][cores[a][0]:cores[a][1] + 1].any() and db["inf"][cores[b][0]:cores[b][1] + 1].any()):
                continue
            ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(da["T"][ma], db["T"][mb]))))))
            length = max((cores[a][1] - cores[a][0]) * da["ds"], (cores[b][1] - cores[b][0]) * db["ds"])
            lov = lum.overlaps(a, b, *cores[a]) or lum.overlaps(b, a, *cores[b])
            base = dict(kind="touching", vessels=[a, b], ends={}, angle=ang, length_px=float(length),
                        lumen_overlap=bool(lov))
            if length <= TOUCH_SPLIT_PX:
                out.append(dict(base, xy=p, cores=cores, lcores=dict(cores), part="whole", at={a: ma, b: mb}))
                continue
            # a long merged stretch: a junction where the two meet and one where they part; the stretch between
            # is one line in the image (the arms into it are flagged merged_with)
            (la, ha), (lb, hb) = cores[a], cores[b]
            same = np.hypot(*(da["C"][la] - db["C"][lb])) <= np.hypot(*(da["C"][la] - db["C"][hb]))
            ends_b = (lb, hb) if same else (hb, lb)
            for part, ia, ib in (("start", la, ends_b[0]), ("end", ha, ends_b[1])):
                ca = (ia, min(ia + int(math.ceil(TOUCH_END_PX / da["ds"])), ha)) if ia == la else                     (max(ia - int(math.ceil(TOUCH_END_PX / da["ds"])), la), ia)
                cb = (ib, min(ib + int(math.ceil(TOUCH_END_PX / db["ds"])), hb)) if ib == lb else                     (max(ib - int(math.ceil(TOUCH_END_PX / db["ds"])), lb), ib)
                q = 0.5 * (da["C"][ia] + db["C"][ib])
                if not _in_frame(q, shape):
                    continue
                cc = {a: ca, b: cb}
                out.append(dict(base, xy=q, cores=cc, lcores=dict(cc), part=part, at={a: ia, b: ib}))
    return out


def _vis_end_events(S, shape):
    """A seen stretch of A ending inside the frame (not at A's own end) with
    another seen vessel B ahead of it within 1 lambda (wall to wall)."""
    from scipy.spatial import cKDTree
    H, W = shape
    out = []
    vis = {v: d["inf"] & d["seen"] for v, d in S.v.items()}
    pts, own, loc, rad = [], [], [], []
    for v, d in S.v.items():
        i = np.flatnonzero(vis[v])
        pts.append(d["C"][i])
        own.append(np.full(len(i), v, np.int64))
        loc.append(i)
        rad.append(d["r"][i])
    if not pts or not sum(len(p) for p in pts):
        return out
    P, own, loc, rP = (np.concatenate(x) for x in (pts, own, loc, rad))
    tree = cKDTree(P)
    rmax = float(rP.max())
    cosmax = math.cos(math.radians(VIS_END_ANGLE))
    for v, d in S.v.items():
        for r0, r1 in _segments(vis[v]):
            for idx, sgn in ((r0, -1), (r1, 1)):
                nxt = idx + sgn
                if nxt < 0 or nxt >= d["n"] or not d["inf"][nxt]:
                    continue                                   # the vessel's end (a node) or the frame's edge
                p = d["C"][idx]
                edge = min(p[0] + 0.5, W - 0.5 - p[0], p[1] + 0.5, H - 0.5 - p[1])
                if edge < d["r"][idx] + 2.0 * d["st"][idx] + 2.0:
                    continue
                t = sgn * d["T"][idx]
                ra = float(d["r"][idx])
                best = None
                for j in tree.query_ball_point(p, ra + rmax + LAMBDA_PX):
                    if own[j] == v:
                        continue
                    wv = P[j] - p
                    dist = float(np.hypot(*wv))
                    gw = dist - ra - rP[j]
                    if gw > LAMBDA_PX:
                        continue
                    ahead = dist < 1e-9 or float(np.dot(wv, t)) / dist >= cosmax
                    if not (ahead or gw <= d["st"][idx]):
                        continue
                    if best is None or gw < best[0]:
                        best = (gw, int(own[j]), int(loc[j]), P[j].copy())
                if best is None:
                    continue
                gw, b, jq, q = best
                # A's core: from its seen end to its closest approach to q, ahead within reach
                n_ahead = int(math.ceil((LAMBDA_PX + ra + float(S.v[b]["r"][jq])) / d["ds"]))
                rng = np.arange(idx, idx + sgn * n_ahead + sgn, sgn)
                rng = rng[(rng >= 0) & (rng < d["n"])]
                j_cl = int(rng[np.argmin(np.hypot(*(d["C"][rng] - q).T))]) if len(rng) else idx
                db = S.v[b]
                nb = int(math.ceil((ra + 2.0 * db["st"][jq]) / db["ds"]))
                nl = int(math.ceil(ra / db["ds"]))
                cores = {v: (min(idx, j_cl), max(idx, j_cl)), b: (max(jq - nb, 0), min(jq + nb, db["n"] - 1))}
                lcores = {v: (min(idx, j_cl), max(idx, j_cl)), b: (max(jq - nl, 0), min(jq + nl, db["n"] - 1))}
                out.append(dict(kind="vis_end", xy=np.asarray(q, float), vessels=[v, b], cores=cores, lcores=lcores,
                                ends={}, at={v: idx, b: jq}, end_xy=p.copy(), gap_px=float(max(gw, 0.0)),
                                seg=(p.copy(), np.asarray(q, float), ra + MERGE_SIG * float(d["st"][idx]))))
    return out


# ======================================================================== clustering
def _footprint(ev, S):
    """The event's region for joining: where its vessels' lumens overlap.
    Each vessel's overlap core is a band of half-width r + MERGE_SIG sigma,
    extended by MERGE_SIG sigma along it (flat caps); the region is the
    union of the pairwise intersections of the bands (a thin vessel crossing
    a wide vein: its path across the vein, not the vein's whole band), plus
    a node's disc (the thinnest incident lumen + sigma) and a vis_end's gap
    from A's end to B; the union of the bands if no two intersect (vessels
    the blur merges without their lumens overlapping)."""
    import shapely
    bands = []
    for v, (lo, hi) in ev[FOOTPRINT_CORES].items():
        d = S.v[v]
        a, b = max(lo - S.sig_samples(v, lo), 0), min(hi + S.sig_samples(v, hi), d["n"] - 1)
        w = float(d["r"][a:b + 1].max() + MERGE_SIG * d["st"][a:b + 1].max())
        pts = d["C"][a:b + 1]
        if b > a and np.hypot(*(pts[-1] - pts[0])) > 1e-6:
            bands.append(shapely.LineString(pts).buffer(w, cap_style="flat"))
        else:
            bands.append(shapely.Point(pts[0]).buffer(w, quad_segs=4))
    geoms = []
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            x = shapely.intersection(bands[i], bands[j])
            if not x.is_empty and x.area > 0:
                geoms.append(x)
    if "disc" in ev:
        c, rr = ev["disc"]
        geoms.append(shapely.Point(c).buffer(rr, quad_segs=8))
    if "seg" in ev:
        p, q, ra = ev["seg"]
        if np.hypot(*(q - p)) > 1e-6:
            geoms.append(shapely.LineString([p, q]).buffer(ra))
    if not geoms:
        geoms = bands
    # joining is local: an elongated overlap (a shallow crossing, a long touching stretch, a thin vessel along a
    # wide vein) counts within CLIP_LAMBDA lambda of the event's centre, or its widest lumen + 2 sigma
    rc = max(CLIP_LAMBDA * LAMBDA_PX, max(float(S.v[v]["r"][lo:hi + 1].max() + 2.0 * S.v[v]["st"][lo:hi + 1].max())
                                           for v, (lo, hi) in ev[FOOTPRINT_CORES].items()))
    return shapely.intersection(shapely.unary_union(geoms), shapely.Point(ev["xy"]).buffer(rc, quad_segs=16))


def _extent(geom, c) -> float:
    """Distance (px) from c to the farthest point of geom."""
    import shapely
    pts = shapely.get_coordinates(shapely.boundary(geom))
    return float(np.hypot(*(pts - c).T).max()) if len(pts) else 0.0


def _clusters(events, S):
    """Join events whose footprints intersect, nearest centres first, when
    every two events of the result stay closer than the sum of their
    extents (complete linkage: a vessel crossed every 30 px is not one giant
    junction), or when one junction's event centres all lie inside the
    other's region (a crossing inside a shallow crossing's long overlap)."""
    import shapely
    if not events:
        return [], []
    geoms = [_footprint(ev, S) for ev in events]
    cen = np.array([ev["xy"] for ev in events], float)
    ext = np.array([_extent(g, c) for g, c in zip(geoms, cen)])
    tree = shapely.STRtree(geoms)
    pi, pj = tree.query(geoms, predicate="intersects")
    m = pi < pj
    pi, pj = pi[m], pj[m]
    order = np.argsort(np.hypot(*(cen[pi] - cen[pj]).T), kind="stable")
    members = {i: [i] for i in range(len(events))}
    region = {i: geoms[i] for i in range(len(events))}
    root = list(range(len(events)))
    for o in order:
        a, b = root[pi[o]], root[pj[o]]
        if a == b:
            continue
        A, B = members[a], members[b]
        D = np.hypot(cen[A][:, None, 0] - cen[B][None, :, 0], cen[A][:, None, 1] - cen[B][None, :, 1])
        ok = bool(np.all(D <= ext[A][:, None] + ext[B][None, :]))
        if not ok:
            ok = bool(shapely.contains_xy(region[a], cen[B][:, 0], cen[B][:, 1]).all() or
                      shapely.contains_xy(region[b], cen[A][:, 0], cen[A][:, 1]).all())
        if ok:
            members[a] = A + B
            region[a] = shapely.union(region[a], region[b])
            for x in B:
                root[x] = a
            del members[b], region[b]
    return [sorted(g) for _, g in sorted(members.items())], geoms


# ======================================================================== assembly
def _ang(t) -> float:
    return math.degrees(math.atan2(float(t[1]), float(t[0])))


def _angle_between(t1, t2) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(t1, t2), -1.0, 1.0))))


def _drop_redundant(evs):
    """vis_ends explained by a crossing / touching of the same pair or by a
    node of both vessels in the same junction; one vis_end per pair."""
    pairs = {frozenset(ev["vessels"]) for ev in evs if ev["kind"] in ("crossing", "touching")}
    nodes_inc = [set(ev["vessels"]) for ev in evs if ev["kind"] == "node"]
    kept, seen = [], set()
    for ev in evs:
        if ev["kind"] == "vis_end":
            key = frozenset(ev["vessels"])
            a, b = ev["vessels"]
            if key in pairs or key in seen or any(a in s and b in s for s in nodes_inc):
                continue
            seen.add(key)
        kept.append(ev)
    return kept


def _intervals(evs, S):
    """Per vessel, the junction's unresolved cores merged where no arm fits
    between them (closer than the two arm margins)."""
    iv = defaultdict(list)
    for ev in evs:
        for v, ext in ev["cores"].items():
            iv[v].append(ext)
    out = {}
    for v, lst in iv.items():
        lst = sorted(lst)
        m = [list(lst[0])]
        for lo, hi in lst[1:]:
            if lo <= m[-1][1] + S.margin(v, m[-1][1]) + S.margin(v, lo):
                m[-1][1] = max(m[-1][1], hi)
            else:
                m.append([lo, hi])
        out[v] = [tuple(x) for x in m]
    return out


CORE_JOIN = 0.5                   # junctions whose claimed cores on a vessel overlap by this share of the shorter one join,
JOIN_DIAM_PX = 2 * LAMBDA_PX      # ... if their events then stay within this (or either's own spread) of each other


def _clip_intervals(iv, S, centre, radius):
    """A junction's claim on each of its vessels: its merged unresolved
    cores (iv: vid -> [(lo, hi)]) clipped to the stretch within `radius`
    of its centre around the core's closest point (a vessel still unresolved
    beyond it, e.g. a shallow crossing's long overlap or a daughter running
    along its sibling, leaves by a 'clipped' arm there)."""
    out = {}
    for v, lst in iv.items():
        C = S.v[v]["C"]
        cl = []
        for lo, hi in lst:
            dist = np.hypot(*(C[lo:hi + 1] - centre).T)
            near = int(np.argmin(dist))
            ok = dist <= radius
            lo2 = lo + (near if not ok[:near + 1].any() else int(np.argmax(ok[:near + 1])))
            okr = ok[near:][::-1]
            hi2 = lo + (near if not okr.any() else len(ok) - 1 - int(np.argmax(okr)))
            cl.append((lo2, hi2))
        out[v] = cl
    return out


def _merge_core_overlaps(groups, S):
    """Join junctions whose claimed cores (_clip_intervals) on a common
    vessel overlap by at least CORE_JOIN of the shorter one (one lies mostly
    inside the other's unresolved stretch: the image cannot separate them).
    A smaller overlap is split between the two at its middle, where both put
    their arm ('linked', _assemble).  (v2 correctness review, defect 1: the
    events are joined by their lumen overlap, the arms were built from the
    wider unresolved cores, and two junctions claimed one stretch of a
    vessel: arms went missing, or both sat at one sample.  Joining on the
    unclipped cores chained junctions along vessels running side by side,
    radius up to 820 px; joining every strong overlap chained the thin
    vessels crossing a wide vein into one junction of 97 events.)  The join
    is complete-linkage: every two events of the result stay within
    JOIN_DIAM_PX of each other (or within the larger spread of the two
    junctions), nearest centres first.  Repeated until stable.  groups:
    [(events, region)]; returns (groups, geometry: [(centre, radius,
    intervals, claimed intervals)])."""
    import shapely
    while True:
        geo = []
        for evs, region in groups:
            centre = _centre(np.array([ev["xy"] for ev in evs], float))
            radius = _extent(region, centre)
            iv = _intervals(evs, S)
            geo.append((centre, radius, iv, _clip_intervals(iv, S, centre, radius)))
        by_v = defaultdict(list)
        for cid, g in enumerate(geo):
            for v, lst in g[3].items():
                by_v[v] += [(lo, hi, cid) for lo, hi in lst]
        par = list(range(len(groups)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        cand = set()
        for v, lst in by_v.items():
            if len(lst) < 2:
                continue
            lst.sort()
            for a in range(len(lst)):
                lo1, hi1, c1 = lst[a]
                for b in range(a + 1, len(lst)):
                    lo2, hi2, c2 = lst[b]
                    if lo2 > hi1:
                        break
                    ov = min(hi1, hi2) - lo2 + 1
                    if c1 != c2 and ov >= CORE_JOIN * min(hi1 - lo1 + 1, hi2 - lo2 + 1):
                        cand.add((min(c1, c2), max(c1, c2)))
        pts = {c: np.array([ev["xy"] for ev in groups[c][0]], float) for c in range(len(groups))}

        def diam(P):
            return float(np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]).max())
        merged = False
        for c1, c2 in sorted(cand, key=lambda p: float(np.hypot(*(geo[p[0]][0] - geo[p[1]][0])))):
            r1, r2 = find(c1), find(c2)
            if r1 == r2:
                continue
            P = np.vstack([pts[r1], pts[r2]])
            if diam(P) <= max(JOIN_DIAM_PX, diam(pts[r1]), diam(pts[r2])):
                par[r2] = r1
                pts[r1] = P
                merged = True
        if not merged:
            return groups, geo
        comp = defaultdict(list)
        for cid in range(len(groups)):
            comp[find(cid)].append(cid)
        new = []
        for root in sorted(comp):
            cids = comp[root]
            if len(cids) == 1:
                new.append(groups[cids[0]])
                continue
            evs = _drop_redundant([ev for c in cids for ev in groups[c][0]])
            new.append((evs, shapely.unary_union([groups[c][1] for c in cids])))
        groups = new


def _arm(S, v, i, flow, centre, crowded, clipped=False):
    d = S.v[v]
    sgn = 1 if flow == "out" else -1
    t = sgn * d["T"][i]
    p = d["C"][i]
    nw = max(1, int(round(LAMBDA_PX / d["ds"])))
    win = np.arange(i, i + sgn * nw + sgn, sgn)
    win = win[(win >= 0) & (win < d["n"])]
    win = win[d["inf"][win]]
    if d["inf"][i] and len(win):
        cnr = float(np.median(d["cnr"][win]))
        vis = "visible" if cnr >= S.cnr_vis else "faint" if cnr >= S.cnr_seen else "invisible"
        cw = d["contrast_own"][win]
        contrast = float(np.median(cw[np.isfinite(cw)])) if np.isfinite(cw).any() else float("nan")
        cw = d["contrast"][win]
        contrast_img = float(np.median(cw[np.isfinite(cw)])) if np.isfinite(cw).any() else float("nan")
    else:
        cnr, vis, contrast, contrast_img = float("nan"), "out_of_frame", float("nan"), float("nan")
    cap = int(math.ceil(RUN_CAP_PX / d["ds"]))
    j = np.arange(i, i + sgn * cap + sgn, sgn)
    j = j[(j >= 0) & (j < d["n"])]
    ok = d["inf"][j] & (d["cnr"][j] >= S.cnr_seen)
    run = 0.0
    if len(ok) and ok[0]:
        okb = _bridge(ok, max(1, int(round(S.bridge_px / d["ds"]))))
        stop = int(np.argmin(okb)) if not okb.all() else len(okb)
        run = float(max(stop - 1, 0) * d["ds"])
    return dict(vid=int(v), cls=d["cls"], flow=flow, stagnant=d["stagnant"], index=int(i),
                s_px=float(d["sig"][i]), xy=[float(p[0]), float(p[1])],
                dist_px=float(np.hypot(*(p - centre))), angle_deg=_ang(t), dir=[float(t[0]), float(t[1])],
                width_px=float(2.0 * d["r"][i]), blur_px=float(d["st"][i]), blur_own_px=float(d["s"][i]),
                depth_dc=float(d["z"][i]), od_peak=float(d["peak"][i]), contrast=contrast,
                contrast_image=contrast_img, cnr=cnr,
                visibility=vis, run_px=run, in_frame=bool(d["inf"][i]), crowded=bool(crowded), clipped=bool(clipped))


def _centre(P: np.ndarray) -> np.ndarray:
    """A junction's centre: of its events' centres and their mean, the point
    with the least summed distance to the events (a medoid: a shallow
    crossing far along a compound does not drag it off its knot; the mean
    on ties, e.g. two events)."""
    cand = np.vstack([P.mean(0, keepdims=True), P])
    cost = np.hypot(cand[:, None, 0] - P[None, :, 0], cand[:, None, 1] - P[None, :, 1]).sum(1)
    return cand[int(np.argmin(np.round(cost, 6)))]


def _ev_type(ev):
    if ev["kind"] == "node":
        return {"fork": "bifurcation", "confluence": "confluence", "anastomosis": "anastomosis"}[ev["ntype"]]
    return {"crossing": "crossing", "touching": "touching", "vis_end": "pseudo-T"}[ev["kind"]]


def _type_visible(evs, arms, merged_pairs=()) -> tuple[str, int]:
    """What a perfect observer of this image calls the junction, from its
    seen arms (visible or faint).  At most 2 seen arms: 'none' (a line, a
    bend or an end).  An event is seen when its own vessels' arms show it: a
    node, >= 3 of its vessels with a seen arm (a vessel with no arm here,
    e.g. a link wholly inside the junction or one never resolved from a
    neighbour, does not count: v2 junction audit); a crossing or touching,
    both vessels and >= 3 arms; a vis_end, both vessels.  One seen event that
    explains every seen arm: its type, except a crossing / touching with 3
    seen arms ('pseudo-T': the image shows a T) and a vis_end with >= 4 (the
    fading vessel shows again beyond the other: 'crossing').  Seen arms of
    other events too, two or more seen events, or none among several events:
    'compound'.  Seen arms of different vessels still unresolved where they
    leave (merged_pairs: index pairs, 'merged_with') are one line in the
    image and count once (for an event, as an arm of each vessel only if
    the merged line is shared by none of the event's other vessels: a
    daughter running unresolved along its sibling is no third arm).
    Returns (type_visible, the number of seen lines)."""
    seen = [i for i, a in enumerate(arms) if a["visibility"] in ("visible", "faint")]
    par = {i: i for i in seen}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j in merged_pairs:
        if i in par and j in par:
            par[find(i)] = find(j)
    lines = defaultdict(set)                      # line (merged arm component) -> its vessels
    for i in seen:
        lines[find(i)].add(arms[i]["vid"])
    n_vis = len(lines)
    return _type_visible_lines(evs, lines, n_vis), n_vis


def _type_visible_lines(evs, lines, n_vis) -> str:
    if n_vis <= 2:
        return "none"
    per = defaultdict(int)                        # per vessel: its seen lines no other vessel shares
    for vs in lines.values():
        if len(vs) == 1:
            per[next(iter(vs))] += 1

    def lines_of(vs):
        return sum(1 for ls in lines.values() if ls & set(vs))

    def visible(ev):
        vs = ev["vessels"]
        if ev["kind"] == "node":
            return sum(1 for v in vs if per[v] > 0) >= 3
        if ev["kind"] in ("crossing", "touching"):
            return all(per[v] > 0 for v in vs) and sum(per[v] for v in vs) >= 3
        return all(per[v] > 0 for v in vs)
    vis_evs = [ev for ev in evs if visible(ev)]
    if not vis_evs:                               # >= 3 lines, but no event shows by its own vessels' arms
        if len(evs) >= 2:
            return "compound"
        if evs[0]["kind"] == "node":
            return _ev_type(evs[0])
        return "pseudo-T" if n_vis == 3 else "crossing"
    if len(vis_evs) >= 2:
        return "compound"
    ev = vis_evs[0]
    own = lines_of(ev["vessels"])
    if own < n_vis:
        return "compound"
    t = _ev_type(ev)
    if ev["kind"] in ("crossing", "touching") and own == 3:
        return "pseudo-T"
    if ev["kind"] == "vis_end" and own >= 4:
        return "crossing"
    return t


def _claims(geo, S, groups=None) -> dict:
    """Each vessel's stretches shared out among the junctions that claim it
    (_clip_intervals), so that no two junctions claim one sample: where
    claims overlap, a sample goes to the claim whose anchor is nearest
    along the vessel (a 1-D Voronoi split, so each claim keeps one
    contiguous run around its anchor; a claim nested in another's long
    unresolved stretch cuts it in two).  The anchor: the median of the
    junction's own events' samples on the vessel in the claim (groups:
    [(events, region)], events' 'at': {vid: sample}), else the claim's
    sample nearest the junction's centre: every event keeps its vessels'
    arms.
    Returns {cid: {vid: [run]}}, run = dict(lo, hi (samples), lo0, hi0 (the
    core it comes from), in_lim, in_link, out_lim, out_link): where the
    arms may go at most, halfway to the neighbouring run of another
    junction along the vessel, or, where the neighbour's run starts at the
    next sample, right there and linked to it (its cid; the vessel is not
    resolved between the two junctions, and both put their arm at the
    meeting point)."""
    anchors = defaultdict(set)                   # (cid, vid) -> samples of its events on the vessel
    for cid, (evs, _) in enumerate(groups or []):
        for ev in evs:
            for v, i in ev.get("at", {}).items():
                anchors[(cid, int(v))].add(int(i))
    by_v = defaultdict(list)
    for cid, (centre, radius, iv, civ) in enumerate(geo):
        for v in civ:
            for (lo0, hi0), (lo, hi) in zip(iv[v], civ[v]):
                by_v[v].append((lo, hi, lo0, hi0, cid))
    out = defaultdict(lambda: defaultdict(list))
    for v, lst in by_v.items():
        C = S.v[v]["C"]
        n = S.v[v]["n"]
        a0, a1 = min(t[0] for t in lst), max(t[1] for t in lst)
        owner = np.full(a1 - a0 + 1, -1, np.int64)
        best = np.full(a1 - a0 + 1, np.inf)
        for kk, (lo, hi, lo0, hi0, cid) in enumerate(lst):
            own = [i_ for i_ in anchors.get((cid, int(v)), ()) if lo <= i_ <= hi]
            anc = int(np.median(own)) if own else lo + int(np.argmin(np.hypot(*(C[lo:hi + 1] - geo[cid][0]).T)))
            dist = np.abs(np.arange(lo, hi + 1) - anc).astype(float)
            seg = slice(lo - a0, hi - a0 + 1)
            better = dist < best[seg]
            best[seg] = np.where(better, dist, best[seg])
            owner[seg] = np.where(better, kk, owner[seg])
        runs = []                                    # (lo, hi, claim index), in order along the vessel
        i = 0
        while i < len(owner):
            if owner[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < len(owner) and owner[j + 1] == owner[i]:
                j += 1
            runs.append((a0 + i, a0 + j, int(owner[i])))
            i = j + 1
        for r, (lo, hi, kk) in enumerate(runs):
            _, _, lo0, hi0, cid = lst[kk]
            in_lim, in_link, out_lim, out_link = 0, None, n - 1, None
            if r > 0:
                plo, phi, pk = runs[r - 1]
                if phi == lo - 1:
                    in_lim, in_link = lo, (lst[pk][4] if lst[pk][4] != cid else None)
                else:
                    in_lim = (phi + lo + 1) // 2
            if r + 1 < len(runs):
                nlo, nhi, nk = runs[r + 1]
                if nlo == hi + 1:
                    out_lim, out_link = hi, (lst[nk][4] if lst[nk][4] != cid else None)
                else:
                    out_lim = (hi + nlo) // 2
            out[cid][v].append(dict(lo=lo, hi=hi, lo0=lo0, hi0=hi0, in_lim=in_lim, in_link=in_link,
                                    out_lim=out_lim, out_link=out_link))
    return out


def _assemble(cid, evs, geo, runs, S, shape, kind):
    """One junction from its events, its geometry geo = (centre, radius,
    its vessels' merged cores, the claimed part of them: _merge_core_overlaps)
    and its share of its vessels (_claims: runs)."""
    centre, radius, ivs, civs = geo
    arms, internal = [], []
    # a vessel's side is closed (no arm) only where it ends at a node event OF THIS junction: a core that runs
    # into the vessel's end at a node of another junction (or outside the frame) still leaves by an arm (v2
    # correctness review, defect 1: such sides were dropped and a crossing next to a fork read as a T)
    node_ends = {(int(v), int(e)) for ev in evs if ev["kind"] == "node" for v, e in ev["ends"].items()}
    for v in sorted(runs):
        d = S.v[v]
        for ru in runs[v]:
            lo, hi = ru["lo"], ru["hi"]
            lo_closed = lo == 0 and (int(v), 0) in node_ends
            hi_closed = hi == d["n"] - 1 and (int(v), 1) in node_ends
            if lo_closed and hi_closed:
                internal.append(int(v))
                continue
            # the arm point: the arm margin beyond the claimed run (the core within the junction's radius of its
            # centre; 'clipped' if the vessel is still unresolved there, e.g. a shallow crossing's long overlap),
            # but at most halfway to the next junction's run on this vessel ('crowded' if that leaves less than
            # half the margin); where the next junction's run starts right there, at the meeting point, where the
            # neighbour puts its arm too ('linked' to it: the vessel is not resolved between them)
            side = []
            if not lo_closed:
                m = S.margin(v, lo)
                link = ru["in_link"]
                i = lo if link is not None else max(lo - m, ru["in_lim"], 0)
                side.append(["in", i, lo - i < 0.5 * m and i > 0, link is None and lo > ru["lo0"], link])
            if not hi_closed:
                m = S.margin(v, hi)
                link = ru["out_link"]
                i = hi if link is not None else min(hi + m, ru["out_lim"], d["n"] - 1)
                side.append(["out", i, i - hi < 0.5 * m and i < d["n"] - 1, link is None and hi < ru["hi0"], link])
            if len(side) == 2 and side[0][1] >= side[1][1]:       # never both arms at one sample
                side[0][1], side[1][1] = max(lo - 1, 0), min(hi + 1, d["n"] - 1)
            for flow, i, crowded, clipped, link in side:
                a = _arm(S, v, i, flow, centre, crowded, clipped)
                a["linked"] = link                                # the neighbouring junction's cid (mapped to its id)
                arms.append(a)
    # partition and lineage
    order = []
    for a in arms:
        if a["vid"] not in order:
            order.append(a["vid"])
    for a in arms:
        a["group"] = order.index(a["vid"])
    partition = [[i for i, a in enumerate(arms) if a["vid"] == v] for v in order]
    lineage = [dict(node=ev["nid"], kind=ev["ntype"], parents=[int(x) for x in ev["parents"]],
                    children=[int(x) for x in ev["children"]]) for ev in evs if ev["kind"] == "node"]
    related = {frozenset((p_, c_)) for ln in lineage for p_ in ln["parents"] for c_ in ln["children"]}
    siblings = {frozenset((x, y)) for ln in lineage for grp in (ln["parents"], ln["children"])
                for x in grp for y in grp if x != y}
    seen_vis = [i for i, a in enumerate(arms) if a["visibility"] in ("visible", "faint")]
    n_vis = len(seen_vis)
    # the pairing a straightness rule makes on the visible arms
    cand = []
    for x in range(len(seen_vis)):
        for y in range(x + 1, len(seen_vis)):
            i, j = seen_vis[x], seen_vis[y]
            turn = 180.0 - _angle_between(arms[i]["dir"], arms[j]["dir"])
            if turn <= VISUAL_TURN_MAX:
                cand.append((turn, i, j))
    used, pairs_visual = set(), []
    for turn, i, j in sorted(cand):
        if i in used or j in used:
            continue
        used |= {i, j}
        vi, vj = arms[i]["vid"], arms[j]["vid"]
        key = frozenset((vi, vj))
        rel = "same" if vi == vj else "lineage" if key in related else "sibling" if key in siblings else "unrelated"
        if rel == "sibling":
            # two branches in line with a trunk thinner than both (or not seen): the image shows a through
            # vessel with a side branch; 'through' (v2 junction audit: a sibling pairing, but not a merge the
            # image could avoid)
            for ln in lineage:
                grp, trunk = ((ln["children"], ln["parents"]) if vi in ln["children"] and vj in ln["children"]
                              else (ln["parents"], ln["children"]) if vi in ln["parents"] and vj in ln["parents"]
                              else (None, None))
                if grp is None:
                    continue
                tw = [a["width_px"] for a in arms if a["vid"] in trunk and a["visibility"] in ("visible", "faint")]
                if not tw or max(tw) < THROUGH_W * min(arms[i]["width_px"], arms[j]["width_px"]):
                    rel = "through"
                break
        pairs_visual.append(dict(arms=[i, j], turn_deg=round(turn, 2), relation=rel))
    same_pairs = [g for g in partition if len(g) == 2 and all(arms[i]["visibility"] in ("visible", "faint")
                                                               for i in g)]
    paired = {frozenset(p["arms"]) for p in pairs_visual}
    missed = sum(frozenset(g) not in paired for g in same_pairs)
    # forks and confluences: continuation and symmetry
    reasons, fork_margin = [], None
    for ev in evs:
        if ev["kind"] != "node" or ev["ntype"] == "anastomosis":
            continue
        trunk = ev["parents"] if ev["ntype"] == "fork" else ev["children"]
        branches = ev["children"] if ev["ntype"] == "fork" else ev["parents"]
        ta = [a for a in arms if a["vid"] in trunk]
        ba = [a for a in arms if a["vid"] in branches]
        if len(ta) != 1 or len(ba) < 2:
            continue
        tdir = -np.asarray(ta[0]["dir"])                         # along the trunk, towards the junction
        turns = sorted((_angle_between(tdir, b["dir"]), b["width_px"]) for b in ba)
        margin = turns[1][0] - turns[0][0]
        fork_margin = margin if fork_margin is None else min(fork_margin, margin)
        ws = sorted(w for _, w in turns[:2])
        if ws[0] / max(ws[1], 1e-9) > SYM_R and margin < SYM_TURN:
            reasons.append("symmetric_fork")
    xing = [ev["angle"] for ev in evs if ev["kind"] in ("crossing", "touching")]
    if any(a < SHALLOW_DEG for a in xing):
        reasons.append("shallow_angle")
    if any(a["visibility"] == "invisible" for a in arms):
        reasons.append("invisible_arm")
    merged_pairs = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            ai, aj = arms[i], arms[j]
            if ai["vid"] == aj["vid"] or ai["visibility"] not in ("visible", "faint") or \
                    aj["visibility"] not in ("visible", "faint"):
                continue
            gap = math.hypot(ai["xy"][0] - aj["xy"][0], ai["xy"][1] - aj["xy"][1]) - \
                0.5 * (ai["width_px"] + aj["width_px"])
            if gap < float(g_res(0.5 * ai["width_px"], 0.5 * aj["width_px"], ai["blur_px"], aj["blur_px"])):
                ai.setdefault("merged_with", []).append(aj["vid"])
                aj.setdefault("merged_with", []).append(ai["vid"])
                merged_pairs.append((i, j))
                if "merged_arms" not in reasons:
                    reasons.append("merged_arms")
    if any(p["relation"] == "unrelated" for p in pairs_visual):
        reasons.append("misleading_geometry")
    if any(a["visibility"] == "out_of_frame" for a in arms):
        reasons.append("at_border")
    if any(a["crowded"] for a in arms):
        reasons.append("crowded")
    reasons = sorted(set(reasons))
    # types
    typ = _ev_type(evs[0]) if len(evs) == 1 else "compound"
    tvis, n_lines = _type_visible(evs, arms, merged_pairs)
    # difficulty
    va = [arms[i] for i in seen_vis] if n_vis >= 2 else arms

    def ratio(key):
        x = [max(a[key], 1e-4) for a in va if a[key] is not None and np.isfinite(a[key])]
        return float(max(x) / min(x)) if len(x) >= 2 else None
    angs = [_angle_between(arms[i]["dir"], arms[j]["dir"]) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    dz = [abs(ev["depth"][0] - ev["depth"][1]) for ev in evs if ev["kind"] == "crossing"]
    members = []
    for ev in evs:
        m = dict(type=_ev_type(ev), vessels=[int(x) for x in ev["vessels"]],
                 xy=[float(ev["xy"][0]), float(ev["xy"][1])],
                 core_px={str(v): [float(S.v[v]["sig"][lo]), float(S.v[v]["sig"][hi])]
                          for v, (lo, hi) in ev["cores"].items()})
        if ev["kind"] == "node":
            m.update(node=ev["nid"], parents=[int(x) for x in ev["parents"]], children=[int(x) for x in ev["children"]])
        elif ev["kind"] == "crossing":
            m.update(angle_deg=ev["angle"], above=ev["above"], depth_dc=ev["depth"])
        elif ev["kind"] == "touching":
            m.update(angle_deg=ev["angle"], length_px=ev["length_px"], lumen_overlap=ev["lumen_overlap"],
                     part=ev["part"])
        else:
            m.update(end_xy=[float(ev["end_xy"][0]), float(ev["end_xy"][1])], gap_px=ev["gap_px"],
                     fading=int(ev["vessels"][0]), at=int(ev["vessels"][1]))
        members.append(m)
    return dict(
        kind=kind, x=float(centre[0]), y=float(centre[1]), radius=radius,
        claim_px={str(v): [[float(S.v[v]["sig"][ru["lo"]]), float(S.v[v]["sig"][ru["hi"]])] for ru in lst]
                  for v, lst in runs.items()},
        arm_radius=max([a["dist_px"] for a in arms], default=0.0),
        in_frame=_in_frame(centre, shape), type=typ, type_visible=tvis, n_events=len(evs), members=members,
        arms=arms, n_arms=len(arms), n_visible_arms=n_vis, n_visible_lines=n_lines, partition=partition,
        lineage=lineage, internal=internal,
        pairs_visual=pairs_visual, visual_missed=int(missed),
        ambiguous=bool(set(reasons) - {"at_border", "crowded"}), ambiguous_reasons=reasons,
        difficulty=dict(
            crossing_angle=float(min(xing)) if xing else None,
            min_arm_angle=float(min(angs)) if angs else None,
            fork_turn_margin=None if fork_margin is None else float(fork_margin),
            depth_gap_dc=float(min(dz)) if dz else None,
            blur_ratio=ratio("blur_px"), width_ratio=ratio("width_px"), contrast_ratio=ratio("contrast"),
            nearest_junction_px=None, nearest_gap_px=None, n_visible_arms=n_vis, n_arms=len(arms)))


# ======================================================================== the API
def _clean(o):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def _optics(optics):
    if optics is None:
        return R.Optics()
    if isinstance(optics, dict):
        return R.Optics(**optics)
    return optics


def _band_rms(noise, kind):
    if noise is None:
        return dict(DEFAULT_BAND_RMS["frame" if kind == "frame" else "average"])
    if isinstance(noise, dict):
        return {float(b): float(v) for b, v in noise.items()}
    return {float(b): float(noise) for b in DEFAULT_BAND_RMS["average"]}


def image_junctions(graph: VesselGraph, k: float, origin=(0.0, 0.0), shape=(1200, 1920), od=None, noise=None,
                    kind: str = "average", *, optics=None, blur_px: float | None = None, fill=None,
                    crossings=None, pad_px: float = PAD_PX, keep_hidden: bool = False, cnr_seen=None,
                    cnr_visible=None, seen_bridge_px=None, seen_min_px=None, in_frame_only: bool = False) -> list[dict]:
    """The image junctions of one image of a scene (module docstring).

    graph: the VesselGraph (d_c).  k: px per d_c.  origin: the d_c position
    of pixel (0, 0).  shape: (H, W) px.
    od: the image's rendered vessel OD (Np, (H, W): labels.npz od_<kind> or
        Scene.od_clean[kind]); the CNR is then scaled by what the rendered
        image really shows (a frame's red-cell gaps, a vessel vanishing under
        a black vein).  None: the closed form only.
    noise: the background's RMS per DoG band ({band px: Np}, as
        scene.background_band_rms), or one number for every band; None:
        DEFAULT_BAND_RMS[kind].
    kind: 'average' or 'frame' (picks the defaults; recorded).
    optics: this image's render.Optics or its dict (the frame has its own
        focal offset: scene.json 'optics_frame'); default render.Optics().
    blur_px: the blur image formation adds (scene.json formed[kind]
        blur_px); default DEFAULT_IMAGE_BLUR[kind].
    fill: {vid: imaging.Filling} (Scene.fills[kind]) for the closed form.
    crossings: precomputed graph.crossings() (d_c fields; Scene.crossings).
    keep_hidden: also return junctions none of whose arms is visible (left
        out by default: they are not in the image; visible_subset(all) gives
        the default list from the keep_hidden one).
    cnr_seen, cnr_visible, seen_bridge_px, seen_min_px, in_frame_only: the
        seen rules (_Samples.set_seen; defaults CNR_VISIBLE, CNR_REQUIRED, 1
        lambda, lambda / 2, False).  truth.py's observable junctions use
        cnr_seen = cnr_visible = its CNR_OBSERVE.
    Returns JSON-serialisable dicts sorted by (y, x), ids 0.. in that order:
      id, kind, x, y (centre px: its events' medoid), radius (px: the
      farthest point of its footprint), arm_radius (the farthest arm point),
      in_frame, type, type_visible, n_events, members (per event: type,
      vessels, xy, core_px per vessel (arclength range, px), and node /
      parents / children, or angle_deg / above / depth_dc, or angle_deg /
      length_px / lumen_overlap, or end_xy / gap_px / fading / at), arms
      (vid, cls, flow, stagnant, index / s_px (along the vessel), xy,
      dist_px, angle_deg, dir, width_px, blur_px, blur_own_px, depth_dc,
      od_peak, contrast (the vessel's own), contrast_image (raw, may hold a
      neighbour's OD), cnr, visibility, run_px, in_frame, crowded, clipped,
      linked (the id of the junction the vessel runs on into without a
      resolved gap, else None), group, merged_with), claim_px (per vessel,
      the arclength ranges (px) this junction holds), n_arms,
      n_visible_arms, n_visible_lines (seen arms, merged ones counted
      once), partition (lists of arm
      indices, one per vessel), lineage, internal (vessels wholly inside),
      pairs_visual, visual_missed, ambiguous, ambiguous_reasons, difficulty
      (crossing_angle, min_arm_angle, fork_turn_margin, depth_gap_dc,
      blur_ratio, width_ratio, contrast_ratio, nearest_junction_px,
      nearest_gap_px (centre distance minus both radii), n_visible_arms,
      n_arms)."""
    S = junction_samples(graph, k, origin, shape, od, noise, kind, optics=optics, blur_px=blur_px, fill=fill,
                         pad_px=pad_px)
    return junctions_from_samples(S, graph, crossings=crossings, keep_hidden=keep_hidden, cnr_seen=cnr_seen,
                                  cnr_visible=cnr_visible, seen_bridge_px=seen_bridge_px, seen_min_px=seen_min_px,
                                  in_frame_only=in_frame_only)


def junction_samples(graph: VesselGraph, k: float, origin=(0.0, 0.0), shape=(1200, 1920), od=None, noise=None,
                     kind: str = "average", *, optics=None, blur_px: float | None = None, fill=None,
                     pad_px: float = PAD_PX) -> _Samples:
    """Every vessel near the frame sampled every SAMPLE_PX px with its CNR
    (image_junctions' arguments; the costly half of it, 4-5 s per full
    image): junctions_from_samples applies the rules, and truth.py reads the
    per-sample visibility from the same samples.  The samples keep k,
    origin, kind, optics, blur_px and band_rms (the noise used)."""
    k = float(k)
    origin = np.asarray(origin, float)
    shape = (int(shape[0]), int(shape[1]))
    optics = _optics(optics)
    blur_px = float(DEFAULT_IMAGE_BLUR.get(kind, 0.85) if blur_px is None else blur_px)
    S = _Samples(graph, k, origin, shape, optics, blur_px, fill, pad_px)
    rms = _band_rms(noise, kind)
    S.visibility(None if od is None else np.asarray(od, np.float32), rms, optics, blur_px)
    S.k, S.origin, S.kind, S.optics, S.blur_px, S.band_rms = k, origin, kind, optics, blur_px, dict(rms)
    S.has_od = od is not None
    return S


def junctions_from_samples(S: _Samples, graph: VesselGraph, *, crossings=None, keep_hidden: bool = False,
                           cnr_seen=None, cnr_visible=None, seen_bridge_px=None, seen_min_px=None,
                           in_frame_only: bool = False, internals: dict | None = None) -> list[dict]:
    """image_junctions on samples from junction_samples, under the given seen
    rules (set on S first; S keeps them afterwards).  internals: a dict to
    receive the intermediate results truth.py builds on (rec_res: the
    unresolved records (a, b, i, j) of the seen samples; events; groups;
    geo; claims {cid: {vid: [run]}}; id_of_cid)."""
    k, origin, shape, kind = S.k, S.origin, S.shape, S.kind
    S.set_seen(cnr_seen, cnr_visible, seen_bridge_px, seen_min_px, in_frame_only)
    rec_res, rec_lum = _pair_records(S)
    res, lum = _Runs(rec_res), _Runs(rec_lum)
    events = _node_events(graph, S, res, lum, k, origin, shape)
    events += _crossing_events(graph, S, res, lum, k, origin, shape, crossings)
    events += _touching_events(S, res, lum, shape, events)
    events += _vis_end_events(S, shape)
    clusters, geoms = _clusters(events, S)
    import shapely
    groups = []
    for cl in clusters:
        evs = _drop_redundant([events[i] for i in cl])
        kept = [i for i in cl if any(events[i] is e for e in evs)]
        groups.append((evs, shapely.unary_union([geoms[i] for i in kept])))
    groups, geo = _merge_core_overlaps(groups, S)
    runs = _claims(geo, S, groups)
    out = []
    for cid, (evs, region) in enumerate(groups):
        j = _clean(_assemble(cid, evs, geo[cid], runs.get(cid, {}), S, shape, kind))
        j["_cid"] = cid
        out.append(j)
    out = [j for j in out if j["n_events"] and (keep_hidden or j["n_visible_arms"] > 0)]
    out.sort(key=lambda j: (round(j["y"], 3), round(j["x"], 3)))
    _nearest(out)
    new_id = {j["_cid"]: n for n, j in enumerate(out)}
    for n, j in enumerate(out):
        j["id"] = n
        del j["_cid"]
        for a in j["arms"]:                     # linked: the id of the junction the vessel runs on into (None: hidden)
            a["linked"] = None if a.get("linked") is None else new_id.get(a["linked"])
    if internals is not None:
        internals.update(rec_res=rec_res, rec_lum=rec_lum, events=events, groups=groups, geo=geo, claims=runs,
                         id_of_cid=new_id)
    return out


def _nearest(out: list):
    """difficulty nearest_junction_px / nearest_gap_px within the list."""
    if len(out) >= 2:
        from scipy.spatial import cKDTree
        P = np.array([[j["x"], j["y"]] for j in out])
        dd, ii = cKDTree(P).query(P, 2)
        for n, j in enumerate(out):
            m = int(ii[n, 1])
            j["difficulty"]["nearest_junction_px"] = float(dd[n, 1])
            j["difficulty"]["nearest_gap_px"] = float(dd[n, 1] - j["radius"] - out[m]["radius"])


def visible_subset(junctions: list) -> list[dict]:
    """The default list (keep_hidden=False) from a keep_hidden=True one of
    the same image: the junctions with a visible arm, re-numbered in the
    same order, 'linked' re-mapped (None where it pointed to a dropped
    junction) and the nearest-junction distances taken within the subset;
    equal to image_junctions(..., keep_hidden=False).  The input is not
    changed."""
    import copy
    keep = [j for j in junctions if j["n_events"] and j["n_visible_arms"] > 0]
    new_id = {j["id"]: n for n, j in enumerate(keep)}
    out = copy.deepcopy(keep)
    _nearest(out)
    for n, j in enumerate(out):
        j["id"] = n
        for a in j["arms"]:
            a["linked"] = None if a.get("linked") is None else new_id.get(a["linked"])
    return out


def scene_samples(scene, kind: str = "average") -> _Samples:
    """junction_samples for a make_scene() Scene (as scene_junctions)."""
    SC = _scene()
    p = scene.params
    opt = R.Optics(**p["optics_frame" if kind == "frame" and "optics_frame" in p else "optics"])
    rms = SC.background_band_rms(scene.formed[kind], scene.images[kind], kind)
    return junction_samples(scene.graph, p["k"], p.get("origin", (0.0, 0.0)), p["shape"], od=scene.od_clean[kind],
                            noise=rms, kind=kind, optics=opt, blur_px=p["formed"][kind]["blur_px"],
                            fill=scene.fills.get(kind))


def scene_junctions(scene, kind: str = "average", samples: _Samples | None = None, **kw) -> list[dict]:
    """image_junctions for a make_scene() Scene: its graph, the kind's OD,
    optics (the frame's own focal offset), image blur, red-cell filling,
    in-frame crossings and the exact background band RMS
    (scene.background_band_rms).  samples: scene_samples(scene, kind), to
    reuse them; keyword rules as junctions_from_samples."""
    S = samples if samples is not None else scene_samples(scene, kind)
    return junctions_from_samples(S, scene.graph, crossings=scene.crossings or None, **kw)


def check_junctions(junctions: list, graph: VesselGraph | None = None) -> list[str]:
    """Consistency problems of one image's junctions (empty when sound):
    two arms of one vessel at the same point (a zero-length pass); an arm
    inside another junction's unresolved core on its vessel (two junctions
    claiming one stretch), unless it is 'linked' to that junction (their
    cores overlap a little and both put their arm at the overlap's middle),
    and then that junction must have the vessel's opposite arm within 2 px;
    with the graph, a vessel of a crossing or touching member without its
    'in' (or 'out') arm although its upstream (downstream) end is not a node
    of this junction (a vessel's continuity broken: the arms cannot be
    followed between junctions)."""
    out = []
    cores = defaultdict(list)                       # vid -> [(lo, hi, junction id)]
    by_id = {}
    for n, j in enumerate(junctions):
        jid = j.get("id", n)
        by_id[jid] = j
        if "claim_px" in j:                         # the claimed cores (v2 fix); else the events' cores
            for v, lst in j["claim_px"].items():
                cores[int(v)] += [(float(lo), float(hi), jid) for lo, hi in lst]
            continue
        for m in j["members"]:
            for v, (lo, hi) in m["core_px"].items():
                cores[int(v)].append((float(lo), float(hi), jid))
    for n, j in enumerate(junctions):
        jid = j.get("id", n)
        by_v = defaultdict(list)
        for a in j["arms"]:
            by_v[a["vid"]].append(a)
            link = a.get("linked")
            for lo, hi, j2 in cores[a["vid"]]:
                if j2 != jid and j2 != link and lo < a["s_px"] < hi:
                    out.append(f"junction {jid}: arm of vessel {a['vid']} at s {a['s_px']:.1f} px inside junction "
                               f"{j2}'s core [{lo:.1f}, {hi:.1f}]")
            if link is not None and link in by_id:
                want = "out" if a["flow"] == "in" else "in"
                if not any(b["vid"] == a["vid"] and b["flow"] == want and abs(b["s_px"] - a["s_px"]) <= 2.0
                           for b in by_id[link]["arms"]):
                    out.append(f"junction {jid}: arm of vessel {a['vid']} linked to junction {link}, which has no "
                               f"'{want}' arm of it there")
        for v, lst in by_v.items():
            s_in = [a["s_px"] for a in lst if a["flow"] == "in"]
            s_out = [a["s_px"] for a in lst if a["flow"] == "out"]
            if any(abs(x - y) < 1e-6 for x in s_in for y in s_out):
                out.append(f"junction {jid}: vessel {v} has its in and out arms at one point")
        if graph is None:
            continue
        nodes = {m["node"] for m in j["members"] if "node" in m}
        for m in j["members"]:
            if m["type"] not in ("crossing", "touching"):
                continue
            for v in m["vessels"]:
                if v in j.get("internal", ()) or v not in graph.vessels:
                    continue
                e = graph.vessels[v]
                flows = {a["flow"] for a in by_v.get(v, [])}
                for flow, node in (("in", e.u), ("out", e.v)):
                    if flow not in flows and node not in nodes:
                        out.append(f"junction {jid}: vessel {v} of a {m['type']} has no '{flow}' arm and its "
                                   f"node {node} is not in this junction")
    return out


def band_rms_from_image(image, od, valid=None, kind: str = "average") -> dict:
    """The background's RMS per DoG band (Np) estimated from a saved image
    and its OD raster (labels.npz od_<kind>), when image formation's own
    background and noise are not at hand: ln(image) + od is the background
    plus noise up to the image-formation blur's residue at vessel edges, so
    a robust SD (1.4826 MAD) per band over the eroded valid region (0.84-1.09x
    the exact value on two test scenes)."""
    from . import imaging as IM
    I = np.asarray(image, np.float64)
    v = np.isfinite(I) & (I > 0)
    if valid is not None:
        v &= np.asarray(valid, bool)
    if kind == "frame":
        v &= I < IM.FULL_SCALE
    L = np.where(v, np.log(np.where(v, I, 1.0)) + np.asarray(od, np.float64), 0.0)
    Lf = IM.fill_invalid(L, v)
    out = {}
    for s in IM.BANDS[:-1]:
        B = IM._dog(Lf, s)
        e = ndi.binary_erosion(v, iterations=int(min(3 * s + 2, 40)))
        m = e if e.sum() > 1000 else v
        x = B[m]
        out[s] = float(1.4826 * np.median(np.abs(x - np.median(x))))
    return out


def inputs_from_saved(folder: str, kind: str = "average", noise=None) -> tuple[dict, np.ndarray]:
    """(kwargs for image_junctions, the image) for a save_scene() folder:
    truth.json, scene.json (k, shape, the kind's optics and image blur),
    labels.npz (od_<kind>, valid_<kind>), crossings.json, still_<kind>.tif;
    noise defaults to band_rms_from_image on the still."""
    import tifffile
    g = VesselGraph.load(os.path.join(folder, "truth.json"))
    with open(os.path.join(folder, "scene.json"), encoding="utf-8") as fh:
        sj = json.load(fh)
    lab = np.load(os.path.join(folder, "labels.npz"))
    od = lab[f"od_{kind}"].astype(np.float32)
    valid = lab[f"valid_{kind}"] if f"valid_{kind}" in lab.files else None
    image = tifffile.imread(os.path.join(folder, f"still_{kind}.tif")).astype(np.float32)
    key = "optics_frame" if kind == "frame" and "optics_frame" in sj else "optics"
    crs = None
    pc = os.path.join(folder, "crossings.json")
    if os.path.exists(pc):
        with open(pc, encoding="utf-8") as fh:
            crs = json.load(fh)
    kw = dict(graph=g, k=float(sj["k"]), origin=sj.get("origin", [0.0, 0.0]), shape=tuple(sj["shape"]), od=od,
              noise=noise if noise is not None else band_rms_from_image(image, od, valid, kind), kind=kind,
              optics=R.Optics(**sj[key]), blur_px=float(sj["formed"][kind]["blur_px"]), crossings=crs)
    return kw, image


def save_junctions(junctions: list, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(format="vesselscene-junctions", version=1, junctions=junctions), fh)


def load_junctions(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["junctions"]


def rules() -> dict:
    """The definitions' constants (recorded in scene.json, junction_rules)."""
    return dict(lambda_px=LAMBDA_PX, cnr_required=CNR_REQUIRED, cnr_visible=CNR_VISIBLE, sample_px=SAMPLE_PX,
                pad_px=PAD_PX, merge_sig=MERGE_SIG, arm_margin_sig=ARM_MARGIN_SIG, shallow_deg=SHALLOW_DEG,
                sym_r=SYM_R, sym_turn_deg=SYM_TURN, visual_turn_max_deg=VISUAL_TURN_MAX,
                vis_end_angle_deg=VIS_END_ANGLE, touch_split_px=TOUCH_SPLIT_PX, touch_end_px=TOUCH_END_PX,
                clip_lambda=CLIP_LAMBDA, footprint_cores=FOOTPRINT_CORES, dip_min=DIP_MIN, types=list(TYPES),
                visibility_names=list(VIS_NAMES), core_join=CORE_JOIN, join_diam_px=JOIN_DIAM_PX,
                through_w=THROUGH_W, version=2)


def summary(junctions: list) -> dict:
    """Counts of one image's junctions (in frame): n, per truth type, per
    type_visible, per number of visible arms (5 = 5 or more), the ambiguous
    share (all / shown: type_visible not 'none') and its reasons among the
    shown ones, 'clean' (one event, every arm visible, not ambiguous, no arm
    crowded), and how often a straightness rule would pair arms of unrelated
    vessels among the shown ones (merge risk)."""
    inf = [j for j in junctions if j.get("in_frame", True)]
    seen = [j for j in inf if j["type_visible"] != "none"]

    def cnt(it):
        d = {}
        for x in it:
            d[x] = d.get(x, 0) + 1
        return dict(sorted(d.items(), key=lambda kv: str(kv[0])))
    clean = sum(1 for j in inf if j["n_events"] == 1 and not j["ambiguous"]
                and j["n_visible_arms"] == j["n_arms"] and not any(a.get("crowded") for a in j["arms"]))
    return dict(n=len(inf), types=cnt(j["type"] for j in inf), types_visible=cnt(j["type_visible"] for j in inf),
                n_visible_arms=cnt(min(int(j["n_visible_arms"]), 5) for j in inf),
                ambiguous=round(float(np.mean([j["ambiguous"] for j in inf])), 4) if inf else None,
                n_shown=len(seen),
                ambiguous_shown=round(float(np.mean([j["ambiguous"] for j in seen])), 4) if seen else None,
                reasons_shown=cnt(r for j in seen for r in j["ambiguous_reasons"]), clean=int(clean),
                relations_shown=cnt(p.get("relation") for j in seen for p in j["pairs_visual"]),
                linked_arms=int(sum(1 for j in inf for a in j["arms"] if a.get("linked") is not None)),
                misleading_shown=int(sum(any(p.get("relation") == "unrelated" for p in j["pairs_visual"])
                                         for j in seen)))


# ======================================================================== rasters
def _heat_key(t: str) -> str:
    return "heat_" + t.replace("-", "_")


def junction_rasters(junctions: list, shape, sigma_min: float = 2.0, visible_only: bool = False) -> dict:
    """Truth rasters of the junctions, (H, W):
      heat_<type> (float32, peak 1; heat_pseudo_T) for each of TYPES: a Gaussian at each
        junction centre of that type, sigma = max(sigma_min, radius / 2),
        max-combined; heat_any: every junction; heat_visible: those whose
        type_visible is not 'none' (what a detector of this image should find).
      arm_dir (H, W, 2) float32: each arm's outward unit direction on the
        segment from its junction's centre to its point (1 px wide; 0
        elsewhere); arm_group (int16): its partition group + 1; arm_vis
        (uint8): 1 visible, 2 faint, 3 invisible, 4 out of frame.
    visible_only: only the junctions whose type_visible is not 'none'."""
    import cv2
    H, W = int(shape[0]), int(shape[1])
    out = {_heat_key(t): np.zeros((H, W), np.float32) for t in TYPES}
    out["heat_any"] = np.zeros((H, W), np.float32)
    out["heat_visible"] = np.zeros((H, W), np.float32)
    arm_dir = np.zeros((H, W, 2), np.float32)
    arm_group = np.zeros((H, W), np.int16)
    arm_vis = np.zeros((H, W), np.uint8)
    for j in junctions:
        if visible_only and j["type_visible"] == "none":
            continue
        cx, cy = j["x"], j["y"]
        sg = max(sigma_min, 0.5 * j["radius"])
        R_ = int(math.ceil(3 * sg))
        x0, x1 = max(int(math.floor(cx)) - R_, 0), min(int(math.ceil(cx)) + R_ + 1, W)
        y0, y1 = max(int(math.floor(cy)) - R_, 0), min(int(math.ceil(cy)) + R_ + 1, H)
        if x1 > x0 and y1 > y0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gk = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sg ** 2).astype(np.float32)
            for key in (_heat_key(j["type"]), "heat_any") + (("heat_visible",) if j["type_visible"] != "none" else ()):
                np.maximum(out[key][y0:y1, x0:x1], gk, out=out[key][y0:y1, x0:x1])
        for a in j["arms"]:
            x_a, y_a = a["xy"]
            ax0, ax1 = int(max(min(cx, x_a) - 2, 0)), int(min(max(cx, x_a) + 3, W))
            ay0, ay1 = int(max(min(cy, y_a) - 2, 0)), int(min(max(cy, y_a) + 3, H))
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            m = np.zeros((ay1 - ay0, ax1 - ax0), np.uint8)
            cv2.line(m, (int(round(cx)) - ax0, int(round(cy)) - ay0), (int(round(x_a)) - ax0, int(round(y_a)) - ay0),
                     1, 1)
            sel = m.astype(bool)
            arm_dir[ay0:ay1, ax0:ax1][sel] = a["dir"]
            arm_group[ay0:ay1, ax0:ax1][sel] = a["group"] + 1
            arm_vis[ay0:ay1, ax0:ax1][sel] = VIS_NAMES.index(a["visibility"]) + 1
    out.update(arm_dir=arm_dir, arm_group=arm_group, arm_vis=arm_vis)
    return out


# ======================================================================== overlay
TYPE_COLORS = {                                   # BGR
    "bifurcation": (40, 200, 40), "confluence": (230, 140, 30), "anastomosis": (200, 40, 200),
    "crossing": (0, 215, 255), "touching": (230, 230, 0), "pseudo-T": (0, 130, 255), "compound": (40, 40, 240)}
GROUP_COLORS = [(60, 60, 255), (255, 200, 40), (60, 230, 60), (240, 80, 240), (0, 200, 255), (255, 120, 120),
                (160, 255, 160), (200, 160, 255), (120, 200, 200), (255, 255, 255)]


def _base_view(image):
    import cv2
    img = np.asarray(image)
    if img.ndim == 3:
        return img.astype(np.uint8).copy()
    if img.dtype == np.uint8:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        g8 = _scene().flat_view(img.astype(np.float32))
    except Exception:                                                   # noqa: BLE001
        x = np.asarray(img, np.float64)
        ok = np.isfinite(x)
        lo, hi = (np.percentile(x[ok], [1, 99]) if ok.any() else (0.0, 1.0))
        g8 = (np.clip((np.where(ok, x, hi) - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)


def draw_junctions(image, junctions: list, path: str | None = None, crop=None, scale: float = 1.0,
                   min_visible_arms: int = 0, label: bool | None = None, title: str | None = None,
                   legend: bool = True) -> np.ndarray:
    """The junctions over an image (a float still in DN with NaN, shown as
    scene.flat_view shows it; or a uint8 grey / BGR image): the region
    (circle of the junction's radius, at most 3 lambda; dashed if
    ambiguous) and a centre dot in the type's colour (TYPE_COLORS), the arms
    as a thin line from the centre to their point and an arrow 0.6 lambda
    along their direction, coloured by partition group (same colour = same
    vessel; thick arrow: visible, thin: faint, dotted: invisible; out of
    frame not drawn).  crop = (y0, x0, h, w) px,
    scale enlarges; min_visible_arms skips junctions with fewer visible arms;
    label writes the junction ids (default: when scale >= 2).  Returns the
    BGR image (and writes it to path)."""
    import cv2
    base = _base_view(image)
    H, W = base.shape[:2]
    y0, x0, h, w = crop if crop is not None else (0, 0, H, W)
    base = base[y0:y0 + h, x0:x0 + w]
    f = float(scale)
    if f != 1.0:
        base = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    out = base.copy()
    sh = 4
    off = np.array([x0, y0], float)
    label = (f >= 2) if label is None else label

    def P(xy):
        return tuple(np.rint(((np.asarray(xy, float) - off) * f + (f - 1) / 2) * (1 << sh)).astype(np.int64).tolist())

    def dotted(p, q, col, th):
        p, q = np.asarray(p, float), np.asarray(q, float)
        n = max(1, int(float(np.hypot(*(q - p))) / 3.0))
        for i in range(0, n, 2):
            a, b = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
            cv2.line(out, P(a), P(b), col, th, cv2.LINE_AA, shift=sh)
    for j in junctions:
        if j["n_visible_arms"] < min_visible_arms:
            continue
        c = np.array([j["x"], j["y"]])
        rad = max(j["radius"], 3.0)
        if not (x0 - 4 <= c[0] < x0 + w + 4 and y0 - 4 <= c[1] < y0 + h + 4):
            continue
        col = TYPE_COLORS.get(j["type"], (200, 200, 200))
        for a in j["arms"]:
            if a["visibility"] == "out_of_frame":
                continue
            gc = GROUP_COLORS[a["group"] % len(GROUP_COLORS)]
            tip = np.asarray(a["xy"]) + 0.6 * LAMBDA_PX * np.asarray(a["dir"])
            if a["visibility"] == "invisible":
                dotted(c, tip, gc, 1)
            else:
                th = 2 if a["visibility"] == "visible" else 1
                cv2.line(out, P(c), P(a["xy"]), gc, 1, cv2.LINE_AA, shift=sh)
                cv2.arrowedLine(out, P(a["xy"]), P(tip), gc, th, cv2.LINE_AA, shift=sh, tipLength=0.3)
        rr = int(min(rad, 3 * LAMBDA_PX) * f * (1 << sh))
        cc = P(c)
        if j["ambiguous"]:
            for a0 in range(0, 360, 30):
                cv2.ellipse(out, cc, (rr, rr), 0, a0, a0 + 15, col, 1, cv2.LINE_AA, shift=sh)
        else:
            cv2.circle(out, cc, rr, col, 1, cv2.LINE_AA, shift=sh)
        cv2.circle(out, cc, int(2.5 * max(f, 1.0) * (1 << sh)), (0, 0, 0), -1, cv2.LINE_AA, shift=sh)
        cv2.circle(out, cc, int(1.8 * max(f, 1.0) * (1 << sh)), col, -1, cv2.LINE_AA, shift=sh)
        if label and "id" in j:
            q = ((c - off) * f).astype(int)
            for colr, th in (((0, 0, 0), 2), ((255, 255, 255), 1)):
                cv2.putText(out, str(j["id"]), (int(q[0] + 4), int(q[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colr, th,
                            cv2.LINE_AA)
    if legend:
        yy = 16
        for name in ([title] if title else []) + list(TYPE_COLORS):
            colr = TYPE_COLORS.get(name)
            if colr is not None:
                cv2.circle(out, (14, yy - 4), 5, colr, -1, cv2.LINE_AA)
            for cc_, th in (((0, 0, 0), 2), ((255, 255, 255), 1)):
                cv2.putText(out, name, (26, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc_, th, cv2.LINE_AA)
            yy += 16
        for txt in ("rays: arms, same colour = same vessel", "thick visible, thin faint, dotted invisible",
                    "dashed circle: ambiguous"):
            for cc_, th in (((0, 0, 0), 2), ((255, 255, 255), 1)):
                cv2.putText(out, txt, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cc_, th, cv2.LINE_AA)
            yy += 15
    if path:
        cv2.imwrite(path, out)
    return out
