"""Statistics for the visual tells the scorecard missed (realism reviews, 2026-09-30).

The frozen suite (stats.py) scores a synthetic still against the pooled 9
reference stills.  The realism review found differences plain to the eye that
it does not measure, or that the pooled reference hides (the 500-row strips
from another site widen the real range).  These statistics measure them, on
the same flattened log image F (stats.flatten) and the same multi-scale
detection (vessel mask M, pruned skeleton S, profiles) as the suite.

v0 tells (still_tells, first review):

  F_sd            spread (SD) of F over the valid pixels: overall contrast (T1)
  dark128_0.3/.45 area share darker than -0.3 / -0.45 Np after a 128 px
                  background (the 32 px flatten removes part of a wide
                  vessel's darkness): big black vessels (T2)
  darkw_p99       width (px) of the widest dark (< -0.3 Np) vessels: 2 x the
                  99th percentile of the distance transform on the dark mask's
                  skeleton (T2)
  focus_var       SD over 4 x 6 tiles of the median log edge width of the
                  profiles (vessels up to 1.5 lambda): focus varying over the
                  field (T3)
  bg_patch_cv     CV over 64 px tiles of the RMS of the 1-2 px DoG band away
                  from vessels: patchiness of the fine texture (T9)
  c_thin, n_thin  median contrast and share of the profiles 0.5-1 lambda
                  wide; c_wide: median contrast of those 2-4 lambda wide (T6)
  par_excess      the suite's parallel-neighbour excess at 3-30 px (T5)
  cprof_p90       90th percentile of the profile contrast

v1 tells (the five gaps the user confirmed, 2026-09-30; lambda = the image's
median profile FWHM; "paths" are continuation paths, below):

1. Layout: a web of long vessels crossing each other, or trees thinning into
   a mesh (layout_tells).
  web_lcc         share of the darker half of the skeleton (skeleton pixels
                  darker in F than the skeleton's median) in its largest
                  8-connected component (gaps <= 2 px closed): a web
                  percolates, separate trees do not
  web_lcc25       the same for the darkest quarter
  web_span        share of the darker half in components whose bounding box
                  spans at least half the frame's width or height: vessels
                  that cross the whole field
  node_rate       junction nodes (branch points + crossings, after merging
                  junction clusters joined by <= 1 lambda) per 100 lambda of
                  skeleton; branch_rate, xing_rate: the two parts
  xing_frac       crossings / (crossings + branch points)
  xing_angle_med, xing_shallow   median crossing angle (deg) and share of
                  crossings below 30 deg
  cont_share_5/10/20   share of skeleton length on continuation paths at
                  least 5 / 10 / 20 lambda long; cont_thin_10/20: the same for
                  the thin skeleton (mask width <= lambda)
  cont_straight   median chord / arc of the paths >= 10 lambda
  calibre_change  median |ln(w(s + 5 lambda) / w(s))| of the mask width w
                  along paths >= 6 lambda (smoothed over 1 lambda): do vessels
                  keep their calibre (web) or taper (trees)?
  thin_coh        length-weighted mean orientation coherence |<exp 2i theta>|
                  of the thin path pixels in 10 lambda tiles
  Continuation paths: at every (merged) junction node the arms are paired
  greedily, straightest first, when the turn is below 30 deg and the arms are
  collinear within 0.5 lambda (+ 0.25 x their distance); two junction nodes
  joined by a branch of 1-3 lambda are merged when their four outer arms form
  two straight lines (a shallow crossing, where the skeleton runs along both
  vessels for a while); a node with two or more straight pairs is a crossing,
  one with at most one pair a branch point.  Path ends are then joined across
  gaps by stats.bridge_paths (good continuation, <= 3 lambda; a bridge over
  another path counts as a crossing).

2. Pairs and bundles (pair_tells).  For points every 2 px along paths >= 3
   lambda, the across-profile of F (0.7 px smoothed) is averaged over +-1
   lambda along the vessel (only lines parallel within ~15-20 deg survive),
   and its local minima are found (prominence >= 0.008 Np and >= 0.15 x that
   of the line at the point, >= 3 px apart) within 2 lambda beyond the mask
   edges; the line at the point is the minimum nearest the skeleton, its width
   the minimum's width at half prominence; its group is the chain of minima
   with consecutive spacing <= 2 lambda.
  pair_frac       share of thin line points (line width <= lambda) whose group
                  has >= 2 lines (a parallel neighbour within 2 lambda)
  pair_run5       share of thin line points with >= 2 lines where at least half
                  of the centred 5 lambda window along the path also has them
  bundle_frac, bundle_run5   the same with >= 3 lines (bundles)
  pair_wide_frac  share of wide line points (width >= 2 lambda) with a
                  parallel neighbour: companions along big veins

3. Wide vessels (wide_focus_tells), from the suite's profiles:
  wide_fwhm, wide_edge, wide_edge_rel   the widest decile of the profiles
                  (FWHM >= its 90th percentile): that FWHM (px), their median
                  edge width (px; sigma of a blurred step) and median edge / FWHM
  wide98_fwhm, wide98_edge, wide98_edge_rel   the same for the widest 2 %
  wide_thin_edge  median edge width of the widest decile / that of the thin
                  profiles (0.5-1 lambda): depth-of-focus proxy (are the big
                  vessels blurred relative to the thin ones?)
4. Focus:
  soft_tiles      share of the 4 x 6 tiles (>= 25 profiles < 1.5 lambda) whose
                  median edge width exceeds 1.5 x the sharpest tile's median
  tile_edge_ratio p90 / p10 over the tiles of the median edge width

5. A single frame against its own average.  frame_tells (v0; T4): the frame
   is registered to the average tile by tile, and along the average's medium
   vessels (detection scales 2-4 px, contrast > 0.06 Np) the frame's darkness
   is compared with the average's (r = frame / average, smoothed 1.5 px, the
   frame's value the darkest in 3 x 3):
  drop / weak     share of that skeleton with r < 0.3 (as good as absent in
                  the frame) / r < 0.6
  r_cv            robust CV of r along it: graininess (noise included)
   frame_average_tells (v1): the average's geometry is carried into the frame
   by `mapping` (the stabilization's own displacement fields for a real burst:
   fields_mapping; identity for a synthetic frame, which is rendered at the
   registration reference):
  sigma_extra     extra Gaussian blur of the average over the frame (px):
                  sqrt(median(s_avg^2 - s_frame^2)), signed, where s is the edge
                  sigma of a box-with-Gaussian-edges fit to the same vessel's
                  across-profile (averaged over 7 px along it, cubic sampling)
                  in each image; at profile points of the average with FWHM
                  >= 0.8 lambda, contrast >= 0.05 Np, > 1 lambda from junctions;
                  _med / _wide: FWHM below / above 2 lambda
  edge_ratio      median s_frame / s_avg (< 1: the frame is sharper)
  grain_*         along medium and wide vessels (mask width >= lambda, runs >= 3
                  lambda away from junctions): the frame's and the average's
                  mean ln intensity over the central half of the lumen,
                  high-passed along the vessel; robust SD, with the noise's
                  robust SD subtracted in quadrature (noise: white noise with
                  each image's own noise-versus-intensity model, estimated from
                  the Laplacian of vessel-free pixels, pushed through the same
                  sampling):
                  grain_frame / grain_avg   each image alone (scales < lambda)
                  grain_fine / grain_coarse the frame minus the average (static
                                            structure cancels): along-vessel
                                            scales < ~2.5 px / 2.5 px - lambda
                  grain_rel                 frame minus average, all scales <
                                            lambda, / the vessels' median contrast
                  grain_sham                the same measured beside the vessels
                                            (should be 0: checks the noise model)

v2 tells (2026-10-01; the v1 realism review's proposals, as targets for the
v2 round; baseline table: vesselscene/_scratch/v2/baseline.md).  Stills
(v2_still_tells, also in still_tells):

  knot_tells       dark_knot_rate, knot_p10, knot_med, dark_knot_frac: is a
                   crossing darker than its darker arm (dark beads at crossings)?
  hier_tells       n_long, n_dark20 (paths >= 10 lambda, per 1200 x 1920 px),
                   darkest_long (Np), dom (darkest / next five)
  vein_tells       vein_depth (Np), vein_fwhm, vein_edge (px): the plateau of the
                   most prominent wide vessel (median re-centred profile)
  wide_shape_tells wide_kink (elbows along wide vessels), wide_rib,
                   wide_rib_peak (periodic ripple inside wide vessels)
  anat_tells       pass-through of stats: along_cv_contrast, along_cv_fwhm
                   (uniform wires), faint_loop_density(_lam), fwhm_p10

A frame against its own average (crisp_tells; frame_record also runs
frame_average_tells and still_tells on the frame): sharper (share of frames
sharper than their average), crisp_c_thin/_med, crisp_fwhm_thin/_med,
crisp_edge_med, crisp_seen_thin (the same vessels fitted in both),
frame_c_thin, frame_cnr_thin (do fine vessels stand out of the noise?),
faint_dash / _absent / _rcv / _gap_rate (faint lines continuous or dashed:
faint_tells on the frame carried onto the average, warp_to_average),
bead_med, bead_med_rel, bead_len (red-cell aggregates along medium vessels),
wide_band_frame (bands across wide lumens).  The reference covers 8 frames
of each of the three bursts with fields (PAIR_BURSTS).

These are detector responses, not truth, and rough: they are meant for
comparing images measured the same way (baseline tables:
vesselscene/_scratch/v1/baseline_tells.md, _scratch/v2/baseline.md).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from . import stats as st

DATA_PATH = Path(__file__).with_name("data") / "real_tells.json"
DENSE_SITE = ("09-16_15-50-30", "09-16_15-50-36", "09-16_15-50-52", "09-17_13-51-26")   # the full-frame stills
V0_KEYS = ("F_sd", "dark128_0.3", "dark128_0.45", "darkw_p99", "focus_var", "bg_patch_cv", "c_thin", "n_thin",
           "c_wide", "par_excess", "cprof_p90")
LAYOUT_KEYS = ("web_lcc", "web_lcc25", "web_span", "node_rate", "branch_rate", "xing_rate", "xing_frac",
               "xing_angle_med", "xing_shallow", "cont_share_5", "cont_share_10", "cont_share_20", "cont_thin_10",
               "cont_thin_20", "cont_straight", "calibre_change", "thin_coh")
PAIR_KEYS = ("pair_frac", "pair_run5", "bundle_frac", "bundle_run5", "pair_wide_frac")
WIDE_KEYS = ("wide_fwhm", "wide_edge", "wide_edge_rel", "wide98_fwhm", "wide98_edge", "wide98_edge_rel",
             "wide_thin_edge")
FOCUS_KEYS = ("soft_tiles", "tile_edge_ratio")
STILL_KEYS_V1 = V0_KEYS + LAYOUT_KEYS + PAIR_KEYS + WIDE_KEYS + FOCUS_KEYS
FRAME_KEYS = ("drop", "weak", "r_cv")
PAIR_FRAME_KEYS = ("sigma_extra", "sigma_extra_med", "sigma_extra_wide", "edge_ratio", "grain_frame", "grain_avg",
                   "grain_fine", "grain_coarse", "grain_rel", "grain_sham")
GAPS = {  # key -> which of the five v1 gaps it measures (0 = a v0 tell)
    **{k: 0 for k in V0_KEYS}, **{k: 1 for k in LAYOUT_KEYS}, **{k: 2 for k in PAIR_KEYS}, "par_excess": 2,
    **{k: 3 for k in WIDE_KEYS}, "darkw_p99": 3, "dark128_0.3": 3, "dark128_0.45": 3,
    **{k: 4 for k in FOCUS_KEYS}, "focus_var": 4, **{k: 5 for k in FRAME_KEYS + PAIR_FRAME_KEYS}}
# the real burst whose raw frames are compared with its average (stabilization fields available)
PAIR_BURST = ("09-16_15-50-52", "nonrigid")
PAIR_FRAMES = (5, 20, 30, 50, 60, 70, 90, 110)        # raw-frame speeds 13-250 px/s (transforms.csv)
# v2 (2026-10-01): all three bursts with raw frames and non-rigid fields; 8 frames each (PAIR_FRAMES for
# 15-50-52, 8 frames evenly spaced over the used frames of the others: select_frames)
PAIR_BURSTS = (("09-16_15-22-26", "nonrigid"), ("09-16_15-31-37", "nonrigid"), ("09-16_15-50-52", "nonrigid"))
KNOT_KEYS = ("dark_knot_rate", "knot_p10", "knot_med", "dark_knot_frac")
HIER_KEYS = ("n_long", "n_dark20", "dom", "darkest_long")
VEIN_KEYS = ("vein_depth", "vein_fwhm", "vein_edge")
WIDE2_KEYS = ("wide_kink", "wide_rib", "wide_rib_peak")
ANAT_KEYS = ("along_cv_contrast", "along_cv_fwhm", "faint_loop_density", "faint_loop_density_lam", "fwhm_p10")
V2_STILL_KEYS = KNOT_KEYS + HIER_KEYS + VEIN_KEYS + WIDE2_KEYS + ANAT_KEYS
STILL_KEYS = STILL_KEYS_V1 + V2_STILL_KEYS
CRISP_KEYS = ("sharper", "crisp_edge_med", "crisp_fwhm_thin", "crisp_fwhm_med", "crisp_c_thin", "crisp_c_med",
              "crisp_seen_thin", "frame_c_thin", "frame_cnr_thin", "faint_dash", "faint_absent", "faint_rcv",
              "faint_gap_rate", "bead_med", "bead_med_rel", "bead_len", "wide_band_frame")
FRAME_STILL_KEYS = ("dark_knot_rate", "knot_p10", "n_long", "darkest_long", "fwhm_p10", "c_thin")   # on the raw frame
# v2 round priorities (1 single frames, 2 dark knots, 3 anatomy detail, 4 widest vein)
PRIORITY = {**{k: 1 for k in CRISP_KEYS}, **{k: 2 for k in KNOT_KEYS}, **{k: 3 for k in HIER_KEYS + ANAT_KEYS},
            "wide_kink": 3, "darkest_long": 4, **{k: 4 for k in VEIN_KEYS}, "wide_rib": 4, "wide_rib_peak": 4}
_NAN = float("nan")


def _nan_dict(keys):
    return {k: _NAN for k in keys}


# ====================================================================== stills
def still_tells(I: np.ndarray, valid: np.ndarray | None = None, res: dict | None = None, v1: bool = True,
                v2: bool | None = None) -> dict:
    """The still statistics above for one image in DN (NaN = no data).  res:
    a stats.analyse(..., keep_maps=True) result to reuse.  v1=False gives only
    the v0 tells (V0_KEYS); v2 (default: as v1) adds V2_STILL_KEYS
    (v2_still_tells)."""
    v2 = v1 if v2 is None else v2
    from skimage.morphology import skeletonize
    I = np.asarray(I, np.float64)
    valid = (np.isfinite(I) & (I > 0)) if valid is None else valid
    res = res if res is not None and "maps" in res else st.analyse(I, valid, keep_maps=True)
    m = res["maps"]
    F, M, Sk, v = m["F"], m["M"], m["S"], m["valid"]
    lam = res["lam"]
    out = dict(lam=float(lam))
    v1_ = st.eroded_valid(v, 3)
    out["F_sd"] = float(np.std(F[v1_]))
    F128 = st.flatten(I, valid, block=128)[0]
    vv = st.eroded_valid(v, 8)
    for t in (0.3, 0.45):
        out[f"dark128_{t}"] = float((F128[vv] < -t).mean())
    dm = ndi.binary_opening((F128 < -0.3) & vv, iterations=1)
    w = 2 * ndi.distance_transform_edt(dm)[skeletonize(dm)] if dm.sum() > 100 else np.zeros(0)
    out["darkw_p99"] = float(np.quantile(w, 0.99)) if len(w) > 20 else 0.0
    pr = res["prof"]
    thin = (pr["fwhm"] >= 0.5 * lam) & (pr["fwhm"] < lam)
    wide = (pr["fwhm"] >= 2 * lam) & (pr["fwhm"] < 4 * lam)
    out["c_thin"] = float(np.median(pr["contrast"][thin])) if thin.sum() > 20 else _NAN
    out["n_thin"] = float(thin.mean()) if len(thin) else _NAN
    out["c_wide"] = float(np.median(pr["contrast"][wide])) if wide.sum() > 20 else _NAN
    out["cprof_p90"] = float(np.quantile(pr["contrast"], 0.9)) if len(pr["contrast"]) else _NAN
    meds = _tile_edge_medians(res, log=True)
    out["focus_var"] = float(np.std(meds)) if len(meds) >= 4 else _NAN
    # patchiness of the fine texture away from vessels
    H, W = F.shape
    band = ndi.gaussian_filter(F, 1.0) - ndi.gaussian_filter(F, 2.0)
    keep = v1_ & ~ndi.binary_dilation(M, iterations=4)
    rms = []
    for y0 in range(0, H - 63, 64):
        for x0 in range(0, W - 63, 64):
            kk = keep[y0:y0 + 64, x0:x0 + 64]
            if kk.sum() > 1000:
                rms.append(band[y0:y0 + 64, x0:x0 + 64][kk].std())
    rms = np.array(rms)
    out["bg_patch_cv"] = float(rms.std() / rms.mean()) if len(rms) > 10 else _NAN
    out["par_excess"] = float(res.get("par_excess_3_30", np.nan))
    cont = continuation_paths(Sk, lam) if (v1 or v2) else None
    if v1:
        out.update(layout_tells(res, cont))
        out.update(pair_tells(res, cont))
        out.update(wide_focus_tells(res))
    if v2:
        out.update(v2_still_tells(I, valid, res, cont))
    return out


def _tile_edge_medians(res, min_n=25, log=False):
    """Median edge width (px; with log, median log edge width) of the profiles
    narrower than 1.5 lambda in each of 4 x 6 tiles (2 x 6 on frames up to 600
    rows) with >= min_n of them."""
    pr, lam = res["prof"], res["lam"]
    Sk = res["maps"]["S"]
    ys, xs = np.nonzero(Sk)
    if not len(pr["idx"]):
        return np.zeros(0)
    py, px = ys[pr["idx"]], xs[pr["idx"]]
    H, W = Sk.shape
    nty = 4 if H > 600 else 2
    ty = np.minimum((py * nty) // H, nty - 1)
    tx = np.minimum((px * 6) // W, 5)
    meds = []
    for a in range(nty):
        for b in range(6):
            mm = (ty == a) & (tx == b) & (pr["fwhm"] < 1.5 * lam)
            if mm.sum() >= min_n:
                meds.append(np.median(np.log(pr["edge"][mm])) if log else np.median(pr["edge"][mm]))
    return np.array(meds)


# ====================================================================== 1. layout
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _arm_dir(c, lam, reach=1.5):
    """Unit direction (dy, dx) of a branch leaving its node (c ordered away
    from it), from 0.25 lambda to reach x lambda of arc."""
    d = st._arc(c)
    L = d[-1] if len(d) else 0.0
    if L <= 0:
        return np.array([np.nan, np.nan])
    d1 = min(reach * lam, L)
    d0 = min(0.25 * lam, 0.3 * L)
    p0 = np.array([np.interp(d0, d, c[:, 0]), np.interp(d0, d, c[:, 1])])
    p1 = np.array([np.interp(d1, d, c[:, 0]), np.interp(d1, d, c[:, 1])])
    v = p1 - p0
    n = math.hypot(*v)
    if n < 1e-6:
        v = c[-1] - c[0]
        n = math.hypot(*v)
    return v / n if n > 1e-6 else np.array([np.nan, np.nan])


def _offset(p, u, q):
    """Distance of q from the line through p along unit u."""
    d = q - p
    return abs(d[0] * u[1] - d[1] * u[0])


def continuation_paths(S, lam, turn_max=30.0, merge=1.0, xmerge=3.0, reach=1.5, bridge=True):
    """Continuation paths of a skeleton through junctions and crossings (module
    docstring, gap 1).

    Returns dict(yx (N, 2) skeleton pixels, paths [pixel-index arrays in path
    order], path_len (px, resampled arc incl. the jumps across nodes and gaps),
    crossings [angle deg], cross_yx, n_branch (branch points), n_xnode
    (crossing nodes), n_bridge_cross, net (stats.trace_skeleton)).
    """
    net = st.trace_skeleton(S, merge_len=0.5 * lam)
    yx, br, ends, kind = net["yx"], net["branches"], net["ends"], net["node_kind"]
    nb = len(br)
    blen = np.array([st._arc(yx[b].astype(float))[-1] if len(b) > 1 else 0.0 for b in br])
    nn = len(kind)
    parent = list(range(nn))

    def isj(n):
        return n >= 0 and kind[n] == 1

    def arm(bi, side):
        c = yx[br[bi]].astype(float)
        c = c if side == 0 else c[::-1]
        return _arm_dir(c, lam, reach), c[0]

    def straight(ua, pa, ub, pb):
        if not (np.all(np.isfinite(ua)) and np.all(np.isfinite(ub))):
            return None
        cth = float(ua @ -ub)
        if cth < math.cos(math.radians(turn_max)):
            return None
        if _offset(pa, -ua, pb) > 0.5 * lam + 0.25 * math.hypot(*(pb - pa)):
            return None
        return math.degrees(math.acos(min(1.0, cth)))

    for bi, (a, b) in enumerate(ends):                    # 1: merge junctions joined by short branches
        if isj(a) and isj(b) and a != b and blen[bi] <= merge * lam:
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                parent[rb] = ra

    def ext_arms():
        arms = {}
        for bi, (a, b) in enumerate(ends):
            ra = _find(parent, a) if isj(a) else -1
            rb = _find(parent, b) if isj(b) else -1
            if ra >= 0 and ra == rb:
                continue                                  # internal to a merged node
            if ra >= 0:
                arms.setdefault(ra, []).append((bi, 0))
            if rb >= 0:
                arms.setdefault(rb, []).append((bi, 1))
        return arms

    arms = ext_arms()
    for bi, (a, b) in enumerate(ends):                    # 2: shallow crossings (X across a 1-3 lambda branch)
        if not (isj(a) and isj(b)) or not (merge * lam < blen[bi] <= xmerge * lam):
            continue
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb:
            continue
        A = [x for x in arms.get(ra, []) if x[0] != bi]
        B = [x for x in arms.get(rb, []) if x[0] != bi]
        if len(A) != 2 or len(B) != 2:
            continue
        dA, dB = [arm(*x) for x in A], [arm(*x) for x in B]
        for perm in ((0, 1), (1, 0)):
            if all(straight(*dA[i], *dB[perm[i]]) is not None for i in (0, 1)):
                parent[rb] = ra
                arms = ext_arms()
                break
    link, crossings, cross_yx, n_branch = {}, [], [], 0   # 3: pair arms, straightest first
    for n, al in arms.items():
        if len(al) < 2:
            continue
        D = [arm(*x) for x in al]
        cand = []
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                t = straight(*D[i], *D[j])
                if t is not None:
                    cand.append((t, i, j))
        used, pairs = set(), []
        for _, i, j in sorted(cand):
            if i in used or j in used:
                continue
            used.update((i, j))
            pairs.append((i, j))
            link[al[i]] = al[j]
            link[al[j]] = al[i]
        if len(al) >= 3:
            if len(pairs) >= 2:
                for p in range(len(pairs)):
                    for q in range(p + 1, len(pairs)):
                        u1 = D[pairs[p][0]][0] - D[pairs[p][1]][0]
                        u2 = D[pairs[q][0]][0] - D[pairs[q][1]][0]
                        ang = abs(math.atan2(u1[0], u1[1]) - math.atan2(u2[0], u2[1])) % math.pi
                        crossings.append(math.degrees(min(ang, math.pi - ang)))
                        cross_yx.append(np.mean([D[x][1] for x in pairs[p] + pairs[q]], 0))
            else:
                n_branch += 1
    n_xnode = len(crossings)
    internal = np.array([isj(a) and isj(b) and _find(parent, a) == _find(parent, b) for a, b in ends], bool)
    used = internal.copy()                                # 4: chains of external branches
    chains = []

    def follow(b, s_in):
        seq = []
        while True:
            used[b] = True
            seq.append(br[b] if s_in == 0 else br[b][::-1])
            nxt = link.get((b, 1 - s_in))
            if nxt is None or used[nxt[0]]:
                break
            b, s_in = nxt
        return np.concatenate(seq)

    for b in range(nb):
        if not used[b]:
            for s in (0, 1):
                if (b, s) not in link:
                    chains.append(follow(b, s))
                    break
    for b in range(nb):
        if not used[b]:
            chains.append(follow(b, 0))
    n_bridge = 0
    if bridge and chains:                                 # 5: across gaps
        bj = st.bridge_paths(yx, [(p, [i]) for i, p in enumerate(chains)], lam, np.shape(S))
        chains = [p for p, _ in st.link_paths(dict(branches=chains), bj["link"])]
        crossings += list(bj["cross_angle"])
        cross_yx += list(bj["cross_yx"])
        n_bridge = len(bj["cross_angle"])
    plen = np.array([st._arc(st.resample_path(yx[p]))[-1] if len(p) > 1 else 0.0 for p in chains])
    return dict(yx=yx, paths=chains, path_len=plen, crossings=np.array(crossings, float),
                cross_yx=np.array(cross_yx, float).reshape(-1, 2), n_branch=int(n_branch), n_xnode=int(n_xnode),
                n_bridge_cross=int(n_bridge), net=net, internal=internal)


def layout_tells(res: dict, cont: dict | None = None) -> dict:
    """Gap 1 statistics (module docstring) from a stats.analyse(keep_maps=True) result."""
    m = res["maps"]
    lam = res["lam"]
    S, M, F, valid = m["S"], m["M"], m["F"], m["valid"]
    out = _nan_dict(LAYOUT_KEYS)
    if S.sum() < 50:
        return out
    cont = cont if cont is not None else continuation_paths(S, lam)
    # web connectivity of the darker half / quarter of the skeleton: seeds are
    # the skeleton pixels darker than the skeleton's median (quartile) of F,
    # grown through skeleton and vessel-mask pixels darker than its 75th
    # percentile (hysteresis, so that a vessel of even darkness near the
    # threshold is not cut into specks, and through the mask because the
    # skeleton often breaks at crossings); share of the seed pixels in the
    # largest component
    Fs = ndi.gaussian_filter(F, 1.0)
    fs = Fs[S]
    H, W = S.shape
    k3 = np.ones((3, 3), bool)
    grow = (S | M) & (Fs <= np.quantile(fs, 0.75))     # the mask too: the skeleton often breaks at crossings
    for tag, q in (("", 0.5), ("25", 0.25)):
        seed = S & (Fs <= np.quantile(fs, q))
        lab, n = ndi.label(grow, structure=k3)
        keep = np.zeros(n + 1, bool)
        keep[np.unique(lab[seed])] = True
        keep[0] = False
        lab = np.where(keep[lab], lab, 0)
        Ss = seed
        L = float(Ss.sum())
        if n == 0 or L < 20:
            continue
        sizes = np.bincount(lab[Ss].ravel(), minlength=n + 1)[1:]
        out[f"web_lcc{tag}"] = float(sizes.max() / L)
        if tag == "":
            span = 0.0
            for i, sl in enumerate(ndi.find_objects(lab)):
                if sl is not None and ((sl[0].stop - sl[0].start) >= 0.5 * H or (sl[1].stop - sl[1].start) >= 0.5 * W):
                    span += sizes[i]
            out["web_span"] = float(span / L)
    # nodes and crossings per skeleton length
    yx = cont["yx"]
    L_lam = len(yx) / lam
    ca = cont["crossings"]
    out["branch_rate"] = float(cont["n_branch"] / L_lam * 100)
    out["xing_rate"] = float(len(ca) / L_lam * 100)
    out["node_rate"] = out["branch_rate"] + out["xing_rate"]
    out["xing_frac"] = float(len(ca) / max(len(ca) + cont["n_branch"], 1))
    if len(ca) >= 3:
        out["xing_angle_med"] = float(np.median(ca))
        out["xing_shallow"] = float((ca < 30.0).mean())
    # continuation lengths: every skeleton pixel takes the longest path through it
    best = np.zeros(len(yx))
    for p, Lp in zip(cont["paths"], cont["path_len"]):
        best[p] = np.maximum(best[p], Lp)
    width = 2 * ndi.distance_transform_edt(M)[yx[:, 0], yx[:, 1]]
    thin = width <= lam
    for c in (5, 10, 20):
        out[f"cont_share_{c}"] = float((best >= c * lam).mean())
    for c in (10, 20):
        out[f"cont_thin_{c}"] = float((best[thin] >= c * lam).mean()) if thin.sum() > 50 else _NAN
    # straightness and calibre along long paths; orientation coherence of thin paths
    sa, lr, P, T = [], [], [], []
    for p, Lp in zip(cont["paths"], cont["path_len"]):
        if len(p) < 5:
            continue
        c = yx[p].astype(float)
        if Lp >= 10 * lam:
            q = st.resample_path(c)
            sa.append(math.hypot(*(q[-1] - q[0])) / max(st._arc(q)[-1], 1e-9))
        if Lp >= 6 * lam:
            w = ndi.gaussian_filter1d(width[p].astype(float), lam, mode="nearest")
            s = st._arc(c)
            i0 = np.searchsorted(s, np.arange(0.0, s[-1] - 5 * lam, lam))
            i1 = np.minimum(np.searchsorted(s, s[i0] + 5 * lam), len(s) - 1)
            lr.append(np.abs(np.log(np.maximum(w[i1], 1.0) / np.maximum(w[i0], 1.0))))
        cs = ndi.gaussian_filter1d(c, 3.0, axis=0, mode="nearest")
        tg = np.gradient(cs, axis=0)
        P.append(p)
        T.append(np.arctan2(tg[:, 0], tg[:, 1]))
    if sa:
        out["cont_straight"] = float(np.median(sa))
    if lr:
        out["calibre_change"] = float(np.median(np.concatenate(lr)))
    if P:
        P, T = np.concatenate(P), np.concatenate(T)
        th = thin[P]
        s = max(8, int(10 * lam))
        key = (yx[P, 0] // s) * 100000 + yx[P, 1] // s
        z = np.exp(2j * T)
        vals, wts = [], []
        for kk in np.unique(key[th]):
            mm = th & (key == kk)
            if mm.sum() >= 2 * lam:
                vals.append(abs(z[mm].mean()))
                wts.append(mm.sum())
        if vals:
            out["thin_coh"] = float(np.average(vals, weights=wts))
    return out


# ====================================================================== 2. pairs and bundles
def line_groups(F, pts, tan, wM, lam, reach=2.0, along=1.0, prom_abs=0.008, prom_rel=0.15, gap=2.0,
                min_sep=3.0, step=0.5):
    """Parallel dark lines beside each point (module docstring, gap 2).

    F: flattened log image; pts (n, 2) yx; tan (n, 2) unit tangents (dy, dx);
    wM (n,) mask width (px).  Returns (g, wl): the number of lines in the group
    of the line at each point (0 = no line found there) and that line's width
    at half prominence (px, NaN = none)."""
    from scipy.signal import find_peaks, peak_widths
    n = len(pts)
    g = np.zeros(n, int)
    wl = np.full(n, np.nan)
    if n == 0:
        return g, wl
    half = 0.5 * wM + reach * lam
    t = np.arange(-float(np.max(half)), float(np.max(half)) + 1e-9, step)
    u = np.arange(-along * lam, along * lam + 1e-9, 1.0)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    dist = max(1, int(round(min_sep / step)))
    for b0 in range(0, n, 1500):
        sl = slice(b0, min(n, b0 + 1500))
        P, Tn, Nn = pts[sl].astype(float), tan[sl], nrm[sl]
        cy = P[:, 0, None, None] + t[None, :, None] * Nn[:, 0, None, None] + u[None, None, :] * Tn[:, 0, None, None]
        cx = P[:, 1, None, None] + t[None, :, None] * Nn[:, 1, None, None] + u[None, None, :] * Tn[:, 1, None, None]
        prof = ndi.map_coordinates(F, [cy.ravel(), cx.ravel()], order=1, mode="nearest").reshape(cy.shape).mean(2)
        for i in range(prof.shape[0]):
            k = b0 + i
            mk = np.abs(t) <= half[k]
            p, tt = prof[i][mk], t[mk]
            pk, props = find_peaks(-p, prominence=prom_abs, distance=dist)
            if not len(pk):
                continue
            pr = props["prominences"]
            tk = tt[pk]
            j0 = int(np.argmin(np.abs(tk)))
            if abs(tk[j0]) > 0.5 * wM[k] + 2.0:
                continue
            wl[k] = float(peak_widths(-p, pk[j0:j0 + 1], rel_height=0.5)[0][0] * step)
            keep = pr >= max(prom_abs, prom_rel * pr[j0])
            keep[j0] = True
            j0 = int(np.flatnonzero(keep).tolist().index(j0))
            tk = tk[keep]
            lo = hi = j0
            while lo > 0 and tk[lo] - tk[lo - 1] <= gap * lam:
                lo -= 1
            while hi + 1 < len(tk) and tk[hi + 1] - tk[hi] <= gap * lam:
                hi += 1
            g[k] = hi - lo + 1
    return g, wl


def _window_majority(flag, pid, arc, ok, L, need=0.5):
    """flag & (share of flag over the centred window of L px of arc along the
    same path >= need); points whose window leaves the path are False."""
    out = np.zeros(len(flag), bool)
    for i in np.unique(pid):
        m = np.flatnonzero(pid == i)
        a = arc[m]
        if a[-1] - a[0] < L:
            continue
        f = (flag[m] & ok[m]).astype(float)
        c = np.r_[0.0, np.cumsum(f)]
        lo = np.searchsorted(a, a - L / 2)
        hi = np.searchsorted(a, a + L / 2, side="right")
        full = (a - L / 2 >= a[0]) & (a + L / 2 <= a[-1])
        out[m] = full & ((c[hi] - c[lo]) / np.maximum(hi - lo, 1) >= need) & flag[m]
    return out


def pair_tells(res: dict, cont: dict | None = None, sub: int = 2, run_lam: float = 5.0) -> dict:
    """Gap 2 statistics (module docstring)."""
    m = res["maps"]
    lam = res["lam"]
    S, M = m["S"], m["M"]
    out = _nan_dict(PAIR_KEYS)
    cont = cont if cont is not None else continuation_paths(S, lam)
    yx = cont["yx"]
    width = 2 * ndi.distance_transform_edt(M)[yx[:, 0], yx[:, 1]]
    ok_img = st.eroded_valid(m["valid"], 3 * lam)
    P, T, Wm, A, PID = [], [], [], [], []
    for i, p in enumerate(cont["paths"]):
        c = yx[p].astype(float)
        a = st._arc(c)
        if len(p) < 5 or a[-1] < 3 * lam:
            continue
        tg = np.gradient(ndi.gaussian_filter1d(c, 3.0, axis=0, mode="nearest"), axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        idx = np.arange(0, len(p), sub)
        P.append(c[idx])
        T.append(tg[idx])
        Wm.append(ndi.gaussian_filter1d(width[p], 3.0, mode="nearest")[idx])
        A.append(a[idx])
        PID.append(np.full(len(idx), i))
    if not P:
        return out
    P, T, Wm, A, PID = (np.concatenate(x) for x in (P, T, Wm, A, PID))
    ok = ok_img[P[:, 0].astype(int), P[:, 1].astype(int)]
    F = ndi.gaussian_filter(m["F"], 0.7)
    g, wl = line_groups(F, P, T, np.maximum(Wm, 2.0), lam)
    thin = ok & (wl <= lam)
    wide = ok & (wl >= 2 * lam)
    if thin.sum() >= 50:
        for k, tag in ((2, "pair"), (3, "bundle")):
            out[f"{tag}_frac"] = float((g[thin] >= k).mean())
            out[f"{tag}_run5"] = float(_window_majority(g >= k, PID, A, ok, run_lam * lam)[thin].mean())
    if wide.sum() >= 20:
        out["pair_wide_frac"] = float((g[wide] >= 2).mean())
    return out


# ====================================================================== 3-4. wide vessels, focus
def wide_focus_tells(res: dict) -> dict:
    """Gaps 3 and 4 (module docstring), from the suite's profiles."""
    out = _nan_dict(WIDE_KEYS + FOCUS_KEYS)
    pr, lam = res["prof"], res["lam"]
    fw, e = pr["fwhm"], pr["edge"]
    if len(fw) >= 100:
        for tag, q in (("wide", 0.9), ("wide98", 0.98)):
            thr = np.quantile(fw, q)
            sel = fw >= thr
            out[f"{tag}_fwhm"] = float(thr)
            out[f"{tag}_edge"] = float(np.median(e[sel]))
            out[f"{tag}_edge_rel"] = float(np.median(e[sel] / fw[sel]))
        thin = (fw >= 0.5 * lam) & (fw < lam)
        if thin.sum() >= 20:
            out["wide_thin_edge"] = float(out["wide_edge"] / np.median(e[thin]))
    meds = _tile_edge_medians(res)
    if len(meds) >= 4:
        out["soft_tiles"] = float((meds > 1.5 * meds.min()).mean())
        out["tile_edge_ratio"] = float(np.quantile(meds, 0.9) / np.quantile(meds, 0.1))
    return out


# ====================================================================== 5. frames
def _register_tiles(fr, avg, tile=240):
    """The frame warped onto the average, one translation per tile (phase
    correlation after a global shift); NaN where unknown."""
    import cv2
    a0 = np.log(np.nan_to_num(avg, nan=np.nanmedian(avg)).clip(1)).astype(np.float32)
    b0 = np.log(np.asarray(fr, np.float32).clip(1))
    a0 -= cv2.GaussianBlur(a0, (0, 0), 15)
    b0 -= cv2.GaussianBlur(b0, (0, 0), 15)
    H, W = a0.shape
    out = np.full((H, W), np.nan, np.float32)
    (gx, gy), _ = cv2.phaseCorrelate(b0, a0)
    M = np.float32([[1, 0, gx], [0, 1, gy]])
    frg = cv2.warpAffine(np.asarray(fr, np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=np.nan)
    bg = cv2.warpAffine(b0, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    for y0 in range(0, H - tile + 1, tile // 2):
        for x0 in range(0, W - tile + 1, tile // 2):
            (dx, dy), resp = cv2.phaseCorrelate(bg[y0:y0 + tile, x0:x0 + tile], a0[y0:y0 + tile, x0:x0 + tile], win)
            if resp < 0.1 or abs(dx) > 8 or abs(dy) > 8:
                continue
            w = cv2.warpAffine(frg, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR,
                               borderValue=np.nan)
            c = tile // 4
            out[y0 + c:y0 + tile - c, x0 + c:x0 + tile - c] = w[y0 + c:y0 + tile - c, x0 + c:x0 + tile - c]
    return out


def frame_tells(frame: np.ndarray, average: np.ndarray, res_avg: dict | None = None) -> dict:
    """drop / weak / r_cv of a single frame against its average (module
    docstring).  res_avg: stats.analyse(average, keep_maps=True) to reuse."""
    avg = np.asarray(average, np.float64)
    frr = _register_tiles(frame, avg)
    vfr = np.isfinite(frr) & (frr > 0) & (frr < 4095)
    Ffr = st.flatten(np.where(vfr, frr, np.nan), vfr)[0]
    va = np.isfinite(avg) & (avg > 0)
    res = res_avg if res_avg is not None and "maps" in res_avg else st.analyse(avg, va, keep_maps=True)
    m = res["maps"]
    Fa, Sk, k = m["F"], m["S"], m["k"]
    sc = np.asarray(st.SCALES)[k]
    Fa_s = ndi.gaussian_filter(Fa, 1.5)
    Ff_s = ndi.minimum_filter(ndi.gaussian_filter(np.nan_to_num(Ffr), 1.5), 3)
    ok = ndi.binary_erosion(vfr & va, iterations=6)
    sel = Sk & ok & ((sc == 2) | (sc == 4)) & (Fa_s < -0.06)
    r = (-Ff_s[sel]) / (-Fa_s[sel])
    if not len(r):
        return dict(drop=_NAN, weak=_NAN, r_cv=_NAN, n=0)
    med = float(np.median(r))
    return dict(drop=float((r < 0.3).mean()), weak=float((r < 0.6).mean()),
                r_cv=float(1.4826 * np.median(np.abs(r - med)) / max(med, 1e-6)), n=int(sel.sum()))


def _FieldSet():
    """FieldSet (analysis/stabilize/fields.py) of the LIMBUS repository, which
    carries a raw frame onto its stabilized average.  Optional: only needed
    to rebuild the real frame references; found on the path, else in the
    LIMBUS checkout of realism.find_data_root."""
    try:
        from analysis.stabilize.fields import FieldSet
    except ImportError:
        import sys
        from . import realism as RM
        root = RM.find_data_root()
        if root is None or not (root / "analysis").is_dir():
            raise ImportError("needs the LIMBUS repository (github.com/karimghabra/limbus): "
                              "set LIMBUS_DATA to its checkout") from None
        sys.path.insert(0, str(root))
        from analysis.stabilize.fields import FieldSet
    return FieldSet


def fields_mapping(fields_path, frame_index):
    """Average -> raw-frame coordinates of one frame of a stabilized burst:
    x_frame = x + d(x) with the pipeline's own displacement field (affine +
    bicubic local residual; analysis/stabilize/fields.py, output(x) =
    frame(x + d(x))).  Returns f((N, 2) xy) -> (N, 2) xy."""
    fs = _FieldSet()(str(fields_path))
    if not fs.has(frame_index):
        raise KeyError(f"frame {frame_index} has no field in {fields_path}")

    def f(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return xy + fs.at(frame_index, xy)
    return f


def _identity(xy):
    return np.asarray(xy, np.float64).reshape(-1, 2)


_SQ2PI = math.sqrt(2 * math.pi)


def fit_edges(T, P, t0=None, h0=None, s0=None, iters=40):
    """Batched Levenberg-Marquardt fit of a box with Gaussian edges on a linear
    baseline, p(t) = b0 + b1 t - c [Phi((t - t0 + h) / s) - Phi((t - t0 - h) / s)],
    to profiles P (n, m) sampled at T (n, m).  t0, h0, s0: starting values
    ((n,) or None).  Returns dict(b0, b1, c, t0, h, s, rms) of (n,) arrays;
    s is the edge sigma (px)."""
    from scipy.special import ndtr
    T = np.asarray(T, float)
    P = np.asarray(P, float)
    n, m = P.shape
    th = np.zeros((n, 6))
    hi = np.quantile(P, 0.9, axis=1)
    th[:, 0] = hi
    th[:, 2] = np.maximum(hi - P.min(1), 1e-3)
    th[:, 3] = 0.0 if t0 is None else t0
    th[:, 4] = np.maximum(0.5 * (np.full(n, 10.0) if h0 is None else np.asarray(h0, float)), 1.0)
    th[:, 5] = np.log(np.maximum(np.full(n, 2.0) if s0 is None else np.asarray(s0, float), 0.5))
    lamb = np.full(n, 1e-2)

    def model(th):
        b0, b1, c, tc, h, ls = th.T
        s = np.exp(ls)[:, None]
        z1 = (T - tc[:, None] + h[:, None]) / s
        z2 = (T - tc[:, None] - h[:, None]) / s
        box = ndtr(z1) - ndtr(z2)
        f = b0[:, None] + b1[:, None] * T - c[:, None] * box
        p1, p2 = np.exp(-0.5 * z1 * z1) / _SQ2PI, np.exp(-0.5 * z2 * z2) / _SQ2PI
        J = np.stack([np.ones_like(T), T, -box, c[:, None] * (p1 - p2) / s, -c[:, None] * (p1 + p2) / s,
                      c[:, None] * (p1 * z1 - p2 * z2)], 2)
        return f, J

    f, J = model(th)
    r = P - f
    cost = (r * r).sum(1)
    eye = np.eye(6)[None]
    for _ in range(iters):
        JtJ = np.einsum("nmi,nmj->nij", J, J)
        g = np.einsum("nmi,nm->ni", J, r)
        A = JtJ + lamb[:, None, None] * eye * np.maximum(np.diagonal(JtJ, 0, 1, 2)[:, :, None], 1e-12)
        try:
            d = np.linalg.solve(A, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        tn = th + d
        tn[:, 4] = np.maximum(tn[:, 4], 0.25)
        tn[:, 5] = np.clip(tn[:, 5], math.log(0.2), math.log(40.0))
        fn, Jn = model(tn)
        rn = P - fn
        cn = (rn * rn).sum(1)
        better = np.isfinite(cn) & (cn < cost)
        th = np.where(better[:, None], tn, th)
        J = np.where(better[:, None, None], Jn, J)
        r = np.where(better[:, None], rn, r)
        cost = np.where(better, cn, cost)
        lamb = np.clip(np.where(better, lamb * 0.3, lamb * 5.0), 1e-7, 1e7)
    b0, b1, c, tc, h, ls = th.T
    return dict(b0=b0, b1=b1, c=c, t0=tc, h=h, s=np.exp(ls), rms=np.sqrt(cost / m))


def noise_model(I, valid, exclude=None, n_bins=24):
    """Pixel noise SD (image units) as a function of local intensity:
    robust variance of the 5-point Laplacian / 20 (white noise) of valid,
    non-excluded pixels per intensity bin, fitted with a quadratic in I.
    Returns (sd(I) callable, coefficients)."""
    I = np.asarray(I, float)
    Iv = np.where(valid, I, 0.0)
    lap = ndi.convolve(Iv, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float))
    mu = ndi.uniform_filter(Iv, 5)
    keep = ndi.binary_erosion(valid, iterations=3)
    if exclude is not None:
        keep &= ~exclude
    x, y = mu[keep], lap[keep]
    coef = np.array([np.var(y) / 20.0, 0.0, 0.0]) if len(x) else np.array([1e-6, 0.0, 0.0])
    if len(x) >= 1000:
        qs = np.quantile(x, np.linspace(0.02, 0.98, n_bins + 1))
        xb, vb = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            mm = (x >= a) & (x < b)
            if mm.sum() >= 200:
                yy = y[mm]
                xb.append(np.median(x[mm]))
                vb.append((1.4826 * np.median(np.abs(yy - np.median(yy)))) ** 2 / 20.0)
        if len(xb) >= 3:
            xb, vb = np.array(xb), np.array(vb)
            coef = np.linalg.lstsq(np.stack([np.ones_like(xb), xb, xb * xb], 1), vb, rcond=None)[0]
    floor = max(1e-12, 1e-6 * float(np.median(x)) ** 2) if len(x) else 1e-12

    def sd(v):
        v = np.asarray(v, float)
        return np.sqrt(np.maximum(coef[0] + coef[1] * v + coef[2] * v * v, floor))
    return sd, coef


def _log_image(I, valid):
    L = np.zeros(I.shape)
    L[valid] = np.log(I[valid])
    return st.fill_invalid(L, valid)


class _Spline:
    """Cubic-spline coefficients of an image, computed once for many samplings."""

    def __init__(self, L):
        self.c = ndi.spline_filter(np.asarray(L, float), 3, mode="nearest")


def _sample(L, yx, order=3):
    """L at points yx (..., 2): cubic spline (L an image or a _Spline) or
    order-1 interpolation."""
    yx = np.asarray(yx, float)
    crd = [yx[..., 0].ravel(), yx[..., 1].ravel()]
    if isinstance(L, _Spline):
        v = ndi.map_coordinates(L.c, crd, order=3, mode="nearest", prefilter=False)
    else:
        v = ndi.map_coordinates(L, crd, order=order, mode="nearest")
    return v.reshape(yx.shape[:-1])


def _map_yx(mapping, g):
    """Apply an xy mapping to an array of yx points (..., 2)."""
    return mapping(g[..., ::-1].reshape(-1, 2)).reshape(g.shape)[..., ::-1]


def _tangents(S, pts, r=4.0):
    from scipy.spatial import cKDTree
    yx = np.argwhere(S).astype(float)
    tree = cKDTree(yx)
    out = np.zeros((len(pts), 2))
    for i, idx in enumerate(tree.query_ball_point(np.asarray(pts, float), r)):
        q = yx[idx] - yx[idx].mean(0)
        out[i] = np.linalg.eigh(q.T @ q)[1][:, 1]
    return out


def _junction_zone(S, r):
    nb = ndi.convolve(S.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    J = S & (nb >= 3)
    return ndi.distance_transform_edt(~J) <= r if J.any() else np.zeros(S.shape, bool)


def _rsd(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) else _NAN


def frame_average_tells(frame: np.ndarray, average: np.ndarray, mapping=None, res_avg: dict | None = None,
                        frame_valid: np.ndarray | None = None, n_sites: int = 2500, seed: int = 0,
                        min_run: float = 3.0, fine: float = 2.5) -> dict:
    """sigma_extra / edge_ratio / grain_* of one raw frame against its average
    (module docstring, gap 5).

    frame: the raw frame in DN (saturated 4095 = no data); average: its
    stabilized mean in DN (NaN = no data); mapping: average xy -> frame xy for
    (N, 2) arrays (fields_mapping(...) for a real burst; None = identity, for
    a synthetic frame); res_avg: stats.analyse(average, keep_maps=True) to
    reuse.  Also returns n_sites, n_runs and the per-image edge sigmas."""
    mapping = mapping or _identity
    rng = np.random.default_rng(seed)
    A = np.asarray(average, float)
    va = np.isfinite(A) & (A > 0)
    Fr = np.asarray(frame, float)
    vf = ((Fr > 0) & (Fr < 4095)) if frame_valid is None else frame_valid
    res = res_avg if res_avg is not None and "maps" in res_avg else st.analyse(A, va, keep_maps=True)
    m = res["maps"]
    lam = res["lam"]
    S, M = m["S"], m["M"]
    La, Lf = _log_image(A, va), _log_image(Fr, vf)
    SA, SF = _Spline(La), _Spline(Lf)
    H, W = A.shape
    Hf, Wf = Fr.shape
    vfd = ndi.binary_erosion(vf, iterations=2)

    def frame_ok(gf):
        yi = np.rint(gf[..., 0]).astype(int)
        xi = np.rint(gf[..., 1]).astype(int)
        inside = (yi >= 2) & (yi < Hf - 2) & (xi >= 2) & (xi < Wf - 2)
        return inside & vfd[np.clip(yi, 0, Hf - 1), np.clip(xi, 0, Wf - 1)]

    out = dict(lam=float(lam), **_nan_dict(PAIR_FRAME_KEYS))
    jz = _junction_zone(S, lam)
    v8 = st.eroded_valid(va, 2 * lam)
    # ---- (a) edge sigma of the same vessels
    pr = res["prof"]
    ys, xs = np.nonzero(S)
    pts = np.stack([ys, xs], 1)[pr["idx"]] if len(pr["idx"]) else np.zeros((0, 2), int)
    fw, c = pr["fwhm"], pr["contrast"]
    sel = (fw >= 0.8 * lam) & (c >= 0.05)
    if len(pts):
        sel &= ~jz[pts[:, 0], pts[:, 1]] & v8[pts[:, 0], pts[:, 1]]
    idx = np.flatnonzero(sel)
    if len(idx) > n_sites:
        idx = np.sort(rng.choice(idx, n_sites, replace=False))
    out["n_sites"] = 0
    if len(idx) >= 20:
        P0 = pts[idx].astype(float)
        tan = _tangents(S, P0, r=max(4.0, 0.3 * lam))
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        fwi = fw[idx]
        Tmax = np.maximum(fwi / 2 + 13.0, 1.2 * fwi)
        tt = np.linspace(-1, 1, 81)[None, :] * Tmax[:, None]
        uu = np.arange(-3, 4, 1.0)
        grid = (P0[:, None, None, :] + tt[:, :, None, None] * nrm[:, None, None, :]
                + uu[None, None, :, None] * tan[:, None, None, :])
        gf = _map_yx(mapping, grid)
        okf = frame_ok(gf).all((1, 2))
        pa = _sample(SA, grid).mean(2)
        pf = _sample(SF, gf).mean(2)
        fa = fit_edges(tt, pa, h0=fwi, s0=np.full(len(idx), 2.5))
        ff = fit_edges(tt, pf, t0=fa["t0"], h0=2 * fa["h"], s0=fa["s"])
        good = (okf & (fa["c"] > 0.03) & (ff["c"] > 0.3 * fa["c"]) & (fa["s"] > 0.3) & (ff["s"] > 0.3)
                & (fa["s"] < 0.6 * fwi + 2) & (ff["s"] < 0.6 * fwi + 2) & (np.abs(ff["t0"] - fa["t0"]) < 3)
                & (fa["rms"] < 0.25 * fa["c"]))
        out["n_sites"] = int(good.sum())
        wide = fwi >= 2 * lam
        for tag, mm in (("", good), ("_med", good & ~wide), ("_wide", good & wide)):
            if mm.sum() < 20:
                continue
            dv = float(np.median(fa["s"][mm] ** 2 - ff["s"][mm] ** 2))
            out[f"sigma_extra{tag}"] = math.copysign(math.sqrt(abs(dv)), dv)
            out[f"s_avg{tag}"] = float(np.median(fa["s"][mm]))
            out[f"s_frame{tag}"] = float(np.median(ff["s"][mm]))
            if tag == "":
                out["edge_ratio"] = float(np.median(ff["s"][mm] / fa["s"][mm]))
    # ---- (b) grain along medium and wide vessels
    Mw = 2 * ndi.distance_transform_edt(M)
    sd_f, coef_f = noise_model(Fr, vf)
    sd_a, _ = noise_model(np.where(va, A, 0.0), va)
    out["noise_frame_coef"] = [float(x) for x in coef_f]
    runs = []
    for q in m.get("paths", []):
        q = st.resample_path(np.asarray(q, float))
        if len(q) < min_run * lam:
            continue
        qi = np.rint(q).astype(int)
        qi[:, 0] = np.clip(qi[:, 0], 0, H - 1)
        qi[:, 1] = np.clip(qi[:, 1], 0, W - 1)
        w = ndi.gaussian_filter1d(Mw[qi[:, 0], qi[:, 1]], 2.0, mode="nearest")
        ok = (w >= lam) & ~jz[qi[:, 0], qi[:, 1]] & v8[qi[:, 0], qi[:, 1]]
        lab, nl = ndi.label(ok)
        for r in range(1, nl + 1):
            j = np.flatnonzero(lab == r)
            if len(j) >= min_run * lam:
                runs.append((q[j], w[j]))
    out["n_runs"] = 0
    if runs:
        NF = _Spline(rng.standard_normal((Hf, Wf)))     # white noise, sampled like the images
        NA = _Spline(rng.standard_normal((H, W)))
        cut = int(round(0.5 * lam))
        bands = {"all": (0.0, lam), "fine": (0.0, fine), "coarse": (fine, lam)}

        def hp(x, band):
            lo, hi_ = band
            y = (ndi.gaussian_filter1d(x, lo, mode="nearest") if lo > 0 else x) - ndi.gaussian_filter1d(x, hi_, mode="nearest")
            return y[cut:len(y) - cut]

        acc = {k: [] for k in ("f", "a", "d", "nf", "na", "sh", "c")}
        acc_b = {b: dict(d=[], nd=[]) for b in bands}
        for q, w in runs:
            tg = np.gradient(ndi.gaussian_filter1d(q, 2.0, axis=0, mode="nearest"), axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
            tt = np.linspace(-1, 1, 7)[None, :] * np.maximum(1.0, 0.25 * w)[:, None]
            g = q[:, None, :] + tt[:, :, None] * nr[:, None, :]              # central half of the lumen
            gsh = g + (0.5 * w + lam)[:, None, None] * nr[:, None, :]      # sham: the same, beside the vessel
            gf, gshf = _map_yx(mapping, g), _map_yx(mapping, gsh)
            okr = frame_ok(gf).all(1) & frame_ok(gshf).all(1)
            lab, nl = ndi.label(okr)                       # the longest stretch seen in the frame
            if nl == 0:
                continue
            keep = np.flatnonzero(lab == 1 + int(np.argmax(np.bincount(lab[lab > 0])[1:])))
            if len(keep) < min_run * lam:
                continue
            q, w, nr = q[keep], w[keep], nr[keep]
            g, gsh, gf, gshf = g[keep], gsh[keep], gf[keep], gshf[keep]
            ca, cf = _sample(SA, g).mean(1), _sample(SF, gf).mean(1)
            If, Ia = np.exp(_sample(Lf, gf, 1)), np.exp(_sample(La, g, 1))
            nf = (_sample(NF, gf) * sd_f(If) / If).mean(1)
            na = (_sample(NA, g) * sd_a(Ia) / Ia).mean(1)
            sh = _sample(SF, gshf).mean(1) - _sample(SA, gsh).mean(1)
            If2, Ia2 = np.exp(_sample(Lf, gshf, 1)), np.exp(_sample(La, gsh, 1))
            nsh = (_sample(NF, gshf) * sd_f(If2) / If2).mean(1) - (_sample(NA, gsh) * sd_a(Ia2) / Ia2).mean(1)
            b = bands["all"]
            acc["f"].append((_rsd(hp(cf, b)), _rsd(hp(nf, b))))
            acc["a"].append((_rsd(hp(ca, b)), _rsd(hp(na, b))))
            acc["sh"].append((_rsd(hp(sh, b)), _rsd(hp(nsh, b))))
            for bn, bb in bands.items():
                acc_b[bn]["d"].append(_rsd(hp(cf - ca, bb)))
                acc_b[bn]["nd"].append(_rsd(hp(nf - na, bb)))
            side = (0.5 * w + 0.5 * lam)[:, None] * np.array([-1.0, 1.0])[None, :]      # the brighter shoulder
            bgv = _sample(SA, q[:, None, :] + side[:, :, None] * nr[:, None, :]).max(1)
            acc["c"].append(float(np.median(bgv - ca)))
        out["n_runs"] = len(acc["c"])
        if acc["c"]:
            def sub(pairs):
                a = np.array(pairs, float)
                return float(math.sqrt(max(np.median(a[:, 0]) ** 2 - np.median(a[:, 1]) ** 2, 0.0)))
            out["grain_frame"] = sub(acc["f"])
            out["grain_avg"] = sub(acc["a"])
            out["grain_sham"] = sub(acc["sh"])
            for bn in bands:
                v = sub(list(zip(acc_b[bn]["d"], acc_b[bn]["nd"])))
                if bn == "all":
                    out["grain_rel"] = v / max(float(np.median(acc["c"])), 1e-6)
                    out["grain_diff"] = v
                else:
                    out[f"grain_{bn}"] = v
            out["vessel_contrast"] = float(np.median(acc["c"]))
    return out


# ====================================================================== v2: stills
def knot_tells(res: dict, cont: dict | None = None, r_node: float = 0.4, r_arm=(1.5, 3.0), dark: float = 0.05,
               min_arm: float = 0.03) -> dict:
    """Darkness of the detected crossings (v2, priority 2; ported from the v1
    review's knots.py).  At each crossing of continuation_paths inside the
    valid area (>= 3 lambda from its edge): Fn = the darkest F (1 px Gaussian)
    within r_node lambda of the node, Fa = the 10th percentile of F on the
    skeleton pixels r_arm lambda away (the darker arm); crossings whose arms
    are fainter than min_arm Np are skipped.  knot = Fn - Fa (< 0: darker than
    its darker arm).  knot_med / knot_p10: median / 10th percentile; a dark
    knot has knot < -dark; dark_knot_frac: their share; dark_knot_rate: their
    number per 100 lambda of skeleton (0 when no crossing is measured)."""
    from scipy.spatial import cKDTree
    m = res["maps"]
    lam = res["lam"]
    out = _nan_dict(KNOT_KEYS)
    out["n_knots"] = 0
    S = m["S"]
    if S.sum() < 50:
        return out
    cont = cont if cont is not None else continuation_paths(S, lam)
    yx = cont["yx"]
    Fs = ndi.gaussian_filter(m["F"], 1.0)
    H, W = Fs.shape
    ev = st.eroded_valid(m["valid"], 3 * lam)
    tree = cKDTree(yx)
    r0 = max(2, int(math.ceil(r_node * lam)))
    vals = []
    for (cy, cx) in cont["cross_yx"]:
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < H and 0 <= ix < W) or not ev[iy, ix]:
            continue
        Fn = float(Fs[max(iy - r0, 0):iy + r0 + 1, max(ix - r0, 0):ix + r0 + 1].min())
        idx = np.asarray(tree.query_ball_point([cy, cx], r_arm[1] * lam), int)
        if not len(idx):
            continue
        d = np.hypot(yx[idx, 0] - cy, yx[idx, 1] - cx)
        ann = idx[d >= r_arm[0] * lam]
        if len(ann) < 6:
            continue
        Fa = float(np.quantile(Fs[yx[ann, 0], yx[ann, 1]], 0.1))
        if Fa > -min_arm:
            continue
        vals.append(Fn - Fa)
    a = np.array(vals, float)
    out["n_knots"] = int(len(a))
    nd = int((a < -dark).sum())
    out["dark_knot_rate"] = float(nd / (len(yx) / lam) * 100) if len(yx) else _NAN
    if len(a) >= 5:
        out["knot_med"] = float(np.median(a))
        out["knot_p10"] = float(np.quantile(a, 0.1))
        out["dark_knot_frac"] = float(nd / len(a))
    return out


def hier_tells(res: dict, cont: dict | None = None, min_len: float = 10.0) -> dict:
    """Hierarchy of the long vessels (v2, priorities 3-4; the v1 review's
    hier.py).  Over the continuation paths >= min_len lambda, each path's
    median F (1 px Gaussian): n_long = their number and n_dark20 = those
    darker than -0.20 Np, both per 1200 x 1920 px of valid area;
    darkest_long = the darkest path's median F (Np); dom = darkest / median
    of the next five (1: many equally dark vessels; large: one dominates)."""
    m = res["maps"]
    lam = res["lam"]
    out = _nan_dict(HIER_KEYS)
    S = m["S"]
    if S.sum() < 50:
        return out
    cont = cont if cont is not None else continuation_paths(S, lam)
    yx = cont["yx"]
    Fs = ndi.gaussian_filter(m["F"], 1.0)
    meds = np.sort([np.median(Fs[yx[p, 0], yx[p, 1]]) for p, L in zip(cont["paths"], cont["path_len"])
                    if L >= min_len * lam])
    sc = 1200 * 1920 / max(float(m["valid"].sum()), 1.0)
    out["n_long"] = float(len(meds) * sc)
    out["n_dark20"] = float((meds < -0.20).sum() * sc)
    if len(meds):
        out["darkest_long"] = float(meds[0])
    if len(meds) >= 6:
        out["dom"] = float(meds[0] / np.median(meds[1:6]))
    return out


def _arm_dir_px(c, d0, d1):
    """Unit direction (dy, dx) of a branch leaving its node (c ordered away
    from it), from d0 to d1 px of arc (clamped to the branch)."""
    d = st._arc(c)
    L = d[-1] if len(d) else 0.0
    if L <= 0:
        return np.array([np.nan, np.nan])
    d1 = min(d1, L)
    d0 = min(d0, 0.3 * d1)
    v = np.array([np.interp(d1, d, c[:, 0]) - np.interp(d0, d, c[:, 0]),
                  np.interp(d1, d, c[:, 1]) - np.interp(d0, d, c[:, 1])])
    n = math.hypot(*v)
    return v / n if n > 1e-6 else np.array([np.nan, np.nan])


def wide_paths(res: dict, seed: float = 2.0, grow: float = 1.5, turn_max: float = 80.0) -> list:
    """Centrelines of the wide vessels, continuous through their junctions.

    The skeleton pixels whose vessel-mask width (2 x distance transform of
    M) is >= grow lambda, in components holding a pixel >= seed lambda wide
    (hysteresis), spurs shorter than the local width pruned; at each junction
    their arms are paired straightest first up to turn_max deg (thin
    tributaries are not in this skeleton, so a wide trunk continues through
    a confluence however much it turns), and the branches are chained.

    Returns [dict(q (n, 2) yx resampled every 1 px and smoothed over a
    quarter width, w (n,) mask width along it (px), w_med, length)]."""
    m = res["maps"]
    lam = res["lam"]
    S, M = m["S"], m["M"]
    Wimg = 2 * ndi.distance_transform_edt(M)
    k3 = np.ones((3, 3), bool)
    lab, n = ndi.label(S & (Wimg >= grow * lam), structure=k3)
    if n == 0:
        return []
    hit = np.zeros(n + 1, bool)
    hit[np.unique(lab[S & (Wimg >= seed * lam)])] = True
    hit[0] = False
    Sw = st.prune_spurs(hit[lab], 0.5 * Wimg, min_len=lam, fac=2.0, min_comp=int(2 * lam))
    if Sw.sum() < 2 * lam:
        return []
    net = st.trace_skeleton(Sw, merge_len=0.5 * lam)
    yx, br = net["yx"], net["branches"]
    wpx = Wimg[yx[:, 0], yx[:, 1]]
    link = {}
    for node, al in net["arms"].items():
        if len(al) < 2:
            continue
        D = []
        for bi, side in al:
            c = yx[br[bi]].astype(float)
            c = c if side == 0 else c[::-1]
            w = float(np.median(wpx[br[bi]]))
            D.append(_arm_dir_px(c, 0.5 * w, 1.5 * w))
        cand = []
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                if np.all(np.isfinite(D[i])) and np.all(np.isfinite(D[j])):
                    t = math.degrees(math.acos(float(np.clip(D[i] @ -D[j], -1, 1))))
                    if t < turn_max:
                        cand.append((t, i, j))
        used = set()
        for _, i, j in sorted(cand):
            if i in used or j in used:
                continue
            used.update((i, j))
            link[al[i]] = al[j]
            link[al[j]] = al[i]
    done = np.zeros(len(br), bool)

    def follow(b, s_in):
        seq = []
        while True:
            done[b] = True
            seq.append(br[b] if s_in == 0 else br[b][::-1])
            nxt = link.get((b, 1 - s_in))
            if nxt is None or done[nxt[0]]:
                break
            b, s_in = nxt
        return np.concatenate(seq)

    chains = []
    for b in range(len(br)):
        if not done[b]:
            for s in (0, 1):
                if (b, s) not in link:
                    chains.append(follow(b, s))
                    break
    for b in range(len(br)):
        if not done[b]:
            chains.append(follow(b, 0))
    out = []
    for p in chains:
        if len(p) < 5:
            continue
        c = yx[p].astype(float)
        w_med = float(np.median(wpx[p]))
        q = st.resample_path(c)
        if len(q) < 5:
            continue
        q = ndi.gaussian_filter1d(q, max(2.0, 0.25 * w_med), axis=0, mode="nearest")
        a_c, a_q = st._arc(c), st._arc(q)
        w = np.interp(a_q / max(a_q[-1], 1e-9), a_c / max(a_c[-1], 1e-9), ndi.gaussian_filter1d(wpx[p], 3.0, mode="nearest"))
        out.append(dict(q=q, w=w, w_med=w_med, length=float(a_q[-1])))
    return out


def _normals(q):
    tg = np.gradient(q, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
    return np.stack([-tg[:, 1], tg[:, 0]], 1)


def _log_minus_bg(I, valid, sigma=150.0):
    """ln I minus its NaN-aware Gaussian mean over sigma px (NaN outside
    valid): the v1 review's rawview image, no percentile background."""
    import cv2
    L = np.where(valid, np.log(np.where(valid, I, 1.0)), 0.0).astype(np.float32)
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(L, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(v, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    G = L.astype(np.float64) - num / np.maximum(den, 1e-6)
    return np.where(valid, G, np.nan)


def wide_shape_tells(res: dict, chains: list | None = None, L: np.ndarray | None = None, kink_deg: float = 20.0,
                     span: float = 2.0, seg: int = 80) -> dict:
    """wide_kink, wide_rib, wide_rib_peak (v2) along wide_paths.

    wide_kink: share of the wide-path length (1 px samples >= span widths
    from the path ends) where the chords over the span widths before and
    after the point differ in heading by more than kink_deg (an elbow of
    theta deg marks about 4 (1 - kink_deg / theta) widths; a smooth bend only
    when its radius is below 5.7 widths).  NaN below 100 px evaluated.
    wide_rib, wide_rib_peak: L (the log image) averaged across the central
    half of the lumen and sampled every 1 px along the wide paths (cubic
    spline), in seg-px Welch segments (half overlap, linear detrend, Hann),
    pooled power spectrum P: max of P over periods 4-10 px / mean of P over
    periods 10-40 px, and / the median of P over 4-10 px (a periodic ripple,
    the renderer's ribs, makes a peak).  NaN with fewer than 3 segments."""
    m = res["maps"]
    lam = res["lam"]
    out = _nan_dict(WIDE2_KEYS)
    chains = wide_paths(res) if chains is None else chains
    ev = kk = 0
    for ch in chains:
        q = ch["q"]
        n = int(round(span * ch["w_med"]))
        if n < 2 or len(q) < 2 * n + 1:
            continue
        a = q[n:-n] - q[:-2 * n]
        b = q[2 * n:] - q[n:-n]
        cs = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
        ang = np.degrees(np.arccos(np.clip(cs, -1, 1)))
        ev += len(ang)
        kk += int((ang > kink_deg).sum())
    out["n_wide_px"] = int(ev)
    if ev >= 100:
        out["wide_kink"] = float(kk / ev)
    if L is None:
        L = m["F"] + m["B"]                               # ln I, filled outside the valid area
    SL = _Spline(L)
    v8 = st.eroded_valid(m["valid"], 2 * lam)
    H, W = v8.shape
    P = np.zeros(seg // 2 + 1)
    nseg = 0
    win = np.hanning(seg)
    xs = np.arange(seg, dtype=float)
    for ch in chains:
        q, w = ch["q"], ch["w"]
        if len(q) < seg:
            continue
        nr = _normals(q)
        tt = np.linspace(-1, 1, 5)[None, :] * (0.25 * w)[:, None]
        x = _sample(SL, q[:, None, :] + tt[:, :, None] * nr[:, None, :]).mean(1)
        qi = np.rint(q).astype(int)
        ok = v8[np.clip(qi[:, 0], 0, H - 1), np.clip(qi[:, 1], 0, W - 1)]
        lab, nl = ndi.label(ok)
        for r in range(1, nl + 1):
            j = np.flatnonzero(lab == r)
            for s0 in range(0, len(j) - seg + 1, seg // 2):
                y = x[j[s0:s0 + seg]]
                y = y - np.polyval(np.polyfit(xs, y, 1), xs)
                P += np.abs(np.fft.rfft(y * win)) ** 2
                nseg += 1
    out["n_rib_seg"] = int(nseg)
    if nseg >= 3:
        f = np.fft.rfftfreq(seg, 1.0)
        per = np.full_like(f, np.inf)
        per[1:] = 1 / f[1:]
        b1 = (per >= 4) & (per < 10)
        b2 = (per >= 10) & (per <= 40)
        out["wide_rib"] = float(P[b1].max() / max(P[b2].mean(), 1e-30))
        out["wide_rib_peak"] = float(P[b1].max() / max(np.median(P[b1]), 1e-30))
    return out


def vein_tells(I: np.ndarray, valid: np.ndarray, res: dict, chains: list | None = None, step: float = 6.0) -> dict:
    """The widest vein's plateau (v2, priority 4; the v1 review's
    veinprof.py, automated).  The vein: among the wide_paths at least
    max(6 widths, 8 lambda) long, the one with the largest median width x
    median darkness (-G along it), G = ln I minus its 150 px NaN-aware
    Gaussian mean.  Perpendicular profiles of G every `step` px along it
    (+-(2.5 w + 25) px, 0.5 px samples, w its median mask width), each
    re-centred on its darkest point (3 px smoothed, within +-0.5 w), and
    their pointwise median m(t).  Baseline: median of m at |t| in
    [FWHM, FWHM + 15 px] (FWHM first from the profile's outer 15 px, then
    once more from this baseline).  vein_depth = min m - baseline (Np, < 0),
    vein_fwhm (px), vein_edge = mean of the two 10-90 % edge widths (px)."""
    m = res["maps"]
    lam = res["lam"]
    out = _nan_dict(VEIN_KEYS)
    chains = wide_paths(res) if chains is None else chains
    G = _log_minus_bg(np.asarray(I, float), valid)
    best, score = None, -np.inf
    for ch in chains:
        if ch["length"] < max(6 * ch["w_med"], 8 * lam):
            continue
        qi = np.rint(ch["q"]).astype(int)
        qi[:, 0] = np.clip(qi[:, 0], 0, G.shape[0] - 1)
        qi[:, 1] = np.clip(qi[:, 1], 0, G.shape[1] - 1)
        g = G[qi[:, 0], qi[:, 1]]
        g = g[np.isfinite(g)]
        if len(g) < 10:
            continue
        sc = ch["w_med"] * max(-float(np.median(g)), 0.0)
        if sc > score:
            best, score = ch, sc
    if best is None:
        return out
    q, w = best["q"], best["w_med"]
    nr = _normals(q)
    half = 2.5 * w + 25
    t = np.arange(-half, half + 1e-9, 0.5)
    keep_t = np.abs(t) <= half - 0.5 * w - 1
    i0 = int(math.ceil(0.5 * w))
    idx = np.arange(i0, len(q) - i0, int(step))
    if len(idx) < 5:
        return out
    crd = q[idx, None, :] + t[None, :, None] * nr[idx, None, :]
    pr = ndi.map_coordinates(G, [crd[..., 0].ravel(), crd[..., 1].ravel()], order=1, mode="constant",
                             cval=np.nan).reshape(crd.shape[:2])
    good = np.isfinite(pr).all(1)
    pr = pr[good]
    if len(pr) < 5:
        return out
    ps = ndi.gaussian_filter1d(pr, 6.0, axis=1)
    cen = np.abs(t) <= 0.5 * w
    tmin = t[cen][np.argmin(ps[:, cen], axis=1)]
    tk = t[keep_t]
    A = np.array([np.interp(tk + a, t, p) for a, p in zip(tmin, pr)])
    mp = np.median(A, 0)

    def width_at(base):
        d = mp - base
        dep = float(d.min())
        inside = np.flatnonzero(d <= 0.5 * dep)
        return dep, (tk[inside[-1]] - tk[inside[0]]) if len(inside) else _NAN, d
    outer = np.abs(tk) >= tk.max() - 15
    dep, fw, _ = width_at(np.median(mp[outer]))
    if np.isfinite(fw):
        band = (np.abs(tk) >= fw) & (np.abs(tk) <= fw + 15)
        if band.sum() >= 5:
            dep, fw, d = width_at(np.median(mp[band]))
            c = int(np.argmin(d))

            def edge(seq):
                a = np.argmax(seq >= 0.9 * dep) if (seq >= 0.9 * dep).any() else 0
                b = np.argmax(seq >= 0.1 * dep) if (seq >= 0.1 * dep).any() else len(seq) - 1
                return (b - a) * 0.5
            out["vein_depth"] = dep
            out["vein_fwhm"] = float(fw)
            out["vein_edge"] = float(0.5 * (edge(d[:c + 1][::-1]) + edge(d[c:])))
            out["vein_w_mask"] = float(w)
            out["vein_yx"] = [float(x) for x in q[len(q) // 2]]
            out["vein_n_prof"] = int(len(pr))
    return out


def anat_tells(res: dict) -> dict:
    """Pass-through of suite statistics the v2 builders target (stats.analyse):
    along_cv_contrast / along_cv_fwhm (robust CV of contrast / FWHM along the
    linked paths: uniform wires), faint_loop_density (faint-mesh loops per
    1e5 px^2) and faint_loop_density_lam (loops per lambda^2: the scorecard's
    'lam:faint_loop_density'), fwhm_p10 (px: the thinnest lines)."""
    out = {k: float(res.get(k, np.nan)) for k in ("along_cv_contrast", "along_cv_fwhm", "faint_loop_density",
                                                   "fwhm_p10")}
    out["faint_loop_density_lam"] = out["faint_loop_density"] * 1e-5 * float(res["lam"]) ** 2
    return out


def v2_still_tells(I: np.ndarray, valid: np.ndarray | None = None, res: dict | None = None,
                   cont: dict | None = None) -> dict:
    """All v2 still statistics (V2_STILL_KEYS) of one image in DN (NaN = no
    data; for a raw frame pass valid = (I > 0) & (I < 4095))."""
    I = np.asarray(I, np.float64)
    valid = (np.isfinite(I) & (I > 0)) if valid is None else valid
    res = res if res is not None and "maps" in res else st.analyse(I, valid, keep_maps=True)
    cont = cont if cont is not None else continuation_paths(res["maps"]["S"], res["lam"])
    out = {}
    out.update(knot_tells(res, cont))
    out.update(hier_tells(res, cont))
    chains = wide_paths(res)
    out.update(wide_shape_tells(res, chains, _log_image(I, valid)))
    out.update(vein_tells(I, valid, res, chains))
    out.update(anat_tells(res))
    return out


# ====================================================================== v2: frames
def warp_to_average(frame: np.ndarray, mapping, shape, frame_valid: np.ndarray | None = None,
                    rows: int = 128) -> np.ndarray:
    """The frame carried onto the average's grid: out(x) = frame(mapping(x))
    (cubic spline; NaN where the frame has no data, saturation and its
    spline support included).  mapping None = identity (a synthetic frame,
    returned as is with NaN for no data)."""
    Fr = np.asarray(frame, np.float64)
    vf = ((Fr > 0) & (Fr < 4095)) if frame_valid is None else frame_valid
    if mapping is None:
        return np.where(vf, Fr, np.nan)
    Hf, Wf = Fr.shape
    H, W = shape
    fill = np.where(vf, Fr, np.median(Fr[vf]))
    c = ndi.spline_filter(fill, 3, mode="nearest")
    bad = (~vf).astype(np.float64)
    out = np.full((H, W), np.nan)
    xx = np.arange(W, dtype=float)
    for y0 in range(0, H, rows):
        yy = np.arange(y0, min(H, y0 + rows), dtype=float)
        g = np.stack(np.meshgrid(xx, yy), -1).reshape(-1, 2)
        q = mapping(g)
        crd = [q[:, 1], q[:, 0]]
        v = ndi.map_coordinates(c, crd, order=3, mode="nearest", prefilter=False)
        b = ndi.map_coordinates(bad, crd, order=1, mode="constant", cval=1.0) > 0
        b |= (q[:, 0] < 1) | (q[:, 0] > Wf - 2) | (q[:, 1] < 1) | (q[:, 1] > Hf - 2)
        b |= ndi.map_coordinates(ndi.binary_dilation(~vf, iterations=1).astype(float), crd, order=0,
                                 mode="constant", cval=1.0) > 0
        out[y0:y0 + len(yy)] = np.where(b, np.nan, v).reshape(len(yy), W)
    return out


def _model_fwhm(h, s, iters=40):
    """FWHM of the box with Gaussian edges Phi((t + h) / s) - Phi((t - h) / s)."""
    from scipy.special import ndtr
    h, s = np.asarray(h, float), np.asarray(s, float)
    f0 = ndtr(h / s) - ndtr(-h / s)
    lo, hi = np.zeros_like(h), h + 6 * s
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = ndtr((mid + h) / s) - ndtr((mid - h) / s)
        above = fm > 0.5 * f0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return lo + hi


def faint_tells(warped: np.ndarray, res_avg: dict, lo: float = 0.012, hi: float = 0.05, absent: float = 0.3,
                dash: float = 2.0, min_gap: int = 3, min_run: float = 2.0) -> dict:
    """Faint thin lines of the average in its frame (v2, priority 1; the v1
    review's morse2.py).  warped: the frame on the average's grid
    (warp_to_average).  On the average's faint + main skeleton where the
    average's contrast ca = -F (1 px Gaussian) is lo..hi Np: r = cf / ca with
    cf = -F of the frame (1.5 px Gaussian).  faint_absent / faint_dash: share
    with r < absent / r > dash; faint_rcv: robust CV of r; faint_gap_rate:
    along the skeleton's branches, runs of such pixels >= min_run lambda long,
    the gaps (>= min_gap px with r < absent, not at a run's end) per 100
    lambda of run length (dashes separated by gaps: 'Morse code')."""
    m = res_avg["maps"]
    lam = res_avg["lam"]
    out = dict(faint_dash=_NAN, faint_absent=_NAN, faint_rcv=_NAN, faint_gap_rate=_NAN, faint_ratio=_NAN, n_faint=0)
    vf = np.isfinite(warped) & (warped > 0)
    if vf.sum() < 1000:
        return out
    Ff = st.flatten(np.where(vf, warped, np.nan), vf)[0]
    Fa = m["F"]
    Sk = m["faint_S"] | m["S"]
    Fas = ndi.gaussian_filter(Fa, 1.0)
    Ffs = ndi.gaussian_filter(Ff, 1.5)
    ev = st.eroded_valid(m["valid"] & vf, 10)
    ca, cf = -Fas, -Ffs
    sel = Sk & ev & (ca >= lo) & (ca <= hi)
    if sel.sum() < 100:
        return out
    r = cf[sel] / ca[sel]
    med = float(np.median(r))
    out.update(n_faint=int(sel.sum()), faint_ratio=med, faint_absent=float((r < absent).mean()),
               faint_dash=float((r > dash).mean()),
               faint_rcv=float(1.4826 * np.median(np.abs(r - med)) / max(med, 1e-6)))
    net = st.trace_skeleton(Sk, merge_len=0.5 * lam)
    yx = net["yx"]
    rr = cf[yx[:, 0], yx[:, 1]] / np.maximum(ca[yx[:, 0], yx[:, 1]], 1e-9)
    ss = sel[yx[:, 0], yx[:, 1]]
    n_gap, length = 0, 0.0
    for b in net["branches"]:
        lab, nl = ndi.label(ss[b])
        for k in range(1, nl + 1):
            j = b[lab == k]
            if len(j) < min_run * lam:
                continue
            length += len(j)
            ab, na = ndi.label(rr[j] < absent)
            for g in range(1, na + 1):
                gi = np.flatnonzero(ab == g)
                if len(gi) >= min_gap and gi[0] > 0 and gi[-1] < len(j) - 1:
                    n_gap += 1
    if length >= 10 * lam:
        out["faint_gap_rate"] = float(n_gap / (length / lam) * 100)
    return out


def crisp_tells(frame: np.ndarray, average: np.ndarray, mapping=None, res_avg: dict | None = None,
                frame_valid: np.ndarray | None = None, pair: dict | None = None, warped: np.ndarray | None = None,
                n_sites: int = 3000, seed: int = 0, min_run: float = 3.0, fine: float = 2.5) -> dict:
    """The frame crispness set (v2, priority 1): one frame against its own
    average (mapping, frame_valid as frame_average_tells; pair: its
    frame_average_tells result, computed when None; warped: the frame on the
    average's grid, computed when None).

    sharper           1 when the frame's medium vessels are sharper than the
                      average's (frame_average_tells sigma_extra_med > 0), else 0;
                      its mean over frames is the share of frames sharper than
                      their average
    Same vessels, fitted in both images (fit_edges; across-profiles averaged
    over 7 px along, at profile points of the average with FWHM 0.5-2
    lambda, contrast >= 0.03 Np, > 1 lambda from junctions; thin: FWHM < 1
    lambda, medium: 1-2 lambda); depth d = c (2 Phi(h / s) - 1), the fitted
    profile's peak (Np; 0 when the frame's fit moved > 3 px or inverted):
    crisp_c_thin/_med    median d_frame / d_avg (> 1: the frame has more
                         contrast than the average on the same vessel)
    crisp_seen_thin      share of the thin sites with d_frame >= 0.5 d_avg
    crisp_fwhm_thin/_med median FWHM_frame / FWHM_avg (fitted profiles) over
                         the sites seen in the frame
    crisp_edge_med       median edge sigma s_frame / s_avg, medium sites
    frame_c_thin         median d_frame at the thin sites (Np)
    frame_cnr_thin       frame_c_thin / the frame's pixel noise in ln I
                         (stats.noise_sigma of its flattened log image): do the
                         fine vessels stand out of the noise?
    faint_*              faint_tells (faint lines: continuous or dashed)
    Beads along the medium vessels (mask width lambda-2 lambda; runs and noise
    handling as grain_* of frame_average_tells; frame minus average, so the
    static structure cancels):
    bead_med             noise-corrected robust SD of the lumen-centre ln I
                         band-passed along the vessel at fine..lambda px (red-cell
                         aggregate scale)
    bead_med_rel         bead_med / the medium vessels' median contrast
    bead_len             along-vessel correlation length (px) of that signal
                         (high-passed at lambda): lag where its autocovariance,
                         minus the noise's, falls to half (NaN when the signal is
                         < 20 % of the variance)
                         (diagnostics, not compared: bead_sham = bead_med measured
                         beside the vessels, where nothing moves (should be 0:
                         checks the noise model, like grain_sham);
                         bead_signal_share = the signal's share of that variance)
    wide_band_frame      bead_med_rel over the wide vessels (>= 2 lambda): bands
                         across wide lumens ('bamboo', the v1 review's T8)
    """
    from scipy.special import ndtr
    mapping_ = mapping or _identity
    rng = np.random.default_rng(seed)
    A = np.asarray(average, float)
    va = np.isfinite(A) & (A > 0)
    Fr = np.asarray(frame, float)
    vf = ((Fr > 0) & (Fr < 4095)) if frame_valid is None else frame_valid
    res = res_avg if res_avg is not None and "maps" in res_avg else st.analyse(A, va, keep_maps=True)
    pair = pair if pair is not None else frame_average_tells(frame, average, mapping, res, frame_valid)
    m = res["maps"]
    lam = res["lam"]
    S, M = m["S"], m["M"]
    out = _nan_dict(CRISP_KEYS)
    se = pair.get("sigma_extra_med", _NAN)
    out["sharper"] = float(se > 0) if np.isfinite(se) else _NAN
    La, Lf = _log_image(A, va), _log_image(Fr, vf)
    SA, SF = _Spline(La), _Spline(Lf)
    Hf, Wf = Fr.shape
    H, W = A.shape
    vfd = ndi.binary_erosion(vf, iterations=2)

    def frame_ok(gf):
        yi = np.rint(gf[..., 0]).astype(int)
        xi = np.rint(gf[..., 1]).astype(int)
        inside = (yi >= 2) & (yi < Hf - 2) & (xi >= 2) & (xi < Wf - 2)
        return inside & vfd[np.clip(yi, 0, Hf - 1), np.clip(xi, 0, Wf - 1)]

    jz = _junction_zone(S, lam)
    v8 = st.eroded_valid(va, 2 * lam)
    # ---- (a) the same thin and medium vessels fitted in both images
    pr = res["prof"]
    ys, xs = np.nonzero(S)
    pts = np.stack([ys, xs], 1)[pr["idx"]] if len(pr["idx"]) else np.zeros((0, 2), int)
    fw, cc = pr["fwhm"], pr["contrast"]
    sel = (fw >= 0.5 * lam) & (fw < 2 * lam) & (cc >= 0.03)
    if len(pts):
        sel &= ~jz[pts[:, 0], pts[:, 1]] & v8[pts[:, 0], pts[:, 1]]
    idx = np.flatnonzero(sel)
    if len(idx) > n_sites:
        idx = np.sort(rng.choice(idx, n_sites, replace=False))
    out["n_crisp_sites"] = 0
    if len(idx) >= 20:
        P0 = pts[idx].astype(float)
        tan = _tangents(S, P0, r=max(4.0, 0.3 * lam))
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        fwi = fw[idx]
        Tmax = np.maximum(fwi / 2 + 13.0, 1.2 * fwi)
        tt = np.linspace(-1, 1, 81)[None, :] * Tmax[:, None]
        uu = np.arange(-3, 4, 1.0)
        grid = (P0[:, None, None, :] + tt[:, :, None, None] * nrm[:, None, None, :]
                + uu[None, None, :, None] * tan[:, None, None, :])
        gf = _map_yx(mapping_, grid)
        okf = frame_ok(gf).all((1, 2))
        pa = _sample(SA, grid).mean(2)
        pf = _sample(SF, gf).mean(2)
        fa = fit_edges(tt, pa, h0=fwi, s0=np.full(len(idx), 2.0))
        ff = fit_edges(tt, pf, t0=fa["t0"], h0=2 * fa["h"], s0=fa["s"])
        da = fa["c"] * (2 * ndtr(fa["h"] / fa["s"]) - 1)
        df = ff["c"] * (2 * ndtr(ff["h"] / ff["s"]) - 1)
        lost = (np.abs(ff["t0"] - fa["t0"]) > 3) | (ff["c"] <= 0)
        df = np.where(lost, 0.0, df)
        good = (okf & (fa["c"] > 0.02) & (da > 0.02) & (fa["rms"] < 0.25 * da) & (fa["s"] > 0.3)
                & (fa["s"] < 0.6 * fwi + 2) & (np.abs(fa["t0"]) < 0.5 * fwi + 2))
        out["n_crisp_sites"] = int(good.sum())
        rd = df / np.maximum(da, 1e-9)
        seen = good & (rd >= 0.5) & (ff["s"] > 0.3) & (ff["s"] < 0.6 * fwi + 2)
        wa, wf = _model_fwhm(fa["h"], fa["s"]), _model_fwhm(ff["h"], ff["s"])
        thin, med = fwi < lam, fwi >= lam
        for tag, cl in (("thin", thin), ("med", med)):
            g = good & cl
            if g.sum() >= 20:
                out[f"crisp_c_{tag}"] = float(np.median(rd[g]))
            g = seen & cl
            if g.sum() >= 20:
                out[f"crisp_fwhm_{tag}"] = float(np.median(wf[g] / wa[g]))
                if tag == "med":
                    out["crisp_edge_med"] = float(np.median(ff["s"][g] / fa["s"][g]))
        g = good & thin
        if g.sum() >= 20:
            out["crisp_seen_thin"] = float((rd[g] >= 0.5).mean())
            out["frame_c_thin"] = float(np.median(df[g]))
            Ffl = st.flatten(np.where(vf, Fr, np.nan), vf)[0]
            nz = st.noise_sigma(np.where(st.eroded_valid(vf, 3), Ffl, 0.0))
            out["frame_noise"] = nz
            out["frame_cnr_thin"] = float(out["frame_c_thin"] / max(nz, 1e-9))
    # ---- (b) beads along medium vessels, bands across wide ones
    Mw = 2 * ndi.distance_transform_edt(M)
    sd_f, _ = noise_model(Fr, vf)
    sd_a, _ = noise_model(np.where(va, A, 0.0), va)
    NF = _Spline(rng.standard_normal((Hf, Wf)))
    NA = _Spline(rng.standard_normal((H, W)))
    cut = int(round(0.5 * lam))
    nlag = int(math.ceil(2 * lam))

    def hp(x, lo_, hi_):
        y = (ndi.gaussian_filter1d(x, lo_, mode="nearest") if lo_ > 0 else x) - ndi.gaussian_filter1d(x, hi_, mode="nearest")
        return y[cut:len(y) - cut]

    acc = {"med": dict(d=[], nd=[], c=[], C=np.zeros(nlag + 1), Cn=np.zeros(nlag + 1), N=np.zeros(nlag + 1),
                       sd=[], snd=[]),
           "wide": dict(d=[], nd=[], c=[])}
    for qq in m.get("paths", []):
        qq = st.resample_path(np.asarray(qq, float))
        if len(qq) < min_run * lam:
            continue
        qi = np.rint(qq).astype(int)
        qi[:, 0] = np.clip(qi[:, 0], 0, H - 1)
        qi[:, 1] = np.clip(qi[:, 1], 0, W - 1)
        w = ndi.gaussian_filter1d(Mw[qi[:, 0], qi[:, 1]], 2.0, mode="nearest")
        base = ~jz[qi[:, 0], qi[:, 1]] & v8[qi[:, 0], qi[:, 1]]
        for cls, okc in (("med", base & (w >= lam) & (w < 2 * lam)), ("wide", base & (w >= 2 * lam))):
            lab, nl = ndi.label(okc)
            for rr_ in range(1, nl + 1):
                j = np.flatnonzero(lab == rr_)
                if len(j) < min_run * lam:
                    continue
                q, wj = qq[j], w[j]
                nr = _normals(ndi.gaussian_filter1d(q, 2.0, axis=0, mode="nearest"))
                tt = np.linspace(-1, 1, 7)[None, :] * np.maximum(1.0, 0.25 * wj)[:, None]
                g = q[:, None, :] + tt[:, :, None] * nr[:, None, :]
                gf = _map_yx(mapping_, g)
                okr = frame_ok(gf).all(1)
                lb, nb = ndi.label(okr)
                if nb == 0:
                    continue
                keep = np.flatnonzero(lb == 1 + int(np.argmax(np.bincount(lb[lb > 0])[1:])))
                if len(keep) < min_run * lam:
                    continue
                q, wj, nr, g, gf = q[keep], wj[keep], nr[keep], g[keep], gf[keep]
                ca, cf = _sample(SA, g).mean(1), _sample(SF, gf).mean(1)
                If, Ia = np.exp(_sample(Lf, gf, 1)), np.exp(_sample(La, g, 1))
                nf = (_sample(NF, gf) * sd_f(If) / If).mean(1)
                na = (_sample(NA, g) * sd_a(Ia) / Ia).mean(1)
                a_ = acc[cls]
                a_["d"].append(_rsd(hp(cf - ca, fine, lam)))
                a_["nd"].append(_rsd(hp(nf - na, fine, lam)))
                side = (0.5 * wj + 0.5 * lam)[:, None] * np.array([-1.0, 1.0])[None, :]
                bgv = _sample(SA, q[:, None, :] + side[:, :, None] * nr[:, None, :]).max(1)
                a_["c"].append(float(np.median(bgv - ca)))
                if cls == "med":
                    gsh = g + (0.5 * wj + lam)[:, None, None] * nr[:, None, :]   # sham: beside the vessel
                    gshf = _map_yx(mapping_, gsh)
                    if frame_ok(gshf).all():
                        Ish, Iash = np.exp(_sample(Lf, gshf, 1)), np.exp(_sample(La, gsh, 1))
                        a_["sd"].append(_rsd(hp(_sample(SF, gshf).mean(1) - _sample(SA, gsh).mean(1), fine, lam)))
                        a_["snd"].append(_rsd(hp((_sample(NF, gshf) * sd_f(Ish) / Ish).mean(1)
                                                 - (_sample(NA, gsh) * sd_a(Iash) / Iash).mean(1), fine, lam)))
                    d0, n0 = hp(cf - ca, 0.0, lam), hp(nf - na, 0.0, lam)
                    for lg in range(min(nlag, len(d0) - 1) + 1):
                        a_["C"][lg] += float((d0[:len(d0) - lg] * d0[lg:]).sum())
                        a_["Cn"][lg] += float((n0[:len(n0) - lg] * n0[lg:]).sum())
                        a_["N"][lg] += len(d0) - lg

    def sub(d, nd):
        if not d:
            return _NAN
        return float(math.sqrt(max(np.median(d) ** 2 - np.median(nd) ** 2, 0.0)))
    a_ = acc["med"]
    out["n_bead_runs"] = len(a_["c"])
    if len(a_["c"]) >= 3:
        out["bead_med"] = sub(a_["d"], a_["nd"])
        out["bead_med_rel"] = out["bead_med"] / max(float(np.median(a_["c"])), 1e-6)
        if len(a_["sd"]) >= 3:
            out["bead_sham"] = sub(a_["sd"], a_["snd"])
        ok = a_["N"] > 0
        Cs = np.where(ok, (a_["C"] - a_["Cn"]) / np.maximum(a_["N"], 1), 0.0)
        Cd0 = a_["C"][0] / max(a_["N"][0], 1)
        out["bead_signal_share"] = float(Cs[0] / Cd0) if Cd0 > 0 else _NAN
        if ok[0] and Cs[0] >= 0.2 * Cd0 and Cs[0] > 0:
            rho = Cs / Cs[0]
            below = np.flatnonzero(rho <= 0.5)
            if len(below):
                k = int(below[0])
                out["bead_len"] = float(k - 1 + (rho[k - 1] - 0.5) / max(rho[k - 1] - rho[k], 1e-9))
    a_ = acc["wide"]
    out["n_band_runs"] = len(a_["c"])
    if len(a_["c"]) >= 2:
        out["wide_band_frame"] = sub(a_["d"], a_["nd"]) / max(float(np.median(a_["c"])), 1e-6)
    # ---- (c) faint lines
    wp = warped if warped is not None else warp_to_average(Fr, mapping, A.shape, vf)
    out.update(faint_tells(wp, res))
    return out


def select_frames(fields_path, n: int = 8) -> list:
    """n frame indices evenly spaced over the used frames of a stabilized burst."""
    fs = _FieldSet()(str(fields_path))
    used = sorted(int(i) for i in fs.frame_index if fs.is_used(int(i)))
    return [used[int((k + 0.5) * len(used) / n)] for k in range(n)] if used else []


def frame_record(frame: np.ndarray, average: np.ndarray, mapping=None, res_avg: dict | None = None) -> dict:
    """Everything v2 measures on one frame: frame_average_tells and
    crisp_tells against its average, and still_tells (v0-v2) of the frame
    itself (under 'still'; FRAME_STILL_KEYS are the ones compared)."""
    A = np.asarray(average, float)
    va = np.isfinite(A) & (A > 0)
    res = res_avg if res_avg is not None and "maps" in res_avg else st.analyse(A, va, keep_maps=True)
    Fr = np.asarray(frame, float)
    vf = (Fr > 0) & (Fr < 4095)
    pair = frame_average_tells(Fr, A, mapping, res)
    rec = dict(pair)
    rec.update(crisp_tells(Fr, A, mapping, res, pair=pair))
    rs = st.analyse(np.where(vf, Fr, np.nan), vf, keep_maps=True)
    sti = still_tells(np.where(vf, Fr, np.nan), vf, rs)
    rec["still"] = sti
    return rec


# ====================================================================== reference
def _burst_speed(still_path) -> dict:
    """Frame index -> eye speed (px/s) from the burst's transforms.csv."""
    import csv
    tp = Path(still_path).parent / "transforms.csv"
    if not tp.is_file():
        return {}
    with open(tp, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["time_s"]) for r in rows])
    dx = np.array([float(r["dx_px"]) for r in rows])
    dy = np.array([float(r["dy_px"]) for r in rows])
    v = np.hypot(np.gradient(dx, t), np.gradient(dy, t))
    return {int(r["index"]): float(s) for r, s in zip(rows, v)}


def build_real(root=None, out=DATA_PATH, log=print, pair_frames=None, parts=("stills", "frames", "pairs"),
               bursts=PAIR_BURSTS, base: dict | None = None) -> dict:
    """The real reference values (about 15 min for everything; parts and
    bursts select a subset, base a document to update):

    stills  still_tells (v0-v2) of the 9 reference stills
    frames  frame_tells (v0) of 3 raw frames of 15-50-52 and 2 of 15-31-37
    pairs   frame_record (frame_average_tells, crisp_tells and still_tells of
            the raw frame itself, under 'still') of 8 raw frames of each burst
            in `bursts`, carried onto its non-rigid average through the
            stabilization's own fields, with each frame's speed
            (transforms.csv).  Frames: pair_frames (a dict burst -> indices;
            a tuple applies to 15-50-52), else PAIR_FRAMES for 15-50-52 and
            select_frames for the others."""
    import tifffile
    from . import realism as RM
    root = Path(root) if root else RM.find_data_root()
    doc = dict(base) if base else {}
    if "stills" in parts:
        stills = {}
        for burst, method in RM.REFERENCE_STILLS:
            I = tifffile.imread(RM.still_path(root, burst, method)).astype(np.float64)
            stills[burst] = still_tells(I)
            log(burst, {k: round(v, 4) for k, v in stills[burst].items() if isinstance(v, float)})
        doc["stills"] = stills
    if "frames" in parts:
        frames = {}
        for burst, method, idx in (("09-16_15-50-52", "nonrigid", (35, 70, 104)),
                                   ("09-16_15-31-37", "nonrigid", (40, 119))):
            avg = tifffile.imread(RM.still_path(root, burst, method)).astype(np.float64)
            va = np.isfinite(avg) & (avg > 0)
            res = st.analyse(avg, va, keep_maps=True)
            for f in idx:
                fr = tifffile.imread(root / "reference_data" / f"burst_2026-{burst}" / f"frame_{f:06d}.tif")
                frames[f"{burst} f{f}"] = frame_tells(fr, avg, res)
                log(burst, f, frames[f"{burst} f{f}"])
        doc["frames"] = frames
    if "pairs" in parts:
        pairs = dict(doc.get("frame_pairs", {}))
        for burst, method in bursts:
            sp = RM.still_path(root, burst, method)
            avg = tifffile.imread(sp).astype(np.float64)
            va = np.isfinite(avg) & (avg > 0)
            res = st.analyse(avg, va, keep_maps=True)
            speed = _burst_speed(sp)
            if isinstance(pair_frames, dict) and burst in pair_frames:
                idx = pair_frames[burst]
            elif pair_frames is not None and not isinstance(pair_frames, dict) and burst == PAIR_BURST[0]:
                idx = pair_frames
            else:
                idx = PAIR_FRAMES if burst == PAIR_BURST[0] else select_frames(sp.parent / "fields.npz")
            for f in idx:
                fr = tifffile.imread(root / "reference_data" / f"burst_2026-{burst}" / f"frame_{f:06d}.tif")
                o = frame_record(fr.astype(np.float64), avg, fields_mapping(sp.parent / "fields.npz", f), res)
                o["speed_px_s"] = speed.get(f, _NAN)
                o["burst"] = burst
                pairs[f"{burst} f{f}"] = o
                log(burst, "pair", f, {k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
        doc["frame_pairs"] = pairs
    doc.update(format="vesselscene real_tells v3", dense_site=list(DENSE_SITE),
               pair_bursts=[b for b, _ in PAIR_BURSTS],
               keys=dict(v0=list(V0_KEYS), layout=list(LAYOUT_KEYS), pairs=list(PAIR_KEYS), wide=list(WIDE_KEYS),
                         focus=list(FOCUS_KEYS), frame=list(FRAME_KEYS), frame_pair=list(PAIR_FRAME_KEYS),
                         v2_still=list(V2_STILL_KEYS), crisp=list(CRISP_KEYS), frame_still=list(FRAME_STILL_KEYS)))
    if "stills" in doc:
        doc["summary"] = summarise(doc)
    if out:
        Path(out).write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8")
    return doc


def merge_real(*docs, out=DATA_PATH) -> dict:
    """One reference document from partial build_real documents (later
    ones win; frame pairs are united), summarised."""
    doc = {}
    pairs = {}
    for d in docs:
        pairs.update(d.get("frame_pairs", {}))
        doc.update({k: v for k, v in d.items() if k != "frame_pairs"})
    doc["frame_pairs"] = pairs
    doc["summary"] = summarise(doc)
    if out:
        Path(out).write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8")
    return doc


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def summarise(doc: dict) -> dict:
    """Per key: mean, SD, min, max over all real stills and over the dense
    site; for the frame-pair keys (PAIR_FRAME_KEYS, CRISP_KEYS and, as
    'frame:<key>', FRAME_STILL_KEYS of the raw frames) over all frame pairs
    ('all'), over those of 15-50-52 ('dense') and per burst ('by_burst')."""
    def agg(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        if not len(v):
            return dict(mean=_NAN, sd=_NAN, min=_NAN, max=_NAN, n=0)
        return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else _NAN,
                    min=float(v.min()), max=float(v.max()), n=int(len(v)))
    out = {}
    stills = doc["stills"]
    dense = [n for n in doc.get("dense_site", DENSE_SITE) if n in stills]
    for k in STILL_KEYS:
        out[k] = dict(all=agg(r.get(k) for r in stills.values()), dense=agg(stills[n].get(k) for n in dense))
    for k in FRAME_KEYS:
        out[k] = dict(all=agg(r.get(k) for r in doc.get("frames", {}).values()),
                      dense=agg(v.get(k) for n, v in doc.get("frames", {}).items() if n.startswith("09-16_15-50-52")))
    fp = doc.get("frame_pairs", {})
    bursts = sorted({n.split(" ")[0] for n in fp})

    def by(get):
        return dict(all=agg(get(r) for r in fp.values()),
                    dense=agg(get(r) for n, r in fp.items() if n.startswith(PAIR_BURST[0])),
                    by_burst={b: agg(get(r) for n, r in fp.items() if n.startswith(b)) for b in bursts})
    for k in PAIR_FRAME_KEYS + CRISP_KEYS:
        out[k] = by(lambda r, k=k: r.get(k))
    for k in FRAME_STILL_KEYS:
        out["frame:" + k] = by(lambda r, k=k: r.get("still", {}).get(k))
    return out


def load_real(path=DATA_PATH) -> dict:
    return json.loads(Path(path).read_text())


def compare(syn_stills: list, syn_frames: list = (), real: dict | None = None, syn_pairs: list = (),
            keys: tuple | None = None, syn_frame_stills: list = ()) -> list:
    """Rows (key, real pooled mean +- sd [min, max], dense-site mean +- sd
    [min, max], synthetic mean [min, max], z against pooled and dense) for
    the still tells, the frame tells (syn_frames: frame_tells dicts), the
    frame-average and crispness tells (syn_pairs: frame_average_tells and/or
    crisp_tells dicts, e.g. frame_record) and the still tells of the frames
    themselves (syn_frame_stills: still_tells dicts of synthetic frames;
    keys FRAME_STILL_KEYS, reported as 'frame:<key>').  keys limits the still
    keys (default: those present in the reference).  For the frame-pair rows
    'pooled' is every frame of the three bursts, 'dense' the frames of
    15-50-52, and `separates` is judged against the pooled frames."""
    real = real or load_real()
    rows = []
    stills = real["stills"]
    first = next(iter(stills.values()))
    for key in (keys or STILL_KEYS):
        if key not in first:
            continue
        allv = np.array([r.get(key, np.nan) for r in stills.values()], float)
        den = np.array([stills[n].get(key, np.nan) for n in real["dense_site"]], float)
        sv = np.array([s.get(key, np.nan) for s in syn_stills], float)
        rows.append(_row(key, "still", allv, den, sv))
    if syn_frames:
        for key in FRAME_KEYS:
            allv = np.array([r[key] for r in real["frames"].values()], float)
            den = np.array([v[key] for n, v in real["frames"].items() if n.startswith("09-16_15-50-52")], float)
            sv = np.array([s[key] for s in syn_frames], float)
            rows.append(_row(key, "frame", allv, den, sv))
    fp = real.get("frame_pairs", {})
    if syn_pairs and fp:
        for key in PAIR_FRAME_KEYS + CRISP_KEYS:
            if not any(key in s for s in syn_pairs):
                continue
            allv = np.array([r.get(key, np.nan) for r in fp.values()], float)
            den = np.array([r.get(key, np.nan) for n, r in fp.items() if n.startswith(PAIR_BURST[0])], float)
            sv = np.array([s.get(key, np.nan) for s in syn_pairs], float)
            rows.append(_row(key, "frame-average", allv, den, sv, sep_ref="all"))
    if syn_frame_stills and fp:
        for key in FRAME_STILL_KEYS:
            allv = np.array([r.get("still", {}).get(key, np.nan) for r in fp.values()], float)
            den = np.array([r.get("still", {}).get(key, np.nan) for n, r in fp.items() if n.startswith(PAIR_BURST[0])],
                           float)
            sv = np.array([s.get(key, np.nan) for s in syn_frame_stills], float)
            rows.append(_row("frame:" + key, "frame-still", allv, den, sv, sep_ref="all"))
    return rows


def _row(key, kind, allv, den, sv, sep_ref="dense"):
    allv, den, sv = allv[np.isfinite(allv)], den[np.isfinite(den)], sv[np.isfinite(sv)]

    def z(ref):
        if len(ref) < 2 or not len(sv):
            return _NAN
        sd = ref.std(ddof=1)
        return float((sv.mean() - ref.mean()) / sd) if sd > 0 else _NAN
    return dict(key=key, kind=kind, gap=GAPS.get(key, 0), priority=PRIORITY.get(key.replace("frame:", ""), 0),
                real_mean=float(allv.mean()) if len(allv) else _NAN,
                real_sd=float(allv.std(ddof=1)) if len(allv) > 1 else _NAN,
                real_min=float(allv.min()) if len(allv) else _NAN, real_max=float(allv.max()) if len(allv) else _NAN,
                dense_mean=float(den.mean()) if len(den) else _NAN,
                dense_sd=float(den.std(ddof=1)) if len(den) > 1 else _NAN,
                dense_min=float(den.min()) if len(den) else _NAN, dense_max=float(den.max()) if len(den) else _NAN,
                syn_mean=float(sv.mean()) if len(sv) else _NAN, syn_min=float(sv.min()) if len(sv) else _NAN,
                syn_max=float(sv.max()) if len(sv) else _NAN, z_pooled=z(allv), z_dense=z(den),
                in_dense=bool(len(den) and len(sv) and den.min() <= sv.mean() <= den.max()),
                separates=separates(allv if sep_ref == "all" else den, sv))


def separates(ref, sv, z_min=3.0) -> str:
    """'yes' when the synthetic mean is more than z_min reference SDs away AND
    the synthetic range [min, max] does not overlap the reference range;
    'partly' when only one of the two holds; 'no' otherwise."""
    ref, sv = np.asarray(ref, float), np.asarray(sv, float)
    ref, sv = ref[np.isfinite(ref)], sv[np.isfinite(sv)]
    if len(ref) < 2 or not len(sv):
        return "n/a"
    sd = ref.std(ddof=1)
    zz = abs(sv.mean() - ref.mean()) / sd if sd > 0 else (np.inf if sv.mean() != ref.mean() else 0.0)
    apart = sv.max() < ref.min() or sv.min() > ref.max()
    return "yes" if (zz > z_min and apart) else ("partly" if (zz > z_min or apart) else "no")


def markdown(rows) -> str:
    L = ["| tell | kind | real pooled mean [min, max] | dense site mean [min, max] | synthetic mean [min, max] | "
         "z pooled | z dense | in dense range |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['key']} | {r['kind']} | {r['real_mean']:.4g} [{r['real_min']:.4g}, {r['real_max']:.4g}] | "
                 f"{r['dense_mean']:.4g} [{r['dense_min']:.4g}, {r['dense_max']:.4g}] | {r['syn_mean']:.4g} "
                 f"[{r['syn_min']:.4g}, {r['syn_max']:.4g}] | {r['z_pooled']:+.1f} | {r['z_dense']:+.1f} | "
                 f"{'yes' if r['in_dense'] else 'no'} |")
    return "\n".join(L)
