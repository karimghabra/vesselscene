"""The coarse ('magnocellular') channel: deep, blurred vessels the stage-1 background absorbs.

Stage 1's vesselness mask is a band-CNR threshold at fine scales, and its background is a masked Gaussian mean
at 8 px. A deep vessel, blurred to 10-30 px, has a low band CNR, so it stays outside the mask and B is fitted
ON it: its OD is lost from the target before any fit (tier2diag: target_keep 0.64-0.82 on s004 and pathologic,
in every width class, the very wide class 41-71 % of the squared residual).

One orientation-selective coarse detector is used twice, as the magnocellular pathway would be (low spatial
frequency, high contrast sensitivity, coarse orientation):

(a) the background: ridges of log I at sigma 4-16 px (Hessian, a dark line: largest eigenvalue > 0 and
    elongated, |l2| < aniso l1, response > k robust sd above the median) that are NOVEL (mostly outside
    stage 1's mask; a known vessel's coarse response would only move B's support away from it, E18), dilated
    by sigma, are OR-ed into
    stage 1's mask; B is re-estimated on the mask's NEGATIVE only (the residual rule), OD = B - log I. This
    target is used to read out (b), and its mask by the retarget before the final fit (geometry frozen, E3);
    the joint fit and the MDL test stay on stage 1's target (fitting the fuller target from the start moved
    the pathologic centrelines: pilot pos 0.680 -> 0.597).
(b) the causes: without them a fuller target only drags the sharp vessels' centrelines toward the deep OD
    (E1; the oracle-OD lesion moved pathologic pos 0.66 -> 0.16). The coarse channel of neuromimetic stages
    3-6 (`_channel_traces`, scales 8.5-24 px) reads the new target out; a trace that is NOVEL against the
    proposed network (at least novel_px of it outside every proposed edge's footprint, r + s + 3 px) is added
    as a DEEP edge with free ends: no node where it passes under a sharp vessel (a crossing of different
    depths is an overlap). Its profile comes from cross-sections of the new target (fit_profiles, box ->
    cylinder). The joint fit and the MDL / wide-edge / flank tests then judge it like every other edge.
    Provisional (pipeline Config.deep, off): on tier 2 it adds +0.0032 +- 0.0026 on top of (a), mostly on one
    image (batch 6); (a) alone is confirmed (+0.0014 +- 0.0005, 9 of 10 images up).

Only image-derived data are read (log I, valid, stage 1's masks, the proposed network).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import use_vesselmap
from .proposals import box_to_cyl

use_vesselmap()


def ridge_mask(L: np.ndarray, ok: np.ndarray, known: np.ndarray, sigmas=(4, 6, 9, 13, 16), k: float = 3.0,
               aniso: float = 0.35, novel: float = 0.5) -> np.ndarray:
    """Coarse dark-line mask of the log image (module docstring, a), dilated by each hit's sigma. Only NOVEL
    ridges count: a connected ridge whose pixels lie in `known` (stage 1's mask) for at least `novel` of them
    is the coarse response of a vessel the fine channels already found, and leaves B alone."""
    import cv2
    from scipy import ndimage as ndi
    Lf = np.where(ok, L, np.median(L[ok])).astype(np.float64)
    m = np.zeros(L.shape, bool)
    for s in sigmas:
        hxx = ndi.gaussian_filter(Lf, s, order=(0, 2))
        hyy = ndi.gaussian_filter(Lf, s, order=(2, 0))
        hxy = ndi.gaussian_filter(Lf, s, order=(1, 1))
        tr, det = hxx + hyy, hxx * hyy - hxy ** 2
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc                  # dark line in L: l1 > 0 across it
        resp = s * s * np.maximum(l1, 0) * (np.abs(l2) < aniso * np.abs(l1))
        med = float(np.median(resp[ok]))
        sd = 1.4826 * float(np.median(np.abs(resp[ok] - med))) + 1e-12
        hit = (resp > k * sd + med) & ok
        n, lab = cv2.connectedComponents(hit.astype(np.uint8), connectivity=8)
        if n > 1:
            tot = np.bincount(lab.ravel(), minlength=n)
            inside = np.bincount(lab.ravel(), weights=known.ravel().astype(float), minlength=n)
            fresh = inside < novel * np.maximum(tot, 1)
            fresh[0] = False
            hit = fresh[lab]
        r = int(round(s))
        m |= cv2.dilate(hit.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return m


def coarse_stage1(s1: dict) -> dict:
    """Stage 1 with the coarse ridges OR-ed into the mask and B re-estimated on its negative (a)."""
    from experiments.neuromimetic import neuromimetic as N
    from scipy import ndimage as ndi
    L, ok = s1["L"], s1["ok"]
    mask = s1["mask"] | ridge_mask(L, ok, s1["mask"])
    bg = ok & ~mask
    B = N._masked_mean(L, bg, N.DEFAULT.bg_sigma)
    OD = np.where(ok, B - L, 0.0).astype(np.float32)
    if not ok.all():
        iy, ix = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        OD = OD[iy, ix]
    return dict(s1, OD=OD, B=B, bg=bg, mask=mask)


def deep_edges(net, s1c: dict, novel_px: float = 20.0, margin: float = 3.0, a_min: float = 0.1) -> list:
    """Add the coarse channel's novel traces of the new target to net as free deep edges (b); returns ids."""
    from experiments.neuromimetic import neuromimetic as N
    from scipy.spatial import cKDTree
    cfg = N.DEFAULT
    el = cfg.elong if isinstance(cfg.elong, tuple) else (cfg.elong,) * len(cfg.channels)
    ccfg = replace(cfg, channels=(cfg.channels[-1],), elong=(el[-1],))
    OD = s1c["OD"]
    traces = N._channel_traces(OD, s1c["bg"], s1c["ok"], ccfg)
    if not traces:
        return []
    acc = []                                                   # the proposed vessels as the coarsest channel:
    for k in sorted(net.edges):                                # a deep trace's run parallel to one (its wall
        smp = net.sample(k, 1.0)                               # echo) is dropped (merge_channels)
        acc.append(dict(xy=smp["xy"], c=np.full(len(smp["xy"]), np.inf), w=2.0 * (smp["r"] + smp["s"]),
                        chan=len(cfg.channels)))
    traces = N.merge_channels(traces, cfg, accepted=acc, shape=OD.shape)[len(acc):]
    if net.edges:
        xy, w = [], []
        for k in sorted(net.edges):
            smp = net.sample(k, 1.0)
            xy.append(smp["xy"])
            w.append(smp["r"] + smp["s"] + margin)
        exy, ew = np.concatenate(xy), np.concatenate(w)
        tree = cKDTree(exy)
    added = []
    for t in traces:
        xy = np.asarray(t["xy"], float)
        if len(xy) < 2:
            continue
        if net.edges:
            d, i = tree.query(xy)
            if int((d > ew[i]).sum()) < novel_px:
                continue                                       # a proposed vessel's own coarse response
        a, r, s_ = N.fit_profiles(OD, xy, np.asarray(t["w"], float))
        rc, sc, ac = box_to_cyl(a, r, s_)
        if float(np.median(ac)) < a_min:
            continue                                           # too faint for a vessel: a background lump
        wd = float(np.median(rc) + np.median(sc))
        added.append(net.add_edge_dense(xy, rc, sc, ac, info=dict(band=[0.0, max(1.5, wd)], deep=True)))
    return added
