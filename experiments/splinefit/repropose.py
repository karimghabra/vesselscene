"""Re-proposal from the prediction error: arms the fitted network does not explain (attention where error remains).

After the joint fit and the MDL prune, the render R of the pruned network is the top-down prediction and the
positive residual D = max(OD - R, 0) (OD: the cleaned target of the residual rule, never a vesselness map) is
what is left unexplained. The batch-3 recordings located it: thin vessels running beside and under wide veins,
a crossing's missing 4th arm (typed a T), a compound's missing third vessel (typed a crossing). Junction typing
is a recall problem at the junctions.

D is passed once more through neuromimetic stages 3-6 (simple cells, surround, association field, readout;
`_channel_traces`) at re_k x the readout thresholds, merged against the network's own edges (its vessels count
as the coarsest channel, so the misfit echoes along their walls are not new vessels: `merge_channels`). A new
trace is kept only if it is NOVEL (at least neuromimetic's re_novel x min_len px of it outside the render's
support, halo included, as retarget's: the misfit echoes along a fitted vessel's walls and halo stay inside)
and an ARM: one of its ends lies within reach (r + s + gap_px) of an existing edge's
centreline. That end is attached to the network: to the edge's nearest node when one lies within max(snap_px, r + s),
else to a new node splitting the edge there (a T or fork on the vessel); a trace passing over a vessel stays
an overlap (a crossing). Its r, s, a come from cross-sections of D (`fit_profiles`, the blurred box converted
to vesselmap's cylinder chord as in proposals). The new arms then join the joint fit (40 iterations) and pay
the same MDL test as every other edge: no extra look-elsewhere cost (neuromimetic's re_mdl = 20 rejected every
arm on whole images, the true ones included). When every arm is rejected the network reverts to its state
before the re-proposal.

Only image-derived data are read (OD, the render, stage 1's masks).
"""
from __future__ import annotations

import numpy as np

from . import use_vesselmap
from .proposals import box_to_cyl

use_vesselmap()


def _edge_samples(net):
    """All 1 px samples of the network: (xy, r + s, edge id) stacked."""
    xy, w, eid = [], [], []
    for k in sorted(net.edges):
        smp = net.sample(k, 1.0)
        xy.append(smp["xy"])
        w.append(smp["r"] + smp["s"])
        eid.append(np.full(len(smp["xy"]), k))
    return np.concatenate(xy), np.concatenate(w), np.concatenate(eid)


def arms(net, OD: np.ndarray, R: np.ndarray, s1: dict, sig: np.ndarray, re_k: float = 1.3, gap_px: float = 6.0,
         snap_px: float = 4.0, sup_thr: float = 0.5, sup_dil: int = 3) -> list:
    """Add the residual's arms to net (in place); sig: the local noise sigma. Returns the new edge ids."""
    import cv2
    from experiments.neuromimetic import neuromimetic as N
    from scipy.spatial import cKDTree
    if not net.edges:
        return []
    cfg = N.DEFAULT
    ok = s1["ok"]
    k = 2 * sup_dil + 1                                 # the render's support (halo included), as retarget's
    sup = cv2.dilate((R > sup_thr * sig).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    D = np.where(ok, np.maximum(OD - R, 0.0), 0.0).astype(np.float32)
    new = N._channel_traces(D, s1["bg"], ok, cfg, t_high=cfg.t_high * re_k, t_low=cfg.t_low * re_k)
    if not new:
        return []
    exy, ew, eeid = _edge_samples(net)
    acc = []
    for k in sorted(net.edges):                         # the network's vessels as the coarsest channel
        smp = net.sample(k, 1.0)
        acc.append(dict(xy=smp["xy"], c=np.full(len(smp["xy"]), np.inf), w=2.0 * (smp["r"] + smp["s"]),
                        chan=len(cfg.channels)))
    merged = N.merge_channels(new, cfg, accepted=acc, shape=OD.shape)[len(acc):]
    tree = cKDTree(exy)
    added = []
    for t in merged:
        xy = np.asarray(t["xy"], float)
        if len(xy) < 2:
            continue
        near = []
        for end in (0, 1):
            d, i = tree.query(xy[0] if end == 0 else xy[-1])
            near.append(d <= ew[i] + gap_px)
        if not any(near):
            continue                                    # not an arm of the network
        ix = np.clip(np.round(xy).astype(int), 0, [OD.shape[1] - 1, OD.shape[0] - 1])
        if int((~sup[ix[:, 1], ix[:, 0]]).sum()) < cfg.re_novel * cfg.min_len:
            continue                                    # inside the render's support: a misfit echo, not novel
        a, r, s_ = N.fit_profiles(D, xy, np.asarray(t["w"], float))
        rc, sc, ac = box_to_cyl(a, r, s_)
        ends = [None, None]
        for end in (0, 1):
            if not near[end]:
                continue
            d, i = tree.query(xy[0] if end == 0 else xy[-1])
            eid, q = int(eeid[i]), exy[i]
            e = net.edges[eid]
            cand = [n for n in (e.u, e.v, *e.through) if n in net.nodes]
            dn = [float(np.hypot(*(net.nodes[n].xy - q))) for n in cand]
            if cand and min(dn) <= max(snap_px, ew[i]):     # within the parent's lumen: the same junction
                ends[end] = cand[int(np.argmin(dn))]
            else:
                ends[end] = net.split_edge(eid, q)
                exy, ew, eeid = _edge_samples(net)      # the split edge's ids changed
                tree = cKDTree(exy)
        if ends[0] is None and ends[1] is None or ends[0] == ends[1]:
            continue
        # the arm runs on to its node (the residual stops at the parent's wall)
        if ends[0] is not None:
            xy = np.vstack([net.nodes[ends[0]].xy, xy])
            rc, sc, ac = (np.r_[v[0], v] for v in (rc, sc, ac))
        if ends[1] is not None:
            xy = np.vstack([xy, net.nodes[ends[1]].xy])
            rc, sc, ac = (np.r_[v, v[-1]] for v in (rc, sc, ac))
        if float(np.hypot(*np.diff(xy, axis=0).T).sum()) < cfg.min_len:
            continue
        wd = float(np.median(rc) + np.median(sc))
        eid = net.add_edge_dense(xy, rc, sc, ac, u=ends[0], v=ends[1],
                                 info=dict(band=[0.0, max(1.5, wd)], new=True))
        added.append(eid)
        exy, ew, eeid = _edge_samples(net)
        tree = cKDTree(exy)
    return added
