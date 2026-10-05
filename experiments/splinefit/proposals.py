"""Neuromimetic proposals -> an initial spline network (vesselmap VesselNetwork).

The neuromimetic annotator (experiments/neuromimetic/neuromimetic.py, `run` with a debug dict) gives traces
(one per followed vessel, running straight through crossings), edges (the traces cut at every junction they
take part in, with j = (junction at start, junction at end), None for a free end), typed junctions and per
point OD profiles (contrast pa, half width pr and blur ps of a blurred box, `fit_profiles`). This module turns
them into the network the renderer fits:

* a junction typed 'crossing' with four incident edge ends is NOT a node: its ends pair into the two
  straightest lines (same trace first), which pass through as continuous edges; the crossing is recorded;
* a junction with two incident ends is a pass-through (a joint) when they turn by less than `joint_turn_deg`;
* every other junction with at least three incident ends is a node; two ends of the same trace that continue
  each other (turn < `cont_turn_deg`) become ONE edge passing through the node (vesselmap's info['through']:
  the parent vessel continues, the branch ends on it), the rest end at the node;
* free ends (no junction, or a junction left with one end) become degree-1 nodes;
* each chain of pieces linked through crossings, joints and through-nodes is one edge: a clamped cubic
  B-spline whose control spacing vesselmap adapts to calibre and curvature (`spacing_for_path`, tolerance
  SPACING_TOL x calibre), with r, s, a
  profile splines from the proposal's cleaned-OD cross-sections, converted from the blurred box to vesselmap's
  blurred cylinder chord (half width x `BOX_TO_CYL`, contrast divided by the cylinder's blurred peak).

Only the proposal (image-derived) is read; nothing here sees a truth file.
"""
from __future__ import annotations

import math

import numpy as np

from . import use_vesselmap

use_vesselmap()
from vesselmap.network import VesselNetwork, pos_spacing_for, spacing_for_path   # noqa: E402

BOX_TO_CYL = 1.2            # a blurred box of half width r ~ a cylinder chord of half width 1.2 r (FWHM / area)
CROSS_TURN_DEG = 45.0       # a crossing's paired ends must continue each other within this turn
CONT_TURN_DEG = 40.0        # a trace continuing through a node within this turn is one edge (parent)
JOINT_TURN_DEG = 60.0       # two ends at a junction left with two: one vessel when they turn less
SPACING_TOL = 0.15          # x calibre (r + s): how closely a spline must follow its trace
END_LOOK_PX = 8.0           # px along an edge for its end direction


def _plen(P):
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


def _out_dir(xy, end, look=END_LOOK_PX):
    """Unit vector pointing out of the polyline at its start (end=0) or end (1)."""
    P = xy if end == 1 else xy[::-1]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    j = int(np.searchsorted(s, s[-1] - min(look, 0.6 * s[-1])))
    v = P[-1] - P[min(j, len(P) - 1)]
    n = math.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _turn(u, v):
    """Turn (deg) of a line entering along -u and leaving along v (u, v both pointing out of a junction)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(-np.dot(u, v))))))


def _cyl_peak(r, s):
    """Peak of vesselmap's blurred cylinder chord (unit centre chord), numpy."""
    from scipy.special import erf
    use_vesselmap()
    from vesselmap.render import PROFILE_C, PROFILE_W
    c, w = PROFILE_C.numpy().astype(float), PROFILE_W.numpy().astype(float)
    r, s = np.asarray(r, float)[..., None], np.asarray(s, float)[..., None]
    return (erf(r * c / (math.sqrt(2.0) * s)) * w).sum(-1)


def box_to_cyl(a, r, s):
    """Blurred-box profile (peak OD a, half width r, blur s) -> vesselmap cylinder parameters (r, s, a)."""
    rc = np.maximum(BOX_TO_CYL * np.asarray(r, float), 0.4)
    s = np.maximum(np.asarray(s, float), 0.65)
    ac = np.maximum(np.asarray(a, float), 1e-3) / np.maximum(_cyl_peak(rc, s), 0.05)
    return rc, s, ac


def _pair_crossing(ends, dirs, traces):
    """Best pairing of four ends into two lines: same-trace pairs first, then the smallest worst turn."""
    best = None
    for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        t = max(_turn(dirs[a], dirs[b]), _turn(dirs[c], dirs[d]))
        same = (traces[a] == traces[b]) + (traces[c] == traces[d])
        key = (-same, t)
        if best is None or key < best[0]:
            best = (key, [(a, b), (c, d)], t)
    return best[1], best[2]


def build_network(prop: dict, shape) -> tuple[VesselNetwork, dict]:
    """The initial network from a neuromimetic debug dict (edges, junctions with pa/pr/ps profiles).

    Returns (net, info) with info = dict(crossings=[dict(x, y, e1, e2)] the proposal's crossings by edge id,
    node_type={nid: neuromimetic junction type}, node_r={nid: junction radius})."""
    edges = [e for e in prop["edges"] if len(e["xy"]) >= 2 and _plen(e["xy"]) > 0.5]
    junctions = prop["junctions"]
    inc = {}                                           # junction -> [(edge index, end)]
    for k, e in enumerate(edges):
        for end in (0, 1):
            ji = e["j"][end]
            if ji is not None:
                inc.setdefault(ji, []).append((k, end))
    link = {}                                          # (k, end) <-> (k2, end2): one vessel passes
    via = {}                                           # (k, end) -> ('crossing' | 'node' | 'joint', junction)
    node_of = {}                                       # junction -> True when it is a node
    crossings = []
    for ji in sorted(inc):
        ends = inc[ji]
        J = junctions[ji]
        dirs = [_out_dir(edges[k]["xy"], end) for k, end in ends]
        trs = [edges[k]["trace"] for k, _ in ends]
        if len(ends) == 4 and J["type"] == "crossing":
            pairs, t = _pair_crossing(ends, dirs, trs)
            if t <= CROSS_TURN_DEG:
                for a, b in pairs:
                    link[ends[a]], link[ends[b]] = ends[b], ends[a]
                    via[ends[a]] = via[ends[b]] = ("crossing", ji)
                crossings.append(dict(ji=ji, x=J["x"], y=J["y"], ends=[ends[p[0]] for p in pairs]))
                continue
        if len(ends) == 2:
            if _turn(dirs[0], dirs[1]) <= JOINT_TURN_DEG and ends[0][0] != ends[1][0]:
                link[ends[0]], link[ends[1]] = ends[1], ends[0]
                via[ends[0]] = via[ends[1]] = ("joint", ji)
                continue
        if len(ends) <= 1:
            continue                                   # a free end
        node_of[ji] = True
        # continuity: same-trace ends that continue each other pass through (greedy, straightest first)
        cand = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                if trs[a] == trs[b] and ends[a][0] != ends[b][0]:
                    t = _turn(dirs[a], dirs[b])
                    if t <= CONT_TURN_DEG:
                        cand.append((t, a, b))
        used = set()
        for t, a, b in sorted(cand):
            if a in used or b in used or len(ends) - len(used) - 2 < 1:
                continue                               # at least one end must remain ending at the node
            used |= {a, b}
            link[ends[a]], link[ends[b]] = ends[b], ends[a]
            via[ends[a]] = via[ends[b]] = ("node", ji)
    # chains: walk from unlinked ends through the links
    seen = np.zeros(len(edges), bool)
    chains = []

    def walk(k, end):
        pieces = []
        while True:
            seen[k] = True
            pieces.append((k, end))                    # traverse edge k starting at `end`
            nxt = (k, 1 - end)
            if nxt not in link:
                return pieces, None
            k2, e2 = link[nxt]
            if seen[k2]:
                return pieces, nxt                     # closed loop
            k, end = k2, e2

    for k in range(len(edges)):
        for end in (0, 1):
            if not seen[k] and (k, end) not in link:
                chains.append(walk(k, end)[0])
    for k in range(len(edges)):                        # loops left: open them at an end of their first edge
        if not seen[k]:
            chains.append(walk(k, 0)[0])
    net = VesselNetwork(shape)
    jnode = {}
    node_type, node_r = {}, {}
    for ji in sorted(node_of):
        J = junctions[ji]
        jnode[ji] = net.add_node(J["x"], J["y"])
        node_type[jnode[ji]] = J["type"]
        node_r[jnode[ji]] = float(J["r"])
    piece_edge = {}                                    # edge index -> network edge id (for crossings)
    for pieces in chains:
        xs, prof, through = [], [], []
        for i, (k, end) in enumerate(pieces):
            e = edges[k]
            sl = slice(None) if end == 0 else slice(None, None, -1)
            xy = np.asarray(e["xy"], float)[sl]
            pa, pr, ps = (np.asarray(e[q], float)[sl] for q in ("pa", "pr", "ps"))
            if i > 0:
                kind, ji = via[(k, end)]
                if kind == "node":
                    through.append((ji, sum(len(x) for x in xs) - 1))
                if np.hypot(*(xy[0] - xs[-1][-1])) < 0.5:
                    xy, pa, pr, ps = xy[1:], pa[1:], pr[1:], ps[1:]
            xs.append(xy)
            prof.append(np.stack([pa, pr, ps], 1))
        xy = np.concatenate(xs)
        P = np.concatenate(prof)
        k0, e0 = pieces[0]
        k1, e1 = pieces[-1][0], 1 - pieces[-1][1]
        j0, j1 = edges[k0]["j"][e0], edges[k1]["j"][e1]
        u = jnode.get(j0) if j0 is not None else None
        v = jnode.get(j1) if j1 is not None else None
        # move the ends onto their nodes along a straight bridge (no kink at the polyline's own end)
        if u is not None:
            q = net.nodes[u].xy
            if np.hypot(*(xy[0] - q)) > 0.5:
                xy, P = np.r_[q[None], xy], np.r_[P[:1], P]
        if v is not None:
            q = net.nodes[v].xy
            if np.hypot(*(xy[-1] - q)) > 0.5:
                xy, P = np.r_[xy, q[None]], np.r_[P, P[-1:]]
        if _plen(xy) < 2.0:
            continue
        rc, sc, ac = box_to_cyl(P[:, 0], P[:, 1], P[:, 2])
        thr = []
        for ji, idx in through:                        # a through node sits on the centreline (cut point)
            if ji not in jnode:
                continue
            n = jnode[ji]
            j = int(np.argmin(np.hypot(*(xy - net.nodes[n].xy).T)))
            net.nodes[n].x, net.nodes[n].y = (float(c) for c in xy[j])
            thr.append(n)
        band = [0.0, float(max(1.5, np.median(rc) + np.median(sc)))]
        info = dict(band=band, source="neuromimetic")
        if thr:
            info["through"] = thr
        # control spacing: the largest (<= vesselmap's calibre bound) whose spline follows the trace within
        # SPACING_TOL x calibre (at least 0.6 px): a wide vessel's trace wobbles by a fraction of its width,
        # and a spline following the wobble folds its nearest-sample render (radial streaks)
        wd = float(np.median(rc) + np.median(sc))
        spacing = spacing_for_path(xy, pos_spacing_for(wd), tol=max(0.6, SPACING_TOL * wd))
        eid = net.add_edge_dense(xy, rc, sc, ac, u=u, v=v, info=info, spacing=spacing)
        for k, _ in pieces:
            piece_edge[k] = eid
    # edges ending at a through node must end on its (moved) position
    for eid in net.edges:
        net.sync_ends(eid)
    cr = []
    for c in crossings:
        es = sorted({piece_edge.get(k) for k, _ in c["ends"]} - {None})
        if len(es) == 2:
            cr.append(dict(x=float(c["x"]), y=float(c["y"]), e1=int(es[0]), e2=int(es[1])))
    # drop nodes nobody uses (junctions whose edges were all too short)
    for n in [n for n in net.nodes if not net.incident(n) and not net.passing(n)]:
        del net.nodes[n]
        net._touch()
    node_type = {n: t for n, t in node_type.items() if n in net.nodes}
    node_r = {n: r for n, r in node_r.items() if n in net.nodes}
    return net, dict(crossings=cr, node_type=node_type, node_r=node_r)
