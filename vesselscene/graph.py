"""The ground truth of a synthetic scene: every vessel a spline, on a flow digraph.

Units.  Anatomy is in capillary diameters (d_c, nominally 6 um).  A scene is
imaged at k pixels per d_c: image pixel coordinates are k * (x, y), pixel
centres at integers, x to the right and y down.  The anatomy extends past the
image (a padded domain), so vessels enter and leave the frame as they do in
the real stills.

The network is a directed graph in the sense of blood flow.  A node is where
vessels meet - a fork (one vessel ends, two or more begin), a confluence (two
or more end, one begins), an anastomosis (several in and out) - or where the
network leaves the padded domain: an inlet (blood enters) or an outlet.  Each
edge is one vessel, from its upstream node u to its downstream node v, so a
vessel ends at every fork and confluence.  A crossing, two vessels
overlapping in the image at different depths, is not a node.

A vessel's centreline is a clamped cubic B-spline in (x, y) from node u to
node v, in vesselmap.spline's basis (so a map can be exported to vesselmap).
Its lumen radius, its depth below the surface and its relative haematocrit
(1 = systemic blood; lower in capillaries) are clamped cubic B-splines over
its normalized arclength tau = s / L in [0, 1].  Depth is a property along
the vessel, like its radius: it only decides how the vessel looks (tissue
above it blurs it and scatters light into its shadow) and which vessel is on
top where two cross.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline

from vesselscene import spline as sp

CLASSES = ("capillary", "arteriole", "venule", "episcleral_artery", "episcleral_vein",
           "anastomosis", "perforator")
SIDES = ("A", "V", "C", "D")          # arterial tree, venous tree, capillary mesh, deep
NODE_KINDS = ("inlet", "outlet", "fork", "confluence", "anastomosis")


# ------------------------------------------------------------------ splines
def _basis(n_ctrl: int, u, deriv: int = 0) -> np.ndarray:
    """(len(u) x n_ctrl) float64 design matrix of vesselmap's clamped uniform
    B-spline basis (vesselmap.spline) at parameters u in [0, 1]."""
    k = sp.degree_for(n_ctrl)
    spl = BSpline(sp._knots(n_ctrl, k), np.eye(n_ctrl), k, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    return np.nan_to_num(spl(np.clip(np.asarray(u, float), 0.0, 1.0)))


def fit_curve(xy: np.ndarray, spacing: float, smooth: float = 1e-4) -> np.ndarray:
    """Control points (n, 2) of a clamped cubic B-spline through a dense
    polyline, with its ends pinned, about `spacing` d_c apart."""
    xy = np.asarray(xy, float)
    L = float(sp.arclength(xy)[-1])
    n = sp.n_ctrl_for_length(L, spacing)
    m = max(4 * n, sp.n_samples_for_length(L, 0.25))
    return sp.fit_ctrl(sp.resample_polyline(xy, m), n, smooth=smooth).astype(float)


def fit_profile(values, n_ctrl: int | None = None) -> np.ndarray:
    """Control values of a profile over tau in [0, 1]: a scalar (constant),
    or values sampled uniformly in tau (fitted, ends pinned)."""
    v = np.atleast_1d(np.asarray(values, float))
    if v.size == 1:
        return np.full(4, float(v[0]))
    n = n_ctrl or int(np.clip(v.size // 4, 4, 64))
    return sp.fit_ctrl(v[:, None], n, smooth=1e-3)[:, 0].astype(float)


def eval_profile(ctrl: np.ndarray, tau) -> np.ndarray:
    return _basis(len(ctrl), tau) @ np.asarray(ctrl, float)


# ------------------------------------------------------------------ objects
@dataclass
class Node:
    nid: int
    xy: np.ndarray                  # (2,) d_c
    depth: float = 0.0              # d_c below the surface
    kind: str = "fork"              # one of NODE_KINDS; see VesselGraph.classify_nodes
    pressure: float | None = None   # boundary pressure (inlets 1, outlets 0) or solved value
    info: dict = field(default_factory=dict)


@dataclass
class Vessel:
    vid: int
    u: int                          # upstream node
    v: int                          # downstream node
    ctrl: np.ndarray                # (n, 2) centreline control points, d_c; ctrl[0] at u, ctrl[-1] at v
    r_ctrl: np.ndarray              # lumen radius over tau, d_c
    z_ctrl: np.ndarray              # depth over tau, d_c
    h_ctrl: np.ndarray              # relative haematocrit over tau (0..1)
    cls: str = "capillary"          # one of CLASSES
    side: str = "C"                 # one of SIDES
    flow: float = 0.0               # volumetric flow u -> v (arbitrary units), >= 0
    info: dict = field(default_factory=dict)

    def length(self, oversample: int = 16) -> float:
        n = len(self.ctrl)
        xy = _basis(n, np.linspace(0, 1, max(64, oversample * n))) @ self.ctrl
        return float(sp.arclength(xy)[-1])

    def sample(self, spacing: float = 0.25, oversample: int = 16) -> dict:
        """Points evenly spaced in arclength (about `spacing` d_c apart):
        s (arclength, d_c), tau (s / L), u (spline parameter), xy, t (unit
        tangent, downstream), kappa (signed curvature, 1/d_c: positive
        turning from +x towards +y), r, z, h."""
        n = len(self.ctrl)
        u0 = np.linspace(0.0, 1.0, max(64, oversample * n))
        s0 = sp.arclength(_basis(n, u0) @ self.ctrl)
        L = float(s0[-1])
        m = max(2, int(math.ceil(L / spacing)) + 1)
        s = np.linspace(0.0, L, m)
        u = np.interp(s, s0, u0) if L > 0 else np.linspace(0, 1, m)
        xy = _basis(n, u) @ self.ctrl
        d1 = _basis(n, u, 1) @ self.ctrl
        d2 = _basis(n, u, 2) @ self.ctrl
        sp1 = np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-12)
        t = d1 / sp1[:, None]
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp1 ** 3
        tau = s / L if L > 0 else np.linspace(0, 1, m)
        return dict(s=s, tau=tau, u=u, xy=xy, t=t, kappa=kappa, L=L,
                    r=eval_profile(self.r_ctrl, tau), z=eval_profile(self.z_ctrl, tau),
                    h=np.clip(eval_profile(self.h_ctrl, tau), 0.0, 1.0))

    def reverse(self):
        """Swap direction (u <-> v), keeping the geometry and profiles."""
        self.u, self.v = self.v, self.u
        self.ctrl = self.ctrl[::-1].copy()
        self.r_ctrl, self.z_ctrl, self.h_ctrl = (c[::-1].copy() for c in
                                                 (self.r_ctrl, self.z_ctrl, self.h_ctrl))


# ------------------------------------------------------------------ the graph
class VesselGraph:
    """Nodes and vessels (edges u -> v in the direction of flow).  `meta`
    holds scene-level facts: 'domain' (x0, y0, x1, y1) in d_c, the padded
    region the anatomy was generated in; 'frame' (x0, y0, x1, y1) in d_c, the
    part the image sees; 'k' (px per d_c); anything else the generator
    records (parameters, seeds)."""

    def __init__(self, meta: dict | None = None):
        self.nodes: dict[int, Node] = {}
        self.vessels: dict[int, Vessel] = {}
        self.meta: dict = dict(meta or {})
        self._next_node = 0
        self._next_vessel = 0

    # ---------------------------------------------------------- building
    def add_node(self, xy, depth: float = 0.0, kind: str = "fork", pressure=None, **info) -> int:
        nid = self._next_node
        self._next_node += 1
        self.nodes[nid] = Node(nid, np.asarray(xy, float).copy(), float(depth), kind, pressure, info)
        return nid

    def add_vessel(self, u: int, v: int, path=None, ctrl=None, radius=1.0, depth=None, hct=1.0,
                   cls: str = "capillary", side: str = "C", spacing: float | None = None,
                   **info) -> int:
        """A vessel from node u to node v.  Its centreline is `ctrl` (control
        points) or fitted to `path` (a dense polyline in d_c, which is moved
        to start exactly at u and end exactly at v).  radius / depth / hct: a
        scalar, or values sampled uniformly along the vessel (tau 0..1).
        Depth defaults to the nodes' depths, linearly between them.
        Control spacing defaults to about 2 radii, at least 1.5 d_c."""
        pu, pv = self.nodes[u].xy, self.nodes[v].xy
        if ctrl is None:
            path = np.asarray(path if path is not None else [pu, pv], float).copy()
            if len(path) == 2:
                path = np.linspace(path[0], path[1], 8)
            # pin the ends onto the nodes, spreading the correction along the path
            w = sp.arclength(path)
            w = w / w[-1] if w[-1] > 0 else np.linspace(0, 1, len(path))
            path += (1 - w)[:, None] * (pu - path[0]) + w[:, None] * (pv - path[-1])
            rmean = float(np.mean(np.atleast_1d(radius)))
            ctrl = fit_curve(path, spacing or max(1.5, 2.0 * rmean))
        ctrl = np.asarray(ctrl, float).copy()
        ctrl[0], ctrl[-1] = pu, pv
        if depth is None:
            depth = np.linspace(self.nodes[u].depth, self.nodes[v].depth, 16)
        vid = self._next_vessel
        self._next_vessel += 1
        self.vessels[vid] = Vessel(vid, u, v, ctrl, fit_profile(radius), fit_profile(depth),
                                   fit_profile(hct), cls, side, 0.0, info)
        return vid

    def remove_vessel(self, vid: int):
        del self.vessels[vid]

    def remove_isolated_nodes(self):
        used = {n for e in self.vessels.values() for n in (e.u, e.v)}
        for nid in [n for n in self.nodes if n not in used]:
            del self.nodes[nid]

    # ---------------------------------------------------------- queries
    def out_vessels(self, nid: int) -> list[int]:
        return [k for k, e in self.vessels.items() if e.u == nid]

    def in_vessels(self, nid: int) -> list[int]:
        return [k for k, e in self.vessels.items() if e.v == nid]

    def degrees(self) -> dict[int, tuple[int, int]]:
        """nid -> (in-degree, out-degree)."""
        deg = {n: [0, 0] for n in self.nodes}
        for e in self.vessels.values():
            deg[e.v][0] += 1
            deg[e.u][1] += 1
        return {n: (a, b) for n, (a, b) in deg.items()}

    def classify_nodes(self):
        """Node kinds from their in/out degrees (inlet, outlet, fork,
        confluence, anastomosis).  A node with one vessel in and one out is
        left as 'anastomosis' and reported by check(): a vessel does not end
        without a fork."""
        for nid, (i, o) in self.degrees().items():
            n = self.nodes[nid]
            n.kind = ("inlet" if i == 0 else "outlet" if o == 0 else "fork" if i == 1 and o >= 2
                      else "confluence" if o == 1 and i >= 2 else "anastomosis")

    def to_digraph(self):
        """networkx.MultiDiGraph: nodes (kind, xy, depth, pressure) and one
        edge per vessel u -> v (key = vid) with its class, side, flow,
        length and mean radius / depth / haematocrit."""
        import networkx as nx
        G = nx.MultiDiGraph(**{k: v for k, v in self.meta.items() if _jsonable(v)})
        for n in self.nodes.values():
            G.add_node(n.nid, kind=n.kind, x=float(n.xy[0]), y=float(n.xy[1]), depth=n.depth,
                       pressure=n.pressure)
        for e in self.vessels.values():
            smp = e.sample(0.5)
            G.add_edge(e.u, e.v, key=e.vid, vid=e.vid, cls=e.cls, side=e.side, flow=e.flow,
                       length=smp["L"], radius=float(smp["r"].mean()),
                       depth=float(smp["z"].mean()), hct=float(smp["h"].mean()))
        return G

    # ---------------------------------------------------------- flow
    def solve_flow(self, orient: bool = True, stagnant: float = 1e-3):
        """Poiseuille flow with the boundary pressures of the nodes that have
        one (Node.pressure set: inlets 1, outlets 0), conductance r^4 / L per
        vessel (harmonic mean of r^4 along it).  Sets every node's pressure
        and every vessel's flow; with orient, reverses the vessels that carry
        flow against their direction, so every edge points downstream.  Ties
        (|dP| tiny) are broken by node id, so the digraph stays acyclic.
        Vessels with |flow| below `stagnant` x the median are flagged
        info['stagnant'].  Returns the flows (vid -> signed flow before
        orienting)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        ids = list(self.nodes)
        idx = {n: i for i, n in enumerate(ids)}
        fixed = {n: self.nodes[n].pressure for n in ids if self.nodes[n].pressure is not None}
        if not fixed:
            raise ValueError("no boundary pressures set")
        g = {}
        for k, e in self.vessels.items():
            smp = e.sample(0.5)
            L = max(smp["L"], 1e-6)
            g[k] = 1.0 / (L * np.mean(1.0 / np.maximum(smp["r"], 1e-3) ** 4))
        free = [n for n in ids if n not in fixed]
        fi = {n: i for i, n in enumerate(free)}
        rows, cols, vals = [], [], []
        b = np.zeros(len(free))
        for k, e in self.vessels.items():
            for a, c in ((e.u, e.v), (e.v, e.u)):
                if a in fi:
                    rows.append(fi[a]); cols.append(fi[a]); vals.append(g[k])
                    if c in fi:
                        rows.append(fi[a]); cols.append(fi[c]); vals.append(-g[k])
                    else:
                        b[fi[a]] += g[k] * fixed[c]
        P = dict(fixed)
        if free:
            A = coo_matrix((vals, (rows, cols)), shape=(len(free), len(free))).tocsr()
            # a node on no path between boundaries has a zero row: pin it
            diag = A.diagonal()
            if np.any(diag == 0):
                A = A + coo_matrix((np.where(diag == 0, 1.0, 0.0), (range(len(free)), range(len(free)))),
                                   shape=A.shape).tocsr()
            x = spsolve(A, b)
            for n, i in fi.items():
                P[n] = float(x[i])
        for n in ids:
            self.nodes[n].pressure = float(P[n])
        signed = {k: g[k] * (P[e.u] - P[e.v]) for k, e in self.vessels.items()}
        med = float(np.median(np.abs(list(signed.values())))) if signed else 0.0
        for k, e in self.vessels.items():
            q = signed[k]
            dp = P[e.u] - P[e.v]
            backwards = dp < 0 if abs(dp) > 1e-12 else idx[e.u] > idx[e.v]
            if orient and backwards:
                e.reverse()
            e.flow = abs(q)
            e.info["stagnant"] = bool(abs(q) < stagnant * med)
        self.classify_nodes()
        return signed

    # ---------------------------------------------------------- geometry
    def crossings(self, spacing: float = 0.5, near_node: float | None = None) -> list[dict]:
        """Where two vessels' centrelines cross in the image, away from a node
        they share: xy (d_c), the two vessels and their arclengths there, the
        depth of each, which is on top ('above': the vid of the shallower),
        and the crossing angle (deg)."""
        from scipy.spatial import cKDTree
        smp = {k: e.sample(spacing) for k, e in self.vessels.items()}
        keys = list(smp)
        pts = np.concatenate([smp[k]["xy"][:-1] for k in keys]) if keys else np.zeros((0, 2))
        own = np.concatenate([np.full(len(smp[k]["xy"]) - 1, k) for k in keys]) if keys else np.zeros(0, int)
        seg = np.concatenate([np.arange(len(smp[k]["xy"]) - 1) for k in keys]) if keys else np.zeros(0, int)
        tree = cKDTree(pts)
        out = []
        seen = set()
        for i, j in tree.query_pairs(2.0 * spacing):
            a, b = int(own[i]), int(own[j])
            if a == b:
                continue
            if a > b:
                i, j, a, b = j, i, b, a
            ea, eb = self.vessels[a], self.vessels[b]
            shared = {ea.u, ea.v} & {eb.u, eb.v}
            p0, p1 = smp[a]["xy"][seg[i]], smp[a]["xy"][seg[i] + 1]
            q0, q1 = smp[b]["xy"][seg[j]], smp[b]["xy"][seg[j] + 1]
            hit = _seg_intersect(p0, p1, q0, q1)
            if hit is None:
                continue
            ta, tb = hit
            xy = p0 + ta * (p1 - p0)
            if shared:
                rad = near_node if near_node is not None else 3.0 * max(smp[a]["r"].max(), smp[b]["r"].max())
                if any(np.hypot(*(xy - self.nodes[n].xy)) < rad for n in shared):
                    continue
            key = (a, b, int(round(xy[0] * 4)), int(round(xy[1] * 4)))
            if key in seen:
                continue
            seen.add(key)
            sa = smp[a]["s"][seg[i]] + ta * (smp[a]["s"][seg[i] + 1] - smp[a]["s"][seg[i]])
            sb = smp[b]["s"][seg[j]] + tb * (smp[b]["s"][seg[j] + 1] - smp[b]["s"][seg[j]])
            za = float(np.interp(sa, smp[a]["s"], smp[a]["z"]))
            zb = float(np.interp(sb, smp[b]["s"], smp[b]["z"]))
            if shared and near_node is None:
                # vessels that share a node and overlap at one depth within its junction reach are one blood
                # volume (the junction), not a crossing (v2 correctness review, defect 6: siblings 6 d_c from
                # their node at 9.5 deg were reported, making the junction truth a 'compound')
                ra = float(np.interp(sa, smp[a]["s"], smp[a]["r"]))
                rb = float(np.interp(sb, smp[b]["s"], smp[b]["r"]))
                if abs(za - zb) < ra + rb and any(np.hypot(*(xy - self.nodes[n].xy)) < self.junction_reach(n)
                                                  for n in shared):
                    continue
            ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(smp[a]["t"][seg[i]], smp[b]["t"][seg[j]]))))))
            out.append(dict(x=float(xy[0]), y=float(xy[1]), vessels=[a, b], s=[float(sa), float(sb)],
                            depth=[za, zb], above=a if za <= zb else b, angle_deg=ang))
        return out

    def bounds(self):
        allxy = np.concatenate([e.ctrl for e in self.vessels.values()]) if self.vessels else np.zeros((1, 2))
        return allxy.min(0), allxy.max(0)

    def end_direction(self, vid: int, end: int) -> np.ndarray:
        """Unit direction (d_c) in which vessel vid leaves its node at end 0
        (u) or 1 (v): a clamped B-spline leaves along its first control leg."""
        c = np.asarray(self.vessels[vid].ctrl, float)
        c = c if end == 0 else c[::-1]
        for i in range(1, len(c)):
            d = c[i] - c[0]
            n = math.hypot(d[0], d[1])
            if n > 1e-9:
                return d / n
        return np.array([1.0, 0.0])

    def junction_reach(self, nid: int, gap: float = 0.5) -> float:
        """How far (d_c, from the node) the lumens of the vessels meeting at a
        node may legitimately overlap: two straight tubes of radii r_i, r_j
        leaving a node theta apart stay within r_i + r_j + gap of each other
        for (r_i + r_j + gap) / sin(theta) along them.  Returns 1.25 x the
        largest of that over pairs of incident vessels (sin floored at 0.25;
        r_i + r_j + gap for pairs more than 150 deg apart, i.e. a vessel and
        its continuation) + 1 d_c.  Beyond it, vessels that share the node are
        separate blood volumes and must not overlap in 3-D (check); the
        renderer's union region (render.junction_radius) is the same rule in
        px, extended to wherever the lumens actually keep overlapping.  0 for
        a node with fewer than two vessels."""
        inc = [(k, 0) for k, e in self.vessels.items() if e.u == nid] + \
              [(k, 1) for k, e in self.vessels.items() if e.v == nid]
        arms = []
        for vid, end in inc:
            e = self.vessels[vid]
            r = float(eval_profile(e.r_ctrl, [float(end)])[0])
            arms.append((self.end_direction(vid, end), r))
        R = 0.0
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                (ti, ri), (tj, rj) = arms[i], arms[j]
                ang = math.acos(float(np.clip(np.dot(ti, tj), -1.0, 1.0)))
                w = ri + rj + gap
                R = max(R, w / max(math.sin(ang), 0.25) if ang < math.radians(150) else w)
        return 1.25 * R + 1.0 if len(arms) >= 2 else 0.0

    # ---------------------------------------------------------- checks
    def check(self, overlap: bool = False, gap: float = 0.5) -> list[str]:
        """Invariants of a valid ground truth; returns the problems found
        (empty: valid).  With overlap, also that no two vessels' lumens
        overlap in 3-D (image distance < r_a + r_b + gap while their depths
        differ by less than that) away from the nodes they share (beyond the
        node's junction_reach), and that no vessel's lumen overlaps itself
        (two stretches of one vessel more than pi (r_a + r_b) apart along it
        closer than r_a + r_b in 3-D: the renderer would add them)."""
        import networkx as nx
        bad = []
        for k, e in self.vessels.items():
            for end, nid in ((e.ctrl[0], e.u), (e.ctrl[-1], e.v)):
                if nid not in self.nodes:
                    bad.append(f"vessel {k}: node {nid} missing")
                elif np.hypot(*(end - self.nodes[nid].xy)) > 1e-6:
                    bad.append(f"vessel {k}: end not on node {nid}")
            smp = e.sample(1.0)
            if np.any(smp["r"] <= 0):
                bad.append(f"vessel {k}: radius <= 0")
            if np.any(smp["z"] < -1e-6):
                bad.append(f"vessel {k}: negative depth")
            if e.cls not in CLASSES or e.side not in SIDES:
                bad.append(f"vessel {k}: class/side {e.cls}/{e.side}")
            for tau, nid in ((0.0, e.u), (1.0, e.v)):
                if nid in self.nodes and abs(float(eval_profile(e.z_ctrl, [tau])[0]) - self.nodes[nid].depth) > 0.05:
                    bad.append(f"vessel {k}: depth at its end differs from node {nid}")
        G = self.to_digraph()
        if not nx.is_directed_acyclic_graph(G):
            bad.append("the digraph has a cycle")
        for nid, (i, o) in self.degrees().items():
            n = self.nodes[nid]
            if i + o == 0:
                bad.append(f"node {nid}: isolated")
            elif i == 1 and o == 1:
                bad.append(f"node {nid}: one vessel in and one out (a vessel must not end without a fork)")
            if n.kind in ("inlet", "outlet") and "domain" in self.meta:
                x0, y0, x1, y1 = self.meta["domain"]
                if min(n.xy[0] - x0, x1 - n.xy[0], n.xy[1] - y0, y1 - n.xy[1]) > 1.0:
                    bad.append(f"node {nid}: {n.kind} inside the domain (a free end)")
        if overlap:
            bad += self._overlaps(gap)
        return bad

    def _overlaps(self, gap: float) -> list[str]:
        if not self.vessels:
            return []
        smp = {k: e.sample(0.5) for k, e in self.vessels.items()}
        keys = list(smp)
        pts = np.concatenate([smp[k]["xy"] for k in keys])
        own = np.concatenate([np.full(len(smp[k]["xy"]), k) for k in keys])
        r = np.concatenate([smp[k]["r"] for k in keys])
        z = np.concatenate([smp[k]["z"] for k in keys])
        s = np.concatenate([smp[k]["s"] for k in keys])
        i, j = close_pairs(pts, r, gap)
        d = np.hypot(*(pts[i] - pts[j]).T)
        same = own[i] == own[j]
        # two stretches of one vessel: a true lumen overlap (no gap), far enough apart along it that the
        # closeness is not just the vessel's own bend (a U-turn at |kappa| r <= 0.9 keeps its legs > 2 r apart)
        rs = r[i] + r[j]
        selfhit = same & (np.abs(s[i] - s[j]) > math.pi * rs) & (d < rs) & (np.abs(z[i] - z[j]) < rs)
        bad = {(int(a), int(a), "itself") for a in own[i][selfhit]}
        lim = r[i] + r[j] + gap
        m = ~same & (d < lim) & (np.abs(z[i] - z[j]) < lim)
        i, j = i[m], j[m]
        a, b = own[i], own[j]
        exempt = np.zeros(len(i), bool)
        U = {k: e.u for k, e in self.vessels.items()}
        V = {k: e.v for k, e in self.vessels.items()}
        reach = {}
        for na, nb in ((U, U), (U, V), (V, U), (V, V)):
            n1 = np.array([na[x] for x in a], int)
            n2 = np.array([nb[x] for x in b], int)
            sh = np.flatnonzero((n1 == n2) & ~exempt)
            for q in sh:
                n = int(n1[q])
                if n not in reach:
                    reach[n] = self.junction_reach(n, gap)
                nxy = self.nodes[n].xy
                if max(np.hypot(*(pts[i[q]] - nxy)), np.hypot(*(pts[j[q]] - nxy))) < reach[n]:
                    exempt[q] = True
        a, b = a[~exempt], b[~exempt]
        bad |= {(int(min(x, y)), int(max(x, y)), "") for x, y in zip(a, b)}
        return [f"vessel {x}: its lumen overlaps itself in 3-D" if w == "itself" else
                f"vessels {x} and {y}: lumens overlap in 3-D" for x, y, w in sorted(bad)]

    # ---------------------------------------------------------- io
    def to_dict(self) -> dict:
        return dict(
            format="vesselscene-truth", version=1,
            units=dict(length="d_c (capillary diameters)", depth="d_c below the surface",
                       hct="relative haematocrit (1 = systemic)"),
            meta={k: v for k, v in self.meta.items() if _jsonable(v)},
            nodes=[dict(nid=n.nid, xy=n.xy.tolist(), depth=n.depth, kind=n.kind,
                        pressure=n.pressure, info=n.info) for n in self.nodes.values()],
            vessels=[dict(vid=e.vid, u=e.u, v=e.v, ctrl=e.ctrl.tolist(), r_ctrl=e.r_ctrl.tolist(),
                          z_ctrl=e.z_ctrl.tolist(), h_ctrl=e.h_ctrl.tolist(), cls=e.cls,
                          side=e.side, flow=e.flow, info=e.info) for e in self.vessels.values()])

    @classmethod
    def from_dict(cls, d: dict) -> "VesselGraph":
        g = cls(d.get("meta"))
        for n in d["nodes"]:
            g.nodes[n["nid"]] = Node(n["nid"], np.asarray(n["xy"], float), n["depth"], n["kind"],
                                     n["pressure"], n.get("info", {}))
        for e in d["vessels"]:
            g.vessels[e["vid"]] = Vessel(e["vid"], e["u"], e["v"], np.asarray(e["ctrl"], float),
                                         np.asarray(e["r_ctrl"], float), np.asarray(e["z_ctrl"], float),
                                         np.asarray(e["h_ctrl"], float), e["cls"], e["side"],
                                         e["flow"], e.get("info", {}))
        g._next_node = max(g.nodes, default=-1) + 1
        g._next_vessel = max(g.vessels, default=-1) + 1
        return g

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, default=_np_default)

    @classmethod
    def load(cls, path: str) -> "VesselGraph":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def close_pairs(xy: np.ndarray, r: np.ndarray, gap: float, rsplit: float = 2.0):
    """Index pairs (i, j), i < j, of points closer than r_i + r_j + gap (a
    KD-tree query sized by the small radii, plus a per-point query for the
    few points with a radius above rsplit)."""
    from scipy.spatial import cKDTree
    xy, r = np.asarray(xy, float), np.asarray(r, float)
    if len(xy) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    tree = cKDTree(xy)
    out = [tree.query_pairs(2 * rsplit + gap, output_type="ndarray")]
    big = np.flatnonzero(r > rsplit)
    if len(big):
        # the big points' partners, in chunks and filtered by the real distance as they come: a small partner lies
        # within r_i + rsplit + gap, a big one within r_i + rmax + gap (the same pairs as before; v2: one query of
        # every big point over r_i + rmax + gap made 28 M candidate pairs, 425 MB, in a pathologic frame)
        rmax = float(r.max())
        btree = cKDTree(xy[big])
        for c0 in range(0, len(big), 2048):
            bc = big[c0:c0 + 2048]
            for tr_, imap, rad in ((tree, None, r[bc] + rsplit + gap), (btree, big, r[bc] + rmax + gap)):
                hits = tr_.query_ball_point(xy[bc], rad)
                lens = np.fromiter((len(h) for h in hits), np.int64, len(hits))
                if not lens.sum():
                    continue
                jj = np.fromiter(itertools.chain.from_iterable(hits), np.int64, int(lens.sum()))
                if imap is not None:
                    jj = imap[jj]
                ii = np.repeat(bc, lens)
                keep = (jj != ii) & (np.hypot(*(xy[ii] - xy[jj]).T) < r[ii] + r[jj] + gap)
                out.append(np.stack([ii[keep], jj[keep]], 1))
    P = np.concatenate([p.reshape(-1, 2) for p in out]).astype(np.int64)
    i, j = np.minimum(P[:, 0], P[:, 1]), np.maximum(P[:, 0], P[:, 1])
    keep = np.hypot(*(xy[i] - xy[j]).T) < r[i] + r[j] + gap
    # unique pairs, sorted by (i, j): one int64 key per pair (much faster than np.unique(axis=0))
    n = np.int64(len(xy))
    key = np.unique(i[keep] * n + j[keep])
    return key // n, key % n


def _seg_intersect(p0, p1, q0, q1):
    """Parameters (ta, tb) in [0, 1] where segments p0p1 and q0q1 cross, or None."""
    r, s = p1 - p0, q1 - q0
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    w = q0 - p0
    ta = (w[0] * s[1] - w[1] * s[0]) / den
    tb = (w[0] * r[1] - w[1] * r[0]) / den
    if 0.0 <= ta < 1.0 and 0.0 <= tb < 1.0:
        return float(ta), float(tb)
    return None


def _jsonable(v) -> bool:
    try:
        json.dumps(v, default=_np_default)
        return True
    except TypeError:
        return False


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
