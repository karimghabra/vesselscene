"""Comparison annotator: the standard curvature-analysis pipeline (Hessian ridges, skeleton, post-hoc typing).

This is the method family the neuromimetic annotator is measured against (DESIGN.md, "Why not curvature
alone"), built as a competent practitioner would, with its few thresholds tuned on the dev scenes:

1. **Log and flat field.**  ``L = log I`` (no-data pixels filled from the nearest valid one, so filters see no
   step; they and saturated glare, grown by 3 px, are masked).  The background ``B`` is a grey closing of
   ``L`` with a disk wider than the widest vessel (an upper envelope: dark lines narrower than the disk are
   filled), smoothed.  ``OD = B - L`` makes vessels bright, in nepers.
2. **Multi-scale Hessian ridges** (LIMBUS ``vesselmap.ridges.ridge_maps``, reused as is): at each scale ``s``
   the ridge strength ``s^2 max(0, -l1 - |l2|)`` (``l1 <= l2``; the anisotropy term rejects blobs), divided
   by a null that is the larger of white sensor noise at that scale and the local RMS of the opposite-polarity
   valley strength (texture makes valleys as often as ridges, vessels only ridges).  The scale with the
   largest normalised strength wins; its z-score is what is thresholded.
3. **Readout.**  Non-maximum suppression across the ridge (along the Hessian normal), hysteresis on z,
   skeletonised, components shorter than ``MIN_COMPONENT_PX`` dropped.  Gap closing: the anisotropy term
   cancels the response where two vessels meet (both eigenvalues large), so lines stop a few px short of
   every junction.  Each line end is joined straight to the nearest skeleton pixel ahead of it (within
   ``BRIDGE_PX`` and a cone of ``BRIDGE_DEG``), then the result is thinned again.  Without this step dev
   junction recall is about 0.03; with it about 0.6.
4. **Skeleton graph.**  Pixels are linked 8-connected, a diagonal link being dropped where a 4-connected
   path of skeleton pixels joins the same two pixels (otherwise every staircase step reads as a branch).
   Pixels with >= 3 links are junction pixels, clustered 8-connected into nodes; pixels with one link are
   ends.  Arms are traced between nodes: one polyline per arm.  Spurs (an end arm shorter than a
   scale-aware length) are pruned, nodes left with two arms dissolve, until stable.
5. **Post-hoc junction typing.**  Junction nodes joined by an arm shorter than ``MERGE_PX`` (about lambda)
   are merged into one node (a crossing read through a Hessian is typically two forks a few px apart).  A
   node's arms point from its centre to each arm's point ``ARM_LOOK_PX`` along it.  3 arms -> 'pseudo-T' (the
   most frequent 3-way class in the observable truth; a still shows no flow direction), 4 arms that pair into
   two near-collinear through lines -> 'crossing', otherwise 'compound'.

Polylines run from node centre to node centre.  Everything is deterministic: fixed filters, no random
numbers (the cached noise constant of ``ridge_maps`` is a fixed-seed simulation), stable orders.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

LIMBUS = "/home/user/karimghabra/limbus"
if LIMBUS not in sys.path:
    sys.path.insert(0, LIMBUS)
from vesselmap.image import robust_noise_map          # noqa: E402  (MAD of the Laplacian, per tile)
from vesselmap.ridges import nms, ridge_maps           # noqa: E402

cv2.setNumThreads(2)

LAMBDA_PX = 11.9
SAT_DN = 4095.0
# tuned on the dev scenes (average and frame) by grid search on the mean of centreline F1, junction F1,
# coarse type accuracy and edge cover
BG_DISK_PX = 61            # closing disk diameter: wider than the widest vessel
BG_SMOOTH_PX = 8.0
SCALES = (1.0, 1.5, 2.2, 3.3, 5.0, 7.5, 11.0, 16.0)
Z_LO, Z_HI = 1.5, 2.0      # hysteresis on the ridge z-score
MIN_COMPONENT_PX = 12      # skeleton components shorter than this are dropped
SPUR_PX = 10.0             # an end arm shorter than max(SPUR_PX, 1.5 x its ridge scale) is a spur
MERGE_PX = LAMBDA_PX       # junction nodes joined by a shorter arm are one junction
ARM_LOOK_PX = 24.0         # arm direction: node centre -> the arm's point this far along it
BRIDGE_PX = 2.5 * LAMBDA_PX  # end-gap closing: reach,
BRIDGE_DEG = 35.0            # cone half-angle about the end's direction,
BRIDGE_LOOK_PX = 8           # read from the pixel this many steps back along the line
THROUGH_COS = 0.9          # two arms are one through line when their directions are within ~26 deg of opposite


# ----------------------------------------------------------------------------------------- junction typing
def type_from_arms(dirs, through_cos: float | None = None) -> str:
    """Post-hoc type of a node from its arms' outward directions: 3 -> 'pseudo-T'; 4 that pair into two
    near-collinear through lines (each pair within acos(through_cos) of opposite) -> 'crossing'; otherwise
    'compound'."""
    through_cos = THROUGH_COS if through_cos is None else through_cos
    u = np.asarray(dirs, float).reshape(-1, 2)
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    if len(u) == 3:
        return "pseudo-T"
    if len(u) == 4:
        worst = min(max(u[a] @ u[b], u[c] @ u[d]) for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)))
        if worst <= -through_cos:
            return "crossing"
    return "compound"


# ------------------------------------------------------------------------------------------ ridge readout
def optical_density(image: np.ndarray, valid: np.ndarray):
    """OD = B - log I (vessels positive), the masked pixels and the log image's noise map."""
    ok = valid & np.isfinite(image) & (image < SAT_DN) & (image > 0)
    bad = ~ok
    if bad.any():
        _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
        image = image[iy, ix]
        ok = ~ndi.binary_dilation(bad, iterations=3)
    L = np.log(np.maximum(image, 1.0)).astype(np.float32)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_DISK_PX, BG_DISK_PX))
    pad = BG_DISK_PX // 2 + 1
    Lp = cv2.copyMakeBorder(L, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    B = cv2.morphologyEx(Lp, cv2.MORPH_CLOSE, disk)[pad:-pad, pad:-pad]
    B = ndi.gaussian_filter(B, BG_SMOOTH_PX, mode="reflect")
    return (B - L).astype(np.float32), ok, robust_noise_map(L)


def ridge_skeleton(image: np.ndarray, valid: np.ndarray):
    """Hessian ridges -> NMS -> hysteresis -> skeleton; also the selected scale per pixel."""
    od, ok, noise = optical_density(image, valid)
    rho, z, sc, nx, ny = ridge_maps(od, noise, SCALES, valid=ok)
    cand = nms(z, nx, ny) & (z > Z_LO) & ok
    mask = apply_hysteresis_threshold(np.where(cand, z, 0.0), Z_LO, Z_HI)
    mask = ndi.binary_closing(mask, structure=np.ones((2, 2))) | mask     # NMS leaves 1 px diagonal gaps
    skel = skeletonize(mask)
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    if n:
        size = ndi.sum(skel, lab, np.arange(1, n + 1))
        skel &= np.r_[False, size >= MIN_COMPONENT_PX][lab]
    if BRIDGE_PX > 0:
        skel = skeletonize(bridge_ends(skel))
    return skel, sc


def bridge_ends(skel: np.ndarray) -> np.ndarray:
    """Close the gaps the ridge measure leaves at junctions: every line end is joined by a straight segment to
    the nearest skeleton pixel ahead of it (within BRIDGE_PX and a cone of BRIDGE_DEG about the end's
    direction, own arm excluded).  A line stopping short of another becomes a T, two collinear ends across a
    crossing become one line again."""
    from scipy.spatial import cKDTree
    ys, xs, nbrs = pixel_links(skel)
    if not len(ys):
        return skel
    deg = np.array([len(n) for n in nbrs])
    tree = cKDTree(np.stack([ys, xs], 1))
    cos_min = np.cos(np.deg2rad(BRIDGE_DEG))
    out = skel.copy()
    for e in np.flatnonzero(deg == 1):
        arm, prev, cur = [e], -1, e                 # the end's own arm, up to its junction or other end
        while True:
            step = [k for k in nbrs[cur] if k != prev]
            if len(step) != 1 or deg[step[0]] > 2:
                break
            prev, cur = cur, step[0]
            arm.append(cur)
        if len(arm) < 4:
            continue
        inner = arm[min(len(arm) - 1, int(BRIDGE_LOOK_PX))]
        d = np.array([ys[e] - ys[inner], xs[e] - xs[inner]], float)
        d /= np.hypot(*d) + 1e-12
        own = set(arm)
        best = None
        for q in sorted(tree.query_ball_point([ys[e], xs[e]], BRIDGE_PX)):
            if q in own:
                continue
            r = np.array([ys[q] - ys[e], xs[q] - xs[e]], float)
            dist = np.hypot(*r)
            c = r @ d / dist
            if c < cos_min:
                continue
            score = dist * (2.0 - c)
            if best is None or score < best[0]:
                best = (score, q)
        if best is not None:
            q = best[1]
            cv2.line(out.view(np.uint8), (int(xs[e]), int(ys[e])), (int(xs[q]), int(ys[q])), 1, 1)
    return out


# ---------------------------------------------------------------------------------------- skeleton graph
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def pixel_links(skel: np.ndarray):
    """8-connected links between skeleton pixels, a diagonal dropped where a 4-path joins its two pixels.
    Returns (ys, xs, nbrs): pixel coordinates and each pixel's linked pixel indices (sorted)."""
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    idx = -np.ones((h + 2, w + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(len(ys))
    on = idx >= 0
    nbrs = [[] for _ in range(len(ys))]
    for dy, dx in _OFFS:
        j = idx[ys + 1 + dy, xs + 1 + dx]
        keep = j >= 0
        if dy and dx:      # diagonal: skip if either shared 4-neighbour is on the skeleton
            keep &= ~(on[ys + 1 + dy, xs + 1] | on[ys + 1, xs + 1 + dx])
        for i in np.flatnonzero(keep):
            nbrs[i].append(int(j[i]))
    return ys, xs, [sorted(n) for n in nbrs]


def skeleton_graph(skel: np.ndarray):
    """Nodes (junction clusters and ends) and arms (pixel paths between them).

    Returns nodes: {nid: (y, x) array of member pixels}, arms: [[u, v, (m, 2) y x path incl. both end pixels]].
    Closed loops without any node get one node on themselves."""
    ys, xs, nbrs = pixel_links(skel)
    deg = np.array([len(n) for n in nbrs])
    junc = np.zeros(skel.shape, bool)
    junc[ys[deg >= 3], xs[deg >= 3]] = True
    jl, nj = ndi.label(junc, structure=np.ones((3, 3)))
    node_of = np.full(len(ys), -1)
    node_of[deg >= 3] = jl[ys[deg >= 3], xs[deg >= 3]] - 1
    nid = nj
    for i in np.flatnonzero(deg <= 1):
        node_of[i] = nid
        nid += 1
    arms, done = [], set()

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while node_of[cur] < 0:
            step = [k for k in nbrs[cur] if k != prev]
            if not step:
                break
            prev, cur = cur, step[0]
            path.append(cur)
            if cur == start:
                break
        return path

    for i in np.flatnonzero(node_of >= 0):
        for k in nbrs[i]:
            if node_of[k] >= 0 and node_of[k] == node_of[i]:
                continue                                  # inside one junction cluster
            key = (min(i, k), max(i, k))
            if key in done:
                continue
            path = walk(int(i), int(k))
            done.add(key)
            done.add((min(path[-1], path[-2]), max(path[-1], path[-2])))
            arms.append([int(node_of[i]), int(node_of[path[-1]]), np.stack([ys[path], xs[path]], 1)])
    # loops of degree-2 pixels only: cut at their first pixel
    covered = np.zeros(skel.shape, bool)
    for _, _, p in arms:
        covered[p[:, 0], p[:, 1]] = True
    for i in range(len(ys)):
        if node_of[i] < 0 and not covered[ys[i], xs[i]] and deg[i] == 2:
            node_of[i] = nid
            path = walk(i, nbrs[i][0])
            arms.append([nid, nid, np.stack([ys[path], xs[path]], 1)])
            covered[ys[path], xs[path]] = True
            nid += 1
    nodes = {}
    for i in np.flatnonzero(node_of >= 0):
        nodes.setdefault(int(node_of[i]), []).append((ys[i], xs[i]))
    return {k: np.array(v, float) for k, v in nodes.items()}, arms


def _length(p) -> float:
    return float(np.hypot(*np.diff(np.asarray(p, float), axis=0).T).sum()) if len(p) > 1 else 0.0


def _join(a, b, node):
    """Concatenate arms a and b at their shared node into one arm (paths y x)."""
    pa = a[2] if a[1] == node else a[2][::-1]
    ua = a[0] if a[1] == node else a[1]
    pb = b[2] if b[0] == node else b[2][::-1]
    vb = b[1] if b[0] == node else b[0]
    return [ua, vb, np.vstack([pa, pb[1:]])]


def simplify(nodes, arms, scale):
    """Prune spurs and dissolve degree-2 nodes until stable; then merge junction nodes joined by short arms."""
    arms = [a for a in arms if len(a[2]) >= 2]
    while True:
        deg = {}
        for u, v, _ in arms:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        spur = []
        for k, (u, v, p) in enumerate(arms):
            if u == v:
                continue
            if (deg[u] == 1) != (deg[v] == 1):        # an end arm on a junction
                s = float(np.max(scale[p[:, 0].astype(int), p[:, 1].astype(int)]))
                if _length(p) < max(SPUR_PX, 1.5 * s):
                    spur.append(k)
        if spur:   # shortest first, one per junction per round, so two short spurs do not both vanish
            spur.sort(key=lambda k: (_length(arms[k][2]), k))
            hit, drop = set(), set()
            for k in spur:
                u, v, _ = arms[k]
                j = u if deg[u] > 1 else v
                if j not in hit:
                    hit.add(j)
                    drop.add(k)
            arms = [a for k, a in enumerate(arms) if k not in drop]
            continue
        two = [n for n in sorted(deg) if deg[n] == 2 and not any(a[0] == a[1] == n for a in arms)]
        if not two:
            break
        n = two[0]
        ks = [k for k, a in enumerate(arms) if n in (a[0], a[1])]
        joined = _join(arms[ks[0]], arms[ks[1]], n)
        arms = [a for k, a in enumerate(arms) if k not in ks] + [joined]
    # merge junction nodes joined by an arm shorter than MERGE_PX (union-find, shortest arms first)
    deg = {}
    for u, v, _ in arms:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    parent = {n: n for n in deg}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    order = sorted(range(len(arms)), key=lambda k: (_length(arms[k][2]), k))
    internal = set()
    for k in order:
        u, v, p = arms[k]
        if u != v and deg[u] >= 3 and deg[v] >= 3 and _length(p) < MERGE_PX:
            internal.add(k)
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    groups = {}
    for n in deg:
        groups.setdefault(find(n), []).append(n)
    centre = {g: np.mean(np.vstack([nodes[n] for n in mem]), axis=0) for g, mem in groups.items()}
    out = [[find(u), find(v), p] for k, (u, v, p) in enumerate(arms) if k not in internal]
    return centre, out


def annotate(image: np.ndarray, valid: np.ndarray) -> dict:
    """Hessian-ridge skeleton tracing with post-hoc junction typing (module docstring)."""
    skel, scale = ridge_skeleton(np.asarray(image, np.float32), np.asarray(valid, bool))
    nodes, arms = skeleton_graph(skel)
    centre, arms = simplify(nodes, arms, scale)
    arms.sort(key=lambda a: tuple(a[2][0]) + tuple(a[2][-1]) + (len(a[2]),))
    deg = {}
    for u, v, _ in arms:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    dirs, polylines = {}, []
    for u, v, p in arms:
        yx = p.astype(float)
        for node, path in ((u, yx), (v, yx[::-1])):
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
            j = min(int(np.searchsorted(s, ARM_LOOK_PX)), len(path) - 1)
            dirs.setdefault(node, []).append((path[j] - centre[node])[::-1])
        # run from junction centre to junction centre (an end keeps its own pixel)
        line = np.vstack([centre[u][None]] * (deg[u] >= 3) + [yx] + [centre[v][None]] * (deg[v] >= 3))
        polylines.append(line[:, ::-1].copy())
    junctions = [(float(centre[n][1]), float(centre[n][0]), type_from_arms(dirs[n]))
                 for n in sorted(deg, key=lambda n: (centre[n][0], centre[n][1])) if deg[n] >= 3]
    return dict(polylines=polylines, junctions=junctions)
