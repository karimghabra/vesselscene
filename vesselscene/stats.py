"""Annotation-free image statistics of dark-vessel stills: the realism suite.

Every statistic here needs only the image, so a synthetic scene, a real
stabilized still and a single raw frame are measured by exactly the same
code.  `analyse` is the frozen suite v1 of the synthetic-generator plan
(imstats.py, ported unchanged in behaviour: the same seed gives the same
numbers) plus the cheap additions of plan section 7.1:

    tortuosity by scale   arc/chord over windows of 1, 2 and 4 lambda of arc
                          along skeleton paths linked through junctions and
                          across gaps by good continuation (the old branch
                          arc/chord is measured on ~26 px branches and cannot
                          see tortuosity);
    junction geometry     branch angles at 3-arm nodes; crossings (4-arm X
                          nodes, plus gaps bridged across another vessel's
                          skeleton, which is how most crossings look to a
                          ridge detector), their density and angles;
    contrast law          contrast against width in the sharpest third of the
                          profiles (in-focus vessels: blur does not confound it);
    along-vessel spread   robust CV of contrast and FWHM along linked paths;
    faint mesh            locally whitened ridges at sigma 1.5-3 px, hysteresis
                          down to ~0.01-0.02 Np, unioned with the main
                          detection, and the loops (closed meshes) they form.

All of these are detector responses, not truth: the skeleton misses or
fragments blurred deep vessels and faint capillaries, so crossing counts and
path lengths are underestimates on real and synthetic images alike.  They
are meaningful only as comparisons between images measured the same way.

Intensity normalisation (applied identically to real and synthetic):
    L = ln I                      (natural log; I in any linear unit, NaN = no data)
    B = large-scale background:   80th percentile of L in 32x32 px blocks
                                  (valid pixels only), Gaussian-smoothed over
                                  1.5 blocks, bicubic-upsampled
    F = L - B                     "flattened log image", in nepers (Np).
Vessels are negative in F; -F at a vessel centre is roughly its optical
density (natural-log units, the unit of vesselmap's contrast a).

lambda (`lam`) is the image's own median profile FWHM (fwhm_p50, about 12 px
on the real stills).  Structural statistics are also reported in lambda units
(keys "lam:...") because the pixel scale of a synthetic scene (k px per
capillary diameter) is a free parameter.

`scalars(res)` flattens an analyse() result into {key: float}; `LABELS` and
`TIERS` describe the keys the scorecard (vesselscene.realism) compares.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy import stats as sst
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects, skeletonize

STATS_VERSION = "vesselscene.stats/1.1 (imstats suite v1 + plan 7.1 a,b,d,e,f)"
SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
C0_ABS = 0.03          # absolute contrast floor (nepers) for ridge thresholds
Q_TEX = 0.95           # opposite-polarity quantile for texture-calibrated thresholds
LAM_DEFAULT = 12.0     # lambda fallback (px) when an image yields no profiles
TORT_LAMS = (1, 2, 4)  # tortuosity windows, in lambda of arc length
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))  # opposite = k + 4


# --------------------------------------------------------------- normalise
def fill_invalid(L, valid, sigma=8.0):
    """Fill invalid pixels of L by normalised convolution with growing
    Gaussians (so filtering near NaN borders does not ring)."""
    if valid.all():
        return L.astype(np.float64)
    Lz = np.where(valid, L, 0.0)
    out = Lz.copy()
    todo = ~valid
    s = sigma
    while todo.any() and s < 4 * max(L.shape):
        num = ndi.gaussian_filter(Lz, s)
        den = ndi.gaussian_filter(valid.astype(float), s)
        ok = todo & (den > 1e-3)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        s *= 2
    if todo.any():
        out[todo] = np.median(L[valid])
    return out


def flatten(I, valid=None, block=32, q=80, smooth=1.5):
    """F = ln I - B (see module docstring).  Returns (F, B, valid); F is
    filled (not NaN) outside `valid` so filters can run over it."""
    I = np.asarray(I, np.float64)
    if valid is None:
        valid = np.isfinite(I) & (I > 0)
    L = np.full(I.shape, np.nan)
    L[valid] = np.log(I[valid])
    H, W = I.shape
    nby, nbx = math.ceil(H / block), math.ceil(W / block)
    P = np.full((nby * block, nbx * block), np.nan)
    P[:H, :W] = L
    blocks = P.reshape(nby, block, nbx, block).transpose(0, 2, 1, 3).reshape(nby, nbx, -1)
    with np.errstate(all="ignore"):
        cnt = np.isfinite(blocks).sum(-1)
        g = np.nanpercentile(np.where(cnt[..., None] > block * block // 8, blocks, np.nan), q, axis=-1)
    gv = np.isfinite(g)
    g = fill_invalid(np.nan_to_num(g), gv, sigma=1.0)
    g = ndi.gaussian_filter(g, smooth, mode="nearest")
    B = cv2.resize(g.astype(np.float32), (nbx * block, nby * block), interpolation=cv2.INTER_CUBIC)[:H, :W]
    Lf = fill_invalid(np.nan_to_num(L), valid)
    F = Lf - B
    return F.astype(np.float64), B.astype(np.float64), valid


def eroded_valid(valid, r):
    """Valid pixels at least r px from invalid data and from the image border."""
    if valid.all():
        m = np.ones_like(valid)
    else:
        d = ndi.distance_transform_edt(valid)
        m = d > r
    r = int(math.ceil(r))
    m[:r, :] = m[-r:, :] = False
    m[:, :r] = m[:, -r:] = False
    return m


# ------------------------------------------------------------------- ridges
def hessian(F, s):
    """Scale-normalised dark- and bright-line strengths at scale s.

    D  = s^2 (l1 - |l2|)+   dark line (vessel): large positive curvature across
    Bm = s^2 (-l2 - |l1|)+  bright line: the opposite polarity, which vessels
                            barely produce, so it measures texture and noise
    phi = across-direction of a dark line.  A Gaussian line of contrast c
    at its matched scale gives D = 0.354 c.
    """
    Fxx = ndi.gaussian_filter(F, s, order=(0, 2), mode="reflect")
    Fyy = ndi.gaussian_filter(F, s, order=(2, 0), mode="reflect")
    Fxy = ndi.gaussian_filter(F, s, order=(1, 1), mode="reflect")
    tr = 0.5 * (Fxx + Fyy)
    disc = np.sqrt((0.5 * (Fxx - Fyy)) ** 2 + Fxy ** 2)
    l1, l2 = tr + disc, tr - disc
    s2 = s * s
    D = s2 * np.clip(l1 - np.abs(l2), 0, None)        # dark line (vessel)
    Bm = s2 * np.clip(-l2 - np.abs(l1), 0, None)      # bright line (opposite polarity)
    phi = 0.5 * np.arctan2(2 * Fxy, Fxx - Fyy)        # across-direction of a dark line
    return D, Bm, phi


def noise_sigma(F):
    """Pixel noise (Np): robust SD of the 5-point Laplacian / sqrt(20)
    (exact for white noise; smooth structure barely contributes)."""
    lap = ndi.convolve(F, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float))
    m = np.median(lap)
    return float(1.4826 * np.median(np.abs(lap - m)) / math.sqrt(20.0))


# ------------------------------------------------------ skeleton (frozen v1)
def skeleton_graph_stats(S, area):
    """Endpoints, junctions (clusters of pixels with >=3 neighbours), branch
    lengths and branch tortuosity of a binary skeleton (suite v1)."""
    S = S.astype(bool)
    nb = ndi.convolve(S.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    nb = np.where(S, nb, 0)
    endp = S & (nb == 1)
    jpx = S & (nb >= 3)
    jl, nj = ndi.label(jpx, structure=np.ones((3, 3)))
    # branches: skeleton minus dilated junctions
    br = S & ~ndi.binary_dilation(jpx, structure=np.ones((3, 3)))
    bl, nbr = ndi.label(br, structure=np.ones((3, 3)))
    lengths = np.bincount(bl.ravel())[1:].astype(float) if nbr else np.zeros(0)
    tort = []
    if nbr:
        nb2 = ndi.convolve(br.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
        ends = br & (nb2 == 1)
        ys, xs = np.nonzero(ends)
        labs = bl[ys, xs]
        order = np.argsort(labs)
        labs, ys, xs = labs[order], ys[order], xs[order]
        uniq, start, cnt = np.unique(labs, return_index=True, return_counts=True)
        for u, st, c in zip(uniq, start, cnt):
            if c == 2 and lengths[u - 1] >= 20:
                chord = math.hypot(ys[st] - ys[st + 1], xs[st] - xs[st + 1])
                tort.append(lengths[u - 1] * 1.06 / max(chord, 1.0))  # 1.06: mean px step on 8-conn lines
    return dict(skel_px=int(S.sum()), endpoints=int(endp.sum()), junctions=int(nj),
                n_branches=int(nbr), branch_len=lengths, branch_tort=np.array(tort),
                endpoint_density=endp.sum() / area * 1e5, junction_density=nj / area * 1e5,
                skel_density=S.sum() / area * 1e3)


def prune_spurs(S, sig_map, min_len=8.0, fac=2.0, min_comp=10):
    """Remove terminal skeleton branches shorter than max(min_len, fac*sigma)
    (spurs of wide masks) and components shorter than min_comp px."""
    S = S.astype(bool).copy()
    k3 = np.ones((3, 3), np.uint8)
    nb = np.where(S, ndi.convolve(S.astype(np.uint8), k3, mode="constant") - 1, 0)
    jpx = S & (nb >= 3)
    jd = ndi.binary_dilation(jpx, structure=k3)
    br = S & ~jd
    bl, n = ndi.label(br, structure=k3)
    if n:
        lengths = np.bincount(bl.ravel(), minlength=n + 1).astype(float)
        endp = S & (nb == 1)
        has_end = np.zeros(n + 1, bool)
        has_end[bl[endp]] = True
        touch = np.zeros(n + 1, bool)
        touch[bl[ndi.binary_dilation(jd, structure=k3) & br]] = True
        msig = np.asarray(ndi.mean(sig_map, bl, index=np.arange(n + 1)))
        lim = np.maximum(min_len, fac * np.nan_to_num(msig))
        rm = has_end & touch & (lengths < lim)
        rm[0] = False
        S[rm[bl]] = False
    S = skeletonize(S)
    cl, nc = ndi.label(S, structure=k3)
    if nc:
        sz = np.bincount(cl.ravel())
        small = sz < min_comp
        small[0] = False
        S[small[cl]] = False
    return S


def enclosed_loops(S, v1, min_area=20, return_map=False):
    """Areas (px^2) of the background regions fully enclosed by the skeleton
    S (4-connected, inside the eroded valid area v1, not touching its
    border), keeping those of at least min_area px.  With return_map, also a
    boolean image of those loop interiors."""
    holes, nh = ndi.label(~S & v1)            # 4-connected background regions
    keep = np.zeros(0, int)
    if nh:
        sizes = np.bincount(holes.ravel())[1:]
        border = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1],
                                 holes[~ndi.binary_erosion(v1, iterations=2) & v1]])
        inner = np.setdiff1d(np.arange(1, nh + 1), border)
        la = sizes[inner - 1] if len(inner) else np.zeros(0)
        keep = inner[la >= min_area] if len(inner) else keep
        la = la[la >= min_area]
    else:
        la = np.zeros(0)
    if not return_map:
        return la
    lut = np.zeros(nh + 1, bool)
    lut[keep] = True
    return la, lut[holes]


# ------------------------------------------------------ skeleton topology
def skeleton_pixel_graph(S):
    """Pixel graph of a thin skeleton: 8-adjacency minus the diagonal
    shortcuts of staircase corners (a diagonal link is dropped when the two
    pixels are also joined through a shared 4-neighbour), so a pixel on a
    line has exactly 2 neighbours and only real junctions have 3 or more.

    Returns yx (N, 2) pixel coordinates (row-major order) and nbrs (N, 8)
    neighbour indices in _RING order (-1 = none).
    """
    S = np.asarray(S, bool)
    yx = np.argwhere(S)
    idx = np.full(S.shape, -1, np.int64)
    idx[yx[:, 0], yx[:, 1]] = np.arange(len(yx))
    P = np.pad(S, 1)
    Pi = np.pad(idx, 1, constant_values=-1)
    y, x = yx[:, 0] + 1, yx[:, 1] + 1
    nbrs = np.full((len(yx), 8), -1, np.int64)
    for k, (dy, dx) in enumerate(_RING):
        q = Pi[y + dy, x + dx]
        if dy and dx:
            q = np.where(P[y + dy, x] | P[y, x + dx], -1, q)
        nbrs[:, k] = q
    return yx, nbrs


def _arc(c):
    """Cumulative arc length along a polyline (n, 2)."""
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))] if len(c) > 1 else np.zeros(len(c))


def trace_skeleton(S, merge_len=6.0):
    """Branches and nodes of a skeleton.

    Junction pixels (>= 3 neighbours in skeleton_pixel_graph) are grouped
    into clusters; clusters joined by a branch no longer than merge_len px
    are merged into one node (the two 3-way junctions a skeleton makes where
    two vessels cross become one 4-arm node).  Endpoints are nodes too.

    Returns dict:
        yx        (N, 2) skeleton pixel coordinates
        branches  list of int arrays: ordered pixel indices from end 0 to end 1
                  (closed rings with no node repeat their first pixel)
        ends      (nb, 2) node id of each branch end (-1 = none: a ring)
        node_yx   (n_nodes, 2) node centroids
        node_kind (n_nodes,) 1 = junction, 0 = endpoint
        arms      {node id: [(branch, side), ...]} for junction nodes
    """
    yx, nbrs = skeleton_pixel_graph(S)
    N = len(yx)
    deg = (nbrs >= 0).sum(1)
    nb = nbrs.tolist()
    dg = deg.tolist()
    jmask = np.zeros(np.shape(S), bool)
    jmask[yx[deg >= 3, 0], yx[deg >= 3, 1]] = True
    jl, nj = ndi.label(jmask, structure=np.ones((3, 3)))
    pnode = np.full(N, -1, np.int64)
    pnode[deg >= 3] = jl[yx[deg >= 3, 0], yx[deg >= 3, 1]] - 1
    endpix = np.flatnonzero(deg == 1)
    pnode[endpix] = nj + np.arange(len(endpix))
    n_nodes = nj + len(endpix)
    vis = [[False] * 8 for _ in range(N)]

    def walk(p, k):
        path = [p]
        prev, cur = p, nb[p][k]
        vis[p][k] = True
        vis[cur][(k + 4) % 8] = True
        while True:
            path.append(cur)
            if dg[cur] != 2:
                break
            k2 = next((kk for kk in range(8) if nb[cur][kk] >= 0 and nb[cur][kk] != prev), -1)
            if k2 < 0 or vis[cur][k2]:
                break
            q = nb[cur][k2]
            vis[cur][k2] = True
            vis[q][(k2 + 4) % 8] = True
            prev, cur = cur, q
        return path

    raw = []
    for p in np.flatnonzero(deg != 2).tolist():
        for k in range(8):
            if nb[p][k] >= 0 and not vis[p][k]:
                raw.append(walk(p, k))
    for p in np.flatnonzero(deg == 2).tolist():           # rings without nodes
        if not any(vis[p]):
            k = next(kk for kk in range(8) if nb[p][kk] >= 0)
            raw.append(walk(p, k))

    # merge junction clusters joined by short branches (union-find)
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    branches, ends = [], []
    for path in raw:
        a, b = int(pnode[path[0]]), int(pnode[path[-1]])
        if a >= 0 and a < nj and b >= 0 and b < nj:
            if len(path) == 2 and a == b:
                continue                                  # link inside a cluster
            if a != b and _arc(yx[path].astype(float))[-1] <= merge_len:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                continue
        branches.append(np.asarray(path, np.int64))
        ends.append((a, b))
    ends = np.array(ends, np.int64).reshape(-1, 2)
    root = np.array([find(a) for a in range(n_nodes)], np.int64)
    ends = np.where(ends >= 0, root[np.maximum(ends, 0)], -1)
    # node centroids and arms
    acc = np.zeros((n_nodes, 3))
    np.add.at(acc, root[pnode[deg >= 3]], np.c_[yx[deg >= 3], np.ones((deg >= 3).sum())])
    np.add.at(acc, root[pnode[endpix]], np.c_[yx[endpix], np.ones(len(endpix))])
    node_yx = acc[:, :2] / np.maximum(acc[:, 2:], 1)
    node_kind = (np.arange(n_nodes) < nj).astype(int)
    arms = {}
    for bi, (a, b) in enumerate(ends):
        for side, n in ((0, a), (1, b)):
            if n >= 0 and node_kind[n]:
                arms.setdefault(int(n), []).append((bi, side))
    return dict(yx=yx, branches=branches, ends=ends, node_yx=node_yx, node_kind=node_kind, arms=arms)


def _arm_angle(c, lam):
    """Direction (radians, image frame: atan2(dy, dx)) of a branch leaving a
    node, from arc 0.25 lambda to 1 lambda (skipping the skeleton's bend into
    the junction); c is ordered away from the node."""
    c = c.astype(float)
    d = _arc(c)
    L = d[-1]
    if L <= 0:
        return float("nan")
    d1 = min(lam, L)
    d0 = min(0.25 * lam, 0.3 * L)
    p0 = np.array([np.interp(d0, d, c[:, 0]), np.interp(d0, d, c[:, 1])])
    p1 = np.array([np.interp(d1, d, c[:, 0]), np.interp(d1, d, c[:, 1])])
    v = p1 - p0
    if np.hypot(*v) < 1e-6:
        v = c[-1] - c[0]
    return float(math.atan2(v[0], v[1]))


def junction_geometry(net, lam, link_dev=40.0, cross_dev=35.0):
    """Angles at junction nodes, and good-continuation links for paths.

    3-arm nodes: the angular gaps between consecutive arms (sum 360 deg);
        min gap = the branching angle (between the two daughters of a Y, or
        side branch to through vessel of a T); 180 - max gap = how far the
        straightest pair is from collinear (0 for a T).
    4-arm nodes: a crossing when both opposite pairs are within cross_dev deg
        of collinear; crossing angle = acute angle between the two lines.
    Links: at every node, arm pairs are joined greedily in order of
        deviation from collinear, up to link_dev deg.

    Returns dict(min_gap, straight_dev (deg arrays over 3-arm nodes),
    cross_angle (deg, per crossing), cross_yx, n_arms (per junction node),
    link {(branch, side): (branch, side)}).
    """
    yx, br = net["yx"], net["branches"]
    min_gap, straight, cross, cross_yx, n_arms = [], [], [], [], []
    link = {}
    for n, arms in net["arms"].items():
        ang = []
        for bi, side in arms:
            c = yx[br[bi]]
            ang.append(_arm_angle(c if side == 0 else c[::-1], lam))
        ang = np.asarray(ang)
        ok = np.isfinite(ang)
        arms = [a for a, o in zip(arms, ok) if o]
        ang = ang[ok]
        m = len(ang)
        n_arms.append(m)
        if m == 3:
            a = np.sort(ang % (2 * np.pi))
            gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
            min_gap.append(np.degrees(gaps.min()))
            straight.append(180.0 - np.degrees(gaps.max()))
        elif m == 4:
            a = np.sort(ang % (2 * np.pi))
            dev13 = abs(np.pi - (a[2] - a[0]))
            dev24 = abs(np.pi - (a[3] - a[1]))
            if max(dev13, dev24) <= np.radians(cross_dev):
                tA = 0.5 * np.angle(np.exp(2j * a[0]) + np.exp(2j * a[2]))
                tB = 0.5 * np.angle(np.exp(2j * a[1]) + np.exp(2j * a[3]))
                d = abs(tA - tB) % np.pi
                cross.append(np.degrees(min(d, np.pi - d)))
                cross_yx.append(net["node_yx"][n])
        if m >= 2:
            u = np.stack([np.sin(ang), np.cos(ang)], 1)          # (dy, dx) unit vectors
            pairs = []
            for i in range(m):
                for j in range(i + 1, m):
                    th = math.acos(float(np.clip(u[i] @ u[j], -1, 1)))
                    pairs.append((math.pi - th, i, j))
            used = set()
            for dev, i, j in sorted(pairs):
                if dev > math.radians(link_dev):
                    break
                if i in used or j in used:
                    continue
                used.update((i, j))
                link[arms[i]] = arms[j]
                link[arms[j]] = arms[i]
    return dict(min_gap=np.array(min_gap), straight_dev=np.array(straight), cross_angle=np.array(cross),
                cross_yx=np.array(cross_yx).reshape(-1, 2), n_arms=np.array(n_arms, int), link=link)


def link_paths(net, link):
    """Chains of branches joined through nodes by the good-continuation
    links: the image's best guess at continuous vessels.  Returns a list of
    (pixel-index array in path order, [branch ids])."""
    br = net["branches"]
    used = set()
    paths = []

    def follow(b, s_in):
        seq, pix = [], []
        while True:
            used.add(b)
            p = br[b] if s_in == 0 else br[b][::-1]
            pix.append(p)
            seq.append(b)
            nxt = link.get((b, 1 - s_in))
            if nxt is None or nxt[0] in used:
                break
            b, s_in = nxt
        return np.concatenate(pix), seq

    for b in range(len(br)):
        if b in used:
            continue
        for s in (0, 1):
            if (b, s) not in link:
                paths.append(follow(b, s))
                break
    for b in range(len(br)):                      # closed chains
        if b not in used:
            paths.append(follow(b, 0))
    return paths


def bridge_paths(yx, paths, lam, shape, r_gap=3.0, max_dev=30.0):
    """Join path ends across gaps by good continuation, and find crossings.

    Where two vessels cross, the ridge detector usually breaks one of them
    (the crossing point is blob-like, not line-like), so the skeleton shows
    a gap or two offset T-junctions instead of an X.  Two path ends a, b
    within r_gap * lambda are joined when their outward directions are
    antiparallel within max_dev deg, the lateral offset of b from a's ray is
    at most 0.5 lambda + |d| sin(max_dev), and neither lies more than 0.5
    lambda behind the other; candidates are accepted greedily by cost.
    A join is a crossing when the straight bridge passes over the skeleton
    of another path; its angle is the acute angle between the joined
    vessel's direction and the crossed path's local direction.

    Returns dict(link {(path, side): (path, side)}, cross_angle (deg array),
    cross_yx (n, 2), n_bridges).
    """
    ends = []
    for i, (p, _) in enumerate(paths):
        c = yx[p].astype(float)
        if len(c) < 3 or _arc(c)[-1] < 0.5 * lam or np.all(c[0] == c[-1]):
            continue
        for side, cc in ((0, c), (1, c[::-1])):
            a = _arm_angle(cc, lam)                     # pointing into the path
            if np.isfinite(a):
                ends.append((i, side, cc[0], -np.array([math.sin(a), math.cos(a)])))
    out = dict(link={}, cross_angle=np.zeros(0), cross_yx=np.zeros((0, 2)), n_bridges=0)
    if len(ends) < 2:
        return out
    P = np.array([e[2] for e in ends])
    U = np.array([e[3] for e in ends])
    cos_max = math.cos(math.radians(max_dev))
    cand = []
    for a, b in cKDTree(P).query_pairs(r_gap * lam):
        if ends[a][0] == ends[b][0]:
            continue
        if U[a] @ -U[b] < cos_max:                     # not antiparallel
            continue
        d = P[b] - P[a]
        dist = float(np.hypot(*d))
        along_a, along_b = float(d @ U[a]), float(-d @ U[b])
        if min(along_a, along_b) < -0.5 * lam:
            continue
        lat = max(abs(d[0] * U[a][1] - d[1] * U[a][0]), abs(d[0] * U[b][1] - d[1] * U[b][0]))
        if lat > 0.5 * lam + dist * math.sin(math.radians(max_dev)):
            continue
        dev = math.degrees(math.acos(float(np.clip(U[a] @ -U[b], -1, 1))))
        cand.append((lat / lam + dev / max_dev + dist / (r_gap * lam), a, b))
    owner = np.zeros(shape, np.int64)                  # path id + 1 per skeleton pixel
    for i, (p, _) in enumerate(paths):
        owner[yx[p, 0], yx[p, 1]] = i + 1
    owner = ndi.grey_dilation(owner, size=(3, 3))
    used, link, cross, cyx = set(), {}, [], []
    for _, a, b in sorted(cand):
        if a in used or b in used:
            continue
        used.update((a, b))
        ia, sa = ends[a][:2]
        ib, sb = ends[b][:2]
        link[(ia, sa)] = (ib, sb)
        link[(ib, sb)] = (ia, sa)
        n = max(2, int(math.ceil(2 * np.hypot(*(P[b] - P[a])))) + 1)
        t = np.linspace(0, 1, n)[:, None]
        q = np.rint(P[a] + t * (P[b] - P[a])).astype(int)
        hit = owner[q[:, 0], q[:, 1]] - 1
        k = np.flatnonzero((hit >= 0) & (hit != ia) & (hit != ib))
        if not len(k):
            continue
        h, j = q[k[len(k) // 2]], hit[k[len(k) // 2]]
        cj = yx[paths[j][0]].astype(float)
        m = int(np.argmin(np.hypot(*(cj - h).T)))
        s = _arc(cj)
        w = np.abs(s - s[m]) <= 0.5 * lam
        seg = cj[w]
        if len(seg) < 3:
            continue
        tj = seg[-1] - seg[0]
        ua = U[a] - U[b]                               # the joined vessel's direction
        ang = abs(math.atan2(ua[0], ua[1]) - math.atan2(tj[0], tj[1])) % math.pi
        cross.append(math.degrees(min(ang, math.pi - ang)))
        cyx.append(h)
    out.update(link=link, cross_angle=np.array(cross), cross_yx=np.array(cyx, float).reshape(-1, 2),
               n_bridges=len(link) // 2)
    return out


def resample_path(c, step=1.0, smooth=1.5):
    """A pixel chain smoothed along its length (Gaussian, `smooth` samples,
    removing the pixel staircase) and resampled every `step` px of arc."""
    c = np.asarray(c, float)
    keep = np.r_[True, np.any(np.diff(c, axis=0) != 0, axis=1)]
    c = c[keep]
    if len(c) < 3:
        return c
    c = ndi.gaussian_filter1d(c, smooth, axis=0, mode="nearest")
    d = _arc(c)
    if d[-1] < step:
        return c[[0, -1]]
    s = np.arange(0.0, d[-1] + 1e-9, step)
    return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], 1)


def tortuosity_windows(paths_xy, lam, lams=TORT_LAMS):
    """Arc/chord over windows of c*lambda of arc (c in lams) slid along each
    resampled path (step 1 px) in quarter-window steps.  Returns
    {c: array of ratios}; windows longer than a path are not formed."""
    out = {}
    for c in lams:
        m = max(2, int(round(c * lam)))
        step = max(1, m // 4)
        r = []
        for q in paths_xy:
            if len(q) <= m:
                continue
            i = np.arange(0, len(q) - m, step)
            chord = np.hypot(*(q[i + m] - q[i]).T)
            r.append(m / np.maximum(chord, 1e-6))
        out[c] = np.concatenate(r) if r else np.zeros(0)
    return out


# ------------------------------------------------------------------ profiles
def profiles(Fs, pts, phi, sig, noise, rng, n_max=4000):
    """Cross-section profiles at skeleton points: contrast, FWHM, edge width.

    contrast = lower shoulder - minimum; FWHM at half contrast; edge width
    = c / (sqrt(2 pi) max slope), which is sigma for a blurred step; sharp =
    FWHM / edge (3.58 for a Gaussian profile); asym = shoulder asymmetry;
    idx = index of the profile's point in `pts`.
    """
    if len(pts) > n_max:
        pts_i = rng.choice(len(pts), n_max, replace=False)
    else:
        pts_i = np.arange(len(pts))
    out = []
    step = 0.5
    for i in pts_i:
        y, x = pts[i]
        s = sig[i]
        Lp = max(10.0, 4 * s + 4)
        t = np.arange(-Lp, Lp + 1e-9, step)
        nx, ny = math.cos(phi[i]), math.sin(phi[i])
        cy, cx = y + t * ny, x + t * nx
        p = ndi.map_coordinates(Fs, [cy, cx], order=1, mode="nearest")
        n = len(t)
        mid = n // 2
        w = int(max(2.0, s) / step)
        i0 = mid - w + int(np.argmin(p[mid - w: mid + w + 1]))
        pmin = p[i0]
        tol = max(3 * noise, 0.002)

        def shoulder(direction):
            m, im = pmin, i0
            j = i0
            while 0 <= j + direction < n:
                j += direction
                if p[j] > m:
                    m, im = p[j], j
                elif p[j] < m - max(0.2 * (m - pmin), tol):
                    break
            return m, im

        bl, il = shoulder(-1)
        br, ir = shoulder(+1)
        c = min(bl, br) - pmin
        if c <= max(0.01, 4 * noise) or il == i0 or ir == i0:
            continue
        h = pmin + 0.5 * c
        # half crossings
        jl = i0
        while jl > il and p[jl] < h:
            jl -= 1
        jr = i0
        while jr < ir and p[jr] < h:
            jr += 1
        if p[jl] < h or p[jr] < h:
            continue
        # sub-sample interpolation
        tl = t[jl] + (h - p[jl]) / (p[jl + 1] - p[jl] + 1e-12) * step
        tr = t[jr - 1] + (h - p[jr - 1]) / (p[jr] - p[jr - 1] + 1e-12) * step
        fwhm = tr - tl
        dp = np.gradient(p, step)
        gl = -dp[il:i0 + 1].min()
        gr = dp[i0:ir + 1].max()
        if gl <= 0 or gr <= 0:
            continue
        el = (bl - pmin) / (math.sqrt(2 * math.pi) * gl)
        er = (br - pmin) / (math.sqrt(2 * math.pi) * gr)
        e = 0.5 * (el + er)
        out.append((s, c, fwhm, e, fwhm / e, abs(bl - br) / (max(bl, br) - pmin), i))
    a = np.array(out, float).reshape(-1, 7)
    return dict(scale=a[:, 0], contrast=a[:, 1], fwhm=a[:, 2], edge=a[:, 3], sharp=a[:, 4], asym=a[:, 5],
                idx=a[:, 6].astype(np.int64))


def sharp_contrast(pr, lam, n_bins=5, min_n=60):
    """Contrast against width among the sharpest third of the profiles: in
    each FWHM quintile the third with the largest FWHM / edge (least blur for
    their width).  Blur lowers contrast, so only in-focus vessels show the
    absorption law: contrast rising with width is a Beer-Lambert column,
    flat is a scattering-limited shadow.  Returns slope of ln c on ln FWHM,
    Spearman rho, and median contrast at FWHM < 0.8, 0.8-1.6 and > 1.6
    lambda (NaN with fewer than 10 profiles in a bin)."""
    nan = float("nan")
    res = dict(sharp3_c_slope=nan, sharp3_rho=nan, sharp3_c_small=nan, sharp3_c_mid=nan, sharp3_c_large=nan)
    fw, c, sh = pr["fwhm"], pr["contrast"], pr["sharp"]
    if len(fw) < min_n:
        return res
    q = np.quantile(fw, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(q, fw, side="right") - 1, 0, n_bins - 1)
    sel = np.zeros(len(fw), bool)
    for i in range(n_bins):
        m = b == i
        if m.sum() >= 9:
            sel |= m & (sh >= np.quantile(sh[m], 2 / 3))
    if sel.sum() < 20:
        return res
    res["sharp3_c_slope"] = float(np.polyfit(np.log(fw[sel]), np.log(c[sel]), 1)[0])
    res["sharp3_rho"] = float(sst.spearmanr(fw[sel], c[sel])[0])
    for name, lo, hi in (("small", 0.0, 0.8), ("mid", 0.8, 1.6), ("large", 1.6, np.inf)):
        m = sel & (fw >= lo * lam) & (fw < hi * lam)
        res[f"sharp3_c_{name}"] = float(np.median(c[m])) if m.sum() >= 10 else nan
    return res


def along_path_cv(pr, pix_path, min_n=6):
    """Robust CV (1.4826 MAD / median) of contrast and FWHM along each linked
    path with at least min_n profiles; returns the per-path arrays."""
    pid = pix_path[pr["idx"]] if len(pr["idx"]) else np.zeros(0, int)
    out_c, out_w = [], []
    ok = pid >= 0
    if ok.sum() == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(pid[ok], kind="stable")
    ids, c, w = pid[ok][order], pr["contrast"][ok][order], pr["fwhm"][ok][order]
    uniq, start, cnt = np.unique(ids, return_index=True, return_counts=True)
    for st, n in zip(start, cnt):
        if n < min_n:
            continue
        for v, out in ((c[st:st + n], out_c), (w[st:st + n], out_w)):
            med = np.median(v)
            out.append(1.4826 * np.median(np.abs(v - med)) / med)
    return np.array(out_c), np.array(out_w)


# ------------------------------------------------------------------ faint mesh
def faint_mesh(F, valid, M, v1, sig, lam, sigmas=(1.5, 2.0, 3.0), c_floor=0.02, k_hi=2.5, k_lo=1.25,
               window=16.0):
    """Faint-vessel detector and the loops (meshes) it closes.

    Whitened ridges: at each sigma the dark-line strength D is divided by
    N = the local RMS of the opposite-polarity response Bm (texture + noise;
    Gaussian window `window` px, valid pixels only), floored at
    0.354 c_floor / k_hi so that in quiet regions the strong threshold is
    the matched response of a c_floor Np line.  W = max over sigmas of D / N
    is hysteresis-thresholded (seeds W > k_hi, grown through W > k_lo; the
    weak level reaches ~0.01 Np lines connected to a seed), components
    smaller than 2 lambda px are dropped, and the union with the main
    detection M (so wide vessels stay solid instead of splitting into edge
    pairs) is skeletonised and pruned like the main one.

    Returns dict(S = faint skeleton, loops = loop areas (px^2), loop_map,
    extra = skeleton px farther than 2 px from M).
    """
    from skimage.filters import apply_hysteresis_threshold
    Wm = np.zeros(F.shape)
    for s in sigmas:
        D, Bm, _ = hessian(F, s)
        vm = eroded_valid(valid, 3 * s + 1)
        num = ndi.gaussian_filter(np.where(vm, Bm * Bm, 0.0), window)
        den = ndi.gaussian_filter(vm.astype(float), window)
        N = np.maximum(np.sqrt(num / np.maximum(den, 1e-6)), 0.354 * c_floor / k_hi)
        Wm = np.maximum(Wm, np.where(vm, D / N, 0.0))
    det = apply_hysteresis_threshold(Wm, k_lo, k_hi)
    det = remove_small_objects(det, max_size=int(max(19, 2 * lam)))
    Sf = prune_spurs(skeletonize(det | M) & v1, sig)
    la, lmap = enclosed_loops(Sf, v1, return_map=True)
    extra = Sf & ~ndi.binary_dilation(M, iterations=2)
    return dict(S=Sf, loops=la, loop_map=lmap, extra=extra)


# ------------------------------------------------------------------ pairs, spectra
def parallel_pairs(pts, th, area, rng, rmax=40.0, umax=2.0, dth=np.deg2rad(15), n_q=15000):
    """Pair correlation of skeleton points with a PARALLEL neighbour at
    perpendicular distance v (|along offset| <= umax): observed / random
    expectation for the image's own skeleton density and orientation
    distribution.  1 = no preference; >1 = parallel companions."""
    N = len(pts)
    if N < 50:
        return None
    xy = pts[:, ::-1].astype(float)                   # (x, y)
    tree = cKDTree(xy)
    q = rng.choice(N, min(n_q, N), replace=False)
    nbrs = tree.query_ball_point(xy[q], rmax)
    ii = np.repeat(q, [len(v) for v in nbrs])
    jj = np.concatenate([np.asarray(v, int) for v in nbrs])
    d = xy[jj] - xy[ii]
    t = np.stack([np.cos(th[ii]), np.sin(th[ii])], 1)
    u = (d * t).sum(1)
    v = np.abs(d[:, 0] * -t[:, 1] + d[:, 1] * t[:, 0])
    dt = np.abs(th[ii] - th[jj]) % np.pi
    dt = np.minimum(dt, np.pi - dt)
    edges = np.arange(2.0, rmax + 0.01, 2.0)
    sel = (np.abs(u) <= umax) & (v >= 2.0)
    h_par, _ = np.histogram(v[sel & (dt < dth)], edges)
    h_perp, _ = np.histogram(v[sel & (dt > np.deg2rad(60))], edges)
    rho = N / area
    # orientation-match probabilities from random point pairs
    a = rng.integers(0, N, 20000)
    b = rng.integers(0, N, 20000)
    dd = np.abs(th[a] - th[b]) % np.pi
    dd = np.minimum(dd, np.pi - dd)
    p_par = float((dd < dth).mean())
    p_perp = float((dd > np.deg2rad(60)).mean())
    width = np.diff(edges)
    exp_par = len(q) * rho * (2 * (2 * umax + 1)) * width * p_par
    exp_perp = len(q) * rho * (2 * (2 * umax + 1)) * width * p_perp
    return dict(v=0.5 * (edges[1:] + edges[:-1]), g_par=h_par / np.maximum(exp_par, 1e-9),
                g_perp=h_perp / np.maximum(exp_perp, 1e-9), p_par=p_par)


def radial_psd(F, valid, tile=256):
    """Radially averaged power spectrum of F from half-overlapping Hann-
    windowed tiles that are fully valid; 20 log-spaced bins 2/tile..0.5
    cycles/px."""
    H, W = F.shape
    tile = min(tile, 1 << int(math.floor(math.log2(min(H, W)))))
    win = np.outer(np.hanning(tile), np.hanning(tile))
    wnorm = (win ** 2).sum()
    acc, n = 0, 0
    step = tile // 2
    for y in range(0, H - tile + 1, step):
        for x in range(0, W - tile + 1, step):
            if valid[y:y + tile, x:x + tile].mean() < 0.99:
                continue
            p = F[y:y + tile, x:x + tile]
            p = p - p.mean()
            acc = acc + np.abs(np.fft.fft2(p * win)) ** 2 / wnorm
            n += 1
    if n == 0:
        return None
    P = np.fft.fftshift(acc / n)
    fy = np.fft.fftshift(np.fft.fftfreq(tile))
    fr = np.hypot(*np.meshgrid(fy, fy, indexing="ij"))
    bins = np.geomspace(2.0 / tile, 0.5, 21)
    idx = np.digitize(fr.ravel(), bins)
    ps = np.array([P.ravel()[idx == k].mean() if (idx == k).any() else np.nan for k in range(1, len(bins))])
    fc = np.sqrt(bins[1:] * bins[:-1])
    return dict(f=fc, psd=ps, tile=tile, n_tiles=n)


def psd_slope(f, p, lo, hi):
    """Negative log-log slope of the PSD between frequencies lo and hi."""
    m = (f >= lo) & (f <= hi) & np.isfinite(p) & (p > 0)
    if m.sum() < 2:
        return float("nan")
    return float(-np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0])


def _q(a, q):
    return float(np.quantile(a, q)) if len(a) else float("nan")


# ---------------------------------------------------------------- analyse
def analyse(I, valid=None, seed=0, keep_maps=False):
    """All statistics of one image I (linear intensity, any unit; NaN or
    <= 0 = no data unless `valid` is given).  Returns a dict of scalars,
    small arrays (profiles, pair curves, spectrum, per-scale dicts) and, with
    keep_maps, the maps used for overlays (F, B, M, S, k, Rmax, valid, faint
    skeleton, loop map, linked paths).  `scalars(res)` flattens it.

    The suite v1 part (everything up to the spectrum) reproduces imstats.py
    exactly for the same seed; the section 7.1 additions use no randomness
    and are computed after it.
    """
    rng = np.random.default_rng(seed)
    F, B, valid = flatten(I, valid)
    H, W = F.shape
    res = {"shape": (H, W)}
    v1 = eroded_valid(valid, 3)
    area = float(v1.sum())
    res["valid_frac"] = float(valid.mean())
    fv = F[v1]
    res["F_p01"], res["F_p05"], res["F_p50"], res["F_p95"] = [float(np.percentile(fv, q)) for q in (1, 5, 50, 95)]
    res["F_std"] = float(fv.std())
    res["F_skew"] = float(sst.skew(fv))
    res["dark_frac_0.1"] = float((fv < np.median(fv) - 0.1).mean())
    res["noise"] = noise_sigma(np.where(valid, F, 0.0))
    # ---- ridges per scale
    Ds, phis = [], []
    per = {}
    for s in SCALES:
        D, Bm, phi = hessian(F, s)
        vm = eroded_valid(valid, 3 * s + 1)
        if vm.sum() < 1000:
            vm = v1
        d, b = D[vm], Bm[vm]
        T_tex = float(np.quantile(b, Q_TEX))
        T_abs = 0.354 * C0_ABS
        T = max(T_tex, T_abs)
        M = (D > T) & vm
        Ms = remove_small_objects(M, max_size=19)
        Sk = skeletonize(Ms)
        per[s] = dict(T_tex=T_tex, T=T,
                      frac_tex=float((d > T_tex).mean()), frac_abs=float((d > T_abs).mean()),
                      frac=float(M[vm].mean()),
                      skel_density=float(Sk[vm].sum() / vm.sum() * 1e3),
                      D_p50=float(np.median(d)), D_p90=float(np.quantile(d, 0.9)), D_p99=float(np.quantile(d, 0.99)),
                      dark_bright=float((d ** 2).mean() / max((b ** 2).mean(), 1e-20)))
        # detection-free orientation: doubled-angle mean of the dark-line direction weighted by D
        th = phi + np.pi / 2
        z = (d * np.exp(2j * th[vm])).sum() / max(d.sum(), 1e-20)
        per[s]["anis_D"] = float(abs(z))
        per[s]["dir_D_deg"] = float(np.rad2deg(np.angle(z) / 2) % 180)
        Ds.append(D / T)
        phis.append(phi)
    res["per_scale"] = per
    Rst = np.stack(Ds)
    k = np.argmax(Rst, 0)
    Rmax = np.take_along_axis(Rst, k[None], 0)[0]
    phi = np.take_along_axis(np.stack(phis), k[None], 0)[0]
    del Rst, Ds, phis
    sig = np.asarray(SCALES)[k]
    M = remove_small_objects((Rmax > 1) & v1, max_size=29)
    res["vessel_frac"] = float(M[v1].mean())
    res["vessel_frac_by_scale"] = {s: float((M & (k == i))[v1].mean()) for i, s in enumerate(SCALES)}
    S = prune_spurs(skeletonize(M) & v1, sig)
    g = skeleton_graph_stats(S, area)
    res.update({kk: g[kk] for kk in ("skel_density", "endpoint_density", "junction_density")})
    res["junction_per_endpoint"] = g["junctions"] / max(g["endpoints"], 1)
    bl = g["branch_len"]
    res["branch_len_med"] = float(np.median(bl)) if len(bl) else float("nan")
    res["branch_len_p90"] = float(np.quantile(bl, 0.9)) if len(bl) else float("nan")
    res["branch_tort_med"] = float(np.median(g["branch_tort"])) if len(g["branch_tort"]) else float("nan")
    res["branch_len"] = bl
    # mesh (closed loops of the skeleton) and spacing between vessels
    la = enclosed_loops(S, v1)
    res["loop_density"] = float(len(la) / area * 1e5)
    res["loop_area_med"] = float(np.median(la)) if len(la) else float("nan")
    res["loop_areas"] = la
    dist = ndi.distance_transform_edt(~M)
    dv = dist[v1 & ~M]
    res["dist_to_vessel_med"] = float(np.median(dv)) if len(dv) else float("nan")
    res["dist_to_vessel_p90"] = float(np.quantile(dv, 0.9)) if len(dv) else float("nan")
    # box-counting dimension of the vessel mask and of the skeleton (boxes 4..64 px)
    for name, X in (("fd_mask", M), ("fd_skel", S)):
        sizes_b, counts = [], []
        for b in (4, 8, 16, 32, 64):
            hh, ww = (X.shape[0] // b) * b, (X.shape[1] // b) * b
            if hh < b or ww < b:
                continue
            blk = X[:hh, :ww].reshape(hh // b, b, ww // b, b).any(axis=(1, 3))
            sizes_b.append(b)
            counts.append(max(int(blk.sum()), 1))
        res[name] = float(-np.polyfit(np.log(sizes_b), np.log(counts), 1)[0]) if len(sizes_b) >= 3 else float("nan")
    ys, xs = np.nonzero(S)
    pts = np.stack([ys, xs], 1)
    th = (phi[ys, xs] + np.pi / 2) % np.pi
    sk_sig = sig[ys, xs]
    res["skel_len_by_scale"] = {s: float((sk_sig == s).mean()) for s in SCALES}
    # orientation of skeleton (length weighted)
    z = np.exp(2j * th).mean()
    res["skel_anis"] = float(abs(z))
    res["skel_dir_deg"] = float(np.rad2deg(np.angle(z) / 2) % 180)
    for s in SCALES:
        m = sk_sig == s
        if m.sum() > 50:
            zz = np.exp(2j * th[m]).mean()
            per[s]["skel_anis"] = float(abs(zz))
            per[s]["skel_dir_deg"] = float(np.rad2deg(np.angle(zz) / 2) % 180)
        else:
            per[s]["skel_anis"] = float("nan")
            per[s]["skel_dir_deg"] = float("nan")
    res["orient_hist"] = np.histogram(th, np.linspace(0, np.pi, 13))[0] / max(len(th), 1)
    # parallel companions
    pp = parallel_pairs(pts, th, area, rng)
    res["pairs"] = pp
    if pp is not None:
        m = (pp["v"] >= 3) & (pp["v"] <= 30)
        res["par_excess_3_30"] = float(np.mean(pp["g_par"][m]))
        res["perp_excess_3_30"] = float(np.mean(pp["g_perp"][m]))
        res["par_peak_v"] = float(pp["v"][np.argmax(pp["g_par"])])
    # profiles
    Fs = ndi.gaussian_filter(F, 0.7)
    pr = profiles(Fs, pts, phi[ys, xs], sk_sig, res["noise"], rng)
    res["prof"] = pr
    for q, name in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        res[f"contrast_{name}"] = float(np.quantile(pr["contrast"], q)) if len(pr["contrast"]) else float("nan")
        res[f"fwhm_{name}"] = float(np.quantile(pr["fwhm"], q)) if len(pr["fwhm"]) else float("nan")
        res[f"edge_{name}"] = float(np.quantile(pr["edge"], q)) if len(pr["edge"]) else float("nan")
    res["sharp_p50"] = float(np.median(pr["sharp"])) if len(pr["sharp"]) else float("nan")
    res["n_profiles"] = int(len(pr["contrast"]))
    for s in SCALES:
        m = pr["scale"] == s
        per[s]["n_prof"] = int(m.sum())
        per[s]["contrast_med"] = float(np.median(pr["contrast"][m])) if m.sum() > 10 else float("nan")
        per[s]["fwhm_med"] = float(np.median(pr["fwhm"][m])) if m.sum() > 10 else float("nan")
        per[s]["edge_med"] = float(np.median(pr["edge"][m])) if m.sum() > 10 else float("nan")
        per[s]["sharp_med"] = float(np.median(pr["sharp"][m])) if m.sum() > 10 else float("nan")
    # contrast-width correlation
    if len(pr["contrast"]) > 20:
        res["rho_contrast_fwhm"] = float(sst.spearmanr(pr["contrast"], pr["fwhm"])[0])
        res["rho_edge_fwhm"] = float(sst.spearmanr(pr["edge"], pr["fwhm"])[0])
    # background texture away from vessels, per DoG band
    bg = {}
    for s in SCALES:
        band = ndi.gaussian_filter(F, s) - ndi.gaussian_filter(F, 2 * s)
        r = int(min(s + 3, 10))
        keep = v1 & ~ndi.binary_dilation(M, iterations=r)
        bg[s] = dict(rms=float(band[keep].std()) if keep.sum() > 500 else float("nan"),
                     frac=float(keep[v1].mean()))
    res["bg_band"] = bg
    res["bg_frac"] = float((v1 & ~ndi.binary_dilation(M, iterations=3))[v1].mean())
    # spectrum
    ps = radial_psd(F, valid)
    res["psd"] = ps
    if ps is not None:
        res["psd_slope_lo"] = psd_slope(ps["f"], ps["psd"], 1 / 128, 1 / 16)
        res["psd_slope_mid"] = psd_slope(ps["f"], ps["psd"], 1 / 16, 1 / 4)
        res["psd_slope_hi"] = psd_slope(ps["f"], ps["psd"], 1 / 4, 0.5)

    # ================= plan 7.1 additions (deterministic) =================
    lam = res["fwhm_p50"] if np.isfinite(res["fwhm_p50"]) else LAM_DEFAULT
    res["lam"] = float(lam)
    # (b) junction geometry and (d) tortuosity along linked paths
    net = trace_skeleton(S, merge_len=0.5 * lam)
    jg = junction_geometry(net, lam)
    yx = net["yx"]
    paths0 = link_paths(net, jg["link"])                    # continuous through junctions
    bj = bridge_paths(yx, paths0, lam, S.shape)            # ... and across gaps / crossings
    paths = [(p, sum((paths0[i][1] for i in seq), []))
             for p, seq in link_paths(dict(branches=[p for p, _ in paths0]), bj["link"])]
    pxy = [resample_path(yx[p]) for p, _ in paths]
    plen = np.array([_arc(q)[-1] if len(q) > 1 else 0.0 for q in pxy])
    tw = tortuosity_windows(pxy, lam)
    res["tort_windows"] = tw
    for c in TORT_LAMS:
        res[f"tort_{c}lam"] = float(np.median(tw[c])) if len(tw[c]) >= 10 else float("nan")
    res["tort_4lam_p90"] = float(np.quantile(tw[4], 0.9)) if len(tw[4]) >= 10 else float("nan")
    res["n_paths"] = int(len(paths))
    long = plen[plen >= 1.0]
    res["path_len_med_lam"] = float(np.median(long) / lam) if len(long) else float("nan")
    res["path_len_p90_lam"] = float(np.quantile(long, 0.9) / lam) if len(long) else float("nan")
    res["jangle_min"] = jg["min_gap"]
    res["jangle_straight"] = jg["straight_dev"]
    ca = np.r_[jg["cross_angle"], bj["cross_angle"]]
    cyx = np.r_[jg["cross_yx"], bj["cross_yx"]]
    res["cross_angles"] = ca
    res["jangle_min_med"] = _q(jg["min_gap"], 0.5)
    res["jangle_min_p10"] = _q(jg["min_gap"], 0.1)
    res["jangle_straight_med"] = _q(jg["straight_dev"], 0.5)
    res["frac_T"] = float((jg["straight_dev"] <= 30.0).mean()) if len(jg["straight_dev"]) else float("nan")
    na = jg["n_arms"]
    res["n_junction_nodes"] = int((na >= 3).sum())
    res["n_cross_x"] = int(len(jg["cross_angle"]))
    res["n_cross_gap"] = int(len(bj["cross_angle"]))
    res["n_bridges"] = int(bj["n_bridges"])
    res["cross_density"] = float(len(ca) / area * 1e5)
    res["cross_frac"] = float(len(ca) / max((na == 3).sum(), 1))
    res["node4_frac"] = float((na >= 4).sum() / max((na >= 3).sum(), 1))
    res["cross_angle_med"] = _q(ca, 0.5) if len(ca) >= 3 else float("nan")
    res["cross_angle_p25"] = _q(ca, 0.25) if len(ca) >= 3 else float("nan")
    # (f) contrast law among the sharpest third
    res.update(sharp_contrast(pr, lam))
    # (e) variability along linked paths
    pix_path = np.full(len(yx), -1, np.int64)
    for i, (p, _) in enumerate(paths):
        if plen[i] >= 2 * lam:
            pix_path[p] = i
    # profile idx refers to rows of `pts` (np.nonzero order), the same order as net["yx"]
    cvc, cvw = along_path_cv(pr, pix_path)
    res["along_cv_contrast"] = float(np.median(cvc)) if len(cvc) >= 5 else float("nan")
    res["along_cv_fwhm"] = float(np.median(cvw)) if len(cvw) >= 5 else float("nan")
    res["n_along"] = int(len(cvc))
    # (a) faint mesh
    fm = faint_mesh(F, valid, M, v1, sig, lam)
    fl = fm["loops"]
    res["faint_loop_density"] = float(len(fl) / area * 1e5)
    res["faint_loop_area_med_lam"] = float(np.median(fl) / lam ** 2) if len(fl) else float("nan")
    res["faint_skel_density"] = float(fm["S"].sum() / area * 1e3)
    res["faint_extra_density"] = float(fm["extra"].sum() / area * 1e3)
    res["faint_loop_areas"] = fl
    if keep_maps:
        res["maps"] = dict(F=F, B=B, M=M, S=S, k=k, Rmax=Rmax, valid=valid, faint_S=fm["S"],
                           faint_extra=fm["extra"], faint_loops=fm["loop_map"],
                           paths=[yx[p] for p, _ in paths], cross_yx=cyx)
    return res


# ---------------------------------------------------------------- scalars
def _num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def scalars(res):
    """Flatten an analyse() result into {key: float}.  Nested values get
    prefixed keys: 'ps:<sigma>:<name>' (per ridge scale), 'vf:<s>' (vessel
    area fraction whose best scale is s), 'sl:<s>' (share of skeleton length
    at best scale s), 'bg:<s>' (background DoG RMS in band s..2s px), and
    'lam:<name>' for statistics normalised by lambda = fwhm_p50."""
    out = {k: float(v) for k, v in res.items() if _num(v)}
    for s, d in res.get("per_scale", {}).items():
        for kk, v in d.items():
            if _num(v):
                out[f"ps:{float(s):g}:{kk}"] = float(v)
    for s, v in res.get("vessel_frac_by_scale", {}).items():
        out[f"vf:{float(s):g}"] = float(v)
    for s, v in res.get("skel_len_by_scale", {}).items():
        out[f"sl:{float(s):g}"] = float(v)
    for s, d in res.get("bg_band", {}).items():
        out[f"bg:{float(s):g}"] = float(d["rms"])
    lam = res.get("fwhm_p50", float("nan"))
    g = out.get
    nan = float("nan")
    out.update({
        "lam:skel_density": g("skel_density", nan) * 1e-3 * lam,
        "lam:junction_density": g("junction_density", nan) * 1e-5 * lam ** 2,
        "lam:endpoint_density": g("endpoint_density", nan) * 1e-5 * lam ** 2,
        "lam:loop_density": g("loop_density", nan) * 1e-5 * lam ** 2,
        "lam:faint_loop_density": g("faint_loop_density", nan) * 1e-5 * lam ** 2,
        "lam:cross_density": g("cross_density", nan) * 1e-5 * lam ** 2,
        "lam:dist_to_vessel_med": g("dist_to_vessel_med", nan) / lam,
        "lam:branch_len_med": g("branch_len_med", nan) / lam,
        "lam:fwhm_p10": g("fwhm_p10", nan) / lam,
        "lam:fwhm_p90": g("fwhm_p90", nan) / lam,
        "lam:edge_p10": g("edge_p10", nan) / lam,
        "lam:edge_p50": g("edge_p50", nan) / lam,
    })
    return out


# key -> (label, tier).  Tier 1 = image formation (px), 2 = vessel population,
# 3 = organisation and topology.  "new" marks the plan 7.1 additions.
SC = tuple(f"{s:g}" for s in SCALES)
LABELS = {
    # ---- tier 1
    "noise": ("pixel noise (Np, Laplacian MAD)", 1),
    "F_std": ("std of flattened log image F", 1),
    "F_skew": ("skewness of F", 1),
    "F_p01": ("1st percentile of F (Np)", 1),
    "dark_frac_0.1": ("frac. pixels > 0.1 Np darker than median", 1),
    "psd_slope_lo": ("PSD slope 1/128-1/16 c/px", 1),
    "psd_slope_mid": ("PSD slope 1/16-1/4 c/px", 1),
    "psd_slope_hi": ("PSD slope 1/4-1/2 c/px", 1),
    "contrast_p10": ("profile contrast p10 (Np)", 1),
    "contrast_p50": ("profile contrast p50 (Np)", 1),
    "contrast_p90": ("profile contrast p90 (Np)", 1),
    "fwhm_p10": ("profile FWHM p10 (px)", 1),
    "fwhm_p50": ("profile FWHM p50 = lambda (px)", 1),
    "fwhm_p90": ("profile FWHM p90 (px)", 1),
    "edge_p10": ("edge width p10 (px)", 1),
    "edge_p50": ("edge width p50 (px)", 1),
    "sharp_p50": ("FWHM / edge width, median", 1),
    "rho_edge_fwhm": ("Spearman(edge width, FWHM)", 1),
    "vf:1": ("vessel area frac, sigma*=1", 1),
    "sl:1": ("share of skeleton length, sigma*=1", 1),
    "ps:1:frac_abs": ("frac D>abs thr (0.03 Np), sigma=1", 1),
    "ps:1:skel_density": ("skeleton px/1e3 px2, sigma=1", 1),
    **{f"bg:{s}": (f"background DoG RMS, band {s}-{2 * float(s):g} px (Np)", 1) for s in SC},
    **{f"ps:{s}:dark_bright": (f"dark/bright ridge energy, sigma={s}", 1) for s in SC},
    # ---- tier 2
    "vessel_frac": ("vessel area fraction (multi-scale)", 2),
    **{f"vf:{s}": (f"vessel area frac, sigma*={s}", 2) for s in SC[1:]},
    **{f"sl:{s}": (f"share of skeleton length, sigma*={s}", 2) for s in SC[1:]},
    **{f"ps:{s}:frac_abs": (f"frac D>abs thr (0.03 Np), sigma={s}", 2) for s in SC[1:]},
    **{f"ps:{s}:skel_density": (f"skeleton px/1e3 px2, sigma={s}", 2) for s in SC[1:]},
    "skel_density": ("skeleton length density (px/1e3 px2)", 2),
    "lam:skel_density": ("skeleton length density x lambda", 2),
    "dist_to_vessel_med": ("median distance to nearest vessel (px)", 2),
    "lam:dist_to_vessel_med": ("median distance to nearest vessel / lambda", 2),
    "lam:fwhm_p10": ("FWHM p10 / lambda", 2),
    "lam:fwhm_p90": ("FWHM p90 / lambda", 2),
    "lam:edge_p10": ("edge width p10 / lambda", 2),
    "lam:edge_p50": ("edge width p50 / lambda", 2),
    "fd_mask": ("box-count dim. of vessel mask (4-64 px)", 2),
    "fd_skel": ("box-count dim. of skeleton (4-64 px)", 2),
    "rho_contrast_fwhm": ("Spearman(contrast, FWHM)", 2),
    "sharp3_c_slope": ("sharpest third: slope ln contrast / ln FWHM [new]", 2),
    "sharp3_rho": ("sharpest third: Spearman(contrast, FWHM) [new]", 2),
    "sharp3_c_small": ("sharpest third: contrast at FWHM < 0.8 lambda [new]", 2),
    "sharp3_c_mid": ("sharpest third: contrast at FWHM 0.8-1.6 lambda [new]", 2),
    "sharp3_c_large": ("sharpest third: contrast at FWHM > 1.6 lambda [new]", 2),
    "along_cv_contrast": ("robust CV of contrast along paths [new]", 2),
    "along_cv_fwhm": ("robust CV of FWHM along paths [new]", 2),
    # ---- tier 3
    "junction_density": ("junctions /1e5 px2", 3),
    "endpoint_density": ("endpoints /1e5 px2", 3),
    "lam:junction_density": ("junction density x lambda^2", 3),
    "lam:endpoint_density": ("endpoint density x lambda^2", 3),
    "junction_per_endpoint": ("junctions per endpoint", 3),
    "branch_len_med": ("median skeleton branch length (px)", 3),
    "lam:branch_len_med": ("median skeleton branch length / lambda", 3),
    "branch_tort_med": ("median branch arc/chord (>=20 px)", 3),
    "loop_density": ("closed loops /1e5 px2", 3),
    "lam:loop_density": ("closed loops x lambda^2", 3),
    "loop_area_med": ("median loop area (px2)", 3),
    "skel_anis": ("orientation anisotropy (skeleton)", 3),
    "par_excess_3_30": ("parallel-neighbour g(v), v=3-30 px", 3),
    "perp_excess_3_30": ("perpendicular-neighbour g(v), v=3-30 px", 3),
    "tort_1lam": ("arc/chord over 1 lambda of path [new]", 3),
    "tort_2lam": ("arc/chord over 2 lambda of path [new]", 3),
    "tort_4lam": ("arc/chord over 4 lambda of path [new]", 3),
    "tort_4lam_p90": ("arc/chord over 4 lambda, p90 [new]", 3),
    "path_len_med_lam": ("linked path length median / lambda [new]", 3),
    "path_len_p90_lam": ("linked path length p90 / lambda [new]", 3),
    "jangle_min_med": ("3-arm nodes: min angle between arms, median (deg) [new]", 3),
    "jangle_min_p10": ("3-arm nodes: min angle between arms, p10 (deg) [new]", 3),
    "jangle_straight_med": ("3-arm nodes: straightest pair off collinear (deg) [new]", 3),
    "frac_T": ("3-arm nodes with a through pair within 30 deg [new]", 3),
    "cross_density": ("crossings /1e5 px2 (X nodes + bridged gaps) [new]", 3),
    "lam:cross_density": ("crossings x lambda^2 [new]", 3),
    "cross_frac": ("crossings per 3-arm junction node [new]", 3),
    "node4_frac": ("share of junction nodes with >= 4 arms [new]", 3),
    "cross_angle_med": ("crossing angle median (deg) [new]", 3),
    "cross_angle_p25": ("crossing angle p25 (deg) [new]", 3),
    "faint_loop_density": ("faint-mesh loops /1e5 px2 [new]", 3),
    "lam:faint_loop_density": ("faint-mesh loops x lambda^2 [new]", 3),
    "faint_loop_area_med_lam": ("faint-mesh loop area median / lambda^2 [new]", 3),
    "faint_skel_density": ("faint+main skeleton px/1e3 px2 [new]", 3),
    "faint_extra_density": ("faint-only skeleton px/1e3 px2 [new]", 3),
}
TIERS = {k: t for k, (_, t) in LABELS.items()}
TIER_NAMES = {1: "image formation", 2: "vessel population", 3: "organisation and topology"}

# distributions compared by quantile functions (Wasserstein-1): name -> getter
DISTRIBUTIONS = {
    "contrast": lambda r: r["prof"]["contrast"],
    "log FWHM": lambda r: np.log(r["prof"]["fwhm"]),
    "log edge width": lambda r: np.log(r["prof"]["edge"]),
    "log FWHM/edge": lambda r: np.log(r["prof"]["sharp"]),
    "log branch length / lambda": lambda r: np.log(np.maximum(r["branch_len"], 1) / r["lam"]),
    "arc/chord over 4 lambda": lambda r: r["tort_windows"][4],
    "3-arm min angle (deg)": lambda r: r["jangle_min"],
    "crossing angle (deg)": lambda r: r["cross_angles"],
}
Q_LEVELS = np.linspace(0.005, 0.995, 100)


def quantile_functions(res, min_n=20):
    """{distribution name: 100 quantiles (Q_LEVELS) or None if < min_n samples}."""
    out = {}
    for name, f in DISTRIBUTIONS.items():
        try:
            a = np.asarray(f(res), float)
        except (KeyError, TypeError):
            a = np.zeros(0)
        a = a[np.isfinite(a)]
        out[name] = np.quantile(a, Q_LEVELS) if len(a) >= min_n else None
    return out
