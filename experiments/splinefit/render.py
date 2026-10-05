"""A junction-aware differentiable render of the whole spline network, in optical density.

Base: LIMBUS vesselmap's NetworkModel (vesselmap/render.py, same author): every edge is a clamped cubic B-spline
centreline with r (half width), s (blur) and a (centre OD) profile splines; a pixel near edge e gets
a_e P(d; r_e, s_e) T_e, P the cylinder chord (6 nested boxes) blurred by a Gaussian in closed form (erf), T_e
end caps that fade along the axis with the same blur; the edges' contributions are summed, an image-wide halo
(1 - h) delta + h G(s_h) is applied, and a smooth background grid (bicubic, 64 px) is subtracted:
pred = bg - OD_render. Here the target is -OD (OD = B - log I of neuromimetic stage 1, the image cleaned by a
background fitted on the vesselness mask's negative), so bg is only a small, regularised correction.

What the plain sum gets wrong, and this module adds: the render is exact for vessels that do not touch, but
* at a fork / confluence (a node) the lumens of the edges meeting there are ONE blood volume: the chord at a
  pixel is that of the UNION of the lumens (for co-planar axes, the max of the edges' chords), not their sum.
  An additive render over-predicts the junction centre by about the thinner vessel's OD (the 'knot cue' of
  NOTES_neuromimetic.md), which pulls widths, contrasts and node positions;
* at a crossing (two vessels at different depths, no node) the ODs nearly add: vesselscene's renderer
  composites the volumes by depth and its crossings are darker than the darker arm by about 0.87 of the
  lighter arm's contrast (docs/reference.md, 'Crossings'), not 1.

Both are corrections confined to small windows ('sites') around nodes and crossings. For a site with member
edges m (incident or passing), in a 'sharp' domain (every member rendered with a small blur s0 = 0.5 px, so
it can be sampled on the pixel grid):
    node:      D = max_m cap_m - sum_m butt_m
    crossing:  D = -(1 - kappa) (sum_m c_m - max_m c_m)
where butt_m is the member's lumen with a (s0-blurred) butt end at the node, exactly what the closed-form sum
renders there before its blur, and cap_m the same lumen closed by a round (hemispherical) end, so that the
union of the members covers the node; passing members and crossing members have no end. D vanishes where the
lumens do not overlap, so the correction blurred to the site's blur, G(sqrt(s^2 - s0^2)) * D, is local and
has no seams; it is multiplied by a window (1 inside the overlap reach, cosine taper outside) and added to the
summed sharp-core image before the halo. kappa (crossing additivity) is one parameter per image (initial 0.87).
Nodes closer than their lumens' reach are one site (a compound region): an edge between two of them is
capped at both ends.

Sites are re-detected at every rebuild (no gradient): nodes of degree >= 2 from the topology, crossings from
the geometry (centreline samples of two edges within 1 px of each other, away from any node they share), so
the same renderer serves a proposal, a pruned network and an oracle (true) network.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from . import use_vesselmap

use_vesselmap()
from vesselmap.render import NetworkModel, profile          # noqa: E402

SQRT2 = math.sqrt(2.0)
S0 = 0.5                    # px, blur of the 'sharp' domain in which lumens are unioned
KAPPA0 = 0.87               # crossing additivity (vesselscene docs/reference.md, measured on real crossings)
R_SITE_MAX = 40.0           # px, largest overlap reach of a site
TAPER = 4.0                 # px, cosine taper of a site's window beyond its reach
CROSS_D = 1.0               # px, centreline samples of two edges this close make a crossing


def _gauss_kernels(sig: torch.Tensor, half: int) -> torch.Tensor:
    """(n, 2 half + 1) normalised 1-D Gaussians of std sig (differentiable in sig)."""
    x = torch.arange(-half, half + 1, dtype=sig.dtype)
    k = torch.exp(-0.5 * (x[None, :] / sig[:, None]) ** 2)
    return k / k.sum(1, keepdim=True)


class JunctionModel(NetworkModel):
    """NetworkModel with junction (union) and crossing (kappa) corrections; see the module docstring.

    Extra arguments: junctions (bool, False = vesselmap's plain additive render), kappa (initial crossing
    additivity), fit_kappa."""

    def __init__(self, net, logI, weight, *args, junctions: bool = True, kappa: float = KAPPA0,
                 fit_kappa: bool = True, **kw):
        self.junctions = junctions
        self._sites_ready = False
        super().__init__(net, logI, weight, *args, **kw)
        k = float(np.clip(kappa, 0.05, 0.99))
        self.raw_kappa = torch.nn.Parameter(torch.tensor(math.log(k / (1 - k)), dtype=torch.float32),
                                            requires_grad=fit_kappa)

    def kappa(self):
        return torch.sigmoid(self.raw_kappa)

    # ------------------------------------------------------------ sites
    def rebuild(self):
        super().rebuild()
        if getattr(self, "junctions", True):
            self._build_sites()

    @torch.no_grad()
    def _build_sites(self):
        C, T, R, S, A, arc, Le = (t.numpy().astype(float) for t in self.samples())
        ne = len(self.eids)
        first, last = np.array(self.samp_first, int), np.array(self.samp_last, int)
        eidx = {eid: k for k, eid in enumerate(self.eids)}
        nidx = {n: i for i, n in enumerate(self.nids)}
        net = self.net
        # ---- node sites: nodes of degree >= 2, clustered when their lumens' reaches overlap
        nodes = []
        for n in self.nids:
            inc = [(eidx[e], end) for e, end in net.incident(n) if e in eidx]
            pas = [eidx[e] for e in net.passing(n) if e in eidx]
            if len(inc) + 2 * len(pas) < 2:
                continue
            nodes.append((n, inc, pas))
        P = np.array([self.node_xy[nidx[n]].numpy() for n, _, _ in nodes]).reshape(-1, 2)

        def rnode(inc, pas):
            rr = [R[first[k] if end == 0 else last[k]] for k, end in inc]
            for k in pas:
                rr.append(float(R[first[k]:last[k] + 1].max()))
            return max(rr) if rr else 1.0
        rn = np.array([rnode(inc, pas) for _, inc, pas in nodes])
        par = list(range(len(nodes)))

        def find(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i
        if len(nodes) > 1:
            for i, j in sorted(cKDTree(P).query_pairs(2.0 * R_SITE_MAX)):
                if np.hypot(*(P[i] - P[j])) < max(6.0, 1.2 * (rn[i] + rn[j])):
                    a, b = find(i), find(j)
                    if a != b:
                        par[max(a, b)] = min(a, b)
        groups = {}
        for i in range(len(nodes)):
            groups.setdefault(find(i), []).append(i)
        sites = []                                     # (centre, members [(k, cap0, cap1, around)], kind)
        for g in sorted(groups):
            mem = {}
            for i in groups[g]:
                n, inc, pas = nodes[i]
                for k, end in inc:
                    c0, c1, _ = mem.get(k, (False, False, None))
                    mem[k] = (c0 or end == 0, c1 or end == 1, None)
                for k in pas:
                    if k not in mem:
                        mem[k] = (False, False, P[i])
            c = P[groups[g]].mean(0)
            sites.append((c, [(k, c0, c1, ar) for k, (c0, c1, ar) in sorted(mem.items())], "node",
                          P[groups[g]]))
        # ---- crossing sites: two edges' centrelines within CROSS_D px, away from nodes they share
        if ne > 1 and len(C):
            tree = cKDTree(C)
            pr = np.array(sorted(tree.query_pairs(CROSS_D)), int).reshape(-1, 2)
            se = self.samp_edge_t.numpy()
            pr = pr[se[pr[:, 0]] != se[pr[:, 1]]]
            ends_of = [{net.edges[eid].u, net.edges[eid].v, *net.edges[eid].through} for eid in self.eids]
            seen = []
            for a, b in pr:
                ka, kb = int(se[a]), int(se[b])
                p = 0.5 * (C[a] + C[b])
                if any(q[0] == (min(ka, kb), max(ka, kb)) and np.hypot(*(q[1] - p)) < 8.0 for q in seen):
                    continue
                shared = ends_of[ka] & ends_of[kb]
                reach = R[a] + R[b] + 3 * max(S[a], S[b]) + 8.0
                if any(np.hypot(*(self.node_xy[nidx[n]].numpy() - p)) < reach for n in shared if n in nidx):
                    continue
                seen.append(((min(ka, kb), max(ka, kb)), p))
                sites.append((p, [(ka, False, False, p), (kb, False, False, p)], "cross", p[None]))
        self._compile_sites(sites, C, T, R, S, arc, Le, first, last)

    def _compile_sites(self, sites, C, T, R, S, arc, Le, first, last):
        """Per site: its window, its members' local samples and every (pixel, member) association."""
        H, W = self.H, self.W
        ent_row, ent_col, ent_samp, ent_cap0, ent_cap1 = [], [], [], [], []
        site_rows, site_kind, site_samp, site_buck, site_pos = [], [], [], [], []
        row_pix, row_win, row_site = [], [], []
        nrow = 0
        self.site_info = []
        for si, (c, mem, kind, cents) in enumerate(sites):
            # members' local samples: within reach of the site's node(s)
            loc = []
            rmax, smax = 0.5, S0
            for k, c0, c1, around in mem:
                a0, a1 = first[k], last[k] + 1
                d = np.min(np.hypot(*(C[a0:a1, None, :] - cents[None, :, :]).transpose(2, 0, 1)), 1)
                loc.append((k, c0, c1, a0, d))
                j = a0 + int(np.argmin(d))
                rmax, smax = max(rmax, R[j]), max(smax, S[j])
            # the overlap reach: for each pair of members, how far their lumens keep overlapping
            dirs = []
            for k, c0, c1, a0, d in loc:
                j = int(np.argmin(d))
                t = T[a0 + j]
                if c0 and not c1:
                    dirs.append((t, R[a0 + j]))
                elif c1 and not c0:
                    dirs.append((-t, R[a0 + j]))
                else:
                    dirs.append((t, R[a0 + j]))
                    dirs.append((-t, R[a0 + j]))
            reach = rmax + 2.0
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    cosang = float(np.clip(np.dot(dirs[i][0], dirs[j][0]), -1, 1))
                    th = math.acos(cosang)
                    if kind == "cross":
                        sn = max(abs(math.sin(th)), 0.05)
                        reach = max(reach, (dirs[i][1] + dirs[j][1]) / sn + 2.0)
                    else:
                        sn = max(math.sin(0.5 * th), 0.05)
                        reach = max(reach, (dirs[i][1] + dirs[j][1]) / (2 * sn) + 2.0)
            spread = float(np.max(np.hypot(*(cents - c).T))) if len(cents) > 1 else 0.0
            reach = min(reach, R_SITE_MAX) + spread
            half = int(math.ceil(reach + TAPER + 3.0 * smax + 1.0))
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
            cx, cy = int(round(c[0])), int(round(c[1]))
            px, py = xx.ravel() + cx, yy.ravel() + cy
            inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            rho = np.hypot(px - c[0], py - c[1])
            win = np.where(rho <= reach, 1.0, np.where(rho >= reach + TAPER, 0.0,
                                                        0.5 * (1 + np.cos(math.pi * (rho - reach) / TAPER))))
            core = win > 0
            pix_xy = np.stack([px, py], 1).astype(float)
            n_local = len(px)
            rows_here = np.flatnonzero(core)
            # rows of this site: every patch pixel (blur needs the whole patch); entries only on the core
            for col, (k, c0, c1, a0, d) in enumerate(loc):
                keep = np.flatnonzero(d <= half * 1.5 + 2 * R[a0:a0 + len(d)].max() + 4)
                if not len(keep):
                    continue
                dd, jj = cKDTree(C[a0 + keep]).query(pix_xy[rows_here])
                ent_row.append(nrow + rows_here)
                ent_col.append(np.full(len(rows_here), col))
                ent_samp.append(a0 + keep[jj])
                ent_cap0.append(np.full(len(rows_here), bool(c0)))
                ent_cap1.append(np.full(len(rows_here), bool(c1)))
            row_pix.append(np.where(inside, py * W + px, -1))
            row_win.append(win)
            row_site.append(np.full(n_local, si))
            # blur of the site: mean of its members' blur at their samples nearest the site
            site_samp.append([a0 + int(np.argmin(d)) for k, c0, c1, a0, d in loc])
            site_rows.append((nrow, half))
            site_kind.append(kind)
            nrow += n_local
            self.site_info.append(dict(kind=kind, x=float(c[0]), y=float(c[1]), half=half, reach=float(reach),
                                       edges=[int(self.eids[k]) for k, *_ in loc]))
        cat = lambda L, dt: torch.tensor(np.concatenate(L) if L else np.zeros(0), dtype=dt)
        self.s_ent_row = cat(ent_row, torch.long)
        self.s_ent_col = cat(ent_col, torch.long)
        self.s_ent_samp = cat(ent_samp, torch.long)
        self.s_ent_cap0 = cat(ent_cap0, torch.bool)
        self.s_ent_cap1 = cat(ent_cap1, torch.bool)
        rp = np.concatenate(row_pix) if row_pix else np.zeros(0, int)
        # pixel coordinates of every row (also outside the image: the members are still defined there)
        rxy = []
        for (r0, half), info in zip(site_rows, self.site_info):
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
            rxy.append(np.stack([xx.ravel() + int(round(info["x"])), yy.ravel() + int(round(info["y"]))], 1))
        self.s_row_xy = torch.tensor(np.concatenate(rxy) if rxy else np.zeros((0, 2)), dtype=torch.float32)
        self.s_row_pix = torch.tensor(rp, dtype=torch.long)
        self.s_row_win = torch.tensor(np.concatenate(row_win) if row_win else np.zeros(0), dtype=torch.float32)
        self.s_n_rows = nrow
        self.s_ncol = int(max([len(m) for _, m, _, _ in sites], default=1))
        self.s_kind_cross = torch.tensor([k == "cross" for k in site_kind], dtype=torch.bool)
        self.s_rows = site_rows
        self.s_samp = site_samp
        # buckets of equal patch size, so each blur is one grouped convolution
        bk = {}
        for si, (r0, half) in enumerate(site_rows):
            bk.setdefault(half, []).append(si)
        self.s_buckets = [(half, torch.tensor(sorted(v), dtype=torch.long),
                           torch.tensor([site_rows[s][0] for s in sorted(v)], dtype=torch.long))
                          for half, v in sorted(bk.items())]
        self._sites_ready = True

    # ------------------------------------------------------------ the correction image
    def junction_correction(self, smp=None):
        """Sum over sites of G(s_site) * (window D), flattened (H W); zero without sites."""
        V = torch.zeros(self.H * self.W, dtype=torch.float32)
        if not self.junctions or not self._sites_ready or not len(self.s_ent_row):
            return V
        C, T, R, S, A, arc, Le = smp if smp is not None else self.samples()
        j = self.s_ent_samp
        xy = self.s_row_xy[self.s_ent_row]
        dv = xy - C[j]
        u = (dv * T[j]).sum(1)
        d = (dv[:, 0] * T[j, 1] - dv[:, 1] * T[j, 0]).abs()
        along = arc[j] + u
        Lk = Le[self.samp_edge_t[j]]
        c0, c1 = self.s_ent_cap0, self.s_ent_cap1
        zero = torch.zeros_like(along)
        beyond = torch.where(c0, F.relu(-along), zero) + torch.where(c1, F.relu(along - Lk), zero)
        s0 = torch.full_like(d, S0)
        Rj, Aj = R[j], A[j]
        dc = torch.sqrt(d * d + beyond * beyond + 1e-12)
        cap = Aj * profile(dc, Rj, s0)
        step = torch.ones_like(d)
        step = torch.where(c0, step * 0.5 * torch.erfc(-along / (SQRT2 * S0)), step)
        step = torch.where(c1, step * 0.5 * torch.erfc((along - Lk) / (SQRT2 * S0)), step)
        butt = Aj * profile(d, Rj, s0) * step
        n, m = self.s_n_rows, self.s_ncol
        M_cap = torch.zeros(n, m).index_put((self.s_ent_row, self.s_ent_col), cap)
        M_butt = torch.zeros(n, m).index_put((self.s_ent_row, self.s_ent_col), butt)
        mx = M_cap.max(1).values
        sm = M_butt.sum(1)
        kap = self.kappa()
        # per-row kind: rows of crossing sites take -(1 - kappa)(sum - max), node rows max - sum
        rk = self._row_cross()
        D = torch.where(rk, -(1.0 - kap) * (sm - mx), mx - sm) * self.s_row_win
        # site blur: sqrt(mean member blur^2 - s0^2)
        out = []
        for half, sids, r0s in self.s_buckets:
            P = 2 * half + 1
            sig = []
            for si in sids.tolist():
                sj = torch.tensor(self.s_samp[si], dtype=torch.long)
                sig.append(torch.sqrt(torch.clamp((S[sj] ** 2).mean() - S0 * S0, min=0.04)))
            sig = torch.stack(sig)
            idx = r0s[:, None] + torch.arange(P * P)[None, :]
            patches = D[idx].view(len(sids), 1, P, P)
            kh = int(min(half, math.ceil(3.0 * float(sig.max().detach())) + 1))
            K = _gauss_kernels(sig, kh)
            x = patches.view(1, len(sids), P, P)
            x = F.conv2d(x, K.view(len(sids), 1, 1, -1), padding=(0, kh), groups=len(sids))
            x = F.conv2d(x, K.view(len(sids), 1, -1, 1), padding=(kh, 0), groups=len(sids))
            out.append((idx.reshape(-1), x.reshape(-1)))
        ridx = torch.cat([o[0] for o in out])
        val = torch.cat([o[1] for o in out])
        pix = self.s_row_pix[ridx]
        ok = pix >= 0
        return V.index_add(0, pix[ok], val[ok])

    def _row_cross(self):
        if getattr(self, "_rk", None) is None or self._rk[0] is not self.s_row_win:
            rk = torch.zeros(self.s_n_rows, dtype=torch.bool)
            for si, (r0, half) in enumerate(self.s_rows):
                if bool(self.s_kind_cross[si]):
                    rk[r0:r0 + (2 * half + 1) ** 2] = True
            self._rk = (self.s_row_win, rk)
        return self._rk[1]

    # ------------------------------------------------------------ rendering (overrides)
    def vessel_image(self, entries=None):
        V = super().vessel_image(entries)
        if self.junctions and self._sites_ready:
            V = V + self.junction_correction().view(self.H, self.W)
        return V
