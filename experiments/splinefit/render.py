"""A junction-aware differentiable render of the whole spline network, in optical density.

Base: LIMBUS vesselmap's NetworkModel (vesselmap/render.py, same author): every edge is a clamped cubic B-spline
centreline with r (half width), s (blur) and a (centre OD) profile splines; a pixel near edge e gets
a_e P(d; r_e, s_e) T_e, P the cylinder chord (6 nested boxes) blurred by a Gaussian in closed form (erf), T_e
end caps that fade along the axis with the same blur; the edges' contributions are summed, an image-wide halo
(1 - h) delta + h G(s_h) is applied, and a smooth background grid (bicubic, 64 px) is subtracted:
pred = bg - OD_render. Here the target is -OD (OD = B - log I, the image cleaned by a background fitted on the
vesselness mask's negative, fit.py), so bg is at most a small, regularised correction (frozen at 0 by
default).

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
summed sharp-core image before the halo. Cost: D is evaluated only where a member's sharp lumen reaches
(r + 3 s0 + ENT_MARGIN px), and the site blurs, quantised to 10 % steps (SIG_LEVEL), are one image-wide
OpenCV Gaussian per level; the site blur takes no gradient (it follows the members' s). kappa (crossing
additivity) is one parameter per image (initial 0.87).
Nodes closer than their lumens' reach are one site (a compound region): an edge between two of them is
capped at both ends.

Sites are re-detected at every rebuild (no gradient): nodes of degree >= 2 from the topology, crossings from
the geometry (centreline samples of two edges within 1 px of each other, running >= CROSS_MIN_DEG apart, away
from any node they share), so
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
from vesselmap.render import NetworkModel, gaussian_blur, profile          # noqa: E402

SQRT2 = math.sqrt(2.0)
S0 = 0.5                    # px, blur of the 'sharp' domain in which lumens are unioned
KAPPA0 = 0.87               # crossing additivity (vesselscene docs/reference.md, measured on real crossings)
R_SITE_MAX = 40.0           # px, largest overlap reach of a site
SIG_LEVEL = 0.1             # site blurs are quantised to exp(k SIG_LEVEL) px (one image blur per level)
TAPER = 4.0                 # px, cosine taper of a site's window beyond its reach
CROSS_D = 1.0               # px, centreline samples of two edges this close make a crossing
CROSS_MIN_DEG = 15.0        # ... running at least this far apart in direction (else a duplicate, not a crossing)
ENT_MARGIN = 3.0            # px, slack of a site entry's reach (the geometry moves between rebuilds)


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

    # ------------------------------------------------------------ mask term (E23, batch 8)
    @torch.no_grad()
    def set_mask(self, weight_w: float, hi: float = 3.0, lo: float = 1.5, tau: float = 0.5, coarse: float = 3.0):
        """The binary-mask term: the target's vessel mask (hysteresis on the target OD in noise units, seeds >
        hi, grown > lo; at the pixel scale and after a `coarse` px Gaussian, united) against the render passed
        through the same threshold softly, sigmoid((z - lo) / tau) at both scales. The mask comes from the
        image's own cleaned OD, never from vesselness (the residual rule). Added to the loss as weight_w x the
        summed binary cross-entropy over the valid pixels."""
        from scipy import ndimage as ndi
        self.mask_w = float(weight_w)
        if self.mask_w <= 0:
            return
        OD = (-self.logI).numpy().astype(np.float32)
        w = self.weight.numpy()
        ok = w > 0
        sig_px = np.where(ok, 1.0 / np.sqrt(np.maximum(w, 1e-12)), np.inf)

        def hyst(z):
            lab, _ = ndi.label((z > lo) & ok)
            keep = np.unique(lab[(z > hi) & ok])
            return np.isin(lab, keep[keep > 0])

        Mf = hyst(OD / sig_px)
        Ac = ndi.gaussian_filter(OD, coarse)
        hp = Ac - ndi.gaussian_filter(Ac, 3.0 * coarse)
        bgv = hp[ok & ~Mf] if (ok & ~Mf).any() else hp[ok]
        sig_c = float(1.4826 * np.median(np.abs(bgv - np.median(bgv))) + 1e-6)
        Mc = hyst(Ac / sig_c)
        self._mask_obs = torch.tensor((Mf | Mc) & ok, dtype=torch.float32)
        self._mask_ok = torch.tensor(ok)
        self._mask_inv_sig = torch.tensor(np.where(ok, 1.0 / sig_px, 0.0), dtype=torch.float32)
        self._mask_inv_sig_c = 1.0 / sig_c
        self._mask_par = (lo, tau, coarse)

    def mask_term(self, R):
        lo, tau, coarse = self._mask_par
        mf = torch.sigmoid((R * self._mask_inv_sig - lo) / tau)
        mc = torch.sigmoid((gaussian_blur(R, coarse) * self._mask_inv_sig_c - lo) / tau)
        m = (1 - (1 - mf) * (1 - mc)).clamp(1e-6, 1 - 1e-6)
        M, ok = self._mask_obs, self._mask_ok
        bce = -(M * torch.log(m) + (1 - M) * torch.log(1 - m))
        return self.mask_w * bce[ok].sum()

    def loss(self, track=None, priors=None):
        total, nll, entries, pred = super().loss(track=track, priors=priors)
        scale = getattr(self, "nll_scale", 1.0)        # E23b: 0 in the mask-only geometry stage
        if scale != 1.0:
            total = total - (1.0 - scale) * nll
        if getattr(self, "mask_w", 0.0) > 0:
            R = (self.background() - pred).view(self.H, self.W)
            total = total + self.mask_term(R)
        return total, nll, entries, pred

    # ------------------------------------------------------------ sites
    def rebuild(self):
        super().rebuild()
        self._build_sites()                            # also without the correction: crossings are reported

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
                if abs(float(T[a, 0] * T[b, 1] - T[a, 1] * T[b, 0])) < math.sin(math.radians(CROSS_MIN_DEG)):
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
                    dirs.append((t, R[a0 + j], k))
                elif c1 and not c0:
                    dirs.append((-t, R[a0 + j], k))
                else:
                    dirs.append((t, R[a0 + j], k))
                    dirs.append((-t, R[a0 + j], k))
            reach = rmax + 2.0
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    if dirs[i][2] == dirs[j][2]:
                        continue                       # a member does not overlap itself
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
                # the sharp (s0) lumen is zero beyond r + 3 s0: only pixels that close (plus the motion
                # allowed between two rebuilds) carry an entry
                near = dd <= R[a0 + keep[jj]] + 3.0 * S0 + ENT_MARGIN
                rh = rows_here[near]
                ent_row.append(nrow + rh)
                ent_col.append(np.full(len(rh), col))
                ent_samp.append(a0 + keep[jj[near]])
                ent_cap0.append(np.full(len(rh), bool(c0)))
                ent_cap1.append(np.full(len(rh), bool(c1)))
            row_pix.append(np.where(inside, py * W + px, -1))
            row_win.append(win)
            row_site.append(np.full(n_local, si))
            # blur of the site: mean of its members' blur at their samples nearest the site
            site_samp.append([a0 + int(np.argmin(d)) for k, c0, c1, a0, d in loc])
            site_rows.append((nrow, half))
            site_kind.append(kind)
            nrow += n_local
            self.site_info.append(dict(kind=kind, x=float(c[0]), y=float(c[1]), half=half, reach=float(reach),
                                       edges=[int(self.eids[k]) for k, *_ in loc],
                                       centres=[(float(q[0]), float(q[1])) for q in cents],
                                       core=float(rmax + 2.0 * smax)))
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
        # the rows any member reaches (D is zero elsewhere), their site, pixel, window and kind
        er = self.s_ent_row.numpy()
        urow, inv = np.unique(er, return_inverse=True)
        rsite = np.zeros(max(nrow, 1), np.int64)
        for si, (r0, half) in enumerate(site_rows):
            rsite[r0:r0 + (2 * half + 1) ** 2] = si
        self.s_urow = torch.tensor(urow, dtype=torch.long)
        self.s_ent_u = torch.tensor(inv.reshape(-1), dtype=torch.long)
        self.s_usite = torch.tensor(rsite[urow], dtype=torch.long)
        upix = rp[urow] if len(urow) else np.zeros(0, int)
        self.s_uin = torch.tensor(upix >= 0, dtype=torch.bool)
        self.s_upix = torch.tensor(np.maximum(upix, 0), dtype=torch.long)
        self.s_uwin = self.s_row_win[self.s_urow]
        self.s_ucross = self.s_kind_cross[self.s_usite] if len(site_kind) else torch.zeros(0, dtype=torch.bool)
        self.s_samp_flat = torch.tensor([j for v in site_samp for j in v], dtype=torch.long)
        self.s_samp_site = torch.tensor([si for si, v in enumerate(site_samp) for _ in v], dtype=torch.long)
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
        nu = len(self.s_urow)
        mx = torch.zeros(nu).scatter_reduce(0, self.s_ent_u, cap, reduce="amax", include_self=True)
        sm = torch.zeros(nu).index_add(0, self.s_ent_u, butt)
        kap = self.kappa()
        # crossing rows take -(1 - kappa)(sum - max), node rows max - sum; zero where no member reaches
        D = torch.where(self.s_ucross, -(1.0 - kap) * (sm - mx), mx - sm) * self.s_uwin
        # site blur sqrt(mean member blur^2 - s0^2) (no gradient through it), quantised to 10 % levels: each
        # level is one image-wide Gaussian blur (OpenCV, vesselmap's gaussian_blur) of its sites' D
        with torch.no_grad():
            ns = len(self.s_rows)
            s2 = torch.zeros(ns).index_add(0, self.s_samp_site, S[self.s_samp_flat] ** 2)
            cnt = torch.zeros(ns).index_add(0, self.s_samp_site, torch.ones(len(self.s_samp_site)))
            sig = torch.sqrt(torch.clamp(s2 / cnt.clamp(min=1) - S0 * S0, min=0.04))
            lev = torch.round(torch.log(sig) / SIG_LEVEL).long()
        ulev = lev[self.s_usite]
        for L in sorted(set(lev.tolist())):
            sel = (ulev == L) & self.s_uin
            if not bool(sel.any()):
                continue
            img = torch.zeros(self.H * self.W).index_add(0, self.s_upix[sel], D[sel])
            V = V + gaussian_blur(img.view(self.H, self.W), math.exp(L * SIG_LEVEL)).reshape(-1)
        return V

    # ------------------------------------------------------------ rendering (overrides)
    def vessel_image(self, entries=None):
        V = super().vessel_image(entries)
        if self.junctions and self._sites_ready:
            V = V + self.junction_correction().view(self.H, self.W)
        return V
