"""Closed-form renderer of a VesselGraph: the optical density of its vessels.

Units.  Anatomy is in capillary diameters d_c (graph.py).  The image has k px
per d_c; pixel (0, 0) sits at the d_c position `origin`, so a point xy maps
to pixel coordinates k * (xy - origin), pixel centres at integers, x to the
right and y down.  Blur widths are in px, depths in d_c, optical density (OD)
in nepers (Np): the still is I = I0 exp(-OD) before background and noise.

Physics.
* Contrast (chord law).  Light that crosses a chord c (d_c) of blood at
  relative haematocrit h is transmitted as T(c) = f + (1 - f) exp(-mu h c).
  f is the share of detected light that the vessel does not attenuate:
  light blood barely absorbs (f_spectral: green-filtered light is not
  perfectly monochrome) and light scattered past the vessel by the tissue
  above it, which grows with depth as 1 - exp(-z / l_bypass):
  f = 1 - (1 - f_spectral) exp(-z / l_bypass).  The profile g(c) = -ln T(c)
  is monotone in the chord, so a vessel's cross-section is still vesselmap's
  staircase of 6 nested boxes (half-widths r c_k, render._C) with cumulative
  levels L_k = g(2 r H_k) at the annulus mid-chords (render._H) and box
  heights W_k = L_k - L_{k+1}.  At f = 0 this is vesselmap's linear profile
  with centre OD a = mu h 2r.  This is how a vessel fades as it dives under
  tissue: its contrast drops with depth while its blur grows.  A vessel
  whose depth changes along it (slope dz/ds, both in d_c) is an inclined
  tube: its projected width is still 2r but the vertical chord through it is
  longer by sqrt(1 + (dz/ds)^2) (Optics.incline, default on; the factor is
  capped at Optics.incline_max = 2, i.e. 60 deg), so a steeply diving vessel
  (a perforator) darkens where it dives.
* Blur.  s(z)^2 = s0^2 + defocus^2 ((z - z_focus(xy))^2 + spread^2) +
  (scatter z)^2 px, z_focus(xy) a focal surface (tilt, curvature and a
  smooth random part, shifted by this frame's axial offset;
  Optics.focal_depth).  spread (Optics.focus_spread) is the SD of the
  focal surface's axial offset over the frames an average is made of: the
  average of frames focused at z_f + delta_i is blurred by the mixture of
  their blurs, rendered by its second moment (mean over delta of
  defocus^2 (z - z_f - delta)^2 = defocus^2 ((z - z_f)^2 + spread^2)).
* Tubes (render study, vesselmap/render_study/README.md section 1).  Each
  vessel's projected centreline is cut into short pieces j (length l_j,
  midpoint C_j, tangent T_j, signed curvature kappa_j, and r, s, W at the
  midpoint).  The 2-D Gaussian blur of the tube is exactly a line integral
  along the centreline; per piece it has the closed form
      OD(x) = sum_j A_j(u_j) sum_k W_kj [B_k(d_j) - kappa_j D_k(d_j)]
  (u_j, d_j: the pixel's offsets along and across piece j; A: erf along the
  piece; B: erf box across; D: its first moment, the Jacobian of the curved
  tube).  Pieces are short where the centreline bends (sagitta
  kappa l^2 / 8 <= 0.02 px), where the blur changes by 2 % or the radius by
  1 %, and where the centre OD changes by 1 % of the vessel's max (but not
  shorter than s / 4 for smooth changes; a step, e.g. a plasma gap in
  hct_mod, gets one cut at the step); at most 8 px.  Every piece within
  reach adds, so hairpin legs and coil strands add; |kappa| r is clamped to
  0.9 (a centreline that bends tighter than its radius is not a tube).
* Junctions (render study section 2).  Where vessels meet at a node (fork,
  confluence, anastomosis: they share the node's depth by construction) the
  blood is the UNION of their lumens, not the sum: for same-depth tubes the
  sharp OD is max_e F_e.  Around each node, within R = 1.25 max over pairs
  (w_i + w_j) / max(sin theta_ij, 0.25) + 1 px (w_i + w_j for pairs more
  than 150 deg apart), every incident vessel has a stub: its tube stretch
  (quads between normals) plus a round cap at the node.  Polygons are
  blurred in closed form, summed over their sides:
      G_s * 1_polygon (x) = sum_sides sign(A x B) [g(h, t_B) - g(h, t_A)],
      g(h, t) = atan(t) / 2 pi - T(h, t)        (T: Owen's T function)
  with Owen's T by 16-point Gauss-Legendre quadrature (fused CUDA kernel).
  The union is drawn by inclusion-exclusion (build_plan junctions='excess',
  the default): each vessel is drawn whole by its own tube pieces, each end
  at a junction gets half-disk caps (one per box), and the double count is
  removed exactly: max_e F_e = sum_e F_e - integral dt of
  sum_{|S|>=2} (-1)^|S| 1[intersection over S of {F_e > t}], one small
  polygon per clique S of overlapping stubs and band of levels.  This is
  the same union as the render study's recipe (junctions='union': stub
  pieces removed, the union of the stubs drawn per band of levels; kept as
  a cross-check) but touches only the overlaps, so it is ~3x cheaper and
  keeps each vessel's own staircase and blur along the whole stub.
  The union region reaches as far as the incident lumens actually keep
  overlapping in 3-D (junction_overlap_run), not only the angle formula.
  Each stub is cut at piece breaks into segments of nearly constant
  staircase, blur and depth (a steep connector between depth sheets, red-cell
  columns in a frame), each with its own length-weighted levels.
  Approximations: an overlap polygon uses the mean blur of its clique of
  segments (at a node all vessels share the depth, so the blurs agree
  there); caps use the vessel's values exactly at its end.  Two junctions
  joined by a vessel shorter than R_u + R_v are merged into one cluster (the
  vessel is one whole stub of it, cut into segments like any other).
* Crossings (v2: composite_od, Optics.composite).  Vessels stacked at
  different depths do not add their ODs.  The chord law's f(z) splits the
  detected light: f_spectral never interacts with blood (light blood barely
  absorbs; it also stands for light sent back above every vessel and for
  stray light), g(z) = 1 - exp(-z / l_bypass) of the rest is sent back by
  the tissue above depth z, the rest reaches z.  Of the light sent back
  from between two stacked vessels only the upper one takes its share, and
  f_spectral crosses both unchanged:
      q = g_1 + sum_j (g_{j+1} - g_j) prod_{i <= j} e_i,
      OD = -ln(f_spectral + (1 - f_spectral) q)
  over the blood volumes stacked at a pixel, sorted by depth (e_i: a
  volume's transmission of the light that reaches it, from its own blurred
  OD).  A crossing is lighter than the sum of its vessels' ODs, and nothing
  is darker than -ln f_spectral.  A blood volume at a pixel: a vessel and
  every vessel that shares a node with it or lies at its depth (|dz| <
  r_a + r_b + Z_MARGIN; same_volume): a junction's union, a chain of
  vessels; crossing vessels are kept apart in depth by the anatomy.  Per
  pixel the volumes are found strongest first, their exact sums (tube
  pieces and junction polygons) accumulated and composited at their
  OD-weighted mean depth (_composite: up to MAX_LAYERS volumes; about 3x
  the pairs of the additive render).  With f_spectral = 0 and l_bypass =
  inf the ODs add (v0; Optics(composite=False) adds anyway).
* Halo.  Light scattered in the tissue gives every vessel a wide skirt: the
  image-wide PSF on the OD is (1 - halo_h - tail_h) delta + tail_h
  G(tail_px) + halo_h G(halo_px) (tail_h = 0 by default: one Gaussian).

GPU.  Pixel-piece and pixel-side pairs are enumerated per bounding
rectangle in chunks of <= 4M pairs, evaluated by fused CUDA kernels (torch
jiterator; eager torch on CPU or if NVRTC is unavailable) and accumulated
with index_add_ (not bit-deterministic on CUDA: ~1e-7 OD run to run).

oracle_od is a slow float64 reference (sharp lumen union on a supersampled
grid, exact Gaussian blur with the local blur of every bit of lumen, each
vessel blurred on its own and the vessels composited per pixel) used by the
tests.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import shapely
import torch
from scipy.interpolate import BSpline

from vesselscene import spline as sp
# vesselmap's nested boxes approximating a semicircle chord profile sqrt(1 - rho^2)
# (vesselmap/render.py of the LIMBUS project; the same values, so profiles agree)
_VM_K = 6
_VM_C = np.sin(0.5 * np.pi * np.arange(1, _VM_K + 1) / _VM_K)           # box half-widths / r
_VM_MID = 0.5 * (np.r_[0.0, _VM_C[:-1]] + _VM_C)
_VM_H = np.sqrt(1.0 - _VM_MID ** 2)                                     # chord / 2r at the mid-points

from .graph import Vessel, VesselGraph

BOX_C = np.asarray(_VM_C, float)          # box half-widths / r (vesselmap's 6 nested boxes)
BOX_H = np.asarray(_VM_H, float)          # chord / 2r at the annulus mid-points (levels)

NSIG = 4.0             # reach of a piece / polygon: r + 4 s across, l/2 + 4 s along
SAGITTA = 0.02         # px: max sagitta kappa l^2 / 8 of a piece
L_MAX = 8.0            # px: longest piece
L_MIN = 0.1            # px: shortest piece from the curvature rule
EDGE_TOL = 0.1        # px: max turn of a piece's end at the lumen edge, kappa l r (v1 fix: wide vessels' ribs)
TOL_S = 0.02           # max relative blur change within a piece
TOL_R = 0.01           # max relative radius change within a piece
TOL_A = 0.01           # max centre-OD change within a piece, relative to the vessel's max
KAPPA_R_MAX = 0.9      # |kappa| r clamp (the tube Jacobian needs < 1)
CAP_TOL = 0.01         # px: sagitta of the polygon approximating a round cap
SEG_TOL = 0.02         # a junction stub is cut where its staircase changes by this share of its peak (or its blur)
SEG_TOL_S = 0.05       # ... or its blur by this share
SEG_TOL_Z = 0.5        # d_c: ... or its depth by this much (at least r / 4)
MAX_CLUSTER_NODES = 8  # merged junction clusters grow no larger (short vessels are cut instead)
LEVEL_SNAP = 1e-3      # junctions='union': levels within this fraction of a cluster's max are merged
CHUNK = 1 << 22        # pairs per GPU chunk (4M)
S_MIN = 0.05           # px: smallest blur
R_LAYER = 1.02         # oracle: ratio between blur layers (linear interpolation in sigma between them)


# ------------------------------------------------------------------ optics
@dataclass
class Optics:
    """Imaging constants.  Blur s(z)^2 = s0^2 + defocus^2 ((z - z_focus(xy))^2 +
    focus_spread^2) + (scatter z)^2 in px (z in d_c); z_focus(xy) is the focal
    surface (focal_depth: tilt focus_gx / focus_gy, curvature focus_curv and
    the smooth random part focus_modes about (focus_x0, focus_y0), shifted by
    focus_offset; all 0 by default: flat).  focus_offset: this frame's axial
    displacement of the focal surface (the eye moves along the optical axis
    during a burst); focus_spread: the SD of that displacement over the frames
    of an average, whose blur is the second moment of the frames' blurs
    (0 for a single frame).  mu: Np per d_c of blood chord at
    haematocrit 1.  f_spectral: share of the detected light no blood
    absorbs (light blood barely absorbs; it also stands for light sent back
    above every vessel and stray light): the floor -ln f_spectral of any
    stack of vessels.  l_bypass: d_c; light scattered past a vessel by the
    tissue above it grows as 1 - exp(-z / l_bypass) (inf: none).
    composite: vessels stacked at different depths are composited by depth
    (composite_od), not added.  Halo: image-wide
    (1 - halo_h - tail_h) delta + tail_h G(tail_px) + halo_h G(halo_px) on
    the optical density (tail_h = 0: the single halo).
    incline: the chord of a vessel whose depth changes along it is longer by
    sqrt(1 + (dz/ds)^2) (the vertical path through an inclined tube), at most
    incline_max (default 2, i.e. 60 deg; plan G9: steeper stretches are not
    rendered with the 1/cos chord.  Light in tissue is not collimated, and
    uncapped, the short connectors between depth sheets rendered as dark
    dots up to 4.5x their vessel's contrast, which the stills do not show)."""
    s0_px: float = 1.4
    z_focus: float = 10.0
    defocus_px_per_dc: float = 0.06
    scatter_px_per_dc: float = 0.03
    mu: float = 0.12
    f_spectral: float = 0.05
    l_bypass: float = 60.0
    halo_h: float = 0.3
    halo_px: float = 8.0
    tail_h: float = 0.0
    tail_px: float = 3.0
    incline: bool = True
    incline_max: float = 2.0
    focus_gx: float = 0.0
    focus_gy: float = 0.0
    focus_curv: float = 0.0
    focus_x0: float = 0.0
    focus_y0: float = 0.0
    focus_modes: tuple = ()        # ((a d_c, kx 1/d_c, ky 1/d_c, phase rad), ...): smooth random part of the surface
    focus_offset: float = 0.0      # d_c: this frame's axial offset of the focal surface
    focus_spread: float = 0.0      # d_c: SD of the offset over the frames of an average (0: a single frame)
    composite: bool = True         # composite vessels that cross at different depths (composite_od); False: ODs add

    def _modes(self) -> np.ndarray:
        m = np.asarray(self.focus_modes if self.focus_modes is not None else (), float)
        return m.reshape(-1, 4)

    def modes_bound(self) -> float:
        """Upper bound (d_c) on |the random part of the focal surface|."""
        return float(np.abs(self._modes()[:, 0]).sum())

    def focal_depth(self, xy=None, modes: bool = True):
        """Depth (d_c) in focus at lateral position xy (d_c, (..., 2)): the
        focal surface z_focus + gx (x - x0) + gy (y - y0) + curv |xy - xy0|^2
        + sum_i a_i cos(kx_i (x - x0) + ky_i (y - y0) + phi_i) + focus_offset.
        The camera looks obliquely at a curved globe near the canthus and focus
        is not controlled, so the depth in focus varies over the field (tilt,
        curvature and a smooth random part, drawn per scene) and from frame to
        frame (the offset); flat (z_focus) by default.  modes=False leaves the
        random part out (reach bounds)."""
        base = self.z_focus + self.focus_offset
        md = self._modes() if modes else np.zeros((0, 4))
        if xy is None:
            return base
        if self.focus_gx == 0 and self.focus_gy == 0 and self.focus_curv == 0 and not len(md):
            return np.full(np.shape(xy)[:-1], float(base))
        xy = np.asarray(xy, float)
        dx, dy = xy[..., 0] - self.focus_x0, xy[..., 1] - self.focus_y0
        z = base + self.focus_gx * dx + self.focus_gy * dy + self.focus_curv * (dx * dx + dy * dy)
        for a, kx, ky, ph in md:
            z = z + a * np.cos(kx * dx + ky * dy + ph)
        return z

    def blur(self, z, xy=None):
        """Gaussian blur sigma (px) of a vessel at depth z (d_c) and lateral
        position xy (d_c; None: on the axis, z_focus + focus_offset), with the
        spread of the frames' focus in quadrature, at least S_MIN = 0.05 px
        (the closed forms divide by it)."""
        z = np.asarray(z, float)
        zf = self.focal_depth(xy)
        s = np.sqrt(self.s0_px ** 2 + (self.defocus_px_per_dc * (z - zf)) ** 2
                    + (self.defocus_px_per_dc * self.focus_spread) ** 2 + (self.scatter_px_per_dc * z) ** 2)
        return np.maximum(s, S_MIN)

    def unabsorbed(self, z):
        """f(z): share of the detected light a vessel at depth z does not attenuate."""
        z = np.maximum(np.asarray(z, float), 0.0)
        return 1.0 - (1.0 - self.f_spectral) * (1.0 - self.bypass(z))

    def bypass(self, z):
        """g(z) = 1 - exp(-z / l_bypass): share of the absorbable light (the
        light that is not f_spectral) that the tissue above depth z sends
        back to the camera before it reaches z (0 with l_bypass = inf)."""
        z = np.maximum(np.asarray(z, float), 0.0)
        lb = self.l_bypass
        return (1.0 - np.exp(-z / lb)) if (lb is not None and np.isfinite(lb) and lb > 0) else np.zeros_like(z)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["focus_modes"] = [list(map(float, m)) for m in self._modes()]
        return d


def chord_levels(a_lin, f) -> np.ndarray:
    """Cumulative levels L_1..L_6 (Np, last axis) of the nested-box staircase:
    L_k = g(2 r H_k), g(c) = -ln(f + (1 - f) exp(-mu h c)), with
    a_lin = mu h 2 r the centre OD of the linear law.  Box heights are
    W_k = L_k - L_{k+1} (L_7 = 0)."""
    a_lin = np.asarray(a_lin, float)
    f = np.broadcast_to(np.asarray(f, float), a_lin.shape)
    x = np.minimum(a_lin[..., None] * BOX_H, 60.0)
    return x - np.log1p(f[..., None] * np.expm1(x))


def box_weights(a_lin, f) -> np.ndarray:
    """Box heights W_1..W_6 (last axis) for chord_levels(a_lin, f)."""
    L = chord_levels(a_lin, f)
    return L - np.concatenate([L[..., 1:], np.zeros_like(L[..., :1])], -1)


# ------------------------------------------------------------------ a vessel, sampled
def _spline(ctrl) -> BSpline:
    ctrl = np.asarray(ctrl, float)
    n = len(ctrl)
    deg = sp.degree_for(n)
    return BSpline(sp._knots(n, deg), ctrl, deg, extrapolate=True)


def _ev(spl, u) -> np.ndarray:
    """Evaluate on [0, 1] (clamped: the same values as vesselmap's basis)."""
    return spl(np.clip(u, 0.0, 1.0))


class _Track:
    """One vessel in image units: its centreline and profiles evaluated
    exactly at any px arclength, and a dense grid (about `spacing` px) on
    which the piece-splitting rules are measured."""

    def __init__(self, e: Vessel, k: float, origin, optics: Optics, hmod=None, spacing: float = 0.25):
        self.e, self.k, self.optics, self.hmod = e, float(k), optics, hmod
        self.origin = np.asarray(origin, float)
        n = len(e.ctrl)
        self.spl = _spline(e.ctrl)
        self.d1 = self.spl.derivative(1)
        self.d2 = self.spl.derivative(2) if sp.degree_for(n) >= 2 else None
        self.prof = [_spline(np.asarray(c, float)[:, None]) for c in (e.r_ctrl, e.z_ctrl, e.h_ctrl)]
        self.dz = self.prof[1].derivative(1)
        # the lumen's depth-wise extent over its width (anatomy._aspects: veins are flattened): the chord's factor
        self.aspect = float(np.clip(e.info.get("aspect", 1.0), 0.05, 1.0)) if isinstance(e.info, dict) else 1.0
        Lc = self.k * float(np.hypot(*np.diff(np.asarray(e.ctrl, float), axis=0).T).sum())
        m = int(max(64, 8 * n, math.ceil(Lc / spacing) + 1))
        u = np.linspace(0.0, 1.0, m)
        speed = np.hypot(*_ev(self.d1, u).T)
        S = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))])
        self.u = u
        self.sig = self.k * S                      # px arclength of the dense grid
        self.Lpx = float(self.sig[-1])
        self._N = None

    def at(self, sig) -> dict:
        """Everything at px arclengths sig: C (px), T, kap (1/px), r (px),
        r_dc, z, h (with hct_mod), s (px), f, a (= mu h 2 r_dc, times
        sqrt(1 + (dz/ds)^2) with optics.incline: the centre chord's OD under
        the linear law, Np)."""
        sig = np.clip(np.asarray(sig, float), 0.0, self.Lpx)
        u = np.interp(sig, self.sig, self.u)
        xy = _ev(self.spl, u)
        D1 = _ev(self.d1, u)
        D2 = _ev(self.d2, u) if self.d2 is not None else np.zeros_like(D1)
        speed = np.maximum(np.hypot(D1[:, 0], D1[:, 1]), 1e-12)
        T = D1 / speed[:, None]
        kap = (D1[:, 0] * D2[:, 1] - D1[:, 1] * D2[:, 0]) / speed ** 3 / self.k
        tau = sig / self.Lpx if self.Lpx > 0 else np.zeros_like(sig)
        r_dc, z, h = (_ev(p, tau)[:, 0] for p in self.prof)
        r_dc = np.maximum(r_dc, 1e-6)
        h = np.clip(h, 0.0, 1.0)
        if self.hmod is not None:
            h = h * np.maximum(np.asarray(self.hmod(sig / self.k), float), 0.0)
        o = self.optics
        a = o.mu * h * 2.0 * r_dc * self.aspect
        if o.incline and self.Lpx > 0:
            slope = _ev(self.dz, tau)[:, 0] * self.k / self.Lpx          # dz/ds, d_c per d_c
            a = a * np.minimum(np.sqrt(1.0 + slope ** 2), max(float(o.incline_max), 1.0))
        return dict(C=self.k * (xy - self.origin), T=T, kap=kap, r=self.k * r_dc, r_dc=r_dc, z=z,
                    h=h, s=o.blur(z, xy), f=o.unabsorbed(z), a=a)

    def _monitor(self):
        """Cumulative piece count along the dense grid.  A piece per sagitta
        0.02 px or 8 px of arc; per 2 % of blur or 1 % of radius; per 1 % of
        the vessel's max centre OD, but where the centre OD changes smoothly
        pieces need not be shorter than s / 4 (the midpoint staircase blurred
        by s is then accurate).  A dense step that changes blur, radius or
        centre OD by more than the tolerance (a step in hct_mod: a plasma
        gap) gets exactly one cut inside it, not dozens."""
        if self._N is None:
            q = self.at(self.sig)
            dsig = np.diff(self.sig)
            kap = np.abs(q["kap"])
            kmax = np.maximum(kap[1:], kap[:-1])
            lc = np.clip(np.sqrt(8.0 * SAGITTA / np.maximum(kmax, 1e-12)), L_MIN, L_MAX)
            # and a piece's end turns by at most EDGE_TOL px at the lumen edge (kappa l r): two straight pieces
            # meet in a wedge there that the curvature Jacobian corrects only to first order; wide vessels got
            # transverse ribs every L_MAX px (0.01 Np SD at r = 20 px on a 320 px bend; realism review v1, T6)
            rr = np.maximum(q["r"][1:], q["r"][:-1])
            lc = np.minimum(lc, np.clip(EDGE_TOL / np.maximum(kmax * rr, 1e-12), L_MIN, L_MAX))
            L1 = chord_levels(q["a"], q["f"])[:, 0]
            ref = max(float(L1.max()), 1e-9)
            v_sr = np.maximum(np.abs(np.diff(np.log(q["s"]))) / TOL_S,
                              np.abs(np.diff(np.log(q["r"]))) / TOL_R)
            v_a = np.abs(np.diff(L1)) / (TOL_A * ref)
            smid = 0.5 * (q["s"][1:] + q["s"][:-1])
            n = np.maximum.reduce([dsig / lc, np.minimum(v_sr, 1.0), np.minimum(v_a, 4.0 * dsig / smid)])
            # a step: a large change in one dense interval, much larger than in its neighbours
            nb = np.maximum(np.concatenate([[0.0], v_a[:-1]]), np.concatenate([v_a[1:], [0.0]]))
            n = np.where((v_sr >= 1.0) | ((v_a >= 1.0) & (v_a > 4.0 * nb)), np.maximum(n, 1.0), n)
            self._N = np.concatenate([[0.0], np.cumsum(n)])
            self.peak = ref
        return self._N

    def breaks(self, a: float, b: float) -> np.ndarray:
        """Piece boundaries (px arclength) from a to b by the splitting rules."""
        N = self._monitor()
        Na, Nb = np.interp([a, b], self.sig, N)
        npc = max(1, int(math.ceil(Nb - Na - 1e-9)))
        br = np.interp(np.linspace(Na, Nb, npc + 1), N, self.sig)
        br[0], br[-1] = a, b
        return br


def _vessel_reach(e: Vessel, k: float, optics: Optics) -> float:
    """Upper bound (px) on r + 4 s along a vessel (splines stay in the hull
    of their control values; s is convex in the depth offset from the focal
    surface, so its max is at an extreme of z - z_focus(xy): z at its ends,
    the focal depth at the corners of the control box or, for a curved
    surface, its vertex clamped into the box, widened by the bound on the
    surface's random part)."""
    z = np.asarray(e.z_ctrl, float)
    c = np.asarray(e.ctrl, float)
    lo, hi = c.min(0), c.max(0)
    # the surface's vertex: (x0, y0) - g / (2 curv) with a tilt (it was taken at (x0, y0), which under-bounds a
    # tilted curved surface: 29.5 px against a true 32.6; correctness review v1, defect 9); separable, so the
    # extremes over the box are at its corners and the vertex clamped into it
    vx, vy = optics.focus_x0, optics.focus_y0
    if optics.focus_curv != 0.0:
        vx, vy = vx - optics.focus_gx / (2.0 * optics.focus_curv), vy - optics.focus_gy / (2.0 * optics.focus_curv)
    pts = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]],
                    np.clip([vx, vy], lo, hi)])
    zf = np.atleast_1d(optics.focal_depth(pts, modes=False))
    mb = optics.modes_bound()
    dz = max(abs(z.max() - (zf.min() - mb)), abs(z.min() - (zf.max() + mb)))
    s_max = math.sqrt(optics.s0_px ** 2 + (optics.defocus_px_per_dc * dz) ** 2 +
                      (optics.defocus_px_per_dc * optics.focus_spread) ** 2 +
                      (optics.scatter_px_per_dc * float(z.max())) ** 2)
    return k * float(np.max(e.r_ctrl)) + NSIG * max(s_max, S_MIN)


def _end_dir(e: Vessel, end: int) -> np.ndarray:
    """Unit direction (d_c) leaving the node at end 0 (u) or 1 (v): a clamped
    B-spline leaves along its first control leg."""
    c = np.asarray(e.ctrl, float)
    c = c if end == 0 else c[::-1]
    for i in range(1, len(c)):
        d = c[i] - c[0]
        n = math.hypot(d[0], d[1])
        if n > 1e-9:
            return d / n
    return np.array([1.0, 0.0])


# ------------------------------------------------------------------ junction layout
class _UF:
    """Union-find over node ids, with a size cap."""

    def __init__(self):
        self.p, self.size = {}, {}

    def find(self, a):
        self.p.setdefault(a, a)
        self.size.setdefault(a, 1)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b, cap):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] + self.size[rb] > cap:
            return False
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def junction_radius(graph: VesselGraph, nid: int, k: float, inc=None) -> float:
    """R (px) of the union region around a node: 1.25 max over pairs of
    incident vessels of (w_i + w_j) / max(sin theta_ij, 0.25) (w_i + w_j if
    they leave more than 150 deg apart) + 1 px, w the radius at the node."""
    inc = inc if inc is not None else [(vid, 0) for vid in graph.out_vessels(nid)] + \
        [(vid, 1) for vid in graph.in_vessels(nid)]
    dirs = []
    for vid, end in inc:
        e = graph.vessels[vid]
        dirs.append((_end_dir(e, end), k * float(e.r_ctrl[0] if end == 0 else e.r_ctrl[-1])))
    R = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            (ti, wi), (tj, wj) = dirs[i], dirs[j]
            ang = math.acos(float(np.clip(np.dot(ti, tj), -1.0, 1.0)))
            R = max(R, (wi + wj) / max(math.sin(ang), 0.25) if ang < math.radians(150) else wi + wj)
    return 1.25 * R + 1.0


OVERLAP_STEP = 0.5     # px: sampling of the incident vessels when measuring how far their lumens overlap


def junction_overlap_run(graph: VesselGraph, nid: int, k: float, inc=None, tracks=None,
                         optics: "Optics | None" = None, origin=(0.0, 0.0), reach: float | None = None) -> float:
    """How far (px of arclength from the node, on either vessel) the lumens
    of the vessels meeting at a node keep overlapping in 3-D: the longest
    run, starting at the node, of centreline points of one vessel that lie
    within r_a + r_b (image distance, px) of another's centreline while
    their depths differ by less than r_a + r_b (d_c).  Near the node this is
    one blood volume (a union); where it goes on further than the formula of
    junction_radius (siblings that leave at a very small angle, or a thin
    vessel running inside a wide one) the union region must cover it too.
    tracks: {vid: _Track} to reuse and extend (all with this origin); reach:
    stop measuring there (px)."""
    inc = inc if inc is not None else [(vid, 0) for vid in graph.out_vessels(nid)] + \
        [(vid, 1) for vid in graph.in_vessels(nid)]
    if len(inc) < 2:
        return 0.0
    optics = optics or Optics()
    arms = []
    for vid, end in inc:
        tr = (tracks or {}).get(vid)
        if tr is None:
            tr = _Track(graph.vessels[vid], k, origin, optics)
            if tracks is not None:
                tracks[vid] = tr
        L = tr.Lpx
        top = L if reach is None else min(L, reach)
        d = np.arange(0.0, top + 1e-9, OVERLAP_STEP)
        q = tr.at(d if end == 0 else L - d)
        arms.append((vid, d, q["C"], q["r"], q["r_dc"], q["z"]))
    run = 0.0
    for i in range(len(arms)):
        for j in range(len(arms)):
            if i == j or arms[i][0] == arms[j][0]:
                continue
            _, da, Ca, ra, rda, za = arms[i]
            _, db, Cb, rb, rdb, zb = arms[j]
            D = np.hypot(Ca[:, None, 0] - Cb[None, :, 0], Ca[:, None, 1] - Cb[None, :, 1])
            ov = ((D < ra[:, None] + rb[None, :]) &
                  (np.abs(za[:, None] - zb[None, :]) < rda[:, None] + rdb[None, :])).any(1)
            if not ov[0]:
                continue
            stop = int(np.argmin(ov)) if not ov.all() else len(ov) - 1
            run = max(run, float(da[stop]))
    return run


def _junction_layout(graph: VesselGraph, k: float, kept: dict):
    """Which stretches of which vessels take part in a junction union.

    A junction is a node with >= 2 incident vessels in the WHOLE graph (so
    rendering a subset keeps the geometry).  Around it, each incident vessel
    has a stub: the stretch within R of the node (arclength), with a round
    cap at the node.  R is junction_radius, or further if the incident
    lumens keep overlapping in 3-D beyond it (junction_overlap_run).  Two junctions joined by a vessel shorter than R_u + R_v
    are merged into one cluster (up to MAX_CLUSTER_NODES nodes), and that
    vessel is a stub along its whole length; if the cap stops the merge the
    vessel is cut between the two clusters.  Returns the stubs (dicts: vid,
    cid, a, b = px arclength range, cap0 / cap1 = a junction at arclength 0 /
    at the end) and counts."""
    inc = defaultdict(list)
    for vid, e in graph.vessels.items():
        inc[e.u].append((vid, 0))
        inc[e.v].append((vid, 1))
    relevant = {n for vid in kept for n in (graph.vessels[vid].u, graph.vessels[vid].v) if len(inc[n]) >= 2}
    R = {n: junction_radius(graph, n, k, inc[n]) for n in relevant}
    # the union region must also cover lumens that keep overlapping in 3-D further out (a thin vessel
    # leaving a wide one at a small angle runs inside it): measured, not only the angle formula
    if kept:
        any_tr = next(iter(kept.values()))
        cache = dict(kept)
        for n in relevant:
            reach = 3.0 * R[n] + 10.0
            run = junction_overlap_run(graph, n, k, inc[n], cache, any_tr.optics, any_tr.origin, reach)
            if run >= reach - 2 * OVERLAP_STEP:
                run = junction_overlap_run(graph, n, k, inc[n], cache, any_tr.optics, any_tr.origin, None)
            if run + 1.0 > R[n]:
                R[n] = run + 1.0
    Lpx = {}

    def length(vid):
        if vid not in Lpx:
            Lpx[vid] = kept[vid].Lpx if vid in kept else k * graph.vessels[vid].length()
        return Lpx[vid]

    uf = _UF()
    n_short, seen = 0, set()
    for n in relevant:
        uf.find(n)
        for vid, _ in inc[n]:
            if vid in seen:
                continue
            seen.add(vid)
            e = graph.vessels[vid]
            if e.u in relevant and e.v in relevant and e.u != e.v and R[e.u] + R[e.v] >= length(vid):
                n_short += 1
                uf.union(e.u, e.v, MAX_CLUSTER_NODES)
    stubs = []
    n_whole = n_split = 0
    for vid, tr in kept.items():
        e = graph.vessels[vid]
        L = tr.Lpx
        ju, jv = e.u in relevant, e.v in relevant
        if ju and jv and R[e.u] + R[e.v] >= L:
            if uf.find(e.u) == uf.find(e.v):
                stubs.append(dict(vid=vid, cid=uf.find(e.u), a=0.0, b=L, cap0=True, cap1=True))
                n_whole += 1
            else:
                sc = L * R[e.u] / (R[e.u] + R[e.v])
                stubs += [dict(vid=vid, cid=uf.find(e.u), a=0.0, b=sc, cap0=True, cap1=False),
                          dict(vid=vid, cid=uf.find(e.v), a=sc, b=L, cap0=False, cap1=True)]
                n_split += 1
            continue
        if ju and R[e.u] >= L:
            stubs.append(dict(vid=vid, cid=uf.find(e.u), a=0.0, b=L, cap0=True, cap1=False))
            n_whole += 1
            continue
        if jv and R[e.v] >= L:
            stubs.append(dict(vid=vid, cid=uf.find(e.v), a=0.0, b=L, cap0=False, cap1=True))
            n_whole += 1
            continue
        if ju:
            stubs.append(dict(vid=vid, cid=uf.find(e.u), a=0.0, b=R[e.u], cap0=True, cap1=False))
        if jv:
            stubs.append(dict(vid=vid, cid=uf.find(e.v), a=L - R[e.v], b=L, cap0=False, cap1=True))
    stats = dict(junction_nodes=len(relevant), short_vessels=n_short, whole_stubs=n_whole,
                 split_short=n_split, clusters=len({st["cid"] for st in stubs}), stubs=len(stubs))
    _junction_layout.last_R = R                   # the union radii (px) of the last layout, for _node_table
    return stubs, stats


SHARE_MARGIN_SIG = 3.0     # same_volume: vessels sharing a node are one volume within its union radius + this many
                           # blurs (+ 2 px), as far as the junction's blurred union reaches


def _node_table(graph: VesselGraph, k: float, origin, kept: dict, R: dict):
    """(xy (N, 2) px, reach (N,) px) per node id (N = max node id + 1): where
    vessels sharing the node are one blood volume (same_volume): its union
    radius R (render's junction region, px) + SHARE_MARGIN_SIG x the largest
    blur of the incident vessels at the node + 2 px; -1 for nodes with no
    union (fewer than two vessels)."""
    N = max(graph.nodes, default=0) + 1
    xy = np.zeros((N, 2))
    reach = np.full(N, -1.0)
    smax = defaultdict(float)
    for vid, tr in kept.items():
        e = graph.vessels[vid]
        q = tr.at([0.0, tr.Lpx])
        smax[e.u] = max(smax[e.u], float(q["s"][0]))
        smax[e.v] = max(smax[e.v], float(q["s"][1]))
    org = np.asarray(origin, float)
    for n, rr in R.items():
        xy[n] = k * (np.asarray(graph.nodes[n].xy, float) - org)
        reach[n] = float(rr) + SHARE_MARGIN_SIG * smax.get(n, 0.0) + 2.0
    return xy, reach


# ------------------------------------------------------------------ polygons
def _n_arc(w: float, s: float) -> int:
    """Segments of a full circle of radius w: sagitta <= CAP_TOL px, or
    vertices <= 1.5 s apart (the polygons are area-matched, so what is left
    is a ripple of period p that the blur damps by exp(-2 pi^2 s^2 / p^2),
    < 2e-4 at p = 1.5 s); at least 16."""
    n_sag = math.pi / math.sqrt(2.0 * CAP_TOL / max(w, 1e-6))
    n_blur = 2.0 * math.pi * w / (1.5 * max(s, 1e-3))
    return min(max(math.ceil(min(n_sag, n_blur)), 16), 256)


_ARC_CACHE = {}


def _arc(P, N, Tout, w, s, forward=True) -> np.ndarray:
    """Interior vertices of the half circle of radius w around P that bulges
    along Tout, from P + w N to P - w N (forward) or back.  Area-matched
    (the vertices sit slightly outside the circle)."""
    nh = (_n_arc(w, s) + 1) // 2
    key = (nh, forward)
    if key not in _ARC_CACHE:
        phi = 0.5 * math.pi - np.arange(1, nh) * (math.pi / nh)
        if not forward:
            phi = phi[::-1]
        _ARC_CACHE[key] = (np.sin(phi)[:, None], np.cos(phi)[:, None],
                           math.sqrt(2 * math.pi / (2 * nh * math.sin(math.pi / nh))))
    sn, cs, f = _ARC_CACHE[key]
    return P + (w * f) * (sn * N + cs * Tout)


def _circle(c, w, s) -> np.ndarray:
    """A round cap: a polygon with the circle's area (see _n_arc)."""
    n = _n_arc(w, s)
    t = np.arange(n) * (2 * math.pi / n)
    rho = w * math.sqrt(2 * math.pi / (n * math.sin(2 * math.pi / n)))
    return np.stack([c[0] + rho * np.cos(t), c[1] + rho * np.sin(t)], 1)


def _stub_ring(C, N, w, arc0=None, arc1=None) -> np.ndarray:
    """Outline of a stub: the strip between the normals (C +- w N) with a
    half-disk at either end (arc = (P, N, Tout, radius, blur))."""
    parts = [C + w[:, None] * N]
    if arc1 is not None:
        parts.append(_arc(*arc1, forward=True))
    parts.append((C - w[:, None] * N)[::-1])
    if arc0 is not None:
        parts.append(_arc(*arc0, forward=False))
    return np.vstack(parts)


def _strip(C, N, w) -> list:
    """The tube stretch between normals as polygons: one strip, or its quads
    if the strip folds on itself (a bend tighter than the width)."""
    ring = np.vstack([C + w[:, None] * N, (C - w[:, None] * N)[::-1]])
    g = shapely.Polygon(ring)
    if g.is_valid:
        return [g]
    quads = []
    for i in range(len(C) - 1):
        q = shapely.Polygon(np.array([C[i] + w[i] * N[i], C[i + 1] + w[i + 1] * N[i + 1],
                                      C[i + 1] - w[i + 1] * N[i + 1], C[i] - w[i] * N[i]]))
        quads.append(q if q.is_valid else q.convex_hull)
    return quads


def _ring_sides(ring: np.ndarray):
    """Sides (A, B) of a closed ring and its orientation sign (+1 ccw)."""
    A, B = ring, np.roll(ring, -1, axis=0)
    area2 = float(np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]))
    return A, B, (1.0 if area2 > 0 else -1.0)


class _Sides:
    """Accumulates polygon sides with a weight (Np), blur and pixel rect, and
    the polygon's owner (vessel id, depth d_c, radius d_c: which blood volume
    it belongs to when crossing vessels are composited)."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.A, self.B, self.per = [], [], []      # sides; per polygon: (n sides, s, weight, rect, owner)
        self.n_poly = 0

    def rect(self, x0, y0, x1, y1, s):
        x0, y0 = math.ceil(x0 - NSIG * s), math.ceil(y0 - NSIG * s)
        x1, y1 = math.floor(x1 + NSIG * s), math.floor(y1 + NSIG * s)
        return max(x0, 0), max(y0, 0), min(x1, self.W - 1), min(y1, self.H - 1)

    def add(self, A, B, weight, s, rect, owner=(-1, 0.0, 0.0)):
        x0, y0, x1, y1 = rect
        if x1 < x0 or y1 < y0 or not len(A) or weight == 0:
            return
        self.n_poly += 1
        self.A.append(A)
        self.B.append(B)
        self.per.append((len(A), s, weight, x0, y0, x1, y1) + tuple(owner))

    def add_ring(self, ring, weight, s, owner=(-1, 0.0, 0.0)):
        A, B, sg = _ring_sides(ring)
        lo, hi = ring.min(0), ring.max(0)
        self.add(A, B, sg * weight, s, self.rect(lo[0], lo[1], hi[0], hi[1], s), owner)

    def add_geoms(self, geoms, weights, blurs, owners=None):
        """Polygonal parts of shapely geometries, oriented (exteriors ccw,
        holes cw) so the side sum gives their signed Gaussian mass."""
        geoms = np.asarray(geoms, dtype=object)
        if not len(geoms):
            return
        weights, blurs = np.asarray(weights, float), np.asarray(blurs, float)
        owners = [(-1, 0.0, 0.0)] * len(geoms) if owners is None else list(owners)
        parts, owner = shapely.get_parts(geoms, return_index=True)
        while np.isin(shapely.get_type_id(parts), (4, 5, 6, 7)).any():      # nested collections
            sub, si = shapely.get_parts(parts, return_index=True)
            parts, owner = sub, owner[si]
        poly = shapely.get_type_id(parts) == 3
        parts, owner = shapely.orient_polygons(parts[poly]), owner[poly]
        if not len(parts):
            return
        bounds = shapely.bounds(parts)
        rings, rp = shapely.get_rings(parts, return_index=True)
        xy, ri = shapely.get_coordinates(rings, return_index=True)
        ok = ri[:-1] == ri[1:]
        A, B, side_part = xy[:-1][ok], xy[1:][ok], rp[ri[:-1][ok]]
        order = np.argsort(side_part, kind="stable")
        A, B, side_part = A[order], B[order], side_part[order]
        cuts = np.searchsorted(side_part, np.arange(len(parts) + 1))
        for p in range(len(parts)):
            i0, i1 = cuts[p], cuts[p + 1]
            o = owner[p]
            s = float(blurs[o])
            self.add(A[i0:i1], B[i0:i1], float(weights[o]), s, self.rect(*bounds[p], s), owners[o])

    def arrays(self) -> dict:
        keys = ("ax", "ay", "bx", "by", "s", "delta", "x0", "y0", "x1", "y1", "vid", "z", "rdc")
        if not self.per:
            return {key: np.zeros(0) for key in keys}
        A, B = np.vstack(self.A), np.vstack(self.B)
        per = np.asarray(self.per, float)
        rep_ = lambda col: np.repeat(per[:, col], per[:, 0].astype(int))       # noqa: E731
        return dict(ax=A[:, 0], ay=A[:, 1], bx=B[:, 0], by=B[:, 1], s=rep_(1), delta=rep_(2),
                    x0=rep_(3), y0=rep_(4), x1=rep_(5), y1=rep_(6), vid=rep_(7).astype(np.int64), z=rep_(8),
                    rdc=rep_(9))


def _cliques(adj: list[set]) -> list[tuple]:
    """All cliques of >= 2 vertices of a small graph (adjacency sets)."""
    out = []

    def grow(cl, cand):
        for i in sorted(cand):
            new = cl + (i,)
            if len(new) >= 2:
                out.append(new)
            grow(new, {j for j in cand if j > i and j in adj[i]})

    grow((), set(range(len(adj))))
    return out


# ------------------------------------------------------------------ the render plan
class RenderPlan:
    """What will be drawn: tube pieces (arrays, px) and polygon sides
    (arrays, px; each side carries its polygon's weight in Np, its blur and
    the rectangle of pixels it is evaluated on), plus counts for reports.
    Built on the CPU."""

    def __init__(self):
        self.tubes = None       # dict of float64 arrays: cx cy tx ty kap l r s a f
        self.sides = None       # dict: ax ay bx by s delta x0 y0 x1 y1 (rect of pixels, inclusive)
        self.stats = {}
        # per tube piece, for truth rasters and visibility (not used by evaluate_plan):
        # vid (vessel id), z (depth, d_c), r_dc (radius, d_c), h (haematocrit incl. hct_mod)
        self.tube_info = None
        # crossing vessels composited by depth (composite_od): None, or dict(f_s, lb, nodes: vessel_nodes)
        self.composite = None


_TUBE_KEYS = ("cx", "cy", "tx", "ty", "kap", "l", "r", "s", "a", "f")
_TUBE_INFO_KEYS = ("vid", "z", "r_dc", "h")


def _tube_dict(vid: int, pc: dict) -> dict:
    """The tube arrays of one vessel's pieces, plus their per-piece info."""
    n = len(pc["l"])
    return dict(cx=pc["C"][:, 0], cy=pc["C"][:, 1], tx=pc["T"][:, 0], ty=pc["T"][:, 1], kap=pc["kap"],
                l=pc["l"], r=pc["r"], s=pc["s"], a=pc["a"], f=pc["f"],
                vid=np.full(n, vid, np.int64), z=pc["z"], r_dc=pc["r_dc"], h=pc["h"])


def _pieces(tr: _Track, a: float, b: float) -> dict:
    """Tube pieces of a vessel between px arclengths a and b."""
    br = tr.breaks(a, b)
    lo, hi = br[:-1], br[1:]
    q = tr.at(0.5 * (lo + hi))
    q["l"] = hi - lo
    q["br"] = br
    q["kap"] = np.clip(q["kap"], -KAPPA_R_MAX / q["r"], KAPPA_R_MAX / q["r"])
    return q


def build_plan(graph: VesselGraph, k: float, shape, optics: Optics, origin=(0.0, 0.0),
               hct_mod=None, vids=None, pad_px=None, junctions: str = "excess", composite: bool | None = None
               ) -> RenderPlan:
    """Pieces and polygons for vessel_od (see there).

    composite (default optics.composite): evaluate_plan composites vessels
        that cross at different depths (composite_od) instead of adding
        their ODs.

    junctions = 'excess' (default): every vessel is drawn along its whole
        length by tube pieces, each end at a junction gets a half-disk round
        cap (its 6 boxes), and the double counting where incident lumens
        overlap is removed in closed form by inclusion-exclusion over the
        stubs: max_e F_e = sum_e F_e - integral over t of
        sum_{|S| >= 2} (-1)^|S| 1[intersection of the stubs' level sets
        {F_e > t}], one polygon per clique of overlapping stubs and band of
        levels.  Exact for the union; only the (small) overlap polygons use
        the stubs' staircases at the node end and the clique's mean blur.
    junctions = 'union': the render study's recipe as written: stub pieces
        are removed and the union of the stubs is drawn per band of levels
        (larger polygons, slower; each stub uses its staircase at its middle
        and the cluster's mean blur).  Kept to cross-check 'excess'."""
    H, W = int(shape[0]), int(shape[1])
    k = float(k)
    origin = np.asarray(origin, float)
    hct_mod = hct_mod or {}
    render = set(graph.vessels) if vids is None else {int(v) for v in vids} & set(graph.vessels)
    plan = RenderPlan()

    # -- cull vessels whose control hull is beyond reach of the frame
    kept = {}
    for vid in sorted(render):
        e = graph.vessels[vid]
        reach = _vessel_reach(e, k, optics) if pad_px is None else float(pad_px)
        p = k * (np.asarray(e.ctrl, float) - origin)
        lo, hi = p.min(0), p.max(0)
        if hi[0] < -reach or lo[0] > W - 1 + reach or hi[1] < -reach or lo[1] > H - 1 + reach:
            continue
        kept[vid] = _Track(e, k, origin, optics, hct_mod.get(vid),
                           spacing=0.05 if (vid in hct_mod and not getattr(hct_mod[vid], "smooth", False)) else 0.25)
    stubs, jstats = _junction_layout(graph, k, kept)
    R_union = _junction_layout.last_R
    plan.stats.update(vessels=len(render), vessels_kept=len(kept), **jstats)
    sides = _Sides(W, H)
    if junctions == "union":
        tubes = _plan_union(kept, stubs, sides)
    elif junctions == "excess":
        tubes = _plan_excess(kept, stubs, sides, plan.stats)
    else:
        raise ValueError(f"junctions = {junctions!r}")
    plan.tubes = {key: np.concatenate([t[key] for t in tubes]) if tubes else np.zeros(0) for key in _TUBE_KEYS}
    plan.tube_info = {key: np.concatenate([t[key] for t in tubes]) if tubes else np.zeros(0)
                      for key in _TUBE_INFO_KEYS}
    plan.sides = sides.arrays()
    plan.stats.update(pieces=int(len(plan.tubes["l"])), polygons=sides.n_poly, sides=int(len(plan.sides["ax"])))
    if (optics.composite if composite is None else composite) and len(kept) >= 2:
        nxy, nreach = _node_table(graph, k, origin, kept, R_union)
        plan.composite = dict(f_s=float(optics.f_spectral), lb=float(optics.l_bypass if optics.l_bypass is not None
                                                                    else math.inf), nodes=vessel_nodes(graph),
                              node_xy=nxy, node_reach=nreach)
    return plan


def vessel_nodes(graph: VesselGraph) -> np.ndarray:
    """(max vessel id + 1, 2) int64: the end nodes (u, v) of every vessel id
    (-1, -2 for ids not in the graph: they share no node)."""
    out = np.tile(np.array([-1, -2], np.int64), (max(graph.vessels, default=0) + 1, 1))
    for vid, e in graph.vessels.items():
        out[vid] = (e.u, e.v)
    return out


Z_MARGIN = 1.0         # d_c: one blood volume where |dz| < r_a + r_b + Z_MARGIN (same_volume)


def same_volume(o_nodes, o_z, o_r, b_vid, b_nodes, b_z, b_r, pix_xy=None, node_tab=None):
    """Whether a piece of vessel o (its end nodes o_nodes (..., 2), depth
    o_z and radius o_r, d_c) belongs to the blood volume of vessel b at a
    pixel (b's strongest piece there: its end nodes, depth b_z, radius b_r;
    torch, elementwise; b_vid < 0: none): o is b, or shares a node with it
    and the pixel lies within that node's junction (node_tab = (xy (N, 2),
    reach (N,)) px, _node_table; pix_xy = (x, y) of the pixels: one connected
    stretch of blood, a vessel diving right after a fork is not stacked on
    its siblings), or their depths differ by less than r_o + r_b + Z_MARGIN.
    Two vessels that share a node and cross far from it are separate blood
    volumes there and are composited (v2 correctness review, defect 2: they
    added anywhere, a 0.28 Np black block where a tributary crossed its own
    wide vein 57 d_c from their confluence).  Without node_tab, any shared
    node counts (the v2 rule).  Vessels that
    cross lie at different depths: the anatomy keeps their lumens apart by
    more than r_a + r_b (at its 1st percentile by 1.1-1.9 d_c more in the v1
    scenes), so they are composited; successive vessels of a chain and
    junctions joined by a short vessel share their depth and add.  The
    margin keeps a junction's tube pieces and its overlap polygons (whose
    depth is their segment's mean, within max(0.5, r / 4) d_c of its
    pieces) in the same volume.  (A rule by graph adjacency alone was not
    transitive enough: a chain's segments three junctions apart fell into
    different volumes, depending on which led.)"""
    ou, ov, bu, bv = o_nodes[..., 0], o_nodes[..., 1], b_nodes[..., 0], b_nodes[..., 1]
    if node_tab is None:
        share = (ou == bu) | (ou == bv) | (ov == bu) | (ov == bv)
    else:
        nxy, nreach = node_tab
        px, py = pix_xy

        def near(nid):
            ok = (nid >= 0) & (nid < len(nreach))
            i = torch.where(ok, nid, torch.zeros_like(nid))
            return ok & ((px - nxy[i, 0]) ** 2 + (py - nxy[i, 1]) ** 2 <= nreach[i] * nreach[i].abs())
        share = ((ou == bu) & (ov == bv)) | (((ou == bu) | (ou == bv)) & near(ou)) | \
            (((ov == bu) | (ov == bv)) & near(ov))
    return (b_vid >= 0) & (share | ((o_z - b_z).abs() < o_r + b_r + Z_MARGIN))


def _plan_excess(kept: dict, stubs: list, sides: _Sides, stats: dict) -> list:
    """Tubes along whole vessels, half-disk caps, and the inclusion-exclusion
    overlap polygons (see build_plan)."""
    pcs = {vid: _pieces(tr, 0.0, tr.Lpx) for vid, tr in kept.items()}
    tubes = [_tube_dict(vid, pc) for vid, pc in pcs.items()]

    ends = {}

    def end_piece(pc, which, vid=None):
        """(P, N, Tout, r, levels, s) of the tube's flat end at arclength 0 / L:
        the geometry of the painted end piece (so the cap meets it); the
        staircase and blur of that piece, or (with vid) of the vessel exactly
        at its end, which the round cap continues."""
        i = 0 if which == 0 else -1
        T = pc["T"][i]
        P = pc["C"][i] - 0.5 * pc["l"][i] * T if which == 0 else pc["C"][i] + 0.5 * pc["l"][i] * T
        N = np.array([-T[1], T[0]])
        if vid is None:
            lev, s = chord_levels(pc["a"][i], pc["f"][i]), float(pc["s"][i])
        else:
            if (vid, which) not in ends:
                tr = kept[vid]
                q = tr.at([0.0 if which == 0 else tr.Lpx])
                ends[(vid, which)] = (chord_levels(q["a"][0], q["f"][0]), float(q["s"][0]))
            lev, s = ends[(vid, which)]
        return P, N, (-T if which == 0 else T), float(pc["r"][i]), lev, s

    # -- caps: 6 half-disks per junction end (part of the vessel's own sum)
    cap_ends = {(st["vid"], 0) for st in stubs if st["cap0"]} | {(st["vid"], 1) for st in stubs if st["cap1"]}
    for vid, which in sorted(cap_ends):
        P, N, Tout, r, Lv, s = end_piece(pcs[vid], which, vid)
        Wv = Lv - np.r_[Lv[1:], 0.0]
        j = 0 if which == 0 else -1
        own = (vid, float(pcs[vid]["z"][j]), float(pcs[vid]["r_dc"][j]))
        for kk in range(6):
            w = r * BOX_C[kk]
            ring = np.vstack([P + w * N, _arc(P, N, Tout, w, s), P - w * N])
            sides.add_ring(ring, float(Wv[kk]), s, own)

    # -- stub outlines Q_m(K), K = 1..6 boxes, per segment m of a stub.  A stub whose staircase or blur
    # changes along it (a steep connector between depth sheets: incline factor 1 at its nodes, up to
    # incline_max between; red-cell columns in a frame) is cut at piece breaks into segments of nearly
    # constant staircase, each with its own levels and blur.  The segments of one vessel partition its
    # lumen, so the inclusion-exclusion runs over cliques with at most one segment per vessel.
    rings, info = [], []
    for si, st in enumerate(stubs):
        tr, pc = kept[st["vid"]], pcs[st["vid"]]
        br = pc["br"]
        i0 = max(int(np.searchsorted(br, st["a"], "right")) - 1, 0)
        i1 = min(int(np.searchsorted(br, st["b"], "left")), len(br) - 1)
        i1 = max(i1, i0 + 1)
        q = tr.at(br[i0:i1 + 1])
        C, T, r = q["C"], q["T"], q["r"]
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        arcs = [None, None]
        for which, at_end in ((0, i0 == 0), (1, i1 == len(br) - 1)):
            if not at_end:
                continue
            P, Np_, Tout, rp, _, _ = end_piece(pc, which)
            j = 0 if which == 0 else -1
            C[j], N[j], r[j] = P, Np_, rp                     # the tube's flat end, as painted
            if st["cap0" if which == 0 else "cap1"]:
                arcs[which] = (P, Np_, Tout, rp)
        Lp = chord_levels(pc["a"][i0:i1], pc["f"][i0:i1])        # (pieces, 6)
        sp_ = pc["s"][i0:i1]
        lp = pc["l"][i0:i1]
        scale = max(float(Lp[:, 0].max()), 1e-9)
        zp, rp_ = pc["z"][i0:i1], pc["r_dc"][i0:i1]
        cuts = [0]
        ref_L, ref_s, ref_z = Lp[0], sp_[0], zp[0]
        for p in range(1, len(Lp)):
            if np.abs(Lp[p] - ref_L).max() > SEG_TOL * scale or abs(math.log(sp_[p] / ref_s)) > SEG_TOL_S or \
                    abs(zp[p] - ref_z) > max(SEG_TOL_Z, 0.25 * rp_[p]):
                cuts.append(p)
                ref_L, ref_s, ref_z = Lp[p], sp_[p], zp[p]
        cuts.append(len(Lp))
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            wl = lp[c0:c1] / lp[c0:c1].sum()
            info.append(dict(si=si, vid=st["vid"], cid=st["cid"], L=(wl[:, None] * Lp[c0:c1]).sum(0),
                             s=float((wl * sp_[c0:c1]).sum()), z=float((wl * zp[c0:c1]).sum()),
                             r=float((wl * rp_[c0:c1]).sum())))
            Cs, Ns, rs = C[c0:c1 + 1], N[c0:c1 + 1], r[c0:c1 + 1]
            first, last = c0 == 0, c1 == len(Lp)
            for kk in range(6):
                a0 = None if (arcs[0] is None or not first) else \
                    arcs[0][:3] + (arcs[0][3] * BOX_C[kk], float(pc["s"][0]))
                a1 = None if (arcs[1] is None or not last) else \
                    arcs[1][:3] + (arcs[1][3] * BOX_C[kk], float(pc["s"][-1]))
                rings.append(_stub_ring(Cs, Ns, rs * BOX_C[kk], a0, a1))
    if not stubs:
        return tubes
    nm = len(info)
    coords = np.vstack(rings)
    ridx = np.repeat(np.arange(len(rings)), [len(x) for x in rings])
    Q = shapely.polygons(shapely.linearrings(coords, indices=ridx))
    bad = ~shapely.is_valid(Q)
    if bad.any():
        Q[bad] = shapely.make_valid(Q[bad], method="structure")
    Q = Q.reshape(nm, 6)                                       # Q[member, K - 1]
    stats["folded_stubs"] = int(bad.reshape(nm, 6).any(1).sum())
    stats["stub_segments"] = nm

    # -- overlaps: cliques of segments of different vessels whose outer outlines intersect, per cluster
    by_c = defaultdict(list)
    for mi, m in enumerate(info):
        by_c[m["cid"]].append(mi)
    bb = shapely.bounds(Q[:, 5])
    pa, pb = [], []
    for members in by_c.values():
        mem = np.asarray(members)
        if len(mem) < 2:
            continue
        x, y = np.triu_indices(len(mem), 1)
        a_, b_ = mem[x], mem[y]
        vid_m = np.array([info[i]["vid"] for i in mem])
        z_m = np.array([info[i]["z"] for i in mem])
        r_m = np.array([info[i]["r"] for i in mem])
        # segments at depths further apart than their radii do not share blood: they add, like a crossing
        # (a steep connector passing over the vessel it joins before it reaches the node)
        ok = (vid_m[x] != vid_m[y]) & (np.abs(z_m[x] - z_m[y]) < r_m[x] + r_m[y]) & \
            (bb[a_, 0] <= bb[b_, 2]) & (bb[b_, 0] <= bb[a_, 2]) & (bb[a_, 1] <= bb[b_, 3]) & (bb[b_, 1] <= bb[a_, 3])
        pa += a_[ok].tolist()
        pb += b_[ok].tolist()
    hit = shapely.intersects(Q[pa, 5], Q[pb, 5]) if pa else np.zeros(0, bool)
    adj = defaultdict(set)
    for a, b, h in zip(pa, pb, hit):
        if h:
            adj[a].add(b)
            adj[b].add(a)
    items = []                                   # (clique, K tuple, weight, blur)
    for members in by_c.values():
        loc = {si: i for i, si in enumerate(members)}
        for cl in _cliques([{loc[j] for j in adj[si] if j in loc} for si in members]):
            S = tuple(members[i] for i in cl)
            Ls = [info[si]["L"] for si in S]
            th = np.unique(np.concatenate([[0.0]] + Ls))
            sgn = -1.0 if len(S) % 2 else 1.0             # (-1)^|S| enters with a minus: - excess
            s = float(np.mean([info[si]["s"] for si in S]))
            Km = (np.asarray(Ls)[:, :, None] > th[None, None, :-1]).sum(1)      # (|S|, bands)
            dt = np.diff(th)
            for band in np.flatnonzero((Km.min(0) >= 1) & (dt > 0)):
                items.append((S, tuple(Km[:, band].tolist()), -sgn * float(dt[band]), s))
    stats["overlap_items"] = len(items)
    # intersections, one new operation per item: an item's (S[:-1], K[:-1])
    # parent is always an item of the smaller clique (its bands are coarser)
    geo = {}
    for m in sorted({len(it[0]) for it in items}):
        batch = [it for it in items if len(it[0]) == m]
        if m == 2:
            left = Q[[it[0][0] for it in batch], [it[1][0] - 1 for it in batch]]
        else:
            left = np.array([geo.get((it[0][:-1], it[1][:-1])) for it in batch], dtype=object)
        right = Q[[it[0][-1] for it in batch], [it[1][-1] - 1 for it in batch]]
        ok = np.array([g is not None and not g.is_empty for g in left], bool)
        res = np.full(len(batch), None, dtype=object)
        if ok.any():
            res[ok] = shapely.intersection(left[ok], right[ok])
        for it, g in zip(batch, res):
            geo[(it[0], it[1])] = g if g is not None and not g.is_empty else None
    keep = [(geo[(it[0], it[1])], it[2], it[3], (info[it[0][0]]["vid"], info[it[0][0]]["z"], info[it[0][0]]["r"]))
            for it in items if geo.get((it[0], it[1])) is not None]
    if keep:
        g, w, s, own = zip(*keep)
        sides.add_geoms(list(g), w, s, own)
    return tubes


def _plan_union(kept: dict, stubs: list, sides: _Sides) -> list:
    """The render study's recipe: pieces outside the stubs, and per cluster
    the union of the stubs per band of levels (see build_plan)."""
    lo_cut = defaultdict(float)
    hi_cut = {vid: tr.Lpx for vid, tr in kept.items()}
    for st in stubs:
        if st["a"] <= 1e-9:
            lo_cut[st["vid"]] = max(lo_cut[st["vid"]], st["b"])
        if st["b"] >= kept[st["vid"]].Lpx - 1e-9:
            hi_cut[st["vid"]] = min(hi_cut[st["vid"]], st["a"])
    tubes, first_last = [], {}
    for vid, tr in kept.items():
        a, b = lo_cut[vid], hi_cut[vid]
        if b - a <= 1e-9:
            continue
        pc = _pieces(tr, a, b)
        l = pc["l"]
        first_last[vid] = (pc["C"][0] - 0.5 * l[0] * pc["T"][0], pc["T"][0], pc["r"][0],
                           pc["C"][-1] + 0.5 * l[-1] * pc["T"][-1], pc["T"][-1], pc["r"][-1])
        tubes.append(_tube_dict(vid, pc))
    groups = defaultdict(list)
    for st in stubs:
        tr = kept[st["vid"]]
        a, b = st["a"], st["b"]
        br = tr.breaks(a, b) if b > a else np.array([a, b])
        q = tr.at(br)
        C, T, r = q["C"].copy(), q["T"].copy(), q["r"].copy()
        fl = first_last.get(st["vid"])
        if fl is not None and st["a"] <= 1e-9 and not st["cap1"]:
            C[-1], T[-1], r[-1] = fl[0], fl[1], fl[2]            # meet the first kept piece's start edge
        if fl is not None and st["cap1"] and not st["cap0"]:
            C[0], T[0], r[0] = fl[3], fl[4], fl[5]               # meet the last kept piece's end edge
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        rep = tr.at([0.5 * (a + b)])
        caps = ([C[0]] if st["cap0"] else []) + ([C[-1]] if st["cap1"] else [])
        caps_r = ([r[0]] if st["cap0"] else []) + ([r[-1]] if st["cap1"] else [])
        groups[st["cid"]].append(dict(C=C, N=N, r=r, caps=caps, caps_r=caps_r, vid=st["vid"],
                                      z=float(rep["z"][0]), r_dc=float(rep["r_dc"][0]),
                                      L=chord_levels(rep["a"], rep["f"])[0], s=float(rep["s"][0])))
    for sts in groups.values():
        s_c = float(np.mean([st["s"] for st in sts]))
        allL = np.concatenate([st["L"] for st in sts])
        top = float(allL.max())
        if top <= 1e-9:
            continue
        # snap nearly equal levels (<= LEVEL_SNAP of the max) to one value
        order = np.unique(allL)
        grp, cur = [], [order[0]]
        for v in order[1:]:
            if v - cur[0] <= LEVEL_SNAP * top:
                cur.append(v)
            else:
                grp.append(cur)
                cur = [v]
        grp.append(cur)
        rep = {v: float(np.mean(g)) for g in grp for v in g}
        for st in sts:
            st["L"] = np.array([rep[v] for v in st["L"]])
        th = np.concatenate([[0.0], np.unique([float(np.mean(g)) for g in grp])])
        geoms, weights = [], []
        for ta, tb in zip(th[:-1], th[1:]):
            parts = []
            for st in sts:
                K = int((st["L"] > ta + 1e-15).sum())
                if K == 0:
                    continue
                parts += _strip(st["C"], st["N"], st["r"] * BOX_C[K - 1])
                parts += [shapely.Polygon(_circle(c, rc * BOX_C[K - 1], s_c)) for c, rc in zip(st["caps"], st["caps_r"])]
            if parts:
                geoms.append(shapely.union_all(parts))
                weights.append(tb - ta)
        own = (sts[0]["vid"], float(np.mean([st["z"] for st in sts])), float(np.mean([st["r_dc"] for st in sts])))
        sides.add_geoms(geoms, weights, np.full(len(geoms), s_c), [own] * len(geoms))
    return tubes


# ------------------------------------------------------------------ kernels
_TUBE_SRC = """
template <typename T> T tube_pair(T u, T d, T l, T s, T r, T kap, T a, T f) {
  const T cs[6] = {%s};
  const T hs[6] = {%s};
  T is = T(1) / s;
  T A = normcdf((u + T(0.5) * l) * is) - normcdf((u - T(0.5) * l) * is);
  T g0 = T(0.3989422804014327) * is;
  T F = T(0);
  T Lnext = T(0);
  #pragma unroll
  for (int k = 5; k >= 0; --k) {
    T x = fmin(a * hs[k], T(60));
    T Lk = x - log1p(f * expm1(x));
    T Wk = Lk - Lnext;
    Lnext = Lk;
    T w = r * cs[k];
    T B = normcdf((w - d) * is) + normcdf((w + d) * is) - T(1);
    T ep = (d + w) * is, em = (d - w) * is;
    T D = d * B + s * s * g0 * (exp(T(-0.5) * ep * ep) - exp(T(-0.5) * em * em));
    F += Wk * (B - kap * D);
  }
  return A * F;
}
""" % (", ".join(f"{v:.12f}" for v in BOX_C), ", ".join(f"{v:.12f}" for v in BOX_H))

_OWN_SRC = """
template <typename T> T own_pair(T u, T d, T l, T s, T r, T a, T f) {
  const T cs[6] = {%s};
  const T hs[6] = {%s};
  T hl = T(0.5) * l;
  T du = u - fmin(fmax(u, -hl), hl);
  T dist = sqrt(du * du + d * d);
  T is = T(1) / s;
  T F = T(0);
  T Lnext = T(0);
  #pragma unroll
  for (int k = 5; k >= 0; --k) {
    T x = fmin(a * hs[k], T(60));
    T Lk = x - log1p(f * expm1(x));
    T Wk = Lk - Lnext;
    Lnext = Lk;
    T w = r * cs[k];
    F += Wk * (normcdf((w - dist) * is) + normcdf((w + dist) * is) - T(1));
  }
  return F;
}
""" % (", ".join(f"{v:.12f}" for v in BOX_C), ", ".join(f"{v:.12f}" for v in BOX_H))
_JIT_NARGS = {"tube_pair": 8, "side_mass": 4, "own_pair": 7}

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W
_SIDE_SRC = """
template <typename T> T owt01(T h, T a) {
  const T xs[16] = {%s};
  const T ws[16] = {%s};
  T acc = T(0);
  #pragma unroll
  for (int i = 0; i < 16; ++i) { T ax = a * xs[i]; T q = T(1) + ax * ax; acc += ws[i] * exp(T(-0.5) * h * h * q) / q; }
  return a * acc * T(0.15915494309189535);
}
template <typename T> T owt(T h, T a) {
  h = fabs(h);
  if (h > T(8.5)) return T(0);
  T sg = a < T(0) ? T(-1) : T(1);
  a = fabs(a);
  if (a <= T(1)) return sg * owt01(h, a);
  T ph = normcdf(h), pah = normcdf(a * h);
  return sg * (T(0.5) * ph + T(0.5) * pah - ph * pah - owt01(a * h, T(1) / a));
}
template <typename T> T side_mass(T ax, T ay, T bx, T by) {
  T dx = bx - ax, dy = by - ay;
  T n = sqrt(dx * dx + dy * dy);
  if (n < T(1e-12)) return T(0);
  T ux = dx / n, uy = dy / n;
  T hsg = ax * uy - ay * ux;
  T h = fmax(fabs(hsg), T(1e-12));
  T ta = (ax * ux + ay * uy) / h, tb = (bx * ux + by * uy) / h;
  T gb = atan(tb) * T(0.15915494309189535) - owt(h, tb);
  T ga = atan(ta) * T(0.15915494309189535) - owt(h, ta);
  T sg = hsg > T(0) ? T(1) : (hsg < T(0) ? T(-1) : T(0));
  return sg * (gb - ga);
}
""" % (", ".join(f"{v:.15f}" for v in _GL_X), ", ".join(f"{v:.15f}" for v in _GL_W))

_JIT = {}


def _jit(name):
    """Fused CUDA kernel via torch's jiterator (NVRTC), or None."""
    if name not in _JIT:
        try:
            src = {"tube_pair": _TUBE_SRC, "side_mass": _SIDE_SRC, "own_pair": _OWN_SRC}[name]
            fn = torch.cuda.jiterator._create_jit_fn(src)
            x = [torch.full((4,), 0.5, device="cuda") for _ in range(_JIT_NARGS[name])]
            fn(*x)
            torch.cuda.synchronize()
            _JIT[name] = fn
        except Exception:                                  # noqa: BLE001 - any failure: eager fallback
            _JIT[name] = None
    return _JIT[name]


_ISQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def tube_pair_eager(u, d, l, s, r, kap, a, f):
    """The per-(pixel, piece) closed form in eager torch (same as the kernel)."""
    ncdf = torch.special.ndtr
    A = ncdf((u + 0.5 * l) / s) - ncdf((u - 0.5 * l) / s)
    hs = torch.as_tensor(BOX_H, dtype=u.dtype, device=u.device)
    cs = torch.as_tensor(BOX_C, dtype=u.dtype, device=u.device)
    x = torch.clamp(a[:, None] * hs, max=60.0)
    L = x - torch.log1p(f[:, None] * torch.expm1(x))
    Wk = L - torch.cat([L[:, 1:], torch.zeros_like(L[:, :1])], 1)
    w = r[:, None] * cs
    dd, ss = d[:, None], s[:, None]
    B = ncdf((w - dd) / ss) + ncdf((w + dd) / ss) - 1.0
    phi = lambda t: torch.exp(-0.5 * (t / ss) ** 2) * (_ISQ2PI / ss)      # noqa: E731
    D = dd * B + ss * ss * (phi(dd + w) - phi(dd - w))
    return A * (Wk * (B - kap[:, None] * D)).sum(1)


def owens_t(h, a):
    """Owen's T(h, a) in torch: 16-point Gauss-Legendre with the a > 1
    reflection (error < 1e-7); 0 for |h| > 8.5."""
    xg = torch.as_tensor(_GL_X, dtype=h.dtype, device=h.device)
    wg = torch.as_tensor(_GL_W, dtype=h.dtype, device=h.device)

    def t01(hh, aa):
        ax = aa[:, None] * xg
        q = 1.0 + ax * ax
        return aa / (2 * math.pi) * (wg * torch.exp(-0.5 * (hh * hh)[:, None] * q) / q).sum(1)

    h = h.abs()
    sg = torch.where(a < 0, -1.0, 1.0).to(h.dtype)
    a = a.abs()
    out = torch.zeros_like(h)
    near = h <= 8.5
    small = near & (a <= 1)
    out[small] = t01(h[small], a[small])
    big = near & (a > 1)
    hb, ab = h[big], a[big]
    ph, pah = torch.special.ndtr(hb), torch.special.ndtr(ab * hb)
    out[big] = 0.5 * ph + 0.5 * pah - ph * pah - t01(ab * hb, 1.0 / ab)
    return sg * out


def side_mass_eager(ax, ay, bx, by):
    """Signed Gaussian mass of the triangle (pixel, A, B), A and B relative to
    the pixel in units of the blur (same as the kernel)."""
    dx, dy = bx - ax, by - ay
    n = torch.sqrt(dx * dx + dy * dy)
    ok = n >= 1e-12
    n = torch.where(ok, n, torch.ones_like(n))
    ux, uy = dx / n, dy / n
    hsg = ax * uy - ay * ux
    h = torch.clamp(hsg.abs(), min=1e-12)
    ta, tb = (ax * ux + ay * uy) / h, (bx * ux + by * uy) / h
    g = lambda t: torch.atan(t) / (2 * math.pi) - owens_t(h, t)          # noqa: E731
    return torch.where(ok, torch.sign(hsg) * (g(tb) - g(ta)), torch.zeros_like(n))


# ------------------------------------------------------------------ pairs on the device
def _rect_chunks(x0, y0, nx, ny, chunk=CHUNK):
    """(item, ix, iy) for every pixel of every item's rectangle, in chunks of
    about `chunk` pairs (an item larger than a chunk goes alone)."""
    cnt = nx * ny
    csum = torch.cumsum(cnt, 0).cpu().numpy()
    n = len(csum)
    start = 0
    while start < n:
        base = csum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(csum, base + chunk, side="right"))
        end = max(end, start + 1)
        idx = torch.arange(start, end, device=x0.device)
        c = cnt[start:end]
        pid = torch.repeat_interleave(idx, c)
        off = torch.cumsum(c, 0) - c
        loc = torch.arange(int(c.sum()), device=x0.device) - torch.repeat_interleave(off, c)
        nxp = nx[pid]
        yield pid, x0[pid] + loc % nxp, y0[pid] + torch.div(loc, nxp, rounding_mode="floor")
        start = end


def _device(device):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    return dev


EPS_TOUCH = 1e-4       # Np: a vessel takes part in the compositing at a pixel where one of its pieces adds more
_VQ = float(1 << 22)   # value quantisation of the per-pixel argmax keys (value << 31 | piece)
_LOW31 = (1 << 31) - 1


def composite_od(D, z, f_s: float, lb: float):
    """The optical density of K layers of blood stacked at one pixel (torch,
    (K, n) -> (n,)): D their own (blurred) optical densities, z their depths
    (d_c).  Every layer obeys the chord law T_i = f_s + (1 - f_s) q_i, q_i =
    g_i + (1 - g_i) e_i, g_i = 1 - exp(-z_i / lb) (Optics.unabsorbed): f_s
    is the light no blood absorbs (it crosses every layer unchanged), g_i
    the absorbable light the tissue above layer i sends back before it
    reaches it, e_i the transmission of the light that reaches it.  The
    absorbable light sent back from between layers j and j + 1 (sorted by
    depth) crossed the layers above it only:
        q = g_1 + sum_j (g_{j+1} - g_j) prod_{i <= j} e_i,   g_{K+1} = 1,
    and OD = -ln(f_s + (1 - f_s) q).  One layer gives its own OD back; with
    f_s = 0 and lb = inf the ODs add (Beer-Lambert)."""
    order = torch.argsort(z, dim=0)
    D = torch.gather(D, 0, order)
    z = torch.gather(z, 0, order)
    if lb is not None and math.isfinite(lb) and lb > 0:
        g = 1.0 - torch.exp(-torch.clamp(z, min=0.0) / lb)
    else:
        g = torch.zeros_like(z)
    fs = min(max(float(f_s), 0.0), 1.0 - 1e-9)
    q = torch.clamp((torch.exp(-D) - fs) / (1.0 - fs), 0.0, 1.0)
    e = torch.clamp((q - g) / torch.clamp(1.0 - g, min=1e-12), 0.0, 1.0)
    acc = g[0].clone()
    prod = torch.ones_like(acc)
    K = D.shape[0]
    for i in range(K):
        prod = prod * e[i]
        acc = acc + ((g[i + 1] if i + 1 < K else 1.0) - g[i]) * prod
    return -torch.log(torch.clamp(fs + (1.0 - fs) * acc, min=1e-30))


def _tube_values(P, pid, ix, iy, fn, dt):
    """The closed form of every (piece, pixel) pair, zero beyond the piece's reach."""
    vx = ix.to(dt) - P["cx"][pid]
    vy = iy.to(dt) - P["cy"][pid]
    tx, ty = P["tx"][pid], P["ty"][pid]
    u = vx * tx + vy * ty
    d = vy * tx - vx * ty
    l, s, r = P["l"][pid], P["s"][pid], P["r"][pid]
    args = (u, d, l, s, r, P["kap"][pid], P["a"][pid], P["f"][pid])
    v = fn(*args) if fn is not None else tube_pair_eager(*args)
    return torch.where((u.abs() <= 0.5 * l + NSIG * s) & (d.abs() <= r + NSIG * s), v, torch.zeros_like(v))


def _side_values(Q, pid, ix, iy, fn, dt):
    """Signed Gaussian mass of every (polygon side, pixel) pair, times the polygon's weight."""
    s = Q["s"][pid]
    fx, fy = ix.to(dt), iy.to(dt)
    args = ((Q["ax"][pid] - fx) / s, (Q["ay"][pid] - fy) / s, (Q["bx"][pid] - fx) / s, (Q["by"][pid] - fy) / s)
    v = fn(*args) if fn is not None else side_mass_eager(*args)
    return Q["delta"][pid] * v


def _own_values(P, pid, ix, iy, dt, fn=None):
    """A piece's vessel's own blurred cross-section at the pixel's distance
    from the piece (its segment, clamped at the ends): the vessel's OD at
    its nearest centreline point, whatever the piece's length (the full
    closed form of a piece also depends on how long it is).  Selects the
    strongest vessel at a pixel and the depth it has there."""
    vx = ix.to(dt) - P["cx"][pid]
    vy = iy.to(dt) - P["cy"][pid]
    tx, ty = P["tx"][pid], P["ty"][pid]
    u = vx * tx + vy * ty
    d = vy * tx - vx * ty
    hl, s, r = 0.5 * P["l"][pid], P["s"][pid], P["r"][pid]
    if fn is not None:
        v = fn(u, d, P["l"][pid], s, r, P["a"][pid], P["f"][pid])
    else:
        du = u - torch.clamp(u, -hl, hl)
        dist = torch.sqrt(du * du + d * d)[:, None]
        w = r[:, None] * torch.as_tensor(BOX_C, dtype=dt, device=dist.device)
        ss = s[:, None]
        v = (P["W"][pid] * (torch.special.ndtr((w - dist) / ss) + torch.special.ndtr((w + dist) / ss) - 1.0)).sum(1)
    return torch.where((u.abs() <= hl + NSIG * s) & (d.abs() <= r + NSIG * s), v, torch.zeros_like(v))


def _key(v, pid):
    return (torch.clamp(torch.round(v * _VQ), max=_LOW31).to(torch.int64) << 31) | pid


def evaluate_plan(plan: RenderPlan, shape, device="cuda") -> np.ndarray:
    """Sum the tube pieces and junction polygons of a plan into an OD image;
    with plan.composite, composite the blood volumes that overlap at a pixel
    by depth (_composite, composite_od) instead of adding them."""
    H, W = int(shape[0]), int(shape[1])
    dev = _device(device)
    dt = torch.float32 if dev.type == "cuda" else torch.float64
    out = torch.zeros(H * W, dtype=dt, device=dev)
    pairs = 0
    comp = plan.composite
    tb = plan.tubes
    T = None
    if len(tb["l"]):
        P = {key: torch.as_tensor(v, dtype=dt, device=dev) for key, v in tb.items()}
        Ua = 0.5 * tb["l"] + NSIG * tb["s"]
        Da = tb["r"] + NSIG * tb["s"]
        hx = np.abs(tb["tx"]) * Ua + np.abs(tb["ty"]) * Da
        hy = np.abs(tb["ty"]) * Ua + np.abs(tb["tx"]) * Da
        x0 = np.maximum(np.ceil(tb["cx"] - hx), 0).astype(np.int64)
        x1 = np.minimum(np.floor(tb["cx"] + hx), W - 1).astype(np.int64)
        y0 = np.maximum(np.ceil(tb["cy"] - hy), 0).astype(np.int64)
        y1 = np.minimum(np.floor(tb["cy"] + hy), H - 1).astype(np.int64)
        ok = (x1 >= x0) & (y1 >= y0) & (tb["a"] > 0)
        sel = torch.as_tensor(np.flatnonzero(ok), device=dev)
        P = {key: v[sel] for key, v in P.items()}
        tdev = lambda a: torch.as_tensor(a[ok], device=dev)           # noqa: E731
        X0, Y0, NX, NY = tdev(x0), tdev(y0), tdev(x1 - x0 + 1), tdev(y1 - y0 + 1)
        fn = _jit("tube_pair") if dev.type == "cuda" else None
        if comp is not None and len(sel):
            ti = plan.tube_info
            P["vid"] = torch.as_tensor(np.asarray(ti["vid"])[ok].astype(np.int64), device=dev)
            P["z"] = torch.as_tensor(np.asarray(ti["z"], float)[ok], dtype=dt, device=dev)
            P["rdc"] = torch.as_tensor(np.asarray(ti["r_dc"], float)[ok], dtype=dt, device=dev)
            if dev.type != "cuda" or _jit("own_pair") is None:                # the eager _own_values needs the box heights
                P["W"] = torch.as_tensor(box_weights(tb["a"][ok], tb["f"][ok]), dtype=dt, device=dev)
            vmin = torch.full((H * W,), 1 << 62, dtype=torch.int64, device=dev)
            vmax = torch.full((H * W,), -1, dtype=torch.int64, device=dev)
        if len(sel):
            for pid, ix, iy in _rect_chunks(X0, Y0, NX, NY):
                v = _tube_values(P, pid, ix, iy, fn, dt)
                pix = iy * W + ix
                out.index_add_(0, pix, v)
                pairs += len(pid)
                if comp is not None:
                    m = v > EPS_TOUCH
                    pm, vid = pix[m], P["vid"][pid[m]]
                    vmin.scatter_reduce_(0, pm, vid, reduce="amin")
                    vmax.scatter_reduce_(0, pm, vid, reduce="amax")
            T = dict(P=P, rect=(X0, Y0, NX, NY), fn=fn)
    sd = plan.sides
    S = None
    if len(sd["ax"]):
        Q = {key: torch.as_tensor(sd[key], dtype=dt, device=dev) for key in ("ax", "ay", "bx", "by", "s", "delta")}
        X0 = torch.as_tensor(sd["x0"].astype(np.int64), device=dev)
        Y0 = torch.as_tensor(sd["y0"].astype(np.int64), device=dev)
        NX = torch.as_tensor((sd["x1"] - sd["x0"] + 1).astype(np.int64), device=dev)
        NY = torch.as_tensor((sd["y1"] - sd["y0"] + 1).astype(np.int64), device=dev)
        fn = _jit("side_mass") if dev.type == "cuda" else None
        for pid, ix, iy in _rect_chunks(X0, Y0, NX, NY):
            out.index_add_(0, iy * W + ix, _side_values(Q, pid, ix, iy, fn, dt))
            pairs += len(pid)
        if comp is not None:
            Q["vid"] = torch.as_tensor(np.asarray(sd["vid"]).astype(np.int64), device=dev)
            nodes = np.asarray(comp["nodes"])                       # a polygon without an owner shares no node
            Q["nodes"] = torch.as_tensor(np.where(Q["vid"].cpu().numpy()[:, None] >= 0,
                                                  nodes[np.maximum(np.asarray(sd["vid"]).astype(np.int64), 0)],
                                                  np.array([-3, -4], np.int64)), device=dev)
            Q["z"] = torch.as_tensor(np.asarray(sd["z"], float), dtype=dt, device=dev)
            Q["rdc"] = torch.as_tensor(np.asarray(sd["rdc"], float), dtype=dt, device=dev)
            S = dict(Q=Q, rect=(X0, Y0, NX, NY), fn=fn)
    plan.stats["pairs"] = pairs
    if comp is not None and T is not None:
        multi = (vmax >= 0) & (vmin != vmax)
        plan.stats["composite_px"] = int(multi.sum())
        if bool(multi.any()):
            plan.stats["pairs"] = pairs + _composite(out, multi, T, S, comp, (H, W), dev, dt, plan.stats)
    return out.view(H, W).cpu().numpy().astype(np.float32)


def _active(rect, multi, H, W):
    """Indices of the items whose pixel rectangle contains a pixel of `multi`."""
    X0, Y0, NX, NY = rect
    I = torch.zeros((H + 1, W + 1), dtype=torch.int32, device=multi.device)
    I[1:, 1:] = multi.view(H, W).to(torch.int32).cumsum(0).cumsum(1)
    x1, y1 = X0 + NX, Y0 + NY
    n = I[y1, x1] - I[Y0, x1] - I[y1, X0] + I[Y0, X0]
    return torch.nonzero(n > 0).flatten()


MAX_LAYERS = 8         # blood volumes composited per pixel at most (the rest joins the strongest; stats: composite_overflow)


def _composite(out, multi, T, S, comp, shape, dev, dt, stats) -> int:
    """Composite the blood volumes that overlap at the pixels of `multi`
    (touched by more than one vessel), in place in `out`.  Per pixel the
    volumes are found one after the other: the vessel whose own OD there is
    the largest (_own_values: its strongest piece, nearly its nearest, gives
    the depth that decides which pieces join it), then the strongest outside
    the volumes found so far, until none is left (at most MAX_LAYERS); a
    volume is that vessel and every piece within its depth (same_volume).
    Each volume's exact sum (tube pieces and junction polygons, the polygons
    by their owner's segment depth) is composited by depth (composite_od),
    the volume at the OD-weighted mean depth of its pieces there.  What no
    volume claims (pieces under EPS_TOUCH of vessels that never lead a volume, a
    polygon off every volume's depth) joins the first volume: its size is
    stats['composite_rest'] (max |Np|).  Returns the number of pairs
    evaluated."""
    H, W = shape
    HW = H * W
    P, fn = T["P"], T["fn"]
    pv, pz, pr = P["vid"], P["z"], P["rdc"]
    nodes = torch.as_tensor(comp["nodes"], device=dev)
    pn = nodes[pv]                                        # each piece's vessel's end nodes
    ntab = None
    if comp.get("node_xy") is not None:
        ntab = (torch.as_tensor(comp["node_xy"], dtype=dt, device=dev),
                torch.as_tensor(comp["node_reach"], dtype=dt, device=dev))
    fo = _jit("own_pair") if dev.type == "cuda" else None

    def chosen(key):
        ok = key >= 0
        p = torch.where(ok, key & _LOW31, torch.zeros_like(key))
        return torch.where(ok, pv[p], torch.full_like(p, -1)), pz[p], pr[p], pn[p]

    def layer_of(o_nodes, o_z, o_r, pix):
        """Index of the first volume a piece belongs to (len(b): none)."""
        j = torch.full_like(o_z, len(b), dtype=torch.int64)
        pxy = None if ntab is None else ((pix % W).to(dt), torch.div(pix, W, rounding_mode="floor").to(dt))
        for i in range(len(b) - 1, -1, -1):
            j = torch.where(same_volume(o_nodes, o_z, o_r, b[i][0][pix], b[i][3][pix], b[i][1][pix], b[i][2][pix],
                                        pxy, ntab), i, j)
        return j

    pairs = 0
    b = []
    todo = multi.clone()                                  # pixels that may hold one more volume
    while len(b) < MAX_LAYERS:
        act = _active(T["rect"], todo, H, W)
        if not len(act):
            break
        key = torch.full((HW,), -1, dtype=torch.int64, device=dev)
        for pid, ix, iy in _rect_chunks(*tuple(a[act] for a in T["rect"])):
            keep = todo[iy * W + ix]                      # only the pixels that may hold another volume
            pid, ix, iy = act[pid[keep]], ix[keep], iy[keep]
            pix = iy * W + ix
            v = _own_values(P, pid, ix, iy, dt, fo)
            m = v > EPS_TOUCH
            if b:
                m &= layer_of(pn[pid], pz[pid], pr[pid], pix) == len(b)
            key.scatter_reduce_(0, pix[m], _key(v[m], pid[m]), reduce="amax")
            pairs += len(pid)
        nb = chosen(key)
        todo = nb[0] >= 0
        if not bool(todo.any()):
            break
        b.append(nb)
        if len(b) == 1:
            todo = multi.clone()
    stats["composite_layers"] = [int((bb[0] >= 0).sum()) for bb in b]
    stats["composite_overflow"] = int(todo.sum()) if len(b) == MAX_LAYERS else 0
    K = len(b)
    if K < 2:
        return pairs
    D = torch.zeros((K + 1) * HW, dtype=dt, device=dev)   # per volume, and the unclaimed rest
    DZ = torch.zeros((K + 1) * HW, dtype=dt, device=dev)  # ... times the depth of each bit (the volume's mean depth)
    act = _active(T["rect"], multi, H, W)
    for pid, ix, iy in _rect_chunks(*tuple(a[act] for a in T["rect"])):
        keep = multi[iy * W + ix]
        pid, ix, iy = act[pid[keep]], ix[keep], iy[keep]
        pix = iy * W + ix
        v = _tube_values(P, pid, ix, iy, fn, dt)
        j = layer_of(pn[pid], pz[pid], pr[pid], pix)
        D.index_add_(0, j * HW + pix, v)
        DZ.index_add_(0, j * HW + pix, v * pz[pid])
        pairs += len(pid)
    if S is not None:
        Q = S["Q"]
        sact = _active(S["rect"], multi, H, W)
        for pid, ix, iy in _rect_chunks(*tuple(a[sact] for a in S["rect"])):
            keep = multi[iy * W + ix]
            pid, ix, iy = sact[pid[keep]], ix[keep], iy[keep]
            pix = iy * W + ix
            v = _side_values(Q, pid, ix, iy, S["fn"], dt)
            j = layer_of(Q["nodes"][pid], Q["z"][pid], Q["rdc"][pid], pix)
            D.index_add_(0, j * HW + pix, v)
            DZ.index_add_(0, j * HW + pix, v * Q["z"][pid])
            pairs += len(pid)
    D, DZ = D.view(K + 1, HW), DZ.view(K + 1, HW)
    two = multi & (b[1][0] >= 0)
    idx = torch.nonzero(two).flatten()
    if len(idx):
        Dl, Zs = D[:K, idx].clone(), DZ[:K, idx].clone()
        rest = D[K, idx]
        Dl[0] += rest
        Zs[0] += DZ[K, idx]
        stats["composite_rest"] = float(rest.abs().max())
        lead = torch.stack([torch.where(bb[0][idx] >= 0, bb[1][idx], b[0][1][idx]) for bb in b])
        Z = torch.where(Dl > 1e-6, Zs / torch.where(Dl > 1e-6, Dl, torch.ones_like(Dl)), lead)
        out[idx] = composite_od(Dl, Z, comp["f_s"], comp["lb"]).to(dt)
    return pairs


# ------------------------------------------------------------------ public API
def vessel_od(graph: VesselGraph, k: float, shape, optics: Optics, origin=(0.0, 0.0), device="cuda",
              hct_mod=None, vids=None, pad_px=None) -> np.ndarray:
    """Optical density (Np, float32 (H, W)) of the vessels, before the halo,
    in the image whose pixel (0, 0) is at the d_c position `origin` (pixel
    coordinates k * (xy - origin)).

    hct_mod: {vid: callable(s_dc ndarray) -> multiplier ndarray} applied to
        the haematocrit along a vessel (s_dc: arclength from its upstream
        node, d_c), e.g. red-cell columns and plasma gaps in a single frame.
        Modulated vessels are sampled at 0.05 px so steps are placed to
        ~0.05 px (0.25 px if the multiplier has attribute smooth = True).
        Tube pieces and the junction stubs' segments follow the multiplier;
        the round caps use its value at the vessel's end.
    vids: render only these vessels.  Junction geometry (R, merged
        clusters) still comes from the whole graph; at a node the union is
        taken over the rendered incident vessels, so a vessel rendered alone
        keeps its round cap at the node (sum over single vessels therefore
        double-counts the small overlaps at nodes).
    pad_px: vessels whose control polygon lies entirely more than pad_px
        outside the frame are skipped before any sampling.  Default: each
        vessel's own reach k r_max + 4 s_max, which loses nothing; vessels
        that leave the frame are drawn with every piece within reach, so
        there is no fade at the border.
    """
    plan = build_plan(graph, k, shape, optics, origin, hct_mod, vids, pad_px)
    return evaluate_plan(plan, shape, device)


def _halo_blur(od: np.ndarray, sigma: float, border) -> np.ndarray:
    import cv2
    rad = int(math.ceil(4.0 * sigma))
    return cv2.GaussianBlur(od, (2 * rad + 1, 2 * rad + 1), sigma, sigmaY=sigma, borderType=border)


def _psf_terms(optics: Optics) -> list:
    """The image-wide PSF on the OD as (weight, sigma px) Gaussian terms
    besides the delta: the halo and, if set, the intermediate tail."""
    out = []
    for w, s in ((optics.halo_h, optics.halo_px), (getattr(optics, "tail_h", 0.0), getattr(optics, "tail_px", 0.0))):
        if w > 0 and s > 0.3:
            out.append((float(w), float(s)))
    return out


def apply_halo(od: np.ndarray, optics: Optics, border: str = "reflect") -> np.ndarray:
    """(1 - h - t) od + t G(tail_px) * od + h G(halo_px) * od: the skirts
    light scattered in the tissue gives every vessel (the halo; the tail,
    off by default, is a narrower second component: together a heavier-
    tailed PSF than one Gaussian).  The Gaussians are truncated at 4 sigma;
    the border is reflected (render_od pads the frame instead, so there it
    does not matter)."""
    import cv2
    od = np.asarray(od)
    dt = np.float64 if od.dtype == np.float64 else np.float32
    od = od.astype(dt)
    terms = _psf_terms(optics)
    if not terms:
        return od.copy()
    mode = {"reflect": cv2.BORDER_REFLECT, "constant": cv2.BORDER_CONSTANT}[border]
    out = (1.0 - sum(w for w, _ in terms)) * od
    for w, s in terms:
        out = out + w * _halo_blur(od, s, mode)
    return out.astype(dt)


def halo_pad(optics: Optics) -> int:
    """Margin (px) the halo needs around a frame: 4 sigmas of its widest term."""
    terms = _psf_terms(optics)
    return int(math.ceil(4.0 * max(s for _, s in terms))) if terms else 0


def render_od(graph: VesselGraph, k: float, shape, optics: Optics, origin=(0.0, 0.0), device="cuda",
              hct_mod=None, vids=None) -> np.ndarray:
    """vessel_od plus the halo.  The vessels are rendered on the frame grown
    by 4 halo sigmas, so vessels just outside the frame still cast their
    halo into it, then cropped."""
    H, W = int(shape[0]), int(shape[1])
    p = halo_pad(optics)
    org = np.asarray(origin, float) - p / float(k)
    od = vessel_od(graph, k, (H + 2 * p, W + 2 * p), optics, org, device, hct_mod, vids)
    return apply_halo(od, optics)[p:p + H, p:p + W]


# ------------------------------------------------------------------ oracle
def _cell_cover(t, nx, ny) -> np.ndarray:
    """Area fraction of a unit square cell (centred at 0) on the side
    n . x <= t of a straight line, n = (nx, ny) a unit vector."""
    p = np.maximum(np.abs(nx), np.abs(ny))
    q = np.minimum(np.abs(nx), np.abs(ny))
    at = np.abs(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        corner = 1.0 - ((p + q) / 2 - at) ** 2 / (2 * p * q)
    half = np.where(at <= (p - q) / 2, 0.5 + at / p, np.where(at >= (p + q) / 2, 1.0, corner))
    return np.where(t >= 0, half, 1.0 - half)


def oracle_od(graph: VesselGraph, k: float, shape, optics: Optics, origin=(0.0, 0.0), ss: int = 8,
              hct_mod=None, vids=None, halo: bool = True) -> np.ndarray:
    """Slow float64 reference for tests on small scenes.

    The sharp lumens are drawn on an ss x supersampled grid: every
    centreline point (0.02 px apart) carries its own staircase (same chord
    law: r, h, z and so f and W at that point); a fine pixel takes the
    staircase of its nearest centreline point.  Vessel ends at a free node
    are cut flat, ends at a junction node are round.  A vessel that crosses
    itself adds (each 1 px chunk of centreline owns the pixels whose nearest
    point it holds).  Where the lumens of two vessels that meet at a node
    (or are joined through one vessel between) overlap in 3-D (at a fine
    pixel, their depths there differ by less than the sum of their radii:
    one blood volume around a junction) only the largest staircase value is
    kept (the union).  This rule is per fine pixel, so it does not depend
    on which nodes lie on the grid or how junctions are grouped.  Every bit
    of lumen is then blurred by the exact 2-D Gaussian of its own blur
    (linear interpolation between blur layers 2 % apart), the fine grid is
    sampled at the pixel centres; with optics.composite each vessel is
    blurred on its own and the vessels stacked at a pixel are composited by
    depth (_oracle_composite, the closed form's rule; else they add), and
    the halo is applied as in render_od.  Independent of the closed form's
    pieces, stubs, R and clusters."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    H, W = int(shape[0]), int(shape[1])
    k = float(k)
    origin = np.asarray(origin, float)
    hct_mod = hct_mod or {}
    ph = halo_pad(optics) if halo else 0
    render = list(graph.vessels) if vids is None else [v for v in vids if v in graph.vessels]
    tracks = {vid: _Track(graph.vessels[vid], k, origin, optics, hct_mod.get(vid), spacing=0.25)
              for vid in render}
    smax = max([float(tr.at(tr.sig)["s"].max()) for tr in tracks.values()] + [1.0])
    pb = int(math.ceil(5.0 * smax + 2))
    Hc, Wc = H + 2 * ph, W + 2 * ph                          # coarse grid incl. halo pad, pixel -ph..
    x_start, y_start = -ph - pb, -ph - pb                     # fine grid origin (px)
    nfx, nfy = (Wc + 2 * pb - 1) * ss + 1, (Hc + 2 * pb - 1) * ss + 1
    deg = defaultdict(int)
    for e in graph.vessels.values():
        deg[e.u] += 1
        deg[e.v] += 1

    entries = {}        # vid -> (flat fine index, value, blur)
    for vid, tr in tracks.items():
        e = graph.vessels[vid]
        n = max(2, int(math.ceil(tr.Lpx / 0.02)) + 1)
        q = tr.at(np.linspace(0.0, tr.Lpx, n))
        C, r = q["C"], q["r"]
        Lv = chord_levels(q["a"], q["f"])                       # (n, 6)
        Wv = Lv - np.concatenate([Lv[:, 1:], np.zeros((n, 1))], 1)
        own_first = deg[e.u] >= 2                                # round cap at a junction, flat at a free end
        own_last = deg[e.v] >= 2
        idx_all, val_all, s_all, z_all, r_all = [], [], [], [], []
        step = 50                                                # 1 px chunks
        for i0 in range(0, n - 1, step):
            i1 = min(i0 + step, n - 1)
            j0, j1 = max(i0 - 1, 0), min(i1 + 1, n - 1)
            pts = C[j0:j1 + 1]
            rm = float(r[j0:j1 + 1].max())
            lo = np.floor((pts.min(0) - rm - 0.1 - [x_start, y_start]) * ss).astype(int)
            hi = np.ceil((pts.max(0) + rm + 0.1 - [x_start, y_start]) * ss).astype(int)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, [nfx - 1, nfy - 1])
            if np.any(hi < lo):
                continue
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            gx, gy = gx.ravel(), gy.ravel()
            P = np.stack([x_start + gx / ss, y_start + gy / ss], 1)
            d, j = cKDTree(pts).query(P, distance_upper_bound=rm + 0.75 / ss)
            ok = np.isfinite(d)
            j = np.where(ok, j + j0, 0)
            last_chunk = i1 == n - 1
            own = (j >= i0) & ((j < i1) | (last_chunk & (j == i1)))
            if not own_first:
                own &= j != 0
            if not own_last:
                own &= j != n - 1
            ok &= own
            ok &= d < r[j] + 0.75 / ss
            if not ok.any():
                continue
            jj, dd = j[ok], d[ok]
            # anti-aliased: the fraction of the fine cell inside each box, the
            # box edge taken straight across the cell (normal: away from the
            # nearest centreline point)
            v = P[ok] - C[jj]
            nrm = np.where(dd > 1e-9, dd, 1.0)
            nx_, ny_ = np.where(dd > 1e-9, v[:, 0] / nrm, 1.0), np.where(dd > 1e-9, v[:, 1] / nrm, 0.0)
            w = r[jj, None] * BOX_C
            cov = (_cell_cover((w - dd[:, None]) * ss, nx_[:, None], ny_[:, None])
                   - _cell_cover((-w - dd[:, None]) * ss, nx_[:, None], ny_[:, None]))
            idx_all.append(gy[ok] * nfx + gx[ok])
            val_all.append((cov * Wv[jj]).sum(1))
            s_all.append(q["s"][jj])
            z_all.append(q["z"][jj])
            r_all.append(q["r_dc"][jj])
        if idx_all:
            entries[vid] = [np.concatenate(a) for a in (idx_all, val_all, s_all, z_all, r_all)]

    # union: where lumens of vessels joined at a node (or through one short vessel) overlap in 3-D (in
    # projection, and their depths differ by less than r_a + r_b at those points) they are one blood volume
    # and only the largest staircase value is kept; lumens at different depths (crossings) add, and so do
    # two stretches of one vessel.  Decided per fine pixel, independent of R, clusters and crops.
    if len(entries) >= 2:
        vids_e = list(entries)
        idx = np.concatenate([entries[v][0] for v in vids_e])
        val = np.concatenate([entries[v][1] for v in vids_e])
        zz = np.concatenate([entries[v][3] for v in vids_e])
        rr = np.concatenate([entries[v][4] for v in vids_e])
        own = np.concatenate([np.full(len(entries[v][0]), i) for i, v in enumerate(vids_e)])
        pos = np.concatenate([np.arange(len(entries[v][0])) for v in vids_e])
        o = np.lexsort((-val, idx))
        idx_s = idx[o]
        start = np.r_[True, idx_s[1:] != idx_s[:-1]]
        grp = np.cumsum(start) - 1
        first = np.flatnonzero(start)
        rank = np.arange(len(o)) - first[grp]
        size = np.bincount(grp)[grp]
        multi = size >= 2
        if multi.any():
            # only vessels joined at a node, or through one vessel between (a merged junction), are one
            # blood volume; any other 3-D overlap is invalid anatomy (graph.check) and adds, like a crossing
            by_node = defaultdict(set)
            for v in graph.vessels.values():
                by_node[v.u].add(v.vid)
                by_node[v.v].add(v.vid)
            nb1 = {v: by_node[graph.vessels[v].u] | by_node[graph.vessels[v].v] for v in graph.vessels}
            M = max(graph.vessels) + 1
            codes = set()
            for i, v in enumerate(vids_e):
                near = set(nb1[v])
                for w in nb1[v]:
                    near |= nb1[w]
                codes.update(v * M + w for w in near if w != v)
            codes = np.fromiter(codes, np.int64, len(codes))
            vid_arr = np.asarray(vids_e, np.int64)
            alive = np.ones(len(o), bool)
            for p_ in range(1, int(rank.max()) + 1):
                cur = np.flatnonzero(rank == p_)
                for q_ in range(p_):
                    prev = cur - (p_ - q_)
                    a_, b_ = o[cur], o[prev]
                    kill = alive[prev] & (own[a_] != own[b_]) & (np.abs(zz[a_] - zz[b_]) < rr[a_] + rr[b_])
                    if kill.any():
                        kill[kill] = np.isin(vid_arr[own[a_[kill]]] * M + vid_arr[own[b_[kill]]], codes)
                    alive[cur[kill]] = False
            dead = o[~alive]
            for i, v in enumerate(vids_e):
                d_ = dead[own[dead] == i]
                if len(d_):
                    entries[v][1][pos[d_]] = 0.0

    # blur every bit of lumen by its own sigma: layers 2 % apart, linear weights
    cy_idx = (np.arange(Hc) + pb) * ss                         # fine rows of the coarse pixel centres
    cx_idx = (np.arange(Wc) + pb) * ss

    def blurred(parts_):
        """The coarse image of the lumen bits (fine index, value, blur) of `parts_`."""
        img_out = np.zeros((Hc, Wc))
        if not parts_:
            return img_out
        idx = np.concatenate([v[0] for v in parts_])
        val = np.concatenate([v[1] for v in parts_])
        sg = np.concatenate([v[2] for v in parts_])
        keep = val != 0
        idx, val, sg = idx[keep], val[keep], sg[keep]
        if not len(idx):
            return img_out
        s_lo, s_hi = float(sg.min()), float(sg.max())
        nl = max(1, int(math.ceil(math.log(s_hi / s_lo) / math.log(R_LAYER))) + 1) if s_hi > s_lo * 1.0001 else 1
        grid = s_lo * R_LAYER ** np.arange(nl) if nl > 1 else np.array([s_lo])
        if nl > 1:
            pos = np.clip(np.log(sg / s_lo) / math.log(R_LAYER), 0, nl - 1 - 1e-12)
            i0 = np.floor(pos).astype(int)
            wgt = (sg - grid[i0]) / (grid[i0 + 1] - grid[i0])
            parts = [(i0, 1.0 - wgt), (i0 + 1, wgt)]
        else:
            parts = [(np.zeros(len(sg), int), np.ones(len(sg)))]
        for layer in range(nl):
            fid, fval = [], []
            for li, wl in parts:
                m = (li == layer) & (wl > 0)
                fid.append(idx[m])
                fval.append(val[m] * wl[m])
            fid, fval = np.concatenate(fid), np.concatenate(fval)
            if not len(fid):
                continue
            gy, gx = np.divmod(fid, nfx)
            sgf = grid[layer] * ss
            m = int(math.ceil(5 * sgf)) + 1
            y0, y1 = max(gy.min() - m, 0), min(gy.max() + m, nfy - 1)
            x0, x1 = max(gx.min() - m, 0), min(gx.max() + m, nfx - 1)
            img = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
            np.add.at(img, (gy - y0, gx - x0), fval)
            img = gaussian_filter(img, sgf, mode="constant", truncate=5.0)
            ry = np.flatnonzero((cy_idx >= y0) & (cy_idx <= y1))
            rx = np.flatnonzero((cx_idx >= x0) & (cx_idx <= x1))
            if len(ry) and len(rx):
                img_out[np.ix_(ry, rx)] += img[np.ix_(cy_idx[ry] - y0, cx_idx[rx] - x0)]
        return img_out

    if optics.composite and len(entries) >= 2:
        out = _oracle_composite(graph, optics, entries, tracks, blurred, (Hc, Wc), ph, k, origin)
    else:
        out = blurred(list(entries.values()))
    if halo and ph:
        out = apply_halo(out, optics)
    return out[ph:ph + H, ph:ph + W]


def _oracle_composite(graph, optics, entries, tracks, blurred, shape, ph, k=None, origin=(0.0, 0.0)) -> np.ndarray:
    """The oracle's compositing of crossing vessels, by the closed form's
    rule (_composite) on exact per-vessel images: each vessel's lumen blurred
    on its own (after the junction union), with its depth and radius at a
    pixel the OD-weighted means of its blurred lumen (blurred val z / blurred
    val); per pixel the strongest vessel's blood volume (itself and the
    vessels at its depth: same_volume), then the strongest outside it, and so
    on, each volume at the OD-weighted mean depth of its blood, composited by
    depth (composite_od).  A vessel's depth is one value per pixel here (the
    closed form tests each piece), so they differ where a vessel's depth
    changes within its reach by more than the margin of same_volume and it
    does not share a node with the leading vessel."""
    Hc, Wc = shape
    vids = list(entries)
    V = len(vids)
    nd = vessel_nodes(graph)[np.asarray(vids, np.int64)]
    # (V, V, 2): the node v shares with w through its end 0 / 1 (-1: none), and whether v is w
    shared = np.where((nd[:, None, :, None] == nd[None, :, None, :]).any(3), nd[:, None, :], -1)
    ident = (nd[:, None, 0] == nd[None, :, 0]) & (nd[:, None, 1] == nd[None, :, 1])
    if k is not None:
        _junction_layout(graph, k, dict(tracks))
        nxy, nreach = _node_table(graph, k, origin, tracks, _junction_layout.last_R)
    else:                                       # no k: any shared node counts (the v2 rule)
        nxy, nreach = np.zeros((1, 2)), None
    D = np.stack([blurred([entries[v]]) for v in vids])                 # (V, Hc, Wc)
    DZ = np.stack([blurred([(entries[v][0], entries[v][1] * entries[v][3], entries[v][2])]) for v in vids])
    DR = np.stack([blurred([(entries[v][0], entries[v][1] * entries[v][4], entries[v][2])]) for v in vids])
    total = D.sum(0)
    touch = D > EPS_TOUCH
    multi = touch.sum(0) >= 2
    out = total.copy()
    py, px = np.nonzero(multi)
    if not len(py):
        return out
    Dm, Tm = D[:, py, px], touch[:, py, px]
    safe = np.where(np.abs(Dm) > 1e-12, Dm, 1.0)
    Zm, Rm = DZ[:, py, px] / safe, DR[:, py, px] / safe
    n = len(py)
    ar = np.arange(n)
    free = np.ones((V, n), bool)
    vols, lead = [], []
    while len(vols) < MAX_LAYERS:
        cand = np.where(Tm & free, Dm, -np.inf)
        b = np.argmax(cand, 0)
        has = np.isfinite(cand[b, ar])
        if not has.any():
            break
        zb, rb = Zm[b, ar], Rm[b, ar]
        same = ident[:, b].copy()
        for end in (0, 1):
            nid = shared[:, b, end]                                          # (V, n)
            if nreach is None:
                same |= nid >= 0
                continue
            ok = (nid >= 0) & (nid < len(nreach))
            ii = np.where(ok, nid, 0)
            d2 = (px[None] - ph - nxy[ii, 0]) ** 2 + (py[None] - ph - nxy[ii, 1]) ** 2
            same |= ok & (d2 <= nreach[ii] * np.abs(nreach[ii]))
        same |= np.abs(Zm - zb[None]) < Rm + rb[None] + Z_MARGIN
        same &= free & has[None]
        vols.append(same)
        lead.append(np.where(has, zb, np.nan))
        free &= ~same
    two = np.zeros(n, bool) if len(vols) < 2 else np.isfinite(lead[1])
    if not two.any():
        return out
    vols[0] = vols[0] | free                                            # unclaimed vessels join the first volume
    DZm = DZ[:, py, px]
    Dl = np.stack([(Dm * v).sum(0) for v in vols])
    Zs = np.stack([(DZm * v).sum(0) for v in vols])
    Zl = np.where(Dl > 1e-6, Zs / np.where(Dl > 1e-6, Dl, 1.0), np.where(np.isfinite(lead), lead, lead[0]))
    lb = optics.l_bypass if optics.l_bypass is not None else math.inf
    comp = composite_od(torch.as_tensor(Dl[:, two]), torch.as_tensor(Zl[:, two]), optics.f_spectral, lb).numpy()
    out[py[two], px[two]] = comp
    return out
