"""A neuromimetic, deterministic vessel annotator (see DESIGN.md).

Work in progress: stages 1-3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

cv2.setNumThreads(2)
FFT_WORKERS = 2
LAMBDA_PX = 11.9                        # vesselscene's lambda (truth.LAMBDA_PX): the length scale of a junction


@dataclass(frozen=True)
class Config:
    # 1. photoreceptors / horizontal cells
    saturation: float = 4095.0          # 12-bit full scale: glare carries no vessel information
    env_radius: float = 28.0            # px, grey closing radius of the provisional upper envelope
    bg_sigma: float = 8.0               # px, masked normalised convolution of the background
    bg_iters: int = 2
    mask_z: float = 2.0                 # band CNR above which a pixel is vessel (mask for the background)
    mask_dilate: int = 2                # px
    # 2. ganglion cells
    bands: tuple = (1.0, 2.0, 4.0, 8.0, 16.0)
    rms_sigma: float = 24.0             # px, window of the local contrast (RMS) estimate
    # 3. simple cells
    n_orient: int = 16
    scales: tuple = (1.5, 2.1, 3.0, 4.2, 6.0, 8.5, 12.0, 17.0)
    channels: tuple = ((0, 1, 2), (3, 4), (5, 6, 7))   # scale indices of the fine, medium, coarse channels
    elong: float = 3.0
    odd_alpha: float = 0.7
    # 4. non-classical surround
    surround: bool = True
    cross_k: float = 1.0                # subtract this x the isotropic (blob) part of the orientation tuning
    flank_k: float = 0.3                # subtract this x the weaker flank's iso-orientation energy
    # 5. association field
    association: bool = True
    assoc_len: float = 8.0              # px, std of the lobes' weight along the tangent
    assoc_steps: int = 3
    assoc_gain: float = 1.0
    # 6. readout
    t_high: float = 3.5
    t_low: float = 1.7
    min_len: float = 10.0               # px, shortest trace kept
    spur_len: float = 5.0               # px, terminal skeleton branches shorter than this (+ width) are cut
    # 7. graph assembly
    gap_att: float = 6.0                # px beyond the other vessel's half width an end may attach
    gap_join: float = 14.0              # px, collinear end-to-end gaps bridged
    join_turn_deg: float = 40.0
    r0: float = 4.0                     # px added to the half widths for an event's reach
    link_k: float = 1.0                 # events link within r0 + link_k x the mean of their thinner widths
    share_k: float = 1.0                # events on a common trace link within r0 + share_k x the crossing widths
    merge_k: float = 0.0                # junction discs closer than merge_k (R_a + R_b) are one region
    pass_k: float = 0.0                 # traces passing within pass_k R of a junction add their arms
    arm_min: float = 6.0                # px, shortest arm counted
    cross_turn_deg: float = 40.0        # a crossing's through pairs turn at most this much
    od_width: bool = True               # vessel widths from OD cross-sections (else 2.5 x the filter scale)
    # 8. iterative refinement
    verify: bool = True
    rounds: int = 2


DEFAULT = Config()


# ======================================================================== helpers
def _gblur(X: np.ndarray, s: float) -> np.ndarray:
    """Gaussian blur with reflecting borders; sigmas above 8 px on a 2^k downsampled grid."""
    X = np.asarray(X, np.float32)
    if s <= 0:
        return X.copy()
    H, W = X.shape
    f = 1
    while s / f > 8.0 and min(H, W) // (2 * f) >= 16:
        f *= 2
    if f == 1:
        k = 2 * int(math.ceil(3.5 * s)) + 1
        return cv2.GaussianBlur(X, (k, k), s, borderType=cv2.BORDER_REFLECT)
    h, w = -(-H // f), -(-W // f)
    Xp = cv2.copyMakeBorder(X, 0, h * f - H, 0, w * f - W, cv2.BORDER_REFLECT)
    small = cv2.resize(Xp, (w, h), interpolation=cv2.INTER_AREA)
    sb = math.sqrt(max((s / f) ** 2 - 1.0 / 12 - 0.25, 0.25))   # the box average and the upsampling blur too
    k = 2 * int(math.ceil(3.5 * sb)) + 1
    small = cv2.GaussianBlur(small, (k, k), sb, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w * f, h * f), interpolation=cv2.INTER_LINEAR)[:H, :W]


def _masked_mean(X: np.ndarray, w: np.ndarray, s: float, levels: int = 4, a: float = 0.03) -> np.ndarray:
    """Normalised convolution sum G_s*(X w) / sum G_s*w over the scales s 2^k, weighted a^k: the masked local
    mean, with holes wider than the kernel filled from the next coarser scale (smoothly)."""
    w = np.asarray(w, np.float32)
    Xw = np.asarray(X, np.float32) * w
    num = np.zeros(X.shape, np.float32)
    den = np.zeros(X.shape, np.float32)
    for k in range(levels):
        c = a ** k
        num += c * _gblur(Xw, s * 2 ** k)
        den += c * _gblur(w, s * 2 ** k)
    tot = float(w.sum())
    g = float(Xw.sum()) / tot if tot > 0 else 0.0
    eps = a ** levels
    return (num + eps * g) / (den + eps)


def _robust_rms(R: np.ndarray, w: np.ndarray) -> float:
    v = np.abs(R[w]) if w.any() else np.abs(R).ravel()
    return float(1.4826 * np.median(v)) + 1e-9


def _local_rms(R: np.ndarray, w: np.ndarray, s: float) -> np.ndarray:
    """Robust local RMS of a zero-mean band R over the pixels w: R^2 Winsorised at (3 x the global robust RMS)^2,
    then its masked local mean."""
    g = _robust_rms(R, w)
    Q = np.minimum(R * R, np.float32((3.0 * g) ** 2))
    return np.sqrt(np.maximum(_masked_mean(Q, w, s), (0.05 * g) ** 2))


# ======================================================================== 1. photoreceptors, horizontal cells
def _upper_envelope(L: np.ndarray, radius: float) -> np.ndarray:
    """Provisional background: grey closing (a dark structure narrower than the disc is filled from its
    surround) of the lightly smoothed log image, on a grid downsampled 4x, then smoothed."""
    H, W = L.shape
    f = 4 if min(H, W) >= 128 else 1
    h, w = -(-H // f), -(-W // f)
    Lp = cv2.copyMakeBorder(L, 0, h * f - H, 0, w * f - W, cv2.BORDER_REFLECT)
    small = cv2.resize(Lp, (w, h), interpolation=cv2.INTER_AREA)
    r = max(1, int(round(radius / f)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pad = r + 1
    sp = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    sp = cv2.morphologyEx(sp, cv2.MORPH_CLOSE, disc)[pad:-pad, pad:-pad]
    sp = _gblur(sp, max(1.0, r / 2))
    return cv2.resize(sp, (w * f, h * f), interpolation=cv2.INTER_LINEAR)[:H, :W]


def photoreceptors(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT) -> dict:
    """Stage 1, photoreceptors and horizontal cells: a logarithmic (Weber) response adapted to the local mean
    of the background around it.

    Biology: cones respond roughly to log intensity; horizontal cells pool the cone signals over a wide
    surround and feed the pooled mean back, so the output is the deviation of log intensity from the local
    background (light adaptation).
    Maths: L = ln I, invalid (NaN) and saturated (glare, full scale) pixels filled from the nearest valid one.
    A provisional background B0 is the robust upper envelope (grey closing over a disc wider than the widest
    vessel). Then twice: OD = B - L, the vesselness mask M = dilate(band CNR(OD) > mask_z or OD > 3 RMS) (the
    ganglion-cell rule of stage 2 with the current background pixels), and B is refitted ONLY on the pixels
    outside M (its negative) by masked normalised convolution, G*(L w) / G*w with w = valid & ~M, holes filled
    from coarser scales. The target is OD = B - L in nepers, vessels positive (Beer-Lambert: overlapping
    vessels add their densities), 0 on invalid pixels.
    Returns dict(OD, B, L, ok (valid data), bg (background pixels), mask)."""
    I = np.asarray(image, np.float32)
    ok = np.asarray(valid, bool) & np.isfinite(I) & (I > 0) & (I < cfg.saturation)
    L = np.zeros(I.shape, np.float32)
    L[ok] = np.log(I[ok])
    if not ok.all():
        iy, ix = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        L = L[iy, ix]
    B = _upper_envelope(L, cfg.env_radius)
    OD = B - L
    S = _gblur(OD, 1.0)
    m = float(np.median(S[ok]))
    bg = ok & (S < m + 2.0 * _robust_rms(S - m, ok))
    mask = ~bg
    for _ in range(cfg.bg_iters):
        Z, _rms = _band_cnr(OD, bg, cfg)
        S = _gblur(OD, 1.0)
        sr = _local_rms(S - np.median(S[bg]), bg, cfg.rms_sigma * 2)
        mask = (Z > cfg.mask_z) | (S > 3.0 * sr)
        if cfg.mask_dilate:
            k = 2 * cfg.mask_dilate + 1
            mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        bg = ok & ~mask
        B = _masked_mean(L, bg, cfg.bg_sigma)
        OD = B - L
    OD = np.where(ok, OD, 0.0).astype(np.float32)
    return dict(OD=OD, B=B, L=L, ok=ok, bg=bg, mask=mask)


# ======================================================================== 2. ganglion cells
def _band_cnr(OD: np.ndarray, bg: np.ndarray, cfg: Config):
    Z = np.full(OD.shape, -np.inf, np.float32)
    rms = {}
    for s in cfg.bands:
        D = _gblur(OD, s) - _gblur(OD, 2 * s)
        r = _local_rms(D, bg, max(cfg.rms_sigma, 4 * s))
        rms[s] = r
        np.maximum(Z, D / r, out=Z)
    return Z, rms


def ganglion_cells(OD: np.ndarray, bg: np.ndarray, cfg: Config = DEFAULT) -> dict:
    """Stage 2, OFF-centre ganglion cells with contrast gain control.

    Biology: OFF-centre retinal ganglion cells respond to a dark centre against a brighter surround
    (difference of Gaussians); their gain is divided by the local contrast (RMS) of the scene, so the output
    is a contrast-to-noise ratio rather than a raw contrast.
    Maths: per band s, D_s = (G_s - G_2s) * OD, divided by the robust local RMS of the same band over the
    background pixels (Winsorised, masked normalised convolution of D_s^2); Z = max_s D_s / RMS_s. This is the
    band-matched CNR the truth uses to decide what is observable.
    Returns dict(Z, rms={s: local RMS map})."""
    Z, rms = _band_cnr(OD, bg, cfg)
    return dict(Z=Z, rms=rms)


# ======================================================================== 3. simple cells
def orientations(cfg: Config = DEFAULT) -> np.ndarray:
    """Tangent angles theta_k = k pi / n (x right, y down)."""
    return np.arange(cfg.n_orient) * (np.pi / cfg.n_orient)


def simple_cells(OD: np.ndarray, bg: np.ndarray, cfg: Config = DEFAULT) -> list:
    """Stage 3, V1 simple cells: an orientation score U(x, y, theta) per spatial-frequency channel.

    Biology: simple cells in V1 have elongated receptive fields with an excitatory centre stripe and two
    inhibitory flanks (even-symmetric, line detectors) or one excitatory and one inhibitory half
    (odd-symmetric, edge detectors), at all orientations and in separate spatial-frequency channels. A
    quadrature pair of them gives the local energy and phase: a line is where the even cell dominates its
    odd partner.
    Maths: with t = (cos theta, sin theta) the tangent and n the normal, the even filter is
    -sigma^2 d^2/dn^2 of an anisotropic Gaussian (std sigma across, elong sigma along), the odd partner
    sigma d/dn of the same Gaussian; both applied exactly in the Fourier domain (reflect-padded FFT, so there
    is no truncation and the 16 orientations are equivalent). Each channel is divided by the robust local RMS
    of its scale over the background pixels (pooled over orientations): the kernel's noise gain and the local
    texture level both cancel, the response is a CNR. Line evidence per scale z = z_even - odd_alpha |z_odd|
    (phase gating: at the wall of a wide, flat-floored vein the small-scale even response is an edge echo with
    a large odd partner and is rejected; at a centre the odd response vanishes by symmetry).
    The scales are grouped into channels (cfg.channels: fine, medium, coarse); per channel U = max over its
    scales and S = the argmax scale. Keeping the channels apart matters: a thin vessel crossing a wide vein
    lives in the fine channel, where the vein's flat floor gives no response, whereas a single max over all
    scales lets the vein's broad orientation tuning swamp it.
    Returns [dict(U (n_orient, H, W), S (scale index into cfg.scales), sigma (the channel's typical scale))]."""
    H, W = OD.shape
    th = orientations(cfg)
    smax = max(cfg.scales)
    P = int(min(math.ceil(3 * cfg.elong * smax), H - 1, W - 1))
    X = np.pad(OD.astype(np.float32), P, mode="reflect")
    Hp, Wp = sfft.next_fast_len(X.shape[0], real=True), sfft.next_fast_len(X.shape[1], real=True)
    F = sfft.rfft2(X, s=(Hp, Wp), workers=FFT_WORKERS).astype(np.complex64)
    wy = (2 * np.pi * sfft.fftfreq(Hp)).astype(np.float32)[:, None]
    wx = (2 * np.pi * sfft.rfftfreq(Wp)).astype(np.float32)[None, :]
    out = []
    for chan in cfg.channels:
        U = np.full((len(th), H, W), -np.inf, np.float32)
        S = np.zeros((len(th), H, W), np.uint8)
        for si in chan:
            s = cfg.scales[si]
            E = np.empty((len(th), H, W), np.float32)
            O = np.empty((len(th), H, W), np.float32)
            for k, t in enumerate(th):
                c, sn = np.float32(math.cos(t)), np.float32(math.sin(t))
                wn = -wx * sn + wy * c
                wt = wx * c + wy * sn
                G = np.exp(-0.5 * (s * s) * (wn * wn + (cfg.elong ** 2) * wt * wt)).astype(np.float32)
                e = sfft.irfft2(F * ((s * s) * wn * wn * G), s=(Hp, Wp), workers=FFT_WORKERS)
                o = sfft.irfft2(F * (1j * s * wn * G).astype(np.complex64), s=(Hp, Wp), workers=FFT_WORKERS)
                E[k] = e[P:P + H, P:P + W]
                O[k] = o[P:P + H, P:P + W]
            ws = max(cfg.rms_sigma, 4 * s)
            re = _local_rms_stack(E, bg, ws)
            ro = _local_rms_stack(O, bg, ws)
            Z = E / re[None] - cfg.odd_alpha * np.abs(O) / ro[None]
            better = Z > U
            U = np.where(better, Z, U)
            S[better] = si
        out.append(dict(U=U, S=S, sigma=float(np.exp(np.mean(np.log([cfg.scales[i] for i in chan]))))))
    return out


def _local_rms_stack(E: np.ndarray, bg: np.ndarray, s: float) -> np.ndarray:
    """Robust local RMS of a stack of zero-mean channels (pooled over the first axis) over the pixels bg."""
    g = float(1.4826 * np.median(np.abs(E[:, bg]))) + 1e-9
    Q = np.minimum(E * E, np.float32((3.0 * g) ** 2)).mean(0)
    return np.sqrt(np.maximum(_masked_mean(Q, bg, s), (0.05 * g) ** 2))


# ======================================================================== 4. non-classical surround
def _segment_kernel(points: np.ndarray, weights: np.ndarray, lat: float = 0.7) -> tuple[np.ndarray, tuple]:
    """A correlation kernel (cv2.filter2D) that sums src(p + q) w over the points q (x, y), each splatted as a
    small Gaussian of std lat; normalised to sum 1. Returns (kernel, anchor)."""
    r = int(math.ceil(np.abs(points).max() + 3 * lat)) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    K = np.zeros_like(xx)
    for (qx, qy), w in zip(points, weights):
        K += w * np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * lat * lat))
    K /= K.sum()
    return K.astype(np.float32), (r, r)


def _corr(X: np.ndarray, K: np.ndarray, anchor) -> np.ndarray:
    return cv2.filter2D(X, cv2.CV_32F, K, anchor=anchor, borderType=cv2.BORDER_CONSTANT)


def surround(U: np.ndarray, sigma: float, cfg: Config = DEFAULT) -> np.ndarray:
    """Stage 4, the non-classical receptive field: cross-orientation and iso-orientation (flank) suppression.

    Biology: a V1 cell's response is divided (normalised) by the pooled activity of cells of all orientations
    at its location (cross-orientation suppression), and suppressed by cells of its own orientation in the
    surround (iso-orientation surround suppression): a contour on an empty background is salient, the same
    contour within texture is not.
    Maths: with P = max(U, 0), the isotropic part of the orientation tuning at a pixel is the mean of its
    lower half over theta (a blob responds at all orientations, a line or a crossing at one or two: about
    0.04 of the peak for a line here, so several peaks per pixel survive); U1 = U - cross_k iso. The flank
    energy F_theta is P_theta averaged over a strip on each side of the line, at d1 = 2.5 sigma + 1 to
    d1 + max(6, 1.5 sigma) px (sigma: the channel's scale), and only the weaker of the two flanks counts
    (texture surrounds a pixel on both sides, a neighbour in a vessel bundle on one only):
    U2 = U1 - flank_k min(F_left, F_right)."""
    n = U.shape[0]
    P = np.maximum(U, 0)
    iso = np.sort(P, axis=0)[: n // 2].mean(0)
    out = U - cfg.cross_k * iso[None]
    if cfg.flank_k <= 0:
        return out
    th = orientations(cfg)
    d1 = 2.5 * sigma + 1.0
    d2 = d1 + max(6.0, 1.5 * sigma)
    step = max(1.0, sigma / 2)
    ds = np.arange(d1, d2 + 0.5 * step, step)
    us = np.arange(-1.5 * step, 1.5 * step + 0.01, step)
    for k, t in enumerate(th):
        tx, ty = math.cos(t), math.sin(t)
        nx, ny = -ty, tx
        pts = np.array([(d * nx + u * tx, d * ny + u * ty) for d in ds for u in us])
        Kl, al = _segment_kernel(pts, np.ones(len(pts)), lat=max(0.7, step / 2))
        Kr, ar = _segment_kernel(-pts, np.ones(len(pts)), lat=max(0.7, step / 2))
        out[k] -= cfg.flank_k * np.minimum(_corr(P[k], Kl, al), _corr(P[k], Kr, ar))
    return out


# ======================================================================== 5. association field
def association_field(U: np.ndarray, sigma: float, cfg: Config = DEFAULT) -> np.ndarray:
    """Stage 5, horizontal connections in V1: the association field as a deterministic diffusion in (x, y,
    theta).

    Biology: long-range horizontal connections link cells with co-linear (co-circular) receptive fields
    (Field, Hayes and Hess 1993); bipole cells (Grossberg and Mingolla) fire for a gap only when both of their
    lobes, ahead and behind along the preferred orientation, are driven, so contours are completed across
    gaps but are not extended past their ends. Williams and Jacobs' stochastic completion field is the same
    computation as the steady state of a random walk of particles moving along their orientation.
    Maths: P = max(C, 0) pooled over the neighbouring orientation layers (max of P_k and 0.8 P_k+-1: the walk
    may turn by one layer, co-circularity); the lobes A+-_k = sum_d w(d) P_k(p +- d t_k) with
    w(d) = exp(-d^2 / 2 l^2), l = assoc_len sqrt(max(1, sigma / 2)) for a channel of scale sigma,
    d = 1..3 l, laterally spread 0.7 px, normalised; the bipole
    B_k = sqrt(A+_k A-_k) (both lobes needed). A fixed number of steps C <- (U + g B(C)) / (1 + g) with
    C0 = max(U, 0): on a continuous line C = U, across a gap C rises to g / (1 + g) of the line, a texture
    blob without collinear support falls to U / (1 + g)."""
    n = U.shape[0]
    th = orientations(cfg)
    U0 = np.maximum(U, 0)
    g = float(cfg.assoc_gain)
    ell = cfg.assoc_len * math.sqrt(max(1.0, sigma / 2.0))
    ds = np.arange(1.0, 3 * ell + 0.5, 1.0)
    w = np.exp(-ds ** 2 / (2 * ell ** 2))
    kern = []
    for t in th:
        tx, ty = math.cos(t), math.sin(t)
        pts = np.stack([ds * tx, ds * ty], 1)
        kern.append((_segment_kernel(pts, w), _segment_kernel(-pts, w)))
    C = U0.copy()
    for _ in range(cfg.assoc_steps):
        Pm = np.maximum(C, 0.8 * np.maximum(np.roll(C, 1, 0), np.roll(C, -1, 0)))
        Bp = np.empty_like(C)
        for k in range(n):
            (Kf, af), (Kb, ab) = kern[k]
            Bp[k] = np.sqrt(np.maximum(_corr(Pm[k], Kf, af), 0) * np.maximum(_corr(Pm[k], Kb, ab), 0))
        C = (U0 + g * Bp) / (1.0 + g)
    return C


# ======================================================================== 6. readout
def _shift(X: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """X sampled at (x + dx, y + dy), bilinear, borders replicated."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(X, M, (X.shape[1], X.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _se2_components(cand: np.ndarray):
    """Connected components of a (theta, y, x) bool volume, 26-connected, theta circular. Returns the label
    volume (0 = none) and the number of labels (labels renumbered 1..n in order of first appearance)."""
    n = cand.shape[0]
    ext = np.concatenate([cand, cand[:1]], 0)
    lab, nl = ndi.label(ext, structure=np.ones((3, 3, 3), bool))
    dsu = _DSU(nl + 1)
    a, b = lab[0], lab[n]
    m = (a > 0) & (b > 0)
    for i, j in zip(a[m].tolist(), b[m].tolist()):
        dsu.union(i, j)
    # layer n duplicates layer 0: also join labels that touch layer 0's twin across layer n-1
    root = np.array([dsu.find(i) for i in range(nl + 1)], np.int64)
    lab = root[lab[:n]]
    u, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape).astype(np.int32), len(u) - 1 if u[0] == 0 else len(u)


_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_paths(sk: np.ndarray):
    """The node-to-node pixel paths of a 1-px skeleton under m-adjacency (a diagonal step only where neither
    4-neighbour between them is set, so staircases are not branch points). Returns a list of (n, 2) int
    arrays of (y, x), in a deterministic order."""
    H, W = sk.shape
    ys, xs = np.nonzero(sk)
    if not len(ys):
        return []
    idx = -np.ones((H + 2, W + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(len(ys))
    nbrs = [[] for _ in range(len(ys))]
    for dy, dx in _OFFS:
        j = idx[ys + 1 + dy, xs + 1 + dx]
        ok = j >= 0
        if dy and dx:
            ok &= (idx[ys + 1 + dy, xs + 1] < 0) & (idx[ys + 1, xs + 1 + dx] < 0)
        for i in np.flatnonzero(ok).tolist():
            nbrs[i].append(int(j[i]))
    deg = np.array([len(v) for v in nbrs])
    node = deg != 2
    seen_edge = set()
    paths = []
    for s in np.flatnonzero(node).tolist():
        for nb in nbrs[s]:
            if (s, nb) in seen_edge:
                continue
            path = [s, nb]
            seen_edge.add((s, nb))
            prev, cur = s, nb
            while not node[cur]:
                nxt = [q for q in nbrs[cur] if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            seen_edge.add((cur, prev))
            paths.append(path)
    on_path = np.zeros(len(ys), bool)
    for p in paths:
        on_path[p] = True
    for s in range(len(ys)):                         # cycles without a node
        if on_path[s]:
            continue
        path = [s]
        on_path[s] = True
        prev, cur = -1, s
        while True:
            nxt = [q for q in nbrs[cur] if q != prev and (not on_path[q] or q == s)]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if cur == s:
                break
            on_path[cur] = True
        paths.append(path)
    return [np.stack([ys[p], xs[p]], 1) for p in paths]


def _resample(P: np.ndarray, step: float = 1.0) -> np.ndarray:
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) < 2:
        return P.copy()
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if d[-1] <= 1e-9:
        return P[:1].copy()
    m = max(2, int(round(d[-1] / step)) + 1)
    s = np.linspace(0, d[-1], m)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1)


def _smooth(P: np.ndarray, win: int = 5) -> np.ndarray:
    """Moving average of a polyline with its end points kept."""
    if len(P) <= 2 or win <= 1:
        return P.copy()
    h = win // 2
    Q = np.pad(P, ((h, h), (0, 0)), mode="reflect", reflect_type="odd")
    k = np.ones(win) / win
    out = np.stack([np.convolve(Q[:, 0], k, "valid"), np.convolve(Q[:, 1], k, "valid")], 1)
    out[0], out[-1] = P[0], P[-1]
    return out


def _plen(P: np.ndarray) -> float:
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


def readout(C: np.ndarray, S: np.ndarray, cfg: Config = DEFAULT, t_high=None, t_low=None, chan: int = 0) -> list:
    """Stage 6, readout: non-maximum suppression across space and orientation, hysteresis in SE(2), tracing.

    Maths: a sample (theta_k, y, x) is a candidate where C_k is a maximum along the normal n_k (bilinear
    neighbours at +-1 px) and over orientation (C_k >= C_k-1, C_k > C_k+1: several orientations may peak at
    one pixel, two crossing lines live in different layers) and C_k > t_low. Candidates are linked in SE(2)
    (26-neighbours in (theta, y, x), theta circular), and a component is kept when it reaches t_high: a line
    is traced in its own orientation layers, so it passes straight through a crossing instead of merging
    with the other line. Each component's (x, y) footprint is thinned to a 1-px skeleton, short terminal
    spurs (< spur_len + its width) are cut, and the node-to-node paths are smoothed and resampled at 1 px.
    Returns traces: dicts with xy (n, 2), c (strength), w (width estimate 2.5 sigma of the selected scale)."""
    from skimage.morphology import skeletonize
    t_high = cfg.t_high if t_high is None else t_high
    t_low = cfg.t_low if t_low is None else t_low
    n, H, W = C.shape
    th = orientations(cfg)
    cand = np.zeros(C.shape, bool)
    for k, t in enumerate(th):
        nx, ny = -math.sin(t), math.cos(t)
        a, b = _shift(C[k], nx, ny), _shift(C[k], -nx, -ny)
        cand[k] = (C[k] >= a) & (C[k] > b) & (C[k] > t_low)
    cand &= (C >= np.roll(C, 1, 0)) & (C > np.roll(C, -1, 0))
    lab, nl = _se2_components(cand)
    if nl == 0:
        return []
    kk, yy, xx = np.nonzero(lab)
    ll = lab[kk, yy, xx]
    cv = C[kk, yy, xx]
    mx = np.zeros(nl + 1, np.float32)
    np.maximum.at(mx, ll, cv)
    order = np.argsort(ll, kind="stable")
    cuts = np.flatnonzero(np.diff(ll[order])) + 1
    sig = np.asarray(cfg.scales, np.float32)
    Cmax = C.max(0)
    Smax = sig[np.take_along_axis(S, C.argmax(0)[None], 0)[0]]
    traces = []
    for grp in np.split(order, cuts):
        l_ = int(ll[grp[0]])
        if mx[l_] < t_high or len(grp) < cfg.min_len * 0.7:
            continue
        y0, y1, x0, x1 = yy[grp].min(), yy[grp].max(), xx[grp].min(), xx[grp].max()
        m = np.zeros((y1 - y0 + 3, x1 - x0 + 3), bool)
        m[yy[grp] - y0 + 1, xx[grp] - x0 + 1] = True
        sk = skeletonize(m)
        paths = _skeleton_paths(sk)
        paths = _prune_spurs(paths, cfg, Smax, y0 - 1, x0 - 1)
        for p in paths:
            xy = np.stack([p[:, 1] + x0 - 1, p[:, 0] + y0 - 1], 1).astype(float)
            if _plen(xy) < cfg.min_len:
                continue
            xy = _resample(_smooth(xy, 5), 1.0)
            ix = np.clip(np.round(xy).astype(int), 0, [W - 1, H - 1])
            traces.append(dict(xy=xy, c=Cmax[ix[:, 1], ix[:, 0]], w=2.5 * Smax[ix[:, 1], ix[:, 0]], chan=chan))
    return traces


def _tangents(P: np.ndarray, k: int = 3) -> np.ndarray:
    n = len(P)
    i0, i1 = np.clip(np.arange(n) - k, 0, n - 1), np.clip(np.arange(n) + k, 0, n - 1)
    v = P[i1] - P[i0]
    return v / np.maximum(np.hypot(*v.T), 1e-9)[:, None]


def merge_channels(traces: list, cfg: Config = DEFAULT) -> list:
    """Merge the traces of the spatial-frequency channels: strongest first, a point is dropped where an
    accepted trace runs parallel (< 20 deg) within 3 px, or parallel (< 10 deg) within 0.6 of its width when
    that trace is from a coarser channel (the wall echoes of a wide vein; a thin vessel crossing the vein at a
    shallow angle is kept); the uncovered runs of at least min_len are kept."""
    from scipy.spatial import cKDTree
    order = sorted(range(len(traces)), key=lambda i: (-float(np.mean(traces[i]["c"])),
                                                      float(traces[i]["xy"][0, 0]), float(traces[i]["xy"][0, 1])))
    cos_par, cos_wall = math.cos(math.radians(20.0)), math.cos(math.radians(10.0))
    acc, A_xy, A_t, A_w, A_ch = [], [], [], [], []
    tree = None
    for i in order:
        t = traces[i]
        P = t["xy"]
        Tg = _tangents(P)
        cov = np.zeros(len(P), bool)
        if tree is not None:
            rmax = max(3.0, 0.6 * float(np.max(A_w)))
            for k, lst in enumerate(tree.query_ball_point(P, rmax)):
                for q in lst:
                    d = math.hypot(*(P[k] - A_xy[q]))
                    c = abs(float(np.dot(Tg[k], A_t[q])))
                    if (d <= 3.0 and c >= cos_par) or (A_ch[q] > t["chan"] and d <= 0.6 * A_w[q] and c >= cos_wall):
                        cov[k] = True
                        break
        keep = ~cov
        lab, nl = ndi.label(keep)
        added = False
        for r in range(1, nl + 1):
            ii = np.flatnonzero(lab == r)
            if len(ii) < cfg.min_len:
                continue
            piece = {k_: (v[ii[0]:ii[-1] + 1].copy() if isinstance(v, np.ndarray) else v) for k_, v in t.items()}
            acc.append(piece)
            added = True
        if added:
            A_xy = np.concatenate([a["xy"] for a in acc])
            A_t = np.concatenate([_tangents(a["xy"]) for a in acc])
            A_w = np.concatenate([a["w"] for a in acc])
            A_ch = np.concatenate([np.full(len(a["xy"]), a["chan"]) for a in acc])
            tree = cKDTree(A_xy)
    return acc


def _prune_spurs(paths, cfg, Smax, oy, ox):
    """Cut terminal branches shorter than spur_len + the local width and rejoin paths through nodes left with
    two branches."""
    def key(p):
        return (int(p[0][0]), int(p[0][1]))
    for _ in range(2):
        if len(paths) <= 1:
            return paths
        ends = {}
        for i, p in enumerate(paths):
            for e in (tuple(p[0]), tuple(p[-1])):
                ends.setdefault(e, []).append(i)
        drop = set()
        for i, p in enumerate(paths):
            a, b = tuple(p[0]), tuple(p[-1])
            na, nb = len(ends[a]), len(ends[b])
            free_a = na == 1 and _degree_at(paths, a) == 1
            free_b = nb == 1 and _degree_at(paths, b) == 1
            L = len(p)
            wloc = 2.5 * float(Smax[min(p[len(p) // 2][0] + oy, Smax.shape[0] - 1),
                                    min(p[len(p) // 2][1] + ox, Smax.shape[1] - 1)])
            if (free_a ^ free_b) and L < cfg.spur_len + 0.5 * wloc:
                drop.add(i)
        if not drop:
            break
        paths = [p for i, p in enumerate(paths) if i not in drop]
        paths = _merge_through(paths)
    return paths


def _degree_at(paths, pt):
    return sum((tuple(p[0]) == pt) + (tuple(p[-1]) == pt) for p in paths)


def _merge_through(paths):
    """Join paths that meet at a pixel shared by exactly two path ends (and nothing else)."""
    changed = True
    paths = [np.asarray(p) for p in paths]
    while changed:
        changed = False
        ends = {}
        for i, p in enumerate(paths):
            ends.setdefault(tuple(p[0]), []).append((i, 0))
            ends.setdefault(tuple(p[-1]), []).append((i, 1))
        for pt in sorted(ends):
            lst = ends[pt]
            if len(lst) == 2 and lst[0][0] != lst[1][0]:
                (i, ei), (j, ej) = lst
                a = paths[i] if ei == 1 else paths[i][::-1]
                b = paths[j] if ej == 0 else paths[j][::-1]
                new = np.concatenate([a, b[1:]], 0)
                paths = [p for k, p in enumerate(paths) if k not in (i, j)] + [new]
                changed = True
                break
    return paths


# ======================================================================== widths from the OD
def _bilinear(X: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ndi.map_coordinates(X, [y, x], order=1, mode="nearest")


def od_widths(OD: np.ndarray, xy: np.ndarray, w0: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Full width at half maximum and peak OD of a vessel along a trace, from cross-sections of the cleaned
    OD target (not from any filter response): per sample every `step` px, the profile along the normal over
    +-max(6, 1.5 w0) px, its baseline the median of the outer fifth on each side, its half-maximum crossings
    on each side; then a running median of 7 along the trace and interpolation back to every point."""
    n = len(xy)
    if n < 3:
        return np.asarray(w0, float).copy(), np.zeros(n)
    T = _tangents(xy, 2)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    ii = np.arange(0, n, step)
    U = float(min(max(6.0, 1.5 * float(np.median(w0))), 60.0))
    us = np.arange(-U, U + 0.01, 0.5)
    X = xy[ii, None, 0] + us[None, :] * N[ii, None, 0]
    Y = xy[ii, None, 1] + us[None, :] * N[ii, None, 1]
    Pf = _bilinear(OD, X.ravel(), Y.ravel()).reshape(X.shape)
    m = len(us)
    c = m // 2
    k = max(2, m // 5)
    base = 0.5 * (np.median(Pf[:, :k], 1) + np.median(Pf[:, -k:], 1))
    pk = Pf[:, c - 2:c + 3].max(1) - base
    half = base + 0.5 * pk
    below = Pf < half[:, None]
    right = np.argmax(below[:, c:], 1).astype(float)
    right[~below[:, c:].any(1)] = m - c
    left = np.argmax(below[:, c::-1], 1).astype(float)
    left[~below[:, c::-1].any(1)] = c
    fw = (left + right) * 0.5
    fw = np.where(pk > 0, fw, np.nan)
    fw = ndi.median_filter(np.nan_to_num(fw, nan=float(np.nanmedian(fw)) if np.isfinite(fw).any() else 2.0), 7,
                           mode="nearest")
    pk = ndi.median_filter(pk, 7, mode="nearest")
    return np.interp(np.arange(n), ii, fw), np.interp(np.arange(n), ii, pk)


# ======================================================================== 7. graph assembly (end-stopped cells)
def _end_dir(P: np.ndarray, which: int, k: int = 6) -> np.ndarray:
    """Outward unit direction of a polyline at its start (which=0) or end (1), over k px."""
    if which == 0:
        a, b = P[min(k, len(P) - 1)], P[0]
    else:
        a, b = P[max(len(P) - 1 - k, 0)], P[-1]
    v = b - a
    n = math.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _angle(u, v) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v))))))


def _join_gaps(traces: list, cfg: Config) -> list:
    """Bridge collinear end-to-end gaps (mutual best pairs, repeated): the completion the association field
    did not reach."""
    from scipy.spatial import cKDTree
    for _ in range(50):
        if len(traces) < 2:
            break
        P = np.array([t["xy"][e * -1 if e else 0] for t in traces for e in (0, 1)])
        D = np.array([_end_dir(t["xy"], e) for t in traces for e in (0, 1)])
        tree = cKDTree(P)
        pairs = tree.query_pairs(cfg.gap_join, output_type="ndarray")
        best = {}
        for a, b in sorted(map(tuple, pairs)):
            ta, tb = a // 2, b // 2
            if ta == tb:
                continue
            g = P[b] - P[a]
            gl = math.hypot(*g)
            turn = _angle(D[a], -D[b])
            if gl > 2:
                gu = g / gl
                turn = max(turn, _angle(D[a], gu), _angle(-D[b], gu))
            if turn > cfg.join_turn_deg:
                continue
            cost = gl + 0.3 * turn
            for x, y in ((a, b), (b, a)):
                if x not in best or cost < best[x][0]:
                    best[x] = (cost, y)
        done, used = False, set()
        new = []
        for a in sorted(best):
            b = best[a][1]
            if a < b and best.get(b, (0, -1))[1] == a and a // 2 not in used and b // 2 not in used:
                ta, tb = traces[a // 2], traces[b // 2]
                A = ta if a % 2 == 1 else _rev(ta)
                B = tb if b % 2 == 0 else _rev(tb)
                new.append(_concat(A, B))
                used |= {a // 2, b // 2}
                done = True
        if not done:
            break
        traces = [t for i, t in enumerate(traces) if i not in used] + new
    return traces


def _rev(t: dict) -> dict:
    return {k: (v[::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()}


def _concat(A: dict, B: dict) -> dict:
    gap = _resample(np.stack([A["xy"][-1], B["xy"][0]]), 1.0)[1:-1]
    m = len(gap)
    out = dict(xy=np.concatenate([A["xy"], gap, B["xy"]], 0))
    for k in ("c", "w"):
        fill = np.full(m, 0.5 * (A[k][-1] + B[k][0]), float)
        out[k] = np.concatenate([A[k], fill, B[k]])
    return out


def _events(traces: list, cfg: Config) -> list:
    """Where a trace ends on another one (T: end-stopping) or two traces cross (X). Each event: dict(p, tr
    (trace ids), wide / thin (the wider and the thinner vessel's width there), kind, att ((trace, end,
    attachment point) for a T))."""
    from scipy.spatial import cKDTree
    from shapely import STRtree, LineString
    if not traces:
        return []
    pts = np.concatenate([t["xy"] for t in traces])
    own = np.concatenate([np.full(len(t["xy"]), i) for i, t in enumerate(traces)])
    idx = np.concatenate([np.arange(len(t["xy"])) for t in traces])
    wall = np.concatenate([t["w"] for t in traces])
    tree = cKDTree(pts)
    wmax = float(wall.max())
    ev = []
    for i, t in enumerate(traces):
        n = len(t["xy"])
        for e in (0, 1):
            p = t["xy"][0 if e == 0 else -1]
            d = _end_dir(t["xy"], e)
            ie = 0 if e == 0 else n - 1
            cand = tree.query_ball_point(p, cfg.gap_att + wmax / 2)
            best = None
            for q in sorted(cand):
                j, m = int(own[q]), int(idx[q])
                if j == i and abs(m - ie) < 3 * (cfg.gap_att + t["w"][ie]):
                    continue
                v = pts[q] - p
                dist = math.hypot(*v)
                if dist > cfg.gap_att + wall[q] / 2 or float(np.dot(v, d)) < -max(2.0, wall[q] / 2):
                    continue
                key = (dist - wall[q] / 2, j, m)
                if best is None or key < best[0]:
                    best = (key, j, m, q)
            if best is None:
                continue
            _, j, m, q = best
            end_end = min(m, len(traces[j]["xy"]) - 1 - m) < 3
            pos = 0.5 * (p + pts[q]) if end_end else pts[q].copy()
            ev.append(dict(p=pos, tr=(i, j), wt={i: float(t["w"][ie]), j: float(wall[q])},
                           wide=max(t["w"][ie], wall[q]), thin=min(t["w"][ie], wall[q]), kind="T",
                           att=(i, e, pts[q].copy()) if float(np.dot(pts[q] - p, d)) > 0 else None))
    lines = [LineString(t["xy"]) for t in traces]
    st = STRtree(lines)
    a_idx, b_idx = st.query(lines, predicate="intersects")
    for a, b in sorted(zip(a_idx.tolist(), b_idx.tolist())):
        if a >= b:
            continue
        inter = lines[a].intersection(lines[b])
        for g in getattr(inter, "geoms", [inter]):
            if g.is_empty:
                continue
            c = np.asarray(g.centroid.coords[0])
            wa = traces[a]["w"][int(np.argmin(np.hypot(*(traces[a]["xy"] - c).T)))]
            wb = traces[b]["w"][int(np.argmin(np.hypot(*(traces[b]["xy"] - c).T)))]
            ev.append(dict(p=c, tr=(a, b), wt={a: float(wa), b: float(wb)}, wide=max(wa, wb), thin=min(wa, wb),
                           kind="X", att=None))
    return ev


def _cluster_events(ev: list, cfg: Config) -> list:
    """Complete-linkage clusters of events, approximating where the lumen-overlap regions of the events
    meet. Two events on a common trace are one junction when closer than share_k x the mean width of the
    vessels crossing that trace + r0 (their stretches of the shared vessel overlap: a thin vessel crossing
    two adjacent veins is one compound); otherwise when closer than r0 + link_k x the mean of their thinner
    widths. Complete linkage (every pair in a cluster must qualify) and a cap of 2 lambda keep a cluster
    from chaining along a vessel."""
    if not ev:
        return []
    if len(ev) == 1:
        return [[0]]
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n = len(ev)
    P = np.array([e["p"] for e in ev])
    th = np.array([e["thin"] for e in ev])
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    lim = cfg.r0 + cfg.link_k * 0.5 * (th[:, None] + th[None, :])
    for a in range(n):
        for b in range(a + 1, n):
            sh = set(ev[a]["tr"]) & set(ev[b]["tr"])
            if not sh or D[a, b] > 2 * LAMBDA_PX:
                continue
            i = min(sh)
            oa = max([w for k, w in ev[a]["wt"].items() if k != i] or [ev[a]["thin"]])
            ob = max([w for k, w in ev[b]["wt"].items() if k != i] or [ev[b]["thin"]])
            lim[a, b] = lim[b, a] = max(lim[a, b], cfg.r0 + cfg.share_k * 0.5 * (oa + ob))
    Dn = D / lim
    Dn[D > 2 * LAMBDA_PX] = np.maximum(Dn[D > 2 * LAMBDA_PX], 1.5)
    np.fill_diagonal(Dn, 0.0)
    lab = fcluster(linkage(squareform(Dn, checks=False), method="complete"), t=1.0, criterion="distance")
    groups = {}
    for k, l_ in enumerate(lab.tolist()):
        groups.setdefault(l_, []).append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def _arms(traces, tids, c, R, cfg):
    """The arms leaving the disc (c, R): per participating trace, the stretches before and after its run
    through the disc that are at least arm_min long. Returns [(trace id, side, unit direction, length)]."""
    arms = []
    for i in sorted(tids):
        P = traces[i]["xy"]
        d = np.hypot(*(P - c).T)
        m = int(np.argmin(d))
        inside = d <= R
        a = m
        while a > 0 and inside[a - 1]:
            a -= 1
        b = m
        while b < len(P) - 1 and inside[b + 1]:
            b += 1
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        if a > 0 and seg[a] >= cfg.arm_min:
            q = P[max(a - 4, 0)]
            arms.append((i, 0, (q - c) / max(math.hypot(*(q - c)), 1e-9), seg[a], a))
        if b < len(P) - 1 and seg[-1] - seg[b] >= cfg.arm_min:
            q = P[min(b + 4, len(P) - 1)]
            arms.append((i, 1, (q - c) / max(math.hypot(*(q - c)), 1e-9), seg[-1] - seg[b], b))
    return arms


def _type_of(arms, cfg) -> str:
    n = len(arms)
    if n == 3:
        return "pseudo-T"
    if n == 4:
        best = None
        for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            ua, ub, uc, ud = (arms[k][2] for k in (a, b, c, d))
            t1, t2 = _angle(ua, -ub), _angle(uc, -ud)
            same = (arms[a][0] == arms[b][0]) + (arms[c][0] == arms[d][0])
            key = (-same, max(t1, t2))
            if best is None or key < best[0]:
                best = (key, ua - ub, uc - ud, max(t1, t2))
        _, ax1, ax2, turn = best
        ax1 = ax1 / max(np.linalg.norm(ax1), 1e-9)
        ax2 = ax2 / max(np.linalg.norm(ax2), 1e-9)
        sep = min(_angle(ax1, ax2), _angle(ax1, -ax2))
        if turn <= cfg.cross_turn_deg and sep >= 15.0:
            return "crossing"
    return "compound"


def end_stopping(traces: list, cfg: Config = DEFAULT, OD: np.ndarray | None = None):
    """Stage 7, end-stopped (hypercomplex) cells: line terminations make the graph.

    Biology: end-stopped cells in V1/V2 respond to the end of a line within their receptive field; a line
    that ends on another one signals a junction (T or Y), while two lines that both continue through a point
    are seen as crossing (X).
    Maths: collinear end-to-end gaps are first bridged (mutual best pairs, turn < join_turn_deg). Events: an
    end lying within gap_att + w/2 of another trace, ahead of the end (T), and the intersections of two
    traces (X). Events form one junction by complete linkage when closer than r0 + link_k (w_thin,a +
    w_thin,b) / 2 (the thinner vessels' widths: the size of their lumen overlap); its centre is their mean,
    its radius the farthest reach (r0 + half the wider vessel). Its arms are the participating traces'
    stretches leaving the disc, at least arm_min long: 3 arms is a 3-way junction ('pseudo-T' by default),
    4 arms that pair into two straight lines (turn < cross_turn_deg; a trace passing through pairs its own
    two arms) at clearly different orientations a 'crossing', otherwise (4 unpaired, 5 or more) a
    'compound'; with fewer than 3 arms there is no junction. A trace ending on another is extended to its
    attachment point, then every trace is cut at its point nearest each of its junctions (where the truth's
    edges end too), so each polyline runs junction to junction. Returns (edges: [dict(xy, c, w, j=(j0,
    j1))], junctions: [dict(x, y, r, type, arms, tids)], traces)."""
    traces = [t for t in traces if _plen(t["xy"]) >= cfg.min_len]
    traces = _join_gaps(traces, cfg)
    traces.sort(key=lambda t: (-len(t["xy"]), float(t["xy"][0, 0]), float(t["xy"][0, 1])))
    if OD is not None and cfg.od_width:
        for t in traces:
            if "w_od" not in t:
                t["w_od"], t["a_od"] = od_widths(OD, t["xy"], t["w"])
            t["w"] = np.clip(t["w_od"], 1.5, None)
    ev = _events(traces, cfg)
    groups = _cluster_events(ev, cfg)
    cand = []
    for g in groups:
        E = [ev[k] for k in g]
        c = np.mean([e["p"] for e in E], 0)
        R = max(math.hypot(*(e["p"] - c)) + cfg.r0 + 0.5 * e["wide"] for e in E)
        cand.append([c, R, set(g)])
    if cfg.merge_k > 0:                               # junctions whose discs overlap are one region
        changed = True
        while changed:
            changed = False
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    ca, Ra, ga = cand[a]
                    cb, Rb, gb = cand[b]
                    if math.hypot(*(ca - cb)) <= cfg.merge_k * (Ra + Rb):
                        g = ga | gb
                        E = [ev[k] for k in sorted(g)]
                        c = np.mean([e["p"] for e in E], 0)
                        R = max(math.hypot(*(e["p"] - c)) + cfg.r0 + 0.5 * e["wide"] for e in E)
                        cand[a] = [c, R, g]
                        del cand[b]
                        changed = True
                        break
                if changed:
                    break
    if cfg.pass_k > 0:
        from scipy.spatial import cKDTree
        allp = np.concatenate([t["xy"] for t in traces])
        own = np.concatenate([np.full(len(t["xy"]), i) for i, t in enumerate(traces)])
        tree = cKDTree(allp)
    junctions = []
    for c, R, g in cand:
        E = [ev[k] for k in sorted(g)]
        tids = {i for e in E for i in e["tr"]}
        if cfg.pass_k > 0:                            # every trace through the region adds its arms
            tids |= {int(own[q]) for q in tree.query_ball_point(c, cfg.pass_k * R)}
        tids = sorted(tids)
        arms = _arms(traces, tids, c, R, cfg)
        if len(arms) < 3:
            continue
        junctions.append(dict(x=float(c[0]), y=float(c[1]), r=float(R), type=_type_of(arms, cfg), arms=arms,
                              tids=tids, strength=float(np.mean([traces[i]["c"].mean() for i in tids]))))
    ext = {}
    for e in ev:
        if e["att"] is not None:
            i, end, q = e["att"]
            ext[(i, end)] = q
    for (i, end), q in sorted(ext.items(), key=lambda kv: kv[0]):
        t = traces[i]
        p = t["xy"][0 if end == 0 else -1]
        if math.hypot(*(q - p)) <= 1.0:
            continue
        seg = _resample(np.stack([p, q]), 1.0)[1:]
        k = 0 if end == 0 else -1
        if end == 0:
            t["xy"] = np.concatenate([seg[::-1], t["xy"]])
            t["c"], t["w"] = np.r_[np.full(len(seg), t["c"][0]), t["c"]], np.r_[np.full(len(seg), t["w"][0]), t["w"]]
        else:
            t["xy"] = np.concatenate([t["xy"], seg])
            t["c"], t["w"] = np.r_[t["c"], np.full(len(seg), t["c"][k])], np.r_[t["w"], np.full(len(seg), t["w"][k])]
    edges = _split(traces, junctions, cfg)
    return edges, junctions, traces


def _split(traces, junctions, cfg):
    """Cut every trace at its point nearest each of its junctions; drop free stubs shorter than arm_min and
    pieces inside one junction."""
    per = {}
    for ji, J in enumerate(junctions):
        c = np.array([J["x"], J["y"]])
        for i in J["tids"]:
            d = np.hypot(*(traces[i]["xy"] - c).T)
            per.setdefault(i, []).append((int(np.argmin(d)), ji))
    edges = []
    for i, t in enumerate(traces):
        cuts = sorted(per.get(i, []))
        n = len(t["xy"])
        bounds = [(0, None)] + cuts + [(n - 1, None)]
        for (a, ja), (b, jb) in zip(bounds[:-1], bounds[1:]):
            if b - a < 1:
                continue
            xy = t["xy"][a:b + 1]
            L = _plen(xy)
            if ((ja is None) or (jb is None)) and L < cfg.arm_min:
                continue
            if ja is not None and ja == jb and L < 2 * junctions[ja]["r"]:
                continue
            edges.append(dict(xy=xy.copy(), c=t["c"][a:b + 1].copy(), w=t["w"][a:b + 1].copy(), j=(ja, jb),
                              trace=i, chan=t.get("chan", 0)))
    return edges


# ======================================================================== the annotator
def _output(edges, junctions):
    js = sorted(junctions, key=lambda J: (-J["strength"], J["y"], J["x"]))
    return dict(polylines=[np.asarray(e["xy"], float) for e in edges],
                junctions=[(float(J["x"]), float(J["y"]), J["type"]) for J in js])


def propose(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT) -> dict:
    """Stages 1-6: the cleaned OD target and the traces of every channel, merged."""
    s1 = photoreceptors(image, valid, cfg)
    chans = simple_cells(s1["OD"], s1["bg"], cfg)
    traces = []
    for ci, ch in enumerate(chans):
        U = ch["U"]
        if cfg.surround:
            U = surround(U, ch["sigma"], cfg)
        C = association_field(U, ch["sigma"], cfg) if cfg.association else np.maximum(U, 0)
        ch["C"] = C
        traces += readout(C, ch["S"], cfg, chan=ci)
    return dict(s1=s1, chans=chans, traces=merge_channels(traces, cfg))


def run(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT, debug: dict | None = None,
        proposal: dict | None = None) -> dict:
    pr = proposal if proposal is not None else propose(image, valid, cfg)
    traces = [dict(t) for t in pr["traces"]]
    edges, junctions, traces = end_stopping(traces, cfg, pr["s1"]["OD"])
    if debug is not None:
        debug.update(pr, traces=traces, edges=edges, junctions=junctions)
    return _output(edges, junctions)


def annotate(image, valid):
    return run(image, valid, DEFAULT)


ABLATIONS = dict(full={}, no_verify=dict(verify=False), no_association=dict(association=False),
                 no_surround=dict(surround=False), v1_only=dict(surround=False, association=False, verify=False))
