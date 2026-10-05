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
    env_radius: float = 60.0            # px, grey closing radius of the provisional upper envelope (> widest half lumen)
    bg_sigma: float = 8.0               # px, masked normalised convolution of the background
    bg_iters: int = 2
    mask_z: float = 2.0                 # band CNR above which a pixel is vessel (mask for the background)
    mask_dilate: int = 2                # px
    # 2. ganglion cells
    bands: tuple = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    rms_sigma: float = 24.0             # px, window of the local contrast (RMS) estimate
    # 3. simple cells
    n_orient: int = 16
    scales: tuple = (1.5, 2.1, 3.0, 4.2, 6.0, 8.5, 12.0, 17.0, 24.0)
    channels: tuple = ((0, 1, 2), (3, 4), (5, 6, 7, 8))   # scale indices of the fine, medium, coarse channels
    chan_factor: tuple = (1, 2, 4)      # each channel is computed on a grid downsampled this much (stages 3-5)
    elong: tuple = (3.0, 2.5, 2.0)      # along / across std ratio per channel (coarse: shorter, fewer star rays)
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
    border_px: int = 3                  # no line candidates this close to the frame or to invalid pixels
    look_px: int = 3                    # the tracker looks this far ahead (bridging 1-2 px gaps)
    pass_px: int = 4                    # ... and passes through another trace's claim for at most this long
    claim_k: float = 0.5                # a sample claims +-max(1, claim_k sigma) px across (its lumen)
    dup_bins: int = 1                   # channel merge: parallel = within this many 11.25 deg bins (duplicates)
    wall_bins: int = 0                  # ... and for the wall echoes inside a coarser vessel
    # 7. graph assembly
    gap_att: float = 6.0                # px beyond the other vessel's half width an end may attach
    att_cone_deg: float = 45.0          # ... within this cone ahead of the end
    ext_max: float = 12.0               # px, longest extension of an end to its attachment
    gap_join: float = 14.0              # px, collinear end-to-end gaps bridged
    join_turn_deg: float = 40.0
    r0: float = 4.0                     # px added to the half widths for an event's reach
    link_k: float = 1.0                 # events link within r0 + link_k x the mean of their thinner widths
    share_k: float = 1.0                # events on a common trace link within r0 + share_k x the crossing widths
    merge_k: float = 0.0                # junction discs closer than merge_k (R_a + R_b) are one region
    pass_k: float = 0.0                 # traces passing within pass_k R of a junction add their arms
    arm_min: float = 6.0                # px, shortest arm counted
    cross_turn_deg: float = 40.0        # a crossing's through pairs turn at most this much
    compound_r: float = 20.0            # px: a junction region at least this large is typed compound (dev prior)
    od_width: bool = True               # vessel widths from OD cross-sections (else 2.5 x the filter scale)
    # 8. iterative refinement
    verify: bool = True
    rounds: int = 2
    repropose: bool = True
    re_k: float = 1.0                   # thresholds of the re-proposal readout, x t_high / t_low
    corr_px: float = 6.0                # px^2, correlation area of the OD noise (texture) for the gains
    mdl_k: float = 2.0                  # description cost of an edge: mdl_k (1 + length / mdl_len)
    cnr_min: float = 0.5                # an edge's OD profile must reach this band CNR (~0.23 x the truth's scale)
    mdl_len: float = 20.0


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
    if not ok.all():                                  # filters see the nearest valid OD, not a step to 0
        iy, ix = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        OD = OD[iy, ix]
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
    H0, W0 = OD.shape
    th = orientations(cfg)
    out = []
    elongs = cfg.elong if isinstance(cfg.elong, tuple) else (cfg.elong,) * len(cfg.channels)
    for ci, (chan, el) in enumerate(zip(cfg.channels, elongs)):
        f = int(cfg.chan_factor[ci]) if ci < len(cfg.chan_factor) else 1
        Xc, bgc = _down(OD, f), (_down(bg.astype(np.float32), f) > 0.5) if f > 1 else bg
        H, W = Xc.shape
        smax = max(cfg.scales[i] for i in chan) / f
        P = int(min(math.ceil(3 * el * smax), H - 1, W - 1))
        X = np.pad(Xc.astype(np.float32), P, mode="reflect")
        Hp, Wp = sfft.next_fast_len(X.shape[0], real=True), sfft.next_fast_len(X.shape[1], real=True)
        F = sfft.rfft2(X, s=(Hp, Wp), workers=FFT_WORKERS).astype(np.complex64)
        wy = (2 * np.pi * sfft.fftfreq(Hp)).astype(np.float32)[:, None]
        wx = (2 * np.pi * sfft.rfftfreq(Wp)).astype(np.float32)[None, :]
        U = np.full((len(th), H, W), -np.inf, np.float32)
        S = np.zeros((len(th), H, W), np.uint8)
        for si in chan:
            s = cfg.scales[si] / f
            E = np.empty((len(th), H, W), np.float32)
            O = np.empty((len(th), H, W), np.float32)
            for k, t in enumerate(th):
                c, sn = np.float32(math.cos(t)), np.float32(math.sin(t))
                wn = -wx * sn + wy * c
                wt = wx * c + wy * sn
                G = np.exp(-0.5 * (s * s) * (wn * wn + (el ** 2) * wt * wt)).astype(np.float32)
                e = sfft.irfft2(F * ((s * s) * wn * wn * G), s=(Hp, Wp), workers=FFT_WORKERS)
                o = sfft.irfft2(F * (1j * s * wn * G).astype(np.complex64), s=(Hp, Wp), workers=FFT_WORKERS)
                E[k] = e[P:P + H, P:P + W]
                O[k] = o[P:P + H, P:P + W]
            ws = max(cfg.rms_sigma, 4 * cfg.scales[si]) / f
            re = _local_rms_stack(E, bgc, ws)
            ro = _local_rms_stack(O, bgc, ws)
            Z = E / re[None] - cfg.odd_alpha * np.abs(O) / ro[None]
            better = Z > U
            U = np.where(better, Z, U)
            S[better] = si
        out.append(dict(U=U, S=S, f=f, shape=(H0, W0),
                        sigma=float(np.exp(np.mean(np.log([cfg.scales[i] for i in chan]))))))
    return out


def _down(X: np.ndarray, f: int) -> np.ndarray:
    """Area average over f x f blocks (borders reflected up to a multiple of f)."""
    if f == 1:
        return X
    H, W = X.shape
    h, w = -(-H // f), -(-W // f)
    Xp = cv2.copyMakeBorder(np.asarray(X, np.float32), 0, h * f - H, 0, w * f - W, cv2.BORDER_REFLECT)
    return cv2.resize(Xp, (w, h), interpolation=cv2.INTER_AREA)


def _up(X: np.ndarray, f: int, shape, nearest: bool = False) -> np.ndarray:
    """Back to the full grid (the pixel centres of _down: bilinear, or nearest for labels), per layer."""
    if f == 1:
        return X
    H, W = shape
    h, w = X.shape[-2:]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return np.stack([cv2.resize(x, (w * f, h * f), interpolation=interp)[:H, :W] for x in X])


def _channel_traces(OD: np.ndarray, bg: np.ndarray, ok: np.ndarray, cfg: Config, t_high=None, t_low=None,
                    keep: list | None = None) -> list:
    """Stages 3-6 for every channel: simple cells, surround, association field (at the channel's grid), and
    the readout on the full grid."""
    traces = []
    for ci, ch in enumerate(simple_cells(OD, bg, cfg)):
        f = ch["f"]
        U = surround(ch["U"], ch["sigma"], cfg, f) if cfg.surround else ch["U"]
        C = association_field(U, ch["sigma"], cfg, f) if cfg.association else np.maximum(U, 0)
        C, S = _up(C, f, ch["shape"]), _up(ch["S"], f, ch["shape"], nearest=True)
        if keep is not None:
            keep.append(dict(C=C, S=S, sigma=ch["sigma"]))
        traces += readout(C, S, cfg, t_high=t_high, t_low=t_low, chan=ci, valid=ok)
    return traces


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


def surround(U: np.ndarray, sigma: float, cfg: Config = DEFAULT, f: int = 1) -> np.ndarray:
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
    d1 = (2.5 * sigma + 1.0) / f                      # px of the channel's grid (downsampled f x)
    d2 = d1 + max(6.0, 1.5 * sigma) / f
    step = max(1.0, sigma / 2 / f)
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
def association_field(U: np.ndarray, sigma: float, cfg: Config = DEFAULT, f: int = 1) -> np.ndarray:
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
    ell = cfg.assoc_len * math.sqrt(max(1.0, sigma / 2.0)) / f      # on the channel's grid
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


def readout(C: np.ndarray, S: np.ndarray, cfg: Config = DEFAULT, t_high=None, t_low=None, chan: int = 0,
            valid: np.ndarray | None = None) -> list:
    """Stage 6, readout: non-maximum suppression across space and orientation, hysteresis in SE(2) by
    tracing along the tangent.

    Maths: a sample (theta_k, y, x) is a candidate where C_k is a maximum along the normal n_k (bilinear
    neighbours at +-1 px), a maximum over orientation (C_k >= C_k-1, C_k > C_k+1: several orientations may
    peak at one pixel, two crossing lines live in different layers) and C_k > t_low, at least border_px from
    the frame and from invalid pixels. Seeds are the candidates above t_high, strongest first (ties by
    position). From a seed a trace steps 1 px along its tangent in both directions; the next sample is the
    best candidate (C, less 15 % per px of lateral offset and 10 % per layer of turn) among the positions
    0, +-1 px across the predicted one in layers k and k+-1 (SE(2) connectivity, theta circular). So a trace
    follows its own orientation layer straight through a crossing (the other line is >= 2 layers away) and
    takes the straighter branch at a fork; it looks up to look_px ahead (1-2 px gaps), passes through another
    trace's claim for at most pass_px samples (a shallow crossing) and stops where no candidate continues it
    or where it keeps running along another trace (a T or a merge: the end is put at the first contact).
    Each accepted sample claims its position and +-max(1, claim_k sigma) px across it in layers k, k+-1
    (its lumen: no second trace runs inside a wide vessel). Traces shorter than min_len are dropped; the rest are smoothed and
    resampled at 1 px. Returns traces: dicts with xy (n, 2), c (strength), w (2.5 sigma of the selected
    scale), chan."""
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
    inner = np.ones((H, W), bool) if valid is None else valid.copy()
    b = cfg.border_px
    inner[:b], inner[-b:], inner[:, :b], inner[:, -b:] = False, False, False, False
    if valid is not None and not valid.all():
        inner &= ndi.binary_erosion(valid, iterations=b)
    cand &= inner[None]
    TX, TY = np.cos(th), np.sin(th)
    owner = np.full(C.shape, -1, np.int32)
    flat = np.flatnonzero(cand & (C > t_high))
    seeds = flat[np.lexsort((flat, -C.ravel()[flat]))]
    sig = np.asarray(cfg.scales, np.float32)
    paths = []

    def claim(k, y, x, tid):
        nx, ny = -TY[k], TX[k]
        lat = max(1, int(round(cfg.claim_k * sig[S[k, y, x]])))
        for dk in (-1, 0, 1):
            kk = (k + dk) % n
            for o in range(-lat, lat + 1):
                yy, xx = int(round(y + o * ny)), int(round(x + o * nx))
                if 0 <= yy < H and 0 <= xx < W and owner[kk, yy, xx] < 0:
                    owner[kk, yy, xx] = tid

    def walk(k, y, x, sgn, tid):
        out, inside = [], 0
        dx, dy = sgn * TX[k], sgn * TY[k]
        for _ in range(4 * (H + W)):
            best = None
            for step in range(1, cfg.look_px + 1):        # look ahead over gaps of up to look_px - 1 px
                for dk in (0, -1, 1):
                    kk = (k + dk) % n
                    tx, ty = TX[kk], TY[kk]
                    if tx * dx + ty * dy < 0:
                        tx, ty = -tx, -ty
                    nx, ny = -ty, tx
                    for o in (0, -1, 1):
                        qy, qx = int(round(y + step * ty + o * ny)), int(round(x + step * tx + o * nx))
                        if (qy == y and qx == x) or not (0 <= qy < H and 0 <= qx < W) or not cand[kk, qy, qx]:
                            continue
                        if (qx - x) * dx + (qy - y) * dy <= 0 or owner[kk, qy, qx] == tid:
                            continue
                        sc = C[kk, qy, qx] * (1 - 0.15 * abs(o)) * (1 - 0.1 * abs(dk))
                        if best is None or sc > best[0]:
                            best = (sc, kk, qy, qx, tx, ty)
                if best is not None:
                    break
            if best is None:
                break
            _, kk, qy, qx, tx, ty = best
            out.append((kk, qy, qx))
            if owner[kk, qy, qx] >= 0:                # in another trace's claim: a shallow crossing is passed,
                inside += 1                           # running along it for pass_px is a merge (T): stop
                if inside > cfg.pass_px:
                    del out[len(out) - inside + 1:]
                    break
            else:
                inside = 0
                claim(kk, qy, qx, tid)
            k, y, x, dx, dy = kk, qy, qx, tx, ty
        return out

    for sd in seeds.tolist():
        k, rem = divmod(sd, H * W)
        y, x = divmod(rem, W)
        if owner[k, y, x] >= 0:
            continue
        tid = len(paths)
        claim(k, y, x, tid)
        f = walk(k, y, x, 1, tid)
        bk = walk(k, y, x, -1, tid)
        paths.append(bk[::-1] + [(k, y, x)] + f)
    traces = []
    for pth in paths:
        if len(pth) < 3:
            continue
        P = np.array(pth)
        xy = P[:, [2, 1]].astype(float)
        if _plen(xy) < cfg.min_len:
            continue
        xy = _resample(_smooth(xy, 5), 1.0)
        kk = np.interp(np.linspace(0, len(P) - 1, len(xy)), np.arange(len(P)), np.arange(len(P))).round().astype(int)
        ks, ys, xs = P[kk, 0], P[kk, 1], P[kk, 2]
        traces.append(dict(xy=xy, c=C[ks, ys, xs].astype(float), w=2.5 * sig[S[ks, ys, xs]].astype(float),
                           chan=chan))
    return traces


def _tangents(P: np.ndarray, k: int = 3) -> np.ndarray:
    n = len(P)
    i0, i1 = np.clip(np.arange(n) - k, 0, n - 1), np.clip(np.arange(n) + k, 0, n - 1)
    v = P[i1] - P[i0]
    return v / np.maximum(np.hypot(*v.T), 1e-9)[:, None]


def merge_channels(traces: list, cfg: Config = DEFAULT, accepted: list | None = None, shape=None) -> list:
    """Merge the traces of the spatial-frequency channels (and new proposals into accepted traces):
    strongest first, a point is dropped where an accepted trace runs parallel (orientation within dup_bins
    bins of 11.25 deg) within 3 px, or parallel (within wall_bins) within 0.6 of its width when that trace is
    from a coarser channel (the wall echoes of a wide vein; a thin vessel crossing the vein at a shallow angle
    is kept); the uncovered runs of at least min_len are kept. Coverage is drawn into rasters per orientation
    bin."""
    acc = list(accepted or [])
    allxy = [t["xy"] for t in acc + list(traces)]
    if not allxy:
        return acc
    if shape is None:
        mx = np.concatenate(allxy).max(0)
        shape = (int(mx[1]) + 2, int(mx[0]) + 2)
    H, W = shape
    nb = 16
    dup = np.zeros((nb, H, W), np.uint8)              # any accepted trace within 3 px
    wall = np.zeros((nb, H, W), np.uint8)             # 1 + the coarsest channel covering within 0.6 w

    def draw(t):
        P, Tg = t["xy"], _tangents(t["xy"])
        ch = int(t.get("chan", 0)) + 1
        hw = int(round(max(3.0, 0.6 * float(np.median(t["w"])))))
        bins = np.round(np.mod(np.arctan2(Tg[:, 1], Tg[:, 0]), np.pi) / (np.pi / nb)).astype(int) % nb
        for i in range(0, len(P) - 1, 3):
            j = min(i + 3, len(P) - 1)
            a_, b_ = tuple(np.round(P[i]).astype(int)), tuple(np.round(P[j]).astype(int))
            k = int(bins[i])
            cv2.line(dup[k], a_, b_, 1, 7)
            if hw > 3:
                sub = wall[k]
                m = np.zeros_like(sub)
                cv2.line(m, a_, b_, 1, 2 * hw + 1)
                np.maximum(sub, m * ch, out=sub)

    for t in acc:
        draw(t)
    order = sorted(range(len(traces)), key=lambda i: (-float(np.mean(traces[i]["c"])),
                                                      float(traces[i]["xy"][0, 0]), float(traces[i]["xy"][0, 1])))
    for i in order:
        t = traces[i]
        P = t["xy"]
        Tg = _tangents(P)
        ix = np.clip(np.round(P).astype(int), 0, [W - 1, H - 1])
        bins = np.round(np.mod(np.arctan2(Tg[:, 1], Tg[:, 0]), np.pi) / (np.pi / nb)).astype(int) % nb
        ch = int(t.get("chan", 0)) + 1
        cov = np.zeros(len(P), bool)
        for d in range(-cfg.dup_bins, cfg.dup_bins + 1):
            kb = (bins + d) % nb
            cov |= dup[kb, ix[:, 1], ix[:, 0]] > 0
            if abs(d) <= cfg.wall_bins:
                cov |= wall[kb, ix[:, 1], ix[:, 0]] > ch
        lab, nl = ndi.label(~cov)
        for r in range(1, nl + 1):
            ii = np.flatnonzero(lab == r)
            if len(ii) < cfg.min_len:
                continue
            piece = {k_: (v[ii[0]:ii[-1] + 1].copy() if isinstance(v, np.ndarray) and len(v) == len(P) else v)
                     for k_, v in t.items()}
            acc.append(piece)
            draw(piece)
    return acc


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
            lat = abs(float(D[a][0] * g[1] - D[a][1] * g[0]))
            wa = float(traces[ta]["w"][-1 if a % 2 else 0])
            if gl > 2 and not (gl <= 6 and lat <= max(3.0, 0.3 * wa)):   # a short offset gap is still a gap
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
    out = dict(xy=np.concatenate([A["xy"], gap, B["xy"]], 0), chan=min(A.get("chan", 0), B.get("chan", 0)))
    na, nb = len(A["xy"]), len(B["xy"])
    for k, v in A.items():
        if k != "xy" and isinstance(v, np.ndarray) and len(v) == na and k in B and len(B[k]) == nb:
            fill = np.full(m, 0.5 * (v[-1] + B[k][0]), float)
            out[k] = np.concatenate([v, fill, B[k]])
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
                if dist > cfg.gap_att + wall[q] / 2:
                    continue
                inside = dist <= max(2.0, 0.5 * wall[q])          # the end already lies in its lumen
                if not inside and float(np.dot(v, d)) < dist * math.cos(math.radians(cfg.att_cone_deg)):
                    continue
                key = (dist - wall[q] / 2, j, m)
                if best is None or key < best[0]:
                    best = (key, j, m, q)
            if best is None:
                continue
            _, j, m, q = best
            end_end = min(m, len(traces[j]["xy"]) - 1 - m) < 3
            pos = 0.5 * (p + pts[q]) if end_end else pts[q].copy()
            # a short gap outside the other lumen is closed by extending the end to the other centreline;
            # an end already inside a (wide) lumen stays where it is
            att = (i, e, pts[q].copy()) if (best[0][0] > 0 and math.hypot(*(pts[q] - p)) <= cfg.ext_max) else None
            ev.append(dict(p=pos, tr=(i, j), wt={i: float(t["w"][ie]), j: float(wall[q])},
                           wide=max(t["w"][ie], wall[q]), thin=min(t["w"][ie], wall[q]), kind="T", att=att))
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
    'compound'; with fewer than 3 arms there is no junction. A junction whose region is at least compound_r
    in radius is typed 'compound' whatever its traced arms (on dev, 81 % of such regions are compounds in the
    truth: wide vessels gather lines that are too faint to trace). A trace ending on another is extended to its
    attachment point, then every trace is cut at its point nearest each of its junctions (where the truth's
    edges end too), so each polyline runs junction to junction. Returns (edges: [dict(xy, c, w, j=(j0,
    j1))], junctions: [dict(x, y, r, type, arms, tids)], traces)."""
    traces = [t for t in traces if _plen(t["xy"]) >= cfg.min_len]
    traces = _join_gaps(traces, cfg)
    traces.sort(key=lambda t: (-len(t["xy"]), float(t["xy"][0, 0]), float(t["xy"][0, 1])))
    for t in traces:
        if "w_ro" not in t:
            t["w_ro"] = np.asarray(t["w"], float).copy()   # the readout's width (2.5 sigma of its scale)
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
        typ = _type_of(arms, cfg)
        if R >= cfg.compound_r:                       # a region this large gathers more lines than are traced
            typ = "compound"
        junctions.append(dict(x=float(c[0]), y=float(c[1]), r=float(R), type=typ, arms=arms, geom=_type_of(arms, cfg),
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
        n = len(t["xy"])
        for k_, v in list(t.items()):
            if k_ != "xy" and isinstance(v, np.ndarray) and len(v) == n:
                pad = np.full(len(seg), v[0] if end == 0 else v[-1])
                t[k_] = np.r_[pad, v] if end == 0 else np.r_[v, pad]
        t["xy"] = np.concatenate([seg[::-1], t["xy"]]) if end == 0 else np.concatenate([t["xy"], seg])
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
                              trace=i, rng=(a, b), chan=t.get("chan", 0)))
    return edges


# ======================================================================== 8. iterative refinement
_RG = np.geomspace(0.5, 30.0, 22)                 # half widths r (px) of the blurred-box profile grid
_SG = np.geomspace(0.7, 10.0, 10)                 # blurs s (px)


def _box(u: np.ndarray, r, s) -> np.ndarray:
    """A lumen of half width r blurred by a Gaussian of std s, peak-normalised: (Phi((r-u)/s) - Phi((-r-u)/s))
    / (Phi(r/s) - Phi(-r/s)) -- the closed-form cross-section of a tube in OD (chord law aside)."""
    from scipy.special import ndtr
    return (ndtr((r - u) / s) - ndtr((-r - u) / s)) / (ndtr(r / s) - ndtr(-r / s))


def _box_fit(prof: np.ndarray, us: np.ndarray, r_max: float = np.inf):
    """Least-squares fit of profiles (n, m) sampled at us by a box(u; r, s) + b over the (r, s) grid
    (r <= r_max), (a, b) linear. Returns (a, r, s) arrays (a = 0 where nothing positive fits)."""
    M = _box(us[None, None, :], _RG[:, None, None], _SG[None, :, None]).reshape(-1, len(us))
    M[np.repeat(_RG > max(r_max, _RG[0]), len(_SG))] = 0.0
    k = float(len(us))
    m1, m2 = M.sum(1), (M * M).sum(1)
    py, p2 = prof.sum(1), (prof * prof).sum(1)
    pp = prof @ M.T
    det = m2[None] * k - m1[None] ** 2
    A = (pp * k - m1[None] * py[:, None]) / np.maximum(det, 1e-9)
    Bc = (py[:, None] - A * m1[None]) / k
    ss = p2[:, None] - 2 * A * pp - 2 * Bc * py[:, None] + A * A * m2[None] + 2 * A * Bc * m1[None] + Bc * Bc * k
    ss = np.where((A > 0) & (m2[None] > 0), ss, np.inf)
    j = np.argmin(ss, 1)
    ok = np.isfinite(ss[np.arange(len(j)), j])
    a = np.where(ok, A[np.arange(len(j)), j], 0.0)
    return a, _RG[j // len(_SG)], _SG[j % len(_SG)]


def fit_profiles(OD: np.ndarray, xy: np.ndarray, w: np.ndarray, step: int = 6):
    """Contrast, half width and blur along a trace from cross-sections of the cleaned OD target (never a
    filter response): every `step` px, the median cross-section over +-max(6, w) px of the trace, along the
    normals over +-(w + 6) px, fitted by a box(u; r, s) + b with r <= 0.75 w + 2 (_box_fit; w is the scale
    the trace was read out at, so a neighbouring wide vessel cannot lend its profile to a thin one); linear
    interpolation between the fits and a running median of 3, so one vessel is rendered without seams.
    Returns (a, r, s) per point."""
    n = len(xy)
    wm = float(np.median(w))
    T = _tangents(xy, 2)
    U = float(np.clip(wm + 6.0, 8.0, 60.0))
    us = np.arange(-U, U + 0.01, 0.5)
    X = xy[:, None, 0] - us[None, :] * T[:, None, 1]
    Y = xy[:, None, 1] + us[None, :] * T[:, None, 0]
    Pf = _bilinear(OD, X.ravel(), Y.ravel()).reshape(X.shape)
    cs = np.unique(np.r_[np.arange(0, n, step), n - 1])
    h = int(max(6, round(wm)))
    prof = np.stack([np.median(Pf[max(0, c - h):c + h + 1], 0) for c in cs])
    a, r, s_ = _box_fit(prof, us, 0.75 * wm + 2.0)
    if len(cs) >= 3:
        a, r, s_ = (ndi.median_filter(v, 3, mode="nearest") for v in (a, r, s_))
    i = np.arange(n)
    return np.interp(i, cs, a), np.interp(i, cs, r), np.interp(i, cs, s_)


def _tube(shape, xy: np.ndarray, a: np.ndarray, r: np.ndarray, s: np.ndarray):
    """Flat pixel indices and OD values of a blurred-box tube along a polyline with per-point contrast a,
    half width r and blur s (each pixel takes the parameters of its nearest centreline point), with butt
    ends: the pieces of one vessel cut at a junction join without a seam."""
    from scipy.spatial import cKDTree
    H, W = shape
    none = (np.zeros(0, np.int64), np.zeros(0, np.float64))
    if len(xy) < 2:
        return none
    reach = float(np.max(r + 3.0 * s)) + 1.0
    x0, y0 = int(max(0, np.floor(xy[:, 0].min() - reach))), int(max(0, np.floor(xy[:, 1].min() - reach)))
    x1, y1 = int(min(W - 1, np.ceil(xy[:, 0].max() + reach))), int(min(H - 1, np.ceil(xy[:, 1].max() + reach)))
    if x1 < x0 or y1 < y0:
        return none
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    Q = np.round(xy - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [Q], False, 1, thickness=int(2 * np.ceil(reach)) + 1)
    yy, xx = np.nonzero(m)
    pix = np.stack([xx + x0, yy + y0], 1).astype(float)
    t_ = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    tt = np.linspace(0, t_[-1], max(2, int(round(t_[-1] / 0.5)) + 1))
    P = np.stack([np.interp(tt, t_, xy[:, 0]), np.interp(tt, t_, xy[:, 1])], 1)
    pa, pr, ps = (np.interp(tt, t_, v) for v in (a, r, s))
    d, k = cKDTree(P).query(pix, distance_upper_bound=reach)
    ok = np.isfinite(d)
    k = np.minimum(k, len(P) - 1)
    t0 = (P[1] - P[0]) / max(np.hypot(*(P[1] - P[0])), 1e-9)
    t1 = (P[-1] - P[-2]) / max(np.hypot(*(P[-1] - P[-2])), 1e-9)
    ok &= ~((k == 0) & (((pix - P[0]) @ t0) < -0.5))
    ok &= ~((k == len(P) - 1) & (((pix - P[-1]) @ t1) > 0.5))
    kk = k[ok]
    val = pa[kk] * _box(d[ok], pr[kk], ps[kk])
    keep = val > 1e-5
    idx = pix[ok, 1].astype(np.int64) * W + pix[ok, 0].astype(np.int64)
    return idx[keep], val[keep]


class _Render:
    """The candidate graph rendered additively in OD (Beer-Lambert: overlapping vessels add their densities)
    and its residual against the cleaned OD target; per-edge noise-weighted gains for the MDL test."""

    def __init__(self, OD, sigma2, edges, cfg):
        self.shape, self.cfg = OD.shape, cfg
        self.t = OD.ravel().astype(np.float64)
        self.iw = 1.0 / sigma2.ravel().astype(np.float64)
        self.R = np.zeros(self.t.size)
        self.tubes = [_tube(self.shape, e["xy"], e["pa"], e["pr"], e["ps"]) for e in edges]
        for idx, val in self.tubes:
            np.add.at(self.R, idx, val)
        self.alive = np.ones(len(edges), bool)

    def gain(self, k: int) -> float:
        """Noise-weighted fall of the squared residual that edge k brings, given the others:
        sum((r + p)^2 - r^2) / s^2 = sum(2 r p + p^2) / s^2 over its tube, r the residual with k rendered,
        divided by the correlation area corr_px. Negative where k duplicates what others already explain."""
        idx, p = self.tubes[k]
        if not len(idx):
            return 0.0
        r = self.t[idx] - self.R[idx]
        if not self.alive[k]:
            r = r - p
        return float(((2 * r * p + p * p) * self.iw[idx]).sum()) / self.cfg.corr_px

    def remove(self, k: int):
        idx, p = self.tubes[k]
        np.subtract.at(self.R, idx, p)
        self.alive[k] = False

    def residual(self) -> np.ndarray:
        return (self.t - self.R).reshape(self.shape).astype(np.float32)


def _band_resp(r: float, s: float, bands) -> np.ndarray:
    """Centre response of a unit-peak blurred box (half width r, blur s) in the DoG bands G_b - G_2b, across
    the vessel (stage 2's channels): h [(2 Phi(r / sqrt(s^2 + b^2)) - 1) - (2 Phi(r / sqrt(s^2 + 4 b^2)) - 1)]
    with h = 1 / (2 Phi(r / s) - 1)."""
    from scipy.special import ndtr
    b = np.asarray(bands, float)
    h = 1.0 / max(2 * ndtr(r / s) - 1, 1e-6)
    return h * (2 * ndtr(r / np.sqrt(s * s + b * b)) - 2 * ndtr(r / np.sqrt(s * s + 4 * b * b)))


def _mdl_prune(OD, sigma2, edges, cfg, rms=None):
    """Greedy MDL with a visibility floor: remove, one at a time, the edge with a free end (a whole trace, a
    spur, a false arm or extension; an edge between two junctions belongs to a vessel that continues) that
    falls furthest short of either test, updating the render and its overlapping neighbours, until every
    such edge passes both:
      gain >= mdl_k (1 + length / mdl_len)  (the squared residual it removes pays for its description; a
      duplicate of what other edges already render has a negative gain: explaining away), and
      CNR = a max_b resp_b(r, s) / RMS_b >= cnr_min  (its OD profile in stage 2's band b against the local
      background RMS of that band along it: the band-matched visibility the truth itself uses).
    Returns (keep flags, the render)."""
    rd = _Render(OD, sigma2, edges, cfg)
    n = len(edges)
    H, W = OD.shape
    cost = np.array([cfg.mdl_k * (1.0 + _plen(e["xy"]) / cfg.mdl_len) for e in edges])
    cnr = np.full(n, np.inf)
    if rms is not None and cfg.cnr_min > 0:
        bands = sorted(rms)
        for k, e in enumerate(edges):
            ix = np.clip(np.round(e["xy"]).astype(int), 0, [W - 1, H - 1])
            noise = np.array([float(np.median(rms[b][ix[:, 1], ix[:, 0]])) for b in bands])
            resp = _band_resp(float(np.median(e["pr"])), float(np.median(e["ps"])), bands)
            cnr[k] = float(np.median(e["pa"])) * float(np.max(resp / np.maximum(noise, 1e-6)))
            e["cnr"] = cnr[k]

    def value(k):
        g = rd.gain(k)
        vg = g / cost[k] if cost[k] > 0 else (np.inf if g >= 0 else -np.inf)
        return min(vg, cnr[k] / cfg.cnr_min if cfg.cnr_min > 0 else np.inf)

    v = np.array([value(k) for k in range(n)])
    bb = np.array([(idx.min() // W, idx.max() // W, (idx % W).min(), (idx % W).max()) if len(idx) else (0, -1, 0, -1)
                   for idx, _ in rd.tubes]).reshape(-1, 4)
    free = np.array([e["j"][0] is None or e["j"][1] is None for e in edges], bool)
    while True:                                       # only edges with a free end are candidates: an edge
        cand = np.flatnonzero(rd.alive & free & (v < 1.0))   # between two junctions is part of a vessel
        if not len(cand):
            break
        k = int(cand[np.lexsort((cand, v[cand]))[0]])
        rd.remove(k)
        v[k] = np.inf
        near = rd.alive & (bb[:, 0] <= bb[k, 1]) & (bb[:, 1] >= bb[k, 0]) & (bb[:, 2] <= bb[k, 3]) & (bb[:, 3] >= bb[k, 2])
        sk = set(rd.tubes[k][0].tolist())
        for j in np.flatnonzero(near):
            if not sk.isdisjoint(rd.tubes[j][0].tolist()):
                v[j] = value(j)
    return rd.alive.tolist(), rd


def _knots(junctions, edges, rd, cfg):
    """The OD at a junction's centre against the additive render: two vessels crossing at different depths
    add their densities (a dark knot; the additive render explains it, residual ~0), the lumens of a fork are
    a union (the additive render over-predicts by about the thinner vessel's density: residual ~ -a_min).
    Stores knot = the mean residual in a disc of radius max(1.5, r_min / 2) over a_min."""
    res = rd.residual()
    H, W = res.shape
    for ji, J in enumerate(junctions):
        near = [e for e in edges if ji in e["j"]]
        if len(near) < 2:
            J["knot"] = float("nan")
            continue
        amin = max(min(float(np.median(e["pa"])) for e in near), 1e-4)
        rad = max(1.5, 0.5 * min(float(np.median(e["pr"])) for e in near))
        x, y = J["x"], J["y"]
        y0, y1 = int(max(0, y - rad - 1)), int(min(H, y + rad + 2))
        x0, x1 = int(max(0, x - rad - 1)), int(min(W, x + rad + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= rad * rad
        J["knot"] = float(res[y0:y1, x0:x1][m].mean() / amin) if m.any() else float("nan")


def _remove_ranges(traces: list, edges: list, keep: list, cfg: Config) -> list:
    """Cut the pruned edges' index ranges out of their traces; the surviving runs become the traces."""
    gone = {}
    for e, k in zip(edges, keep):
        if not k:
            gone.setdefault(e["trace"], []).append(e["rng"])
    kept_pts = {}
    for e, k in zip(edges, keep):
        if k:
            kept_pts.setdefault(e["trace"], []).append(e["rng"])
    out = []
    for i, t in enumerate(traces):
        if i not in gone:
            out.append(t)
            continue
        n = len(t["xy"])
        m = np.zeros(n, bool)
        for a, b in kept_pts.get(i, []):
            m[a:b + 1] = True
        lab, nl = ndi.label(m)
        for r in range(1, nl + 1):
            ii = np.flatnonzero(lab == r)
            if _plen(t["xy"][ii[0]:ii[-1] + 1]) < cfg.min_len:
                continue
            out.append({k_: (v[ii[0]:ii[-1] + 1].copy() if isinstance(v, np.ndarray) and len(v) == n else v)
                        for k_, v in t.items()})
    return out


def _assemble_profiled(traces, cfg, OD):
    """Stage 7, then every trace's OD profile (fit_profiles) and its slice on each edge."""
    edges, junctions, traces = end_stopping(traces, cfg, OD)
    for t in traces:
        t["pa"], t["pr"], t["ps"] = fit_profiles(OD, t["xy"], t.get("w_ro", t["w"]))
    for e in edges:
        t = traces[e["trace"]]
        a, b = e["rng"]
        e["pa"], e["pr"], e["ps"] = t["pa"][a:b + 1], t["pr"][a:b + 1], t["ps"][a:b + 1]
    return edges, junctions, traces


def refine(pr: dict, traces: list, cfg: Config = DEFAULT, debug: dict | None = None):
    """Stage 8, iterative refinement (the lesson of diffusion models, deterministic like DDIM): a fixed
    number of rounds of render, compare, prune and re-propose.

    Each round: the graph (stage 7) is built from the traces; its edges get their width, blur and contrast
    from cross-sections of the cleaned OD target (fit_profile); the graph is rendered additively in OD
    (Beer-Lambert: crossing vessels add their densities); the residual is taken against OD = B - L of
    stage 1 -- the image cleaned by the background fitted on the vesselness mask's negative, never a
    vesselness map, whose response is distorted exactly at forks and crossings; edges whose removal barely
    changes the noise-weighted squared residual are cut out of their traces (MDL, _mdl_prune); the residual
    itself is passed again through stages 3-6 to propose what the graph does not yet explain (a vessel
    masked by a stronger neighbour, an arm lost at a junction), and the new traces join the old ones. After
    the last round the graph is assembled once more, pruned and re-assembled; the knot cue (OD at junction
    centres against the additive render) is stored on each junction."""
    s1 = pr["s1"]
    OD, bg = s1["OD"], s1["bg"]
    hp = OD - _gblur(OD, 8.0)
    sig2 = np.maximum(_local_rms(hp, bg, 32.0), 1e-4) ** 2
    sig2 = np.where(s1["ok"], sig2, 1e12)              # invalid pixels carry no weight
    for rnd in range(cfg.rounds + 1):
        edges, junctions, traces = _assemble_profiled(traces, cfg, OD)
        keep, rd = _mdl_prune(OD, sig2, edges, cfg, s1.get("rms"))
        traces = _remove_ranges(traces, edges, keep, cfg)
        if rnd < cfg.rounds and cfg.repropose:
            res = rd.residual()
            new = _channel_traces(res, bg, s1["ok"], cfg, t_high=cfg.t_high * cfg.re_k, t_low=cfg.t_low * cfg.re_k)
            # the graph's own vessels count as coarser than any proposal: misfit echoes along their walls
            # (parallel, inside 0.6 of their width) are not new vessels
            acc = [dict(t, chan=len(cfg.channels)) for t in traces]
            merged = merge_channels(new, cfg, accepted=acc, shape=OD.shape)
            traces = traces + merged[len(acc):]
    edges, junctions, traces = _assemble_profiled(traces, cfg, OD)
    rd = _Render(OD, sig2, edges, cfg)
    _knots(junctions, edges, rd, cfg)
    if debug is not None:
        debug.update(residual=rd.residual(), sig2=sig2)
    return edges, junctions, traces


# ======================================================================== the annotator
def _output(edges, junctions):
    js = sorted(junctions, key=lambda J: (-J["strength"], J["y"], J["x"]))
    return dict(polylines=[np.asarray(e["xy"], float) for e in edges],
                junctions=[(float(J["x"]), float(J["y"]), J["type"]) for J in js])


def propose(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT) -> dict:
    """Stages 1-6: the cleaned OD target and the traces of every channel, merged."""
    s1 = photoreceptors(image, valid, cfg)
    s1["rms"] = ganglion_cells(s1["OD"], s1["bg"], cfg)["rms"]
    chans = []
    traces = _channel_traces(s1["OD"], s1["bg"], s1["ok"], cfg, keep=chans)
    return dict(s1=s1, chans=chans, traces=merge_channels(traces, cfg, shape=s1["OD"].shape))


def run(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT, debug: dict | None = None,
        proposal: dict | None = None) -> dict:
    pr = proposal if proposal is not None else propose(image, valid, cfg)
    traces = [dict(t) for t in pr["traces"]]
    if cfg.verify:
        edges, junctions, traces = refine(pr, traces, cfg, debug)
    else:
        edges, junctions, traces = end_stopping(traces, cfg, pr["s1"]["OD"])
    if debug is not None:
        debug.update(pr, traces=traces, edges=edges, junctions=junctions)
    return _output(edges, junctions)


ABLATIONS = dict(full={}, no_verify=dict(verify=False), no_association=dict(association=False),
                 no_surround=dict(surround=False), v1_only=dict(surround=False, association=False, verify=False))


def annotate(image, valid):
    """The full annotator, stages 1-8."""
    return run(image, valid, DEFAULT)


def annotate_no_verify(image, valid):
    """Stages 1-7: no render / residual / prune / re-propose refinement."""
    return run(image, valid, replace(DEFAULT, **ABLATIONS["no_verify"]))


def annotate_no_association(image, valid):
    """All stages but the association field (5): no contour completion."""
    return run(image, valid, replace(DEFAULT, **ABLATIONS["no_association"]))


def annotate_no_surround(image, valid):
    """All stages but the non-classical surround (4)."""
    return run(image, valid, replace(DEFAULT, **ABLATIONS["no_surround"]))


def annotate_v1_only(image, valid):
    """Stages 1-3 + 6-7: the orientation score read out without any contextual stage."""
    return run(image, valid, replace(DEFAULT, **ABLATIONS["v1_only"]))
