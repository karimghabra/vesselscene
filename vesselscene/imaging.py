"""Image formation: from vessel optical density to a still in DN, like
mean_stabilized.tif, or to one raw frame; calibrated on the real stills.

The model (each term measured on the real data by measure_real, saved in
data/real_imaging.json, where ImagingParams takes its defaults):

    frame  = level * E(x) * exp(tau(x)) * exp(-OD(x))              the scene
             blurred by the eye's motion during the exposure,
             times the lamp's flicker and the camera's fixed gain pattern
             (PRNU-like, dust specks), plus its fixed row/column offsets,
             with shot noise (Poisson in electrons at the gain), read noise
             and row/column noise, through the camera gamma, quantised to
             12 bits and clipped (specular glare saturates).
    still  = the mean of n_frames such frames, each moved back onto the scene
             by the pipeline's bilinear shift, glare left out, NaN where
             fewer than half the frames saw the pixel.

    E    illumination: a broad bump (the round, non-diffuse green lamp) and a
         radial vignetting, the family fitted to the stills' vessel-free
         background (fit_illumination).
    tau  scleral texture (Np): a Gaussian random field whose power in the
         1-2 / 2-4 / ... / 64-128 px difference-of-Gaussian bands is that of
         the bright half of the stills' vessel-masked residuals (noise
         removed), with their orientation anisotropy, patchy (a log-normal
         envelope on the fine scales, calibrated to the residuals' tail
         weight), plus sparse lumps of both signs.
    OD   the renderer's optical density (Np) as seen in ln DN.  A camera gamma
         scales every log contrast by gamma, and the texture and contrast
         targets were measured in ln DN, so gamma is folded into tau and OD;
         it is used only where it acts on its own: the noise.

An average still differs from a frame by what is simulated, not assumed: an
eye trajectory (fixational drift and saccades, centred on its median as the
pipeline's is) gives the coverage of every pixel (the NaN corners and glare
holes; noise rising where fewer frames saw a pixel), the camera-fixed things
seen through the moving eye (glare bloom smeared into streaks, the fixed
pattern and dust averaged away, the flicker left where coverage changes),
and the blur of the average: the frames' motion blur, bilinear interpolation
and registration error (~0.85 px SD with the defaults).  A single frame gets
its own motion blur, full per-frame noise, unaveraged dust and saturated
glare, and no NaN.

The blood of a single frame (v2, 2026-10-01): one instant of the red cells,
which the average holds only as their mean.  filling() gives the cells along
each vessel: single cells and plasma gaps in the capillaries, clumps of
aggregated cells and plasma-rich stretches in the venules (the beads of a raw
frame), moving at the conjunctiva's measured red-cell speed (about 0.5 mm/s
in every calibre) and smeared over the exposure; rbc_mottle() the cells'
granularity across the lumens of the continuous vessels (the cell count of
each column fluctuates; Poisson-like, through the chord law, blurred with
each vessel).  The frame's noise is the camera's, white like the raw
frames' (their power spectrum is flat from 0.2 cycles/px to Nyquist).

What anchors the noise: the reference bursts rebuilt from their raw frames
with the pipeline's own shifts reproduce mean_stabilized.tif to 2e-8, so the
half difference of the even- and odd-frame means (rebuild_halves) is the
still's own noise.  Its white part (bilinear-shaped) is what the camera model
predicts, within 2-7 % on the three bursts with raw frames; the rest of it is
registration residue of the tissue and the lamp's flicker (noise_structure).

Coordinates: pixels, x to the right, y down, as graph.py.  Images are
float32 DN.  Log quantities are in nepers (Np, natural log).
"""
from __future__ import annotations

import csv
import dataclasses
import glob
import json
import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy import stats as sst

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_imaging.json")
BANDS = (1, 2, 4, 8, 16, 32, 64)        # DoG band s: G_s - G_2s, i.e. structure of ~s..2s px
FULL_SCALE = 4095                       # 12-bit camera
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)


# ======================================================================== helpers
def _gauss_fft(X: np.ndarray, s: float) -> np.ndarray:
    """Gaussian blur of a 2-D array with reflecting borders, by FFT (fast for
    the large sigmas of the texture bands)."""
    if s <= 0:
        return X.astype(np.float64)
    H, W = X.shape
    p = int(min(max(H, W), math.ceil(3 * s) + 2))
    Xp = np.pad(X.astype(np.float64), p, mode="reflect" if p < min(H, W) else "symmetric")
    return _filter_fft(Xp, lambda fy, fx: np.exp(-2 * math.pi ** 2 * s * s * (fx * fx + fy * fy)))[p:p + H, p:p + W]


def _filter_fft(X: np.ndarray, transfer) -> np.ndarray:
    """Multiply the spectrum of X by transfer(fy, fx) (cycles/px) and invert."""
    H, W = X.shape
    fy = sfft.fftfreq(H)[:, None]
    fx = sfft.rfftfreq(W)[None, :]
    return sfft.irfft2(sfft.rfft2(X, workers=-1) * transfer(fy, fx), s=(H, W), workers=-1)


def _filter_padded(X: np.ndarray, transfer, pad: int) -> np.ndarray:
    """_filter_fft with reflecting padding (no wrap-around at the borders)."""
    H, W = X.shape
    p = int(min(pad, H - 1, W - 1))
    Xp = np.pad(X.astype(np.float64), p, mode="reflect")
    return _filter_fft(Xp, transfer)[p:p + H, p:p + W]


def _dog(X: np.ndarray, s: float) -> np.ndarray:
    return _gauss_fft(X, s) - _gauss_fft(X, 2 * s)


def fill_invalid(L: np.ndarray, valid: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Replace invalid pixels by normalized convolution of the valid ones,
    widening the kernel until every hole is filled (as imstats.fill_invalid)."""
    if valid.all():
        return L.astype(np.float64)
    Lz = np.where(valid, L, 0.0)
    out = Lz.copy()
    todo = ~valid
    s = sigma
    while todo.any() and s < 4 * max(L.shape):
        num = _gauss_fft(Lz, s)
        den = _gauss_fft(valid.astype(float), s)
        ok = todo & (den > 1e-3)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        s *= 2
    if todo.any():
        out[todo] = np.median(L[valid])
    return out


def lap_noise(F: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Pixel noise (same units as F) from the MAD of the Laplacian, the
    estimator of imstats.noise_sigma (it reads correlated noise low; a
    synthetic still measured the same way is compared like for like)."""
    lap = ndi.convolve(np.asarray(F, np.float64), LAP, mode="reflect")
    v = lap[valid] if valid is not None else lap.ravel()
    m = np.median(v)
    return float(1.4826 * np.median(np.abs(v - m)) / math.sqrt(20.0))


def ridge_mask(L: np.ndarray, valid: np.ndarray, factor: float = 1.0,
               scales=(1.0, 2.0, 4.0, 8.0, 16.0), c_abs: float = 0.03, grow: float = 1.5) -> np.ndarray:
    """Dark-line (vessel) mask of a log image: the multi-scale Hessian ridge
    detector of imstats (strength D = s^2 (l1 - |l2|)+ above a threshold set
    by the opposite-polarity response and an absolute 0.03 Np floor), each
    scale's detections grown by grow * s + 2 px so the blurred flanks go too.
    factor < 1 lowers the thresholds to catch the faint capillary mesh."""
    Lf = fill_invalid(L, valid)
    M = np.zeros(L.shape, bool)
    for s in scales:
        Fxx = ndi.gaussian_filter(Lf, s, order=(0, 2), mode="reflect")
        Fyy = ndi.gaussian_filter(Lf, s, order=(2, 0), mode="reflect")
        Fxy = ndi.gaussian_filter(Lf, s, order=(1, 1), mode="reflect")
        tr = 0.5 * (Fxx + Fyy)
        disc = np.sqrt((0.5 * (Fxx - Fyy)) ** 2 + Fxy ** 2)
        l1, l2 = tr + disc, tr - disc
        D = s * s * np.clip(l1 - np.abs(l2), 0, None)
        Bm = s * s * np.clip(-l2 - np.abs(l1), 0, None)
        T = factor * max(float(np.quantile(Bm[valid], 0.95)), 0.354 * c_abs)
        m = (D > T) & valid
        m = ndi.binary_opening(m, iterations=1) if s <= 1 else m
        r = int(round(grow * s + 2))
        M |= ndi.binary_dilation(m, structure=_disk(r))
    return M


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _dog_power_radial(f: np.ndarray, s: float) -> np.ndarray:
    """|DoG_s(f)|^2 for the band G_s - G_2s at radial frequency f (cycles/px)."""
    a = -2 * math.pi ** 2 * f * f
    return (np.exp(a * s * s) - np.exp(a * 4 * s * s)) ** 2


_FR = np.linspace(0.0, 0.5, 20001)        # radial frequency grid for band integrals
_DFR = _FR[1] - _FR[0]


def _band_matrix(blur: float = 0.0) -> np.ndarray:
    """A[k, j] = variance in DoG band k of a unit-weight texture component j
    whose power spectrum is |DoG_j(f)|^2 (isotropic, 2-D integral written as
    a radial one), after a Gaussian blur `blur` px."""
    Kb = np.exp(-4 * math.pi ** 2 * blur * blur * _FR ** 2)
    P = np.array([_dog_power_radial(_FR, s) for s in BANDS])
    return np.einsum("kf,jf,f->kj", P, P, 2 * math.pi * _FR * Kb) * _DFR


# ======================================================================== measuring real stills
def _still_info(path: str) -> dict:
    """Metadata next to a mean_stabilized.tif: frames averaged, method, fps,
    ROI, motion (from metrics.json and transforms.csv)."""
    d = os.path.dirname(path)
    info = dict(path=path, burst=os.path.basename(d), method=os.path.basename(os.path.dirname(d)))
    try:
        with open(os.path.join(d, "metrics.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        fr, b, mo = m.get("frames", {}), m.get("burst", {}), m.get("motion", {})
        info["n_frames"] = int(fr.get("registered") or 0)
        info["fps"] = float(b.get("fps") or 0.0)
        info["roi_top_row"] = int(b.get("roi_top_row") or 0)
        info["speed_px_s"] = float(mo.get("speed_median_px_s") or 0.0)
        info["saccades"] = int(mo.get("saccades") or 0)
        info["duration_s"] = float(b.get("duration_s") or 0.0)
    except (OSError, ValueError):
        pass
    tp = os.path.join(d, "transforms.csv")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("registered") == "1"]
        if len(rows) > 2:
            dx = np.array([float(r["dx_px"]) for r in rows])
            dy = np.array([float(r["dy_px"]) for r in rows])
            st = np.hypot(np.diff(dx), np.diff(dy))
            info["traj_sd_px"] = float(np.hypot(dx.std(), dy.std()) / math.sqrt(2))
            info["traj_step_px"] = float(np.median(st))
    return info


def _illum_model(p, u, v):
    """ln E on normalized coordinates (u, v) = (x - W/2, y - H/2) / 1000 px:
    c + amp * exp(-q/2) + vig * r^2, q the rotated elliptical Gaussian."""
    c, amp, cx, cy, sx, sy, th, vig = p
    ct, st = math.cos(th), math.sin(th)
    du, dv = u - cx, v - cy
    a = (ct * du + st * dv) / sx
    b = (-st * du + ct * dv) / sy
    return c + amp * np.exp(-0.5 * (a * a + b * b)) + vig * (u * u + v * v)


def fit_illumination(Ls: np.ndarray, w: np.ndarray, step: int = 16) -> dict:
    """Fit _illum_model to a smooth log background Ls (weights w: vessel-free
    coverage) on a coarse grid.  Returns the parameters (amp, centre and widths
    in px, angle in deg, vignetting in Np at 1000 px from the centre), the RMS
    residual and the span of the fitted field over the frame."""
    from scipy.optimize import least_squares
    H, W = Ls.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    L = Ls[ys, xs]
    ww = w[ys, xs]
    ok = ww > 0.2
    u = (xs - W / 2) / 1000.0
    v = (ys - H / 2) / 1000.0
    L, u, v, ww = L[ok], u[ok], v[ok], np.sqrt(ww[ok])
    iy = int(np.argmax(L))
    best = None
    for s0 in (0.4, 0.8, 1.6):
        p0 = [float(np.percentile(L, 5)), float(np.ptp(L)) + 0.05, float(u[iy]), float(v[iy]), s0, s0, 0.0, 0.0]
        # a bright spot (amp >= 0) and vignetting that darkens (<= 0): the
        # unconstrained family trades a huge bump against negative vignetting
        lo = [-np.inf, 0.0, -2.0, -2.0, 0.15, 0.15, -math.pi / 2, -1.0]
        hi = [np.inf, 1.5, 2.0, 2.0, 3.0, 3.0, math.pi / 2, 0.0]
        r = least_squares(lambda p: (_illum_model(p, u, v) - L) * ww, p0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    fitted = _illum_model(p, (np.arange(W)[None, :] - W / 2) / 1000.0, (np.arange(H)[:, None] - H / 2) / 1000.0)
    resid = (_illum_model(p, u, v) - L)
    return dict(amp=float(p[1]), cx_px=float(p[2] * 1000), cy_px=float(p[3] * 1000), sx_px=float(p[4] * 1000),
                sy_px=float(p[5] * 1000), theta_deg=float(np.rad2deg(p[6])), vignette=float(p[7]),
                rms=float(np.sqrt(np.mean(resid ** 2))), span=float(np.ptp(fitted)))


def measure_still(path: str, noise_band_ratio=None) -> dict:
    """Stage A measurements on one stabilized still (float32 DN, NaN = no
    data): DN level, no-data mask, illumination fit, texture band RMS and
    orientation, heavy tails and patchiness of the vessel-masked residual,
    and the pixel noise (imstats estimator).

    The texture is measured on L = ln I with the vessels masked out
    (ridge_mask at the imstats threshold and at half of it, grown with scale)
    and filled by normalized convolution, per DoG band, on the kept pixels:
    rms (imstats threshold mask), rms_sens (half threshold), rms_bright (the
    half above the median only: faint or deep vessels the mask missed are
    dark, so they do not reach it), tex_* the same with the noise removed,
    kurt / kurt_bright (tail weight), local_cv (SD of the log band RMS over
    64 px blocks: patchiness), anis and fibre_dir_deg (orientation).
    noise_band_ratio[k] (band variance of the still's noise per unit squared
    Laplacian noise estimate, from the half-split noise fields) removes the
    noise from the band variances."""
    import tifffile
    I = tifffile.imread(path).astype(np.float64)
    valid = np.isfinite(I) & (I > 0)
    H, W = I.shape
    out = _still_info(path)
    out["shape"] = [H, W]
    out["nan_frac"] = float(1 - valid.mean())
    lab, n = ndi.label(~valid)
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        border = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        comps.append((area, border))
    out["nan_holes_interior"] = int(sum(1 for a, b in comps if not b and a >= 20))
    out["nan_holes_px"] = [a for a, b in sorted(comps, reverse=True) if not b and a >= 20][:6]
    v = I[valid]
    out["dn_p01"], out["dn_p50"], out["dn_p99"] = (float(np.percentile(v, q)) for q in (1, 50, 99))
    L = np.log(np.where(valid, I, 1.0))
    Lf = fill_invalid(L, valid)
    # pixel noise, imstats' estimator on the flattened log image
    Fhp = Lf - _gauss_fft(Lf, 16)
    out["noise_lap"] = lap_noise(Fhp, ndi.binary_erosion(valid, iterations=3))
    # vessel masks: imstats threshold, and half of it (faint mesh)
    res = {}
    for tag, fac in (("", 1.0), ("_sens", 0.5)):
        M = ridge_mask(L, valid, factor=fac)
        keep = valid & ~M
        res[tag] = keep
        out["keep_frac" + tag] = float(keep.mean())
    keep = res["_sens"]
    Lb = fill_invalid(L, keep)
    # level and illumination: smooth vessel-free background
    out["bg_dn_p50"] = float(np.exp(np.median(L[keep])))
    wmap = _gauss_fft(keep.astype(float), 48)
    Ls = _gauss_fft(np.where(keep, L, 0.0), 48) / np.maximum(wmap, 1e-3)
    ill = fit_illumination(Ls, wmap)
    out["illum"] = ill
    # texture bands
    ratio = noise_band_ratio or {}
    bands = {}
    for s in BANDS:
        B = _dog(Lb, s)
        er = ndi.binary_erosion(valid, iterations=int(3 * s + 2))
        ks = {}
        for tag in ("", "_sens"):
            kk = res[tag] & ndi.binary_erosion(res[tag], structure=_disk(int(min(s, 6)))) & er
            ks[tag] = kk if kk.sum() >= 2000 else res[tag]
            var = float(B[ks[tag]].var())
            nvar = float(ratio.get(str(s), 0.0)) * out["noise_lap"] ** 2
            bands.setdefault(s, {})["rms" + tag] = math.sqrt(var)
            bands[s]["tex_rms" + tag] = math.sqrt(max(var - nvar, 0.0))
        k = ks["_sens"]
        vals = B[k]
        bands[s]["kurt"] = float(sst.kurtosis(vals, fisher=True))
        # the bright half only: the dark vessel residue (faint capillaries,
        # blurred deep vessels the mask missed) leaves it alone
        med = np.median(vals)
        vb = float(np.mean((vals[vals > med] - med) ** 2))
        nvar = float(ratio.get(str(s), 0.0)) * out["noise_lap"] ** 2
        bands[s]["rms_bright"] = math.sqrt(vb)
        bands[s]["tex_rms_bright"] = math.sqrt(max(vb - nvar, 0.0))
        # the bright side mirrored: tail weight without the dark vessel residue
        up = vals[vals > np.median(vals)] - np.median(vals)
        bands[s]["kurt_bright"] = float(sst.kurtosis(np.r_[up, -up], fisher=True))
        bands[s]["skew"] = float(sst.skew(vals))
        # patchiness: spread of the local band RMS over 64 px blocks
        bs = 64
        hh, ww_ = (H // bs) * bs, (W // bs) * bs
        B2 = np.where(k, B, np.nan)[:hh, :ww_].reshape(hh // bs, bs, ww_ // bs, bs)
        cnt = np.isfinite(B2).sum(axis=(1, 3))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loc = np.sqrt(np.nanmean(B2 ** 2, axis=(1, 3)))
        loc = loc[cnt > bs * bs // 3]
        bands[s]["local_cv"] = float(np.std(np.log(loc))) if loc.size > 8 else float("nan")
        # orientation: doubled-angle mean of the band's gradient
        gy, gx = np.gradient(_gauss_fft(B, max(0.5, s / 2)))
        z = (gx + 1j * gy)[k] ** 2
        zz = z.sum() / max(np.abs(z).sum(), 1e-30)
        bands[s]["anis"] = float(abs(zz))
        bands[s]["fibre_dir_deg"] = float((np.rad2deg(np.angle(zz) / 2) + 90) % 180)
    out["bands"] = {str(s): b for s, b in bands.items()}
    # dark lumps: fraction of the residual's 4-8 px band below -3 sd, both tails
    B4 = _dog(Lb, 4)[keep]
    sd = B4.std()
    out["tail_lo_3sd"] = float((B4 < -3 * sd).mean())
    out["tail_hi_3sd"] = float((B4 > 3 * sd).mean())
    return out


def measure_camera(raw_dirs, n_per_burst: int = 6, block: int = 32) -> dict:
    """Per-frame noise of the camera from raw frames: in 32 px blocks, the
    Laplacian-MAD noise against the block mean, both moved to linear DN
    through the burst's gamma (manifest.json), fitted as
    var = x / g_e + read^2; g_e * 10^(gain_dB / 20) is the conversion at
    0 dB (e-/DN).  Also the temporal row/column noise (row medians of
    consecutive-frame differences beyond the iid expectation)."""
    import tifffile
    rows_out = []
    for d in raw_dirs:
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        st = man["camera_state_at_start"]
        gam = float(st.get("gamma", 1.0))
        with open(os.path.join(d, "frames.csv"), encoding="utf-8", newline="") as fh:
            fr = list(csv.DictReader(fh))
        files = sorted(glob.glob(os.path.join(d, "frame_*.tif")))
        idx = np.linspace(0, len(files) - 2, n_per_burst).astype(int)
        X, V, rown, coln, pix = [], [], [], [], []
        for i in idx:
            img = tifffile.imread(files[i]).astype(np.float64)
            lap = ndi.convolve(img, LAP, mode="reflect")
            H, W = img.shape
            h, w = (H // block) * block, (W // block) * block
            Lb = lap[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
            Ib = img[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
            ok = Ib.max(1) < 0.97 * FULL_SCALE
            m = Ib.mean(1)[ok]
            s = 1.4826 * np.median(np.abs(Lb - np.median(Lb, 1, keepdims=True)), 1)[ok] / math.sqrt(20)
            x = FULL_SCALE * (m / FULL_SCALE) ** (1 / gam)
            X.append(x)
            V.append((s * x / (gam * np.maximum(m, 1))) ** 2)
            nxt = tifffile.imread(files[i + 1]).astype(np.float64)
            dd = nxt - img
            dl = ndi.convolve(dd, LAP, mode="reflect")
            sp = 1.4826 * np.median(np.abs(dl - np.median(dl))) / math.sqrt(20) / math.sqrt(2)
            for axis, store, n_ in ((1, rown, W), (0, coln, H)):
                med = np.median(dd, axis)
                hp = med - ndi.uniform_filter1d(med, 9)
                meas = hp.std() / math.sqrt(2) / math.sqrt(1 - 1 / 9)
                iid = 1.2533 * sp / math.sqrt(n_)
                store.append(math.sqrt(max(meas ** 2 - iid ** 2, 0.0)))
            pix.append(sp)
        x, var = np.concatenate(X), np.concatenate(V)
        qs = np.quantile(x, np.linspace(0.02, 0.98, 15))
        bx, bv = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (x >= a) & (x < b)
            if m.sum() > 20:
                bx.append(np.median(x[m]))
                bv.append(np.median(var[m]))
        bx, bv = np.array(bx), np.array(bv)
        slope, icpt = np.polyfit(bx, bv, 1)
        gain_db = float(np.median([float(r["gain"]) for r in fr]))
        g_e = 1.0 / slope if slope > 0 else float("nan")
        # lamp flicker: frame-to-frame change of the mean brightness (robust,
        # so blinks do not count; consecutive differences, so slow drift and
        # the eye's motion do not either)
        means = np.array([float(tifffile.imread(f)[::4, ::4].mean()) for f in files])
        rel = np.diff(means) / means[:-1]
        flicker = float(1.4826 * np.median(np.abs(rel - np.median(rel))) / math.sqrt(2))
        rows_out.append(dict(burst=os.path.basename(d), gamma=gam, gain_db=gain_db,
                             exposure_us=float(st.get("exposure_us", 0.0)),
                             e_per_dn=g_e, e_per_dn_0db=g_e * 10 ** (gain_db / 20),
                             read_dn=math.sqrt(max(icpt, 0.0)),
                             frame_noise_dn=float(np.median(pix)),
                             frame_noise_ln=float(np.median(gam * np.sqrt(var) / x)),
                             row_noise_dn=float(np.median(rown)), col_noise_dn=float(np.median(coln)),
                             flicker=flicker))
    e0 = float(np.median([r["e_per_dn_0db"] for r in rows_out]))
    return dict(bursts=rows_out, e_per_dn_0db=e0, flicker=float(np.median([r["flicker"] for r in rows_out])),
                row_noise_dn=float(np.median([r["row_noise_dn"] for r in rows_out])),
                col_noise_dn=float(np.median([r["col_noise_dn"] for r in rows_out])))


def _shift_frame(img, d):
    """The stabilization pipeline's warp (analysis/stabilize/register.shift):
    bilinear, undoing a content displacement d = (dx, dy)."""
    import cv2
    m = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])
    return cv2.warpAffine(np.ascontiguousarray(img, dtype=np.float32), m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _glare_mask_frame(img, frac=0.98, dilate=30):
    import cv2
    c = img >= frac * FULL_SCALE
    if not c.any():
        return c
    k = max(3, dilate | 1)
    return cv2.dilate(c.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def rebuild_halves(still_dir: str, raw_dir: str) -> dict:
    """Rebuild a translation-stabilized mean from its raw frames with the
    pipeline's own shifts (transforms.csv) as two interleaved halves.  Returns
    the full mean, the half difference D = (ln A - ln B) / 2 rescaled to the
    noise of the full mean (Np), the coverage and the agreement with the
    stored mean_stabilized.tif."""
    import tifffile
    with open(os.path.join(still_dir, "transforms.csv"), encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["registered"] == "1"]
    H, W = tifffile.imread(os.path.join(raw_dir, rows[0]["filename"])).shape
    acc = {k: np.zeros((H, W)) for k in ("sA", "cA", "sB", "cB")}
    for j, r in enumerate(rows):
        img = tifffile.imread(os.path.join(raw_dir, r["filename"])).astype(np.float32)
        d = (float(r["dx_px"]), float(r["dy_px"]))
        w = _shift_frame(img, d).astype(np.float64)
        seen = _shift_frame((~_glare_mask_frame(img)).astype(np.float32), d) >= 0.5
        h = "A" if j % 2 == 0 else "B"
        acc["s" + h] += w * seen
        acc["c" + h] += seen
    n = len(rows)
    cover = acc["cA"] + acc["cB"]
    valid = cover >= 0.5 * n
    mean = np.full((H, W), np.nan)
    mean[valid] = (acc["sA"] + acc["sB"])[valid] / cover[valid]
    real = tifffile.imread(os.path.join(still_dir, "mean_stabilized.tif")).astype(np.float64)
    both = valid & np.isfinite(real)
    agree = float(np.median(np.abs(mean[both] - real[both]) / real[both]))
    ok = valid & (acc["cA"] > 0.2 * n) & (acc["cB"] > 0.2 * n)
    with np.errstate(all="ignore"):
        A = acc["sA"] / acc["cA"]
        B = acc["sB"] / acc["cB"]
        D = np.where(ok, (np.log(A) - np.log(B)) / 2 * np.sqrt(4 * acc["cA"] * acc["cB"] / cover ** 2), np.nan)
    return dict(mean=mean, D=D, cover=cover, n=n, rel_agreement=agree)


def _acov(D: np.ndarray, m: np.ndarray, K: int) -> np.ndarray:
    """Masked autocovariance of D over lags |dy|, |dx| <= K."""
    H, W = D.shape
    sh = (sfft.next_fast_len(H + K), sfft.next_fast_len(W + K))
    X = sfft.rfft2(np.where(m, D, 0.0), sh)
    Mf = sfft.rfft2(m.astype(float), sh)
    num = sfft.irfft2(X * np.conj(X), sh)
    den = sfft.irfft2(Mf * np.conj(Mf), sh)
    iy = np.arange(-K, K + 1) % sh[0]
    ix = np.arange(-K, K + 1) % sh[1]
    return num[np.ix_(iy, ix)] / np.maximum(den[np.ix_(iy, ix)], 1.0)


def noise_structure(D: np.ndarray, vessel_free: np.ndarray, K: int = 12, smooth: float = 16.0) -> dict:
    """Level and spatial structure of a still's noise field D (Np).

    D is split into a smooth part (Gaussian `smooth` px, normalized
    convolution over the kept pixels) and the rest.  The smooth part of a
    half-split difference is the halves sampling the camera-fixed
    illumination and the lamp's frame-to-frame flicker differently; in the
    full still it is part of the illumination, so it is reported apart
    (smooth_sd).  On the rest: SD, the imstats (Laplacian) estimate, lag
    correlations, and a fit of its autocovariance as white noise through the
    average bilinear-interpolation kernel + row streaks (constant along x)
    + column streaks + an isotropic Gaussian-correlated part, with their
    variance shares; and the DoG band variances per squared Laplacian
    estimate (to remove noise from the texture bands of the stills)."""
    from scipy.optimize import least_squares
    m = np.isfinite(D) & vessel_free
    m &= ndi.binary_erosion(np.isfinite(D), iterations=4)
    Dz = np.where(m, D - D[m].mean(), 0.0)
    Ds = _gauss_fft(Dz, smooth) / np.maximum(_gauss_fft(m.astype(float), smooth), 1e-3)
    smooth_sd = float(Ds[m].std())
    Dz = np.where(m, Dz - Ds, 0.0)
    C = _acov(Dz, m, K)
    lags = np.arange(-K, K + 1)
    kb = np.where(lags == 0, 1.0, np.where(np.abs(lags) == 1, 0.25, 0.0))
    rb, rr, rc = np.outer(kb, kb), np.outer(kb, np.ones_like(kb)), np.outer(np.ones_like(kb), kb)
    YY, XX = np.meshgrid(lags, lags, indexing="ij")

    def model(p):
        s2, wr, wc, wl, ell = p
        return s2 * ((1 - wr - wc - wl) * rb + wr * rr + wc * rc + wl * np.exp(-(YY ** 2 + XX ** 2) / (2 * ell ** 2)))

    fit = least_squares(lambda p: ((model(p) - C) / C[K, K]).ravel(), [C[K, K], 0.1, 0.05, 0.05, 4.0],
                        bounds=([0, 0, 0, 0, 1.0], [1, 0.9, 0.9, 0.9, 60.0]))
    s2, wr, wc, wl, ell = (float(x) for x in fit.x)
    lap = lap_noise(Dz, m)
    ratio = {}
    for s in BANDS[:3]:
        B = _dog(Dz, s)
        k = m & ndi.binary_erosion(m, iterations=int(2 * s + 1))
        ratio[str(s)] = float(B[k].var() / lap ** 2)
    return dict(sd=float(np.sqrt(C[K, K])), smooth_sd=smooth_sd, lap=lap, white=1 - wr - wc - wl, rows=wr, cols=wc,
                lowf=wl,
                lowf_ell_px=ell, fit_rms=float(np.sqrt(np.mean(((model(fit.x) - C) / C[K, K]) ** 2))),
                corr_x=[float(C[K, K + i] / C[K, K]) for i in (1, 2, 5, 10)],
                corr_y=[float(C[K + i, K] / C[K, K]) for i in (1, 2, 5, 10)],
                band_ratio=ratio)


def camera_pattern(still_dir: str, raw_dir: str) -> dict:
    """Camera-fixed pattern (fixed-pattern noise): the mean over frames of
    (frame - the stabilized mean moved back to the frame's position).  The
    tissue cancels, the camera's own pattern stays, the temporal noise is
    reduced by sqrt(N).  Returns pixel, row and column pattern SDs (DN and
    relative to the level)."""
    import cv2
    import tifffile
    with open(os.path.join(still_dir, "transforms.csv"), encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["registered"] == "1"]
    mean = tifffile.imread(os.path.join(still_dir, "mean_stabilized.tif")).astype(np.float32)
    ok = np.isfinite(mean)
    mz = np.where(ok, mean, 0).astype(np.float32)
    H, W = mean.shape
    acc = np.zeros((H, W))
    cnt = np.zeros((H, W))
    pix = []
    for r in rows:
        img = tifffile.imread(os.path.join(raw_dir, r["filename"])).astype(np.float64)
        d = (float(r["dx_px"]), float(r["dy_px"]))
        M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
        back = cv2.warpAffine(mz, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        bok = cv2.warpAffine(ok.astype(np.float32), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0.5
        bok &= img < 0.97 * FULL_SCALE
        acc += np.where(bok, img - back, 0.0)
        cnt += bok
        if len(pix) < 3:
            pix.append(lap_noise(img))
    n = len(rows)
    P = np.where(cnt > 0.8 * n, acc / np.maximum(cnt, 1), np.nan)
    m = np.isfinite(P)
    m &= ndi.binary_erosion(m, iterations=3)
    Pz = np.where(m, P, 0.0)
    lvl = float(np.nanmedian(mean))
    fine = lap_noise(Pz, m)
    temporal = float(np.median(pix)) / math.sqrt(n)
    pix_dn = math.sqrt(max(fine ** 2 - temporal ** 2, 0.0))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rowp = np.nanmedian(np.where(m, P - ndi.gaussian_filter1d(Pz, 3, axis=0), np.nan), 1)
        colp = np.nanmedian(np.where(m, P - ndi.gaussian_filter1d(Pz, 3, axis=1), np.nan), 0)
    # dust: small dark specks (<= 20 px) well below the pattern's spread
    B = ndi.gaussian_filter(Pz, 1.0) - ndi.gaussian_filter(Pz, 6.0)
    sB = 1.4826 * float(np.median(np.abs(B[m])))
    lab, nd = ndi.label((B < -6 * sB) & m)
    specks = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        reg = lab[sl] == i
        if reg.sum() <= 20:
            specks.append(float(-B[sl][reg].min() / lvl))
    return dict(level_dn=lvl, pixel_dn=pix_dn, pixel_rel=pix_dn / lvl, row_dn=float(np.nanstd(rowp)),
                col_dn=float(np.nanstd(colp)), n=n, dust_per_mpx=len(specks) / (m.sum() / 1e6),
                dust_depth=float(np.median(specks)) if specks else 0.0,
                note="pixel_rel is an upper bound: registration residue of the tissue also lands in this estimate")


def measure_real(paths, raw_root: str | None = None, out: str | None = DATA_PATH, log=print) -> dict:
    """Stage A of the plan: measure the image formation on the real stills.

    paths: mean_stabilized.tif files (one per burst: the reference set).
    raw_root: folder of raw bursts (reference_data); for the stills whose raw
    frames are there, the camera noise (gain, read noise, row/column noise),
    the still's own noise field (half split), its structure and the camera's
    fixed pattern are measured too.  The result (small) is written to `out`
    (vesselscene/data/real_imaging.json) and is where ImagingParams gets its
    defaults."""
    paths = list(paths)
    halves = {}
    camera = None
    if raw_root:
        pairs = []
        for p in paths:
            b = os.path.basename(os.path.dirname(p))
            tdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(p))), "translation", b)
            rdir = os.path.join(raw_root, b)
            if os.path.isdir(rdir) and os.path.exists(os.path.join(tdir, "transforms.csv")):
                pairs.append((b, tdir, rdir))
        if pairs:
            camera = measure_camera([r for _, _, r in pairs])
            log(f"camera: {camera['e_per_dn_0db']:.2f} e-/DN at 0 dB")
        for b, tdir, rdir in pairs:
            h = rebuild_halves(tdir, rdir)
            L = np.log(np.where(np.isfinite(h["mean"]), h["mean"], 1.0))
            vf = ~ridge_mask(L, np.isfinite(h["mean"]), factor=0.5)
            ns = noise_structure(h["D"], vf)
            ns["n"] = h["n"]
            ns["rel_agreement"] = h["rel_agreement"]
            ns["pattern"] = camera_pattern(tdir, rdir)
            halves[b] = ns
            log(f"noise {b}: sd {ns['sd']:.2e} Np, white {ns['white']:.2f} rows {ns['rows']:.2f} cols {ns['cols']:.2f}; "
                f"pattern {ns['pattern']['pixel_rel']:.1e}")
    ratio = None
    if halves:
        ratio = {k: float(np.mean([h["band_ratio"][k] for h in halves.values()])) for k in ("1", "2", "4")}
    stills = {}
    for p in paths:
        m = measure_still(p, ratio)
        m.pop("path", None)
        stills[m["burst"].replace("burst_", "")] = m
        log(f"still {m['burst']}: level {m['bg_dn_p50']:.0f} DN, noise {m['noise_lap']:.2e}, "
            f"tex 1..64: " + " ".join(f"{m['bands'][str(s)]['tex_rms_sens'] * 1e3:.1f}" for s in BANDS))
    res = dict(format="vesselscene-real-imaging", version=1, bands_px=list(BANDS),
               units=dict(dn="12-bit camera DN (mean of frames)", log="Np (natural log)"),
               noise_band_ratio=ratio, camera=camera, still_noise=halves, stills=stills,
               summary=_summarise(stills, halves, camera))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_round(res), fh, indent=1)
        if os.path.abspath(out) == os.path.abspath(DATA_PATH):
            global _MEASURED
            _MEASURED = None
    return res


def _summarise(stills: dict, halves: dict, camera) -> dict:
    """Medians and ranges over the stills of what ImagingParams uses."""
    S = list(stills.values())

    def agg(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if v.size == 0:
            return None
        return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))

    out = dict(level_dn=agg(s["bg_dn_p50"] for s in S), noise_lap=agg(s["noise_lap"] for s in S),
               nan_frac=agg(s["nan_frac"] for s in S), nan_holes=agg(s["nan_holes_interior"] for s in S),
               n_frames=agg(s.get("n_frames") for s in S), speed_px_s=agg(s.get("speed_px_s") for s in S),
               traj_sd_px=agg(s.get("traj_sd_px") for s in S),
               saccade_rate_hz=agg(s.get("saccades", 0) / max(s.get("duration_s", 0) or 1, 1e-3) for s in S))
    for k in ("amp", "sx_px", "sy_px", "vignette", "span", "rms"):
        out["illum_" + k] = agg(s["illum"][k] for s in S)
    out["illum_r_px"] = agg(math.hypot(s["illum"]["cx_px"], s["illum"]["cy_px"]) for s in S)
    # one still's whole fit, the one whose span is the median: medians of the
    # separate parameters of different stills need not make a typical field
    # (among the full 1200-row frames when there are any: the default image size)
    full = [s for s in S if s["shape"][0] >= 1000] or S
    spans = np.array([s["illum"]["span"] for s in full])
    typ = full[int(np.argmin(np.abs(spans - np.median(spans))))]
    out["illum_typical"] = dict(typ["illum"], burst=typ["burst"])
    for s_ in BANDS:
        b = [s["bands"][str(s_)] for s in S]
        out[f"band{s_}"] = {k: agg(x[k] for x in b) for k in ("rms", "tex_rms", "tex_rms_sens", "rms_bright", "tex_rms_bright", "kurt", "kurt_bright", "local_cv", "anis")}
    out["tail_lo_3sd"] = agg(s["tail_lo_3sd"] for s in S)
    out["tail_hi_3sd"] = agg(s["tail_hi_3sd"] for s in S)
    if halves:
        H = list(halves.values())
        for k in ("white", "rows", "cols", "lowf"):
            out["still_noise_" + k] = agg(h[k] for h in H)
        out["pattern_pixel_rel"] = agg(h["pattern"]["pixel_rel"] for h in H)
        out["pattern_row_dn"] = agg(h["pattern"]["row_dn"] for h in H)
        out["pattern_col_dn"] = agg(h["pattern"]["col_dn"] for h in H)
        out["dust_per_mpx"] = agg(h["pattern"]["dust_per_mpx"] for h in H)
        out["dust_depth"] = agg(h["pattern"]["dust_depth"] for h in H)
        out["still_noise_lap_over_sd"] = agg(h["lap"] / h["sd"] for h in H)
    return out


def _round(o, sig: int = 5):
    if isinstance(o, dict):
        return {str(k): _round(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, sig) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if not np.isfinite(f) or f == 0 else float(f"{f:.{sig}g}")
    if isinstance(o, np.integer):
        return int(o)
    return o


# ======================================================================== parameters
_MEASURED = None


def measured(path: str = DATA_PATH) -> dict:
    """The saved Stage A measurements (real_imaging.json), cached."""
    global _MEASURED
    if path != DATA_PATH:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if _MEASURED is None:
        try:
            with open(path, encoding="utf-8") as fh:
                _MEASURED = json.load(fh)
        except OSError:
            _MEASURED = {}
    return _MEASURED


def _m(key: str, fallback, stat: str = "median"):
    """Default from the measured summary (key path with '.'), else fallback."""
    d = measured().get("summary", {})
    try:
        for k in key.split("."):
            d = d[k]
        v = d[stat] if isinstance(d, dict) else d
        return type(fallback)(v) if v is not None else fallback
    except (KeyError, TypeError, ValueError):
        return fallback


def _mcam(key: str, fallback):
    try:
        return type(fallback)(measured()["camera"][key])
    except (KeyError, TypeError, ValueError):
        return fallback


def _tex_default():
    """Texture band RMS: the bright half of the vessel-masked residual
    (noise removed), which the dark residue of faint and deep vessels the
    mask missed leaves alone."""
    return tuple(_m(f"band{s}.tex_rms_bright", f) for s, f in
                 zip(BANDS, (7.2e-4, 1.59e-3, 2.97e-3, 5.0e-3, 7.7e-3, 9.7e-3, 1.3e-2)))


@dataclass
class ImagingParams:
    """Everything image formation needs besides the optical density.
    Defaults are the medians measured on the real reference stills
    (real_imaging.json); ImagingParams.random draws wide ranges for training.
    Pixel lengths are image pixels; log amplitudes are Np as seen in ln DN."""
    # --- brightness and illumination: ln E = amp * bump + vignette * (r / 1000 px)^2
    level: float = field(default_factory=lambda: _m("level_dn", 2460.0))   # median DN of the vessel-free background
    # the illumination fitted to the still of median span (summary.illum_typical)
    illum_amp: float = field(default_factory=lambda: _m("illum_typical.amp", 0.31))   # Np, the lamp's bump in ln E
    illum_cx: float = field(default_factory=lambda: _m("illum_typical.cx_px", -670.0))  # px, bump centre from the
    illum_cy: float = field(default_factory=lambda: _m("illum_typical.cy_px", -600.0))  # image centre
    illum_sx: float = field(default_factory=lambda: _m("illum_typical.sx_px", 440.0))   # px, bump SD (in ln E)
    illum_sy: float = field(default_factory=lambda: _m("illum_typical.sy_px", 985.0))   # along its axes
    illum_theta: float = field(default_factory=lambda: _m("illum_typical.theta_deg", -4.0))   # deg
    vignette: float = field(default_factory=lambda: _m("illum_typical.vignette", -0.06))  # Np at 1000 px from the centre
    # --- scleral texture tau (Np)
    tex_rms: tuple = field(default_factory=_tex_default)   # DoG band RMS, bands 1-2 ... 64-128 px, noise excluded
    tex_scale: float = 1.0          # multiplies every band (e.g. < 1 once the faint mesh is rendered)
    tex_anis: float = field(default_factory=lambda: _m("band4.anis", 0.18))   # anisotropy of the texture gradients, < 16 px
    tex_anis_coarse: float = field(default_factory=lambda: _m("band16.anis", 0.25))   # ... and at 16 px and above
    tex_dir: float = 70.0           # deg, fibre direction (0 = +x, 90 = +y): varies per site
    # SD of the log envelope over the scales 0.5-1.4 / 2-2.8 / 4-5.7 / 8-11 px, calibrated by simulating
    # stills (OD = 0) so the bright-side kurtosis of their DoG bands 1 / 2 / 4 / 8 px comes out near the
    # real medians 3.2 / 1.8 / 0.86 / 0.30 (reached: ~2.9 / 2.0 / 1.0 / 0.25)
    tex_patchy: tuple = (0.65, 0.45, 0.35, 0.40)
    tex_patch_px: float = 100.0     # envelope smoothing (px): rough and smooth patches of ~200 px
    lump_density: float = 3.0       # sparse lumps per 10^5 px^2 (hard negatives; not calibrated)
    lump_amp: float = 0.006         # Np, median |amplitude|
    lump_r: tuple = (2.0, 12.0)     # px, radius range (log-uniform)
    lump_dark: float = 0.5          # fraction of the lumps that are dark
    # the texture is mostly the reflectance of the tissue beneath the vessels, and over a vessel only the light that
    # crossed the blood came from there: of the detected light exp(-OD) = f + (1 - f) T, the textured part is
    # (1 - f) T, so ln I carries tau x (1 - f e^OD) / (1 - f) inside a vessel (1 on the background, 0 where
    # the vessel is black).  tex_bypass = f (0: the texture multiplies everything, as v0).  In the real stills the
    # structure along the lumen of medium and wide vessels is 0.68-0.71 of that beside them (_scratch/v1/optics)
    tex_bypass: float = 0.0
    # --- camera (Basler acA1920-40um, IMX249, Mono12)
    gain_db: float = 20.4           # analogue gain: 15-50-52, the median-noise reference still (bursts 11.6-23.9 dB)
    gamma: float = 0.8              # camera gamma: 15-50-52 (bursts used 0.5, 0.8, 1.0); only the noise sees it
    e_per_dn0: float = field(default_factory=lambda: _mcam("e_per_dn_0db", 7.6))   # e-/DN at 0 dB (measured)
    read_e: float = 7.0             # read noise, e- (datasheet order; too small to measure here)
    row_noise_dn: float = field(default_factory=lambda: _mcam("row_noise_dn", 1.3))  # temporal, per frame
    col_noise_dn: float = field(default_factory=lambda: _mcam("col_noise_dn", 1.5))
    fpn_pixel: float = field(default_factory=lambda: _m("pattern_pixel_rel", 2.2e-3))   # camera-fixed, relative
    fpn_row_dn: float = field(default_factory=lambda: _m("pattern_row_dn", 2.7))
    fpn_col_dn: float = field(default_factory=lambda: _m("pattern_col_dn", 0.73))
    dust_rate: float = field(default_factory=lambda: _m("dust_per_mpx", 17.7) * 2.304)   # specks per 1920 x 1200
    dust_px: float = 1.2            # their radius (Gaussian SD, px)
    dust_depth: float = field(default_factory=lambda: 2 * _m("dust_depth", 0.01))   # peak darkening (DoG reads ~half)
    flicker: float = field(default_factory=lambda: _mcam("flicker", 0.005))   # frame-to-frame brightness SD
    noise_sigma_c: float = 0.0      # extra Gaussian correlation of the averaged noise (px); bilinear alone fits
    # --- burst, eye motion, registration
    n_frames: int = field(default_factory=lambda: _m("n_frames", 138))
    fps: float = 32.2               # 1920x1200: 32.2; 500 rows: 74
    exposure_ms: float = 12.8
    speed_px_s: float = field(default_factory=lambda: _m("speed_px_s", 225.0))   # median eye speed
    drift_px: float = 12.0          # SD of the positions within one fixation
    saccade_rate: float = field(default_factory=lambda: _m("saccade_rate_hz", 0.33))
    saccade_px: float = 20.0        # SD of the fixation centres
    reg_error_px: float = 0.3       # SD of a frame's residual registration error (the median over the field)
    # the registration residual varies over the field (the eye does not move rigidly: quadrants of a translation-
    # stabilized burst disagree by 1-1.7 px per frame, the non-rigid fields' tile residuals 0.15-10 px): a smooth
    # log-normal field of SD exp(N(ln reg_error_px, reg_field_cv^2)), correlation length reg_field_px (0: uniform)
    reg_field_cv: float = 0.0
    reg_field_px: float = 300.0
    reg_field_max: float = 4.0      # px: cap of the field (a region registered worse than this is gated out)
    max_blur_px: float = 12.0       # frames blurred more than this are gated out, like blinks
    cover_min: float = 0.5          # pipeline: valid where at least half the frames saw the pixel
    # --- glare (specular reflections of the lamp, fixed to the camera)
    glare_rate: float = 1.0         # mean number of spots per 1920 x 1200 frame
    glare_px: float = 4.0           # radius of a spot's clipped core
    glare_aspect: float = 1.5       # up to this elongation
    glare_dilate: int = 30          # the pipeline's mask dilation (diameter, px)
    bloom_amp: float = 0.08         # bloom at the core edge, fraction of the level
    bloom_px: float = 12.0          # its e-folding distance
    # --- seeds: the same scene_seed gives the same texture (the same anatomy),
    #     the same camera_seed the same camera pattern and glare positions
    scene_seed: int | None = None
    camera_seed: int | None = None

    def replace(self, **kw) -> "ImagingParams":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def random(cls, rng: np.random.Generator, **fixed) -> "ImagingParams":
        """Wide ranges around the measured values, for training: every
        measured quantity spans at least the range of the real stills."""
        u = rng.uniform

        def lu(a, b):
            return float(math.exp(u(math.log(a), math.log(b))))

        base = cls()
        tex = tuple(float(t * math.exp(rng.normal(0, 0.15))) for t in base.tex_rms)
        # illum_amp up to 0.6 and tex_scale up to 1.1 (were 0.9 and 1.8: the random preset's illumination span
        # reached 0.82 Np against 0.14-0.77 real, and its texture was cloudier than any still; review T12)
        p = dict(level=u(1300, 3200), illum_amp=u(0.05, 0.6), illum_cx=u(-1200, 1200), illum_cy=u(-900, 900),
                 illum_sx=lu(300, 2500), illum_sy=lu(300, 2500), illum_theta=u(-90, 90), vignette=u(-0.3, 0.0),
                 tex_rms=tex, tex_scale=lu(0.4, 1.1), tex_anis=u(0.0, 0.45), tex_anis_coarse=u(0.0, 0.5),
                 tex_dir=u(0, 180),
                 tex_patchy=tuple(float(x * math.exp(rng.normal(0, 0.35))) for x in base.tex_patchy),
                 tex_patch_px=lu(24, 128), lump_density=lu(0.3, 30),
                 lump_amp=lu(0.002, 0.02), lump_dark=u(0.3, 0.8), tex_bypass=u(0.0, 0.5), gain_db=u(10, 24),
                 gamma=float(rng.choice([0.5, 0.8, 1.0])), read_e=u(4, 12), row_noise_dn=u(0.5, 3),
                 col_noise_dn=u(0.5, 3), fpn_pixel=lu(8e-4, 5e-3), fpn_row_dn=u(1, 4.5), fpn_col_dn=u(0.3, 1.5),
                 dust_rate=u(0, 40), dust_px=u(0.8, 3), dust_depth=lu(0.005, 0.08), flicker=lu(0.001, 0.015),
                 noise_sigma_c=float(u(0, 0.6) if rng.random() < 0.3 else 0.0),
                 n_frames=int(round(lu(14, 700))), fps=float(rng.choice([32.2, 74.0])), speed_px_s=lu(40, 500),
                 drift_px=lu(3, 40), saccade_rate=u(0, 0.7), saccade_px=lu(5, 60), reg_error_px=u(0.1, 0.9),
                 reg_field_cv=u(0.0, 0.8), reg_field_px=lu(150, 600),
                 glare_rate=u(0, 3), glare_px=u(2, 9), glare_aspect=u(1, 4), bloom_amp=u(0, 0.2),
                 bloom_px=u(5, 25), scene_seed=int(rng.integers(2 ** 31)), camera_seed=int(rng.integers(2 ** 31)))
        p.update(fixed)
        return cls(**p)


# ======================================================================== the scene's own fields
def illumination(shape, p: ImagingParams) -> np.ndarray:
    """ln E (Np): the lamp's broad bump plus radial vignetting, zero at the
    median (the level sets the DN)."""
    H, W = shape
    u = (np.arange(W)[None, :] - W / 2) / 1000.0
    v = (np.arange(H)[:, None] - H / 2) / 1000.0
    lnE = _illum_model([0.0, p.illum_amp, p.illum_cx / 1000, p.illum_cy / 1000, max(p.illum_sx, 1) / 1000,
                        max(p.illum_sy, 1) / 1000, math.radians(p.illum_theta), p.vignette], u, v)
    return lnE - np.median(lnE)


def _field(shape, power, rng, pad: int) -> np.ndarray:
    """A stationary Gaussian random field with power spectrum power(fy, fx)
    (per unit frequency area), generated on a padded grid (no wrap-around)."""
    H, W = shape
    Hp, Wp = sfft.next_fast_len(H + 2 * pad), sfft.next_fast_len(W + 2 * pad)
    z = rng.standard_normal((Hp, Wp))
    fy = sfft.fftfreq(Hp)[:, None]
    fx = sfft.rfftfreq(Wp)[None, :]
    X = sfft.irfft2(sfft.rfft2(z, workers=-1) * np.sqrt(np.maximum(power(fy, fx), 0.0)), s=(Hp, Wp), workers=-1)
    return X[pad:pad + H, pad:pad + W]


_SPEC_S = 2.0 ** np.arange(-1.0, 7.01, 0.5)     # texture spectrum basis: centre scales 0.5 ... 128 px
_SPEC_W = 0.5                                   # basis width (SD in octaves of frequency)


def _spec_basis(f: np.ndarray) -> np.ndarray:
    """(n_basis, len(f)): smooth bumps in log frequency, half an octave apart,
    centred on f_j = 1 / (2 pi s_j)."""
    lf = np.log2(np.maximum(f, 1e-9))[None, :]
    return np.exp(-0.5 * ((lf - np.log2(1 / (2 * math.pi * _SPEC_S))[:, None]) / _SPEC_W) ** 2)


def texture_spectrum(p: ImagingParams, blur: float = 0.8) -> np.ndarray:
    """Weights c_j > 0 of the texture's radial power spectrum
    P(f) = sum_j c_j phi_j(f) such that its DoG band variances, after a
    Gaussian blur `blur` px (the average's motion and registration blur),
    are (p.tex_rms * p.tex_scale)^2.  log c is kept smooth (its second
    differences are penalised), so the spectrum has no holes between bands
    and continues the measured trend linearly in log-log beyond them."""
    from scipy.optimize import least_squares
    Kb = np.exp(-4 * math.pi ** 2 * blur * blur * _FR ** 2)
    D = np.array([_dog_power_radial(_FR, s) for s in BANDS])
    M = np.einsum("kf,jf,f->kj", D, _spec_basis(_FR), 2 * math.pi * _FR * Kb) * _DFR
    t = (np.asarray(p.tex_rms, float) * p.tex_scale) ** 2
    x0 = np.full(len(_SPEC_S), math.log(t.mean() / M.sum(1).mean()))

    def res(x):
        return np.r_[np.log(M @ np.exp(x) / t), 0.3 * np.diff(x, 2)]

    return np.exp(least_squares(res, x0).x)


def texture_band_rms(p: ImagingParams, blur: float = 0.8) -> np.ndarray:
    """DoG band RMS (Np) of texture_spectrum after `blur`: what the spectrum
    achieves against p.tex_rms (the fit is not exact where the targets are
    not smooth)."""
    Kb = np.exp(-4 * math.pi ** 2 * blur * blur * _FR ** 2)
    D = np.array([_dog_power_radial(_FR, s) for s in BANDS])
    M = np.einsum("kf,jf,f->kj", D, _spec_basis(_FR), 2 * math.pi * _FR * Kb) * _DFR
    return np.sqrt(M @ texture_spectrum(p, blur))


_TEX_GROUPS = [(_SPEC_S < 1.9), (_SPEC_S >= 1.9) & (_SPEC_S < 3.9), (_SPEC_S >= 3.9) & (_SPEC_S < 7.9),
               (_SPEC_S >= 7.9) & (_SPEC_S < 16)]


def texture(shape, p: ImagingParams, rng: np.random.Generator | None = None, blur: float = 0.8,
            parts: dict | None = None) -> np.ndarray:
    """Scleral texture tau (Np, zero mean): an anisotropic Gaussian field with
    the measured band spectrum (texture_spectrum); its fine part (scales
    below ~16 px) in four groups, each multiplied by a smooth log-normal
    envelope exp(sigma g - sigma^2) of one shared field g (patchy: rough
    here, smooth there, heavier-tailed at fine scales, as the real
    residuals); plus sparse Gaussian lumps of both signs.  The envelope keeps
    the mean power (E[env^2] = 1).  Uses p.scene_seed when set, so the frames
    and the average of a scene share it.  `blur` is the Gaussian blur (px)
    the texture will undergo; the spectrum is set so the bands match after it.
    parts: texture_parts(...) to sum (e.g. after defocus_texture)."""
    parts = parts if parts is not None else texture_parts(shape, p, rng, blur)
    tau = sum(parts["fine"]) + parts["coarse"] + parts["lumps"]
    return (tau - tau.mean()).astype(np.float32)


def texture_parts(shape, p: ImagingParams, rng: np.random.Generator | None = None, blur: float = 0.8) -> dict:
    """The components of texture(): 'fine' (the four enveloped groups of
    scales below 16 px, a list), 'coarse' and 'lumps' (arrays, Np), and
    'spec' (texture_spectrum).  Same random draws as texture()."""
    if p.scene_seed is not None:
        rng = np.random.default_rng([p.scene_seed, 1])
    rng = rng if rng is not None else np.random.default_rng()
    c = texture_spectrum(p, blur)
    thg = math.radians(p.tex_dir + 90.0)      # gradients run across the fibres
    # the fine scales come in four groups, each under its own envelope,
    # weaker at larger scales (the real residuals' tails thin with scale)
    groups = _TEX_GROUPS
    fine_j = _SPEC_S < 16

    def power(sel, anis):
        a = min(2.0 * anis, 0.95)

        def P(fy, fx):
            f = np.sqrt(fx * fx + fy * fy)
            ang = 1.0 + a * np.cos(2 * (np.arctan2(fy, fx) - thg))
            lf = np.log2(np.maximum(f, 1e-9))
            s = 0.0
            for cj, sj in zip(c[sel], _SPEC_S[sel]):
                s = s + cj * np.exp(-0.5 * ((lf - math.log2(1 / (2 * math.pi * sj))) / _SPEC_W) ** 2)
            return s * ang
        return P

    coarse = _field(shape, power(~fine_j, p.tex_anis_coarse), rng, pad=400)
    g = _field(shape, lambda fy, fx: np.exp(-4 * math.pi ** 2 * p.tex_patch_px ** 2 * (fx * fx + fy * fy)), rng,
               pad=int(3 * p.tex_patch_px))
    g /= max(float(g.std()), 1e-12)
    fine = []
    for i, sel in enumerate(groups):
        if sel.any():
            sig = max(float(p.tex_patchy[i]), 0.0)
            fine.append(np.exp(sig * g - sig * sig) * _field(shape, power(sel, p.tex_anis), rng, pad=64))
        else:
            fine.append(np.zeros(shape))
    return dict(fine=fine, coarse=coarse, lumps=lumps(shape, p, rng), spec=c)


def _group_attenuation(c: np.ndarray, sel: np.ndarray, sig) -> np.ndarray:
    """Variance share of texture group `sel` (spectrum weights c) left after
    a Gaussian blur sig (px, array): integral of P_g(f) exp(-4 pi^2 sig^2 f^2)
    over the radial frequency, over the unblurred integral."""
    sig = np.atleast_1d(np.asarray(sig, float))
    B = _spec_basis(_FR)[sel]
    Pg = (np.asarray(c)[sel][:, None] * B).sum(0) * 2 * math.pi * _FR
    K = np.exp(-4 * math.pi ** 2 * sig[:, None] ** 2 * _FR[None, :] ** 2)
    return (K * Pg[None, :]).sum(1) / max(float(Pg.sum()), 1e-30)


def defocus_texture(parts: dict, var_map: np.ndarray, norm_var_map: np.ndarray | None = None,
                    max_gain: float = 1.5) -> np.ndarray:
    """The texture of an image whose depth of the textured tissue is out of
    focus by a spatially varying extra Gaussian blur of variance var_map
    (px^2, (H, W)): each fine group (scales < 16 px) is blurred by it
    (blur_field), rescaled so that under norm_var_map (default var_map: the
    image the band targets were measured on, the average) its variance over
    the frame is what it was unblurred (the measured band power is kept, it
    moves to the sharper regions), by at most max_gain in amplitude (the real
    stills the targets come from are partly soft too: a scene mostly out of
    focus at the texture's depth keeps a smoother texture, not a sandpaper
    patch where it is sharp); the lumps are blurred too (not rescaled),
    the coarse scales unchanged.  In soft regions the fine texture is
    smoother, as the vessels there are."""
    norm_var_map = var_map if norm_var_map is None else norm_var_map
    c = parts["spec"]
    fine = np.zeros_like(parts["coarse"], dtype=np.float64)
    v = np.sqrt(np.maximum(np.asarray(norm_var_map, float), 0.0)).ravel()[::17]
    qs = np.unique(np.quantile(v, np.linspace(0.0, 1.0, 33)))
    for sel, gpart in zip(_TEX_GROUPS, parts["fine"]):
        if not sel.any():
            continue
        # the mean variance share this group keeps under norm_var_map (analytic, from its spectrum)
        att = float(np.mean(np.interp(v, qs, _group_attenuation(c, sel, qs))))
        fine += gpart * min(1.0 / math.sqrt(max(att, 1e-6)), max_gain)
    tau = blur_field(fine + parts["lumps"], None, var_map) + parts["coarse"]
    return (tau - tau.mean()).astype(np.float32)


def blur_field(X: np.ndarray, cov=None, var_map=None, ratio: float = 1.25, s_first: float = 0.3,
               pad: int | None = None) -> np.ndarray:
    """X blurred by a Gaussian of covariance cov (px^2, 2 x 2; None: 0) plus
    an isotropic Gaussian whose variance var_map (px^2, (H, W), >= 0) varies
    over the image.  X is blurred at a ladder of levels of added variance
    (sigmas s_first, then `ratio` apart in total sigma, up to the largest
    var_map; the first level is 0) and every pixel interpolates linearly in
    variance between the two levels around its var_map: a two-Gaussian
    mixture with the right second moment.  One padded FFT, one inverse per
    level (reflecting borders)."""
    X = np.asarray(X, np.float64)
    H, W = X.shape
    cov = np.zeros((2, 2)) if cov is None else np.asarray(cov, float)
    vm = np.zeros((H, W)) if var_map is None else np.maximum(np.asarray(var_map, np.float64), 0.0)
    vmax = float(vm.max())
    base = max(0.5 * float(np.trace(cov)), 0.0)
    s_tot = math.sqrt(base + vmax)
    p = int(min(pad if pad is not None else math.ceil(4 * s_tot + 8 + 3 * math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0))),
                H - 1, W - 1))
    Xp = np.pad(X, p, mode="reflect")
    Hp, Wp = Xp.shape
    fy = sfft.fftfreq(Hp)[:, None]
    fx = sfft.rfftfreq(Wp)[None, :]
    Xf = sfft.rfft2(Xp, workers=-1)
    q = fx * fx + fy * fy
    Kc = np.exp(-2 * math.pi ** 2 * (cov[0, 0] * fx * fx + 2 * cov[0, 1] * fx * fy + cov[1, 1] * fy * fy))

    def level(v):
        return sfft.irfft2(Xf * Kc * np.exp(-2 * math.pi ** 2 * v * q), s=(Hp, Wp), workers=-1)[p:p + H, p:p + W]
    if vmax <= 1e-9:
        return level(0.0)
    # ladder of added variances: 0, then total sigmas from max(sqrt(base + s_first^2), ...) by `ratio`
    lv = [0.0]
    st_ = math.sqrt(base + s_first ** 2)
    while True:
        v = st_ * st_ - base
        if v >= vmax - 1e-9:
            lv.append(vmax)
            break
        lv.append(v)
        st_ *= ratio
    lv = np.array(sorted(set(lv)))
    out = np.zeros((H, W))
    for j, v in enumerate(lv):
        lo = lv[j - 1] if j > 0 else None
        hi = lv[j + 1] if j + 1 < len(lv) else None
        w = np.zeros((H, W))
        if lo is not None:
            m = (vm > lo) & (vm <= v)
            w[m] = (vm[m] - lo) / (v - lo)
        if hi is not None:
            m = (vm > v) & (vm < hi)
            w[m] = (hi - vm[m]) / (hi - v)
        w[vm == v] = 1.0
        if j == 0:
            w[vm <= 0.0] = 1.0
        if w.any():
            out += w * level(v)
    return out


def texture_through_vessels(od, f: float) -> np.ndarray:
    """Share (0..1) of the texture's log amplitude left where vessels of
    optical density od (Np) lie over the textured tissue, when a share f of
    the detected light never crossed the blood (tex_bypass):
    (1 - f e^od) / (1 - f), clipped to [0, 1]."""
    od = np.asarray(od, np.float64)
    if f <= 0:
        return np.ones_like(od)
    f = min(float(f), 0.99)
    return np.clip((1.0 - f * np.exp(np.minimum(od, 20.0))) / (1.0 - f), 0.0, 1.0)


def registration_field(shape, p: ImagingParams, rng: np.random.Generator) -> np.ndarray:
    """SD (px) of the residual registration error over the field: a smooth
    log-normal field (median p.reg_error_px, log-SD p.reg_field_cv,
    correlation length p.reg_field_px, at most p.reg_field_max); uniform
    when reg_field_cv = 0."""
    if p.reg_field_cv <= 0:
        return np.full(shape, float(p.reg_error_px))
    ell = max(float(p.reg_field_px), 4.0)
    step = max(int(ell // 8), 1)
    small = (int(math.ceil(shape[0] / step)) + 1, int(math.ceil(shape[1] / step)) + 1)
    g = _field(small, lambda fy, fx: np.exp(-4 * math.pi ** 2 * (ell / step) ** 2 * (fx * fx + fy * fy)), rng,
               pad=int(3 * ell / step))
    g = (g - g.mean()) / max(float(g.std()), 1e-12)
    yy = np.arange(shape[0]) / step
    xx = np.arange(shape[1]) / step
    g = ndi.map_coordinates(g, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")
    return np.minimum(float(p.reg_error_px) * np.exp(p.reg_field_cv * g), max(float(p.reg_field_max), p.reg_error_px))


def lumps(shape, p: ImagingParams, rng: np.random.Generator) -> np.ndarray:
    """Sparse Gaussian blobs, dark or bright, heavy-tailed (log-normal)
    amplitudes and log-uniform radii: pigment, cysts, thin spots of tissue,
    things that look vessel-like at a glance but are not."""
    H, W = shape
    out = np.zeros(shape)
    n = rng.poisson(p.lump_density * H * W / 1e5)
    if n == 0 or p.lump_amp <= 0:
        return out
    r = np.exp(rng.uniform(math.log(p.lump_r[0]), math.log(p.lump_r[1]), n))
    amp = p.lump_amp * np.exp(rng.normal(0, 0.7, n)) * np.where(rng.random(n) < p.lump_dark, -1.0, 1.0)
    cx, cy = rng.uniform(-10, W + 10, n), rng.uniform(-10, H + 10, n)
    ecc = np.exp(rng.normal(0, 0.3, n))
    ang = rng.uniform(0, math.pi, n)
    for i in range(n):
        R = int(math.ceil(3.5 * r[i] * max(ecc[i], 1 / ecc[i])))
        x0, x1 = max(int(cx[i]) - R, 0), min(int(cx[i]) + R + 1, W)
        y0, y1 = max(int(cy[i]) - R, 0), min(int(cy[i]) + R + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - cx[i], yy - cy[i]
        c, s = math.cos(ang[i]), math.sin(ang[i])
        q = ((c * dx + s * dy) / (r[i] * ecc[i])) ** 2 + ((-s * dx + c * dy) * ecc[i] / r[i]) ** 2
        out[y0:y1, x0:x1] += amp[i] * np.exp(-0.5 * q)
    return out


# ======================================================================== burst: eye motion, coverage, camera
def trajectory(n: int, p: ImagingParams, rng: np.random.Generator) -> dict:
    """Eye positions of n frames (px, content displacement as transforms.csv
    dx, dy) and each frame's motion during its exposure (px vector).
    Fixations separated by saccades (Poisson, rate p.saccade_rate); within a
    fixation an Ornstein-Uhlenbeck drift with SD p.drift_px whose median
    frame-to-frame step is p.speed_px_s / fps.  Frames caught in a saccade or
    blurred beyond p.max_blur_px are dropped, as the pipeline's blur gate
    does.  Positions are centred on their median, as the pipeline's are (so
    the frame's edges keep at least half the frames and the corners do not)."""
    n = max(int(n), 1)
    m = n + 1
    step = p.speed_px_s / p.fps / 1.1774           # per-axis SD of a step: median |step| = speed / fps
    drift = max(p.drift_px, 1e-3)
    rho = 1.0 - step ** 2 / (2 * drift ** 2)
    if rho > 0:
        innov = math.sqrt(max(1 - rho * rho, 0.0)) * drift
        ou = np.zeros((m, 2))
        ou[0] = rng.normal(0, drift, 2)
        e = rng.normal(0, innov, (m, 2))
        for i in range(1, m):
            ou[i] = rho * ou[i - 1] + e[i]
    else:                # steps too large for the drift SD: a plain random walk
        ou = np.cumsum(rng.normal(0, step, (m, 2)), 0)
    n_sac = int(rng.poisson(p.saccade_rate * m / p.fps))
    cuts = np.sort(rng.integers(1, m, n_sac)) if n_sac else np.zeros(0, int)
    seg = np.searchsorted(cuts, np.arange(m), side="right")
    centres = rng.normal(0, p.saccade_px, (n_sac + 1, 2))
    pos = ou + centres[seg]
    vel = np.diff(pos, axis=0)                       # per frame interval
    blur = vel * min(p.exposure_ms * 1e-3 * p.fps, 1.0)
    keep = np.ones(n, bool)
    keep[np.minimum(cuts, n) - 1] = False            # the frame spanning a saccade
    keep &= np.hypot(blur[:, 0], blur[:, 1]) <= p.max_blur_px
    if not keep.any():
        keep[0] = True
    d = pos[:n][keep]
    d = d - np.median(d, 0)          # the pipeline's reference: median dx = dy = 0
    return dict(d=d, blur=blur[keep], n_total=n, n_used=int(keep.sum()), saccades=n_sac)


def _bilinear_hist(d: np.ndarray, weights=None):
    """Histogram of positions with bilinear weights on the integer grid, so
    that sum_i w_i Q_bilinear(x + d_i) = sum_o h[o] Q(x + o) exactly."""
    R = int(math.ceil(np.abs(d).max())) + 2 if len(d) else 2
    h = np.zeros((2 * R + 1, 2 * R + 1))
    fx, fy = np.floor(d[:, 0]), np.floor(d[:, 1])
    ax, ay = d[:, 0] - fx, d[:, 1] - fy
    ix, iy = fx.astype(int) + R, fy.astype(int) + R
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    for wy, oy in ((1 - ay, 0), (ay, 1)):
        for wx, ox in ((1 - ax, 0), (ax, 1)):
            np.add.at(h, (iy + oy, ix + ox), w * wy * wx)
    return h, R


def traj_sum(Q: np.ndarray, h: np.ndarray, R: int) -> np.ndarray:
    """sum_i Q(x + d_i) over the frames (Q zero outside the camera frame):
    the stabilized sum of a camera-fixed image, as one FFT correlation."""
    from scipy.signal import fftconvolve
    return fftconvolve(np.pad(Q.astype(np.float64), R), h[::-1, ::-1], mode="valid")


def glare_spots(shape, p: ImagingParams, rng: np.random.Generator) -> list:
    """Specular reflections of the lamp, fixed in the camera frame: centre,
    core semi-axes and angle.  Uses p.camera_seed when set."""
    if p.camera_seed is not None:
        rng = np.random.default_rng([p.camera_seed, 2])
    H, W = shape
    n = int(rng.poisson(p.glare_rate * H * W / (1920 * 1200)))
    out = []
    for _ in range(n):
        a = p.glare_px * math.exp(rng.normal(0, 0.3))
        asp = rng.uniform(1.0, max(p.glare_aspect, 1.0))
        out.append(dict(x=float(rng.uniform(0, W)), y=float(rng.uniform(0, H)), a=a * asp, b=a,
                        theta=float(rng.uniform(0, math.pi))))
    return out


def glare_maps(shape, spots, p: ImagingParams):
    """(core, mask, bloom) in the camera frame: the clipped core, the
    pipeline's glare mask (the core dilated by a disk of p.glare_dilate px)
    and the bloom (fraction of the level) that leaks past the mask."""
    core = np.zeros(shape, bool)
    if not spots:
        return core, core.copy(), np.zeros(shape)
    H, W = shape
    dist = np.full(shape, np.inf)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in spots:
        c, sn = math.cos(s["theta"]), math.sin(s["theta"])
        dx, dy = xx - s["x"], yy - s["y"]
        r = np.sqrt(((c * dx + sn * dy) / s["a"]) ** 2 + ((-sn * dx + c * dy) / s["b"]) ** 2)
        core |= r <= 1
        dist = np.minimum(dist, np.maximum(r - 1, 0) * s["b"])
    mask = ndi.binary_dilation(core, structure=_disk(p.glare_dilate // 2)) if core.any() else core.copy()
    bloom = p.bloom_amp * np.exp(-dist / max(p.bloom_px, 1e-3))
    return core, mask, np.where(core, 0.0, bloom)


def camera_fixed_pattern(shape, p: ImagingParams, rng: np.random.Generator) -> dict:
    """The camera's fixed pattern: per-pixel relative gain (PRNU-like, with
    dark dust specks on the sensor), per-row and per-column offsets (DN).
    Uses p.camera_seed when set."""
    if p.camera_seed is not None:
        rng = np.random.default_rng([p.camera_seed, 3])
    H, W = shape
    pixel = rng.standard_normal(shape) * p.fpn_pixel
    n = int(rng.poisson(p.dust_rate * H * W / (1920 * 1200)))
    if n and p.dust_depth > 0:
        spots = np.zeros(shape)
        ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
        np.add.at(spots, (ys, xs), p.dust_depth * np.exp(rng.normal(0, 0.5, n)))
        pixel -= ndi.gaussian_filter(spots, p.dust_px) * 2 * math.pi * p.dust_px ** 2
    return dict(pixel=pixel, row=rng.standard_normal(H) * p.fpn_row_dn, col=rng.standard_normal(W) * p.fpn_col_dn)


def frame_noise_sd(dn, p: ImagingParams):
    """Per-frame temporal noise SD (output DN) at output level dn: Poisson and
    read noise in linear DN at the gain, carried through the gamma curve."""
    g_e = p.e_per_dn0 / 10 ** (p.gain_db / 20)
    dn = np.clip(dn, 1.0, FULL_SCALE)
    x = FULL_SCALE * (dn / FULL_SCALE) ** (1 / p.gamma)
    return p.gamma * dn / x * np.sqrt(x / g_e + (p.read_e / g_e) ** 2)


def _blur_gauss(X, cov, pad=32):
    """Gaussian blur with covariance cov (px^2, [[xx, xy], [xy, yy]])."""
    cxx, cxy, cyy = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
    return _filter_padded(X, lambda fy, fx: np.exp(-2 * math.pi ** 2 * (cxx * fx * fx + 2 * cxy * fx * fy
                                                                        + cyy * fy * fy)), pad)


def _blur_line(X, v, pad=32):
    """Blur by uniform motion along the vector v (px): a box of length |v|."""
    return _filter_padded(X, lambda fy, fx: np.sinc(v[0] * fx + v[1] * fy), pad)


def _two_tap(z: np.ndarray) -> np.ndarray:
    """White noise through the average bilinear-interpolation kernel: per
    axis the 2-tap filter whose power spectrum is 2/3 + cos(w)/3, the mean
    over uniform sub-pixel shifts (variance 4/9, lag-1 correlation 1/4)."""
    a, b = (1 + 1 / math.sqrt(3)) / 2, (1 - 1 / math.sqrt(3)) / 2
    z = a * z + b * np.roll(z, 1, axis=0)
    return a * z + b * np.roll(z, 1, axis=1)


def average_blur(tr: dict, p: ImagingParams) -> np.ndarray:
    """Covariance (px^2) of the blur an average adds to the scene: the mean
    motion blur of its frames (a box of length |v| has variance |v|^2 / 12
    along v), the bilinear interpolation (1/6 per axis, the mean of f(1 - f)
    over uniform sub-pixel shifts) and the registration error."""
    b = tr["blur"]
    cov = (b[:, :, None] * b[:, None, :]).mean(0) / 12.0 if len(b) else np.zeros((2, 2))
    return cov + (1 / 6 + p.reg_error_px ** 2) * np.eye(2)


# ======================================================================== forming the image
def form_image(od, params: ImagingParams | None = None, rng: np.random.Generator | None = None,
               kind: str = "average", n_frames: int | None = None, tau=None, lnE=None) -> dict:
    """A still (kind='average': the stabilized mean of n_frames frames, like
    mean_stabilized.tif) or one raw frame (kind='frame') of a scene whose
    vessels have optical density od (H x W, Np; numpy or torch).

    Returns float32 images in DN:
      image       what the camera/pipeline delivers (NaN where no data)
      clean       the same without random noise (blur, glare bloom included)
      background  clean with od = 0 (texture and illumination only)
      noise       image - clean (temporal noise plus the camera pattern)
    and valid (bool), cover (frames per pixel), tau and lnE (Np; pass them
    back in to form another image of the same scene), traj, blur_cov, the
    frames used and the parameters.

    A frame is at the registration reference (no offset) with its own motion
    blur, full per-frame noise, the camera pattern unaveraged, quantised to
    12 bits, glare saturated (not NaN): what one raw frame looks like.  Pass
    the od rendered with that frame's filling (filling(..., kind='frame'))."""
    p = params if params is not None else ImagingParams()
    rng = rng if rng is not None else np.random.default_rng()
    if hasattr(od, "detach"):
        od = od.detach().cpu().numpy()
    od = np.asarray(od, np.float64)
    shape = od.shape
    H, W = shape
    if lnE is None:
        lnE = illumination(shape, p)
    if tau is None:
        tau = texture(shape, p, rng)
    lnE = np.asarray(lnE, np.float64)
    tau = np.asarray(tau, np.float64)
    bg0 = p.level * np.exp(lnE + tau)                # scene radiance in DN, no vessels
    sc0 = bg0 * np.exp(-od) if p.tex_bypass <= 0 else \
        p.level * np.exp(lnE + tau * texture_through_vessels(od, p.tex_bypass) - od)
    spots = glare_spots(shape, p, rng)
    core, gmask, bloom = glare_maps(shape, spots, p)
    fpn = camera_fixed_pattern(shape, p, rng)
    out = dict(kind=kind, tau=tau.astype(np.float32), lnE=lnE.astype(np.float32), params=p.to_dict(), glare=spots)
    if kind == "frame":
        tr = trajectory(2, p, rng)
        v = tr["blur"][0]
        clean = _blur_line(sc0, v) + bloom * p.level
        flick = 1.0 + rng.normal(0, p.flicker)
        back = _blur_line(bg0, v) + bloom * p.level
        clean = np.where(core, 1.3 * FULL_SCALE, clean)
        back = np.where(core, 1.3 * FULL_SCALE, back)
        # per-frame photon and read noise in linear DN, then the gamma curve
        g_e = p.e_per_dn0 / 10 ** (p.gain_db / 20)
        dn = np.clip(clean * flick, 0.0, 2 * FULL_SCALE)
        x = FULL_SCALE * (dn / FULL_SCALE) ** (1 / p.gamma)
        e = rng.poisson(np.minimum(x * g_e, 1e9)).astype(np.float64) + rng.normal(0, p.read_e, shape)
        y = FULL_SCALE * (np.maximum(e / g_e, 0.0) / FULL_SCALE) ** p.gamma
        y = y * (1 + fpn["pixel"]) + fpn["row"][:, None] + fpn["col"][None, :]
        y += rng.normal(0, p.row_noise_dn, H)[:, None] + rng.normal(0, p.col_noise_dn, W)[None, :]
        img = np.clip(np.round(y), 0, FULL_SCALE)
        clean_c = np.minimum(clean, FULL_SCALE)
        out.update(image=img.astype(np.float32), clean=clean_c.astype(np.float32),
                   background=np.minimum(back, FULL_SCALE).astype(np.float32),
                   noise=(img - clean_c).astype(np.float32), valid=np.ones(shape, bool),
                   cover=np.ones(shape, np.float32), traj=np.zeros((1, 2)), blur_cov=np.outer(v, v) / 12.0,
                   frame_blur=v, n_frames=1)
        return out
    if kind != "average":
        raise ValueError(f"kind must be 'average' or 'frame', not {kind!r}")
    n = int(n_frames or p.n_frames)
    tr = trajectory(n, p, rng)
    d = tr["d"]
    N = tr["n_used"]
    h, R = _bilinear_hist(d)
    seen = (~gmask).astype(np.float64)
    cover = traj_sum(seen, h, R)
    valid = cover >= p.cover_min * N - 1e-6
    cov_safe = np.maximum(cover, 1e-6)
    cov = average_blur(tr, p)
    reg_sd = None
    if p.reg_field_cv > 0:
        # the registration residual varies over the field: its variance replaces the uniform reg_error_px^2
        reg_sd = registration_field(shape, p, rng)
        v = reg_sd ** 2
        vmin = float(v.min())
        cov0 = cov + (vmin - p.reg_error_px ** 2) * np.eye(2)
        clean = blur_field(sc0, cov0, v - vmin)
        back = blur_field(bg0, cov0, v - vmin)
        cov = cov + (float(v.mean()) - p.reg_error_px ** 2) * np.eye(2)       # reported: the mean blur
    else:
        clean = _blur_gauss(sc0, cov)
        back = _blur_gauss(bg0, cov)
    # camera-fixed light (glare bloom) seen through the moving eye: smeared
    if spots:
        smear = traj_sum(bloom * seen, h, R) / cov_safe * p.level
        clean = clean + smear
        back = back + smear
    # camera-fixed pattern averaged over the eye positions
    pat = clean * traj_sum(fpn["pixel"] * seen, h, R) / cov_safe
    # lamp flicker: frames differ in brightness; where fewer frames cover a
    # pixel (borders, glare) their mean brightness differs from the rest
    b = rng.normal(0, p.flicker, len(d))
    hb, _ = _bilinear_hist(d, b)
    pat += clean * traj_sum(seen, hb, R) / cov_safe
    pat += traj_sum(np.broadcast_to(fpn["row"][:, None], shape) * seen, h, R) / cov_safe
    pat += traj_sum(np.broadcast_to(fpn["col"][None, :], shape) * seen, h, R) / cov_safe
    # temporal noise: per-frame noise through the average bilinear kernel, / sqrt(frames seen)
    z = rng.standard_normal(shape)
    if p.noise_sigma_c > 0:
        z = ndi.gaussian_filter(z, p.noise_sigma_c)
        z /= max(float(z.std()), 1e-12)
    temporal = _two_tap(z) * frame_noise_sd(clean, p) / np.sqrt(cov_safe)
    temporal += (rng.normal(0, p.row_noise_dn, H)[:, None] + rng.normal(0, p.col_noise_dn, W)[None, :]) / math.sqrt(N)
    noise = temporal + pat
    img = np.where(valid, clean + noise, np.nan)
    out.update(image=img.astype(np.float32), clean=clean.astype(np.float32), background=back.astype(np.float32),
               noise=np.where(valid, noise, np.nan).astype(np.float32), valid=valid,
               cover=cover.astype(np.float32), traj=d, blur_cov=cov, n_frames=N, saccades=tr["saccades"])
    if reg_sd is not None:
        out["reg_sd"] = reg_sd.astype(np.float32)
    return out


# ======================================================================== red-cell granularity (single frames)
N0_CELLS = 0.45 / 0.42      # red cells per d_c^2 column per d_c of systemic blood: haematocrit 0.45, a cell of 90 fL
                            # = 0.42 d_c^3 (d_c = 6 um)


def _splat(x, y, w, H, W, out: np.ndarray | None = None, layer=None) -> np.ndarray:
    """Bilinear splat of weights w at points (x, y) (px), added into `out`
    (H x W, or n_layers x H x W with point i going to layer[i]; a new H x W
    array when None).  Returns out."""
    if out is None:
        out = np.zeros((H, W))
    xf, yf = np.floor(x), np.floor(y)
    ax, ay = x - xf, y - yf
    xb, yb = xf.astype(np.int64), yf.astype(np.int64)
    base = 0 if layer is None else np.asarray(layer, np.int64) * (H * W)
    flat = out.reshape(-1)
    for oy, wy in ((0, 1 - ay), (1, ay)):
        for ox, wx in ((0, 1 - ax), (1, ax)):
            xx, yy = xb + ox, yb + oy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            np.add.at(flat, (base + yy * W + xx)[ok], (w * wy * wx)[ok])
    return out


def _lumen_samples(T: dict, step: float, chunk: int = 400000):
    """Samples covering the lumen footprints of tube pieces (T: arrays cx, cy,
    tx, ty, l, r, a): yields (piece index, x, y, chord absorbance c, sample
    area) in chunks of about `chunk` samples, a grid of about step px."""
    nt = np.maximum(np.ceil(2 * T["r"] / step).astype(int), 1)
    nu = np.maximum(np.ceil(T["l"] / step).astype(int), 1)
    cnt = nt * nu
    starts = np.r_[0, np.cumsum(cnt)]
    i0 = 0
    while i0 < len(cnt):
        i1 = int(np.searchsorted(starts, starts[i0] + chunk, side="right"))
        i1 = min(max(i1 - 1, i0 + 1), len(cnt))
        pid = np.repeat(np.arange(i0, i1), cnt[i0:i1])
        j = np.arange(len(pid)) - np.repeat(starts[i0:i1] - starts[i0], cnt[i0:i1])
        it, iu = j % nt[pid], j // nt[pid]
        dt, du = 2 * T["r"][pid] / nt[pid], T["l"][pid] / nu[pid]
        t = -T["r"][pid] + (it + 0.5) * dt
        u = -0.5 * T["l"][pid] + (iu + 0.5) * du
        x = T["cx"][pid] + u * T["tx"][pid] - t * T["ty"][pid]
        y = T["cy"][pid] + u * T["ty"][pid] + t * T["tx"][pid]
        c = T["a"][pid] * np.sqrt(np.maximum(1.0 - (t / T["r"][pid]) ** 2, 0.0))
        yield pid, x, y, c, dt * du
        i0 = i1


MOTTLE_OFFSET = 128     # px: range of the per-vessel offsets into the cells' field (rbc_mottle with vessel ids)


def rbc_mottle(shape, tubes: dict, rng: np.random.Generator, amp: float, k: float, mu: float,
               speed_dc_s=None, exposure_s: float = 0.0128, r_min_px: float = 0.0, step: float = 0.5,
               ratio: float = 1.25, vessel=None) -> np.ndarray:
    """The red cells' granularity inside the lumens in one frame: delta OD
    (Np, `shape`), zero mean, to add to the frame's rendered OD (before its
    halo and motion blur).

    A lumen holds discrete cells: the column of blood above a pixel holds
    n = N0_CELLS (a / mu) cells per d_c^2 (a: the chord's absorbance under
    the linear law, a / mu its length in d_c of systemic blood), and n
    fluctuates from column to column by amp sqrt(n) (amp^2: the structure
    factor, 1 for independent cells, below 1 for packed ones, above for
    aggregates), over columns of about one cell (a field of correlation
    area d_c^2).  Through the chord law OD = -ln(f + (1 - f) e^-C) that is
        dOD = amp sqrt(mu c / N0_CELLS) (1 - f) e^-C / (f + (1 - f) e^-C) xi,
    c the vessel's own chord absorbance at the point, C the whole column's
    (every vessel over the pixel, capillaries included: where vessels cross,
    the column is darker and each one's cells change its light less; f the
    shallowest vessel's unabsorbed share there), xi a unit field: the
    Poisson-like granularity grows as sqrt(c) and saturates where the column
    is black.  During the exposure the cells move v x exposure along the
    vessel, which averages 1 + v exposure / d_c columns (speed_dc_s: per
    piece, d_c/s; None: no smear).  Each piece's share is blurred by its own
    blur s (px; pieces grouped into blurs `ratio` apart), so a blurred
    vessel's granularity is blurred, and mostly averaged away, with it.

    tubes: the render plan's tube pieces (px, the frame's grid): cx, cy, tx,
    ty (centre, unit tangent), l (length), r (radius), s (blur), a (centre
    chord absorbance, Np), f (unabsorbed share).  Pieces with r < r_min_px
    get no granularity (capillaries: their single cells are the filling's
    columns) but count in the column.  The average of a burst is the mean of
    such frames: its own granularity is 1 / sqrt(n_frames) of this,
    negligible, and not drawn.

    vessel: per piece, its vessel id.  The cells of different vessels are
    independent: each vessel reads the unit field at its own random offset
    (MOTTLE_OFFSET px range, from (one draw of rng, vessel id)), evaluated
    at each lumen sample.  None: one field for every vessel (v2: the grain
    of two crossing vessels correlated at 0.97, correctness review v2,
    defect 4)."""
    H, W = int(shape[0]), int(shape[1])
    out = np.zeros((H, W))
    if amp <= 0 or tubes is None or not len(tubes.get("l", ())):
        return out.astype(np.float32)
    T = {kk: np.asarray(tubes[kk], np.float64) for kk in ("cx", "cy", "tx", "ty", "l", "r", "s", "a", "f")}
    near = (T["a"] > 0) & (T["l"] > 0)
    near &= (T["cx"] > -T["r"] - 4 * T["s"]) & (T["cx"] < W + T["r"] + 4 * T["s"])
    near &= (T["cy"] > -T["r"] - 4 * T["s"]) & (T["cy"] < H + T["r"] + 4 * T["s"])
    sel = near & (T["r"] >= r_min_px)
    if not sel.any():
        return out.astype(np.float32)
    # pass 1: the whole column's absorbance (sharp lumens, every vessel) and the shallowest unabsorbed share
    Ta = {kk: v[near] for kk, v in T.items()}
    Ctot = np.zeros((H, W))
    fmin = np.ones((H, W))
    for pid, x, y, c, area in _lumen_samples(Ta, step):
        xi, yi = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.minimum.at(fmin.ravel(), (yi * W + xi)[ok], np.clip(Ta["f"][pid], 0.0, 0.999)[ok])
        _splat(x, y, c * area, H, W, out=Ctot)                  # bilinear: the sample grid does not alias
    Ctot = ndi.gaussian_filter(Ctot, 0.5)
    # pass 2: each continuous vessel's granularity, in blur classes
    T = {kk: v[sel] for kk, v in T.items()}
    sm = np.ones(int(sel.sum()))
    if speed_dc_s is not None:
        v = np.asarray(speed_dc_s, np.float64)[sel]
        sm = 1.0 / np.sqrt(1.0 + np.maximum(v, 0.0) * exposure_s)          # columns of one d_c averaged by the smear
    smin = max(float(T["s"].min()), 0.3)
    cls = np.rint(np.log(np.maximum(T["s"], smin) / smin) / math.log(ratio)).astype(int)
    used = np.unique(cls)
    layer_of = np.full(int(cls.max()) + 1, -1)
    layer_of[used] = np.arange(len(used))
    maps = np.zeros((len(used), H, W))
    # the cells: a unit field of correlation area d_c^2 (Gaussian, sigma = k / sqrt(4 pi))
    sc = float(k) / math.sqrt(4 * math.pi)
    if vessel is None:
        xi_ = ndi.gaussian_filter(rng.standard_normal((H, W)), sc, mode="reflect") * math.sqrt(4 * math.pi) * sc
    else:
        M = MOTTLE_OFFSET
        xi_ = ndi.gaussian_filter(rng.standard_normal((H + M + 2, W + M + 2)), sc, mode="reflect") * \
            math.sqrt(4 * math.pi) * sc
        base = int(rng.integers(2 ** 31))
        vid = np.asarray(vessel, np.int64)[sel]
        offs = {int(u): np.random.default_rng([base, int(u) & 0x7FFFFFFF]).integers(0, M + 1, 2) for u in np.unique(vid)}
        ox = np.array([offs[int(u)][0] for u in vid], float)
        oy = np.array([offs[int(u)][1] for u in vid], float)
    for pid, x, y, c, area in _lumen_samples(T, step):
        xn, yn = np.clip(np.floor(x + 0.5).astype(int), 0, W - 1), np.clip(np.floor(y + 0.5).astype(int), 0, H - 1)
        C = np.maximum(Ctot[yn, xn], c)
        f = fmin[yn, xn]
        e = np.exp(-C)
        w = math.sqrt(mu / N0_CELLS) * np.sqrt(c) * (1 - f) * e / (f + (1 - f) * e) * sm[pid] * area
        if vessel is not None:                  # each vessel's own cells: its field at its offset, at the sample
            xs = np.clip(x + ox[pid], 0.0, W + M)
            ys = np.clip(y + oy[pid], 0.0, H + M)
            w = w * ndi.map_coordinates(xi_, [ys, xs], order=1, mode="nearest")
        _splat(x, y, w, H, W, out=maps, layer=layer_of[cls[pid]])
    for li, kc in enumerate(used):
        out += _gauss_fft(maps[li] * xi_ if vessel is None else maps[li], smin * ratio ** kc)
    return (amp * out).astype(np.float32)


# ======================================================================== red-cell filling
class Filling:
    """A vessel's blood filling along its arclength: callable(s_dc) ->
    multiplier of its haematocrit (s_dc from the upstream node, d_c),
    linear between samples 0.1 d_c apart, constant beyond the ends.
    Attributes: phi (fraction of the lumen length holding red cells while
    it flows), mean (mean of the multiplier), state ('flowing', 'stop-go',
    'stagnant', 'unperfused', or 'empty' for a frame caught empty), speed
    (the red cells' speed along the vessel, d_c/s; None when not simulated)."""

    def __init__(self, s: np.ndarray, m: np.ndarray, phi: float, state: str, smooth: bool = False,
                 speed: float | None = None):
        self.s = np.asarray(s, float)
        self.m = np.asarray(m, float)
        self.phi = float(phi)
        self.state = state
        self.smooth = bool(smooth)          # no steps (the renderer may sample it coarsely)
        self.speed = None if speed is None else float(speed)
        self.mean = float(self.m.mean()) if self.m.size else 0.0

    def __call__(self, s_dc):
        return np.interp(np.asarray(s_dc, float), self.s, self.m)

    def __repr__(self):
        return f"Filling({self.state}, phi={self.phi:.2f}, mean={self.mean:.2f}, L={self.s[-1]:.1f} d_c)"


FILL_DEFAULTS = dict(
    n_frames=138, fps=32.2,
    exposure_ms=12.8,     # a frame integrates the moving columns over its exposure (smear v * exposure)
    v_cap=100.0,          # d_c/s, red-cell speed in a capillary (~0.6 mm/s at 6 um; conjunctiva 0.3-1.2 mm/s)
    v_max=3000.0,         # d_c/s, cap on any vessel's red-cell speed (~18 mm/s: far above the conjunctiva's few mm/s;
                          # keeps an anatomy with an extreme flow ratio from asking for astronomically long patterns)
    speed_exp=1.0,        # red-cell speed v_cap x (u / u_cap)^speed_exp, u = flow / lumen area from the solved flow, u_cap
                          # the capillaries' median (1: proportional to u, v1).  The conjunctiva's red cells move at about
                          # the same speed in every calibre: axial velocity 0.53 +- 0.15 mm/s over vessels of 21 +- 8 um
                          # (Brennan 2021), 0.51 mm/s in capillaries and the same in 14-24 um post-capillary venules
                          # (Moka & Koutsiaris 2019), venules 0.51 +- 0.17 mm/s (Hwang 2021)
    speed_sd=0.0,         # log-normal spread of a vessel's red-cell speed about that (SD of ln v; 0: none)
    cell_len=1.4,         # d_c, length of one deformed red cell in a capillary (~8 um)
    train_cells=3.0,      # mean number of cells in a column (cells, then a plasma gap)
    phi_cap=0.45,         # fraction of a capillary's length holding cells while it flows
    phi_sd=0.12,          # its spread between capillaries
    r_single=0.6,         # d_c: at or below this radius cells pass in single file
    r_cont=1.6,           # d_c: from this radius the column is continuous (no gaps)
    p_unperfused=0.08,    # capillaries never perfused during the burst (empty in every frame)
    p_stopgo=0.25,        # capillaries with stop-and-go flow ...
    stop_frac=0.35,       # ... stopped this fraction of the time, in episodes of
    stop_s=0.4,           # ... this mean length (s)
    p_empty=0.15,         # capillaries with plasma-only episodes (skimming) ...
    empty_frac=0.3,       # ... this fraction of the time, in episodes of
    empty_s=0.25,         # ... this mean length (s)
    edge=0.15,            # d_c, softness of a column's ends (rounded cells)
    relative=False,       # True: divide by phi, so a flowing vessel's time mean is 1
    h_col_max=2.0,        # relative haematocrit of a packed red-cell column (absolute ~0.9 at systemic 0.45): with
                          # relative, a column is h / phi, so phi is raised to at least h / h_col_max
    grain_amp=0.0,        # wider (continuous) vessels: red-cell aggregates make the blood's density fluctuate along
                          # the vessel by this relative SD, moving with the flow (0: off)
    grain_len=2.0,        # d_c, correlation length of those fluctuations
    grain_r_exp=0.0,      # the aggregates' relative SD grows with the vessel's radius: grain_amp x (r / r_cont) ** this
                          # (red cells aggregate more at the lower shear rates of wider venules; 0: the same everywhere)
    grain_amp_max=1.0,    # ... at most this
    grain_r_agg=float("inf"),   # d_c: beyond this radius the chord through the lumen crosses about 2 r / grain_len
                          # independent aggregates, which average: the relative SD falls as sqrt(grain_r_agg / r)
                          # (v1 fix, realism review T8: grown with r up to 0.35, it drew full-width plasma bands,
                          # 'bamboo', across wide veins in frames)
    grain_r_pow=0.5,      # ... as (grain_r_agg / r) ** grain_r_pow: 0.5 counts the aggregates along the chord only; 1 also
                          # those side by side across the lumen, which a pattern the same across the lumen cannot show
    gap_fill=0.0,         # vessels wider than one cell (radius r_single .. r_cont): a 'plasma gap' does not empty the
                          # whole lumen (cells pass two abreast, a missing cell leaves the other side); the gap keeps
                          # gap_fill x sqrt(t) of the column's blood, t = 0 at r_single .. 1 at r_cont (0: v0, empty)
    grain_kind="gauss",   # the aggregates' pattern along a continuous vessel: 'gauss' (v1: smooth Gaussian noise of
                          # correlation grain_len) or 'clump': clumps of red cells (mean length grain_len) separated by
                          # plasma-rich stretches, a fraction grain_phi of the length in clumps, lengths gamma-distributed
                          # (shape grain_shape), soft ends (grain_edge, d_c): the beads of a raw frame
    grain_phi=0.5,
    grain_shape=2.0,
    grain_edge=0.3,
    col_shape=1.0,        # capillaries: shape of the gamma distribution of the column and gap lengths (1: exponential,
                          # v1; larger: more regular spacing of the cells)
    mottle=0.0,           # single frames: the red cells' granularity inside the lumens of the continuous vessels
                          # (rbc_mottle's amp: the square root of the cells' structure factor; 0: off; used by
                          # scene.frame_mottle, not by filling)
)


def _episodes(n: int, frac: float, mean_len: float, rng) -> np.ndarray:
    """Boolean per frame: a two-state Markov chain in episodes of mean
    length mean_len (frames) covering `frac` of the time."""
    if frac <= 0 or n == 0:
        return np.zeros(n, bool)
    if frac >= 1:
        return np.ones(n, bool)
    p_off = 1.0 / max(mean_len, 1.0)                   # leave an episode
    p_on = p_off * frac / (1 - frac)                   # enter one
    u = rng.random(n + 1)
    out = np.zeros(n, bool)
    s = bool(u[0] < frac)
    for i in range(n):
        out[i] = s
        s = (u[i + 1] >= p_off) if s else (u[i + 1] < p_on)
    return out


def _column_pattern(n: int, dx: float, phi: float, on_mean: float, edge: float, rng, shape: float = 1.0) -> np.ndarray:
    """Red-cell columns on a grid of n samples dx apart: 1 in a column, 0 in
    a plasma gap; columns of mean length on_mean, gaps of mean length
    on_mean (1 - phi) / phi (exponential, or gamma of the given shape:
    more regular), rounded ends."""
    if phi >= 0.999:
        return np.ones(n)
    off_mean = on_mean * (1 - phi) / max(phi, 1e-3)
    L = n * dx
    g = np.zeros(n)
    pos, on = -rng.uniform(0, on_mean + off_mean), bool(rng.random() < phi)
    while pos < L:
        mean_ = on_mean if on else off_mean
        ln = rng.exponential(mean_) if shape == 1.0 else rng.gamma(shape, mean_ / shape)
        if on:
            a, b = max(int(math.ceil(pos / dx)), 0), min(int(math.ceil((pos + ln) / dx)), n)
            if b > a:
                g[a:b] = 1.0
        pos += ln
        on = not on
    return ndi.gaussian_filter1d(g, edge / dx) if edge > 0 else g


def _grain(e, vid, burst_seed, q, N, fps, kind, frame, v, amp=None) -> "Filling":
    """Red-cell aggregates in a continuous vessel: 1 + amp (default
    grain_amp) x a smooth random pattern (correlation grain_len) moving
    downstream at v; a frame sees it at its instant (smeared over the
    exposure), the average the mean over the burst (nearly 1)."""
    from scipy.signal import fftconvolve
    dx = 0.25
    L = e.length()
    M = int(math.ceil(L / dx)) + 1
    vr = np.random.default_rng([int(burst_seed), int(vid), 7])
    X = np.arange(N) * v / fps
    qi = np.round(X / dx).astype(int)
    Q = int(qi.max()) if N else 0
    if q.get("grain_kind", "gauss") == "clump":
        # clumps of aggregated red cells and plasma-rich stretches between them (Meyer 2018: erythrocytes aggregate
        # in the low-order drainage vessels, laminar flow is lost there), standardised to mean 0, SD 1
        ph = float(np.clip(q["grain_phi"], 0.05, 0.95))
        c = _column_pattern(M + Q, dx, ph, q["grain_len"], q["grain_edge"], vr, shape=q["grain_shape"])
        g = (c - ph) / math.sqrt(ph * (1 - ph))
    else:
        g = ndi.gaussian_filter1d(vr.standard_normal(M + Q), q["grain_len"] / dx)
        g = g / max(float(g.std()), 1e-9)
    smear = int(round(v * q["exposure_ms"] * 1e-3 / dx))
    if smear > 1:
        g = ndi.uniform_filter1d(g, smear, mode="nearest")
    P = 1.0 + (q["grain_amp"] if amp is None else amp) * g
    if kind == "frame":
        m = P[np.arange(M) + Q - qi[frame]]
    else:
        c = np.bincount(qi, minlength=Q + 1).astype(float)
        m = fftconvolve(P, c)[Q:Q + M] / N
    return Filling(np.arange(M) * dx, np.clip(m, 0.0, None), 1.0, "flowing", smooth=True, speed=v)


def filling(graph, rng: np.random.Generator | None = None, kind: str = "average", frame: int | None = None,
            burst_seed: int | None = None, **p) -> dict:
    """Red-cell filling of the vessels for the renderer's hct_mod:
    {vid: Filling (callable(s_dc) -> multiplier of the vessel's haematocrit)}.

    Physiology.  In a capillary (radius <= r_single d_c) red cells pass in
    single file: columns of a few cells separated by plasma gaps, a fraction
    phi (~0.45) of the length holding cells (column and gap lengths
    exponential, or gamma of shape col_shape: more regular), moving
    downstream (u -> v) at the vessel's red-cell speed: v_cap (u / u_cap)^
    speed_exp, u = flow / lumen area when the graph's flow is solved, u_cap
    the capillaries' median, spread log-normally by speed_sd per vessel (the
    conjunctiva's red cells move at about 0.5 mm/s in every calibre: v2 uses
    speed_exp 0.1; v1 used 1, which drove medium venules at 3-18 mm/s and
    smeared their red cells over 20-150 px in a frame); 0 when the vessel is
    stagnant.  Some capillaries flow stop-and-go, some have plasma-only
    episodes, a few are not perfused at all during the burst.  Towards
    r_cont the gaps close; wider vessels are continuous and are left out of
    the dict (their multiplier is 1), unless grain_amp > 0: then their
    blood's density fluctuates along them by that relative SD (red-cell
    aggregates: smooth noise of correlation grain_len, or with
    grain_kind='clump' clumps of cells and plasma-rich stretches, the beads
    of a raw frame), moving with the flow (_grain).  The cells' granularity
    across a lumen (one frame's columns of cells) is not a filling: see
    rbc_mottle.  Each Filling carries its red cells' speed (d_c/s).

    kind='frame': one exposure (frame index `frame`, random if None): the
    multiplier is 1 in a column and 0 in a gap, smeared along the vessel by
    the columns' motion during the exposure (v * exposure_ms), 0 along a
    vessel caught empty or never perfused.
    kind='average': the mean over the burst's n_frames of exactly those
    frames: lower than 1 but continuous in the capillaries (about phi times
    the time not empty), close to 1 in the wide vessels, still 0 in the
    vessels never perfused; slow or stagnant columns leave structure.

    The multiplier is the blood present relative to a lumen full of blood at
    the vessel's haematocrit h.  If the generator's h already includes the
    capillaries' low tube haematocrit, pass relative=True: the multiplier is
    then divided by phi, so a flowing vessel's time mean is 1; a column then
    holds h / phi, which may not exceed packed cells (h_col_max), so phi is
    at least h / h_col_max.

    Everything is drawn from burst_seed (per vessel), so the frames and the
    average of one burst agree: the average is the mean of all its frames.
    rng only draws burst_seed and the frame index when they are not given.
    Keyword parameters: FILL_DEFAULTS."""
    from scipy.signal import fftconvolve
    from vesselscene.graph import eval_profile
    q = dict(FILL_DEFAULTS)
    unknown = set(p) - set(q)
    if unknown:
        raise TypeError(f"unknown filling parameters {sorted(unknown)}")
    q.update(p)
    rng = rng if rng is not None else np.random.default_rng()
    if burst_seed is None:
        burst_seed = int(rng.integers(2 ** 31))
    N = max(int(q["n_frames"]), 1)
    if kind == "frame":
        frame = int(rng.integers(N)) if frame is None else int(frame) % N
    elif kind != "average":
        raise ValueError(f"kind must be 'average' or 'frame', not {kind!r}")
    fps = float(q["fps"])
    dx = 0.1
    ves = graph.vessels
    radius = {vid: float(np.mean(eval_profile(e.r_ctrl, np.linspace(0, 1, 16)))) for vid, e in ves.items()}
    # mean red-cell speed per vessel: flow / area, scaled to v_cap in the capillaries
    speed = {vid: e.flow / (math.pi * radius[vid] ** 2) if e.flow > 0 else 0.0 for vid, e in ves.items()}
    cap = [speed[v] for v in ves if radius[v] <= q["r_single"] and speed[v] > 0]
    pos = [s for s in speed.values() if s > 0]
    ref = float(np.median(cap)) if cap else (float(np.median(pos)) if pos else 0.0)
    if ref > 0:                                  # red-cell speed (d_c/s): v_cap (u / u_cap)^speed_exp, capped at v_max
        def _rel(vid, sp_):
            x = (sp_ / ref) ** q["speed_exp"] if sp_ > 0 else 0.0
            if q["speed_sd"] > 0 and x > 0:
                z = float(np.random.default_rng([int(burst_seed), int(vid), 11]).standard_normal())
                x *= math.exp(q["speed_sd"] * z)
            return min(x, q["v_max"] / q["v_cap"]) * ref
        speed = {vid: _rel(vid, sp_) for vid, sp_ in speed.items()}
    out = {}
    for vid, e in ves.items():
        r = radius[vid]
        if r >= q["r_cont"]:
            if q["grain_amp"] > 0 and speed[vid] > 0 and not e.info.get("stagnant"):
                amp = min(q["grain_amp"] * (min(r, q["grain_r_agg"]) / q["r_cont"]) ** q["grain_r_exp"],
                          q["grain_amp_max"])
                if r > q["grain_r_agg"]:
                    amp *= (q["grain_r_agg"] / r) ** q["grain_r_pow"]
                out[vid] = _grain(e, vid, burst_seed, q, N, fps, kind, frame, (q["v_cap"] * speed[vid] / ref)
                                  if ref > 0 else q["v_cap"], amp=amp)
            continue
        vr = np.random.default_rng([int(burst_seed), int(vid)])
        t = float(np.clip((r - q["r_single"]) / max(q["r_cont"] - q["r_single"], 1e-6), 0.0, 1.0))
        capillary = t == 0.0
        phi_c = float(np.clip(vr.normal(q["phi_cap"], q["phi_sd"]), 0.1, 0.95))
        if q["relative"]:
            # the vessel's h is its tube haematocrit = phi x the columns' own: a column cannot be denser than
            # packed cells, so the cells must fill at least h / h_col_max of the length
            h_tube = float(np.mean(np.clip(eval_profile(e.h_ctrl, np.linspace(0, 1, 16)), 0.0, 1.0)))
            phi_c = max(phi_c, min(h_tube / q["h_col_max"], 1.0))
        phi = phi_c + (1 - phi_c) * t ** 0.7
        on_mean = q["cell_len"] * q["train_cells"] * (1 + 8 * t * t)
        floor = float(np.clip(q["gap_fill"], 0.0, 1.0)) * math.sqrt(t)       # blood left in a gap (0: capillaries)
        phi_eff = floor + (1.0 - floor) * phi                                 # time-mean filling while flowing
        L = e.length()
        M = int(math.ceil(L / dx)) + 1
        s = np.arange(M) * dx
        if ref > 0:
            v = q["v_cap"] * speed[vid] / ref
        else:
            v = q["v_cap"] * (max(r, 0.3) / 0.5) * math.exp(vr.normal(0, 0.4))
        state = "flowing"
        if e.info.get("stagnant"):
            v, state = 0.0, "stagnant"
        u_draw = vr.random(3)
        if capillary and u_draw[0] < q["p_unperfused"]:
            out[vid] = Filling(s, np.zeros(M), phi, "unperfused", speed=0.0)
            continue
        flowing = np.ones(N, bool)
        empty = np.zeros(N, bool)
        if capillary and u_draw[1] < q["p_stopgo"] and v > 0:
            flowing = ~_episodes(N, q["stop_frac"], q["stop_s"] * fps, vr)
            state = "stop-go"
        if capillary and u_draw[2] < q["p_empty"]:
            empty = _episodes(N, q["empty_frac"], q["empty_s"] * fps, vr)
        X = np.concatenate([[0.0], np.cumsum(flowing[1:] * v / fps)])      # column displacement at each frame
        qi = np.round(X / dx).astype(int)
        Q = int(qi.max())
        P = _column_pattern(M + Q, dx, phi, on_mean, q["edge"], vr, shape=q["col_shape"])   # P[j]: position (j - Q) dx
        if floor > 0:
            P = floor + (1.0 - floor) * P
        # a frame is exposed for exposure_ms: the columns move v * exposure meanwhile, so the
        # frame sees them smeared along the vessel (the average, the mean of such frames, too)
        smear = int(round(v * q["exposure_ms"] * 1e-3 / dx))
        if smear > 1:
            P = ndi.uniform_filter1d(P, smear, mode="nearest")
        if kind == "frame":
            if empty[frame]:
                m, st = np.zeros(M), "empty"
            else:
                m, st = P[np.arange(M) + Q - qi[frame]], state
        else:
            c = np.bincount(qi[~empty], minlength=Q + 1).astype(float)
            m = fftconvolve(P, c)[Q:Q + M] / N if c.sum() else np.zeros(M)
            st = state
        if q["relative"]:
            m = m / phi_eff
        out[vid] = Filling(s, np.clip(m, 0.0, None), phi_eff, st, speed=v)
    return out
