"""Scale selection across size channels (spec 4.6b), on toy stimuli.

Two size channels of the same orientation (the sheet along the line, theta = 0,
so every profile below is read along image row ROW):

  fine    sigma 1.5 px, elongation 3   (an 18 px long cell)
  coarse  sigma 6 px,   elongation 4   (a ~100 px long cell: +-2 sd of its 24 px Gaussian)

End-zone inhibition with a centre pool (the spec states it in words; this is
the toy's form of it):

  E+ , E-  mean fine rate in the coarse cell's two end zones (24-48 px from its centre)
  Cz       mean fine rate in its centre pool (|t| < 12 px)
  I        relu(max(E+, E-) - Cz)
  G'       max(0, G - w I)

and the rejected first version, plain cross-scale inhibition along the axis:

  G'_plain = max(0, G - w_plain * mean fine rate over the whole axis (|t| < 48 px)).

Noise: white 0.01 OD (as the spec's synthetic vessels) plus a faint smooth
tissue texture, so the coarse channel's noise units are not inflated by
averaging white noise alone.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter
from scipy.signal import fftconvolve

from . import neuro as N

SHAPE = (120, 240)              # (h, w); the line runs along row ROW
ROW = 60
END = 120                       # the thin line ends at this column (as the spec's synthetic line)
FINE = (1.5, 3.0)               # sigma, elongation
COARSE = (6.0, 4.0)
ALONG = COARSE[0] * COARSE[1]   # 24 px: the coarse cell's along-axis sd
ZONE_IN, ZONE_OUT, CENTRE = 24, 48, 12
W_INH = 1.0                     # weight of end-zone inhibition (scale_weight; toy)
THRESH = 3.0                    # a tracking threshold, noise units (t_high, spec 6.3)


def _texture(shape, seed, amp=0.003, scale=8.0):
    t = gaussian_filter(np.random.default_rng(seed).normal(0.0, 1.0, shape), scale)
    return amp * t / t.std()


def _noisy(clean, seed):
    return N.add_noise(clean, 0.01, seed) + _texture(clean.shape, seed + 10)


def tapered_od(shape, x0, x1, xt0, xt1, r_wide, r_thin, row, a=0.15, blur=1.0):
    """A horizontal vessel from x0 to x1 whose radius goes from r_wide to r_thin
    between columns xt0 and xt1 (a wide vessel tapering into a thin one)."""
    x, y = N.grid(shape)
    r = np.interp(x, [xt0, xt1], [r_wide, r_thin])
    inside = (x >= x0) & (x <= x1)
    od = np.where(inside, N.chord(np.abs(y - row), r, a), 0.0)
    # round caps
    for xe in (x0, x1):
        re = float(np.interp(xe, [xt0, xt1], [r_wide, r_thin]))
        cap = N.chord(np.hypot(x - xe, y - row), re, a)
        od = np.maximum(od, cap)
    return gaussian_filter(od, blur)


STIMULI = {
    # name: clean OD
    "thin_end": lambda: N.vessel_od(SHAPE, [((-60, ROW), (END, ROW), 1.5, 0.15)], blur=1.0),
    "thin_cont": lambda: N.vessel_od(SHAPE, [((-60, ROW), (300, ROW), 1.5, 0.15)], blur=1.0),
    "wide_end": lambda: N.vessel_od(SHAPE, [((-60, ROW), (END, ROW), 7.0, 0.15)], blur=1.0),
    "taper": lambda: tapered_od(SHAPE, -60, 300, 112, 128, 7.0, 1.5, ROW),
}


@lru_cache(maxsize=None)
def _noise_sd(sigma, elong):
    big = _noisy(np.zeros((360, 720)), 99)
    return float(fftconvolve(big, N.line_kernel(0.0, sigma, elong), mode="valid").std())


def channel(od, sigma, elong):
    """One size channel's theta = 0 sheet, in noise units (drive, may be < 0)."""
    return fftconvolve(od, N.line_kernel(0.0, sigma, elong), mode="same") / _noise_sd(sigma, elong)


def _zone_mean(a, lo, hi):
    """Mean of a over columns x+lo .. x+hi (inclusive), for every x (correlation along rows)."""
    lo, hi = int(lo), int(hi)
    half = max(abs(lo), abs(hi))
    w = np.zeros(2 * half + 1)
    w[half + lo:half + hi + 1] = 1.0 / (hi - lo + 1)
    return correlate1d(a, w, axis=1, mode="constant")


@lru_cache(maxsize=None)
def compute(name: str, seed: int = 3):
    clean = STIMULI[name]()
    od = _noisy(clean, seed)
    F = np.maximum(channel(od, *FINE), 0.0)          # fine rates
    G = np.maximum(channel(od, *COARSE), 0.0)        # coarse rates
    Ep = _zone_mean(F, ZONE_IN, ZONE_OUT)
    Em = _zone_mean(F, -ZONE_OUT, -ZONE_IN)
    Cz = _zone_mean(F, -CENTRE, CENTRE)
    I = np.maximum(np.maximum(Ep, Em) - Cz, 0.0)
    G2 = np.maximum(G - W_INH * I, 0.0)
    axis_mean = _zone_mean(F, -ZONE_OUT, ZONE_OUT)
    return dict(clean=clean, od=od, F=F, G=G, Ep=Ep, Em=Em, Cz=Cz, I=I, G2=G2, axis_mean=axis_mean)


def profile(name, key, row=ROW):
    return compute(name)[key][row]


def reach(curve, thresh=THRESH, start=0):
    """Last column (from start on) before the curve first falls below thresh."""
    below = np.nonzero(curve[start:] < thresh)[0]
    return start + (below[0] - 1 if len(below) else len(curve) - 1 - start)


def plain_weight_needed(margin=20, name="thin_end"):
    """Smallest plain cross-scale weight that silences the coarse cell `margin` px past the end."""
    c = compute(name)
    G, A = c["G"][ROW, END + margin], c["axis_mean"][ROW, END + margin]
    return G / max(A, 1e-9)
