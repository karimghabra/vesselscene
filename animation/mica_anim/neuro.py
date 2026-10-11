"""Small numpy versions of the spec's computations, used to drive the animation.

Everything shown on screen (images, tuning curves, maps) is computed here from
the same equations as the spec, on toy stimuli: a few vessels drawn with the
chord law, white noise in OD, 16 orientations.  These are teaching versions,
not MICA's layers: one size channel, no association field unless asked for,
and parameters chosen so the effects read clearly at animation scale.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

N_ORIENT = 16
THETAS = np.arange(N_ORIENT) * np.pi / N_ORIENT  # preferred orientations, 0..180 deg


# --------------------------------------------------------------------- stimuli

def grid(shape):
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    return x, y


def dist_to_segment(x, y, p0, p1):
    """Distance from every pixel to the segment p0-p1, and the along-segment
    coordinate t in [0, 1]."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L2 = max(d @ d, 1e-12)
    t = np.clip(((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / L2, 0.0, 1.0)
    return np.hypot(x - (p0[0] + t * d[0]), y - (p0[1] + t * d[1])), t


def chord(dist, r, a):
    """OD across a cylinder of blood of radius r: proportional to the chord
    length through it, 2 sqrt(r^2 - x^2) (Beer-Lambert), peak a."""
    return a * np.sqrt(np.clip(1.0 - (dist / r) ** 2, 0.0, None))


def vessel_od(shape, segments, blur=1.0, combine="sum"):
    """OD image of straight vessels.  segments: [(p0, p1, r, a), ...].
    combine='sum' adds ODs (vessels at different depths, a crossing);
    'max' takes the union (lumens merged at a fork)."""
    x, y = grid(shape)
    layers = [chord(dist_to_segment(x, y, p0, p1)[0], r, a) for p0, p1, r, a in segments]
    od = np.sum(layers, axis=0) if combine == "sum" else np.max(layers, axis=0)
    return gaussian_filter(od, blur) if blur > 0 else od


def ring_od(shape, centre, radius, r, a, blur=1.0):
    x, y = grid(shape)
    d = np.abs(np.hypot(x - centre[0], y - centre[1]) - radius)
    return gaussian_filter(chord(d, r, a), blur)


def blob_od(shape, centre, sigma, a):
    x, y = grid(shape)
    return a * np.exp(-((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / (2 * sigma ** 2))


def add_noise(od, sd=0.01, seed=0):
    return od + np.random.default_rng(seed).normal(0.0, sd, od.shape)


def to_intensity(od, background=0.85):
    """What the camera records: I = B exp(-OD)."""
    return background * np.exp(-od)


# ---------------------------------------------------------------- simple cells

def line_kernel(theta, sigma=2.0, elong=3.0, size=None):
    """Even-symmetric simple cell tuned to a dark line at orientation theta:
    a second derivative of a Gaussian across the line (an excitatory centre
    between inhibitory flanks), a Gaussian of sigma*elong along it.  Zero mean,
    unit L1 norm of its positive part."""
    half = int(np.ceil(3.5 * sigma * max(elong, 1.0))) if size is None else size
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    t = x * np.cos(theta) + y * np.sin(theta)       # along the line
    n = -x * np.sin(theta) + y * np.cos(theta)      # across it
    k = (1.0 - n ** 2 / sigma ** 2) * np.exp(-n ** 2 / (2 * sigma ** 2)) * np.exp(-t ** 2 / (2 * (sigma * elong) ** 2))
    k -= k.mean()
    return k / k[k > 0].sum()


def simple_cells(od, sigma=2.0, elong=3.0):
    """Responses of the 16 orientation layers: U[k] = w_k * OD (a linear
    filter; positive on a dark line at orientation theta_k)."""
    return np.stack([fftconvolve(od, line_kernel(th, sigma, elong), mode="same") for th in THETAS])


def noise_units(U, bg_mask):
    """Divide each layer by its spread over background pixels (section 4.0)."""
    sd = np.array([u[bg_mask].std() for u in U])
    return U / sd[:, None, None]


def floor(curve, mode="half"):
    """Step 4's orientation-blind part of one tuning curve: the mean of its
    lower half (8 of 16, today's) or its lowest quarter (4 of 16, the fix)."""
    s = np.sort(np.asarray(curve))
    n = {"half": len(s) // 2, "quarter": len(s) // 4}[mode]
    return s[:n].mean()


def surround(curve, mode="half"):
    return np.maximum(np.asarray(curve) - floor(curve, mode), 0.0)


def tuning_curve(U, p):
    """The hypercolumn at pixel p = (x, y): the 16 responses read through it."""
    x, y = p
    return U[:, int(round(y)), int(round(x))]
