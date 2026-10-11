"""V2 junction computations of spec 4.2-4.3: small numpy versions for chapters 8 and 9.

Everything here follows the spec's equations on toy stimuli (chord-law vessels
6 px wide, peak OD 0.15, white noise 0.01 OD, as the spec's synthetic vessels;
16 orientations x 2 directions):

  C        rectified simple cells in noise units (one size channel, no surround)
  lobes    A_ahead[j](p): weighted mean of the orientation-pooled response of layer
           k = j mod 16 along the direction u_j, from d0 to d1 px ahead of p
           (with a little width across, spec 4.7).  A+_k = A_ahead[k], A-_k = A_ahead[k + 16].
  ends     E[j](p) = max(0, min(C[k], A_behind[j]) - beta A_ahead[j])        (spec 4.2)
           j < 16: E_fwd[k], free side +t_k;  j >= 16: E_bwd[k], free side -t_k.
           Gate: C[k] > t_low, and the cell wins against its nearest rivals (the cells
           beside it across the vessel and the neighbouring orientations: as built, 4.2).
  divisive E[j](p) = [A_behind - beta A_ahead]+ / (A_behind + A_ahead + c)   (spec 4.2, next variant)
           same winner gate, with C > DIV_GATE (toy: a ratio has no absolute floor)
  bipole   b_k(p) = sqrt(A+_k A-_k), inhibited by end cells of its own orientation near p
           (as built, 4.2)
  arms     rays (3 px wide) from end cells pointing into p, within rho = r0 + w/2, and
           passing lines (bipole peaks over orientation); merged within 22.5 deg;
           J3 = the 3rd-largest arm strength                                   (spec 4.3)

Thresholds are the toy's (TOY_TAU_*, DIV_THR): the spec's start values (6.3) are in MICA's
noise units, and this toy has one size channel and no surround.

Angles are image angles (x right, y down), as in neuro.py: t_k = (cos th_k, sin th_k).
A direction phi in the image is the scene direction (cos phi, -sin phi).
Stimuli are rendered on a padded canvas (vessels continue past the visible frame),
and every map is cropped back, so nothing fires at the frame's edge.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, maximum_filter
from scipy.signal import fftconvolve

from . import neuro as N

N_DIR = 32
PHIS = np.arange(N_DIR) * np.pi / 16        # free-side directions u_j, image angles over 360 deg
SIGMA, ELONG = 2.0, 2.0                     # the simple cell of these chapters: 2 px across, 4 px along
LOBE_D0, LOBE_D1, LOBE_SD = 3.0, 13.0, 6.0  # lobe weights exp(-d^2 / 2 sd^2), d in [d0, d1] px
LOBE_W = 1.5                                # the lobe's Gaussian width across, px (spec 4.7: lobes need width)
R0, WIDTH = 4.0, 6.0                        # junction disc rho = r0 + w / 2 (toy: one width, 6 px)
MERGE_BINS = 2                              # arms within 22.5 deg (2 bins of 11.25 deg) are merged
INH_NEAR = 2                                # end -> bipole inhibition pooled over 5 x 5 px
RAY_LAT = 1.5                               # the ray kernel is 3 px wide
T_LOW, T_HIGH, TAU_J = 1.5, 3.0, 3.0        # spec 6.3 start values, noise units
PAD = 28                                    # canvas margin, px
VESSEL_R, VESSEL_A, NOISE = 3.0, 0.15, 0.01


def unit(phi):
    return np.array([np.cos(phi), np.sin(phi)])


def scene_vec(phi):
    """Image direction phi as a scene vector (y up)."""
    return np.array([np.cos(phi), -np.sin(phi), 0.0])


def rho():
    return R0 + WIDTH / 2


# ------------------------------------------------------------------ V1 input

def v1(od, bg_mask, sigma=SIGMA, elong=ELONG):
    """Rectified simple cells in noise units, (16, h, w)."""
    U = N.noise_units(N.simple_cells(od, sigma, elong), bg_mask)
    return np.maximum(U, 0.0)


def pooled(C):
    """Orientation pooling of step 5: max(C[k], 0.8 C[k-1], 0.8 C[k+1])."""
    return np.maximum(C, 0.8 * np.maximum(np.roll(C, 1, axis=0), np.roll(C, -1, axis=0)))


def winners(C):
    """Winner-take-all among the nearest rivals (4.2, as built): the cell is a crest
    across the vessel (beats the cells 1 px either side along its normal) and a peak
    over orientation at its pixel."""
    h, w = C.shape[1:]
    y, x = np.mgrid[0:h, 0:w].astype(float)
    out = np.zeros(C.shape, bool)
    for k, th in enumerate(N.THETAS):
        n = np.array([-np.sin(th), np.cos(th)])
        a = map_coordinates(C[k], [y + n[1], x + n[0]], order=1, mode="nearest")
        b = map_coordinates(C[k], [y - n[1], x - n[0]], order=1, mode="nearest")
        out[k] = (C[k] >= a) & (C[k] >= b)
    out &= (C >= np.roll(C, 1, axis=0)) & (C >= np.roll(C, -1, axis=0))
    return out


# ------------------------------------------------------------------ lobes

def lobe_samples(step=1.0):
    """Distances along a lobe and their normalised weights (for drawing it)."""
    d = np.arange(LOBE_D0, LOBE_D1 + 1e-9, step)
    w = np.exp(-d ** 2 / (2 * LOBE_SD ** 2))
    return d, w / w.sum()


def lobe_kernel(phi):
    """Correlation kernel K with (P (x) K)(p) = weighted mean of P(p + d u), d in [d0, d1]:
    a ray along u, with a narrow Gaussian across so it has some width."""
    half = int(np.ceil(LOBE_D1 + 3 * LOBE_W)) + 1
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    u = unit(phi)
    t = x * u[0] + y * u[1]
    n = -x * u[1] + y * u[0]
    wt = np.exp(-t ** 2 / (2 * LOBE_SD ** 2)) * ((t >= LOBE_D0 - 0.5) & (t <= LOBE_D1 + 0.5))
    K = wt * np.exp(-n ** 2 / (2 * LOBE_W ** 2))
    return K / K.sum()


def lobes(C):
    """A_ahead[j] for the 32 directions: the pooled response of layer j mod 16 along u_j."""
    P = pooled(C)
    out = np.empty((N_DIR,) + C.shape[1:])
    for j, phi in enumerate(PHIS):
        K = lobe_kernel(phi)
        out[j] = fftconvolve(P[j % 16], K[::-1, ::-1], mode="same")   # correlation
    return np.maximum(out, 0.0)


def behind(A):
    """A_behind[j] = A_ahead[j + 16]."""
    return np.roll(A, 16, axis=0)


# ------------------------------------------------------------------ end-stopped cells (4.2)

def end_cells(C, A, gate, beta=1.0):
    """Subtractive end-stopping, 32 direction channels."""
    C32 = np.concatenate([C, C])
    E = np.maximum(np.minimum(C32, behind(A)) - beta * A, 0.0)
    return E * np.concatenate([gate, gate])


def end_cells_divisive(C, A, gate, beta=1.0, c=2.0):
    """Divisive end-stopping (spec 4.2, next variant): shunting by the local line activity."""
    Ab = behind(A)
    E = np.maximum(Ab - beta * A, 0.0) / (Ab + A + c)
    return E * np.concatenate([gate, gate])


def end_peaks(E, thr, min_dist=5, min_turn_deg=45.0):
    """Ends: local maxima in space of each direction channel E[j] above thr, then a greedy
    merge: a weaker end is dropped if a stronger one lies within min_dist px with a free
    direction within min_turn_deg.  [(x, y, j, value)], strongest first."""
    cands = []
    for j in range(N_DIR):
        Ej = np.maximum(E[j], 0.0)
        nb = np.maximum(np.maximum(E[(j - 1) % N_DIR], E[(j + 1) % N_DIR]), Ej)
        pk = (Ej == maximum_filter(nb, size=2 * min_dist + 1)) & (Ej > thr)
        ys, xs = np.nonzero(pk)
        cands += [(float(x), float(y), j, float(Ej[y, x])) for y, x in zip(ys, xs)]
    cands.sort(key=lambda t: -t[3])
    out = []
    for c in cands:
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) > min_dist
               or abs((c[2] - o[2] + 16) % 32 - 16) * 11.25 >= min_turn_deg for o in out):
            out.append(c)
    return out


# ------------------------------------------------------------------ junction cells (4.3)

def bipole(A):
    return np.sqrt(A[:16] * A[16:])


def rays(E, r, lateral=RAY_LAT):
    """e_j(p) = max over d in [0, r] of E[j](p - d u_j): the 'ray' kernel per end direction,
    given a little width (+-lateral px across), so an end just beside the ray still counts."""
    h, w = E.shape[1:]
    y, x = np.mgrid[0:h, 0:w].astype(float)
    lats = np.linspace(-lateral, lateral, 5) if lateral > 0 else [0.0]
    out = np.zeros_like(E)
    for j in range(N_DIR):
        u = unit(PHIS[j])
        n = np.array([-u[1], u[0]])
        best = E[j].copy()
        for d in np.arange(1.0, r + 1e-9, 1.0):
            for l in lats:
                s = map_coordinates(E[j], [y - d * u[1] + l * n[1], x - d * u[0] + l * n[0]], order=1, mode="constant")
                np.maximum(best, s, out=best)
        out[j] = best
    return out


def arm_bins(e, b, tau_end=T_HIGH, tau_pass=T_HIGH, b_peak=None):
    """Arm candidates per pixel on the 32 outward directions.
    An end at q = p - d u_j whose free side u_j points at p gives an arm leaving p
    along -u_j (bin j + 16).  A bipole peak over orientation gives two arms, +t_k
    (bin k) and -t_k (bin k + 16).  Returns (strength, kind): 1 = end, 2 = pass."""
    end_s = np.roll(np.where(e > tau_end, e, 0.0), 16, axis=0)
    bp = b if b_peak is None else b_peak             # the peak over orientation is read before inhibition
    peak = (bp >= np.roll(bp, 1, axis=0)) & (bp >= np.roll(bp, -1, axis=0)) & (b > tau_pass)
    pb = np.where(peak, b, 0.0)
    pass_s = np.concatenate([pb, pb])
    S = np.maximum(end_s, pass_s)
    kind = np.where(S <= 0, 0, np.where(pass_s >= end_s, 2, 1))
    return S, kind


def merge_arms(S, n_max=6, merge_bins=MERGE_BINS):
    """Greedy merge on the circle of 32 bins: keep the strongest arm, drop those within
    22.5 deg of it, repeat.  Returns (strengths, bins), (n_max, h, w), strongest first."""
    S = S.copy()
    vals = np.zeros((n_max,) + S.shape[1:])
    bins = np.full((n_max,) + S.shape[1:], -1, int)
    idx = np.arange(N_DIR)[:, None, None]
    for i in range(n_max):
        m = S.argmax(axis=0)
        v = np.take_along_axis(S, m[None], 0)[0]
        vals[i] = v
        bins[i] = np.where(v > 0, m, -1)
        dist = np.abs((idx - m[None] + 16) % 32 - 16)
        S = np.where(dist <= merge_bins, 0.0, S)
    return vals, bins


def end_inhibition(e, reach=MERGE_BINS, near=INH_NEAR):
    """Inhibition of the bipole of orientation k by end cells of the same orientation
    (within the arm-merge angle) whose rays reach p or a pixel near it: the max over
    |k' - k| <= reach of max(e_k', e_k'+16), pooled over a (2 near + 1)^2 square."""
    ek = np.maximum(e[:16], e[16:])
    out = ek.copy()
    for s in range(1, reach + 1):
        out = np.maximum(out, np.maximum(np.roll(ek, s, axis=0), np.roll(ek, -s, axis=0)))
    if near > 0:
        out = maximum_filter(out, size=(1, 2 * near + 1, 2 * near + 1))
    return out


def junction_cells(E, A, tau_end=T_HIGH, tau_pass=T_HIGH, end_inhibits_bipole=True):
    """J3 map, and what is needed to read the arms at any pixel.  As built (4.2), an
    end cell inhibits the bipole of its own orientation: a line that ends near p does
    not pass through it."""
    e = rays(E, rho())
    b0 = bipole(A)
    b = np.maximum(b0 - end_inhibition(e), 0.0) if end_inhibits_bipole else b0
    S, kind = arm_bins(e, b, tau_end, tau_pass, b_peak=b0)
    vals, bins = merge_arms(S)
    return dict(e=e, b=b, b0=b0, S=S, kind=kind, vals=vals, bins=bins, J3=vals[2], n=(vals > 0).sum(axis=0))


def junction_peaks(J3, tau_J=TAU_J, r0=R0):
    """Local maxima of J3 above tau_J, at least r0 apart (non-maximum suppression)."""
    size = int(2 * r0 + 1)
    pk = (J3 == maximum_filter(J3, size=size)) & (J3 > tau_J)
    ys, xs = np.nonzero(pk)
    return sorted([(float(x), float(y), float(J3[y, x])) for y, x in zip(ys, xs)], key=lambda t: -t[2])


def ray_source(E, x, y, j, r=None, lateral=RAY_LAT):
    """Where the ray of direction j from (x, y) found its end: (qx, qy, E) at the maximum of
    E[j](p - d u_j + l n_j), for drawing."""
    r = rho() if r is None else r
    u = unit(PHIS[j])
    n = np.array([-u[1], u[0]])
    best = (x, y, -1.0)
    for d in np.arange(0.0, r + 1e-9, 0.5):
        for l in np.linspace(-lateral, lateral, 5):
            qx, qy = x - d * u[0] + l * n[0], y - d * u[1] + l * n[1]
            v = float(map_coordinates(E[j], [[qy], [qx]], order=1, mode="constant")[0])
            if v > best[2]:
                best = (qx, qy, v)
    return best


def arms_at(J, x, y):
    """The merged arms at pixel (x, y): [(phi, strength, 'end' | 'pass')], strongest first."""
    xi, yi = int(round(x)), int(round(y))
    out = []
    for v, bn in zip(J["vals"][:, yi, xi], J["bins"][:, yi, xi]):
        if v <= 0 or bn < 0:
            continue
        out.append((float(PHIS[bn]), float(v), "pass" if J["kind"][bn, yi, xi] == 2 else "end"))
    return out


def type_junction(arms, min_cross_deg=15.0):
    """First-pass typing of spec 4.3, from the merged arms."""
    n = len(arms)
    ends = [a for a in arms if a[2] == "end"]
    passes = [a for a in arms if a[2] == "pass"]
    if n < 3:
        return "none"
    if n >= 5:
        return "compound"
    if n == 3:
        if len(passes) == 2 and len(ends) == 1 and _opposite(passes[0][0], passes[1][0]):
            return "T"
        if len(ends) == 3:
            return "Y"
        return "compound"
    if len(passes) == 4:
        lines = _pair_opposites([p[0] for p in passes])
        if lines is not None:
            d = np.rad2deg(abs(((lines[0] - lines[1]) + np.pi / 2) % np.pi - np.pi / 2))
            if d >= min_cross_deg:
                return "crossing"
    if len(ends) == 4 and _pair_opposites([a[0] for a in ends]) is not None:
        return "crossing (broken)"
    return "compound"


def _opposite(a, b, tol=np.deg2rad(22.6)):
    return abs(((a - b) % (2 * np.pi)) - np.pi) <= tol


def _pair_opposites(phis):
    """If 4 directions pair into 2 roughly straight lines: their orientations."""
    p = list(phis)
    for i in range(1, 4):
        rest = [q for k, q in enumerate(p) if k not in (0, i)]
        if _opposite(p[0], p[i]) and _opposite(rest[0], rest[1]):
            return (p[0] % np.pi, rest[0] % np.pi)
    return None


# ------------------------------------------------------------------ stimuli and the whole stack

def render(shape, groups, pad=PAD, blur=1.0):
    """OD of vessel groups on a padded canvas.  groups: [(combine, [(p0, p1, r, a), ...])];
    inside a group lumens merge ('max', a fork) or add ('sum', a crossing); groups add.
    Points are in the visible frame's pixels."""
    h, w = shape
    big = (h + 2 * pad, w + 2 * pad)
    od = np.zeros(big)
    for combine, segs in groups:
        segs = [((p0[0] + pad, p0[1] + pad), (p1[0] + pad, p1[1] + pad), r, a) for p0, p1, r, a in segs]
        od = od + N.vessel_od(big, segs, blur=blur, combine=combine)
    return od


def stack(clean_big, pad=PAD, noise=NOISE, seed=0, beta=1.0, c_div=2.0, extra=None):
    """Every map of chapters 8-9 for one stimulus, cropped to the visible frame."""
    od = N.add_noise(clean_big, noise, seed)
    bg = distance_transform_edt(clean_big < 0.004) > 8
    C = v1(od, bg)
    A = lobes(C)
    Wn = winners(C)
    G = Wn & (C > T_LOW)
    E = end_cells(C, A, G, beta)
    crop = (slice(None), slice(pad, -pad), slice(pad, -pad))
    out = dict(od=od[pad:-pad, pad:-pad], clean=clean_big[pad:-pad, pad:-pad],
               C=C[crop], A=A[crop], G=G[crop], E=E[crop], P=pooled(C)[crop])
    if extra and "div" in extra:
        out["Ediv"] = end_cells_divisive(C, A, Wn & (C > DIV_GATE), beta, c_div)[crop]
    return out


# ================================================================== the chapters' toy stimuli

D_ANG = np.pi / 16                          # 11.25 deg: every toy vessel lies on one of the 16 orientations


def _at(p, k, L):
    """The point L px from p along image angle k * 11.25 deg."""
    return (p[0] + L * np.cos(k * D_ANG), p[1] + L * np.sin(k * D_ANG))


# ---------------------------------------------------------------- chapter 8, level 1: one line that ends

LINE_SHAPE = (60, 124)
LINE_END = np.array([90.0, 23.0])
LINE_J = 30                                  # free side u_30: 337.5 deg in the image = up-right on screen
LINE_K = LINE_J % 16


@lru_cache(maxsize=None)
def single_line(seed=3):
    """A vessel that ends at LINE_END, its free side u_30.  Maps, and the direction j."""
    u = unit(PHIS[LINE_J])
    far = LINE_END - 260 * u
    big = render(LINE_SHAPE, [("max", [(tuple(far), tuple(LINE_END), VESSEL_R, VESSEL_A)])])
    S = stack(big, seed=seed)
    S["u"], S["j"], S["k"], S["end"] = u, LINE_J, LINE_K, LINE_END
    return S


def probe_values(S, x, y):
    """(C, A-, A+, E) of channel S['j'] at sub-pixel (x, y): A- behind, A+ ahead."""
    from .style import bilinear
    j, k = S["j"], S["k"]
    C = float(bilinear(S["C"][k:k + 1], x, y)[0])
    Aa = float(bilinear(S["A"][j:j + 1], x, y)[0])
    Ab = float(bilinear(S["A"][(j + 16) % 32:(j + 16) % 32 + 1], x, y)[0])
    E = max(0.0, min(C, Ab) - Aa) if C > T_LOW else 0.0       # the formula on the three values shown
    return C, Ab, Aa, E


# ---------------------------------------------------------------- the toy scene: a T, a Y, an X, free ends

SCENE_SHAPE = (130, 240)
T_C, Y_C, X_C = (48.0, 68.0), (122.0, 66.0), (194.0, 60.0)
T_NECK = 10.0                                # the branch is faint over its last 10 px before its parent


def scene_groups():
    neck = _at(T_C, -1, T_NECK)
    R, A = VESSEL_R, VESSEL_A
    return [
        ("max", [(_at(T_C, -9, 140), _at(T_C, 7, 140), R, A),          # T: the parent passes
                 (neck, _at(T_C, -1, 38), 2.4, 0.10),                   # the branch, free end at the right
                 (T_C, neck, 2.0, 0.03)]),                              # its faint neck
        ("max", [(Y_C, _at(Y_C, 8, 140), R, A),                         # Y: parent from below
                 (Y_C, _at(Y_C, -12, 40), 2.6, 0.13),                   # two daughters with free ends
                 (Y_C, _at(Y_C, -4, 40), 2.4, 0.12)]),
        ("sum", [(_at(X_C, -5, 140), _at(X_C, 11, 140), R, A),          # X: one line passes through
                 (_at(X_C, 1, 28), _at(X_C, 17, 34), 2.6, 0.13)]),      # the other has two free ends
    ]


TOY_TAU_END = TOY_TAU_PASS = 8.0             # toy thresholds (spec 6.3 starts at t_high = 3)
TOY_TAU_J = 10.0                             # toy (spec 6.3 start: 3)


@lru_cache(maxsize=None)
def toy_scene(seed=1):
    big = render(SCENE_SHAPE, scene_groups())
    S = stack(big, seed=seed)
    S["ends"] = end_peaks(S["E"], TOY_TAU_END)
    J = junction_cells(S["E"], S["A"], TOY_TAU_END, TOY_TAU_PASS)
    S["J"] = J
    S["junctions"] = [(x, y, v, type_junction(arms_at(J, x, y)), arms_at(J, x, y))
                      for x, y, v in junction_peaks(J["J3"], TOY_TAU_J)]
    return S


# ---------------------------------------------------------------- a toy tracer (the old way, step 6)

def trace_along(S, p0, p1, t_track, step=1.0):
    """Follow the straight centreline from p0 toward p1 while the vessel's own cell stays above
    t_track and wins over orientation (within one step): the points reached, in pixels."""
    from .style import bilinear
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.hypot(*(p1 - p0))
    u = (p1 - p0) / L
    k = int(round((np.arctan2(u[1], u[0]) % np.pi) / D_ANG)) % 16
    h, w = S["C"].shape[1:]
    pts = []
    for s in np.arange(0.0, L + 1e-9, step):
        q = p0 + s * u
        if not (0 <= q[0] <= w - 1 and 0 <= q[1] <= h - 1):
            break
        c = bilinear(S["C"], q[0], q[1])
        win = int(np.argmax(c))
        if c[k] < t_track or min((win - k) % 16, (k - win) % 16) > 1:
            break
        pts.append(q)
    return np.array(pts)


# ---------------------------------------------------------------- chapter 8, level 3: dips and contrast

DIP_SHAPE = (60, 200)
DIP_Y, DIP_X, DIP_LEN = 30.0, 110.0, 24
DIP_DEPTHS = np.linspace(0.0, 1.0, 21)
C_DIV = 2.0                                  # toy c for the divisive form, noise units
DIV_GATE = 6.0                               # toy gate for the divisive form: C > 6 noise units (a ratio has no
                                             # absolute floor; the subtractive threshold of 8 gates C implicitly)


@lru_cache(maxsize=None)
def _noise_sd():
    noise = N.add_noise(np.zeros((DIP_SHAPE[0] + 2 * PAD, DIP_SHAPE[1] + 2 * PAD)), NOISE, 11)
    return np.array([u.std() for u in N.simple_cells(noise, SIGMA, ELONG)])


@lru_cache(maxsize=None)
def dip_lobes(a_ref=0.15):
    """Noise-free: for each dip depth, (C, A-, A+) at the place of the strongest subtractive
    end near the dip on channel j = 0.  Everything up to here is linear in contrast, so
    other contrasts follow by scaling."""
    out = []
    for d in DIP_DEPTHS:
        segs = [((-40.0, DIP_Y), (DIP_X, DIP_Y), VESSEL_R, a_ref)]
        if d < 1:
            segs += [((DIP_X, DIP_Y), (DIP_X + DIP_LEN, DIP_Y), VESSEL_R, a_ref * (1 - d)),
                     ((DIP_X + DIP_LEN, DIP_Y), (DIP_SHAPE[1] + 60.0, DIP_Y), VESSEL_R, a_ref)]
        big = render(DIP_SHAPE, [("max", segs)])
        C = np.maximum(N.simple_cells(big, SIGMA, ELONG) / _noise_sd()[:, None, None], 0.0)
        A = lobes(C)
        yy = PAD + int(DIP_Y)
        xs = np.arange(PAD + int(DIP_X) - 20, PAD + int(DIP_X) + 8)
        out.append((C[0, yy, xs], A[16, yy, xs], A[0, yy, xs]))      # profiles along the axis
    C0 = float(np.maximum(N.simple_cells(render(DIP_SHAPE, [("max", [((-40.0, DIP_Y), (260.0, DIP_Y), VESSEL_R, a_ref)])]),
                                         SIGMA, ELONG)[0, PAD + int(DIP_Y), PAD + 40] / _noise_sd()[0], 0))
    return np.array(out), C0


def dip_response(depths_idx=None, strength=64.0, c=C_DIV, beta=1.0):
    """Subtractive and divisive end responses near a dip, for a vessel whose line cell
    gives `strength` noise units: arrays over DIP_DEPTHS."""
    L, C0 = dip_lobes()
    s = strength / C0
    Cc, Ab, Aa = (L[:, 0] * s, L[:, 1] * s, L[:, 2] * s)          # (depths, positions)
    gate = Cc > T_LOW
    sub = (np.maximum(np.minimum(Cc, Ab) - beta * Aa, 0) * gate).max(axis=1)
    div = (np.maximum(Ab - beta * Aa, 0) / (Ab + Aa + c) * (Cc > DIV_GATE)).max(axis=1)
    return sub, div


DIV_THR = 0.5                                # toy threshold for the divisive form


@lru_cache(maxsize=None)
def varying_vessel(seed=5, amp=0.3, c=C_DIV):
    """A vessel whose strength varies along it (55-145 %), ending at the right: subtractive
    and divisive end maps, and their ends."""
    shape = (70, 240)
    u = unit(PHIS[LINE_J])
    end = np.array([156.0, 34.0])
    big = render(shape, [("max", [(tuple(end - 260 * u), tuple(end), VESSEL_R, 0.10)])])
    h, w = big.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    s = -((x - PAD - end[0]) * u[0] + (y - PAD - end[1]) * u[1])       # distance back from the end
    m = 1 + amp * np.sin(2 * np.pi * s / 64 + 0.7) + 0.5 * amp * np.sin(2 * np.pi * s / 27 + 2.0)
    S = stack(big * m, seed=seed, extra=("div",), c_div=c)
    def inside(ends, m=6):
        return [e for e in ends if m <= e[0] <= shape[1] - 1 - m and m <= e[1] <= shape[0] - 1 - m]
    S["sub_ends"] = inside(end_peaks(S["E"], TOY_TAU_END, min_dist=12, min_turn_deg=361))
    S["div_ends"] = inside(end_peaks(S["Ediv"], DIV_THR, min_dist=12, min_turn_deg=361))
    S["end"], S["u"] = end, u
    return S


# ---------------------------------------------------------------- chapter 9: the corner analogy

@lru_cache(maxsize=None)
def corner_maps(shape=(130, 240), sigma=2.5):
    """Structure-tensor eigenvalues (Harris) of a toy image: a dark square and a dark bar.
    lambda_1 lights edges and corners, lambda_2 only corners."""
    from scipy.ndimage import gaussian_filter, sobel
    img = np.zeros(shape)
    img[30:100, 30:100] = 1.0
    img[55:75, 130:215] = 1.0
    img = gaussian_filter(img, 1.0)
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    Jxx, Jyy, Jxy = (gaussian_filter(g, sigma) for g in (gx * gx, gy * gy, gx * gy))
    tr, det = Jxx + Jyy, Jxx * Jyy - Jxy ** 2
    disc = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0))
    return img, tr / 2 + disc, tr / 2 - disc
