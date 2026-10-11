"""Teaching version of spec 4.10 (incremental grouping with attention) and
4.10b (linking facing label ends), for chapter 14.

One vessel at a time, strongest seed first.  A label a(k, y, x) spreads
from the seed through the vessel's own orientation sheet (or one either
side) along the line, wherever the attended evidence
    Ca[k] = C[k] * (1 + gamma * spread[k](a))
is above t_low; with a binary label that is a continuation threshold of
t_low / (1 + gamma) just ahead of the label (as built, spec 4.10).  At a
junction cell the label enters only the arm paired with the arriving arm.
Inhibition of return claims a square of 1.25 x the width around each grouped
ridge point, in its own sheet and one either side.

Images are noise-free; C is expressed in noise units of a reference white
noise (REF_NOISE OD), computed exactly from each kernel's L2 norm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

from . import neuro as N
from .grouping import line_cells

THETAS = N.THETAS
K = N.N_ORIENT
GAMMA, T_HIGH, T_LOW = 0.3, 8.0, 2.0
CLAIM_SHEETS = 2       # as built: one either side; the toy's broader tuning needs two
REF_NOISE = 0.01


# ------------------------------------------------------------------ evidence
def evidence(od, sigma=2.0, elong=3.0, ref=REF_NOISE):
    """Line evidence C[k] >= 0 in noise units of `ref` OD white noise.  sigma
    may be a tuple of size channels: the maximum over channels is kept."""
    out = None
    for sg in np.atleast_1d(sigma):
        U = line_cells(od, float(sg), elong)
        sd = np.array([np.sqrt((N.line_kernel(th, float(sg), elong) ** 2).sum()) for th in THETAS]) * ref
        c = np.maximum(U / sd[:, None, None], 0.0)
        out = c if out is None else np.maximum(out, c)
    return out


def orientation_peaks(C, rel=0.5, size=5):
    """Keep each pixel's tuning-curve peaks (local maxima over orientation, at
    least `rel` of its strongest) and one sheet either side; a crossing keeps
    both of its lobes."""
    up, dn = np.roll(C, -1, axis=0), np.roll(C, 1, axis=0)
    local = maximum_filter(C.max(axis=0), size=size)          # the strongest response nearby
    peak = (C >= up) & (C >= dn) & (C > 0) & (C >= rel * local[None])
    keep = peak | np.roll(peak, 1, axis=0) | np.roll(peak, -1, axis=0)
    return np.where(keep, C, 0.0)


def ridges(C, t=T_LOW):
    """Ridge tops: local maxima across each sheet's own normal, above t."""
    h, w = C.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    R = np.zeros(C.shape, bool)
    for k, th in enumerate(THETAS):
        nx, ny = -np.sin(th), np.cos(th)
        a = map_coordinates(C[k], [yy + ny, xx + nx], order=1, mode="nearest")
        b = map_coordinates(C[k], [yy - ny, xx - nx], order=1, mode="nearest")
        R[k] = (C[k] >= a) & (C[k] >= b) & (C[k] > t)
    return R


def _offsets(k, along=2, across=1):
    """The spread kernel of sheet k: steps along the line (ahead and behind)
    and one pixel across, as integer offsets (dx, dy)."""
    th = THETAS[k]
    t, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    offs = set()
    for s in range(-along, along + 1):
        for o in range(-across, across + 1):
            d = np.rint(s * t + o * n).astype(int)
            if d.any():
                offs.add((int(d[0]), int(d[1])))
    return sorted(offs)


OFFSETS = [_offsets(k) for k in range(K)]


# ---------------------------------------------------------------- junctions
@dataclass
class Junction:
    """A junction cell (spec 4.3) with its arms' outward directions (image
    angles, degrees) and widths; pairs come from good continuation (4.4)."""
    x: float
    y: float
    radius: float
    arms: list                     # [(angle_deg, width_px), ...]
    kind: str = "crossing"         # "crossing" or "fork"
    pairs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pairs = pair_arms(self.arms, self.kind)

    def arm_of(self, x, y, tol=35.0):
        a = np.degrees(np.arctan2(y - self.y, x - self.x))
        d = [abs((a - u + 180) % 360 - 180) for u, _ in self.arms]
        i = int(np.argmin(d))
        return i if d[i] <= tol else None


def cont(u_a, w_a, u_b, w_b):
    """Good continuation of arm a into arm b (spec 4.4, straight form):
    turn = angle between u_a and -u_b; 20 deg and ln 1.5 tolerances."""
    turn = abs((u_a - (u_b + 180.0) + 180) % 360 - 180)
    return float(np.exp(-turn ** 2 / (2 * 20.0 ** 2)) * np.exp(-np.log(w_a / w_b) ** 2 / (2 * np.log(1.5) ** 2)))


def pair_arms(arms, kind):
    n = len(arms)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = cont(arms[i][0], arms[i][1], arms[j][0], arms[j][1])
    pairs = {}
    if kind == "crossing" and n == 4:
        best, bp = -1, None
        for (a, b, c, d) in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            if S[a, b] * S[c, d] > best:
                best, bp = S[a, b] * S[c, d], ((a, b), (c, d))
        for a, b in bp:
            pairs[a], pairs[b] = b, a
    else:                                  # fork: the parent is the best-continuing pair; the branch pairs with nothing
        i, j = np.unravel_index(np.argmax(S), S.shape)
        pairs[int(i)], pairs[int(j)] = int(j), int(i)
    return pairs


# --------------------------------------------------------------- the grouping
def half_height_width(od, x, y, theta, reach=12.0, cap=20.0):
    """The OD's width at half height across orientation theta at (x, y),
    each side measured against its own baseline (spec 4.4)."""
    n = np.array([-np.sin(theta), np.cos(theta)])
    s = np.arange(-reach, reach + 0.01, 0.25)
    prof = map_coordinates(od, [y + s * n[1], x + s * n[0]], order=1, mode="nearest")
    c = len(s) // 2
    i0 = c - 8 + int(np.argmax(prof[c - 8:c + 9]))
    pk = prof[i0]
    half = 0.0
    for side in (1, -1):
        seg = prof[i0::side] if side > 0 else prof[i0::-1]
        base = seg.min()
        lvl = base + 0.5 * (pk - base)
        below = np.nonzero(seg < lvl)[0]
        half += (below[0] if below.size else len(seg)) * 0.25
    return float(np.clip(half, 2.0, cap))


@dataclass
class Vessel:
    vid: int
    seed: tuple                    # (k, y, x)
    width: float
    paths: list                    # two fronts, ahead and behind: [[(x, y, k, step), ...], [...]]
    queued: list = field(default_factory=list)   # branch seeds met at forks (x, y)
    gaps: list = field(default_factory=list)     # gap crossings: (front, index of the first point after it)
    stops: list = field(default_factory=list)    # why each front stopped

    def polyline(self):
        """The whole labelled ridge, from one end to the other: (n, 2) and the arrival steps."""
        a, b = self.paths
        pts = [p for p in a[::-1]] + [p for p in b[1:]]
        return np.array([(p[0], p[1]) for p in pts]), np.array([p[3] for p in pts])

    @property
    def length(self):
        xy, _ = self.polyline()
        return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def _bil(img, x, y):
    h, w = img.shape
    if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
        return -1.0
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    return float(img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
                 + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


class Grouper:
    """Attentive grouping on a noise-free toy (or a vesselscene crop)."""

    def __init__(self, od, junctions=(), gamma=GAMMA, t_high=T_HIGH, t_low=T_LOW, gaps=True,
                 sigma=2.0, elong=3.0, ref=REF_NOISE, max_vessels=80, min_len=0.0, border=6, compete=True):
        self.od = od
        C = evidence(od, sigma, elong, ref)
        self.C = orientation_peaks(C) if compete else C
        self.R = ridges(self.C, t_high)
        self.junctions = list(junctions)
        self.gamma, self.t_high, self.t_low, self.gaps = gamma, t_high, t_low, gaps
        self.claimed = np.zeros(self.C.shape, bool)
        self.vessels: list[Vessel] = []
        self.max_vessels, self.min_len, self.border = max_vessels, min_len, border
        self.fork_claims = []

    def next_seed(self):
        cand = self.R & ~self.claimed
        b = self.border
        cand[:, :b], cand[:, -b:], cand[:, :, :b], cand[:, :, -b:] = False, False, False, False
        if not cand.any():
            return None
        k, y, x = np.unravel_index(np.argmax(np.where(cand, self.C, -1)), self.C.shape)
        return int(k), int(y), int(x)

    def _free(self, k, x, y):
        h, w = self.od.shape
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < w and 0 <= yi < h and not self.claimed[k % K, yi, xi]

    def _step(self, x, y, k, head, thr):
        """One step of the label along its sheet (or one either side): the best
        attended evidence 1-2 px ahead, within a pixel across."""
        best = None
        self._blocked_by_claim = False
        for kk in (k - 1, k, k + 1):
            kk %= K
            t = np.array([np.cos(THETAS[kk]), np.sin(THETAS[kk])])
            if t @ head < 0:
                t = -t
            n = np.array([-t[1], t[0]])
            for d in (1.0, 1.5):
                for o in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    qx, qy = x + d * t[0] + o * n[0], y + d * t[1] + o * n[1]
                    c = _bil(self.C[kk], qx, qy)
                    if c > thr and not self._free(kk, qx, qy):
                        self._blocked_by_claim = True
                    if c > thr and self._free(kk, qx, qy) and (best is None or c > best[0]):
                        best = (c, qx, qy, kk, t)
        return best

    def _gap(self, x, y, k, head, width, cone=20.0):
        """Where the line breaks, look further along the heading (up to 3 vessel
        widths, in a 20 degree cone) and cross if the vessel resumes."""
        for s in np.arange(3.0, min(3.0 * width, 30.0) + 0.01, 1.0):
            for da in (0, -5, 5, -10, 10, -15, 15, -20, 20):
                if abs(da) > cone:
                    continue
                a = np.radians(da)
                d = np.array([head[0] * np.cos(a) - head[1] * np.sin(a), head[0] * np.sin(a) + head[1] * np.cos(a)])
                qx, qy = x + s * d[0], y + s * d[1]
                for kk in (k, k - 1, k + 1):
                    kk %= K
                    if _bil(self.C[kk], qx, qy) > self.t_high and self._free(kk, qx, qy):
                        return qx, qy, kk
        return None

    def _front(self, x, y, k, head, width, step0, gain, queued, gaps, fid, max_steps=900):
        thr = self.t_low / (1 + self.gamma) if gain else self.t_low
        pts = [(x, y, k, step0)]
        inside = {}                       # junction index -> exit direction
        done = set()
        reason = "end"
        for it in range(1, max_steps):
            # junction cells: inside a junction's disc the label continues only into the paired arm
            jhit = None
            for ji, j in enumerate(self.junctions):
                if ji in done:
                    continue
                if np.hypot(x - j.x, y - j.y) < j.radius:
                    jhit = (ji, j)
                    break
            if jhit is not None:
                ji, j = jhit
                done.add(ji)
                a = j.arm_of(x - head[0] * 3, y - head[1] * 3)
                if a is None or a not in j.pairs:          # arrived on a fork's branch: stop here
                    reason = "fork branch"
                    break
                b = j.pairs[a]
                u = np.array([np.cos(np.radians(j.arms[b][0])), np.sin(np.radians(j.arms[b][0]))])
                if j.kind == "fork":
                    self.fork_claims.append((j.x, j.y, j.radius + 2.0))
                    for i, (ang, _) in enumerate(j.arms):
                        if i not in j.pairs:
                            r = j.radius + 10
                            queued.append((j.x + r * np.cos(np.radians(ang)), j.y + r * np.sin(np.radians(ang))))
                # walk through the disc along the paired arm
                kb = int(np.argmin([abs(((np.degrees(th) - j.arms[b][0]) + 90) % 180 - 90) for th in THETAS]))
                x, y = j.x, j.y
                pts.append((x, y, kb, step0 + it))
                while np.hypot(x - j.x, y - j.y) < j.radius + 1.0:
                    x, y = x + u[0], y + u[1]
                    pts.append((x, y, kb, step0 + it))
                head, k = u, kb
                continue
            h, w = self.od.shape
            if not (2 <= x <= w - 3 and 2 <= y <= h - 3):
                reason = "border"
                break
            nxt = self._step(x, y, k, head, thr)
            if nxt is None:
                g = self._gap(x, y, k, head, width) if self.gaps else None
                if g is None:
                    reason = "claimed" if self._blocked_by_claim else "weak"
                    break
                x, y, k = g
                gaps.append((fid, len(pts)))
                pts.append((x, y, k, step0 + it))
                continue
            _, x, y, k, t = nxt
            head = t
            pts.append((x, y, k, step0 + it))
        return pts, reason

    def spread(self, seed, vid, gain=True):
        k0, y0, x0 = seed
        width = half_height_width(self.od, x0, y0, THETAS[k0])
        t = np.array([np.cos(THETAS[k0]), np.sin(THETAS[k0])])
        queued, gaps, paths, stops = [], [], [], []
        # the seed's own pixel is claimed while its two fronts run, so they do not meet head-on
        for fid, head in enumerate((t, -t)):
            pts, why = self._front(float(x0), float(y0), k0, head, width, 0, gain, queued, gaps, fid)
            paths.append(pts)
            stops.append(why)
        return Vessel(vid, seed, width, paths, queued, gaps, stops)

    def claim(self, v: Vessel):
        """Inhibition of return: a square of 1.25 x the width around each
        grouped point, in its own sheet and one either side."""
        Kn, h, w = self.C.shape
        half = 0.625 * v.width
        for path in v.paths:
            for (x, y, k, _) in path:
                x0, x1 = max(0, int(np.floor(x - half))), min(w, int(np.ceil(x + half)) + 1)
                y0, y1 = max(0, int(np.floor(y - half))), min(h, int(np.ceil(y + half)) + 1)
                for kk in range(k - CLAIM_SHEETS, k + CLAIM_SHEETS + 1):
                    self.claimed[kk % Kn, y0:y1, x0:x1] = True
        yy, xx = np.mgrid[0:h, 0:w]
        for (x, y, r) in self.fork_claims:          # a fork's merged lumen goes to the parent that passed through
            self.claimed[:, np.hypot(xx - x, yy - y) < r] = True
        self.fork_claims = []

    def run(self, gain=True):
        vid = 0
        while len(self.vessels) < self.max_vessels:
            seed = self.next_seed()
            if seed is None:
                break
            v = self.spread(seed, vid + 1, gain)
            self.claim(v)
            k0, y0, x0 = seed
            self.claimed[:, y0, x0] = True
            if v.length >= self.min_len:
                vid += 1
                self.vessels.append(v)
        return self.vessels


# ------------------------------------------------------------------ the toy
TOY_SHAPE = (150, 240)


def _polyline_od(shape, pts, r, amp):
    """OD of a vessel along a polyline, with amplitude amp(s) along arc length."""
    x, y = N.grid(shape)
    best = np.full(shape, np.inf)
    s_at = np.zeros(shape)
    s0 = 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d, t = N.dist_to_segment(x, y, p0, p1)
        L = float(np.hypot(*(np.asarray(p1) - p0)))
        m = d < best
        best[m] = d[m]
        s_at[m] = s0 + t[m] * L
        s0 += L
    return N.chord(best, r, 1.0) * amp(s_at), s_at


def toy_paths():
    xs = np.linspace(-12.0, 252.0, 89)
    ya = 70.0 + 16.0 * np.sin((xs - 10.0) / 260.0 * 2 * np.pi * 0.8)
    A = np.stack([xs, ya], 1)
    B = np.array([[64.0, -12.0], [78.0, 40.0], [92.0, 100.0], [104.0, 162.0]])
    # the fork on A at x = 182: the branch leaves at about 45 degrees, curving gently
    i = int(np.argmin(np.abs(xs - 182.0)))
    F = A[i]
    tA = A[i + 1] - A[i - 1]
    tA /= np.linalg.norm(tA)
    a0 = np.arctan2(tA[1], tA[0]) + np.radians(45.0)
    s = np.linspace(0, 1, 30)
    ang = a0 + np.radians(12.0) * s
    step = 3.6
    Cb = [F]
    for a in ang:
        Cb.append(Cb[-1] + step * np.array([np.cos(a), np.sin(a)]))
    return dict(A=A, B=B, C=np.array(Cb), fork=F, fork_dir=tA)


def _dip(s, s0, s1, depth):
    """A smooth dip of the given depth (0..1) between arc lengths s0 and s1."""
    u = np.clip((s - s0) / (s1 - s0), 0, 1)
    ramp = np.clip(np.minimum(u, 1 - u) / 0.22, 0, 1)          # raised-cosine ramps, a flat floor between
    bump = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return 1.0 - depth * bump


@lru_cache(maxsize=None)
def toy(dip_depth=None, weak=(128.0, 168.0)):
    """The chapter-14 toy: vessel A (strong) crossed by B, a weak stretch on A,
    then a fork where A continues and branch C leaves."""
    P = toy_paths()
    shape = TOY_SHAPE
    xa = P["A"][:, 0]
    # arc length on A of the weak stretch's ends
    segs = np.r_[0, np.cumsum(np.hypot(*np.diff(P["A"], axis=0).T))]
    s0, s1 = np.interp(weak, xa, segs)
    depth = 0.983 if dip_depth is None else dip_depth
    odA, sA = _polyline_od(shape, P["A"], 3.5, lambda s: 0.15 * _dip(s, s0, s1, depth))
    odB, _ = _polyline_od(shape, P["B"], 3.0, lambda s: 0.12 + 0 * s)
    odC, _ = _polyline_od(shape, P["C"], 2.6, lambda s: 0.11 + 0 * s)
    od = gaussian_filter(np.maximum(odA, odC) + odB, 1.0)
    # junction cells (positions from the construction; pairs from good continuation)
    xc, yc = _intersect(P["A"], P["B"])
    angA = np.degrees(np.arctan2(*(_tangent(P["A"], xc)[::-1])))
    angB = np.degrees(np.arctan2(P["B"][2, 1] - P["B"][1, 1], P["B"][2, 0] - P["B"][1, 0]))
    cross = Junction(xc, yc, 7.0, [(angA, 7.0), (angA + 180, 7.0), (angB, 6.0), (angB + 180, 6.0)], "crossing")
    F, tA = P["fork"], P["fork_dir"]
    angF = np.degrees(np.arctan2(tA[1], tA[0]))
    angC = np.degrees(np.arctan2(*(P["C"][3] - P["C"][0])[::-1]))
    fork = Junction(F[0], F[1], 7.0, [(angF + 180, 7.0), (angF, 7.0), (angC, 5.2)], "fork")
    return dict(od=od, paths=P, junctions=(cross, fork), weak_s=(s0, s1), sA=sA, depth=depth)


def _tangent(P, x):
    i = int(np.argmin(np.abs(P[:, 0] - x)))
    i = min(max(i, 1), len(P) - 2)
    t = P[i + 1] - P[i - 1]
    return t / np.linalg.norm(t)


def _intersect(A, B):
    for p0, p1 in zip(A[:-1], A[1:]):
        for q0, q1 in zip(B[:-1], B[1:]):
            d1, d2 = p1 - p0, q1 - q0
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                continue
            t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / den
            u = ((q0[0] - p0[0]) * d1[1] - (q0[1] - p0[1]) * d1[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                p = p0 + t * d1
                return float(p[0]), float(p[1])
    raise ValueError("no crossing")


@lru_cache(maxsize=None)
def toy_grouping(gain=True, gaps=True):
    T = toy()
    g = Grouper(T["od"], T["junctions"], gaps=gaps, min_len=25.0)
    g.run(gain=gain)
    return g


# ------------------------------------------------------------- gap crossing
GAP_SHAPE = (90, 220)


@lru_cache(maxsize=None)
def gap_toy(gap=28.0, r=5.0, a=0.04):
    """A vessel broken by a gap the line cells cannot bridge, and a distractor
    whose end lies outside the label's 20 degree cone."""
    x, y = N.grid(GAP_SHAPE)
    xa, xb = 100.0 - gap / 2, 100.0 + gap / 2
    d1, _ = N.dist_to_segment(x, y, (-10.0, 46.0), (xa, 44.0))
    d2, _ = N.dist_to_segment(x, y, (xb, 43.6), (230.0, 41.0))
    d3, _ = N.dist_to_segment(x, y, (xb + 2.0, 70.0), (230.0, 100.0))
    od = gaussian_filter(N.chord(d1, r, a) + N.chord(d2, r, a) + N.chord(d3, r, a), 1.0)
    g = Grouper(od, (), gaps=True, min_len=20.0)
    g.run()
    g0 = Grouper(od, (), gaps=False, min_len=20.0)
    g0.run()
    return dict(od=od, g=g, g_nogap=g0, ends=((xa, 44.0), (xb, 43.6)), distractor=(xb + 2.0, 70.0))


# --------------------------------------------------------- a vesselscene crop
@lru_cache(maxsize=None)
def real_crop(x0=0, y0=60, w=300, h=180, ref=0.01, sigma=(1.5, 2.5, 4.0)):
    """This chapter's grouping on a crop of the default vesselscene average."""
    from . import vsdata
    S = vsdata.load()
    od = np.nan_to_num(S.od[y0:y0 + h, x0:x0 + w])
    g = Grouper(od, (), ref=ref, min_len=20.0, max_vessels=200, sigma=sigma)
    g.run()
    return g
