"""Small test scenes for the closed-form renderer (64 x 64 px at k = 3).

Geometry is written in px and converted to d_c; every scene returns
(graph, optics, note).  Free ends (inlets / outlets) lie far outside the
frame and its halo margin, so only the feature under test is in view."""
from __future__ import annotations

import numpy as np

from vesselscene.graph import VesselGraph
from vesselscene.render import Optics

K = 3.0
SHAPE = (64, 64)


def optics_for_blur(s: float, **kw) -> Optics:
    """Optics with blur s px at every depth (no defocus or scattering)."""
    return Optics(s0_px=s, defocus_px_per_dc=0.0, scatter_px_per_dc=0.0, **kw)


def turtle(parts, step=0.05, p0=(0.0, 0.0), h0=0.0):
    """Path (px) from ('line', L) and ('arc', radius, degrees) parts."""
    p, h, out = np.asarray(p0, float), np.deg2rad(h0), [np.asarray(p0, float)]
    for part in parts:
        if part[0] == "line":
            for _ in range(int(part[1] / step)):
                p = p + step * np.array([np.cos(h), np.sin(h)])
                out.append(p)
        else:
            rad, deg = part[1], np.deg2rad(part[2])
            n = int(abs(deg) * rad / step)
            dh = deg / n
            for _ in range(n):
                h += dh
                p = p + rad * abs(dh) * np.array([np.cos(h), np.sin(h)])
                out.append(p)
    return np.array(out)


def ray(p0, deg, length, curv=0.0, step=0.05):
    n = int(length / step)
    h = np.deg2rad(deg) + curv * np.arange(n) * step
    d = step * np.stack([np.cos(h), np.sin(h)], 1)
    return np.vstack([p0, np.asarray(p0, float) + np.cumsum(d, 0)])


def add_free(g: VesselGraph, path_px, r_px, z=10.0, h=1.0, spacing=None, z_end=None):
    """A vessel between two free ends along path_px (px)."""
    path = np.asarray(path_px, float) / K
    a = g.add_node(path[0], depth=z, kind="inlet", pressure=1.0)
    b = g.add_node(path[-1], depth=z if z_end is None else z_end, kind="outlet", pressure=0.0)
    return g.add_vessel(a, b, path=path, radius=r_px / K, hct=h, spacing=spacing)


def star(node_px, arms, z=10.0):
    """Vessels meeting at one node.  arms: (deg, r_px, into_node, h, curv)."""
    g = VesselGraph()
    N = g.add_node(np.asarray(node_px, float) / K, depth=z)
    for deg, r, into, h, curv in arms:
        path = ray(node_px, deg, 110, curv) / K
        far = g.add_node(path[-1], depth=z, kind="inlet" if into else "outlet")
        if into:
            g.add_vessel(far, N, path=path[::-1], radius=r / K, hct=h)
        else:
            g.add_vessel(N, far, path=path, radius=r / K, hct=h)
    return g


# ---------------------------------------------------------------- scenes
def straight():
    g = VesselGraph()
    add_free(g, ray((-60, 20), 18, 190), 1.5)
    return g, Optics(), "straight, r 1.5 px, default optics"


def gentle():
    g = VesselGraph()
    x = np.arange(-60, 124, 0.05)
    add_free(g, np.stack([x, 32 + 4 * np.sin(2 * np.pi * (x - 17) / 60)], 1), 1.5)
    return g, optics_for_blur(1.5), "gentle wiggle, rho ~ 23 px"


def _rot(xy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return xy @ np.array([[c, -s], [s, c]]).T


def tight_bend():
    rho, turn = 4.0, 135.0
    xy = turtle([("line", 100), ("arc", rho, turn), ("line", 100)])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    i = np.argmin(np.abs(s - (100 + rho * np.deg2rad(turn) / 2)))
    t = xy[i + 1] - xy[i - 1]
    xy = _rot(xy - xy[i], -np.arctan2(t[1], t[0]))
    if xy[i + 40, 1] < 0:
        xy[:, 1] *= -1
    g = VesselGraph()
    add_free(g, xy + [32, 22], 1.5, spacing=0.4)
    return g, optics_for_blur(2.0), "tight bend, rho 4 px, 135 deg, r 1.5, s 2"


def hairpin():
    gap = 5.0
    xy = turtle([("line", 110), ("arc", gap / 2, 180), ("line", 110)]) - [110, 0] + [46, 32 - gap / 2]
    g = VesselGraph()
    add_free(g, xy, 1.5, spacing=0.4)
    return g, optics_for_blur(2.0), "hairpin, legs 5 px apart, r 1.5, s 2"


def coil():
    b, c = 5.5, 2.3
    t = np.arange(-7 * np.pi, 9 * np.pi, 0.002)
    xy = np.stack([c * t + b * np.sin(t), -b * np.cos(t)], 1)
    xy = xy + [32 - c * 2 * np.pi, 32]
    g = VesselGraph()
    add_free(g, xy, 1.0, spacing=0.4)
    return g, optics_for_blur(1.2), "coil that crosses itself, r 1.0, s 1.2"


def fork():
    g = star((32, 30), [(90, 2.0, True, 1.0, 0.002), (235, 1.6, False, 1.0, -0.004),
                        (305, 1.4, False, 1.0, 0.004)])
    return g, optics_for_blur(1.5), "fork: parent + 2 children at one node"


def joint70():
    g = star((34, 34), [(180, 1.8, True, 1.0, 0.0), (290, 1.8, False, 1.0, 0.0)])
    return g, optics_for_blur(1.5), "70 deg joint (degree-2 node)"


def unequal():
    g = star((32, 32), [(100, 3.0, True, 1.0, 0.0), (230, 1.0, False, 0.6, 0.003),
                        (320, 2.2, False, 1.0, -0.002)])
    return g, optics_for_blur(1.4), "unequal radii (3.0 / 1.0 / 2.2 px) and haematocrit at a node"


def crossing():
    g = VesselGraph()
    add_free(g, ray((-70, 30), 3, 200), 3.0, z=60.0)
    d60 = np.array([np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
    add_free(g, ray(np.array([32, 34]) - 110 * d60, 60, 220), 1.0, z=5.0)
    opt = Optics(s0_px=1.0, z_focus=5.0, defocus_px_per_dc=0.07, scatter_px_per_dc=0.0)
    return g, opt, "crossing at different depths (deep s ~3.6 px, shallow s 1.0)"


def touching():
    g = VesselGraph()
    add_free(g, ray((-70, 30), 0, 200), 2.0, z=10.0)
    add_free(g, ray((-70, 33.5), 0, 200), 1.5, z=30.0)
    return g, Optics(), "parallel vessels touching, at different depths"


def depth_ramp():
    g = VesselGraph()
    x = np.arange(-50, 114, 0.05)
    add_free(g, np.stack([x, 30 + 0.004 * (x - 32) ** 2], 1), 2.0, z=0.0, z_end=80.0)
    return g, Optics(), "depth ramp 0 -> 80 d_c: blur 1.5 -> 5 px, f 0.05 -> 0.75"


def riser():
    """A perforator: flat at 5 d_c, then diving to 120 d_c over ~40 px (slope
    up to ~6): blur 1.4 -> 7 px, f 0.12 -> 0.9, chord longer by up to ~6x."""
    g = VesselGraph()
    tau = np.linspace(0, 1, 256)
    x = np.clip((tau - 0.45) / 0.25, 0, 1)
    z = 5.0 + 115.0 * x * x * (3 - 2 * x)
    path = ray((-60, 28), 4, 180) / K
    a = g.add_node(path[0], depth=z[0], kind="inlet")
    b = g.add_node(path[-1], depth=z[-1], kind="outlet")
    g.add_vessel(a, b, path=path, radius=1.8 / K, depth=z)
    return g, Optics(), "riser: steep dive 5 -> 120 d_c (perforator), inclined chord"


def nonlinear():
    g = star((32, 32), [(95, 4.0, True, 1.0, 0.0), (225, 2.5, False, 1.0, 0.002),
                        (320, 3.2, False, 1.0, -0.002)], z=30.0)
    opt = Optics(mu=0.4, f_spectral=0.35, l_bypass=40.0)
    return g, opt, "nonlinear contrast: thick fork, mu 0.4, f_spectral 0.35, f(z=30) ~ 0.74"


def leaving():
    g = VesselGraph()
    add_free(g, ray((20, -60), 75, 200), 2.0)
    add_free(g, ray((-60, 40), -8, 200), 1.2)
    return g, Optics(), "vessels leaving the frame on every side"


def merged():
    """Parent -> N1 -> (child a, short link) ; link -> N2 -> (children b, c)."""
    g = VesselGraph()
    n1 = np.array([26.0, 34.0])
    n2 = n1 + 7.0 * np.array([np.cos(np.deg2rad(-20)), np.sin(np.deg2rad(-20))])
    N1 = g.add_node(n1 / K, depth=10)
    N2 = g.add_node(n2 / K, depth=10)
    p = ray(n1, 180, 110) / K
    a = g.add_node(p[-1], depth=10, kind="inlet")
    g.add_vessel(a, N1, path=p[::-1], radius=2.0 / K)
    q = ray(n1, 120, 110) / K
    b = g.add_node(q[-1], depth=10, kind="outlet")
    g.add_vessel(N1, b, path=q, radius=1.4 / K)
    g.add_vessel(N1, N2, path=np.linspace(n1, n2, 50) / K, radius=1.7 / K)
    for deg, r in ((-75, 1.2), (25, 1.4)):
        q = ray(n2, deg, 110) / K
        c = g.add_node(q[-1], depth=10, kind="outlet")
        g.add_vessel(N2, c, path=q, radius=r / K)
    return g, optics_for_blur(1.5), "two forks joined by a 7 px vessel (one merged cluster)"


SCENES = [straight, gentle, tight_bend, hairpin, coil, fork, joint70, unequal, crossing, touching,
          depth_ramp, riser, nonlinear, leaving, merged]


# ---------------------------------------------------------------- regression scenes (review defects)
def _fork_at(g, n_px, arms, z):
    """Arms (deg, r_px, into) from an existing node at n_px (px), free far ends."""
    N = g.add_node(np.asarray(n_px, float) / K, depth=z)
    for deg, r, into in arms:
        path = ray(n_px, deg, 110) / K
        far = g.add_node(path[-1], depth=z, kind="inlet" if into else "outlet")
        if into:
            g.add_vessel(far, N, path=path[::-1], radius=r / K)
        else:
            g.add_vessel(N, far, path=path, radius=r / K)
    return N


def steep_link():
    """Two forks joined by a short vessel (9 px, 3 d_c) that climbs from 10
    to 17 d_c between them: its incline factor is 1 at both nodes and the
    cap of 2 in the middle, and it is shorter than its two junction radii (a
    whole stub).  The first version took the overlap staircase of such a
    vessel from its middle piece: ~20 % too light at the junctions."""
    g = VesselGraph()
    n1, n2 = np.array([27.0, 32.0]), np.array([36.0, 32.0])
    N1 = _fork_at(g, n1, [(180, 2.2, True), (120, 1.6, False)], 10.0)
    N2 = _fork_at(g, n2, [(-60, 1.5, False), (40, 1.8, False)], 17.0)
    tau = np.linspace(0, 1, 32)
    z = 10.0 + 7.0 * tau * tau * (3 - 2 * tau)
    g.add_vessel(N1, N2, path=np.linspace(n1, n2, 40) / K, radius=2.0 / K, depth=z)
    opt = Optics(s0_px=1.4, z_focus=10.0, defocus_px_per_dc=0.08, scatter_px_per_dc=0.0)
    return g, opt, "two forks joined by a 9 px link climbing 10 -> 17 d_c (incline 1 -> 2 -> 1)"


def sibling_inside():
    """A thin vessel leaves a wide one's node at 35 deg, then bends to run
    inside the wide one's lumen (2.5 px off its axis), at the same depth, to
    the frame edge, far beyond the angle formula's junction radius.  One
    blood volume: the renderer must union it all the way (the first version
    added it beyond R).  (graph.check flags such anatomy; the generator
    removes it, but the renderer must still be right.)"""
    g = VesselGraph()
    n = np.array([14.0, 32.0])
    N = g.add_node(n / K, depth=10.0)
    p = ray(n, 180, 110) / K
    a = g.add_node(p[-1], depth=10.0, kind="inlet")
    g.add_vessel(a, N, path=p[::-1], radius=5.5 / K)
    q = ray(n, 0, 150) / K
    b = g.add_node(q[-1], depth=10.0, kind="outlet")
    g.add_vessel(N, b, path=q, radius=5.0 / K)
    xy = turtle([("arc", 14.0, 35.0), ("line", 130)], p0=n, h0=-35.0)
    c = g.add_node(xy[-1] / K, depth=10.0, kind="outlet")
    g.add_vessel(N, c, path=xy / K, radius=1.5 / K)
    return g, optics_for_blur(1.5), "thin vessel running inside a wide sibling beyond R"


def short_link(gap_px: float = 1.5):
    """Two forks joined by a vessel only gap_px long (the review's case)."""
    g = VesselGraph()
    n1 = np.array([28.0, 32.0])
    n2 = n1 + gap_px * np.array([1.0, 0.0])
    N1, N2 = g.add_node(n1 / K, depth=10), g.add_node(n2 / K, depth=10)
    p = ray(n1, 180, 110) / K
    a = g.add_node(p[-1], depth=10, kind="inlet")
    g.add_vessel(a, N1, path=p[::-1], radius=2.5 / K)
    q = ray(n1, 110, 110) / K
    b = g.add_node(q[-1], depth=10, kind="outlet")
    g.add_vessel(N1, b, path=q, radius=1.6 / K)
    g.add_vessel(N1, N2, path=np.linspace(n1, n2, 20) / K, radius=2.0 / K)
    for deg, r in ((-60, 1.3), (30, 1.7)):
        q = ray(n2, deg, 110) / K
        c = g.add_node(q[-1], depth=10, kind="outlet")
        g.add_vessel(N2, c, path=q, radius=r / K)
    return g, optics_for_blur(1.5), f"two forks joined by a {gap_px} px vessel"


def dive_at_fork():
    """A fork whose child dives from 10 to 40 d_c right after the node (its
    incline factor reaches the cap of 2 within the junction's stubs)."""
    g = VesselGraph()
    n = np.array([30.0, 34.0])
    N = g.add_node(n / K, depth=10.0)
    p = ray(n, 90, 110) / K
    a = g.add_node(p[-1], depth=10.0, kind="inlet")
    g.add_vessel(a, N, path=p[::-1], radius=2.2 / K)
    q = ray(n, 220, 110) / K
    b = g.add_node(q[-1], depth=10.0, kind="outlet")
    g.add_vessel(N, b, path=q, radius=1.8 / K)
    q = ray(n, 320, 110) / K
    c = g.add_node(q[-1], depth=40.0, kind="outlet")
    tau = np.linspace(0, 1, 64)
    x = np.clip(tau / 0.12, 0, 1)
    g.add_vessel(N, c, path=q, radius=1.6 / K, depth=10.0 + 30.0 * x * x * (3 - 2 * x))
    opt = Optics(s0_px=1.4, z_focus=10.0, defocus_px_per_dc=0.06, scatter_px_per_dc=0.0)
    return g, opt, "fork with a child diving 10 -> 40 d_c just after the node"


# ---------------------------------------------------------------- crossings composited by depth (v2)
def strong_optics(s0=1.2, **kw) -> Optics:
    """Optics where compositing matters: half the light unabsorbable
    (f_spectral 0.5), a short bypass length (30 d_c), strong absorption."""
    base = dict(s0_px=s0, z_focus=10.0, defocus_px_per_dc=0.06, scatter_px_per_dc=0.0, mu=0.6, f_spectral=0.5,
                l_bypass=30.0)
    base.update(kw)
    return Optics(**base)


def cross_sharp_over_blurred():
    """A sharp thin vessel (5 d_c) over a wide blurred deep one (40 d_c): the
    v1 review's dark beads."""
    g = VesselGraph()
    add_free(g, ray((-70, 28), 4, 200), 6.0, z=40.0)
    d = np.array([np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70))])
    add_free(g, ray(np.array([30, 32]) - 110 * d, 70, 220), 1.5, z=5.0)
    return g, strong_optics(), "sharp thin over wide blurred deep vessel, strong compositing"


def cross_shallow():
    """Two medium vessels crossing at 14 deg at 8 and 22 d_c: a long overlap
    (the 'spindle')."""
    g = VesselGraph()
    add_free(g, ray((-70, 30), 0, 200), 2.5, z=8.0)
    d = np.array([np.cos(np.deg2rad(14)), np.sin(np.deg2rad(14))])
    add_free(g, ray(np.array([32, 32]) - 110 * d, 14, 220), 2.0, z=22.0)
    return g, strong_optics(), "shallow (14 deg) crossing at different depths"


def cross_triple():
    """Three vessels at 6, 20 and 45 d_c crossing near one point: three layers."""
    g = VesselGraph()
    for deg, r, z, c in ((5, 2.5, 20.0, (32, 30)), (65, 1.5, 6.0, (30, 32)), (125, 4.0, 45.0, (33, 33))):
        d = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
        add_free(g, ray(np.asarray(c, float) - 110 * d, deg, 220), r, z=z)
    return g, strong_optics(), "three vessels at three depths crossing"


def cross_at_fork():
    """A shallow vessel crossing a fork (one blood volume) just beside its node."""
    g = star((34, 30), [(95, 2.4, True, 1.0, 0.0), (230, 1.8, False, 1.0, 0.003), (320, 2.0, False, 1.0, -0.002)],
             z=25.0)
    d = np.array([np.cos(np.deg2rad(-20)), np.sin(np.deg2rad(-20))])
    add_free(g, ray(np.array([30, 36]) - 110 * d, -20, 220), 1.4, z=6.0)
    return g, strong_optics(), "shallow vessel crossing a fork beside its node"


def cross_siblings():
    """Second-ring neighbours that cross: a fork at N1 (parent, child a, link)
    and one at N2 (link, children b, c), 8 px apart; child a runs on at 12
    d_c, child b curves back across it diving to 32 d_c.  a and b share no
    node but the link: one blood volume where their depths agree (near the
    forks), composited where b has dived."""
    g = VesselGraph()
    n1 = np.array([6.0, 14.0])
    n2 = n1 + np.array([8.0, 0.0])
    N1, N2 = g.add_node(n1 / K, depth=12.0), g.add_node(n2 / K, depth=12.0)
    p = ray(n1, 180, 110) / K
    a = g.add_node(p[-1], depth=12.0, kind="inlet")
    g.add_vessel(a, N1, path=p[::-1], radius=2.4 / K)
    g.add_vessel(N1, N2, path=np.linspace(n1, n2, 30) / K, radius=2.2 / K)
    q = ray(n1, 40, 150) / K                                       # child a: down-right, stays at 12 d_c
    b = g.add_node(q[-1], depth=12.0, kind="outlet")
    g.add_vessel(N1, b, path=q, radius=1.6 / K)
    q = ray(n2, -10, 150) / K                                      # child c: away
    c = g.add_node(q[-1], depth=12.0, kind="outlet")
    g.add_vessel(N2, c, path=q, radius=1.5 / K)
    q = ray(n2, 15, 150, curv=0.02) / K                            # child b: curls down across child a, diving
    tau = np.linspace(0, 1, 64)
    x = np.clip((tau - 0.04) / 0.12, 0, 1)
    d = g.add_node(q[-1], depth=32.0, kind="outlet")
    g.add_vessel(N2, d, path=q, radius=1.8 / K, depth=12.0 + 20.0 * x * x * (3 - 2 * x))
    return g, strong_optics(), "second-ring neighbours (forks joined by a link) crossing at different depths"


def cross_dive_at_fork():
    """dive_at_fork under strong compositing: the diving child's stub and the
    fork's overlap polygons must stay in one blood volume."""
    g, _, _ = dive_at_fork()
    return g, strong_optics(s0=1.4), "fork with a diving child, strong compositing"


def cross_shared_node():
    """A tributary that crosses its own collecting vessel far from their
    confluence: b leaves the node N (outside the frame, right) beside a,
    dives from 8 to 30 d_c, curls back and crosses a about 80 px (27 d_c)
    from N, beyond N's junction.  There they are two blood volumes and are
    composited, not added (v2 correctness review, defect 2: a shared node
    anywhere made one volume)."""
    from scipy.interpolate import CubicSpline
    g = VesselGraph()
    n_px = np.array([110.0, 32.0])
    N = g.add_node(n_px / K, depth=8.0)
    p = ray((-80.0, 32.0), 0, 190.0)
    a = g.add_node(p[0] / K, depth=8.0, kind="inlet")
    g.add_vessel(a, N, path=p / K, radius=2.0 / K, depth=8.0)
    knots = np.array([n_px, [80.0, 24.0], [52.0, 16.0], [32.0, 32.0], [22.0, 60.0], [14.0, 110.0]])
    t = np.linspace(0.0, len(knots) - 1.0, 400)
    q = CubicSpline(np.arange(len(knots)), knots)(t)
    o = g.add_node(q[-1] / K, depth=30.0, kind="outlet")
    tau = np.linspace(0, 1, 64)
    x = np.clip(tau / 0.2, 0, 1)
    g.add_vessel(N, o, path=q / K, radius=1.8 / K, depth=8.0 + 22.0 * x * x * (3 - 2 * x))
    return g, strong_optics(), "a tributary crossing its own collecting vessel far from their confluence"


COMPOSITE_SCENES = [cross_sharp_over_blurred, cross_shallow, cross_triple, cross_at_fork, cross_siblings,
                    cross_dive_at_fork, cross_shared_node]


# ---------------------------------------------------------------- wide vessels at high OD (v2: ribs, node seams)
WIDE_SHAPE = (128, 128)


def wide_optics() -> Optics:
    """High OD: in-band chord OD ~5 through a 13 d_c lumen, 30 % unabsorbable (OD up to 1.2)."""
    return Optics(s0_px=1.5, defocus_px_per_dc=0.0, scatter_px_per_dc=0.0, mu=0.4, f_spectral=0.3,
                  l_bypass=float("inf"))


def wide_bend():
    """r 20 px on a bend whose curvature rises smoothly to 1/70 px (kappa r
    up to 0.29) and falls again: the v1 review's ribs (transverse stripes at
    the 7-8 px piece joints) and the curvature Jacobian at high OD."""
    s = np.arange(0.0, 420.0, 0.05)
    kap = (1.0 / 70.0) * np.exp(-0.5 * ((s - 210.0) / 55.0) ** 2)
    h = np.cumsum(kap) * 0.05
    xy = np.cumsum(0.05 * np.stack([np.cos(h), np.sin(h)], 1), 0)
    i = int(np.argmin(np.abs(s - 210.0)))
    xy = xy - xy[i] + [64.0, 40.0]
    g = VesselGraph()
    add_free(g, xy, 20.0, spacing=0.5)
    return g, wide_optics(), "wide (r 20 px) vessel on a smooth bend, high OD"


def wide_tributary():
    """A thin tributary (r 3 px) joining a wide vein (r 18 -> 19 px) at 50 deg:
    the junction's caps and overlap polygons inside a near-black lumen (the
    v1 review's node seam)."""
    g = VesselGraph()
    c = np.array([64.0, 64.0])
    N = g.add_node(c / K, depth=10.0)
    p = ray(c - [200, 0], 0, 200) / K
    a = g.add_node(p[0], depth=10.0, kind="inlet")
    g.add_vessel(a, N, path=p, radius=18.0 / K)
    q = ray(c, 0, 200) / K
    b = g.add_node(q[-1], depth=10.0, kind="outlet")
    g.add_vessel(N, b, path=q, radius=19.0 / K)
    d = np.array([np.cos(np.deg2rad(-130)), np.sin(np.deg2rad(-130))])
    t = ray(c + 200 * d, 50, 200) / K
    e = g.add_node(t[0], depth=10.0, kind="inlet")
    g.add_vessel(e, N, path=t, radius=3.0 / K)
    return g, wide_optics(), "thin tributary joining a wide vein at 50 deg, high OD"


def wide_confluence():
    """Two wide veins (r 12 px) joining into one (r 17 px) at 50 deg, high OD."""
    g = VesselGraph()
    c = np.array([64.0, 64.0])
    N = g.add_node(c / K, depth=10.0)
    for ang in (155, 205):
        d = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        t = ray(c + 200 * d, ang - 180, 200) / K
        e = g.add_node(t[0], depth=10.0, kind="inlet")
        g.add_vessel(e, N, path=t, radius=12.0 / K)
    q = ray(c, 0, 200) / K
    b = g.add_node(q[-1], depth=10.0, kind="outlet")
    g.add_vessel(N, b, path=q, radius=17.0 / K)
    return g, wide_optics(), "two wide veins joining, high OD"


WIDE_SCENES = [wide_bend, wide_tributary, wide_confluence]
