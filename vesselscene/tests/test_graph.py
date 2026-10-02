"""Tests of the truth contract (graph.py): VesselGraph.check must accept valid
anatomy and catch each kind of invalid anatomy (the other test files only
assert check() == [], which a check that always returned [] would pass)."""
from __future__ import annotations

import math

import numpy as np

from vesselscene.graph import VesselGraph, close_pairs


def _ray(p0, deg, length, n=200):
    t = np.linspace(0.0, length, n)
    return np.asarray(p0, float) + t[:, None] * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def _fork(child_deg=(-30.0, 30.0), r=(1.5, 1.0, 1.0), z=10.0):
    """Parent from the left into a node at (50, 50), two children leaving it."""
    g = VesselGraph(dict(domain=(0.0, 0.0, 100.0, 100.0)))
    N = g.add_node((50, 50), depth=z)
    a = g.add_node((0, 50), depth=z, kind="inlet", pressure=1.0)
    g.add_vessel(a, N, path=_ray((0, 50), 0, 50), radius=r[0])
    for deg, rc in zip(child_deg, r[1:]):
        p = _ray((50, 50), deg, 50 / max(abs(math.cos(math.radians(deg))), 1e-3))
        p = p[(p[:, 0] <= 100) & (p[:, 1] >= 0) & (p[:, 1] <= 100)]
        b = g.add_node(p[-1], depth=z, kind="outlet", pressure=0.0)
        g.add_vessel(N, b, path=p, radius=rc)
    g.solve_flow()
    return g


def test_a_valid_fork_passes():
    g = _fork()
    assert g.check(overlap=True) == []


def test_same_depth_crossing_is_flagged():
    g = _fork()
    a = g.add_node((70, 0), depth=10.0, kind="inlet", pressure=1.0)
    b = g.add_node((70, 100), depth=10.0, kind="outlet", pressure=0.0)
    v = g.add_vessel(a, b, radius=1.0)
    g.solve_flow()
    bad = g.check(overlap=True)
    assert any("overlap" in s and str(v) in s for s in bad)
    # the same vessel deep below the others is a valid crossing
    g.nodes[a].depth = g.nodes[b].depth = 40.0
    g.vessels[v].z_ctrl[:] = 40.0
    assert g.check(overlap=True) == []


def test_pass_through_node_and_free_end_are_flagged():
    g = _fork()
    # split the first child at an interior node: one vessel in, one out
    e = next(e for e in g.vessels.values() if e.u != 1 and g.nodes[e.u].kind != "inlet")
    smp = e.sample(0.5)
    m = len(smp["xy"]) // 2
    M = g.add_node(smp["xy"][m], depth=10.0)
    end = e.v
    g.remove_vessel(e.vid)
    g.add_vessel(e.u, M, path=smp["xy"][:m + 1], radius=1.0)
    g.add_vessel(M, end, path=smp["xy"][m:], radius=1.0)
    g.solve_flow()
    assert any("one vessel in and one out" in s for s in g.check())
    # a vessel ending inside the domain
    g2 = _fork()
    n = g2.add_node((50, 90), depth=10.0, kind="outlet", pressure=0.0)
    far = next(k for k, nd in g2.nodes.items() if nd.kind == "inlet")
    g2.add_vessel(far, n, path=_ray((0, 50), 30, 40), radius=0.8)
    g2.nodes[n].xy = g2.vessels[max(g2.vessels)].ctrl[-1].copy()
    g2.solve_flow()
    assert any("free end" in s for s in g2.check())


def test_cycle_is_flagged():
    g = _fork()
    kids = [e for e in g.vessels.values() if g.nodes[e.v].kind == "outlet"]
    a, b = kids[0], kids[1]
    # join the two outlets back to the fork node with a loop (edges forced to point into a cycle)
    g.nodes[a.v].pressure = g.nodes[b.v].pressure = None
    w = g.add_vessel(a.v, b.v, radius=0.5, depth=40.0)
    g.vessels[w].z_ctrl[:] = 40.0
    x = g.add_vessel(b.v, a.u, path=np.array([g.nodes[b.v].xy, [95, 50], g.nodes[a.u].xy]), radius=0.5,
                     depth=60.0)
    g.vessels[x].z_ctrl[:] = 60.0
    assert any("cycle" in s for s in g.check())


def test_a_vessel_overlapping_itself_is_flagged():
    """A loop that closes on itself at one depth (the renderer would add the
    two stretches); a hairpin whose legs do not touch is fine."""
    g = VesselGraph(dict(domain=(-100.0, -100.0, 200.0, 200.0)))
    t = np.linspace(-np.pi, 3 * np.pi, 800)
    xy = np.stack([50 + 1.0 * t + 5.0 * np.sin(t), 50 - 5.0 * np.cos(t)], 1)
    xy = np.vstack([_ray(xy[0], 180, 150)[::-1], xy, _ray(xy[-1], 0, 150)])
    a = g.add_node(xy[0], depth=10.0, kind="inlet", pressure=1.0)
    b = g.add_node(xy[-1], depth=10.0, kind="outlet", pressure=0.0)
    v = g.add_vessel(a, b, path=xy, radius=0.5, spacing=0.8)
    g.solve_flow()
    assert f"vessel {v}: its lumen overlaps itself in 3-D" in g.check(overlap=True)
    # a hairpin with legs 3 r apart (a U-turn of radius 1.5 r)
    g2 = VesselGraph(dict(domain=(-100.0, -100.0, 200.0, 200.0)))
    ph = np.linspace(-np.pi / 2, np.pi / 2, 100)
    arc = np.stack([60 + 1.5 * np.cos(ph), 50 + 1.5 * np.sin(ph)], 1)
    xy = np.vstack([_ray((-60, 48.5), 0, 120), arc, _ray((60, 51.5), 180, 120)])
    a = g2.add_node(xy[0], depth=10.0, kind="inlet", pressure=1.0)
    b = g2.add_node(xy[-1], depth=10.0, kind="outlet", pressure=0.0)
    g2.add_vessel(a, b, path=xy, radius=1.0, spacing=0.8)
    g2.solve_flow()
    assert not any("itself" in s for s in g2.check(overlap=True))


def test_sibling_running_inside_another_beyond_the_junction_is_flagged():
    """A thin vessel leaving a wide one's node that bends back to run inside
    it: beyond the junction's reach that is not a junction but a vessel
    inside a vessel (the first check exempted 4 max(r) + 2 d_c around the
    node: 49 d_c for a wide vein, and missed it)."""
    g = VesselGraph(dict(domain=(-100.0, -100.0, 400.0, 200.0)))
    N = g.add_node((0, 50), depth=60.0)
    a = g.add_node((-100, 50), depth=60.0, kind="inlet", pressure=1.0)
    g.add_vessel(a, N, path=_ray((-100, 50), 0, 100), radius=10.0)
    b = g.add_node((300, 50), depth=60.0, kind="outlet", pressure=0.0)
    g.add_vessel(N, b, path=_ray((0, 50), 0, 300), radius=10.0)
    # leaves at 50 deg, turns back after 12 d_c, runs 3 d_c off the vein's axis up to 34 d_c from the node,
    # then leaves the vein at 90 deg: every overlap lies within the old exemption (4 max(r) + 2 = 42 d_c)
    p1 = _ray((0, 50), -50, 12)
    p2 = np.linspace(p1[-1], [24, 47], 40)[1:]
    p3 = _ray((24, 47), 0, 10)[1:]
    p4 = _ray(p3[-1], -90, 90)[1:]
    c = g.add_node(p4[-1], depth=60.0, kind="outlet", pressure=0.0)
    thin = g.add_vessel(N, c, path=np.vstack([p1, p2, p3, p4]), radius=3.0)
    g.solve_flow()
    reach = g.junction_reach(N)
    assert 10 < reach < 30
    s_thin, s_vein = g.vessels[thin].sample(0.5), g.vessels[1].sample(0.5)
    D = np.hypot(*(s_thin["xy"][:, None] - s_vein["xy"][None]).transpose(2, 0, 1))
    far = (D < 3.0 + 10.0 + 0.5) & (np.hypot(*(s_thin["xy"] - [0, 50]).T)[:, None] > reach)
    assert far.any()                                        # overlap beyond the junction's reach ...
    assert np.hypot(*(s_thin["xy"][far.any(1)] - [0, 50]).T).max() < 4 * 10.0 + 2     # ... hidden by the old rule
    assert any("overlap" in s and str(thin) in s for s in g.check(overlap=True))


def test_siblings_overlapping_only_near_their_node_pass():
    """Children leaving at 16 deg overlap near the node for about
    (r_i + r_j) / sin(theta): that is the junction, not an overlap."""
    g = _fork(child_deg=(-8.0, 8.0), r=(2.0, 1.5, 1.5))
    assert g.check(overlap=True) == []
    N = next(n for n, (i, o) in g.degrees().items() if i == 1 and o == 2)
    assert g.junction_reach(N) > (1.5 + 1.5) / math.sin(math.radians(16))


def test_close_pairs_matches_brute_force():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 40, (400, 2))
    r = np.where(rng.random(400) < 0.1, rng.uniform(2, 6, 400), rng.uniform(0.3, 1.5, 400))
    i, j = close_pairs(xy, r, 0.5)
    D = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    lim = r[:, None] + r[None] + 0.5
    bi, bj = np.nonzero(np.triu(D < lim, 1))
    assert set(zip(i.tolist(), j.tolist())) == set(zip(bi.tolist(), bj.tolist()))


def test_crossings_skip_one_volume_overlaps_at_a_shared_node():
    """v2 correctness review, defect 6: two vessels leaving one node at a
    small angle (20 deg) whose centrelines cross 5 d_c from it at the same depth are
    one blood volume (the junction), not a crossing; at different depths the
    same geometry is a crossing."""
    for z_b, n_expected in ((10.0, 0), (30.0, 1)):
        g = VesselGraph()
        n = g.add_node((0.0, 0.0), depth=10.0)
        e = g.add_node((60.0, 0.0), depth=10.0, kind="outlet", pressure=0.0)
        f = g.add_node((40.0, 14.0), depth=z_b, kind="outlet", pressure=0.0)
        g.add_vessel(n, e, radius=1.0, depth=10.0)
        t = np.linspace(0.0, 1.0, 200)
        path = np.stack([40.0 * t, 14.0 * t * t - 0.7 * np.sin(math.pi * np.minimum(t / 0.15, 1.0))], 1)
        g.add_vessel(n, f, path=path, radius=1.0, depth=np.r_[10.0, np.full(9, z_b)] if z_b != 10.0 else 10.0)
        cr = g.crossings()
        assert len(cr) == n_expected, cr
