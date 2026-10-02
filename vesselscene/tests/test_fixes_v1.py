"""Regression tests for the fixes after the v1 reviews (2026-09-30): the
correctness review's defects 1-9 and the realism review's tells T1, T2, T4,
T6, T7 and T8 that were fixed in code.  Each test fails on the code as it
was before its fix (the docstrings say how)."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from vesselscene import anatomy as A
from vesselscene import imaging as IM
from vesselscene import render as R
from vesselscene import scene as SC
from vesselscene.graph import VesselGraph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMALL = (480, 300)
WEB = ("long", "collecting", "companion", "bundle")


@pytest.fixture(scope="module")
def full():
    return A.generate(A.preset("healthy", 0), 0)


def _scene_params(preset, seed, shape=(1200, 1920)):
    """The anatomy parameters make_scene draws for (seed, preset)."""
    ss = np.random.SeedSequence([int(seed), SC.SCENE_SALT])
    P = A.preset(preset, np.random.default_rng(ss.spawn(6)[0]))
    return replace(P, frame_px=(shape[1], shape[0]))


# ------------------------------------------------------------------ defect 1: pairs pushed onto each other
def _paired_overlap(g):
    """{kind: (paired length in the frame, length whose projected lumens overlap the partner's)} over the web's
    companions and bundle members, along their stretches beside and parallel to (within 20 deg, a wall gap under
    4 (r + r)) their partner vessels (review_correct/pairs.py)."""
    fx0, fy0, fx1, fy1 = g.meta["frame"]
    out = {}
    for e in g.vessels.values():
        if "partner" not in e.info:
            continue
        kind = e.info.get("web") or "v0pair"
        hosts = [g.vessels[v] for v in e.info.get("partner_path", [e.info["partner"]]) if v in g.vessels]
        if not hosts:
            continue
        sh = [A._vsample(h, 0.5) for h in hosts]
        xy = np.concatenate([s["xy"] for s in sh])
        rh = np.concatenate([s["r"] for s in sh])
        th = np.concatenate([s["t"] for s in sh])
        s = A._vsample(e, 0.5)
        d, i = cKDTree(xy).query(s["xy"])
        rel = (d - s["r"] - rh[i]) / (s["r"] + rh[i])
        par = np.abs(np.sum(th[i] * s["t"], 1)) > math.cos(math.radians(20))
        inside = (s["xy"][:, 0] > fx0) & (s["xy"][:, 0] < fx1) & (s["xy"][:, 1] > fy0) & (s["xy"][:, 1] < fy1)
        m = (rel < 4.0) & par & inside
        L, ov = out.get(kind, (0.0, 0.0))
        out[kind] = (L + 0.5 * m.sum(), ov + 0.5 * (m & (rel < 0)).sum())
    return out


def test_pairs_and_bundles_do_not_lie_on_each_other(full):
    """Companions and bundle members meander with their partner (a shared
    displacement, anatomy._shared_meander).  Each tree drew its own before:
    on healthy seed 0 the lumens overlapped in projection over 27 % of the
    bundles' paired length and 9 % of the companions'."""
    res = _paired_overlap(full)
    assert full.meta["diagnostics"]["shared_meander_nodes"] > 0
    # (v2: bundles at about half v1's rate, so seed 0's full frame may have none paired in the frame; a
    # bundle-rich frame is checked as well)
    rich = _paired_overlap(A.generate(replace(A.preset("healthy", 3), frame_px=(960, 600), bundle_rate=8.0), 3))
    for kind, r_ in (("bundle", rich), ("companion", res), ("companion", rich)):
        L, ov = r_[kind]
        assert L > (120 if kind == "bundle" else 300), (kind, L)
        # (v2: a bundle member's spacing drifts and neighbours may touch or overlap a little in projection, at
        # different depths - the stills' irregular bundles; v1 measured 27 % before the shared meander)
        assert ov / L < (0.2 if kind == "bundle" else 0.05), (kind, ov / L)


def test_collecting_companions_keep_their_own_gap():
    """The collecting veins' companions have their own parameter
    (collect_pair_gap; it was a hard-coded (0.2, 1.0) while the README and
    the pairs test assumed web_pair_gap)."""
    P = A.AnatomyParams()
    assert P.collect_pair_gap == (0.2, 1.0) and P.web_pair_gap != P.collect_pair_gap


# ------------------------------------------------------------------ defect 2: no twigs, a mesh without flow
def test_twigs_feed_the_mesh_even_when_collecting_veins_and_bundles_fill_the_budget():
    """With many collecting veins and bundles the length budget was full
    before any plain long vessel was grown: no twigs, a capillary mesh
    carrying 1e-20 of the flow and inlets / outlets inside the domain (random
    seed 4, healthy seed 25).  Now part of the budget is kept for plain long
    vessels, a side without twigs gets them from any long vessel, and every
    vessel carries flow."""
    P = replace(A.preset("healthy", 3), frame_px=SMALL, collect_rate=6.0, bundle_rate=10.0)
    g = A.generate(P, 3)
    w = g.meta["diagnostics"]["web"]
    assert w["twigs"] > 0 and w.get("long", 0) > 0
    assert g.meta["diagnostics"]["resolve"]["dead_removed"] == 0
    assert g.check(overlap=True) == []
    qmax = max(e.flow for e in g.vessels.values())
    caps = [e for e in g.vessels.values() if e.cls == "capillary"]
    assert len(caps) > 10 and min(e.flow for e in caps) > 1e-9 * qmax


def test_vessels_without_flow_are_removed():
    """A component with no pressure difference (here a triangle of vessels
    off every path) carries no flow; _remove_dead removes it, so no source
    or sink is left inside the domain."""
    g = VesselGraph(dict(domain=(0.0, 0.0, 100.0, 100.0)))
    a = g.add_node((0.0, 50.0), depth=10, pressure=1.0, border=True, p_boundary=1.0)
    b = g.add_node((100.0, 50.0), depth=10, pressure=0.0, border=True, p_boundary=0.0)
    g.add_vessel(a, b, radius=1.0, depth=10.0, cls="venule", side="V")
    n = [g.add_node(xy, depth=10) for xy in ((40, 20), (60, 20), (50, 35))]
    for u, v in ((0, 1), (1, 2), (2, 0)):
        g.add_vessel(n[u], n[v], radius=0.5, depth=10.0)
    g.solve_flow(orient=True)
    assert A._remove_dead(g) == 3
    assert len(g.vessels) == 1 and g.check() == []


# ------------------------------------------------------------------ defect 3: wide vessels removed, corners
def _two_vessel_graph():
    g = VesselGraph(dict(frame=(0.0, 0.0, 200.0, 100.0)))
    a, b = g.add_node((-50, 50), depth=60), g.add_node((250, 50), depth=60)
    c, d = g.add_node((20, -50), depth=20), g.add_node((180, 150), depth=20)
    wide = g.add_vessel(a, b, radius=6.0, depth=60.0, cls="episcleral_vein", side="D")
    thin = g.add_vessel(c, d, radius=0.7, depth=20.0, cls="venule", side="V", web="bundle")
    return g, wide, thin


def test_last_resort_removes_the_less_conspicuous_vessel():
    """In the overlap pass's last round the victim is the vessel whose
    removal costs the image least (calibre x length in the frame x a depth
    fade), not the first non-web vessel: an episcleral vein of r 6 lost
    against a bundle member of r 0.7 (pathologic seed 0 lost veins of r
    6-10.5)."""
    g, wide, thin = _two_vessel_graph()
    cache = {}
    A._samples(g, list(g.vessels), cache)
    assert A._pick_victim(g, cache, wide, thin, last=True) == thin
    assert A._pick_victim(g, cache, thin, wide, last=True) == thin
    assert A._pick_victim(g, cache, wide, thin, last=False) is None        # a long web vessel is kept before


def test_merge_does_not_fold_a_thin_vessel_back():
    """After a removal, two thin vessels meeting at a node that fold back
    (> 90 deg) are not merged into one vessel with a U-turn (54 % of merged
    vessels turned > 60 deg within 6 d_c; a bundle vessel made a 154 deg
    U-turn): _merge_pair declines, and both go."""
    g = VesselGraph()
    a, n, b = g.add_node((0, 0), depth=10), g.add_node((30, 0), depth=10), g.add_node((2, 6), depth=10)
    k1 = g.add_vessel(a, n, radius=0.6, depth=10.0)
    k2 = g.add_vessel(n, b, radius=0.6, depth=10.0)
    assert A._merge_pair(g, k1, k2, n) is None
    g2 = VesselGraph()
    a, n, b = g2.add_node((0, 0), depth=10), g2.add_node((30, 0), depth=10), g2.add_node((60, 8), depth=10)
    k1 = g2.add_vessel(a, n, radius=0.6, depth=10.0)
    k2 = g2.add_vessel(n, b, radius=0.6, depth=10.0)
    assert A._merge_pair(g2, k1, k2, n) is not None


def test_coincident_short_vessels_are_not_kept():
    """Two vessels between the same two nodes, one of them short, lie on each
    other (merges made such pairs 1 d_c long): _graph_remove keeps one."""
    g = VesselGraph()
    p = g.add_node((0, 0), depth=10, pressure=1.0, border=True, p_boundary=1.0)
    q = g.add_node((40, 0), depth=10, pressure=0.0, border=True, p_boundary=0.0)
    u, v = g.add_node((19, 0), depth=10), g.add_node((20, 0), depth=10)
    g.add_vessel(p, u, radius=1.0, depth=10.0)
    g.add_vessel(u, v, radius=0.8, depth=10.0)
    g.add_vessel(u, v, path=[(19, 0), (19.5, 0.3), (20, 0)], radius=0.5, depth=10.0)
    g.add_vessel(v, q, radius=1.0, depth=10.0)
    A._graph_remove(g, [])
    pairs = [frozenset((e.u, e.v)) for e in g.vessels.values()]
    assert len(pairs) == len(set(pairs))


# ------------------------------------------------------------------ defect 4: a self-overlap left behind
def test_folds_detects_a_loop():
    t = np.linspace(0, 2.2 * math.pi, 400)
    loop = np.stack([30 * np.sin(t), 30 * (1 - np.cos(t))], 1)
    s = A._arclen(loop)
    assert A._folds(loop, np.full(len(loop), 2.0), s)
    line = np.stack([np.linspace(0, 200, 400), 5 * np.sin(np.linspace(0, 6, 400))], 1)
    assert not A._folds(line, np.full(len(line), 2.0), A._arclen(line))


def test_random_seed_1_full_frame_is_valid():
    """random seed 1 (as make_scene draws it) left a collecting venule of r
    3.2-4.6 that turned 534 deg and overlapped itself (self_left 1, and
    check(overlap=True) failed).  Tortuosity no longer folds a non-capillary
    vessel onto itself, and any self-overlap left is removed."""
    g = A.generate(_scene_params("random", 1), 1)
    assert g.check(overlap=True) == []
    for e in g.vessels.values():
        if e.cls != "capillary" and e.info.get("web") in WEB:
            s = A._vsample(e, 0.5)
            ang = np.unwrap(np.arctan2(s["t"][:, 1], s["t"][:, 0]))
            assert abs(ang[-1] - ang[0]) < 2 * math.pi, e.vid


# ------------------------------------------------------------------ defect 5: labels of the frame
def test_frame_has_its_own_label_rasters():
    """The frame's focus, filling and blur differ from its average's (18 % of
    in-frame vessels change tier): the kind-dependent rasters are saved for
    the frame too (<key>_frame), its tier map from the frame's visibility."""
    sc = SC.make_scene(4, "healthy", shape=(160, 256), device=DEV)
    lab = sc.labels
    for key in SC.KIND_LABELS:
        assert lab[key + "_frame"].shape == lab[key].shape, key
    dom, t = lab["dominant_frame"], lab["tier_frame"]
    strong = (dom >= 0) & (lab["dominant_od_frame"] >= SC.OD_MIN)
    assert strong.mean() > 0.05
    for vid in np.unique(dom[strong])[:40]:
        want = SC.TIER_NAMES.index(sc.visibility[int(vid)]["frame"]["tier"]) \
            if sc.visibility[int(vid)]["frame"]["tier"] in SC.TIER_NAMES else SC.TIER_NAMES.index("dont_care")
        assert (t[strong & (dom == vid)] == want).all()
    assert (t[~strong] == 0).all()


# ------------------------------------------------------------------ defect 6: kinks through web junctions
@pytest.mark.parametrize("which", ["full", "small_healthy", "small_pathologic"])
def test_long_vessels_have_no_kink_at_their_junctions(which, full):
    """The course of a long vessel turns little over +-2 d_c about each node
    it passes (the old test only looked at the first control legs, which
    _web_c1 sets; kinks of 46 and 108 deg survived where a node had been
    contracted with a companion's or a twig's).  30 deg: a thin tortuous
    vessel's interior turns up to about 19 deg over 4 d_c (90th percentile,
    correctness review)."""
    g = full if which == "full" else A.generate(replace(A.preset(which.split("_")[1], 2), frame_px=SMALL), 2)
    ends = {}
    for k, e in g.vessels.items():
        if e.info.get("web") in WEB:
            ends.setdefault(e.u, []).append((k, 0, e.info["tree"]))
            ends.setdefault(e.v, []).append((k, 1, e.info["tree"]))
    turns = []
    for nid, lst in ends.items():
        if g.nodes[nid].info.get("border"):
            continue
        by = {}
        for k, end, t in lst:
            by.setdefault(t, []).append((k, end))
        for arms in by.values():
            if len(arms) != 2:
                continue
            tv = []
            for k, end in arms:
                s = A._vsample(g.vessels[k], 0.25)
                if s["L"] < 2.0:
                    break
                i = int(np.searchsorted(s["s"], 2.0)) if end == 0 else int(np.searchsorted(s["s"], s["L"] - 2.0))
                tv.append(s["t"][min(i, len(s["t"]) - 1)] * (1 if end == 0 else -1))
            if len(tv) == 2:
                turns.append(math.degrees(math.acos(float(np.clip(-tv[0] @ tv[1], -1, 1)))))
    assert len(turns) > 5
    assert max(turns) < 30.0, max(turns)
    assert np.percentile(turns, 90) < 10.0


# ------------------------------------------------------------------ defect 8: arcades crossing their host
def test_no_arcade_crosses_the_vessels_it_joins(full):
    h = VesselGraph.from_dict(full.to_dict())
    assert A._drop_crossing_arcades(h) == 0
    assert sum(1 for e in full.vessels.values() if e.info.get("web") == "arcade") > 10


# ------------------------------------------------------------------ defect 9: render reach bound
def test_reach_bound_with_a_tilted_curved_focal_surface():
    """The deepest point of a tilted, curved focal surface is at x0 - g / (2c),
    not at (x0, y0): the reach bound of every vessel must cover its true
    max of k r + 4 s (the review's counterexample: bound 29.5 px, true 32.6)."""
    g = VesselGraph()
    a, b = g.add_node((300, 40), depth=5.0), g.add_node((420, 60), depth=5.0)
    e = g.vessels[g.add_vessel(a, b, radius=1.0, depth=5.0)]
    o = R.Optics(defocus_px_per_dc=0.08, focus_gx=0.5, focus_gy=0.0, focus_curv=-4e-4, focus_x0=100.0,
                 focus_y0=50.0, z_focus=20.0)
    s = A._vsample(e, 0.5)
    true = float(np.max(4.0 * s["r"] + R.NSIG * o.blur(s["z"], s["xy"])))
    assert R._vessel_reach(e, 4.0, o) >= true - 1e-6


# ------------------------------------------------------------------ T1: crossings composited by depth
def _cross_od(bypass: bool, shared: bool = False):
    g = VesselGraph(dict(k=3.0))
    a0, a1 = g.add_node((-20, 20), depth=12, pressure=1.0), g.add_node((60, 20), depth=12, pressure=0.0)
    b0, b1 = g.add_node((20, -20), depth=30, pressure=1.0), g.add_node((20, 60), depth=30, pressure=0.0)
    g.add_vessel(a0, a1, radius=3.0, depth=12.0, hct=1.0, cls="venule", side="V")
    g.add_vessel(b0, b1, radius=1.2, depth=30.0, hct=1.0, cls="venule", side="V")
    old = SC.CROSSING_BYPASS
    SC.CROSSING_BYPASS = bypass
    try:
        od, _, _ = SC._render(g, 3.0, (120, 120), R.Optics(halo_h=0.0, mu=0.135, l_bypass=80.0), None, DEV)
        single = [R.vessel_od(g, 3.0, (120, 120), R.Optics(halo_h=0.0, mu=0.135, l_bypass=80.0), device=DEV,
                              vids=[v]) for v in (0, 1)]
    finally:
        SC.CROSSING_BYPASS = old
    return od, single


def test_crossing_is_composited_by_depth():
    """Where a wide shallow vessel lies over a deeper one, the light
    scattered in above the upper one crossed neither: the crossing is
    lighter than the sum of the two ODs, by the closed form
    T' = T'_t T'_b + g_t / (1 - g_t) (1 - T'_t)(1 - T'_b) (the sum made
    black knots at crossings); away from the crossing nothing changes."""
    add, (s0, s1) = _cross_od(False)
    comp, _ = _cross_od(True)
    y, x = 60, 60
    assert add[y, x] == pytest.approx(s0[y, x] + s1[y, x], rel=0.02)
    o = R.Optics(mu=0.135, l_bypass=80.0)
    fs = o.f_spectral
    T1, T2 = math.exp(-s0[y, x]), math.exp(-s1[y, x])
    q1, q2 = (T1 - fs) / (1 - fs), (T2 - fs) / (1 - fs)
    gt = 1 - math.exp(-12.0 / 80.0)
    qn = q1 * q2 + gt / (1 - gt) * (1 - q1) * (1 - q2)
    want = -math.log(fs + (1 - fs) * qn)
    assert comp[y, x] == pytest.approx(want, rel=0.03)
    assert comp[y, x] < add[y, x] - 0.01
    assert np.allclose(comp[:, 100:], add[:, 100:], atol=1e-5)          # the upper vessel alone
    assert np.allclose(comp[100:, :], add[100:, :], atol=1e-5)          # the lower vessel alone


# ------------------------------------------------------------------ T2: flattened veins
def test_flattened_vein_renders_its_shorter_chord():
    """A vein with aspect a (lumen depth / width) renders as a round one with
    a times the absorbing chord: the same width, a lower plateau."""
    def one(aspect=None, hct=1.0):
        g = VesselGraph(dict(k=3.0))
        a, b = g.add_node((-10, 20), depth=20), g.add_node((50, 20), depth=20)
        vid = g.add_vessel(a, b, radius=4.0, depth=20.0, hct=hct, cls="venule", side="V")
        if aspect is not None:
            g.vessels[vid].info["aspect"] = aspect
        return R.vessel_od(g, 3.0, (120, 120), R.Optics(halo_h=0.0), device=DEV)
    flat, thin_blood, round_ = one(0.5), one(None, 0.5), one(None, 1.0)
    assert np.allclose(flat, thin_blood, atol=1e-4)
    assert flat.max() < 0.8 * round_.max()


def test_wide_veins_are_flattened_arterioles_are_round(full):
    P = full.meta["params"]
    for e in full.vessels.values():
        r = float(np.mean(e.r_ctrl))
        a = e.info["aspect"]
        if e.side == "A" or e.cls == "capillary":
            assert a == 1.0
        elif (e.side == "V" or e.cls == "episcleral_vein") and r >= P["vein_flat_r"][1]:
            assert P["vein_aspect"][0] - 1e-6 <= a <= P["vein_aspect"][1] + 1e-6


# ------------------------------------------------------------------ T6: ribs in wide bent vessels
def test_wide_bent_vessel_has_no_ribs():
    """A wide vessel (r 20 px) on a 320 px bend: the OD along the lumen off
    its axis is smooth (pieces end-turn at most EDGE_TOL px at the lumen
    edge; with 7 px pieces the joints drew transverse ribs of SD 0.01 Np)."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    k, Rc, r = 4.0, 80.0, 5.0
    g = VesselGraph()
    th = np.linspace(-0.6, 0.6, 200)
    path = np.stack([100 + Rc * np.sin(th), 60 + Rc * (1 - np.cos(th))], 1)
    a, b = g.add_node(path[0], depth=20.0), g.add_node(path[-1], depth=20.0)
    vid = g.add_vessel(a, b, path=path, radius=r, depth=20.0, hct=1.0, cls="venule", side="V")
    o = R.Optics(s0_px=1.55, mu=0.135, l_bypass=80.0, halo_h=0.0, z_focus=20.0)
    od = R.vessel_od(g, k, (560, 800), o, device=DEV)
    s = g.vessels[vid].sample(0.25)
    xy, t = s["xy"] * k, s["t"]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    m = (s["s"] > 20) & (s["s"] < s["L"] - 20)
    for off in (-0.5, 0.5):
        p = xy + off * r * k * nrm
        v = map_coordinates(od, [p[:, 1], p[:, 0]], order=1)[m]
        assert np.std(v - gaussian_filter1d(v, 48.0)) < 0.002


# ------------------------------------------------------------------ T7, T8: the frame
def test_frame_focal_offset_stays_within_the_spread():
    o = R.Optics(focus_spread=5.0, focus_offset=1.0)
    rng = np.random.default_rng(0)
    offs = np.array([SC.frame_optics(o, rng).focus_offset - 1.0 for _ in range(2000)])
    assert np.abs(offs).max() < 5.0 and 1.5 < np.std(offs) < 4.0


def test_aggregate_grain_averages_across_a_wide_lumen():
    """Beyond grain_r_agg the aggregates' relative SD falls as sqrt(r_agg / r)
    (a chord through a wide lumen crosses many independent aggregates); it
    grew with r up to 0.35 and drew full-width bands across wide veins."""
    def amp(r):
        g = VesselGraph()
        a = g.add_node((0, 0), depth=20, pressure=1.0)
        b = g.add_node((400, 0), depth=20, pressure=0.0)
        g.add_vessel(a, b, radius=r, depth=20.0, cls="venule", side="V")
        g.solve_flow()
        f = IM.filling(g, kind="frame", frame=3, burst_seed=1, relative=True, **SC.CALIBRATED_FILL)[0]
        return float(np.std(f.m) / np.mean(f.m))
    a2, a4, a8 = amp(SC.CALIBRATED_FILL["grain_r_agg"]), amp(5.0), amp(10.0)
    assert a4 < a2 and a8 < a4
    ra = SC.CALIBRATED_FILL["grain_r_agg"]
    assert a8 == pytest.approx(a4 * math.sqrt(5.0 / 10.0), rel=0.35)
    assert ra < 4.0
