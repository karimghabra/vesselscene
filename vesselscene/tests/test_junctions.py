"""Tests of the image-junction ground truth (junctions.py) on hand-built
VesselGraphs whose answers are known: a fork, crossings at 60 and 10 deg, a
crossing next to a fork, two touching vessels, vessels diving or fading out
of sight beside another, a vessel leaving the frame; the visibility taken
from a rendered OD (a frame's empty stretch); rasters, overlay, JSON; a
small make_scene."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from vesselscene import junctions as J
from vesselscene import render as R
from vesselscene.graph import VesselGraph

K = 4.0
SHAPE = (160, 256)                  # 40 x 64 d_c
NOISE = 0.004                       # Np per DoG band: every vessel below is well visible, a dive is not
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _line(g, a, b, r, z, cls="venule", side="V", hct=1.0):
    na = g.add_node(a, depth=z if np.isscalar(z) else z[0])
    nb = g.add_node(b, depth=z if np.isscalar(z) else z[-1])
    return g.add_vessel(na, nb, radius=r, depth=z, hct=hct, cls=cls, side=side)


def _fork():
    """A parent from the left into a node at (30, 20) d_c = (120, 80) px; one daughter straight on, a thinner one
    50 deg off; every end outside the frame."""
    g = VesselGraph(dict(k=K))
    p0, n = g.add_node((-10, 20), depth=10), g.add_node((30, 20), depth=10)
    d1 = g.add_node((80, 20), depth=10)
    a = math.radians(50)
    d2 = g.add_node((30 + 40 * math.cos(a), 20 + 40 * math.sin(a)), depth=10)
    vp = g.add_vessel(p0, n, radius=1.2)
    v1 = g.add_vessel(n, d1, radius=1.1)
    v2 = g.add_vessel(n, d2, radius=0.7)
    return g, (vp, v1, v2), n


def _crossing(angle_deg):
    """a horizontal at depth 10, b through (32, 20) d_c at angle_deg, deeper (30)."""
    g = VesselGraph(dict(k=K))
    va = _line(g, (-10, 20), (80, 20), 1.0, 10.0)
    t = math.radians(angle_deg)
    c, d = np.array([32.0, 20.0]), 50 * np.array([math.cos(t), math.sin(t)])
    vb = _line(g, c - d, c + d, 1.0, 30.0, cls="arteriole", side="A")
    return g, va, vb


def _junctions(g, **kw):
    return J.image_junctions(g, K, (0.0, 0.0), SHAPE, noise=NOISE, **kw)


def _groups_by_vid(j):
    return sorted(sorted({j["arms"][i]["vid"] for i in grp}) for grp in j["partition"])


def test_constants_mirror_scene():
    from vesselscene import scene as SC
    assert J.LAMBDA_PX == SC.LAMBDA_PX
    assert (J.CNR_REQUIRED, J.CNR_VISIBLE) == (SC.CNR_REQUIRED, SC.CNR_VISIBLE)


def test_resolution_gap_is_the_25_percent_dip():
    """g_res: the wall gap at which two equal parallel vessels, blurred, show a 25 % dip between them."""
    from scipy.special import ndtr
    W = R.box_weights(np.array([0.3]), np.array([0.1]))[0]

    def prof(x, r, s):
        w = r * R.BOX_C
        x = np.abs(np.asarray(x, float))[..., None]
        return (W * (ndtr((w - x) / s) + ndtr((w + x) / s) - 1)).sum(-1)
    for r, s in ((1.0, 2.0), (2.0, 2.0), (3.0, 1.6), (0.6, 3.0)):
        g = float(J.g_res(r, r, s, s))
        d = 2 * r + g
        x = np.linspace(0, d, 401)
        P = prof(x, r, s) + prof(x - d, r, s)
        dip = (P[0] - P.min()) / P[0]
        assert abs(dip - J.DIP_MIN) < 0.04, (r, s, dip)
    assert float(J.g_res(10.0, 10.0, 2.0, 2.0)) == 0.0                 # wide lumens: a dip as soon as they part
    assert float(J.g_res(0.5, 0.5, 2.0, 2.0)) > float(J.g_res(2.0, 2.0, 2.0, 2.0))


def test_fork_is_a_bifurcation_with_three_singleton_arms():
    g, (vp, v1, v2), n = _fork()
    js = _junctions(g)
    assert len(js) == 1
    j = js[0]
    assert j["type"] == "bifurcation" and j["type_visible"] == "bifurcation"
    assert np.hypot(j["x"] - 120.0, j["y"] - 80.0) < 1e-6                # the node
    assert j["n_arms"] == 3 and j["n_visible_arms"] == 3
    assert _groups_by_vid(j) == [[vp], [v1], [v2]]                       # parent and daughters: different vessels
    assert j["lineage"] == [dict(node=n, kind="fork", parents=[vp], children=[v1, v2])]
    arm = {a["vid"]: a for a in j["arms"]}
    assert arm[vp]["flow"] == "in" and arm[v1]["flow"] == "out" and arm[v2]["flow"] == "out"
    assert math.cos(math.radians(arm[vp]["angle_deg"])) < -0.99          # the parent's arm points back (left)
    assert abs(arm[v1]["angle_deg"]) < 1.0 and abs(arm[v2]["angle_deg"] - 50.0) < 1.0
    for a in j["arms"]:                                                  # measured outside the union, a few px out
        assert a["width_px"] / 2 < a["dist_px"] < 30 and a["visibility"] == "visible"
    assert not j["ambiguous"]
    assert j["difficulty"]["fork_turn_margin"] == pytest.approx(50.0, abs=1.5)
    # the straight daughter continues the parent for a straightness rule: a lineage pair, not a merge
    assert [p["relation"] for p in j["pairs_visual"]] == ["lineage"]


def test_crossing_at_60_deg_pairs_each_vessel_through():
    g, va, vb = _crossing(60)
    js = _junctions(g)
    assert len(js) == 1
    j = js[0]
    assert j["type"] == "crossing" and j["type_visible"] == "crossing"
    assert j["n_arms"] == 4 and _groups_by_vid(j) == [[va], [vb]]
    assert all(len(grp) == 2 for grp in j["partition"])
    assert {a["flow"] for a in j["arms"] if a["vid"] == va} == {"in", "out"}
    (m,) = j["members"]
    assert m["angle_deg"] == pytest.approx(60.0, abs=1.0) and m["above"] == va
    assert j["difficulty"]["crossing_angle"] == pytest.approx(60.0, abs=1.0)
    assert j["difficulty"]["depth_gap_dc"] == pytest.approx(20.0, abs=0.5)
    assert not j["ambiguous"] and j["lineage"] == []
    assert sorted(p["relation"] for p in j["pairs_visual"]) == ["same", "same"]


def test_shallow_crossing_is_ambiguous():
    g, va, vb = _crossing(10)
    (j,) = _junctions(g)
    assert j["type"] == "crossing" and j["n_arms"] == 4 and _groups_by_vid(j) == [[va], [vb]]
    assert j["ambiguous"] and "shallow_angle" in j["ambiguous_reasons"]
    (j60,) = _junctions(_crossing(60)[0])
    assert j["radius"] > 1.8 * j60["radius"]                             # the lumens overlap over a long stretch
    # ... longer than the junction's region (CLIP_LAMBDA): its arms leave it still unresolved from each other
    assert "merged_arms" in j["ambiguous_reasons"] and all(a["clipped"] for a in j["arms"])


def test_crossing_next_to_a_fork_is_one_compound_junction():
    g, (vp, v1, v2), n = _fork()
    vc = _line(g, (33, -10), (33, 60), 0.8, 30.0, cls="arteriole", side="A")   # 12 px past the node, deeper
    js = _junctions(g)
    assert len(js) == 1
    j = js[0]
    assert j["type"] == "compound" and j["type_visible"] == "compound"
    types = sorted(m["type"] for m in j["members"])
    assert types[0] == "bifurcation" and "crossing" in types
    assert _groups_by_vid(j) == sorted([[vp], [v1], [v2], [vc]])
    grp = {j["arms"][g_[0]]["vid"]: len(g_) for g_ in j["partition"]}
    assert grp[vc] == 2 and grp[vp] == grp[v1] == grp[v2] == 1           # {p}{d1}{d2}{c_in, c_out}
    assert j["lineage"][0]["parents"] == [vp]


def _kiss(width):
    """b runs beside a, 20 d_c deeper, their lumens overlapping in projection around x = 32 d_c over a stretch
    set by `width` (d_c)."""
    g = VesselGraph(dict(k=K))
    va = _line(g, (-10, 20), (80, 20), 1.0, 10.0)
    xs = np.linspace(-10, 80, 200)
    ys = 21.6 + 4.0 * (1 - np.exp(-((xs - 32) / width) ** 2))
    na, nb = g.add_node((xs[0], ys[0]), depth=30), g.add_node((xs[-1], ys[-1]), depth=30)
    vb = g.add_vessel(na, nb, path=np.stack([xs, ys], 1), radius=1.0, depth=30.0, cls="arteriole", side="A")
    assert not g.crossings()
    return g, va, vb


def test_touching_parallel_vessels():
    g, va, vb = _kiss(6.0)                                   # a short kiss: one junction
    (j,) = _junctions(g)
    assert j["type"] == "touching" and j["n_arms"] == 4 and _groups_by_vid(j) == [[va], [vb]]
    (m,) = j["members"]
    assert m["lumen_overlap"] and m["part"] == "whole" and m["length_px"] <= J.TOUCH_SPLIT_PX
    assert j["ambiguous"] and "shallow_angle" in j["ambiguous_reasons"]


def test_a_long_merged_stretch_is_where_they_meet_and_where_they_part():
    g, va, vb = _kiss(30.0)
    js = _junctions(g)
    assert len(js) == 2 and sorted(j["members"][0]["part"] for j in js) == ["end", "start"]
    for j in js:
        assert j["type"] == "touching" and _groups_by_vid(j) == [[va], [vb]]
        merged = [a for a in j["arms"] if a.get("merged_with")]
        assert sorted(a["vid"] for a in merged) == [va, vb]              # the two arms into the merged stretch
        assert "merged_arms" in j["ambiguous_reasons"]
    assert abs(js[0]["x"] - js[1]["x"]) > J.TOUCH_SPLIT_PX


def _approach(y1, knots, fade):
    """b comes down at x = 36 d_c towards a (y = 20 d_c), turns right at y1 and runs on beside it; it goes out of
    sight before the turn: diving to 160 d_c (fade = 'dive') or emptying (haematocrit 0.02, 'hct')."""
    g = VesselGraph(dict(k=K))
    va = _line(g, (-10, 20), (80, 20), 1.2, 12.0)
    p1 = np.stack([np.full(60, 36.0), np.linspace(-10, y1, 60)], 1)
    th = np.linspace(math.pi, math.pi / 2, 20)
    p2 = np.stack([38.5 + 2.5 * np.cos(th), y1 + 2.5 * np.sin(th)], 1)
    p3 = np.stack([np.linspace(38.5, 80, 60), np.full(60, y1 + 2.5)], 1)
    path = np.concatenate([p1, p2[1:], p3[1:]])
    prof = np.interp(np.linspace(0, 1, 40), knots, [0, 0, 1, 1])
    if fade == "dive":
        zs, hs = 10 + 150 * prof, 1.0
    else:
        zs, hs = 10.0, 1 - 0.98 * prof
    na = g.add_node(path[0], depth=float(np.atleast_1d(zs)[0]))
    nb = g.add_node(path[-1], depth=float(np.atleast_1d(zs)[-1]))
    vb = g.add_vessel(na, nb, path=path, radius=0.9, depth=zs, hct=hs, cls="arteriole", side="A")
    return g, va, vb


def test_vessel_diving_out_of_sight_beside_another_looks_like_a_t():
    g, va, vb = _approach(15.0, [0, 0.34, 0.42, 1], "dive")
    assert not g.crossings()
    (j,) = _junctions(g)
    assert j["type_visible"] == "pseudo-T" and j["n_visible_arms"] == 3
    assert _groups_by_vid(j) == [[va], [vb]]
    vis = {(a["vid"], a["visibility"]) for a in j["arms"]}
    assert (vb, "invisible") in vis and (vb, "visible") in vis
    assert "invisible_arm" in j["ambiguous_reasons"]


def test_vessel_fading_short_of_another_is_a_pseudo_t():
    """b empties (its haematocrit falls) a few px short of a's wall: no overlap, no crossing, no node, but the
    image shows a T."""
    g, va, vb = _approach(14.0, [0, 0.33, 0.38, 1], "hct")
    (j,) = _junctions(g)
    assert j["type"] == "pseudo-T" and j["type_visible"] == "pseudo-T"
    (m,) = j["members"]
    assert m["fading"] == vb and m["at"] == va and 0 < m["gap_px"] < J.LAMBDA_PX
    assert _groups_by_vid(j) == [[va], [vb]]


def test_a_vessel_leaving_the_frame_makes_no_junction():
    g = VesselGraph(dict(k=K))
    _line(g, (-10, 20), (40, 70), 1.0, 10.0)                 # enters on the left, leaves at the bottom
    p0, n = g.add_node((-30, -20), depth=10), g.add_node((-5, -5), depth=10)    # a fork outside the frame
    g.add_vessel(p0, n, radius=1.0)
    g.add_vessel(n, g.add_node((70, 5), depth=10), radius=0.9)
    g.add_vessel(n, g.add_node((30, 80), depth=10), radius=0.8)
    assert _junctions(g) == []


def test_visibility_follows_the_rendered_od():
    """The arms' visibility comes from what the image shows: with b emptied beyond the crossing (a frame's
    red-cell gap), the closed form still says crossing, the rendered OD a pseudo-T."""
    g, va, vb = _crossing(60)
    opt = R.Optics()
    L = g.vessels[vb].length()
    full = R.render_od(g, K, SHAPE, opt, device=DEV)
    empty = R.render_od(g, K, SHAPE, opt, device=DEV,
                        hct_mod={vb: lambda s: (np.asarray(s, float) < 0.5 * L - 4).astype(float)})
    (j_cf,) = _junctions(g)
    (j_full,) = _junctions(g, od=full, optics=opt)
    (j_empty,) = _junctions(g, od=empty, optics=opt, kind="frame")
    assert j_cf["type_visible"] == j_full["type_visible"] == "crossing"
    assert j_empty["type"] == "crossing" and j_empty["type_visible"] == "pseudo-T" and j_empty["kind"] == "frame"
    out = [a for a in j_empty["arms"] if a["vid"] == vb and a["flow"] == "out"][0]
    assert out["visibility"] == "invisible"
    for a in j_full["arms"]:                     # the rendered OD agrees with the closed form where nothing changed
        b = [x for x in j_cf["arms"] if x["vid"] == a["vid"] and x["flow"] == a["flow"]][0]
        assert a["cnr"] == pytest.approx(b["cnr"], rel=0.1)


def test_rasters_overlay_and_json(tmp_path):
    g, (vp, v1, v2), n = _fork()
    js = _junctions(g)
    ras = J.junction_rasters(js, SHAPE)
    for t in J.TYPES:
        assert ras["heat_" + t.replace("-", "_")].shape == SHAPE
    assert ras["heat_bifurcation"][80, 120] == pytest.approx(1.0, abs=0.02)
    assert ras["heat_crossing"].max() == 0 and ras["heat_visible"].max() > 0.98
    on = ras["arm_vis"] > 0
    assert on.sum() > 10 and np.allclose(np.hypot(*ras["arm_dir"][on].T), 1.0, atol=1e-5)
    assert set(np.unique(ras["arm_group"][on])) == {1, 2, 3}
    img = np.full(SHAPE, 2000.0, np.float32) * np.exp(-R.render_od(g, K, SHAPE, R.Optics(), device=DEV))
    p = tmp_path / "ov.png"
    out = J.draw_junctions(img, js, str(p), scale=2)
    assert p.exists() and out.shape == (2 * SHAPE[0], 2 * SHAPE[1], 3)
    s = json.dumps(js, allow_nan=False)                                  # JSON-serialisable, no NaN
    J.save_junctions(js, str(tmp_path / "j.json"))
    assert J.load_junctions(str(tmp_path / "j.json")) == json.loads(s)


def test_scene_junctions_on_a_small_scene():
    from vesselscene import scene as SC
    sc = SC.make_scene(4, "healthy", shape=SHAPE, device=DEV)
    for kind in ("average", "frame"):
        js = J.scene_junctions(sc, kind)
        assert js, kind
        for j in js:
            assert j["kind"] == kind and j["type"] in J.TYPES and j["type_visible"] in J.VISIBLE_TYPES
            assert sorted(i for grp in j["partition"] for i in grp) == list(range(j["n_arms"]))
            for grp in j["partition"]:
                assert len({j["arms"][i]["vid"] for i in grp}) == 1
            assert all(a["vid"] in sc.graph.vessels for a in j["arms"])
            assert j["n_visible_arms"] >= 1
            assert (j["type_visible"] == "none") == (j["n_visible_lines"] <= 2)
        json.dumps(js, allow_nan=False)
        assert J.check_junctions(js, sc.graph) == [], kind           # v2 correctness review, defect 1


def _crossing_before_fork(dx, angle_deg):
    """_fork moved 10 d_c right (node at (40, 20) d_c) and a thin deep vessel crossing the parent dx d_c upstream
    of the node at angle_deg: the crossing's unresolved core on the parent runs into the fork's."""
    g = VesselGraph(dict(k=K))
    p0, n = g.add_node((-10, 20), depth=10), g.add_node((40, 20), depth=10)
    d1 = g.add_node((90, 20), depth=10)
    a = math.radians(50)
    d2 = g.add_node((40 + 40 * math.cos(a), 20 + 40 * math.sin(a)), depth=10)
    vp = g.add_vessel(p0, n, radius=1.2)
    g.add_vessel(n, d1, radius=1.1)
    g.add_vessel(n, d2, radius=0.7)
    t = math.radians(angle_deg)
    c, d = np.array([40.0 - dx, 20.0]), 60 * np.array([math.cos(t), math.sin(t)])
    vc = _line(g, c - d, c + d, 0.8, 30.0, cls="arteriole", side="A")
    return g, vp, vc


@pytest.mark.parametrize("dx, angle", [(7, 20), (9, 8), (14, 12)])
def test_a_crossing_beside_a_fork_keeps_the_parent_continuous(dx, angle):
    """v2 correctness review, defect 1: when a crossing's core on a vessel ran
    into that vessel's end node held by another junction, the side was taken
    as closed and the vessel had no arm there (a crossing read as a T, the
    vessel's continuity broken), or both its arms sat at one sample.  Now the
    parent leaves the crossing by an 'out' arm and enters the fork's junction
    by an 'in' arm; where their claims touch, both arms are linked to each
    other at the meeting point; no arm lies in another junction's claim."""
    g, vp, vc = _crossing_before_fork(dx, angle)
    js = _junctions(g)
    assert J.check_junctions(js, g) == []
    xing = [j for j in js if any(m["type"] == "crossing" and vp in m["vessels"] for m in j["members"])]
    assert len(xing) == 1
    jx = xing[0]
    flows = sorted(a["flow"] for a in jx["arms"] if a["vid"] == vp)
    if any(m["type"] == "bifurcation" for m in jx["members"]):        # crossing and fork in one junction
        assert flows == ["in"]
    else:
        assert flows == ["in", "out"]
        out = next(a for a in jx["arms"] if a["vid"] == vp and a["flow"] == "out")
        fork = next(j for j in js if any(m["type"] == "bifurcation" for m in j["members"]))
        into = next(a for a in fork["arms"] if a["vid"] == vp and a["flow"] == "in")
        assert out["s_px"] <= into["s_px"] + 2.0
        if out["linked"] is not None:
            assert out["linked"] == fork["id"] and into["linked"] == jx["id"]
    for j in js:                                                       # no junction swallows the whole scene
        assert j["radius"] < 4 * J.LAMBDA_PX


def test_type_visible_needs_three_seen_lines():
    """type_visible follows the seen arms (v2 junction audit): a junction with
    at most two seen arms is 'none'; seen arms of different vessels still
    unresolved where they leave count as one line; a crossing with three
    seen lines reads as a T."""
    ev_x = dict(kind="crossing", vessels=[1, 2])
    ev_f = dict(kind="node", ntype="fork", vessels=[1, 2, 3])

    def arm(v, vis="visible"):
        return dict(vid=v, visibility=vis)
    assert J._type_visible([ev_x], [arm(1), arm(1), arm(2), arm(2, "invisible")])[0] == "pseudo-T"
    assert J._type_visible([ev_x], [arm(1), arm(1), arm(2), arm(2)]) == ("crossing", 4)
    assert J._type_visible([ev_x], [arm(1), arm(2, "faint")])[0] == "none"
    # a fork whose two daughters leave unresolved from each other: one line, not a Y
    assert J._type_visible([ev_f], [arm(1), arm(2), arm(3)], [(1, 2)]) == ("none", 2)
    assert J._type_visible([ev_f], [arm(1), arm(2), arm(3)]) == ("bifurcation", 3)
    # a crossing beside a fork whose arms the crossing's alone do not explain: compound
    assert J._type_visible([ev_x, ev_f], [arm(1), arm(1), arm(2), arm(2), arm(3)])[0] == "compound"
