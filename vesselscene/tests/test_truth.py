"""Tests of the observable and complete ground truth (truth.py) on hand-built
VesselGraphs whose answers are known: a vessel fading mid-way (its run ends
'fade'), a fork with one daughter below the threshold (no observable junction,
a hidden_junction node), a crossing with one vessel invisible (no junction,
the line passes through), a vessel leaving the frame ('border'), touching
vessels and sibling daughters running together (merged); and on generated
scenes: the frame shows less than its average, the thresholds change the
counts monotonically, files, rasters, the CLI recomputation without
re-rendering, scoring.  Regression tests of the truth review's defects (D1-D4
and the smaller issues) are marked 'review'."""
from __future__ import annotations

import json
import math
import os

import networkx as nx
import numpy as np
import pytest
import torch

from vesselscene import junctions as J
from vesselscene import truth as T
from vesselscene.graph import VesselGraph

K = 4.0
SHAPE = (160, 256)                  # 40 x 64 d_c
NOISE = 0.004                       # Np per DoG band (as test_junctions: full-haematocrit vessels well visible)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _line(g, a, b, r, z, cls="venule", side="V", hct=1.0):
    na = g.add_node(a, depth=z if np.isscalar(z) else z[0])
    nb = g.add_node(b, depth=z if np.isscalar(z) else z[-1])
    return g.add_vessel(na, nb, radius=r, depth=z, hct=hct, cls=cls, side=side)


def _obs(g, **kw):
    """The observable truth of a hand-built graph (closed form, NOISE per band), with its profiles."""
    S = J.junction_samples(g, K, (0.0, 0.0), SHAPE, noise=NOISE)
    ctx = T._Ctx(g, S)
    ctx.junctions_all = J.junctions_from_samples(S, g, keep_hidden=True)
    obs, prof = T.observable_truth(ctx, "average", return_profiles=True, **kw)
    assert T.check_observable(obs, prof) == []
    json.dumps(obs, allow_nan=False)
    return obs, prof


def _obs_cnr(g, cnr_fn, **kw):
    """As _obs, with the CNR of some samples set by cnr_fn(vid, samples) -> CNR array or None."""
    S = J.junction_samples(g, K, (0.0, 0.0), SHAPE, noise=NOISE)
    for v, d in S.v.items():
        c = cnr_fn(v, d)
        if c is not None:
            d["cnr"] = np.asarray(c, float)
    S.set_seen()
    ctx = T._Ctx(g, S)
    ctx.junctions_all = J.junctions_from_samples(S, g, keep_hidden=True)
    obs, prof = T.observable_truth(ctx, "average", return_profiles=True, **kw)
    assert T.check_observable(obs, prof) == []
    json.dumps(obs, allow_nan=False)
    return obs, prof


def _nodes(obs, kind):
    return [n for n in obs["graph"]["nodes"] if n["kind"] == kind]


def _fork(daughter_hct=1.0):
    """A parent from the left into a node at (30, 20) d_c = (120, 80) px; one daughter straight on, a thinner
    one 50 deg off (its haematocrit daughter_hct)."""
    g = VesselGraph(dict(k=K))
    p0, n = g.add_node((-10, 20), depth=10), g.add_node((30, 20), depth=10)
    d1 = g.add_node((80, 20), depth=10)
    a = math.radians(50)
    d2 = g.add_node((30 + 40 * math.cos(a), 20 + 40 * math.sin(a)), depth=10)
    vp = g.add_vessel(p0, n, radius=1.2)
    v1 = g.add_vessel(n, d1, radius=1.1)
    v2 = g.add_vessel(n, d2, radius=0.7, hct=daughter_hct)
    return g, (vp, v1, v2), n


def test_a_vessel_fading_mid_way_ends_by_fade():
    """Its haematocrit falls from full to 0 between 45 % and 60 % of its length: the run starts at the frame's
    left edge ('border') and ends where its CNR falls below CNR_OBSERVE ('fade'); the stretch between that and
    CNR_DONT_CARE is don't care, not observable."""
    g = VesselGraph(dict(k=K))
    na, nb = g.add_node((-10, 20), depth=10), g.add_node((80, 20), depth=10)
    hs = np.interp(np.linspace(0, 1, 40), [0, 0.45, 0.6, 1], [1, 1, 0.0, 0.0])
    v = g.add_vessel(na, nb, radius=1.0, depth=10.0, hct=np.maximum(hs, 1e-3))
    obs, prof = _obs(g)
    (r,) = obs["runs"]
    assert r["vid"] == v and r["start"]["label"] == "border" and r["end"]["label"] == "fade"
    assert r["start"]["xy"][0] < 1.0                                    # at the left edge of the frame
    p = T.profile_of(prof, v)
    s_end = r["s1_px"]
    after = (p["s_px"] > s_end + 1) & p["in_frame"]
    assert (p["cnr"][after][:3] < T.CNR_OBSERVE).all()                  # it ends where the CNR drops below 3
    before = (p["s_px"] < s_end - 1) & (p["s_px"] > s_end - 10)
    assert (p["cnr"][before] >= T.CNR_OBSERVE).all()
    assert p["dont_care"].any() and not (p["dont_care"] & p["observable"]).any()
    assert obs["junctions"] == [] and obs["hidden_junctions"] == []
    assert sorted(n["kind"] for n in obs["graph"]["nodes"]) == ["border", "fade"]
    (e,) = obs["graph"]["edges"]
    assert e["vid"] == v and e["vids"] == [v]


def test_a_fork_with_a_daughter_below_the_threshold_is_a_hidden_junction():
    g, (vp, v1, v2), n = _fork(1.0)
    obs, _ = _obs(g)                                                   # control: all three seen, a bifurcation
    (j,) = obs["junctions"]
    assert j["type_observable"] == "bifurcation" and j["n_lines"] == 3 and obs["hidden_junctions"] == []
    g, (vp, v1, v2), n = _fork(0.05)                                   # the thin daughter nearly empty
    obs, prof = _obs(g)
    assert T.profile_of(prof, v2)["cnr"].max() < T.CNR_OBSERVE
    assert obs["junctions"] == []                                      # two lines: no observable junction
    (h,) = obs["hidden_junctions"]
    assert h["vids"] == sorted([vp, v1]) and h["vessel_node"] == n and h["truth_type"] == "bifurcation"
    assert h["degree"] == 2 and math.hypot(h["x"] - 120.0, h["y"] - 80.0) < 1e-6
    runs = {r["vid"]: r for r in obs["runs"]}
    assert set(runs) == {vp, v1}
    assert runs[vp]["end"]["label"] == runs[v1]["start"]["label"] == "hidden_junction"
    assert runs[vp]["end"]["node"] == runs[v1]["start"]["node"] == h["node"]
    # the hidden junction is a don't-care junction for scoring
    assert any(d["source"] == "hidden_junction" for d in obs["dont_care_junctions"])


def test_a_crossing_with_one_vessel_invisible_is_no_junction():
    """a horizontal at depth 10, b crossing at 60 deg nearly empty: the observable line of a passes through the
    crossing: one run, one edge from border to border, no node there."""
    g = VesselGraph(dict(k=K))
    va = _line(g, (-10, 20), (80, 20), 1.0, 10.0)
    t = math.radians(60)
    c, d = np.array([32.0, 20.0]), 50 * np.array([math.cos(t), math.sin(t)])
    vb = _line(g, c - d, c + d, 1.0, 30.0, cls="arteriole", side="A", hct=0.03)
    assert len(g.crossings()) == 1
    obs, prof = _obs(g)
    assert T.profile_of(prof, vb)["observable"].sum() == 0
    assert obs["junctions"] == [] and obs["hidden_junctions"] == []
    (r,) = obs["runs"]
    assert r["vid"] == va and r["start"]["label"] == r["end"]["label"] == "border" and r["junctions"] == []
    (e,) = obs["graph"]["edges"]
    assert e["vid"] == va and e["length_px"] > 250
    # the complete truth still has the crossing
    full = J.junctions_from_samples(J.junction_samples(g, K, (0, 0), SHAPE, noise=NOISE), g, keep_hidden=True)
    assert any(m["type"] == "crossing" for jc in full for m in jc["members"])


def test_a_vessel_leaving_the_frame_ends_at_the_border():
    g = VesselGraph(dict(k=K))
    v = _line(g, (-10, 20), (40, 70), 1.0, 10.0)                     # enters on the left, leaves at the bottom
    obs, _ = _obs(g)
    (r,) = obs["runs"]
    assert r["start"]["label"] == r["end"]["label"] == "border"
    x, y = r["end"]["xy"]
    assert abs(y - (SHAPE[0] - 0.5)) < 1.5                              # at the bottom edge
    assert sorted(n["kind"] for n in obs["graph"]["nodes"]) == ["border", "border"]
    assert obs["junctions"] == []


def _kiss(width):
    """b runs beside a, 20 d_c deeper, their lumens overlapping in projection around x = 32 d_c over a stretch
    set by `width` (d_c) (as test_junctions)."""
    g = VesselGraph(dict(k=K))
    va = _line(g, (-10, 20), (80, 20), 1.0, 10.0)
    xs = np.linspace(-10, 80, 200)
    ys = 21.6 + 4.0 * (1 - np.exp(-((xs - 32) / width) ** 2))
    na, nb = g.add_node((xs[0], ys[0]), depth=30), g.add_node((xs[-1], ys[-1]), depth=30)
    vb = g.add_vessel(na, nb, path=np.stack([xs, ys], 1), radius=1.0, depth=30.0, cls="arteriole", side="A")
    return g, va, vb


def test_touching_vessels_merge_into_one_line():
    """Two vessels unresolved over a long stretch: one line in the image.  The weaker (deeper) one is merged
    into the stronger's line between where they meet and where they part (two observable junctions, three
    lines each); the edge between them carries both vessel ids."""
    g, va, vb = _kiss(30.0)
    obs, prof = _obs(g)
    assert len(obs["junctions"]) == 2
    for j in obs["junctions"]:
        assert j["n_lines"] == 3 and j["type"] == "touching" and j["type_observable"] == "pseudo-T"
    ids = {j["node"] for j in obs["junctions"]}
    between = [e for e in obs["graph"]["edges"] if {e["source"], e["target"]} == ids]
    assert len(between) == 1 and between[0]["vid"] == va and between[0]["vids"] == sorted([va, vb])
    (m,) = obs["merged"]
    assert m["vid"] == vb and m["into"] == va and m["s1_px"] - m["s0_px"] > 2 * T.LAMBDA_PX
    runs_b = [r for r in obs["runs"] if r["vid"] == vb]
    assert len(runs_b) == 2 and runs_b[0]["end"]["label"] == runs_b[1]["start"]["label"] == "junction"
    p = T.profile_of(prof, vb)
    assert ((p["merged_into"] == va) == (p["observable"] & ~p["line"] & ~p["absorbed"])).all()


def _siblings(off=1.5, run_len=14.0):
    """A fork at (25, 20) d_c whose thinner daughter starts beside its sibling (off d_c apart) for run_len d_c
    and then parts from it."""
    g = VesselGraph(dict(k=K))
    p0, n = g.add_node((-10, 20), depth=10), g.add_node((25, 20), depth=10)
    d1 = g.add_node((80, 20), depth=10)
    xs = np.linspace(25, 75, 120)
    x1 = 25 + run_len
    ys = 20 + np.where(xs < 28, off * (xs - 25) / 3, off) + np.where(xs > x1, 0.5 * (xs - x1), 0.0)
    path = np.stack([xs, ys], 1)
    vp = g.add_vessel(p0, n, radius=1.2)
    v1 = g.add_vessel(n, d1, radius=1.1)
    v2 = g.add_vessel(n, g.add_node(path[-1], depth=10), path=path, radius=0.8)
    return g, vp, v1, v2, n


def test_siblings_running_together_merge_and_part_like_a_fork():
    """At the fork the two daughters are one line (2 lines: no observable junction; the parent runs on into
    the stronger daughter: hidden_junction); the weaker daughter's run starts where it parts from its sibling,
    at a merge junction the image shows as a bifurcation (its end labelled 'junction', the label of its node;
    merged_with names the line it was merged into)."""
    g, vp, v1, v2, n = _siblings()
    obs, _ = _obs(g)
    (h,) = obs["hidden_junctions"]
    assert h["vids"] == sorted([vp, v1]) and h["vessel_node"] == n
    (j,) = obs["junctions"]
    assert j["source"] == "merge" and j["type"] == "merge" and j["type_observable"] == "bifurcation"
    assert j["n_lines"] == 3 and j["x"] > 25 * K + 10                    # where they part, not at the node
    (r2,) = [r for r in obs["runs"] if r["vid"] == v2]
    assert r2["start"]["label"] == "junction" and r2["start"]["merged_with"] == v1 and r2["start"]["node"] == j["node"]
    assert r2["start"]["junction"] == j["id"]
    lines = {(ln["vid"], ln["flow"]): ln["vids"] for ln in j["lines"]}
    assert lines[(v1, "in")] == sorted([v1, v2]) and lines[(v1, "out")] == [v1] and lines[(v2, "out")] == [v2]


def test_observable_junction_types_follow_the_lines():
    """3 lines: bifurcation / confluence by the flow when they are one node's vessels, else pseudo-T; 4+:
    crossing when two vessels pass through, else compound."""
    ev_f = [dict(type="bifurcation", vessels=[1, 2, 3], node=7, parents=[1], children=[2, 3])]
    ev_c = [dict(type="confluence", vessels=[1, 2, 3], node=7, parents=[1, 2], children=[3])]

    def ln(v, flow):
        return dict(vid=v, flow=flow)
    assert T._obs_type([ln(1, "in"), ln(2, "out"), ln(3, "out")], ev_f, None) == "bifurcation"
    assert T._obs_type([ln(1, "in"), ln(2, "in"), ln(3, "out")], ev_c, None) == "confluence"
    assert T._obs_type([ln(1, "in"), ln(1, "out"), ln(3, "in")], ev_f, None) == "pseudo-T"
    assert T._obs_type([ln(1, "in"), ln(1, "out"), ln(4, "in"), ln(4, "out")], [], None) == "crossing"
    assert T._obs_type([ln(1, "in"), ln(2, "out"), ln(3, "out"), ln(4, "in")], ev_f, None) == "compound"
    assert T._obs_type([ln(1, "in"), ln(2, "out")], ev_f, None) == "none"


# ======================================================================== review regressions (hand-built)
def test_review_d1_a_wide_vessel_inside_a_chain_of_claims_keeps_its_line():
    """review D1: a 36 px wide vessel crossed every 8 d_c by thin vessels at 45 deg lies wholly inside the
    crossings' claims (its own crossing core is its width / tan 45 deg long).  It used to be on no line and was
    absorbed (209 px); now its line runs through all 7 crossings."""
    g = VesselGraph(dict(k=K))
    vv = _line(g, (6, 20), (58, 20), 4.5, 30.0, cls="episcleral_vein")
    t = math.radians(45)
    xs = np.arange(8, 58, 8.0)
    for x in xs:
        c, dd = np.array([x, 20.0]), 40 * np.array([math.cos(t), math.sin(t)])
        _line(g, c - dd, c + dd, 0.8, 5.0, cls="arteriole", side="A")
    obs, prof = _obs(g)
    p = T.profile_of(prof, vv)
    assert p["observable"].sum() > 200
    (r,) = [r for r in obs["runs"] if r["vid"] == vv]
    assert r["length_px"] > 200 and len(r["junctions"]) == len(xs)
    assert (p["line"] == p["observable"]).all() and not p["absorbed"].any()
    for j in obs["junctions"]:
        assert vv in {ln["vid"] for ln in j["lines"]} and vv not in j["hidden_vessels"]
    # every edge of the vein carries a polyline (attachment to attachment, linked between touching claims)
    ev = [e for e in obs["graph"]["edges"] if e["vid"] == vv]
    assert all(len(e["xy"]) >= 2 for e in ev)
    assert sum(e["length_px"] for e in ev) > 0.8 * r["length_px"]


def test_review_d2_a_parent_running_into_its_only_observable_daughter_through_a_crossing_is_an_x():
    """review D2: a fork (parent, a daughter straight on, a second daughter 50 deg off) with a vessel crossing
    at the node: all seen, 5 lines, compound; the crossing vessel invisible, a bifurcation; the second daughter
    invisible, the image is an X: 'crossing' (it was 'compound': two vessel ids each in and out were required)."""
    def fork(drop):
        g = VesselGraph(dict(k=K))
        p0, n = g.add_node((-10, 20), depth=10), g.add_node((32, 20), depth=10)
        d1 = g.add_node((80, 20), depth=10)
        a = math.radians(-50)
        d2 = g.add_node((32 + 40 * math.cos(a), 20 + 40 * math.sin(a)), depth=10)
        vp = g.add_vessel(p0, n, radius=1.2)
        v1 = g.add_vessel(n, d1, radius=1.1)
        v2 = g.add_vessel(n, d2, radius=0.9, hct=0.02 if drop == "daughter" else 1.0)
        t = math.radians(70)
        c, dd = np.array([33.0, 20.0]), 40 * np.array([math.cos(t), math.sin(t)])
        vc = _line(g, c - dd, c + dd, 1.0, 30.0, cls="arteriole", side="A", hct=0.02 if drop == "crosser" else 1.0)
        return g, (vp, v1, v2, vc)
    expect = {None: ("compound", 5), "crosser": ("bifurcation", 3), "daughter": ("crossing", 4)}
    for drop, (typ, n) in expect.items():
        g, (vp, v1, v2, vc) = fork(drop)
        obs, _ = _obs(g)
        (j,) = obs["junctions"]
        assert (j["type_observable"], j["n_lines"]) == (typ, n), drop
        if drop == "daughter":
            assert j["hidden_vessels"] == [v2]
            pairs = T._through_pairs(j["lines"], j["members"])
            assert sorted(tuple(sorted((j["lines"][a]["vid"], j["lines"][b]["vid"]))) for a, b in pairs) == \
                sorted([tuple(sorted((vp, v1))), (vc, vc)])
    # the rule itself: a parent and its child pair, a vessel of another node does not; a merged line stands for
    # every vessel id it carries
    fork_ev = [dict(type="bifurcation", vessels=[1, 2, 3], node=7, parents=[1], children=[2, 3])]

    def ln(v, flow, vids=None):
        return dict(vid=v, flow=flow, vids=vids or [v])
    assert T._obs_type([ln(1, "in"), ln(2, "out"), ln(4, "in"), ln(4, "out")], fork_ev, None) == "crossing"
    assert T._obs_type([ln(1, "in"), ln(5, "out"), ln(4, "in"), ln(4, "out")], fork_ev, None) == "compound"
    assert T._obs_type([ln(1, "in"), ln(2, "out"), ln(3, "out"), ln(4, "in")], fork_ev, None) == "compound"
    assert T._obs_type([ln(1, "in", [1, 6]), ln(6, "out"), ln(4, "in"), ln(4, "out")], [], None) == "crossing"


def test_review_d3_one_junctions_claims_close_together_make_no_loop():
    """review D3: one junction's two claims on a vessel 1 sample apart made a 0 px junction-to-itself edge,
    counted twice in its degree (X's typed compound with 6 lines); junctions.py's 1-D split of overlapping claims
    leaving A B A on a vessel made the vessel run A -> B -> A.  Now the gap belongs to the junction, and the
    piece of A farther from A's centre to B; claims of one junction farther apart than ARM_MIN_PX stay apart (a
    vessel leaving a junction and coming back is a loop)."""
    class _S:
        def __init__(self, n):
            self.v = {0: dict(n=n, ds=1.0, C=np.stack([np.arange(n, dtype=float), np.zeros(n)], 1))}
    S = _S(80)
    G = T._G(S)
    cen = {1: np.array([12.0, 0.0]), 2: np.array([32.0, 0.0]), 3: np.array([36.0, 0.0]), 4: np.array([65.0, 0.0])}
    claims = [(1, 0, 5, 10), (1, 0, 12, 20), (2, 0, 30, 34), (3, 0, 35, 37), (2, 0, 38, 45), (4, 0, 52, 58),
              (4, 0, 70, 75)]
    owner, segs = T._owners(S, G, claims, {1, 2, 3, 4}, T.ARM_MIN_PX, cen)
    assert segs[0] == [(5, 20, 1), (30, 34, 2), (35, 45, 3), (52, 58, 4), (70, 75, 4)]
    assert owner[11] == 1 and (owner[60:70] == -1).all()
    owner, segs = T._owners(S, G, claims, {1}, T.ARM_MIN_PX, cen)          # inactive junctions own nothing
    assert segs[0] == [(5, 20, 1)]


def test_review_d4_run_end_labels_follow_their_nodes():
    """review D4: a capillary T-joining a venule, its last 8 px faded: the venule's runs were labelled
    'hidden_junction' at what became a confluence junction, the capillary's end 'merged'.  Now every end
    carries its node's kind (and its junction id); check_observable catches a mismatch."""
    g = VesselGraph(dict(k=K))
    a0, n, b0 = g.add_node((-10, 20), depth=10), g.add_node((32, 20), depth=10), g.add_node((80, 20), depth=10)
    va = g.add_vessel(a0, n, radius=1.4)
    vb = g.add_vessel(n, b0, radius=1.4)
    vc = g.add_vessel(g.add_node((32, 60), depth=10), n, radius=0.8, cls="capillary", side="C")

    def faded(v, d):
        if v != vc:
            return None
        c = d["cnr"].copy()
        c[d["sig"] > d["L"] - 8.0] = 1.0
        return c
    obs, _ = _obs_cnr(g, faded)
    (j,) = obs["junctions"]
    assert j["type_observable"] == "confluence" and j["n_lines"] == 3
    runs = {r["vid"]: r for r in obs["runs"]}
    for end in (runs[va]["end"], runs[vb]["start"], runs[vc]["end"]):
        assert end["label"] == "junction" and end["junction"] == j["id"] and end["node"] == j["node"]
    bad = json.loads(json.dumps(obs))
    bad["runs"][0]["end"]["label"] = "hidden_junction"
    assert any("labelled hidden_junction at a junction node" in p for p in T.check_observable(bad))


def test_review_runs_stay_inside_the_frame():
    """review (smaller issues): a crossing 4-8 px from the top edge, its second vessel leaving the frame there:
    the claim's fill reached 7-10 px outside the frame.  Now lines are in-frame samples only."""
    H, W = SHAPE
    for yb in (1.0, 2.0):
        g = VesselGraph(dict(k=K))
        _line(g, (-10, yb), (80, yb), 1.5, 10.0)
        t = math.radians(35)
        c, dd = np.array([32.0, yb]), 40 * np.array([math.cos(t), math.sin(t)])
        _line(g, c - dd, c + dd, 1.0, 30.0, cls="arteriole", side="A")
        obs, prof = _obs(g)
        for r in obs["runs"]:
            P = np.asarray(r["xy"])
            assert (P >= -0.5).all() and (P[:, 0] <= W - 0.5).all() and (P[:, 1] <= H - 0.5).all()
        assert not (prof["line"] & ~prof["in_frame"]).any()


def test_review_a_line_through_a_junction_where_it_is_not_observable():
    """review (smaller issues): a thin vessel crossing under a wide black vein, not observable under it: its line
    passes through the crossing (an X), the samples under the vein are 'unobserved': not on the centreline
    raster, in the don't-care mask, and a tracing that stops at the vein's walls scores recall 1."""
    g = VesselGraph(dict(k=K))
    vv = _line(g, (-10, 20), (80, 20), 3.5, 10.0)
    t = math.radians(80)
    c, dd = np.array([32.0, 20.0]), 50 * np.array([math.cos(t), math.sin(t)])
    vt = _line(g, c - dd, c + dd, 1.0, 20.0, cls="arteriole", side="A")

    def under(v, d):
        if v != vt:
            return None
        cn = d["cnr"].copy()
        cn[np.abs(d["C"][:, 1] - 80.0) < 4.0 * K + 2] = 0.5
        return cn
    obs, prof = _obs_cnr(g, under)
    (j,) = obs["junctions"]
    assert j["type_observable"] == "crossing"
    (rt,) = [r for r in obs["runs"] if r["vid"] == vt]
    ((k0, k1),) = rt["unobserved"]
    assert k1 - k0 + 1 >= 30
    P = np.asarray(rt["xy"])
    mid = np.rint(P[(k0 + k1) // 2]).astype(int)
    ras = T.observable_rasters(obs, prof)
    assert ras["centreline"][mid[1], mid[0]] in (-1, vv)               # not the thin vessel under the vein
    assert ras["dont_care"][mid[1], mid[0]] or ras["lumen"][mid[1], mid[0]]
    tracing = [np.asarray(r["xy"]) for r in obs["runs"] if r["vid"] != vt] + [P[:k0], P[k1 + 1:]]
    s = T.score_tracing(obs, tracing, prof=prof)
    assert s["centreline_recall"] == pytest.approx(1.0) and s["centreline_precision"] == pytest.approx(1.0)
    s2 = T.score_tracing(obs, [np.asarray(r["xy"]) for r in obs["runs"]], prof=prof)
    assert s2["centreline_precision"] == pytest.approx(1.0)              # tracing through it is no error either


def test_the_run_count_is_not_monotone_in_the_threshold():
    """Why not every count can be monotone (review, smaller issues): a vessel at CNR 6 with a 20 px stretch at
    CNR 4 is one run at CNR 3 and two at CNR 5 (a line breaking in two), while its observable samples shrink."""
    g = VesselGraph(dict(k=K))
    _line(g, (-10, 20), (80, 20), 1.0, 10.0)

    def dip(vid, d):
        c = np.full(d["n"], 6.0)
        i0 = int(np.argmin(np.abs(d["C"][:, 0] - 128)))
        c[i0:i0 + 20] = 4.0
        return c
    o3, p3 = _obs_cnr(g, dip)
    o5, p5 = _obs_cnr(g, dip, cnr_observe=5.0)
    assert len(o3["runs"]) == 1 and len(o5["runs"]) == 2
    assert p5["observable"].sum() < p3["observable"].sum() and not (p5["observable"] & ~p3["observable"]).any()


# ======================================================================== generated scenes
@pytest.fixture(scope="module")
def scene():
    from vesselscene import scene as SC
    return SC.make_scene(4, "healthy", shape=(320, 512), device=DEV)


def test_make_scene_carries_both_truths(scene):
    """make_scene: the observable truth and the profiles per image; the default junction list is the visible
    subset of all junctions (equal to the keep_hidden=False computation)."""
    from vesselscene import scene as SC
    H, W = 320, 512
    for kind in ("average", "frame"):
        obs = scene.observable[kind]
        assert obs["kind"] == kind and obs["runs"] and obs["junctions"]
        assert T.check_observable(obs, scene.vis_profiles[kind]) == [], kind
        assert J.visible_subset(scene.junctions_all[kind]) == scene.junctions[kind]
        assert J.scene_junctions(scene, kind) == scene.junctions[kind]
        assert len(scene.junctions_all[kind]) >= len(scene.junctions[kind])
        prof = scene.vis_profiles[kind]
        assert prof["meta"]["noise_source"] == "exact"
        assert set(prof["vids"].tolist()) <= set(scene.graph.vessels)
        for key in T.raster_keys(kind):
            assert key in scene.labels and scene.labels[key].shape == (H, W), key
        # the runs are the line samples; each run's vessel is on the centreline raster
        cl = scene.labels[f"observable_centreline_{kind}"]
        for r in obs["runs"][:20]:
            x, y = np.rint(r["xy"][len(r["xy"]) // 2]).astype(int)
            if 0 <= x < W and 0 <= y < H:
                assert cl[y, x] == r["vid"] or cl[y, x] >= 0
        lum, dc = scene.labels[f"observable_lumen_{kind}"], scene.labels[f"observable_dont_care_{kind}"]
        assert lum.dtype == bool and not (lum & dc).any()
        assert lum[cl >= 0].mean() > 0.95                              # the lines lie in the observable lumen
        heat = scene.labels[f"observable_junction_heat_any_{kind}"]
        for j in obs["junctions"]:
            x, y = int(round(j["x"])), int(round(j["y"]))
            if 0 <= x < W and 0 <= y < H:
                assert heat[y, x] > 0.5
    assert SC.KINDS == ("average", "frame")


def test_review_d1_every_observable_vessel_of_a_scene_is_on_a_line(scene):
    """review D1 on the test scene: its two widest vessels (31 and 35 px, 81-146 px observable) were on no line
    and merged nowhere (absorbed, or lost inside a chain of junction claims).  Every vessel with MIN_RUN_PX of
    observable samples is now on a line, merged into one or internal to a junction (check_observable with the
    profiles), nothing longer than ARM_MIN_PX is absorbed, and every vessel at least 20 px wide with 2 lambda
    observable has a run of its own."""
    for kind in ("average", "frame"):
        obs, prof = scene.observable[kind], scene.vis_profiles[kind]
        assert T.check_observable(obs, prof) == [], kind
        with_run = {r["vid"] for r in obs["runs"]}
        wide = 0
        for i, v in enumerate(prof["vids"].tolist()):
            a, b = int(prof["offsets"][i]), int(prof["offsets"][i + 1])
            o = prof["observable"][a:b]
            if o.sum() >= 2 * T.LAMBDA_PX and 2 * np.median(prof["radius_px"][a:b][o]) >= 20:
                wide += 1
                assert v in with_run, (kind, v)
        assert wide >= 2
        assert obs["totals"]["absorbed_px"] < 0.01 * obs["totals"]["observable_px"]


def test_the_frame_shows_less_than_its_average(scene):
    """The single frame (its noise, red-cell gaps, own focus) has fewer observable samples, less observable
    length and no more vessels than its average.  Its run and junction counts need not be lower on this
    320 x 512 scene: the frame breaks dotted capillaries into pieces at red-cell gaps longer than bridge_px
    (more fade ends; a piece ending next to another vessel is a pseudo-T); on the full-size final scenes every
    frame has fewer runs and fewer junctions than its average (README, 'Ground truth: observable and
    complete')."""
    a, f = scene.observable["average"]["summary"], scene.observable["frame"]["summary"]
    assert f["run_length_px"] < a["run_length_px"]
    assert f["observable_px"] < a["observable_px"]
    assert f["n_vessels"] <= a["n_vessels"]
    assert f["end_labels"].get("fade", 0) > a["end_labels"].get("fade", 0)
    pa, pf = scene.vis_profiles["average"], scene.vis_profiles["frame"]
    assert pf["observable"].sum() < pa["observable"].sum()


def test_thresholds_change_the_counts_monotonically(scene):
    """CNR 2 -> 3 -> 5 (and a longer minimum run): the observable samples shrink (each a subset of the one
    before), and with them the observable length and the number of vessels with an observable stretch."""
    ctx = T.scene_context(scene, "average")
    prev = None
    for cnr, run in ((2.0, T.MIN_RUN_PX), (3.0, T.MIN_RUN_PX), (5.0, T.MIN_RUN_PX), (5.0, 3 * T.LAMBDA_PX)):
        obs, prof = T.observable_truth(ctx, "average", cnr, run, return_profiles=True)
        assert T.check_observable(obs, prof) == []
        assert obs["rules"]["cnr_observe"] == cnr and obs["rules"]["min_run_px"] == run
        cur = (prof["observable"].copy(), obs["summary"])
        if prev is not None:
            assert not (cur[0] & ~prev[0]).any()                       # a subset
            assert cur[0].sum() < prev[0].sum()
            assert cur[1]["observable_px"] < prev[1]["observable_px"]
            assert cur[1]["n_vessels_observable"] <= prev[1]["n_vessels_observable"]
            # (the runs' length and number are not monotone: a weaker vessel merged into a line at CNR 2 has
            # a line of its own at 3 when its partner is no longer observable; a line fading earlier breaks
            # in two; the junctions are not either: a vessel fading next to another makes a pseudo-T)
        prev = cur


def test_save_load_graphml_and_overlay(scene, tmp_path):
    from vesselscene import scene as SC
    from vesselscene import views as V
    files = SC.save_scene(scene, str(tmp_path))
    for kind in ("average", "frame"):
        for name in (f"observable_{kind}", f"observable_graphml_{kind}", f"observable_overlay_{kind}",
                     f"complete_visibility_{kind}", f"junctions_all_{kind}"):
            assert os.path.getsize(files[name]) > 0, name
        obs = T.load_observable(files[f"observable_{kind}"])
        assert obs == json.loads(json.dumps(scene.observable[kind]))
        G = nx.read_graphml(files[f"observable_graphml_{kind}"])
        assert G.number_of_edges() == len(obs["graph"]["edges"]) and G.number_of_nodes() == len(obs["graph"]["nodes"])
        # review: one GraphML key (one type) per attribute name ('junction', 'vessel_node' mixed int and '')
        import xml.etree.ElementTree as ET
        keys = [(k.get("for"), k.get("attr.name")) for k in ET.parse(files[f"observable_graphml_{kind}"]).getroot()
                if k.tag.endswith("key")]
        assert len(keys) == len(set(keys)), keys
        assert {d["junction"] for _, d in G.nodes(data=True)} >= {-1}
        Gn = T.to_networkx(obs)
        assert sorted(d for _, d in Gn.degree()) == sorted(n["degree"] for n in obs["graph"]["nodes"])
        prof = T.load_profiles(files[f"complete_visibility_{kind}"])
        assert np.array_equal(prof["observable"], scene.vis_profiles[kind]["observable"])
        assert prof["meta"]["rules"]["cnr_observe"] == T.CNR_OBSERVE
    meta = json.load(open(files["scene"], encoding="utf-8"))
    assert meta["counts"]["observable"]["frame"]["n_runs"] == scene.observable["frame"]["summary"]["n_runs"]
    assert meta["truth_rules"]["cnr_observe"] == T.CNR_OBSERVE
    back = V.load_scene(str(tmp_path))
    assert set(back.observable) == {"average", "frame"} and set(back.junctions_all) == {"average", "frame"}
    assert back.labels["observable_lumen_average"].dtype == bool
    ct = T.complete_truth(str(tmp_path))
    assert set(ct["junctions_all"]) == {"average", "frame"} and len(ct["graph"].vessels) == len(scene.graph.vessels)


def test_cli_truth_recomputes_without_rendering(scene, tmp_path):
    """`python -m vesselscene truth DIR --cnr C`: from the saved profiles (exact), other thresholds; the default
    thresholds reproduce the saved truth byte for byte (review: 423 junction-line values copied unrounded from
    the float32 recomputation, and the noise source, differed): the JSON, the GraphML and the rasters."""
    from vesselscene import scene as SC
    from vesselscene.__main__ import main
    d = str(tmp_path)
    SC.save_scene(scene, d, overlay=False, vesselmap_export=False)
    before = open(os.path.join(d, "observable_average.json"), encoding="utf-8").read()
    assert main(["truth", d, "--kind", "average"]) == 0
    assert open(os.path.join(d, "observable_average.json"), encoding="utf-8").read() == before
    rec = T.load_observable(os.path.join(d, "observable_average_recomputed.json"))
    sav = json.loads(before)
    assert rec["rules"]["noise_source"] == "exact"
    assert open(os.path.join(d, "observable_average_recomputed.json"), encoding="utf-8").read() == before
    assert open(os.path.join(d, "observable_average_recomputed.graphml"), "rb").read() ==         open(os.path.join(d, "observable_average.graphml"), "rb").read()
    lab = np.load(os.path.join(d, "labels.npz"))
    with np.load(os.path.join(d, "observable_average_recomputed_rasters.npz")) as ras:
        for key in ras.files:
            assert np.array_equal(ras[key], lab[f"observable_{key}_average"]), key
    assert [(r["vid"], r["s0_px"], r["s1_px"], r["start"]["label"], r["end"]["label"]) for r in rec["runs"]] == \
        [(r["vid"], r["s0_px"], r["s1_px"], r["start"]["label"], r["end"]["label"]) for r in sav["runs"]]
    assert [(j["x"], j["y"], j["type_observable"]) for j in rec["junctions"]] == \
        [(j["x"], j["y"], j["type_observable"]) for j in sav["junctions"]]
    assert main(["truth", d, "--kind", "average", "--cnr", "6", "--min-run", "20"]) == 0
    hi = T.load_observable(os.path.join(d, "observable_average_cnr6_run20.json"))
    assert hi["rules"]["cnr_observe"] == 6.0 and hi["summary"]["run_length_px"] < sav["summary"]["run_length_px"]
    ras = np.load(os.path.join(d, "observable_average_cnr6_run20_rasters.npz"))
    assert ras["lumen"].sum() < np.load(os.path.join(d, "labels.npz"))["observable_lumen_average"].sum()


def test_scoring_against_the_truths(scene):
    """The truth's own runs as a tracing score 1; nothing traced scores recall 0; a fit following every vessel
    finds the observable, don't-care and invisible samples; the calibration curve of a tracing of the
    observable lines crosses 0.5 near CNR_OBSERVE."""
    obs, prof = scene.observable["average"], scene.vis_profiles["average"]
    lines = [np.asarray(r["xy"]) for r in obs["runs"]]
    s = T.score_tracing(obs, lines, [(j["x"], j["y"], j["type_observable"]) for j in obs["junctions"]], prof=prof)
    assert s["centreline_recall"] == pytest.approx(1.0) and s["centreline_precision"] == pytest.approx(1.0)
    assert s["junction_recall"] == pytest.approx(1.0) and s["junction_precision"] == pytest.approx(1.0)
    assert s["junction_type_accuracy"] == pytest.approx(1.0)
    s0 = T.score_tracing(obs, [np.zeros((0, 2))], [], prof=prof)
    assert s0["centreline_recall"] == 0.0 and s0["junction_recall"] == 0.0
    allv = []
    for i in range(len(prof["vids"])):
        a, b = int(prof["offsets"][i]), int(prof["offsets"][i + 1])
        allv.append(np.stack([prof["x_px"][a:b], prof["y_px"][a:b]], 1))
    f = T.score_fit(prof, allv)
    assert f["recall_observable"] == pytest.approx(1.0) and f["recall_invisible"] == pytest.approx(1.0)
    obs_only = T.score_fit(prof, lines)            # the lines miss the merged vessels' own centrelines
    tot = obs["totals"]
    assert obs_only["recall_observable"] >= tot["line_px"] / tot["observable_px"] - 0.1
    assert obs_only["recall_invisible"] < 0.5
    cal = T.traced_by_cnr(prof, lines, tol_px=1.0)
    assert 1.5 < cal["cnr50"] < 6.0
