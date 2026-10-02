"""Tests of the scene integration: make_scene / save_scene / to_vesselnetwork /
label_maps on small frames (fast), and the truth rasters on a hand-built
crossing whose answer is known."""
from __future__ import annotations

import json
import math
import os

import networkx as nx
import numpy as np
import pytest
import torch

from vesselscene import render as R
from vesselscene import scene as SC
from vesselscene.graph import VesselGraph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = (160, 256)


@pytest.fixture(scope="module")
def small_scene():
    return SC.make_scene(4, "healthy", shape=SHAPE, device=DEV)


def test_scene_images_and_shapes(small_scene):
    sc = small_scene
    assert set(sc.images) == {"average", "frame"}
    for kind, img in sc.images.items():
        assert img.shape == SHAPE and img.dtype == np.float32
        v = np.isfinite(img)
        assert v.mean() > 0.9 and np.nanmin(img) >= 0 and np.nanmax(img) <= 4095 * 1.01
        assert sc.od_clean[kind].shape == SHAPE and sc.od_clean[kind].min() >= -1e-4
    assert np.isfinite(sc.images["frame"]).all()              # a raw frame has no NaN
    # the same scene: illumination and texture shared by both kinds; the texture's fine scales are blurred by
    # each image's own focus at the texture's depth (the frame's offset, the average's spread), the coarse not
    assert np.array_equal(sc.formed["average"]["lnE"], sc.formed["frame"]["lnE"])
    from scipy import ndimage as ndi
    ta, tf = (ndi.gaussian_filter(sc.formed[k]["tau"].astype(float), 8.0) for k in ("average", "frame"))
    assert np.corrcoef(ta.ravel(), tf.ravel())[0, 1] > 0.95
    # the frame is noisier than the average by an order of magnitude
    n_a = np.nanstd(sc.formed["average"]["noise"] / np.maximum(sc.formed["average"]["clean"], 1))
    n_f = np.nanstd(sc.formed["frame"]["noise"] / np.maximum(sc.formed["frame"]["clean"], 1))
    assert n_f > 5 * n_a
    # the average is fuller: along the perfused capillaries, far less of the length is (nearly) empty in the
    # average than in the frame (never-perfused capillaries are empty in both and are left out)
    caps = [vid for vid, e in sc.graph.vessels.items() if e.cls == "capillary" and vid in sc.fills["frame"]
            and sc.fills["average"][vid].state != "unperfused"]
    assert len(caps) >= 10
    low = {kind: np.mean(np.concatenate([sc.fills[kind][v].m < 0.2 for v in caps])) for kind in ("average", "frame")}
    assert low["average"] < 0.5 * low["frame"] and low["frame"] > 0.05


def test_scene_is_reproducible():
    a = SC.make_scene(5, "healthy", shape=(96, 128), device=DEV)
    b = SC.make_scene(5, "healthy", shape=(96, 128), device=DEV)
    for kind in ("average", "frame"):
        va = np.isfinite(a.images[kind])
        assert np.array_equal(va, np.isfinite(b.images[kind]))
        # GPU sums vary by ~1e-7 OD run to run; the images agree far below the noise.  The raw frame is quantized
        # (2 DN steps): those 1e-7 can flip a pixel that sits on a rounding boundary by one step (seen in 2 of 3
        # standalone runs: one pixel of 12288, 2 DN), so the frame may differ by a step at a few pixels
        d = np.abs(a.images[kind] - b.images[kind])
        if kind == "frame":
            assert np.nanmean(d > 0) < 1e-3 and np.nanmax(d) <= 4.0
        else:
            assert np.nanmax(d) < 0.05
    assert a.params["optics"] == b.params["optics"] and a.params["burst_seed"] == b.params["burst_seed"]
    # the truth too: the label rasters and the visibility tiers
    for key in ("dominant", "lumen_top", "lumen_second", "centreline", "cls", "tier"):
        assert np.mean(a.labels[key] == b.labels[key]) > 0.9999, key
    assert {v: r["average"]["tier"] for v, r in a.visibility.items()} == \
        {v: r["average"]["tier"] for v, r in b.visibility.items()}


def test_labels_are_consistent(small_scene):
    sc = small_scene
    lab = sc.labels
    ids = set(sc.graph.vessels)
    for key in ("dominant", "runner_up", "lumen_top", "lumen_second", "centreline"):
        u = set(np.unique(lab[key]).tolist()) - {-1}
        assert u <= ids, key
    both = (lab["dominant"] >= 0) & (lab["runner_up"] >= 0)
    assert (lab["dominant"][both] != lab["runner_up"][both]).all()
    assert (lab["dominant_od"] >= lab["runner_up_od"] - 1e-6).all()
    g, k = sc.graph, float(sc.params["k"])
    smp = {}

    def z_at(vid, y, x):
        if vid not in smp:
            smp[vid] = g.vessels[vid].sample(0.1)
        d = np.hypot(*(k * smp[vid]["xy"] - [x, y]).T)
        return float(smp[vid]["z"][np.argmin(d)])

    def share_node(a, b):
        return bool({g.vessels[a].u, g.vessels[a].v} & {g.vessels[b].u, g.vessels[b].v})
    # the centreline pixel of a vessel lies in its own lumen: the vessel is lumen_top or lumen_second there,
    # unless two shallower lumens cover it (the labels keep two layers; the web's wide collecting veins, pairs
    # and bundles put some vessels under two others: 3-5 % of centreline pixels on this frame)
    cl = lab["centreline"]
    m = cl >= 0
    ok = (lab["lumen_top"] == cl) | (lab["lumen_second"] == cl)
    # (> 0.85: in the v1 fix pass at least one collecting vein per scene and intact pairs and bundles put more
    # vessels under two others: 10 % of this frame's centreline pixels; the rule below is checked on them)
    assert ok[m].mean() > 0.85
    bad = np.argwhere(m & ~ok)
    for y, x in bad[np.random.default_rng(1).permutation(len(bad))[:150]]:
        v, t, s2 = int(cl[y, x]), int(lab["lumen_top"][y, x]), int(lab["lumen_second"][y, x])
        assert t >= 0 and s2 >= 0
        for w in (t, s2):
            if not share_node(v, w):
                assert z_at(w, y, x) <= z_at(v, y, x) + 0.05
    # where two lumens overlap the top one is the shallower THERE (depth of each at its nearest centreline
    # point to the pixel; vessels sharing a node have the same depth at it and are left out)
    two = np.argwhere((lab["lumen_top"] >= 0) & (lab["lumen_second"] >= 0))
    checked = 0
    for y, x in two[np.random.default_rng(0).permutation(len(two))[:400]]:
        t, s2 = int(lab["lumen_top"][y, x]), int(lab["lumen_second"][y, x])
        et, es = g.vessels[t], g.vessels[s2]
        if {et.u, et.v} & {es.u, es.v}:
            continue
        assert z_at(t, y, x) <= z_at(s2, y, x) + 0.05
        checked += 1
    assert checked >= 20 or len(two) < 100
    # class / depth maps only where the dominant OD is strong
    strong = (lab["dominant"] >= 0) & (lab["dominant_od"] >= SC.OD_MIN)
    assert (lab["cls"][strong] < 255).all() and (lab["cls"][~strong] == 255).all()
    assert np.isfinite(lab["depth"][strong]).all()
    # the dominant OD is close to the rendered vessel OD at strong single-vessel pixels
    od = sc.od_clean["average"]
    single = strong & (lab["runner_up_od"] < 0.05 * lab["dominant_od"])
    r = od[single] / lab["dominant_od"][single]
    assert 0.5 < np.median(r) < 1.2          # the halo (30 %) moves OD out of the core
    for name in SC.NODE_MAPS:
        hm = lab[f"heat_{name}"]
        assert hm.shape == SHAPE and hm.max() <= 1.0 + 1e-6


def test_crossings_mark_the_shallower(small_scene):
    for c in small_scene.crossings:
        a, b = c["vessels"]
        za, zb = c["depth"]
        assert c["above"] == (a if za <= zb else b)


def test_visibility_tiers(small_scene):
    sc = small_scene
    tiers = {r["average"]["tier"] for r in sc.visibility.values()}
    assert tiers <= {"required", "dont_care", "invisible", "out_of_frame"}
    req = [v for v, r in sc.visibility.items() if r["average"]["tier"] == "required"]
    assert req
    for v in req:
        r = sc.visibility[v]["average"]
        assert r["max_cnr"] >= SC.CNR_REQUIRED and r["visible_run_px"] >= SC.LAMBDA_PX
        assert r["visible_px"] >= r["visible_run_px"]


def test_save_and_load(small_scene, tmp_path):
    VesselNetwork = pytest.importorskip("vesselmap.network").VesselNetwork
    from vesselscene import views as V
    files = SC.save_scene(small_scene, str(tmp_path))
    for name in ("still_average", "still_frame", "truth", "digraph", "labels", "scene", "overlay_class", "map",
                 "vessels", "crossings"):
        assert os.path.getsize(files[name]) > 0, name
    g = VesselGraph.load(files["truth"])
    assert len(g.vessels) == len(small_scene.graph.vessels)
    G = nx.read_graphml(files["digraph"])
    assert G.number_of_edges() == len(g.vessels)
    meta = json.load(open(files["scene"], encoding="utf-8"))
    assert meta["seed"] == 4 and meta["shape"] == list(SHAPE) and "timings" in meta
    import tifffile
    fr = tifffile.imread(files["still_frame"])
    assert fr.dtype == np.uint16                              # a raw frame, like frame_*.tif
    assert np.array_equal(fr, np.rint(small_scene.images["frame"]).astype(np.uint16))
    net = VesselNetwork.load(files["map"])
    assert 0 < len(net.edges) <= len(g.vessels)
    assert net.to_digraph().number_of_edges() == len(net.edges)      # vesselmap's own tools run on it
    back = V.load_scene(str(tmp_path))
    assert set(back.images) == {"average", "frame"}
    assert np.array_equal(back.labels["dominant"], small_scene.labels["dominant"])
    # the image-junction truth (v2): one JSON per image, the average's overlay, the counts in scene.json
    for name in ("junctions_average", "junctions_frame", "junctions_overlay"):
        assert os.path.getsize(files[name]) > 0, name
    assert {k: len(v) for k, v in back.junctions.items()} == {k: len(v) for k, v in small_scene.junctions.items()}
    assert meta["counts"]["junctions"]["average"]["n"] == len(small_scene.junctions["average"])
    assert meta["junction_rules"]["lambda_px"] == SC.LAMBDA_PX
    assert back.labels["junction_heat_any"].dtype == np.float16 and "junction_arm_dir_frame" in back.labels


def test_scene_junctions(small_scene):
    """make_scene's junction truth: per image, on vessels of the graph, its rasters (the frame's with _frame)."""
    from vesselscene import junctions as J
    sc = small_scene
    assert set(sc.junctions) == {"average", "frame"}
    H, W = SHAPE
    for kind, js in sc.junctions.items():
        assert js, kind
        suf = "" if kind == "average" else "_frame"
        heat = sc.labels[f"junction_heat_any{suf}"]
        assert heat.shape == SHAPE and heat.max() <= 1.0 + 1e-6
        for j in js:
            assert j["type"] in J.TYPES and j["type_visible"] in J.VISIBLE_TYPES
            assert all(a["vid"] in sc.graph.vessels for a in j["arms"])
            x, y = int(round(j["x"])), int(round(j["y"]))
            if 0 <= x < W and 0 <= y < H:
                assert heat[y, x] > 0.5                      # each junction's centre is on its heatmap
            groups = sorted(i for grp in j["partition"] for i in grp)
            assert groups == list(range(len(j["arms"])))     # the partition covers every arm once
        # the node events are graph nodes
        nodes = {m["node"] for j in js for m in j["members"] if m["type"] in ("bifurcation", "confluence",
                                                                              "anastomosis")}
        assert nodes <= set(sc.graph.nodes), kind
    # the frame's junctions differ from the average's (its own focus, filling and noise), the geometry not
    assert sc.labels["junction_arm_dir_frame"].shape == SHAPE + (2,)


def _crossing_graph():
    """Two straight vessels crossing at 90 deg: a thin shallow one (z 10) and
    a wide deep one (z 40)."""
    g = VesselGraph(dict(k=3.0))
    a0, a1 = g.add_node((-20, 20), depth=10, pressure=1.0), g.add_node((60, 20), depth=10, pressure=0.0)
    b0, b1 = g.add_node((20, -20), depth=40, pressure=1.0), g.add_node((20, 60), depth=40, pressure=0.0)
    va = g.add_vessel(a0, a1, radius=0.6, depth=10.0, hct=0.6, cls="capillary")
    vb = g.add_vessel(b0, b1, radius=3.0, depth=40.0, hct=1.0, cls="venule", side="V")
    return g, va, vb


def test_label_maps_on_a_known_crossing():
    g, va, vb = _crossing_graph()
    k, shape = 3.0, (120, 120)
    opt = R.Optics()
    od, plan, pad = SC._render(g, k, shape, opt, None, DEV)
    lab = SC.label_maps(g, plan, shape, pad, k, 0.0, DEV)
    y, x = int(round(3 * 20)), int(round(3 * 20))
    # at the crossing the wide vessel dominates the OD, the shallow one is on top
    assert lab["dominant"][y, x] == vb and lab["runner_up"][y, x] == va
    assert lab["lumen_top"][y, x] == va and lab["lumen_second"][y, x] == vb
    # along the thin vessel away from the crossing, it dominates
    assert lab["dominant"][y, 10] == va and lab["dominant"][y, 110] == va
    # the label OD of a vessel alone equals its own render without the halo, near its centreline
    od0 = R.vessel_od(g, k, shape, opt, device=DEV, vids=[vb])
    col = slice(20, 100)
    assert np.allclose(lab["dominant_od"][90:100, x], od0[90:100, x], rtol=0.02, atol=2e-4)
    cl = SC.centreline_raster(g, k, shape)
    assert cl[y, x] == va                     # the shallower is drawn on top
    assert cl[100, x] == vb and cl[y, 100] == va


def test_to_vesselnetwork_is_exact():
    pytest.importorskip("vesselmap.network")          # the export needs the LIMBUS repository
    g, va, vb = _crossing_graph()
    k = 3.0
    net = SC.to_vesselnetwork(g, k, (0.0, 0.0), (120, 120), R.Optics(), blur_px=0.85)
    assert len(net.edges) == 2
    for e in net.edges.values():
        v = g.vessels[e.info["true_vessel"]]
        assert np.allclose(e.ctrl, k * v.ctrl)
        smp = net.sample(list(net.edges.keys())[list(net.edges.values()).index(e)], 1.0)
        assert np.allclose(smp["r"], k * float(np.mean(v.r_ctrl)), rtol=1e-3)
        s_expect = math.sqrt(float(R.Optics().blur(float(np.mean(v.z_ctrl)))) ** 2 + 0.85 ** 2)
        assert np.allclose(smp["s"], s_expect, rtol=1e-3)
        assert e.info["fit_err_r"] < 1e-3
    assert net.meta["optics"]["halo_weight"] == R.Optics().halo_h


def _formed_flat(shape, rms_dn=3.0, level=2000.0, seed=0):
    """A form_image-like record: a flat background with white noise."""
    rng = np.random.default_rng(seed)
    return dict(background=np.full(shape, level, np.float32), noise=rng.normal(0, rms_dn, shape).astype(np.float32),
                valid=np.ones(shape, bool), n_frames=100)


def test_visibility_is_like_for_like():
    """The CNR numerator is the vessel's own response in the DoG band (the
    same filter the background RMS is measured with), not its full peak
    contrast (which overstated the CNR about 5x): it matches the DoG of the
    rendered vessel measured on the image."""
    from scipy.ndimage import gaussian_filter
    from vesselscene import imaging as IM
    g = VesselGraph(dict(k=3.0))
    a0, a1 = g.add_node((-30, 20), depth=10, pressure=1.0), g.add_node((70, 20), depth=10, pressure=0.0)
    g.add_vessel(a0, a1, radius=1.2, depth=10.0, hct=0.8, cls="venule", side="V")
    k, shape, opt, blur = 3.0, (120, 160), R.Optics(), 0.85
    od, plan, pad = SC._render(g, k, shape, opt, None, DEV)
    odb = gaussian_filter(od.astype(np.float64), blur, mode="nearest")
    tb = plan.tubes
    i = int(np.argmin(np.hypot(tb["cx"] - pad - 80, tb["cy"] - pad - 60)))
    one = {key: v[i:i + 1] for key, v in tb.items()}
    pk, resp = SC.band_response(one, opt, blur, IM.BANDS[:-1])
    for bi, b in enumerate(IM.BANDS[:4]):
        img = IM._dog(odb, b)[60, 80]
        assert resp[0, bi] == pytest.approx(img, rel=0.05, abs=2e-4), b
    assert pk[0] == pytest.approx(odb[60, 80], rel=0.05)
    assert resp[0].max() < 0.6 * pk[0]                        # the band response is well below the peak
    # the ratio: CNR = response / background band RMS, best band
    formed = _formed_flat(shape)
    vis = SC.visibility(plan, shape, pad, opt, blur, formed, formed["background"] + formed["noise"], "average")
    rms = SC.background_band_rms(formed, formed["background"] + formed["noise"], "average")
    best = max(resp[0, bi] / rms[b] for bi, b in enumerate(IM.BANDS[:-1]))
    assert vis[0]["max_cnr"] == pytest.approx(best, rel=0.05)


def test_visible_run_is_contiguous():
    assert SC._longest_run(np.ones(6), np.array([1, 1, 0, 1, 1, 1], bool)) == 3.0
    assert SC._longest_run(np.array([5.0, 5, 5]), np.array([0, 0, 0], bool)) == 0.0


def test_vessel_just_outside_the_frame_is_scored():
    """A vessel whose centreline runs 1.5 px outside the top edge but whose
    lumen darkens the edge rows is scored (the first version tested piece
    centres only: 'out_of_frame', coded 'invisible' in the tier map)."""
    k, shape = 3.0, (60, 90)
    g = VesselGraph(dict(k=k))
    a0, a1 = g.add_node((-20, -0.5), depth=10, pressure=1.0), g.add_node((50, -0.5), depth=10, pressure=0.0)
    g.add_vessel(a0, a1, radius=1.0, depth=10.0, hct=1.0, cls="venule", side="V")
    opt = R.Optics()
    od, plan, pad = SC._render(g, k, shape, opt, None, DEV)
    assert od[0].max() > 0.05                                 # it darkens the top rows
    formed = _formed_flat(shape)
    vis = SC.visibility(plan, shape, pad, opt, 0.85, formed, formed["background"] + formed["noise"], "average")
    assert 0 in vis and vis[0]["tier"] in ("required", "dont_care")


# ------------------------------------------------------------------ v1: focus over the field, frame against average
def test_focus_presets_span_the_calibrated_range():
    """The healthy preset's focal surface varies enough over a 1920 x 1200
    frame (k = 4) that whole regions go soft (README "Calibration, v1
    optics"), it is drawn per scene (an annotation must not rely on it), and
    the single frame's offset is a draw of the burst's jitter."""
    from dataclasses import replace
    xs, ys = np.meshgrid(np.linspace(0, 479.75, 49), np.linspace(0, 299.75, 31))
    XY = np.stack([xs, ys], -1)
    soft, span, z_off, tilts = [], [], [], []
    for seed in range(200):
        o = replace(SC.optics_preset("healthy", np.random.default_rng(seed)), focus_x0=239.875, focus_y0=149.875)
        assert SC.FOCUS_JITTER[0] <= o.focus_spread <= SC.FOCUS_JITTER[1] and len(o.focus_modes) in (0, 4)
        f = SC.frame_optics(o, np.random.default_rng(seed + 1000))
        assert f.focus_spread == 0.0
        z_off.append(f.focus_offset / o.focus_spread)
        s = f.blur(15.0, XY)                                   # a conjunctival vessel's blur over the field
        soft.append(np.mean(s > 1.5 * s.min()))
        zf = f.focal_depth(XY)
        span.append(zf.max() - zf.min())
        tilts.append(math.hypot(o.focus_gx, o.focus_gy))
    soft, span = np.array(soft), np.array(span)
    assert np.median(soft) > 0.4 and np.mean(soft >= 0.2) > 0.9       # soft regions in nearly every scene
    assert 150 < np.median(span) < 300 and np.std(span) > 30          # d_c across the frame, varying per scene
    assert SC.FOCUS_TILT[0] <= min(tilts) and max(tilts) <= SC.FOCUS_TILT[1]
    # the frame's offset is focus_spread x tanh(n), n ~ N(0, 1) (v1 fix, realism review T7): within the spread, SD
    # E[tanh(n)^2]^0.5 = 0.63 of it (it was N(0, spread))
    assert np.std(z_off) == pytest.approx(0.63, abs=0.08) and abs(np.mean(z_off)) < 0.15
    assert np.max(np.abs(z_off)) < 1.0


def _vessel_field(k=4.0, shape=(480, 768), seed=0):
    """A hand-built anatomy (independent of anatomy.generate): 14 long, gently
    curved vessels of radius 1.3-3 d_c at depths 8-30 d_c crossing the frame
    in all directions, with flows (so their red cells and aggregates move)."""
    rng = np.random.default_rng(seed)
    g = VesselGraph(meta=dict(k=k))
    H, W = shape[0] / k, shape[1] / k
    for i in range(14):
        ang = rng.uniform(0, np.pi)
        c = np.array([rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H])
        t = np.linspace(-1.2 * W, 1.2 * W, 400)
        d = np.array([np.cos(ang), np.sin(ang)])
        n = np.array([-d[1], d[0]])
        path = c + t[:, None] * d + (6.0 * np.sin(t / rng.uniform(25, 60)))[:, None] * n
        z = float(rng.uniform(8, 30))
        a = g.add_node(path[0], depth=z, kind="inlet", pressure=1.0)
        b = g.add_node(path[-1], depth=z, kind="outlet", pressure=0.0)
        r = float(rng.uniform(1.3, 3.0))
        vid = g.add_vessel(a, b, path=path, radius=r, hct=1.0)
        g.vessels[vid].flow = r * r
    return g


def test_frame_is_sharper_and_grainier_than_its_average():
    """End to end: a frame at the burst's mean focus against its average,
    measured by tells.frame_average_tells as on the real pair (15-50-52:
    extra blur of the average 0.56-1.14 px over 8 frames, the frame grainier
    along medium vessels, 0.0063 Np frame - average).  The average is
    blurred by the frames' focus mixture and the registration field, the
    frame is not.  (A hand-built anatomy, so the test does not follow
    anatomy.generate.)"""
    from vesselscene import tells as T
    sc = SC.make_scene(3, "healthy", shape=(480, 768), device=DEV, labels=False, frame_focus_offset=0.0,
                       graph=_vessel_field())
    assert sc.params["optics_frame"]["focus_offset"] == sc.params["optics"]["focus_offset"]
    assert sc.params["optics_frame"]["focus_spread"] == 0.0 < sc.params["optics"]["focus_spread"]
    rs = sc.formed["average"]["reg_sd"]
    assert rs.max() > 1.5 * rs.min()                                    # the registration blur varies over the field
    fr = np.clip(np.rint(sc.images["frame"]), 0, 4095)
    p = T.frame_average_tells(fr, sc.images["average"].astype(float))
    assert p["n_sites"] > 300
    assert 0.5 < p["sigma_extra_med"] < 2.0
    assert p["edge_ratio"] < 1.0
    # grainier: the frame - average difference along the vessels has the red-cell aggregates' grain
    assert p["grain_frame"] > 1.2 * p["grain_avg"] and p["grain_coarse"] > 0.003


def test_cli_junctions_does_not_overwrite_the_exact_truth(small_scene, tmp_path):
    """v2 correctness review, defect 5: `python -m vesselscene junctions`
    recomputes the junctions from the saved files (an approximation) and
    used to overwrite the exact junctions_<kind>.json, leaving the label
    rasters and counts stale.  It now writes junctions_<kind>_recomputed.json;
    --force replaces the JSON together with the rasters and counts."""
    from vesselscene import junctions as J
    from vesselscene.__main__ import main
    d = str(tmp_path)
    SC.save_scene(small_scene, d)
    p = os.path.join(d, "junctions_average.json")
    before = open(p, encoding="utf-8").read()
    assert main(["junctions", d, "--kinds", "average"]) == 0
    assert open(p, encoding="utf-8").read() == before
    rec = J.load_junctions(os.path.join(d, "junctions_average_recomputed.json"))
    assert rec
    assert main(["junctions", d, "--kinds", "average", "--force"]) == 0
    assert J.load_junctions(p) == rec
    meta = json.load(open(os.path.join(d, "scene.json"), encoding="utf-8"))
    assert meta["counts"]["junctions"]["average"]["n"] == J.summary(rec)["n"]
    lab = np.load(os.path.join(d, "labels.npz"))
    ras = J.junction_rasters(rec, lab["junction_heat_any"].shape)
    assert np.allclose(lab["junction_heat_any"], ras["heat_any"].astype(np.float16))
    assert "junction_heat_any_frame" in lab.files                   # the frame's rasters are kept
