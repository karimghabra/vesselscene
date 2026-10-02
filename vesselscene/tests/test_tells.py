"""Tests of the v1 tells (vesselscene.tells): each statistic on toy
skeletons or rendered toy images whose answer is known, the frame-versus-
average estimators on images with a known extra blur and a known red-cell
modulation, and the stored real reference.  Tests that need the real data
skip when it is not available."""
import math

import cv2
import numpy as np
import pytest
import scipy.ndimage as ndi

from vesselscene import stats as S
from vesselscene import tells as T

LAM = 12.0


# ------------------------------------------------------------------ helpers
def _skel(shape, polylines):
    img = np.zeros(shape, np.uint8)
    for pl in polylines:
        for (x0, y0), (x1, y1) in zip(pl[:-1], pl[1:]):
            cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, 1)
    from skimage.morphology import skeletonize
    return skeletonize(img.astype(bool))


def _ray(c, ang_deg, r):
    a = math.radians(ang_deg)
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a))


def _od(shape, segs):
    """Optical density of straight vessels: segs = [((x0, y0), (x1, y1), width px, OD)]; union by max."""
    od = np.zeros(shape)
    for p0, p1, w, c in segs:
        m = np.zeros(shape, np.uint8)
        cv2.line(m, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), 1,
                 thickness=max(1, int(round(w))), lineType=cv2.LINE_AA)
        od = np.maximum(od, c * m)
    return od


def _image(od, blur=1.5, level=2000.0, noise=0.0, seed=0):
    I = level * np.exp(-ndi.gaussian_filter(od, blur))
    if noise:
        I = I + np.random.default_rng(seed).standard_normal(I.shape) * noise * np.sqrt(I)
    return I


def _tree(segs, p, ang, length, w, depth):
    q = (p[0] + length * math.cos(math.radians(ang)), p[1] + length * math.sin(math.radians(ang)))
    segs.append((p, q, w, 0.12 + 0.04 * w / 4))
    if depth:
        for d in (-25, 25):
            _tree(segs, q, ang + d, 0.7 * length, max(2.0, 0.75 * w), depth - 1)


# ------------------------------------------------------------------ 1. layout
def test_continuation_through_crossing_not_through_fork():
    c = (200, 200)
    x = _skel((400, 400), [[_ray(c, 180, 180), _ray(c, 0, 180)], [_ray(c, 240, 180), _ray(c, 60, 180)]])
    cp = T.continuation_paths(x, LAM)
    assert len(cp["crossings"]) == 1 and abs(cp["crossings"][0] - 60) < 6
    assert cp["n_branch"] == 0
    assert (cp["path_len"] > 300).sum() == 2          # both lines run straight through
    y = _skel((400, 400), [[_ray(c, a, 150), c] for a in (90, 210, 330)])
    cp = T.continuation_paths(y, LAM)
    assert cp["n_branch"] == 1 and len(cp["crossings"]) == 0
    assert np.all(cp["path_len"] < 170)                # a symmetric fork turns 60 deg: no continuation
    t = _skel((400, 400), [[(20, 200), (380, 200)], [(200, 200), (200, 380)]])
    cp = T.continuation_paths(t, LAM)
    assert cp["n_branch"] == 1 and len(cp["crossings"]) == 0
    assert np.sort(cp["path_len"])[-1] > 340           # the through vessel of a T continues


def test_shallow_crossing_along_a_shared_stretch():
    """Two vessels crossing at 20 deg share a 2 lambda stretch of skeleton:
    '>-<'.  The X-merge joins the two junctions and both lines continue."""
    p1, p2 = (170.0, 200.0), (170.0 + 2 * LAM, 200.0)
    a = 10.0
    pl = [[_ray(p1, 180 - a, 150), p1, p2, _ray(p2, -a, 150)], [_ray(p1, 180 + a, 150), p1, p2, _ray(p2, a, 150)]]
    cp = T.continuation_paths(_skel((400, 500), pl), LAM)
    assert len(cp["crossings"]) == 1 and abs(cp["crossings"][0] - 2 * a) < 8
    assert (cp["path_len"] > 250).sum() == 2


def test_layout_web_vs_trees():
    """A web of long crossing vessels percolates and spans the frame; separate
    trees hanging from a border do not, and their calibre changes along them."""
    shape = (480, 720)
    rng = np.random.default_rng(3)
    web = []
    for i in range(14):
        th = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(150, 570), rng.uniform(100, 380)
        d = 900 * np.array([math.cos(th), math.sin(th)])
        web.append(((cx - d[0], cy - d[1]), (cx + d[0], cy + d[1]), rng.choice([4.0, 6.0, 9.0]), 0.2))
    trees = []                                     # 5 separate trees hanging from the top border
    for x0 in (72, 216, 360, 504, 648):
        _tree(trees, (x0, 0), 90 + rng.uniform(-5, 5), 100, 9.0, 2)
    out = {}
    for name, segs in (("web", web), ("trees", trees)):
        I = _image(_od(shape, segs), noise=0.3, seed=1)
        res = S.analyse(I, keep_maps=True)
        out[name] = T.layout_tells(res)
    assert out["web"]["web_lcc"] > 0.6 and out["trees"]["web_lcc"] < 0.35
    assert out["web"]["web_span"] > 0.7 and out["trees"]["web_span"] < 0.3
    assert out["web"]["xing_rate"] > out["trees"]["xing_rate"] + 1.0
    assert out["web"]["cont_share_10"] > out["trees"]["cont_share_10"]
    for o in out.values():
        assert set(T.LAYOUT_KEYS) <= set(o)


# ------------------------------------------------------------------ 2. pairs and bundles
@pytest.mark.parametrize("n_lines", [1, 2, 3])
def test_line_groups_counts_parallel_lines(n_lines):
    H, W = 200, 400
    y = np.arange(H)[:, None] + 0 * np.arange(W)[None, :]
    F = np.zeros((H, W))
    ys = [100 + 9 * (i - (n_lines - 1) / 2) for i in range(n_lines)]
    for y0 in ys:
        F -= 0.1 * np.exp(-0.5 * ((y - y0) / 1.5) ** 2)
    F -= 0.1 * np.exp(-0.5 * ((np.arange(W)[None, :] + 0 * y - 200) / 1.5) ** 2)   # a crossing vessel
    y_mid = ys[n_lines // 2]
    pts = np.array([[y_mid, x] for x in range(60, 340, 5)], float)
    tan = np.tile([0.0, 1.0], (len(pts), 1))
    g, wl = T.line_groups(F, pts, tan, np.full(len(pts), 6.0), LAM)
    assert np.all(g == n_lines)
    assert np.all(np.abs(wl - 2.355 * 1.5) < 1.5)


def test_pair_tells_doubled_lines():
    shape = (480, 720)
    singles, doubles = [], []
    for i, x in enumerate(range(60, 700, 80)):
        singles.append(((x, 0), (x + 40, 480), 3.0, 0.15))
        doubles += [((x, 0), (x + 40, 480), 3.0, 0.15), ((x + 9, 0), (x + 49, 480), 3.0, 0.15)]
    frac = {}
    for name, segs in (("single", singles), ("double", doubles)):
        res = S.analyse(_image(_od(shape, segs), blur=1.2, noise=0.2), keep_maps=True)
        frac[name] = T.pair_tells(res)
    assert frac["double"]["pair_frac"] > 0.5 > frac["single"]["pair_frac"] + 0.3
    assert frac["double"]["pair_run5"] > frac["single"]["pair_run5"]


# ------------------------------------------------------------------ 3-4. wide vessels and focus
def test_wide_edges_and_soft_tiles():
    shape = (600, 960)
    rng = np.random.default_rng(0)
    segs = [((x, 0), (x + rng.uniform(-60, 60), 600), rng.choice([3.0, 5.0, 8.0]), 0.2) for x in range(30, 960, 45)]
    segs += [((0, 300), (960, 330), 40.0, 0.6)]
    od = _od(shape, segs)
    sharp = _image(od, blur=1.2)
    soft = _image(od, blur=3.5)
    half = _image(od, blur=1.2)
    half[:, 480:] = soft[:, 480:]
    r = {k: T.wide_focus_tells(S.analyse(v, keep_maps=True)) for k, v in (("sharp", sharp), ("soft", soft), ("half", half))}
    assert r["soft"]["wide98_edge"] > r["sharp"]["wide98_edge"] + 1.0
    assert r["sharp"]["soft_tiles"] < 0.15 and r["soft"]["soft_tiles"] < 0.15
    assert r["half"]["soft_tiles"] >= 0.3
    assert r["half"]["tile_edge_ratio"] > r["sharp"]["tile_edge_ratio"] + 0.4


# ------------------------------------------------------------------ 5. frame versus average
def test_fit_edges_recovers_the_edge_sigma():
    rng = np.random.default_rng(0)
    from scipy.special import ndtr
    n = 200
    s = rng.uniform(1.0, 4.0, n)
    h = rng.uniform(3.0, 15.0, n)
    t = np.linspace(-1, 1, 81)[None, :] * (h + 14)[:, None]
    P = 0.02 * t / 30 - 0.2 * (ndtr((t + h[:, None]) / s[:, None]) - ndtr((t - h[:, None]) / s[:, None]))
    P += rng.normal(0, 0.004, P.shape)
    f = T.fit_edges(t, P, h0=2 * h * rng.uniform(0.7, 1.3, n), s0=np.full(n, 2.5))
    assert np.median(np.abs(f["s"] / s - 1)) < 0.08
    assert np.median(np.abs(f["h"] / h - 1)) < 0.05


def _vessel_scene(shape=(480, 720), seed=0):
    rng = np.random.default_rng(seed)
    segs = []
    for i in range(10):
        x = 40 + 70 * i
        segs.append(((x, 0), (x + rng.uniform(-80, 80), shape[0]), rng.choice([6.0, 10.0, 16.0, 24.0]), 0.3))
    segs += [((0, 120), (720, 160), 5.0, 0.2), ((0, 360), (720, 330), 12.0, 0.25)]
    return _od(shape, segs)


def test_frame_average_sigma_extra_and_sign():
    od = _vessel_scene()
    frame = _image(od, blur=1.5, noise=0.25, seed=2)
    frame[frame >= 4095] = 4094
    avg = _image(od, blur=math.hypot(1.5, 1.2))
    r = T.frame_average_tells(frame, avg)
    assert r["n_sites"] >= 100
    assert abs(r["sigma_extra"] - 1.2) < 0.3 and r["edge_ratio"] < 0.97
    # the other way round: the "frame" blurrier than the "average"
    frame2 = _image(od, blur=math.hypot(1.5, 1.5), noise=0.25, seed=3)
    r2 = T.frame_average_tells(frame2, _image(od, blur=1.5))
    assert r2["sigma_extra"] < -1.0 and r2["edge_ratio"] > 1.03


def test_frame_average_grain_sees_red_cell_modulation():
    od = _vessel_scene(seed=1)
    avg = _image(od, blur=1.5)
    rng = np.random.default_rng(5)
    mod = ndi.gaussian_filter(rng.standard_normal(od.shape), 3.0)
    mod *= 0.25 / mod.std()
    plain = _image(od, blur=1.5, noise=0.25, seed=6)
    grainy = _image(ndi.gaussian_filter(od, 1.5) * (1 + mod), blur=0.01, noise=0.25, seed=6)
    r0 = T.frame_average_tells(plain, avg)
    r1 = T.frame_average_tells(grainy, avg)
    assert r0["n_runs"] >= 5
    assert r0["grain_coarse"] < 0.004 and r0["grain_sham"] < 0.004           # noise is subtracted
    assert r1["grain_coarse"] > 3 * max(r0["grain_coarse"], 0.002)
    assert r1["grain_rel"] > 2 * r0["grain_rel"]


def test_fields_mapping_is_the_stabilization_displacement(tmp_path):
    fields = pytest.importorskip("analysis.stabilize.fields")
    A = np.zeros((3, 2, 3))
    A[:, 0, 2], A[:, 1, 2] = (1.5, -2.0, 0.0), (0.5, 3.0, 0.0)
    L = np.zeros((3, 2, 2, 2), np.float32)
    p = tmp_path / "fields.npz"
    fields.save_fields(p, [4, 7, 9], [True] * 3, A, L, fields.Grid(10, 10, 100, 2, 2), (200, 300), "affine")
    f = T.fields_mapping(p, 7)
    xy = np.array([[10.0, 20.0], [250.0, 150.0]])
    assert np.allclose(f(xy), xy + [-2.0, 3.0])
    with pytest.raises(KeyError):
        T.fields_mapping(p, 5)


# ------------------------------------------------------------------ bookkeeping and the reference
def test_v0_keys_alone_and_all_keys_with_v1():
    I = _image(_vessel_scene(), noise=0.2)
    res = S.analyse(I, keep_maps=True)
    v0 = T.still_tells(I, res=res, v1=False)
    assert set(v0) == set(T.V0_KEYS) | {"lam"}
    v1 = T.still_tells(I, res=res)
    assert set(T.STILL_KEYS) <= set(v1)
    assert all(v0[k] == v1[k] or (np.isnan(v0[k]) and np.isnan(v1[k])) for k in T.V0_KEYS)


def test_separates():
    ref = np.array([1.0, 1.1, 0.9, 1.05])
    assert T.separates(ref, np.array([2.0, 2.1, 1.9])) == "yes"
    assert T.separates(ref, np.array([1.0, 1.02])) == "no"
    assert T.separates(ref, np.array([1.0, 1.8])) == "partly"


@pytest.mark.skipif(not T.DATA_PATH.is_file(), reason="real_tells.json not built")
def test_real_reference_has_every_key():
    real = T.load_real()
    assert set(T.DENSE_SITE) <= set(real["stills"])
    for rec in real["stills"].values():
        assert set(T.STILL_KEYS) <= set(rec)
    pairs = real.get("frame_pairs", {})
    assert len(pairs) >= 5
    for rec in pairs.values():
        assert set(T.PAIR_FRAME_KEYS) <= set(rec) and rec["n_sites"] > 200
    # the real frames are sharper than their average
    assert np.median([r["sigma_extra"] for r in pairs.values()]) > 0.3
    rows = T.compare([real["stills"][n] for n in T.DENSE_SITE], real=real, syn_pairs=list(pairs.values()))
    assert {r["key"] for r in rows} >= set(T.STILL_KEYS) | set(T.PAIR_FRAME_KEYS)
    assert all(r["separates"] != "yes" for r in rows)       # the dense site does not separate from itself


# ------------------------------------------------------------------ v2 statistics (2026-10-01)
def _poly_od(shape, pts, w, od):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(np.array(pts)).astype(np.int32)], False, 1, thickness=int(w), lineType=cv2.LINE_8)
    return od * m


def _thin_background(shape, od=0.15):
    """Thin oblique vessels, so that lambda (the median FWHM) stays small."""
    bg = np.zeros(shape)
    for x in range(40, shape[1], 70):
        bg = np.maximum(bg, _poly_od(shape, [(x, 0), (x + 60, shape[0])], 4, od))
    return bg


def test_knot_tells_additive_crossings_are_dark_knots():
    """Vessels whose densities add where they cross make dark knots; the
    union of the same lumens (the darker one wins) does not.  The skeleton
    is given (knot_tells only reads F along it), so the test does not depend
    on the suite's detection of crossings."""
    shape = (480, 720)
    lines = [[(0, y), (720, y + 40)] for y in range(60, 440, 90)]
    lines += [[(x, 0), (x - 60, 480)] for x in range(100, 700, 110)]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    dens = []
    for (x0, y0), (x1, y1) in lines:
        n = np.array([y0 - y1, x1 - x0], float)
        n /= np.hypot(*n)
        d = (xx - x0) * n[0] + (yy - y0) * n[1]
        dens.append(0.15 * np.exp(-0.5 * (d / 2.0) ** 2))
    Sk = _skel(shape, lines)
    out = {}
    for name, od in (("union", np.max(dens, 0)), ("add", np.sum(dens, 0))):
        res = dict(lam=LAM, maps=dict(S=Sk, F=-od, valid=np.ones(shape, bool)))
        out[name] = T.knot_tells(res)
    assert out["add"]["n_knots"] >= 15 and out["union"]["n_knots"] == out["add"]["n_knots"]
    assert out["union"]["dark_knot_rate"] == 0.0 and out["add"]["dark_knot_rate"] > 1.0
    assert out["add"]["knot_p10"] < -0.1 and abs(out["union"]["knot_p10"]) < 0.01
    assert out["add"]["dark_knot_frac"] > 0.9


def test_hier_tells_dominance_of_the_darkest_vessel():
    shape = (480, 720)

    def lines(dark_first):
        segs = []
        for i, y in enumerate(range(30, 480, 60)):
            segs.append(((0, y), (720, y + 20), 6.0, 0.6 if (dark_first and i == 3) else 0.25))
        return _od(shape, segs)
    even = T.hier_tells(S.analyse(_image(lines(False), noise=0.2), keep_maps=True))
    one = T.hier_tells(S.analyse(_image(lines(True), noise=0.2), keep_maps=True))
    assert even["n_long"] >= 6 * 1200 * 1920 / (480 * 720) * 0.9
    assert abs(even["dom"] - 1.0) < 0.15
    assert one["dom"] > 1.6 and one["darkest_long"] < even["darkest_long"] - 0.1


def test_wide_kink_sees_an_elbow_not_a_smooth_bend():
    shape = (600, 960)
    bg = _thin_background(shape)
    a = math.radians(55)
    elbow = [(0, 300), (480, 300), (480 + 600 * math.cos(a), 300 + 600 * math.sin(a))]
    th = np.linspace(-0.6, 0.6, 200)
    bend = [(480 + 700 * math.sin(t), 300 + 700 * (1 - math.cos(t))) for t in th]   # radius 700 px, 20 widths
    out = {}
    for name, pts in (("elbow", elbow), ("bend", bend)):
        I = _image(np.maximum(bg, _poly_od(shape, pts, 36, 0.45)), blur=3.0, noise=0.3)
        res = S.analyse(I, keep_maps=True)
        out[name] = T.wide_shape_tells(res)
    # an elbow of 55 deg marks about 4 (1 - 20 / 55) widths of the path
    assert 0.08 < out["elbow"]["wide_kink"] < 0.3
    assert out["bend"]["wide_kink"] < 0.02


def test_wide_rib_sees_a_periodic_ripple():
    shape = (600, 960)
    od = np.maximum(_thin_background(shape), _poly_od(shape, [(0, 300), (960, 330)], 36, 0.45))
    xx = np.arange(shape[1])[None, :]
    out = {}
    for amp in (0.0, 0.01):
        I = _image(od * (1 + amp / 0.45 * np.sin(2 * np.pi * xx / 7.0)), blur=1.0, noise=0.1)
        out[amp] = T.wide_shape_tells(S.analyse(I, keep_maps=True))
    assert out[0.0]["n_rib_seg"] >= 10
    assert out[0.01]["wide_rib"] > 20 * out[0.0]["wide_rib"]
    assert out[0.01]["wide_rib_peak"] > 20 and out[0.0]["wide_rib_peak"] < 5


def test_vein_tells_recovers_the_plateau():
    """The widest vessel is found and its plateau, width and edges measured."""
    shape = (600, 960)
    od = np.maximum(_thin_background(shape), _poly_od(shape, [(0, 250), (960, 350)], 44, 0.45))
    I = _image(od, blur=4.0, noise=0.2)
    v = T.vein_tells(I, np.isfinite(I), S.analyse(I, keep_maps=True))
    assert abs(v["vein_depth"] + 0.45) < 0.03
    assert abs(v["vein_fwhm"] - 44) < 4
    assert 4.0 < v["vein_edge"] < 14.0          # 10-90 % of a 4 px Gaussian edge: 10.3 px
    assert abs(v["vein_yx"][0] - 300) < 30


def test_v2_still_tells_has_every_key():
    I = _image(_vessel_scene(), noise=0.2)
    out = T.still_tells(I)
    assert set(T.STILL_KEYS) <= set(out)
    assert set(T.V2_STILL_KEYS) <= set(T.v2_still_tells(I))
    assert not set(T.V2_STILL_KEYS) & set(T.still_tells(I, v2=False))


def test_crisp_tells_frame_sharper_than_its_average():
    """A frame with less blur than its average: more contrast and narrower
    profiles on the same thin vessels, sharper edges on the medium ones.
    The same blur plus noise: ratios near 1 (the fits are not biased by
    noise)."""
    shape = (480, 720)
    rng = np.random.default_rng(4)
    segs = [((x, 0), (x + rng.uniform(-60, 60), 480), rng.choice([4.0, 6.0, 9.0, 14.0]), 0.25)
            for x in range(30, 720, 40)]
    segs += [((0, 140), (720, 170), 5.0, 0.2), ((0, 330), (720, 300), 8.0, 0.25)]
    od = _od(shape, segs)
    avg = _image(od, blur=2.2)
    sharp = _image(od, blur=1.2, noise=0.25, seed=2)
    same = _image(od, blur=2.2, noise=0.25, seed=3)
    rs = T.crisp_tells(sharp, avg)
    r0 = T.crisp_tells(same, avg)
    assert rs["sharper"] == 1.0 and rs["n_crisp_sites"] >= 200
    assert rs["crisp_c_thin"] > 1.08 and rs["crisp_fwhm_thin"] < 0.95 and rs["crisp_edge_med"] < 0.85
    assert abs(r0["crisp_c_thin"] - 1) < 0.05 and abs(r0["crisp_fwhm_thin"] - 1) < 0.05
    assert abs(r0["crisp_edge_med"] - 1) < 0.08 and r0["crisp_seen_thin"] > 0.9
    assert r0["frame_cnr_thin"] > 1.0


def test_faint_tells_dashed_lines():
    """Faint lines continuous in the frame (as in the average, plus noise)
    against the same lines broken into dashes of three times the density: the
    dashed frame has more gaps and more over-dark pixels."""
    shape = (480, 720)
    rng = np.random.default_rng(1)
    segs = [((x, 0), (x + rng.uniform(-80, 80), 480), 3.0, 0.035) for x in range(30, 720, 45)]
    segs += [((0, 240), (720, 260), 10.0, 0.3)]
    od = _od(shape, segs)
    avg = _image(od, blur=1.5)
    yy = np.arange(shape[0])[:, None] + 0 * np.arange(shape[1])[None, :]
    on = ((yy // 12) % 2 == 0)
    od_d = np.where(od < 0.1, od * 3 * on, od)
    cont = T.faint_tells(_image(od, blur=1.5, noise=0.15, seed=2), S.analyse(avg, keep_maps=True))
    dash = T.faint_tells(_image(od_d, blur=1.5, noise=0.15, seed=2), S.analyse(avg, keep_maps=True))
    assert cont["n_faint"] > 1000
    assert dash["faint_gap_rate"] > cont["faint_gap_rate"] + 5
    assert dash["faint_dash"] > cont["faint_dash"] + 0.1
    assert dash["faint_rcv"] > cont["faint_rcv"]


def test_bead_statistics_see_red_cell_aggregates():
    """Medium vessels whose density is modulated along them in the frame
    (aggregates) against plain ones; the correlation length follows the
    modulation's."""
    shape = (480, 720)
    segs = [((0, y), (720, y + 10), 8.0, 0.3) for y in range(40, 480, 50)]          # medium: lambda..2 lambda
    segs += [((x, 0), (x + 40, 480), 3.0, 0.15) for x in range(10, 720, 35)]      # thin ones set lambda
    od = _od(shape, segs)
    avg = _image(od, blur=1.5)
    rng = np.random.default_rng(7)
    out = {}
    for name, corr in (("plain", None), ("short", 1.5), ("long", 4.0)):
        if corr is None:
            od_f = od
        else:
            g = ndi.gaussian_filter1d(rng.standard_normal((shape[0], shape[1])), corr, axis=1)
            g /= g.std()
            od_f = od * (1 + 0.25 * g)
        out[name] = T.crisp_tells(_image(od_f, blur=1.0, noise=0.25, seed=3), avg)
    assert out["plain"]["n_bead_runs"] >= 5
    assert out["short"]["bead_med_rel"] > 3 * max(out["plain"]["bead_med_rel"], 0.005)
    assert out["long"]["bead_len"] > out["short"]["bead_len"] + 1.0
    assert out["short"]["bead_sham"] < 0.3 * out["short"]["bead_med"]          # nothing beside the vessels


def test_warp_to_average_follows_the_mapping():
    rng = np.random.default_rng(0)
    clean = 2000 + 300 * ndi.gaussian_filter(rng.standard_normal((200, 300)), 4.0)
    fr = clean.copy()
    fr[50, 60] = 4095                                     # saturated: no data

    def shift(xy):
        return np.asarray(xy, float) + [2.5, -1.25]
    w = T.warp_to_average(fr, shift, (200, 300))
    ref = ndi.shift(clean, (1.25, -2.5), order=3, mode="nearest")
    inner = np.isfinite(w)
    inner[:10] = inner[-10:] = False
    inner[:, :10] = inner[:, -10:] = False
    inner[40:62, 50:70] = False                           # the spline support of the filled saturated pixel
    assert np.abs(w - ref)[inner].max() < 1.0
    assert np.isnan(w[51, 57]) and np.isfinite(w[100, 150]) and np.isnan(w[:, -2:]).all()
    assert np.array_equal(np.isnan(T.warp_to_average(fr, None, fr.shape)), fr >= 4095)


def test_select_frames_spreads_over_the_used_frames(tmp_path):
    fields = pytest.importorskip("analysis.stabilize.fields")
    idx = list(range(0, 40, 2))
    used = [i % 4 != 2 for i in idx]
    A = np.zeros((len(idx), 2, 3))
    L = np.zeros((len(idx), 2, 2, 2), np.float32)
    p = tmp_path / "fields.npz"
    fields.save_fields(p, idx, used, A, L, fields.Grid(10, 10, 100, 2, 2), (200, 300), "affine")
    sel = T.select_frames(p, n=4)
    ok = [i for i, u in zip(idx, used) if u]
    assert len(sel) == 4 and all(i in ok for i in sel) and sel == sorted(sel)
    assert sel[0] <= ok[2] and sel[-1] >= ok[-3]


@pytest.mark.skipif(not T.DATA_PATH.is_file(), reason="real_tells.json not built")
def test_real_reference_v2():
    real = T.load_real()
    for rec in real["stills"].values():
        assert set(T.V2_STILL_KEYS) <= set(rec)
    pairs = real["frame_pairs"]
    bursts = {n.split(" ")[0] for n in pairs}
    assert {b for b, _ in T.PAIR_BURSTS} <= bursts
    for b in bursts:
        recs = [r for n, r in pairs.items() if n.startswith(b)]
        assert len(recs) >= 6
        for r in recs:
            assert set(T.CRISP_KEYS) <= set(r) and set(T.FRAME_STILL_KEYS) <= set(r["still"])
    summ = real["summary"]
    assert set(summ["sharper"]["by_burst"]) == bursts
    # the dense site's widest vessels are dark but not black, and every dense still has a dark long vessel
    for n in T.DENSE_SITE:
        assert -0.7 < real["stills"][n]["vein_depth"] < -0.25
        assert real["stills"][n]["darkest_long"] < -0.3
    rows = T.compare([real["stills"][n] for n in T.DENSE_SITE], real=real, syn_pairs=list(pairs.values()),
                     syn_frame_stills=[r["still"] for r in pairs.values()])
    keys = {r["key"] for r in rows}
    assert keys >= set(T.V2_STILL_KEYS) | set(T.CRISP_KEYS) | {"frame:" + k for k in T.FRAME_STILL_KEYS}
    assert all(r["separates"] != "yes" for r in rows)
