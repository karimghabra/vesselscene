"""Tests of image formation (vesselscene.imaging): the parameters come from
the real-still measurements, the average still and the single frame have the
noise, blur, coverage and artefacts the model promises, the helpers agree
with the stabilization pipeline, and the red-cell filling of the frames
averages to the filling of the still.

Small images keep this fast; the full-size comparison against the real
stills (imstats) is a script, not a test (see the module docstring)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from vesselscene import imaging as im
from vesselscene.graph import VesselGraph

SHAPE = (300, 480)


def _params(**kw):
    kw.setdefault("scene_seed", 1)
    kw.setdefault("camera_seed", 1)
    kw.setdefault("glare_rate", 0.0)
    return im.ImagingParams(**kw)


def _lines_od(shape, a=0.1, fwhm=10.0):
    """Two Gaussian-profile vertical lines (Np)."""
    H, W = shape
    x = np.arange(W)[None, :]
    sd = fwhm / 2.3548
    od = a * np.exp(-0.5 * ((x - W / 3) / sd) ** 2) + 2 * a * np.exp(-0.5 * ((x - 2 * W / 3) / (2 * sd)) ** 2)
    return np.broadcast_to(od, shape).copy()


# ------------------------------------------------------------------ measured defaults
def test_measured_json_is_small_and_complete():
    assert os.path.exists(im.DATA_PATH)
    assert os.path.getsize(im.DATA_PATH) < 100_000
    d = im.measured()
    assert d["format"] == "vesselscene-real-imaging"
    assert len(d["stills"]) >= 5
    S = d["summary"]
    for k in ("level_dn", "noise_lap", "nan_frac", "n_frames", "speed_px_s", "illum_span"):
        assert S[k]["min"] <= S[k]["median"] <= S[k]["max"]
    assert 5.0 < d["camera"]["e_per_dn_0db"] < 10.0          # IMX249 at 0 dB: ~8 e-/DN


def test_defaults_come_from_the_measurements():
    d = im.measured()["summary"]
    p = im.ImagingParams()
    assert p.level == pytest.approx(d["level_dn"]["median"])
    assert p.n_frames == int(d["n_frames"]["median"])
    assert p.e_per_dn0 == pytest.approx(im.measured()["camera"]["e_per_dn_0db"])
    assert p.tex_rms[3] == pytest.approx(d["band8"]["tex_rms_bright"]["median"])
    assert len(p.tex_rms) == len(im.BANDS)


def test_random_params_are_valid_and_form_an_image():
    rng = np.random.default_rng(3)
    for _ in range(4):
        p = im.ImagingParams.random(rng)
        assert 0.5 <= p.gamma <= 1.0 and p.n_frames >= 14 and p.level > 0
        assert all(t > 0 for t in p.tex_rms)
        out = im.form_image(np.zeros((160, 240)), p.replace(n_frames=min(p.n_frames, 60)), rng)
        v = out["valid"]
        assert v.mean() > 0.5 and np.isfinite(out["image"][v]).all()
    q = im.ImagingParams.random(rng, level=1234.0)
    assert q.level == 1234.0


# ------------------------------------------------------------------ helpers against the pipeline
def test_traj_sum_is_the_pipeline_shift():
    """sum over positions of the bilinear-histogram correlation equals the
    stabilization pipeline's cv2 warp (register.shift) summed over frames."""
    rng = np.random.default_rng(0)
    Q = rng.random((60, 80)).astype(np.float32)
    d = np.array([[2.3, -1.6], [-3.75, 0.4], [0.0, 0.0]])
    h, R = im._bilinear_hist(d)
    got = im.traj_sum(Q, h, R)
    want = sum(im._shift_frame(Q, di).astype(np.float64) for di in d)
    inner = (slice(6, -6), slice(6, -6))
    assert np.abs(got[inner] - want[inner]).max() < 2e-3        # cv2 uses 1/32 px weights


def test_two_tap_is_the_average_bilinear_kernel():
    z = np.random.default_rng(1).standard_normal((512, 512))
    y = im._two_tap(z)
    assert y.var() == pytest.approx(4 / 9, rel=0.02)
    c = np.mean(y[:, 1:] * y[:, :-1]) / y.var()
    assert c == pytest.approx(0.25, abs=0.02)


def test_noise_structure_reads_white_bilinear_noise():
    y = im._two_tap(np.random.default_rng(2).standard_normal((400, 600))) * 1e-3
    ns = im.noise_structure(y, np.ones(y.shape, bool))
    assert ns["white"] > 0.9
    assert ns["corr_x"][0] == pytest.approx(0.25, abs=0.03)
    assert ns["sd"] == pytest.approx(1e-3 * 2 / 3, rel=0.05)


def test_lap_noise_of_white_noise():
    z = np.random.default_rng(4).standard_normal((300, 300)) * 0.02
    assert im.lap_noise(z) == pytest.approx(0.02, rel=0.05)


# ------------------------------------------------------------------ illumination and texture
def test_illumination_fit_recovers_its_family():
    p = _params(illum_amp=0.5, illum_cx=-150, illum_cy=80, illum_sx=500, illum_sy=800, illum_theta=20, vignette=-0.1)
    lnE = im.illumination((600, 960), p)
    fit = im.fit_illumination(lnE, np.ones(lnE.shape), step=8)
    assert fit["rms"] < 2e-3
    assert fit["span"] == pytest.approx(np.ptp(lnE), rel=0.05)


def test_texture_spectrum_meets_the_band_targets():
    p = im.ImagingParams()
    for blur in (0.0, 0.85):
        got = im.texture_band_rms(p, blur)
        assert np.allclose(got, p.tex_rms, rtol=0.06), (blur, got / np.asarray(p.tex_rms))


def test_texture_field_band_rms_and_seed():
    p = _params(lump_density=0.0)
    tau = im.texture((512, 768), p, blur=0.0).astype(float)
    want = im.texture_band_rms(p, blur=0.0)
    for j, s in enumerate(im.BANDS[:4]):
        b = im._dog(tau, s)[3 * s + 8:-(3 * s + 8), 3 * s + 8:-(3 * s + 8)]
        assert b.std() == pytest.approx(want[j], rel=0.3), s
    # the same scene_seed gives the same texture (one anatomy); another differs
    assert np.array_equal(tau, im.texture((512, 768), p, blur=0.0).astype(float))
    assert not np.array_equal(tau, im.texture((512, 768), p.replace(scene_seed=2), blur=0.0).astype(float))


# ------------------------------------------------------------------ the average still
def test_average_still_basics():
    p = _params()
    od = _lines_od(SHAPE)
    out = im.form_image(od, p, np.random.default_rng(0))
    img, v = out["image"], out["valid"]
    assert img.dtype == np.float32 and img.shape == SHAPE
    assert np.isnan(img[~v]).all() and np.isfinite(img[v]).all()
    assert v.mean() > 0.97
    assert np.allclose(img[v], (out["clean"] + out["noise"])[v], rtol=1e-5)
    assert np.all(out["background"] >= out["clean"] - 1e-3)
    assert np.nanmedian(out["background"]) == pytest.approx(p.level, rel=0.1)
    # the vessel contrast in ln DN is the OD (blurred by < 1 px: a wide line keeps its centre)
    x0 = int(2 * SHAPE[1] / 3)
    c = np.log(out["background"][:, x0] / out["clean"][:, x0])
    assert np.median(c) == pytest.approx(0.2, rel=0.03)
    cov = out["blur_cov"]
    assert 0.2 < np.trace(cov) / 2 < 2.0                        # ~0.85 px SD with the defaults


def test_average_noise_level_and_shape():
    p = _params(fpn_pixel=0.0, fpn_row_dn=0.0, fpn_col_dn=0.0, dust_rate=0.0, flicker=0.0)
    out = im.form_image(np.zeros(SHAPE), p, np.random.default_rng(5))
    v = out["valid"] & np.isfinite(out["image"])
    n = np.log(out["image"] / out["clean"])
    n = np.where(v, n, 0.0)
    pred = float(np.median(im.frame_noise_sd(out["clean"], p) / out["clean"])) / math.sqrt(out["n_frames"])
    sd = n[v].std()
    assert sd == pytest.approx(pred * 2 / 3, rel=0.1)           # bilinear averaging: SD x 2/3
    c = np.mean(n[:, 1:] * n[:, :-1]) / np.mean(n * n)
    assert c == pytest.approx(0.25, abs=0.04)
    # the default camera (15-50-52: 20.4 dB, gamma 0.8, 138 frames) is inside the real range
    lap = im.lap_noise(np.where(v, np.log(out["image"]), 0.0), v)
    lo, hi = im.measured()["summary"]["noise_lap"]["min"], im.measured()["summary"]["noise_lap"]["max"]
    assert lo < lap < hi


def test_more_frames_less_noise():
    p = _params()
    rng = np.random.default_rng(6)
    sds = []
    for n in (25, 400):
        out = im.form_image(np.zeros(SHAPE), p, rng, n_frames=n)
        v = out["valid"]
        sds.append(np.log(out["image"][v] / out["clean"][v]).std())
    assert sds[0] / sds[1] == pytest.approx(math.sqrt(400 / 25), rel=0.25)


def test_glare_makes_holes_and_less_coverage():
    # a steady fixation (drift 3 px, no saccade): the camera-fixed glare mask covers the same tissue
    # in most frames, so it leaves a hole (a moving eye smears it below half coverage: no hole)
    p = _params(glare_rate=25.0, glare_px=6.0, n_frames=100, camera_seed=3, drift_px=3.0, saccade_rate=0.0)
    out = im.form_image(np.zeros((600, 900)), p, np.random.default_rng(7))
    assert len(out["glare"]) > 0
    bad = ~out["valid"]
    interior = bad[40:-40, 40:-40]
    assert interior.any()                                        # a glare hole away from the corners
    assert out["cover"][~bad].min() >= 0.5 * out["n_frames"] - 1e-3
    assert 0 < bad.mean() < 0.1


def test_nan_only_at_corners_without_glare():
    """The trajectory is centred on its median, as the pipeline's is: the
    edges keep half the frames, the corners lose them."""
    p = _params(n_frames=150, saccade_rate=0.0, drift_px=15.0)
    out = im.form_image(np.zeros((400, 600)), p, np.random.default_rng(8))
    bad = ~out["valid"]
    mid = bad[60:-60, 60:-60]
    assert not mid.any()
    assert bad[:5, :5].any() or bad[-5:, -5:].any() or bad[:5, -5:].any() or bad[-5:, :5].any() or not bad.any()


def test_same_scene_two_images_share_texture_not_noise():
    p = _params()
    a = im.form_image(np.zeros(SHAPE), p, np.random.default_rng(1))
    b = im.form_image(np.zeros(SHAPE), p, np.random.default_rng(2), tau=a["tau"], lnE=a["lnE"])
    assert np.array_equal(a["tau"], b["tau"])
    va = a["valid"] & b["valid"]
    assert not np.allclose(a["image"][va], b["image"][va])


# ------------------------------------------------------------------ the single frame
def test_single_frame():
    p = _params(glare_rate=30.0, glare_px=5.0, camera_seed=5)
    od = _lines_od(SHAPE)
    fr = im.form_image(od, p, np.random.default_rng(9), kind="frame")
    img = fr["image"]
    assert np.isfinite(img).all() and fr["valid"].all()
    assert np.array_equal(img, np.round(img)) and img.min() >= 0 and img.max() <= im.FULL_SCALE
    if fr["glare"]:
        assert (img == im.FULL_SCALE).any()                     # specular glare saturates a frame
    # per-frame noise: Poisson + read noise through the gamma, as measured on raw frames
    flat = im.form_image(np.zeros(SHAPE), _params(dust_rate=0.0, fpn_pixel=0.0), np.random.default_rng(10),
                         kind="frame")
    ln = im.lap_noise(np.log(np.maximum(flat["image"], 1.0)))
    want = float(np.median(im.frame_noise_sd(flat["clean"], p) / flat["clean"]))
    assert ln == pytest.approx(want, rel=0.1)
    cam = {b["burst"]: b for b in im.measured()["camera"]["bursts"]}["burst_2026-09-16_15-50-52"]
    assert want == pytest.approx(cam["frame_noise_ln"], rel=0.15)   # the default camera is that burst's


def test_frame_is_much_noisier_than_average():
    p = _params()
    rng = np.random.default_rng(11)
    av = im.form_image(np.zeros(SHAPE), p, rng)
    fr = im.form_image(np.zeros(SHAPE), p, rng, kind="frame", tau=av["tau"], lnE=av["lnE"])
    v = av["valid"]
    na = im.lap_noise(np.where(v, np.log(av["image"]), 0.0), v)
    nf = im.lap_noise(np.log(np.maximum(fr["image"], 1.0)))
    assert 5 < nf / na < 40                                     # ~ sqrt(N) / (2/3) with N = 138


def test_frame_noise_sd_gamma():
    p1, p5 = _params(gamma=1.0, gain_db=13.9), _params(gamma=0.5, gain_db=13.9)
    # at the same output DN, gamma 0.5 halves the log noise of the linear signal it came from
    dn = np.array([2500.0])
    x = im.FULL_SCALE * (dn / im.FULL_SCALE) ** 2
    g = p5.e_per_dn0 / 10 ** (13.9 / 20)
    want = 0.5 * np.sqrt(x / g + (p5.read_e / g) ** 2) / x
    assert im.frame_noise_sd(dn, p5) / dn == pytest.approx(want, rel=1e-6)
    assert im.frame_noise_sd(dn, p1)[0] > 0


# ------------------------------------------------------------------ red-cell filling
def _small_graph():
    g = VesselGraph()
    a = g.add_node((0, 0), kind="inlet", pressure=1.0)
    b = g.add_node((40, 0))
    c = g.add_node((90, 12), kind="outlet", pressure=0.0)
    d = g.add_node((90, -12), kind="outlet", pressure=0.0)
    e = g.add_node((60, 30), kind="outlet", pressure=0.0)
    g.add_vessel(a, b, radius=2.5, cls="arteriole", side="A")
    g.add_vessel(b, c, radius=0.5)
    g.add_vessel(b, d, radius=1.0, cls="arteriole", side="A")
    g.add_vessel(b, e, radius=0.45)
    g.solve_flow()
    return g


def test_filling_average_is_the_mean_of_its_frames():
    g = _small_graph()
    for seed in range(4):
        av = im.filling(g, None, "average", burst_seed=seed, n_frames=60)
        frames = [im.filling(g, None, "frame", frame=i, burst_seed=seed, n_frames=60) for i in range(60)]
        assert set(av) == {1, 2, 3}                                # the wide vessel is continuous: left out
        for vid in av:
            mean = np.mean([f[vid].m for f in frames], 0)
            assert np.allclose(mean, av[vid].m, atol=1e-9)


def test_filling_frame_columns_and_average_fuller():
    g = _small_graph()
    rng = np.random.default_rng(0)
    for seed in range(6):
        fr = im.filling(g, rng, "frame", burst_seed=seed)
        av = im.filling(g, rng, "average", burst_seed=seed)
        f = fr[1]                                                  # a capillary
        if f.state in ("unperfused", "empty"):
            continue
        s = np.linspace(0, f.s[-1], 2000)
        m = f(s)
        assert m.min() < 0.05 and m.max() > 0.95                   # red-cell columns and plasma gaps
        if av[1].state != "unperfused":
            a = av[1](s)
            assert a.std() < m.std()                                # the average is smoother ...
            assert 0.05 < a.mean() < 0.95                           # ... lower than full, but not empty


def test_filling_unperfused_stays_empty_and_relative_mode():
    g = _small_graph()
    av = im.filling(g, None, "average", burst_seed=1, p_unperfused=1.0)
    fr = im.filling(g, None, "frame", burst_seed=1, p_unperfused=1.0, frame=3)
    for vid in (1, 3):                                             # capillaries
        assert av[vid].state == fr[vid].state == "unperfused"
        assert av[vid].m.max() == 0 and fr[vid].m.max() == 0
    rel = im.filling(g, None, "average", burst_seed=2, relative=True, n_frames=600, p_empty=0.0, p_stopgo=0.0,
                     p_unperfused=0.0)
    for vid, f in rel.items():
        assert f.mean == pytest.approx(1.0, abs=0.15), (vid, f)
    with pytest.raises(TypeError):
        im.filling(g, None, "average", not_a_parameter=1)


def test_filling_columns_are_not_denser_than_packed_cells():
    """With relative=True a column holds h / phi of systemic blood; it may
    not exceed packed cells (h_col_max = 2, i.e. about 0.9 absolute).  The
    first version drew phi independently of h: columns up to 2.8x."""
    g = _small_graph()
    for e in g.vessels.values():
        e.h_ctrl = np.full_like(e.h_ctrl, 0.8)                     # a high tube haematocrit
    worst = 0.0
    for seed in range(20):
        for kind in ("frame", "average"):
            fl = im.filling(g, None, kind, frame=5, burst_seed=seed, relative=True, phi_cap=0.2, phi_sd=0.1)
            for vid, f in fl.items():
                worst = max(worst, 0.8 * float(f.m.max()))
    assert 1.0 < worst <= 2.0 + 1e-9


def test_filling_grain_in_continuous_vessels():
    """grain_amp > 0: a continuous vessel's blood fluctuates along it in a
    frame (red-cell aggregates moving with the flow); the average is still
    exactly the mean of its frames, and nearly flat."""
    g = _small_graph()
    kw = dict(burst_seed=3, n_frames=40, grain_amp=0.15)
    av = im.filling(g, None, "average", **kw)
    frames = [im.filling(g, None, "frame", frame=i, **kw) for i in range(40)]
    assert 0 in av and av[0].smooth                               # the wide vessel is in the dict now
    f0 = frames[7][0]
    assert 0.08 < f0.m.std() < 0.25 and abs(f0.m.mean() - 1.0) < 0.1
    assert np.allclose(np.mean([f[0].m for f in frames], 0), av[0].m, atol=1e-9)
    assert av[0].m.std() < 0.5 * f0.m.std()


def test_filling_grain_grows_with_radius():
    """grain_r_exp: the aggregates' relative SD is grain_amp x (r / r_cont) **
    grain_r_exp, capped at grain_amp_max (v1 integration: wider venules,
    lower shear, more aggregation); the pattern itself is the same."""
    g = _small_graph()                                             # vessel 0: radius 2.5 d_c, continuous
    base = dict(burst_seed=3, n_frames=40, frame=7, r_cont=1.2, grain_amp=0.1)
    f0 = im.filling(g, None, "frame", **base)[0].m - 1.0
    f1 = im.filling(g, None, "frame", grain_r_exp=1.0, **base)[0].m - 1.0
    f2 = im.filling(g, None, "frame", grain_r_exp=1.0, grain_amp_max=0.15, **base)[0].m - 1.0
    assert f1.std() / f0.std() == pytest.approx(2.5 / 1.2, rel=1e-3)
    assert f2.std() / f0.std() == pytest.approx(1.5, rel=1e-3)
    assert np.corrcoef(f0, f1)[0, 1] > 0.999
    # the average is still the mean of its frames
    kw = dict(burst_seed=3, n_frames=30, r_cont=1.2, grain_amp=0.1, grain_r_exp=1.0, grain_amp_max=0.35)
    av = im.filling(g, None, "average", **kw)
    frames = [im.filling(g, None, "frame", frame=i, **kw) for i in range(30)]
    assert np.allclose(np.mean([f[0].m for f in frames], 0), av[0].m, atol=1e-9)


# ------------------------------------------------------------------ measuring
def test_measure_still_on_a_synthetic_still(tmp_path):
    import tifffile
    p = _params(glare_rate=2.0, camera_seed=4)
    out = im.form_image(_lines_od((400, 640)), p, np.random.default_rng(12))
    fp = tmp_path / "mean_stabilized.tif"
    tifffile.imwrite(str(fp), out["image"])
    m = im.measure_still(str(fp))
    assert m["shape"] == [400, 640]
    assert m["nan_frac"] == pytest.approx(np.isnan(out["image"]).mean())
    assert m["bg_dn_p50"] == pytest.approx(p.level, rel=0.15)
    assert m["noise_lap"] == pytest.approx(im.lap_noise(np.where(out["valid"], np.log(out["image"]), 0),
                                                        out["valid"]), rel=0.3)
    for s in im.BANDS[:5]:
        b = m["bands"][str(s)]
        assert b["rms"] > 0 and 0 <= b["anis"] <= 1


from vesselscene import realism as _RM
_ROOT = _RM.find_data_root()
REAL = (str(_ROOT / "stabilization" / "translation" / "burst_2026-09-16_15-50-52" / "mean_stabilized.tif")
        if _ROOT else "")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real stills not on this machine")
def test_measure_still_reproduces_the_saved_numbers():
    m = im.measure_still(REAL, im.measured()["noise_band_ratio"])
    saved = im.measured()["stills"]["2026-09-16_15-50-52"]
    # the saved entry is the nonrigid still of the same burst: close, not identical
    assert m["noise_lap"] == pytest.approx(saved["noise_lap"], rel=0.2)
    assert m["bands"]["8"]["tex_rms_bright"] == pytest.approx(saved["bands"]["8"]["tex_rms_bright"], rel=0.3)


# ------------------------------------------------------------------ v1: spatially varying blur of the average, texture focus
def _line_image(shape=(160, 240)):
    """Thin Gaussian lines (sigma 0.9 px) at several orientations, in
    intensity units around 1."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = np.ones(shape)
    for ang, off in ((0.2, 60.0), (1.3, 120.0), (2.4, 30.0), (0.9, 180.0)):
        d = (xx - off) * math.sin(ang) - (yy - H / 2) * math.cos(ang)
        X -= 0.4 * np.exp(-0.5 * (d / 0.9) ** 2)
    return X


def test_blur_field_constant_is_the_gaussian():
    """A uniform var_map is the Gaussian of that variance (a level of the
    ladder, or a mix of two levels around it: within 1 % of the lines' depth)."""
    X = _line_image()
    for v in (0.5, 1.37, 4.0):
        got = im.blur_field(X, None, np.full(X.shape, v))
        ref = im._blur_gauss(X, v * np.eye(2), pad=40)
        assert np.abs(got - ref)[20:-20, 20:-20].max() < 0.01 * 0.4, v
    cov = np.array([[0.6, 0.1], [0.1, 0.3]])
    assert np.allclose(im.blur_field(X, cov, np.zeros(X.shape), pad=40), im._blur_gauss(X, cov, pad=40), atol=1e-9)


def test_blur_field_is_the_average_of_frames_with_local_registration_errors():
    """The average of frames each displaced by a random shift whose SD varies
    over the field (registration error sd(x): 0.3 px on the left, 1.5 px on
    the right), sampled exactly (cubic spline) and averaged, is the
    spatially varying Gaussian blur_field(X, var_map = sd^2): the 2-D
    Gauss-Hermite product of shifts integrates the Gaussian exactly, so what
    is left is the ladder's two-level mixing (< 1.5 % of the lines' depth)."""
    from scipy import ndimage as ndi
    X = _line_image()
    H, W = X.shape
    sd = 0.3 + 1.2 / (1.0 + np.exp(-(np.arange(W) - W / 2) / 25.0))[None, :] * np.ones((H, 1))
    x, w = np.polynomial.hermite_e.hermegauss(9)
    w = w / w.sum()
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    C = ndi.spline_filter(X, 3, mode="mirror")
    avg = np.zeros(X.shape)
    for xi, wi in zip(x, w):
        for yi, wj in zip(x, w):
            avg += wi * wj * ndi.map_coordinates(C, [yy + sd * yi, xx + sd * xi], order=3, mode="mirror",
                                                 prefilter=False)
    got = im.blur_field(X, None, sd ** 2)
    err = np.abs(got - avg)[12:-12, 12:-12].max() / 0.4
    assert err < 0.015, f"{100 * err:.2f} %"
    # and it differs from a uniform blur at the mean variance by much more
    uni = im._blur_gauss(X, float((sd ** 2).mean()) * np.eye(2), pad=40)
    assert np.abs(uni - avg)[12:-12, 12:-12].max() / 0.4 > 5 * err


def test_registration_field():
    p = _params(reg_error_px=0.6, reg_field_cv=0.5, reg_field_px=150.0)
    f = im.registration_field((600, 900), p, np.random.default_rng(3))
    assert f.shape == (600, 900) and f.min() > 0
    lf = np.log(f)
    assert np.exp(np.median(lf)) == pytest.approx(0.6, rel=0.2)
    assert lf.std() == pytest.approx(0.5, rel=0.25)
    # smooth: neighbours 30 px apart nearly equal, 600 px apart not
    a = lf - lf.mean()
    assert np.mean(a[:, 30:] * a[:, :-30]) / np.mean(a * a) > 0.8
    assert np.allclose(im.registration_field((50, 60), _params(reg_error_px=0.4), None), 0.4)


def test_average_with_a_registration_field():
    """form_image with reg_field_cv > 0: a wide line stays at its contrast,
    thin lines are blurred more where the field is large, and blur_cov reports
    the mean."""
    p = _params(reg_error_px=0.8, reg_field_cv=0.6, reg_field_px=120.0, fpn_pixel=0.0, dust_rate=0.0)
    od = _lines_od(SHAPE)
    out = im.form_image(od, p, np.random.default_rng(0))
    rs = out["reg_sd"]
    assert rs.shape == SHAPE and rs.max() > 1.5 * rs.min()
    x0 = int(2 * SHAPE[1] / 3)
    c = np.log(out["background"][:, x0] / out["clean"][:, x0])
    assert np.median(c) == pytest.approx(0.2, rel=0.05)
    # the reported blur: the frames' mean motion blur (>= 0) + bilinear 1/6 + the field's mean variance
    iso = np.trace(out["blur_cov"]) / 2 - 1 / 6 - float((rs ** 2).mean())
    assert -1e-6 < iso < 1.0


def test_defocus_texture():
    """The fine texture blurred where the textured tissue is out of focus:
    smoother there, the band power moved to the sharp side, the total kept
    under the normalising map; no blur = texture() exactly."""
    from scipy import ndimage as ndi
    p = _params()
    shape = (384, 640)
    parts = im.texture_parts(shape, p, blur=0.0)
    tau0 = im.texture(shape, p, parts=parts).astype(float)
    assert np.array_equal(tau0, im.texture(shape, p, blur=0.0))
    assert np.allclose(im.defocus_texture(parts, np.zeros(shape)), tau0, atol=1e-6)
    x = np.arange(shape[1])
    vm = np.broadcast_to(9.0 / (1.0 + np.exp(-(x - shape[1] / 2) / 10.0)), shape).copy()     # 0 left, 9 px^2 right
    tau = im.defocus_texture(parts, vm).astype(float)
    band = lambda t: ndi.gaussian_filter(t, 1.0) - ndi.gaussian_filter(t, 2.0)          # noqa: E731
    b0, b1 = band(tau0), band(tau)
    L, Rt = (slice(20, -20), slice(20, 260)), (slice(20, -20), slice(380, -20))
    # soft side smoother (the 1-2 px DoG band also sees the coarser fine groups and the unblurred coarse part:
    # about half is left at 3 px of extra blur)
    assert b1[Rt].std() < 0.6 * b1[L].std()
    assert b1[L].std() > 1.1 * b0[L].std()                    # sharp side gained power
    fine0 = sum(parts["fine"])
    fine1 = tau - parts["coarse"]
    assert np.std(fine1) == pytest.approx(np.std(fine0 + parts["lumps"]), rel=0.15)


def test_filling_gaps_in_vessels_wider_than_one_cell_are_partial():
    """gap_fill > 0: in a vessel between single file and continuous (radius
    1.0 d_c here, t = 0.5 of the way) a plasma gap keeps gap_fill sqrt(t) of
    the column's blood (beads, not dashes); a capillary's gaps stay empty;
    the average is still the mean of its frames, and relative mode keeps a
    flowing vessel's time mean at 1."""
    g = _small_graph()
    kw = dict(burst_seed=5, n_frames=60, gap_fill=0.6, p_empty=0.0, p_stopgo=0.0, p_unperfused=0.0)
    for i in (0, 7, 30):
        fr = im.filling(g, None, "frame", frame=i, **kw)
        f0 = im.filling(g, None, "frame", frame=i, **dict(kw, gap_fill=0.0))
        m, m0 = fr[2].m, f0[2].m                                   # the 1.0 d_c arteriole (r_single 0.6, r_cont 1.6)
        floor = 0.6 * math.sqrt((1.0 - 0.6) / (1.6 - 0.6))
        assert np.allclose(m, floor + (1.0 - floor) * m0)          # the same columns, the gaps partly filled
        assert m.min() >= floor - 1e-9
        assert fr[1].m.min() < 0.05 or fr[1].state != "flowing"   # the capillary: gaps still empty
    av = im.filling(g, None, "average", **kw)
    frames = [im.filling(g, None, "frame", frame=i, **kw) for i in range(60)]
    assert np.allclose(np.mean([f[2].m for f in frames], 0), av[2].m, atol=1e-9)
    f0s = [im.filling(g, None, "frame", frame=i, **dict(kw, gap_fill=0.0))[2].m.min() for i in range(60)]
    assert min(f0s) < 0.1 and min(f[2].m.min() for f in frames) >= floor - 1e-9      # deep gaps now filled
    rel = im.filling(g, None, "average", relative=True, **dict(kw, n_frames=600))
    assert rel[2].mean == pytest.approx(1.0, abs=0.15)


# ------------------------------------------------------------------ v2: red-cell speed, clumps (single frames)
def test_filling_red_cell_speed_follows_the_conjunctival_measurements():
    """speed_exp: the red cells' speed is v_cap (u / u_cap)^speed_exp, u the
    flow / lumen area; 1 (v1) scales it with u, 0 gives every flowing vessel
    v_cap (the conjunctiva: about 0.5 mm/s in every calibre); speed_sd
    spreads it log-normally per vessel, the same in a frame and its average."""
    g = _small_graph()
    radius = {vid: float(np.mean(e.r_ctrl)) for vid, e in g.vessels.items()}
    u = {vid: e.flow / (math.pi * radius[vid] ** 2) for vid, e in g.vessels.items()}
    kw = dict(burst_seed=2, n_frames=20, grain_amp=0.1, p_unperfused=0.0)
    f1 = im.filling(g, None, "frame", frame=3, **kw)
    assert all(f.speed is not None for f in f1.values())
    u_cap = float(np.median([u[1], u[3]]))                          # the capillaries (radius <= r_single)
    assert f1[0].speed / f1[1].speed == pytest.approx(u[0] / u[1], rel=0.03)
    assert f1[1].speed == pytest.approx(100.0 * u[1] / u_cap, rel=0.03)
    f0 = im.filling(g, None, "frame", frame=3, speed_exp=0.0, **kw)
    assert all(f.speed == pytest.approx(100.0) for f in f0.values())
    lv = [math.log(im.filling(g, None, kind, frame=3, speed_exp=0.0, speed_sd=0.3, **dict(kw, burst_seed=s))[0].speed
                   / 100.0) for s in range(150) for kind in ("frame",)]
    assert np.std(lv) == pytest.approx(0.3, rel=0.2) and abs(np.mean(lv)) < 0.1
    fr = im.filling(g, None, "frame", frame=3, speed_exp=0.0, speed_sd=0.3, **kw)
    av = im.filling(g, None, "average", speed_exp=0.0, speed_sd=0.3, **kw)
    assert all(fr[v].speed == av[v].speed for v in fr)
    # slower red cells smear a frame's aggregates less: more of their pattern survives
    def sd(v):
        return np.mean([im.filling(g, None, "frame", frame=3, speed_exp=0.0, **dict(kw, grain_len=1.0, v_cap=v,
                                                                                  burst_seed=s))[0].m.std()
                        for s in range(10)])
    assert sd(25.0) > 1.5 * sd(800.0)


def test_column_pattern_gamma_shape_is_more_regular():
    """col_shape / grain_shape: column and gap lengths gamma-distributed (1 =
    exponential, v1); a larger shape keeps the mean lengths and the filled
    share but spaces the columns more regularly (CV 1 / sqrt(shape))."""
    rng = np.random.default_rng(0)
    for shape, cv_want in ((1.0, 1.0), (4.0, 0.5)):
        g = im._column_pattern(400000, 0.1, 0.4, 1.5, 0.0, rng, shape=shape)
        assert g.mean() == pytest.approx(0.4, abs=0.02)
        edges = np.flatnonzero(np.diff(g) != 0)
        runs = np.diff(edges) * 0.1
        on = runs[(g[edges[:-1] + 1] > 0.5)]
        assert on.mean() == pytest.approx(1.5, rel=0.08)
        assert on.std() / on.mean() == pytest.approx(cv_want, rel=0.15)


def test_clump_grain_beads_and_its_average():
    """grain_kind='clump': a continuous vessel's blood comes in clumps of
    aggregated cells and plasma-rich stretches (a two-level pattern of
    relative SD grain_amp about 1, before the exposure smear); the average is
    exactly the mean of its frames and nearly flat; grain_r_pow sets how fast
    the amplitude falls beyond grain_r_agg."""
    g = _small_graph()                                              # vessel 0: radius 2.5 d_c
    kw = dict(burst_seed=4, n_frames=40, grain_amp=0.2, grain_kind="clump", grain_len=1.5, grain_edge=0.0,
              exposure_ms=0.0, speed_exp=0.0)
    frames = [im.filling(g, None, "frame", frame=i, **kw) for i in range(40)]
    av = im.filling(g, None, "average", **kw)
    m = frames[5][0].m
    assert abs(m.mean() - 1.0) < 0.08 and m.std() == pytest.approx(0.2, rel=0.2)
    lv = np.unique(np.round(m, 6))
    assert len(lv) == 2                                             # sharp clumps: two levels without edge / smear
    assert np.allclose(np.mean([f[0].m for f in frames], 0), av[0].m, atol=1e-9)
    assert av[0].m.std() < 0.4 * m.std()
    # beyond grain_r_agg the amplitude falls as (r_agg / r) ** grain_r_pow
    base = dict(kw, grain_r_agg=1.25)
    a5 = im.filling(g, None, "frame", frame=5, grain_r_pow=0.5, **base)[0].m - 1.0
    a1 = im.filling(g, None, "frame", frame=5, grain_r_pow=1.0, **base)[0].m - 1.0
    r0 = float(np.mean(eval_profile_r(g.vessels[0])))
    assert a1.std() / a5.std() == pytest.approx(math.sqrt(1.25 / r0), rel=0.02)


def eval_profile_r(e):
    from vesselscene.graph import eval_profile
    return eval_profile(e.r_ctrl, np.linspace(0, 1, 16))


def _tube(n=40, r=6.0, s=1.5, a=0.4, f=0.2, y=50.0, x0=20.0, l=4.0):
    """A straight horizontal vessel as render-plan tube pieces (px)."""
    cx = x0 + l * (np.arange(n) + 0.5)
    one = np.ones(n)
    return dict(cx=cx, cy=y * one, tx=one, ty=0 * one, l=l * one, r=r * one, s=s * one, a=a * one, f=f * one)


def test_rbc_mottle_grain_inside_the_lumen_only():
    """rbc_mottle: zero-mean granularity confined to the lumen (and its blur),
    linear in amp, smaller where the piece is blurred more or its cells are
    smeared by their speed, absent for pieces below r_min_px."""
    shape = (100, 200)
    tb = _tube()
    d = im.rbc_mottle(shape, tb, np.random.default_rng(0), 0.5, 4.0, 0.135)
    inside, outside = d[46:55, 30:170], d[70:, :]
    assert inside.std() > 0.005 and np.abs(outside).max() < 1e-6
    assert abs(inside.mean()) < 0.3 * inside.std()
    d2 = im.rbc_mottle(shape, tb, np.random.default_rng(0), 1.0, 4.0, 0.135)
    assert np.allclose(d2, 2 * d, atol=1e-6)
    blurred = im.rbc_mottle(shape, dict(tb, s=4.0 * np.ones(40)), np.random.default_rng(0), 0.5, 4.0, 0.135)
    assert blurred[46:55, 30:170].std() < 0.6 * inside.std()
    smeared = im.rbc_mottle(shape, tb, np.random.default_rng(0), 0.5, 4.0, 0.135, speed_dc_s=np.full(40, 300.0))
    assert smeared[46:55, 30:170].std() == pytest.approx(inside.std() / math.sqrt(1 + 300 * 0.0128), rel=0.02)
    assert np.abs(im.rbc_mottle(shape, tb, np.random.default_rng(0), 0.5, 4.0, 0.135, r_min_px=7.0)).max() == 0


def test_rbc_mottle_follows_the_cell_count_and_the_chord_law():
    """Poisson-like: the granularity grows as sqrt(chord absorbance) where
    the vessel is light (4x the absorbance, 2x the grain) and falls where it
    is black (the chord law saturates)."""
    shape = (100, 200)
    def sd(a, f):
        return im.rbc_mottle(shape, _tube(a=a, f=f), np.random.default_rng(1), 0.5, 4.0, 0.135)[46:55, 30:170].std()
    assert sd(0.08, 0.0) / sd(0.02, 0.0) == pytest.approx(2.0, rel=0.02)      # linear law (f = 0): sqrt(c)
    assert sd(6.0, 0.0) / sd(3.0, 0.0) == pytest.approx(math.sqrt(2.0), rel=0.02)
    assert sd(6.0, 0.3) < 0.5 * sd(3.0, 0.3)                                  # a black vessel: the chord law saturates
    # where a dark vessel crosses (here one without granularity of its own: r below r_min_px), the column is
    # darker and the first vessel's cells change the light less: its granularity is damped there, not elsewhere
    a_ = _tube(a=0.4, f=0.3)
    b_ = dict(cx=np.full(20, 100.0), cy=10.0 + 4.0 * (np.arange(20) + 0.5), tx=np.zeros(20), ty=np.ones(20),
              l=np.full(20, 4.0), r=np.full(20, 5.0), s=np.full(20, 1.5), a=np.full(20, 3.0), f=np.full(20, 0.3))
    both = {kk: np.r_[a_[kk], b_[kk]] for kk in a_}
    d1 = im.rbc_mottle(shape, a_, np.random.default_rng(3), 0.5, 4.0, 0.135, r_min_px=5.5)
    d2 = im.rbc_mottle(shape, both, np.random.default_rng(3), 0.5, 4.0, 0.135, r_min_px=5.5)
    xr, away = slice(97, 104), slice(30, 70)
    assert d2[47:54, xr].std() < 0.3 * d1[47:54, xr].std()
    assert np.allclose(d2[47:54, away], d1[47:54, away], atol=1e-7)


def test_scene_frame_mottle_only_in_the_frame_and_the_continuous_lumens():
    """scene.make_scene with filling 'mottle' > 0: the single frame's OD gains
    the red cells' granularity inside the continuous vessel (and its blur),
    not along the capillary (its cells are the filling's columns), not
    outside the vessels; the average is untouched; drawn from its own stream
    (the frame's filling and noise draws are unchanged)."""
    import torch
    from vesselscene import scene as SC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k, shape = 4.0, (120, 200)
    g = VesselGraph(meta=dict(k=k))
    for y, r in ((10.0, 2.5), (20.0, 0.5)):
        a = g.add_node((-5.0, y), depth=12.0, kind="inlet", pressure=1.0)
        b = g.add_node((55.0, y), depth=12.0, kind="outlet", pressure=0.0)
        vid = g.add_vessel(a, b, radius=r, hct=1.0, cls="venule" if r > 1 else "capillary")
        g.vessels[vid].flow = r * r
    base = dict(SC.CALIBRATED_FILL, p_unperfused=0.0, p_empty=0.0, p_stopgo=0.0)
    out = {}
    for m in (0.0, 0.5):
        out[m] = SC.make_scene(2, "healthy", shape=shape, device=dev, graph=g, labels=False,
                               fill_params=dict(base, mottle=m))
    d = out[0.5].od_clean["frame"].astype(float) - out[0.0].od_clean["frame"].astype(float)
    assert np.abs(d[34:46, 20:180]).std() > 0.004                       # inside the venule (y 40 px, r 10 px)
    assert np.abs(d[100:, :]).max() < 1e-4                              # far from both vessels
    assert np.abs(d[78:83, 20:180]).std() < 0.2 * np.abs(d[34:46, 20:180]).std()   # the capillary (y 80 px)
    assert np.allclose(out[0.5].od_clean["average"], out[0.0].od_clean["average"], atol=1e-5)   # GPU sums: ~1e-7
    assert np.isfinite(out[0.5].images["frame"]).all()
    assert "mottle_frame" in out[0.5].timings


def _xtube(horizontal, r, n=40, l=4.0, s=1.5, a=0.4, f=0.35, c=100.0):
    t = l * (np.arange(n) + 0.5) + 20
    one = np.ones(n)
    if horizontal:
        return dict(cx=t, cy=c * one, tx=one, ty=0 * one, l=l * one, r=r * one, s=s * one, a=a * one, f=f * one)
    return dict(cx=c * one, cy=t, tx=0 * one, ty=one, l=l * one, r=r * one, s=s * one, a=a * one, f=f * one)


def test_rbc_mottle_crossing_vessels_have_independent_cells():
    """v2 correctness review, defect 4: one shared field made the red-cell
    grain of two vessels crossing at different depths coherent (correlation
    0.97 at the crossing).  With vessel ids every vessel reads the field at
    its own offset: the two grains are independent where they cross, and a
    single vessel's grain keeps its amplitude."""
    shape, k, mu, rmin = (200, 200), 4.0, 0.185, 4.8
    cat = lambda p, q: {kk: np.r_[p[kk], q[kk]] for kk in p}                      # noqa: E731
    A_, B_ = _xtube(True, 6.0), _xtube(False, 6.0)
    Bx, Ax = _xtube(False, 4.79), _xtube(True, 4.79)          # in the column, no grain of their own
    ids = np.r_[np.zeros(40, int), np.ones(40, int)]
    win = (slice(96, 105), slice(96, 105))
    cs, cs0 = [], []
    for seed in range(8):
        for vessel, out in ((ids, cs), (None, cs0)):
            dA = im.rbc_mottle(shape, cat(A_, Bx), np.random.default_rng(seed), 0.6, k, mu, r_min_px=rmin, vessel=vessel)
            dB = im.rbc_mottle(shape, cat(Ax, B_), np.random.default_rng(seed), 0.6, k, mu, r_min_px=rmin, vessel=vessel)
            out.append(np.corrcoef(dA[win].ravel(), dB[win].ravel())[0, 1])
    assert np.mean(cs0) > 0.9                                  # the shared field (vessel=None)
    assert abs(np.mean(cs)) < 0.3
    sd = [im.rbc_mottle(shape, A_, np.random.default_rng(s_), 0.6, k, mu, vessel=v)[95:106, 30:170].std()
          for s_ in range(6) for v in (None, np.zeros(40, int))]
    assert np.mean(sd[1::2]) == pytest.approx(np.mean(sd[0::2]), rel=0.15)
