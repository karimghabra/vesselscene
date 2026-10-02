"""Tests of the realism suite (vesselscene.stats) and the scorecard
(vesselscene.realism).

Toy skeletons check the new topology statistics against known geometry;
the old vesselmap synthetic scene is the negative control (it must fail);
a real still scored against the reference with its own site left out is the
positive control (it must pass most statistics).  Tests that need the real
stills skip when the LIMBUS data folder is not found.
"""
import math

import cv2
import numpy as np
import pytest

from vesselscene import realism as R
from vesselscene import stats as S

LAM = 12.0
ROOT = R.find_data_root()
needs_data = pytest.mark.skipif(ROOT is None or not R.DATA_PATH.is_file(),
                                reason="real stills (stabilization/) or real_stats.json not available")
LEFT_OUT = ("09-16_15-31-37", "nonrigid")        # a 500x1920 still: fast; its site (B) has a second still


def _lines(shape, segs):
    img = np.zeros(shape, np.uint8)
    for (x0, y0), (x1, y1) in segs:
        cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, 1)
    from skimage.morphology import skeletonize
    return skeletonize(img.astype(bool))


def _ray(c, ang_deg, r):
    a = math.radians(ang_deg)
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a))


# ------------------------------------------------------------------ topology on toys
@pytest.mark.parametrize("angle", [35.0, 60.0, 90.0])
def test_x_crossing_angle(angle):
    c = (150, 150)
    Sk = _lines((300, 300), [(_ray(c, 180, 120), _ray(c, 0, 120)),
                             (_ray(c, 180 + angle, 120), _ray(c, angle, 120))])
    net = S.trace_skeleton(Sk, merge_len=0.5 * LAM)
    jg = S.junction_geometry(net, LAM)
    assert len(jg["cross_angle"]) == 1
    assert abs(jg["cross_angle"][0] - angle) < 6
    paths = S.link_paths(net, jg["link"])
    assert len(paths) == 2                           # each line continues through the crossing


def test_t_junction_angles():
    Sk = _lines((300, 300), [((30, 150), (270, 150)), ((150, 150), (150, 280))])
    net = S.trace_skeleton(Sk, merge_len=0.5 * LAM)
    jg = S.junction_geometry(net, LAM)
    assert len(jg["min_gap"]) == 1 and len(jg["cross_angle"]) == 0
    assert abs(jg["min_gap"][0] - 90) < 6            # side branch at right angles
    assert jg["straight_dev"][0] < 6                 # the through pair is collinear
    assert len(S.link_paths(net, jg["link"])) == 2   # through vessel + side branch


def test_gap_crossing_is_bridged():
    """A vessel broken where it passes a wide vessel (how most real crossings
    look to the ridge detector) is joined across the gap and counted as a
    crossing with the right angle."""
    Sk = _lines((300, 300), [((20, 150), (280, 150)), ((150, 20), (150, 138)), ((150, 162), (150, 280))])
    net = S.trace_skeleton(Sk, merge_len=0.5 * LAM)
    jg = S.junction_geometry(net, LAM)
    paths = S.link_paths(net, jg["link"])
    assert len(paths) == 3
    bj = S.bridge_paths(net["yx"], paths, LAM, Sk.shape)
    assert bj["n_bridges"] == 1 and len(bj["cross_angle"]) == 1
    assert abs(bj["cross_angle"][0] - 90) < 5


def test_tortuosity_by_scale():
    t = np.arange(0, 600.0, 0.25)
    straight = np.stack([100 + 0.3 * t, t], 1)
    wavy = np.stack([100 + 8 * np.sin(2 * np.pi * t / 60), t], 1)   # period 5 lambda
    tw = S.tortuosity_windows([S.resample_path(np.rint(straight))], LAM)
    assert np.median(tw[1]) < 1.01 and np.median(tw[4]) < 1.01
    tw = S.tortuosity_windows([S.resample_path(wavy)], LAM)
    assert np.median(tw[4]) > np.median(tw[1]) + 0.05   # tortuosity shows at the larger window


# ------------------------------------------------------------------ analyse
@pytest.fixture(scope="module")
def old_synthetic():
    syn = pytest.importorskip("vesselmap.synthetic")
    return [S.analyse(syn.make_scene(seed)[0], seed=3) for seed in (0, 1)]


def test_analyse_old_synthetic(old_synthetic):
    res = old_synthetic[0]
    sc = S.scalars(res)
    for k in list(S.LABELS):
        assert k in sc, k
    for k in ("noise", "fwhm_p50", "contrast_p50", "tort_4lam", "jangle_min_med", "along_cv_contrast",
              "faint_loop_density", "sharp3_c_small", "cross_density"):
        assert np.isfinite(sc[k]), k
    assert 0.015 < sc["noise"] < 0.03                  # vesselmap's white noise, ~20x the stills
    q = S.quantile_functions(res)
    assert q["contrast"] is not None and len(q["contrast"]) == len(S.Q_LEVELS)


@pytest.fixture(scope="module")
def left_out_real():
    import tifffile
    b, m = LEFT_OUT
    I = tifffile.imread(R.still_path(ROOT, b, m)).astype(np.float64)
    return S.analyse(I, seed=R.STILL_SEED)


@needs_data
def test_real_still_reproduces_reference(left_out_real):
    """analyse is deterministic: a fresh run equals the stored record."""
    stored = R.load_real()["records"][LEFT_OUT[0]]["scalars"]
    fresh = S.scalars(left_out_real)
    for k, v in stored.items():
        if np.isfinite(v):
            assert fresh[k] == pytest.approx(v, rel=1e-6, abs=1e-12), k


@needs_data
def test_load_real():
    real = R.load_real()
    assert real["names"] == [b for b, _ in R.REFERENCE_STILLS]
    assert real["summary"]["fwhm_p50"]["n"] == 9
    assert 10 < real["summary"]["fwhm_p50"]["mean"] < 14            # lambda = 11.9 +- 1.9 px
    assert 0.35 < real["summary"]["junction_per_endpoint"]["mean"] < 0.5
    fr = R.load_real("frames")
    assert len(fr["names"]) == 9
    assert fr["summary"]["noise"]["mean"] > 3 * real["summary"]["noise"]["mean"]   # single frames are noisier
    loo = R.load_real(exclude=["09-16_15-50-52"])
    assert loo["summary"]["fwhm_p50"]["n"] == 8
    with pytest.raises(KeyError):
        R.load_real(exclude=["no-such-still"])


# ------------------------------------------------------------------ scorecard
def _site_out(name):
    real = R.load_real()
    site = real["records"][name]["site"]
    return R.load_real(exclude=[m for m in real["names"] if real["records"][m]["site"] == site])


@needs_data
def test_left_out_real_still_passes_most(left_out_real):
    card = R.scorecard([left_out_real], _site_out(LEFT_OUT[0]))
    assert card.n_scored >= 90
    assert card.n_pass / card.n_scored >= 0.7           # measured: 83/102
    assert card.n_far() <= 3                            # measured: 0 with |z| > 4
    d = card.summary()["distributions+spectra"]
    assert d[0] >= d[1] - 4                             # measured: 8/11
    for k in ("sl:1", "vf:1", "contrast_p10", "contrast_p50", "psd_slope_hi", "sharp_p50", "tort_4lam_p90"):
        assert next(r for r in card.rows if r["key"] == k)["passed"], k


@needs_data
def test_old_synthetic_fails(old_synthetic, left_out_real):
    """Negative control: vesselmap's old synthetic scenes must be flagged."""
    card = R.scorecard(old_synthetic, R.load_real())
    rows = {r["key"]: r for r in card.rows}
    for k in ("noise", "sl:1", "vf:1", "contrast_p50", "psd_slope_hi", "sharp_p50", "junction_per_endpoint",
              "tort_4lam_p90", "along_cv_contrast"):
        assert rows[k]["passed"] is False, k
    assert card.n_far() >= 12                           # measured: 17-23 per seed
    real_card = R.scorecard([left_out_real], _site_out(LEFT_OUT[0]))
    assert card.n_pass / card.n_scored < real_card.n_pass / real_card.n_scored - 0.15
    assert card.summary()["distributions+spectra"][0] <= 3
    md = card.markdown()
    assert "**FAIL**" in md and "noise" in md


@needs_data
def test_figures(tmp_path, old_synthetic):
    import tifffile
    syn = pytest.importorskip("vesselmap.synthetic")
    real = R.load_real()
    card = R.scorecard(old_synthetic, real)
    p = R.figure_scorecard({"old synthetic": card}, tmp_path / "card.png")
    assert p.stat().st_size > 10_000
    p = R.figure_spectra(real, old_synthetic, tmp_path / "psd.png")
    assert p.stat().st_size > 10_000
    b, m = LEFT_OUT
    Ir = tifffile.imread(R.still_path(ROOT, b, m)).astype(np.float64)[:256, :384]
    Is = syn.make_scene(0, shape=(256, 384))[0]
    p = R.figure_side_by_side(Ir, Is, tmp_path / "sbs.png", crop_size=(96, 160))
    assert p.stat().st_size > 10_000
