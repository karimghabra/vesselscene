"""Tests of the closed-form renderer against the float64 supersampled oracle.

Every scene (scenes_render.py, 64 x 64 px at k = 3) must render within 1 % of
the oracle's peak optical density, with and without the halo."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import ndtr, owens_t as sp_owens_t

from vesselscene import render as R
from vesselscene.graph import VesselGraph
from vesselscene.tests import scenes_render as S

VM_W = R.BOX_H - np.r_[R.BOX_H[1:], 0.0]          # vesselmap's box heights (vesselmap/render.py _W)

TOL = 0.01          # of the oracle's peak OD
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rel_err(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("scene", S.SCENES, ids=[f.__name__ for f in S.SCENES])
def test_scene_against_oracle(scene):
    g, opt, note = scene()
    got = R.render_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8)
    assert ref.max() > 0.02, note
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} % of peak"
    got0 = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref0 = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got0, ref0) < TOL, f"{note} (no halo): {100 * rel_err(got0, ref0):.2f} % of peak"


def test_no_fade_at_the_border():
    """A frame is a crop of a larger frame: vessels leaving it are drawn
    exactly as they would be inside a bigger image."""
    g, opt, _ = S.leaving()
    small = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    big = R.vessel_od(g, S.K, (128, 128), opt, origin=(-32 / S.K, -32 / S.K), device=DEV)
    assert np.abs(small - big[32:96, 32:96]).max() < 1e-5 * big.max()
    small_h = R.render_od(g, S.K, S.SHAPE, opt, device=DEV)
    big_h = R.render_od(g, S.K, (128, 128), opt, origin=(-32 / S.K, -32 / S.K), device=DEV)
    assert np.abs(small_h - big_h[32:96, 32:96]).max() < 1e-5 * big_h.max()
    # along the near-horizontal vessel the peak is the same at the border and in the middle
    g2 = VesselGraph()
    S.add_free(g2, S.ray((-80, 30), 0, 240), 1.5)
    od = R.vessel_od(g2, S.K, S.SHAPE, opt, device=DEV)
    peaks = od.max(0)
    assert np.abs(peaks - peaks[32]).max() < 1e-4 * peaks[32]


def test_vids_renders_only_those_vessels():
    g, opt, _ = S.crossing()
    a = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV, vids=[0])
    b = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV, vids=[1])
    both = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    add = R.evaluate_plan(R.build_plan(g, S.K, S.SHAPE, opt, composite=False), S.SHAPE, DEV)
    assert np.abs(a + b - add).max() < 1e-5 * add.max()        # without compositing, crossings add
    assert (a + b - both).min() > -1e-5 * add.max()              # composited by depth: never darker than the sum
    assert (a + b - both).max() > 1e-3 * add.max()               # ... and lighter where they cross
    assert a.max() > 0.01 and b.max() > 0.01
    # vessel 1 alone is zero away from it; empty selection is zero
    far = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV, vids=[])
    assert far.max() == 0
    # at a fork each vessel alone matches the oracle of that vessel alone
    # (its round cap at the node included)
    g, opt, _ = S.fork()
    parts = []
    for v in g.vessels:
        one = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV, vids=[v])
        ref = R.oracle_od(g, S.K, S.SHAPE, opt, vids=[v], halo=False)
        assert rel_err(one, ref) < TOL
        parts.append(one)
    full = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    assert (np.sum(parts, 0) - full).max() > 0.05 * full.max()   # the union removes the double count


def test_hct_mod_multiplies():
    g, opt, _ = S.straight()
    lin = R.Optics(f_spectral=0.0, l_bypass=math.inf)            # f = 0: OD linear in haematocrit
    base = R.vessel_od(g, S.K, S.SHAPE, lin, device=DEV)
    half = R.vessel_od(g, S.K, S.SHAPE, lin, device=DEV, hct_mod={0: lambda s: np.full_like(s, 0.5)})
    assert np.abs(half - 0.5 * base).max() < 1e-5 * base.max()
    # plasma gaps (a step every 5 d_c) and a smooth modulation, against the oracle
    for fn in (lambda s: np.where(np.mod(s, 5.0) < 3.0, 1.0, 0.15),
               lambda s: 0.55 + 0.45 * np.cos(2 * np.pi * s / 4.0)):
        got = R.render_od(g, S.K, S.SHAPE, opt, device=DEV, hct_mod={0: fn})
        ref = R.oracle_od(g, S.K, S.SHAPE, opt, hct_mod={0: fn})
        assert rel_err(got, ref) < TOL
        assert np.abs(got - R.render_od(g, S.K, S.SHAPE, opt, device=DEV)).max() > 0.1 * ref.max()


def test_modes_and_devices_agree():
    g, opt, _ = S.fork()
    ex = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    un = R.evaluate_plan(R.build_plan(g, S.K, S.SHAPE, opt, junctions="union"), S.SHAPE, DEV)
    assert rel_err(un, ex) < TOL
    if DEV == "cuda":
        cpu = R.vessel_od(g, S.K, S.SHAPE, opt, device="cpu")
        assert np.abs(cpu - ex).max() < 1e-5


def test_chord_law_reduces_to_vesselmap_profile():
    a = np.array([0.05, 0.3, 1.2])
    W = R.box_weights(a, np.zeros(3))
    assert np.allclose(W, a[:, None] * VM_W, atol=1e-12)
    # with f > 0 the staircase is still nested and monotone, and saturates at -ln f
    L = R.chord_levels(np.array([1.5]), np.array([0.3]))
    assert np.all(np.diff(L[0]) < 0) and np.all(R.box_weights(np.array([1.5]), np.array([0.3])) > 0)
    assert np.all(L[0] < 1.5 * R.BOX_H)                            # f > 0 lowers the contrast
    L = R.chord_levels(np.array([50.0]), np.array([0.3]))
    assert abs(L[0, 0] + math.log(0.3)) < 1e-6
    o = R.Optics()
    assert o.unabsorbed(0.0) == pytest.approx(o.f_spectral)
    assert o.unabsorbed(1e6) == pytest.approx(1.0)
    assert o.blur(o.z_focus) == pytest.approx(math.hypot(o.s0_px, o.scatter_px_per_dc * o.z_focus))


def test_inclined_chord():
    """A vessel whose depth changes along it is an inclined tube: at f = 0
    and constant blur its OD scales by sqrt(1 + (dz/ds)^2)."""
    g = VesselGraph()
    path = S.ray((-60, 30), 0, 190) / S.K
    L = float(np.hypot(*(path[-1] - path[0])))
    a = g.add_node(path[0], depth=0.0)
    b = g.add_node(path[-1], depth=0.75 * L)
    g.add_vessel(a, b, path=path, radius=1.5 / S.K)
    kw = dict(defocus_px_per_dc=0.0, scatter_px_per_dc=0.0, f_spectral=0.0, l_bypass=math.inf)
    tilt = R.vessel_od(g, S.K, S.SHAPE, R.Optics(**kw), device=DEV)
    flat = R.vessel_od(g, S.K, S.SHAPE, R.Optics(incline=False, **kw), device=DEV)
    assert tilt.max() / flat.max() == pytest.approx(math.hypot(1.0, 0.75), rel=1e-3)


def test_polygon_mass_closed_form():
    """Owen's T and the polygon side sum: a square equals the separable erf
    product (eager float64 and the fused kernel)."""
    rng = np.random.default_rng(0)
    h, a = np.abs(rng.normal(0, 2, 5000)), rng.standard_cauchy(5000) * 3
    got = R.owens_t(torch.tensor(h), torch.tensor(a)).numpy()
    ok = h <= 8.5
    assert np.abs(got[ok] - sp_owens_t(h[ok], a[ok])).max() < 2e-7
    P = rng.uniform(-3, 3, (4000, 2))
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    s = 0.7
    ref = (ndtr((1 - P[:, 0]) / s) - ndtr((-1 - P[:, 0]) / s)) * (ndtr((1 - P[:, 1]) / s) - ndtr((-1 - P[:, 1]) / s))
    A = (sq[None] - P[:, None]) / s
    B = np.roll(A, -1, axis=1)
    t = lambda x: torch.tensor(x.reshape(-1))                     # noqa: E731
    m = R.side_mass_eager(t(A[..., 0]), t(A[..., 1]), t(B[..., 0]), t(B[..., 1])).numpy().reshape(len(P), 4).sum(1)
    assert np.abs(m - ref).max() < 1e-6
    if torch.cuda.is_available() and R._jit("side_mass") is not None:
        c = lambda x: torch.tensor(x.reshape(-1), dtype=torch.float32, device="cuda")   # noqa: E731
        mk = R._jit("side_mass")(c(A[..., 0]), c(A[..., 1]), c(B[..., 0]), c(B[..., 1]))
        assert np.abs(mk.cpu().numpy().reshape(len(P), 4).sum(1) - ref).max() < 1e-5


# ------------------------------------------------------------------ regressions (review of 2026-09-30)
def _max_union_ref(g, k, shape, opt, ss=16, pad=12):
    """An independent reference for vessels that all share one depth and one
    blur and form one blood volume: the sharp OD is the MAX over the vessels
    of their staircases (render study section 2), blurred by the one Gaussian."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    H, W = shape
    xs = -pad + np.arange((W + 2 * pad) * ss) / ss
    ys = -pad + np.arange((H + 2 * pad) * ss) / ss
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    best = np.zeros(len(P))
    for e in g.vessels.values():
        tr = R._Track(e, k, (0, 0), opt)
        n = int(math.ceil(tr.Lpx / 0.02)) + 1
        q = tr.at(np.linspace(0, tr.Lpx, n))
        d, j = cKDTree(q["C"]).query(P, distance_upper_bound=5.0)
        ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        Lv = R.chord_levels(q["a"], q["f"])
        Wv = Lv - np.concatenate([Lv[:, 1:], np.zeros((n, 1))], 1)
        val = (Wv[jj] * (d[:, None] < q["r"][jj, None] * R.BOX_C)).sum(1)
        best = np.maximum(best, np.where(ok, val, 0.0))
    img = gaussian_filter(best.reshape(len(ys), len(xs)), float(opt.s0_px) * ss, mode="constant", truncate=5.0)
    return img[pad * ss::ss, pad * ss::ss][:H, :W]


def test_short_steep_link_between_junctions():
    """A vessel shorter than its two junction radii whose staircase changes
    along it (incline factor 1 at its nodes, 2 in the middle): the overlap
    polygons must follow its own levels along it (the first version used its
    middle piece's: ~20 % too light; one segment per stub gives ~39 %)."""
    g, opt, note = S.steep_link()
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert R.build_plan(g, S.K, S.SHAPE, opt).stats["whole_stubs"] == 1
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} %"


def test_sibling_overlapping_beyond_the_angle_radius_is_unioned():
    """Where the lumens of vessels meeting at a node keep overlapping in 3-D
    beyond the angle formula's R, the union region extends over it (the first
    version added them there)."""
    g, opt, note = S.sibling_inside()
    nid = next(n for n, (i, o) in g.degrees().items() if i + o == 3)
    R0 = R.junction_radius(g, nid, S.K)
    run = R.junction_overlap_run(g, nid, S.K, optics=opt)
    assert run > R0 + 10                                # the case: overlap far beyond the formula
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} %"


@pytest.mark.parametrize("gap", [1.5, 4.0])
def test_oracle_and_closed_form_are_the_union_at_merged_forks(gap):
    """Two forks joined by a very short vessel: the oracle itself must be
    the max-union (the first version added overlaps across the merged
    cluster, 17.6 % off), and the closed form too."""
    g, opt, _ = S.short_link(gap)
    ref = _max_union_ref(g, S.K, S.SHAPE, opt)
    ora = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    assert rel_err(ora, ref) < TOL
    assert rel_err(got, ref) < TOL


def test_oracle_does_not_depend_on_the_crop():
    """The oracle's union is decided per fine pixel: a crop whose grid leaves
    out a junction node renders the overlap the same (the first version took
    no union for a node outside its grid)."""
    g, opt, _ = S.fork()
    big = R.oracle_od(g, S.K, S.SHAPE, opt, ss=6, halo=False)
    # a crop that starts 3 px right of the node: the node (32, 30) is outside it
    org = (35.0 / S.K, 10.0 / S.K)
    crop = R.oracle_od(g, S.K, (40, 29), opt, origin=org, ss=6, halo=False)
    assert np.abs(crop - big[10:50, 35:64]).max() < 0.003 * big.max()


def test_frame_filling_at_a_junction():
    """Red-cell columns and plasma gaps (soft 0.15 d_c edges, as
    imaging.filling makes them) on every vessel of a fork: the caps and
    overlap corrections follow the filling along the stubs (the first
    version used each vessel's end piece: 1-8 % off in frames)."""
    from scipy.special import ndtr
    g, opt, _ = S.fork()

    def gaps(ph):
        return lambda s: 0.1 + 0.9 * (ndtr(np.mod(s + ph, 3.0) / 0.15) - ndtr((np.mod(s + ph, 3.0) - 1.8) / 0.15))
    fns = {v: gaps(0.7 * v) for v in g.vessels}
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV, hct_mod=fns)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False, hct_mod=fns)
    assert rel_err(got, ref) < TOL


def test_dive_right_after_a_fork():
    g, opt, note = S.dive_at_fork()
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} %"


@pytest.mark.parametrize("k", [1.5, 8.0])
def test_fork_at_other_scales(k):
    """The same fork (in d_c) at the ends of the k range (the oracle tests
    otherwise run only at k = 3)."""
    g, opt, _ = S.fork()
    org = g.nodes[0].xy - np.array([32.0, 30.0]) / k
    got = R.vessel_od(g, k, S.SHAPE, opt, origin=org, device=DEV)
    ref = R.oracle_od(g, k, S.SHAPE, opt, origin=org, ss=8, halo=False)
    assert ref.max() > 0.02
    assert rel_err(got, ref) < TOL


def test_tilted_focal_surface():
    """A focal surface tilted across the frame: one vessel is sharp at one
    end and blurred at the other, and the closed form still matches the
    oracle (which takes the blur of every bit of lumen at its own place)."""
    g = VesselGraph()
    S.add_free(g, S.ray((-60, 32), 0, 190), 1.5, z=10.0)
    opt = R.Optics(s0_px=1.2, z_focus=10.0, defocus_px_per_dc=0.1, scatter_px_per_dc=0.0,
                   focus_gx=1.2, focus_x0=0.0, focus_y0=0.0)
    xy = np.array([[2.0 / S.K, 32.0 / S.K], [62.0 / S.K, 32.0 / S.K]])
    s_l, s_r = opt.blur(10.0, xy)
    assert s_l < 1.3 < 2.5 < s_r                                  # sharp on the left, blurred on the right
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got, ref) < TOL
    assert got[:, 4].max() > 1.3 * got[:, 60].max()               # the blurred end is fainter
    # the reach bound covers the blur anywhere along the vessel
    e = g.vessels[0]
    smp = e.sample(0.2)
    assert R._vessel_reach(e, S.K, opt) >= S.K * smp["r"].max() + R.NSIG * opt.blur(smp["z"], smp["xy"]).max() - 1e-9


# ------------------------------------------------------------------ v1 optics: random focal surface, frame offset, spread
def _modes_optics(**kw):
    """A focal surface with a tilt, a curvature and a smooth random part (two
    cosines), at the scale of the tests (64 px = 21 d_c): the depth in focus
    changes by several d_c across the frame."""
    base = dict(s0_px=1.2, z_focus=10.0, defocus_px_per_dc=0.25, scatter_px_per_dc=0.0, focus_gx=0.1,
                focus_curv=-2e-3, focus_x0=10.0, focus_y0=10.0,
                focus_modes=((3.0, 0.35, 0.1, 0.4), (-2.0, -0.1, 0.3, 1.1)))
    base.update(kw)
    return R.Optics(**base)


def test_focal_surface_random_part_offset_and_spread():
    """focal_depth adds the cosines and the frame's offset; blur adds the
    spread in quadrature; to_dict / Optics(**d) round-trips (as scene.json
    does); the reach bound covers the random part."""
    o = _modes_optics()
    xy = np.array([[3.0, 4.0], [15.0, 2.0]])
    dx, dy = xy[:, 0] - 10.0, xy[:, 1] - 10.0
    want = 10.0 + 0.1 * dx - 2e-3 * (dx * dx + dy * dy) + 3.0 * np.cos(0.35 * dx + 0.1 * dy + 0.4)         - 2.0 * np.cos(-0.1 * dx + 0.3 * dy + 1.1)
    assert np.allclose(o.focal_depth(xy), want)
    assert np.allclose(R.Optics(**o.to_dict()).focal_depth(xy), want)
    assert o.modes_bound() == pytest.approx(5.0)
    f = replace(o, focus_offset=4.0)
    assert np.allclose(f.focal_depth(xy), want + 4.0)
    sp = replace(o, focus_spread=6.0)
    assert np.allclose(sp.blur(12.0, xy) ** 2, o.blur(12.0, xy) ** 2 + (0.25 * 6.0) ** 2)
    g = VesselGraph()
    S.add_free(g, S.ray((-60, 32), 10, 190), 1.5, z=10.0, z_end=16.0)
    e = g.vessels[0]
    smp = e.sample(0.2)
    for opt in (o, f, sp):
        assert R._vessel_reach(e, S.K, opt) >= S.K * smp["r"].max() + R.NSIG * opt.blur(smp["z"], smp["xy"]).max() - 1e-9


def test_spread_is_the_second_moment_of_the_frames_blur():
    """The average of frames focused at offsets delta ~ N(0, spread) has the
    frames' mean squared blur (Gauss-Hermite nodes integrate it exactly)."""
    o = _modes_optics(focus_spread=7.0)
    xy = np.random.default_rng(1).uniform(0, 21, (50, 2))
    z = np.random.default_rng(2).uniform(5, 30, 50)
    x, w = np.polynomial.hermite_e.hermegauss(5)
    w = w / w.sum()
    frames = [replace(o, focus_spread=0.0, focus_offset=7.0 * xi) for xi in x]
    m2 = sum(wi * f.blur(z, xy) ** 2 for wi, f in zip(w, frames))
    assert np.allclose(o.blur(z, xy) ** 2, m2)


@pytest.mark.parametrize("kw", [dict(), dict(focus_offset=3.0), dict(focus_spread=5.0)],
                         ids=["modes", "offset", "spread"])
def test_random_focal_surface_against_oracle(kw):
    """A vessel under a focal surface with a random part (and a frame's
    offset, or an average's spread): sharp where the surface meets its depth,
    blurred elsewhere, and still within 1 % of the oracle."""
    g = VesselGraph()
    S.add_free(g, S.ray((-60, 20), 12, 190), 1.5, z=10.0, z_end=14.0)
    S.add_free(g, S.ray((10, -60), 80, 190), 2.0, z=18.0)
    opt = _modes_optics(**kw)
    s = R.build_plan(g, S.K, S.SHAPE, opt).tubes["s"]
    assert s.max() > 1.5 * s.min()
    got = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got, ref) < TOL, f"{kw}: {100 * rel_err(got, ref):.2f} % of peak"


def test_average_of_focus_jittered_frames():
    """The average's OD rendered with focus_spread (the second moment of the
    frames' blur) against the mean of the ODs of frames focused at
    Gauss-Hermite offsets (the exact mixture).  The mixture is peakier, so
    the single Gaussian is a little lower at a thin vessel's centre.  Max
    difference, % of peak, for s0 1.8 px and defocus x spread 0.6 / 1.1 /
    1.5 / 2.2 px: 0.5 / 1.9 / 3.9 / 8.5 (s0 1.3, 1.5 px: 5.1).  Checked here
    at 1.1 px (the calibrated scenes: README)."""
    g = VesselGraph()
    S.add_free(g, S.ray((-60, 32), 3, 190), 1.2, z=10.0)
    S.add_free(g, S.ray((32, -60), 88, 190), 3.5, z=16.0)
    base = R.Optics(s0_px=1.8, defocus_px_per_dc=0.2, scatter_px_per_dc=0.0, z_focus=12.0)
    avg = R.vessel_od(g, S.K, S.SHAPE, replace(base, focus_spread=5.5), device=DEV)
    x, w = np.polynomial.hermite_e.hermegauss(9)
    w = w / w.sum()
    mix = sum(wi * R.vessel_od(g, S.K, S.SHAPE, replace(base, focus_offset=5.5 * xi), device=DEV)
              for wi, xi in zip(w, x))
    err = np.abs(avg - mix).max() / mix.max()
    assert err < 0.025, f"{100 * err:.2f} % of peak"
    # the frame at the mean focus is sharper than the average: higher peak on the thin vessel
    one = R.vessel_od(g, S.K, S.SHAPE, base, device=DEV)
    assert one[25:40, 5:20].max() > 1.04 * avg[25:40, 5:20].max()


# ------------------------------------------------------------------ v2: crossings composited by depth
def _no_composite(opt):
    return replace(opt, composite=False)


@pytest.mark.parametrize("scene", S.COMPOSITE_SCENES, ids=[f.__name__ for f in S.COMPOSITE_SCENES])
def test_composited_crossings_against_oracle(scene):
    """Vessels crossing at different depths, composited by depth
    (composite_od: half the light unabsorbable, a 30 d_c bypass length): the
    closed form's per-pixel blood volumes against the oracle's per-vessel
    images, with and without the halo.  Where vessels cross (not at a fork)
    the composite is markedly lighter than the sum of the ODs and never
    darker."""
    g, opt, note = scene()
    got = R.render_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8)
    assert ref.max() > 0.05, note
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} % of peak"
    got0 = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    ref0 = R.oracle_od(g, S.K, S.SHAPE, opt, ss=8, halo=False)
    assert rel_err(got0, ref0) < TOL, f"{note} (no halo): {100 * rel_err(got0, ref0):.2f} % of peak"
    add0 = R.vessel_od(g, S.K, S.SHAPE, _no_composite(opt), device=DEV)
    assert (add0 - got0).min() > -1e-4 * ref0.max()
    if scene is not S.cross_dive_at_fork:              # a fork with a diving child: one blood volume, no crossing
        assert (add0 - got0).max() > 0.05 * ref0.max(), note


def test_composite_od_layers():
    """composite_od: one layer is itself; f_s = 0 with no bypass adds the
    ODs (Beer-Lambert); two layers give the v1 formula T' = T'_t T'_b +
    g_t / (1 - g_t) (1 - T'_t)(1 - T'_b); the order of the inputs does not
    matter; the composite is never darker than the sum and never darker
    than f_s allows."""
    rng = np.random.default_rng(3)
    z = torch.tensor(rng.uniform(0.0, 100.0, (3, 500)))
    o = R.Optics(f_spectral=0.3, l_bypass=50.0)
    D = torch.tensor(rng.uniform(0.0, 1.0, (3, 500)) * -np.log(o.unabsorbed(z.numpy())))      # within the chord law
    one = R.composite_od(D[:1], z[:1], 0.3, 50.0)
    assert torch.allclose(one, D[0], atol=1e-12)
    zero = torch.zeros_like(D)
    assert torch.allclose(R.composite_od(torch.stack([D[0], zero[0]]), z[:2], 0.3, 50.0), D[0], atol=1e-12)
    assert torch.allclose(R.composite_od(D, z, 0.0, math.inf), D.sum(0), atol=1e-12)
    perm = R.composite_od(D[[2, 0, 1]], z[[2, 0, 1]], 0.3, 50.0)
    full = R.composite_od(D, z, 0.3, 50.0)
    assert torch.allclose(perm, full, atol=1e-12)
    assert torch.all(full <= D.sum(0) + 1e-12) and torch.all(full <= -math.log(0.3) + 1e-12)
    # two layers: the v1 closed form (scene.crossing_bypass)
    fs, lb = 0.05, 80.0
    Dt, Db = torch.tensor([0.4, 0.12]), torch.tensor([0.2, 0.3])
    zt, zb = torch.tensor([12.0, 30.0]), torch.tensor([30.0, 45.0])
    q1, q2 = (torch.exp(-Dt) - fs) / (1 - fs), (torch.exp(-Db) - fs) / (1 - fs)
    gt = 1 - torch.exp(-zt / lb)
    want = -torch.log(fs + (1 - fs) * (q1 * q2 + gt / (1 - gt) * (1 - q1) * (1 - q2)))
    assert torch.allclose(R.composite_od(torch.stack([Dt, Db]), torch.stack([zt, zb]), fs, lb), want, atol=1e-12)


def test_composite_under_a_saturated_vein():
    """Under a near-black wide vessel a deeper one adds almost nothing; a
    shallower one crossing over it still shows (it intercepts the light the
    tissue sends back between the two depths)."""
    thin, fs, lb = torch.tensor([0.15]), 0.05, 80.0
    vein = torch.tensor([1.2])                                    # at 20 d_c: f = 0.26, at most 1.35 Np
    below = R.composite_od(torch.stack([vein, thin]), torch.tensor([[20.0], [60.0]]), fs, lb) - vein
    vein = torch.tensor([0.8])                                    # at 40 d_c: f = 0.42, at most 0.86 Np
    above = R.composite_od(torch.stack([vein, thin]), torch.tensor([[40.0], [5.0]]), fs, lb) - vein
    assert float(below) < 0.25 * float(thin) < 0.5 * float(thin) < float(above)


def test_composite_devices_agree():
    """The compositing passes give the same image on the CPU (float64, eager)
    as on the GPU (float32, fused kernels)."""
    g, opt, _ = S.cross_triple()
    gpu = R.vessel_od(g, S.K, S.SHAPE, opt, device=DEV)
    cpu = R.vessel_od(g, S.K, S.SHAPE, opt, device="cpu")
    assert np.abs(cpu - gpu).max() < 1e-5 * gpu.max() + 1e-6


def test_composite_off_adds():
    """Optics.composite = False (or build_plan(composite=False)): crossing
    ODs add, as in v0, against the oracle without compositing."""
    g, opt, note = S.cross_triple()
    o = _no_composite(opt)
    got = R.vessel_od(g, S.K, S.SHAPE, o, device=DEV)
    ref = R.oracle_od(g, S.K, S.SHAPE, o, ss=8, halo=False)
    assert rel_err(got, ref) < TOL
    parts = sum(R.vessel_od(g, S.K, S.SHAPE, o, device=DEV, vids=[v]) for v in g.vessels)
    assert np.abs(parts - got).max() < 1e-5 * got.max()


# ------------------------------------------------------------------ v2: wide vessels at high OD (ribs, node seams)
def _hp(a):
    from scipy.ndimage import gaussian_filter
    return a - gaussian_filter(a, 3.0)


@pytest.mark.parametrize("scene", S.WIDE_SCENES, ids=[f.__name__ for f in S.WIDE_SCENES])
def test_wide_vessels_at_high_od_against_oracle(scene):
    """Wide lumens (r 12-20 px, k = 3) at OD ~1.2 against the oracle: a
    smooth bend (kappa r up to 0.29), a thin tributary joining a wide vein,
    two wide veins joining.  Within 1 % of the peak, and the fine structure
    inside the lumen (OD minus its 3 px Gaussian) is the oracle's: no ribs
    at piece joints, no seam where a junction's caps and overlap polygons
    meet the tubes (the v1 review's T6: 0.01 Np ribs, a bright node seam)."""
    from scipy.ndimage import binary_erosion
    g, opt, note = scene()
    got = R.vessel_od(g, S.K, S.WIDE_SHAPE, opt, device=DEV)
    ref = R.oracle_od(g, S.K, S.WIDE_SHAPE, opt, ss=6, halo=False)
    assert ref.max() > 1.0, note
    assert rel_err(got, ref) < TOL, f"{note}: {100 * rel_err(got, ref):.2f} % of peak"
    m = binary_erosion(ref > 0.5 * ref.max(), iterations=3)
    assert np.std((_hp(got) - _hp(ref))[m]) < 0.001, note
    assert np.abs(_hp(got) - _hp(ref))[m].max() < 0.003, note


def test_wide_bend_needs_the_lumen_edge_piece_rule(monkeypatch):
    """Without the lumen-edge rule (pieces cut only by the centreline
    sagitta, up to L_MAX = 8 px) the wide bend draws transverse ribs: the
    test above would catch it."""
    g, opt, _ = S.wide_bend()
    ref = R.oracle_od(g, S.K, S.WIDE_SHAPE, opt, ss=6, halo=False)
    monkeypatch.setattr(R, "EDGE_TOL", 1e9)
    got = R.vessel_od(g, S.K, S.WIDE_SHAPE, opt, device=DEV)
    assert rel_err(got, ref) > 0.02
