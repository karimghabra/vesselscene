"""A whole synthetic scene: anatomy -> red-cell filling -> closed-form render
-> image formation, with its ground truth.

    make_scene(seed, preset)    one scene: the VesselGraph (anatomy.generate),
                                every parameter and seed, the images (an
                                average still like mean_stabilized.tif and a
                                single raw frame of the same anatomy), their
                                optical densities, and the truth rasters
    save_scene(scene, folder)   the files (see save_scene)
    to_vesselnetwork(graph, k)  the truth as vesselmap's VesselNetwork (one
                                edge per vessel), so vesselmap's tools and
                                metrics run on it

One anatomy, two kinds of image.  The average still and the single frame
share the anatomy, the scleral texture field, the illumination, the camera's
fixed pattern and glare; they differ in what the burst does.  Every
capillary's red-cell filling is simulated over the burst (imaging.filling:
columns of cells and plasma gaps moving at the vessel's red-cell speed,
stop-go flow, plasma-only episodes, a few capillaries never perfused): the
average still renders the burst mean of the filling (continuous, "fuller"),
the frame one instant of it (broken columns, some capillaries empty, red-cell
aggregates), with a frame's motion blur and its full noise (about 20x the
still's).  Focus moves too: the eye moves along the axis during a burst, so
the frame is rendered at its own focal offset (frame_optics) and the average
with the frames' mixture of focus (Optics.focus_spread); the average is also
blurred by the residual registration error of its frames, which varies over
the field (imaging.registration_field).  The texture's fine scales follow
each image's focus at the textured tissue's depth (texture_blur_map,
imaging.defocus_texture).  So a frame is sharper than its average in most of
the field and grainier, and may be blurrier where its focus is off.

Ground truth (labels, all in the image frame, pixel (0, 0) at d_c (0, 0)):
  * dominant / runner_up: per pixel, the vessel contributing the most and the
    second most optical density.  A vessel's contribution at a pixel is its
    own blurred cross-section at the nearest point of its centreline (the
    6-box staircase of its chord law, the vessel's blur plus the image's own
    blur; the halo is left out), so where vessels cross the one that darkens
    the pixel most wins, and at a junction the widest chord wins.
  * lumen_top / lumen_second: the shallowest and next shallowest vessel whose
    projected lumen (|distance| <= r) covers the pixel: the depth order.
  * centreline: vessel ids on a 1 px raster of the centrelines (shallower
    drawn over deeper).
  * cls / side / depth / radius / tier of the dominant vessel where its OD
    reaches OD_MIN.
  * These rasters (KIND_LABELS) are the average's; the frame, with its own
    focal offset, red-cell filling and blur, has its own under the same names
    with the suffix _frame (dominant_frame, tier_frame, ...).  lumen_top /
    lumen_second, centreline and the node heatmaps are geometry, the same for
    both.
  * heat_fork / heat_confluence / heat_anastomosis / heat_crossing: Gaussian
    node heatmaps (NODE_SIGMA px) at the nodes and crossings in the frame.
  * visibility per vessel and image kind: CNR = max over the DoG bands of
    the vessel's own band-filtered response (its cross-section after the PSF:
    its blur, the image's blur and the halo) divided by the RMS of the
    scene's own vessel-free background (texture + noise, ln DN) in the same
    band: like for like.  Tiers (plan section 6): required (max CNR >= 3 in
    frame and a contiguous run of >= 1 lambda = 11.9 px with CNR >= 1.5),
    dont_care (visible, CNR >= 1.5, but not required), invisible,
    out_of_frame.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy import ndimage as ndi

from . import anatomy as A
from . import imaging as IM
from . import render as R
from .graph import CLASSES, SIDES, VesselGraph, eval_profile

VERSION = "0.1"
KINDS = ("average", "frame")
PRESETS = ("healthy", "pathologic", "random")
SCENE_SALT = 0x5CE4E              # scene-level streams: SeedSequence([seed, SCENE_SALT]) (anatomy uses seed alone)
LAMBDA_PX = 11.9                  # real median vessel FWHM (px): the visibility length unit
CNR_REQUIRED, CNR_VISIBLE = 3.0, 1.5
OD_MIN = 0.005                    # Np: class / depth / radius / tier maps where the dominant OD reaches this
NODE_SIGMA = 2.0                  # px: node heatmap Gaussians
TIER_NAMES = ("none", "invisible", "dont_care", "required")     # tier map codes 0..3
NODE_MAPS = ("fork", "confluence", "anastomosis", "crossing")
KIND_LABELS = ("dominant", "runner_up", "dominant_od", "runner_up_od", "cls", "side", "depth", "radius", "tier")
                                  # rasters that depend on the image kind: the frame's are saved as <key>_frame
_VQ = float(1 << 22)              # OD quantisation of the label keys
_ZMAX = 4000.0                    # d_c: depth keys count down from here
_LOW31 = (1 << 31) - 1

# Scene-level calibration against the real stills (README, "Calibration"; the modules keep their own
# neutral defaults).  s0 1.4 -> 1.8 px: edge width p10 1.68 -> 2.15 px (real 2.07) and FWHM p50 7.8 -> 10.7 px
# (real 11.9); halo 0.3 -> 0.45: FWHM / edge and the sigma*=2 vessel area.  Texture x 0.85: the rendered
# deep vessels and mesh now carry part of the 4-64 px background power the texture was measured with.
CALIBRATED_OPTICS = dict(s0_px=1.55, halo_h=0.45, incline_max=1.4, mu=0.135, l_bypass=80.0, scatter_px_per_dc=0.015)
# v1 integration (README "Calibration, v1 integration"): s0 1.8 -> 1.55 px and the scatter blur 0.03 -> 0.015 px
# per d_c of depth (the renderer's default, never calibrated before).  The sharpest real raw-frame edges are 1.76 px
# (edge p10 of the 15-50-52 frames; 1.79 over 9 frames), below the old core, which was calibrated on averages
# before v1 added the frames' focus mixture and the registration field to them; and the web puts the
# conjunctival vessels at 12-46 d_c (v0 9-20), where 0.03 px/d_c blurred them by 0.4-1.4 px.  On the web anatomy
# the stills' edge p10 / p50 went 2.65 / 3.93 -> 2.24 / 3.47 px (real 2.19 / 3.36), FWHM p10 8.5 -> 7.1 px (6.8).
# Realism pass 2026-09-30 (README, "Calibration, second pass"): mu 0.12 -> 0.135 Np per d_c at haematocrit 1
# and the scattering bypass length 60 -> 80 d_c: against the dense full-frame site the vessels were 0.5-0.8x
# too light (F SD 0.058 against 0.080, widest vessels' contrast 0.18 against 0.29); incline_max 2 -> 1.4 (the
# steep-dive 'clubs'); the focal surface below.
# v2 renderer (2026-10-01; render.composite_od, _scratch/v2/render/): vessels stacked at a crossing are composited
# by depth, exactly against the oracle.  Real crossings are lighter than the sum of their vessels' ODs: crossing
# additivity (the crossing's darkness over its darker arm, per the lighter arm's contrast) 0.84-0.87 in the dense
# stills against 1.01 in the v1 scenes (difference -0.14, image-bootstrap 95 % CI -0.24 .. -0.01).  The chord
# law's f_spectral (the light no blood absorbs: spectral leakage, light sent back above the vessels, stray light)
# 0.05 -> 0.35 gives -0.15 on the same v1 crossings (40 paired, -0.21 .. -0.11); mu 0.135 -> 0.185 keeps the thin
# vessels' contrast ((1 - f) mu at 20-30 d_c), l_bypass 80 -> 100 d_c the deep vessels as pale as before.
# The 12 healthy v1 anatomies, v1 renderer and optics -> v2: dark knots' 10th percentile -0.115 -> -0.086 Np (real
# -0.085), dark knots per 100 lambda 0.80 -> 0.70 (real 0.47; frames 1.91 -> 1.62, real 0.88); the wide vessels 0.06
# Np lighter (widest vein -0.434 -> -0.373, real -0.471; F SD 0.075 -> 0.068, real 0.080).  Rounder veins restore
# them: the anatomy's vein_aspect 0.4-0.65 was set against the v1 chord law (tar-black veins); with these optics
# round collecting veins give widest vein -0.464, area darker than -0.45 Np 0.017 (real 0.017), F SD 0.081 (seeds
# 0-5; the square root of the aspect: -0.410, F SD 0.075).
CALIBRATED_OPTICS.update(f_spectral=0.35, mu=0.185, l_bypass=100.0)
# v2 integration (_scratch/v2/integ/changelog.md): the tissue halo 0.45 -> 0.375.  Real thin vessels are 1.5-2x darker
# than the synthetic ones (3x more thin, high-contrast profiles in real raw frames, 5x in averages; single-frame
# builder), and the halo moves (h) of every vessel's OD into an 8 px skirt.  6 seeds, against the dense site / the 24
# pooled real frames: still FWHM p10 7.47 -> 6.93 px (real 6.81), long lines n_long 54 -> 62 (68), frame FWHM p10
# 9.8 -> 9.2 px (6.8; +4.1 -> +3.3 SD), dense scorecard 55 -> 54 of 102.  0.3 went further (n_long 70, frame FWHM p10
# 8.7) but lost 6 scorecard rows (background band power, ridge fractions, along-vessel contrast CV -4.1 SD).
CALIBRATED_OPTICS.update(halo_h=0.375)
# Red-cell filling (imaging.filling), the same pass: a single frame lost 2.4 % of the average's medium vessels
# (0.2 % in real frames): single cells with short gaps (train_cells 3 -> 1), continuous columns from radius
# 1.2 d_c (1.6), fewer plasma-only (0.15 -> 0.03) and stop-go (0.25 -> 0.1) capillaries, and red-cell aggregates
# (grain 0.16) in the continuous vessels: a real frame is its average, grainier.
CALIBRATED_FILL = dict(train_cells=1.0, r_cont=1.2, p_empty=0.03, p_stopgo=0.1, grain_amp=0.12, grain_r_exp=1.0,
                       grain_amp_max=0.35, grain_r_agg=2.0)
# v1 fix (realism review T8): the aggregates' grain grows with r up to 2 d_c (0.2), then falls as sqrt(2 / r) (a chord
# through a wide lumen averages independent aggregates): 0.14 at r 4, 0.1 at r 8 (was up to 0.35: 'bamboo' bands).
# 2.5 was tried: medium vessels' graininess r_cv +6.6 against +3.9 SD, the wide vessels' frame - average grain the same
# v1 (README "Calibration, v1 optics"): with the average blurred by the frames' focus jitter and the registration
# field, the frame and its average differ by more than before, so the red-cell aggregates' grain 0.16 -> 0.14
# (at 0.16 the frame - average difference along medium vessels at 2.5 px - lambda was 0.0083 against 0.0063 real,
# the graininess r_cv 0.143 against 0.119; 0.12 gave 0.0059 and 0.113).  r_cont 1.4 (gaps in 1.2-1.4 d_c vessels)
# matched the frames' 'weak' share (1.2 %, real 1.17 %) but drew regular dashes along those vessels that real frames
# do not show: not kept.  imaging.filling's gap_fill (partial gaps in 0.6-1.2 d_c vessels) is left at 0: at 0.6 it
# made the frames' medium vessels too continuous (weak 0.12 %, absent 0 %, against 1.2 % and 0.19 % real), and
# real frames do show some dashed stretches.
# v1 integration: on the web anatomy the uniform grain 0.14 left the frames' grain along wide vessels low
# (`grain_coarse` 0.0026 against 0.0063 real) with the medium vessels' graininess (`r_cv` 0.102 against 0.119)
# and the weak share (0.6 % against 1.2 %) low too; a uniform 0.25 fixed the first but overshot `r_cv` (0.168) and
# `weak` (1.6 %), 0.18 left all three off.  The web's wide continuous vessels are deeper and darker than v0's twigs
# (the chord law compresses their OD fluctuation), and red cells aggregate more at the lower shear rates of wider
# venules: the aggregates' relative SD is now 0.12 x (r / r_cont), at most 0.35 (0.12 at r 1.2 d_c, 0.2 at 2,
# 0.35 from 3.5): grain_coarse 0.0056, r_cv 0.130, weak 0.66 % (6 healthy scenes).
# v2 single frames (2026-10-01; _scratch/v2/frames/; real: 8 raw frames of each of 15-22-26, 15-31-37, 15-50-52
# carried onto their averages).  The red cells' speed: v1 scaled it with flow / area, which drove medium venules at
# 3-18 mm/s and smeared their red cells over 20-150 px in a frame (no beads; bead_len 7.5 px against 3.5 real).
# The conjunctiva's red cells move at about 0.5 mm/s in every calibre (0.53 +- 0.15 mm/s over 21 +- 8 um vessels,
# Brennan 2021; capillaries 0.51 mm/s and the same in 14-24 um venules, Moka & Koutsiaris 2019; venules
# 0.51 +- 0.17, Hwang 2021): v_cap 85 d_c/s (0.51 mm/s at 6 um), speed_exp 0.1, speed_sd 0.3.  The aggregates
# come as clumps and plasma-rich stretches (Meyer 2018: erythrocytes aggregate in the low-order drainage vessels),
# 1.5 d_c long, soft ends 0.15 d_c, amplitude 0.09 x r / r_cont (at most 0.15 at 2 d_c), falling as 2 / r beyond (a
# pattern the same across the lumen cannot show the aggregates side by side, which average: 'bamboo' otherwise);
# the cells' granularity across the lumens is scene.frame_mottle (mottle 0.6: structure factor 0.36, a dense
# suspension; saturating on the whole column where vessels cross, so that crossings do not gain dark beads).
# Capillaries: column and gap lengths gamma of shape 5 (v1 exponential) and soft column ends 0.4 d_c
# (parachute-shaped cells taper): both calibrated on the faint-line statistics, not measured.  The 12 v1 healthy
# anatomies (v1 -> v2; real: 24 frames pooled / 15-50-52): faint lines dashed 7.0 -> 3.3 % (3.5 / 3.4 %), gaps per
# 100 lambda 5.6 -> 2.9 (3.3 / 2.8), absent 4.9 -> 1.6 % (5.0 / 3.7 %: now too continuous); beads: bead_len 7.5 ->
# 3.9 px (3.5 / 3.7), bead_med_rel 0.024 -> 0.033 (0.038 / 0.042), grain_fine 0.0012 -> 0.0024 (0.0026 / 0.0031),
# grain_coarse 0.0041 -> 0.0052 (0.0054 / 0.0063), bands across wide lumens 0.008 -> 0.015 (0.021 / 0.020).  The
# average is the mean of such frames (filling); its still statistics barely move (6 scenes, same code, against the
# dense site: xing_shallow -0.9, cprof_p90 -0.8, pair_frac -0.6, along_cv_fwhm -0.5, knot_med -0.5 SD, the rest
# within 0.4).
CALIBRATED_FILL.update(v_cap=85.0, speed_exp=0.1, speed_sd=0.3, v_max=500.0, grain_kind="clump", grain_len=1.5,
                       grain_edge=0.15, grain_amp=0.09, grain_r_pow=1.0, col_shape=5.0, edge=0.4, mottle=0.6)
# v2 fix pass (realism review v2, tell 5): the capillaries' dashes were more regular than real ones (gap-length CV
# along thin paths 0.34-0.43 against 0.47-0.57 in 15-50-52, 15-22-26, 15-31-37): column and gap lengths gamma of
# shape 3 (CV 0.58 before the blur) instead of 5 (0.45).
CALIBRATED_FILL.update(col_shape=3.0)
# The focal surface (realism review T3; v1 gap 4, README "Calibration, v1 optics").  The camera looks obliquely at
# the globe near the canthus and focus is not controlled: tilt 0.2-0.6 d_c of depth per d_c (the focal plane 11-31
# deg to the tissue; v0 0.03-0.18 left every scene sharper than every real still: tile focus variation 0.09
# against 0.13-0.45), the globe's curvature -1e-4 .. -4e-4 per d_c, and a smooth random part (4 cosines, total
# SD 0-35 d_c, wavelengths 150-500 d_c).  Calibrated on the tile focus statistics of the 4 dense stills and of 6
# real raw frames (focus_var 0.09 -> 0.15, real 0.15; soft tiles 0.03 -> 0.15, real 0.22).
FOCUS_TILT = (0.2, 0.6)
# The depth in focus at the frame centre: 6-35 d_c (was 6-20: the stills' widest black vessels have 2-3 px
# edges, so some stills are focused at the depth of the large vessels, with the fine conjunctival ones blurred).
FOCUS_DEPTH = (6.0, 35.0)
FOCUS_CURV = (-4e-4, -1e-4)
FOCUS_GAIN = (0.04, 0.08)          # px of blur per d_c of defocus (v1 integration: was 0.05-0.10; with the web anatomy
                                   # the stills' edge p50 3.68 -> 3.55 px, real 3.36; focus_var 0.18 -> 0.17, real 0.15)
FOCUS_MODES_SD = (0.0, 35.0)
FOCUS_MODES_WAVE = (150.0, 500.0)
# The burst's axial focus jitter (v1 gap 5): SD (d_c) of the frames' focal offset; the average is blurred by the
# frames' mixture of focus, a single frame has its own offset.  In the real pair (15-50-52, 8 raw frames against
# the non-rigid average) a frame is sharper than the average in one part of the field and blurrier in another
# (per-tile extra blur of the average -1.2 .. +1.7 px, 27 % of tiles negative), and the per-frame affine scale
# varies by 0.08 % SD (the eye moves along the axis).  2-8 d_c: per-tile p10 / p50 / p90 -1.4 / 1.0 / 1.9 px,
# 27 % negative.
FOCUS_JITTER = (2.0, 8.0)
# The depth (d_c) of the textured tissue whose fine texture follows the focus map.
TEX_DEPTH = (15.0, 45.0)
CALIBRATED_IMAGING = dict(tex_scale=0.85)
# The camera's analogue gain, drawn per scene (v2 fix pass, realism review v2, tell 9: every frame had the gain of
# 15-50-52, 20.4 dB, at the noisy end of the real frames: high-pass SD 0.0186-0.0191 Np against 0.0186 / 0.0153 /
# 0.0094 for 15-50-52 / 15-22-26 / 15-31-37, whose bursts ran at 20.4 / 13.1 / 13.9 dB; the reference stills'
# bursts 11.6-23.9 dB).  None: ImagingParams' 20.4 dB.
GAIN_DB = (12.0, 20.4)
# The average's residual registration blur (v1 gap 5), drawn per scene: a still may come from the non-rigid or
# the translation pipeline (quadrants of a translation-stabilized burst disagree by 1.0-1.7 px per frame, the
# non-rigid tiles' residuals 0.15-10 px), and some regions register worse: a log-normal field, median SD 0.2-0.6 px
# (log-uniform), log-SD 0.4-0.8 over the field, correlation length 500 px, at most 4 px.  Calibrated with the jitter
# on the frame/average extra blur of the real pair (median 0.93 px, 0.56-1.14 over 8 frames; v0 0.69, now 1.04
# [0.55, 1.34] over 10 scenes) and the stills' focus statistics above.
REG_ERROR = (0.2, 0.6)
REG_CV = (0.4, 0.8)
REG_FIELD_PX = 500.0
# The texture seen through vessels (imaging.tex_bypass): the tissue texture is mostly beneath the vessels, and
# over a vessel only the light that crossed the blood came from there.  0.3 = the chord law's unabsorbed share
# at a depth of ~25 d_c (1 - 0.95 exp(-25 / 80)).  In the real stills the structure along the lumen of medium and
# wide vessels is 0.68-0.71 of that beside them; with 0.3 the texture's own share inside is ~0.73 of outside (the
# rest of the synthetic lumen structure is the anatomy's radius and haematocrit variation along vessels).
TEX_BYPASS = 0.3
CROSSING_BYPASS = True              # composite crossing vessels by depth (crossing_bypass; v1 fix, realism review T1):
                                    # the light scattered in above the upper vessel crossed neither; False: ODs add


# ======================================================================== presets
def optics_preset(name: str = "healthy", rng=None) -> R.Optics:
    """Per-scene optics.  Focus is not controlled in the recordings, so the
    focal surface (depth at the centre, tilt, curvature, a smooth random
    part), the defocus gain and the burst's axial focus jitter are drawn per
    scene: the same anatomy can be sharp or blurred, in whole regions, and an
    annotation must not depend on it.  The returned optics are the
    average's (focus_spread = the jitter SD); make_scene draws the single
    frame's own offset.  'healthy' / 'pathologic': the calibrated central
    ranges; 'random': wide (training).  Values in px (blur) and d_c (depths)."""
    rng = np.random.default_rng(rng)
    u = rng.uniform
    base = replace(R.Optics(), **CALIBRATED_OPTICS)
    if name in ("healthy", "pathologic"):
        o = replace(base, z_focus=float(u(*FOCUS_DEPTH)), defocus_px_per_dc=float(u(*FOCUS_GAIN)))
        o = _focal_surface(o, rng, FOCUS_TILT, FOCUS_CURV, FOCUS_MODES_SD, FOCUS_MODES_WAVE)
        return replace(o, focus_spread=float(u(*FOCUS_JITTER)))
    if name == "random":
        o = replace(base, s0_px=float(u(1.1, 2.2)), z_focus=float(u(0.0, 45.0)),
                    defocus_px_per_dc=float(u(0.02, 0.18)), scatter_px_per_dc=float(u(0.0, 0.08)),
                    mu=float(base.mu * math.exp(rng.normal(0, 0.25))), f_spectral=float(u(0.15, 0.5)),
                    l_bypass=float(math.exp(u(math.log(25.0), math.log(250.0)))),
                    halo_h=float(u(0.15, 0.6)), halo_px=float(u(4.0, 16.0)))
        o = _focal_surface(o, rng, (0.0, 0.7), (-6e-4, 0.0), (0.0, 40.0), FOCUS_MODES_WAVE)
        return replace(o, focus_spread=float(u(0.0, 1.5 * FOCUS_JITTER[1])))
    raise ValueError(f"unknown preset {name!r}")


def _focal_surface(o: R.Optics, rng, tilt, curv, modes_sd=(0.0, 0.0), wave=(150.0, 500.0), n_modes=4) -> R.Optics:
    """A focal surface: tilt |grad z_focus| ~ U(tilt) d_c per d_c in a random
    direction, curvature ~ U(curv) d_c per d_c^2 (negative: the globe curves
    away from the camera, so towards the edges the depth in focus is
    shallower; its radius, ~12 mm = 2000 d_c, gives -1/(2R) = -2.5e-4), and a
    smooth random part: n_modes cosines in random directions, wavelengths
    log-uniform in `wave` (d_c), phases uniform, amplitudes Rayleigh-like
    with a total SD ~ U(modes_sd) d_c (the surface of the eye near the
    canthus is not a sphere: the caruncle, the fornix, the lid pressing on
    it).  The centre (focus_x0, focus_y0) is set to the frame centre by
    make_scene."""
    g = float(rng.uniform(*tilt))
    a = float(rng.uniform(0.0, 2.0 * math.pi))
    o = replace(o, focus_gx=g * math.cos(a), focus_gy=g * math.sin(a), focus_curv=float(rng.uniform(*curv)))
    sd = float(rng.uniform(*modes_sd))
    if sd <= 0 or n_modes <= 0:
        return o
    amp = rng.normal(0.0, 1.0, n_modes)
    amp *= sd * math.sqrt(2.0 / n_modes) / max(math.sqrt(float(np.mean(amp ** 2))), 1e-9)    # SD of the sum = sd
    lam_ = np.exp(rng.uniform(math.log(wave[0]), math.log(wave[1]), n_modes))
    ang = rng.uniform(0.0, 2.0 * math.pi, n_modes)
    ph = rng.uniform(0.0, 2.0 * math.pi, n_modes)
    k = 2.0 * math.pi / lam_
    modes = tuple((float(amp[i]), float(k[i] * math.cos(ang[i])), float(k[i] * math.sin(ang[i])), float(ph[i]))
                  for i in range(n_modes))
    return replace(o, focus_modes=modes)


def frame_optics(opt: R.Optics, rng) -> R.Optics:
    """The single frame's optics: the average's focal surface displaced along
    the axis by this frame's offset focus_spread x tanh(n), n ~ N(0, 1) (a
    typical frame, within one SD of the burst's mean focus), no spread.  (It
    was N(0, focus_spread): 3 of 12 frames came out blurrier than their
    average over the whole field, against 0 of 8 real frames of 15-50-52;
    realism review v1, T7.  Bounded by the spread, a frame's extra defocus
    relative to its average, gain^2 (delta^2 - spread^2 - 2 delta mean(z -
    z_f)), is <= 0 unless the whole field lies on one side of the focus.)"""
    rng = np.random.default_rng(rng)
    n = float(rng.normal(0.0, 1.0))
    return replace(opt, focus_offset=float(opt.focus_offset + opt.focus_spread * math.tanh(n)), focus_spread=0.0)


def frame_mottle(plan: R.RenderPlan, pad: int, shape, optics: R.Optics, fill: dict, fp: dict, k: float,
                 burst_seed: int, frame_index: int) -> np.ndarray:
    """The single frame's red-cell granularity inside the lumens of the
    continuous vessels (imaging.rbc_mottle, amplitude fp['mottle']; radius
    from fp['r_cont'], the capillaries' cells being their filling's columns),
    each piece blurred by its own blur, smeared by its red cells' speed
    (fill[vid].speed) over the exposure, through the halo: delta OD (Np) on
    the frame's grid.  Drawn from (burst_seed, frame_index), so the frame's
    noise stream is untouched.  The average (the mean of its frames) keeps
    1 / sqrt(n_frames) of it: not drawn."""
    H, W = int(shape[0]), int(shape[1])
    ti = plan.tube_info
    v_cap = float(fp.get("v_cap", IM.FILL_DEFAULTS["v_cap"]))
    sp = np.array([(getattr(fill.get(int(v)), "speed", None) if fill.get(int(v)) is not None else None) for v in
                   np.asarray(ti["vid"]).astype(int)], dtype=object)
    sp = np.array([v_cap if x is None else float(x) for x in sp], float)
    rng = np.random.default_rng([int(burst_seed), int(frame_index), 0xCE11])
    d = IM.rbc_mottle((H + 2 * pad, W + 2 * pad), plan.tubes, rng, float(fp["mottle"]), float(k), float(optics.mu),
                      speed_dc_s=sp, exposure_s=float(fp.get("exposure_ms", 12.8)) * 1e-3,
                      r_min_px=float(fp.get("r_cont", 1.2)) * float(k), vessel=np.asarray(ti["vid"]).astype(np.int64))
    return R.apply_halo(d, optics)[pad:pad + H, pad:pad + W].astype(np.float32)


def texture_blur_map(opt: R.Optics, z_tex: float, k: float, shape, ref: R.Optics | None = None,
                     step: int = 16) -> np.ndarray:
    """Extra blur variance (px^2, (H, W)) of the textured tissue at depth
    z_tex (d_c) under `opt`, relative to the sharpest the texture is anywhere
    in the frame under `ref` (default: opt without the offset and spread):
    s(z_tex, xy)^2 - min s_ref^2, at least 0.  Evaluated every `step` px and
    interpolated (the surface is smooth)."""
    H, W = int(shape[0]), int(shape[1])
    ref = ref if ref is not None else replace(opt, focus_offset=0.0, focus_spread=0.0)
    ys = np.r_[np.arange(0, H, step), H - 1].astype(float)
    xs = np.r_[np.arange(0, W, step), W - 1].astype(float)
    XY = np.stack(np.meshgrid(xs, ys), -1) / float(k)
    s2 = np.asarray(opt.blur(z_tex, XY), float) ** 2
    s2_ref = float((np.asarray(ref.blur(z_tex, XY), float) ** 2).min())
    small = np.maximum(s2 - s2_ref, 0.0)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((ys, xs), small)
    yy, xx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    return f(np.stack([yy, xx], -1)).astype(np.float64)


def imaging_preset(name: str = "healthy", rng=None) -> IM.ImagingParams:
    """Image formation: the measured defaults (the camera of 15-50-52, the
    median-noise still) for 'healthy' / 'pathologic', ImagingParams.random
    for 'random'; the texture and camera seeds are drawn here so both image
    kinds of a scene share them."""
    rng = np.random.default_rng(rng)
    seeds = dict(scene_seed=int(rng.integers(2 ** 31)), camera_seed=int(rng.integers(2 ** 31)))
    if name == "random":
        return IM.ImagingParams.random(rng, **seeds)
    if name in ("healthy", "pathologic"):
        reg = dict(reg_error_px=float(math.exp(rng.uniform(math.log(REG_ERROR[0]), math.log(REG_ERROR[1])))),
                   reg_field_cv=float(rng.uniform(*REG_CV)), reg_field_px=float(REG_FIELD_PX), tex_bypass=TEX_BYPASS)
        reg.update(CALIBRATED_IMAGING)
        if GAIN_DB is not None:                  # drawn last: the draws above are unchanged
            reg["gain_db"] = float(rng.uniform(*GAIN_DB))
        return IM.ImagingParams(**seeds, **reg)
    raise ValueError(f"unknown preset {name!r}")


# ======================================================================== the scene
@dataclass
class Scene:
    """A synthetic scene.  images[kind]: float32 DN, NaN = no data (kind in
    'average', 'frame'); od_clean[kind]: the vessels' optical density (Np,
    halo included, as seen in ln DN) before image formation; labels: the
    truth rasters (module docstring); visibility: {vid: {kind: dict(max_cnr,
    visible_px, frame_px, tier)}}; crossings: the in-frame crossings (px);
    formed[kind]: the rest of imaging.form_image's output (valid, clean,
    background, noise, blur_cov, ...); params: everything needed to redo it;
    junctions[kind]: the image-junction truth (junctions.image_junctions:
    every place where vessels meet or overlap in that image, its type and
    which arms belong to the same vessel); junctions_all[kind]: the same
    with the hidden ones (keep_hidden; the complete truth); observable[kind]:
    the observable truth (truth.observable_truth: what a careful person could
    see and measure in that image, the annotation target); vis_profiles[kind]:
    the per-vessel per-sample visibility profiles (truth.visibility_profiles,
    the complete truth's)."""
    graph: VesselGraph
    params: dict
    images: dict = field(default_factory=dict)
    od_clean: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    visibility: dict = field(default_factory=dict)
    crossings: list = field(default_factory=list)
    formed: dict = field(default_factory=dict)
    fills: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    junctions: dict = field(default_factory=dict)
    junctions_all: dict = field(default_factory=dict)
    observable: dict = field(default_factory=dict)
    vis_profiles: dict = field(default_factory=dict)


def _render(g: VesselGraph, k: float, shape, optics: R.Optics, hct_mod, device):
    """render.render_od, keeping the plan (its pieces carry the vessel ids,
    for the labels and the visibility), with the renderer's depth-ordered
    compositing of crossing vessels (render.composite_od, if CROSSING_BYPASS;
    v2: exact against the oracle, replacing crossing_bypass).  Returns (od,
    plan, pad)."""
    H, W = shape
    p = R.halo_pad(optics)
    org = np.array([0.0, 0.0]) - p / float(k)
    plan = R.build_plan(g, k, (H + 2 * p, W + 2 * p), optics, org, hct_mod, composite=bool(CROSSING_BYPASS))
    od = R.evaluate_plan(plan, (H + 2 * p, W + 2 * p), device)
    od = R.apply_halo(od, optics)[p:p + H, p:p + W]
    return od.astype(np.float32), plan, p


def crossing_bypass(g: VesselGraph, plan: R.RenderPlan, shape, optics: R.Optics, device="cuda") -> np.ndarray:
    """How much lighter (Np, >= 0) each pixel is where two vessels cross at
    different depths than the renderer's sum of their optical densities.

    The chord law's unabsorbed share f(z) = 1 - (1 - f_s) exp(-z / l_bypass)
    is, apart from the spectral part f_s, light scattered into the path above
    depth z.  Where a vessel t lies over a deeper vessel b, the light
    scattered in above t crossed neither, the light scattered in between
    crossed only t, the rest both.  With T'_i = (T_i - f_s) / (1 - f_s) the
    transmissions of the absorbed band, g_t = 1 - exp(-z_t / l_bypass):
        T' = T'_t T'_b + g_t / (1 - g_t) (1 - T'_t)(1 - T'_b),
    lighter than the product T'_t T'_b the sum of ODs gives; under a
    near-black vessel what lies deeper adds almost nothing (a deeper vessel
    of OD 0.22 under a vein of OD 0.8 at f = 0.3: 0.08 Np instead of 0.22).
    (Realism review v1, T1: the sum made black knots and spindles at
    crossings.)  Per pixel the two vessels contributing most OD (their own
    blurred cross-sections, as the labels' dominant / runner-up) are
    composited so; vessels that share a node (a junction: one blood volume,
    the renderer's union) are left alone, as are third and further layers.
    The blurred contributions stand in for the transmissions: exact for a
    sharp vessel over a blurred one."""
    H, W = shape
    f_s = float(optics.f_spectral)
    lb = optics.l_bypass
    if not (lb is not None and np.isfinite(lb) and lb > 0):
        return np.zeros(shape, np.float32)
    P, idx, pairs, dev = _tube_pairs(plan, shape, 0, 0.0, device)
    if not len(idx):
        return np.zeros(shape, np.float32)
    npx = H * W
    best = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    for pid, ix, iy in pairs():
        v, _ = _pair_values(P, pid, ix, iy)
        m = v >= 1e-3
        key = (torch.clamp(torch.round(v * _VQ), max=_LOW31).to(torch.int64) << 31) | pid
        best.scatter_reduce_(0, (iy * W + ix)[m], key[m], reduce="amax")
    ok = best >= 0
    best_vid = torch.full_like(best, -1)
    best_vid[ok] = P["vid"][best[ok] & _LOW31]
    # vessels meeting at a node are one blood volume there (the renderer's union), not a crossing: the second
    # layer is the strongest vessel that shares no node with the first
    nmax = max(g.vessels, default=0) + 2
    U = torch.full((nmax,), -1, dtype=torch.int64, device=dev)
    V = torch.full((nmax,), -2, dtype=torch.int64, device=dev)
    ids = torch.as_tensor(list(g.vessels), dtype=torch.int64, device=dev)
    U[ids] = torch.as_tensor([e.u for e in g.vessels.values()], dtype=torch.int64, device=dev)
    V[ids] = torch.as_tensor([e.v for e in g.vessels.values()], dtype=torch.int64, device=dev)
    bvx = torch.clamp(best_vid, min=0)
    Ub, Vb = torch.where(ok, U[bvx], -3), torch.where(ok, V[bvx], -4)
    second = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    for pid, ix, iy in pairs():
        v, _ = _pair_values(P, pid, ix, iy)
        pix = iy * W + ix
        vid = P["vid"][pid]
        uu, vv, ub, vb = U[vid], V[vid], Ub[pix], Vb[pix]
        m = (v >= 1e-3) & (vid != best_vid[pix]) & (uu != ub) & (uu != vb) & (vv != ub) & (vv != vb)
        key = (torch.clamp(torch.round(v * _VQ), max=_LOW31).to(torch.int64) << 31) | pid
        second.scatter_reduce_(0, pix[m], key[m], reduce="amax")
    both = (best >= 0) & (second >= 0)
    out = np.zeros(npx, np.float32)
    if not bool(both.any()):
        return out.reshape(H, W)
    pix = torch.nonzero(both).flatten()
    k1, k2 = best[pix], second[pix]
    v1 = ((k1 >> 31).double() / _VQ).cpu().numpy()
    v2 = ((k2 >> 31).double() / _VQ).cpu().numpy()
    p1 = (k1 & _LOW31).cpu().numpy()
    p2 = (k2 & _LOW31).cpu().numpy()
    pix = pix.cpu().numpy()
    ti = plan.tube_info
    z_all = np.asarray(ti["z"])[idx]
    vid_all = np.asarray(ti["vid"])[idx].astype(np.int64)
    z1, z2 = z_all[p1], z_all[p2]
    a1, a2 = vid_all[p1], vid_all[p2]
    zt = np.minimum(z1, z2)
    gt = 1.0 - np.exp(-np.maximum(zt, 0.0) / lb)
    T1, T2 = np.exp(-v1), np.exp(-v2)
    q1 = np.clip((T1 - f_s) / (1.0 - f_s), 1e-6, 1.0)
    q2 = np.clip((T2 - f_s) / (1.0 - f_s), 1e-6, 1.0)
    qn = q1 * q2 + gt / np.maximum(1.0 - gt, 1e-6) * (1.0 - q1) * (1.0 - q2)
    Tn = f_s + (1.0 - f_s) * np.minimum(qn, 1.0)
    d = np.log(np.maximum(Tn, 1e-12)) - np.log(np.maximum(T1 * T2, 1e-12))
    d = np.clip(d, 0.0, None)
    out[pix] = d.astype(np.float32)
    return out.reshape(H, W)


def _image_blur(fo: dict) -> float:
    """Isotropic equivalent (px SD) of the blur image formation adds."""
    cov = np.asarray(fo.get("blur_cov", np.zeros((2, 2))), float)
    return float(math.sqrt(max(np.trace(cov) / 2.0, 0.0)))


def make_scene(seed: int, preset: str = "healthy", shape=(1200, 1920), k: float | None = None, kinds=KINDS,
               device: str = "cuda", graph: VesselGraph | None = None, anatomy_params=None,
               optics: R.Optics | None = None, imaging: IM.ImagingParams | None = None, fill_params=None,
               labels: bool = True, log=None, frame_focus_offset: float | None = None,
               junctions: bool = True, truth: bool = True) -> Scene:
    """One synthetic scene (module docstring).

    seed: everything follows from it: the anatomy is anatomy.generate(params,
        seed); the scene-level streams (preset draws, optics, imaging, the
        burst's filling, the noise of each image) come from
        SeedSequence([seed, SCENE_SALT]).
    preset: 'healthy', 'pathologic' or 'random' (anatomy.preset,
        optics_preset, imaging_preset).
    shape: (H, W) px.  k: px per d_c (default: the preset's, AnatomyParams.k
        = 4 for healthy and pathologic; random draws it).
    graph: an anatomy to re-use (then anatomy_params and k are ignored).
    anatomy_params / optics / imaging / fill_params: override the preset.
        optics are the average's (focus_spread: the SD of the frames' focal
        offset); the single frame's offset is drawn from it (frame_optics).
    labels: compute the truth rasters and visibility (a few seconds).
    frame_focus_offset: the single frame's axial focal offset (d_c) instead
        of the draw (e.g. 0: the frame at the burst's mean focus).
    junctions: with labels, also the image-junction truth of each image
        (junctions.scene_junctions; 10-30 s per full image on the CPU) and
        its rasters (labels junction_<key>, the frame's junction_<key>_frame).
    truth: with junctions, also the observable and complete truth of each
        image (truth.py; 6-10 s per full image): Scene.observable[kind],
        Scene.vis_profiles[kind], Scene.junctions_all[kind] and the labels
        observable_<key>_<kind> (truth.observable_rasters)."""
    say = log or (lambda *a: None)
    T = {}
    t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        t1 = time.perf_counter()
        T[name] = round(T.get(name, 0.0) + t1 - t0, 3)
        t0 = t1

    kinds = tuple(kinds)
    H, W = int(shape[0]), int(shape[1])
    ss = np.random.SeedSequence([int(seed), SCENE_SALT])
    r_pre, r_opt, r_img, r_fill, r_avg, r_frame = [np.random.default_rng(c) for c in ss.spawn(6)]
    r_noise = dict(average=r_avg, frame=r_frame)
    # ---- anatomy
    if graph is None:
        P = anatomy_params if anatomy_params is not None else A.preset(preset, r_pre)
        P = replace(P, frame_px=(W, H), **({"k": float(k)} if k else {}))
        say(f"anatomy: preset {preset}, seed {seed}, k {P.k:.2f}")
        g = A.generate(P, int(seed))
    else:
        g = graph
    kk = float(g.meta["k"])
    lap("anatomy")
    opt = optics if optics is not None else optics_preset(preset, r_opt)
    if opt.focus_x0 == 0.0 and opt.focus_y0 == 0.0:          # unset: the focal surface centred on the frame
        opt = replace(opt, focus_x0=0.5 * (W - 1) / kk, focus_y0=0.5 * (H - 1) / kk)
    # the average is made of frames focused at the surface + offsets of SD focus_spread; the single frame has one
    opt_kind = dict(average=opt, frame=frame_optics(opt, r_opt))
    if frame_focus_offset is not None:
        opt_kind["frame"] = replace(opt_kind["frame"], focus_offset=float(opt.focus_offset + frame_focus_offset))
    z_tex = float(r_opt.uniform(*(TEX_DEPTH if preset != "random" else (5.0, 60.0)))) if TEX_DEPTH else None
    ip = imaging if imaging is not None else imaging_preset(preset, r_img)
    if ip.scene_seed is None or ip.camera_seed is None:
        ip = replace(ip, scene_seed=ip.scene_seed if ip.scene_seed is not None else int(r_img.integers(2 ** 31)),
                     camera_seed=ip.camera_seed if ip.camera_seed is not None else int(r_img.integers(2 ** 31)))
    fp = dict(n_frames=int(ip.n_frames), fps=float(ip.fps), exposure_ms=float(ip.exposure_ms), relative=True)
    fp.update(CALIBRATED_FILL)
    fp.update(fill_params or {})
    burst_seed = int(r_fill.integers(2 ** 31))
    frame_index = int(r_fill.integers(max(int(fp["n_frames"]), 1)))
    sc = Scene(graph=g, params=dict(
        format="vesselscene-scene-v1", version=VERSION, seed=int(seed), preset=preset, shape=[H, W], k=kk,
        origin=[0.0, 0.0], kinds=list(kinds), anatomy=g.meta.get("params"), optics=opt.to_dict(),
        optics_frame=opt_kind["frame"].to_dict(), texture_depth=z_tex,
        imaging=ip.to_dict(), filling=dict(fp), burst_seed=burst_seed, frame_index=frame_index,
        seeds=dict(anatomy=f"anatomy.generate(params, seed={int(seed)})",
                   scene=f"numpy SeedSequence([{int(seed)}, {SCENE_SALT}]).spawn(6): preset, optics, imaging, "
                         "filling, noise of the average, noise of the frame",
                   texture=ip.scene_seed, camera=ip.camera_seed, burst=burst_seed)))
    # ---- the texture: one field per scene, its fine scales blurred per image by the focus at its depth
    tex_parts = IM.texture_parts((H, W), ip, None, blur=0.8)
    var_ref = texture_blur_map(opt_kind["average"], z_tex, kk, (H, W)) if z_tex is not None else None
    lap("texture")
    # ---- per kind: filling, render, image formation
    plans, lnE = {}, None
    for kind in kinds:
        fill = IM.filling(g, kind=kind, frame=frame_index, burst_seed=burst_seed, **fp)
        sc.fills[kind] = fill
        lap(f"filling_{kind}")
        ok_ = opt_kind[kind]
        od, plan, pad = _render(g, kk, (H, W), ok_, fill, device)
        lap(f"render_{kind}")
        if kind == "frame" and fp.get("mottle", 0.0) > 0:          # the frame's red cells: granular lumens
            od = od + frame_mottle(plan, pad, (H, W), ok_, fill, fp, kk, burst_seed, frame_index)
            lap("mottle_frame")
        plans[kind] = (plan, pad)
        sc.od_clean[kind] = od
        say(f"{kind}: {plan.stats['pieces']} pieces, {plan.stats['polygons']} polygons, "
            f"{plan.stats.get('pairs', 0) / 1e6:.0f} M pairs")
        if z_tex is None:                                     # texture defocus off: one texture for both kinds
            tau = IM.texture((H, W), ip, parts=tex_parts)
        else:
            vmap = var_ref if kind == "average" else texture_blur_map(ok_, z_tex, kk, (H, W),
                                                                      ref=replace(opt, focus_spread=0.0))
            tau = IM.defocus_texture(tex_parts, vmap, norm_var_map=var_ref)
        lap(f"texture_{kind}")
        fo = IM.form_image(od, ip, r_noise.get(kind, r_avg), kind=kind, tau=tau, lnE=lnE)
        lnE = fo["lnE"]
        sc.images[kind] = fo.pop("image")
        fo.pop("params", None)
        sc.formed[kind] = fo
        lap(f"form_{kind}")
    sc.params["render_stats"] = {kind: dict(plans[kind][0].stats) for kind in kinds}
    sc.params["formed"] = {kind: dict(n_frames=int(sc.formed[kind].get("n_frames", 1)),
                                      blur_px=_image_blur(sc.formed[kind]),
                                      blur_cov=np.asarray(sc.formed[kind].get("blur_cov")).tolist(),
                                      saccades=int(sc.formed[kind].get("saccades", 0)),
                                      glare_spots=len(sc.formed[kind].get("glare", [])))
                           for kind in kinds}
    if labels:
        ref = "average" if "average" in kinds else kinds[0]
        plan, pad = plans[ref]
        lab = label_maps(g, plan, (H, W), pad, kk, _image_blur(sc.formed[ref]), device)
        # the other kinds' own rasters (the frame has its own focus, filling and blur: 18 % of in-frame vessels
        # change tier between the average and the frame; correctness review v1, defect 5), with a _<kind> suffix
        other = {}
        for kind in kinds:
            if kind != ref:
                plan_k, pad_k = plans[kind]
                other[kind] = label_maps(g, plan_k, (H, W), pad_k, kk, _image_blur(sc.formed[kind]), device)
        lap("labels")
        cr = [c for c in g.crossings() if _in_frame(kk * c["x"], kk * c["y"], H, W)]
        sc.crossings = [dict(c, x_px=kk * c["x"], y_px=kk * c["y"]) for c in cr]
        lab.update(node_heatmaps(g, kk, (H, W), sc.crossings))
        lap("nodes_crossings")
        for kind in kinds:
            plan, pad = plans[kind]
            vis = visibility(plan, (H, W), pad, opt_kind[kind], _image_blur(sc.formed[kind]), sc.formed[kind],
                             sc.images[kind], kind)
            for vid, rec in vis.items():
                sc.visibility.setdefault(vid, {})[kind] = rec
        for vid in g.vessels:
            for kind in kinds:
                sc.visibility.setdefault(vid, {}).setdefault(kind, dict(max_cnr=0.0, visible_px=0.0,
                                                                         visible_run_px=0.0, frame_px=0.0,
                                                                         tier="out_of_frame"))
        for kind, lk in [(ref, lab)] + list(other.items()):
            tier_of = np.zeros(max(g.vessels, default=0) + 2, np.uint8)
            for vid, rec in sc.visibility.items():
                t = rec[kind]["tier"]
                # a vessel whose lumen never reaches the frame but whose blurred skirt dominates pixels at its
                # edge: visible there, not required (it was coded 'invisible')
                tier_of[vid] = TIER_NAMES.index(t) if t in TIER_NAMES else TIER_NAMES.index("dont_care")
            dom = lk["dominant"]
            strong = (dom >= 0) & (lk["dominant_od"] >= OD_MIN)
            lk["tier"] = np.where(strong, tier_of[np.maximum(dom, 0)], 0).astype(np.uint8)
            for key in ("cls", "side", "depth", "radius"):
                lk[key] = np.where(strong, lk[key], 255 if key in ("cls", "side") else np.nan).astype(lk[key].dtype)
        for kind, lk in other.items():              # the lumen layers are geometry: the same for every kind
            for key in KIND_LABELS:
                lab[f"{key}_{kind}"] = lk[key]
        lab["centreline"] = centreline_raster(g, kk, (H, W))
        sc.labels = lab
        lap("visibility")
        if junctions:
            # the image-junction truth (v2): per image, from its own OD, optics, filling and background noise; all
            # junctions (hidden ones too: the complete truth) once, the default list is their visible subset
            from . import junctions as J
            for kind in kinds:
                S = J.scene_samples(sc, kind)
                sc.junctions_all[kind] = J.junctions_from_samples(S, g, crossings=sc.crossings or None,
                                                                  keep_hidden=True)
                sc.junctions[kind] = J.visible_subset(sc.junctions_all[kind])
                suf = "" if kind == ref else f"_{kind}"
                for key, v in J.junction_rasters(sc.junctions[kind], (H, W)).items():
                    lab[f"junction_{key}{suf}"] = v
                lap(f"junctions_{kind}")
                if truth:
                    # the observable truth (the annotation target) and the per-sample visibility profiles (truth.py)
                    from . import truth as TR
                    obs, prof = TR.observable_truth(TR.scene_context(sc, kind, samples=S), kind,
                                                    return_profiles=True)
                    sc.observable[kind], sc.vis_profiles[kind] = obs, prof
                    for key, v in TR.observable_rasters(obs, prof, (H, W)).items():
                        lab[f"observable_{key}_{kind}"] = v
                    lap(f"truth_{kind}")
    sc.timings = T
    sc.params["timings"] = T
    return sc


def _in_frame(x, y, H, W, margin=0.0):
    return (-0.5 - margin <= x <= W - 0.5 + margin) and (-0.5 - margin <= y <= H - 0.5 + margin)


# ======================================================================== truth rasters
def _tube_pairs(plan: R.RenderPlan, shape, pad: int, blur_px: float, device):
    """Device tensors of the plan's tube pieces moved into the frame (offset
    by the render pad), with the image blur added to their own, and a
    generator of (piece, ix, iy) pairs over each piece's reach."""
    H, W = shape
    dev = R._device(device)
    tb, ti = plan.tubes, plan.tube_info
    n = len(tb["l"])
    s = np.sqrt(tb["s"] ** 2 + blur_px ** 2)
    cx, cy = tb["cx"] - pad, tb["cy"] - pad
    Ua = 0.5 * tb["l"] + R.NSIG * s
    Da = tb["r"] + R.NSIG * s
    hx = np.abs(tb["tx"]) * Ua + np.abs(tb["ty"]) * Da
    hy = np.abs(tb["ty"]) * Ua + np.abs(tb["tx"]) * Da
    x0 = np.maximum(np.ceil(cx - hx), 0).astype(np.int64)
    x1 = np.minimum(np.floor(cx + hx), W - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(cy - hy), 0).astype(np.int64)
    y1 = np.minimum(np.floor(cy + hy), H - 1).astype(np.int64)
    ok = (x1 >= x0) & (y1 >= y0) if n else np.zeros(0, bool)      # empty vessels too: their lumen is anatomy
    idx = np.flatnonzero(ok)
    dt = torch.float32 if dev.type == "cuda" else torch.float64
    P = dict(cx=cx[idx], cy=cy[idx], tx=tb["tx"][idx], ty=tb["ty"][idx], l=tb["l"][idx], r=tb["r"][idx], s=s[idx])
    P = {key: torch.as_tensor(v, dtype=dt, device=dev) for key, v in P.items()}
    P["W"] = torch.as_tensor(R.box_weights(tb["a"][idx], tb["f"][idx]), dtype=dt, device=dev)
    P["vid"] = torch.as_tensor(ti["vid"][idx].astype(np.int64), device=dev)
    P["zq"] = torch.as_tensor(np.round(np.clip(_ZMAX - ti["z"][idx], 0, _ZMAX) * 2 ** 16).astype(np.int64),
                              device=dev)
    rect = [torch.as_tensor(a[idx], device=dev) for a in (x0, y0, x1 - x0 + 1, y1 - y0 + 1)]

    def pairs():
        if len(idx):
            yield from R._rect_chunks(*rect)
    return P, idx, pairs, dev


def _pair_values(P, pid, ix, iy):
    """Per (pixel, piece): the piece's blurred cross-section at the pixel's
    distance from the piece (its segment, clamped at the ends): the vessel's
    own OD at its nearest centreline point; and that distance."""
    vx = ix.to(P["cx"].dtype) - P["cx"][pid]
    vy = iy.to(P["cx"].dtype) - P["cy"][pid]
    tx, ty = P["tx"][pid], P["ty"][pid]
    u = vx * tx + vy * ty
    d = vy * tx - vx * ty
    hl = 0.5 * P["l"][pid]
    du = u - torch.clamp(u, -hl, hl)
    dist = torch.sqrt(du * du + d * d)
    s = P["s"][pid][:, None]
    w = P["r"][pid][:, None] * torch.as_tensor(R.BOX_C, dtype=s.dtype, device=s.device)
    ncdf = torch.special.ndtr
    dd = dist[:, None]
    v = (P["W"][pid] * (ncdf((w - dd) / s) + ncdf((w + dd) / s) - 1.0)).sum(1)
    return v, dist


def label_maps(g: VesselGraph, plan: R.RenderPlan, shape, pad: int, k: float, blur_px: float,
               device="cuda") -> dict:
    """The per-pixel truth rasters of a rendered scene (module docstring):
    dominant / runner_up (+ their OD), lumen_top / lumen_second, and the
    dominant vessel's class, side, depth (d_c) and radius (d_c).  Two
    passes over the render plan's tube pieces on the device; each pixel
    keeps the max of (quantised value << 31 | piece) by scatter_reduce."""
    H, W = shape
    P, idx, pairs, dev = _tube_pairs(plan, shape, pad, blur_px, device)
    npx = H * W
    best = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    top = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    rmax = P["r"] if len(idx) else None
    for pid, ix, iy in pairs():
        v, dist = _pair_values(P, pid, ix, iy)
        pix = iy * W + ix
        m = v >= 1e-4
        key = (torch.clamp(torch.round(v * _VQ), max=_LOW31).to(torch.int64) << 31) | pid
        best.scatter_reduce_(0, pix[m], key[m], reduce="amax")
        m = dist <= rmax[pid]
        top.scatter_reduce_(0, pix[m], (P["zq"][pid] << 31 | pid)[m], reduce="amax")

    def vid_of(key):
        out = torch.full_like(key, -1)
        ok = key >= 0
        out[ok] = P["vid"][key[ok] & _LOW31]
        return out
    best_vid, top_vid = vid_of(best), vid_of(top)
    second = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    second_top = torch.full((npx,), -1, dtype=torch.int64, device=dev)
    for pid, ix, iy in pairs():
        v, dist = _pair_values(P, pid, ix, iy)
        pix = iy * W + ix
        vid = P["vid"][pid]
        m = (v >= 1e-4) & (vid != best_vid[pix])
        key = (torch.clamp(torch.round(v * _VQ), max=_LOW31).to(torch.int64) << 31) | pid
        second.scatter_reduce_(0, pix[m], key[m], reduce="amax")
        m = (dist <= rmax[pid]) & (vid != top_vid[pix])
        second_top.scatter_reduce_(0, pix[m], (P["zq"][pid] << 31 | pid)[m], reduce="amax")
    out = {}
    ti = plan.tube_info
    for name, key in (("dominant", best), ("runner_up", second), ("lumen_top", top), ("lumen_second", second_top)):
        kc = key.cpu().numpy()
        ok = kc >= 0
        piece = np.where(ok, kc & _LOW31, 0)
        vid = np.where(ok, ti["vid"][idx][piece] if len(idx) else -1, -1).astype(np.int32)
        out[name] = vid.reshape(H, W)
        if name in ("dominant", "runner_up"):
            out[name + "_od"] = np.where(ok, (kc >> 31) / _VQ, 0.0).astype(np.float32).reshape(H, W)
        if name == "dominant":
            gi = idx[piece] if len(idx) else piece
            cls_i = {vid_: CLASSES.index(e.cls) for vid_, e in g.vessels.items()}
            side_i = {vid_: SIDES.index(e.side) for vid_, e in g.vessels.items()}
            lut_c = np.full(max(g.vessels, default=0) + 2, 255, np.uint8)
            lut_s = np.full(max(g.vessels, default=0) + 2, 255, np.uint8)
            for vid_, c in cls_i.items():
                lut_c[vid_] = c
                lut_s[vid_] = side_i[vid_]
            vv = np.maximum(vid, 0)
            out["cls"] = np.where(ok, lut_c[vv], 255).astype(np.uint8).reshape(H, W)
            out["side"] = np.where(ok, lut_s[vv], 255).astype(np.uint8).reshape(H, W)
            out["depth"] = np.where(ok, ti["z"][gi] if len(idx) else 0, np.nan).astype(np.float32).reshape(H, W)
            out["radius"] = np.where(ok, ti["r_dc"][gi] if len(idx) else 0, np.nan).astype(np.float32).reshape(H, W)
    return out


def centreline_raster(g: VesselGraph, k: float, shape, spacing_px: float = 0.25) -> np.ndarray:
    """Vessel ids (int32, -1 elsewhere) on a 1 px raster of the centrelines;
    where two cross, the shallower is drawn last (on top)."""
    H, W = shape
    out = np.full((H, W), -1, np.int32)
    xs, ys, zs, ids = [], [], [], []
    for vid, e in g.vessels.items():
        c = k * np.asarray(e.ctrl)
        if c[:, 0].max() < -2 or c[:, 0].min() > W + 1 or c[:, 1].max() < -2 or c[:, 1].min() > H + 1:
            continue
        smp = e.sample(spacing_px / k)
        p = np.rint(k * smp["xy"]).astype(np.int64)
        m = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        if m.any():
            xs.append(p[m, 0])
            ys.append(p[m, 1])
            zs.append(smp["z"][m])
            ids.append(np.full(int(m.sum()), vid, np.int32))
    if xs:
        x, y, z, i = (np.concatenate(a) for a in (xs, ys, zs, ids))
        o = np.argsort(-z, kind="stable")                 # deepest first, shallowest written last
        out[y[o], x[o]] = i[o]
    return out


def node_heatmaps(g: VesselGraph, k: float, shape, crossings: list, sigma: float = NODE_SIGMA) -> dict:
    """Gaussian heatmaps (float32, peak 1) of the forks, confluences,
    anastomoses (several in and several out) and crossings in the frame."""
    H, W = shape
    maps = {f"heat_{n}": np.zeros((H, W), np.float32) for n in NODE_MAPS}
    pts = {n: [] for n in NODE_MAPS}
    deg = g.degrees()
    for nid, n in g.nodes.items():
        i, o = deg.get(nid, (0, 0))
        kind = "fork" if (i == 1 and o >= 2) else "confluence" if (o == 1 and i >= 2) else \
            "anastomosis" if (i >= 1 and o >= 1) else None
        if kind:
            pts[kind].append(k * n.xy)
    pts["crossing"] = [np.array([c["x_px"], c["y_px"]]) for c in crossings]
    R_ = int(math.ceil(3 * sigma))
    yy, xx = np.mgrid[-R_:R_ + 1, -R_:R_ + 1]
    for name, lst in pts.items():
        M = maps[f"heat_{name}"]
        for p in lst:
            if not _in_frame(p[0], p[1], H, W, margin=R_):
                continue
            cx, cy = int(round(p[0])), int(round(p[1]))
            gk = np.exp(-0.5 * ((xx + cx - p[0]) ** 2 + (yy + cy - p[1]) ** 2) / sigma ** 2)
            y0, y1, x0, x1 = cy - R_, cy + R_ + 1, cx - R_, cx + R_ + 1
            sy0, sx0 = max(0, -y0), max(0, -x0)
            y0c, x0c, y1c, x1c = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
            if y1c <= y0c or x1c <= x0c:
                continue
            sub = gk[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
            M[y0c:y1c, x0c:x1c] = np.maximum(M[y0c:y1c, x0c:x1c], sub)
    maps["node_counts"] = {n: int(sum(_in_frame(p[0], p[1], H, W) for p in lst)) for n, lst in pts.items()}
    return maps


# ======================================================================== visibility
def background_band_rms(formed: dict, image: np.ndarray, kind: str) -> dict:
    """RMS (Np) of the scene's own vessel-free background (texture, lumps,
    illumination, camera pattern and noise, in ln DN) per DoG band s (G_s -
    G_2s) over its valid pixels: what a vessel's contrast competes with."""
    bg = np.asarray(formed["background"], np.float64)
    noise = np.asarray(formed["noise"], np.float64)
    valid = np.asarray(formed.get("valid", np.isfinite(image)), bool) & np.isfinite(noise) & (bg > 0)
    if kind == "frame":
        valid &= image < IM.FULL_SCALE
    L = np.log(np.maximum(np.where(valid, bg + noise, 1.0), 1.0))
    Lf = IM.fill_invalid(L, valid)
    out = {}
    for s in IM.BANDS[:-1]:
        B = IM._dog(Lf, s)
        e = ndi.binary_erosion(valid, iterations=int(min(3 * s + 2, 40)))
        m = e if e.sum() > 1000 else valid
        out[s] = float(B[m].std())
    return out


def band_response(tb: dict, optics: R.Optics, blur_px: float, bands) -> tuple[np.ndarray, np.ndarray]:
    """Per tube piece: its peak OD after the PSF (its own blur, the image's
    blur and the halo terms) and, per DoG band s (G_s - G_2s, as the
    background's band RMS is measured), the band-filtered response at its
    centreline (the 1-D DoG across a straight vessel: each nested box
    blurred by the PSF and by G_s minus by G_2s).  Returns (peak (n,),
    response (n, n_bands))."""
    from scipy.special import ndtr
    s = np.sqrt(tb["s"] ** 2 + blur_px ** 2)
    Wk = R.box_weights(tb["a"], tb["f"])
    w = tb["r"][:, None] * R.BOX_C

    def peak(sig):
        return (Wk * (2.0 * ndtr(w / np.maximum(sig, 1e-6)[:, None]) - 1.0)).sum(1)
    terms = [(1.0 - sum(w_ for w_, _ in R._psf_terms(optics)), 0.0)] + R._psf_terms(optics)
    pk = sum(wt * peak(np.sqrt(s ** 2 + st ** 2)) for wt, st in terms)
    resp = np.zeros((len(s), len(bands)))
    for bi, b in enumerate(bands):
        for wt, st in terms:
            S2 = s ** 2 + st ** 2
            resp[:, bi] += wt * (peak(np.sqrt(S2 + b * b)) - peak(np.sqrt(S2 + 4 * b * b)))
    return pk, resp


def _longest_run(l: np.ndarray, ok: np.ndarray) -> float:
    """Longest total length of consecutive pieces with ok."""
    best = cur = 0.0
    for li, oi in zip(l, ok):
        cur = cur + li if oi else 0.0
        best = max(best, cur)
    return best


def visibility(plan: R.RenderPlan, shape, pad: int, optics: R.Optics, blur_px: float, formed: dict,
               image: np.ndarray, kind: str | None = None) -> dict:
    """Per vessel: max CNR, visible length (px, CNR >= CNR_VISIBLE), its
    longest contiguous visible run, length in frame, tier (module docstring).

    CNR of a piece = max over the DoG bands s of (the piece's own response in
    band s) / (the RMS of the scene's vessel-free background in band s): a
    like-for-like signal-to-noise ratio of a detector matched to the vessel's
    scale.  (The first version divided the full peak contrast by one band's
    RMS, which overstated it about 5x.)  A piece counts as in the frame
    where its lumen reaches the frame (centre within r + 2 s of it), so a
    vessel running just outside the edge but darkening it is scored.
    required: max CNR >= CNR_REQUIRED and a contiguous run of pieces with
    CNR >= CNR_VISIBLE at least LAMBDA_PX long; dont_care: some piece has
    CNR >= CNR_VISIBLE; otherwise invisible."""
    H, W = shape
    tb, ti = plan.tubes, plan.tube_info
    if not len(tb["l"]):
        return {}
    kind = kind or ("frame" if formed.get("n_frames", 2) == 1 else "average")
    rms = background_band_rms(formed, image, kind)
    bs = np.array(sorted(rms), float)
    noise = np.maximum(np.array([rms[b] for b in bs]), 1e-9)
    c, resp = band_response(tb, optics, blur_px, bs)
    cnr_b = resp / noise[None, :]
    cnr = cnr_b.max(1)
    best_band = bs[np.argmax(cnr_b, 1)]
    cx, cy = tb["cx"] - pad, tb["cy"] - pad
    reach = tb["r"] + 2.0 * np.sqrt(tb["s"] ** 2 + blur_px ** 2)
    inside = (cx >= -0.5 - reach) & (cx <= W - 0.5 + reach) & (cy >= -0.5 - reach) & (cy <= H - 0.5 + reach)
    vid = ti["vid"]
    out = {}
    order = np.argsort(vid, kind="stable")                         # stable: each vessel's pieces stay in order
    cuts = np.flatnonzero(np.diff(vid[order])) + 1
    for grp in np.split(order, cuts):
        m = grp[inside[grp]]
        v = int(vid[grp[0]])
        if not len(m):
            continue
        mx = float(cnr[m].max())
        vis_ok = inside[grp] & (cnr[grp] >= CNR_VISIBLE)
        vis = float(tb["l"][grp][vis_ok].sum())
        run = _longest_run(tb["l"][grp], vis_ok)
        tier = "required" if (mx >= CNR_REQUIRED and run >= LAMBDA_PX) else \
            "invisible" if mx < CNR_VISIBLE else "dont_care"
        out[v] = dict(max_cnr=round(mx, 3), visible_px=round(vis, 1), visible_run_px=round(run, 1),
                      frame_px=round(float(tb["l"][m].sum()), 1), peak_od=round(float(c[m].max()), 5),
                      band_px=float(best_band[m][np.argmax(cnr[m])]), tier=tier)
    return out


# ======================================================================== vesselmap export
def to_vesselnetwork(graph: VesselGraph, k: float, origin=(0.0, 0.0), shape=(1200, 1920),
                     optics: R.Optics | None = None, blur_px: float = 0.0, hct_mod=None, margin_px: float = 8.0):
    """The truth as vesselmap's VesselNetwork (image px): one Edge per vessel
    that comes within margin_px of the frame, its centreline control points
    exactly k (ctrl - origin) (the same clamped cubic basis), and profiles
    r (px), s (blur, px: the vessel's depth blur combined with blur_px, the
    blur image formation adds, e.g. 0.85 px for an average) and a (the
    linear-law centre OD whose nested boxes best match the rendered chord
    staircase, least squares over the boxes' areas), refitted on vesselmap's
    profile basis (spline parameter, about 30 px per control value).  The
    halo goes to net.meta['optics'] (vesselmap's (1 - h) G(s) + h
    G(sqrt(s^2 + s_h^2)) is exactly the renderer's).  Edge.info carries
    true_vessel, cls, side, flow_rate, and fit_err_* (max abs error of the refit
    profiles).  Nodes outside the frame are kept (vessels leave the image).
    (info['flow'] is vesselmap's measured-velocity record, so the truth's
    volumetric flow goes to info['flow_rate'].)"""
    from vesselscene import spline as sp
    try:
        from vesselmap.network import Edge, VesselNetwork, PROFILE_SPACING, S_MIN
    except ImportError as exc:                     # vesselmap lives in the LIMBUS repository
        raise ImportError("to_vesselnetwork needs vesselmap (github.com/karimghabra/limbus) "
                          "on the Python path") from exc
    optics = optics or R.Optics()
    H, W = int(shape[0]), int(shape[1])
    origin = np.asarray(origin, float)
    net = VesselNetwork((H, W))
    hct_mod = hct_mod or {}
    area = np.diff(np.r_[0.0, R.BOX_C])                  # box half-width increments (annulus widths / r)
    keep, nmap = [], {}
    for vid, e in graph.vessels.items():
        c = k * (np.asarray(e.ctrl) - origin)
        if c[:, 0].max() < -margin_px or c[:, 0].min() > W - 1 + margin_px or \
                c[:, 1].max() < -margin_px or c[:, 1].min() > H - 1 + margin_px:
            continue
        smp = e.sample(0.5 / k)
        p = k * (smp["xy"] - origin)
        if not ((p[:, 0] >= -margin_px) & (p[:, 0] <= W - 1 + margin_px) & (p[:, 1] >= -margin_px)
                & (p[:, 1] <= H - 1 + margin_px)).any():
            continue
        keep.append(vid)
    for vid in keep:
        e = graph.vessels[vid]
        for n in (e.u, e.v):
            if n not in nmap:
                xy = k * (graph.nodes[n].xy - origin)
                nmap[n] = net.add_node(float(xy[0]), float(xy[1]))
    for vid in keep:
        e = graph.vessels[vid]
        n = len(e.ctrl)
        ctrl = k * (np.asarray(e.ctrl, float) - origin)
        tr = R._Track(e, k, origin, optics, hct_mod.get(vid), spacing=0.25)
        m = max(256, 16 * n)
        u = np.linspace(0.0, 1.0, m)
        sig = np.interp(u, tr.u, tr.sig)                   # px arclength at spline parameter u
        q = tr.at(sig)
        Lv = R.chord_levels(q["a"], q["f"])                # (m, 6) rendered staircase levels
        Hk = R.BOX_H
        a_fit = (Lv * Hk * area).sum(1) / (Hk * Hk * area).sum()
        s_fit = np.sqrt(q["s"] ** 2 + blur_px ** 2)
        L = float(tr.Lpx)
        n_p = sp.n_ctrl_for_length(L, PROFILE_SPACING, minimum=2)
        prof, err = [], {}
        for name, vals in (("r", q["r"]), ("s", s_fit), ("a", a_fit)):
            cv = sp.fit_ctrl(vals[:, None], max(n_p, 2), smooth=1e-4, pin_ends=False)[:, 0]
            fit = sp.design(len(cv), m) @ cv
            err[f"fit_err_{name}"] = float(np.abs(fit - vals).max())
            prof.append(cv)
        prof[1] = np.maximum(prof[1], S_MIN)
        info = dict(true_vessel=int(vid), cls=e.cls, side=e.side, flow_rate=float(e.flow),
                    orientation="truth", tier="truth", depth_mean=float(np.mean(q["z"])),
                    s_below_floor=bool(np.min(s_fit) < S_MIN), **err)
        eid = net._eid
        net._eid += 1
        net.edges[eid] = Edge(nmap[e.u], nmap[e.v], ctrl, np.asarray(prof[0]), np.asarray(prof[1]),
                              np.asarray(prof[2]), info)
    net._touch()
    net.meta.update(source="vesselscene truth", k=float(k), origin=origin.tolist(),
                    optics=dict(halo_weight=float(optics.halo_h), halo_sigma=float(optics.halo_px)),
                    psf_tail_not_exported=dict(weight=float(optics.tail_h), sigma=float(optics.tail_px)),
                    note="one edge per vessel of the flow digraph (forks end vessels); a: linear-law centre OD "
                         "matching the chord-law staircase; s includes the image-formation blur; vesselmap has "
                         "one halo, so the renderer's intermediate PSF tail (if any) is not represented")
    return net


# ======================================================================== saving
def _versions() -> dict:
    import scipy
    out = dict(vesselscene=VERSION, anatomy=A.VERSION, python=platform.python_version(), numpy=np.__version__,
               scipy=scipy.__version__, torch=torch.__version__,
               cuda=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    try:
        from . import stats as st
        out["stats"] = st.STATS_VERSION
    except Exception:                                            # noqa: BLE001
        pass
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out["git_head"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                         timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "vesselscene"],
                               capture_output=True, text=True, timeout=10).stdout
        out["git_dirty_vesselscene"] = bool(dirty.strip())
    except Exception:                                            # noqa: BLE001
        out["git_head"] = None
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _graphml(g: VesselGraph, k: float, path: str, visibility: dict | None = None):
    """The flow digraph as GraphML: nodes (kind, x/y/depth in d_c, x_px/y_px,
    pressure), one edge per vessel u -> v (key = vid) with its class, side,
    flow, length and mean radius / depth / haematocrit, and its visibility
    tier per image kind.  Graph attributes: scalars of meta only."""
    import networkx as nx
    G = g.to_digraph()
    G.graph = {k_: v for k_, v in G.graph.items() if isinstance(v, (int, float, str, bool))}
    G.graph["k"] = float(k)
    for n, d in G.nodes(data=True):
        d["x_px"] = k * d["x"]
        d["y_px"] = k * d["y"]
        if d.get("pressure") is None:
            d["pressure"] = float("nan")
    for u, v, key, d in G.edges(keys=True, data=True):
        vis = (visibility or {}).get(key, {})
        for kind, rec in vis.items():
            d[f"tier_{kind}"] = rec["tier"]
            d[f"max_cnr_{kind}"] = float(rec["max_cnr"])
        e = g.vessels[key]
        d["stagnant"] = bool(e.info.get("stagnant", False))
        for name in ("tree", "layer", "order"):
            if isinstance(e.info.get(name), (int, float, str)):
                d[name] = e.info[name]
    nx.write_graphml(G, path)


def _have_vesselmap() -> bool:
    try:
        import vesselmap.network  # noqa: F401
        return True
    except ImportError:
        return False


def save_scene(scene: Scene, folder: str, overlay: bool = True, vesselmap_export: bool = True) -> dict:
    """Write a scene:
        still_average.tif                    float32 DN, NaN = no data (as mean_stabilized.tif)
        still_frame.tif                      uint16 DN: a raw frame (as frame_*.tif; saturated glare 4095)
        truth.json                           the VesselGraph (d_c; VesselGraph.load)
        digraph.graphml                      the flow digraph, one edge per vessel
        labels.npz                           the truth rasters, od_<kind> (float16 OD), valid_<kind>
        vessels.csv                          one row per vessel: class, nodes, flow, size, visibility
        crossings.json                       the in-frame crossings (px and d_c; which vessel is on top)
        scene.json                           every parameter, seed, version, timing, count
        overlay.png, overlay_id.png          the truth over the average still (by class / by vessel id)
        map.json                             vesselmap VesselNetwork export (to_vesselnetwork)
        junctions_<kind>.json                the image-junction truth of each image (junctions.load_junctions)
        junctions_average.png                the average's junctions over it (junctions.draw_junctions)
        junctions_all_<kind>.json            all junctions, hidden ones too (the complete truth)
        observable_<kind>.json               the observable truth of each image (truth.load_observable): runs,
                                             junctions, hidden junctions, the image graph (node-link)
        observable_<kind>.graphml            its image graph as GraphML (truth.save_graphml)
        observable_<kind>.png                the observable truth over the image (truth.draw_observable)
        complete_visibility_<kind>.npz       per-vessel per-sample visibility profiles (truth.load_profiles)
    labels.npz also holds the junction rasters (junctions.junction_rasters) as
    junction_<key> (the average's) and junction_<key>_frame, and the observable
    truth's (truth.observable_rasters) as observable_<key>_<kind>.
    Returns {name: path}."""
    import tifffile
    os.makedirs(folder, exist_ok=True)
    t0 = time.perf_counter()
    g = scene.graph
    k = float(scene.params["k"])
    H, W = scene.params["shape"]
    files = {}
    for kind, img in scene.images.items():
        p = os.path.join(folder, f"still_{kind}.tif")
        img = np.asarray(img, np.float32)
        if kind == "frame" and np.isfinite(img).all():
            img = np.clip(np.rint(img), 0, 65535).astype(np.uint16)      # a raw frame: integer DN, like frame_*.tif
        tifffile.imwrite(p, img)
        files[f"still_{kind}"] = p
    p = os.path.join(folder, "truth.json")
    g.save(p)
    files["truth"] = p
    p = os.path.join(folder, "digraph.graphml")
    _graphml(g, k, p, scene.visibility)
    files["digraph"] = p
    if scene.labels:
        lab = {key: v for key, v in scene.labels.items() if isinstance(v, np.ndarray)}
        for key in [k_ for k_ in lab if k_.split("_frame")[0] in ("depth", "radius", "dominant_od", "runner_up_od")
                                               or k_.startswith(("heat_", "junction_heat_", "junction_arm_dir"))]:
            if key in lab:
                lab[key] = lab[key].astype(np.float16)
        for kind, od in scene.od_clean.items():
            lab[f"od_{kind}"] = np.asarray(od, np.float16)
            lab[f"valid_{kind}"] = np.asarray(scene.formed[kind]["valid"], bool)
        lab["class_names"] = np.array(CLASSES)
        lab["side_names"] = np.array(SIDES)
        lab["tier_names"] = np.array(TIER_NAMES)
        p = os.path.join(folder, "labels.npz")
        np.savez_compressed(p, **lab)
        files["labels"] = p
    # per-vessel table
    p = os.path.join(folder, "vessels.csv")
    kinds = list(scene.images)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["vid", "u", "v", "cls", "side", "flow", "stagnant", "length_dc", "r_mean_dc", "z_mean_dc",
                     "z_min_dc", "z_max_dc", "hct_mean"] + [f"{c}_{kd}" for kd in kinds
                                                             for c in ("tier", "max_cnr", "visible_px", "visible_run_px",
                                                                       "frame_px")])
        for vid, e in g.vessels.items():
            smp = e.sample(1.0)
            row = [vid, e.u, e.v, e.cls, e.side, f"{e.flow:.6g}", int(bool(e.info.get("stagnant"))),
                   f"{smp['L']:.2f}", f"{smp['r'].mean():.3f}", f"{smp['z'].mean():.2f}", f"{smp['z'].min():.2f}",
                   f"{smp['z'].max():.2f}", f"{smp['h'].mean():.3f}"]
            for kd in kinds:
                rec = scene.visibility.get(vid, {}).get(kd, {})
                row += [rec.get("tier", ""), rec.get("max_cnr", ""), rec.get("visible_px", ""),
                        rec.get("visible_run_px", ""), rec.get("frame_px", "")]
            wr.writerow(row)
    files["vessels"] = p
    p = os.path.join(folder, "crossings.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(scene.crossings), fh)
    files["crossings"] = p
    junction_counts = {}
    if scene.junctions:
        from . import junctions as J
        for kind, js in scene.junctions.items():
            p = os.path.join(folder, f"junctions_{kind}.json")
            J.save_junctions(js, p)
            files[f"junctions_{kind}"] = p
            junction_counts[kind] = J.summary(js)
        if overlay and "average" in scene.junctions and "average" in scene.images:
            p = os.path.join(folder, "junctions_average.png")
            J.draw_junctions(scene.images["average"], scene.junctions["average"], p,
                             title=f"seed {scene.params.get('seed')} {scene.params.get('preset')} average: junctions")
            files["junctions_overlay"] = p
        for kind, js in (scene.junctions_all or {}).items():
            p = os.path.join(folder, f"junctions_all_{kind}.json")
            J.save_junctions(js, p)
            files[f"junctions_all_{kind}"] = p
    observable_counts = {}
    if scene.observable:
        from . import truth as TR
        for kind, obs in scene.observable.items():
            p = os.path.join(folder, f"observable_{kind}.json")
            TR.save_observable(obs, p)
            files[f"observable_{kind}"] = p
            p = os.path.join(folder, f"observable_{kind}.graphml")
            TR.save_graphml(obs, p)
            files[f"observable_graphml_{kind}"] = p
            observable_counts[kind] = obs.get("summary")
            if overlay and kind in scene.images:
                p = os.path.join(folder, f"observable_{kind}.png")
                TR.draw_observable(scene.images[kind], obs, p, prof=scene.vis_profiles.get(kind),
                                   title=f"seed {scene.params.get('seed')} {scene.params.get('preset')} {kind}")
                files[f"observable_overlay_{kind}"] = p
        for kind, prof in (scene.vis_profiles or {}).items():
            p = os.path.join(folder, f"complete_visibility_{kind}.npz")
            TR.save_profiles(prof, p)
            files[f"complete_visibility_{kind}"] = p
    if vesselmap_export and not _have_vesselmap():
        vesselmap_export = False                   # optional: needs the LIMBUS repository's vesselmap
    if vesselmap_export:
        ref = "average" if "average" in scene.images else kinds[0]
        net = to_vesselnetwork(g, k, (0.0, 0.0), (H, W), R.Optics(**scene.params["optics"]),
                               blur_px=scene.params["formed"][ref]["blur_px"], hct_mod=scene.fills.get(ref))
        p = os.path.join(folder, "map.json")
        net.save(p)
        files["map"] = p
    if overlay and scene.images:
        ref = "average" if "average" in scene.images else kinds[0]
        for by in ("class", "id"):
            p = os.path.join(folder, "overlay.png" if by == "class" else "overlay_id.png")
            draw_overlay(scene, p, kind=ref, by=by)
            files[f"overlay_{by}"] = p
    # summary
    vis_counts = {kd: {} for kd in kinds}
    for vid, rec in scene.visibility.items():
        for kd in kinds:
            t = rec.get(kd, {}).get("tier", "out_of_frame")
            vis_counts[kd][t] = vis_counts[kd].get(t, 0) + 1
    cls_in = {}
    for vid, rec in scene.visibility.items():
        if any(r.get("tier") != "out_of_frame" for r in rec.values()):
            c = g.vessels[vid].cls
            cls_in[c] = cls_in.get(c, 0) + 1
    fill_states = {}
    for kd, fl in scene.fills.items():
        st_ = {}
        for f in fl.values():
            st_[f.state] = st_.get(f.state, 0) + 1
        fill_states[kd] = st_
    meta = dict(scene.params)
    meta.update(versions=_versions(), files={n: os.path.basename(v) for n, v in files.items()},
                counts=dict(vessels=len(g.vessels), nodes=len(g.nodes), vessels_in_frame_by_class=cls_in,
                            visibility=vis_counts, crossings_in_frame=len(scene.crossings),
                            nodes_in_frame=scene.labels.get("node_counts") if scene.labels else None,
                            filling_states=fill_states, junctions=junction_counts or None,
                            observable=observable_counts or None),
                anatomy_meta={key: v for key, v in g.meta.items() if key not in ("params",)},
                label_rules=dict(od_min=OD_MIN, node_sigma_px=NODE_SIGMA, cnr_required=CNR_REQUIRED,
                                 cnr_visible=CNR_VISIBLE, lambda_px=LAMBDA_PX, tier_names=list(TIER_NAMES)))
    if scene.junctions:
        from . import junctions as J
        meta["junction_rules"] = J.rules()
    if scene.observable:
        meta["truth_rules"] = next(iter(scene.observable.values()))["rules"]
    meta["timings"] = dict(scene.timings, save=round(time.perf_counter() - t0, 3))
    p = os.path.join(folder, "scene.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(meta), fh, indent=1)
    files["scene"] = p
    return files


# ======================================================================== overlay
CLASS_COLORS = {                                    # BGR, for cv2
    "capillary": (80, 200, 255), "arteriole": (60, 60, 235), "venule": (235, 120, 40),
    "episcleral_artery": (150, 20, 170), "episcleral_vein": (140, 60, 20), "anastomosis": (40, 200, 60),
    "perforator": (160, 160, 160)}


def flat_view(img: np.ndarray, vmin: float = -0.35, vmax: float = 0.08) -> np.ndarray:
    """uint8 grey view of F = ln I - background (stats.flatten) on vmin..vmax
    Np, NaN white: the common display of the real and synthetic stills."""
    from . import stats as st
    I = np.asarray(img, np.float64)
    valid = np.isfinite(I) & (I > 0)
    F = st.flatten(I, valid)[0]
    g8 = np.clip((F - vmin) / (vmax - vmin), 0, 1) * 255
    g8[~valid] = 255
    return g8.astype(np.uint8)


def _id_color(vid: int):
    h = (vid * 0.6180339887) % 1.0
    import colorsys
    r, g_, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(255 * b), int(255 * g_), int(255 * r)


def draw_overlay(scene: Scene, path: str | None = None, kind: str = "average", by: str = "class",
                 crop=None, scale: float = 1.0) -> np.ndarray:
    """The truth over the still: every vessel's centreline coloured by class
    (or by id), 1-3 px by radius, dashed where it is invisible in `kind`;
    every in-frame crossing marked by a ring, with the vessel ON TOP redrawn
    across it in white over its own colour; forks (small filled dots) and
    confluences (small squares).  crop = (y0, x0, h, w) px; scale enlarges.
    Returns the BGR image (and writes it to path)."""
    import cv2
    g = scene.graph
    k = float(scene.params["k"])
    H, W = scene.params["shape"]
    base = cv2.cvtColor(flat_view(scene.images[kind]), cv2.COLOR_GRAY2BGR)
    y0, x0, h, w = crop if crop is not None else (0, 0, H, W)
    base = base[y0:y0 + h, x0:x0 + w]
    f = float(scale)
    if f != 1.0:
        base = cv2.resize(base, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    ov = base.copy()
    sh = 4                                                   # cv2 fixed-point bits
    off = np.array([x0, y0], float)

    def P(xy_px):
        return np.rint(((np.asarray(xy_px) - off) * f + (f - 1) / 2) * (1 << sh)).astype(np.int32)
    smp_cache = {}                                           # one sampling per vessel (crossings re-use it)

    def samples(vid):
        if vid not in smp_cache:
            smp_cache[vid] = g.vessels[vid].sample(0.5 / k)
        return smp_cache[vid]
    order = sorted(g.vessels, key=lambda v: -float(np.mean(g.vessels[v].z_ctrl)))   # deepest first
    for vid in order:
        e = g.vessels[vid]
        c = k * np.asarray(e.ctrl)
        if c[:, 0].max() < x0 - 5 or c[:, 0].min() > x0 + w + 5 or c[:, 1].max() < y0 - 5 or \
                c[:, 1].min() > y0 + h + 5:
            continue
        smp = samples(vid)
        pts = P(k * smp["xy"][::2])
        col = CLASS_COLORS.get(e.cls, (200, 200, 200)) if by == "class" else _id_color(vid)
        th = int(np.clip(round(0.5 * k * float(np.median(smp["r"])) * f / 2), 1, 4))
        tier = scene.visibility.get(vid, {}).get(kind, {}).get("tier", "")
        if tier == "invisible":
            for i in range(0, len(pts) - 1, 8):
                cv2.polylines(ov, [pts[i:i + 5]], False, col, th, cv2.LINE_AA, shift=sh)
        else:
            cv2.polylines(ov, [pts], False, col, th, cv2.LINE_AA, shift=sh)
    out = cv2.addWeighted(ov, 0.8, base, 0.2, 0)
    for cr in scene.crossings:
        x, y = cr["x_px"], cr["y_px"]
        if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
            continue
        top = cr["above"]
        e = g.vessels[top]
        smp = samples(top)
        s_here = cr["s"][cr["vessels"].index(top)]
        m = np.abs(smp["s"] - s_here) <= 6.0 / k
        if m.sum() >= 2:
            pts = P(k * smp["xy"][m])
            col = CLASS_COLORS.get(e.cls, (200, 200, 200)) if by == "class" else _id_color(top)
            cv2.polylines(out, [pts], False, (255, 255, 255), 4, cv2.LINE_AA, shift=sh)
            cv2.polylines(out, [pts], False, col, 2, cv2.LINE_AA, shift=sh)
        c = P([x, y])
        cv2.circle(out, (int(c[0]), int(c[1])), int((5 * f) * (1 << sh)), (0, 0, 0), 1, cv2.LINE_AA, shift=sh)
    deg = g.degrees()
    for nid, n in g.nodes.items():
        x, y = k * n.xy
        if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
            continue
        i, o = deg[nid]
        c = P([x, y])
        cc = (int(c[0]), int(c[1]))
        if i == 1 and o >= 2:
            cv2.circle(out, cc, int(2 * f * (1 << sh)), (0, 0, 0), -1, cv2.LINE_AA, shift=sh)
        elif o == 1 and i >= 2:
            d = int(2 * f * (1 << sh))
            cv2.rectangle(out, (cc[0] - d, cc[1] - d), (cc[0] + d, cc[1] + d), (0, 0, 0), -1, cv2.LINE_AA,
                          shift=sh)
    if by == "class":
        yy = 18
        for name, col in CLASS_COLORS.items():
            cv2.rectangle(out, (8, yy - 11), (22, yy + 1), col, -1)
            cv2.putText(out, name, (28, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, name, (28, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            yy += 17
        for txt in ("ring: crossing; white-edged stroke = vessel on top", "dot: fork   square: confluence",
                    "dashed: invisible (CNR < 1.5)"):
            cv2.putText(out, txt, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, txt, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            yy += 17
    if path:
        cv2.imwrite(path, out)
    return out
