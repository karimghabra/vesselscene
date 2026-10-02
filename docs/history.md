# vesselscene: development log

The realism rounds v0 to v2 (2026-09-29 to 2026-10-01): what was measured against the real stills, what was changed and why, and what remained. Newest first. Paths under `_scratch/` refer to working files that are not part of the repository; the real stills belong to the LIMBUS project and are not distributed.

## v2 fix pass (2026-10-01, after the v2 reviews; the current package)

After the v2 integration, a correctness review and a fresh-eyes realism review (`_scratch/v2/review_correct/`,
`_scratch/v2/review/`). This pass reproduced and fixed the six confirmed defects, then worked down the realism
tells as far as a measured, physically grounded change allowed. Change log with every step, the rejected
experiments and the evidence: `_scratch/v2/final/changelog.md`.

**Defects fixed** (each reproduced first; regression tests in brackets):

| # | defect | fix |
|---|---|---|
| 1 | junction truth: neighbouring junctions claimed one stretch of a vessel, so arms went missing (a crossing next to a fork read as a T) or sat at one sample; 1773 consistency problems in one saved average | claims shared out along each vessel, sides closed only at this junction's own nodes, `linked` arms where claims meet, a bounded join of strongly overlapping junctions; `junctions.check_junctions` (Image-junction section) [`test_junctions.py::test_a_crossing_beside_a_fork_keeps_the_parent_continuous`, `::test_scene_junctions_on_a_small_scene`] |
| 2 | renderer: two vessels sharing a node were one blood volume anywhere, so a tributary crossing its own vein far from the confluence added to it (0.28 Np black block in pathologic 2) | a shared node counts only within that node's junction, in the renderer and the oracle [`test_render.py::test_composited_crossings_against_oracle[cross_shared_node]`] |
| 3 | anatomy: the depth fit lost a self-overlap bump on a long vessel; healthy 6 failed `check(overlap=True)` | the fitted profile is checked and refitted with finer control points, else the other direction, else removal [`test_anatomy.py::test_self_overlap_bump_survives_the_depth_fit`] |
| 4 | frames: one red-cell granularity field for every vessel (two crossing vessels' grain correlated at 0.97) | each vessel reads the field at its own offset [`test_imaging.py::test_rbc_mottle_crossing_vessels_have_independent_cells`] |
| 5 | the CLI's `junctions` overwrote the exact truth with an approximation and left the rasters stale | writes `junctions_<kind>_recomputed.json`; `--force` replaces the file, rasters and counts [`test_scene.py::test_cli_junctions_does_not_overwrite_the_exact_truth`] |
| 6 | `graph.crossings` reported same-depth sibling overlaps inside a node's reach as crossings | skipped [`test_graph.py::test_crossings_skip_one_volume_overlaps_at_a_shared_node`] |

Also from the junction audit: `type_visible` from the seen lines, arm `contrast` the vessel's own, a `through`
relation [`test_junctions.py::test_type_visible_needs_three_seen_lines`]; dead oracle code removed.

**Realism changes kept** (besides defects 2 and 4, which change images): the camera gain is drawn per scene,
12-20.4 dB (`scene.GAIN_DB`; every frame had the noise of the noisiest real burst), and capillary columns and
gaps are gamma of shape 3 instead of 5 (`col_shape`). Tried and rejected (6 seeds, `changelog.md`): a sparser
deep network (plexus spacing 80 -> 120 d_c, 1.6 -> 1.0 artery trunks; the 16-24 px profiles are 2x the real
share, but the change moved four other statistics the wrong way), a union region capped where the lumens stop
overlapping in 3-D, and compositing node-sharing vessels stacked apart in depth (broke
`test_dive_right_after_a_fork`).

**Results** (the final set `_scratch/v2/final_scenes/`, 12 healthy + 3 pathologic + 3 random, same seeds as v1
and v2; `_scratch/v2/final/final_scorecard.txt`; healthy means, z against the dense site for still keys and
against the 24 pooled real frames for frame keys):

| comparison (scalars passed (far); distributions + spectra) | v1 | v2 integration | v2 final |
|---|---|---|---|
| healthy averages vs the dense site (4 stills) | 50/102 (5); 8/11 | 59/102 (9); 9/11 | 57/102 (9); 7/11 |
| healthy + pathologic averages vs the 9 real stills | 91/102 (1); 10/11 | 85/102 (2); 10/11 | 84/102 (2); 10/11 |
| healthy frames vs the 9 real raw frames | 76/102 (4); 9/11 | 67/102 (3); 9/11 | 80/102 (2); 9/11 |
| healthy + pathologic frames vs the 9 real raw frames | 73/102 (4); 9/11 | 60/102 (4); 7/11 | 77/102 (2); 9/11 |

| item | statistic (real) | v1 | v2 integration | v2 final |
|---|---|---|---|---|
| 1 frames | frame FWHM p10, px (6.77) | 9.81 (+4.1) | 9.23 (+3.3) | 8.30 (+2.1) |
| | frame dark knots per 100 lambda (0.88) | 1.64 (+2.0) | 1.83 (+2.5) | 1.07 (+0.5) |
| | thin-vessel CNR in frames (5.92) | 4.46 (-1.0) | 4.28 (-1.1) | 5.47 (-0.3) |
| | frame long paths (57.4) | 44.8 (-0.8) | 47.3 (-0.6) | 51.3 (-0.4) |
| | `bead_len`, px (3.53) | 7.55 (+11.0) | 3.71 (+0.5) | 3.79 (+0.7) |
| | `faint_absent` (0.050) | 0.049 | 0.019 (-0.6) | 0.037 (-0.3) |
| | frames sharper than their average (0.75) | 0.92 | 1.00 | 0.92 |
| 2 knots | still dark knots per 100 lambda (0.47) | 0.83 (+1.4) | 0.60 (+0.5) | 0.56 (+0.4) |
| 3 anatomy | `xing_shallow` (0.063) | 0.095 (+2.8) | 0.079 (+1.3) | 0.099 (+3.1) |
| | `pair_frac` (0.261) | 0.206 (-7.7) | 0.174 (-12.2) | 0.171 (-12.6) |
| | `along_cv_fwhm` (0.147) | 0.117 (-2.1) | 0.117 (-2.1) | 0.115 (-2.2) |
| | `branch_rate` (7.25) | 6.06 (-2.6) | 5.83 (-3.1) | 5.73 (-3.3) |
| 4 vein | widest vein, Np (-0.471) | -0.435 (+1.5) | -0.477 (-0.2) | -0.474 (-0.1) |
| | `vein_fwhm`, px (45.8) | 39.0 (-3.6) | 53.2 (+3.9) | 53.7 (+4.2) |

The frames gained most (lower noise in most scenes, independent cells at crossings). The stills barely moved;
two rows got worse: the averages' pixel noise and high-frequency spectrum now sit below the dense site (whose
bursts ran at the top gain; against the 9 pooled stills the noise passes, z -0.3), and `xing_shallow` (+1.3 ->
+3.1: tributaries that ride along their vein at another depth beyond the confluence's junction are now
composited, and read as shallow crossings, where the v2 rule summed them).

**Junction truth, 12 healthy scenes** (`_scratch/v2/final/jstats.txt`; v2 integration -> final, per image):
`check_junctions` problems 1689 -> 0 (average) and 1765 -> 0 (frame); shown junctions with fewer than 3 seen
lines, 'none' with 3 or more, pseudo-Ts with 4 or more, forks shown with fewer than 3: 0; invisible arms
reporting a contrast >= 0.08 Np 43 -> 0 (average). Junctions 682 -> 666 per average (494 shown), 730 -> 715 per
frame (421 shown); clean 86 -> 99 per average; ambiguous among the shown 0.58 -> 0.57 (frames 0.71 -> 0.69);
radius p50 / p90 / max 14.9 / 36 / 107 px; 25 % of the arms are linked to a neighbouring junction, 37 %
crowded; shown junctions whose centre lies inside another shown one's circle 26 % -> 21 % (the rest are linked
neighbours). A straightness rule pairs two different vessels at 3 % of the crossings with 4 seen lines and
two branches of one fork at 21 % of the forks with 3 lines ('through' 2 %). 9-17 s per healthy image.

**Still different** (`_scratch/v2/final/blocks_v2_vs_final.png`, `final_*.png`):

- Tell 1, hard-edged black blocks: fixed in pathologic 2 (the triangle: a tributary crossing its vein 57 d_c
  from their confluence). Not fixed: the black D's of healthy 1 (745 on 744) and healthy 3 (49 on 50) and the
  pathologic 0 / 1 polygons. Mechanism (healthy 3): tributary 49 runs 12 d_c above its vein 50 inside the
  vein's projected lumen within 92 px of their confluence; the junction's union region there is 116-216 px long
  (the angle formula at a small angle), its overlap polygons join only segments within r_a + r_b in depth, so
  the two add, and the compositing keeps node-sharing vessels in one volume inside the junction: summed, 0.95
  Np against the vein's 0.60. Making them composite broke the diving-child-at-a-fork case; the fix needs the
  union and the compositing to agree on stacked lumens inside a junction (or the anatomy to keep tributaries
  out of their vein's projected lumen at another depth). The oracle also differs from the closed form by up to
  9.8 % of the peak at that rider (the two decide 'union' differently there: 2-D polygons within the stub
  region against a per-pixel 3-D test): a renderer-accuracy defect found in this pass, not fixed.
- Tell 2, too few thin dark lines: a third fewer thin (4-10 px) profiles and 2x fewer dark ones; the 16-24 px
  blurred profiles are 27 % of all in healthy 0 against 12-16 % real (episcleral vessels make 38-40 % of the synthetic
  profiles). Not the deep network's density alone (rejected experiment above).
- Tell 3, the widest vein as a ribbon (flat floor, 9-10 px walls against 26-32 px real): the real widest veins
  look like scatter-blurred deep trunks (a 46 px FWHM with 26-32 px edges means a blur sigma of 16-19 px).
- Tells 4, 5, 7, 8, 11: uniform wires (`along_cv_fwhm` -2.2), regular capillary dashes (the dash probe's
  gap-length CV did not move measurably with `col_shape` 3: 20-30 paths per image), missing pairs (-12.6),
  kinked mesh polylines, lens-shaped spindles at shallow crossings (`final_crossings.png`).

## Realism: v2 (2026-10-01; the integrated round)

The round's five items, agreed, in priority order: (1) single
frames: a real raw frame is crisp, with beaded red-cell clumps and fine
vessels clearly visible, the v1 frame soft, its capillaries broken into
dashes; (2) dark knots where vessels cross (about 1.8x the real rate; black
blocks across pathologic veins; long black spindles at shallow crossings);
(3) anatomy detail: bundles too regular, medium vessels uniform wires, too
few fine wiggly vessels and loops, too few long thin grey lines, elbows; (4)
the widest vein's black level (too black in some scenes, missing in others)
and ribs; (5) a new image-junction ground truth for intersection
identification. Statistics for each (`tells.py` v2 keys, `T.PRIORITY[key]`)
were built first on the 4 dense stills and on 24 raw frames of three bursts
(8 each of 15-22-26, 15-31-37, 15-50-52) carried onto their averages by the
stabilization fields (`data/real_tells.json`); then four builders worked in
parallel (single frames: `imaging.py` and the frame step of `scene.py`;
anatomy: `anatomy.py`; renderer: `render.py`; junction truth: the new
`junctions.py`), followed by an integration and a bounded calibration pass.

Scenes: `_scratch/v2/scenes/` (healthy 0-11, pathologic 0-2, random 0-2, as
written by `save_scene`, plus `measure.json`); the integration harness,
figures, tables and change log: `_scratch/v2/integ/` (`run.py`, `figs.py`,
`changelog.md`); the builders' studies: `_scratch/v2/frames/`,
`_scratch/v2/anatomy/`, `_scratch/v2/render/`, `_scratch/v2/junctions/`; the
v1 baseline of the new statistics: `_scratch/v2/baseline.md`.

**What changed, by module** (details and evidence in the modules' comments
and the builders' folders):

| item | change | evidence |
|---|---|---|
| 1 frames | red cells move at the conjunctiva's measured speed, about 0.5 mm/s in every calibre (`imaging.filling`: `v_cap` 85 d_c/s, `speed_exp` 0.1, `speed_sd` 0.3; v1 scaled the speed with flow / area, 3-18 mm/s in medium venules smeared the cells over 20-150 px in one 12.8 ms exposure) | Brennan 2021 (0.53 +- 0.15 mm/s), Moka & Koutsiaris 2019, Hwang 2021 |
| 1 frames | red-cell clumps and plasma-rich stretches in the continuous vessels (`grain_kind = 'clump'`: 1.5 d_c long, soft ends, amplitude 0.09 r / r_cont falling as 2 / r beyond 2 d_c, so wide veins get no bamboo bands) | aggregation in low-order drainage vessels (Meyer 2018); real frames' along-vessel spectrum of frame - average |
| 1 frames | the cells' granularity across the lumens (`imaging.rbc_mottle`, `scene.frame_mottle`, `mottle` 0.6): each column's cell count fluctuates (Poisson-like, sqrt of its absorbance), through the chord law on the whole column (crossings gain no dark beads), blurred with each vessel, smeared by its cells' speed; frame only | structure factor 0.36 (a dense suspension) |
| 1 frames | capillary columns and gaps gamma-distributed (shape 5; v1 exponential), soft column ends 0.4 d_c | calibrated on the dashed-line statistics |
| 2 knots | crossings composited by depth inside the renderer, exact against the oracle (`render.composite_od`, Rendering above), with the chord law's `f_spectral` 0.05 -> 0.35, `mu` 0.135 -> 0.185, `l_bypass` 80 -> 100 | real crossings 0.87 of additive (42 crossings of the dense stills), v1 1.01 |
| 2, 3, 4 anatomy | a long vessel meanders as one course through its junctions; a collecting vein keeps its course, the tributary takes the angle; wide vessels bend no tighter than 6 diameters (the elbows); long vessels undulate in depth (SD 4 d_c over 60-150 d_c, inside the depth allocation); venules vary in calibre more than arterioles (0.20 / 0.15 over about 2.5 diameters); bundles of 2-3 with unequal calibres, drifting spacing, members joining and leaving at different places; thoroughfare channels (the shortest capillary routes between arteriole and venule tips widened to 1.3-2 d_c, higher haematocrit, at most 25 % of the mesh: the visible fine network and its loops); gently S-bent mesh capillaries; the collecting vein through the central 60 % of the frame, gently arcing, at mid-depth, its flattening drawn once per vessel tree; courses, companions and bundles keep off long shallow overlaps with dark or deep vessels | `_scratch/v2/anatomy/` |
| 4 ribs | none left: v1's lumen-edge piece rule already removes them (new oracle test: r 20 px on a smooth bend at OD 1.2 within 0.01 %; without the rule 4.3 % and 0.012 Np ribs) | `test_render.py` |
| 5 junctions | `junctions.py` (section above); wired into `make_scene` / `save_scene`, the CLI and `views.load_scene` | `test_junctions.py`, `test_scene.py` |

**Integration and calibration pass** (`_scratch/v2/integ/changelog.md`; healthy seeds 0-5 per step, renders
on the same anatomies unless the step changes the anatomy). The integrated set came out with veins too light
(widest vein -0.39 Np against the dense site's -0.47; `dark128_0.45` -8.9 SD; `F_sd` -2.5), because the
anatomy's venous flattening had been set against v1's chord law. Three changes were kept:

| step | change | effect (6 seeds, z against the dense site) |
|---|---|---|
| veins | `AnatomyParams.vein_aspect` 0.4-0.65 -> 0.7-0.9 (round veins overshoot every darkness statistic) | widest vein -0.365 -> -0.472 Np (+4.3 -> -0.0), `F_sd` -3.2 -> -0.0, `dark128_0.45` -10.0 -> -2.0; darkest long vessel per scene -0.26 .. -0.33 -> -0.35 .. -0.38 Np |
| halo | `CALIBRATED_OPTICS.halo_h` 0.45 -> 0.375 (0.3 went further on the frames but lost 6 scorecard rows) | still FWHM p10 7.47 -> 6.93 px (real 6.81), `n_long` 54 -> 62 (68), frame FWHM p10 9.8 -> 9.2 px (+4.1 -> +3.3 SD against the 24 real frames) |
| bulges | `AnatomyParams.r_var_clip` 2: the calibre noise is soft-limited to 2 SD (it reached 3-3.5 SD: 2-2.8x fusiform 'spindles') | vessels wider than 1.5x their ends 27 -> 0.5 per frame; `xing_shallow` +5.9 -> +1.7 (the spindles read as shallow crossings), `calibre_change` +1.9 -> +1.0, `along_cv_fwhm` -1.2 -> -1.7 |

A diagnostic without the cells' granularity showed it makes part of the frames' excess dark knots (1.59 -> 1.31
per 100 lambda on 6 seeds; real 0.88); it was kept (it is the cells' physics and makes the beads).

**Results** (the final set: 12 healthy, 3 pathologic, 3 random scenes; `_scratch/v2/scenes/table.md`; v1 = the
committed `_scratch/v1/fin2` scenes measured with the same statistics; healthy scenes; z against the dense site
for stills and against the 24 pooled real frames for frames; separates = |z| > 3 and the ranges apart).

| comparison (scalars pass (far, \|z\| > 4); distributions + spectra) | v1 | v2 |
|---|---|---|
| healthy averages vs the dense site (4 stills) | 50/102 (5); 8/11 | 59/102 (9); 9/11 |
| healthy + pathologic averages vs the 9 real stills | 91/102 (1); 10/11 | 85/102 (2); 10/11 |
| healthy frames vs the 9 real raw frames (v0's reference) | 76/102 (4); 9/11 | 67/102 (3); 9/11 |
| healthy + pathologic frames vs the 9 real raw frames | 73/102 (4); 9/11 | 60/102 (4); 7/11 |

The frames' suite lost rows to densities per lambda^2 (junctions, crossings, loops, endpoints): the v2 web
has more structure, and the frames' lambda is still too large (below), which inflates them.

*1. Single frames* (real: 24 frames; the burst means are 15-22-26 / 15-31-37 / 15-50-52):

| statistic | real 24 frames (burst means) | v1 | v2 |
|---|---|---|---|
| red-cell bead length `bead_len` (px) | 3.53 (3.19 / 3.68 / 3.72) | 7.55 (+11.0, separates) | 3.71 (+0.5) |
| bead depth `bead_med` / `bead_med_rel` | 0.0058 / 0.038 | 0.0045 / 0.024 | 0.0057 / 0.032 |
| frame - average grain, fine / coarse | 0.0026 / 0.0054 | 0.0012 / 0.0041 | 0.0022 / 0.0051 |
| bands across wide lumens `wide_band_frame` | 0.021 | 0.008 | 0.012 |
| dashed faint lines `faint_dash`; gaps per 100 lambda | 0.035 (0.063 / 0.007 / 0.034); 3.3 | 0.070; 5.6 | 0.034; 2.8 |
| faint line absent `faint_absent` | 0.050 (0.105 / 0.009 / 0.037) | 0.049 | 0.019 (too continuous) |
| frames sharper than their average | 18 of 24 (5/8, 5/8, 8/8) | 11 of 12 | 12 of 12 |
| average's extra blur `sigma_extra_med` (px) | 0.59 (0.48 / 0.34 / 0.93) | 0.90 | 0.92 |
| thin lines' contrast / CNR in the frame | 0.089 / 5.9 | 0.094 / 4.5 | 0.090 / 4.3 |
| frame's thinnest lines `frame:fwhm_p10` (px) | 6.77 (6.10 / 6.76 / 7.45) | 9.81 (+4.1, separates) | 9.23 (+3.3, separates) |
| frame's lambda (px) | 13.2 | 16.2 (+2.4) | 16.8 (+2.9) |
| frame's long lines `frame:n_long` | 57.4 | 44.8 | 47.3 |
| frame's dark knots per 100 lambda | 0.88 (1.17 / 0.57 / 0.91) | 1.64 (+2.0) | 1.83 (+2.5) |

*2. Dark knots at crossings* (the dense stills; crossing additivity: how much darker a crossing is than its
darker arm, per the lighter arm's contrast, on crossings found by `_scratch/v2/render/crossprof.py`):

| statistic | real | v1 | v2 |
|---|---|---|---|
| dark knots per 100 lambda | 0.47 [0.29, 0.84] | 0.83 (+1.4) | 0.60 (+0.5) |
| darkest 10 % / median knot (Np) | -0.085 / -0.023 | -0.122 / -0.041 | -0.089 / -0.033 |
| share of crossings that are dark knots | 0.28 | 0.42 | 0.34 |
| crossing additivity (median) | 0.87 (42 crossings) | 1.01 | 0.87 (91) |
| pathologic: dark knots per 100 lambda; darkest 10 %; widest vein (Np) | (healthy 0.47; -0.085; -0.47) | 1.61; -0.188; -0.567 | 0.47; -0.079; -0.455 |
| pathologic frames: dark knots per 100 lambda | (healthy 0.88) | 3.73 | 2.72 |

*3. Anatomy detail* (dense site):

| statistic | real | v1 | v2 |
|---|---|---|---|
| elbows: kinked wide path `wide_kink` | 0.0021 | 0.025 (+5.3) | 0.0041 (+0.5) |
| long thin grey lines `n_long` | 68.4 | 48.1 (-1.2) | 58.6 (-0.6) |
| faint loops per lambda^2 `faint_loop_density_lam` | 0.0064 | 0.0038 (-4.1) | 0.0053 (-1.7) |
| bundles `bundle_frac` (3+ parallel lines) | 0.014 | 0.030 (+4.9) | 0.011 (-0.9) |
| parallel neighbours `par_excess` | 1.38 | 1.09 (-3.4) | 1.39 (+0.1) |
| pairs `pair_frac` / `pair_run5` | 0.261 / 0.128 | 0.206 (-7.7) / 0.096 | 0.174 (-12.2) / 0.078 (-3.5) |
| shallow crossings `xing_shallow` | 0.063 | 0.095 (+2.8) | 0.079 (+1.3) |
| uniform wires: along-vessel CV of FWHM / contrast | 0.147 / 0.222 | 0.117 (-2.1) / 0.210 | 0.117 (-2.1) / 0.196 (-1.1) |
| branch points per 100 lambda `branch_rate` | 7.25 | 6.06 (-2.6) | 5.83 (-3.1) |
| thinnest lines `fwhm_p10` (px) | 6.81 | 7.33 | 7.03 |

*4. The widest vein* (dense site; per-scene ranges in brackets):

| statistic | real | v1 | v2 |
|---|---|---|---|
| widest vein's level `vein_depth` (Np) | -0.471 [-0.489, -0.435] | -0.435 [-0.533, -0.325] | -0.477 [-0.548, -0.354] |
| darkest long vessel (Np); scenes with none darker than -0.33 | -0.380 [-0.447, -0.333]; 0 of 4 | -0.348 [-0.420, -0.278]; 3 of 12 | -0.366 [-0.392, -0.308]; 1 of 12 |
| area darker than -0.45 Np `dark128_0.45` | 0.017 | 0.0081 (-7.1) | 0.017 (+0.0) |
| overall contrast `F_sd`; wide vessels' contrast `c_wide` | 0.080; 0.286 | 0.075; 0.313 | 0.082; 0.312 |
| widest dark vessels `darkw_p99` (px) | 62 | 46.7 (-5.1, separates) | 57.5 (-1.5) |
| widest vein's FWHM `vein_fwhm` (px) | 45.8 | 39.0 (-3.6) | 53.2 (+3.9; median 46: seed 1's vein runs beside a blurred deep vessel and their profiles merge, 136 px) |
| ribs `wide_rib` | 0.0074 | 0.0031 | 0.0069 |

*5. Image-junction truth* (per frame, means over the 12 healthy scenes, `_scratch/v2/scenes/junction_summary.md`):

| | average | frame |
|---|---|---|
| junctions in frame (shown: `type_visible` not none) | 682 (541) | 730 (466) |
| truth type: crossing / compound / confluence / bifurcation / touching / pseudo-T / anastomosis | 272 / 272 / 56 / 52 / 19 / 8 / 3 | 278 / 283 / 53 / 49 / 25 / 39 / 3 |
| junctions with 1 / 2 / 3 / 4 / 5+ visible arms | 16 / 110 / 137 / 236 / 184 | 49 / 193 / 175 / 179 / 134 |
| ambiguous (all / shown); main reasons among the shown | 0.65 / 0.58; merged arms, invisible arm, misleading geometry | 0.80 / 0.71; invisible arm, merged arms |
| clean (one event, every arm visible, not ambiguous, no arm crowded) | 86 | 54 |
| a straightness rule pairs two vessels' arms at crossings with 4 visible arms / pairs the two branches of a fork with 3 | 4 % / 22 % | 3 % / 21 % |
| junction radius p50 / p90; arms cut short by the next junction | 14.6 / 36.0 px; 45 % | 14.3 / 34.1 px; 46 % |

Crossing angles in the truth: median 60 deg; per frame 37 below 15 deg and 103 at 15-30 deg. Pathologic
averages: 376 junctions (307 shown), ambiguous 0.71; random: 632 (385), 0.71.

Figures (`_scratch/v2/scenes/figs/`; one grey scale, F on -0.35..+0.08 Np): `v2_side_by_side.png` (healthy
seed 0 against 15-50-52 and 13-51-26: full frames, two 1:1 crops, a 2x zoom), `v2_frame_vs_average.png`
(two synthetic frame / average pairs next to raw frames of 15-50-52 and 15-31-37 carried onto their averages,
1:1 and 3x), `v2_crossings.png` (crossings at 2x: the darkest and the median knots, real dense stills and v2
averages, same detector), `v2_pathologic.png`, `v2_before_after.png` (v1 and v2 of seeds 0-2),
`v2_junctions.png` (the junction truth over the whole average and a 3x zoom next to the plain image).

What changed visibly: the widest veins are dark to black with a round profile (every healthy scene has one,
-0.35 .. -0.55 Np), no black blocks across pathologic veins; dark knots at crossings are about as frequent and
as dark as in the stills; collecting veins curve smoothly through confluences; more long thin lines and a
visible fine network with loops; bundles are irregular doubled and tripled lines; frames show red-cell beads
along the medium vessels and gently dashed capillaries.

**What still differs** (specific):

- **Thin dark lines are too few.** Profiles 4-10 px wide and darker than 0.084 Np: 160-210 per average still
  against 358-571 in the dense stills (seeds 0-2); their median contrast 0.059 against 0.065-0.069. In frames
  this keeps the thinnest lines too wide (`frame:fwhm_p10` +3.3 SD) and lambda too large (+2.9); in stills it
  shows as fewer branch points (`branch_rate` -3.1 SD) and in the dark / bright ridge ratio at 1-8 px (the
  suite's `ps:*:dark_bright`, -7 to -12 SD against the dense site). The halo explains part (0.3 would close
  more but costs the profile-shape and background statistics); the rest is probably the thin vessels'
  haematocrit and depth spread (a dark tail of small, full-haematocrit vessels near focus), not yet modelled.
- **Pairs**: `pair_frac` -12 SD (0.17 against 0.26). The v2 web adds unpaired thin lines (channels, more long
  vessels); the companion gap was measured on the stills and left.
- **Frames' dark knots**: 1.8 per 100 lambda against 0.9 (stills: 0.60 against 0.47); about a sixth of the
  excess is the cells' granularity, the rest not identified (the frames' noise makes 3-4x more detected
  crossings than the averages, in real and synthetic frames alike).
- **Capillaries a little too continuous in frames** (`faint_absent` 1.9 % against 3.7-5 %), and their dashes
  more regular than the real ones (gamma columns of shape 5, calibrated, not measured); beads about 20 %
  weaker than 15-50-52's strongest clumps (calibrated on the pooled frames).
- **Uniform wires**: the along-vessel FWHM CV stays at -2.1 SD (0.117 against 0.147); the bulge limit took a
  little of it back: real calibre changes are frequent but not fusiform.
- **Pathologic**: hooked pigtail fragments of tortuous small vessels where they dive; the widest pathologic
  veins are wide (78 px) and their blurred deep neighbours make long bright bands in the flattened view.
- **Junction definitions are open** (Image-junction ground truth, above): with 600-1300 crossings per frame,
  45 % of arms are cut short by the next junction and 58-71 % of the shown junctions are ambiguous.
- **Evidence is thin**: each calibration step on 6 seeds, the dense site 4 stills of 2 sites, 24 real frames
  of 3 bursts; z values against the dense site overstate separation.

## Realism: v1 fix pass (2026-09-30, after the v1 reviews)

Two reviews of v1 (a correctness review: 9 confirmed defects; a fresh-eyes
realism review: tells T1-T12) were worked through. Scenes:
`_scratch/v1/fin2/` (healthy seeds 0-11, pathologic 0-2, random 0-2, as
written by `save_scene`, plus `measure.json`); figures and scorecards:
`_scratch/v1/final/` (`final_*.png`, `final_scorecard.txt`); change log, the
calibration renders' harness and probes: `_scratch/v1/fix/` (`changelog.md`,
`light.py`, `final_score.py`, `final_figs.py`, `truth_summary.py`,
`knots_tags.py`, `morse_tags.py`, `veinprof_tags.py`; renders in
`_scratch/v1/fx1` ... `fx11`).

**Correctness fixes** (regression tests in `tests/test_fixes_v1.py`):

| defect | fix | measured |
|---|---|---|
| 1 pair and bundle members pushed onto each other by independent tortuosity | a follower takes its leader's displacement along the shared stretch (`_shared_meander`); its own nodes move with it; companions' offset set for the host's widening (1.6 x nominal radius) | projected overlap over the paired length in the frame, 12 healthy + 3 pathologic scenes: companions 14 % -> 0.5 %, bundles 18 % -> 2.8 % |
| 2 no twigs, a mesh without flow | 30 % of the length budget for plain long vessels; twigs from collecting veins' tributaries; a fallback of up to 6 twigs per side that has none; zero-flow vessels removed, an interior source / sink raises | 0 zero-flow vessels, 0 interior sources / sinks in the 18 final scenes; healthy seed 25: 19-28 twigs (was 0) |
| 3 the last-resort removal took wide vessels; corners from merges | victim by visual cost; one steeper-bump round first; no fold-back merges of thin vessels; corners rounded; coincident short vessels removed; episcleral depth range grows with dilation x size gain | healthy: removals with r >= 1.5 d_c are now 0-3 per frame, all r <= 4.8 (v1: up to 6.5); pathologic seed 4: 22 -> 3; **pathologic seed 2 still loses 14** (r up to 8, 304 d_c in frame) |
| 4 a self-overlap left behind | no fold of a non-capillary vessel by tortuosity; self pass after the C1 pass; a leftover is removed | random seed 1 full frame valid; check(overlap=True) clean on all 18 final scenes (one intermediate set had 5 failures, from the arcade removal running last: it now runs before the late overlap passes) |
| 5 labels only of the average | `<key>_frame` rasters | - |
| 6 kinks at web nodes | kinks up to 150 deg smoothed at chain level and made C1 | first-leg kinks 0 deg; over +-2 d_c p90 3 deg, max 31 deg in frame (healthy 4) and 80 deg outside the frame (healthy 2, r 0.7: a tight S-bend right after a C1 node) |
| 7 collecting companions' gap | `collect_pair_gap` | - |
| 8 arcades crossing their host | removed (an anastomosis; the host's pieces merge back) | 0-15 per frame removed |
| 9 render reach bound | the surface's vertex at x0 - g / (2c) | test |

**Realism steps** (healthy seeds 0-5 per step, scored against the dense site;
full log in `_scratch/v1/fix/changelog.md`):

| tell | change | kept |
|---|---|---|
| T1 knots | crossings composited by depth (`scene.crossing_bypass`) | yes |
| T1 knots, gap 1 | a long vessel's course that meets another at < 30 deg within 8 d_c turns to run beside it (`web_para_mode = 'align'`); turning away instead removed the side-by-side runs | align: yes |
| T2 tar-black veins | flattened venous lumens (`_aspects`), and at least one collecting venule per frame (`collect_min`); a milder flattening (0.5-0.8 from r 2.5) left the scenes too dark | yes |
| T4 elbows | a long vessel keeps its planned course through its forks (rotated by the Murray angle), the branch walks clear of it | yes |
| T5 dotted capillaries | capillary red-cell speed 150 d_c/s: no change; filled share 0.6: the dashes match but the frame / average statistics worsen | no |
| T6 ribs | lumen-edge piece rule | yes |
| T7 frame blurrier | frame offset within the spread | yes |
| T8 bamboo | aggregate grain averages across wide lumens (r_agg 2 d_c; 2.5 tried) | yes |

**Scorecards** (scalars pass (far, |z| > 4); distributions + spectra;
`_scratch/v1/final/final_scorecard.txt`):

| comparison | v0 | v1 as reviewed | final |
|---|---|---|---|
| healthy averages vs the dense site (4 stills) | 46/102 (11); 6/11 | 50/102 (9); 7/11 | 50/102 (5); 8/11 |
| healthy + pathologic averages vs the 9 stills | 84/102 (1); 9/11 | 74/102 (1); 9/11 | 91/102 (1); 10/11 |
| healthy frames vs the 9 raw frames | 83/102 (0); 11/11 | 73/102 (1); 9/11 | 76/102 (4); 9/11 |
| healthy + pathologic frames vs the 9 raw frames | 81/102 (0); 11/11 | 52/102 (3); 6/11 | 73/102 (4); 9/11 |

**The five gaps and the review's tells** (12 healthy scenes each; z against
the dense site, the real pair or 15-50-52's raw frames):

| gap | statistic | real | v0 | v1 as reviewed | final |
|---|---|---|---|---|---|
| 1 | `web_span` | 0.67 | 0.33 (-4.9) | 0.66 (-0.2) | 0.63 (-0.6) |
| 1 | `branch_rate` | 7.25 | 5.37 (-4.0) | 6.52 (-1.6) | 6.06 (-2.6) |
| 1 | `xing_shallow` | 0.063 | 0.133 (+6.1) | 0.113 (+4.3) | 0.095 (+2.8) |
| 2 | `pair_frac` | 0.261 | 0.145 (-16.4) | 0.130 (-18.4) | 0.207 (-7.7) |
| 2 | `pair_run5` | 0.128 | 0.077 (-3.6) | 0.042 (-6.0) | 0.096 (-2.3) |
| 2 | `par_excess` | 1.38 | 0.97 (-4.9) | 1.09 (-3.5) | 1.09 (-3.4) |
| 2 | `bundle_frac` | 0.0138 | 0.0041 (-2.9) | 0.0087 (-1.5) | 0.0299 (+4.9) |
| 3 | `c_wide` | 0.286 | 0.297 (+0.6) | 0.392 (+5.7) | 0.313 (+1.4) |
| 3 | `F_sd` | 0.080 | 0.083 (+1.0) | 0.092 (+3.7) | 0.075 (-1.4) |
| 3 | widest vein's plateau (review's `veinprof`, Np) | -0.41 / -0.44 | | -0.66 .. -0.84 (2 scenes) | median -0.42 (-0.61 .. -0.13, 12 scenes) |
| 3 | `wide98_fwhm` (px) | 46.3 | 35.7 (-6.2) | 45.4 (-0.5) | 46.6 (+0.2) |
| 3 | `darkw_p99` (px) | 62 | 39 (-7.5) | 52 (-3.4) | 47 (-5.1) |
| 3 | `dark128_0.45` | 0.0167 | 0.0078 (-7.3) | 0.0201 (+2.8) | 0.0081 (-7.1) |
| 3 | `sharp_p50` | 3.52 | 3.69 (+4.5) | 3.73 (+5.7) | 3.74 (+5.9) |
| 4 | `focus_var` | 0.151 | 0.092 (-2.6) | 0.160 (+0.4) | 0.183 (+1.4) |
| 5 | `sigma_extra_med` (px); frames blurrier than their average | 0.93; 0 of 8 | 0.69 (-1.0) | 0.72 (-0.9); 3 of 12 | 0.90 (-0.1); 1 of 12 (-0.08 px) |
| 5 | `grain_coarse` | 0.0063 | 0.0076 (+3.0) | 0.0059 (-0.9) | 0.0041 (-4.9) |
| 5 | `r_cv` | 0.119 | 0.126 (+2.6) | 0.134 (+5.7) | 0.129 (+3.6) |
| 5 | `weak` | 1.2 % | 0.8 % (-2.6) | 0.7 % (-3.4) | 0.7 % (-2.9) |
| T1 | dark knots per 100 lambda; share of crossings > 0.05 Np darker than their darker arm (review's `knots`) | 0.47; 28 % | | 1.02; 48 % | 0.83; 42 % |
| T5 | faint lines: frame dashes > 2x the average; absent; robust CV (review's `morse2`) | 3.1-4.2 %; 4.0-4.7 %; 0.30-0.32 | | 7.0 %; 5.3 %; 0.37 | 7.5 %; 5.5 %; 0.37 |

Pathologic (3 scenes, against the healthy dense site, for information): the
frames' wide-vessel grain `grain_coarse` +41.5 -> -1.4 SD (the 'bamboo'
bands are gone), `c_wide` +12.7 -> +0.1, `F_sd` +24 -> +6.2.

What changed visibly (`final_side_by_side.png`, `final_before_after.png`):
the widest veins are dark grey with a rounded profile instead of tar-black
(and every healthy frame has one, though in 2 of 12 only a narrow
tributary is in the frame); companions and bundles show as doubled and
tripled thin lines; fewer and milder knots at crossings; the collecting
veins curve smoothly through their confluences.

**What still differs from the real stills** (specific):

- **Knots**: still about 1.8x the real rate (0.83 against 0.47 per 100
  lambda). The remaining ones are mostly a sharp medium vessel crossing a
  blurred wide deep vessel (dark beads along it, `final_before_after.png`
  seed 2), where the compositing changes little (the upper vessel is
  shallow, little light bypasses it).
- **Bundles too regular and too frequent** now that they stay intact
  (`bundle_frac` +4.9 SD; 3-5 evenly spaced identical lines); the bundle rate
  (3 per frame, a proposal) was set while bundles were half destroyed.
- **The dark-area statistics disagree with the vein profile**: by the
  review's own widest-vein profile the plateau now matches (-0.42 against
  -0.41 / -0.44 Np), but the area darker than -0.45 Np against a 128 px
  background is half the dense site's (`dark128_0.45` -7.1 SD) and the widest
  dark vessels narrower (`darkw_p99` 47 against 62 px): either real veins
  have darker stretches than their median profile, or the flattening is
  too strong for the vessels just below collecting size. Not resolved.
- **Profiles too boxy** (`sharp_p50` +5.9 SD) and the faint fine structure:
  too few thin wiggly vessels, loops (-3.3 SD) and junctions (-3.3 SD), dark
  /bright ridge energy at 1-2 px -8.6 / -6.0 SD (the texture is a smooth
  field).
- **Single frames** are still about their average plus noise: faint
  capillaries are dotted (T5 unchanged: 7.5 % dashes against 3.1-4.2 %);
  raw frames are blurrier than the real ones at their sharpest (edge p10
  2.04 against 1.76 px, FWHM p10 10 against 7.2 px, skeleton density far
  below); the frame - average grain along wide vessels is now low
  (`grain_coarse` -4.9 SD), the price of removing the bamboo bands.
- **Focus**: soft regions now slightly too common on average (`soft_tiles`
  0.38 against 0.22, +1.6 SD), the extreme real cases (15-31-37) still rare.
- **Pathologic deep layer**: the overlap pass still removes up to 14 wide
  deep vessels in a pathologic frame (seed 2).
- **Evidence is thin**: each step on 6 seeds with the anatomy redrawn by
  most anatomy changes (scorecard pass counts moved by 5-10 between draws of
  the same code); 4 dense stills from 2 sites, one burst's frames.

Timings (18 scenes made four at a time, contended): healthy make_scene
79-160 s (anatomy 36-99 s, render 6.5-26 s per image, crossing compositing
included), save 10-31 s; pathologic 156-226 s; random 40-260 s. Not re-timed
alone; the crossing compositing and the frame's label rasters add about 2
label passes per image (a few seconds each).

## Realism: v1 results (2026-09-30)

Five criticisms of the v0 scenes, in priority order: (1) layout:
trees and a border comb instead of a web of long crossing vessels; (2) too few
parallel pairs and bundles; (3) the widest vessels not black or wide enough;
(4) focus too uniform over the frame; (5) a single frame just its average plus
noise. Each has statistics in `tells.py` (`T.GAPS[key]` gives a key's gap).
Two builders worked in parallel (anatomy: the web; optics and imaging: focal
surface, focus jitter, registration field, texture); then an integration and
a bounded calibration pass (this section and "Calibration, v1 integration").

Scenes: `_scratch/v1/scenes/` (healthy seeds 0-11, pathologic 0-2, random 0-2;
each scene folder as written by `save_scene`, plus `measure.json` with the
tells). Harness, change log and helper scripts: `_scratch/v1/integ/`
(`run.py gen / light / measure / table`, `figs.py`, `changelog.md`). Tables:
`_scratch/v1/scenes/table.md` (every gap statistic, the pathologic and random
scenes too) and `scorecard_v{0,1}_{pooled,dense,frame}.md`. Three states are
compared: **v0** (`_scratch/final/scenes`: healthy 0-9, pathologic 1), **v1
as integrated** (both builders' code before the calibration:
`_scratch/v1/scenes_integrated`) and **v1** (after it: the package defaults).

| comparison: scalars pass (far, \|z\| > 4); distributions | v0 | v1 as integrated | v1 |
|---|---|---|---|
| healthy averages vs the dense site (4 full-frame stills) | 46/102 (11); 6/11 | 46/102 (6); 4/11 | 50/102 (9); 7/11 |
| healthy + pathologic averages vs the 9 real stills | 84/102 (1); 9/11 | 77/102 (1); 10/11 | 74/102 (1); 9/11 |
| healthy frames vs the 9 real raw frames | 83/102 (0); 11/11 | 74/102 (2); 9/11 | 73/102 (1); 9/11 |
| healthy + pathologic frames vs the 9 real raw frames | 81/102 (0); 11/11 | 65/102 (3); 7/11 | 52/102 (3); 6/11 |

(v0: 10 healthy + 1 pathologic scenes; v1: 12 healthy + 3 pathologic. The
pathologic scenes are not meant to look like the healthy real stills; in v1
they are much darker, which pulls the pooled rows down.)

The five gaps (healthy scenes; mean over the scenes, z against the dense
site's SD; for gap 5 against the real frame / average pair of 15-50-52 or its
raw frames; "yes" = |z| > 3 and the ranges apart, "partly" = one of the two):

| gap | statistic | real | v0 | v1 as integrated | v1 |
|---|---|---|---|---|---|
| 1 | `web_span` (dark skeleton in half-frame networks) | 0.67 | 0.33 (-4.9) | 0.61 (-0.8) | 0.66 (-0.2) |
| 1 | `branch_rate` (junctions per length) | 7.25 | 5.37 (-4.0, yes) | 7.01 (-0.5) | 6.52 (-1.6) |
| 1 | `cont_share_10` (long continuation paths) | 0.744 | 0.805 (+4.9, yes) | 0.763 (+1.6) | 0.772 (+2.2) |
| 1 | `calibre_change` | 0.181 | 0.236 (+3.3, yes) | 0.206 (+1.5) | 0.196 (+0.9) |
| 1 | `xing_shallow` (crossings under 30 deg) | 0.063 | 0.133 (+6.1, partly) | 0.108 (+3.9, partly) | 0.113 (+4.3, partly) |
| 2 | `par_excess` | 1.38 | 0.97 (-4.9, yes) | 0.84 (-6.5, partly) | 1.09 (-3.5, partly) |
| 2 | `pair_frac` | 0.261 | 0.145 (-16.4, yes) | 0.146 (-16.2, yes) | 0.130 (-18.4, yes) |
| 2 | `pair_run5` | 0.128 | 0.077 (-3.6) | 0.053 (-5.3) | 0.042 (-6.0, yes) |
| 2 | `bundle_frac` | 0.0138 | 0.0041 (-2.9) | 0.0067 (-2.1) | 0.0087 (-1.5) |
| 3 | `darkw_p99` (widest dark vessels, px) | 62 | 39 (-7.5, yes) | 46 (-5.3) | 52 (-3.4, partly) |
| 3 | `dark128_0.45` | 0.0167 | 0.0078 (-7.3) | 0.0128 (-3.3) | 0.0201 (+2.8) |
| 3 | `wide98_fwhm` (px) | 46.3 | 35.7 (-6.2) | 43.7 (-1.5) | 45.4 (-0.5) |
| 3 | `wide_edge_rel` | 0.165 | 0.205 (+3.5) | 0.194 (+2.5) | 0.166 (+0.1) |
| 3 | `c_wide` (contrast of 2-4 lambda vessels) | 0.286 | 0.297 (+0.6) | 0.321 (+1.9) | 0.392 (+5.7, partly) |
| 3 | `sharp_p50` (FWHM / edge: "boxy") | 3.52 | 3.69 (+4.5, yes) | 3.67 (+4.0) | 3.73 (+5.7, partly) |
| 4 | `focus_var` | 0.151 | 0.092 (-2.6) | 0.149 (-0.1) | 0.160 (+0.4) |
| 4 | `soft_tiles` | 0.22 | 0.03 (-1.9) | 0.25 (+0.3) | 0.20 (-0.2) |
| 5 | `sigma_extra_med` (the average's extra blur, px) | 0.93 | 0.69 (-1.0) | 0.79 (-0.6) | 0.72 (-0.9) |
| 5 | `grain_coarse` (frame - average along vessels) | 0.0063 | 0.0076 (+3.0) | 0.0036 (-6.2, yes) | 0.0059 (-0.9) |
| 5 | `weak` (the average's medium vessels < 60 % in the frame) | 1.2 % | 0.8 % (-2.6) | 0.5 % (-4.7, yes) | 0.7 % (-3.4, partly) |
| 5 | `r_cv` (graininess along medium vessels) | 0.119 | 0.126 (+2.6) | 0.114 (-1.8) | 0.134 (+5.7, partly) |
| - | lambda (median FWHM, px) | 11.9 | 11.4 (-0.5) | 14.9 (+3.3, yes) | 13.5 (+1.8) |
| - | edge width p10 / p50 (px) | 2.19 / 3.36 | 2.24 / 3.09 | 2.69 / 4.00 (+2.3 / +2.4) | 2.35 / 3.54 (+0.7 / +0.7) |
| - | `F_sd` (overall contrast) | 0.080 | 0.083 (+1.0) | 0.086 (+1.8) | 0.092 (+3.7, partly) |
| - | raw frames: FWHM p10 / edge p10 (px; the 15-50-52 frames) | 7.2 / 1.76 | 8.7 / 1.89 | 10.4 / 2.12 | 9.6 / 1.99 |

Per gap:

1. **Layout** no longer separates on any statistic (v0: 5 "yes"): a web of
   long vessels crossing the frame, joined by forks and arcades, replaced the
   trees and the comb. Crossings at shallow angles are still about twice as
   common as in the stills (`xing_shallow` +4.3).
2. **Pairs are not fixed.** The truth has companions, venae comitantes and
   bundles, but the thin lines the pair tell looks at have a parallel
   neighbour 12-30 px away over 10-27 % of their length, against 32-45 % in
   the dense stills (`_scratch/v1/integ/pair_spacing.py`; the stills' doubled
   lines are 16-24 px apart). Placing the companions at that spacing
   (calibration step 5) moved `par_excess` from -6.5 to -3.5 but not
   `pair_frac`. What is missing is also a denser population of thin, sharp
   lines (profiles under 8 px FWHM: 14 % against 20 %; fewer loops and faint
   mesh), not only pairs.
3. **Widest vessels**: wide enough (`wide98_fwhm` -0.5), with real edge
   widths (`wide_edge_rel` +0.1); the widest dark ones are still narrower (52
   against 62 px) and about a third of the frames have no collecting vein;
   and the wide vessels are now too dark (`c_wide` +5.7, overall `F_sd`
   +3.7). Profiles are still boxier than real (`sharp_p50` +5.7).
4. **Focus** matches: whole regions go soft, superficial vessels included.
5. **Frame vs average**: the average is blurrier than its frame (0.72 px
   against 0.93, the sign changing across tiles as in the real pair); the
   frame's red-cell grain along wide vessels matches (`grain_coarse` -0.9);
   the medium vessels are slightly too grainy (`r_cv` +5.7) and fade less
   often (`weak` -3.4). Raw frames are still too blurred: FWHM p10 9.6 px
   against 7.2 px in the 15-50-52 frames, edge p10 1.99 against 1.76 px.

**What still looks different** (figures below, same grey scale):

- Dark knots where two medium vessels cross, most visibly at shallow angles
  (`fig_side_by_side.png` crop 2, `fig_frame_vs_average.png`). Stacked
  vessels' optical densities add (Beer-Lambert). Treating the light scattered
  above each vessel as one shared bypass would make a typical medium-vessel
  crossing only about 7 % lighter (estimated by hand, not rendered), so the
  knots' conspicuity is not explained yet.
- Fewer thin, crisp lines than in 15-50-52 (the fine, wiggly conjunctival
  vessels and capillary loops); the synthetic thin vessels are smoother
  (branch arc/chord in frames 1.032 against 1.047).
- More medium-dark vessels of similar calibre; the real hierarchy is one or
  two black veins and many thin grey lines.
- About a third of the healthy frames have no black collecting vein.

Figures (`_scratch/v1/scenes/`):

- `fig_side_by_side.png`: healthy seed 0's average next to 15-50-52 and
  13-51-26: full frame, two 1:1 crops, a 2x zoom.
- `fig_frame_vs_average.png`: the synthetic frame and its average next to the
  real raw frame 70 of 15-50-52, carried onto its non-rigid average by the
  stabilization's fields, and that average (full, 1:1, 2x).
- `fig_truth_overlay.png`: the truth by vessel id, and a 3x zoom by class.
- `fig_before_after.png`: v0 and v1 of healthy seeds 0-2.
- `fig_gap_z.png`: z of every gap statistic, v0 / as integrated / v1.
- `fig_scorecard_{dense,pooled,frame}.png`: the suite's scorecard, v0 and v1.

## Realism: v0 results (the realism pass before v1)

Scenes: `_scratch/final/scenes/` (healthy seeds 0-9, pathologic seed 1, random
seed 3; figures `final_*.png` next to them; the numbers in
`final_scorecard.txt`). "Before" is the integration's scenes as reviewed
(healthy 0-2, pathologic 1); "after" is this pass (healthy 0-9 and
pathologic 1; the dense-site rows use the healthy ones).

| comparison | before: scalars pass (\|z\| > 4); distributions | after |
|---|---|---|
| averages vs the 9 real stills | 91 / 102 (1); 9 / 11 | 84 / 102 (1); 9 / 11 |
| healthy averages vs the dense site (4 full-frame stills) | 36 / 102 (14); 6 / 11 | 46 / 102 (11); 6 / 11 |
| single frames vs 9 real raw frames | 86 / 102 (0); 10 / 11 | 81 / 102 (0); 11 / 11 |

Against the 9 stills the pass count went down: the synthetic contrast now
matches the dense site the healthy preset imitates, which sits above the
pooled mean of the sparser, lower-contrast strips (profile contrast p10 z
+2.6, skeleton branch length +2.7). Against the dense site it went up, with
fewer far-off statistics. The tells, healthy scenes against the dense site
(z in brackets):

| tell | real dense site | before | after |
|---|---|---|---|
| overall contrast (F SD) | 0.080 | 0.058 (-6.8) | 0.083 (+1.0) |
| contrast of thin vessels (0.5-1 lambda) | 0.080 | 0.062 (-3.5) | 0.086 (+1.1) |
| contrast of wide vessels (2-4 lambda) | 0.29 | 0.18 (-5.5) | 0.30 (+0.6) |
| area darker than -0.45 Np (128 px background) | 0.017 | 0.0014 (-12.7) | 0.0078 (-7.3) |
| width of the widest dark vessels (px) | 62 | 35 (-8.8) | 39 (-7.5) |
| focus variation over tiles | 0.15 | 0.063 (-3.9) | 0.092 (-2.6) |
| patchiness of the fine texture | 0.44 | 0.33 (-2.3) | 0.40 (-1.0) |
| parallel neighbours | 1.38 | 0.78 (-7.2) | 0.97 (-4.9) |
| frame: share of the average's medium vessels absent | 0.19 % | 2.4 % (+15) | 0.15 % (-0.3) |
| frame: share weak | 1.2 % | 6.8 % (+38) | 0.8 % (-2.6) |
| frame: graininess along vessels | 0.119 | 0.133 (+5.4) | 0.126 (+2.6) |

Still failing and what they mean (after):

- FWHM / edge width median z +5.6 against the 9 stills (3.77 against 3.47):
  profiles too boxy (T7 of the review). FWHM p10 / lambda +2.9: the thinnest
  visible vessels are relatively too wide.
- Skeleton branch length between junctions +2.7 (dense site +8.5), linked
  path length +2.3: fewer junctions and crossings per length than real;
  the architecture (next section).
- Dark/bright ridge energy at sigma 1-2 about -2.3, at 4-8 px +5..+7 against
  the dense site: the balance of bright and dark fine structure.
- Parallel neighbours -1.7 (dense site -5.0): better, still too few pairs.
- Frames: robust CV of FWHM along paths -3.8, FWHM p10 +3.0.

**What still looks different** (the figures, same grey scale):

- The dense stills are a lattice of long medium vessels crossing the whole
  field in many directions. The synthetic stills now have crossings at many
  angles and the right contrast, but more of their vessels visibly end
  (trees thinning into the faint capillary mesh), and parts of the field are
  emptier.
- No 45-60 px black vessels with sharp edges: the widest dark ones are
  narrower or blurred (episcleral).
- Crossings of a sharp vessel over a deep one, and some junctions, make dark
  knots that are more conspicuous than in the stills.
- Companion vessels join their tree and the mesh by short straight
  connectors, which can make right-angle kinks (most visible in the random
  preset).
- The background texture is smooth Gaussian clouds; the sclera's lobules
  and bright fibres are not modelled.
- Single frames look like their average plus noise and grain, as real frames
  do (the review's broken dashes are gone); the real raw frame shown is
  unregistered and so not the same view as its still.

## Calibration

### v2 integration (2026-10-01: after the frames, anatomy, renderer and junction builders)

Four steps on healthy seeds 0-5 (`_scratch/v2/integ/changelog.md`, renders in `_scratch/v2/cal_*`): the veins'
flattening (`vein_aspect` 0.4-0.65 -> 0.7-0.9; round veins were tried and overshot), the tissue halo (0.45 ->
0.375; 0.3 tried), a diagnostic without the frame's red-cell granularity (not kept), and a soft limit on the
calibre noise (`r_var_clip` 2). The tables are in [Realism: v2](#realism-v2-2026-10-01-the-integrated-round).
The builders' own calibrations (the red cells' speed, clumps, granularity and capillary columns; f_spectral,
mu and l_bypass; the anatomy's bundle, channel and meander parameters) are in their modules' comments.

### v1 integration (2026-09-30: after the anatomy and optics builders)

The two builders' code ran together without a test failure (144 passed); the
optics had been calibrated on the v0 anatomies and the web on the optics as
they were during its build, so the integrated scenes were first measured as
they came (`_scratch/v1/scenes_integrated`). Largest gaps then: pairs
(`pair_frac` -16), widest vessels (`darkw_p99` -5.3), every width too large
(lambda 14.9 px against 11.9, edge width p10 / p50 +2.3 / +2.4 SD; raw frames
FWHM p10 10.4 px against 7.2), the frames' red-cell grain (`grain_coarse` -6,
`weak` -4.7). Each step below was rendered on healthy seeds 0-5 (the optics-only
steps on the same anatomies), measured with the suite and the tells, and
looked at; the full log is `_scratch/v1/integ/changelog.md`.

| step | change | effect (z against the dense site) | kept |
|---|---|---|---|
| 1 | k 4 -> 3.2 px/d_c (plan prior 3-3.5; in lambda units the web's densities matched) | lambda only 14.9 -> 14.1 (FWHM is blur-dominated), junction density -3.1 -> -4.8 (the mesh drops below detection), contrast up (1.6x the vessels per frame); dense scorecard 36/102 | no |
| 2 | PSF core s0 1.8 -> 1.55 px, scatter blur 0.03 -> 0.015 px/d_c | edge p10 2.65 -> 2.39, FWHM p10 8.5 -> 7.8, `par_excess` -7.1 -> -5.4 | yes |
| 2b | + defocus gain 0.05-0.10 -> 0.04-0.08 px/d_c | edge p50 3.93 -> 3.55 (+0.7); focus statistics unchanged | yes |
| 3 | thinner long vessels (web_r_V 0.5-1.1, web_r_A 0.4-0.7) | FWHM p50 unchanged (the median profile is not a long vessel: below); frames' grain lower | no |
| 4 | deep plexus sparser: plexus_spacing 50 -> 80 d_c (about 10 -> 6 veins per frame) | FWHM p50 +2.3 -> +1.5, 16-24 px profiles 21 -> 17 % (real 13 %), suite 53/102 | yes |
| 5 | companions at wall gap 1.2-2.5 (r_a + r_v) (was 0.5-1.5), bundle gaps 1.5-3 d_c (0.8-1.8), collecting veins D 7-11 at their upstream end (6-9) | `par_excess` -7.6 -> -4.8, `bundle_frac` -1.5 -> 0.0, `darkw_p99` -8.6 -> -5.3 | yes |
| 6 | defocus gain 0.03-0.06 with focal tilt 0.3-0.8 (less depth blur within a region) | no change beyond noise | no |
| 7-10 | red-cell aggregate grain: uniform 0.25 / 0.18, then 0.14 or 0.12 x (r / r_cont) up to 0.35 | uniform: `grain_coarse` and `r_cv` / `weak` cannot both match; 0.12 x r / r_cont: none of the three separates on 6 scenes | 0.12 x r / r_cont |

The evidence behind the kept steps:

- **s0 and scatter**: the sharpest real raw-frame edges are 1.76 px (edge p10
  of the 15-50-52 frames; 1.79 over the 9 reference frames), below the v0
  core of 1.8 px. That core was calibrated on averages before v1 added the
  frames' focus mixture and the registration field to them. The scatter term
  (0.03 px/d_c, the renderer's default, never calibrated) mattered little
  while the conjunctival vessels sat at 9-20 d_c; the web puts them at 12-46.
- **Who makes lambda**: attributing the detected profiles of two scenes to the
  truth (`_scratch/v1/integ/who_lambda.py`): 27-46 % were deep episcleral
  vessels (FWHM 18-22 px, 60-75 d_c deep), 53-58 % long venules (median r
  1.6-1.9 d_c), only 5-28 % thin vessels. FWHM histogram of the profiles, real
  dense / v1 as integrated: < 8 px 20 / 9 %, 8-12 px 31 / 22 %, 12-16 px 21 /
  27 %, 16-24 px 13 / 22 %, > 24 px 16 / 20 %: hence step 4, not step 1 or 3.
- **Pair spacing** (`pair_spacing.py`: for thin line points, the distance to
  the nearest parallel dark line found by the pair tell's own profile search):
  real dense stills 8-12 px 0.2-1.2 %, 12-16 px 2-8 %, 16-24 px 22-27 %,
  24-30 px 8-9 % of thin line length; v1 as integrated 0-1.5 / 0.2-1.4 / 7-9 /
  2-4 %. The stills' doubled lines are 16-24 px apart (4-6 d_c centre to
  centre); the web's companions at 0.5-1.5 (r_a + r_v) sat 9-14 px apart,
  where two blurred 8-10 px lines merge. The plan's G3 range (1.5-4) was
  closer to the stills than the review's by-eye 0.5-1.5.
- **Grain**: the optics builder's 0.14 was calibrated on the v0 anatomy, whose
  medium vessels were shallow twigs; the web's continuous vessels are deeper
  and darker (the chord law compresses their OD fluctuation). Red cells
  aggregate more at the lower shear rates of wider venules, so the relative SD
  now grows with the radius (`imaging.filling(grain_r_exp, grain_amp_max)`).

Not calibrated further (reported in "Realism: v1 results"): pairs, the
darkness of the wide vessels (`c_wide`, `F_sd`: made worse by step 5's wider
collecting veins together with the anatomy's own long-vessel radii), the raw
frames' thin-line width, the crossing knots. Six seeds per step: the
differences between steps that change the anatomy include the change of
random anatomy (step 3 vs 4 vs 5 are different draws).

### v1 optics and imaging (2026-09-30: gaps 3-5 of the v1 review)

The optics part of the five criticisms: wide-vessel profiles too
boxy (gap 3), focus too uniform over the frame (gap 4), a single frame too
much like its average (gap 5). Harness `_scratch/v1/optics/calib.py`: the
10 v0 healthy anatomies (`_scratch/final/scenes/healthy_s000-009/truth.json`)
re-rendered with each configuration (so only optics and imaging change),
measured with `stats.analyse`, `tells` and the frame / average pair tells,
against the 4 dense stills, 6 real raw frames (15-50-52 f5/30/70/110,
15-31-37 f40/119) and the real pair (8 frames of 15-50-52). Seven
configurations (c1-c7); c7 is the package default.

**Gap 3, what makes the profiles boxy.** The mean normalised cross-profile
of the sharpest, middle and softest third of the profiles in each FWHM bin
(9-55 px) is the same in the real dense stills and in v0 (figure
`_scratch/v1/optics/meanprof_real_vs_v0.png`): no flat tops from chord-law
saturation, no missing wall or plasma-layer softening, no size-dependent
blur of in-focus vessels. The lower envelope of the edge width at each FWHM
agrees within 1-6 %. What differs is the spread: at every width real
profiles are more often blurred (edge p50 6-10 % above v0 at 9-28 px). Real
widths with the v0 edge distribution at each width give FWHM / edge 3.73,
v0 widths reweighted to the real ones 3.72 (real 3.52): the width mix
explains nothing, the edge-at-width everything. So the cause is too few
blurred vessels at each width, i.e. focus and registration (gaps 4 and 5),
plus the anatomy's depth mix; the chord law, the 6-box staircase and the
PSF were left as they are. FWHM / edge per bin now (real / v0 / v1): 7-9 px
3.25 / 3.41 / 3.27; 12-16 px 3.62 / 3.97 / 3.63; 16-21 px 3.86 / 4.38 / 3.83;
21-28 px 4.01 / 4.48 / 3.96 (`_scratch/v1/optics/envelope.py`).

| statistic | real dense (sd) | v0 optics | v1 (c7) |
|---|---|---|---|
| focus variation over tiles (`focus_var`) | 0.151 (0.022) | 0.092 (-2.6) | 0.152 (+0.0) |
| soft tiles | 0.22 (0.10) | 0.03 (-1.9) | 0.17 (-0.5) |
| tile edge p90 / p10 | 1.46 (0.12) | 1.26 (-1.6) | 1.46 (+0.0) |
| FWHM / edge median (`sharp_p50`) | 3.52 (0.04) | 3.68 (+4.5) | 3.60 (+2.1) |
| edge p50 (px) | 3.36 (0.26) | 3.09 (-1.0) | 3.69 (+1.3) |
| patchiness of the fine texture | 0.44 (0.05) | 0.40 (-1.0) | 0.45 (+0.2) |
| frames: focus variation over tiles (real frames) | 0.133 (0.033) | 0.083 (-1.5) | 0.109 (-0.7) |
| pair: extra blur of the average (px) | 0.93 (0.24) | 0.69 (-1.0) | 1.05 (+0.5) |
| pair: edge ratio frame / average | 0.973 (0.016) | 0.980 (+0.4) | 0.977 (+0.2) |
| pair: grain of the frame (Np) | 0.0101 | 0.0115 (+1.9) | 0.0110 (+1.2) |
| pair: frame - average, 2.5 px - lambda | 0.0063 | 0.0076 (+3.0) | 0.0069 (+1.4) |
| frame: graininess `r_cv` | 0.119 (0.003) | 0.126 (+2.6) | 0.122 (+1.0) |
| frame: medium vessels weak (< 60 %) | 0.0117 | 0.0078 (-2.6) | 0.0042 (-5.2) |

(v0 optics = the v1 code with the v0 settings, c0_v0; means of 10 scenes, z
against the dense site's SD; real frames and pair from 15-50-52 and
15-31-37.) The steps: c1 (jitter 5-25 d_c, uniform-ish registration field):
the stills got MORE uniform (the average's mixture adds the same blur
everywhere) and frames often blurrier than their average (-2 px); c2-c3
stronger tilt, smaller jitter, per-scene registration field; c4-c6 tilt
0.2-0.6, jitter 2-8, registration 0.2-0.6 px, grain 0.12; c5/c6 also r_cont
1.4 (matches `weak` but draws regular dashes along 1.2-1.4 d_c vessels in a
frame, which real frames do not show: rejected); c7 grain 0.14, r_cont 1.2
(the package default); c8 partial plasma gaps in 0.6-1.2 d_c vessels
(`filling(gap_fill=0.6)`: frames too continuous, weak 0.12 % and absent 0 %
against 1.2 % and 0.19 %; the parameter stays, default 0).

Still different: `weak` (a frame's medium vessels below 60 % of their
darkness in the average) 0.4 % against 1.2 % (v0 0.8 %: the lower grain);
the along-lumen structure of the average (`grain_avg` 0.0061 against 0.0049,
+12 SD) comes from the anatomy's radius and haematocrit variation along
vessels (constant r and h per vessel: 0.0038 -> 0.0026 Np), not from the
optics; `grain_fine` (pixel scale) stays low, within the error of the noise
subtraction (`grain_sham`). Wide-vessel statistics (`darkw_p99`,
`dark128_0.45`, `wide_edge_rel`) are the anatomy's (the widest vessels are
deep): unchanged by the optics.

Render time: the texture's defocus adds about 1.5 s per image and the
registration field about 0.5 s to the average's image formation; the
closed-form render is unchanged (average 4.5-6, frame 5.6-8 s): 21 s per
scene without anatomy, labels and saving (v0 settings 19 s).

On the anatomy as it was at 11:30-11:40 on 2026-09-30 (the web anatomy in
progress, 4 full scenes `_scratch/v1/optics/current/`, figures
`fig_current_*.png`), preliminary: stills' focus variation 0.178 and soft
tiles 0.44 (real 0.151 and 0.22: more than on the v0 anatomy, whose vessels
were deeper and more blurred anyway), FWHM / edge 3.77; frames against their
average 0.38 px (-0.67 .. 1.48: 2 of 4 frames blurrier than their average
overall, against 0 of 8 real frames), frame - average grain 0.0023 against
0.0063, `r_cv` 0.093 against 0.119. The focus jitter, registration field and
grain were calibrated on the v0 anatomies; they should be re-checked with
`_scratch/v1/optics/calib.py` on the final anatomy (point `SC` at its
scenes).

### Second pass (2026-09-30, after the realism review)

Each step was rendered on healthy seeds 0-2 and pathologic seed 1, scored
against the 9 stills, the dense site and the frames, measured with the tells,
and looked at (`_scratch/realism_loop/<step>/look_*.png`; the harness is
`_scratch/realism_loop/run_iter.py`). Tells of the healthy scenes, with z
against the dense site in brackets (real dense-site values in the first row):

| iteration | avg vs 9 stills: pass (far) | avg vs dense site: pass (far) | frame vs 9 frames: pass (far) | F_sd | c_thin | c_wide | dark128_0.45 | darkw_p99 | focus_var | par_excess | bg_patch_cv | drop | weak | r_cv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real dense site | | | | 0.0797 | 0.0801 | 0.286 | 0.0167 | 62 | 0.151 | 1.38 | 0.444 | 0.00186 | 0.0117 | 0.119 |
| it00_fixed | 92/102 (1) | 35/102 (12) | 86/102 (0) | 0.0578 (-7) | 0.0623 (-3) | 0.183 (-6) | 0.00141 (-13) | 35.3 (-9) | 0.0614 (-4) | 0.805 (-7) | 0.335 (-2) | 0.024 (+15) | 0.0654 (+37) | 0.13 (+4) |
| it01_contrast_fill | 74/102 (1) | 36/102 (18) | 52/102 (0) | 0.115 (+11) | 0.0723 (-1) | 0.365 (+4) | 0.0338 (+14) | 55.7 (-2) | 0.0814 (-3) | 0.746 (-8) | 0.42 (-0) | 0.00239 (+0) | 0.0132 (+1) | 0.0761 (-16) |
| it02a_optics_focus | 78/102 (1) | 43/102 (9) | 69/102 (0) | 0.0889 (+3) | 0.0718 (-2) | 0.27 (-1) | 0.0133 (-3) | 48.2 (-5) | 0.114 (-2) | 0.749 (-8) | 0.378 (-1) | 0.00917 (+5) | 0.0308 (+13) | 0.0847 (-13) |
| it02b_venules | 78/102 (1) | 40/102 (10) | 70/102 (0) | 0.0856 (+2) | 0.09 (+2) | 0.253 (-2) | 0.0093 (-6) | 43.4 (-6) | 0.103 (-2) | 0.565 (-10) | 0.369 (-2) | 0.00586 (+3) | 0.0149 (+2) | 0.0568 (-24) |
| it03a_pairs_uturn | 76/102 (1) | 37/102 (14) | 65/102 (1) | 0.0977 (+6) | 0.0767 (-1) | 0.33 (+2) | 0.0194 (+2) | 50.8 (-4) | 0.133 (-1) | 0.829 (-7) | 0.402 (-1) | 0.00393 (+1) | 0.0177 (+4) | 0.0693 (-19) |
| it03b_pairs_uturn_venules | 72/102 (1) | 44/102 (14) | 66/102 (1) | 0.094 (+4) | 0.0905 (+2) | 0.305 (+1) | 0.0152 (-1) | 51.2 (-4) | 0.118 (-1) | 0.635 (-9) | 0.402 (-1) | 0.00139 (-0) | 0.00847 (-2) | 0.0519 (-26) |
| it04a_tips | 62/102 (7) | 30/102 (26) | 51/102 (3) | 0.122 (+13) | 0.135 (+10) | 0.417 (+7) | 0.0291 (+10) | 47.3 (-5) | 0.108 (-2) | 0.694 (-8) | 0.455 (+0) | 0.00195 (+0) | 0.00704 (-3) | 0.0393 (-30) |
| it04b_tips_more_arterioles (folder it04b_tips_arcade) | 66/102 (8) | 27/102 (30) | 53/102 (3) | 0.133 (+16) | 0.131 (+10) | 0.476 (+10) | 0.0399 (+19) | 45 (-6) | 0.102 (-2) | 0.801 (-7) | 0.461 (+0) | 0.00153 (-0) | 0.00553 (-4) | 0.0396 (-30) |
| it05a_t8_back06 | 75/102 (1) | 38/102 (14) | 72/102 (2) | 0.0861 (+2) | 0.0894 (+2) | 0.29 (+0) | 0.00959 (-6) | 46.6 (-5) | 0.12 (-1) | 0.762 (-7) | 0.374 (-1) | 0.00203 (+0) | 0.00957 (-1) | 0.0517 (-26) |
| it05b_t8_back03 | 78/102 (1) | 45/102 (13) | 71/102 (2) | 0.0911 (+4) | 0.0884 (+2) | 0.285 (-0) | 0.0138 (-2) | 49.7 (-4) | 0.111 (-2) | 0.667 (-9) | 0.394 (-1) | 0.0013 (-0) | 0.00896 (-2) | 0.0502 (-26) |
| it06a_turn_grain | 73/102 (2) | 40/102 (16) | 64/102 (1) | 0.0935 (+4) | 0.0917 (+2) | 0.342 (+3) | 0.0134 (-3) | 45.4 (-5) | 0.125 (-1) | 0.882 (-6) | 0.398 (-1) | 0.000783 (-1) | 0.00526 (-4) | 0.105 (-5) |
| it06b_turn_grain_noturnA | 75/102 (2) | 42/102 (17) | 66/102 (2) | 0.0929 (+4) | 0.0898 (+2) | 0.331 (+2) | 0.0132 (-3) | 45 (-6) | 0.129 (-1) | 0.741 (-8) | 0.405 (-1) | 0.00233 (+0) | 0.00831 (-2) | 0.106 (-5) |
| it07a_focusdepth_mu145 | 78/102 (1) | 41/102 (13) | 65/102 (1) | 0.0871 (+2) | 0.0873 (+1) | 0.326 (+2) | 0.0103 (-5) | 45 (-6) | 0.115 (-2) | 0.861 (-6) | 0.394 (-1) | 0.000578 (-1) | 0.00348 (-6) | 0.104 (-6) |
| it07b_plus_art_sheets | 76/102 (1) | 45/102 (11) | 68/102 (1) | 0.0862 (+2) | 0.082 (+0) | 0.287 (+0) | 0.00949 (-6) | 44.5 (-6) | 0.108 (-2) | 0.783 (-7) | 0.398 (-1) | 0.002 (+0) | 0.0079 (-3) | 0.105 (-6) |
| it08a_bypass110_turn05 | 70/102 (2) | 39/102 (17) | 55/102 (2) | 0.105 (+8) | 0.0894 (+2) | 0.354 (+4) | 0.0197 (+2) | 49.6 (-4) | 0.104 (-2) | 0.797 (-7) | 0.394 (-1) | 0.00135 (-0) | 0.00753 (-3) | 0.109 (-4) |
| it08b_bypass110_turn05_back06 | 69/102 (1) | 35/102 (19) | 57/102 (1) | 0.103 (+7) | 0.0921 (+2) | 0.373 (+5) | 0.0197 (+3) | 49.2 (-4) | 0.112 (-2) | 0.7 (-8) | 0.412 (-1) | 0.00161 (-0) | 0.00645 (-4) | 0.106 (-5) |
| it09 (package defaults) | 75/102 (1) | 42/102 (12) | 61/102 (1) | 0.0909 (+3) | 0.086 (+1) | 0.329 (+2) | 0.0136 (-3) | 46.9 (-5) | 0.0947 (-3) | 0.914 (-6) | 0.411 (-1) | 0.00193 (+0) | 0.00824 (-2) | 0.106 (-5) |
| it10_mu135_tilt018_grain02 | 80/102 (1) | 43/102 (12) | 73/102 (1) | 0.086 (+2) | 0.0867 (+1) | 0.317 (+2) | 0.0112 (-5) | 46.6 (-5) | 0.0999 (-2) | 0.949 (-5) | 0.404 (-1) | 0.00174 (-0) | 0.0115 (-0) | 0.142 (+9) |

(z of the real dense site's own spread; e.g. "0.0578 (-7)" is 0.058, 7 dense-site SDs below. `drop`, `weak`,
`r_cv`: frames against their own averages.)

What each step changed and did:

| step | change | effect |
|---|---|---|
| it00 | the correctness fixes only (the review's baseline) | 92 / 102 against the 9 stills, but against the dense site 35 / 102: everything 0.5-0.8x too light, no black vessels, a flat focus, single frames losing 2.4 % of the average's medium vessels (real 0.2 %) |
| it01 | mu 0.12 -> 0.15, bypass length 60 -> 150 d_c; frame filling: single cells, continuous from r 1.2, fewer empty / stop-go capillaries | contrast overshoots (F SD 0.115 against 0.080, wide vessels 0.37 against 0.29); frame dropout fixed (0.24 % against 0.19 %) |
| it02a | mu 0.17, bypass 80; the focal surface | contrast about right; focus variation over tiles 0.06 -> 0.11 (real 0.15) |
| it02b | + venules: wider tips (r 1.0-1.6), denser (spacing 20), 3 sheets, roots every 120 d_c | more medium vessels, but trees made U-turns with right-angle kinks |
| it03 | + long, close pairs (through forks, gap 0.5-1.5); no growth back towards a tree's root | no U-turns, but the trees formed a comb: all hanging from the top and bottom borders |
| it04 | tips r 1.2-2.0; (b) also more arterioles (spacing 60, roots every 180 d_c) | far too dark (F SD 0.12-0.13); rejected |
| it05 | artefact fixes: incline cap 1.4, minimum bend radius 2 d_c, contraction 2 (R_u + R_v), smooth pathological dilation | clubs, V kinks and the worst knots gone; 45 / 102 against the dense site |
| it06 | sheets turned from the dominant direction; red-cell aggregates in frames | vessels cross at many angles; frame graininess 0.05 -> 0.105 (real 0.119) |
| it07 | focus depth 6-35 d_c, mu 0.145; (b) 2 arteriolar sheets | (b) 45 / 102 but arteriole density 2x Houben's: not kept |
| it08 | bypass 110 | too dark again; rejected |
| it09 | the package defaults at that point (+ companion arterioles beside venules) | parallel neighbours 0.91 (baseline 0.81; real 1.38) |
| it10 | mu 0.135, tilt up to 0.18, grain 0.2 | 80 / 102 against the 9 stills, 43 / 102 against the dense site, frames 73 / 102; then grain 0.2 -> 0.16 (graininess 0.142 against 0.119) |

Where the calibrated values live: `anatomy.AnatomyParams` defaults (each
marked calibrated, with the value it replaced), `scene.CALIBRATED_OPTICS`
(s0 1.55 px (v1 integration; 1.8 before), scatter 0.015 px/d_c (v1; the
renderer's 0.03 before), halo 0.45, incline cap 1.4, mu 0.135, bypass length 80 d_c),
`scene.FOCUS_DEPTH / FOCUS_TILT / FOCUS_CURV` (and, v1, `FOCUS_GAIN / FOCUS_MODES_SD /
FOCUS_MODES_WAVE / FOCUS_JITTER / TEX_DEPTH / REG_ERROR / REG_CV / REG_FIELD_PX /
TEX_BYPASS`), `scene.CALIBRATED_FILL` (filling) and `scene.CALIBRATED_IMAGING`
(texture x 0.85). The renderer and
the filling keep their own neutral defaults. All stay within the plan's
physiological ranges except where the table says so: venule length density
1 / 20 d_c against Houben's 1 / 26 (the frame's density comes out 0.04-0.05
with the wider tips), the incline cap and the focal surface (no data). (The
companion wall gap of this pass, 0.5-1.5 (r_a + r_v) against the plan's
1.5-4, was replaced in the v1 integration by 1.2-2.5, measured on the stills.)

### First pass (integration, before the review)

| step | change | effect (averages, 3 seeds, 9 stills) |
|---|---|---|
| baseline | module defaults, k = 3 | 65 / 102 pass |
| k, optics | k 3 -> 4; s0 1.4 -> 1.8 px, halo 0.3 -> 0.45 | 79 / 102; FWHM and edge widths pass |
| deep layer, shape | more deep trunks; weaker orientation pull; tortuosity 0.08 -> 0.2; radius variation 0.1 -> 0.18 with a shorter correlation; pairs 0.3 -> 0.6; texture x 0.85 | 88 / 102 |
| orientation | roots equally on both borders | 89-91 / 102, anisotropy passes |

## Correctness review fixes (2026-09-30)

| defect (review) | fix | regression tests |
|---|---|---|
| 1. short vessels between two junctions rendered 4-34 % too light (overlap levels from the middle piece) | every junction stub is cut into segments of nearly constant staircase and blur, each with its own levels; the review's 8 worst cases now 0.3-3.7 % (the residual: a 12 px vessel 14 px wide) | `test_render.py::test_short_steep_link_between_junctions` (0.7 %; 39 % with one segment per stub) |
| 2. siblings overlapping in 3-D beyond R were added, and `check()` exempted 4 max(r) + 2 d_c | render: the union region extends to the measured overlap run (`junction_overlap_run`); check: the geometric `junction_reach`, both samples within it; the anatomy's overlap pass uses the same rule and clears them | `test_render.py::test_sibling_overlapping_beyond_the_angle_radius_is_unioned`, `test_graph.py::test_sibling_running_inside_another_beyond_the_junction_is_flagged`, `::test_siblings_overlapping_only_near_their_node_pass`, the anatomy contract tests |
| 3. visibility CNR about 5x too generous; visible length not contiguous | like-for-like band response / band RMS (`scene.band_response`); the longest contiguous visible run | `test_scene.py::test_visibility_is_like_for_like`, `::test_visible_run_is_contiguous`, `::test_visibility_tiers` |
| 4. `oracle_od` not a valid reference at merged clusters and crop edges | the oracle's union is decided per fine pixel (neighbours at a node or through one vessel, overlapping in 3-D: max; otherwise add) | `test_render.py::test_oracle_and_closed_form_are_the_union_at_merged_forks[1.5, 4.0]`, `::test_oracle_does_not_depend_on_the_crop` |
| 5. single-frame junctions 1-8 % off | the stub segments follow the filling (now 0.1-1.3 % on the review's 12 frame junctions) | `test_render.py::test_frame_filling_at_a_junction` |
| 6. `check()` ignored a vessel overlapping itself | self rule in `check()`; `anatomy._resolve_self` bumps or removes | `test_graph.py::test_a_vessel_overlapping_itself_is_flagged` |
| 7. tests that would pass while wrong | new `test_graph.py` (each invariant must be caught), oracle tests at k = 1.5 / 8, incline at a fork, the "average fuller" test on perfused capillaries' empty length, lumen order by local depth, reproducibility of the labels | `test_graph.py` (8 tests), `test_render.py::test_fork_at_other_scales[1.5, 8.0]`, `::test_dive_right_after_a_fork`, `test_scene.py::test_scene_images_and_shapes`, `::test_labels_are_consistent`, `::test_scene_is_reproducible` |
| 8. red-cell columns denser than packed cells (up to 2.8x systemic) | phi >= h / h_col_max (2) | `test_imaging.py::test_filling_columns_are_not_denser_than_packed_cells` |
| 9. vessels just outside the frame tiered out_of_frame, coded invisible | pieces count where their lumen reaches the frame; out_of_frame skirts coded dont_care | `test_scene.py::test_vessel_just_outside_the_frame_is_scored` |
| 10. flows stale after the late curvature refit | flow solved again at the end of `generate` | `test_anatomy.py::test_flows_match_the_final_geometry` |
| 11. `still_frame.tif` float32; docstring k | uint16; k = 4 | `test_scene.py::test_save_and_load` |

Defects 1, 2 (render side) and 4 were fixed in `render.py` before this pass's
tests were written; they were verified with the review's own scripts
(`_scratch/review/check_*.py`; outputs in `_scratch/fix/`).

