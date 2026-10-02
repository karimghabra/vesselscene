"""The anatomy of a synthetic scene: a physiological vessel network as a VesselGraph.

What it builds, in capillary diameters d_c (graph.py), for one still of the
bulbar conjunctiva near the canthus (the limbus usually out of frame):

* a padded domain around the frame (large vessels enter and leave the frame),
  a smooth orientation field (a dominant direction, near vertical by default)
  and a smooth log-normal density modulation;
* a deep (episcleral) layer: a few large, straight trunks running across the
  domain (veins wide and straight, arteries narrower and more tortuous),
  sparse branches, perforating branches that dive out of sight, risers that
  climb into the conjunctiva and feed or drain its trees, arteriovenous
  anastomoses (AVAs) and vein-vein anastomoses; with layout 'web' also the
  deep plexus, many medium veins joined by anastomoses (an interlacing venous
  network, Selbach 2005);
* layout 'web' (v1, the presets' default): the conjunctival layer is a web of
  long vessels (_Web).  Long arterioles and venules run from border to border
  across the whole frame and beyond (fairly straight, gently curving, about
  the dominant radial direction with a spread of angles), fork in Y's whose
  branches run on to the border, and are joined by arcades (anastomoses
  between neighbouring vessels of one side, leaving each obliquely); every
  arteriole has a companion venule (some a venule on each side) and some
  venules a companion arteriole, running beside their host for 150-500 d_c
  at a wall gap of 1.2-2.5 (r_a + r_v) and meandering with it (_shared_meander);
  bundles of 3-5 thin parallel vessels; large collecting venules (to 16 d_c,
  at least one per scene, their lumens flattened: _aspects) with tributaries
  and companions; a long vessel meeting another at a shallow angle turns to
  run beside it (web_para_mode).  Each
  long vessel is cut into stretches, and the stretches of all long vessels
  and episcleral systems get their depths together (_allocate): where two
  cross they lie at depths that clear each other, thinner vessels shallower,
  a vessel changing depth gently along its course.  Short twigs (terminal
  arterioles / venules, space colonization with a limited reach) leave the
  medium long vessels, climb to twig sheets under the capillary mesh and
  feed / drain it.  A long vessel ends at every junction on its way (one
  vessel = one edge), but its centreline runs through them tangent-
  continuously (_web_through, _web_c1);
* v2 (2026-10-01, anatomy detail): a long vessel meanders as one course
  through its junctions (_course_meander; wide ones bend no tighter than
  wide_bend_D diameters, a collecting vein keeps its course through its
  confluences and the tributary takes the angle) and undulates in depth
  along it (_undulations, cleared by the allocation); calibre varies over a
  few diameters, more in venules than arterioles; bundles of 2-4 members are
  irregular (unequal calibres, drifting spacing, members joining and
  leaving); a collecting vein runs through the central part of the frame and
  never beside or across another one; courses, companions and bundles keep
  off long overlaps in projection with other long and deep vessels (dark
  spindles); preferential (thoroughfare) channels widen the capillary
  routes between arteriole and venule tips (_channels: the visible fine
  network and its loops);
* layout 'forest' (v0): conjunctival arteriole and venule forests grown by
  open space colonization (Runions 2005) from roots on the domain border and
  from the riser tops, in separate depth "sheets" (one arteriolar, three
  venular), so vessels of one sheet never cross each other while vessels
  of different sheets cross freely at different depths; half the sheets grow
  along a direction turned 60-90 deg from the dominant one, and no tree grows
  back towards its root; arteriole-venule pairs: some arterioles get a
  companion venule, and some venules a companion arteriole, running beside
  them through their forks for 40-250 d_c at a wall gap of 0.5-1.5 (r_a + r_v);
* a shallow capillary mesh (Poisson-Voronoi, stretched along the dominant
  direction, some edges dropped) that arteriole tips feed (forks) and venule
  tips drain (confluences); side capillaries leave the small tree vessels;
* pruning of everything on no inlet -> outlet path, so no vessel ends inside
  the domain; one vessel = one edge between forks / confluences;
* calibre by a Murray-type law r0^g = sum ri^g (bottom-up in the conjunctival
  trees from their tips, in the long vessels from where they leave the domain
  upstream, top-down in the deep trees from their trunks; anastomoses take
  no part),
  branch angles from the flow split (Murray 1926, Zamir 1978), tortuosity by
  class, along-vessel calibre variation, depth by layer (larger deeper) with
  smooth ramps where a vessel changes layer, haematocrit (Fahraeus), and a
  Poiseuille solve (graph.solve_flow) that orients every vessel downstream;
* a final pass that removes every 3-D lumen overlap between vessels (smooth
  depth bumps where vessels of different layers cross; where no bump can
  clear a conflict, the thinner, capillary-scale vessel is removed and the
  topology repaired) and every self-overlap, so graph.check(overlap=True) is
  empty; the flow is solved again after the last geometry change.

Scale: the frame is (W/k) x (H/k) d_c at k px per d_c; the image pixel of
a point xy (d_c) is k * xy (pixel centres at integers), so a renderer uses
origin (0, 0).  Haematocrit h is the tube haematocrit relative to systemic
blood (capillaries 0.3-0.8), i.e. already includes the Fahraeus reduction;
each vessel's info carries its tree, Strahler order (signed: arterial side
negative), layer, roles, pair partner(s), flow_as_tree and a relative
velocity (flow / pi r^2) for the single-frame red-cell filling.

Entry points: AnatomyParams (every knob with its source), preset(name, rng)
('healthy', 'pathologic', 'random') and generate(params, seed).  generate is
deterministic for a seed: each stage draws from its own numpy SeedSequence
stream, so switching one module off leaves the others unchanged.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Voronoi, cKDTree

from .graph import VesselGraph, close_pairs, fit_profile

VERSION = "0.1"
STAGES = ("setup", "fields", "deep", "risers", "ava", "sheets", "mesh", "attach", "radii",
          "pairs", "geometry", "depth", "profiles", "resolve", "web", "aspect", "channels")   # appended: the others
                                                                                     # keep theirs
RELAX_TWIGS = 6               # twigs of the fallback for a side that got none (anatomy._twig_plans)
RAMP = -1                     # layer label of points that ramp between their end depths


# =================================================================== parameters
@dataclass
class AnatomyParams:
    """Every knob of the network generator.  Lengths in d_c (capillary
    diameters, about 6 um) unless stated.  "Plan" is the generator plan
    (synthetic_generator_plan.md); its section 3.3 rules table gives the
    sources.  (P) marks my proposals, (C) values to calibrate on the stills."""

    # ---- scale and frame
    k: float = 4.0                          # px per d_c. Plan 3.2: calibration prior LogU[2, 6], central 3-3.5;
                                            # training [1.5, 8].  Calibrated (scene.py, 2026-09-30): 4 fits the
                                            # stills' FWHM p10/p50/p90 and edge widths better than 3 or 4.5
    frame_px: tuple = (1920, 1200)          # (W, H) of the camera frame, px
    pad_frac: float = 0.4                   # domain = frame padded by max(pad_frac * max(FOV), pad_min) on every side
                                            # (plan G1)
    pad_min: float = 150.0
    tissue_margin: float = 60.0             # forests and capillary mesh fill the frame plus this margin; beyond:
                                            # stalks and deep trunks only (P)

    # ---- orientation and density (plan G0/G1)
    orientation_deg: float | None = None    # dominant direction, deg from +x towards +y (image y down); None: drawn
                                            # near vertical
    orientation_sd_deg: float = 12.0        # spread of that draw about 90 deg (realstats: 75-91 deg in 7 of 9 stills)
    orient_field_sd_deg: float = 35.0       # smooth variation of the local direction (C: anisotropy 0.18 +- 0.07;
                                            # calibrated from 15: skeleton anisotropy was 0.36)
    orient_field_corr: float = 250.0        # its correlation length (P)
    orient_weight: float = 0.05             # pull of tree growth towards the local direction (C; calibrated from 0.3)
    density_cv: float = 0.4                 # log-normal density modulation, CV 0.3-0.5 (plan G0)
    density_corr: float = 150.0             # its correlation length, 100-200 d_c (plan G0)
    limbus_sign: int | None = None          # +1 / -1: the limbus lies off-frame along +/- the dominant direction;
                                            # None: drawn

    # ---- depth (rules table, depth rows: epithelium 7 d_c, conjunctival vessels 7-40, episcleral 40-100, larger
    # deeper)
    epithelium: float = 7.0                 # no vessel centre shallower than this + r (Zhang 2013)
    mesh_depth: float = 9.0                 # capillary mesh centre depth, just under the epithelium (pig casts: a
                                            # sheet)
    sheet_gap: float = 1.5                  # extra depth between the lumens of neighbouring sheets (P)
    sheet_quantile: float = 0.95            # a sheet is as thick as this quantile of its vessels' diameters (P)
    tissue_depth_sd: float = 1.5            # smooth tissue-thickness variation shared by mesh and conjunctival sheets
                                            # (P)
    tissue_depth_corr: float = 200.0
    deep_top: float = 45.0                  # shallowest episcleral lumen top (Howlett 2014; Akagi 2019)
    deep_spread: float = 15.0               # extra depth spacing spread between stacked episcleral systems (P)
    deep_depth_sd: float = 4.0              # smooth depth variation of the deep layer (plan G8: GRF SD 2-6)
    z_wobble: float = 0.3                   # along-vessel depth SD in the conjunctiva (P)
    deep_z_wobble: float = 3.0              # along-vessel depth SD of deep vessels (P)
    ramp_angle_deg: float = 30.0            # steepest inclination of layer ramps and fades where there is room;
                                            # a steep tube renders darker (longer chord) (plan G9: > 60 deg is faded)
                                            # (P)
    perforator_angle_deg: float = 40.0      # perforators dive more steeply (P)
    bump_angle_deg: float = 45.0            # steepest depth bump where the overlap pass separates two vessels (P)
    relax_angle_deg: float = 45.0           # tree nodes are moved in depth so no vessel must climb more steeply (P)
    relax_max: float = 6.0                  # ... but by at most this much (d_c): further, they cross other sheets (P)
    dive_rate: float = 1.0 / 500.0          # depth excursions (fades) per d_c of deep vessel (plan G9, P)
    dive_amp: tuple = (15.0, 50.0)          # their amplitude, d_c
    dive_len: tuple = (40.0, 150.0)         # their length, d_c

    # ---- deep (episcleral) layer (plan G2)
    n_deep_veins: float = 2.4               # Poisson mean of vein trunks per domain (C; calibrated from 1.5: too
                                            # few wide blurred vessels, FWHM p90 and sigma*=16 share low)
    n_deep_arteries: float = 1.6            # Poisson mean of artery trunks per domain (C; from 1.0)
    vein_diam: tuple = (8.0, 17.0)          # episcleral vein trunk D (Asanad 2020; W)
    artery_diam: tuple = (5.0, 12.0)        # episcleral artery trunk D (plan 3.3; no data for arteries)
    vein_persistence: tuple = (20.0, 40.0)  # heading correlation length / D (plan G2 (P)); veins straighter (Meyer
                                            # 1988)
    artery_persistence: tuple = (5.0, 15.0)
    vein_heading_sd_deg: float = 8.0        # stationary heading SD of the OU walk (P)
    artery_heading_sd_deg: float = 16.0
    deep_angle_sd_deg: float = 30.0         # deep trunks' direction about the dominant one (P)
    deep_branch_LD: tuple = (10.0, 30.0)    # spacing of deep branches along a trunk / D, log-uniform (plan G2: [8,
                                            # 30] (P); calibrated from (15, 40))
    deep_branch_q: tuple = (0.15, 0.5)      # a branch's share of its parent's flow
    perforator_frac: float = 0.3            # deep branches that dive through the sclera out of sight (Meyer 1988)
    perforator_depth: tuple = (80.0, 140.0) # at least this deep, and below every episcleral system
    ava_density: float = 2.7e-5             # AVAs per d_c^2 of tissue = 0.75 /mm^2 (Selbach 2005: 0.5-1 /mm^2,
                                            # primate; W)
    ava_radius_frac: tuple = (0.3, 0.5)     # AVA calibre / the smaller vessel's (plan 3.3, my proposal)
    vv_anastomoses: float = 0.5             # Poisson mean of vein-vein anastomoses per pair of deep veins (P)
    n_risers_A: float = 1.0                 # Poisson mean of arterial risers from deep arteries (plan G2; Meyer &
                                            # Watson 1987)
    n_risers_V: float = 1.0                 # venous risers into deep veins
    riser_angle_deg: tuple = (20.0, 45.0)   # risers climb at this inclination (plan G2 (P))

    # ---- conjunctival forests (plan G3)
    n_art_sheets: int = 1                   # arteriolar depth sheets (P; 2 was tried in the realism pass of
                                            # 2026-09-30: arteriole length density 2x Houben's, V:A 1.5 against 3)
    n_ven_sheets: int = 3                   # venular depth sheets: lets venules cross venules at another depth
                                            # (critique 2.2; calibrated from 2: more crossing medium vessels
                                            # against the dense full-frame site)
    art_spacing: float = 81.0               # mean spacing of arterioles = 1 / length density 0.0124 /d_c (Houben
                                            # 2005)
    ven_spacing: float = 20.0               # venules: 1 / length density (Houben 2005: 0.0386 /d_c, i.e. 26;
                                            # calibrated to 20: the frame's venule density comes out 0.04 with
                                            # the wider tips); V:A length ~ 3:1
    attractor_spacing: float = 10.0         # space-colonization attraction points, mean spacing (plan G3)
    attractor_aniso: float = 1.0            # their spacing along / across the dominant direction (plan G3: stretched)
                                            # (C; calibrated from 1.3: too anisotropic)
    kill_factor_A: float = 0.7              # kill distance / (sheet spacing): calibrated so arteriole trees in the
    kill_factor_V: float = 0.5              # frame match art_/ven_spacing (healthy seed 0: 0.0124 / 0.040 per d_c)
                                            # (C)
    influence: float = 3.5                  # influence radius / kill distance (plan G3: 3-4)
    sc_step: float = 2.0                    # growth step (plan G3)
    persistence_weight: float = 0.8         # weight of a tip's own heading in its next step (C)
    back_cos: float = -0.3                  # a tree never grows more than acos(back_cos) (~107 deg) away from its
                                            # root's heading: no U-turns back towards the root (a colonization
                                            # artefact: real trees run across the field) (P; realism review 2026-09-30)
    root_spacing_A: float = 250.0           # arterial roots along the main end border (P)
    root_spacing_V: float = 120.0           # venous roots (P; calibrated from 150)
    art_root_bias: float = 0.5              # share of arterial roots on the fornix side (posterior conjunctival
                                            # arteries; the anterior ones enter from the limbal side and the two
                                            # territories overlap).  Calibrated from 0.75: trees rooted mostly on
                                            # one border all ran one way (skeleton anisotropy 0.33 against 0.18)
    sheet_turn_prob: float = 0.5            # a sheet's trees grow along a direction turned from the dominant one by
    sheet_turn_deg: tuple = (60.0, 90.0)    # this many degrees (either way), so trees of different sheets cross at
                                            # many angles as in the stills (all along one direction they formed a
                                            # comb; realism review 2026-09-30) (P)
    ven_root_bias: float = 0.5              # share of a venous sheet's roots on its main end (drawn per sheet) (P;
                                            # calibrated from 0.75, as art_root_bias)
    min_branch_spacing: float = 8.0         # no two forks closer along a vessel (plan invariant 8)
    min_branch_angle_deg: float = 25.0      # a sprout leaves its sibling at least this angle
    clearance: float = 3.0                  # growth keeps nodes of other branches of the sheet this far (P)
    tip_r_A: tuple = (0.5, 0.9)             # terminal arteriole radius (pre-capillary D 1-2, Koutsiaris 2010)
    tip_r_V: tuple = (1.0, 1.6)             # terminal venule radius (post-capillary D 1.5-4, Koutsiaris 2015; Moka
                                            # 2019; calibrated from (0.75, 1.2): too many thin twigs, too few
                                            # medium vessels against the dense site)
    gamma_A: tuple = (2.1, 2.9)             # arterial radius exponent, per scene (Luo 2017; Kreitner 2024)
    gamma_V: tuple = (1.7, 2.3)             # venous (Luo 2017; Maibier 2016; Koutsiaris 2007: Q ~ D^2)
    art_diam_max: float = 7.0               # conjunctival arteriole D 1-7 (Khansari 2016)
    ven_diam_max: float = 12.0              # venules 1-9, collecting venules 4-12 (Khansari 2016; Wang 2016)
    side_spacing: float = 40.0              # side capillaries leave small tree vessels every ~30-50 d_c (plan G4
                                            # (P)); 0: none
    side_rmax: float = 1.2                  # ... only vessels thinner than this radius
    attach_max: float = 60.0                # a tip joins the mesh within this distance, or is pruned

    # ---- arteriole-venule pairs (plan G3; user: the doubled lines are real pairs)
    p_pair: float = 0.7                     # share of arteriole segments with a companion venule (plan G3: 0-0.6;
                                            # raised from 0.3, then 0.6: parallel neighbours are too rare; few
                                            # qualify)
    pair_gap: tuple = (0.5, 1.5)            # wall gap / (r_a + r_v) (plan G3: 1.5-4; realism review: the stills'
                                            # doubled lines run closer, 0.5-1.5)
    pair_length: tuple = (40.0, 250.0)      # run length, d_c (plan: "tens of d_c"; review: runs of 100-400 d_c; the
                                            # companion follows the arteriole through its forks)
    pair_ratio: tuple = (1.2, 2.0)          # r_v / r_a (Khansari 2016; Jiang 2014; used as a check in the plan)
    p_pair_V: float = 0.3                   # share of venule segments with a companion arteriole (the same pairs
                                            # seen from the venous side: long venules had none) (P; review T5)
                                            # (layout 'web': the pairs are the web's; these apply to the twigs x
                                            # twig_pair_frac)

    # ---- v1 layout: the conjunctival web (realism v1, 2026-09-30).  The dense stills are a web of long, fairly
    # straight vessels crossing the whole field and each other, joined by forks and arcades; v0's forests grown
    # by space colonization from the domain border were trees thinning into the mesh (and a comb hanging from
    # the border).  Bulbar conjunctival arterioles and venules run long radial courses from the fornix towards
    # the limbus and anastomose with their neighbours (arcades) (Meighan 1956; Meyer & Watson 1987; Singh 2021;
    # "orderly anastomosing networks", Jiang 2014; Wang 2016); each arteriole has one or more companion venules
    # (Singh 2021); their terminal branches feed / drain the capillary bed just under the epithelium, and larger
    # vessels lie deeper (Kiseleva 2022).
    layout: str = "web"                     # 'web' (v1) or 'forest' (v0: border-rooted forests, kept for comparison)
    web_ven_spacing: float = 20.0           # long venules: 1 / their length density over the tissue region (d_c).
                                            # Venules overall 1/26 (Houben 2005); the twigs add the rest (P)
    web_art_spacing: float = 70.0           # long arterioles (arterioles overall 1/81, Houben 2005) (P)
    web_angle_sd_deg: float = 18.0          # their direction about the dominant (radial) one (P)
    web_iso_frac: float = 0.12              # share of long vessels with any direction (arcades, circumferential
                                            # stretches; real skeleton anisotropy 0.18 +- 0.07) (P)
    web_persistence: tuple = (40.0, 120.0)  # heading correlation length, d_c: long, gently curving courses (P)
    web_heading_sd_deg: float = 14.0        # stationary heading SD of that walk (P)
    web_r_V: tuple = (0.65, 1.4)            # radius of a long venule where it leaves the domain upstream, log-uniform
                                            # (D 1.3-2.8; Khansari 2016: venules D 1-8.6, mean 3); it widens downstream
                                            # by Murray as its twigs, forks and arcades join
    web_r_A: tuple = (0.45, 0.85)           # long arterioles (D 0.9-1.7; Khansari 2016: 1-7, mean 2.5; V/A
                                            # calibre about 1.5, within the paired 1.2-2)
    web_grow_max: float = 2.0               # a long vessel widens (by Murray, as twigs and forks join it) at most
                                            # this much across the domain (P)
    web_fork_spacing: tuple = (200.0, 600.0)    # Y-forks along a long vessel, log-uniform (P)
    web_fork_q: tuple = (0.1, 0.35)         # the branch's share of its parent's r^g (both branches run on) (P)
    web_arcade_spacing: float = 180.0       # arcades between neighbouring long vessels of one side: one per this
                                            # length of long vessel (P)
    web_arcade_r: tuple = (0.3, 0.6)        # arcade radius / the smaller long vessel's (P)
    web_arcade_len: tuple = (25.0, 80.0)    # arcade chord (d_c) (P)
    web_para_clear: float = 8.0             # a long vessel's course (and its Y-branches) that comes within this distance
                                            # (d_c) of another long vessel running within web_para_deg of its heading
                                            # turns (web_para_turn_deg per 2 d_c step) to run beside it: vessels of one
                                            # plane that meet at a shallow angle keep side by side rather than cross
                                            # (v1 fix, realism review T1: crossings under 30 deg were 1.7-2.3x the
                                            # stills', drawn as dark spindles).  6 seeds: shallow crossings +3.7 -> +0.5
                                            # SD, dark knots per 100 lambda 1.20 -> 0.87 (real 0.47), parallel pairs
                                            # pair_frac -12.5 -> -8.2.  Turning away instead ('away') removed the side by
                                            # side runs (par_excess -7.2 SD): rejected (P; 0: off)
    web_para_deg: float = 30.0
    web_para_turn_deg: float = 3.0
    web_para_mode: str = "align"           # 'align' (turn to run beside it) or 'away' (turn away from it)
    web_para_min: float = 1.5               # ... but closer than this + 1.6 x the neighbour's nominal radius (centre to
                                            # centre, d_c) it turns away: aligned on top of its neighbour it overlapped
                                            # it in projection over tens of d_c (v2; at 4 the side-by-side runs, the
                                            # stills' doubled lines, were lost: truth pairing 0.61 -> 0.42)
    web_dark_r: float = 1.0                 # long overlaps in projection are avoided where both vessels are at least this
                                            # wide (r, d_c): dark spindles (v2; P)
    deep_para_clear: float = 3.0            # a long vessel's course turns away from an episcleral vessel it would run
    deep_para_deg: float = 25.0             # along within (its radius + this) d_c and this angle: long overlaps of a
                                            # sharp vessel on a blurred deep one drew dark spindles, and a sharp line
                                            # on a grey band's axis (review v1 T1, T10; 0: off) (v2; P)
    web_tort_gain: float = 0.6              # tortuosity gain of the long vessels (on top of their class's) (P; v2: 1.0
                                            # while each vessel's meander was pinned at every junction on its way;
                                            # one meander along the whole course (_course_meander) at 1.0 drew S-bends
                                            # of 3-4 D and pulled companions off their hosts: pair_frac -11 -> -19 SD)
    web_depth_span: float = 34.0            # depth range of the long vessels, below the twig sheets (d_c): each is
                                            # placed so that the long vessels it crosses lie at other depths (P)
    collect_rate: float = 0.48              # large collecting venules: Poisson mean per 1e5 d_c^2 of tissue region
                                            # (1.2 per full frame at k = 4; P; the dense stills' shallow black veins
                                            # 45-60 px wide)
    collect_min: int = 1                    # at least this many per scene, the rest Poisson (same mean): each of the 4
                                            # dense stills has 1-2 black veins; at Poisson(1.2) 30 % of the healthy
                                            # frames had none (realism review v1, T2/T3) (P)
    collect_r: tuple = (3.5, 5.5)           # their radius where they leave the domain upstream (D 7-11; v1 integration:
                                            # was 3.0-4.5, the widest dark vessels 46 px against the dense stills' 62)
    collect_diam_max: float = 16.0          # and their cap (collecting venules D 4-12, Wang 2016 / Singh 2021;
                                            # episcleral veins 8-17, Asanad 2020) (W)
    collect_tributaries: tuple = (100.0, 300.0)  # tributary long venules join them this far apart (P)
    web_p_pair_A: float = 1.0               # long arterioles with a companion venule (Singh 2021: each artery has
                                            # one or more matching veins; user: the doubled lines are real)
    web_p_pair_V: float = 0.7               # long venules with a companion arteriole (P; v2: 0.35; pair_frac -8 SD with
                                            # half the bundles, and companions that would lie on another vessel are
                                            # now not made)
    web_p_pair_A2: float = 0.7              # long arterioles with a venule on each side (venae comitantes) (P; v2: 0.3)
    p_collect_companion: float = 0.9        # collecting venules with a companion arteriole (P; review T5; v2: 0.7)
    collect_pair_gap: tuple = (0.2, 1.0)    # its wall gap / (r_a + r_v): closer than web_pair_gap in these units,
                                            # since r_v is large (a wall gap of 1-5 d_c beside a vein of r 4-5); a
                                            # parameter since v1 fix (was hard-coded; correctness review v1,
                                            # defect 7) (P)
    web_pair_gap: tuple = (1.2, 2.5)        # wall gap / (r_a + r_v); where their lumens could touch the two lie at
                                            # different depths (_allocate).  v1 integration: was 0.5-1.5 (review, by
                                            # eye); measured on the dense stills, a thin line's nearest parallel
                                            # neighbour is 16-24 px away (22-27 % of thin line length; 8-12 px:
                                            # 0.2-1.2 %), i.e. 4-6 d_c centre to centre; at 0.5-1.5 the pairs sat
                                            # 9-14 px apart and merged into one blurred line (plan G3: 1.5-4)
    web_plain_min: float = 0.3              # at least this share of each side's length budget goes to plain long
                                            # vessels (not collecting veins, bundles, companions): they carry the
                                            # twigs (P)
    web_pair_run: tuple = (250.0, 700.0)    # the companion runs beside its host for this long (review), then
                                            # diverges; it follows the host through its forks (v2: was 150-500;
                                            # pair_frac -7.7 SD, pair_run5 -2.3 SD) (P)
    web_pair_diverge_deg: tuple = (8.0, 20.0)
    web_pair_ratio: tuple = (1.0, 1.6)      # r_v / r_a of a pair (Jiang 2014: 'about 1:2'; unpaired means 1.2,
                                            # Khansari 2016); the thinner stays visible beside its partner (P)
    bundle_rate: float = 1.4                # bundles of thin parallel vessels: Poisson mean per 1e5 d_c^2 of tissue
                                            # region (3.5 per full frame, half of them of 3 lines: about half v1's
                                            # rate of 3-5 line bundles; user: 3-5 thin parallel lines; an arteriole
                                            # with its venae comitantes) (P; v2: was 1.2 bundles of 3-5, set while
                                            # bundles were half destroyed; intact, they were too frequent: bundle_frac
                                            # +4.9 SD; the dense stills' long thin lines run in pairs and triples more
                                            # than in fours and fives)
    bundle_size: tuple = (2, 3)             # an arteriole with one or two venae comitantes (v2: 3-5; at 2-4 and rate 1.0
                                            # the pairs fell, pair_frac -7 -> -14 SD)
    bundle_r: tuple = (0.35, 1.0)           # member radius, log-uniform (D 0.7-2.0; v2: was uniform 0.55-0.9: identical
                                            # lines)
    bundle_drift: float = 1.2               # SD of each member's lateral drift along the run (d_c): spacings drift (v2)
    bundle_drift_corr: tuple = (60.0, 150.0)    # its correlation length (d_c) (P)
    bundle_stagger: float = 0.35            # a member joins / leaves within this share of the run from each end (v2; P)
    bundle_gap: tuple = (1.5, 3.0)          # wall gap between neighbours (d_c; closer members each need a depth of their
                                            # own all along).  v1 integration: was 0.8-1.8 (lines 6-10 px apart, by
                                            # eye); the measured neighbour spacing above: 12-18 px centre to centre
    bundle_run: tuple = (150.0, 500.0)
    twig_spacing: float = 40.0              # twigs (terminal arterioles / venules to the capillary mesh) leave the
                                            # long vessels about this far apart (P)
    twig_rmax: float = 2.5                  # ... long vessels up to this radius (P)
    twig_rmin: float = 0.7                  # ... and from this radius: a twig is thinner than its vessel (the thinner
                                            # long vessels are conduits between the capillary bed's feeders) (P)
    twig_len: tuple = (20.0, 50.0)          # how far a twig reaches from its long vessel (path length) (P)
    twig_stalk: tuple = (15.0, 30.0)        # its first stretch, where it climbs from the long vessel's depth to
                                            # its sheet under the capillary mesh (P)
    twig_sheets_A: int = 1                  # twig depth sheets (as v0's forest sheets)
    twig_sheets_V: int = 1
    twig_spacing_A: float = 90.0            # colonization spacing of the twig sheets (kill distance from it): a
                                            # twig branches a few times, it is a terminal vessel (P)
    twig_spacing_V: float = 70.0
    twig_tip_r_A: tuple = (0.5, 0.8)        # twig tips: pre-capillary arterioles D 1-2 (Koutsiaris 2010)
    twig_tip_r_V: tuple = (0.7, 1.0)        # post-capillary venules D 1.5-4 (Koutsiaris 2015; Moka 2019)
    twig_pair_frac: float = 0.0             # p_pair / p_pair_V for the twigs (v0's companions of forest vessels)
    plexus_spacing: float = 80.0            # deep plexus: medium episcleral veins, an interlacing venous network
                                            # (Selbach 2005; review T11), 1 / their length density over the tissue
                                            # region (about 6 veins per full frame) (P; v1 integration: was 50, 10
                                            # per frame: blurred 16-24 px profiles were 21 % of all against 13 % in
                                            # the dense stills, and 27-46 % of the profiles were deep vessels)
    plexus_diam: tuple = (3.0, 7.0)         # their D (P)
    plexus_persistence: tuple = (8.0, 20.0) # heading correlation length / D (wavier than the trunks) (P)
    plexus_heading_sd_deg: float = 14.0
    plexus_angle_sd_deg: float = 40.0
    plexus_links: float = 1.5               # vein-vein anastomoses per plexus vein (P)
    deep_depth_span: float = 55.0           # depth range of the episcleral systems (d_c; below conj_bottom + 3 sd)

    vein_aspect: tuple = (0.7, 0.9)         # a wide vein's lumen depth / width (elliptical, flattened in the tissue plane:
                                            # thin-walled vessels at low transmural pressure); v1 fix, realism review T2
                                            # (the widest veins' plateau 1.6-1.9x the stills').  With collect_min 1 this
                                            # matched the dense site's overall and wide contrast (F_sd -0.1, c_wide +2.6
                                            # SD on 6 seeds); (0.5, 0.8) from r 2.5 was too dark (F_sd +3.4, c_wide +5.1)
                                            # (P, C).  v2 integration: 0.4-0.65 -> 0.7-0.9.  The v1 flattening made up
                                            # for v1's chord law; with v2's (f_spectral 0.35, the renderer's depth
                                            # compositing) the veins came out too light (6 seeds, z against the dense
                                            # site: widest vein -0.365 Np (+4.3; real -0.471), F_sd -3.2, dark128_0.45
                                            # -10.0, 5 of 6 scenes with no long vessel darker than -0.33 Np); round
                                            # veins overshoot (-0.521, F_sd +1.9, dark128 +3.7); 0.7-0.9: -0.472, F_sd
                                            # -0.0, dark128 -2.0, every scene's darkest long vessel -0.35 .. -0.38
    vein_flat_r: tuple = (1.5, 4.0)         # venous vessels flatten from this radius (aspect 1) to that (the minimum) (P)
    # ---- v2 (2026-10-01): wide trunks curve smoothly through their junctions (realism review v1, T4: collecting
    # veins turned 20-70 deg at thin tributaries; wide_kink +5.3 SD against the dense site, 62 % of the kinked wide
    # path on collecting veins, 87 % within 2 widths of a node)
    wide_bend_D: float = 6.0                # a wide vessel (median r >= wide_bend_r) bends no tighter than this many
    wide_bend_r: float = 2.0                # diameters (large veins run in gentle arcs) (P; review: 3-5 D; 4 left
                                            # wide_kink +6 SD: the stills' wide paths never turn > 20 deg over 4
                                            # widths, i.e. no bend radius under ~6 widths)
    web_depth_undulation: float = 4.0       # SD (d_c) of a long vessel's own depth undulation along its course (half for
    web_depth_undulation_corr: tuple = (60.0, 150.0)    # collecting veins) and its correlation length (d_c): the
                                            # stroma's vessels rise and dive, so they fade and sharpen along their
                                            # length (v2; realism review v1 T3 'uniform wires'; v1 kept one depth per
                                            # 100-180 d_c stretch) (P)
    collect_persist: tuple = (150.0, 300.0) # a collecting venule's heading correlation length (d_c) and heading SD (deg):
    collect_heading_sd_deg: float = 8.0     # gently arcing courses (v1: 200-400 and 4 deg: ruler-straight across the
                                            # frame; the dense stills' big veins arc by 20-60 deg over it) (P)
    collect_depth_u: tuple = (0.35, 0.65)   # a collecting venule's preferred depth within the web's range (0 top, 1
                                            # bottom) (P; v1 drew it by size among the long vessels with noise SD
                                            # 0.15 of the range: a 10 d_c vein at 20 d_c rendered at -0.72 Np, others
                                            # near the bottom too light; 0.7-0.95 left 2 of 8 healthy frames without a
                                            # long vessel darker than -0.33 Np)
    collect_center: float = 0.6             # a collecting venule's course passes through the central part of the
                                            # frame (this share of its width and height): every dense still has its
                                            # black vein well inside the frame; v1 drew 2 of 12 healthy frames whose
                                            # vein only grazed a corner (darkest long path > -0.33 Np) (P)

    # ---- capillary mesh (plan G4)
    mesh_density: float = 6.0e-4            # Poisson-Voronoi seeds per d_c^2 (from Kvernebo 2022 FCD 5.2 /mm; cells
                                            # ~46 d_c)
    mesh_aspect: tuple = (1.0, 1.5)         # cells stretched along the dominant direction (plan G4: 1-3; calibrated
                                            # from (1, 2))
    mesh_drop: float = 0.1                  # share of mesh edges dropped (plan G4: 0-25%)
    cap_diam: float = 1.0                   # capillary D median (Koutsiaris 2007; Moka 2019)
    cap_cv: float = 0.28                    # its CV (Pries 1995)
    cap_diam_range: tuple = (0.75, 1.5)
    channel_frac: float = 1.0               # share of the arteriole / venule tips whose shortest route through the mesh
                                            # to a tip of the other side is a preferential (thoroughfare) channel (v2;
                                            # P: the dense stills show a faint polygonal network of thin lines over
                                            # much of the field, and thin zig-zag vessels)
    channel_diam: tuple = (1.3, 2.0)        # its D (8-12 um: metarterioles, thoroughfare channels; Zweifach 1959) (P)
    channel_max: float = 250.0              # the longest such route (d_c of capillary)
    channel_share: float = 0.25             # at most this share of the mesh's length becomes channels (thoroughfare
                                            # channels are a minority of the capillary bed) (P)
    channel_hct: tuple = (0.55, 0.85)       # its tube haematocrit (relative): the high-flow branch takes more red cells
                                            # (phase separation, Pries 1989) than the capillaries' 0.3-0.8 (P)
    mesh_min_edge: float = 3.0              # shorter Voronoi edges are collapsed (P)
    mesh_bow: float = 0.08                  # capillaries bow sideways by N(0, mesh_bow) * their length (P)
    mesh_sbend: float = 0.05                # ... and bend in an S by N(0, mesh_sbend) * their length (v2; P)

    # ---- geometry (plan G9)
    angle_noise_deg: float = 12.7           # branch-angle noise per daughter (Luo 2017: SD 18 deg on the total)
    tort_amp: float = 0.2                   # A_k: curvature * D SD of the tortuosity process (C; calibrated from
                                            # 0.08: arc/chord over 1-4 lambda was too low)
    tort_corr: float = 6.0                  # its correlation length / D (plan G9: 3-15)
    tort_gain: dict = field(default_factory=lambda: dict(
        capillary=1.0, venule=0.6, arteriole=0.7, episcleral_artery=0.5, episcleral_vein=0.2,
        anastomosis=0.5, perforator=0.3))   # class gains (plan G9 (P); order: Owen 2008, Meyer 1988)
    tort_vessel_sd: float = 0.5             # log-normal spread of the gain between vessels (P)
    p_tortuous: float = 0.05                # share of small vessels that are strongly tortuous (P)
    tortuous_gain: float = 3.0
    kappa_r_max: float = 0.75               # build target for |kappa| r (contract: <= 0.9; render study: < 1)
    min_bend_radius: float = 2.0            # and no bend tighter than this radius (d_c) in any vessel: small vessels
                                            # made V-shaped hairpin kinks at |kappa| r = 0.75 (realism review T8) (P)
    contract_factor: float = 2.0            # edges shorter than this x (R_u + R_v) + 1 are contracted into one junction
                                            # (was 1.5: short wide stubs rendered as black knots; review T8) (P)
    r_var_sd: float = 0.2                   # along-vessel log-radius SD (plan G9: 0.05-0.2; calibrated from 0.1, then
                                            # 0.18; v2: 0.2, along_cv_fwhm -1.5 SD: radius is what moves it, a depth
                                            # undulation or haematocrit noise did not; 0.22 at 6 d_c drew fusiform
                                            # bulges)
    r_var_sd_A: float = 0.15                # the same for arterioles (more even calibre: muscular walls) (v2; P)
    r_var_clip: float = 2.0                 # soft limit of the log-radius excursion, in SDs (eps -> c tanh(eps / c),
                                            # c = r_var_clip x SD; 0: none).  v2 integration: the bridged noise reached
                                            # 3-3.5 SD (up to 2.8x) and drew fusiform bulges, 'spindles' the stills do
                                            # not show (27 per frame wider than 1.5x their ends, 7.5 wider than 1.8x;
                                            # with the limit 0.5 and 0.2).  6 seeds: xing_shallow +5.9 -> +1.7 SD (the
                                            # spindles read as shallow crossings), calibre_change +1.9 -> +1.0, wide_kink
                                            # +1.5 -> +0.2, along_cv_fwhm -1.2 -> -1.7 (P, C)
    r_var_corr: float = 8.0                 # its correlation length, at least 2.5 D of the vessel (plan G9: 10-50;
                                            # calibrated from 25: FWHM CV along vessels was 0.09 against 0.14; v2:
                                            # from 10 and 3 D: real vessels bead and change calibre over a few
                                            # diameters, along_cv_fwhm -1.8 SD)

    # ---- haematocrit (relative; 1 = systemic)
    hct_cap: tuple = (0.3, 0.8)             # capillaries (Fahraeus; Pries 1992; Klitzman & Duling 1979: 0.2-1)
    hct_var: float = 0.05                   # along-vessel SD (P)

    # ---- boundary pressures (plan G6: inlets 1 +- 0.1, outlets 0 +- 0.05)
    p_art_in: tuple = (0.95, 1.0)           # arterial inlets
    p_art_exit: tuple = (0.85, 0.95)        # deep arteries and their branches leaving the domain
    p_ven_entry: tuple = (0.1, 0.2)         # deep veins and tributaries entering the domain
    p_ven_out: tuple = (0.0, 0.05)          # venous outlets

    # ---- pathology knobs (1 = healthy)
    dilation: float = 1.0                   # venous calibre multiplier (venules, deep veins)
    art_dilation: float = 1.0               # arterial calibre multiplier
    cap_dilation: float = 1.0               # capillary calibre multiplier
    tortuosity_gain: float = 1.0            # multiplies every tortuosity amplitude
    size_gain: float = 1.0                  # multiplies the deep trunk calibres (conjunctival caps: *_diam_max)
    deep_count_gain: float = 1.0            # multiplies the number of deep trunks

    # ---- final pass
    resolve_iters: int = 30                 # overlap-removal iterations (a cap; it stops when none is left)


def preset(name: str = "healthy", rng=None) -> AnatomyParams:
    """Parameters for a scene: 'healthy' (defaults, with the scene-level
    draws made explicit), 'pathologic' (dilated, tortuous, larger vessels)
    or 'random' (wide ranges, for training)."""
    rng = np.random.default_rng(rng)
    ori = float(90.0 + rng.normal(0.0, 12.0))
    if name == "healthy":
        return AnatomyParams(orientation_deg=ori, limbus_sign=int(rng.choice([-1, 1])))
    if name == "pathologic":
        return AnatomyParams(
            orientation_deg=ori, limbus_sign=int(rng.choice([-1, 1])),
            dilation=float(rng.uniform(1.4, 2.2)), art_dilation=float(rng.uniform(1.0, 1.3)),
            cap_dilation=float(rng.uniform(1.0, 1.3)), tortuosity_gain=float(rng.uniform(2.0, 4.0)),
            size_gain=float(rng.uniform(1.0, 1.3)), deep_count_gain=float(rng.uniform(1.2, 2.0)),
            ven_diam_max=16.0, art_diam_max=9.0, vein_diam=(10.0, 22.0), artery_diam=(6.0, 14.0),
            p_tortuous=float(rng.uniform(0.15, 0.4)), r_var_sd=float(rng.uniform(0.1, 0.16)),
            r_var_corr=30.0, n_risers_A=1.5, n_risers_V=1.5,   # dilation long and smooth, not spindles (review T8)
            collect_diam_max=22.0, plexus_diam=(4.0, 9.0), web_tort_gain=0.3,   # tortuosity_gain acts too (v2: 0.5)
            bundle_rate=0.25)   # v2: half, as healthy's
    if name == "random":
        lu = lambda a, b: float(math.exp(rng.uniform(math.log(a), math.log(b))))  # noqa: E731
        ori = float(90.0 + rng.normal(0.0, 20.0)) if rng.random() < 0.7 else float(rng.uniform(0, 180))
        p = AnatomyParams(
            k=lu(1.5, 8.0), orientation_deg=ori, limbus_sign=int(rng.choice([-1, 1])),
            orient_field_sd_deg=float(rng.uniform(5, 30)), orient_weight=float(rng.uniform(0.1, 1.0)),
            density_cv=float(rng.uniform(0.2, 0.6)),
            mesh_depth=float(rng.uniform(8.0, 11.0)), deep_top=float(rng.uniform(40, 60)),
            n_deep_veins=float(rng.uniform(0.5, 3.0)), n_deep_arteries=float(rng.uniform(0.3, 2.0)),
            n_art_sheets=int(rng.integers(1, 3)), n_ven_sheets=int(rng.integers(1, 4)),
            art_spacing=lu(50, 130), ven_spacing=lu(16, 45),
            persistence_weight=float(rng.uniform(0.3, 1.5)),
            root_spacing_A=lu(120, 400), root_spacing_V=lu(70, 220),
            gamma_A=(2.1, 2.9), gamma_V=(1.7, 2.3),
            p_pair=float(rng.uniform(0.0, 0.6)),
            # mesh cells 29-58 d_c (v2: up to 2.8e-3, cells 19 d_c: a honeycomb over the frame at small k, review v1
            # T11)
            mesh_density=lu(3e-4, 1.2e-3), mesh_aspect=(1.0, float(rng.uniform(1.0, 2.0))),   # 3: box kinks (T12)
            mesh_drop=float(rng.uniform(0.0, 0.25)),
            tort_amp=lu(0.03, 0.2), tort_corr=float(rng.uniform(3, 15)),
            p_tortuous=float(rng.uniform(0.0, 0.2)), r_var_sd=float(rng.uniform(0.05, 0.2)),
            side_spacing=float(rng.choice([0.0, rng.uniform(25, 60)])))
        # the web (v1): drawn after the v0 draws, so those keep their values
        p = replace(p, web_ven_spacing=lu(28, 80), web_art_spacing=lu(70, 200),
                    web_angle_sd_deg=float(rng.uniform(15, 40)), web_iso_frac=float(rng.uniform(0.0, 0.35)),
                    web_arcade_spacing=lu(80, 300), collect_rate=float(rng.uniform(0.12, 0.8)), collect_min=0,
                    web_p_pair_A=float(rng.uniform(0.2, 0.8)), web_p_pair_V=float(rng.uniform(0.0, 0.4)),
                    bundle_rate=float(rng.uniform(0.0, 0.5)), twig_spacing=lu(50, 150),
                    plexus_spacing=lu(35.0, 160.0), web_tort_gain=float(rng.uniform(0.2, 0.6)))
        if rng.random() < 0.3:     # a share of pathological eyes
            p = replace(p, dilation=float(rng.uniform(1.2, 2.2)), tortuosity_gain=float(rng.uniform(1.5, 4.0)),
                        size_gain=float(rng.uniform(1.0, 1.3)), ven_diam_max=16.0, vein_diam=(10.0, 22.0),
                        collect_diam_max=22.0)
        return p
    raise ValueError(f"unknown preset {name!r}")


# =================================================================== small helpers
def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _arclen(p) -> np.ndarray:
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def _resample(p, h: float, *attrs):
    """Points spaced ~h apart along a polyline (ends kept), and per-point
    attributes carried along (nearest source point)."""
    p = np.asarray(p, float)
    s = _arclen(p)
    L = float(s[-1])
    n = max(2, int(math.ceil(L / h)) + 1)
    q = np.linspace(0.0, L, n)
    out = np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)
    if not attrs:
        return out
    j = np.clip(np.searchsorted(s, q, side="right") - 1, 0, len(p) - 1)
    near = np.where((j + 1 < len(p)) & (np.abs(s[np.minimum(j + 1, len(p) - 1)] - q) < np.abs(s[j] - q)), j + 1, j)
    return (out,) + tuple(np.asarray(a)[near] for a in attrs)


def _dedupe(pts, lay):
    """Drop repeated consecutive points (keeping both ends)."""
    if len(pts) <= 2:
        return pts, lay
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-9])
    keep[-1] = True
    if not keep.all():
        pts, lay = pts[keep], lay[keep]
        if len(pts) > 2 and np.hypot(*(pts[-1] - pts[-2])) < 1e-9:
            pts, lay = np.delete(pts, -2, 0), np.delete(lay, -2)
    return pts, lay


def _smooth_noise(n: int, corr: float, rng) -> np.ndarray:
    """n samples of a zero-mean unit-variance Gaussian process whose
    autocorrelation is exp(-d^2 / 2 corr^2) (corr in samples)."""
    if corr < 0.5:
        return rng.standard_normal(n)
    sk = corr / math.sqrt(2.0)
    pad = int(4 * sk) + 2
    w = rng.standard_normal(n + 2 * pad)
    f = ndi.gaussian_filter1d(w, sk, mode="wrap")[pad:pad + n]
    return f * math.sqrt(2.0 * math.sqrt(math.pi) * sk)


def _bridge(x):
    """x with its linear trend between the end values removed (zero at both ends)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x * 0
    t = np.linspace(0.0, 1.0, len(x))
    return x - (x[0] * (1 - t) + x[-1] * t)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _laplace(p, n_iter: int, lam: float = 0.5, mask=None):
    """Laplacian smoothing of a polyline with fixed ends (optionally only
    where mask is True)."""
    p = np.array(p, float)
    for _ in range(n_iter):
        d = np.zeros_like(p)
        d[1:-1] = 0.5 * (p[:-2] + p[2:]) - p[1:-1]
        if mask is not None:
            d[~mask] = 0
        p += lam * d
    return p


def _curv(p):
    """Unsigned discrete curvature at every point of a polyline (ends 0)."""
    p = np.asarray(p, float)
    k = np.zeros(len(p))
    if len(p) < 3:
        return k
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab = np.hypot(*(b - a).T)
    bc = np.hypot(*(c - b).T)
    ac = np.hypot(*(c - a).T)
    cross = np.abs((b - a)[:, 0] * (c - b)[:, 1] - (b - a)[:, 1] * (c - b)[:, 0])
    k[1:-1] = 2 * cross / np.maximum(ab * bc * ac, 1e-12)
    return k


def _tangents(p, span: int = 2):
    p = np.asarray(p, float)
    n = len(p)
    i0 = np.clip(np.arange(n) - span, 0, n - 1)
    i1 = np.clip(np.arange(n) + span, 0, n - 1)
    return _unit(p[i1] - p[i0])


def _murray_angles(q: float, rng, noise_deg: float):
    """Daughter angles (rad) from the parent's direction for a flow split q
    (share of the smaller daughter), with cube-law-equivalent radii
    (Murray 1926; Zamir 1978; plan 3.3 branch angles), plus noise."""
    q = float(np.clip(q, 1e-3, 0.5))
    r1, r2 = (1 - q) ** (1 / 3), q ** (1 / 3)
    c1 = (1 + r1 ** 4 - r2 ** 4) / (2 * r1 ** 2)
    c2 = (1 + r2 ** 4 - r1 ** 4) / (2 * r2 ** 2)
    a1 = math.acos(float(np.clip(c1, -1, 1)))
    a2 = math.acos(float(np.clip(c2, -1, 1)))
    nz = math.radians(noise_deg)
    return max(0.0, a1 + nz * rng.standard_normal()), max(math.radians(15), a2 + nz * rng.standard_normal())


# =================================================================== geometry of the scene
class _Geo:
    """The frame, the padded domain and the tissue region (d_c rectangles
    (x0, y0, x1, y1))."""

    def __init__(self, P: AnatomyParams):
        W, H = P.frame_px
        k = P.k
        self.frame = (-0.5 / k, -0.5 / k, (W - 0.5) / k, (H - 0.5) / k)
        fw, fh = W / k, H / k
        self.pad = max(P.pad_frac * max(fw, fh), P.pad_min)
        x0, y0, x1, y1 = self.frame
        self.domain = (x0 - self.pad, y0 - self.pad, x1 + self.pad, y1 + self.pad)
        m = min(P.tissue_margin, 0.9 * self.pad)
        self.tissue = (x0 - m, y0 - m, x1 + m, y1 + m)

    @staticmethod
    def inside(xy, rect, margin: float = 0.0):
        xy = np.atleast_2d(xy)
        x0, y0, x1, y1 = rect
        return ((xy[:, 0] >= x0 + margin) & (xy[:, 0] <= x1 - margin) &
                (xy[:, 1] >= y0 + margin) & (xy[:, 1] <= y1 - margin))

    def exit_point(self, p, q):
        """Where the segment p (inside the domain) -> q (outside) leaves it."""
        x0, y0, x1, y1 = self.domain
        d = q - p
        t = 1.0
        for ax, lo, hi in ((0, x0, x1), (1, y0, y1)):
            if d[ax] > 0 and q[ax] > hi:
                t = min(t, (hi - p[ax]) / d[ax])
            elif d[ax] < 0 and q[ax] < lo:
                t = min(t, (lo - p[ax]) / d[ax])
        out = p + max(0.0, t) * d
        out[0] = min(max(out[0], x0), x1)
        out[1] = min(max(out[1], y0), y1)
        return out

    def ray_to_border(self, c, d):
        """The border point reached from interior point c going along d."""
        x0, y0, x1, y1 = self.domain
        ts = []
        for ax, lo, hi in ((0, x0, x1), (1, y0, y1)):
            if d[ax] > 1e-12:
                ts.append((hi - c[ax]) / d[ax])
            elif d[ax] < -1e-12:
                ts.append((lo - c[ax]) / d[ax])
        p = c + min(ts) * d
        p[0] = min(max(p[0], x0), x1)
        p[1] = min(max(p[1], y0), y1)
        return p

    def margin_taper(self, xy):
        """1 inside the frame, fading to 0.2 at the edge of the tissue region,
        so trees do not crowd along that (artificial) edge."""
        xy = np.atleast_2d(xy)
        fx0, fy0, fx1, fy1 = self.frame
        out = np.maximum.reduce([fx0 - xy[:, 0], xy[:, 0] - fx1, fy0 - xy[:, 1], xy[:, 1] - fy1, np.zeros(len(xy))])
        m = max(self.tissue[2] - self.frame[2], 1e-6)
        return 1.0 - 0.8 * _smoothstep(out / m)

    def border_dist(self, xy):
        xy = np.atleast_2d(xy)
        x0, y0, x1, y1 = self.domain
        return np.minimum.reduce([xy[:, 0] - x0, x1 - xy[:, 0], xy[:, 1] - y0, y1 - xy[:, 1]])


class _Field:
    """A smooth Gaussian random field (unit variance before scaling) on a
    coarse grid over the domain, looked up bilinearly."""

    def __init__(self, domain, corr: float, sd: float, rng, mean: float = 0.0):
        x0, y0, x1, y1 = domain
        self.cell = max(4.0, corr / 4.0)
        self.x0, self.y0 = x0 - 2 * self.cell, y0 - 2 * self.cell
        nx = int(math.ceil((x1 - x0) / self.cell)) + 5
        ny = int(math.ceil((y1 - y0) / self.cell)) + 5
        w = rng.standard_normal((ny, nx))
        f = ndi.gaussian_filter(w, corr / self.cell / math.sqrt(2.0), mode="reflect")
        f = (f - f.mean()) / max(float(f.std()), 1e-9)
        self.v = mean + sd * f

    def __call__(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        gx = (xy[:, 0] - self.x0) / self.cell
        gy = (xy[:, 1] - self.y0) / self.cell
        return ndi.map_coordinates(self.v, [gy, gx], order=1, mode="nearest")


class _Fields:
    def __init__(self, P: AnatomyParams, geo: _Geo, theta0: float, rng):
        dom = geo.domain
        self.theta0 = theta0
        self._th = _Field(dom, P.orient_field_corr, math.radians(P.orient_field_sd_deg), rng, theta0)
        sig = math.sqrt(math.log(1 + P.density_cv ** 2))
        self._dn = _Field(dom, P.density_corr, sig, rng)
        self._sig = sig
        self.tissue_depth = _Field(dom, P.tissue_depth_corr, P.tissue_depth_sd, rng)
        self.deep_depth = _Field(dom, P.tissue_depth_corr, P.deep_depth_sd, rng)

    def theta(self, xy):
        return self._th(xy)

    def odir(self, xy, ref=None):
        """Unit vectors of the local direction, flipped to agree with ref."""
        th = self.theta(xy)
        o = np.stack([np.cos(th), np.sin(th)], 1)
        if ref is not None:
            ref = np.atleast_2d(ref)
            o = np.where((np.sum(o * ref, 1) < 0)[:, None], -o, o)
        return o

    def density(self, xy):
        return np.exp(self._dn(xy) - 0.5 * self._sig ** 2)

    def density_max(self):
        return float(np.exp(self._dn.v.max() - 0.5 * self._sig ** 2))


def _walk(p0, phi0, *, step, length, persist, heading_sd, F, geo, rng, obstacles=None, clear=0.0,
          exempt_xy=None, exempt_r=0.0, stop_in=None, stop_out=None, para=None):
    """An Ornstein-Uhlenbeck heading walk: the heading reverts, over the
    correlation length `persist`, to the local field direction plus the
    walk's initial offset from it, with stationary SD heading_sd (rad).
    Steers around obstacle points closer than `clear`; with para = (kd-tree
    of other vessels' points, their unit tangents, clear, cos_max, turn) it
    also turns away by `turn` (rad) per step from any of them closer than
    clear that runs within acos(cos_max) of its heading (it keeps beside a
    near-parallel vessel instead of crossing it at a shallow angle).  Stops on the
    domain border (last point exactly on it), after `length`, on entering
    stop_in / leaving stop_out (rectangles).  Returns (points, reason)."""
    p = np.asarray(p0, float).copy()
    pts = [p.copy()]
    phi = float(phi0)
    off = phi - float(F.theta(p[None])[0])
    a = min(1.0, step / persist)
    sig = heading_sd * math.sqrt(2.0 * a)
    tree = cKDTree(obstacles) if obstacles is not None and len(obstacles) else None
    paras = [] if para is None else (para if isinstance(para, list) else [para])
    s = 0.0
    while s < length:
        ref = float(F.theta(p[None])[0]) + off
        d = (ref - phi + math.pi) % (2 * math.pi) - math.pi
        phi += a * d + sig * rng.standard_normal()
        for para_ in paras:
            # (kd-tree, unit tangents, clear, cos_max, turn[, mode[, per-point extra clearance[, min distance]]]):
            # v2 added the extra clearance (a deep vessel's radius) and, in 'align' mode, a distance under which the
            # walk turns away instead (aligned on top of its neighbour it drew a long overlap: a dark spindle)
            ptree, ptan, pclear, pcos, pturn = para_[:5]
            para_mode = para_[5] if len(para_) > 5 else "away"
            pext = para_[6] if len(para_) > 6 else None
            pmin = para_[7] if len(para_) > 7 else 0.0
            hits = ptree.query_ball_point(p, pclear + (float(pext.max()) if pext is not None else 0.0))
            if hits and pext is not None:
                hits = [h for h in hits if np.hypot(*(ptree.data[h] - p)) < pclear + pext[h]]
            if hits:
                hv = np.array([math.cos(phi), math.sin(phi)])
                c = ptan[hits] @ hv
                near = np.abs(c) > pcos
                if near.any():
                    if para_mode == "align" and pmin > 0:
                        hn = np.asarray(hits)[near]
                        dn = np.hypot(*(ptree.data[hn] - p).T) - (pext[hn] if pext is not None else 0.0)
                        if float(np.min(dn)) < pmin:
                            para_mode = "away"
                    if para_mode == "align":
                        # turn towards the neighbour's direction (up to pturn per step): it runs beside it
                        tn = ptan[hits][near] * np.sign(c[near])[:, None]
                        tm = tn.mean(0)
                        da = math.atan2(hv[0] * tm[1] - hv[1] * tm[0], float(hv @ tm))
                        dlt = float(np.clip(da, -pturn, pturn))
                    else:
                        away = p - ptree.data[np.asarray(hits)[near]].mean(0)
                        dlt = pturn * (1.0 if hv[0] * away[1] - hv[1] * away[0] > 0 else -1.0)
                    phi += dlt
                    # the reference drifts with it, or the OU pull would turn it straight back
                    off += dlt
        q = None
        for _ in range(10):
            cand = p + step * np.array([math.cos(phi), math.sin(phi)])
            if tree is not None:
                hits = tree.query_ball_point(cand, clear)
                if hits and exempt_xy is not None:
                    ob = obstacles[hits]
                    hits = [h for h, o in zip(hits, ob) if np.hypot(*(o - exempt_xy)) > exempt_r]
                if hits:
                    away = cand - obstacles[hits].mean(0)
                    dirv = cand - p
                    sgn = 1.0 if dirv[0] * away[1] - dirv[1] * away[0] > 0 else -1.0
                    phi += 0.3 * sgn
                    continue
            q = cand
            break
        if q is None:
            return np.array(pts), "blocked"
        if not _Geo.inside(q, geo.domain)[0]:
            pts.append(geo.exit_point(p, q))
            return np.array(pts), "border"
        pts.append(q)
        p = q
        s += step
        if stop_in is not None and _Geo.inside(q, stop_in)[0]:
            return np.array(pts), "in"
        if stop_out is not None and not _Geo.inside(q, stop_out)[0]:
            return np.array(pts), "out"
    return np.array(pts), "length"


# =================================================================== the working network
@dataclass
class _E:
    """A working edge: a polyline from node u to node v with a layer label
    per point.  Tree edges run root -> leaf (u towards the tree's root)."""
    u: int
    v: int
    pts: np.ndarray
    lay: np.ndarray
    cls: str
    side: str
    role: str
    tree: int | None = None
    attach: dict = field(default_factory=dict)   # foreign tree id -> node where this edge hangs from it
    leaf_r: float | None = None
    r: float = 0.5
    q: float | None = None                       # deep branch: share of its parent's flow
    fixed_r: float | None = None                 # AVAs, anastomoses
    flags: dict = field(default_factory=dict)


class _Net:
    """Nodes (positions, border pressures) and polyline edges, with the
    split / remove operations the generator needs.  Converted into a
    VesselGraph at the end."""

    def __init__(self):
        self.xy: dict[int, np.ndarray] = {}
        self.nd: dict[int, dict] = {}
        self.E: dict[int, _E] = {}
        self.adj: dict[int, set] = defaultdict(set)
        self.trees: dict[int, dict] = {}
        self.layers: dict[int, dict] = {}
        self._n = 0
        self._e = 0
        self._t = 0

    def node(self, xy, **kw) -> int:
        n = self._n
        self._n += 1
        self.xy[n] = np.asarray(xy, float).copy()
        self.nd[n] = dict(kw)
        self.adj[n]
        return n

    def edge(self, u, v, pts, lay, **kw) -> int:
        pts = np.asarray(pts, float).copy()
        pts[0], pts[-1] = self.xy[u], self.xy[v]
        lay = np.full(len(pts), lay, int) if np.ndim(lay) == 0 else np.asarray(lay, int).copy()
        pts, lay = _dedupe(pts, lay)
        e = self._e
        self._e += 1
        self.E[e] = _E(u, v, pts, lay, **kw)
        self.adj[u].add(e)
        self.adj[v].add(e)
        return e

    def new_tree(self, **kw) -> int:
        t = self._t
        self._t += 1
        self.trees[t] = dict(kw)
        return t

    def remove_edge(self, e):
        E = self.E.pop(e)
        self.adj[E.u].discard(e)
        self.adj[E.v].discard(e)

    def remove_node_if_free(self, n):
        if n in self.xy and not self.adj[n]:
            del self.xy[n], self.nd[n], self.adj[n]

    def is_border(self, n) -> bool:
        return self.nd[n].get("pressure") is not None

    def split(self, e, s_list) -> list[int]:
        """Split edge e at arclengths s_list (sorted, strictly inside); the
        first piece keeps the id.  Returns the new node ids."""
        E = self.E[e]
        s = _arclen(E.pts)
        L = float(s[-1])
        s_list = [float(x) for x in s_list if 1e-6 < x < L - 1e-6]
        if not s_list:
            return []
        cuts = []
        for x in s_list:
            j = int(np.clip(np.searchsorted(s, x) - 1, 0, len(s) - 2))
            t = (x - s[j]) / max(s[j + 1] - s[j], 1e-12)
            cuts.append((j, E.pts[j] + t * (E.pts[j + 1] - E.pts[j])))
        nodes = [self.node(p) for _, p in cuts]
        pieces = []
        start_j, start_p = 0, None
        for (j, p) in cuts:
            body = E.pts[start_j + (1 if start_p is not None else 0):j + 1]
            lay = E.lay[start_j + (1 if start_p is not None else 0):j + 1]
            head = [start_p] if start_p is not None else []
            hl = [E.lay[start_j]] if start_p is not None else []
            pieces.append((np.array(head + list(body) + [p]), np.array(hl + list(lay) + [E.lay[j]])))
            start_j, start_p = j, p
        body = E.pts[start_j + 1:]
        lay = E.lay[start_j + 1:]
        pieces.append((np.array([start_p] + list(body)), np.array([E.lay[start_j]] + list(lay))))
        ends = [E.u] + nodes + [E.v]
        v_old = E.v
        self.adj[v_old].discard(e)
        leaf_r = E.leaf_r
        # an attachment at the far end (a foreign tree this edge hangs from there) moves to the last piece
        far_attach = {T: n for T, n in E.attach.items() if n == v_old and n != E.u}
        for T in far_attach:
            del E.attach[T]
        last = e
        for i, (pp, ll) in enumerate(pieces):
            u, v = ends[i], ends[i + 1]
            if len(pp) < 2:
                pp = np.array([self.xy[u], self.xy[v]])
                ll = np.array([ll[0] if len(ll) else E.lay[0]] * 2)
            if i == 0:
                pp = pp.copy()
                pp[-1] = self.xy[v]
                E.pts, E.lay = _dedupe(pp, ll)
                E.v = v
                E.leaf_r = None
                self.adj[v].add(e)
            else:
                role = "cont" if E.role in ("branch", "trunk", "cont") else E.role
                last = self.edge(u, v, pp, ll, cls=E.cls, side=E.side, role=role, tree=E.tree,
                                 leaf_r=leaf_r if i == len(pieces) - 1 else None, r=E.r,
                                 flags={k: v2 for k, v2 in E.flags.items() if k not in ("partners",)})
        if far_attach:
            self.E[last].attach.update(far_attach)
        return nodes

    def member_parent(self, tree, node):
        """The member edge of `tree` ending (leaf-side) at `node`, or None."""
        for e in self.adj[node]:
            E = self.E[e]
            if E.tree == tree and E.v == node:
                return e
        return None


class _PtIndex:
    """Dense samples of a set of edges (about `spacing` apart) in a KD-tree:
    nearest points, and whether a segment passes close to them."""

    def __init__(self, net: _Net, eids, spacing: float = 1.0, with_r: bool = False):
        xs, es, ss, rs = [], [], [], []
        for e in eids:
            p = net.E[e].pts
            s = _arclen(p)
            n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
            q = np.linspace(0, s[-1], n)
            xs.append(np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1))
            es.append(np.full(n, e))
            ss.append(q)
            if with_r:
                rs.append(np.full(n, net.E[e].r))
        self.xy = np.concatenate(xs) if xs else np.zeros((0, 2))
        self.eid = np.concatenate(es) if es else np.zeros(0, int)
        self.s = np.concatenate(ss) if ss else np.zeros(0)
        self.r = np.concatenate(rs) if rs else np.zeros(len(self.xy))
        self.tree = cKDTree(self.xy) if len(self.xy) else None

    def nearest_edges(self, p, k: int = 32, rmax: float = np.inf, n_edges: int = 8):
        """Candidates (eid, s, xy, distance), one per edge, nearest first."""
        if self.tree is None:
            return []
        k = min(k, len(self.xy))
        d, i = self.tree.query(p, k=k, distance_upper_bound=rmax)
        d, i = np.atleast_1d(d), np.atleast_1d(i)
        out, seen = [], set()
        for dd, ii in zip(d, i):
            if not np.isfinite(dd):
                break
            e = int(self.eid[ii])
            if e in seen:
                continue
            seen.add(e)
            out.append((e, float(self.s[ii]), self.xy[ii], float(dd)))
            if len(out) >= n_edges:
                break
        return out

    def seg_hits(self, a, b, clear: float, exclude=(), skip_a: float = 0.0, skip_b: float = 0.0) -> bool:
        if self.tree is None:
            return False
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.hypot(*(b - a)))
        n = max(2, int(L / 0.75) + 2)
        t = np.linspace(0, 1, n)
        keep = (t * L >= skip_a) & ((1 - t) * L >= skip_b)
        if not keep.any():
            return False
        pts = a + t[keep, None] * (b - a)
        ex = set(exclude)
        for hits in self.tree.query_ball_point(pts, clear):
            for h in hits:
                if int(self.eid[h]) not in ex:
                    return True
        return False


# =================================================================== stage: deep layer
def _deep_estimates(net: _Net, tid: int):
    """Nominal top-down radii of one deep tree (before attachments)."""
    T = net.trees[tid]
    g = T["gamma"]
    root = [e for e, E in net.E.items() if E.tree == tid and net.member_parent(tid, E.u) is None]
    stack = [(e, T["root_r"]) for e in root]
    seen = set()
    while stack:
        e, r = stack.pop()
        if e in seen:
            continue
        seen.add(e)
        net.E[e].r = r
        kids = [c for c in net.adj[net.E[e].v] if c != e and net.E[c].tree == tid and net.E[c].u == net.E[e].v]
        rem = r ** g
        for c in kids:
            if net.E[c].role == "branch":
                rb = r * net.E[c].q ** (1 / g)
                rem -= rb ** g
                stack.append((c, rb))
        for c in kids:
            if net.E[c].role != "branch":
                stack.append((c, max(rem, (0.5 * r) ** g) ** (1 / g)))


def _grow_deep(net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng, gammas) -> list[int]:
    """Episcleral trunks across the domain, with sparse branches to the
    border; some branches dive (perforators).  Layout 'web': also the deep
    plexus, many medium veins (an interlacing venous network, Selbach 2005)
    without branches; plexus_links anastomoses join them (_add_avas)."""
    n_v = rng.poisson(P.n_deep_veins * P.deep_count_gain)
    n_a = rng.poisson(P.n_deep_arteries * P.deep_count_gain)
    kinds = ["DV"] * n_v + ["DA"] * n_a
    if P.layout == "web":
        # as many as make the plexus's length density 1 / plexus_spacing over the tissue region (each crosses it
        # about (w + h) / 2)
        tw, th = geo.tissue[2] - geo.tissue[0], geo.tissue[3] - geo.tissue[1]
        kinds += ["DP"] * rng.poisson(tw * th / P.plexus_spacing / (0.5 * (tw + th)) * P.deep_count_gain)
    tids = []
    tx0, ty0, tx1, ty1 = geo.tissue
    diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
    deep_xy, deep_t, deep_r = [], [], []
    for kind in kinds:
        plexus = kind == "DP"
        vein = kind in ("DV", "DP")
        lo, hi = P.plexus_diam if plexus else P.vein_diam if vein else P.artery_diam
        D = float(rng.uniform(lo, hi)) * P.size_gain * (P.dilation if plexus else 1.0)
        r0 = D / 2
        pl, ph = P.plexus_persistence if plexus else P.vein_persistence if vein else P.artery_persistence
        hsd = math.radians(P.plexus_heading_sd_deg if plexus else P.vein_heading_sd_deg if vein
                           else P.artery_heading_sd_deg) * math.sqrt(P.tortuosity_gain)
        asd = P.plexus_angle_sd_deg if plexus else P.deep_angle_sd_deg
        kind = "DV" if plexus else kind
        path = None
        # v2: away from the deep trunks made so far where it would run along one (overlapping in projection over a
        # long stretch, two deep vessels need depths clear of each other's diameters all along: the pathologic
        # preset's dilated ones could not, and the overlap pass removed them; and they drew dark spindles)
        dpara = None
        if P.layout == "web" and P.deep_para_clear > 0 and deep_xy:
            dpara = (cKDTree(np.concatenate(deep_xy)), np.concatenate(deep_t), P.deep_para_clear + r0,
                     math.cos(math.radians(P.deep_para_deg)), math.radians(3.0), "away", np.concatenate(deep_r), 0.0)
        for _ in range(20):
            ang = F.theta0 + math.radians(asd) * rng.standard_normal()
            u = np.array([math.cos(ang), math.sin(ang)])
            c = np.array([rng.uniform(tx0, tx1), rng.uniform(ty0, ty1)])
            start = geo.ray_to_border(c, -u)
            pts, why = _walk(start, ang, step=2.0, length=3 * diag, persist=float(rng.uniform(pl, ph)) * D,
                             heading_sd=hsd, F=F, geo=geo, rng=rng, para=dpara)
            inside = _Geo.inside(pts, geo.tissue)
            if why == "border" and inside.sum() * 2.0 > 100 and len(pts) > 30:
                path = pts
                break
        if path is None:
            continue
        q_ = _resample(path, 2.0)
        deep_xy.append(q_)
        deep_t.append(_tangents(q_, 1))
        deep_r.append(np.full(len(q_), r0))
        g = gammas["A" if not vein else "V"]
        tid = net.new_tree(kind=kind, gamma=g, root_r=r0, D=D, **({"plexus": True} if plexus else {}))
        lay = len(net.layers)
        net.layers[lay] = dict(name=f"deep{len(tids)}", kind="deep", tree=tid, vein=vein, plexus=plexus)
        net.trees[tid]["layer"] = lay
        tids.append(tid)
        if not vein:
            tp = path
            p_root, p_leaf = float(rng.uniform(*P.p_art_in)), float(rng.uniform(*P.p_art_exit))
        else:
            tp = path[::-1]
            p_root, p_leaf = float(rng.uniform(*P.p_ven_out)), float(rng.uniform(*P.p_ven_entry))
        cls = "episcleral_vein" if vein else "episcleral_artery"
        n0 = net.node(tp[0], pressure=p_root)
        n1 = net.node(tp[-1], pressure=p_leaf)
        trunk = net.edge(n0, n1, tp, lay, cls=cls, side="D", role="trunk", tree=tid,
                         flags=dict(plexus=True) if plexus else {})
        if plexus:                    # no branches: the plexus is joined by anastomoses
            _deep_estimates(net, tid)
            continue
        # branches, leafward from the trunk at the Murray angle of their flow split
        s = _arclen(tp)
        L = float(s[-1])
        pos, x = [], D * math.exp(rng.uniform(*np.log(P.deep_branch_LD)))
        while x < L - 60:
            pos.append(x)
            x += D * math.exp(rng.uniform(*np.log(P.deep_branch_LD)))
        at = np.stack([np.interp(pos, s, tp[:, 0]), np.interp(pos, s, tp[:, 1])], 1) if pos else np.zeros((0, 2))
        pos = [x for x, b in zip(pos, geo.border_dist(at)) if b > 40] if pos else []
        if not pos:
            continue
        nodes = net.split(trunk, pos)
        for nb in nodes:
            ein = net.member_parent(tid, nb)
            t = _unit(net.xy[nb] - net.E[ein].pts[max(0, len(net.E[ein].pts) - 4)])
            q = float(rng.uniform(*P.deep_branch_q))
            _, a2 = _murray_angles(q, rng, P.angle_noise_deg)
            a2 = min(a2, math.radians(85))
            sgn = 1 if rng.random() < 0.5 else -1
            hd = _rot(t, sgn * a2)
            rb = r0 * q ** (1 / g)
            obst = np.concatenate([net.E[e].pts for e in net.E if net.E[e].tree == tid])
            bp, why = _walk(net.xy[nb], math.atan2(hd[1], hd[0]), step=2.0, length=3 * diag,
                            persist=float(rng.uniform(pl, ph)) * 2 * rb, heading_sd=hsd, F=F, geo=geo, rng=rng,
                            obstacles=obst, clear=(r0 + rb + 3.0) * (2.0 if P.layout == "web" else 1.0),
                            exempt_xy=net.xy[nb], exempt_r=3 * (r0 + rb) + 6)
            if why != "border" or len(bp) < 10:
                continue
            pl_ = float(rng.uniform(*P.p_art_exit)) if not vein else float(rng.uniform(*P.p_ven_entry))
            nl = net.node(bp[-1], pressure=pl_)
            perf = bool(rng.random() < P.perforator_frac)
            net.edge(nb, nl, bp, lay, cls="perforator" if perf else cls, side="D", role="branch", tree=tid, q=q,
                     flags=dict(perforator=perf))
        _deep_estimates(net, tid)
    return tids


def _deep_points(net, tids, roles=("trunk", "cont", "branch"), rect=None, spacing=8.0):
    """Sample points (xy, eid, s) along deep member edges (not perforators)."""
    xs, es, ss = [], [], []
    for e, E in net.E.items():
        if E.tree in tids and E.role in roles and not E.flags.get("perforator"):
            s = _arclen(E.pts)
            if s[-1] < 3 * spacing:
                continue
            q = np.arange(spacing, s[-1] - spacing, spacing)
            xy = np.stack([np.interp(q, s, E.pts[:, 0]), np.interp(q, s, E.pts[:, 1])], 1)
            keep = np.ones(len(q), bool) if rect is None else _Geo.inside(xy, rect)
            xs.append(xy[keep])
            es.append(np.full(keep.sum(), e))
            ss.append(q[keep])
    if not xs:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
    return np.concatenate(xs), np.concatenate(es), np.concatenate(ss)


def _add_risers(net: _Net, P, F, geo, rng, tids, sheets):
    """Risers: a vessel leaves a deep artery (vein) and climbs into the
    conjunctiva, where it roots an arteriole (venule) tree.  A riser passes
    through every layer between, so it is placed where its path stays clear
    of the other episcleral vessels; the sheets it climbs through avoid it
    as they grow (see _colonize)."""
    roots = defaultdict(list)
    placed = []
    for side, kind, n_mean in (("A", "DA", P.n_risers_A), ("V", "DV", P.n_risers_V)):
        cand = [t for t in tids if net.trees[t]["kind"] == kind]
        my_sheets = [s for s in sheets if s["side"] == side]
        if not cand or not my_sheets:
            continue
        for _ in range(rng.poisson(n_mean)):
            xy, eids, ss = _deep_points(net, cand, rect=geo.tissue)
            if not len(xy):
                break
            ok = [i for i in range(len(xy)) if 25 < ss[i] < _arclen(net.E[eids[i]].pts)[-1] - 25]
            if not ok:
                break
            deep_e = [e for e, E in net.E.items() if E.tree in tids or E.role == "riser"]
            dpts = np.concatenate([net.E[e].pts for e in deep_e]) if deep_e else np.zeros((0, 2))
            drad = np.concatenate([np.full(len(net.E[e].pts), net.E[e].r if E_.role != "riser" else 3.0)
                                   for e, E_ in ((e, net.E[e]) for e in deep_e)]) if deep_e else np.zeros(0)
            dtree = cKDTree(dpts) if len(dpts) else None
            for _try in range(12):
                i = ok[rng.integers(len(ok))]
                e = int(eids[i])
                t = _unit(xy[i] - np.array([np.interp(ss[i] - 4, _arclen(net.E[e].pts), net.E[e].pts[:, k])
                                            for k in (0, 1)]))
                hd = _rot(t, (1 if rng.random() < 0.5 else -1) * math.radians(rng.uniform(50, 90)))
                # the depth it climbs: from the episcleral systems to the conjunctival sheets (layout 'web': the
                # episcleral systems lie below the web, about deep_top + deep_depth_span / 2)
                dz = (P.deep_top + 0.5 * P.deep_spread - 20.0 if P.layout != "web" else
                      max(P.deep_top, 12.0 + P.web_depth_span) + 0.5 * P.deep_depth_span - 15.0)
                Lr = float(np.clip(dz / math.tan(math.radians(rng.uniform(*P.riser_angle_deg))), 30, 150))
                path, why = _walk(xy[i], math.atan2(hd[1], hd[0]), step=2.0, length=Lr, persist=40.0,
                                  heading_sd=math.radians(10), F=F, geo=geo, rng=rng, stop_out=geo.tissue)
                if len(path) < 8 or why in ("border", "blocked"):
                    continue
                r_host = net.E[e].r
                far = np.hypot(*(path - xy[i]).T) > 3 * r_host + 8
                clear = True
                if dtree is not None and far.any():
                    for p_, hits in zip(path[far], dtree.query_ball_point(path[far], float(drad.max()) + 6.0)):
                        if any(np.hypot(*(dpts[h] - p_)) < drad[h] + 6.0 for h in hits):
                            clear = False
                            break
                if clear:
                    break
            else:
                continue
            nb = net.split(e, [ss[i]])[0]
            sheet = my_sheets[rng.integers(len(my_sheets))]
            nt = net.node(path[-1])
            tid = net.new_tree(kind=side, gamma=None, sheet=sheet["name"], riser_path=path)
            net.edge(nb, nt, path, RAMP, cls="perforator", side="D", role="riser", tree=tid,
                     attach={net.E[e].tree: nb})
            roots[sheet["name"]].append((nt, _unit(path[-1] - path[-2]), tid))
            placed.append(path)
    return roots


def _riser_obstacles(net: _Net, sheet: dict, sheets: list):
    """Points of the risers that climb through `sheet` (those rooting a
    shallower sheet; and its own, except next to their tops)."""
    order = {s["name"]: s["order"] for s in sheets}
    pts = []
    for t, T in net.trees.items():
        path = T.get("riser_path")
        if path is None:
            continue
        o = order.get(T.get("sheet"), -1)
        if o < sheet["order"]:
            pts.append(path)
        elif o == sheet["order"]:
            keep = np.hypot(*(path - path[-1]).T) > 20.0
            pts.append(path[keep])
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts) if pts else None


def _add_avas(net: _Net, P, geo, rng, tids):
    """Arteriovenous anastomoses between deep arteries and veins that pass
    near each other, and vein-vein anastomoses between deep vein systems."""
    arts = [t for t in tids if net.trees[t]["kind"] == "DA"]
    veins = [t for t in tids if net.trees[t]["kind"] == "DV"]
    x0, y0, x1, y1 = geo.tissue
    plans = []
    specs = []
    if arts and veins:
        n = rng.poisson(P.ava_density * (x1 - x0) * (y1 - y0))
        specs.append(("ava", arts, veins, n))
    if len(veins) > 1:
        npx = sum(1 for t in veins if net.trees[t].get("plexus"))
        ntr = len(veins) - npx
        # trunks: per pair of vein systems; the plexus (layout 'web'): per plexus vein
        n = rng.poisson(P.vv_anastomoses * ntr * max(ntr - 1, 0) / 2 + P.plexus_links * npx)
        specs.append(("vv", veins, veins, n))
    for kind, ta, tb, n in specs:
        if n == 0:
            continue
        xa, ea, sa = _deep_points(net, ta, rect=geo.tissue)
        xb, eb, sb = _deep_points(net, tb, rect=geo.tissue)
        if not len(xa) or not len(xb):
            continue
        allp = np.concatenate([net.E[e].pts for e in net.E if net.E[e].tree in tids or net.E[e].role == "riser"])
        allt = cKDTree(allp)
        pairs = cKDTree(xa).query_ball_tree(cKDTree(xb), 90.0)
        cand = [(i, j) for i, js in enumerate(pairs) for j in js
                if np.hypot(*(xa[i] - xb[j])) > 15 and net.E[ea[i]].tree != net.E[eb[j]].tree]
        rng.shuffle(cand)
        got = 0
        ends = np.array([p for pa, pb, *_ in plans for p in (pa, pb)], float).reshape(-1, 2)
        for i, j in cand:
            if got >= n:
                break
            a, b = xa[i], xb[j]
            if len(ends) and min(float(np.hypot(*(ends - a).T).min()), float(np.hypot(*(ends - b).T).min())) < 60:
                continue
            La, Lb = _arclen(net.E[ea[i]].pts)[-1], _arclen(net.E[eb[j]].pts)[-1]
            if not (20 < sa[i] < La - 20 and 20 < sb[j] < Lb - 20):
                continue
            ra, rb = net.E[ea[i]].r, net.E[eb[j]].r
            d = np.hypot(*(b - a))
            nrm = _rot(_unit(b - a), math.pi / 2)
            ctrl = 0.5 * (a + b) + nrm * rng.uniform(-0.25, 0.25) * d
            t = np.linspace(0, 1, max(8, int(d)))[:, None]
            path = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
            mid = path[(np.hypot(*(path - a).T) > 3 * ra + 8) & (np.hypot(*(path - b).T) > 3 * rb + 8)]
            if len(mid) and any(len(h) for h in allt.query_ball_point(mid, max(ra, rb) + 4.0)):
                continue
            frac = float(rng.uniform(*P.ava_radius_frac))
            plans.append((a, b, int(ea[i]), float(sa[i]), int(eb[j]), float(sb[j]), path, frac * min(ra, rb), kind))
            ends = np.concatenate([ends, [a, b]])
            got += 1
    # split every host edge once at all its positions, then connect
    cuts = defaultdict(list)
    for k, (a, b, e1, s1, e2, s2, path, r, kind) in enumerate(plans):
        cuts[e1].append((s1, k, 0))
        cuts[e2].append((s2, k, 1))
    ends = {}
    for e, lst in cuts.items():
        lst.sort()
        tree = net.E[e].tree
        nodes = net.split(e, [x[0] for x in lst])
        for (s, k, w), nid in zip(lst, nodes):
            ends[(k, w)] = (nid, tree)
    for k, (a, b, e1, s1, e2, s2, path, r, kind) in enumerate(plans):
        if (k, 0) not in ends or (k, 1) not in ends:
            continue
        (na, ta), (nb, tb) = ends[(k, 0)], ends[(k, 1)]
        net.edge(na, nb, path, RAMP, cls="anastomosis", side="D", role=kind, tree=None,
                 attach={ta: na, tb: nb}, fixed_r=r)


# =================================================================== stage: the conjunctival web (layout 'web')
def _in_len(pts, rect) -> float:
    """Length of a polyline inside a rectangle (by segment midpoints)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return 0.0
    mid = 0.5 * (pts[1:] + pts[:-1])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return float(seg[_Geo.inside(mid, rect)].sum())


def _web_heading(P: AnatomyParams, F: _Fields, rng, sd_deg=None, iso=None) -> float:
    """A long vessel's direction: about the dominant (radial) one, a share
    in any direction."""
    iso = P.web_iso_frac if iso is None else iso
    if rng.random() < iso:
        return F.theta0 + float(rng.uniform(-0.5 * math.pi, 0.5 * math.pi))
    return F.theta0 + math.radians(P.web_angle_sd_deg if sd_deg is None else sd_deg) * float(rng.standard_normal())


def _web_path(P: AnatomyParams, F: _Fields, geo: _Geo, rng, ang: float, persist=None, hsd=None, tries: int = 10,
              para=None, region=None, min_in=None):
    """The course of a long vessel: from the border point behind a random
    point of the tissue region (or of `region`, a rectangle; looking along
    ang), an OU heading walk (_walk: about the local orientation field plus
    its initial offset) until it leaves the domain.  None if it keeps
    missing the tissue (or, with min_in, runs less than min_in d_c inside
    `region`)."""
    diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
    tx0, ty0, tx1, ty1 = geo.tissue if region is None else region
    hsd = math.radians(P.web_heading_sd_deg) * math.sqrt(P.tortuosity_gain) if hsd is None else hsd
    for _ in range(tries):
        c = np.array([rng.uniform(tx0, tx1), rng.uniform(ty0, ty1)])
        u = np.array([math.cos(ang), math.sin(ang)])
        start = geo.ray_to_border(c, -u)
        pts, why = _walk(start, ang, step=2.0, length=3 * diag,
                         persist=float(rng.uniform(*P.web_persistence)) if persist is None else persist,
                         heading_sd=hsd, F=F, geo=geo, rng=rng, para=para)
        if why == "border" and len(pts) > 30 and _in_len(pts, geo.tissue) > 120 and \
                (min_in is None or _in_len(pts, region) >= min_in):
            return pts
    return None


def _smooth_join(pts, idx, width: int = 6):
    """Laplacian smoothing of a polyline around the given point indices
    (where a companion leaves its host's course)."""
    m = np.zeros(len(pts), bool)
    for i in idx:
        m[max(1, i - width):min(len(pts) - 1, i + width + 1)] = True
    return _laplace(pts, 12, 0.5, m)


class _Web:
    """The long conjunctival vessels (layout 'web'): each a tree of the net
    (kind 'A' / 'V', flagged web) in its own depth layer, from a border node
    (its root: inlet for an arteriole, outlet for a venule) to border nodes
    (where it leaves the domain upstream / downstream), with its Y-forks.
    Radii are Murray bottom-up (_assign_radii) from the radius where it leaves
    the domain (leaf_r) plus what joins it on the way (its branches, twigs;
    not arcades, which are anastomoses), capped at web_grow_max x that
    radius, so a long vessel widens only gently."""

    def __init__(self, net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng, gammas):
        self.net, self.P, self.F, self.geo, self.rng, self.gammas = net, P, F, geo, rng, gammas
        self.trees = []                   # tree ids
        self.length = dict(A=0.0, V=0.0)  # in-tissue length
        self.length_role = defaultdict(float)     # (side, role) -> in-tissue length
        self._para_xy, self._para_t, self._para = [], [], None
        self._para_r, self._para_id = [], []     # per point: the vessel's nominal radius (x 1.6: it widens) and tree
        self._coll_xy = []                        # points of the collecting veins made so far
        self.log = defaultdict(int)

    def para(self):
        """The walk's near-parallel avoidance (_walk para): over the long
        vessels made so far (web_para_mode; v2: turning away within
        web_para_min of one), and (v2) away from the episcleral vessels below
        (a long vessel lying along a deep vein over a long stretch drew a
        dark spindle, or a sharp line on a blurred band's axis: review v1
        T1, T10); None if both are off."""
        P = self.P
        out = []
        if P.web_para_clear > 0 and self._para_xy:
            if self._para is None:
                # the align zone and the turn-away distance grow with the neighbour's radius (v2: a long vessel
                # aligned 4-8 d_c from a collecting vein's axis ran inside its lumen in projection)
                self._para = (cKDTree(np.concatenate(self._para_xy)), np.concatenate(self._para_t), P.web_para_clear,
                              math.cos(math.radians(P.web_para_deg)), math.radians(P.web_para_turn_deg),
                              P.web_para_mode, np.concatenate(self._para_r), P.web_para_min)
            out.append(self._para)
        if P.deep_para_clear > 0:
            if not hasattr(self, "_deep_para"):
                xs, ts, rs = [], [], []
                for E in self.net.E.values():
                    if self.net.layers.get(int(E.lay[0]), {}).get("kind") != "deep" or E.tree is None:
                        continue
                    q = _resample(E.pts, 2.0) if len(E.pts) > 2 else np.asarray(E.pts, float)
                    if len(q) < 3:
                        continue
                    xs.append(q)
                    ts.append(_tangents(q, 1))
                    rs.append(np.full(len(q), float(self.net.trees[E.tree].get("root_r", 2.0))))
                self._deep_para = (cKDTree(np.concatenate(xs)), np.concatenate(ts), P.deep_para_clear,
                                   math.cos(math.radians(P.deep_para_deg)), math.radians(P.web_para_turn_deg), "away",
                                   np.concatenate(rs), 0.0) if xs else None
            if self._deep_para is not None:
                out.append(self._deep_para)
        return out or None

    def _add_para(self, pts, r: float = 1.0, tid: int = -1):
        q = _resample(np.asarray(pts, float), 2.0) if len(pts) > 2 else np.asarray(pts, float)
        if len(q) < 3:
            return
        self._para_xy.append(q)
        self._para_t.append(_tangents(q, 1))
        self._para_r.append(np.full(len(q), 1.6 * float(r)))
        self._para_id.append(np.full(len(q), int(tid)))
        self._para = None
        self._ovl = None

    def overlaps(self, path, r: float, exclude=(), max_len: float = 20.0, deg: float = 25.0, deep: bool = True,
                 extra: float = 0.0) -> bool:
        """Whether a planned course of nominal radius r (+ extra: a bundle's
        half-width) would lie over a long vessel made so far (other than
        `exclude`) or (deep) an episcleral vessel, inside the tissue region:
        its lumen overlapping theirs in projection (centre distance < 1.6 r +
        theirs + 1 d_c; 1.6: they widen downstream) while running within `deg`
        of them, over more than max_len d_c: two dark vessels' long overlap,
        a dark spindle (v2: companions and bundles were placed without looking
        at their neighbours and ran over other long vessels for up to 170
        d_c).  Two thin vessels (both radii under web_dark_r) may overlap:
        thin lines that merge and part are seen in the stills."""
        if getattr(self, "_ovl", None) is None:
            xs, ts, rs, ids = list(self._para_xy), list(self._para_t), list(self._para_r), list(self._para_id)
            self.para()                     # builds the deep set
            if getattr(self, "_deep_para", None) is not None:
                dp = self._deep_para
                xs.append(dp[0].data)
                ts.append(dp[1])
                rs.append(dp[6])
                ids.append(np.full(len(dp[1]), -2))
            if not xs:
                return False
            self._ovl = (cKDTree(np.concatenate(xs)), np.concatenate(ts), np.concatenate(rs), np.concatenate(ids))
        tr, tt, rr, ii = self._ovl
        q = _resample(np.asarray(path, float), 1.0)
        q = q[_Geo.inside(q, self.geo.tissue)]
        if len(q) < 3:
            return False
        tq = _tangents(q, 2)
        # the 24 nearest points within reach (vectorised; the lumens that overlap are among the nearest)
        rs_ = 1.6 * r + extra
        d, j = tr.query(q, k=min(24, len(rr)), distance_upper_bound=rs_ + float(rr.max()) + 1.0)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        ok &= d < rs_ + rr[jj] + 1.0
        if r < self.P.web_dark_r:          # a thin one: only over dark ones (deep: their radius; web: 1.6 r_nom)
            ok &= rr[jj] >= np.where(ii[jj] == -2, 1.0, 1.6) * self.P.web_dark_r
        ok &= np.abs(np.einsum("ijk,ik->ij", tt[jj], tq)) > math.cos(math.radians(deg))
        if exclude:
            ok &= ~np.isin(ii[jj], [int(x) for x in exclude])
        if not deep:
            ok &= ii[jj] != -2
        return float(ok.any(1).sum()) > max_len

    def _pressures(self, side):
        P, rng = self.P, self.rng
        if side == "A":
            return float(rng.uniform(*P.p_art_in)), lambda: float(rng.uniform(*P.p_art_exit))
        return float(rng.uniform(*P.p_ven_out)), lambda: float(rng.uniform(*P.p_ven_entry))

    def tree(self, side: str, pts, r_far: float, *, forks=None, fork_q=None, rmax=None, flags=None,
             role_name="long", hsd=None, persist=None):
        """A long vessel along pts (root first) with its Y-forks: at each,
        the branch leaves at the Murray angle of its share q and walks on to
        the border (steering clear of its own tree); the continuation turns
        by its own Murray angle.  Returns the tree id."""
        net, P, F, geo, rng = self.net, self.P, self.F, self.geo, self.rng
        g = self.gammas[side]
        tid = net.new_tree(kind=side, gamma=g, web=True, role=role_name, r_nom=float(r_far),
                           **({"rmax": float(rmax)} if rmax is not None else {}))
        lay = len(net.layers)
        net.layers[lay] = dict(name=f"web{len(self.trees)}", kind="web", side=side, tree=tid)
        net.trees[tid]["layer"] = lay
        cls = "arteriole" if side == "A" else "venule"
        p_root, p_leaf = self._pressures(side)
        n_root = net.node(pts[0], pressure=p_root)
        diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
        hsd = math.radians(P.web_heading_sd_deg) * math.sqrt(P.tortuosity_gain) if hsd is None else hsd
        # forks along the course, root to leaf
        course = np.asarray(pts, float)
        pieces = []                        # (points, None): the course, root -> fork -> ... -> leaf
        branches = []
        x_next = float(np.exp(rng.uniform(*np.log(forks)))) if forks else np.inf
        seg_pts = course
        big = geo.tissue[0] - 80, geo.tissue[1] - 80, geo.tissue[2] + 80, geo.tissue[3] + 80
        edges_made = []
        while True:
            s = _arclen(seg_pts)
            L = float(s[-1])
            if x_next >= L - 60:
                break
            xy = np.array([np.interp(x_next, s, seg_pts[:, 0]), np.interp(x_next, s, seg_pts[:, 1])])
            if not _Geo.inside(xy, big)[0] or geo.border_dist(xy[None])[0] < 50 or x_next < 40:
                x_next += float(np.exp(rng.uniform(*np.log(forks))))
                continue
            back = np.array([np.interp(x_next - 6, s, seg_pts[:, 0]), np.interp(x_next - 6, s, seg_pts[:, 1])])
            t = _unit(xy - back)
            q = float(rng.uniform(*fork_q))
            a1, a2 = _murray_angles(q, rng, P.angle_noise_deg)
            if role_name == "collecting":
                # a wide trunk keeps its course through the confluence; the thin tributary takes the angle
                # between the two (v2: the trunk turned by its Murray angle, 20 +- 13 deg at q 0.2: elbows)
                a1, a2 = 0.0, a1 + a2
            else:
                # the larger daughter keeps close to its Murray angle (v2: the noise, 12.7 deg SD per daughter, turned
                # long vessels by up to 40 deg at a thin branch; the noise stays on the branch)
                a1 = _murray_angles(q, rng, 0.0)[0]
            a2 = min(a2, math.radians(70))
            sgn = 1 if rng.random() < 0.5 else -1
            r_b = r_far * (q / (1 - q)) ** (1 / g)
            head = seg_pts[s <= x_next]
            own = np.concatenate([head] + [b for b, _ in branches] + [p for p, _ in pieces]) if pieces or branches \
                else head
            # the continuation keeps the planned course beyond the fork, turned rigidly about the fork by its
            # Murray angle a1 (small for a thin branch): the trunk's curvature stays continuous and the branch
            # takes the angle.  (It was a fresh walk from the fork, steering around the branch just walked: wide
            # collecting veins turned 40-70 deg at a thin tributary; realism review v1, T4.)
            cp, why_c = self._continue(seg_pts[s > x_next], xy, -sgn * a1, hsd, persist)
            hb = _rot(t, sgn * a2)
            bp, why = _walk(xy, math.atan2(hb[1], hb[0]), step=2.0, length=3 * diag,
                            persist=float(rng.uniform(*P.web_persistence)) if persist is None else persist,
                            heading_sd=hsd, F=F, geo=geo, rng=rng,
                            # (v2) a collecting vein's tributary keeps clear of the other collecting veins too: two
                            # wide veins crossing need depths clear of each other's diameters and the overlap pass
                            # removed one
                            obstacles=np.concatenate([own, cp] + (self._coll_xy if role_name == "collecting" else [])),
                            clear=2 * (r_far + r_b) + 6.0, exempt_xy=xy, exempt_r=3 * (r_far + r_b) + 10,
                            para=self.para() if role_name in ("long", "collecting") else None)
            if why != "border" or why_c != "border" or len(bp) < 10 or len(cp) < 10:
                x_next += float(np.exp(rng.uniform(*np.log(forks))))
                continue
            pieces.append((np.concatenate([head, [xy]]), None))
            branches.append((bp, r_b))
            seg_pts = cp
            x_next = float(np.exp(rng.uniform(*np.log(forks))))
        pieces.append((seg_pts, None))
        # nodes and edges: trunk pieces root -> fork -> ... -> leaf, branches from the forks
        u = n_root
        for i, (pp, _) in enumerate(pieces):
            last = i == len(pieces) - 1
            v = net.node(pp[-1], pressure=p_leaf()) if last else net.node(pp[-1])
            e = net.edge(u, v, pp, lay, cls=cls, side=side, role="web", tree=tid, leaf_r=r_far if last else None,
                         flags=dict(flags or {}))
            edges_made.append(e)
            if not last:
                bp, r_b = branches[i]
                nl = net.node(bp[-1], pressure=p_leaf())
                edges_made.append(net.edge(v, nl, bp, lay, cls=cls, side=side, role="web", tree=tid, leaf_r=r_b,
                                           flags=dict(flags or {})))
                self.log["forks"] += 1
            u = v
        allp = [net.E[e].pts for e in edges_made]
        for p_ in allp:
            self._add_para(p_, r_far, tid)
        ln = sum(_in_len(p, geo.tissue) for p in allp)
        self.length[side] += ln
        self.length_role[(side, role_name)] += ln
        self.trees.append(tid)
        net.trees[tid]["course"] = np.concatenate([pieces[0][0]] + [p[1:] for p, _ in pieces[1:]])
        if role_name == "collecting":
            self._coll_xy += [_resample(p_, 2.0) for p_ in allp if len(p_) > 1]
        self.log[role_name] += 1
        return tid

    def _continue(self, rest, xy, ang: float, hsd, persist):
        """The trunk beyond a fork: the rest of its planned course (after xy),
        turned rigidly about xy by ang, cut where it leaves the domain or
        extended by a walk along its last heading if the turn brought its end
        inside.  Returns (points from xy, 'border' or the walk's reason)."""
        P, F, geo, rng = self.P, self.F, self.geo, self.rng
        if len(rest) < 2:
            return np.asarray([xy], float), "short"
        ca, sa = math.cos(ang), math.sin(ang)
        rel = np.asarray(rest, float) - xy
        cp = np.concatenate([[xy], xy + np.stack([ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]],
                                                 1)])
        out = np.nonzero(~_Geo.inside(cp, geo.domain))[0]
        if len(out):
            i = int(out[0])
            return np.concatenate([cp[:i], [geo.exit_point(cp[i - 1], cp[i])]]), "border"
        h = _unit(cp[-1] - cp[-2])
        diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
        ext, why = _walk(cp[-1], math.atan2(h[1], h[0]), step=2.0, length=diag,
                         persist=float(rng.uniform(*P.web_persistence)) if persist is None else persist,
                         heading_sd=hsd, F=F, geo=geo, rng=rng)
        return np.concatenate([cp, ext[1:]]), why

    def offset_run(self, course, off: float, sgn: int, run: tuple, avoid_curv: float = 0.4):
        """A stretch of `course` offset sideways by `off` (sgn: which side),
        about `run` long, placed across the tissue region, and the
        diverging walks out of both its ends to the border (turned away from
        the course by web_pair_diverge_deg).  Returns (points, i_start,
        i_end) of the whole path (root end first, as `course`), or None."""
        P, F, geo, rng = self.P, self.F, self.geo, self.rng
        diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
        H = _resample(course, 1.0)
        L = len(H) - 1
        Lr = float(rng.uniform(*run))
        inside = np.nonzero(_Geo.inside(H, geo.tissue))[0]
        if len(inside) < 20 or L < 140:
            return None
        Lr = min(Lr, L - 80)
        if Lr < 60:
            return None
        mid_lo, mid_hi = max(40 + Lr / 2, inside[0]), min(L - 40 - Lr / 2, inside[-1])
        mid = float(rng.uniform(mid_lo, mid_hi)) if mid_hi > mid_lo else 0.5 * (mid_lo + mid_hi)
        i0, i1 = int(max(40, mid - Lr / 2)), int(min(L - 40, mid + Lr / 2))
        sm = _laplace(H, 8)
        tan = _tangents(sm, 3)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        kap = _curv(_laplace(H, 12))
        bad = np.nonzero(kap[i0:i1 + 1] * abs(off) > avoid_curv)[0]
        if len(bad):                          # keep the longest stretch between bends too tight for the offset
            cuts = np.concatenate([[-1], bad, [i1 - i0 + 1]])
            k = int(np.argmax(np.diff(cuts)))
            i0, i1 = i0 + int(cuts[k]) + 1, i0 + int(cuts[k + 1]) - 1
            if i1 - i0 < 50:
                return None
        w = H[i0:i1 + 1] + sgn * off * nrm[i0:i1 + 1]
        dv = math.radians(float(rng.uniform(*P.web_pair_diverge_deg)))
        hb = _unit(-tan[i0] * math.cos(dv) + sgn * nrm[i0] * math.sin(dv))
        hf = _unit(tan[i1] * math.cos(dv) + sgn * nrm[i1] * math.sin(dv))
        hsd = math.radians(P.web_heading_sd_deg) * math.sqrt(P.tortuosity_gain)
        out = []
        for p0, h in ((w[0], hb), (w[-1], hf)):
            pp, why = _walk(p0, math.atan2(h[1], h[0]), step=2.0, length=3 * diag,
                            persist=float(rng.uniform(*P.web_persistence)), heading_sd=hsd, F=F, geo=geo, rng=rng)
            if why != "border" or len(pp) < 3:
                return None
            out.append(pp)
        back, fwd = out
        path = np.concatenate([back[::-1][:-1], w, fwd[1:]])
        j0, j1 = len(back) - 1, len(back) - 1 + len(w) - 1
        path = _smooth_join(path, [j0, j1])
        return path, j0, j1

    def companion(self, host: int, side: str, r_c: float, gap: tuple, run: tuple, sgn: int | None = None):
        """A companion of the other class beside a long vessel for a run of
        `run` (through its forks), at a wall gap of gap x (r_a + r_v), then
        diverging; its own tree (and depth).  Its root lies at its host's
        root end (the pair's flows run countercurrent: an arteriole in, its
        venule out)."""
        net, rng, P = self.net, self.rng, self.P
        T = net.trees[host]
        # the host widens downstream by Murray (up to web_grow_max, a collecting vein to its cap): the offset is
        # set for 1.6 x its nominal radius (it was 1.2 x: companions of collecting veins, placed at a wall gap of
        # 0.2-1.0 (r_a + r_v), ran inside the widened vein in projection)
        r_h = min(T["r_nom"] * 1.6, T.get("rmax", np.inf))
        off = (r_h + r_c) * (1.0 + float(rng.uniform(*gap)))
        sgn = (1 if rng.random() < 0.5 else -1) if sgn is None else sgn
        got = self.offset_run(T["course"], off, sgn, run)
        # its run beside the host not over another long vessel or a deep one: the other side, else no companion (v2)
        if got is not None and self.overlaps(got[0][got[1]:got[2] + 1], r_c, exclude=(host,)):
            got = self.offset_run(T["course"], off, -sgn, run)
            if got is not None and self.overlaps(got[0][got[1]:got[2] + 1], r_c, exclude=(host,)):
                got = None
            sgn = -sgn
        if got is None:
            return None
        path, j0, j1 = got
        ct = self.tree(side, path, r_c, flags=dict(pairtort=True), role_name="companion",
                       rmax=min(0.5 * (P.art_diam_max if side == "A" else P.ven_diam_max), P.web_grow_max * r_c))
        net.trees[ct].update(pair_host=int(host), pair_off=float(off), pair_run=path[j0:j1 + 1],
                             lead=int(host), lead_off=float(off))
        T.setdefault("pair_sides", []).append(int(sgn))
        for e, E in net.E.items():
            if E.tree == host:
                E.flags["pairtort"] = True
        return ct

    def bundle(self):
        """3-5 thin vessels (alternately venules and arterioles: an arteriole
        with its venae comitantes) running side by side over a long run, then
        fanning out; each its own tree and depth.  v2: irregular, as the
        stills' doubled and tripled lines are (v1 drew 3-5 identical lines at
        constant, even spacings: a comb): unequal calibres (log-uniform
        bundle_r), the spacing of each member drifting along the run
        (bundle_drift, never crossing its neighbour: walls >= 0.2 d_c apart, so
        neighbours may touch in projection, at different depths), and members
        joining and leaving the bundle at different places (bundle_stagger:
        only the middle one, the guide, runs the whole length)."""
        P, F, geo, rng, net = self.P, self.F, self.geo, self.rng, self.net
        ang = _web_heading(P, F, rng)
        guide = _web_path(P, F, geo, rng, ang, persist=float(rng.uniform(200, 400)), hsd=math.radians(5.0))
        if guide is None:
            return []
        m = int(rng.integers(P.bundle_size[0], P.bundle_size[1] + 1))
        lo_r, hi_r = math.log(P.bundle_r[0]), math.log(P.bundle_r[1])
        rr = np.exp(rng.uniform(lo_r, hi_r, size=m))
        first = int(rng.integers(2))
        sides = ["V" if (i + first) % 2 == 0 else "A" for i in range(m)]
        rr = np.array([r * (P.dilation if s == "V" else P.art_dilation) for r, s in zip(rr, sides)])
        c = np.concatenate([[0.0], np.cumsum(rr[:-1] + rr[1:] + rng.uniform(*P.bundle_gap, size=m - 1)
                                             * max(P.dilation, 1.0))])
        c -= c.mean()
        H = _resample(guide, 1.0)
        L = len(H) - 1
        inside = np.nonzero(_Geo.inside(H, geo.tissue))[0]
        if len(inside) < 20 or L < 200:
            return []
        Lr = min(float(rng.uniform(*P.bundle_run)), L - 100)
        mid_lo, mid_hi = max(50 + Lr / 2, inside[0]), min(L - 50 - Lr / 2, inside[-1])
        for _ in range(6):                    # (v2) its run not along another long vessel or a deep one
            mid = float(rng.uniform(mid_lo, mid_hi)) if mid_hi > mid_lo else 0.5 * (mid_lo + mid_hi)
            i0, i1 = int(max(50, mid - Lr / 2)), int(min(L - 50, mid + Lr / 2))
            ok = not self.overlaps(H[i0:i1 + 1], float(rr.max()), extra=float(np.abs(c).max()))
            if ok:
                break
        if not ok:
            return []
        tan = _tangents(_laplace(H, 8), 3)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        diag = math.hypot(geo.domain[2] - geo.domain[0], geo.domain[3] - geo.domain[1])
        rev = rng.random() < 0.5             # which end is the bundle's root end
        n = i1 - i0 + 1
        li = int(np.argmin([abs(i - 0.5 * (m - 1)) for i in range(m)]))     # the middle member: the guide
        corr = float(rng.uniform(*P.bundle_drift_corr))
        cs = np.array([c[i] + (np.zeros(n) if i == li else P.bundle_drift * max(P.dilation, 1.0)
                               * _smooth_noise(n, corr, rng)) for i in range(m)]).reshape(m, n)
        for i in range(li + 1, m):           # neighbours never cross (their walls stay apart)
            cs[i] = np.maximum(cs[i], cs[i - 1] + rr[i] + rr[i - 1] + 0.2)
        for i in range(li - 1, -1, -1):
            cs[i] = np.minimum(cs[i], cs[i + 1] - rr[i] - rr[i + 1] - 0.2)
        span = []
        for i in range(m):                   # where each member runs beside the guide
            if i == li:
                span.append((0, n - 1))
            else:
                a = int(rng.uniform(0.0, P.bundle_stagger) * n)
                b = n - 1 - int(rng.uniform(0.0, P.bundle_stagger) * n)
                span.append((a, b) if b - a >= 40 else (0, n - 1))
        self.log["bundles"] += 1
        bid = int(self.log["bundles"])
        tids = []
        for i in range(m):
            a, b = span[i]
            k0, k1 = i0 + a, i0 + b
            w = H[k0:k1 + 1] + cs[i, a:b + 1, None] * nrm[k0:k1 + 1]
            side_i = np.sign(cs[i, a:b + 1].mean()) if abs(cs[i, a:b + 1].mean()) > 1e-6 else 0.0
            fan = math.radians(float(rng.uniform(4, 12))) * (side_i if side_i
                                                              else float(rng.choice([-1, 1])) * 0.4)
            hb = _unit(-tan[k0] * math.cos(fan) + nrm[k0] * math.sin(fan))
            hf = _unit(tan[k1] * math.cos(fan) + nrm[k1] * math.sin(fan))
            out = []
            for p0, h in ((w[0], hb), (w[-1], hf)):
                pp, why = _walk(p0, math.atan2(h[1], h[0]), step=2.0, length=3 * diag,
                                persist=float(rng.uniform(*P.web_persistence)), heading_sd=math.radians(5.0),
                                F=F, geo=geo, rng=rng)
                out.append(pp if why == "border" else None)
            if out[0] is None or out[1] is None:
                continue
            path = np.concatenate([out[0][::-1][:-1], w, out[1][1:]])
            j0 = len(out[0]) - 1
            path = _smooth_join(path, [j0, j0 + len(w) - 1])
            if rev:
                path = path[::-1]
            t = self.tree(sides[i], path, float(rr[i]), flags=dict(pairtort=True), role_name="bundle",
                          rmax=P.web_grow_max * float(rr[i]))
            run_pts = (path[::-1] if rev else path)[j0:j0 + len(w)]
            net.trees[t].update(bundle=bid, pair_run=run_pts)
            tids.append((t, sides[i], i))

        def off(i, j):                        # the mean centre spacing of two members where both run
            a, b = max(span[i][0], span[j][0]), min(span[i][1], span[j][1])
            return float(np.mean(np.abs(cs[i, a:b + 1] - cs[j, a:b + 1]))) if b > a else float(abs(c[i] - c[j]))
        for t, s, i in tids:                  # partner: the nearest member of the other class
            others = sorted((abs(i - j), t2, j) for t2, s2, j in tids if s2 != s)
            if others:
                _, t2, j = others[0]
                net.trees[t]["pair_host"] = int(t2)
                net.trees[t]["pair_off"] = off(i, j)
        lead = [x for x in tids if x[2] == li] or ([min(tids, key=lambda x: (abs(x[2] - 0.5 * (m - 1)), x[2]))]
                                                  if tids else [])
        if lead:                              # the members meander together: they follow the guide's
            lt, _, lj = lead[0]               # displacement (_tortuosity)
            for t, s, i in tids:
                if t != lt:
                    net.trees[t].update(lead=int(lt), lead_off=off(i, lj))
        return [t for t, _, _ in tids]


def _grow_web(net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng, gammas) -> _Web:
    """The long conjunctival vessels (the _Web class): large collecting
    venules with their tributaries and companion arterioles; bundles of thin
    parallel vessels; long arterioles, most with a companion venule, and long
    venules, some with a companion arteriole, until their length densities
    over the tissue region reach 1 / web_art_spacing and 1 / web_ven_spacing."""
    W = _Web(net, P, F, geo, rng, gammas)
    x0, y0, x1, y1 = geo.tissue
    area = (x1 - x0) * (y1 - y0)
    budget = dict(A=area / P.web_art_spacing, V=area / P.web_ven_spacing)
    lu = lambda ab: float(math.exp(rng.uniform(math.log(ab[0]), math.log(ab[1]))))  # noqa: E731
    # collecting venules: straight, wide, fed by long tributaries
    n_coll = P.collect_min + int(rng.poisson(max(P.collect_rate * area / 1e5 - P.collect_min, 0.0)))
    fx0, fy0, fx1, fy1 = geo.frame
    cx, cy, hw, hh = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1), 0.5 * P.collect_center * (fx1 - fx0), \
        0.5 * P.collect_center * (fy1 - fy0)
    centre = (cx - hw, cy - hh, cx + hw, cy + hh)
    coll = []                                  # (course points, tangents, radius) of the collecting veins made
    for _ in range(n_coll):
        r = float(rng.uniform(*P.collect_r)) * P.dilation
        pts = None
        for _try in range(8):
            ang = _web_heading(P, F, rng, sd_deg=P.web_angle_sd_deg * 1.5)
            # through the central part of the frame (v2: a vein that only grazed a corner left the frame without a
            # black vessel); gently curving (v2: hsd 4 deg drew ruler-straight veins across the frame, the stills'
            # big veins arc)
            cand = _web_path(P, F, geo, rng, ang, persist=float(rng.uniform(*P.collect_persist)),
                             hsd=math.radians(P.collect_heading_sd_deg), para=W.para(),
                             region=centre if P.collect_center > 0 else None,
                             min_in=min(hw, hh) if P.collect_center > 0 else None)
            if cand is None or W.overlaps(cand, r, deep=False):
                continue
            # not beside or across another collecting vein: two big veins running side by side and crossing at a
            # shallow angle drew a black spindle, and two collecting venules of one plexus that meet join rather
            # than cross (crossing, each needs a depth clear of the other's whole diameter: the pathologic preset's
            # dilated pairs could not clear and the overlap pass removed one) (v2)
            q = _resample(cand, 2.0)
            tq = _tangents(q, 2)
            near = False
            tis = _Geo.inside(q, geo.tissue)
            for cp_, ct_, cr_ in coll:
                d, j = cKDTree(cp_).query(q)
                m = d < 3.0 * (r + cr_) + 10.0
                if (m.any() and float(np.max(np.abs(np.sum(tq[m] * ct_[j[m]], 1)))) > math.cos(math.radians(35))) \
                        or bool(np.any((d < r + cr_ + 4.0) & tis)):
                    near = True
                    break
            if not near:
                pts = cand
                break
        if pts is None:
            continue
        q = _resample(pts, 2.0)
        if rng.random() < 0.5:
            pts = pts[::-1]
        t = W.tree("V", pts, r, forks=P.collect_tributaries, fork_q=(0.08, 0.3),
                   rmax=0.5 * P.collect_diam_max, role_name="collecting")
        for E in net.E.values():           # its course and tributaries (the next one keeps off them all)
            if E.tree == t and len(E.pts) > 1:
                q = _resample(E.pts, 2.0)
                coll.append((q, _tangents(q, 2), r))
        if t is not None and rng.random() < P.p_collect_companion:
            W.companion(t, "A", lu(P.web_r_A) * P.art_dilation, P.collect_pair_gap, P.web_pair_run)
    for _ in range(rng.poisson(P.bundle_rate * area / 1e5)):
        W.bundle()
    # long arterioles (with companion venules) and long venules (some with companion arterioles), each time of
    # the side further from its length budget (companions count on their own side), until both are reached
    # A share of each side's budget (P.web_plain_min) is kept for plain long vessels: the twigs that feed and
    # drain the capillary mesh leave them, and a scene whose budget the collecting veins and bundles had filled
    # had none (healthy seed 25: 5 collecting veins, 6 bundles, no long vessel, no twig; correctness review v1,
    # defect 2)
    tries = 0
    while tries < 400:
        fill = {k: W.length[k] / max(budget[k], 1e-9) for k in ("A", "V")}
        plain = {k: W.length_role[(k, "long")] / max(budget[k], 1e-9) for k in ("A", "V")}
        short = {k: fill[k] if plain[k] >= P.web_plain_min else plain[k] - 1.0 for k in ("A", "V")}
        side = min(short, key=short.get)
        if short[side] >= 1.0:
            break
        tries += 1
        pts = _web_path(P, F, geo, rng, _web_heading(P, F, rng), para=W.para())
        if pts is None:
            continue
        if rng.random() < 0.5:
            pts = pts[::-1]
        r = lu(P.web_r_A if side == "A" else P.web_r_V) * (P.art_dilation if side == "A" else P.dilation)
        if W.overlaps(pts, r):               # (v2) not lying along another long vessel or a deep one
            continue
        t = W.tree(side, pts, r, forks=P.web_fork_spacing, fork_q=P.web_fork_q,
                   rmax=min(0.5 * (P.art_diam_max if side == "A" else P.ven_diam_max), P.web_grow_max * r))
        p = P.web_p_pair_A if side == "A" else P.web_p_pair_V
        if t is not None and rng.random() < p:
            o = "V" if side == "A" else "A"
            r_c = r * float(rng.uniform(*P.web_pair_ratio)) ** (1 if side == "A" else -1)
            W.companion(t, o, r_c, P.web_pair_gap, P.web_pair_run)
            if side == "A" and rng.random() < P.web_p_pair_A2 and net.trees[t].get("pair_sides"):
                # venae comitantes: a second venule on the arteriole's other side
                r_c2 = r * float(rng.uniform(*P.web_pair_ratio))
                W.companion(t, o, r_c2, P.web_pair_gap, P.web_pair_run, sgn=-net.trees[t]["pair_sides"][0])
    return W


def _web_points(net: _Net, eids, spacing: float):
    """Points (xy, eid, s, unit tangent) every ~spacing along edges."""
    xs, es, ss, ts = [], [], [], []
    for e in eids:
        p = net.E[e].pts
        s = _arclen(p)
        if s[-1] < 2 * spacing:
            continue
        q = np.arange(spacing, s[-1] - spacing * 0.5, spacing)
        xy = np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)
        a = np.stack([np.interp(q - 1.5, s, p[:, 0]), np.interp(q - 1.5, s, p[:, 1])], 1)
        b = np.stack([np.interp(q + 1.5, s, p[:, 0]), np.interp(q + 1.5, s, p[:, 1])], 1)
        xs.append(xy)
        es.append(np.full(len(q), e))
        ss.append(q)
        ts.append(_unit(b - a))
    if not xs:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0), np.zeros((0, 2))
    return np.concatenate(xs), np.concatenate(es), np.concatenate(ss), np.concatenate(ts)


def _split_many(net: _Net, cuts: dict) -> dict:
    """Split every edge once at all its planned positions: cuts {eid: [(s,
    key)]} -> {key: node}."""
    out = {}
    for e, lst in cuts.items():
        lst = sorted(lst)
        nodes = net.split(e, [x[0] for x in lst])
        for (s, key), nid in zip(lst, nodes):
            out[key] = nid
    return out


class _NodeSet:
    """Positions of the nodes placed so far on the long vessels (no new
    junction closer than dmin to one)."""

    def __init__(self, pts):
        self.pts = np.array(pts, float).reshape(-1, 2)

    def clear(self, xy, dmin: float) -> bool:
        return not len(self.pts) or float(np.min(np.hypot(*(self.pts - xy).T))) >= dmin

    def add(self, *xy):
        self.pts = np.concatenate([self.pts, np.array(xy, float).reshape(-1, 2)])


def _partners(net: _Net, a: int, b: int) -> bool:
    """Two long vessels run as a pair (companion and host) or in one bundle."""
    A_, B_ = net.trees[a], net.trees[b]
    return (A_.get("pair_host") == b or B_.get("pair_host") == a or
            (A_.get("bundle") is not None and A_.get("bundle") == B_.get("bundle")))


def _tree_kd(net: _Net, tids) -> dict:
    """KD-tree of the points of each tree's edges."""
    out = {}
    for t in tids:
        p = [E.pts for E in net.E.values() if E.tree == t]
        if p:
            out[t] = cKDTree(np.concatenate(p))
    return out


def _crosses_web(path, own, web: "_Web", tkd: dict, r: float, ends=(), end_r: float = 0.0) -> bool:
    """Whether a connector (a stalk, an arcade) passes over or under a long
    vessel other than `own` (points within end_r of `ends` excepted)."""
    p = np.asarray(path, float)
    if len(ends):
        far = np.ones(len(p), bool)
        for e in ends:
            far &= np.hypot(*(p - e).T) > end_r
        p = p[far]
    if not len(p):
        return False
    for t in web.trees:
        if t in own or t not in tkd:
            continue
        rt = 1.3 * web.net.trees[t]["r_nom"] + r + 3.0
        if float(tkd[t].query(p, distance_upper_bound=rt)[0].min()) < rt:
            return True
    return False


def _add_arcades(net: _Net, P: AnatomyParams, geo: _Geo, rng, web: _Web) -> int:
    """Arcades: a vessel joining two neighbouring long vessels of one side
    (arteriole to arteriole, venule to venule; Singh 2021, Jiang 2014: an
    anastomosing network), leaving each obliquely (30-65 deg) on a smooth
    curve, about one per web_arcade_spacing of long vessel.  Calibre
    web_arcade_r x the smaller vessel's; it climbs between their depths."""
    made = 0
    nodes = _NodeSet([net.xy[n] for e, E in net.E.items() if E.role == "web" for n in (E.u, E.v)])
    tkd = _tree_kd(net, web.trees)
    plans = []
    arc_pts = []
    for side in ("A", "V"):
        eids = [e for e, E in net.E.items() if E.role == "web" and E.side == side]
        xy, es, ss, ts = _web_points(net, eids, 4.0)
        if len(xy) < 10:
            continue
        keep = _Geo.inside(xy, geo.tissue)
        xy, es, ss, ts = xy[keep], es[keep], ss[keep], ts[keep]
        if len(xy) < 10:
            continue
        tree_of = np.array([net.E[e].tree for e in es])
        rnom = np.array([net.trees[t]["r_nom"] for t in tree_of])
        kd = cKDTree(xy)
        n_target = rng.poisson(web.length[side] / P.web_arcade_spacing)
        got, tries = 0, 0
        lo, hi = P.web_arcade_len
        while got < n_target and tries < 40 * max(n_target, 1):
            tries += 1
            i = int(rng.integers(len(xy)))
            js = [j for j in kd.query_ball_point(xy[i], hi) if tree_of[j] != tree_of[i]
                  and np.hypot(*(xy[j] - xy[i])) >= lo and not _partners(net, tree_of[i], tree_of[j])]
            if not js:
                continue
            j = int(js[rng.integers(len(js))])
            a, b = xy[i], xy[j]
            c = b - a
            dc = float(np.hypot(*c))
            ta = ts[i] if ts[i] @ c >= 0 else -ts[i]
            tb = ts[j] if ts[j] @ c >= 0 else -ts[j]
            # it must leave and join obliquely: the chord 20-70 deg from both vessels
            ca, cb = float(ta @ c) / dc, float(tb @ c) / dc
            if not (math.cos(math.radians(70)) < ca < math.cos(math.radians(20)) and
                    math.cos(math.radians(70)) < cb < math.cos(math.radians(20))):
                continue
            na = _unit(c - (c @ ta) * ta)
            nb = _unit(c - (c @ tb) * tb)
            al, be = math.radians(float(rng.uniform(30, 65))), math.radians(float(rng.uniform(30, 65)))
            da = _unit(ta * math.cos(al) + na * math.sin(al))
            db = _unit(tb * math.cos(be) + nb * math.sin(be))
            lam = 0.4 * dc
            p1, p2 = a + lam * da, b - lam * db
            tt = np.linspace(0, 1, max(12, int(2 * dc)))[:, None]
            path = (1 - tt) ** 3 * a + 3 * (1 - tt) ** 2 * tt * p1 + 3 * (1 - tt) * tt ** 2 * p2 + tt ** 3 * b
            # web_arcade_r x the smaller vessel, at least a capillary (and at most 0.8 x the smaller vessel)
            r_arc = min(max(float(rng.uniform(*P.web_arcade_r)) * min(rnom[i], rnom[j]), 0.5 * P.cap_diam),
                        0.8 * min(rnom[i], rnom[j]))
            kap = _curv(_resample(path, 0.5))
            if kap.max() > 1.0 / max(6.0, 4 * r_arc):
                continue
            # clear of both vessels away from its ends, and of other nodes at its ends
            ok = True
            for end, host, rh in ((a, tree_of[i], rnom[i]), (b, tree_of[j], rnom[j])):
                far = np.hypot(*(path - end).T) > 3 * (rh + r_arc) + 6
                if far.any() and float(tkd[host].query(path[far])[0].min()) < 1.3 * rh + r_arc + 2.5:
                    ok = False
                    break
            # and it crosses no other long vessel (it climbs between its two vessels' depths: it would pass
            # through the depth of a vessel it crosses)
            if ok and _crosses_web(path, (tree_of[i], tree_of[j]), web, tkd, r_arc,
                                   (a, b), 3 * (max(rnom[i], rnom[j]) + r_arc) + 6):
                ok = False
            if ok and arc_pts and float(cKDTree(np.concatenate(arc_pts)).query(path)[0].min()) < 2 * r_arc + 4.0:
                ok = False                # nor another arcade
            if not ok:
                continue
            dmin = max(12.0, 3 * (rnom[i] + rnom[j]) + 6)
            if not (nodes.clear(a, dmin) and nodes.clear(b, dmin)):
                continue
            nodes.add(a, b)
            arc_pts.append(path)
            plans.append((int(es[i]), float(ss[i]), int(es[j]), float(ss[j]), path, r_arc, side,
                          int(tree_of[i]), int(tree_of[j])))
            got += 1
    cuts = defaultdict(list)
    for k, (e1, s1, e2, s2, *_) in enumerate(plans):
        cuts[e1].append((s1, (k, 0)))
        cuts[e2].append((s2, (k, 1)))
    ends = _split_many(net, cuts)
    for k, (e1, s1, e2, s2, path, r_arc, side, ta, tb) in enumerate(plans):
        if (k, 0) not in ends or (k, 1) not in ends:
            continue
        na, nb = ends[(k, 0)], ends[(k, 1)]
        net.edge(na, nb, path, RAMP, cls="arteriole" if side == "A" else "venule", side=side, role="arcade",
                 tree=None, attach={ta: na, tb: nb}, fixed_r=r_arc)
        made += 1
    web.log["arcades"] = made
    return made


def _twig_roots(net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng, sheets, web: _Web) -> dict:
    """Twigs: terminal arterioles and venules leave the long vessels (up to
    twig_rmax) about every twig_spacing, at 40-75 deg, climb over a stalk to
    a twig sheet under the capillary mesh and branch there (space
    colonization, reaching twig_len from their vessel); their tips join the
    mesh (_attach).  Returns {sheet name: [(start node, heading, tree)]}."""
    by_side = defaultdict(list)
    for s in sheets:
        by_side[s["side"]].append(s)
    nodes = _NodeSet([net.xy[n] for e, E in net.E.items() if E.role in ("web", "arcade") for n in (E.u, E.v)])
    tkd = _tree_kd(net, web.trees)
    plans = []
    for relax in (False, True):
        _twig_plans(net, P, F, geo, rng, web, by_side, nodes, tkd, plans, relax)
        if relax or {net.E[p[0]].side for p in plans} >= {s for s in by_side if by_side[s]}:
            break                         # every side with a twig sheet has twigs: done
        web.log["twigs_relaxed"] += 1     # a side had none (its long vessels all outside the twig window, e.g.
                                          # all collecting veins and bundles): its mesh would carry no flow
    cuts = defaultdict(list)
    for k, (e, x, *_rest) in enumerate(plans):
        cuts[e].append((x, k))
    ends = _split_many(net, cuts)
    starts = defaultdict(list)
    for k, (e, x, sp, side, wt) in enumerate(plans):
        if k not in ends:
            continue
        nr = ends[k]
        sh = by_side[side][int(rng.integers(len(by_side[side])))]
        tid = net.new_tree(kind=side, gamma=None, sheet=sh["name"], twig=True,
                           max_len=float(rng.uniform(*P.twig_len)))
        nt = net.node(sp[-1])
        net.edge(nr, nt, sp, RAMP, cls="arteriole" if side == "A" else "venule", side=side, role="stalk",
                 tree=tid, attach={wt: nr})
        starts[sh["name"]].append((nt, _unit(sp[-1] - sp[-2]), tid))
        web.log["twigs"] += 1
    return starts


def _twig_plans(net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng, web: _Web, by_side, nodes, tkd, plans,
                relax: bool = False):
    """The twig roots of _twig_roots, appended to plans: along every long
    vessel in the twig window (its tree's nominal radius, or the vessel's own
    where it leaves the domain: a collecting vein's tributary; not bundle
    members), or with relax along every long vessel of a side that had none
    (bundle members too)."""
    have = {net.E[p[0]].side for p in plans}
    items = sorted(net.E.items())
    if relax:                             # the fallback: a few twigs (RELAX_TWIGS per side), from long vessels in
        items = [items[i] for i in rng.permutation(len(items))]      # random order, enough to feed / drain the mesh
    n_relax = defaultdict(int)
    for e, E in items:
        if E.role != "web" or not by_side.get(E.side) or (relax and (E.side in have or n_relax[E.side] >= RELAX_TWIGS)):
            continue
        T = net.trees[E.tree]
        dil = max(P.dilation if E.side == "V" else P.art_dilation, 1.0)      # the window scales with dilation
        own = E.leaf_r if E.leaf_r is not None else T["r_nom"]
        # (not bundle members, unless relaxed: twigs joining would widen them by Murray into their neighbours)
        if not relax and (T.get("role") == "bundle" or not (P.twig_rmin * dil <= T["r_nom"] <= P.twig_rmax * dil or
                                                             P.twig_rmin * dil <= own <= P.twig_rmax * dil)):
            continue
        s = _arclen(E.pts)
        L = float(s[-1])
        otree = tkd[E.tree]
        spacing = P.twig_spacing * (2.0 if relax else 1.0)       # the fallback is sparser
        x = float(rng.uniform(0.3, 1.0)) * spacing
        while x < L - 15:
            p = np.array([np.interp(x, s, E.pts[:, 0]), np.interp(x, s, E.pts[:, 1])])
            q = np.array([np.interp(x + 1.5, s, E.pts[:, 0]), np.interp(x + 1.5, s, E.pts[:, 1])])
            step = spacing * float(rng.uniform(0.6, 1.4))
            dmin = max(12.0, 3 * T["r_nom"] + 8)
            if x > 15 and _Geo.inside(p, geo.tissue)[0] and nodes.clear(p, dmin):
                t = _unit(q - p)
                sgn = 1 if rng.random() < 0.5 else -1
                for side_try in (sgn, -sgn):      # the other side if the first stalk is blocked
                    h = _rot(t, side_try * math.radians(float(rng.uniform(40, 75))))
                    Ls = float(rng.uniform(*P.twig_stalk))
                    sp, why = _walk(p, math.atan2(h[1], h[0]), step=1.5, length=Ls, persist=25.0,
                                    heading_sd=math.radians(10.0), F=F, geo=geo, rng=rng)
                    far = np.hypot(*(sp - p).T) > 3 * T["r_nom"] + 6
                    clear = not far.any() or float(otree.query(sp[far])[0].min()) > 1.3 * T["r_nom"] + 3.0
                    # the stalk climbs from its vessel's depth to the twig sheet: over no other long vessel (relaxed: it
                    # may, when a side would otherwise have no twig at all; the overlap pass separates them in depth)
                    clear = clear and (relax or not _crosses_web(sp, (E.tree,), web, tkd, 1.0))
                    if why == "length" and len(sp) > 4 and clear:
                        nodes.add(p)
                        plans.append((e, x, sp, E.side, E.tree))
                        n_relax[E.side] += 1
                        break
            if relax and n_relax[E.side] >= RELAX_TWIGS:
                break
            x += step


# =================================================================== stage: conjunctival forests
def _make_sheets(P: AnatomyParams, net: _Net, rng) -> list[dict]:
    """Depth sheets of the conjunctival forests (layout 'web': of the twigs),
    ordered shallow to deep: arteriolar sheets usually above venular ones
    (larger vessels deeper)."""
    if P.layout == "web":
        nA, nV, sA, sV = P.twig_sheets_A, P.twig_sheets_V, P.twig_spacing_A, P.twig_spacing_V
    else:
        nA, nV, sA, sV = P.n_art_sheets, P.n_ven_sheets, P.art_spacing, P.ven_spacing
    sh = [dict(side="A", spacing=sA * nA) for _ in range(nA)]
    sh += [dict(side="V", spacing=sV * nV) for _ in range(nV)]
    key = [(0.0 if s["side"] == "A" else 1.0) + (rng.normal(0, 0.6) if rng.random() < 0.3 else 0.0) for s in sh]
    sh = [sh[i] for i in np.argsort(key, kind="stable")]
    for i, s in enumerate(sh):
        s["name"] = f"{s['side']}{i}"
        s["order"] = i
        lay = len(net.layers)
        net.layers[lay] = dict(name=s["name"], kind="sheet", order=i, side=s["side"])
        s["layer"] = lay
    return sh


def _border_roots(net: _Net, P, F, geo, rng, sheet, lsign: int):
    """Roots on the two domain borders the dominant direction points to,
    most of them on the sheet's main end (arterioles: the fornix side,
    away from the limbus; venules: drawn per sheet), so its trees grow
    across the whole frame; each root has a stalk walking into the tissue
    region."""
    th = F.theta0
    if rng.random() < P.sheet_turn_prob:
        th += math.radians(float(rng.uniform(*P.sheet_turn_deg))) * (1 if rng.random() < 0.5 else -1)
    sheet["theta"] = float(th)
    u = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-u[1], u[0]])
    x0, y0, x1, y1 = geo.domain
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    tx0, ty0, tx1, ty1 = geo.tissue
    corners = np.array([[tx0, ty0], [tx1, ty0], [tx0, ty1], [tx1, ty1]]) - ctr
    w = corners @ nrm              # lateral extent of the tissue region across the dominant direction
    wlo, whi = float(w.min()), float(w.max())
    side = sheet["side"]
    spacing = P.root_spacing_A if side == "A" else P.root_spacing_V
    starts = []
    stalk_pts = []
    main = -lsign if side == "A" else int(rng.choice([-1, 1]))
    bias = P.art_root_bias if side == "A" else P.ven_root_bias
    n_bins = max(1, int(round((whi - wlo) / spacing)))
    plan = []
    for end in (-1, 1):             # roots on the border behind -u (walking +u) and behind +u
        keep = 0.9 if end == main else 0.9 * (1 - bias) / max(bias, 1e-6)
        for b in range(n_bins):
            if rng.random() <= keep:
                plan.append((end, wlo + (b + rng.uniform(0.15, 0.85)) * (whi - wlo) / n_bins))
    extra = 0
    while plan or (not starts and extra < 8):
        if plan:
            end, ww = plan.pop(0)
        else:                       # every sheet gets at least one root
            end, ww = main, float(rng.uniform(wlo, whi))
            extra += 1
        c = ctr + ww * nrm
        d = -u * end
        bp = geo.ray_to_border(c, -d)
        if not _Geo.inside(bp, geo.domain, -1e-6)[0]:
            continue
        obst = np.concatenate(stalk_pts) if stalk_pts else None
        ang = math.atan2(d[1], d[0])
        pts, why = _walk(bp, ang, step=2.0, length=4 * geo.pad + 200, persist=40.0,
                         heading_sd=math.radians(10), F=F, geo=geo, rng=rng, obstacles=obst,
                         clear=16.0, stop_in=geo.tissue)
        if why != "in" or len(pts) < 3:
            continue
        pr = float(rng.uniform(*P.p_art_in)) if side == "A" else float(rng.uniform(*P.p_ven_out))
        nb = net.node(pts[0], pressure=pr)
        ns = net.node(pts[-1])
        tid = net.new_tree(kind=side, gamma=None, sheet=sheet["name"])
        net.edge(nb, ns, pts, sheet["layer"], cls="arteriole" if side == "A" else "venule", side=side,
                 role="stalk", tree=tid)
        stalk_pts.append(pts)
        starts.append((ns, _unit(pts[-1] - pts[-2]), tid))
    return starts


def _attractors(P: AnatomyParams, F: _Fields, geo: _Geo, rng) -> np.ndarray:
    """Attraction points over the tissue region: a jittered grid (blue-noise
    like, mean spacing attractor_spacing), its cells longer along the
    dominant direction than across it (attractor_aniso), thinned by the
    density field and towards the edge of the tissue region."""
    x0, y0, x1, y1 = geo.tissue
    a = math.sqrt(max(P.attractor_aniso, 1e-3))
    dmax = F.density_max()
    # cell sides along / across, so the mean point density is dmax / spacing^2 before thinning
    sa, sc = P.attractor_spacing * a / math.sqrt(dmax), P.attractor_spacing / a / math.sqrt(dmax)
    th = F.theta0
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ R          # (along, across) coordinates
    lo, hi = corners.min(0), corners.max(0)
    ia = np.arange(math.floor(lo[0] / sa), math.ceil(hi[0] / sa) + 1)
    ic = np.arange(math.floor(lo[1] / sc), math.ceil(hi[1] / sc) + 1)
    A_, C_ = np.meshgrid(ia, ic, indexing="ij")
    q = np.stack([(A_.ravel() + rng.random(A_.size)) * sa, (C_.ravel() + rng.random(A_.size)) * sc], 1)
    att = q @ R.T
    att = att[_Geo.inside(att, geo.tissue)]
    return att[rng.random(len(att)) < F.density(att) / dmax * geo.margin_taper(att)]


def _colonize(net: _Net, starts, P: AnatomyParams, F: _Fields, geo: _Geo, rng, sheet, obstacles=None):
    """Open space colonization (Runions 2005) of one sheet from its start
    nodes, keeping clear of `obstacles` (riser paths through the sheet).
    A tree whose record has 'max_len' (the twigs of the web layout) stops
    growing that far (path length) from its start.  Adds the tree edges to
    the net; returns the tip nodes."""
    otree = cKDTree(obstacles) if obstacles is not None and len(obstacles) else None
    oclear = P.clearance + 3.0
    clear = P.clearance * max(1.0, P.dilation if sheet["side"] == "V" else P.art_dilation)
    side = sheet["side"]
    d_k = sheet["spacing"] * (P.kill_factor_A if side == "A" else P.kill_factor_V)
    d_i = P.influence * d_k
    step = P.sc_step
    x0, y0, x1, y1 = geo.tissue
    att = _attractors(P, F, geo, rng)
    alive = np.ones(len(att), bool)
    atree = cKDTree(att) if len(att) else None
    cap = 4096
    pos = np.zeros((cap, 2))
    head = np.zeros((cap, 2))
    parent = np.full(cap, -1)
    tree = np.zeros(cap, int)
    since = np.zeros(cap)
    reach = np.zeros(cap)
    kids: list[list[int]] = []
    netid: dict[int, int] = {}
    N = 0
    heading0 = {}
    maxlen = {int(t): float(T["max_len"]) for t, T in net.trees.items() if T.get("max_len") is not None}
    for nid, h, tid in starts:
        pos[N], head[N], tree[N] = net.xy[nid], h, tid
        heading0.setdefault(int(tid), _unit(np.asarray(h, float)))
        kids.append([])
        netid[N] = nid
        N += 1
    if atree is not None and N:
        for hits in atree.query_ball_point(pos[:N], d_k):
            alive[hits] = False
    min_ang = math.radians(P.min_branch_angle_deg)
    exempt_r = 2 * clear
    max_iter = int(3 * math.hypot(x1 - x0, y1 - y0) / step)
    for _ in range(max_iter):
        A = np.nonzero(alive)[0]
        if len(A) == 0 or N == 0:
            break
        ntree = cKDTree(pos[:N])
        d, nn = ntree.query(att[A], distance_upper_bound=d_i)
        ok = np.isfinite(d) & (d > 1e-9)
        if not ok.any():
            break
        idx = nn[ok]
        vec = (att[A[ok]] - pos[idx]) / d[ok][:, None]
        acc = np.zeros((N, 2))
        np.add.at(acc, idx, vec)
        cnt = np.bincount(idx, minlength=N)
        first = {}
        for a_i, n_i in zip(A[ok], idx):
            first.setdefault(int(n_i), int(a_i))
        new_pos, new_par, new_dir = [], [], []
        for nidx in np.nonzero(cnt)[0]:
            nc = len(kids[nidx])
            if nc >= 2:
                continue
            if maxlen and reach[nidx] >= maxlen.get(int(tree[nidx]), np.inf):
                continue                      # a twig reaches no further from its long vessel
            v = acc[nidx]
            if np.hypot(*v) < 0.2 * cnt[nidx]:
                v = att[first[nidx]] - pos[nidx]
            v = _unit(v)
            if nc == 0:
                h = head[nidx]
                o = F.odir(pos[nidx][None], h)[0]
                dv = _unit(v + P.persistence_weight * h + P.orient_weight * o)
            else:
                hc = head[kids[nidx][0]]
                if since[nidx] < P.min_branch_spacing:
                    continue
                if math.acos(float(np.clip(v @ hc, -1, 1))) < min_ang:
                    continue
                # no fork within min_branch_spacing downstream on the existing chain
                c, dist, near_fork = kids[nidx][0], step, False
                while dist < P.min_branch_spacing:
                    if len(kids[c]) != 1:
                        near_fork = len(kids[c]) >= 2
                        break
                    c = kids[c][0]
                    dist += step
                if near_fork:
                    continue
                o = F.odir(pos[nidx][None], v)[0]
                dv = _unit(v + 0.3 * P.orient_weight * o)
                if math.acos(float(np.clip(dv @ hc, -1, 1))) < min_ang:
                    sg = 1.0 if hc[0] * dv[1] - hc[1] * dv[0] > 0 else -1.0
                    dv = _rot(hc, sg * min_ang)
            h0 = heading0.get(int(tree[nidx]))
            if h0 is not None and float(dv @ h0) < P.back_cos:
                continue                      # no growing back towards the root
            q = pos[nidx] + step * dv
            if not _Geo.inside(q, geo.domain, 1.0)[0]:
                continue
            blocked = False
            for j in ntree.query_ball_point(q, clear):
                if j == nidx:
                    continue
                if tree[j] == tree[nidx] and np.hypot(*(pos[j] - pos[nidx])) < exempt_r:
                    continue
                blocked = True
                break
            if not blocked:
                for qq, pp in zip(new_pos, new_par):
                    if np.hypot(*(qq - q)) < clear and not (
                            tree[pp] == tree[nidx] and np.hypot(*(pos[pp] - pos[nidx])) < exempt_r):
                        blocked = True
                        break
            if not blocked and otree is not None and otree.query_ball_point(q, oclear):
                blocked = True
            if blocked:
                continue
            new_pos.append(q)
            new_par.append(nidx)
            new_dir.append(dv)
        if not new_pos:
            break
        for q, par, dv in zip(new_pos, new_par, new_dir):
            if N >= cap:
                cap *= 2
                pos = np.resize(pos, (cap, 2))
                head = np.resize(head, (cap, 2))
                parent = np.resize(parent, cap)
                tree = np.resize(tree, cap)
                since = np.resize(since, cap)
                reach = np.resize(reach, cap)
            pos[N], head[N], parent[N], tree[N] = q, dv, par, tree[par]
            reach[N] = reach[par] + step
            if len(kids[par]) == 0:
                since[N] = since[par] + step
            else:
                since[N] = step
                # par becomes a fork: recount the existing chain below it
                c, dd = kids[par][0], step
                while True:
                    since[c] = dd
                    if len(kids[c]) != 1:
                        break
                    c = kids[c][0]
                    dd += step
            kids.append([])
            kids[par].append(N)
            N += 1
        for hits in atree.query_ball_point(np.array(new_pos), d_k):
            alive[hits] = False
    # chains between real nodes -> net edges
    cls = "arteriole" if side == "A" else "venule"
    real = set(netid) | {i for i in range(N) if len(kids[i]) != 1}
    tips = []
    for r in sorted(real):
        for c in kids[r]:
            chain = [r, c]
            x = c
            while x not in real:
                x = kids[x][0]
                chain.append(x)
            if x not in netid:
                netid[x] = net.node(pos[x])
                if len(kids[x]) == 0:
                    tips.append(netid[x])
            net.edge(netid[r], netid[x], pos[chain], sheet["layer"], cls=cls, side=side, role="tree",
                     tree=int(tree[r]))
    for s_i, (nid, _, _) in enumerate(starts):
        if len(kids[s_i]) == 0:
            tips.append(nid)
    return tips


# =================================================================== stage: capillary mesh
def _grow_mesh(net: _Net, P: AnatomyParams, F: _Fields, geo: _Geo, rng):
    """A Poisson-Voronoi capillary mesh over the tissue region, cells
    stretched along the dominant direction, some edges dropped."""
    x0, y0, x1, y1 = geo.tissue
    cell = 1.0 / math.sqrt(P.mesh_density)
    pad = 2.5 * cell
    X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    dmax = F.density_max()
    n = rng.poisson(P.mesh_density * (X1 - X0) * (Y1 - Y0) * dmax)
    pts = np.stack([rng.uniform(X0, X1, n), rng.uniform(Y0, Y1, n)], 1)
    pts = pts[rng.random(n) < F.density(pts) / dmax]
    a = float(rng.uniform(*P.mesh_aspect))
    th = F.theta0
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c = pts @ R                       # coordinates along / across the dominant direction
    c[:, 0] /= a
    vor = Voronoi(c)
    V = vor.vertices.copy()
    V[:, 0] *= a
    V = V @ R.T
    inside = _Geo.inside(V, geo.tissue)
    edges = [(i, j) for i, j in vor.ridge_vertices if i >= 0 and j >= 0 and inside[i] and inside[j]]
    edges = [e for e in edges if rng.random() >= P.mesh_drop]
    # collapse short edges (union-find)
    par = {}

    def find(i):
        while par.get(i, i) != i:
            i = par[i]
        return i
    for i, j in edges:
        if np.hypot(*(V[i] - V[j])) < P.mesh_min_edge:
            ri, rj = find(i), find(j)
            if ri != rj:
                par[max(ri, rj)] = min(ri, rj)
    groups = defaultdict(list)
    for i in {x for e in edges for x in e}:
        groups[find(i)].append(i)
    cen = {g: V[m].mean(0) for g, m in groups.items()}
    nid = {}
    seen = set()
    lay0 = 0
    sl = math.sqrt(math.log(1 + P.cap_cv ** 2))
    for i, j in edges:
        gi, gj = find(i), find(j)
        if gi == gj or (min(gi, gj), max(gi, gj)) in seen:
            continue
        seen.add((min(gi, gj), max(gi, gj)))
        for g in (gi, gj):
            if g not in nid:
                nid[g] = net.node(cen[g])
        p0, p1 = cen[gi], cen[gj]
        L = float(np.hypot(*(p1 - p0)))
        m = max(2, int(math.ceil(L)) + 1)
        D = float(np.clip(P.cap_diam * math.exp(sl * rng.standard_normal() - 0.5 * sl ** 2), *P.cap_diam_range))
        t = np.linspace(0.0, 1.0, m)[:, None]
        nrm_ = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(L, 1e-9)
        bow = P.mesh_bow * L * rng.standard_normal() * nrm_
        # a gentle arc, not a straight Voronoi edge; v2: and an S-bend (an even mesh of straight edges meeting at
        # 120 deg read as a honeycomb where it was visible: random preset at small k, review v1 T11)
        sb = P.mesh_sbend * L * rng.standard_normal() * nrm_
        # (the S-bend vanishes towards the ends with 4t(1-t): the capillary leaves its node along its chord and does
        # not swing across a sibling there)
        path = p0 + t * (p1 - p0) + 4 * t * (1 - t) * (bow + np.sin(2 * math.pi * t) * sb)
        net.edge(nid[gi], nid[gj], path, lay0, cls="capillary", side="C", role="mesh",
                 r=0.5 * D * P.cap_dilation)
    return a


def _channels(net: _Net, P: AnatomyParams, rng) -> int:
    """Preferential (thoroughfare) channels of the capillary bed: from a share
    (channel_frac) of the mesh nodes where an arteriole tip feeds the mesh or
    a venule tip drains it, the shortest route through the mesh (Dijkstra on
    capillary length, at most channel_max) to a node of the other side (a
    venule's, an arteriole's); its capillaries widen to D
    channel_diam (Zweifach's metarterioles and thoroughfare channels, the
    high-flow routes structural adaptation widens: Pries 1998).  They are the
    visible fine network between the twigs: the true capillaries (D 1 d_c)
    stay faint, so in v1 the twigs ended in nothing visible and the fine
    vessels closed few loops (faint_loop_density -2.5 SD, junction density
    -3.3 SD against the dense site; realism review v1 T3).  Returns the number
    of capillaries widened."""
    import heapq
    if P.channel_frac <= 0:
        return 0
    adj = defaultdict(list)
    elen = {}
    for e, E in net.E.items():
        if E.role == "mesh":
            L = float(_arclen(E.pts)[-1])
            elen[e] = L
            adj[E.u].append((E.v, e, L))
            adj[E.v].append((E.u, e, L))
    budget = P.channel_share * sum(elen.values())     # at most this much of the mesh becomes channels
    side = {}
    for n in adj:
        s = {net.E[x].side for x in net.adj[n] if net.E[x].role != "mesh"} & {"A", "V"}
        if len(s) == 1:
            side[n] = s.pop()
    starts = sorted(side)
    if len({side[n] for n in starts}) < 2:
        return 0
    widened = set()
    for a in [starts[i] for i in rng.permutation(len(starts))]:
        if rng.random() >= P.channel_frac:
            continue
        ends = {n for n, s in side.items() if s != side[a]}     # from a feeder to a drainer or the other way
        dist, prev, pq, found = {a: 0.0}, {}, [(0.0, a)], None
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf):
                continue
            if n in ends:
                found = n
                break
            if d > P.channel_max:
                break
            for m, e, L in adj[n]:
                nd = d + L
                if nd < dist.get(m, np.inf):
                    dist[m] = nd
                    prev[m] = (n, e)
                    heapq.heappush(pq, (nd, m))
        if found is None:
            continue
        D = float(rng.uniform(*P.channel_diam)) * P.cap_dilation
        n = found
        while n != a:
            n, e = prev[n]
            E = net.E[e]
            E.r = max(E.r, 0.5 * D)
            E.flags["channel"] = True
            widened.add(e)
        if sum(elen[e] for e in widened) >= budget:
            break
    return len(widened)


# =================================================================== stage: attachments to the mesh
def _tree_kids(net: _Net):
    """eid -> member children (same tree, u == parent's v)."""
    kids = defaultdict(list)
    for e, E in net.E.items():
        if E.tree is None:
            continue
        p = net.member_parent(E.tree, E.u)
        if p is not None and p != e:
            kids[p].append(e)
    return kids


def _postorder(kids: dict, starts):
    """Edges reachable from `starts` through `kids`, children before
    parents; an edge met again (a loop) is not descended into twice."""
    out, seen = [], set()
    for s0 in starts:
        if s0 in seen:
            continue
        seen.add(s0)
        stack = [(s0, iter(kids.get(s0, ())))]
        while stack:
            x, it = stack[-1]
            nxt = next((c for c in it if c not in seen), None)
            if nxt is None:
                out.append(x)
                stack.pop()
            else:
                seen.add(nxt)
                stack.append((nxt, iter(kids.get(nxt, ()))))
    return out


def _estimate_tree_r(net: _Net, P, gammas, rmean):
    """Rough bottom-up radii of the conjunctival trees (every tip at a mean
    tip radius) - used to choose where side capillaries leave."""
    kids = _tree_kids(net)
    est = {}
    todo = [e for e, E in net.E.items() if E.tree is not None and net.trees[E.tree]["kind"] in ("A", "V")]
    for x in _postorder(kids, todo):
        g = gammas[net.trees[net.E[x].tree]["kind"]]
        ks = [c for c in kids[x] if c in est]
        est[x] = (sum(est[c] ** g for c in ks)) ** (1 / g) if ks else rmean[net.trees[net.E[x].tree]["kind"]]
    return est


class _Approach:
    """Where a vessel from a tree sheet may join the capillary mesh.  It
    joins at the nearest point of a mesh edge (arriving square to it), or
    at a mesh node only if it arrives at a wide angle to the node's edges;
    and its last stretch - where it climbs to the mesh - must stay clear of
    other mesh edges and of the vessels of the sheets it climbs through, so
    it neither runs alongside a capillary nor crosses a vessel at its own
    depth on the way up."""

    def __init__(self, net: _Net, P: AnatomyParams, midx: _PtIndex):
        self.net, self.P, self.midx = net, P, midx
        self.tan = math.tan(math.radians(P.ramp_angle_deg))
        self.order = {l: L["order"] for l, L in net.layers.items() if L["kind"] == "sheet"}
        self._upper = {}

    def upper(self, lay):
        k = self.order.get(lay, 0)
        if k not in self._upper:
            lays = {l for l, o in self.order.items() if o < k}
            es = [e for e, E in self.net.E.items() if E.tree is not None and int(E.lay[0]) in lays
                  and E.role in ("tree", "stalk", "companion")]
            self._upper[k] = _PtIndex(self.net, es, 1.0) if es else None
        return self._upper[k], k

    def target(self, xy, me, sm):
        """(s, point, node or None) on mesh edge me near arclength sm, or None."""
        E = self.net.E[me]
        s_arr = _arclen(E.pts)
        Lm = float(s_arr[-1])
        endn = None
        if sm < 2.0 or Lm - sm < 2.0:
            endn = E.u if sm < 2.0 else E.v
            vec = _unit(np.asarray(xy) - self.net.xy[endn])
            for f in self.net.adj[endn]:
                F_ = self.net.E[f]
                if F_.role != "mesh":
                    continue
                d = _unit(F_.pts[1] - F_.pts[0]) if F_.u == endn else _unit(F_.pts[-2] - F_.pts[-1])
                if float(vec @ d) > math.cos(math.radians(50)):
                    return None
            sm = 0.0 if endn == E.u else Lm
        elif Lm >= 8.0:
            sm = float(np.clip(sm, 4.0, Lm - 4.0))
        tgt = np.array([np.interp(sm, s_arr, E.pts[:, 0]), np.interp(sm, s_arr, E.pts[:, 1])])
        return sm, tgt, endn

    def ok(self, xy, tgt, lay, me, endn) -> bool:
        lx = float(np.hypot(*(np.asarray(tgt) - xy)))
        ex = {me}
        if endn is not None:
            ex |= {f for f in self.net.adj[endn] if self.net.E[f].role == "mesh"}
        if self.midx.seg_hits(xy, tgt, 2.5, exclude=ex, skip_a=max(0.0, lx - 12.0)):
            return False
        idx, k = self.upper(lay)
        if idx is not None:
            lr = 1.5 * (4.0 + 6.5 * k) / self.tan + 3.0
            clear = 3.5 * max(1.0, self.P.dilation)          # dilated venules above are wider
            if idx.seg_hits(xy, tgt, clear, skip_a=max(0.0, lx - lr)):
                return False
        return True


def _attach(net: _Net, P, geo, rng, gammas, tips):
    """Tree tips join the capillary mesh (arteriole tips at forks, venule
    tips at confluences), and side capillaries leave the small tree vessels
    (plan G4).  Each takes the nearest admissible mesh point (_Approach):
    reached without crossing a vessel of its own sheet, without a U-turn
    (tips) or a near-tangent start (side capillaries), not next to a
    connection of the other side (no arteriole straight into a venule), and
    never two vessels from one fork to one node.  Tips that find no mesh
    are left to be pruned."""
    mesh = [e for e, E in net.E.items() if E.role == "mesh"]
    midx = _PtIndex(net, mesh, 1.0)
    sheet_idx = {}
    for lay, Ly in net.layers.items():
        if Ly["kind"] == "sheet":
            sheet_idx[lay] = _PtIndex(net, [e for e, E in net.E.items() if E.tree is not None and E.lay[0] == lay
                                            and E.role in ("tree", "stalk")], 1.0)
    requests = []
    for t in tips:
        es = [e for e in net.adj[t]]
        if len(es) != 1:
            continue
        e = es[0]
        E = net.E[e]
        head = _unit(E.pts[-1] - E.pts[max(0, len(E.pts) - 3)])
        requests.append(dict(kind="tip", node=t, eid=e, xy=net.xy[t], side=E.side, lay=int(E.lay[-1]),
                             anchor={("n", E.u), ("e", e)}, head=head))
    web = P.layout == "web"
    tip_A, tip_V = (P.twig_tip_r_A, P.twig_tip_r_V) if web else (P.tip_r_A, P.tip_r_V)
    if P.side_spacing > 0:
        rmean = dict(A=float(np.mean(tip_A)), V=float(np.mean(tip_V)))
        est = _estimate_tree_r(net, P, gammas, rmean)
        for e in sorted(est):
            E = net.E[e]
            if est[e] > P.side_rmax or E.role != "tree":
                continue
            s = _arclen(E.pts)
            L = float(s[-1])
            x = float(rng.uniform(0.3, 1.0)) * P.side_spacing
            while x < L - 8:
                if x > 8 and _Geo.inside(np.array([np.interp(x, s, E.pts[:, 0]), np.interp(x, s, E.pts[:, 1])]),
                                         geo.tissue)[0]:
                    p = np.array([np.interp(x, s, E.pts[:, 0]), np.interp(x, s, E.pts[:, 1])])
                    q = np.array([np.interp(x + 2, s, E.pts[:, 0]), np.interp(x + 2, s, E.pts[:, 1])])
                    requests.append(dict(kind="side", eid=e, s=x, xy=p, side=E.side, lay=int(E.lay[0]),
                                         anchor={("e", e)}, tan=_unit(q - p)))
                x += P.side_spacing * float(rng.uniform(0.7, 1.3))
    order = rng.permutation(len(requests))
    claims = defaultdict(list)            # mesh eid -> [s, side, key, end node]
    anchors = defaultdict(set)            # key -> the tree nodes that reach it
    done = []
    key_n = 0
    appr = _Approach(net, P, midx)
    for oi in order:
        rq = requests[oi]
        for me, sm, xym, dist in midx.nearest_edges(rq["xy"], k=64, rmax=P.attach_max, n_edges=10):
            got = appr.target(rq["xy"], me, sm)
            if got is None:
                continue
            sm, tgt, endn = got
            Lm = float(_arclen(net.E[me].pts)[-1])
            dvec = tgt - rq["xy"]
            if np.hypot(*dvec) > 3.0:
                dvec = _unit(dvec)
                if rq["kind"] == "tip" and float(dvec @ rq["head"]) < -0.3:
                    continue              # no U-turn back to a capillary behind the tip
                if rq["kind"] == "side" and abs(float(dvec @ rq["tan"])) > 0.8:
                    continue              # a side capillary leaves its vessel at a real angle
            if any(abs(c[0] - sm) < 10 and c[1] != rq["side"] for c in claims[me]):
                continue
            if endn is not None and any(c[3] == endn and c[1] != rq["side"] for cl in claims.values() for c in cl):
                continue
            idx = sheet_idx.get(rq["lay"])
            if idx is not None and idx.seg_hits(rq["xy"], tgt, 2.5, exclude=(rq["eid"],), skip_a=4.0):
                continue
            if not appr.ok(rq["xy"], tgt, rq["lay"], me, endn):
                continue
            lx = float(np.hypot(*(tgt - rq["xy"])))
            same = [c for c in claims[me] if abs(c[0] - sm) < min(4.0, 0.3 * lx) and c[1] == rq["side"]]
            # the two ends of a mesh edge are nodes too: another claim may sit on them
            if sm <= 0 or sm >= Lm:
                same += [c for x, cl in claims.items() if x != me for c in cl
                         if c[1] == rq["side"] and c[3] == (net.E[me].u if sm <= 0 else net.E[me].v)]
            if same and rq["anchor"] & anchors[same[0][2]]:
                continue                  # two vessels from one fork / edge to one node would run side by side
            if same:
                sm, key = same[0][0], same[0][2]
            else:
                key = key_n
                key_n += 1
                end_node = net.E[me].u if sm <= 0 else net.E[me].v if sm >= Lm else None
                claims[me].append([sm, rq["side"], key, end_node])
            anchors[key] |= rq["anchor"]
            done.append((rq, me, sm, key))
            break
    # split the mesh edges
    target = {}
    for me, cl in claims.items():
        E = net.E[me]
        Lm = float(_arclen(E.pts)[-1])
        inner = sorted({c[0] for c in cl if 0 < c[0] < Lm})
        u0, v0 = E.u, E.v
        nodes = dict(zip(inner, net.split(me, inner)))
        for c in cl:
            s_, key = c[0], c[2]
            target[key] = u0 if s_ <= 0 else v0 if s_ >= Lm else nodes[s_]
    # split the tree edges that carry side capillaries (per edge, all positions at once)
    side_nodes = {}
    by_edge = defaultdict(list)
    for rq, me, sm, key in done:
        if rq["kind"] == "side":
            by_edge[rq["eid"]].append(rq["s"])
    for e, lst in by_edge.items():
        lst = sorted(set(lst))
        for s_, nid in zip(lst, net.split(e, lst)):
            side_nodes[(e, s_)] = nid
    rA, rV = tip_A, tip_V
    for rq, me, sm, key in done:
        nt = target[key]
        leaf = float(rng.uniform(*(rA if rq["side"] == "A" else rV))) * (P.art_dilation if rq["side"] == "A"
                                                                          else P.dilation)
        if rq["kind"] == "tip":
            t = rq["node"]
            es = list(net.adj[t])
            if len(es) != 1:
                continue
            e = es[0]
            E = net.E[e]
            seg = np.linspace(E.pts[-1], net.xy[nt], max(2, int(np.hypot(*(net.xy[nt] - E.pts[-1]))) + 2))[1:]
            net.adj[t].discard(e)
            net.remove_node_if_free(t)
            E.pts = np.concatenate([E.pts, seg])
            E.lay = np.concatenate([E.lay, np.full(len(seg), E.lay[-1])])
            E.v = nt
            E.pts[-1] = net.xy[nt]
            E.leaf_r = leaf
            net.adj[nt].add(e)
        else:
            nu = side_nodes.get((rq["eid"], rq["s"]))
            if nu is None:
                continue
            E0 = net.E[rq["eid"]]
            m = max(2, int(np.hypot(*(net.xy[nt] - net.xy[nu]))) + 2)
            net.edge(nu, nt, np.linspace(net.xy[nu], net.xy[nt], m), rq["lay"], cls=E0.cls, side=E0.side,
                     role="side", tree=E0.tree, leaf_r=leaf)


# =================================================================== stage: pruning
def _prune(net: _Net) -> dict:
    """Remove every edge on no path between two border nodes (dead ends,
    dangling loops, islands), and components whose border pressures are
    all equal (no flow)."""
    import networkx as nx
    removed = 0
    while True:
        changed = False
        stack = [n for n in list(net.xy) if len(net.adj[n]) == 1 and not net.is_border(n)]
        while stack:
            n = stack.pop()
            if n not in net.xy or len(net.adj[n]) != 1 or net.is_border(n):
                continue
            e = next(iter(net.adj[n]))
            E = net.E[e]
            other = E.v if E.u == n else E.u
            net.remove_edge(e)
            removed += 1
            changed = True
            net.remove_node_if_free(n)
            if other in net.xy and len(net.adj[other]) == 1 and not net.is_border(other):
                stack.append(other)
        G = nx.Graph()
        S = ("S",)
        for e, E in net.E.items():
            x = ("e", e)
            G.add_edge(E.u, x)
            G.add_edge(x, E.v)
        for n in net.xy:
            if net.is_border(n) and net.adj[n]:
                G.add_edge(S, n)
        alive = set()
        if S in G:
            for comp in nx.biconnected_component_edges(G):
                nodes = {a for ed in comp for a in ed}
                if S in nodes:
                    alive |= {a[1] for a in nodes if isinstance(a, tuple) and len(a) == 2 and a[0] == "e"}
        for e in [e for e in net.E if e not in alive]:
            E = net.E[e]
            net.remove_edge(e)
            removed += 1
            changed = True
            net.remove_node_if_free(E.u)
            net.remove_node_if_free(E.v)
        # components without a pressure difference carry no flow
        H = nx.Graph()
        for e, E in net.E.items():
            H.add_edge(E.u, E.v)
        for comp in list(nx.connected_components(H)):
            ps = [net.nd[n]["pressure"] for n in comp if net.is_border(n)]
            if len(ps) < 2 or max(ps) - min(ps) < 0.05:
                for e in [e for n in comp for e in list(net.adj[n])]:
                    if e in net.E:
                        net.remove_edge(e)
                        removed += 1
                        changed = True
                for n in comp:
                    net.remove_node_if_free(n)
        for n in [n for n in net.xy if not net.adj[n]]:
            net.remove_node_if_free(n)
        if not changed:
            break
    return dict(removed=removed)


# =================================================================== stage: calibre
def _assign_radii(net: _Net, P: AnatomyParams, gammas) -> dict:
    """Murray-type calibre r0^g = sum ri^g: bottom-up in the conjunctival
    trees from their tip radii, clipped to the class caps; top-down in the
    deep trees from the trunk calibre, where risers and anastomoses take
    their fixed share.  Returns the clipped edges."""
    kids = _tree_kids(net)
    foreign = defaultdict(list)
    for e, E in net.E.items():
        for T, n in E.attach.items():
            p = net.member_parent(T, n)
            if p is not None:
                foreign[p].append(e)
    clipped = set()
    rmax = dict(A=P.art_diam_max / 2, V=P.ven_diam_max / 2)     # class caps (the pathologic preset raises them)
    rdef = dict(A=float(np.mean(P.tip_r_A)), V=float(np.mean(P.tip_r_V)))
    done = {}
    conj = [e for e, E in net.E.items() if E.tree is not None and net.trees[E.tree]["kind"] in ("A", "V")]
    # first the trees that end in the mesh (bottom-up from their tips), then the long vessels of the web, which
    # also take in what joins them from other trees (twigs, by their stalk; not arcades: anastomoses)
    for web in (False, True):
        todo = [e for e in conj if bool(net.trees[net.E[e].tree].get("web")) == web]
        for x in _postorder(kids, todo):
            X = net.E[x]
            T = net.trees[X.tree]
            kind = T["kind"]
            g = gammas[kind]
            ks = [c for c in kids[x] if c in done]
            add = [done[c] for c in ks]
            if web:
                for f in foreign[x]:
                    F_ = net.E[f]
                    if F_.role == "arcade":
                        continue          # an anastomosis: flow runs either way; it does not set the calibre
                    val = F_.fixed_r if F_.fixed_r is not None else done.get(f)
                    if val is not None:
                        add.append(val)
            r = (sum(v ** g for v in add)) ** (1 / g) if add else (X.leaf_r or rdef[kind])
            cap = T.get("rmax", rmax[kind])
            if r > cap:
                r = cap
                clipped.add(x)
            done[x] = r
            X.r = r
    floored = set()
    for t, T in net.trees.items():
        if T["kind"] not in ("DA", "DV"):
            continue
        g = T["gamma"]
        roots = [e for e, E in net.E.items() if E.tree == t and net.member_parent(t, E.u) is None]
        stack = [(e, T["root_r"]) for e in roots]
        seen = set()
        while stack:
            e, r = stack.pop()
            if e in seen:
                continue
            seen.add(e)
            net.E[e].r = r
            ks = kids[e]
            rem = r ** g
            for f in foreign[e]:
                F_ = net.E[f]
                rem -= (F_.fixed_r if F_.fixed_r is not None else F_.r) ** g
            for c in ks:
                if net.E[c].role == "branch":
                    rb = r * net.E[c].q ** (1 / g)
                    rem -= rb ** g
                    stack.append((c, rb))
            conts = [c for c in ks if net.E[c].role != "branch"]
            share = rem / max(len(conts), 1)      # a trunk that splits into two continuations shares what is left
            for c in conts:                       # (each took all of it: Murray broken, residual up to 1)
                if share < (0.5 * r) ** g:
                    floored.add(e)
                stack.append((c, max(share, (0.5 * r) ** g) ** (1 / g)))
    for e, E in net.E.items():
        if E.fixed_r is not None:
            E.r = E.fixed_r
    return dict(clipped=clipped, floored=floored)


def _contract_short(net: _Net, factor: float = 1.5) -> int:
    """Merge the two nodes of every edge shorter than factor * (R_a + R_b) + 1
    (R: the widest vessel at each node): forks that close are one junction
    (plan invariant 8), otherwise the lumens of the vessels at the two forks
    would overlap.  The merged node stays where the wider vessels meet; the
    other node's vessels are bent over to it."""
    done = 0
    while True:
        rmax = {n: max(net.E[e].r for e in net.adj[n]) for n in net.xy if net.adj[n]}
        cand = []
        for e, E in net.E.items():
            if E.u == E.v or net.is_border(E.u) or net.is_border(E.v):
                continue
            L = float(_arclen(E.pts)[-1])
            if L < factor * (rmax[E.u] + rmax[E.v]) + 1.0:
                cand.append((L, e))
        cand.sort()
        used, n0 = set(), done
        for L, e in cand:
            if e not in net.E:
                continue
            E = net.E[e]
            a, b = E.u, E.v
            if a in used or b in used:
                continue
            sides = {net.E[f].side for f in net.adj[a] | net.adj[b]} & {"A", "V"}
            if sides == {"A", "V"}:
                continue                      # would join an arteriole straight to a venule
            ta = {net.E[f].tree for f in net.adj[a] if f != e} | {T for f in net.adj[a] for T in net.E[f].attach}
            tb = {net.E[f].tree for f in net.adj[b] if f != e} | {T for f in net.adj[b] for T in net.E[f].attach}
            if (ta & tb) - {None, E.tree}:
                continue                      # would close a loop inside a tree
            keep, drop = (a, b) if rmax[a] >= rmax[b] else (b, a)
            net.remove_edge(e)
            shift = net.xy[keep] - net.xy[drop]
            for f in list(net.adj[drop]):
                F_ = net.E[f]
                s = _arclen(F_.pts)
                Lf = float(s[-1])
                wl = max(min(0.5 * Lf, 2.0 * float(np.hypot(*shift)) + 2.0), 1e-6)
                w = np.zeros(len(s))
                if F_.u == drop:
                    w += 1 - _smoothstep(s / wl)
                    F_.u = keep
                if F_.v == drop:
                    w += 1 - _smoothstep((Lf - s) / wl)
                    F_.v = keep
                F_.pts = F_.pts + np.clip(w, 0, 1)[:, None] * shift
                F_.pts[0], F_.pts[-1] = net.xy[F_.u], net.xy[F_.v]
                F_.attach = {T: (keep if n == drop else n) for T, n in F_.attach.items()}
                net.adj[keep].add(f)
            net.adj[drop].clear()
            net.remove_node_if_free(drop)
            for f in [f for f in net.adj[keep] if net.E[f].u == net.E[f].v]:
                net.remove_edge(f)
            used |= {a, b}
            done += 1
        if done == n0:
            break
    return done


def _strahler(net: _Net) -> dict:
    kids = _tree_kids(net)
    out = {}
    for x in _postorder(kids, [e for e, E in net.E.items() if E.tree is not None]):
        ks = [out[c] for c in kids[x] if c in out]
        if not ks:
            out[x] = 1
        else:
            m = max(ks)
            out[x] = m + 1 if ks.count(m) >= 2 else m
    return out


# =================================================================== stage: arteriole-venule pairs
def _add_pairs(net: _Net, P: AnatomyParams, geo: _Geo, rng, host: str = "A") -> int:
    """Companion venules beside arterioles (host 'A'), or companion
    arterioles beside venules (host 'V'; p_pair_V): the doubled lines.  An offset
    copy of a stretch of the arteriole's course (followed downstream through
    its forks along the wider child, up to pair_length) at a wall gap of
    pair_gap * (r_a + r_v), stopped before a side branch on its side or a
    bend too tight for the offset, joined at one end to the nearest venule
    (a new tributary) and at the other to the capillary mesh.  The companion
    lies in the arteriole's sheet, so it never crosses it.  (With host 'V'
    read venule for arteriole and the other way round: the companion
    arteriole branches off the nearest arteriole and feeds the mesh.)"""
    oside = "V" if host == "A" else "A"
    p_pair = (P.p_pair if host == "A" else P.p_pair_V) * (P.twig_pair_frac if P.layout == "web" else 1.0)
    arts = [e for e, E in net.E.items() if E.side == host and E.role in ("tree", "stalk")
            and net.layers.get(int(E.lay[0]), {}).get("kind") == "sheet"]
    made = 0
    idx = {}

    def indices(lay_a):
        if "v" not in idx:
            ven = [x for x, X in net.E.items() if X.side == oside and X.role in ("tree", "stalk")]
            idx["v"] = _PtIndex(net, ven, 1.0)
            idx["m"] = _PtIndex(net, [x for x, X in net.E.items() if X.role == "mesh"], 1.0)
        if ("s", lay_a) not in idx:
            same = [x for x, X in net.E.items() if int(X.lay[len(X.lay) // 2]) == lay_a or
                    (X.role == "companion" and X.flags.get("sheet") == lay_a)]
            idx[("s", lay_a)] = _PtIndex(net, same, 1.0, with_r=True)
        return idx[("s", lay_a)], idx["v"], idx["m"]

    paired = set()

    def run_path(e, want):
        """The arteriole's course from edge e downstream through its tree's
        forks, along the wider child at each (the companion follows the
        main vessel, not a side branch), up to `want` d_c: (edge ids, points)."""
        path, pts = [e], [net.E[e].pts]
        L = float(_arclen(net.E[e].pts)[-1])
        cur = net.E[e]
        while L < want:
            kids = [x for x in net.adj[cur.v] if x not in path and net.E[x].u == cur.v and net.E[x].tree == cur.tree
                    and net.E[x].side == host and net.E[x].role in ("tree", "stalk")]
            if not kids:
                break
            x = max(kids, key=lambda k_: (net.E[k_].r, -k_))
            path.append(x)
            pts.append(net.E[x].pts[1:])
            L += float(_arclen(net.E[x].pts)[-1])
            cur = net.E[x]
        return path, np.concatenate(pts)

    for e in sorted(arts):
        if rng.random() >= p_pair:
            continue
        if e not in net.E or e in paired:
            continue
        E = net.E[e]
        ra = E.r
        rv = ra * float(rng.uniform(*P.pair_ratio)) ** (1 if host == "A" else -1)    # venules the wider
        off = (float(rng.uniform(*P.pair_gap)) + 1.0) * (ra + rv)
        margin = max(10.0, 3 * (ra + rv) + 6)
        want = float(rng.uniform(*P.pair_length))
        path, raw = run_path(e, want + 2 * margin)
        pts = _resample(raw, 1.0)
        L = float(_arclen(pts)[-1])
        Lf = min(want, L - 2 * margin)
        if Lf < P.pair_length[0]:
            continue
        s0 = float(rng.uniform(margin, max(margin, L - margin - Lf)))
        i0, i1 = int(s0), int(s0 + Lf)
        seg = _laplace(pts, 4)
        tan = _tangents(seg, 3)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        kap = _curv(_laplace(pts, 8))
        bend = np.nonzero(kap[i0:i1 + 1] * off > 0.5)[0]
        if len(bend):                           # stop before the first bend too tight for the offset
            i1 = i0 + int(bend[0]) - 1
            if i1 - i0 < P.pair_length[0]:
                continue
        lay_a = int(E.lay[0])
        sidx, vidx, midx = indices(lay_a)
        win = None
        own = set(path)
        for sg in ((1, -1) if rng.random() < 0.5 else (-1, 1)):
            w = pts[i0:i1 + 1] + sg * off * nrm[i0:i1 + 1]
            first_bad = None
            for n_, hits in enumerate(sidx.tree.query_ball_point(w, off - 0.5)):
                if any(int(sidx.eid[h]) not in own for h in hits):
                    first_bad = n_
                    break
            if first_bad is not None:             # a side branch on this side: keep the run before it
                w = w[:max(first_bad - int(off) - 3, 0)]
            if len(w) >= P.pair_length[0] and (win is None or len(w) > len(win)):
                win = w
            if first_bad is None and win is not None:
                break
        if win is None:
            continue
        best = None
        for end, other in ((win[0], win[-1]), (win[-1], win[0])):
            c = vidx.nearest_edges(end, k=64, rmax=80.0, n_edges=4)
            for ve, vs, vxy, vd in c:
                Lv = float(_arclen(net.E[ve].pts)[-1])
                if vs < 6 or Lv - vs < 6:
                    continue
                if sidx.seg_hits(end, vxy, rv + 1.5, exclude=(), skip_a=0.0):
                    continue
                if best is None or vd < best[4]:
                    best = (end, other, ve, vs, vd, vxy)
                break
        if best is None:
            continue
        vend, mend, ve, vs, vd, vxy = best
        mc = None
        appr = _Approach(net, P, midx)
        for me, ms, mxy, md in midx.nearest_edges(mend, k=64, rmax=P.attach_max, n_edges=8):
            got = appr.target(mend, me, ms)
            if got is None:
                continue
            ms, mxy, endn = got
            if sidx.seg_hits(mend, mxy, rv + 1.0, exclude=(), skip_a=0.0) or \
                    not appr.ok(mend, mxy, lay_a, me, endn):
                continue
            mc = (me, ms, mxy, endn)
            break
        if mc is None:
            continue
        me, ms, mxy, endn = mc
        nm = endn if endn is not None else net.split(me, [ms])[0]
        nv = net.split(ve, [vs])[0]
        if np.hypot(*(vend - win[0])) < 1e-9:
            body = win
        else:
            body = win[::-1]
        l1 = np.linspace(net.xy[nv], vend, max(2, int(np.hypot(*(vend - net.xy[nv]))) + 2))
        l2 = np.linspace(mend, net.xy[nm], max(2, int(np.hypot(*(net.xy[nm] - mend))) + 2))
        cpath = np.concatenate([l1[:-1], body, l2[1:]])
        lay_v = int(net.E[ve].lay[0])
        lay = np.concatenate([np.full(len(l1) - 1, lay_v), np.full(len(body) + len(l2) - 1, lay_a)])
        ce = net.edge(nv, nm, cpath, lay, cls="venule" if oside == "V" else "arteriole", side=oside,
                      role="companion", tree=net.E[ve].tree,
                      leaf_r=rv, r=rv, flags=dict(partner=e, partner_path=list(path), sheet=lay_a, lowtort=True))
        for x in path:
            if x in net.E:
                net.E[x].flags.setdefault("partners", []).append(ce)
                net.E[x].flags["lowtort"] = True
                paired.add(x)
        made += 1
        idx.clear()               # edges were split: rebuild the indices
    return made


# =================================================================== final vessels (chains)
@dataclass
class _Chain:
    edges: list                 # [(eid, forward)]
    u: int
    v: int
    pts: np.ndarray = None
    lay: np.ndarray = None
    rn: np.ndarray = None       # nominal radius per point
    ci: np.ndarray = None       # constituent index per point
    cls: str = "capillary"
    side: str = "C"
    info: dict = field(default_factory=dict)


def _chains(net: _Net) -> tuple[list[_Chain], int]:
    """Merge the edges through nodes with exactly two edges into vessels:
    one vessel per edge of the final graph."""
    real = {n for n in net.xy if len(net.adj[n]) != 2 or net.is_border(n)}
    seen = set()
    out = []
    for n in sorted(real):
        for e in sorted(net.adj[n]):
            if e in seen:
                continue
            seq, cur, x = [], n, e
            while True:
                seen.add(x)
                X = net.E[x]
                fwd = X.u == cur
                seq.append((x, fwd))
                cur = X.v if fwd else X.u
                if cur in real:
                    break
                nxt = [y for y in net.adj[cur] if y != x]
                if not nxt:
                    break
                x = nxt[0]
            if cur == n:
                continue            # a loop through one node carries no flow
            if sum(1 for _, f in seq if not f) > len(seq) / 2:
                seq = [(x, not f) for x, f in reversed(seq)]
                out.append(_Chain(seq, cur, n))
            else:
                out.append(_Chain(seq, n, cur))
    lost = len([e for e in net.E if e not in seen])
    return out, lost


def _chain_arrays(net: _Net, ch: _Chain):
    P_, L_, R_, C_ = [], [], [], []
    for i, (e, f) in enumerate(ch.edges):
        E = net.E[e]
        p, lay = (E.pts, E.lay) if f else (E.pts[::-1], E.lay[::-1])
        if i:
            p, lay = p[1:], lay[1:]
        P_.append(p)
        L_.append(lay)
        R_.append(np.full(len(p), E.r))
        C_.append(np.full(len(p), i))
    ch.pts = np.concatenate(P_)
    ch.lay = np.concatenate(L_)
    ch.rn = np.concatenate(R_)
    ch.ci = np.concatenate(C_)
    ch.pts[0], ch.pts[-1] = net.xy[ch.u], net.xy[ch.v]


def _chain_class(net: _Net, ch: _Chain, order: dict):
    Es = [net.E[e] for e, _ in ch.edges]
    lens = [float(_arclen(E.pts)[-1]) for E in Es]
    roles = {E.role for E in Es}
    if "riser" in roles or any(E.flags.get("perforator") for E in Es):
        cls = "perforator"
    elif roles & {"ava", "vv"}:
        cls = "anastomosis"
    else:
        cls = Es[int(np.argmax(lens))].cls
    ch.cls = cls
    ch.side = dict(capillary="C", arteriole="A", venule="V").get(cls, "D")
    main = max(range(len(Es)), key=lambda i: (Es[i].tree is not None, Es[i].r, lens[i]))
    E0 = Es[main]
    ch.info = dict(tree=E0.tree, roles=sorted(roles), edges=[e for e, _ in ch.edges])
    if E0.tree is not None:
        kind = net.trees[E0.tree]["kind"]
        o = order.get(ch.edges[main][0], 0)
        ch.info["order"] = int(-o if kind in ("A", "DA") else o)
    else:
        ch.info["order"] = 0
    lays = [int(x) for x in ch.lay if x != RAMP]
    if lays:
        ch.info["layer"] = net.layers[max(set(lays), key=lays.count)]["name"]
    else:
        ch.info["layer"] = "ramp"
    if any(E.flags.get("lowtort") for E in Es):
        ch.info["lowtort"] = True
    if any(E.flags.get("channel") for E in Es):
        ch.info["channel"] = True
    if any(E.flags.get("pairtort") for E in Es):
        ch.info["pairtort"] = True
    if E0.role == "web":                  # a long vessel of the conjunctival web (layout 'web')
        ch.info["web"] = str(net.trees[E0.tree].get("role", "long"))
    elif "arcade" in roles:
        ch.info["web"] = "arcade"
    elif E0.tree is not None and net.trees[E0.tree].get("twig"):
        ch.info["twig"] = True
    if E0.tree is not None and net.trees[E0.tree].get("plexus"):
        ch.info["plexus"] = True


# =================================================================== stage: geometry
def _node_ends(chains):
    ends = defaultdict(list)            # node -> [(chain index, at_start)]
    for i, ch in enumerate(chains):
        ends[ch.u].append((i, True))
        ends[ch.v].append((i, False))
    return ends


def _fork_angles(net: _Net, chains, P, rng, gammas):
    """At every fork of a tree (and mirrored at every confluence), turn the
    daughters' first stretch so they leave the parent's direction at the
    angles of their flow split (Murray 1926; Zamir 1978), plus noise; the
    larger daughter stays closer to the parent's course."""
    ends = _node_ends(chains)
    n_done = 0
    for nd, lst in ends.items():
        if len(lst) < 3 or net.is_border(nd):
            continue
        if any(net.E[(chains[ci].edges[0] if s else chains[ci].edges[-1])[0]].role == "web" for ci, s in lst):
            continue                  # the web's junctions were laid out at their Murray angles as they grew
        par, dau = None, []
        for ci, at_start in lst:
            ch = chains[ci]
            e, f = ch.edges[0] if at_start else ch.edges[-1]
            E = net.E[e]
            if E.tree is None or E.role in ("mesh", "riser", "ava", "vv"):
                continue
            # the tree-edge end at this node: v (leaf side) -> parent; u -> daughter
            if E.v == nd:
                par = (ci, at_start)
            elif E.u == nd:
                dau.append((ci, at_start))
        if par is None or len(dau) != 2:
            continue
        pc = chains[par[0]]
        pp = pc.pts if not par[1] else pc.pts[::-1]
        rp = pc.rn[-1] if not par[1] else pc.rn[0]
        back = pp[max(0, len(pp) - 1 - max(3, int(4 * rp + 2)))]
        tp = _unit(net.xy[nd] - back)
        ds = []
        for ci, at_start in dau:
            ch = chains[ci]
            p = ch.pts if at_start else ch.pts[::-1]
            r = ch.rn[0] if at_start else ch.rn[-1]
            ahead = p[min(len(p) - 1, max(3, int(4 * r + 2)))]
            ds.append((ci, at_start, r, _unit(ahead - net.xy[nd]), float(_arclen(p)[-1])))
        ds.sort(key=lambda x: -x[2])
        g = gammas.get(net.trees[net.E[chains[par[0]].edges[-1 if not par[1] else 0][0]].tree]["kind"], 2.5)
        (c1, s1, r1, t1, L1), (c2, s2, r2, t2, L2) = ds
        q = r2 ** g / (r1 ** g + r2 ** g)
        a1, a2 = _murray_angles(q, rng, P.angle_noise_deg)
        cr = lambda a, b: a[0] * b[1] - a[1] * b[0]   # noqa: E731
        sg1, sg2 = np.sign(cr(tp, t1)) or 1.0, np.sign(cr(tp, t2)) or 1.0
        if sg1 == sg2:
            # both on one side: keep the order, set only the outer daughter's angle
            cur1 = math.acos(float(np.clip(tp @ t1, -1, 1)))
            cur2 = math.acos(float(np.clip(tp @ t2, -1, 1)))
            targets = [(c1, s1, r1, t1, L1, sg1 * max(cur1, 0.0)),
                       (c2, s2, r2, t2, L2, sg2 * max(cur2, cur1 + math.radians(25)))]
        else:
            targets = [(c1, s1, r1, t1, L1, sg1 * a1), (c2, s2, r2, t2, L2, sg2 * a2)]
        for ci, at_start, r, t, L, ang in targets:
            want = _rot(tp, ang)
            delta = math.atan2(cr(t, want), float(t @ want))
            delta = float(np.clip(delta, -math.radians(40), math.radians(40)))
            if abs(delta) < 1e-3:
                continue
            ch = chains[ci]
            p = ch.pts if at_start else ch.pts[::-1].copy()
            s = _arclen(p)
            lb = min(0.5 * s[-1], 6 * r + 8)
            w = 1 - _smoothstep((s - 0.3 * lb) / max(0.7 * lb, 1e-6))
            w[0] = 1.0
            rel = p - net.xy[nd]
            ca, sa = np.cos(delta * w), np.sin(delta * w)
            q_ = net.xy[nd] + np.stack([ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]], 1)
            q_[0] = net.xy[nd]
            q_[-1] = p[-1]
            ch.pts = q_ if at_start else q_[::-1].copy()
            n_done += 1
    return n_done


def _layer_groups(net: _Net, chains):
    """KD-trees of chain points per layer (for clearance)."""
    by = defaultdict(lambda: ([], [], []))
    for i, ch in enumerate(chains):
        for lay in np.unique(ch.lay):
            if lay == RAMP:
                continue
            m = ch.lay == lay
            by[int(lay)][0].append(ch.pts[m])
            by[int(lay)][1].append(np.full(m.sum(), i))
            by[int(lay)][2].append(ch.rn[m])
    out = {}
    for lay, (xs, ids, rs) in by.items():
        xy = np.concatenate(xs)
        out[lay] = (cKDTree(xy), np.concatenate(ids), np.concatenate(rs), xy)
    return out


def _tortuosity(net: _Net, chains, P: AnatomyParams, rng):
    """Lateral displacement of every vessel by a smooth Gaussian process:
    curvature * D has SD tort_amp * class gain at correlation length
    tort_corr * D (plan G9), per-vessel gain log-normal, a few vessels
    strongly tortuous; zero near the ends; capped at 0.4 of the free gap to
    the nearest vessel of the same layer, so layers stay planar."""
    groups = _layer_groups(net, chains)
    new = []
    for i, ch in enumerate(chains):
        p = ch.pts
        s = _arclen(p)
        L = float(s[-1])
        if L < 4 or len(p) < 5:
            new.append(np.zeros_like(p))
            continue
        r = float(np.median(ch.rn))
        D = 2 * r
        gain = P.tort_gain.get(ch.cls, 0.5) * P.tortuosity_gain * math.exp(P.tort_vessel_sd * rng.standard_normal())
        if r < 2 and rng.random() < P.p_tortuous:
            gain *= P.tortuous_gain
            ch.info["tortuous"] = True
        if ch.info.get("lowtort"):
            gain *= 0.15
        elif ch.info.get("web") and ch.info["web"] != "arcade":
            gain *= P.web_tort_gain * (0.6 if ch.info.get("pairtort") else 1.0)
        ell = min(P.tort_corr * max(D, 1.0), L / 2.5)
        if ell < 1.5:
            new.append(np.zeros_like(p))
            continue
        ksd = P.tort_amp * gain / max(D, 1.0)
        sd = ksd * ell ** 2 / math.sqrt(3.0)
        if ch.info.get("web") and ch.info["web"] != "arcade":
            sd = min(sd, 0.5 * ell)       # a long vessel meanders, it does not fold back on itself
        h = L / (len(p) - 1)
        d = sd * _bridge(_smooth_noise(len(p), ell / h, rng))
        f_end = [2 * ch.rn[0] + 2, 2 * ch.rn[-1] + 2]
        lt = max(ell, 3.0)
        taper = _smoothstep((s - f_end[0]) / lt) * _smoothstep((L - s - f_end[1]) / lt)
        d *= taper
        # clearance cap against other vessels of the same layer
        free = np.full(len(p), np.inf)
        for lay in np.unique(ch.lay):
            if lay == RAMP or int(lay) not in groups:
                continue
            tr, ids, rs, xy = groups[int(lay)]
            m = np.nonzero(ch.lay == lay)[0]
            kk = min(24, len(ids))
            dd, jj = tr.query(p[m], k=kk)
            dd, jj = np.atleast_2d(dd), np.atleast_2d(jj)
            other = ids[jj] != i
            big = np.where(other, dd - rs[jj], np.inf).min(1)
            nothing = ~other.any(1)
            big[nothing] = dd[nothing, -1]
            free[m] = big - ch.rn[m] - 1.0
        cap = 0.4 * np.maximum(free, 0.0)
        d = np.clip(d, -cap, cap)
        t = _tangents(_laplace(p, 3), 2)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        # no meander may fold the vessel back onto itself (a loop at one depth: a pigtail ring, or a self-overlap
        # no depth bump can clear, e.g. a collecting venule of r 3-4.6 turning 534 deg in random seed 1;
        # correctness review v1, defect 4): halve the displacement until it does not
        for _ in range(6 if ch.cls != "capillary" else 0):
            if float(np.max(np.abs(d))) < 1.0 or not _folds(p + d[:, None] * nrm, ch.rn, s):
                break
            d = 0.5 * d
            ch.info["unfolded"] = ch.info.get("unfolded", 0) + 1
        new.append(d[:, None] * nrm)
    pre, skip = _course_meander(net, chains, new, P, rng)
    moved = _shared_meander(net, chains, new, pre=pre, skip=skip)
    for ch, dv in zip(chains, new):
        ch.pts = ch.pts + dv
    for n, v in moved.items():
        net.xy[n] = net.xy[n] + v
    return len(moved)


def _folds(q, rn, s, margin: float = 1.0) -> bool:
    """Whether a centreline q (radii rn, arclength s along the undisplaced
    course) comes within r_i + r_j + margin of itself at two points more
    than pi (r_i + r_j) apart along it (graph.check's self rule)."""
    step = 2 if len(q) > 400 else 1
    qq, rr, ss = q[::step], rn[::step], s[::step]
    pairs = cKDTree(qq).query_pairs(2.0 * float(rr.max()) + margin, output_type="ndarray")
    if not len(pairs):
        return False
    i, j = pairs[:, 0], pairs[:, 1]
    rs = rr[i] + rr[j]
    return bool(np.any((np.abs(ss[i] - ss[j]) > math.pi * rs) & (np.hypot(*(qq[i] - qq[j]).T) < rs + margin)))


def _course_meander(net: _Net, chains, disp, P: AnatomyParams, rng):
    """One meander along the whole course of a long vessel of the web (a
    tree's planned course, root to leaf through its forks), instead of one per
    vessel pinned at every junction on its way: the per-vessel displacement
    vanished towards each node (the taper), so a long vessel bent back onto
    every node; on a wide collecting vein a tributary or twig every 40-300 d_c
    made an elbow at each (v2, realism review v1 T4: 62 % of the kinked wide
    path lay on collecting veins, 87 % within 2 widths of a node).  The course
    takes one Gaussian-process displacement (the vessel's tortuosity, as in
    _tortuosity, from its median radius), bridged to zero only at the
    course's two ends; it is then limited in curvature as a whole (a wide
    vessel no tighter than wide_bend_D diameters), and every vessel lying on
    the course takes it, so the nodes on the way move with it (the other
    vessels there - branches, twigs, arcades - take the move with a fade,
    _shared_meander).  Followers (companions, bundle members) are left to
    _shared_meander, which gives them their leader's displacement.  disp is
    modified in place; returns ({node: [displacement]}, {(chain, node)} whose
    displacement already holds the node's move)."""
    WEB = ("long", "collecting", "companion", "bundle")
    by_tree = defaultdict(list)
    for i, ch in enumerate(chains):
        t = ch.info.get("tree")
        if t is not None and ch.info.get("web") in WEB:
            by_tree[int(t)].append(i)
    moved = defaultdict(list)
    skip = set()
    for t in sorted(by_tree):
        T = net.trees.get(t, {})
        course = T.get("course")
        if course is None or len(course) < 5:
            continue
        C = _resample(np.asarray(course, float), 0.5)
        sC = _arclen(C)
        L = float(sC[-1])
        kd = cKDTree(C)
        on = []
        for i in by_tree[t]:
            d, j = kd.query(chains[i].pts)
            tol = 1.5 + float(np.max(chains[i].rn))
            if float(np.median(d)) < tol and float(np.mean(d < tol)) > 0.9:
                on.append((i, j))
                chains[i].cs, chains[i].ctree = sC[j], t      # its arclength along the course (_undulations)
        if not on or T.get("lead") is not None:
            continue
        cls = chains[on[0][0]].cls
        # the meander's scale from the wider part of the course (a long vessel widens downstream by Murray; scaled
        # by its median, its wide stretch bent like a thin vessel: S-bends of 3-4 D on 5-6 d_c wide venules)
        rr_ = np.concatenate([chains[i].rn for i, _ in on])
        r = float(np.quantile(rr_, 0.9))
        D = 2 * r
        gain = P.tort_gain.get(cls, 0.5) * P.tortuosity_gain * math.exp(P.tort_vessel_sd * rng.standard_normal())
        if r < 2 and rng.random() < P.p_tortuous:
            gain *= P.tortuous_gain
        gain *= P.web_tort_gain * (0.6 if any(chains[i].info.get("pairtort") for i, _ in on) else 1.0)
        ell = min(P.tort_corr * max(D, 1.0), L / 2.5)
        dC = np.zeros(len(C))
        if ell >= 1.5:
            ksd = P.tort_amp * gain / max(D, 1.0)
            sd = min(ksd * ell ** 2 / math.sqrt(3.0), 0.5 * ell)
            dC = sd * _bridge(_smooth_noise(len(C), ell / 0.5, rng))
            f_end = 2 * r + 2
            lt = max(ell, 3.0)
            dC *= _smoothstep((sC - f_end) / lt) * _smoothstep((L - sC - f_end) / lt)
        tC = _tangents(_laplace(C, 3), 2)
        nC = np.stack([-tC[:, 1], tC[:, 0]], 1)
        for _ in range(6):                    # it meanders, it does not fold back on itself
            if float(np.max(np.abs(dC))) < 1.0 or not _folds(C + dC[:, None] * nC, np.full(len(C), r), sC):
                break
            dC = 0.5 * dC
        Q = C + dC[:, None] * nC
        rho = P.min_bend_radius
        if r >= P.wide_bend_r:
            rho = max(rho, P.wide_bend_D * D)
        # the curvature limit on a copy spaced in proportion to the bend radius (local Laplacian smoothing at 0.5 d_c
        # would need thousands of passes to round a bend of radius 40 d_c), carried back by relative arclength
        sQ = _arclen(Q)
        hc = float(np.clip(rho / 8.0, 0.5, 4.0))
        nq = max(5, int(math.ceil(sQ[-1] / hc)) + 1)
        tc = np.linspace(0.0, 1.0, nq)
        Qc = np.stack([np.interp(tc * sQ[-1], sQ, Q[:, 0]), np.interp(tc * sQ[-1], sQ, Q[:, 1])], 1)
        tmp = _Chain([], 0, 0, pts=Qc, rn=np.full(nq, r))
        _limit_curvature(tmp, P.kappa_r_max, max_iter=200, rho_min=rho)
        tq = sQ / max(sQ[-1], 1e-9)
        Qs = np.stack([np.interp(tq, tc, tmp.pts[:, 0]), np.interp(tq, tc, tmp.pts[:, 1])], 1)
        dv = Qs - C
        for i, j in on:
            ch = chains[i]
            disp[i] = dv[j].copy()
            for k, nd in ((0, ch.u), (-1, ch.v)):
                if net.is_border(nd):
                    disp[i][k] = 0.0          # a border node stays where it is
                else:
                    moved[nd].append(disp[i][k].copy())
                    skip.add((i, nd))
            ch.info["course_meander"] = True
    return moved, skip


def _shared_meander(net: _Net, chains, disp, pre=None, skip=()) -> dict:
    """A companion or bundle member meanders WITH its partner: every web tree
    with a leader (companions: their host; bundles: the middle member) takes
    the leader's displacement (disp: per chain, per point, 2-D; modified in
    place) along the stretch where it runs beside it, blending into its own
    displacement where it diverges (by distance to the leader, beyond 1.25 x
    the pair's offset, and by angle).  Each tree drew its own before, so
    partners were pushed onto each other (projected lumens overlapping over
    14-18 % of the paired length; correctness review v1, defect 1).  The
    follower's own nodes on the shared stretch (its twigs, arcades) move with
    it: returns {node: displacement}; the other vessels at such a node take
    the move with a fade over a few diameters from it.  pre: nodes already
    moved (_course_meander: {node: [displacement]}); skip: (chain, node)
    pairs whose displacement already holds the node's move."""
    by_tree = defaultdict(list)
    for i, ch in enumerate(chains):
        t = ch.info.get("tree")
        if t is not None and ch.info.get("web") not in (None, "arcade"):
            by_tree[int(t)].append(i)
    follower = set()
    moved = defaultdict(list)
    for n_, vs in (pre or {}).items():
        moved[n_].extend(vs)
    for t, cis in by_tree.items():
        T = net.trees.get(t, {})
        lead = T.get("lead")
        if lead is None or not by_tree.get(int(lead)):
            continue
        off = float(T.get("lead_off", T.get("pair_off", 4.0)))
        lc = by_tree[int(lead)]
        lp = np.concatenate([chains[j].pts for j in lc])
        lv = np.concatenate([disp[j] for j in lc])
        lt = np.concatenate([_tangents(chains[j].pts, 2) for j in lc])
        kd = cKDTree(lp)
        d1, d2 = 1.25 * off + 2.0, 1.8 * off + 8.0
        # (v2) along the follower's whole course where its pieces lie on it (_course_meander set their course arclength
        # cs): one weight, smoothed along the course, and no own meander (its planned course already wanders), so its
        # pieces join smoothly at their nodes; the per-piece blend made V-kinks at the nodes on its way (to 126 deg)
        on = [i for i in cis if getattr(chains[i], "cs", None) is not None]
        if on:
            cs_all = np.concatenate([chains[i].cs for i in on])
            pts_all = np.concatenate([chains[i].pts for i in on])
            tan_all = np.concatenate([_tangents(chains[i].pts, 2) for i in on])
            dist, j = kd.query(pts_all)
            cosang = np.abs(np.sum(tan_all * lt[j], 1))
            w = _smoothstep((d2 - dist) / (d2 - d1)) * _smoothstep((cosang - math.cos(math.radians(35)))
                                                                   / (math.cos(math.radians(20)) -
                                                                      math.cos(math.radians(35))))
            o = np.argsort(cs_all, kind="stable")
            grid = np.arange(0.0, float(cs_all.max()) + 1.0, 1.0)
            wg = np.interp(grid, cs_all[o], w[o])
            wg = ndi.gaussian_filter1d(wg, 6.0, mode="nearest")
            k0 = 0
            for i in on:
                n_ = len(chains[i].pts)
                wi = np.interp(chains[i].cs, grid, wg)
                disp[i] = wi[:, None] * lv[j[k0:k0 + n_]]
                k0 += n_
        for i in cis:
            ch = chains[i]
            follower.add(i)
            if i not in on:
                dist, j = kd.query(ch.pts)
                cosang = np.abs(np.sum(_tangents(ch.pts, 2) * lt[j], 1))
                w = _smoothstep((d2 - dist) / (d2 - d1)) * _smoothstep((cosang - math.cos(math.radians(35)))
                                                                       / (math.cos(math.radians(20)) -
                                                                          math.cos(math.radians(35))))
                w = ndi.gaussian_filter1d(w, 4.0, mode="nearest")
                disp[i] = w[:, None] * lv[j] + (1.0 - w[:, None]) * disp[i]
            s = _arclen(ch.pts)
            for k, nd, dist in ((0, ch.u, s), (-1, ch.v, float(s[-1]) - s)):
                if net.is_border(nd):         # a border node stays where it is
                    disp[i] *= _smoothstep(dist / max(min(0.5 * float(s[-1]), 20.0), 1e-6))[:, None]
                else:
                    moved[nd].append(disp[i][k].copy())
    moved = {n: np.mean(v, 0) for n, v in moved.items() if float(np.hypot(*np.mean(v, 0))) > 1e-6}
    # (v2) a follower's pieces meeting at a node took different displacements there (one beside its leader, the other
    # diverging from it), and the node moved by their mean: with the leader's meander along its whole course (no
    # longer pinned at its junctions) the pieces met in a V (126 deg in a pathologic frame).  Each piece now takes the
    # node's move at its end, faded in over a few diameters.
    for i in sorted(follower):
        ch = chains[i]
        s = _arclen(ch.pts)
        L = float(s[-1])
        r = float(np.median(ch.rn))
        for k, nd, dist in ((0, ch.u, s), (-1, ch.v, L - s)):
            if nd not in moved or net.is_border(nd) or (nd == ch.v and ch.u == ch.v):
                continue
            dv = moved[nd] - disp[i][k]
            lb = min(0.5 * L, max(3.0 * float(np.hypot(*dv)) + 6.0, 4.0 * r + 6.0))
            disp[i] = disp[i] + (1.0 - _smoothstep(dist / max(lb, 1e-6)))[:, None] * dv
    for i, ch in enumerate(chains):
        if i in follower:
            continue
        p = ch.pts
        s = _arclen(p)
        L = float(s[-1])
        r = float(np.median(ch.rn))
        for nd, dist in ((ch.u, s), (ch.v, L - s)):
            if nd not in moved or (nd == ch.v and ch.u == ch.v) or (i, nd) in skip:
                continue
            v = moved[nd]
            lb = min(0.5 * L, max(3.0 * float(np.hypot(*v)) + 6.0, 4.0 * r + 6.0))
            fade = 1.0 - _smoothstep(dist / max(lb, 1e-6))
            disp[i] = disp[i] + fade[:, None] * v
    return moved


def _limit_curvature(ch: _Chain, kmax: float, max_iter: int = 60, rho_min: float = 0.0):
    """Smooth locally until |kappa| r <= kmax along the vessel and no bend is
    tighter than rho_min (ends fixed)."""
    p = ch.pts
    reff = np.maximum(ch.rn, kmax * rho_min)
    for _ in range(max_iter):
        k = _curv(p) * reff
        bad = k > kmax
        if not bad.any():
            break
        m = ndi.binary_dilation(bad, iterations=3)
        m[0] = m[-1] = False
        p = _laplace(p, 2, 0.5, m)
    ch.pts = p


def _rotate_end(ch: _Chain, nd_xy, at_start: bool, delta: float, r: float):
    """Turn a vessel's first stretch at a node by delta (rad) about the
    node, fading to no turn over a few diameters."""
    p = ch.pts if at_start else ch.pts[::-1].copy()
    s = _arclen(p)
    lb = min(0.5 * s[-1], 6 * r + 8)
    w = 1 - _smoothstep((s - 0.3 * lb) / max(0.7 * lb, 1e-6))
    rel = p - nd_xy
    ca, sa = np.cos(delta * w), np.sin(delta * w)
    q = nd_xy + np.stack([ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]], 1)
    q[0], q[-1] = nd_xy, p[-1]
    ch.pts = q if at_start else q[::-1].copy()


def _spread(net: _Net, chains, iters: int = 3) -> int:
    """At every node, turn apart any two vessels that leave it at too small
    an angle to clear each other's lumen beyond the node's region (the
    4 max(r) + 2 of graph.check): the thinner turns away."""
    ends = _node_ends(chains)
    n = 0
    for _ in range(iters):
        changed = False
        for nd, lst in ends.items():
            if len(lst) < 2:
                continue
            c0 = net.xy[nd]
            arms = []
            for ci, at_start in lst:
                ch = chains[ci]
                if ch.u == ch.v:
                    continue
                p = ch.pts if at_start else ch.pts[::-1]
                r = float(ch.rn[0] if at_start else ch.rn[-1])
                s = _arclen(p)
                d = min(0.5 * s[-1], 4 * r + 4)
                q = np.array([np.interp(d, s, p[:, 0]), np.interp(d, s, p[:, 1])])
                arms.append((math.atan2(*(q - c0)[::-1]), r, ci, at_start))
            arms.sort()
            m = len(arms)
            for k in range(m):
                a1, r1, c1, s1 = arms[k]
                a2, r2, c2, s2 = arms[(k + 1) % m]
                gap = (a2 - a1) % (2 * math.pi)
                dex = 4 * max(r1, r2) + 4
                need = 2 * math.asin(min(0.95, (r1 + r2 + 1.5) / (2 * dex)))
                if gap >= need:
                    continue
                turn = min(need - gap + math.radians(3), math.radians(45))
                # the thinner arm turns away from the other (arm 1 clockwise, arm 2 anticlockwise)
                if r1 <= r2:
                    _rotate_end(chains[c1], c0, s1, -turn, r1)
                else:
                    _rotate_end(chains[c2], c0, s2, turn, r2)
                n += 1
                changed = True
        if not changed:
            break
    return n


def _web_through(net: _Net, chains, max_deg: float = 150.0) -> int:
    """Where a long vessel of the web passes a junction on its way (a twig
    leaves, an arcade joins), its two pieces are separate vessels of the
    graph; turn their ends about the node (each by half the kink) so the
    long vessel runs smoothly through it (tangent-continuous)."""
    ends = _node_ends(chains)
    n = 0
    for nd, lst in ends.items():
        if len(lst) < 3 or net.is_border(nd):
            continue
        by = defaultdict(list)
        for ci, s in lst:
            E = net.E[(chains[ci].edges[0] if s else chains[ci].edges[-1])[0]]
            if E.role == "web":
                by[E.tree].append((ci, s))
        c0 = net.xy[nd]
        for arms in by.values():
            if len(arms) != 2:
                continue                  # a Y-fork: its own angles
            dirs = []
            for ci, s in arms:
                ch = chains[ci]
                p = ch.pts if s else ch.pts[::-1]
                r = float(ch.rn[0] if s else ch.rn[-1])
                ss = _arclen(p)
                d = min(0.3 * ss[-1], r + 1.5)     # the local tangent at the node (not the vessel's bend beyond)
                q = np.array([np.interp(d, ss, p[:, 0]), np.interp(d, ss, p[:, 1])])
                dirs.append((_unit(q - c0), r, ci, s))
            (a1, r1, c1, s1), (a2, r2, c2, s2) = dirs
            b = -a2
            eps = math.atan2(a1[0] * b[1] - a1[1] * b[0], float(a1 @ b))
            # every kink up to max_deg (was 45: a long vessel's pieces at a node contracted with a companion's or a
            # twig's kept kinks of 46-108 deg; correctness review v1, defect 6)
            if abs(eps) < 1e-3 or abs(eps) > math.radians(max_deg):
                continue
            _rotate_end(chains[c1], c0, s1, 0.5 * eps, r1)
            _rotate_end(chains[c2], c0, s2, -0.5 * eps, r2)
            n += 1
    return n


def _geometry(net: _Net, chains, P, rng, gammas) -> dict:
    """Resample, smooth the growth kinks, set branch angles, add
    tortuosity, open up junctions that are too narrow, run the long vessels
    smoothly through their junctions, and limit curvature."""
    for ch in chains:
        h = 0.5 if np.median(ch.rn) < 1.0 else 0.75
        ch.pts, ch.lay, ch.rn, ch.ci = _resample(ch.pts, h, ch.lay, ch.rn, ch.ci)
        n_it = 2 + int(min(8, 2 * np.max(ch.rn)))
        if ch.cls == "capillary":
            n_it = 16                 # round the corners where mesh edges were merged (~2 d_c)
        ch.pts = _laplace(ch.pts, n_it, 0.5)
    n_ang = _fork_angles(net, chains, P, rng, gammas)
    n_shared = _tortuosity(net, chains, P, rng)
    n_spread = _spread(net, chains)
    n_through = _web_through(net, chains) if P.layout == "web" else 0
    for ch in chains:
        rho = P.min_bend_radius
        rm = float(np.median(ch.rn))
        if rm >= P.wide_bend_r and ch.cls != "capillary":       # wide vessels bend gently (v2)
            rho = max(rho, P.wide_bend_D * 2 * rm)
        _limit_curvature(ch, P.kappa_r_max, rho_min=rho, max_iter=60 if rho <= 4 * P.min_bend_radius else 150)
    return dict(fork_angles=n_ang, spread=n_spread, through=n_through, shared_meander_nodes=n_shared)


# =================================================================== stage: depth
def _layer_bases(net: _Net, chains, P: AnatomyParams, rng, F: _Fields | None = None):
    """Depth of every layer: the mesh under the epithelium; the sheets
    stacked below it, each as thick as its wide lumens (a high quantile of
    the radius along its vessels, sheet_quantile: the very widest reach into
    the next sheet and are separated where they cross it by the overlap
    pass); the deep systems below the conjunctiva, stacked by their widest
    lumen (larger vessels deeper)."""
    rs = defaultdict(list)
    for ch in chains:
        for lay in np.unique(ch.lay):
            if lay != RAMP:
                rs[int(lay)].append(ch.rn[ch.lay == lay])
    rmax = {}
    for lay, v in rs.items():
        v = np.concatenate(v)
        kind = net.layers[lay]["kind"]
        rmax[lay] = float(np.quantile(v, P.sheet_quantile)) if kind == "sheet" else float(v.max())
    base = {0: P.mesh_depth}
    sheets = sorted([l for l, L in net.layers.items() if L["kind"] == "sheet"], key=lambda l: net.layers[l]["order"])
    top = max(P.mesh_depth + rmax.get(0, 0.75), P.epithelium) + 0.5 + P.sheet_gap
    for l in sheets:
        base[l] = top
        top += 2 * rmax.get(l, 1.0) + 0.5 + P.sheet_gap
    conj_bottom = top
    if P.layout == "web":
        # the long vessels from just under the capillary mesh (clear of the twigs in their sheets there), the
        # episcleral systems below them, placed together: each stretch so that the vessels it crosses lie at
        # other depths there (larger vessels deeper)
        top0 = max(P.mesh_depth + rmax.get(0, 0.75), P.epithelium) + 0.5 + P.sheet_gap
        web = [l for l, L in net.layers.items() if L["kind"] == "web" and l in rs]
        deep = [l for l, L in net.layers.items() if L["kind"] == "deep" and l in rs]
        # dilated (congested) vessels need more depth to cross: the span grows with the dilation (P)
        span = P.web_depth_span * max(P.dilation, 1.0)
        spec = {l: dict(lo=top0, hi=top0 + span, overflow=6.0, wobble=P.z_wobble, max_step=8.0,
                        deep=False) for l in web}
        fixed = {l: base[l] for l in sheets if l in rs}
        spec.update({l: dict(lo=0.0, hi=0.0, overflow=0.0, wobble=P.z_wobble, max_step=0.0, deep=False)
                     for l in fixed})
        lo = max(P.deep_top, top0 + span)
        # so do the dilated and enlarged episcleral vessels (v1 fix: the pathologic preset's deep trunks and plexus,
        # r up to 12, had no room to cross, and the overlap pass removed 10-20 of them per frame) (P)
        dspan = P.deep_depth_span * max(P.dilation * P.size_gain, 1.0)
        spec.update({l: dict(lo=lo, hi=lo + dspan, overflow=20.0, wobble=P.deep_z_wobble, max_step=12.0,
                             deep=True,
                             vein=bool(net.layers[l].get("vein"))) for l in deep})
        s1, p1, a1 = _segment_layers(net, chains, web, rng)
        s2, p2, a2 = _segment_layers(net, chains, deep, rng, seg=(200.0, 350.0))
        b, bottom = _allocate(net, chains, s1 + s2, P, F, rng, spec, parent={**p1, **p2}, adj={**a1, **a2},
                              fixed=fixed)
        base.update(b)
        wb = [bottom[l] for l in s1 if l in bottom]
        conj_bottom = max([top] + wb) + P.sheet_gap
        return base, conj_bottom
    # episcleral systems stacked in a random order (veins more often deeper), each as thick as its widest lumen
    deep = [l for l, L in net.layers.items() if L["kind"] == "deep"]
    key = [rng.uniform(0, 1) + (0.5 if net.layers[l]["vein"] else 0.0) for l in deep]
    top = max(P.deep_top, conj_bottom + 3 * P.deep_depth_sd)
    for i in np.argsort(key, kind="stable"):
        l = deep[i]
        top += float(rng.uniform(0, P.deep_spread / max(len(deep), 1)))
        base[l] = top
        top += 2 * rmax.get(l, 4.0) + 0.5 + P.sheet_gap + 2 * P.deep_depth_sd
    return base, conj_bottom


def _segment_layers(net: _Net, chains, lays, rng, seg=(100.0, 180.0)):
    """Cut the layers `lays` (each a long vessel of the web or an episcleral
    system) into stretches about seg long, each a layer of its own (same
    name): a long vessel's depth then follows the vessels it crosses along
    its way (ramps between stretches; depth is a profile along a vessel), and
    long vessels that all cross each other need not be stacked.  Returns
    (the stretch layers, {stretch: its parent layer}, {stretch: the
    stretches it continues into})."""
    lays = set(int(l) for l in lays)
    parent, adj = {}, defaultdict(set)
    end_lab = {}
    out = set()
    used = set()
    for ci, ch in enumerate(chains):
        lab = ch.lay.copy()
        s = _arclen(ch.pts)
        for L in np.unique(ch.lay):
            L = int(L)
            if L not in lays:
                continue
            idx = np.nonzero(ch.lay == L)[0]
            s0, s1 = float(s[idx[0]]), float(s[idx[-1]])
            cuts = []
            x = s0 + float(rng.uniform(*seg))
            while x < s1 - 0.4 * seg[0]:
                cuts.append(x)
                x += float(rng.uniform(*seg))
            ids = [] if L in used else [L]         # every vessel (chain) of the long vessel a stretch of its own
            used.add(L)
            while len(ids) < len(cuts) + 1:
                nid = max(net.layers) + 1
                net.layers[nid] = dict(net.layers[L], seg=len(parent) + len(ids) + 1)
                ids.append(nid)
            k_of = np.searchsorted(np.array(cuts), s[idx]) if cuts else np.zeros(len(idx), int)
            lab[idx] = np.array(ids)[k_of]
            for a, b in zip(ids[:-1], ids[1:]):
                adj[a].add(b)
                adj[b].add(a)
            for x_ in ids:
                parent[x_] = L
                out.add(x_)
        ch.lay = lab
        end_lab[(ci, True)] = int(lab[0])
        end_lab[(ci, False)] = int(lab[-1])
    for nd, lst in _node_ends(chains).items():      # stretches of one vessel meeting at a node continue
        labs = [end_lab[(ci, s_)] for ci, s_ in lst if end_lab[(ci, s_)] in parent]
        for a in labs:
            for b in labs:
                if a != b and parent[a] == parent[b]:
                    adj[a].add(b)
    return sorted(out), parent, adj


def _allocate(net: _Net, chains, lays, P: AnatomyParams, F: _Fields, rng, spec: dict, parent=None, adj=None,
              fixed: dict | None = None):
    """Depths (base: lumen top of a nominal-radius vessel; its layer's depth
    field is added) for layers that may cross each other - the stretches of
    the long vessels of the web and of the episcleral systems, placed
    together: where two of them come closer in the image than their lumens
    (radius up to exp(1.5 r_var_sd) x nominal) plus a gap (graph.check's 0.5
    d_c, a margin, their along-vessel depth wobble), they must lie at depths
    that clear each other there (the two depth fields' values at those
    points included).  Placed one at a time, the widest vessels first, each
    at the admissible depth nearest its preferred one: that of a stretch it
    continues (within max_step, so a vessel changes depth gently), else by
    size within its range (larger vessels deeper; veins deeper among the
    episcleral systems); deeper than its range if nothing fits, and where
    nothing clears, the depth that overlaps least (the overlap pass bumps
    what remains).  spec: layer -> dict(lo, hi, overflow, wobble, max_step,
    deep, vein) (a stretch takes its parent's); parent: stretch -> its vessel's
    layer (stretches of one vessel on one chain do not conflict); adj:
    stretch -> the stretches it continues into; fixed: layers already at
    their depth (the twig sheets), to be cleared too.  Returns ({layer:
    base}, {layer: deepest lumen bottom}) of the layers placed."""
    fixed = {int(l): float(b) for l, b in (fixed or {}).items()}
    lays = [int(l) for l in lays]
    parent = parent or {l: l for l in lays}
    adj = adj or {}
    place = list(lays)
    lays = lays + [l for l in fixed if l not in lays]
    sp = {l: spec[parent.get(l, l)] for l in lays}
    f = math.exp(1.5 * P.r_var_sd)
    xs, rr, ls, cs, se, us = [], [], [], [], [], []
    for ci, ch in enumerate(chains):
        m = np.isin(ch.lay, lays)
        if m.any():
            xs.append(ch.pts[m])
            rr.append(ch.rn[m])
            ls.append(ch.lay[m])
            cs.append(np.full(int(m.sum()), ci))
            zu = getattr(ch, "zu", None)
            us.append(zu[m] if zu is not None else np.zeros(int(m.sum())))
            s = _arclen(ch.pts)[m]
            d_end = np.zeros(len(s))                  # distance to the nearer end of its stretch on this chain
            for L in np.unique(ch.lay[m]):
                k = ch.lay[m] == L
                d_end[k] = np.minimum(s[k] - s[k].min(), s[k].max() - s[k])
            se.append(d_end)
    if not xs:
        return {}, {}
    xy, r, lay, cid, dend, zu = (np.concatenate(a) for a in (xs, rr, ls, cs, se, us))
    deep_pt = np.array([sp[int(l)]["deep"] for l in lay], bool)
    # the depth field under each point, and the vessel's own undulation along it (_undulations; v2)
    off = np.where(deep_pt, F.deep_depth(xy), F.tissue_depth(xy)) + zu
    wob = np.array([sp[int(l)]["wobble"] for l in lay])
    rmax = {l: float(r[lay == l].max()) for l in lays if (lay == l).any()}
    rmed = {l: float(np.median(r[lay == l])) for l in rmax}
    zu_lo = {l: min(0.0, float(zu[lay == l].min())) for l in rmax}      # the undulation stays within the range
    zu_hi = {l: max(0.0, float(zu[lay == l].max())) for l in rmax}
    lays = [l for l in place if l in rmax]
    conf = {}
    if len(xy) > 1:
        i, j = close_pairs(xy, f * r, 1.0 + 2 * float(wob.max()))
        par = np.array([parent.get(int(x), int(x)) for x in lay])
        m = (lay[i] != lay[j]) & ~((cid[i] == cid[j]) & (par[i] == par[j]))
        m &= np.hypot(*(xy[i] - xy[j]).T) < f * (r[i] + r[j]) + 1.0 + wob[i] + wob[j]
        i, j = i[m], j[m]
        if len(i):
            cu = np.array([c.u for c in chains])
            cv = np.array([c.v for c in chains])
            ok = np.ones(len(i), bool)
            for na, nb in ((cu, cu), (cu, cv), (cv, cu), (cv, cv)):
                sh = np.flatnonzero((na[cid[i]] == nb[cid[j]]) & ok)
                if len(sh):
                    nxy = np.array([net.xy[n] for n in na[cid[i[sh]]]])
                    reach = 3 * (r[i[sh]] + r[j[sh]]) + 6
                    near = (np.hypot(*(xy[i[sh]] - nxy).T) < reach) & (np.hypot(*(xy[j[sh]] - nxy).T) < reach)
                    ok[sh[near]] = False
            i, j = i[ok], j[ok]
            # where two points meet: the depth spacing needed with i's layer above j's (n_ij) and the other way
            # (n_ji); z = base + field + r and z_b - z_a >= f (r_a + r_b) + gap
            g_ = 1.0 + wob[i] + wob[j]
            n_ij = (f + 1) * r[i] + (f - 1) * r[j] + g_ + off[i] - off[j]
            n_ji = (f + 1) * r[j] + (f - 1) * r[i] + g_ + off[j] - off[i]
            ms = np.array([sp[int(l)]["max_step"] for l in lay])
            K = np.int64(int(lay.max()) + 1)

            def agg(mask):
                key = lay[i[mask]].astype(np.int64) * K + lay[j[mask]]
                uk, inv = np.unique(key, return_inverse=True)
                mx, my = np.full(len(uk), -np.inf), np.full(len(uk), -np.inf)
                np.maximum.at(mx, inv, n_ij[mask])
                np.maximum.at(my, inv, n_ji[mask])
                return zip((uk // K).tolist(), (uk % K).tolist(), mx.tolist(), my.tolist())

            def add(la_, lb_, x_, y_):              # x_: la_ above lb_, y_: lb_ above la_
                if la_ == lb_:
                    return
                k_, v_ = ((la_, lb_), (x_, y_)) if la_ < lb_ else ((lb_, la_), (y_, x_))
                old = conf.get(k_)
                conf[k_] = v_ if old is None else (max(old[0], v_[0]), max(old[1], v_[1]))

            for la_, lb_, x_, y_ in agg(np.ones(len(i), bool)):
                add(la_, lb_, x_, y_)
            # near the end of its stretch a vessel ramps to the depth of the stretch it continues into (the
            # node's): the conflict binds that stretch too
            for la_, lb_, x_, y_ in agg(dend[i] < 2.6 * ms[i] + 3.0 + 2 * r[i]):
                for L2 in adj.get(la_, ()):
                    add(int(L2), lb_, x_, y_)
            for la_, lb_, x_, y_ in agg(dend[j] < 2.6 * ms[j] + 3.0 + 2 * r[j]):
                for L2 in adj.get(lb_, ()):
                    add(la_, int(L2), x_, y_)
    # preferred depth by the vessel's size within its range (its parent layer's median radius; the same for all
    # its stretches), relative to the vessels of its kind
    pr = defaultdict(list)
    for l in lays:
        pr[parent.get(l, l)].append(rmed[l])
    lr = {p: math.log(max(float(np.median(v)), 1e-3)) for p, v in pr.items()}
    uu = {}
    for kind in (False, True):
        ps = [p for p in lr if spec[p]["deep"] == kind]
        if not ps:
            continue
        lo_r, hi_r = min(lr[p] for p in ps), max(lr[p] for p in ps)
        for p in sorted(ps):
            u = (lr[p] - lo_r) / (hi_r - lo_r) if hi_r > lo_r + 1e-9 else 0.5
            if kind:
                u = 0.6 * u + 0.4 * (1.0 if spec[p].get("vein") else 0.0)
            # from 15 % into the range (the band right under the mesh and the twig sheets is crowded: a vessel
            # placed at its very top has no room left to clear what crosses it)
            uu[p] = float(np.clip(0.15 + 0.7 * u + 0.15 * rng.standard_normal(), 0.0, 1.0))
            if not kind and net.trees.get(net.layers[p].get("tree"), {}).get("role") == "collecting":
                uu[p] = float(rng.uniform(*P.collect_depth_u))     # v2: the collecting veins in the deep plexus
    prmax = defaultdict(float)
    for l in lays:
        prmax[parent.get(l, l)] = max(prmax[parent.get(l, l)], rmax[l])
    base = dict(fixed)

    def pick(p, a_lo, a_hi, forb):
        """The admissible depth in [a_lo, a_hi] (outside every open forbidden interval) nearest p."""
        if a_lo > a_hi + 1e-9:
            return None
        if not forb:
            return float(np.clip(p, a_lo, a_hi))
        fb = np.asarray(forb, float)
        cand = np.concatenate([[p, a_lo, a_hi], fb[:, 0], fb[:, 1]])
        cand = cand[(cand >= a_lo - 1e-9) & (cand <= a_hi + 1e-9)]
        cand = cand[~((cand[:, None] > fb[None, :, 0]) & (cand[:, None] < fb[None, :, 1])).any(1)]
        return float(cand[np.argmin(np.abs(cand - p))]) if len(cand) else None

    nbrs = defaultdict(list)               # layer -> [(other, spacing with it above, with the other above)]
    for (a_, b_), (x_, y_) in conf.items():
        nbrs[a_].append((b_, x_, y_))
        nbrs[b_].append((a_, y_, x_))
    # the widest vessels (> 3 d_c) first, then pairs and bundles (their members run side by side for hundreds
    # of d_c: only the depths clear them; a crossing is short, and the overlap pass can bump what is left of
    # one), then by width; a vessel's stretches together, each after one it continues where possible
    def rank(p):
        T = net.trees.get(net.layers[p].get("tree"), {})
        paired = T.get("role") in ("companion", "bundle") or bool(T.get("pair_sides"))
        return (0 if prmax[p] > 3.0 else 1 if paired else 2, -prmax[p], p)
    order = []
    for p in sorted(prmax, key=rank):
        todo = sorted(l for l in lays if parent.get(l, l) == p)
        done_p = []
        while todo:
            nxt = next((l for l in todo if any(a in done_p for a in adj.get(l, ()))), todo[0])
            todo.remove(nxt)
            done_p.append(nxt)
        order += done_p
    for l in order:
        s_ = sp[l]
        lo, hi = s_["lo"], s_["hi"]
        forb = [(base[c] - n_lc, base[c] + n_cl) for c, n_lc, n_cl in nbrs.get(l, ()) if c in base]
        a_lo = lo + (f - 1) * rmax[l] - zu_lo[l]
        a_hi = max(a_lo, hi - (f + 1) * rmax[l] - zu_hi[l])
        nb = [base[a] for a in adj.get(l, ()) if a in base]
        own = a_lo + uu[parent.get(l, l)] * (a_hi - a_lo)
        # continuing a placed stretch: near its depth, drifting back towards the vessel's own preferred depth
        # (else a vessel pushed deep by one crossing would stay deep for the rest of its course)
        pref = own if not nb else float(np.mean(nb)) + float(np.clip(own - np.mean(nb), -0.5 * s_["max_step"],
                                                                      0.5 * s_["max_step"]))
        pref = float(np.clip(pref, a_lo, a_hi))
        # continuing a placed stretch, a vessel changes depth by at most max_step (its ramp then stays within the
        # reach the conflicts were propagated over)
        w_lo, w_hi = a_lo, a_hi + s_["overflow"]
        if nb:
            w_lo, w_hi = max(w_lo, max(nb) - s_["max_step"]), min(w_hi, min(nb) + s_["max_step"])
            if w_lo > w_hi:
                w_lo = w_hi = float(np.clip(np.mean(nb), a_lo, a_hi + s_["overflow"]))
        b = pick(pref, max(w_lo, a_lo), min(w_hi, a_hi), forb) if max(w_lo, a_lo) <= min(w_hi, a_hi) else None
        if b is None:                       # deeper than planned
            b = pick(pref, w_lo, w_hi, forb)
        if b is None:                       # the least overlap
            grid = np.linspace(w_lo, w_hi, 120)
            fb = np.asarray(forb, float).reshape(-1, 2)
            cost = np.maximum(0.0, np.minimum(grid[:, None] - fb[None, :, 0], fb[None, :, 1] - grid[:, None])).sum(1)
            b = float(grid[int(np.argmin(cost))])
        base[l] = float(b)
    bottom = {l: base[l] + (f + 1) * rmax[l] + zu_hi[l] for l in lays}
    return {l: base[l] for l in lays}, bottom


def _layer_depth(net, base, F: _Fields, lay: int, xy, r):
    L = net.layers[lay]
    if L["kind"] == "mesh":
        return base[lay] + F.tissue_depth(xy)
    if L["kind"] in ("sheet", "web"):
        return base[lay] + F.tissue_depth(xy) + r
    return base[lay] + F.deep_depth(xy) + r


def _undulations(net: _Net, chains, P: AnatomyParams, rng):
    """ch.zu: a long vessel's own depth undulation along its course (d_c),
    added to its layers' depth (in _allocate's clearances and in the
    profiles).  The conjunctival stroma is loose connective tissue and its
    vessels rise towards the epithelium and dive along their course, so a
    long vessel fades and sharpens along its length (contrast through the
    scattered-light bypass, width through the defocus); v1's long vessels kept
    one depth per 100-180 d_c stretch (SD 0.75 d_c within a vessel, 3 d_c along
    a whole long vessel): the 'uniform wires' of the v1 realism review (T3;
    along_cv_fwhm -2.1 SD).  A Gaussian process of SD web_depth_undulation
    (half for collecting veins) and correlation length
    web_depth_undulation_corr along each web tree's course (continuous
    through its junctions); a companion or bundle member takes its leader's
    where it runs beside it (they share the connective-tissue sheath); the
    Y-branches of a long vessel their own, starting from the course's value
    at the fork."""
    for ch in chains:
        ch.zu = np.zeros(len(ch.pts))
    sd0 = P.web_depth_undulation
    if sd0 <= 0:
        return
    trees = defaultdict(list)
    for i, ch in enumerate(chains):
        if getattr(ch, "cs", None) is not None:
            trees[int(ch.ctree)].append(i)
    U, corr_t = {}, {}
    for t in sorted(trees):
        T = net.trees.get(t, {})
        L = max(float(np.max(chains[i].cs)) for i in trees[t])
        corr_t[t] = float(rng.uniform(*P.web_depth_undulation_corr))
        sd = sd0 * (0.5 if T.get("role") == "collecting" else 1.0)
        U[t] = sd * _smooth_noise(int(L) + 3, corr_t[t], rng)       # 1 d_c spacing
    node_u = defaultdict(list)
    order = sorted(trees, key=lambda t: (net.trees.get(t, {}).get("lead") is not None, t))   # leaders first
    for t in order:
        T = net.trees.get(t, {})
        lead = T.get("lead")
        lc = trees.get(int(lead), []) if lead is not None else []
        if lc:
            lp = np.concatenate([chains[j].pts for j in lc])
            lu = np.concatenate([chains[j].zu for j in lc])
            kd = cKDTree(lp)
            off = float(T.get("lead_off", T.get("pair_off", 4.0)))
            d1, d2 = 1.25 * off + 2.0, 1.8 * off + 8.0
        for i in trees[t]:
            ch = chains[i]
            u = np.interp(ch.cs, np.arange(len(U[t]), dtype=float), U[t])
            if lc:
                dist, j = kd.query(ch.pts)
                w = ndi.gaussian_filter1d(_smoothstep((d2 - dist) / (d2 - d1)), 4.0, mode="nearest")
                u = w * lu[j] + (1.0 - w) * u
            ch.zu = u
            for k, nd in ((0, ch.u), (-1, ch.v)):
                if not net.is_border(nd):
                    node_u[nd].append(float(u[k]))
    # the other vessels of the web trees (Y-branches): their own, from the node values at their ends
    sd_b = sd0
    for i, ch in enumerate(chains):
        if getattr(ch, "cs", None) is not None or ch.info.get("web") not in ("long", "collecting", "companion",
                                                                              "bundle"):
            continue
        s = _arclen(ch.pts)
        L = float(s[-1])
        if L < 4 or len(s) < 5:
            continue
        h = L / (len(s) - 1)
        corr = float(rng.uniform(*P.web_depth_undulation_corr))
        ua = float(np.mean(node_u[ch.u])) if node_u.get(ch.u) else 0.0
        ub = float(np.mean(node_u[ch.v])) if node_u.get(ch.v) else 0.0
        ch.zu = sd_b * _bridge(_smooth_noise(len(s), corr / h, rng)) + ua + (ub - ua) * s / L


def _depths(net: _Net, chains, P: AnatomyParams, F: _Fields, rng):
    """Node depths (the host layer's depth there, relaxed where a vessel
    would otherwise climb too steeply) and a depth profile per vessel: its
    layers' depths, ramps where it changes layer (no steeper than
    ramp_angle_deg where there is room), flat at its nodes, a small wobble,
    the fades (dives) of deep vessels and the perforators' dive out of
    sight.  v2: the long vessels' own undulation (_undulations) is part of
    their layers' depth, so the allocation clears it."""
    _undulations(net, chains, P, np.random.default_rng(rng.integers(2 ** 63)))
    base, conj_bottom = _layer_bases(net, chains, P, rng, F)
    ends = _node_ends(chains)
    zn = {}
    perf_nodes = {}
    # perforators dive below every episcleral system (they leave through the sclera)
    deep_floor = max([base[l] + 2 * max((float(c.rn.max()) for c in chains if int(l) in set(c.lay.tolist())),
                                        default=5.0)
                      for l, L_ in net.layers.items() if L_["kind"] == "deep" and l in base] + [0.0])
    for ci, ch in enumerate(chains):
        if ch.cls == "perforator" and ch.info.get("layer", "").startswith("deep") and net.is_border(ch.v):
            perf_nodes[ch.v] = max(float(rng.uniform(*P.perforator_depth)),
                                   deep_floor + 3 * P.deep_depth_sd + float(ch.rn.max()) + 10.0)
    host = {}
    for nd, lst in ends.items():
        cand = []
        mesh = False
        for ci, at_start in lst:
            ch = chains[ci]
            lay = int(ch.lay[0] if at_start else ch.lay[-1])
            r = float(ch.rn[0] if at_start else ch.rn[-1])
            if lay == 0:
                mesh = True
            if lay != RAMP:
                cand.append((r, lay, float(ch.zu[0] if at_start else ch.zu[-1])))
        if nd in perf_nodes:
            zn[nd] = perf_nodes[nd]
            continue
        zu_host = 0.0
        if mesh:
            lay, r = 0, 0.0
        elif cand:
            r, lay, zu_host = max(cand)
        else:
            lay = None
        host[nd] = lay
        if lay is None:
            zn[nd] = None
        else:
            r_host = max([rr for rr, ll, _ in cand if ll == lay] + [r])
            zn[nd] = float(_layer_depth(net, base, F, lay, net.xy[nd][None], r_host)[0]) + \
                (zu_host if net.layers[lay]["kind"] == "web" else 0.0)
    for nd in [n for n, z in zn.items() if z is None]:
        zs = [zn[chains[ci].v if s else chains[ci].u] for ci, s in ends[nd]]
        zs = [z for z in zs if z is not None]
        zn[nd] = float(np.mean(zs)) if zs else P.deep_top
    rmx = {nd: max(float(chains[ci].rn[0] if s else chains[ci].rn[-1]) for ci, s in ends[nd]) for nd in zn}
    for nd in zn:
        zn[nd] = max(zn[nd], rmx[nd] + 1.0, P.epithelium + rmx[nd] if zn[nd] < 40 else 0.0)
    tan_max = math.tan(math.radians(P.ramp_angle_deg))
    _relax_node_depths(net, chains, zn, host, rmx, perf_nodes, math.tan(math.radians(P.relax_angle_deg)), P)
    dch = [i for i, ch in enumerate(chains) if ch.info.get("layer", "").startswith("deep") or ch.cls == "perforator"]
    dtree, dids, drmax = None, None, 0.0
    if dch:
        dtree = cKDTree(np.concatenate([chains[i].pts for i in dch]))
        dids = np.concatenate([np.full(len(chains[i].pts), i) for i in dch])
        drmax = max(float(chains[i].rn.max()) for i in dch)
    for ci, ch in enumerate(chains):
        p = ch.pts
        s = _arclen(p)
        L = float(s[-1])
        n = len(p)
        h = L / max(n - 1, 1)
        z = np.full(n, np.nan)
        for lay in np.unique(ch.lay):
            if lay == RAMP:
                continue
            m = ch.lay == lay
            z[m] = _layer_depth(net, base, F, int(lay), p[m], ch.rn[m])
            if net.layers[int(lay)]["kind"] == "web":
                z[m] += ch.zu[m]  # its undulation (v2)
            if m.sum() > 3:     # radius steps between merged edges of one layer: soften
                z[m] = ndi.gaussian_filter1d(z[m], 2.0 / h, mode="nearest")
        if np.isnan(z).all():
            z = np.interp(s, [0, L], [zn[ch.u], zn[ch.v]])
        elif np.isnan(z).any():
            good = ~np.isnan(z)
            zz = np.concatenate([[zn[ch.u]], z[good], [zn[ch.v]]])
            ss = np.concatenate([[-1e-6], s[good], [L + 1e-6]])
            if not np.isnan(z[0]):
                zz, ss = zz[1:], ss[1:]
            if not np.isnan(z[-1]):
                zz, ss = zz[:-1], ss[:-1]
            z = np.interp(s, ss, zz)
        # layer changes: short ramps no steeper than the ramp angle, centred on the change
        z = _slope_limit(z, h, 2 * tan_max)
        z = ndi.gaussian_filter1d(z, 1.0 / h, mode="nearest")
        deep = ch.info["layer"].startswith("deep") or ch.cls in ("episcleral_artery", "episcleral_vein")
        wob = P.deep_z_wobble if deep else P.z_wobble
        if wob > 0 and L > 10:
            z = z + wob * _bridge(_smooth_noise(n, min(40.0, L / 3) / h, rng))
        if deep and L > 80:
            for _ in range(rng.poisson(L * P.dive_rate)):
                c = float(rng.uniform(30, L - 30))
                w = float(rng.uniform(*P.dive_len))
                amp = float(rng.uniform(*P.dive_amp))
                # a fade only where no other deep vessel passes nearby (it would dive through it)
                reg = np.abs(s - c) < w
                if dtree is not None and reg.any():
                    near = dtree.query_ball_point(p[reg], float(np.max(ch.rn)) + drmax + 3.0)
                    if any(dids[h] != ci for hs in near for h in hs):
                        continue
                sig = max(w / 2.5, 0.61 * amp / tan_max)      # a Gaussian's steepest slope is 0.61 amp / sigma
                z = z + amp * np.exp(-0.5 * ((s - c) / sig) ** 2)
        if ch.cls == "perforator" and ch.v in perf_nodes:
            dz_p = max(perf_nodes[ch.v] - float(np.median(z)), 1.0)
            lr = max(float(rng.uniform(30, 60)), 1.5 * dz_p / math.tan(math.radians(P.perforator_angle_deg)))
            opts_s0 = np.sort(rng.uniform(0.1, 0.8, 16)) * L
            best = (np.inf, float(opts_s0[0]))
            for x in opts_s0:             # dive where no other deep vessel passes (it would dive through it)
                reg = (s > x - 5) & (s < x + lr + 5)
                if dtree is None or not reg.any():
                    best = (0, float(x))
                    break
                near = dtree.query_ball_point(p[reg], float(np.max(ch.rn)) + drmax + 3.0)
                hits = sum(1 for hs in near for h in hs if dids[h] != ci)
                if hits < best[0]:
                    best = (hits, float(x))
                if hits == 0:
                    break
            s0 = best[1]
            z = z + (perf_nodes[ch.v] - z) * _smoothstep((s - s0) / lr)
        z = _end_correct(z, s, zn[ch.u], zn[ch.v], ch.rn[0], ch.rn[-1], tan_max)
        z = np.maximum(z, ch.rn + 1.0)
        z[0], z[-1] = zn[ch.u], zn[ch.v]
        ch.z = z
        ch.s = s
    return zn, dict(base={net.layers[k]["name"] + (f".{net.layers[k]['seg']}" if net.layers[k].get("seg") else ""):
                          round(float(v), 2) for k, v in base.items()},
                    conj_bottom=float(conj_bottom))


def _relax_node_depths(net: _Net, chains, zn: dict, host: dict, rmx: dict, fixed, tan_max: float,
                       P: AnatomyParams, iters: int = 100):
    """Where a vessel would have to climb or dive more steeply than the ramp
    angle between its two nodes (a short terminal venule of a deep sheet
    joining the capillary mesh), move its tree-sheet nodes towards the other
    node's depth; mesh, episcleral and border nodes stay.  The climb spreads
    up the tree: the smallest vessels sit shallower, nearer the capillary
    bed."""
    z0 = dict(zn)
    flex = {}
    for nd, lay in host.items():
        kind = net.layers[lay]["kind"] if lay is not None else None
        flex[nd] = 0.0 if (net.is_border(nd) or nd in fixed or kind != "sheet") else 1.0
    arcs = []
    for ch in chains:
        if ch.u == ch.v:
            continue
        L = float(_arclen(ch.pts)[-1])
        room = max(L - (min(2 * ch.rn[0] + 1, 0.25 * L) + min(2 * ch.rn[-1] + 1, 0.25 * L)) - 2.0, 0.5)
        arcs.append((ch.u, ch.v, room * tan_max / 1.5))
    for _ in range(iters):
        delta = defaultdict(float)
        for a, b, lim in arcs:
            d = zn[a] - zn[b]
            if abs(d) <= lim:
                continue
            wa, wb = flex.get(a, 0.0), flex.get(b, 0.0)
            if wa + wb == 0:
                continue
            ex = 0.5 * (abs(d) - lim) * math.copysign(1.0, d)
            delta[a] -= ex * wa / (wa + wb)
            delta[b] += ex * wb / (wa + wb)
        if not delta or max(abs(v) for v in delta.values()) < 0.05:
            break
        for n, dv in delta.items():
            z = float(np.clip(zn[n] + dv, z0[n] - P.relax_max, z0[n] + P.relax_max))
            zn[n] = max(z, rmx[n] + 1.0, P.epithelium + rmx[n])


def _slope_limit(z, h: float, m: float):
    """Steps in z turned into ramps of slope <= m/2 centred on them: the mean
    of the largest m-Lipschitz function below z and the smallest above."""
    z = np.asarray(z, float)
    lo, hi = z.copy(), z.copy()
    d = m * h
    for i in range(1, len(z)):
        lo[i] = min(lo[i], lo[i - 1] + d)
        hi[i] = max(hi[i], hi[i - 1] - d)
    for i in range(len(z) - 2, -1, -1):
        lo[i] = min(lo[i], lo[i + 1] + d)
        hi[i] = max(hi[i], hi[i + 1] - d)
    return 0.5 * (lo + hi)


def _end_correct(z, s, zu, zv, ru, rv, tan_max):
    """Bring a depth profile to its node depths: flat within ~2r+1 of each
    node, then a smooth ramp no steeper than about tan_max (where there is
    room)."""
    L = float(s[-1])
    du, dv = zu - z[0], zv - z[-1]
    fu, fv = min(2 * ru + 1, 0.25 * L), min(2 * rv + 1, 0.25 * L)
    lu, lv = 1.5 * abs(du) / tan_max + 2.0, 1.5 * abs(dv) / tan_max + 2.0
    if fu + lu + fv + lv <= L:
        wu = 1 - _smoothstep((s - fu) / lu)
        wv = 1 - _smoothstep((L - s - fv) / lv)
        return z + du * wu + dv * wv
    c = min(1.0, 0.5 * L / max(fu + fv, 1e-9))
    fu, fv = fu * c, fv * c
    t = _smoothstep((s - fu) / max(L - fu - fv, 1e-9))
    return zu + (zv - zu) * t


# =================================================================== stage: profiles
def _hct_nominal(cls: str, r: float, P, rng) -> float:
    if cls == "capillary":
        return float(rng.uniform(*P.hct_cap))
    if cls in ("episcleral_artery", "episcleral_vein") or r >= 2.5:
        return 1.0
    h0 = float(np.mean(P.hct_cap))
    return float(h0 + (1 - h0) * _smoothstep((r - 0.5) / 2.0))


def _profiles(net: _Net, chains, P: AnatomyParams, rng):
    """Radius (constituent radii, smooth transitions, along-vessel log-normal
    variation vanishing at the nodes) and haematocrit (by class and calibre,
    equal at each node, small variation) along every vessel."""
    ends = _node_ends(chains)
    hn = {}
    for ch in chains:
        ch.h0 = _hct_nominal(ch.cls, float(np.median(ch.rn)), P, rng)
        if ch.info.get("channel"):        # (v2) a thoroughfare channel carries more red cells than the capillaries
            ch.h0 = float(rng.uniform(*P.channel_hct))
    for nd, lst in ends.items():
        hn[nd] = float(np.mean([chains[ci].h0 for ci, _ in lst]))
    for ch in chains:
        s = ch.s
        L = float(s[-1])
        n = len(s)
        h = L / max(n - 1, 1)
        r = ch.rn.astype(float).copy()
        if len(np.unique(r)) > 1:
            sig = max(1.0, float(np.median(r))) / h
            r = np.exp(ndi.gaussian_filter1d(np.log(r), sig, mode="nearest"))
            r = _end_correct(r, s, ch.rn[0], ch.rn[-1], ch.rn[0], ch.rn[-1], 1.0)
        if P.r_var_sd > 0 and L > 6:
            # correlation at least 2.5 diameters (v1: 3; shorter calibre changes make wide vessels look beaded)
            corr = max(P.r_var_corr, 5.0 * float(np.median(ch.rn)))
            # arterioles (muscular, vasomotion) keep a more even calibre than venules (thin-walled, irregular);
            # companions and bundle members are the stills' even doubled lines.  (v1 halved it for both partners of
            # a pair, and every long arteriole has a companion: all of them were 'uniform wires', T3)
            sd = P.r_var_sd_A if ch.side == "A" else P.r_var_sd
            if ch.info.get("web") in ("companion", "bundle"):
                sd *= 0.8
            eps = sd * _bridge(_smooth_noise(n, corr / h, rng))
            if P.r_var_clip > 0:
                c_ = P.r_var_clip * sd
                eps = c_ * np.tanh(eps / c_)
            f = _smoothstep(s / (2 * ch.rn[0] + 1)) * _smoothstep((L - s) / (2 * ch.rn[-1] + 1))
            r = r * np.exp(eps * f)
        r[0], r[-1] = ch.rn[0], ch.rn[-1]
        ch.r = np.maximum(r, 0.2)
        hv = ch.h0 * (1 + P.hct_var * _smooth_noise(n, 10.0 / h, rng))
        hv = _end_correct(hv, s, hn[ch.u], hn[ch.v], ch.rn[0], ch.rn[-1], 1.0)
        ch.h = np.clip(hv, 0.05, 1.0)
    return hn


# =================================================================== the final graph
def _profile_samples(values, s, n_ctrl: int):
    """Values resampled uniformly in tau, 4 per control value (fit_profile
    then uses n_ctrl control values)."""
    m = 4 * n_ctrl
    q = np.linspace(0, s[-1], m)
    return np.interp(q, s, values)


def _n_prof(L: float) -> int:
    return int(np.clip(L / 3.0 + 4, 4, 64))


def _build_graph(net: _Net, chains, zn, hn, meta) -> tuple[VesselGraph, dict]:
    g = VesselGraph(meta)
    nmap = {}
    for nd in sorted({c.u for c in chains} | {c.v for c in chains}):
        pr = net.nd[nd].get("pressure")
        nmap[nd] = g.add_node(net.xy[nd], depth=zn[nd], kind="fork", pressure=pr,
                              **({"border": True, "p_boundary": float(pr)} if pr is not None else {}))
    vmap = {}
    for ci, ch in enumerate(chains):
        n_c = _n_prof(float(ch.s[-1]))
        info = dict(ch.info)
        info.pop("edges", None)
        rad = _profile_samples(ch.r, ch.s, n_c)
        vid = g.add_vessel(nmap[ch.u], nmap[ch.v], ctrl=_vessel_ctrl(g, nmap[ch.u], nmap[ch.v], ch.pts, rad),
                           radius=rad,
                           depth=_profile_samples(ch.z, ch.s, n_c),
                           hct=_profile_samples(ch.h, ch.s, n_c),
                           cls=ch.cls, side=ch.side, **info)
        _clamp_profiles(g.vessels[vid], dict(r=ch.r, z=ch.z, h=ch.h))
        vmap[ci] = vid
    return g, dict(nodes=nmap, vessels=vmap)


def _fix_curvature(g: VesselGraph, kmax: float = 0.85, tries: int = 6) -> int:
    """Re-smooth (locally, where |kappa| r is too high, ends fixed) and
    re-fit any centreline whose |kappa| r exceeds kmax."""
    n = 0
    for e in g.vessels.values():
        target = 0.7
        for _ in range(tries):
            smp = _vsample(e, 0.25)
            if np.max(np.abs(smp["kappa"]) * smp["r"]) <= kmax:
                break
            tmp = _Chain([], e.u, e.v, pts=smp["xy"], rn=smp["r"])
            _limit_curvature(tmp, target, max_iter=400)
            spacing = max(1.5, 2.0 * float(np.mean(smp["r"])))
            e.ctrl = _fit_curve(tmp.pts, spacing)
            e.ctrl[0], e.ctrl[-1] = g.nodes[e.u].xy, g.nodes[e.v].xy
            target *= 0.8
            n += 1
    return n


# =================================================================== faster equivalents of graph.py helpers
def _fit_ctrl_banded(values, n_ctrl: int, smooth: float):
    """vesselmap.spline.fit_ctrl(values, n_ctrl, smooth, pin_ends=True) - the
    same penalised least squares - solved as the banded system it is (a cubic
    B-spline's normal matrix has 3 off-diagonals): O(n) instead of O(n^3)
    for the long vessels of the web (hundreds of control points)."""
    from scipy.linalg import solveh_banded
    from scipy.sparse import csr_matrix
    from vesselscene import spline as sp
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    m = values.shape[0]
    if n_ctrl < 12 or m < 2:
        return sp.fit_ctrl(values, n_ctrl, smooth=smooth)
    B = csr_matrix(sp.design(n_ctrl, m).astype(np.float64))
    rhs = np.asarray(B.T @ values)
    n = n_ctrl
    # upper banded form (4 rows: offsets 3, 2, 1, 0) of B^T B + smooth m D^T D + 1e-9 I
    BtB = (B.T @ B).tocsr()
    ab = np.zeros((4, n))
    for k in range(4):
        d = BtB.diagonal(k)
        ab[3 - k, k:] = d
    lam = smooth * m
    dtd = {0: np.full(n, 6.0), 1: np.full(n - 1, -4.0), 2: np.ones(n - 2)}
    dtd[0][[0, -1]] = 1.0
    dtd[0][[1, -2]] = 5.0
    dtd[1][[0, -1]] = -2.0
    for k in (0, 1, 2):
        ab[3 - k, k:] += lam * dtd[k]
    ab[3] += 1e-9
    ends = np.stack([values[0], values[-1]])
    # the pinned end control points move to the right-hand side
    Acols = np.zeros((n, 2))
    for k in range(4):
        # column 0: A[i, 0] for i = 0..3 is the k-th upper diagonal's first entry
        if k < n:
            Acols[k, 0] = ab[3 - k, k]
            Acols[n - 1 - k, 1] = ab[3 - k, n - 1]
    rhs_i = rhs[1:-1] - Acols[1:-1] @ ends
    # the inner block (rows / columns 1..n-2): in upper banded form its columns are those of the whole matrix
    # (the entries of row 0 fall in the unused upper-left triangle)
    ci = solveh_banded(ab[:, 1:-1].copy(), rhs_i)
    return np.concatenate([ends[:1], ci, ends[1:]], 0)


def _fit_curve(xy, spacing: float, smooth: float = 1e-4) -> np.ndarray:
    """graph.fit_curve, with the banded solve."""
    from vesselscene import spline as sp
    xy = np.asarray(xy, float)
    L = float(sp.arclength(xy)[-1])
    n = sp.n_ctrl_for_length(L, spacing)
    m = max(4 * n, sp.n_samples_for_length(L, 0.25))
    return _fit_ctrl_banded(sp.resample_polyline(xy, m), n, smooth).astype(float)


def _vessel_ctrl(g: VesselGraph, u: int, v: int, path, radius) -> np.ndarray:
    """The control points VesselGraph.add_vessel would fit to `path` (ends
    pinned onto the nodes, spacing about 2 radii, at least 1.5 d_c)."""
    from vesselscene import spline as sp
    pu, pv = g.nodes[u].xy, g.nodes[v].xy
    path = np.asarray(path, float).copy()
    if len(path) == 2:
        path = np.linspace(path[0], path[1], 8)
    w = sp.arclength(path)
    w = w / w[-1] if w[-1] > 0 else np.linspace(0, 1, len(path))
    path += (1 - w)[:, None] * (pu - path[0]) + w[:, None] * (pv - path[-1])
    ctrl = _fit_curve(path, max(1.5, 2.0 * float(np.mean(np.atleast_1d(radius)))))
    ctrl[0], ctrl[-1] = pu, pv
    return ctrl


def _spl(ctrl):
    from scipy.interpolate import BSpline
    from vesselscene import spline as sp
    c = np.asarray(ctrl, float)
    n = len(c)
    k = sp.degree_for(n)
    return BSpline(sp._knots(n, k), c, k, extrapolate=False)


def _vsample(e, spacing: float = 0.25, oversample: int = 16) -> dict:
    """Vessel.sample(spacing, oversample), evaluating the splines with their
    coefficients (graph._basis builds an n x n design matrix: slow for the
    long vessels of the web).  The same numbers up to rounding."""
    from vesselscene import spline as sp
    n = len(e.ctrl)
    spl = _spl(e.ctrl)
    u0 = np.linspace(0.0, 1.0, max(64, oversample * n))
    s0 = sp.arclength(np.nan_to_num(spl(u0)))
    L = float(s0[-1])
    m = max(2, int(math.ceil(L / spacing)) + 1)
    s = np.linspace(0.0, L, m)
    u = np.clip(np.interp(s, s0, u0) if L > 0 else np.linspace(0, 1, m), 0.0, 1.0)
    xy = np.nan_to_num(spl(u))
    d1 = np.nan_to_num(spl.derivative(1)(u))
    d2 = np.nan_to_num(spl.derivative(2)(u))
    sp1 = np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-12)
    t = d1 / sp1[:, None]
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp1 ** 3
    tau = s / L if L > 0 else np.linspace(0, 1, m)
    ev = lambda c: np.nan_to_num(_spl(c)(np.clip(tau, 0.0, 1.0)))  # noqa: E731
    return dict(s=s, tau=tau, u=u, xy=xy, t=t, kappa=kappa, L=L, r=ev(e.r_ctrl), z=ev(e.z_ctrl),
                h=np.clip(ev(e.h_ctrl), 0.0, 1.0))


def _junction_reach(g: VesselGraph, inc: dict, nid: int, gap: float) -> float:
    """VesselGraph.junction_reach(nid, gap) with the node's incident vessels
    given (inc: node -> [(vid, end)]) instead of searched for."""
    from .graph import eval_profile
    arms = []
    for vid, end in inc.get(nid, ()):
        e = g.vessels[vid]
        r = float(eval_profile(e.r_ctrl, [float(end)])[0])
        arms.append((g.end_direction(vid, end), r))
    R = 0.0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            (ti, ri), (tj, rj) = arms[i], arms[j]
            ang = math.acos(float(np.clip(np.dot(ti, tj), -1.0, 1.0)))
            w = ri + rj + gap
            R = max(R, w / max(math.sin(ang), 0.25) if ang < math.radians(150) else w)
    return 1.25 * R + 1.0 if len(arms) >= 2 else 0.0


# =================================================================== overlap removal
def _samples(g: VesselGraph, vids, cache):
    for v in vids:
        cache[v] = _vsample(g.vessels[v], 0.5)


class _Pairs:
    """The sample pairs of different vessels that are close in the image
    (closer than r_a + r_b + gap, away from the nodes they share - the rule
    of graph.check with a margin).  The overlap pass only changes depths,
    so these are found once; each iteration re-tests their depths."""

    def __init__(self, g: VesselGraph, cache: dict, gap: float):
        keys = sorted(cache)
        self.keys = keys
        self.off, n0 = {}, 0
        for k in keys:
            self.off[k] = (n0, n0 + len(cache[k]["xy"]))
            n0 += len(cache[k]["xy"])
        xy = np.concatenate([cache[k]["xy"] for k in keys])
        r = np.concatenate([cache[k]["r"] for k in keys])
        own = np.concatenate([np.full(len(cache[k]["xy"]), k) for k in keys])
        ix = np.concatenate([np.arange(len(cache[k]["xy"])) for k in keys])
        self.z = np.concatenate([cache[k]["z"] for k in keys])
        i, j = close_pairs(xy, r, gap)
        keep = own[i] != own[j]
        i, j = i[keep], j[keep]
        lim = r[i] + r[j] + gap
        # exemption near shared nodes: graph.check's rule (within the node's junction_reach, the extent over
        # which lumens leaving a node at their angles legitimately overlap), with a margin
        U = {k: g.vessels[k].u for k in keys}
        V = {k: g.vessels[k].v for k in keys}
        ua, va = np.array([U[k] for k in own[i]], int), np.array([V[k] for k in own[i]], int)
        ub, vb = np.array([U[k] for k in own[j]], int), np.array([V[k] for k in own[j]], int)
        reach = {}
        inc = defaultdict(list)
        for k in keys:
            inc[g.vessels[k].u].append((k, 0))
            inc[g.vessels[k].v].append((k, 1))
        ok = np.ones(len(i), bool)
        check_gap = max(gap - 0.35, 0.0)             # the rule of graph.check (gap 0.5; the pass adds a margin)
        for na, nb_ in ((ua, ub), (ua, vb), (va, ub), (va, vb)):
            sh = np.flatnonzero((na == nb_) & ok)
            if len(sh):
                for n in set(na[sh].tolist()) - set(reach):
                    reach[n] = _junction_reach(g, inc, n, check_gap) - 0.5
                nxy = np.array([g.nodes[n].xy for n in na[sh]])
                d = np.maximum(np.hypot(*(xy[i[sh]] - nxy).T), np.hypot(*(xy[j[sh]] - nxy).T))
                ok[sh[d < np.array([reach[n] for n in na[sh]])]] = False
        i, j, lim = i[ok], j[ok], lim[ok]
        a, b = own[i], own[j]
        swap = a > b
        self.a, self.b = np.where(swap, b, a), np.where(swap, a, b)
        self.gi, self.gj = np.where(swap, j, i), np.where(swap, i, j)          # global indices
        self.ia, self.ib = ix[self.gi], ix[self.gj]
        self.lim = lim
        self.gap = gap
        n = len(self.a)
        ves = np.concatenate([self.a, self.b])
        ent = np.concatenate([np.arange(n), np.arange(n)])
        srt = np.argsort(ves, kind="stable")
        self._ves, self._ent = ves[srt], ent[srt]

    def entries(self, vid):
        """Indices of the pairs that involve vessel vid."""
        lo = np.searchsorted(self._ves, vid, "left")
        hi = np.searchsorted(self._ves, vid, "right")
        return self._ent[lo:hi]

    def update(self, cache, vids):
        for k in vids:
            lo, hi = self.off[k]
            self.z[lo:hi] = cache[k]["z"]

    def conflicts(self, gap: float | None = None):
        """(a, b, i, j, need) of the pairs whose depths are too close, or None."""
        lim = self.lim if gap is None else self.lim - self.gap + gap
        dz = np.abs(self.z[self.gi] - self.z[self.gj])
        m = dz < lim
        if not m.any():
            return None
        return self.a[m], self.b[m], self.ia[m], self.ib[m], (lim - dz)[m]


def _zfloor(z, P) -> float:
    """Shallowest allowed lumen top: under the epithelium in the conjunctiva."""
    return 0.5 if z > 40 else max(0.5, P.epithelium - 2.0)


def _bump(e_smp, s_lo, s_hi, wl, wr, amount):
    """A depth bump: `amount` over [s_lo, s_hi], smooth shoulders of widths
    wl / wr outside it, exactly zero within 0.5 d_c of the vessel's ends."""
    s = e_smp["s"]
    L = float(s[-1])
    w = np.ones(len(s))
    w = np.where(s < s_lo, _smoothstep((s - (s_lo - wl)) / max(wl, 1e-6)), w)
    w = np.where(s > s_hi, _smoothstep(((s_hi + wr) - s) / max(wr, 1e-6)), w)
    w[(s < 0.5) | (s > L - 0.5)] = 0.0
    return amount * w


def _resolve(g: VesselGraph, P: AnatomyParams, max_iter: int, gap: float = 0.5, margin: float = 0.35,
             rounds: int = 3) -> dict:
    """Remove every 3-D lumen overlap: where two vessels come too close,
    the cheaper of the two to move (thin, room for a smooth bump) gets a
    depth bump away from the other, keeping their depth order; a conflict
    at both vessels' own ends moves a node (with all its vessels' ends).
    Vessels of one layer rarely meet (growth keeps them apart); a bump
    between them makes a crossing at another depth and is counted
    ('same_layer').  Conflicts that bumps cannot clear remove the thinner
    capillary-scale vessel (a missed faint vessel is better than a merge),
    and the topology is repaired (_graph_remove)."""
    log = dict(iterations=0, bumps=0, node_moves=0, same_layer=0, remaining=0, pairs_first=0, stuck=0,
               surgery_removed=0, surgery_merged=0)
    cache = {}
    _samples(g, list(g.vessels), cache)
    pairs = _Pairs(g, cache, gap + margin)
    log["forced"] = 0
    for rnd in range(rounds + 1 if max_iter > 0 else 0):
        left = _resolve_loop(g, P, cache, pairs, max_iter if rnd == 0 else max(6, max_iter // 3), gap, margin,
                             log)
        if not left or rnd == rounds:
            break
        last = rnd == rounds - 1          # last resort: the less conspicuous of any pair still in conflict goes
        if last:                          # but first steeper bumps (65 deg; mostly deep trunks against the plexus)
            left = _resolve_loop(g, replace(P, bump_angle_deg=max(P.bump_angle_deg, 65.0)), cache, pairs,
                                 max(6, max_iter // 3), gap, margin, log)
            if not left:
                break
        victims = set()
        for va, vb in left:
            v = _pick_victim(g, cache, va, vb, last)
            if v is not None:
                victims.add(v)
        if last:
            log["forced"] += len(victims)
            log.setdefault("forced_r", []).extend(round(float(np.mean(cache[v]["r"])), 2) for v in sorted(victims))
        log.setdefault("removed_r", []).extend(round(float(np.mean(cache[v]["r"])), 2) for v in sorted(victims))
        log.setdefault("removed_len_in_frame", []).extend(round(_in_frame_len(g, cache[v]), 1) for v in sorted(victims))
        if not victims:
            continue
        rep = _graph_remove(g, sorted(victims))
        log["surgery_removed"] += rep["removed"]
        log["surgery_merged"] += rep["merged"]
        cache = {}
        _samples(g, list(g.vessels), cache)
        pairs = _Pairs(g, cache, gap + margin)
    c = pairs.conflicts(gap + 0.05)
    log["remaining"] = 0 if c is None else len(set(zip(c[0].tolist(), c[1].tolist())))
    return log


def _pick_victim(g: VesselGraph, cache: dict, va: int, vb: int, last: bool):
    """Which of two vessels in an unresolved conflict the overlap pass
    removes: before the last round only a capillary-scale or side vessel
    that is not a long vessel of the web; in the last round either.  Of the
    candidates, the one whose removal costs the image least (_visual_cost:
    calibre x length in the frame x a depth fade).  (The key was (web
    vessel?, r, L), so in the last round a non-web vessel went first whatever
    its width, e.g. episcleral veins of r 6-10 against a bundle member;
    correctness review v1, defect 3.)  None if neither qualifies."""
    cand = []
    for v in (va, vb):
        e = g.vessels[v]
        r = float(np.mean(cache[v]["r"]))
        roles = set(e.info.get("roles", []))
        small = r < 1.5 or (cache[v]["L"] < 20 and r < 2.5)
        long_ = e.info.get("web") in ("long", "collecting", "companion", "bundle")
        if last or ((small or e.cls == "capillary" or roles & {"companion", "side"}) and not long_):
            cand.append((_visual_cost(g, cache[v], e), r, cache[v]["L"], v))
    return min(cand)[-1] if cand else None


def _in_frame_len(g: VesselGraph, smp: dict) -> float:
    """Length (d_c) of a sampled vessel inside the frame."""
    x0, y0, x1, y1 = g.meta.get("frame", (-np.inf, -np.inf, np.inf, np.inf))
    xy = smp["xy"]
    m = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return float(m.sum()) * float(smp["L"]) / max(len(m) - 1, 1)


def _visual_cost(g: VesselGraph, smp: dict, e=None) -> float:
    """What removing a vessel costs the image: its mean radius x its length
    in the frame (+ 2 % of its length outside, to break ties off-frame) x
    exp(-z / 80), a proxy for how much a deeper vessel fades (the chord law's
    bypass length, scene.CALIBRATED_OPTICS l_bypass); x 10 for a piece of a
    collecting vein (v2: a 150 d_c piece of the frame's black vein cost less
    than a long thin venule merged over 2580 d_c, and its removal cut the
    darkest vessel of the frame in two)."""
    r = float(np.mean(smp["r"]))
    lf = _in_frame_len(g, smp)
    w = 10.0 if e is not None and e.info.get("web") == "collecting" else 1.0
    return w * r * (lf + 0.02 * (float(smp["L"]) - lf)) * math.exp(-float(np.median(smp["z"])) / 80.0)


def _best_shift(pairs: "_Pairs", mv: int, w, cache, zfloor: float, zmax: float):
    """The smallest amplitude d of a depth bump with profile d * w (w: 1 on
    the conflict, smooth shoulders, 0 at the ends) along vessel mv that
    clears every neighbour of the moved samples (not only the vessel it is
    in conflict with): each neighbour forbids an interval of d; returns the
    admissible d closest to zero, or None."""
    smp = cache[mv]
    moved = np.nonzero(w > 0.02)[0]
    ent = pairs.entries(mv)
    is_a = pairs.a[ent] == mv
    mine = np.where(is_a, pairs.ia[ent], pairs.ib[ent])
    other = np.where(is_a, pairs.gj[ent], pairs.gi[ent])
    mask = np.zeros(len(smp["s"]), bool)
    mask[moved] = True
    sel = mask[mine]
    k = mine[sel]
    wi, zi, zo, lim = w[k], smp["z"][k], pairs.z[other[sel]], pairs.lim[ent][sel]
    lo, hi = (zo - lim - zi) / wi, (zo + lim - zi) / wi
    zp, rp, wp = smp["z"][moved], smp["r"][moved], w[moved]
    dmin = float(np.max((zfloor - (zp - rp)) / wp))
    dmax = float(np.min((zmax - zp) / wp))
    # union of the forbidden intervals (widened by 0.3 d_c of margin), sorted and merged
    if len(lo):
        o = np.argsort(lo)
        lo, hi = lo[o] - 0.3, hi[o] + 0.3
        mlo, mhi = [lo[0]], [hi[0]]
        for a_, b_ in zip(lo[1:], hi[1:]):
            if a_ <= mhi[-1]:
                mhi[-1] = max(mhi[-1], b_)
            else:
                mlo.append(a_)
                mhi.append(b_)
        mlo, mhi = np.array(mlo), np.array(mhi)
    else:
        mlo = mhi = np.zeros(0)
    cand = np.concatenate([[0.0], mlo, mhi])
    cand = cand[(cand >= dmin) & (cand <= dmax)]
    if len(mlo):
        i = np.searchsorted(mlo, cand, side="right") - 1
        inside = (i >= 0) & (cand > mlo[np.maximum(i, 0)]) & (cand < mhi[np.maximum(i, 0)])
        cand = cand[~inside]
    if not len(cand):
        return None
    return float(cand[np.argmin(np.abs(cand))])


def _resolve_loop(g: VesselGraph, P: AnatomyParams, cache: dict, pairs: "_Pairs", max_iter: int, gap: float,
                  margin: float, log: dict) -> list:
    """Sequential bump iterations (see _resolve): conflicts are cleared one
    pair at a time, each with the smallest shift that clears all the
    mover's neighbours there; returns the vessel pairs still in conflict."""
    from .graph import eval_profile
    conj_bottom = float(g.meta.get("depth_layout", {}).get("conj_bottom", 45.0))
    tan_inc = math.tan(math.radians(P.bump_angle_deg))
    tries = defaultdict(int)
    stuck = set()

    def deep(smp):
        return float(np.median(smp["z"])) > conj_bottom + 2

    def refresh(vids):
        _samples(g, vids, cache)
        pairs.update(cache, vids)

    for it in range(max_iter):
        c = pairs.conflicts()
        if c is None:
            return []
        log["iterations"] += 1
        todo = sorted(set(zip(c[0].tolist(), c[1].tolist())))
        if log["pairs_first"] == 0:
            log["pairs_first"] = len(todo)
        if set(todo) <= stuck:
            break
        for va, vb in todo:
            if (va, vb) in stuck:
                continue
            ent = pairs.entries(va)
            ent = ent[pairs.b[ent] == vb] if len(ent) else ent
            if not len(ent):
                continue
            live = np.abs(pairs.z[pairs.gi[ent]] - pairs.z[pairs.gj[ent]]) < pairs.lim[ent]
            if not live.any():
                continue                  # cleared by an earlier move this iteration
            ent = ent[live]
            ea, eb = g.vessels[va], g.vessels[vb]
            same_layer = ea.info.get("layer") == eb.info.get("layer") and ea.info.get("layer") not in (None, "ramp")
            if tries[(va, vb)] == 0 and same_layer:
                log["same_layer"] += 1
            tries[(va, vb)] += 1
            if tries[(va, vb)] > 12:
                stuck.add((va, vb))       # oscillating: left to the surgery
                log["stuck"] += 1
                continue
            opts = []
            for mv, idx, oidx in ((va, pairs.ia[ent], pairs.gj[ent]), (vb, pairs.ib[ent], pairs.gi[ent])):
                sm = cache[mv]
                s = sm["s"]
                L = float(s[-1])
                s_lo, s_hi = float(s[idx].min()), float(s[idx].max())
                plateau = np.nonzero((s >= s_lo) & (s <= s_hi))[0]
                width = max(3.0, 1.5 * (float(sm["r"][idx].max()) + float(np.max(pairs.lim[ent]))) / 2)
                zmx = 400.0 if deep(sm) else conj_bottom + 6.0
                for _ in range(3):            # shoulders wide enough that the bump is no steeper than ramp_angle
                    wl, wr = min(width, s_lo - 0.5), min(width, L - s_hi - 0.5)
                    room = wl >= 0.75 and wr >= 0.75
                    if room:
                        wprof = _bump(sm, s_lo, s_hi, wl, wr, 1.0)
                    else:                     # a node move shifts the end region
                        wprof = np.zeros(len(s))
                        wprof[plateau] = 1.0
                    d = _best_shift(pairs, mv, wprof, cache, _zfloor(float(sm["z"][plateau].min()), P), zmx)
                    if d is None or not room:
                        break
                    need_w = 1.5 * abs(d) / tan_inc
                    if min(wl, wr) >= need_w - 1e-6:
                        break
                    if width >= need_w:       # the vessel's end limits the shoulder: move its node instead
                        room = False
                        wprof = np.zeros(len(s))
                        wprof[plateau] = 1.0
                        d = _best_shift(pairs, mv, wprof, cache, _zfloor(float(sm["z"][plateau].min()), P), zmx)
                        break
                    width = need_w
                if d is None or d == 0.0 or abs(d) > max(10.0, 2.5 * float(np.max(pairs.lim[ent]))):
                    continue                  # no admissible shift, or only a plunge through other vessels
                steep = abs(d) / max(min(wl, wr), 0.75)
                cost = abs(d) * (1.0 + float(np.median(sm["r"]))) * (1.0 + 0.3 * steep) * (1.0 if room else 1e3)
                opts.append((cost, mv, room, s_lo, s_hi, wl, wr, d, idx, zmx))
            if not opts:
                stuck.add((va, vb))
                log["stuck"] += 1
                continue
            opts.sort(key=lambda o: (o[0], o[1]))
            t = tries[(va, vb)]
            pick = opts[1] if (t > 4 and len(opts) == 2 and opts[1][2] and t % 2) else opts[0]
            cost, mv, room, s_lo, s_hi, wl, wr, d, idx, zmx = pick
            sm = cache[mv]
            if room:
                _set_depth(g.vessels[mv], sm, sm["z"] + _bump(sm, s_lo, s_hi, wl, wr, d))
                log["bumps"] += 1
                refresh([mv])
                continue
            cap = 30.0 if zmx > 100 else 4.0
            if abs(d) > cap:
                stuck.add((va, vb))
                log["stuck"] += 1
                continue
            # the conflict sits at both vessels' own ends: move the mover's node, with all its vessels' ends
            e = g.vessels[mv]
            nd_id = e.u if float(sm["s"][idx].mean()) < 0.5 * float(sm["s"][-1]) else e.v
            node = g.nodes[nd_id]
            inc = [x for x in g.vessels.values() if nd_id in (x.u, x.v)]
            r_inc = max(float(eval_profile(x.r_ctrl, [0.0 if x.u == nd_id else 1.0])[0]) for x in inc)
            floor = max(r_inc + 1.0, _zfloor(node.depth, P) + r_inc)
            if node.depth + d < floor:
                d = floor - node.depth       # not above the surface (or into the epithelium)
                if abs(d) < 1e-3:
                    stuck.add((va, vb))
                    log["stuck"] += 1
                    continue
            node.depth = float(node.depth + d)
            moved = []
            for vid in [k for k, x in g.vessels.items() if nd_id in (x.u, x.v)]:
                x = g.vessels[vid]
                smp = cache[vid]
                s = smp["s"]
                L = float(s[-1])
                z = eval_profile(x.z_ctrl, smp["tau"])
                lr = 1.5 * abs(d) / tan_inc + 2
                for end in ("u", "v"):
                    if getattr(x, end) != nd_id:
                        continue
                    f = min(2 * float(smp["r"][0 if end == "u" else -1]) + 1, 0.3 * L)
                    dist = s if end == "u" else L - s
                    w = 1 - _smoothstep((dist - f) / lr)
                    w = w - w[-1 if end == "u" else 0] * dist / max(L, 1e-9)   # leave the far end alone
                    z = z + d * w
                _set_depth(x, smp, z)
                moved.append(vid)
            log["node_moves"] += 1
            refresh(moved)
    c = pairs.conflicts()
    return [] if c is None else sorted(set(zip(c[0].tolist(), c[1].tolist())))


def _self_pairs(smp: dict, margin: float = 0.0):
    """(i, j), i < j: samples of one vessel whose lumens overlap in 3-D
    (graph.check's self rule: more than pi (r_i + r_j) apart along it, closer
    than r_i + r_j + margin in the image and in depth)."""
    xy, r, z, s = smp["xy"], smp["r"], smp["z"], smp["s"]
    i, j = close_pairs(xy, r, margin)
    rs = r[i] + r[j] + margin
    m = (np.abs(s[i] - s[j]) > math.pi * (r[i] + r[j])) & (np.hypot(*(xy[i] - xy[j]).T) < rs) & \
        (np.abs(z[i] - z[j]) < rs)
    return i[m], j[m]


def _resolve_self(g: VesselGraph, P: AnatomyParams, gap: float = 0.5, margin: float = 0.3) -> dict:
    """A vessel whose lumen overlaps itself (a loop closing at one depth;
    rare, in strongly tortuous vessels) would render as two vessels adding.
    Its later stretch gets a smooth depth bump (away from the earlier one,
    else towards and past it) of just the amount needed, if that keeps it
    clear of every other vessel; failing that, a capillary-scale vessel is
    removed and the topology repaired (a missed faint vessel is better than
    a merge).  The fitted depth profile is checked: a bump the fit loses (a
    short loop on a long vessel: correctness review v2, defect 3, 128
    control points 6.6 d_c apart kept 0.54 of a 1.37 d_c bump) is refitted
    with finer control points, else the other direction is tried, else the
    vessel is removed.  Returns counts (self_left: vessels still overlapping
    themselves afterwards, 0 unless a removal failed)."""
    log = dict(self_found=0, self_bumped=0, self_refit=0, self_removed=0, self_left=0)
    todo = []
    for vid, e in g.vessels.items():
        smp = _vsample(e, 0.5)
        i, j = _self_pairs(smp, margin)
        if len(i):
            todo.append(vid)
    log["self_found"] = len(todo)
    if not todo:
        return log
    cache = {}
    _samples(g, list(g.vessels), cache)
    keys = sorted(cache)
    allxy = np.concatenate([cache[k]["xy"] for k in keys])
    allr = np.concatenate([cache[k]["r"] for k in keys])
    allz = np.concatenate([cache[k]["z"] for k in keys])
    own = np.concatenate([np.full(len(cache[k]["xy"]), k) for k in keys])
    start = {k: int(np.searchsorted(own, k)) for k in keys}
    tree = cKDTree(allxy)
    rmax = float(allr.max())
    tan_inc = math.tan(math.radians(P.bump_angle_deg))
    victims = []
    for vid in todo:
        e = g.vessels[vid]
        smp = cache[vid]
        i, j = _self_pairs(smp, margin)
        s, z, r = smp["s"], smp["z"], smp["r"]
        L = float(s[-1])
        jj = np.maximum(i, j)
        ii = np.minimum(i, j)
        s_lo, s_hi = float(s[jj].min()), float(s[jj].max())
        dz = z[jj] - z[ii]
        need = float(np.max(r[i] + r[j] + margin + 0.2 - np.abs(dz)))
        sign0 = 1.0 if np.mean(dz) >= 0 else -1.0
        # others near the stretch that moves (their pairs with this vessel, exempt near shared nodes)
        done = False
        for d in (sign0 * need, -sign0 * (need + 2 * float(np.max(np.abs(dz))))):
            wl = min(1.5 * abs(d) / tan_inc + 1.0, s_lo - 0.5)
            wr = min(1.5 * abs(d) / tan_inc + 1.0, L - s_hi - 0.5)
            if wl < 0.75 or wr < 0.75:
                continue
            w = _bump(smp, s_lo, s_hi, wl, wr, 1.0)
            if np.any(w[ii] > 0.02):
                continue                           # the shoulders reach the earlier stretch: no clean bump
            znew = z + d * w
            if np.any(znew[1:-1] - r[1:-1] < _zfloor(float(np.median(z)), P)):
                continue
            def clear(zc, mv):
                for p, hits in zip(mv, tree.query_ball_point(smp["xy"][mv], r[mv] + rmax + gap)):
                    hits = np.asarray(hits, int)
                    hits = hits[own[hits] != vid]
                    if not len(hits):
                        continue
                    dd = np.hypot(*(allxy[hits] - smp["xy"][p]).T)
                    lim = r[p] + allr[hits] + gap
                    if np.any((dd < lim) & (np.abs(allz[hits] - zc[p]) < lim)):
                        return False
                return True
            if not clear(znew, np.flatnonzero(w > 0.02)):
                continue
            z_old = np.array(e.z_ctrl, float)
            for n_max in (128, 1024):
                _set_depth(e, smp, znew, n_max)
                fit = _vsample(e, 0.5)
                zf = fit["z"] if len(fit["z"]) == len(z) else np.interp(s, fit["s"], fit["z"])
                if not len(_self_pairs(fit, 0.0)[0]) and clear(zf, np.flatnonzero(np.abs(zf - z) > 0.01)):
                    break
                e.z_ctrl = z_old.copy()
            else:
                continue                            # the fit cannot hold this bump: the other direction, or removal
            allz[start[vid]:start[vid] + len(z)] = zf
            log["self_bumped"] += 1
            log["self_refit"] += int(n_max > 128)
            done = True
            break
        if not done:
            # removed whatever its calibre (a wider one was left, so the truth failed check(overlap=True):
            # correctness review v1, defect 4; _tortuosity no longer folds non-capillary vessels, so this is rare)
            victims.append(vid)
            log.setdefault("self_removed_r", []).append(round(float(np.mean(r)), 2))
    if victims:
        rep = _graph_remove(g, victims)
        log["self_removed"] = rep["removed"]
    log["self_left"] = sum(1 for e in g.vessels.values() if len(_self_pairs(_vsample(e, 0.5), 0.0)[0]))
    return log


def _clamp_profiles(e, arr: dict):
    """Keep the fitted profiles within the range of the values they were
    fitted to (a B-spline stays within its control values; the fit's
    control values can ring past a sharp step)."""
    for attr, key in (("r_ctrl", "r"), ("z_ctrl", "z"), ("h_ctrl", "h")):
        v = np.asarray(arr[key], float)
        setattr(e, attr, np.clip(getattr(e, attr), v.min(), v.max()))


def _merge_pair(g: VesselGraph, k1: int, k2: int, n: int) -> int | None:
    """Join the two vessels meeting at interior node n into one vessel
    (path, radius, depth and haematocrit concatenated) and drop n."""
    e1, e2 = g.vessels[k1], g.vessels[k2]
    s1, s2 = _vsample(e1, 0.5), _vsample(e2, 0.5)
    f1 = e1.v == n                      # e1 runs towards n
    f2 = e2.u == n                      # e2 runs away from n
    a = e1.u if f1 else e1.v
    b = e2.v if f2 else e2.u
    if a == b:
        return None
    arr = {}
    for key in ("xy", "r", "z", "h"):
        x1 = s1[key] if f1 else s1[key][::-1]
        x2 = s2[key] if f2 else s2[key][::-1]
        arr[key] = np.concatenate([x1, x2[1:]])
    # the course's turn where the two met: a thin pair folding back (> 90 deg) is not one vessel (it made U-turns
    # and corners: 54 % of merged vessels turned > 60 deg within 6 d_c; correctness review v1, defect 3)
    d_in = -g.end_direction(k1, 1 if f1 else 0)
    d_out = g.end_direction(k2, 0 if f2 else 1)
    turn = math.degrees(math.acos(float(np.clip(d_in @ d_out, -1.0, 1.0))))
    r_n = float(arr["r"][len(s1["r"]) - 1])
    if turn > 90.0 and r_n < 1.5:
        return None
    tmp = _Chain([], a, b, pts=arr["xy"], rn=arr["r"])
    _limit_curvature(tmp, 0.7, max_iter=200, rho_min=max(2.0, 2.0 * r_n) if turn > 30.0 else 0.0)   # round it
    arr["xy"] = tmp.pts
    s = _arclen(arr["xy"])
    nn = int(np.clip(s[-1] / 1.5 + 6, 6, 128))
    big = e1 if s1["L"] * np.mean(s1["r"]) >= s2["L"] * np.mean(s2["r"]) else e2
    info = {k: v for k, v in big.info.items() if k not in ("stagnant",)}
    info["merged_from"] = [int(k1), int(k2)]
    rad = _profile_samples(arr["r"], s, nn)
    vid = g.add_vessel(a, b, ctrl=_vessel_ctrl(g, a, b, arr["xy"], rad), radius=rad,
                       depth=_profile_samples(arr["z"], s, nn), hct=_profile_samples(arr["h"], s, nn),
                       cls=big.cls, side=big.side, **info)
    _clamp_profiles(g.vessels[vid], arr)
    g.remove_vessel(k1)
    g.remove_vessel(k2)
    del g.nodes[n]
    return vid


def _graph_remove(g: VesselGraph, vids) -> dict:
    """Remove vessels and repair the topology: dead ends pruned, interior
    nodes left with two vessels merged, islands without a pressure drop
    removed; then flow re-solved and re-oriented."""
    import networkx as nx
    out = dict(removed=0, merged=0)
    for v in vids:
        if v in g.vessels:
            g.remove_vessel(v)
            out["removed"] += 1
    border = {n for n, nd in g.nodes.items() if nd.info.get("border")}
    while True:
        inc = defaultdict(list)
        for k, e in g.vessels.items():
            inc[e.u].append(k)
            inc[e.v].append(k)
        changed = False
        for n in list(g.nodes):
            ks = inc.get(n, [])
            if not ks:
                del g.nodes[n]
                continue
            if n in border:
                continue
            if len(ks) == 1 or (len(ks) == 2 and ks[0] == ks[1]):
                g.remove_vessel(ks[0])
                out["removed"] += 1
                changed = True
                break
            if len(ks) == 2:
                if _merge_pair(g, ks[0], ks[1], n) is None:
                    g.remove_vessel(ks[0])
                    g.remove_vessel(ks[1])
                    out["removed"] += 2
                else:
                    out["merged"] += 1
                changed = True
                break
        if changed:
            continue
        # two vessels between the same two nodes, one of them short: they lie on each other (merges made such
        # coincident pairs 1 d_c long); the thinner goes
        by_pair = defaultdict(list)
        for k, e in g.vessels.items():
            by_pair[frozenset((e.u, e.v))].append(k)
        for ks in by_pair.values():
            if len(ks) < 2:
                continue
            sm = {k: _vsample(g.vessels[k], 0.5) for k in ks}
            ks = sorted(ks, key=lambda k: float(np.mean(sm[k]["r"])))
            k0, k1_ = ks[0], ks[-1]
            if min(sm[k0]["L"], sm[k1_]["L"]) < 3.0 * (float(np.max(sm[k0]["r"])) + float(np.max(sm[k1_]["r"]))) + 2:
                g.remove_vessel(k0)
                out["removed"] += 1
                changed = True
                break
        if changed:
            continue
        H = nx.Graph()
        for e in g.vessels.values():
            H.add_edge(e.u, e.v)
        for comp in list(nx.connected_components(H)):
            ps = [g.nodes[n].info.get("p_boundary") for n in comp if n in border]
            ps = [p for p in ps if p is not None]
            if len(ps) < 2 or max(ps) - min(ps) < 0.05:
                for k in [k for k, e in g.vessels.items() if e.u in comp]:
                    g.remove_vessel(k)
                    out["removed"] += 1
                changed = True
        if not changed:
            break
    for n in [n for n in g.nodes if n not in {x for e in g.vessels.values() for x in (e.u, e.v)}]:
        del g.nodes[n]
    for n, nd in g.nodes.items():
        nd.pressure = nd.info.get("p_boundary") if n in border else None
    g.solve_flow(orient=True)
    return out


def _set_depth(e, smp, z, n_max: int = 128):
    """Fit the vessel's depth profile to z at its samples smp (control
    points every 1.5 d_c, at most n_max: on a long vessel 128 are too coarse
    for a short bump, which _resolve_self then refits with more)."""
    s = smp["s"]
    z = np.array(z, float)
    z[1:-1] = np.maximum(z[1:-1], smp["r"][1:-1] + 0.6)       # the ends stay at their nodes' depths
    n = int(np.clip(float(s[-1]) / 1.5 + 6, 6, n_max))   # fine enough to hold a local bump
    vals = _profile_samples(z, s, n)
    e.z_ctrl = np.clip(fit_profile(vals, n), vals.min(), vals.max())


# =================================================================== entry point
def _streams(seed: int) -> dict:
    ss = np.random.SeedSequence(seed)
    return {name: np.random.default_rng(child) for name, child in zip(STAGES, ss.spawn(len(STAGES)))}


def generate(params: AnatomyParams | None = None, seed: int = 0) -> VesselGraph:
    """A physiological vessel network for one scene (module docstring).
    The graph's meta holds the domain, frame, tissue region and k, the
    parameters, the seed, per-stage diagnostics and timings."""
    P = replace(params) if params is not None else AnatomyParams()
    R = _streams(seed)
    T = {}
    t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        t1 = time.perf_counter()
        T[name] = round(t1 - t0, 3)
        t0 = t1

    rs = R["setup"]
    ori = P.orientation_deg if P.orientation_deg is not None else 90.0 + P.orientation_sd_deg * rs.standard_normal()
    lsign = P.limbus_sign if P.limbus_sign is not None else int(rs.choice([-1, 1]))
    gammas = dict(A=float(rs.uniform(*P.gamma_A)), V=float(rs.uniform(*P.gamma_V)))
    gammas["DA"], gammas["DV"] = gammas["A"], gammas["V"]
    geo = _Geo(P)
    F = _Fields(P, geo, math.radians(ori), R["fields"])
    net = _Net()
    net.layers[0] = dict(name="mesh", kind="mesh")
    lap("setup")
    tids = _grow_deep(net, P, F, geo, R["deep"], gammas)
    lap("deep")
    sheets = _make_sheets(P, net, R["sheets"])
    riser_roots = _add_risers(net, P, F, geo, R["risers"], tids, sheets)
    _add_avas(net, P, geo, R["ava"], tids)
    lap("risers_avas")
    web, twig_starts = None, {}
    if P.layout == "web":
        # the long conjunctival vessels, their arcades and the twigs that leave them for the capillary bed; the
        # trees on riser tops are twigs too
        web = _grow_web(net, P, F, geo, R["web"], gammas)
        _add_arcades(net, P, geo, R["web"], web)
        twig_starts = _twig_roots(net, P, F, geo, R["web"], sheets, web)
        for tr in net.trees.values():
            if tr.get("riser_path") is not None:
                tr["max_len"] = float(R["web"].uniform(*P.twig_len))
        lap("web")
    elif P.layout != "forest":
        raise ValueError(f"unknown layout {P.layout!r}")
    tips = []
    for sh, srng in zip(sheets, R["sheets"].spawn(len(sheets))):
        roots = twig_starts.get(sh["name"], []) if web is not None else _border_roots(net, P, F, geo, srng, sh, lsign)
        starts = roots + riser_roots.get(sh["name"], [])
        for t in [t for t in net.trees if net.trees[t].get("sheet") == sh["name"]]:
            net.trees[t]["gamma"] = gammas[sh["side"]]
        tips += _colonize(net, starts, P, F, geo, srng, sh, _riser_obstacles(net, sh, sheets))
    lap("forests")
    aspect = _grow_mesh(net, P, F, geo, R["mesh"])
    lap("mesh")
    _attach(net, P, geo, R["attach"], gammas, tips)
    lap("attach")
    pr1 = _prune(net)
    n_channels = _channels(net, P, R["channels"]) if P.layout == "web" else 0
    rad = _assign_radii(net, P, gammas)
    n_contracted = _contract_short(net, P.contract_factor)
    n_pairs = _add_pairs(net, P, geo, R["pairs"])
    n_pairs += _add_pairs(net, P, geo, R["pairs"], host="V")
    pr2 = _prune(net)
    rad = _assign_radii(net, P, gammas)
    n_contracted += _contract_short(net, P.contract_factor)
    pr3 = _prune(net)
    rad = _assign_radii(net, P, gammas)
    order = _strahler(net)
    lap("prune_radii_pairs")
    chains, lost = _chains(net)
    for ch in chains:
        _chain_arrays(net, ch)
        _chain_class(net, ch, order)
    geo_log = _geometry(net, chains, P, R["geometry"], gammas)
    lap("geometry")
    zn, depth_log = _depths(net, chains, P, F, R["depth"])
    hn = _profiles(net, chains, P, R["profiles"])
    lap("depth_profiles")
    meta = dict(domain=tuple(float(x) for x in geo.domain), frame=tuple(float(x) for x in geo.frame),
                tissue=tuple(float(x) for x in geo.tissue), k=float(P.k), frame_px=tuple(P.frame_px),
                seed=int(seed), generator="vesselscene.anatomy", version=VERSION,
                seeds=dict(entropy=int(seed), stages=list(STAGES),
                           rule="numpy SeedSequence(seed).spawn(len(stages)): one stream per stage, in this order"),
                orientation_deg=float(ori), limbus_sign=int(lsign), gammas=gammas, mesh_aspect=float(aspect),
                depth_layout=depth_log,
                params=asdict(P))
    g, maps = _build_graph(net, chains, zn, hn, meta)
    n_curv = _fix_curvature(g)
    lap("build")
    for nd, n in maps["nodes"].items():
        if net.nd[nd].get("pressure") is not None:
            g.nodes[n].pressure = float(net.nd[nd]["pressure"])
        else:
            g.nodes[n].pressure = None
    g.solve_flow(orient=True)
    lap("flow")
    res = _resolve(g, P, P.resolve_iters)
    for _ in range(3):                               # merged vessels may need it; then re-check overlaps (whose
        if not _fix_curvature(g):                    # repairs may merge vessels again)
            break
        res2 = _resolve(g, P, max(6, P.resolve_iters // 3))
        for k_, v_ in res2.items():
            if isinstance(v_, (int, float)) and k_ != "remaining":
                res[k_] = res.get(k_, 0) + v_
        res["remaining"] = res2["remaining"]
    res.update(_resolve_self(g, P))
    if P.layout == "web":
        # arcades the geometry made cross a vessel they join go first (their removal merges the host's pieces,
        # refitted: the late passes below clear what that moved; at the very end it left overlaps)
        res["arcades_crossing_host"] = _drop_crossing_arcades(g)
        res["c1"] = _web_c1(g)                     # long vessels run smoothly through their junctions
        if res["c1"] or res["arcades_crossing_host"]:     # a turned leg or a refit may have moved onto a neighbour
            res2 = _resolve(g, P, 6)
            for k_, v_ in res2.items():
                if isinstance(v_, (int, float)) and k_ != "remaining":
                    res[k_] = res.get(k_, 0) + v_
            res["remaining"] = res2["remaining"]
        late = _resolve_self(g, P)                 # the turned legs and bumps may have closed a self-overlap
        res["self_found_late"] = late["self_found"]
        res["self_bumped_late"] = late["self_bumped"]
        res["self_left_late"] = late["self_left"]
        res["self_removed"] = res.get("self_removed", 0) + late["self_removed"]
    # the late curvature refit changed some vessels' lengths (and so their conductances): flows are solved
    # again so that they match the final geometry (the orientation can only flip for a stagnant vessel), from
    # the boundary pressures alone (solve_flow leaves every node's solved pressure set: solving again with those
    # fixed would only recompute the old flows)
    for nd in g.nodes.values():
        nd.pressure = nd.info.get("p_boundary") if nd.info.get("border") else None
    g.solve_flow(orient=True)
    res["dead_removed"] = _remove_dead(g)
    if res["dead_removed"]:                      # its merges refit vessels too
        res2 = _resolve(g, P, 6)
        res["remaining"] = res2["remaining"]
        res.update({k_: res.get(k_, 0) + v_ for k_, v_ in _resolve_self(g, P).items() if isinstance(v_, int)})
        for nd in g.nodes.values():
            nd.pressure = nd.info.get("p_boundary") if nd.info.get("border") else None
        g.solve_flow(orient=True)
    lap("resolve")
    _finalize(g, net, chains, maps, rad)
    _aspects(g, P, R["aspect"])
    web_log = None
    if web is not None:
        web_log = dict(web.log)
        web_log["pairs"] = _web_partners(g, net)
        web_log["length_in_tissue"] = {k: round(v, 1) for k, v in web.length.items()}
        n_pairs += web_log["pairs"]
    g.meta["diagnostics"] = dict(
        pruned=pr1["removed"] + pr2["removed"] + pr3["removed"], contracted=n_contracted, lost_edges=lost,
        pairs=n_pairs, channels=n_channels, fork_angles=geo_log["fork_angles"], spread=geo_log["spread"],
        through=geo_log["through"],
        shared_meander_nodes=geo_log["shared_meander_nodes"],
        curvature_refits=n_curv, resolve=res, depth=depth_log, web=web_log,
        clipped=len(rad["clipped"]), floored=len(rad["floored"]), flow=_flow_checks(g), **_counts(g))
    g.meta["timings"] = T
    g.meta["runtime_s"] = round(sum(T.values()), 3)
    return g


def _drop_crossing_arcades(g: VesselGraph) -> int:
    """An arcade that crosses one of the two long vessels it joins (away
    from its own nodes) is removed: its clearance was checked when it was
    planned, before tortuosity and the later geometry changes reshaped both
    (4 % of arcades crossed their own host in the frame; correctness review
    v1, defect 8).  An arcade is an anastomosis, so the long vessel's two
    pieces at each end merge back into one (_graph_remove).  Returns the
    number removed."""
    from shapely.geometry import LineString, Point
    WEB = ("long", "collecting", "companion", "bundle")
    by_tree = defaultdict(list)
    at_node = defaultdict(set)
    for k, e in g.vessels.items():
        if e.info.get("web") in WEB and e.info.get("tree") is not None:
            by_tree[int(e.info["tree"])].append(k)
            at_node[e.u].add(int(e.info["tree"]))
            at_node[e.v].add(int(e.info["tree"]))
    lines = {}

    def line(k):
        if k not in lines:
            s = _vsample(g.vessels[k], 0.5)
            lines[k] = (LineString(s["xy"]), float(np.max(s["r"])))
        return lines[k]
    bad = []
    for k, e in g.vessels.items():
        if e.info.get("web") != "arcade":
            continue
        la, ra = line(k)
        hosts = at_node.get(e.u, set()) | at_node.get(e.v, set())
        ends = [Point(g.nodes[n].xy) for n in (e.u, e.v)]
        for t in hosts:
            for h in by_tree[t]:
                lh, rh = line(h)
                hit = la.intersection(lh)
                if hit.is_empty:
                    continue
                pts = [hit] if hit.geom_type == "Point" else list(getattr(hit, "geoms", [hit]))
                far = 3.0 * (ra + rh) + 2.0
                if any(min(p.distance(q) for q in ends) > far for p in pts):
                    bad.append(k)
                    break
            if bad and bad[-1] == k:
                break
    if bad:
        _graph_remove(g, bad)
    return len(bad)


def _aspects(g: VesselGraph, P: AnatomyParams, rng):
    """info['aspect'] of every vessel: its lumen's depth-wise extent over its
    width (the renderer's chord is 2 r aspect; its projected width stays 2 r).
    Venules and veins are thin-walled and at low transmural pressure take an
    elliptical cross-section flattened in the plane of the tissue (collapsible
    tubes); arterioles and capillaries stay round.  A venous vessel's aspect
    falls from 1 at r = vein_flat_r[0] to its own minimum, drawn from
    vein_aspect, at r >= vein_flat_r[1].  (v1 fix, realism review T2: the
    widest shallow veins' plateau was 0.66-0.84 Np against the stills'
    0.41-0.44 while their widths and edges matched; a round lumen of their
    width saturates the chord law.)"""
    lo, hi = P.vein_flat_r
    by_tree = {}
    for vid in sorted(g.vessels):
        e = g.vessels[vid]
        a_min = float(rng.uniform(*P.vein_aspect))
        # one minimum per vessel tree (v2: drawn per graph vessel, a collecting vein's flattening, and so its
        # darkness, jumped at every confluence on its way: 0.43 -> 0.61 -> 0.46)
        t = e.info.get("tree")
        if t is not None:
            a_min = by_tree.setdefault(int(t), a_min)
        if e.side not in ("V",) and e.cls not in ("episcleral_vein",):
            e.info["aspect"] = 1.0
            continue
        r = float(np.mean(e.r_ctrl))
        w = float(_smoothstep((r - lo) / max(hi - lo, 1e-6)))
        e.info["aspect"] = round(1.0 - (1.0 - a_min) * w, 4)


def _remove_dead(g: VesselGraph, rel: float = 1e-9) -> int:
    """Every vessel must carry flow: one with |Q| <= rel x the largest lies
    on no inlet -> outlet path in effect (e.g. a capillary mesh cut off from
    the web: random seed 4 had no twig, its mesh carried 1e-20 of the flow and
    its zero-flow vessels were oriented at random, leaving inlets and outlets
    inside the domain; correctness review v1, defect 2).  Such vessels are
    removed and the topology repaired (_graph_remove); then no node inside the
    domain may be a source or a sink, else the generator raises.  Returns the
    number removed (0 for a valid anatomy)."""
    qmax = max((e.flow for e in g.vessels.values()), default=0.0)
    dead = [k for k, e in g.vessels.items() if e.flow <= rel * qmax]
    if dead:
        _graph_remove(g, dead)
    bad = []
    for nid, (i, o) in g.degrees().items():
        if (i == 0 or o == 0) and not g.nodes[nid].info.get("border"):
            bad.append(nid)
    if bad:
        raise RuntimeError(f"anatomy: {len(bad)} nodes inside the domain are sources or sinks (e.g. {bad[:5]})")
    return len(dead)


def _finalize(g: VesselGraph, net: _Net, chains, maps, rad):
    """Vessel and node records for scoring: partner, tree fork / Murray
    flags, velocity, flow direction against the tree's."""
    remap = {}                            # vessels merged by the overlap pass -> the merged vessel
    for vid, e in g.vessels.items():
        for m in e.info.get("merged_from", []):
            remap[m] = vid

    def current(v):
        while v not in g.vessels and v in remap:
            v = remap[v]
        return v if v in g.vessels else None

    inv = {}
    for ci, vid in maps["vessels"].items():
        for e, _ in chains[ci].edges:
            if current(vid) is not None:
                inv[e] = current(vid)
    for ci, vid in maps["vessels"].items():
        ch = chains[ci]
        if vid not in g.vessels:          # removed or merged by the overlap pass
            continue
        e = g.vessels[vid]
        ps = []
        for x, _ in ch.edges:
            X = net.E[x]
            if "partner" in X.flags and X.flags["partner"] in inv:
                e.info["partner"] = int(inv[X.flags["partner"]])
                pp = [int(inv[y]) for y in X.flags.get("partner_path", []) if y in inv]
                if pp:
                    e.info["partner_path"] = sorted(set(pp))
            for y in X.flags.get("partners", []):
                if y in inv:
                    ps.append(int(inv[y]))
        if ps:
            e.info["partners"] = sorted(set(ps))
        smp = _vsample(e, 1.0)
        e.info["velocity"] = float(e.flow / (math.pi * float(np.mean(smp["r"])) ** 2))
        tree_edges = [(x, f) for x, f in ch.edges if net.E[x].tree is not None]
        if tree_edges:
            x, f = tree_edges[0]
            kind = net.trees[net.E[x].tree]["kind"]
            # built from ch.u to ch.v, where that ran root -> leaf iff f; solve_flow may have reversed it
            flipped = e.u != maps["nodes"][ch.u]
            root_to_leaf = f != flipped
            # the tree's own sense of flow: root -> leaf on the arterial side, leaf -> root on the venous side
            e.info["flow_as_tree"] = bool(root_to_leaf == (kind in ("A", "DA")))
    for e in g.vessels.values():          # vessels made by the overlap pass's merges
        if "velocity" not in e.info:
            e.info["velocity"] = float(e.flow / (math.pi * float(np.mean(_vsample(e, 1.0)["r"])) ** 2))
    # tree forks: Murray bookkeeping
    clipped = rad["clipped"] | rad["floored"]
    node_of = {v: k for k, v in maps["nodes"].items()}
    gdeg = defaultdict(int)
    for e in g.vessels.values():
        gdeg[e.u] += 1
        gdeg[e.v] += 1
    for nid, node in g.nodes.items():
        nd = node_of.get(nid)
        if nd is None or gdeg[nid] != len(net.adj[nd]):
            continue                      # changed by the overlap pass
        mem = [(x, net.E[x]) for x in net.adj[nd]]
        trees = {X.tree for _, X in mem if X.tree is not None}
        for t in trees:
            par = [x for x, X in mem if X.tree == t and X.v == nd]
            kid = [x for x, X in mem if X.tree == t and X.u == nd]
            fk = [x for x, X in mem if t in X.attach and X.attach[t] == nd and X.role != "arcade"]
            if len(par) == 1 and len(kid) + len(fk) >= 2:
                node.info["tree_fork"] = int(t)
                node.info["tree_kind"] = net.trees[t]["kind"]
                node.info["gamma"] = float(net.trees[t]["gamma"])
                node.info["murray_exact"] = not any(x in clipped for x in par + kid)


def _web_c1(g: VesselGraph, kmax: float = 0.85) -> int:
    """Where a long vessel of the web passes a junction (its two pieces are
    two vessels of the graph), make its centreline tangent-continuous there:
    a clamped B-spline leaves its node along its first control leg, so both
    pieces' first legs are turned onto their common direction (each keeping
    its length).  A turn that would bend a piece past the curvature bound is
    not made.  Returns the number of nodes made C1."""
    ends = defaultdict(list)
    for k, e in g.vessels.items():
        if e.info.get("web") in ("long", "collecting", "companion", "bundle"):
            ends[e.u].append((k, 0, e.info["tree"]))
            ends[e.v].append((k, 1, e.info["tree"]))
    n = 0
    for nid, lst in ends.items():
        if g.nodes[nid].info.get("border"):
            continue
        by = defaultdict(list)
        for k, end, t in lst:
            by[t].append((k, end))
        for arms in by.values():
            if len(arms) != 2:
                continue
            (a, ea), (b, eb) = arms
            da, db = g.end_direction(a, ea), g.end_direction(b, eb)
            if float(-da @ db) < math.cos(math.radians(150)):
                continue                  # not a through course (two pieces of one long vessel at a node always
                                          # are; was 30 deg, leaving kinks of 46-108 deg: review v1, defect 6)
            t = _unit(da - db)            # the common direction, pointing along piece a
            old = {}
            for k, end, sgn in ((a, ea, 1.0), (b, eb, -1.0)):
                e = g.vessels[k]
                c = e.ctrl
                i1 = 1 if end == 0 else len(c) - 2
                old[k] = c.copy()
                leg = float(np.hypot(*(c[i1] - c[0 if end == 0 else -1])))
                c = c.copy()
                c[i1] = g.nodes[nid].xy + sgn * leg * t
                e.ctrl = c
            ok = True
            for k in (a, b):
                smp = _vsample(g.vessels[k], 0.25)
                if np.max(np.abs(smp["kappa"]) * smp["r"]) > kmax:
                    ok = False
            if ok:
                n += 1
            else:
                for k, c in old.items():
                    g.vessels[k].ctrl = c
    return n


def _web_partners(g: VesselGraph, net: _Net) -> int:
    """The web's pairs (a companion beside its host, neighbours in a bundle)
    as vessel records: a vessel of the companion's tree lying mostly within
    1.6 x the pair's offset of its host gets info 'partner' (the host vessel
    it runs beside most) and 'partner_path' (the host vessels it runs
    beside); those get 'partners'.  Returns the number of paired vessels."""
    by_tree = defaultdict(list)
    for vid, e in g.vessels.items():
        if e.info.get("tree") is not None:
            by_tree[int(e.info["tree"])].append(vid)
    n = 0
    for ct, T in net.trees.items():
        ht = T.get("pair_host")
        if ht is None or not by_tree.get(ct) or not by_tree.get(ht):
            continue
        off = float(T["pair_off"])
        hs = [(v, _vsample(g.vessels[v], 0.5)["xy"]) for v in by_tree[ht]]
        kd = cKDTree(np.concatenate([x for _, x in hs]))
        hid = np.concatenate([np.full(len(x), v) for v, x in hs])
        for v in by_tree[ct]:
            e = g.vessels[v]
            d, k = kd.query(_vsample(e, 0.5)["xy"])
            near = d < 1.6 * off + 1.0
            if near.mean() < 0.5:
                continue
            vals, counts = np.unique(hid[k[near]], return_counts=True)
            e.info["partner"] = int(vals[int(np.argmax(counts))])
            e.info["partner_path"] = sorted(int(x) for x in vals)
            for x in vals:
                h = g.vessels[int(x)].info
                h["partners"] = sorted(set(h.get("partners", [])) | {int(v)})
            n += 1
    return n


def _flow_checks(g: VesselGraph) -> dict:
    """Flow against the trees' own sense (arterial root -> tips, venous tips
    -> root) and the flow-calibre consistency: Spearman rank correlation of
    flow and radius over the arteriole and venule vessels."""
    from scipy.stats import spearmanr
    ft = [e.info["flow_as_tree"] for e in g.vessels.values() if "flow_as_tree" in e.info]
    out = dict(flow_as_tree=round(float(np.mean(ft)), 4) if ft else None,
               stagnant=int(sum(bool(e.info.get("stagnant")) for e in g.vessels.values())))
    for cls in ("arteriole", "venule"):
        es = [e for e in g.vessels.values() if e.cls == cls]
        if len(es) > 5:
            q = [e.flow for e in es]
            r = [float(np.mean(_vsample(e, 2.0)["r"])) for e in es]
            out[f"spearman_flow_radius_{cls}"] = round(float(spearmanr(q, r)[0]), 3)
    return out


def _counts(g: VesselGraph) -> dict:
    cls = defaultdict(int)
    length = defaultdict(float)
    x0, y0, x1, y1 = g.meta["frame"]
    in_frame = defaultdict(float)
    for e in g.vessels.values():
        cls[e.cls] += 1
        smp = _vsample(e, 1.0)
        length[e.cls] += smp["L"]
        m = _Geo.inside(smp["xy"], (x0, y0, x1, y1))
        in_frame[e.cls] += float(m.sum()) * smp["L"] / max(len(m) - 1, 1)
    area = (x1 - x0) * (y1 - y0)
    kinds = defaultdict(int)
    for n in g.nodes.values():
        kinds[n.kind] += 1
    return dict(vessels=dict(cls), length=dict(length),
                frame_length_density={k: v / area for k, v in in_frame.items()},
                node_kinds=dict(kinds), n_vessels=len(g.vessels), n_nodes=len(g.nodes))


# =================================================================== a quick look
def draw_quicklook(g: VesselGraph, path: str, px_per_dc: float | None = None, show_domain: bool = True):
    """A line drawing of the network (not a render): centrelines coloured
    by class, width by radius, lighter with depth; the frame outlined."""
    import cv2
    colours = dict(capillary=(80, 200, 80), arteriole=(40, 40, 230), venule=(230, 80, 40),
                   episcleral_artery=(20, 20, 140), episcleral_vein=(140, 40, 20),
                   anastomosis=(200, 0, 200), perforator=(0, 160, 220))
    x0, y0, x1, y1 = g.meta["domain"] if show_domain else g.meta["frame"]
    s = px_per_dc or g.meta.get("k", 3.0) * (0.5 if show_domain else 1.0)
    W, H = int((x1 - x0) * s) + 1, int((y1 - y0) * s) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    order = sorted(g.vessels.values(), key=lambda e: -float(np.mean(_vsample(e, 2.0)["z"])))
    for e in order:
        smp = _vsample(e, 1.0)
        pts = ((smp["xy"] - [x0, y0]) * s * 16).astype(np.int32)
        col = np.array(colours.get(e.cls, (0, 0, 0)), float)
        fade = float(np.clip(np.mean(smp["z"]) / 150.0, 0, 0.7))
        col = tuple(int(c) for c in (col * (1 - fade) + 255 * fade))
        w = max(1, int(round(2 * float(np.mean(smp["r"])) * s)))
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, w, cv2.LINE_AA, shift=4)
    fx0, fy0, fx1, fy1 = g.meta["frame"]
    cv2.rectangle(img, (int((fx0 - x0) * s), int((fy0 - y0) * s)), (int((fx1 - x0) * s), int((fy1 - y0) * s)),
                  (0, 0, 0), 1)
    cv2.imwrite(path, img)
    return path
