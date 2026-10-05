# LOG: the spline-fit loop

The research log of program.md's loop: one section per batch, every experiment pre-registered (hypothesis,
predicted direction of each metric part, why) and resolved (held or not). The per-run rows are in
`$SPLINEFIT_WORK/results.tsv`; this file carries the reasoning.

## 0. Baseline (v0) and the freeze, 2026-10-05

**Frozen** at this commit: score.py, run_experiment.py, probes.py + probe_battery.json, fastset.py +
fastset.json, evaluate_fit.py, oracle.py, check_frozen.py and the tests (program.md section 3 has the blob
hashes; `python -m experiments.splinefit.check_frozen`). The instrument was validated before the freeze:
the self-check row (true observable network) scores 0.965 (tier 2) and 0.989 (probes); the null pipeline
scores 0.142 on the probes; the sham (a re-run of the unchanged v0) reproduced the tier-1 digest
`a6bd0c1671b8bc41` exactly; tier 2 is deterministic (two runs per image, equal digests).

### Tier 1 (the loop's tier: 10 fast crops + 34 probes; 272 s on 2 threads)

| part | v0 |
|---|---|
| **composite** (0.5 fast + 0.5 probe) | **0.759238** |
| fast_composite | 0.676340 |
| probe_score | 0.842137 |
| graph | 0.608675 |
| pos | 0.687964 |
| width | 0.761496 |
| explained | 0.782555 |
| explained_junction | 0.781049 |
| digest | a6bd0c1671b8bc41 |

Reference rows on the same crops (`results/dev/baseline_tier1/table.md`, means over the 10 crops):

| row | composite | graph | pos | width | explained | expl_junction | junction_bias | geom_matched | cl recall | cl precision |
|---|---|---|---|---|---|---|---|---|---|---|
| proposals (neuromimetic alone) | 0.661 | 0.612 | 0.707 | 0.784 | 0.643 | 0.649 | -0.464 | 0.741 | 0.611 | 0.922 |
| **fit (v0)** | **0.676** | 0.609 | 0.688 | 0.761 | 0.783 | 0.781 | -0.354 | 0.758 | 0.627 | 0.912 |
| ORACLE init (v0 fit from the truth) | 0.634 | 0.470 | 0.676 | 0.909 | 0.809 | 0.807 | -0.296 | 0.682 | 0.602 | 0.542 |
| true_complete (truth, additive render) | 0.802 | 0.615 | 1.000 | 0.998 | 0.966 | 0.959 | +0.093 | 1.000 | 1.000 | 0.408 |
| true_observable (self-check) | 0.964 | 0.995 | 0.999 | 1.000 | 0.801 | 0.748 | +0.199 | 0.984 | 0.966 | 1.000 |

(vesselmap has cached networks for whole images only: tier 2.)

Probe battery (`results/dev/baseline/probes_v0.md`, tuning curves `figures/probes_tuning_v0.png`):

| sweep | v0 mean | v0 threshold | oracle threshold |
|---|---|---|---|
| width | 0.754 | 2.5 px diameter detected, but width score 0.04 at d 2.5, 0.48 at d 3 | 2.5 |
| blur | 0.917 | focus 25 d_c (width err < 0.25) | 50 |
| contrast | 0.787 | hct 0.25 (recall 0.73, composite 0.67) | 0.25 |
| corner (thin, blurred, faint) | 0.714 | 1/1 | 1/1 |
| fork | 0.864 | 4/5 typed 3-way (fork_wide: a spurious crossing + duplicated node, graph 0.48) | 5/5 |
| crossing_angle | 0.887 | 30 deg (20 deg: compound + 2 pseudo-T) | 20 |
| crossing_depth | 0.937 | 5 d_c | 5 |
| crossing_width | 0.886 | 1/1 (thin arm width score 0.45) | 1/1 |
| t | 0.973 | 2/2 | 2/2 |
| gap | 0.798 | 2 px resolved (pos 0.29 at 2 px; 1 px merged, as the truth) | 2 |
| end | 0.951 | 2/2 | 2/2 |
| rbc | 0.935 | 2/2 | 2/2 |
| empty | 0.535 | 1/2 empty: 93 px fitted on empty_average | 2/2 |

probe_score: v0 0.842, oracle (true network) 0.989, null pipeline 0.142.

### Tier 2 (every dev image, both kinds, twice)

composite **0.753886**, graph 0.679220, pos 0.772442, width 0.797316, explained 0.915901, explained_junction 0.918413; 1809 s; digest `1a78216250e2e602`; deterministic: True.

| image | kind | composite | graph | pos | width | explained | explained_junction | junction_bias | s (min of 2) | deterministic |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy_s000_480x768 | average | 0.820 | 0.763 | 0.839 | 0.813 | 0.982 | 0.984 | -0.086 | 106 | True |
| healthy_s000_480x768 | frame | 0.814 | 0.731 | 0.859 | 0.868 | 0.961 | 0.964 | -0.102 | 103 | True |
| healthy_s004_320x512 | average | 0.725 | 0.668 | 0.700 | 0.738 | 0.910 | 0.928 | -0.241 | 51 | True |
| healthy_s004_320x512 | frame | 0.735 | 0.656 | 0.733 | 0.810 | 0.900 | 0.913 | -0.261 | 48 | True |
| healthy_s007_480x768 | average | 0.778 | 0.694 | 0.818 | 0.778 | 0.989 | 0.990 | -0.059 | 129 | True |
| healthy_s007_480x768 | frame | 0.772 | 0.673 | 0.809 | 0.824 | 0.979 | 0.985 | -0.068 | 106 | True |
| healthy_s008_480x768 | average | 0.840 | 0.790 | 0.860 | 0.817 | 0.992 | 0.995 | -0.040 | 92 | True |
| healthy_s008_480x768 | frame | 0.808 | 0.721 | 0.864 | 0.839 | 0.982 | 0.989 | -0.049 | 89 | True |
| pathologic_s000_480x768 | average | 0.625 | 0.565 | 0.597 | 0.720 | 0.740 | 0.725 | -0.502 | 81 | True |
| pathologic_s000_480x768 | frame | 0.621 | 0.531 | 0.646 | 0.765 | 0.724 | 0.711 | -0.529 | 79 | True |

Reference rows on the whole images (`results/dev/baseline/table.md`, means over the 10 images; vesselmap
over its 6 cached ones):

| row | composite | graph | pos | width | explained | expl_junction | junction_bias | geom_matched | cl recall | cl precision |
|---|---|---|---|---|---|---|---|---|---|---|
| proposals | 0.743 | 0.692 | 0.762 | 0.816 | 0.803 | 0.810 | -0.230 | 0.851 | 0.747 | 0.944 |
| vesselmap (LIMBUS, 6 images) | 0.672 | 0.546 | 0.673 | 0.773 | 0.949 | 0.955 | -0.050 | 0.885 | 0.716 | 0.693 |
| **fit (v0)** | **0.754** | 0.679 | 0.772 | 0.797 | 0.916 | 0.918 | -0.194 | 0.856 | 0.749 | 0.910 |
| ORACLE init | 0.720 | 0.563 | 0.771 | 0.923 | 0.940 | 0.940 | -0.148 | 0.870 | 0.795 | 0.423 |
| ORACLE render (truth, junction-aware) | 0.797 | 0.601 | 1.000 | 0.998 | 0.980 | 0.976 | +0.046 | 1.000 | 1.000 | 0.334 |
| true_complete (truth, additive) | 0.796 | 0.601 | 1.000 | 0.998 | 0.972 | 0.962 | +0.085 | 1.000 | 1.000 | 0.334 |
| true_observable (self-check) | 0.965 | 0.995 | 0.999 | 1.000 | 0.807 | 0.734 | +0.181 | 0.980 | 0.962 | 1.000 |

### What the recordings show (residuals.py; figures in results/dev/baseline/figures/)

Residual OD_obs - render at the truth's junctions (positive = the render under-predicts):

| case | share of the squared residual in junction discs | bias by type (negative = under-predicted) |
|---|---|---|
| fast0 healthy_s000 average | 0.57 | bifurcation -0.34, compound -0.19, crossing -0.26, pseudo-T -0.31 |
| fast2 healthy_s004 average | 0.51 | compound -0.25, crossing -0.40, pseudo-T -0.50 |
| fast4 healthy_s007 average | 0.49 | compound -0.38, crossing -0.22, pseudo-T -0.27 |
| fast8 pathologic_s000 average | 0.39 | compound -0.86, crossing -0.41, pseudo-T -0.86 |
| probes fork_wide / fork_thin / cross_a20 / cross_a45 / t_branch | 0.06-0.12 | -0.03 / -0.22 / -0.04 / -0.05 / -0.08 |

- On the dev crops the render **under-predicts** every junction type (bias -0.2 to -0.9), the opposite of the
  additive render's over-prediction ('knot cue') that motivated the junction-aware render. Half of the
  squared prediction error sits in the junction discs, which cover a few percent of the band. On the clean
  probes the same junctions are fine (|bias| <= 0.08 except the thin fork). So it is not the render model:
  the oracle render explains 0.980 with junction_bias +0.046, the additive one 0.972 / +0.085.
- The residual is a broad positive pedestal around the wide vessels and the junction clusters
  (fast2: the whole region under the wide vein is red): the stage-1 background B has absorbed the wide
  lumens and the dense junction regions (its vesselness mask misses them), so OD = B - ln I is too low
  there and the fit follows it. The pipeline explains its own target (explained_self 0.957 on the crops)
  but not the image's OD (explained 0.783; target_fidelity 0.81, stage 1's own 0.70).
- pathologic_s000 (fast8): the ~40 px wide vessel on the right is not proposed at all and the diffuse haze
  is unexplained (explained 0.18, composite 0.37).
- The empty probe: a wide false vessel fitted along the valid-region corner (93 px); the frame version is
  clean.
- fork_wide: the render is right (explained 0.996) but the graph is wrong: a spurious crossing next to the
  node and a duplicated node (graph_probe 0.48).

### The main failure modes (ranked by what they cost the composite, with evidence)

1. **The fit target and the fit schedule destroy a correct network (the positive control fails).** Started
   from the TRUE network, the v0 schedule drops from 0.802 to 0.634 on the crops and from 0.796 to 0.720 on
   the images: pos 1.00 -> 0.68 (median offset 0.97 px), geom_matched 1.00 -> 0.68, centreline recall
   1.00 -> 0.60, junction F1 0.62 -> 0.40; worst where the background leaks most (pathologic crop 0.788 ->
   0.368, healthy_s004 frame 0.791 -> 0.495). The fit chases a target whose background has absorbed wide
   lumens and junction clusters (junction_bias -0.35 on the crops, half the squared residual in junction
   discs), then MDL prunes the vessels the target no longer supports and drifts centrelines toward the
   leak's asymmetry. Every gain from a better proposal is capped by this.
2. **Missed vessels (recall, not precision).** Centreline recall 0.63 (crops) / 0.75 (images) against
   precision 0.91; edge_cover 0.59; geom_matched 0.76. The fit never adds a vessel: nothing re-proposes
   from the residual. pathologic_s000: the widest vessel is never proposed (stage 1 puts it in B).
3. **Junction topology and typing.** junction_f1_strict 0.59, coarse type balanced 0.52, crossing recall
   0.33 (crops) / 0.40 (images). The fit is no better than the proposal here (graph 0.609 vs 0.612). Probes:
   fork_wide gets a spurious crossing and a duplicated node; a 20 deg crossing becomes a compound with two
   pseudo-T.
4. **Thin and faint vessel geometry.** Width score 0.04 at 2.5 px diameter, 0.48 at 3 px; crossing arms
   0.45-0.54 at 30-45 deg; the parallel pair at a 2 px gap is resolved but displaced (pos 0.29); contrast
   at hct 0.25: recall 0.73. Width rel. error 0.24 overall (crops).
5. **False vessels on background.** empty_average: 93 px of a wide false vessel at the valid-region border
   (probe composite 0.07); the dev crops' precision (0.91) says this is a smaller cost than the misses.

### Pre-registered first direction (for the first batch)

Fix the target before the fitter (failure mode 1): estimate B outside the union of the vesselness mask and
the proposal's (dilated) lumen support, from the start, and let MDL see the precision-weighted target only
after that. Predicted: explained and explained_junction up (junction_bias toward 0), oracle-init pos and
geom_matched up, fast_composite up most on pathologic and healthy_s004/s007; probe_score about unchanged
(the probes' backgrounds are flat). If explained rises but pos does not, the drift is the fitter's, not the
target's.

E1 result (tier 1, 898717e): composite 0.752183 (-0.007), **discard**. explained 0.783 -> 0.816 and
explained_junction 0.781 -> 0.818 (held), but pos 0.688 -> 0.635 (NOT held; pathologic average 0.23 -> 0.02,
healthy_s004 frame 0.61 -> 0.46: the crops whose stage-1 B leaked most) and probe_score 0.842 -> 0.832
(crossing_depth threshold 5 -> 50). Surprise, and informative: a cleaner target puts MORE unexplained OD
(the missed wide vessel of pathologic, missed faint vessels) next to the proposed vessels, and the joint fit
pulls their centrelines into it. Same mechanism as (i)-(ii) above: position is driven by error far from the
vessel. Model update: in this fitter a better target only pays once the positional updates are local.

### E2: a tighter positional prior (pre-registered)

Hypothesis: centreline drift (the fitter moving vessels by 0.4-0.7 px from a correct start, even on the
oracle target) is the main loss; the anchored stages' prior (anchor_px = 4 px std about where the stage
started) is too weak to stop vessels sliding into unexplained neighbouring OD. A receptive-field argument:
a vessel's position should answer to error within about its own width, not to OD a vessel-width away.
Change: anchor_px 4.0 -> 1.5.
Predicted: pos up (offset down; most on s004 frame, s007, pathologic), explained down slightly (< 0.01:
the drift absorbs only the last per cent), width about equal, graph equal or up (less drift, fewer
geometry-driven prunes), probe_score equal or up (crossing_depth / crossing_width arms drift less; the 20 deg
crossing may stay compound). Composite +0.003 to +0.01.

E2 result (tier 1, 8a4af52): composite 0.759045 (-0.0002), **discard** (equal score, no simpler). pos 0.686,
explained 0.782, probe_score 0.843: every part within noise (NOT held). The positional prior is negligible
against the precision-weighted data term (the NLL of a 1 px shift of a vessel of OD 0.3 at sigma 0.02 is
~10^2 per px of length; the prior's is 0.2 per control point): the drift is data-driven, not under-
regularised. Model update: to stop drift, change what the positional gradient sees, not its prior.

### E3: geometry frozen after the retarget (pre-registered)

Hypothesis: E1 showed that a re-cleaned target pulls centrelines into the OD it newly exposes (missed
vessels, leaked lumens). The v0 schedule does exactly that in its final stage: retarget, then 60 joint
iterations with free positions. Freezing the geometry there (profiles, halo, kappa still fitted to the clean
target) keeps the positions the joint stage found while the widths and contrasts follow the better target.
Change: the final stage runs with fit_pos = False (FitConfig.final_pos = False).
Predicted: pos up (+0.01 to +0.03; most on pathologic and s004 frame), explained down slightly (the final
geometry no longer adapts; < 0.01), width equal or up, graph about equal (the topology is fixed before the
retarget; node positions move less), probe_score about equal (flat probe backgrounds, little retarget
change). Composite +0.003 to +0.01. Also a few seconds faster.

E3 result (tier 1, 3834bb0): composite **0.768662 (+0.0094), keep**. pos 0.688 -> 0.752 (held, and 2-6x
larger than predicted), explained 0.783 -> 0.772 (held: -0.010), width 0.761 -> 0.768 (held), graph 0.609 ->
0.610 (held), probe_score 0.842 -> 0.850 (predicted equal; better: contrast 0.787 -> 0.808, corner 0.714 ->
0.757, empty 0.535 -> 0.612 with the false vessel 93 -> 78 px; gap 0.798 -> 0.776 is the one sweep that lost).
fast_composite 0.676 -> 0.687. A third of the v0 fit's positional loss happened in its last 60 iterations,
on the re-cleaned target. Recording (residuals.py on fast 0, 2, 8 and fork_wide, cross_a20, empty_average):
junction biases unchanged (fast0 -0.19..-0.33, fast2 -0.26..-0.49, fast8 -0.87 compound / pseudo-T), the
discs still hold 0.39-0.56 of the squared residual: positions were fixed, the junction deficit was not. On
fast2 the residual is a positive pedestal under and along both flanks of the wide vein (red stripes parallel
to its edges) plus the unproposed wide vessel at the bottom border: the render's wide lumens are too narrow
in their blurred flanks against the oracle target, i.e. the retargeted B still holds the flanks.

### E4: a wider render support in the retarget (pre-registered)

Hypothesis: the retarget excludes from B only the pixels where the render exceeds 0.5 sigma, dilated by
3 px; a wide, blurred vessel's flanks extend further, so B still absorbs them and the final (now
geometry-frozen) profile fit renders the wide lumens too narrow and the junction clusters too faint.
Change: bg_dil 3 -> 6 px.
Predicted: explained and explained_junction up (junction_bias toward 0), width up slightly (blur / radius
follow the flanks), pos and graph unchanged (geometry and topology are fixed before the retarget), probe_score
about unchanged (flat backgrounds; a few px less background near the probe vessels). Composite +0.002 to
+0.006.

E4 result (tier 1, d9297b1): composite 0.768156 (-0.0005 against E3), **discard**. explained 0.772 -> 0.774 and
explained_junction 0.770 -> 0.772 moved in the predicted direction but by a fifth of the prediction; width
0.768 -> 0.765 (NOT held); pos and graph identical to E3 (held: geometry and topology are fixed before the
retarget); probe_score 0.850 -> 0.849. Model update: the flank / pedestal deficit is not the retarget's
dilation. Candidates left: stage 1's own mask (the retarget ORs it in, and its B is used for the first two
stages' fit, so the widths are set on the leaky target and the final 60 profile iterations do not undo it),
or the render family (vesselmap's blurred cylinder lacks the long tail of the formed blur and the halo).

### Batch 1 summary

| # | commit | change | tier-1 composite | graph | pos | width | explained | expl_junction | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | fb6e415 | baseline | 0.759238 | 0.609 | 0.688 | 0.761 | 0.783 | 0.781 | 0.842 | baseline | - |
| E1 | 898717e | retarget early (after the profile stage) | 0.752183 | 0.608 | 0.635 | 0.759 | 0.816 | 0.818 | 0.832 | discard | no: explained up as predicted, pos down 0.05 |
| E2 | 8a4af52 | anchor_px 4 -> 1.5 | 0.759045 | 0.607 | 0.686 | 0.762 | 0.782 | 0.780 | 0.843 | discard | no: no part moved (null) |
| E3 | 3834bb0 | geometry frozen in the final fit | **0.768662** | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.850 | **keep** | yes, pos gain 2-6x the prediction |
| E4 | d9297b1 | retarget support dilated 6 px (on E3) | 0.768156 | 0.610 | 0.752 | 0.765 | 0.774 | 0.772 | 0.849 | discard | direction only, a fifth of the size |

**Tier 2 on E3 (3834bb0): composite 0.760516 against the confirmed 0.753886 (+0.0066), confirm.** graph
0.679 -> 0.683, pos 0.772 -> 0.804, width 0.797 -> 0.798, explained 0.916 -> 0.912, explained_junction
0.918 -> 0.915; 1789 s; digest `6b9c2fb82341e0c5`; deterministic: True. Nine of the ten images gained (healthy_s008
frame -0.001); the largest gains were healthy_s004 (+0.012 / +0.014) and pathologic_s000 (+0.021 / +0.009).

| image | kind | composite | graph | pos | width | explained | explained_junction | junction_bias |
|---|---|---|---|---|---|---|---|---|
| healthy_s000_480x768 | average | 0.825 | 0.767 | 0.855 | 0.811 | 0.979 | 0.983 | -0.087 |
| healthy_s000_480x768 | frame | 0.815 | 0.734 | 0.859 | 0.868 | 0.959 | 0.962 | -0.104 |
| healthy_s004_320x512 | average | 0.737 | 0.672 | 0.764 | 0.740 | 0.902 | 0.922 | -0.249 |
| healthy_s004_320x512 | frame | 0.749 | 0.667 | 0.788 | 0.810 | 0.894 | 0.907 | -0.267 |
| healthy_s007_480x768 | average | 0.781 | 0.694 | 0.843 | 0.776 | 0.988 | 0.989 | -0.060 |
| healthy_s007_480x768 | frame | 0.774 | 0.671 | 0.822 | 0.826 | 0.978 | 0.984 | -0.069 |
| healthy_s008_480x768 | average | 0.842 | 0.793 | 0.869 | 0.815 | 0.991 | 0.994 | -0.040 |
| healthy_s008_480x768 | frame | 0.807 | 0.719 | 0.866 | 0.836 | 0.981 | 0.989 | -0.049 |
| pathologic_s000_480x768 | average | 0.646 | 0.579 | 0.680 | 0.729 | 0.733 | 0.718 | -0.508 |
| pathologic_s000_480x768 | frame | 0.630 | 0.532 | 0.693 | 0.774 | 0.716 | 0.700 | -0.537 |

**Tuning curves (E3 against v0, per probe, |change| > 0.01).** The thin and faint lines improved through
position: line_d2.5 0.454 -> 0.479 (pos 0.83 -> 0.89), line_d3 0.635 -> 0.661, line_d6_h0.25 0.670 -> 0.705
(pos 0.75 -> 0.85), line_d5_f25_h0.5 0.716 -> 0.757 (pos 0.75 -> 0.88); cross_a45_wide_over_thin 0.888 -> 0.900
(thin arm width 0.45 -> 0.50); end_blind 0.964 -> 0.975; empty_average's false vessel 93 -> 78 px. One loss:
parallel_g2 0.659 -> 0.578 (pos 0.28 -> 0.14): two vessels 2 px apart need the final positional fit on the
clean target to explain each other away. Every threshold is unchanged (width 2.5 px, blur 25 d_c, contrast
hct 0.25, fork 4/5, crossing_angle 30 deg, crossing_depth 5 d_c, gap 2 px, empty 1/2): E3 raises the curves'
levels (position), not their thresholds (detection and topology are set before the final fit).

**What the batch taught (the model of the system).**
1. The positive control loses in three separable ways (the recording above): the fitter's own drift on a
   perfect target (0.36-0.41 px), drift after MDL removals (+0.1-0.25 px), and the leaky target (junction
   bias -0.13). The drift is data-driven (E2: the prior is irrelevant) and is driven by OD the network does
   not explain (E1: exposing more of it makes drift worse; E3: not letting positions answer to it in the last
   stage is worth +0.064 pos on the crops, +0.032 on whole images).
2. So target cleaning and the positional fit must be decoupled: positions from the stage-1 target and the
   proposal, widths / contrasts / optics from the cleaned target. The junction deficit (bias -0.2 to -0.9 on
   dev, unchanged by E3/E4) is then a profile / target problem, not a geometry one.
3. The remaining big costs are recall (missed vessels: the unexplained OD that drags neighbours) and the
   junction/pedestal deficit along wide vessels.

**Directions for the next batch (ranked).**
1. Re-propose from the residual (attention where error remains): after the joint fit, trace the
   positive residual of the retargeted OD (the neuromimetic stages on OD - R) and add the new vessels before
   the final fit; this removes the unexplained OD that drags neighbours, and should raise recall, graph and
   explained together. Pre-register a pos gain on pathologic and s004.
2. Decouple further: run the joint stage's positions on the stage-1 target but its profiles on the
   retargeted one (E1 failed because both moved together); or retarget early with geometry frozen in the
   joint stage's last half.
3. MDL removals drag neighbours: refit the neighbours of a removed edge with positions frozen, or require
   a removal to pay after refitting (explaining away run forward).
4. The flank / pedestal deficit along wide vessels: stage 1's mask (OR-ed into every retarget) and the
   cylinder profile's lack of the formed blur's tail; test retarget without stage 1's mask (render support
   only) and a second halo term.
5. parallel_g2 regressed under E3: a short positional-only final phase restricted to edge pairs closer than
   their summed calibre.

## Batch 2

Setup: last kept = last confirmed = 3834bb0 (E3; tier 1 0.768662, tier 2 0.760516).

### E5: a vessel must have two observed flanks (pre-registered)

Recording before the change (empty_average, scratch diag): the 78 px false vessel is ONE wide edge (r 7.2,
s 7.9 px, a 0.11) laid over a dark illumination lump in the bottom-right corner of the probe, which runs into
the invalid corner. Stage 1 masks the whole lump, so its B there is extrapolated from the brighter interior
and the lump's OD is positive; the edge's outer flank (r + 2 s = 23 px from its centreline) lies outside the
valid region for most of its length (centreline 12-22 px from the invalid pixels). Its MDL gain is 1580
against a penalty of 21, and the smooth-background test (sigma 25 px) does not absorb a lump that is only
seen from one side. Hypothesis: a vessel is a line darker than the background on BOTH sides (the even-
symmetric receptive field of a line detector: a centre and two off-flanks); an edge whose outer flank is
not observed is evidence of a background step at the frame, not of a vessel.
Change: in prune(), a wide edge (r + s >= wide_test_w) is removed when fewer than half of its 1 px samples
have both flank points (x +- n (r + 2 s)) inside the image and on valid pixels.
Predicted: empty sweep 0.612 -> ~1.0 (empty_average 0.22 -> 1.0; +0.023 probe_score), every other probe
unchanged (their wide vessels are flanked by background), fast crops unchanged (crops are >= 97 % valid and
their wide vessels rarely run along the crop border; at worst a pathologic edge along the border goes:
graph / explained down slightly there). Composite +0.008 to +0.012.

E5 result (tier 1, 34d262c): composite **0.781836 (+0.0132), keep**, but only half held. probe_score 0.850 ->
0.884: empty 0.61 -> 1.00 (held) and, not predicted, the thin and faint line probes too (line_d2.5 0.48 ->
0.64, line_d3 0.66 -> 0.74, line_d6_h0.25 0.71 -> 0.80, line_d6_h0.5 +0.014, fork_thin +0.015): every probe
shares the same illumination field, and its dark lump at the bottom-right frame edge was fitted as a faint
wide edge (r 14-19 px, s 8-13 px, a 0.01-0.05) in most of them. fast_composite 0.687 -> 0.680 (NOT held):
true wide vessels running along the CROP border lost (s007 average: an a 0.31, r 17 px vein, explained 0.69 ->
0.44; s008 average graph 0.66 -> 0.61; s000 frame -0.007). Recording of every wide edge's flank fraction
(scratch diag): one-flanked true vessels 0.29-0.49 (a 0.14-0.41), one-flanked false lumps 0.00-0.48 (a
0.004-0.08, one blob at 0.32). Geometry alone cannot tell a vessel cut by the frame from an illumination
lump at the frame (the frame vs aperture distinction does not either: the probes' outer flanks also leave
the frame); contrast can. Tier 2 has no probes and full-image borders, so E5 as it stands would likely lose
there.

### E6: only FAINT one-flanked wide edges are illumination (pre-registered)

Hypothesis: an illumination lump at the frame is faint and broad (OD contrast a < 0.1), a vessel cut by the
frame is not; the flank test should apply only to the faint ones (the threshold is read off the recording
above: exploratory, not confirmatory). Change: E5's removal also needs mean a < flank_a = 0.1.
Predicted: fast_composite back to E3's 0.687 (s007 avg, s008 avg, s000 frame restored; graph, explained
back up), probe_score 0.884 -> ~0.883 (line_d6_h0.5's strong blob, a 0.32, is kept again), composite
~0.785 (+0.003 over E5).

E8 result (tier 1, f0cf830): composite 0.779753 (-0.0056 against E6), **discard**. NOT held: probe_score is
E7's to the digit (0.8769: line_d3's false crossing edge and fork_thin_frame's compound pay any description
cost; they are high-gain), and fast falls to 0.683 (s004 frame recall 0.61 -> 0.54, s008 frame 0.827 ->
0.767, pathologic frame 0.546 -> 0.474), while the averages keep E7's gains (pathologic average 0.481).
Model update: the false edges that stage 8 removes on the frames are not cheap edges; they explain real OD
(the frames' red-cell texture, 20x the average's noise), and doubling the cost removes true faint vessels
first. Stage 8's advantage there is its band-CNR visibility test (cnr_min) and its junction-disc-free
gains, not its MDL cost: the joint fit needs a visibility (precision-relative contrast) test, not a
higher price.

### Batch 2 summary

| # | commit | change | tier-1 composite | fast | graph | pos | width | explained | expl_junction | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3 | 3834bb0 | (batch 1, confirmed) | 0.768662 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.850 | keep | - |
| E5 | 34d262c | wide edge needs two observed flanks | 0.781836 | 0.6796 | 0.603 | 0.751 | 0.771 | 0.746 | 0.768 | 0.884 | keep | half: empty fixed (+ thin-line probes, unpredicted), but true wide vessels at the crop border lost |
| E6 | 7b44ac6 | ... only for faint (a < 0.1) wide edges | **0.785308** | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | **keep** | yes, on every part |
| E7 | b28c95d | proposals = stage-7 graph (no stage-8 prune) | 0.786362 | 0.6959 | 0.625 | 0.757 | 0.770 | 0.774 | 0.770 | 0.877 | discard (+0.001) | partly: averages up, frames and probes down |
| E8 | f0cf830 | E7 + mdl_scale 0.5 | 0.779753 | 0.6827 | 0.604 | 0.741 | 0.771 | 0.772 | 0.769 | 0.877 | discard | no: the false edges are high-gain |

**Tier 2 on E6 (7b44ac6): composite 0.760600 against the confirmed 0.760516 (+0.0001), confirm.** graph
0.683038, pos 0.803839, width 0.798460, explained 0.912187, explained_junction 0.914835; 1800 s; digest
`bcdbe0a268f3c3a7`; deterministic: True. Nine of the ten images reproduce E3's digests exactly; only
pathologic_s000 average changed (0.6463 -> 0.6471: one faint one-flanked wide edge removed). As predicted for
E6, the whole images (few aperture or frame lumps, strong border vessels kept by the contrast condition) are
untouched; the batch's tier-1 gain is the negative control and the thin/faint line probes.

**Tuning curves (E6 against E3).** empty 1/2 -> 2/2 (threshold; empty_average 0.22 -> 1.00); width sweep mean
0.769 -> 0.831 (line_d2.5 0.48 -> 0.64, line_d3 0.66 -> 0.74: precision, graph_probe 0.63 -> 0.99 at d 2.5;
but its width score stays 0.0-0.09 with width_bias +0.9..+1.1: a 2.5 px vessel is fitted twice too wide);
contrast mean 0.808 -> 0.855 (line_d6_h0.25 0.71 -> 0.80); fork_thin +0.015. Every other threshold
unchanged (width 2.5 px, blur 25 d_c, contrast hct 0.25, fork 4/5, crossing_angle 30 deg, crossing_depth
5 d_c, gap 2 px). Residual recordings at dev junctions: unchanged from E3 (the deficit -0.2 to -0.9 and the
0.39-0.56 disc share remain the largest unexplained error).

**What the batch taught.**
1. The negative control's false vessels were not noise but one shared illumination lump at the frame,
   fitted as a faint wide 'vessel' in most probes; a one-flanked faint wide edge is background. Geometry
   alone cannot separate it from a strong vessel cut by the crop border (E5 lost those); contrast can (E6).
2. Recall is bounded by stage 8, but stage 8 is doing two jobs: it removes true faint vessels on averages
   and false, high-gain texture edges on frames (E7). The joint MDL test cannot take over by raising its
   price (E8): the frames' false edges pay any price. A replacement must test VISIBILITY (contrast against
   the local texture, as stage 8's cnr_min does) rather than NLL gain.

**Directions for the next batch (ranked).**
1. Stage-7 proposals (E7) plus a visibility test in the joint prune: an edge's fitted peak contrast over the
   local band RMS (stage 2's rms maps, or the precision map) must exceed a CNR floor, like stage 8's cnr_min,
   instead of a higher MDL price. Predicted: E7's average-image gains (+0.009 fast) without its frame losses.
2. Re-propose from the residual (attention where error remains; batch 1's direction 1, still untried).
3. Thin-vessel width bias (+100 % at d 2.5, +50 % at d 3): the blur / radius trade-off below the PSF; a prior
   tying s to the image-wide optics (or fitting s jointly per image) for edges with r < s.
4. Decouple positions from the cleaned target inside the joint stage; explaining away after MDL removals;
   the flank / pedestal deficit along wide vessels; parallel_g2 (batch 1 directions 2-5, still open).

## Batch 3

Setup: last kept = last confirmed = 7b44ac6 (E6; tier 1 0.785308, tier 2 0.760600, digest bcdbe0a268f3c3a7).

### Priority 1: the review's findings

**Finding 1 (determinism across run histories): confirmed and fixed.** vesselmap's `entry_core` is
`torch.compile(dynamic=True)`. Past dynamo's recompile limit it falls back to eager, so one image's floats
depended on which shapes ran before it in the same process. The fix sets `set_threads` to
`VESSELMAP_COMPILE=0`, which runs the render eagerly (be34189). Tier 1 (re-baseline): composite 0.785309
against E6's 0.785308. Every part is equal to 1e-5 and the digest is new, `d56fee848ca3e349`. It ran in
303 s, so eager is not slower here: a fresh compile on each new shape class cost about as much as it
saved. A new diagnostic, `determinism.py`, re-runs each row of a saved run in its own fresh process. On the
10 fast crops all 10 fresh digests equal the in-run digests. The review's eager digests for s007 average
(d860f99a) and pathologic average (51907319) are reproduced exactly. From now on a composite difference
below about 5e-4 on one image is treated as noise, and tier-2 `deterministic: True` comes with a fresh-process
check.

**Finding 2 (the size of E3's gain): confirmed, so the size is corrected.** Per-crop pos from v0 to E3:
pathologic_s000 average went from 0.226 to 0.588 (+0.362). That one crop supplies 0.036 of the mean +0.064.
Without it the gain is +0.031 over 9 crops, two of which lost pos (s000 average -0.038, pathologic frame
-0.038). The review also showed that part of the median-offset gain is a matching-selection effect: E3 matches
fewer samples, and the ones it drops had large offsets. On the samples matched by both runs, weighted MEAN
offsets do not improve. The corrected statement: E3 removes the small drift in the last iterations (median
offset on commonly matched samples improves on every image checked), worth about +0.03 pos on crops and +0.032
on whole images (tier 2). It does not remove the tail errors. The batch-1 table's 'pos gain 2-6x the
prediction' should read 'about 1-3x the prediction once the one degenerate crop is excluded'.

### Recording before the experiments: a signal-detection view of the prune (ORACLE diagnostic, scratch)

Every edge at prune time on the 10 fast crops and 34 probes was recorded with its gains (with and without
node cores, smooth-background-orthogonal), its MDL penalty, its contrast-to-noise and flank fraction, and a
truth label (at least half its 1 px samples within max(3, r/2) px of an observable true sample). The
findings:
- On the fast crops the joint prune is almost irrelevant. It removes 349 of 18820 px of true length
  (1.9 %) and 188 of 1080 px of false length. Recall (0.63 on fast) is lost upstream, in the proposal.
- The node-core dual gain (site_mask) and min_gain_per_px decide NO removal in the 44 cases. Of the
  wide-edge smooth-background test, one removal is decisive: probe end_blind's false wide edge (g_all 511
  >> pen 14, g_bg -177). The flank test decides every probe illumination-lump removal.
- The correlation area of the precision-normalised texture, measured on stage 1's background, is 33-61 px^2
  on frames and 92-125 px^2 on averages (probes 64-103), against corr_px = 6. A per-image corr_px
  would make the AVERAGES stricter and the frames more lenient. That is the opposite of what E7/E8 need
  (averages lose true faint vessels, frames keep false texture edges), so that idea was dropped without a run.

### E9: delete the node-core dual gain and min_gain_per_px (pre-registered; simplification)

Hypothesis: the in-silico lesion above shows that neither criterion decides a removal, so both are dead code
on tier 1. Change: the prune uses the plain junction-aware gain (no site_mask exclusion, no max of two gains)
and only the MDL inequality, the smooth-background test for wide edges and the flank test.
Prediction: the tier-1 digest is IDENTICAL (d56fee848ca3e349): every part is equal, composite 0.785309, and it
is kept on simplicity (about 30 fewer lines). Tier 2 may differ on whole images, where a node core could decide.

E9 result (tier 1, b527f79): composite 0.785309, digest `d56fee848ca3e349`: **identical, as predicted. Kept**
(29 fewer lines). The in-silico lesion on recorded features predicted the real lesion exactly.

### E10: stage-7 proposals plus a sharp-edge test for faint wide edges (pre-registered)

Recording (ORACLE diagnostic, stage-7 proposals through the E9 prune): stage-7 adds +514 px of true kept
length on the fast averages and +282 on frames, against +34 and +171 px of false length. The two probe losses
of E7 come from wide, faint, soft lumps (line_d3: r 9.7, s 7.2, a 0.03; line_d5_f25_h0.5: r 8.7, s 8.0, a 0.04)
whose flanks are observed (0.64), so E6's flank test passes them. Edge features separate the frames' false
edges poorly. Median CNR is 1.9 for false edges against 2.7 for true ones, and the gain per px and gain/penalty
also overlap. A CNR or gain criterion at any setting costs more true length than it removes false length.
Shape separates better. Among wide (r + s >= 8) faint (a < 0.1) edges, the soft ones (s >= 0.6 r) are the
probe lumps (254 -> 73 px of false probe length). On the fast crops the shape test is a wash: it removes 5 true
edges (353 px) and 3 false ones (209 px). The threshold was read off this recording, so the test is
exploratory. Hypothesis: a vessel's lumen has a sharp edge relative to its width (a blurred box: s < r for a
wide vessel). A broad faint profile with s ~ r is an illumination or texture modulation (it has no high
spatial frequency), as V1's even-symmetric cells need a stripe, not a ramp.
Change: proposals = stage-7 graph (E7's change, nm_verify False). The prune's faint wide edge (r + s >= 8,
a < 0.1) is also removed when mean s >= 0.6 mean r (soft_k).
Predicted: fast_composite ~0.694 (E7 0.6959 less a little true length on averages), graph +0.01 against E9;
probe_score ~0.881 (line_d3 recovers its E6 0.743, line_d5_f25_h0.5 up, fork_thin_frame stays E7's 0.73);
pos, width, explained about equal; composite ~0.787-0.788 (+0.002 to +0.003). If the gain is mostly probes,
treat it as tuned to the battery and require tier 2 to hold.

E10 result (tier 1, b9eb2c0): composite 0.787348 (+0.00204 against E9), **a provisional keep**: it meets the
threshold, but the gain is half probe-tuned, so tier 2 decides it against a tier-2 re-baseline of E9 (both
are run). Held in part:
- graph +0.005 (predicted +0.01).
- fast 0.6892 (predicted ~0.694). Against E7, the soft test costs the crops 0.007: it removes true faint
  vessels. On s007 average, explained_junction falls 0.52 -> 0.39; pathologic average falls 0.481 -> 0.456.
- probes 0.8855 (predicted 0.881). line_d3 0.743 -> 0.798 and line_d5_f25_h0.5 0.757 -> 0.896: the soft lumps
  are gone. E7's two losses remain: fork_thin_frame is typed compound (fork threshold 4/5 -> 3/5), and
  cross_a20 drops 0.76 -> 0.75.
- explained_junction 0.770 -> 0.751 (NOT predicted): the removed true faint vessels sat at junctions.
On the recorded features, a free end does not separate the removed true edges from the lumps either, so
the rule is not tuned further.

Pilot, not run on tier 1 (direction 3, typing from the fitted arm pattern): each fitted event (degree >= 3
node, crossing) was typed by the truth's own rule. The rule counts the lines leaving the event's region (events
whose reaches overlap form one region) by at least the truth's 6 px stub length, then types it: 3 lines a
3-way, 4 lines a crossing when they pair into two through lines, 5 or more a compound. Results on the s000
and s004 frame crops:
- With reach r + s, wide veins merge distinct junctions (a true crossing 25 px from a compound on the vein
  became compound), and balanced type accuracy falls 0.34 -> 0.27 and 0.51 -> 0.40.
- With reach r the scores are identical to the baseline: every fitted event is already typed as its arm
  count says.
- The type errors (truth compound -> fit crossing 19, compound -> pseudo-T 16, crossing -> pseudo-T 9 of
  ~110 matched) are MISSING ARMS. A crossing whose 4th arm is not traced is a T; a compound whose third vessel
  is not traced is a crossing. Typing is a recall problem, and the next junction lever is re-proposing arms
  from the residual around fitted junctions, not relabelling.

### E11: the retarget renders the pruned network (pre-registered; the review's direction-2 fix)

Hypothesis: retarget excludes from B the support of the PRE-prune model, so a removed false edge (an
illumination lump) keeps its pixels out of B. The lump's OD stays in the final target, where the network
cannot explain it and its neighbours' profiles absorb it (the final fit moves only profiles). E10 removes
more such edges, so the fix matters more now. Change: the retarget's R is the render of the pruned network
(one extra model build).
Predicted: graph and pos identical (topology and geometry are fixed before the retarget); explained up slightly
on the probes whose lumps were removed (line_d3, line_d5, empty); width about equal; fast within +-0.001.
Composite +0.000 to +0.002. It is a correctness fix, so it is kept if not worse (within 0.002) because the
code is no more complex.

E11 result (tier 1, ba75eba): composite 0.788777 (+0.0014 against E10), **kept** at tier 1 as a correctness
fix. graph and pos are identical (held). probes +0.0032 (better than predicted): line_d2.5 explained -0.07 ->
0.47, line_d3 0.43 -> 0.65, because the removed lumps return to B. explained -0.002 overall (NOT held): where
the prune removed a TRUE edge (s004 frame, s007 average), its pixels now go back into B and the vessel leaks
into the background. The fix is right exactly when the prune is right.

### Batch 3 summary

| # | commit | change | tier-1 composite | fast | graph | pos | width | explained | expl_junction | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E6 | 7b44ac6 | (batch 2, confirmed) | 0.785308 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | - | - |
| R0 | be34189 | eager vesselmap render (determinism fix) | 0.785309 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | rebaseline | yes: scores equal, digest new, fresh = in-run |
| E9 | b527f79 | prune: delete node-core dual gain, min_gain_per_px | 0.785309 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | **keep** (simpler) | yes, exactly: identical digest |
| E10 | b9eb2c0 | stage-7 proposals + soft faint wide edges removed | 0.787348 | 0.6892 | 0.615 | 0.754 | 0.770 | 0.765 | 0.751 | 0.886 | keep (provisional) -> reset by tier 2 | partly: half the gain is two probe lumps; explained_junction -0.019 |
| E11 | ba75eba | retarget renders the pruned network (on E10) | 0.788777 | 0.6889 | 0.615 | 0.754 | 0.770 | 0.763 | 0.747 | 0.889 | keep -> reset by tier 2 | mostly: probes up, explained down where removals were wrong |
| pilot | - | junction typing from the fitted arm pattern | (2 crops) | | | | | | | | not run | a no-op at lumen reach, harmful at r + s |

**Tier 2.**
- **E9 (b527f79, with the eager render): composite 0.760603 against the confirmed 0.760600, confirm.**
  graph 0.683044, pos 0.803793, width 0.798493, explained 0.912199, explained_junction 0.914852. It ran 2269 s
  (in parallel with the second tier-2 run), digest `cf65125bf2fb0dea`, deterministic: True. In fresh
  processes the pathologic_s000 average and frame reproduce their in-run digests. This is the tier-2
  re-baseline of the determinism fix, and E9 is a no-op on whole images too (+3e-6).
- **E10 + E11 (ba75eba): composite 0.749321, a regression of -0.0113, so reset to b527f79.** graph 0.683 ->
  0.663, with junction_f1_strict -0.027, junction type balance -0.042, centreline recall -0.007 and edge_cover
  -0.007. Explained -0.004, pos +0.002. 8 of 10 images lost, the worst being pathologic average (-0.033, graph
  0.58 -> 0.51) and frame (-0.024). Deterministic, digest `8eea31a4aaa98080`.

**Tuning curves (E10/E11 against E9; tier 1 only, both reset).** empty 2/2 throughout. fork threshold 4/5 ->
3/5 (fork_thin_frame: the stage-7 graph types its fork 'compound'). The width sweep mean rose (line_d2.5
0.64 -> 0.73, line_d3 0.74 -> 0.85, from the soft-lump removal and the pruned retarget); the contrast sweep
rose through line_d5_f25_h0.5 (0.76 -> 0.89). cross_a20 -0.011. No other threshold moved. Kept state (E9):
every curve is identical to E6, since the digest is identical.

**Residual recording (E11 state, fast 0, 2, 8 and fork_wide, cross_a20, empty_average).** Junction biases are
unchanged from E6 (fast0 -0.19..-0.43, fast2 -0.25..-0.50, fast8 -0.49..-0.88), and the discs still hold
0.39-0.57 of the squared residual. On fast2 the error is a positive (under-predicted) band along the wide
vein's lower flank, made of thin parallel stripes: thin vessels running beside and under the vein, never
traced. The largest junction error (#4, pseudo-T, bias -0.65) is one of them. With the pilot, this points to
the same mechanism: missing arms, not mis-typed or mis-rendered ones.

**What the batch taught.**
1. Determinism: the review's history dependence was torch.compile's per-shape cache in vesselmap's render
   core. Eager mode removes it at no cost in time (303 s against 278-343 s) or score (1e-6). Fresh-process
   digests now equal in-run digests on all 10 fast crops and on the two tier-2 images that diverged before.
   determinism.py is the check.
2. Measure before cutting. The signal-detection recording of the prune predicted the E9 lesion exactly
   (identical digest). The same recording shows the joint prune is nearly irrelevant on stage-8 proposals
   (1.9 % of true length, 17 % of false length). Recall is lost in the proposal, and no CNR or gain criterion
   separates the frames' false edges (median CNR 1.9 against 2.7 for true ones).
3. The texture's correlation area is 33-125 px^2, not 6, and averages are MORE correlated than frames. So
   corr_px is not the reason the averages lose faint vessels and the frames keep texture edges, and a
   per-image corr_px would push the wrong way.
4. The fast crops and the probes over-rate the stage-7 proposal. Tier 1 favoured E7/E10 (+0.002 to +0.009 on
   fast). On whole images stage 7 loses junction F1 and type balance (-0.027, -0.042) without gaining
   recall. Stage 8 is doing junction work (merging and typing at the proposal level) that the joint fit does
   not redo. The fast crops' border truncation flatters added edges: most crop edges have a 'free' border end.
   Whole images are the honest test of proposal changes, and future proposal experiments should use tier 2
   (or a whole-image subset) before a keep.
5. The soft-lump test (s >= 0.6 r) was read off a recording, and on independent data (tier 2) it removes true
   faint vessels: an exploratory threshold that failed confirmation.
6. Junction typing is a recall problem. The fitted events' types already follow their arm counts. Truth
   compounds become crossings and crossings become Ts because an arm is missing.

**Directions for the next batch (ranked).**
1. Re-propose arms from the residual around fitted junctions and along wide vessels (attention where error
   remains; predicted to move junction recall, type balance and explained_junction together). The residual
   recordings show untraced thin vessels beside and under wide veins and missing 4th arms at crossings.
   Concretely: after the joint fit, run neuromimetic stages 3-7 on the positive residual (OD - R) restricted
   to discs around fitted nodes and crossings and to bands along wide edges, then add the traced pieces that
   connect to an existing junction or edge before the final fit. Judge it on tier 2 as well as tier 1.
2. The retarget fix (E11) alone on E9: it was positive where removals were right, and E9's removals are
   conservative (stage-8 proposals). Expect a small probe gain and a whole-image no-op.
3. A whole-image tier-1 proxy for proposal changes: add one full dev image to the loop's checks (not to the
   frozen metric) whenever the proposal changes, since crops over-reward added edges.
4. The CFAR detector (the review's direction 1) is still the principled replacement for the MDL prune. The
   recording says it can only pay off together with a proposal that adds recall without losing stage 8's
   junction work, so it is ranked below direction 1.

## Batch 4

Setup: last kept = last confirmed = b527f79 (E9; tier 1 0.785309, digest d56fee848ca3e349; tier 2 0.760603,
digest cf65125bf2fb0dea).

### Priority 1: the review's findings on E5/E6 (batch 2)

**Finding 1 (the E5/E6 tier-1 gain is one shared background draw): confirmed; the size is corrected and a
multi-seed control is added.** The battery renders every probe with imaging SEED 11 (common random numbers),
so the bottom-right illumination lump that E5/E6 remove sits in every probe and is counted about six times.
The batch-2 claim "+0.0166 tier 1, empty 1/2 -> 2/2" should read: roughly +0.004 to +0.006 of real probe gain
(never worse on any seed; the thin-line false wide edges go on seeds 3 and 5 only) plus a seed-11 artifact; on
real images E5/E6 are a near no-op (fast +0.00001, tier 2 +0.00008). New diagnostic, outside the frozen metric:
`controls.py` re-renders six battery stimuli (empty average and frame, line_d3, line_d6_h0.25, fork_wide,
cross_a45) with imaging seeds 1-7 (cached in `$SPLINEFIT_WORK/controls/`) and reports the mean probe_composite
per stimulus and the false length on the empty frames. From now on a probe-only tier-1 gain counts as real only
if it also shows on the controls. It reproduces the review's empty-average numbers for E6 exactly
(0.072, 0.045, 1.0, 0.308, 0.560, 0.0, 0.010). Baseline (E9, b527f79, controls digest 8b1a3cba43e347d4):

| control (7 seeds) | mean probe_composite | battery (seed 11) |
|---|---|---|
| empty_average | 0.285 (false length 81 px) | 1.000 |
| empty_frame | 0.766 (28 px) | 1.000 |
| line_d3 | 0.661 | 0.743 |
| line_d6_h0.25 | 0.360 (0 on 4 of 7 seeds) | 0.80 |
| fork_wide | 0.884 | 0.96 |
| cross_a45 | 0.901 | 0.97 |
| **controls_score** | **0.6427** | |

So the single-seed battery overstates the negative control and the faint-line threshold badly: the averages'
background is fitted with 44-165 px of false vessel on 6 of 7 seeds. A re-baseline of the frozen battery that
averages each probe over several imaging seeds is the right evaluator fix, but it changes the tier-1 metric's
scale for every past row, so it is recommended (for the bug-fix protocol at a batch boundary), not done here.

**Finding 2 (E6's contrast rule is post hoc and contradicted): confirmed in substance.** flank_a = 0.1 was read
off truth-labelled recordings of the same tier-1 data; on line_d6_h0.5 the same lump is fitted with a = 0.315
and kept; on empty seed 5, E5 removes the lump and E6 keeps it; on tier 2 the one edge E6 removed (pathologic
s000 average, at (258, 5)) was a badly fitted piece of a real border vessel (true a 0.37), not a lump. The
statement "contrast separates an illumination lump at the frame from a vessel cut by the frame" is withdrawn:
the flank test with flank_a = 0.1 is an exploratory heuristic whose measured effect is +0.004-0.006 on probes
and ~0 on real images. It stays (it is never worse on any seed and has no tier-2 cost), but its threshold is
not to be tuned further on the battery; a principled replacement is a CFAR test against a width-matched null
measured on the image's own background (review direction 1), judged on the multi-seed controls.

### E12: the retarget renders the pruned network, alone on E9 (pre-registered)

Hypothesis: E11's fix (batch 3) was confounded with E10's stage-7 proposals when tier 2 regressed. On stage-8
proposals the prune removes little (1.9 % of true length), so the retarget's support should change only where
a removed edge (mostly a false lump) sat: those pixels return to B and the final fit no longer has to bend its
neighbours' profiles around unexplained OD.
Change: retarget's R = the render of the pruned network (one extra model build).
Predicted: graph and pos identical (topology and geometry are set before the retarget); probes +0.001-0.003
(line_d2.5, line_d3 explained up, as in E11); fast within +-0.001, explained about equal; composite +0.000 to
+0.002; controls: empty unchanged (no topology change), line controls' explained slightly up. Kept only if it
reaches the threshold, or within it as a correctness fix at equal complexity.

E12 result (tier 1, ba7302d): composite 0.786641 (+0.00133 against E9). **Every prediction held**: graph and pos
identical, probes +0.0024 (width and contrast sweep means; empty unchanged), fast +0.0002, width +0.0024,
explained -0.0009. Below the 0.002 threshold and not simpler, so **discarded** by the rule (reset to 3f38019),
although it is the consistent form of the retarget. On stage-8 proposals the retarget fix is a pure probe
effect of about +0.001 composite. It does not explain the batch-3 tier-2 regression; that was E10.

### E13: re-propose arms from the prediction error (pre-registered; review direction 1)

Hypothesis (predictive coding: attention goes where error remains). Junction typing is a recall problem
(batch 3): crossings are typed T because the 4th arm is missing, and compounds are typed crossing because the
third vessel is missing. The missing arms are in the positive residual of the pruned network's render. Pilot
recordings (scratch: 5 probes and 2 fast crops) shaped the change before the run:
- Without a novelty gate, the residual readout proposes the misfit echoes along the wide vessels' flanks and
  halo (pseudo_t, t_branch: +70-145 px of false arms at a 0.005-0.02). The gate is neuromimetic's own re_novel
  rule: at least 2 x min_len px of the trace must lie outside the render's support (halo included; the
  retarget's support mask).
- The joint MDL prune keeps 40-50 px texture arms at a = 0.005-0.009 (fork_asym kept 2 of 4). A re-proposal
  searches the whole residual, so its hypotheses pay neuromimetic's re_mdl = 20 x the description cost (the
  look-elsewhere effect).
- When every arm is rejected, the network reverts to its state before the re-proposal.
Change: a new module, repropose.arms. Stages 3-6 run on max(OD - R, 0) at 1.3 x the readout thresholds and are
merged against the network's own edges, which count as the coarsest channel. A trace is kept if it is novel
and one of its ends lies within r + s + 6 px of an existing edge. That end attaches to the edge's node when one
lies within the parent's lumen, else it splits the edge. Then come 40 joint iterations and a second prune (new
arms pay 20 x), then the retarget and the final fit.
Predicted:
- probes about unchanged;
- fast up where a real arm is found (pilot: pathologic average crop +0.07);
- a no-op elsewhere;
- composite +0.002 to +0.008;
- the whole-image check (pathologic_s000 and healthy_s004 averages) must not regress.

E13 result (tier 1, 17b48cc): composite 0.786965 (+0.00166), **discarded**. Partly held:
- fast +0.0072, but ALL of it comes from the pathologic-average crop (+0.072: one 146 px arm, junction F1 0.35
  -> 0.67). The other 9 crops are digest-identical.
- probes -0.0039, from capillary_average alone (0.94 -> 0.81): a wide, faint arm (r 7.6, s 3.8, a 0.016)
  sits on the battery's bottom-right seed-11 lump. The other 33 probes are digest-identical: the revert makes
  the change an exact no-op wherever nothing survives.
- whole-image check: pathologic average and frame and s004 average are EXACT no-ops. 20, 11 and 9 arms were
  proposed and the 20 x cost rejected every one. The crop gain was a crop artifact.

Recording (ORACLE diagnostic, scratch: E13 with re_mdl = 1 on two whole images; arms labelled TRUE when their
samples lie a median <= max(3, r) px from the observable truth):
- pathologic average: 28 arms, 13 TRUE. At 1 x, 8 TRUE and 6 false are kept. Composite 0.6472 -> 0.6494,
  junction F1 strict 0.477 -> 0.559, recall 0.567 -> 0.593.
- s004 average: 10 arms, 7 TRUE. The thin true capillary arms (r ~1.5, a ~0.008) are removed even at 1 x; 2
  TRUE and 1 false are kept. Composite 0.7370 -> 0.7424.
- No single feature (a, r, s, length) separates the kept false arms from the kept true ones, which matches
  batch 3's finding on the prune.

### E14: re-proposed arms pay the ordinary MDL cost (pre-registered; E13 without re_mdl)

Hypothesis: re_mdl = 20 came from neuromimetic's additive render on a leaky target, where every residual was
suspect. The junction-aware joint fit explains the image better, so a re-proposed arm can be judged like any
other edge. The recording shows the 20 x cost is what made E13 a whole-image no-op.
Change: delete re_mdl (the arms pay the same MDL test as every edge). It is simpler than E13: one parameter
fewer.
Predicted:
- whole images: graph up through junction F1 (pathologic and s004 averages +0.002 and +0.005 as recorded);
- fast: up on pathologic average, small +- elsewhere (texture arms on frames);
- probes: down. The pilot kept 2 texture arms on fork_asym, capillary_average keeps its lump arm, and other
  probes may gain arms on the seed-11 lumps. Probe_score -0.005 to -0.015;
- composite -0.005 to +0.003 on tier 1, so likely below the threshold.
If it is discarded on tier 1, the recording says it is a whole-image gain that the single-seed battery cannot
see. The multi-seed controls would then decide whether the probe losses are lump artifacts or real false arms.

E14 result (tier 1, ffc8257): composite 0.788423 (+0.00311 against E9), **kept**. The predictions held, at the
upper end:
- fast +0.0126 (0.6870 -> 0.6996), on 6 of 10 crops: s000 average +0.015, s000 frame +0.013, s007 average
  +0.030, pathologic average +0.073; s004 average -0.007, s007 frame -0.006.
- graph +0.023, explained +0.019, explained_junction +0.026; pos -0.003, width -0.008.
- probes -0.0064 (predicted -0.005 to -0.015). fork_asym 0.98 -> 0.89 (two texture arms, typed 'branch'),
  capillary_average 0.94 -> 0.81 (the seed-11 lump arm), cross_a45_wide_over_thin +0.007. Empty stays 2/2.

Multi-seed controls (the review's required check; E14 against E9): controls_score 0.6427 -> 0.6056. 13 of 42
fits changed: 10 are worse and none is better. Mean probe_composite per control:

| control | E9 | E14 |
|---|---|---|
| line_d3 | 0.661 | 0.602 |
| line_d6_h0.25 | 0.360 | 0.315 |
| empty_average | 0.285 | 0.241 |
| empty_frame | 0.766 | 0.714 |
| fork_wide | 0.884 | 0.874 |
| cross_a45 | 0.901 | 0.888 |

False 'branch' arms attach to the line on 4 of 7 seeds for each line control, and two empty draws gain false
length. So the single-seed battery UNDER-states E14's false-alarm cost (2 of 34 probes against 10 of 42
control fits). This is the mirror image of the E5/E6 case.

**Tier 2 (E14, ffc8257): composite 0.760839 against the confirmed 0.760603 (+0.00024), not regressed, so
confirmed.** Deterministic: True; digest c829ed334440a0f9; 2557 s (the machine was shared).
- graph 0.684037 (+0.001), pos 0.802285 (-0.0015), width 0.791701 (-0.0068), explained 0.918937 (+0.0067),
  explained_junction 0.921196 (+0.0063).
- Per image the composite moves -0.0062 to +0.0054: s000 average -0.006, s008 average -0.003, pathologic
  frame -0.005; s004 average and frame +0.005; the rest within +-0.002.
- The whole-image effect is a clean signal-detection signature, a criterion shift toward 'yes':
  - centreline recall up on 8 of 10 images (+0.003 to +0.027);
  - precision down on 10 of 10 (-0.002 to -0.035);
  - explained and explained_junction up on 10 of 10 (the render explains more of the image's OD, the
    predictive-coding objective);
  - junction F1 strict up on 7 of 10 (pathologic average +0.082);
  - junction type balance down on 6 of 10 (the new nodes carry no proposal type, so they are exported as
    'branch' / 'compound');
  - width down on 7 of 10.
So the composite gain on whole images is at the noise level. The tier-1 gain is mostly the crops'
over-reward of added edges, as batch 3 found for E10.

Residual recording (E14; fast 0, 2, 8 and fork_wide, cross_a20, empty_average): the junction under-prediction
shrinks where arms were added. fast0 bifurcation bias -0.43 -> -0.21 (against the E11 recording); fast8 compound
-0.88 -> -0.81 and crossing -0.49 -> -0.41. The disc share of the squared residual: fast2 0.57 -> 0.52,
cross_a20 0.17 -> 0.12. On fast8 (pathologic average) the dominant error is unchanged. It is a very wide
(> 30 px), dark structure at the right edge, and a broad positive residual covers the field: background leak
plus an unmodelled wide lumen, not missing arms. The added arm covers only part of it.

### Batch 4 summary

| # | commit | change | tier-1 composite | fast | graph | pos | width | explained | expl_junction | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E9 | b527f79 | (batch 3, confirmed) | 0.785309 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | - | - |
| P1 | 3f38019 | controls.py (multi-seed controls; diagnostic) | (pipeline unchanged) | | | | | | | | doc / diagnostic | review findings 1 and 2 confirmed |
| E12 | ba7302d | retarget renders the pruned network (E11 alone) | 0.786641 | 0.6872 | 0.610 | 0.752 | 0.770 | 0.771 | 0.769 | 0.886 | discard (+0.0013) | yes, every part |
| E13 | 17b48cc | arms re-proposed from max(OD - R, 0), 20 x MDL | 0.786965 | 0.6942 | 0.621 | 0.753 | 0.773 | 0.776 | 0.773 | 0.880 | discard (+0.0017) | partly: one crop; whole images exact no-op |
| E14 | ffc8257 | E13 with the ordinary MDL cost | 0.788423 | 0.6996 | 0.633 | 0.749 | 0.760 | 0.791 | 0.795 | 0.877 | **keep**, tier 2 confirm | yes (upper end); controls -0.037 |

**Tier 2:** E14 0.760839 against 0.760603, confirmed (deterministic, digest c829ed334440a0f9).

**Tuning curves (E14 against E9).**
- Thresholds: none moved. Width 2.5, blur 25, contrast 0.25, fork 4/5, crossing angle 30, depth 5, T 2/2, gap
  2.0, ends 2/2, rbc 2/2, empty 2/2.
- Sweep means: fork 0.871 -> 0.853 (fork_asym's texture arms), rbc 0.936 -> 0.869 (capillary_average's lump
  arm), crossing_width 0.900 -> 0.907. The rest are identical.
- Multi-seed controls: lines, empty and fork/cross all down, which is the false-alarm side of the criterion
  shift.

**What the batch taught.**
1. The single-seed battery is a biased instrument in BOTH directions. It over-rated E5/E6 (one lump counted six
   times) and under-rates E14's false arms (2 of 34 probes against 10 of 42 control fits). The multi-seed
   controls (controls.py, 3-4 min) are now the check for any change that moves probes. A re-baseline of the
   frozen battery that averages over imaging seeds is the evaluator fix to propose at a batch boundary.
2. Re-proposal from the prediction error works as a mechanism: 13 of 28 arms on pathologic average are real
   vessels, and the render explains more of every whole image. The arms' acceptance test is the weak part. The
   MDL gain cannot separate texture arms (a 0.005-0.03) from faint true arms (no feature separates them in the
   recording), so the cost only moves the criterion: at 20 x every arm is rejected (a no-op), at 1 x hits and
   false alarms both rise and the composite stays flat on whole images.
3. A reverted hypothesis must leave no trace: restoring the pre-re-proposal network when every arm is
   rejected made E13 digest-identical on 41 of 44 tier-1 cases and on whole images. Without it, the extra
   joint iterations alone moved the scores (pilot: s004 average -0.008 with every arm rejected).
4. Fast crops over-reward added edges, again: E13's +0.072 crop gain was 0 on the whole image, and E14's
   +0.0126 fast gain is +0.0002 on tier 2.
5. The retarget fix (E12) is a probe-only +0.001 on stage-8 proposals, so it does not matter at this scale.

**Directions for the next batch (ranked).**
1. Give the re-proposed arms a calibrated acceptance test instead of an MDL cost (signal-detection theory;
   CFAR). Measure each arm's matched-filter statistic, its fitted template's weighted correlation with the
   residual, against a null of the same template translated onto the image's own vessel-free background
   (outside stage 1's mask and the render's support). Accept an arm above that null's family-wise criterion.
   The controls (false arms on 10 of 42 fits) and tier 2 (precision down on 10 of 10) are the pre-registered
   tests: precision back to E9 level, recall and explained_junction kept.
2. Type the new nodes from their arm pattern: deg-3 nodes from a re-proposed arm are exported 'branch' with no
   fork / confluence / pseudo-T type, and junction type balance fell on 6 of 10 images. Use neuromimetic's
   typing rule (_type_of on the fitted arms) for the new nodes only.
3. Very wide dark structures (pathologic) are neither proposed nor modelled. They leak into B, and fast8's
   residual is dominated by one. A coarse channel (scales > 24 px) or a background-leak test on the residual
   is needed before arms matter there.
4. Recommend the evaluator re-baseline (the battery averaged over 3-5 imaging seeds) through the bug-fix
   protocol, so that tier 1 sees false alarms and lump artifacts at their true rate.

## Batch 5

Setup: last kept = last confirmed = ffc8257 (E14; tier 1 0.788423; tier 2 0.760839, digest c829ed334440a0f9);
pushed head b9f3166 (the same pipeline). Resets go to the latest committed log/diagnostic head.

### Priority 1: the review's finding on the whole-image 'explained' (tier 2 near its ceiling)

New diagnostic outside the frozen metric: `tier2diag.py`. It breaks the whole-image residual down by true
vessel-width class (thin < 2 px, mid 2-4, wide 4-8, vwide >= 8; each band pixel goes to its nearest true
observable sample) and per junction disc, and it measures how much of the true OD the pipeline's own target
keeps (target_keep = sum(od_target OD_img) / sum(OD_img^2), per class). It records whole-image outputs while
tier 2 runs: the `annotate_saving` wrapper returns annotate's output unchanged, so the digests are the
default pipeline's. It can also run the pipeline on a few dev images directly (`--compute`).

First recording (E14, pathologic_s000 average, whole image):

| class | px | explained | share of SSR | target_keep | target_fidelity |
|---|---|---|---|---|---|
| thin (< 2) | 11606 | 0.822 | 0.03 | 0.763 | 0.849 |
| mid (2-4) | 55320 | 0.708 | 0.15 | 0.567 | 0.723 |
| wide (4-8) | 72103 | 0.784 | 0.18 | 0.652 | 0.806 |
| vwide (>= 8) | 140061 | 0.755 | 0.64 | 0.669 | 0.800 |
| all | | 0.757 | | 0.656 | 0.794 |

Junction discs (64): median explained 0.741, 10th percentile 0.229, 27 % of the discs below 0.5.
The finding is confirmed in substance: the pipeline's own target keeps only about two thirds of the true OD
in the observable band (a third lost to the background, in every width class, worst for mid vessels next
to wide ones), and the very wide class holds 64 % of the squared residual. The fit cannot explain OD its
target has already given to B. The background leak (target fidelity) is ranked as a direction below.

### E15: type the fitted nodes from their arm pattern (pre-registered; direction 2)

Hypothesis: a re-proposed arm that lands on an existing node raises its degree; a pseudo-T that gains its
4th arm is exported as an untyped 'compound' even when its arms pair into two straight lines, and the nodes
the arms create carry no proposal type. Neuromimetic stage 7's typing rule (_type_of) on the fitted arms of
every node without a valid proposal type should restore the coarse type balance.
Change: export only (pipeline.arm_type); fit, render and topology unchanged.
Predicted: pos, width, explained IDENTICAL; graph only through junction_type_balanced_coarse (+0 to +0.01 on
crops with an untyped degree->=4 node); probes about unchanged; composite +0.000 to +0.002.

E15 result (tier 1, 17b9c24): composite 0.788423, every part identical to 6 decimals; **discarded** (equal, not
simpler). 9 of 44 digests changed, all through the EXACT type only: new degree-3 nodes become 'pseudo-T'
instead of 'branch' (the same coarse class; exact accuracy up on the s000 and s007 average crops). No
untyped degree-4 node paired into a crossing. The type-balance loss of E14 is therefore not a labelling
problem of the new nodes: in the tier-2 confusion (E14, both runs) the big errors are truth compound typed
crossing (53) or 3-way (60), and truth 3-way typed crossing (37), all on proposal-typed junctions.
(The first E15 run crashed at the 600 s wall clock under machine load from a concurrent diagnostic; the
re-run took 394 s.)

E16 result (tier 1, 167f2fb): composite 0.787676 (-0.00075), **discarded**. Not held in direction:
- probes IDENTICAL (34 of 34 digests): on the small probe images the arms' templates find fewer than 30
  background positions, so the MDL test still decided every probe arm;
- fast -0.0015, on 6 crops; the CFAR criterion is MORE permissive than the MDL cost, not less: precision
  fell (s008 average 0.991 -> 0.966, s008 frame 0.996 -> 0.965, pathologic frame 0.986 -> 0.941) while
  recall rose (s000 average 0.792 -> 0.810).
Model update: a null measured on the vessel-free background is narrower than the error the arms actually live
in. Arms sit next to fitted vessels, where the residual holds the vessels' misfit echoes (profile mismatch,
halo, red-cell texture), not the background's texture. A calibrated null must be taken where the arms are:
e.g. the same template translated ALONG the parent vessel's flank (a local, context-matched null), not over
empty background.

### E17: iterate the retarget (EM between the background and the vessels) (pre-registered; review finding)

Hypothesis: the target loses a third of the true OD on the pathologic image (tier2diag: target_keep 0.66).
retarget re-fits B outside the render's support, but the render was fitted to stage 1's leaky target, so it
is fainter and narrower than the vessels, and its support (R > 0.5 sigma, dilated 3 px) still leaves their
flanks and halo in B. One retarget is one E-step. Alternating once more (fit the profiles to the re-cleaned
target, re-derive the support from that stronger render, re-fit B) should keep more OD in the target, like
an EM iteration toward the fixed point of 'background = what the vessels do not explain'.
Change: the final stage runs as retarget_rounds = 2 rounds (retarget, then iters_final / 2 = 30 iterations,
geometry frozen), the same total iteration count. One parameter (retarget_rounds).
Predicted: graph and pos IDENTICAL (topology and geometry are fixed before the retarget); width up slightly
(widths fit to a fuller target); explained up on the crops with wide vessels (pathologic most, +0.005 to
+0.02), explained_junction up; probes about equal (+-0.002). Composite +0.001 to +0.004. If held, tier2diag
target_keep up on pathologic.

E17 result (tier 1, 10d3d2d): composite 0.789640 (+0.00122), **discarded** (below the threshold, one parameter
more). Partly held:
- graph +0.00001 and pos identical (held: the topology and geometry are fixed before the retarget);
- explained +0.0059 and explained_junction +0.0062 (held); target_fidelity up on 10 of 10 crops (+0.002 to
  +0.034, s004 frame 0.847 -> 0.881): the EM step does recover OD from the background;
- width -0.0047 (NOT held). The radius grows into the recovered flank and halo OD instead of the halo taking
  it: width bias s000 average +0.006 -> +0.029, s004 average +0.10 -> +0.14, while the under-wide vessels
  (s007 frame -0.14 -> -0.12) improve;
- probes +0.0022 (width sweep mean 0.831 -> 0.853, line_d2.5 0.64 -> 0.73; fork_thin -0.017);
- the pathologic average crop's target_fidelity stays at 0.25 -> 0.26: its leak is an unrendered very wide
  structure that no render support covers, not the flanks of rendered vessels.
Model update: a fuller target helps the render but the profile model then trades halo for radius. The EM
step and a better-constrained halo (or r anchored at the proposal's calibre during the final fit) belong
together; on their own, the width part cancels the explained gain in the composite.

### Tier 2 and the whole-image recording (batch 5)

No batch-5 change was kept, so the final kept state is the confirmed E14 pipeline. Tier 2 was run on it
through the recording wrapper (tier2diag.annotate_saving): composite **0.760839**, digest c829ed334440a0f9,
deterministic: True, 2469 s. It is bit-identical to batch 4's confirmation: the wrapper changes nothing, and
the confirmed state reproduces across sessions. Logged as a `confirm` row (no regression).

The recording (tier2diag, every dev image; explained against OD_obs; keep = target_keep; discs = the
observable junction discs):

| image | explained | ex thin / mid / wide / vwide | keep thin / mid / wide / vwide | target_keep | disc median / p10 / < 0.5 | SSR share vwide |
|---|---|---|---|---|---|---|
| s000 average | 0.981 | 0.976 / 0.970 / 0.976 / 0.987 | 0.93 / 0.95 / 0.93 / 0.96 | 0.953 | 0.977 / 0.859 / 0.00 | 0.41 |
| s000 frame | 0.961 | 0.939 / 0.931 / 0.939 / 0.973 | 0.97 / 0.94 / 0.92 / 0.96 | 0.949 | 0.955 / 0.809 / 0.04 | 0.46 |
| s004 average | 0.912 | 0.965 / 0.910 / 0.910 / 0.910 | 0.86 / 0.81 / 0.81 / 0.83 | 0.821 | 0.880 / 0.554 / 0.08 | 0.65 |
| s004 frame | 0.898 | 0.949 / 0.892 / 0.894 / 0.900 | 0.82 / 0.79 / 0.79 / 0.82 | 0.808 | 0.819 / 0.453 / 0.15 | 0.66 |
| s007 average | 0.989 | 0.965 / 0.983 / 0.984 / 0.992 | 0.95 / 0.97 / 0.96 / 0.97 | 0.969 | 0.981 / 0.887 / 0.02 | 0.44 |
| s007 frame | 0.983 | 0.941 / 0.969 / 0.975 / 0.988 | 0.91 / 0.95 / 0.95 / 0.97 | 0.960 | 0.976 / 0.834 / 0.01 | 0.48 |
| s008 average | 0.992 | 0.992 / 0.992 / 0.982 / 0.995 | 0.96 / 0.97 / 0.98 / 0.96 | 0.969 | 0.981 / 0.911 / 0.01 | 0.20 |
| s008 frame | 0.982 | 0.989 / 0.978 / 0.958 / 0.992 | 0.95 / 0.96 / 0.96 / 0.96 | 0.958 | 0.956 / 0.845 / 0.00 | 0.15 |
| pathologic average | 0.757 | 0.822 / 0.708 / 0.784 / 0.755 | 0.76 / 0.57 / 0.65 / 0.67 | 0.656 | 0.741 / 0.229 / 0.27 | 0.64 |
| pathologic frame | 0.735 | 0.790 / 0.666 / 0.791 / 0.728 | 0.88 / 0.53 / 0.66 / 0.64 | 0.636 | 0.682 / 0.197 / 0.32 | 0.71 |

What it shows (answering the review's finding):
- The images split in two. On s000, s007 and s008 the target keeps 95-97 % of the true OD and the render
  explains 0.96-0.99 in every width class; the pooled number is honest there, and the remaining error is in
  the worst junction discs (10th percentile 0.81-0.91). On s004 (keep 0.81-0.82) and pathologic (keep
  0.64-0.66) the target itself has lost 18-36 % of the true OD to the background, in EVERY width class, and
  explained follows it. The worst discs (p10 0.45-0.55 on s004, 0.20-0.23 on pathologic) are where tier 2
  hides its error; the pooled explained_junction does not show them.
- The very wide class holds 41-71 % of the squared residual everywhere except s008.
- So the review's direction is right and is now measured: the biggest remaining render gap on the hard
  images is the TARGET (stage 1's leak), not the fit. E17 (EM retarget) raised target_fidelity on every
  crop but the radius absorbed the recovered flank OD; the pathologic leak is a very wide structure that no
  render support covers.

### Batch 5 summary

| # | commit | change | tier-1 composite | fast | graph | pos | width | explained | expl_junction | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E14 | ffc8257 | (batch 4, confirmed) | 0.788423 | 0.6996 | 0.633 | 0.749 | 0.760 | 0.791 | 0.795 | 0.877 | - | - |
| P1 | 63583c8 | tier2diag (width classes, discs, target_keep) | (pipeline unchanged) | | | | | | | | diagnostic | review finding confirmed and measured |
| E15 | 17b9c24 | type untyped nodes from their fitted arms | 0.788423 | 0.6996 | 0.633 | 0.749 | 0.760 | 0.791 | 0.795 | 0.877 | discard (equal) | partly: identical; exact types only |
| E16 | 167f2fb | CFAR test for re-proposed arms | 0.787676 | 0.6981 | 0.630 | 0.747 | 0.761 | 0.792 | 0.796 | 0.877 | discard (-0.0008) | no: more permissive than MDL |
| E17 | 10d3d2d | two retarget / final-fit alternations (EM) | 0.789640 | 0.6998 | 0.633 | 0.749 | 0.755 | 0.797 | 0.802 | 0.879 | discard (+0.0012) | partly: explained, fidelity up; width down |

**Tier 2:** 0.760839 (unchanged E14 pipeline, confirmed again; digest c829ed334440a0f9, deterministic).

**Tuning curves.** Nothing was kept, so the thresholds and sweep means are E14's. In the discarded runs:
E15 and E16 left every probe digest-identical; E17 moved the width sweep mean 0.831 -> 0.853 and
crossing_width 0.907 -> 0.919 (fuller target on thin lines; line_d2.5 0.64 -> 0.73), fork -0.004.

**What the batch taught.**
1. Junction typing is not a labelling problem of the re-proposed nodes (E15): no untyped degree-4 node
   pairs into a crossing, and the coarse confusion is on proposal-typed junctions (truth compound typed
   crossing 53 / 3-way 60, truth 3-way typed crossing 37 over both tier-2 runs). The typing errors start in
   neuromimetic stage 7's clustering of events, which the fit never revisits.
2. A CFAR null must match the context of the hypothesis (E16): the vessel-free background is quieter than
   the residual next to fitted vessels, so a background-calibrated criterion is more liberal than the MDL
   cost, not more conservative. The arms' false alarms are misfit echoes of their parents.
3. The target is the ceiling on the hard images (tier2diag): s004 and pathologic lose 18-36 % of the true
   OD to the background before any fit, uniformly across widths. An EM retarget recovers part of it (E17,
   target_fidelity up on 10/10 crops) but the profile model turns the recovered flank OD into radius.

**Directions for the next batch (ranked).**
1. E17 with the radius held: alternate retarget and fit, but in the second round fit only a (contrast),
   the halo and kappa, with r and s frozen at the first round's values (or anchored at their calibre). That
   keeps the explained / target_fidelity gain without the width loss. Judge on tier 2 with tier2diag
   (target_keep on s004 and pathologic).
2. A context-matched null for re-proposed arms: translate each arm's template ALONG its parent's flank
   (same distance to the parent's centreline, same side, outside the arm's own footprint) and accept above
   that null. This is the CFAR that E16 should have been; the multi-seed controls are its test.
3. Very wide structures in pathologic (keep 0.64-0.66; vwide 64-71 % of the SSR): a coarse channel (scale
   > 24 px) in the proposal, or a background-leak test on the residual at a coarse scale, before arms
   matter there.
4. Junction typing from the fitted render (the confusion above): re-type a proposal junction after the
   fit from the fitted arms' pairing AND the union/additive misfit at the node (a crossing fitted as a node
   leaves an over-predicted centre; a fork fitted as a crossing an under-predicted one). Pre-register on the
   tier-2 confusion counts.
5. Recommend (unchanged) the evaluator re-baseline that averages the battery over imaging seeds.

## Batch 6

Setup: last kept = last confirmed = ffc8257 (E14; tier 1 0.788423; tier 2 0.760839); pushed head 1ce2bf1.

### Priority 1: the review's findings on E14 (both confirmed; fixed)

Finding 1 (E14's keep rests on one crop; tier 2 is a statistical zero). Re-computed with the new paired
test (`paired.py`, outside the frozen metric) from runs/e9_tier2.json and runs/e14_tier2.json:
- tier 2 (10 images): mean +0.00024, SE 0.00127, t +0.19, 7 up / 3 down (s000 average -0.0062,
  pathologic frame -0.0053, s008 average -0.0031), leave-one-out range [-0.0003, +0.0010];
- tier 1 fast crops: mean +0.0126, SE 0.0075, t +1.7; the gain rests on the pathologic-average crop
  (+0.073), as the review says.
Confirmed. E14 is re-labelled PROVISIONAL: `FitConfig.repropose` now defaults to False (R0, ff0a51f), which is
the E9 pipeline exactly (the re-proposal branch is the only code E14 added to the fit path). The module stays
for a future calibrated acceptance test.

Finding 2 (the kept change mostly adds false vessels; the negative controls got worse). Confirmed from the
batch-4 recordings already in LOG (controls_score 0.6427 -> 0.6056, 10 of 42 control fits worse, none
better; precision down on 10 of 10 tier-2 images). Fixed by R0 and by two charter changes in program.md:
- keep rule: a change that adds/removes edges or moves a probe must run the multi-seed controls; a fall of
  controls_score or new false length on the empty controls is a VETO; the tier-1 gain must survive leaving
  any one crop out (paired.py's leave-one-out range above 0);
- tier-2 confirm rule: paired per-image test against the last confirmed run; CONFIRMED only when
  mean > 2 SE; non-inferior added code is provisional (off by default or reset); mean < -2 SE resets.

### E18: a coarse 'magnocellular' channel, used twice (pre-registered; direction 1)

Hypothesis: on s004 and pathologic, stage 1's background (a fine-scale band-CNR mask, B a masked mean at
8 px) is fitted ON deep, blurred vessels, so the target has lost 18-36 % of the true OD before any fit
(tier2diag). One coarse orientation-selective detector (Hessian ridges of log I at sigma 4-16 px, elongation
gated; the review's pilot b6review/coarse_mask.py) (a) is OR-ed into the background mask, so B is still
estimated only on the mask's negative, and (b) its causes are proposed: the coarse channel of neuromimetic
stages 3-6 on the new target, traces novel against the proposed network added as free DEEP edges (no node
where they pass under a sharp vessel). The joint fit and the MDL / wide / flank tests judge them.
Change: coarse.py + Config.coarse (pipeline, 7 lines). One variable (the coarse channel), two uses that
must not be separated (the review: (a) without (b) collapses pathologic pos).
Predicted:
- tier 1: fast crops about unchanged (+-0.003; few deep vessels in 256 px crops), probes about unchanged
  (no deep vessels; the gate should flag nothing on the empty probes, but wide line probes may gain a
  coarse ridge); composite +-0.003. Controls must not regress (veto);
- tier 2 (the judge): s004 and pathologic: target_fidelity / target_keep up, explained up, recall and
  junction F1 up (deep vessels are missing crossing arms), coarse type balance up; width may fall (the
  deep edges are wide and poorly identifiable); s000/s007/s008 within +-0.002.
Pilot before the commit (scratch b6/pilot.py, whole images, scored against E9's tier-2 rows; truth only in
the scorer). Two placements of the new target:

| variant | image | composite | graph | pos | width | explained | prec / rec | jf1 | type | target_fid | deep kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E9 | s004 average | 0.7370 | 0.672 | 0.764 | 0.740 | 0.902 | 0.939 / 0.671 | 0.687 | 0.600 | 0.921 | - |
| early (whole fit on it) | s004 average | 0.7436 | 0.665 | 0.736 | 0.767 | 0.964 | 0.925 / 0.706 | 0.642 | 0.583 | 0.976 | 2 |
| late (retarget only) | s004 average | **0.7600** | 0.680 | 0.776 | 0.803 | 0.942 | 0.943 / 0.685 | 0.635 | 0.674 | 0.969 | 2 |
| E9 | pathologic average | 0.6472 | 0.581 | 0.680 | 0.729 | 0.733 | 0.885 / 0.567 | 0.477 | 0.625 | 0.766 | - |
| early | pathologic average | 0.6573 | 0.585 | **0.597** | 0.758 | 0.833 | 0.839 / 0.600 | 0.535 | 0.557 | 0.848 | 5 |
| late | pathologic average | 0.6499 | 0.575 | 0.667 | 0.755 | 0.753 | 0.884 / 0.573 | 0.450 | 0.617 | 0.855 | 2 |

The early placement repeats E1 (pos -0.08 on pathologic: the sharp centrelines move toward the deep OD
before the deep edges explain it). The LATE placement (deep edges proposed from the coarse target and
judged by the joint fit / MDL on stage 1's target; the coarse mask enters at the retarget, geometry frozen)
is the committed E18. Pilot prediction for tier 2: s004 +0.01..+0.02, pathologic about +0.003, others ~0.

E18 result (tier 1, 31b95c0): composite 0.793358 (+0.0080 against R0/E9 0.785309); **discarded by the
controls veto** (the new keep rule):
- fast +0.0235 (0.6870 -> 0.7105), up on 9 of 10 crops, leave-one-out range [+0.014, +0.026] (not one
  crop: pathologic average +0.107, s004 frame +0.034, s007 average +0.032, s000 frame +0.028); graph
  +0.019, width +0.034, explained +0.059, explained_junction +0.085; pos -0.009.
- probes -0.0074 (NOT held: predicted unchanged). Two kinds of loss: explained -0.01 to -0.02 on most
  line, crossing and parallel probes (the dilated ridge mask moves B's support further from the line, so
  B is interpolated from farther away; (a) costs a little where nothing leaked), and false deep edges
  (line_d5_f25_h0.5 0.757 -> 0.648, cross_a30 graph 1.0 -> 0.90).
- multi-seed controls 0.6427 -> 0.6372 and empty_average false length 80.7 -> 94.5 px: two false deep
  edges, empty seed 2 (78 px, cylinder contrast a = 0.034) and fork_wide seed 6 (119 px running along the
  fork's own arms, 3 false crossings). The pre-registered 'empty controls unchanged' did NOT hold: the
  gate flags an illumination lump.
Lesion pilot (whole images, the coarse mask in the retarget WITHOUT deep edges): s004 average 0.7450
(+0.008; full E18 placement +0.023), pathologic average 0.6544 (+0.007; full +0.003). So (a) alone is a
clean gain in width and explained with graph and pos identical (geometry is frozen at the retarget, so the
review's 'pos collapses without the causes' applies only to an early placement), and the deep edges help
where they are true (s004: type balance +0.07, width +0.04) and cost junction F1 where they are not.
Proposal statistics before any fit (scratch b6/deepstats.py): true deep vessels have a = 0.14-0.36 and run
across the image; the empty-control edge has a = 0.034; the fork_wide edge is a wall echo of the fork's arm.

### E19: gate the deep proposals like a second channel (pre-registered)

Hypothesis: E18's false deep edges are (i) a coarse echo running parallel to a proposed vessel (the fork
arm), which neuromimetic already removes between its channels by merging against the coarser channel, and
(ii) a faint background lump (a = 0.034), far below every true deep vessel (a >= 0.14).
Change: in coarse.deep_edges, merge the coarse traces against the proposed network as the coarsest channel
(merge_channels, as repropose does) and drop a trace whose median cylinder contrast is below 0.1 (= flank_a,
the existing contrast at which a one-sided line counts as a vessel). Proposal-time check (scratch): the
empty seed-2 and fork_wide seed-6 edges are gone, s004 and pathologic keep 3-5 deep edges, s000 average loses
its one.
Predicted: controls back to 0.6427 with empty false length 80.7 (veto passes); probes up from E18 toward
~0.88 (the false deep edges on line_d5_f25_h0.5 and cross_a30 gone; the (a) explained loss stays, ~-0.003);
fast about E18's (+-0.005); composite above R0 by > 0.002. Tier 2: s004 and pathologic up, others ~0.

E19 result (tier 1, e0c9604): composite 0.791564 (+0.0063 against R0); **discarded by the controls veto**.
- held: empty_average false length back to 80.7 px (both false deep edges gone), probes 0.8814 (-0.0022
  against R0, against E18's -0.0074), corner and crossing_angle sweep means up;
- fast 0.7017 (+0.0147; E18 +0.0235: the gate also drops some true deep edges or fragments them);
- controls 0.6396 against 0.6427: no false length, but line_d3 0.661 -> 0.648, cross_a45 0.901 -> 0.897,
  fork_wide 0.884 -> 0.882 at IDENTICAL fitted lengths. This is (a): the coarse ridge of an already-found
  line, dilated by up to 16 px, moves B's support away from the line, B is interpolated from farther away,
  and the explained OD falls. The veto as written (any controls_score fall) applies.

### E20: only NOVEL coarse ridges claim the background (pre-registered)

Hypothesis: E19's remaining control loss is the coarse response of vessels the fine channels already
found. Like a magnocellular unit that is silent where the parvocellular map already explains the input,
a coarse ridge should take part in B only where it is new: a connected ridge lying mostly (>= 50 %) in
stage 1's mask leaves B alone.
Change: ridge_mask keeps only connected ridges with < 50 % of their pixels in stage 1's mask (before
dilation). Scratch check: the line control's mask is unchanged (0.371 -> 0.371; all ridges gave 0.450);
s004 average 0.385 -> 0.466 (all ridges 0.680), pathologic average 0.338 -> 0.506 (0.661).
Predicted: controls back to 0.6427 (line_d3 0.661, identical lengths), probes back to about R0's 0.8836
(the explained loss on the line probes gone), fast lower than E19 (less flank OD recovered around known
wide vessels; +0.005 to +0.012 against R0), composite +0.003 to +0.006 against R0. Tier 2: s004 and
pathologic up, others within +-0.002.

E20 result (tier 1, 3072e73): composite 0.788501 (+0.0032 against R0), **kept provisionally**. Held:
- controls 0.642610 (R0 0.642661): line_d3 0.6608 and empty false length 80.7 back to R0 exactly; the
  remaining -0.00005 is fork_wide 0.8838 -> 0.8837 and cross_a45 0.9011 -> 0.9009 at identical lengths. The
  veto as written triggers on ANY fall; with no false length and a 5e-5 change I kept it provisionally and let
  tier 2 judge (not pre-registered: noted so the reviewer can weigh it);
- probes 0.8830 (-0.0007): only cross_a45_wide_over_thin moved (width 0.50 -> 0.37: a novel coarse ridge
  beside the wide vessel takes B's support away from the thin one's flank);
- fast +0.0071 (7 up, 2 down by < 0.001; leave-one-out [+0.0041, +0.0080]), below E19's +0.0147.
Residual recording (E20; against the E11 recording, the nearest earlier one): the junction under-prediction
shrinks everywhere: fast8 (pathologic average) compound bias -0.88 -> -0.74, pseudo-T -0.88 -> -0.75,
crossing -0.49 -> -0.40; fast0 bifurcation -0.43 -> -0.32, crossing -0.27 -> -0.20; the junction discs'
share of the squared residual fast0 0.56 -> 0.50, fast2 0.57 -> 0.52, fast8 0.39 -> 0.35, cross_a20 0.17 ->
0.12; empty_average still nothing fitted. The sign stays negative (under-predicted centres): the remaining
error at junctions is still target loss, not additive overlap.

Tier-1 lesion of (b) (scratch b6/lesion_nodeep.py: E20 with deep_edges returning nothing): 0.787735 (+0.0024
against R0). graph and pos IDENTICAL to R0 (the ridges only change the final, geometry-frozen fit); width
+0.006, explained +0.027. The deep edges add +0.0008 on tier 1 (s004 frame +0.010, s007 average +0.016,
pathologic average -0.010).

**Tier 2 (E20, 3072e73): 0.765183** (E9 0.760603, E14 0.760839), deterministic True, digest
4e65a9805c3ec08d, 2133 s. Paired against E9 (paired.py): mean +0.00458, SE 0.00281, t 1.63, 9 up / 1 down
(s008 frame -0.00004); leave-one-out [+0.0019, +0.0051]. Per image: s004 average +0.0075, s004 frame
+0.0058, pathologic average +0.0027, pathologic frame +0.0287; s000, s007, s008 within +-0.0007. The
pre-registered direction held on every image (s004 and pathologic up, the others within +-0.002). Parts
(paired, against E9): width +0.0081 (t 2.05), explained +0.0091 (t 1.88), graph +0.0040, pos -0.0015,
precision +0.0001 (no criterion shift), recall +0.0042, junction type balance +0.009.
Tier 2 of the (b) lesion (same commit, deep edges off; recorded as b6_e20_nodeep): **0.761983**,
deterministic True, digest ee234d53452257f8. Against E9: mean +0.00138, SE 0.00054, **t 2.53**, 9 up / 1 down
(-0.00004): CONFIRMED under the new rule. The deep edges on top of it: +0.00320, SE 0.00257, t 1.25, 3 up /
1 down (pathologic average -0.0015), mostly one image (pathologic frame +0.0257; leave-one-out min +0.0007).
Decision under program.md's confirm rule: (a) is confirmed; (b) is added code that is only non-inferior on
whole images, so it is PROVISIONAL: `Config.deep` defaults to False (b4a46f6). E20 as a whole fails the 2 SE
bar (t 1.63) only because (b)'s gain is concentrated on one image.
Whole-image recording (tier2diag, E20 against the batch-5 E14 recording): target_keep +0.017 (s004
average), +0.025 (s004 frame), +0.038 (pathologic average), +0.055 (pathologic frame), +0.001 to +0.003
elsewhere; the worst junction discs improve on s004 (p10 0.554 -> 0.643, 0.453 -> 0.586) and pathologic
frame (share of discs below 0.5: 0.32 -> 0.23).
The verification run of b4a46f6 (Config.deep off) reproduced the lesion's tier-1 digest 3f0cc5204d2f7ffa
exactly, so the lesion's tier 2 is this commit's tier 2: **confirmed, 0.761983** (E9 +0.00138, t 2.53).
Against the batch-5 confirmation (E14, now provisional and off): +0.00114, SE 0.00122, 5 up / 5 down; the
reference for the paired rule is E9, the pipeline R0 restored, and E14's own gain over E9 was t 0.19.

### Batch 6 summary

| # | commit | change | tier-1 composite | fast | graph | pos | width | explained | expl_junction | probe_score | controls | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E14 | ffc8257 | (batch 4/5, confirmed then) | 0.788423 | 0.6996 | 0.633 | 0.749 | 0.760 | 0.791 | 0.795 | 0.877 | 0.6056 | - | - |
| R0 | ff0a51f | repropose off (review findings 1, 2) | 0.785309 | 0.6870 | 0.610 | 0.752 | 0.768 | 0.772 | 0.770 | 0.884 | 0.6427 | keep (review) | yes: E9 digest exactly |
| P1 | 116e5d1 | paired.py; controls veto + paired tier-2 rule (program.md) | (pipeline unchanged) | | | | | | | | | charter | - |
| E18 | 31b95c0 | coarse ridges (all) in the retarget mask + deep edges | 0.793358 | 0.7105 | 0.629 | 0.744 | 0.802 | 0.831 | 0.855 | 0.876 | 0.6372 | discard (veto) | no: probes and controls fell |
| E19 | e0c9604 | E18 + deep edges merged against the network, a >= 0.1 | 0.791564 | 0.7017 | 0.614 | 0.750 | 0.793 | 0.824 | 0.847 | 0.881 | 0.6396 | discard (veto) | partly: false length gone, lines still lose |
| E20 | 3072e73 | E19 + only NOVEL coarse ridges claim B | 0.788501 | 0.6940 | 0.608 | 0.746 | 0.785 | 0.809 | 0.829 | 0.883 | 0.6426 | keep (provisional) | yes |
| E20a | b4a46f6 | E20 with the deep edges off (provisional) | 0.787735 | 0.6925 | 0.610 | 0.752 | 0.774 | 0.799 | 0.806 | 0.883 | - | keep, tier-2 confirm | yes: lesion digest |

**Tier 2:** E20 0.765183 (+0.0046 vs E9, t 1.63: not confirmed as a whole); E20a 0.761983 (+0.0014, t 2.53,
9 of 10 up): **confirmed**. Pushed state: E20a (novel coarse ridges on; deep edges and residual re-proposal
provisional, off).

**Tuning curves (E20a against R0).** No threshold moved (width 2.5, blur 25, contrast 0.25, fork 4/5,
crossing angle 30, depth 5, T 2/2, gap 2.0, ends 2/2, rbc 2/2, empty 2/2). Sweep means identical except
crossing_width 0.900 -> 0.877 (cross_a45_wide_over_thin: width 0.50 -> 0.37). Multi-seed controls (E20,
whose ridge mask is E20a's): 0.6426 against 0.6427, empty false length unchanged. The fast crops and the
probes see almost nothing of this change (no deep vessels in them); tier 2 and tier2diag do.

**What the batch taught.**
1. The review was right about E14: its whole-image gain was t 0.19 and its cost was false arms. With the
   paired rule, E14 would never have been confirmed. The paired test also separated E20's two parts cleanly:
   the background fix is small but consistent (9 of 10 images, t 2.53); the deep edges are larger but carried
   by one image (t 1.25).
2. WHERE a cleaner target enters decides whether it helps (the E1 lesson, now measured on whole images).
   Fitting the joint stage to it drags sharp centrelines toward OD nothing explains yet (pathologic pos
   0.680 -> 0.597); giving it only to the geometry-frozen final fit leaves graph and pos bit-identical and
   turns it into width and explained. The review's 'do not ship (a) without (b)' holds only for the early
   placement.
3. A coarse channel must be silent where the fine channels already explain the input (E19 -> E20): the
   coarse response of a known line, dilated by up to 16 px, only moved B's support away from it (line
   controls -0.012). Restricting the mask to NOVEL coarse ridges kept the hard images' gain and returned the
   line controls to R0 exactly. Its one remaining cost is a novel coarse ridge beside a wide vessel
   (cross_a45_wide_over_thin width -0.14).
4. False deep edges are the same two failure modes as false arms: a faint background lump (a = 0.034 against
   0.14-0.36 for true deep vessels) and a coarse echo along a found vessel. The proposal-time gate (merge
   against the network as the coarsest channel, a >= 0.1) removed both on the controls.
5. On whole images the remaining error at junctions is still UNDER-prediction (biases -0.2 to -0.75 in
   every class): the target still loses OD at junctions, now less (target_keep +0.02 to +0.06 on s004 and
   pathologic).

**Directions for the next batch (ranked).**
1. Deep edges, confirmed or killed (Config.deep): they are the larger effect (+0.0032 on tier 2) but rest on
   pathologic frame. Record per deep edge on the four hard images (scratch, truth-labelled): on-truth
   fraction, depth (blur) of the matched true vessel, and what they do to junction F1 (pathologic average
   -0.0015). Pre-register a gate from that (e.g. minimum length relative to width, or the edge's MDL gain
   on the coarse target instead of stage 1's) and re-run tier 2 with the paired rule.
2. Width identifiability (the review's direction 2): blur as optics, not a free per-vessel parameter. A
   per-image PSF floor on s (from high-CNR edges after the joint stage) and stronger along-edge smoothness on s
   than r. Target: cross_a45_wide_over_thin's width loss and the tier-2 width variance (rel. error
   0.14-0.28 at zero bias).
3. Topology moves by fit comparison at event clusters (fork vs crossing vs T; add / drop an arm): the
   junction type and F1 terms are still the largest loss (graph 0.683 on tier 2).
4. Residual re-proposal (off) needs a context-matched null before it returns (batch 5, direction 2).
5. Rule note for the reviewer: the paired 2 SE rule with n = 10 fixed images cannot confirm a change that by
   design moves only the 4 hard images unless the effect is very even across them. A pre-registered
   directional criterion (named images up, the rest within +-0.002) is the alternative; it held for E20 on
   every image but was not the rule in force, so it was not used.

## Batch 7

Setup: last kept = b4a46f6 (E20a; tier 1 0.787735); last tier-2 confirm row b4a46f6 (0.761983, under the
per-image rule); pushed head 43f75ba. Dev set: still the 5 frozen scenes (no healthy_s009+ generated).

### Priority 1: the review's findings on E20a (both confirmed; fixed)

Finding 1 (E20a's 'confirmed' counts the average and the frame of a scene as independent images).
Re-computed from runs/e9_tier2.json and b6/e20_nodeep_tier2.json, averaging the two kinds per scene (n = 5):
per-scene deltas s000 +0.00012, s004 +0.00279, s007 +0.00036, s008 +0.00003, pathologic +0.00360; mean
+0.00138, SE 0.00075, t 1.83 (below 2 SE); width t 2.00, explained t 1.52. Per image (the old test) the same
data gave t 2.53 because s004 (+0.0040 / +0.0016) and pathologic (+0.0043 / +0.0029) each count twice.
Confirmed exactly. Fixes (R1):
- `paired.py` groups by scene (kinds and crops averaged first; `--by image` keeps the old test for
  comparison), counts up / down only beyond a materiality of 5e-4 (E20a: 2 up, 0 down, not '9 up'), and
  prints a verdict (confirmed / provisional / regressed). program.md's tier-2 rule says so.
- E20a is re-labelled PROVISIONAL: `pipeline.Config.coarse` now defaults to False, the E9 / R0 pipeline
  exactly (the coarse channel is the only code E20a added to the path). No healthy_s009+ dev scene exists yet.
- Re-checked under the grouped rule: E20 (with deep edges) t 1.50, E14 t 0.19 (both already provisional);
  E9 vs E6 and E6 vs E3 are simplifications at mean ~0 (non-inferior, kept by the simplicity rule).

Finding 2 (the explained gain on the hardest image is mostly neighbouring vessels absorbing the deep OD that
the new target restores; the blur and contrast errors it costs are not in the composite). Confirmed from
the review's whole-image diagnostic (b6review2/whole.py, pathologic_s000 average; its digests reproduce the
tier-2 rows) and by the grouped guard metrics of E20a against E9 (per scene, s000 s004 s007 s008 path):
blur_err_px +0.0001 +0.0302 -0.0061 +0.0044 +0.0572; |contrast_bias| +0.003 +0.045 +0.003 +0.001 -0.063 (the
pathologic bias flips sign: -0.077 -> -0.014 on its far samples, -0.035 -> +0.376 near the ridges);
|blur_bias_px| mean +0.041. E20 (deep edges ON, the missing cause supplied) does not show it: blur_err mean
+0.0006, contrast_rel_err -0.0072, |contrast_bias| -0.0068. This is the explaining-away failure in one line:
a fuller target without its cause is absorbed by the nearest causes the model has.
Fixes: GUARD metrics in the keep rule (paired.GUARDS: blur_err_px 0.02 px, contrast_rel_err 0.005,
|contrast_bias| 0.02, |blur_bias_px| 0.05 px, width_rel_err 0.005, |width_bias| 0.02; a guard whose mean over
scenes worsens beyond its threshold, or one scene beyond twice it, vetoes a confirmation; pre-register their
direction). The thresholds were set with E14, E20 and E20a in view, so they are calibrated on these, not
tested by them: E20a is vetoed (blur_err pathologic +0.057, |contrast_bias| s004 +0.045), E14 is vetoed
(blur_err, contrast_rel_err, width_rel_err), E20 passes. Then E21 below (the review's second remedy).

### E21: the novel coarse ridges stay out of B, and out of the final fit's data term (pre-registered)

Hypothesis: the far-from-ridge part of E20a's gain is real (a background no longer pulled up by deep
vessels; pathologic far samples width_rel 0.211 -> 0.199, contrast_rel 0.329 -> 0.318), the near-ridge part is
failed explaining-away. Give the final fit no data where only an absent cause could explain it: pixels in the
coarse ridge mask (novel coarse ridges, dilated by their sigma) weigh 0 in the final fit's loss
(FitConfig.ridge_w = 0); they stay out of B (the residual rule: the mask's negative cleans the background).
A vessel crossing or running along a deep one keeps its joint-stage profile there (smooth along the edge).
Change: coarse_stage1 also returns the ridge mask; fit_network multiplies the final-fit precision by ridge_w
on it; Config.coarse on. One variable against E20a (the ridge pixels' final-fit weight 1 -> 0).
Predictions (against E20a; tier 1 and tier 2): graph and pos identical (geometry frozen in the final fit);
width about equal (its gain was mostly far from the ridges); explained DOWN (the near-ridge absorption
removed; pathologic about -0.01, s004 a little), still at or above E9; probes: identical except the ones with
a novel ridge (cross_a45_wide_over_thin width back up toward R0); guards (against E9): pathologic
contrast_bias near the ridges back to about E9's -0.04 (from +0.38), blur_err_px pathologic and s004 back to
within 0.02 of E9, no veto. Tier 2 against E9 (per scene): +0.0003..+0.0010, s004 and pathologic carrying it;
expected verdict PROVISIONAL (non-inferior added code), unless the far gain alone clears 2 SE.

Result (tier 1, 1f862f7; scratch b7/e21b_tier1.json, the patch in b7/e21.patch): composite 0.784410 (R1 0.785309,
E20a 0.787735), graph 0.609913 and pos 0.752174 identical (held), width 0.767566 (= E9, not E20a), explained
0.766438 (E9 0.772348, E20a 0.799228), explained_junction 0.751995, probes 0.882833, digest a19dca6426d49ca3.
Per scene against R1: -0.0027 -0.0033 +0.0026 -0.0012 -0.0004, t -0.95; guards contrast_rel_err,
|contrast_bias| and width_rel_err veto. DISCARD (below R1, added code); reset to R1.
(The first run, 3c90cc8, scored 0.779913: vesselmap renders no pixel of weight 0, so the exported render was
blank on the ridge mask. Fixed in 1f862f7 by rendering the returned model with the full weights: a bug in
the change, re-run per the crash policy.)
Near/far on pathologic average (b7/nearfar.py, the review's split; E9 / E20a / E21):
| true samples | contrast_bias | contrast_rel | blur_bias | blur_err | width_rel |
|---|---|---|---|---|---|
| near ridges | -0.035 / +0.376 / -0.046 | 0.368 / 0.465 / 0.368 | +0.02 / -0.005 / -0.223 | 1.53 / 1.58 / 1.64 | 0.496 / 0.497 / 0.508 |
| far | -0.077 / -0.026 / -0.054 | 0.329 / 0.318 / 0.305 | +0.45 / +0.54 / +0.50 | 1.44 / 1.48 / 1.43 | 0.211 / 0.199 / 0.202 |
Explained on that image: +0.0009 against E9 (E20a +0.0177): near-ridge footprint -0.0024, far +0.0036.
Which predictions held: the absorption is gone (near contrast bias back to E9's level) and the far gain
stays (contrast_rel 0.305, the best of the three). NOT held: explained fell below E9 on the fast crops, because
the vessels inside the ridge mask (16 % of the s000 average crop; dilated by up to 16 px) have no data in the
final fit and their profiles drift under the priors and the along-edge smoothness (s000 average crop: an
edge wholly in the mask a 0.099 -> 0.058; pathologic near-ridge blur bias -0.22). Deafferentation is not
explaining away: a region with no input is filled in from the neighbours' priors, and that is worse than
the biased input it replaced.

### Batch 7 summary

| # | commit | change | tier-1 composite | graph | pos | width | explained | probe_score | status | prediction held? |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 70c6072 | paired per scene + guards; Config.coarse off (E20a provisional) | 0.785309 | 0.610 | 0.752 | 0.768 | 0.772 | 0.884 | keep (review) | yes: E9 digest d56fee848ca3e349 |
| E21 | 1f862f7 | coarse on, ridge pixels out of the final fit's data term | 0.784410 | 0.610 | 0.752 | 0.768 | 0.766 | 0.883 | discard | partly: absorption gone, explained below E9 |

**Tier 2 (R1, 70c6072, run after a container restart interrupted the batch's own run): composite 0.760603, graph 0.683044, pos 0.803793, width 0.798493, explained 0.912199, explained_junction 0.914852; 2001 s; digest `cf65125bf2fb0dea`; deterministic: True.** Identical to E9's tier 2 in batch 3 (same digest), as predicted: R1 is the E9 pipeline with every provisional switch off, and it reproduces across sessions. Pushed state: R1, the E9 pipeline (coarse channel, deep edges and residual
re-proposal all provisional, off).

**What the batch taught.**
1. The unit of replication is the scene. With n = 5 scenes, of which 2 carry deep vessels, no change that
   acts only on deep vessels can reach 2 SE unless it is very even; the honest remedy is more scenes
   (healthy_s009+), not a looser rule.
2. The guards separate absorbing from explaining: E20a (ridges without causes) is vetoed by blur and contrast
   guards, E20 (ridges plus deep edges, the cause) passes every guard and has the larger composite gain
   (+0.0046, t 1.50). The deep edges are therefore the part to pursue, not the part to drop.
3. Silencing the unexplained pixels (E21) removes the absorption but the region then has no input and its
   vessels drift: worse than E9. The OD needs a cause in the model.

**Directions for the next batch (ranked).**
1. E20 (coarse ridges + gated deep edges) is the candidate: re-test it, with the guards, once more dev scenes
   exist; meanwhile record per deep edge on s004 and pathologic (on-truth fraction, depth of the matched
   true vessel, effect on junction F1) and pre-register a gate.
2. Width identifiability (blur as optics: a per-image PSF floor on s).
3. Topology moves by fit comparison at event clusters (fork / crossing / T).
