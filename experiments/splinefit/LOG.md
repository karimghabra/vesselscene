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
