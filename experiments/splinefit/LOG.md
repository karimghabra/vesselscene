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
