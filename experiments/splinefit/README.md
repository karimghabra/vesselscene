# experiments/splinefit: a jointly fitted, junction-aware spline render of the whole image

The neuromimetic annotator (`experiments/neuromimetic`) **proposes** vessels and junctions; this package turns
the proposal into a spline network and fits its differentiable render, jointly, to the whole image's optical
density. The end product is the network: centreline B-splines with radius r, blur s and contrast a profiles,
nodes at forks and confluences, crossings of vessels at different depths as overlaps without a node. The
work is organised as an *autoresearch* loop (Karpathy): a frozen evaluator, one metric, fixed-budget
experiments that are kept or reset, a results log. The charter every experiment follows is
[program.md](program.md); the history and the findings are in [LOG.md](LOG.md).

The full report of both experiments, with annotated held-out scenes (truth, proposals, the fitted network and
vesselmap over the image, renders, residuals and junction close-ups), is [../REPORT.md](../REPORT.md).

**Scope since re-baseline R2: only the averaged still is fitted.** The single frame of a scene is out of
scope; `fastset.KINDS` is the one switch (the fast crops, the scored probes, tier 2 and the tables follow it).
The results below were measured on both kinds during the night and are kept as measured; REPORT.md restates
them for the averaged stills, and [LOG.md](LOG.md) (R2) re-judges the loop's decisions on the averages
(`rejudge_average.py`, output in results/dev/rejudge_average.txt).

## Results

**On six held-out scenes (12 images), the fitted spline render explains the vessel OD far better than the
proposals' own profiles, including at junctions. It keeps their graph and improves the vessel positions. It
beats LIMBUS vesselmap on graph, position, crossings and composite on every image. The junction-aware
composition of the render (a union at nodes) is not what makes the difference: an additive-render lesion
does as well on the development images.** The final pipeline is the state the autoresearch loop confirmed on
the development scenes (results.tsv commit 70c6072, the E9 pipeline; only a docstring changed after it). It
was run once on the held-out scenes.

Means over the 6 held-out scenes (healthy seeds 1, 2, 3, 5, 6 and pathologic seed 1, 480 x 768 px), averaged
still / single frame. Composite = 0.5 graph + 0.5 mean(pos, width, explained) (score.py).

| row | composite | graph | pos | width | explained | explained at junctions | line F1 | junc F1 strict | crossing R | s / image |
|---|---|---|---|---|---|---|---|---|---|---|
| **fit (this package)** | **0.770 / 0.756** | 0.697 / 0.667 | 0.808 / 0.797 | 0.800 / 0.827 | 0.921 / 0.909 | 0.936 / 0.926 | 0.840 / 0.829 | 0.672 / 0.635 | 0.354 / 0.285 | 116 / 99 |
| proposals alone (neuromimetic) | 0.756 / 0.723 | 0.706 / 0.656 | 0.788 / 0.717 | 0.828 / 0.835 | 0.801 / 0.815 | 0.804 / 0.828 | 0.835 / 0.808 | 0.682 / 0.637 | 0.319 / 0.241 | 8 / 8 |
| vesselmap `build_map` | 0.668 / 0.692 | 0.520 / 0.547 | 0.708 / 0.703 | 0.786 / 0.841 | 0.953 / 0.968 | 0.964 / 0.982 | 0.644 / 0.716 | 0.444 / 0.439 | 0.116 / 0.100 | ~600 (cached) |
| fit started from the truth (oracle) | 0.740 / 0.730 | 0.570 / 0.559 | 0.852 / 0.833 | 0.930 / 0.935 | 0.949 / 0.937 | 0.963 / 0.949 | 0.556 / 0.547 | 0.519 / 0.491 | 0.630 / 0.453 | 187 / 178 |
| the true network, rendered | 0.790 / 0.774 | 0.589 / 0.559 | 1.000 / 1.000 | 0.998 / 0.998 | 0.976 / 0.967 | 0.968 / 0.957 | 0.490 / 0.440 | 0.522 / 0.484 | 0.826 / 0.760 | - |

Each cell below is the fit minus the row, as the mean per-image difference over all 12 images. The brackets
give the images where the fit is higher (and ties):

| fit minus | composite | graph | pos | width | explained | explained at junctions | line F1 | junc F1 strict | crossing R |
|---|---|---|---|---|---|---|---|---|---|
| proposals alone | +0.024 (11/12) | +0.001 (7/12) | +0.050 (11/12) | -0.018 (6/12) | +0.107 (12/12) | +0.115 (12/12) | +0.014 (10/12) | -0.006 (6/12) | +0.040 (6/12, 6 ties) |
| vesselmap | +0.083 (12/12) | +0.149 (12/12) | +0.097 (12/12) | +0.000 (8/12) | -0.046 (4/12) | -0.042 (3/12) | +0.155 (12/12) | +0.212 (12/12) | +0.212 (12/12) |
| oracle init | +0.028 (9/12) | +0.118 (12/12) | -0.040 (0/12) | -0.119 (0/12) | -0.028 (0/12) | -0.025 (0/12) | +0.283 (12/12) | +0.148 (12/12) | -0.222 (2/12, 1 tie) |

What this says:
- **The joint fit explains the vessel OD.** The `explained` score is 1 - SSR / sum(OD^2) over a band within
  3 px of the observable vessels, against an oracle background (score.py). The fit scores 0.91-0.92 there,
  against 0.80-0.82 for the proposals' profiles drawn through vesselmap's additive renderer. In the junction
  discs it scores 0.93-0.94 against 0.80-0.83. The junction gain (+0.115) is about the same as the overall
  gain (+0.107), and larger on only 6 of 12 images.
- **The junction-aware composition is not what explains the junctions.** A lesion (`annotate_additive`, the
  same pipeline with vesselmap's plain additive render) was run on the 10 development images during the
  write-up check. Against it, the fit is -0.0003 explained at junctions (higher on 4 of 10) and -0.0006
  composite (5 of 10). The lesion's widths are better on all 10 images. On the true network, the composition
  adds +0.014 at junctions on average, at most +0.043 on one image (dev `oracle_render` against
  `oracle_render_additive`). The junction gain
  therefore comes from fitting profiles and positions jointly, with any render. A post-hoc lesion like this
  was never run by the loop: results/dev/lesion_additive/.
- **The fit's remaining error still sits at junctions, as an under-prediction.** In the held-out figure
  below, 41 of 51 junctions are under-predicted. The four worst are compounds (-0.17 to -0.51), and the
  junction discs hold about 0.6 of the band's squared residual. This points at the target (OD lost to the
  background near junctions; see the lessons) more than at the render.
- **It keeps the proposal's graph.** Graph is +0.001, and junction F1 is within 0.01. Topology still comes
  from the proposal, and the fit does not revisit junction types. Positions improve: +0.05, and +0.08 on
  frames, 11 of 12 images. Widths are slightly worse than the proposals' own profile widths (-0.018).
- **Against vesselmap:** much better graph, positions, crossings and line F1, on every image. vesselmap
  explains more (0.95-0.97), but only on the two images where the fit's own target had lost 11-24 % of the OD
  to the background (s002 and pathologic). There, vesselmap's own background fit kept that OD. On the 8 images
  where the fit's target keeps at least 98 %, vesselmap explains 0.011 less. This is the residual rule's
  concern seen from the other side: what the target loses to the background, no render can explain.
- **Started from the truth, the same fit keeps better geometry but loses vessels.** The oracle runs the
  shipped fit schedule from the complete true network. It explains more (0.94-0.95) and keeps far better
  widths (0.93-0.94 against 0.80-0.83) and positions. But its prune drops observable vessels it was given
  (centreline recall 0.79-0.82, crossing recall 0.45-0.63), and its graph and composite are lower than the
  fit's. The remaining gaps are three, and they are not separated:
  - the target's background leak, largest on s002 and pathologic;
  - width identifiability: the fit's widths are slightly worse than the proposals' starting widths (-0.018),
    and the oracle drifts from its true widths (0.998 to 0.93);
  - topology: missed faint vessels and arms, and the prune.
- **The gain over the proposals carries over from development to held-out scenes.** Fit minus proposals is
  +0.018 composite on development (9 of 10 images) and +0.024 held-out (11 of 12). Explained is +0.109 and
  +0.107. The absolute composites (development 0.7606, held-out 0.7629) come from different scenes and are not
  a like-for-like test. On the four held-out scenes no development agent ever read (healthy 3, 5, 6,
  pathologic 1), the fit minus the proposals is +0.024 composite (7 of 8 images) and +0.117 explained (8 of 8),
  and the fit minus vesselmap is +0.092 (8 of 8). The loop's own development gain (v0 to E9, +0.0067) was not
  tested on the held-out scenes: v0 was not run there.
- **Determinism and speed.**
  - Two runs per image, in one process, gave identical digests on all 12 held-out images.
  - The confirmed development state reproduced its digest across sessions (a separate process, after a
    container restart).
  - A fit takes about 90-145 s (mean 108 s) per 480 x 768 image on 2 CPU threads. Most of the held-out run
    shared the 4-core machine with another job.

![held-out frame, proposals' render](results/heldout/figures/proposals_healthy_s003_frame.jpg)
![held-out frame, fitted render](results/heldout/figures/fit_healthy_s003_frame.jpg)

*Held-out healthy seed 3, single frame (a scene no development agent read). Each figure shows the target
OD_obs, the render, and the residual OD_obs - render; below are the four worst junctions. Top: the proposals'
own profiles (explained 0.82; compound junctions over-predicted by up to +0.62). Bottom: the fitted render
(explained 0.975, 0.98 at junctions). Most of its residual is small. What remains is under-prediction at
junctions (the four worst are compounds) and along faint vessels at the right, which were never proposed.*

Tables: [results/heldout/table.md](results/heldout/table.md) (all held-out scenes and the untouched four,
every part) and [results/heldout/paired.md](results/heldout/paired.md). Made by `heldout_report.py` from
`run_experiment --tier 2` and `evaluate_fit` on the held-out scenes (`SPLINEFIT_DEV=<heldout>
SPLINEFIT_ALLOW_HELDOUT=1`).

## What the autoresearch loop did and learned

About 13 hours from setup to the last batch (about 10.5 hours after the evaluator was frozen): 7 batches, 21 experiments (E1-E21), 27 tier-1 runs and 10 tier-2 runs. The last tier-2 run, the confirmation of the final state, was re-run after a container restart interrupted it. The kept changes of
batches 1-6 were each reviewed adversarially, and the reviews' findings opened the next batch. A strategy
review, looking at the probe tuning curves and residual recordings, came before batches 3 and 6. Batch 7's
fixes were not reviewed. Hypotheses were pre-registered with their predicted effect (exceptions under Caveats). [LOG.md](LOG.md) has every experiment, its prediction and what happened;
`results/results.tsv` is the run log. On the development set, tier 2 went from 0.7539 (v0) to 0.7606. Four
changes survived, plus a determinism fix (be34189: vesselmap's render runs eagerly, not through
torch.compile, whose per-shape cache made results depend on the images run before):
- **E3: freeze the geometry in the final fit** (+0.0066 on tier 2). This was the one large gain. A cleaner
  target exposes OD the network does not explain yet, and free centrelines slide into it. So positions are
  fitted on the first target, and widths, contrasts and optics on the cleaned one.
- **E5 and E6: a faint, wide edge must be seen on both flanks.** This removed false wide vessels laid over
  illumination lumps at the frame in the probes (the negative control). On whole images it changes almost
  nothing: tier 2 +0.0001, from one edge, which was part of a real border vessel. A later review (batch 4)
  found the battery counts one shared lump many times, so the probe gain was overstated.
- **E9: deleting two terms of the MDL prune** that never decided anything. The score is identical, and the
  code simpler.

Ideas that did not survive, kept as provisional switches (off):
- **Re-proposing arms from the fit residual** (E13/E14). This is predictive coding's "attention where error
  remains". About half the arms were real vessels, but no acceptance test (MDL cost, CFAR) separated the
  faint true arms from texture echoes.
- **A coarse 'magnocellular' channel** (E18-E20). Novel coarse ridges join the background mask, and deep edges
  are proposed.
  - The ridges alone give +0.0014 +- 0.0008 per scene (t 1.83). That is carried by s004 and pathologic, and
    the guard metrics vetoed it: without a cause for the deep OD the ridges restore, neighbouring vessels
    absorb it as contrast.
  - With the deep edges, the gain is +0.0046 per image (t 1.50 per scene), mostly on one image.

The main lessons, in the loop's own words (LOG.md):
- **Where a cleaner target enters decides whether it helps** (E1, E3, E20). Positions must not chase OD the
  model cannot yet explain.
- **On the hard images the target is the ceiling.** The pipeline's final (retargeted) target keeps only
  64-82 % of the true OD on s004 and pathologic: the background takes 18-36 %. This was measured with
  tier2diag on the batch-5 pipeline.
- **Typing errors start in the proposal's event clustering** (neuromimetic stage 7), which the fit never
  revisits. The loop's further reading, that junction errors are missing arms, rests on a two-crop pilot and
  residual recordings; adding arms (E14) lowered type balance on 6 of 10 images.
- **Fast crops over-reward added edges** (E7, E10, E13, E14). The single-seed probe battery is biased both
  ways: it over-rated E5/E6's lump removal and under-rated E14's false arms. Whole images (tier 2), with a
  paired per-scene test and multi-seed controls, are the honest test. The paired
  per-scene rule, guard metrics and controls were added to program.md during the night, after reviews caught
  gains that did not hold.

Next steps (batch 7's ranking, then earlier batches' open directions):
1. Deep edges with a pre-registered gate, tested on more scenes. With 5 development scenes, only 2 have deep
   vessels.
2. Blur as optics (a per-image PSF floor), for width identifiability of thin vessels.
3. Topology moves decided by fit comparison at junction clusters: fork, crossing, T, and add or drop an arm
   (batches 6-7).
4. The target at junctions: the fit's residual is an under-prediction there, and vesselmap explains more only
   where the target lost OD to the background. A background mask that keeps junction regions (the residual
   rule applied more thoroughly) is a direct test.
5. A context-matched acceptance test for residual re-proposals (batches 5-6).

Caveats:
- **Synthetic scenes only.** The effective sample is 6 held-out scenes, one of them pathologic.
- **vesselmap ran as `build_map` at its defaults.** On averaged stills it is outside its design (its
  README), and its junctions are typed by the adapter's rule.
- **The development set is 5 scenes**, and the loop ran 27 tier-1 and 10 tier-2 runs on it. The held-out
  result above was measured once, after the freeze.
- **The oracle row is a diagnostic:** it starts from the truth.
- **The held-out scenes were used before.** They were the neuromimetic study's held-out set, so this is the
  second test on the same six scenes (healthy seeds 1 and 2 also had their junction-type counts read during
  that study).
- **Pre-registration was mostly kept.** The E6 and E10 thresholds were read off the data they were tested on,
  and E20's keep despite a guard veto was marked not pre-registered (LOG.md).

## Method (v0)

1. **Target (the residual rule).** OD = B - ln I, with the background B fitted only on the pixels **outside**
   the vesselness mask of neuromimetic stage 1 (`photoreceptors`). The fit never sees a vesselness or
   orientation map or a mask as its target. Precision weights w = 1 / sigma^2, sigma the local RMS of the
   high-passed OD over the background (texture dominates the noise of an average still).
2. **Proposal** (`proposals.py`). `neuromimetic.run` (stages 1-8) gives traced vessels cut at typed
   junctions with OD profiles; a junction typed 'crossing' with four ends becomes two through edges and a
   recorded crossing (no node), other 3+ way junctions become nodes, continuing ends of one trace pass
   through. Each chain is a clamped cubic B-spline with r, s, a profile splines (box profile converted to
   vesselmap's blurred cylinder chord).
3. **Render** (`render.py`, `JunctionModel`, a subclass of LIMBUS vesselmap's `NetworkModel`). Per edge the
   blurred cylinder chord in closed form, summed, then corrected in small windows: at a **node** the lumens
   are one blood volume, so the OD is that of their **union** (max of capped chords minus the sum of butt
   ends); at a **crossing** the ODs add with an image-wide additivity kappa (initial 0.87). An image-wide
   halo follows.
4. **Fit** (`fit.py`). Every parameter (control points, shared node positions, r/s/a profiles, halo, kappa)
   descends the same precision-weighted residual: profiles first, then everything jointly, MDL pruning (an
   edge must explain more NLL than its description costs, with the gain of the junction-aware prediction;
   a wide edge must also beat a smooth background, and a faint wide edge must be seen against background on
   both flanks, batch 2, E5/E6), topology clean-up, **retarget** (B re-fitted
   outside stage 1's mask OR the fitted render's support), and a final fit of the profiles and optics with
   the geometry frozen (batch 1, E3: free centrelines slide into the OD the re-cleaned target exposes).
5. **Export** (`pipeline.py`). One polyline per edge (junction to junction), nodes typed, crossings found
   geometrically in the fitted network, `od_render`, `od_target`.

`pipeline.annotate(image, valid)` reads the image and its valid mask only.

## The evaluator (frozen; program.md lists the files and hashes)

- `score.py`: graph (harness.score on the network's polylines and typed junctions), geometry against the
  TRUE network on observable samples (offset, r, s, a errors; pos, width), render (explained against the
  image's own OD with an oracle background, explained_junction, junction_bias, chi2) and
  `composite = 0.5 * graph_score + 0.5 * mean(pos, width, explained)`.
- `fastset.py` / `fastset.json`: the fast tier, ten 256 x 256 crops of the dev images (one per scene and kind,
  the most junctions).
- `probes.py` / `probe_battery.json`: the psychophysics battery, 34 small canonical stimuli with exact truth
  (width, blur, contrast, forks, crossing angle / depth / width, T and pseudo-T, parallel pairs, ends,
  red-cell gaps, empty backgrounds) and the fitter's tuning curves and thresholds. The data (9 MB) is
  regenerated deterministically on first use and checked against the committed manifest.
- `run_experiment.py`: one experiment. Tier 1 (the loop, ~4.5 min on 2 threads):
  `composite = 0.5 * fast_composite + 0.5 * probe_score`. Tier 2 (confirmation, ~30 min): every dev image,
  both kinds, twice; must be deterministic.
- `evaluate_fit.py`: reference rows (proposals alone, LIMBUS vesselmap, the truth, oracle rows) into
  `results/dev/<name>/table.md`. `oracle.py`: ORACLE diagnostics (fit started from the truth = positive
  control; the truth through the junction-aware render = the render model's ceiling).
- `check_frozen.py` / `frozen.json`: verifies the frozen files and the dev set.

Report, not frozen: `report_figures.py` draws the annotated scenes of ../REPORT.md (it reads the truth, and
stores each output's digest so the figures can be checked against the held-out JSONs), and `report_html.py`
makes the report one self-contained HTML page.

Diagnostic, not frozen: `residuals.py` (records the residual OD_obs - render at the truth's junctions and
crossings; reads the truth to know where to look) and `determinism.py` (re-runs rows of a saved
run_experiment JSON, each in a fresh process, and compares digests: tier 2's 'deterministic' flag only
compares two runs inside one process). vesselmap's torch.compile is off (`set_threads` sets
VESSELMAP_COMPILE=0): its per-shape cache made an image's floats depend on the images run before it.
`controls.py` re-renders six battery stimuli (empty average / frame, line_d3, line_d6_h0.25, fork_wide,
cross_a45; empty_frame dropped since R2) with imaging seeds 1-7 and reports the mean probe_composite per stimulus: the battery uses ONE
imaging draw (seed 11), so a change that moves probes must also move these controls to count as real
(`python -m experiments.splinefit.controls`, ~4 min).
`tier2diag.py` breaks the whole-image residual down by true vessel-width class and per junction disc and
measures target_keep, the fraction of the true OD the pipeline's own target keeps (the background leak).
Record during tier 2 with `DIAG2_RUN=<name> python -m experiments.splinefit.run_experiment --tier 2
--pipeline experiments.splinefit.tier2diag:annotate_saving` (the output, digests and metric are unchanged),
then `python -m experiments.splinefit.tier2diag --run <name> [--compare <other>]`.

Since batch 4 the fit can re-propose ARMS from its own prediction error (`repropose.py`, after the MDL prune):
neuromimetic stages 3-6 on max(OD - R, 0), kept when novel (outside the render's support) and attached to an
existing edge (to its node, or splitting it), then 40 joint iterations and a second prune; if every arm is
rejected the network reverts to its state before the re-proposal. Since batch 6 it is OFF by default
(`FitConfig.repropose`; provisional): its tier-2 gain was +0.0002 +- 0.0013 and it added false arms to the
multi-seed controls.

Since batch 6 a coarse 'magnocellular' channel (`coarse.py`) cleans the target further: Hessian ridges of
log I at sigma 4-16 px that are elongated and NOVEL (mostly outside stage 1's mask) join the retarget's
vesselness mask, so B is re-estimated on the mask's negative without the deep, blurred vessels that stage 1's
fine-scale mask lets into it (s004 and pathologic target_keep +0.02 to +0.06). It is provisional and off
(`pipeline.Config.coarse`, batch 7): +0.0014 +- 0.0008 over the 5 scenes, and without a cause for the deep
OD it restores, neighbouring vessels absorb it as contrast. The same channel can propose free DEEP edges (no
node where they pass under a sharp vessel; `pipeline.Config.deep`, provisional, off). `paired.py` is the
paired per-SCENE test of program.md's tier-2 confirm rule, with guard metrics (blur, contrast and width
errors and biases) that can veto a confirmation.

## How to run

```
cd /home/user/vesselscene && export OMP_NUM_THREADS=2
python -m experiments.splinefit.check_frozen                        # the evaluator is intact
python -m experiments.splinefit.run_experiment --tier 1 > run.log 2>&1
grep "^composite:\|^fast_composite:\|^probe_score:\|^graph:\|^pos:\|^width:\|^explained\|^seconds:\|^status:" run.log
python -m experiments.splinefit.run_experiment --tier 2 --repeat 2 > run2.log 2>&1
python -m experiments.splinefit.probes run --pipeline experiments.splinefit.pipeline:annotate   # tuning curves
python -m experiments.splinefit.residuals --fast 0 2 8 --probe fork_wide cross_a20 empty_average
python -m experiments.splinefit.determinism --json $SPLINEFIT_WORK/runs/last_tier1.json --rows 0 8
python -m experiments.splinefit.evaluate_fit --tier 2 --rows proposals --pipeline experiments.splinefit.pipeline:annotate --out experiments/splinefit/results/dev/<name>
python -m pytest -q experiments/splinefit/tests
```

Work files (results.tsv, runs, the probe data) live in `$SPLINEFIT_WORK` (default the session scratchpad's
`work/splinefit/`); the dev scenes in `$SPLINEFIT_DEV`. Held-out scenes are refused by every entry point.

## Attribution

`render.py` subclasses and `fit.py` adapts LIMBUS vesselmap (`/home/user/limbus`, `vesselmap/render.py`
NetworkModel and `vesselmap/fit.py` optimize, n_params, MDL pruning; same author), used read-only. The loop
is modelled on Andrej Karpathy's autoresearch (`program.md`, `prepare.py` frozen, `results.tsv`).
