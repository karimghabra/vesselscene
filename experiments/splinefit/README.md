# experiments/splinefit: a jointly fitted, junction-aware spline render of the whole image

The neuromimetic annotator (`experiments/neuromimetic`) **proposes** vessels and junctions; this package turns
the proposal into a spline network and fits its differentiable render, jointly, to the whole image's optical
density. The end product is the network: centreline B-splines with radius r, blur s and contrast a profiles,
nodes at forks and confluences, crossings of vessels at different depths as overlaps without a node. The
work is organised as an *autoresearch* loop (Karpathy): a frozen evaluator, one metric, fixed-budget
experiments that are kept or reset, a results log. The charter every experiment follows is
[program.md](program.md); the history and the findings are in [LOG.md](LOG.md).

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

Diagnostic, not frozen: `residuals.py` (records the residual OD_obs - render at the truth's junctions and
crossings; reads the truth to know where to look) and `determinism.py` (re-runs rows of a saved
run_experiment JSON, each in a fresh process, and compares digests: tier 2's 'deterministic' flag only
compares two runs inside one process). vesselmap's torch.compile is off (`set_threads` sets
VESSELMAP_COMPILE=0): its per-shape cache made an image's floats depend on the images run before it.
`controls.py` re-renders six battery stimuli (empty average / frame, line_d3, line_d6_h0.25, fork_wide,
cross_a45) with imaging seeds 1-7 and reports the mean probe_composite per stimulus: the battery uses ONE
imaging draw (seed 11), so a change that moves probes must also move these controls to count as real
(`python -m experiments.splinefit.controls`, ~4 min).
`tier2diag.py` breaks the whole-image residual down by true vessel-width class and per junction disc and
measures target_keep, the fraction of the true OD the pipeline's own target keeps (the background leak).
Record during tier 2 with `DIAG2_RUN=<name> python -m experiments.splinefit.run_experiment --tier 2
--pipeline experiments.splinefit.tier2diag:annotate_saving` (the output, digests and metric are unchanged),
then `python -m experiments.splinefit.tier2diag --run <name> [--compare <other>]`.

Since batch 4 the fit re-proposes ARMS from its own prediction error (`repropose.py`, after the MDL prune):
neuromimetic stages 3-6 on max(OD - R, 0), kept when novel (outside the render's support) and attached to an
existing edge (to its node, or splitting it), then 40 joint iterations and a second prune; if every arm is
rejected the network reverts to its state before the re-proposal.

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
