# program.md: the research charter of the spline-fit loop

Every experiment agent reads this first and follows it. It adapts Karpathy's *autoresearch* `program.md`
(fixed-budget experiments against a frozen evaluator, one metric, keep or reset, a results log, simpler
wins ties, never stop) to a vision problem, and adds the method of an experimental neuroscientist.

## 1. The goal

Explain each whole vesselscene still with a **spline network** whose differentiable render, fitted jointly
to the entire image, accounts for its optical density, junctions included:

- centreline B-splines, each with radius r, blur s and contrast a profiles along it;
- **nodes** at forks and confluences (degree >= 3), where the lumens are one blood volume: the render is the
  OD of the **union** of the lumens, not the sum;
- **crossings** of vessels at different depths as overlaps **without** a node, where the ODs roughly add
  (vesselscene crossing additivity ~0.87).

The neuromimetic annotator (`experiments/neuromimetic/neuromimetic.py`: `run`, `propose`, `annotate`) is the
mechanism that **proposes** vessels and junctions. The end product is the jointly fitted network
(`pipeline.annotate(image, valid)`), scored by `score.py`.

## 2. The residual rule (must hold everywhere)

> "when computing residuals using the candidate graph's render, make sure to not use the vesselness mask as
> the target. instead, use the vesselness mask's negative to clean up the original image for background,
> since vesselness distorts the target OD of bifurcations and crossings, causing erroneous fits."

The fit target is the image's own optical density **OD = B - log I**, with the background B estimated only on
pixels **outside** a vesselness mask (neuromimetic stage 1, `photoreceptors()`; its leak of faint and wide
vessels into B may be improved, e.g. by re-estimating B outside the fitted render's support). Never fit to a
vesselness map, an orientation map or a mask. The pipeline reads **only** `image` and `valid`.

## 3. What you may edit, and what is frozen

**Editable (the pipeline):** `proposals.py`, `render.py`, `fit.py`, `pipeline.py`, and new pipeline modules
in `experiments/splinefit/` (e.g. a re-proposal module). Also `residuals.py` (a diagnostic) and the docs
(`README.md`, `LOG.md`). The pipeline's entry point stays `experiments.splinefit.pipeline:annotate`.

**Frozen (the evaluator; like autoresearch's `prepare.py`).** Never edit these in an experiment. Only a
confirmed bug fix may change one, in its own commit, followed by a logged re-baseline (LOG.md and a
`rebaseline` row in results.tsv). The hashes are git blob hashes (`git hash-object <file>`) at the freeze
commit; check them with `python -m experiments.splinefit.check_frozen` (exit 1 on any mismatch).

| file (experiments/splinefit/) | git blob hash | role |
|---|---|---|
| `score.py` | `165399683b7cae171ceef0de904c3623d1ff87ab` | the metric (graph, geometry, render, composite) |
| `run_experiment.py` | `e36f7e9f53dc35e11fadd77718a4b9c0cb142b35` | one experiment, tier 1 / tier 2 (prints the metric) |
| `probes.py` | `b58c61739ab4c91eb3432981c11ea859b8fe0a96` | the psychophysics battery: stimuli, scoring, tuning curves |
| `probe_battery.json` | `50897efcaae300e2563d112bd7cdf49caa31480b` | the battery's manifest (sha256 of every probe image and truth) |
| `fastset.py` | `6b610f0186ad4bfbcf93539bdc9fa2bec34fcf74` | selection of the fast crops |
| `fastset.json` | `888a1bc7e80cad79e29387985db08926510706e2` | the 10 fast crops |
| `evaluate_fit.py` | `7e0a01dbed84c070b13c14c46fa55e09a5404960` | reference rows and tables |
| `oracle.py` | `4e433437f4a11e477551dcf6037201d28785a68e` | ORACLE diagnostics (positive control, render ceiling) |
| `check_frozen.py` | `eaf43664cd4cadf389ed54d50dfabb36c7ded144` | this check |
| `tests/test_score.py` | `9fda445f6f78015dd70921730f69b387b0612fbc` | the scorer's known answers |
| `tests/test_probes.py` | `0e202d9d66ee42559ee4372e549a2f1253ef363d` | the probe helpers' known answers |
| `tests/test_loop.py` | `5e857a95c4011cfcad7770db310339f5bbdc993e` | the tier-1 arithmetic |
| dev set (5 scenes, every file) | `b9f9795b33357754` (sha256 digest, check_frozen) | the scenes the tiers read |

Also frozen: the dev set (`$SPLINEFIT_DEV`, the scratchpad's `scenes/dev/`: healthy_s000, healthy_s004,
healthy_s007, healthy_s008, pathologic_s000), the probe data (regenerated deterministically from probes.py
into `$SPLINEFIT_WORK/probes/` and verified against `probe_battery.json`), and LIMBUS vesselmap
(`/home/user/limbus`, read-only: subclass or copy it, never modify it). Never edit `experiments/neuromimetic`
files (except a documented real bug fix) or the `vesselscene/` package.

**Never** open, list, glob or score anything under the scratchpad's `heldout/` directory.

## 4. The metric

`run_experiment` prints, grep-able:

- tier 1 (the loop's tier): `composite = 0.5 * fast_composite + 0.5 * probe_score`;
  `fast_composite` = mean `score.composite` over the 10 fast crops of fastset.json (256 x 256 px, one per dev
  scene and kind); `probe_score` = mean `probe_composite` over the 34-stimulus probe battery;
- `score.composite = 0.5 * graph_score + 0.5 * mean(pos, width, explained)`, where graph_score =
  mean(centreline_f1, junction_f1_strict, junction_type_balanced_coarse, edge_cover), pos = 1 - min(1,
  offset_px / 3), width = 1 - min(1, |r_fit - r_true| / r_true), explained = 1 - SSR / SS of the render
  against the oracle-background OD within 3 px of the observable lumens;
- also every part (`graph`, `pos`, `width`, `explained`, `explained_junction`), the probe thresholds
  (`probe_<sweep>: X`) and sweep means (`probe_mean_<sweep>: X`), `seconds` and `digest`.

Higher is better. Tier 2 (confirmation): every dev image, both kinds, twice; composite = mean image
composite; `deterministic: True` is required.

## 5. The loop

Set up once per session: `cd /home/user/vesselscene`, `export OMP_NUM_THREADS=2`, read README.md, LOG.md and
the tail of results.tsv. Find the last kept commit (the last `keep` row) and the last tier-2-confirmed commit
(the last `confirm` row). Then LOOP FOREVER (until the batch's experiment budget is spent):

1. **Pre-register.** Look at the latest residual recordings and probe tuning curves. Write in LOG.md, before
   touching code: the hypothesis (a mechanism, e.g. "the stage-1 background leaks the wide lumens, so the
   fit under-predicts junctions"), the change, and the **predicted direction of each metric part**
   (graph, pos, width, explained, explained_junction, probe_score and the sweeps it should move) and why.
2. **Edit** the pipeline: one variable at a time.
3. **Commit** (`git commit` with a one-line description; the message ends with the two attribution lines).
4. **Run:** `python -m experiments.splinefit.run_experiment --tier 1 > run.log 2>&1` (redirect everything;
   do not flood the context).
5. **Read:** `grep "^composite:\|^fast_composite:\|^graph:\|^pos:\|^width:\|^explained\|^probe_\|^seconds:\|^status:\|^digest:" run.log`.
   No `composite:` line, or `status: crash`: `tail -n 50 run.log` and see the crash policy.
6. **Decide.** Keep if the tier-1 composite improves on the last kept one by **>= 0.002**, or is equal
   (within 0.002) and the code is simpler. Otherwise `git reset --hard <last kept commit>` (only commits made
   since the last push; never rewrite pushed history).
   **Veto (batch 6, review):** a change that adds or removes edges, or moves any probe, must also run the
   multi-seed controls (`python -m experiments.splinefit.controls --json ...`, 3-4 min); if its
   controls_score falls below the last kept state's, or the empty controls gain false length, it is
   discarded whatever tier 1 says (the single-seed battery under-states false alarms). Check also that a
   tier-1 gain does not rest on one crop (`python -m experiments.splinefit.paired A.json B.json`: the
   leave-one-out range must stay above 0).
7. **Log** one row in `$SPLINEFIT_WORK/results.tsv` (tab-separated, never committed; LOG.md carries the
   summary), and in LOG.md record whether each prediction held.
8. **Record** (kept changes): `python -m experiments.splinefit.residuals --fast 0 2 8 --probe fork_wide cross_a20 empty_average`
   and look at the residuals at junctions and crossings; note what changed in LOG.md.
9. Every ~4 kept experiments, and at the end of each batch: **tier 2**
   (`python -m experiments.splinefit.run_experiment --tier 2 --repeat 2 > run2.log 2>&1`, ~30 min, run in the
   background while you think). Compare it PAIRED, per image, with the last confirmed run
   (`python -m experiments.splinefit.paired <last confirmed tier2.json> <new tier2.json>`):
   - mean > 2 SE: a real gain; log a `confirm` row;
   - within +-2 SE (non-inferior): keep only if the batch's kept changes are simplifications (equal or less
     code); added code that is only non-inferior on whole images is PROVISIONAL: turn it off by default (or
     reset) and log a `regress` row saying so;
   - mean < -2 SE, or regressed: reset to the last tier-2-confirmed commit; log a `regress` row. Push kept, confirmed work
   (`git push -u origin claude/elegant-johnson-8n78yv`, retry 4x with 2/4/8/16 s backoff on network errors).

results.tsv columns (tab-separated; the header is the first line):

```
commit	tier	composite	graph	pos	width	explained	explained_junction	probe_score	seconds	status	description
```

`commit` is the short hash (7 chars); `tier` 1 or 2; the metrics with 6 decimals (0 on a crash); `status` is
`keep`, `discard`, `crash`, `confirm` (a tier-2 row that confirms) or `regress` (a tier-2 row that does not),
or `baseline` / `rebaseline`; `description` is `hypothesis | prediction | held?` (no tabs).

**Budget.** Tier 1 must finish within 6 minutes on 2 threads (it takes ~4.5 min at v0). A tier-1 run that
exceeds 10 minutes is killed by run_experiment and counts as a crash; a change that makes the pipeline
slower must pay for itself in score. Tier 2 has a 90-minute limit.

**Crash policy.** A typo or a trivial bug in your own change: fix it and re-run (same experiment). A deeper
failure (numerical blow-up, a broken idea, out of time): log `crash`, reset, move on.

**Simplicity criterion.** All else equal, simpler is better. A gain of 0.002 that adds 30 lines of special
cases is not worth keeping; deleting code at an equal score is a win. Weigh complexity against the size of
the gain.

**Determinism.** No unseeded randomness, fixed iteration counts, stable tie-breaks,
`torch.use_deterministic_algorithms(True)` (set_threads does it). A sham change (a comment, a refactor) must
reproduce the tier-1 `digest` exactly; tier 2 must print `deterministic: True`.

**Never stop** inside a batch: when out of ideas, re-read the residuals and the probe curves, the oracle rows,
NOTES_neuromimetic.md and LIMBUS vesselmap's fit.py, combine near-misses, or try something more radical.

## 6. Think like a neuroscientist

- **Predictive coding (Rao and Ballard).** The render is the top-down prediction; the residual is the
  prediction error that drives every update. Precision weighting is the noise-normalised residual (the
  texture-dominated local sigma); explaining away is the joint fit (two vessels that overlap must share the
  error, not both claim it); coarse-to-fine is the cortical hierarchy (fit position and topology at coarse
  blur first, then widths and profiles); attention goes where error remains: re-propose vessels and junctions
  from the residual, not from the original image.
- **Psychophysics.** The probe battery (probes.py) is a frozen set of canonical stimuli with parametric
  sweeps (vessel width, blur, contrast; Y-forks; X-crossing angle, depth gap, unequal widths; T and pseudo-T;
  parallel pairs at shrinking gaps; ends; red-cell gaps in a frame; empty backgrounds). Read the tuning
  curves and thresholds (`probe_<sweep>`), not only the score: a change should move the thresholds it
  targets and leave the others.
- **Lesions and controls.** Change one variable at a time. Positive control: the oracle init
  (`python -m experiments.splinefit.evaluate_fit --rows --oracle-fit experiments.splinefit.oracle:fit`), the
  fit started from the true network; a fitter that is right keeps the truth's geometry. Negative control:
  the empty probes (nothing must be fitted). Sham: a no-op change must reproduce the baseline digest. A
  lesion (turning a component off, e.g. `pipeline:annotate_additive`) tells what the component buys.
- **Pre-registration.** Write the prediction first, then run; afterwards say whether it held. A surprise
  is information about the system: update the model of what the fitter does, not just the code.
- **Recording.** For every kept change look at the residual images at junctions and crossings
  (`residuals.py`): the sign and shape of the error at a fork (over-predicted centre = additive overlap;
  under-predicted = background leak) or a crossing tells which mechanism is wrong.

## 7. Where things are

- Code: `/home/user/vesselscene/experiments/splinefit/` (README.md: method and how to run; LOG.md: the
  baseline and every batch).
- Work: `$SPLINEFIT_WORK` = `/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/scratchpad/work/splinefit/`
  (results.tsv, runs/, probes/). Scratch files go there, never into the repo.
- Reference tables: `results/dev/baseline/table.md` (proposals, vesselmap, the fit, the truth rows and the
  oracle rows on tier 2), `results/dev/baseline/tier1.md` and `tier2.md` (the v0 run_experiment outputs).
