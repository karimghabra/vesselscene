# Neuromimetic vessel annotation: an experiment

**Question.** Can lessons from how the retina and early visual cortex detect contrast, edges and contours,
and from diffusion models (Stable Diffusion), give a deterministic annotator whose output matches the vessel
graph of a vesselscene still better than curvature (Hessian) analysis?

**Answer, on six held-out scenes: yes, for lines, junctions and crossings; only weakly for junction types.**
- **Lines and junctions.** The neuromimetic annotator beats both curvature annotators on every one of the 12
  held-out images (6 scenes, average and frame) in centreline F1 and strict junction F1. It also beats them on
  edge cover on 23 of 24 image pairs and on purity on 24 of 24.
  - The two curvature annotators are a tuned Hessian ridge + skeleton pipeline and LIMBUS vesselmap itself.
- **Crossings.** It finds 25-32 % of the observable crossings; the curvature annotators find 5-12 %.
- **Junction types.** Telling a crossing from a fork from a compound beats a constant majority label by
  only 0.05-0.10. That margin depends on a prior fitted on the development scenes.
- **What did the work.** Most of the gain comes from a full orientation score, instead of one Hessian
  orientation per pixel, read out by tracing in (x, y, theta). Contour integration (the association field)
  and surround suppression make the annotator robust on noisy single frames.
- **What did not.** The render-and-residual refinement borrowed from diffusion models is neutral, and
  re-proposing vessels from the residual hurts.
- **Determinism and speed.** Every output is bit-identical across runs, and an image takes about 8 s on 2
  CPU threads. vesselmap takes about 10 minutes per image.

![Held-out frame: truth, Hessian, vesselmap, neuromimetic](figures/heldout_frame_comparison.jpg)

*A held-out single frame (healthy seed 1, 2x zoom). Panels: observable truth (green lines, rings at
junctions), the Hessian baseline, vesselmap, and the neuromimetic annotator. Each traced edge has its own
colour. Junctions: cyan circle = 3-way, magenta X = crossing, yellow square = compound.*
- The Hessian traces both walls of the wide vessel and short texture lines.
- vesselmap's splines loop.
- The neuromimetic annotator follows the wide vessel's axis and keeps thin vessels as their own lines, but
  misses part of a faint vessel.

## What was built

[DESIGN.md](DESIGN.md) gives the design, and [NOTES_neuromimetic.md](NOTES_neuromimetic.md) gives each
stage's biology and maths, the parameters, the development log and what failed. In short:

| stage | biology | algorithm (`neuromimetic.py`) |
|---|---|---|
| 1 | photoreceptors, horizontal cells | log image; background fitted only outside the vesselness mask; target OD = background - log I |
| 2 | OFF-centre ganglion cells, contrast gain control | centre-surround bands of OD divided by their local noise: a CNR (used for the mask and a visibility floor) |
| 3 | V1 simple cells | even and odd oriented filters, 16 orientations, 9 scales in three spatial-frequency channels; the full orientation stack is kept |
| 4 | non-classical surround | subtract the isotropic part of the orientation tuning; flank suppression by the weaker side |
| 5 | horizontal connections (association field, bipole cells) | three deterministic steps of collinear completion in (x, y, theta): gaps fill, line ends do not grow |
| 6 | readout | non-maximum suppression across space and orientation; tracing along the tangent in SE(2), so a line passes through a crossing in its own orientation layer |
| 7 | end-stopped cells | line ends on another line make a 3-way junction, two lines through each other a crossing, more arms a compound |
| 8 | diffusion-style refinement | render the graph in OD, residual against the cleaned OD, prune edges that do not pay for themselves (MDL) |

**The residual rule.** Wherever a render of the candidate graph is compared with the image, the target is
the image's own optical density.
- The background is fitted on the vesselness mask's negative, and that background cleans the original image
  into OD = B - log I.
- A vesselness or orientation map is never a target.
- An adversarial review checked this in the code: every residual and profile fit receives the very array
  stage 1 produced.

## How it was tested

- **Scenes** (`make_scene`, CPU, 480 x 768 px, k = 4):
  - Development: healthy seeds 0 and 4 (seed 4 at 320 x 512) and pathologic seed 0. All tuning used only
    these.
  - Held out: healthy seeds 1, 2, 3, 5 and 6, and pathologic seed 1.
  - Each scene gives two images, the averaged still and a single frame. The 12 held-out images hold 500 + 417
    observable junctions.
- **Scoring** (`harness.py`, against each image's observable truth):
  - Centreline recall, precision and F1 within 3 px.
  - Junction F1, plain and strict (match radius at most 2 lambda).
  - Coarse junction type (3-way / crossing / compound) with two references: a constant majority label and
    balanced accuracy, where a constant label scores 0.33.
  - Crossing recall and precision.
  - Edge cover (does one traced edge cover each true edge?) and purity (does a traced edge stay on one true
    edge?), on polylines cut at their junctions.
  - Runtime, and a determinism digest over two runs.
- **Comparisons:**
  - `baseline_hessian.py`: the curvature pipeline (vesselmap's own ridge measure, skeleton graph,
    arm-count typing), tuned on the development scenes.
  - `baseline_vesselmap.py`: LIMBUS `vesselmap.build_map` at its default config, the current curvature-based
    annotator of the LIMBUS project. The adapter only exports its network.
- **Review.** An agent built each annotator and another tried to refute it. The harness, both baselines and
  the neuromimetic annotator were fixed after review; the review findings are in the commit messages and
  NOTES.

## Results on the held-out scenes

Means over the 6 held-out scenes. The ceiling is the truth's own graph scored by the same harness.

| average still | line R | line P | line F1 | junc F1 | junc F1 strict | crossing R / P | coarse type (majority ref) | coarse balanced | edge cover | purity | s / image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ceiling | 0.96 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 (0.47) | 1.00 | 1.00 | 1.00 | - |
| **neuromimetic** | 0.76 | **0.93** | **0.83** | **0.70** | **0.68** | **0.32** / 0.59 | **0.56** (0.46) | **0.60** | **0.70** | **0.91** | 8 |
| Hessian | 0.59 | 0.74 | 0.66 | 0.56 | 0.50 | 0.10 / 0.61 | 0.47 (0.45) | 0.51 | 0.54 | 0.79 | 1 |
| vesselmap | 0.77 | 0.56 | 0.64 | 0.51 | 0.44 | 0.12 / 0.25 | 0.33 (0.48) | 0.37 | 0.62 | 0.77 | 697 |

| single frame | line R | line P | line F1 | junc F1 | junc F1 strict | crossing R / P | coarse type (majority ref) | coarse balanced | edge cover | purity | s / image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ceiling | 0.97 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 (0.46) | 1.00 | 1.00 | 1.00 | - |
| **neuromimetic** | 0.75 | **0.89** | **0.81** | **0.66** | **0.64** | **0.25** / **0.50** | **0.50** (0.45) | **0.50** | **0.69** | **0.89** | 8 |
| Hessian | 0.65 | 0.48 | 0.55 | 0.47 | 0.41 | 0.05 / 0.19 | 0.45 (0.46) | 0.39 | 0.52 | 0.83 | 2 |
| vesselmap | 0.72 | 0.71 | 0.72 | 0.51 | 0.44 | 0.10 / 0.31 | 0.34 (0.48) | 0.39 | 0.65 | 0.74 | 534 |

**Paired per image.** Each cell is the mean difference, neuromimetic minus baseline, with the number of
images where the neuromimetic annotator is higher:

| neuromimetic minus | line F1 | junc F1 strict | crossing R | coarse balanced | edge cover | purity |
|---|---|---|---|---|---|---|
| Hessian | +0.22 (12/12) | +0.21 (12/12) | +0.21 (11/12) | +0.10 (9/12) | +0.16 (12/12) | +0.09 (12/12) |
| vesselmap | +0.14 (12/12) | +0.22 (12/12) | +0.18 (11/12) | +0.17 (11/12) | +0.06 (11/12) | +0.14 (12/12) |

Against vesselmap, line recall is a tie (+0.01, higher on 6 of 12 images). The gain is precision and graph
structure, not more vessels found.

All tables, per scene and per kind, are in [results/heldout](results/heldout):
- `table_all.md`: means and ranges of every annotator and variant;
- `paired.md`: per-image differences by kind;
- the JSONs: every row, with the type confusion counts.

## Which stages matter

Held-out means; each pair is average / frame:

| variant | line F1 | line P | junc F1 | junc P | crossing R | coarse balanced |
|---|---|---|---|---|---|---|
| full (stages 1-8) | 0.83 / 0.81 | 0.93 / 0.89 | 0.70 / 0.66 | 0.80 / 0.78 | 0.32 / 0.25 | 0.60 / 0.50 |
| no refinement (stages 1-7) | 0.84 / 0.81 | 0.92 / 0.87 | 0.71 / 0.65 | 0.78 / 0.71 | 0.34 / 0.30 | 0.58 / 0.51 |
| no association field | 0.84 / 0.78 | 0.91 / 0.80 | 0.71 / 0.61 | 0.78 / 0.58 | 0.39 / 0.18 | 0.60 / 0.47 |
| no surround | 0.83 / 0.81 | 0.91 / 0.88 | 0.70 / 0.64 | 0.76 / 0.70 | 0.34 / 0.26 | 0.58 / 0.51 |
| V1 only (stages 1-3, 6-7) | 0.84 / 0.64 | 0.86 / 0.54 | 0.71 / 0.39 | 0.70 / 0.27 | 0.39 / 0.25 | 0.56 / 0.46 |
| re-proposal from the residual | 0.79 / 0.79 | 0.80 / 0.83 | 0.56 / 0.63 | 0.46 / 0.59 | 0.30 / 0.26 | 0.53 / 0.48 |
| background leak fixed (`bg_traced`) | 0.84 / 0.80 | 0.91 / 0.86 | 0.72 / 0.64 | 0.75 / 0.65 | 0.42 / 0.24 | 0.57 / 0.50 |
| no compound-radius prior | 0.83 / 0.81 | 0.93 / 0.89 | 0.70 / 0.66 | 0.80 / 0.78 | 0.37 / 0.28 | 0.55 / 0.46 |

- **The orientation score and its readout are the main gain.** V1 only (stages 1-3, then the readout and
  graph logic of stages 6-7) already matches the full annotator on averaged stills, and beats both curvature
  annotators there.
  - Separate fine, medium and coarse channels let a thin vessel crossing a wide one survive.
  - An odd-symmetric partner suppresses echoes along vessel walls.
  - Tracing in (x, y, theta) carries a line straight through a crossing in its own orientation layer.
- **Context is what makes it robust.** On single frames (red-cell gaps, grain), V1 alone fires on texture:
  junction precision 0.27.
  - With surround, association field and refinement, it is 0.78. The full annotator is higher on 6 of 6
    frames, by +0.52 on average.
  - The association field alone is worth +0.20 junction precision on frames (6 of 6).
  - This is contour integration doing what it is thought to do in vision: a line is salient by its
    collinear support, so isolated texture loses.
  - The figure below shows it.
- **The diffusion-style refinement is neutral.** Rendering and pruning trade a little recall for a little
  precision.
  - Re-proposing vessels from the residual, the full "denoising loop", adds false T arms and lowers junction
    precision by 0.34 on averages.
  - The useful lesson from diffusion models was the iterative, deterministic refinement of a field towards
    a prior. That is what the association field does in (x, y, theta), as a few steps of a
    completion-field diffusion.

![Held-out frame: V1 only against the full annotator](figures/heldout_frame_ablation.jpg)

*The same held-out frame. Left: truth. Middle: V1 only, the orientation score without contextual stages,
fires on texture. Right: the full annotator.*

## The residual rule and the background

- **The rule holds in the code.** The review confirmed that every residual is OD_target - render, with
  OD_target from the background fitted outside the vesselness mask.
- **The mask misses faint vessels.** 5-32 % of observable centreline pixels fell outside it on the
  development scenes. The background was then pulled towards them, and their OD attenuated to 0.56-0.70 of a
  leak-free refit.
  - `bg_trace_k` grows the mask from the traced lumens and refits. It removes most of the leak.
  - On the held-out averages it raises crossing recall from 0.32 to 0.42, higher on 6 of 6 images. On frames
    it costs junction precision (0.78 to 0.65).
  - It is off by default because it did not help on development. The crossing gain suggests the leak matters
    where the residual rule says it does, at junctions.
- **The crossing knot cue did not work, and the reason points at the render.** In OD, two vessels crossing
  should add their densities, so a crossing shows a darker knot than a fork.
  - The residual at junction centres did not separate the types: the additive render over-predicts every
    junction centre by about twice the thinner vessel's contrast.
  - vesselscene's own image formation explains why. Its chord law T = f + (1 - f) exp(-mu h c) saturates, so
    OD is not additive at dark knots. Lumens meeting at a node are drawn as a union, and crossings are
    composited by depth.
  - A junction-aware render (a union at nodes, the saturating chord law at crossings) is the next step for
    typing from the residual.

## Junction types

- **Bifurcation versus confluence** needs the flow direction, which a still does not show. Every 3-way
  junction is called 'pseudo-T', the commonest 3-way type.
- **The coarse type** (3-way, crossing or compound) beats the image's majority label only modestly: 0.56
  against 0.46 on averages and 0.50 against 0.45 on frames, above it on 4 of 6 images of each kind.
  - That margin comes from one rule fitted on the development truth: a junction region at least 16 px in
    radius is a compound. Without it, coarse accuracy is 0.48 / 0.45, at the reference.
  - The rule held up on the held-out scenes. With it, coarse accuracy is higher on 5 of 6 averages and 3 of
    6 frames.
- **Balanced accuracy** is 0.60 / 0.50 (Hessian 0.51 / 0.39, vesselmap 0.37 / 0.39).
- **Typing is detection-limited.** On the development scenes, a truth compound typically had 5-6 lines, of
  which 3-4 were traced.

## Determinism and speed

- **Determinism.** Every neuromimetic variant gave identical output digests over two runs on all 12 held-out
  images. The review also checked separate processes and 1, 2 and 4 threads. There are no random numbers,
  iteration counts are fixed, and sorts break ties by coordinate.
  - vesselmap is not bit-reproducible: two builds of one development image differed by 0.13 in junction
    recall. Its rows come from one build per image.
- **Speed** (480 x 768 px on 2 threads, on a 4-core machine shared with other jobs; vesselmap's times are
  inflated by that):
  - neuromimetic: about 8 s per image, and 70 s for a 1920 x 1200 frame;
  - Hessian: 1-2 s;
  - vesselmap: 426-1161 s. Its own README gives about 40 min for 1920 x 1200 on 4 cores.

## Caveats

- **Synthetic only.** The scenes are vesselscene's, with its known realism gaps (README, Limitations).
  Nothing here has been measured on real stills.
  - The thresholds are in CNR units comparable to the truth's, but they should be recalibrated on real
    stills.
- **A small test set.** Six held-out scenes, only one of them pathologic, and 480 x 768 crops of the 1920 x
  1200 frame.
  - On the pathologic held-out frame, the Hessian's junction F1 (0.61) beats the neuromimetic's (0.58).
- **vesselmap outside its design.** Its README says the low-noise averaged still is the wrong input: texture
  passes its noise-relative tests and is traced as vessels. Its averaged-still rows therefore understate it.
  On frames, its intended input, its line F1 (0.72) is the better of the two baselines, but still below the
  neuromimetic's 0.81.
- **A held-out leak, contained.** A baseline reviewer once globbed the scene folders and read the junction-type
  counts of two held-out scenes (healthy seeds 1 and 2). No annotator was run or tuned on a held-out image.
  - Two fresh scenes (seeds 5 and 6) were then rendered and all held-out scenes moved out of the agents'
    reach.
  - On the four scenes no agent ever touched, the neuromimetic annotator still wins line F1, strict junction
    F1, crossing recall and edge cover on 4 of 4 scenes in both kinds. The one exception is plain junction F1
    on frames against the Hessian (3 of 4).
- **Development-fitted priors.** Thresholds and the compound-radius rule were tuned on three development
  scenes. Their held-out behaviour is reported above.

## What to try next

1. **A junction-aware render for the refinement**: a union at nodes and the saturating chord law at
   crossings. This would make the residual a typing cue, which is where the residual rule should pay off.
2. **A better background mask**: the association field's output with a lower threshold, or the traced lumens
   (`bg_trace_k`). Faint vessels would then stop leaking into the background.
3. **A hybrid with vesselmap.** Use the orientation score and tracker as vesselmap's proposal stage instead
   of Hessian ridges, and keep vesselmap's spline fit for widths, blur and contrast. The proposals would
   then be deterministic, and crossings would survive into the fit.
4. **Calibration on real stills** from LIMBUS, with the truth's CNR scale as the common unit.

## Reproduce

```bash
pip install -e ".[test]"   # the baselines also need a LIMBUS checkout: $LIMBUS_DATA, or cloned next to this repository as ../limbus
python -m vesselscene scene --seed 1 2 3 5 6 --preset healthy --shape 480,768 --device cpu -o scenes
python -m vesselscene scene --seed 1 --preset pathologic --shape 480,768 --device cpu -o scenes
python -m experiments.neuromimetic.evaluate --scenes scenes/healthy_s001 ... --out results \
    --annotators experiments.neuromimetic.neuromimetic:annotate experiments.neuromimetic.baseline_hessian:annotate \
                 experiments.neuromimetic.baseline_vesselmap:annotate \
    --no-repeat experiments.neuromimetic.baseline_vesselmap:annotate
python -m experiments.neuromimetic.figures scenes/healthy_s001 --kind frame --crop 140,170,200,300 --zoom 2 \
    --annotators experiments.neuromimetic.neuromimetic:annotate -o fig.png
```

Files:
- `harness.py`: the scorer.
- `evaluate.py`: several annotators into one table.
- `figures.py`: the figure panels.
- `neuromimetic.py`: the annotator. Entry points: `annotate`, plus the ablations `annotate_no_verify`,
  `annotate_no_association`, `annotate_no_surround`, `annotate_v1_only`, `annotate_repropose`,
  `annotate_bg_traced` and `annotate_no_compound_prior`.
- `baseline_hessian.py`, `baseline_vesselmap.py`: the baselines.
- `results/heldout/`: every held-out score.
