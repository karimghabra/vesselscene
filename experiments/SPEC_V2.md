# Tech spec: a V2 layer for the vessel annotator (junction cells, good continuation, transparency, grouping) and a motion channel

| | |
|---|---|
| status | proposed |
| author | Claude, for Karim Ghabra |
| date | October 2026 |
| code | `experiments/neuromimetic/` (proposer), `experiments/splinefit/` (fit, evaluator) |
| background | [PIPELINE.md](PIPELINE.md) (how every step works), [REPORT.md](REPORT.md) (results) |

## 1. Summary

The annotator's neural layers stop at the level of primary visual cortex (V1). They find oriented line segments
at several sizes (steps 3-4) and link collinear ones (step 5). Everything above that is done with symbolic rules:

- tracing vessels (step 6);
- finding and typing junctions from where traces end (step 7);
- deciding which arms continue each other (step 9).

Junctions are where the system fails. Over the 500 junctions of the six held-out images, 61 % are found, 34 %
get the right type, and 21 % are found, correctly typed and rendered to 95 %. Only 38 % of 3-way junctions are
found.

This spec adds the computations the visual system performs in V2, between V1 and shape-level areas:

1. **End-stopped cells:** a dense map of line terminations, computed from the existing association field.
2. **Junction cells:** at every pixel, the set of arms meeting there, from line ends and passing lines; a
   junction's presence and type follow from its arms, with no tracing needed.
3. **Good continuation:** which arms continue each other, from turn and width.
4. **Transparency:** fork or crossing, from the optical density (OD) at the centre against the arms'. This is
   the cue the visual system uses at X-junctions. Measured on the OD itself, not on a render.
5. **Local model comparison:** for junctions the cues leave ambiguous, draw each hypothesis with the existing
   junction-aware renderer and keep the one that explains the image best (top-down feedback).

Later phases add:

6. a **surround fix** that stops penalising crossings;
7. **curvature cells**;
8. **grouping by label spreading**, so vessels keep their identity through junctions;
9. a **motion channel** for real bursts. Flowing red cells make the vessel lumen flicker over time, a vessel
   cue independent of darkness.

Everything stays deterministic and hand-built. The goal of phases 1-3 is to raise junction detection and typing
on the development scenes without more false junctions, then confirm on the evaluator's paired test.

## 2. Problem and evidence

### 2.1 Junctions are where the graph is lost

Per-junction scorecard of the fitted output, the 6 held-out averages (`junction_gallery.py`; found = a junction
within the strict radius of 23.8 px):

| junction type | n | found | right type | render >= 95 % | all three |
|---|---|---|---|---|---|
| 3-way | 176 | 38 % | 24 % | 52 % | 11 % |
| crossing | 135 | 73 % | 36 % | 53 % | 23 % |
| compound | 189 | 74 % | 42 % | 54 % | 30 % |
| all | 500 | 61 % | 34 % | 53 % | 21 % |

### 2.2 Why: junctions are built from traces

Step 7 makes a junction only where a trace ends on another trace or two traces cross, and types it by counting
the traces that leave its disc ([PIPELINE.md](PIPELINE.md), step 7). So:

- **A 3-way junction needs a trace to reach its parent.** If the branch's trace stops short, never reaches the
  parent, or is pruned, the junction is lost.
- **Typing is detection-limited** (NOTES_neuromimetic.md). Truth compounds have 5-6 lines and tracing finds 3-4.
  Most crossings typed 3-way are missing their fourth arm.
- **Fork against crossing is never tested.** Four arms that pair into two straight lines make a crossing; nothing
  checks the image at the centre.

![step 7 mechanism](splinefit/results/pipeline/mech7_junctions.jpg)

*Step 7 today: events from traces, a disc, arms counted from traces.*

### 2.3 What was tried and must not be repeated

| attempt | result | what this spec does differently |
|---|---|---|
| the "knot" cue: residual of the additive render at junction centres | did not separate types (medians -2.4 crossings, -1.8 compounds, -1.4 3-ways, in thinner-arm contrasts): the additive render over-predicts every junction | measure transparency on the OD itself, not on a render's residual |
| merging nearby junctions, counting all traces through a region, a crowding rule | traded junction recall for typing | arms from the orientation field, not from traces |
| weaker cross-orientation suppression (cross_k 0.5) | within noise | change which part of the tuning curve counts as orientation-blind (section 4.6), measured on crossings first |
| re-proposing traces from the residual | lowered every score | feedback only at junctions, as a choice between hypotheses |
| linking pieces after tracing (E22) | lent length to false pieces | continuation decided at junction cells, inside grouping |
| a background mask from OD amplitude (E24) | also took dark texture, biased B | motion as the extra mask cue (section 4.9): independent of darkness |

### 2.4 The surround penalises crossings

![step 4 mechanism](splinefit/results/pipeline/mech4_surround.jpg)

*Step 4 on four pixels of the development image: a blob loses half its response (2.7 to 1.4), a vessel 4 %
(13.5 to 13.0), a crossing 44 % (14.7 to 8.3). A crossing's two lobes raise the "orientation-blind" part.*

## 3. Goals and non-goals

**Goals (phases 1-3).**
- Raise the share of true junctions found, above all 3-way junctions.
- Raise the share typed correctly (3-way, crossing, compound).
- Do this without more false junctions (on the probes and on empty background).
- Keep the pipeline deterministic, hand-built and interpretable: every new map is a neural population with a
  stated tuning, and every parameter is set by hand or swept on development scenes only.

**Goals (phases 4-5).** Vessels with identity through junctions (continuous vessels); a darkness-independent
vessel cue on real bursts.

**Non-goals.**
- Training a network (hGRU, CNN). Section 8 says why.
- Changing the spline fit (steps 10-12) or the evaluator.
- Bifurcation against confluence (needs flow direction; section 10 notes how bursts could give it).
- The 95 % per-junction bar itself: this spec targets detection and typing, the first two of its three tests.

## 4. Design

### 4.1 Overview

```
image -> [1 retina: OD, background] -> [2 ganglion cells] -> [3 simple cells U(x, y, theta, sigma)]
      -> [4 surround (4.6 fix)] -> [5 association field C, lobes A+ A- (4.7 curvature)]
                                         |
             +---------------------------+---------------------------+
             v                                                       v
   [V2a end-stopped cells E(x, y, direction)]               [6 tracing -> 4.8 grouping]
             v                                                       |
   [V2b junction cells: arms, strength, type] <-- arm widths (OD)    |
             |        ^                                              |
             |   [V2c good continuation]   [V2d transparency (OD)]   |
             |        ambiguous --> [V2e local model comparison]     |
             v                                                       v
   [7' graph: traces cut at junction cells; types and arm pairings from V2] -> [8 prune] -> [9-12 fit]

[4.9 motion] registered burst -> flicker map -> step 1 mask, step 3 drive
```

All V2 maps are computed per channel on the channel's grid, like steps 3-5, and brought to the full grid.

### 4.2 V2a. End-stopped cells

**Biology.** End-stopped (hypercomplex) cells respond to a line that ends inside their receptive field; the
standard model takes a line detector's response minus the response displaced along the line beyond the end
(Heitger, Rosenthaler, von der Heydt, Peterhans and Kübler 1992). They are the complement of the bipole cells of
step 5, which need support on both sides.

**Computation.** Step 5 already computes, for every layer k, the lobes ahead and behind on its final iteration:
`A+_k(p)` and `A-_k(p)` (weighted means of the pooled response along `+t_k` and `-t_k`). Then, for the two
directions a line in layer k can end:

```
E_fwd[k](p) = max(0, min(C[k](p), A-[k](p)) - beta * A+[k](p))     # line here and behind, none ahead:
                                                                    # an end at p, free side +t_k
E_bwd[k](p) = max(0, min(C[k](p), A+[k](p)) - beta * A-[k](p))     # free side -t_k
```

This gives 32 direction channels (16 orientations x 2), an end's free direction covering 360 degrees.
`beta` = 1 to start. A gate `C[k](p) > t_low` keeps ends on lines only.

**Output.** `ends[c]`, shape (32, H, W) per channel c, in contrast-to-noise units.

**Cost.** Uses step 5's lobes; about 2 extra correlations per layer per channel (step 5 does 2 per iteration).

### 4.3 V2b. Junction cells

**Biology.** V2 has cells selective for angles and junctions (Ito and Komatsu 2004): they combine line
terminations and passing lines at different orientations around one point.

**Computation.** For each pixel p, the arms meeting there are gathered from a disc of radius
`rho(p) = r0 + w(p) / 2` (r0 = 4 px; w = the local width, 2.5 x the winning scale):

```
arms(p) = []
# line ends pointing into p: an end at q whose free side u points at p, within the disc
for each direction j (free side u_j):
    e_j(p) = max over q in the disc along -u_j from p (q = p - d u_j, d <= rho) of E[j](q)   # a "ray" kernel
    if e_j(p) > tau_end: arms += (direction -u_j, strength e_j, kind 'end')
# lines passing through p
for each orientation k:
    b_k(p) = the bipole of step 5 at p, sqrt(A+ x A-)
    if b_k(p) > tau_pass and k is a local peak over orientation: arms += (+t_k, b_k, 'pass'), (-t_k, b_k, 'pass')
merge arms within 22.5 deg of each other (keep the stronger)
n(p)  = number of arms
J3(p) = the 3rd-largest arm strength                     # a junction needs 3 arms; its evidence is the 3rd
```

Junctions are the local maxima of `J3` above `tau_J`, at least r0 apart (non-maximum suppression, like corners
from the 2nd eigenvalue of a structure tensor). The junction's arms are `arms(p*)` at the maximum.

**Type (first pass, refined by 4.4-4.5):**

| arms | pattern | type |
|---|---|---|
| 3 | one passing line + one end | 3-way (T) |
| 3 | three ends | 3-way (Y) |
| 4 | two passing lines, at least 15 deg apart | crossing candidate |
| 4 | four ends that pair into two lines by good continuation (4.4) | crossing candidate (broken at the centre) |
| 4 | otherwise | compound |
| 5 or more | | compound |

The compound radius prior of step 7 (a region of radius >= 16 px is compound) is kept as a separate flag and
measured with and without.

**Why it can find what tracing misses.** Arms come from the orientation field around the point, not from traces
that reached it. A branch with field support near its parent counts even if its trace stopped short. An arm with
no field support is still missed: phase 1 measures how much of the loss is of each kind.

**Cost.** One ray kernel per end direction (32) and channel; a disc gather per pixel. Comparable to step 5.

### 4.4 V2c. Good continuation

**Biology.** Contour integration follows good continuation: two segments are grouped when one continues the
other with a small turn (Field, Hayes and Hess 1993). At junctions it decides which arms belong to one curve.

**Computation.** For two arms a, b of one junction (outward directions u_a, u_b; widths w_a, w_b, each measured as
the OD width at half height along a 2 r-long stretch of the arm, as `od_widths`):

```
turn(a, b) = angle between u_a and -u_b                       # 0 for a straight continuation
cont(a, b) = exp(-turn^2 / (2 * 20 deg^2)) * exp(-ln(w_a / w_b)^2 / (2 * ln(1.5)^2))
```

- **Crossing:** the pairing of 4 arms into two lines with the largest product of `cont` (both above 0.3).
- **Fork:** the parent is the pair with the largest `cont`; the third arm is the branch. A soft Murray check,
  `r_parent^3 ~ r_1^3 + r_2^3`, is reported as a diagnostic, not used to decide, in phase 2.

**Output.** Per junction: arm pairings with scores. Used by step 9 instead of the same-trace rule.

### 4.5 V2d. Transparency: fork or crossing

**Biology.** At an X-junction the visual system tests whether one surface is seen through another: two
absorbers in series make the overlap darker than either arm (Metelli's conditions; Adelson and Anandan 1990).
Blood is such an absorber: two vessels at different depths add their ODs, while at a fork the lumens merge into
one volume, the union, not the sum.

**Computation, on the OD of step 1 (never on a render).** For a candidate crossing with lines L1 and L2:

```
a1, a2 = median OD on each line's two arms, from R to R + 2 w outside the junction centre, along the arm
ac     = max OD within 2 px of the centre
rho    = ac / (a1 + a2)
crossing predicts rho ~ (max(a1, a2) + kappa * min(a1, a2)) / (a1 + a2)       # kappa ~ 0.87-0.94
fork predicts     rho ~  max(a1, a2) / (a1 + a2)                               # union
separation = kappa * min(a1, a2) / (a1 + a2)
```

Decide with a likelihood ratio using the local noise level of the OD (step 1's noise map) and the spread of
`rho` measured in phase 2. When `separation` is small (one vessel much fainter than the other), or the
likelihood ratio is between 1/3 and 3, the junction is **ambiguous** and goes to 4.5b.

The earlier knot cue failed because it divided a render's residual by the thinner contrast, and the additive
render over-predicts every junction. Here the render is not involved.

### 4.5b V2e. Local model comparison (top-down feedback)

**Biology.** Higher areas predict lower-area activity and the mismatch drives the interpretation (predictive
coding, Rao and Ballard 1999). Here the "higher area" is the renderer, used only to choose between a few
hypotheses at one junction.

**Computation.** For an ambiguous junction, in a window of radius `R + 3 w`:
1. build each hypothesis as a small spline network from the junction's arms: crossing (the 4.4 pairing),
   fork with each pair as parent, T;
2. render each with `render.JunctionModel` (union at nodes, kappa at crossings), profiles initialised from the
   arms' OD, positions fixed; fit the profiles for 20 iterations;
3. keep the hypothesis with the lowest weighted residual plus description cost, the fit's prune cost:
   `0.25 * 0.5 * n_params * ln(max(pixels / 6, 2))` per edge.

**Cost.** About 0.2-0.5 s per ambiguous junction. Phase 2 reports how many are ambiguous.

### 4.6 Surround fix for crossings (phase 3)

Step 4's orientation-blind part is the mean of the lower half (8 of 16) of the tuning curve. A crossing's second
lobe enters that half. Two candidates, measured on the step-4 figure's four pixels and then on all truth
crossings and background pixels of the development scenes:
- the mean of the lowest quarter (4 of 16);
- the mean of the lower half after removing the layers within one step of the two largest local peaks.

Accept a candidate only if crossings lose at most 15 % (now 44 %) while blobs still lose at least 40 % (now 48 %).
The global cross_k change already tried (0.5) is not repeated.

### 4.7 Curvature cells (phase 4)

**Biology.** End-stopped cells carry curvature (Dobbins, Zucker and Cynader 1987); V4 cells are tuned to it
(Pasupathy and Connor 2001). Contour integration follows co-circularity (Parent and Zucker 1989).

**Computation.** Bend step 5's lobes along circles: for curvatures `kappa_j` in {0, +-1/40, +-1/20, +-1/10} px^-1,
the lobe points follow the circle of curvature `kappa_j` tangent to `t_k`, and each point reads the layer of the
circle's tangent there. The association response is the maximum over j; the winning `kappa_j` is the curvature
map. Uses: lifting dips on curved vessels, bending the tracer's predicted step, co-circular turns in 4.4.

### 4.8 Grouping by label spreading (phase 4)

**Biology.** Tracing one curve through a tangle is serial: an attentional label spreads along the curve in V1,
enhancing the neurons on it, and follows good continuation at crossings (Jolicoeur, Ullman and Mackay 1986;
Roelfsema, Lamme and Spekreijse 1998).

**Computation.** Step 6 stays as the spreading rule along a ridge. Three changes:
- one object at a time, strongest seed first;
- at a junction cell, the label continues along the arm that 4.4 pairs with the arriving arm (crossing: straight
  through; fork: along the parent), not along the best local step;
- the branch at a fork starts as a new object later.

**Output.** Traces that keep a vessel's identity through crossings and forks: continuous vessels, the property
asked for in batch 8 and not reached by linking pieces afterwards (E22).

### 4.9 Motion channel for real bursts (phase 5)

**Biology.** Common fate: elements that move together are grouped. Direction-selective cells in V1 and area MT
carry it.

**Why here.** In a burst, red cells and plasma gaps move along the vessels, so the lumen flickers from frame to
frame while the tissue around it is still. vesselscene simulates exactly this (`imaging.filling`: red-cell
columns and aggregates moving downstream at each vessel's speed; the average is the burst mean, a frame one
instant). Flicker is a vessel cue that does not depend on darkness, unlike the OD mask that failed in E24.

**Computation**, on registered frames `F_t` (t = 1..N) with their valid masks:

```
L_t  = ln F_t
D_t  = L_t - median_t(L_t)                         # each frame's departure from the burst's median
Phi  = 1.4826 * median_t(|D_t|)                    # robust temporal spread per pixel
Phi2 = max(Phi^2 - noise_var(I), 0)                # minus the camera's noise at that brightness
flow = sqrt(Phi2) / local_rms(sqrt(Phi2), background, 48)    # flicker contrast-to-noise
```

**Uses.**
- step 1: add `flow > 3` (hysteresis to 1.5) to the background mask;
- steps 3-5: an extra drive into the line evidence, weighted by a parameter swept on synthetic bursts;
- later: the direction of motion along a vessel gives flow direction, so bifurcation against confluence.

**Known confounds.**
- Residual registration error makes every high-contrast edge flicker, including vessel walls. Flicker at the
  lumen centre and flicker at the walls have different signatures: score the centre only, or subtract a
  model of edge flicker (gradient magnitude x registration error).
- Capillaries that are not perfused do not flicker. Motion can only add vessels to the mask, never remove them.

## 5. Interfaces and integration

**New module** `experiments/neuromimetic/v2.py`:

| function | input | output |
|---|---|---|
| `end_stopped(C, A_plus, A_minus, cfg)` | step 5's C and lobes, per channel | E: (32, h, w) |
| `junction_cells(C, E, S, cfg)` | C, E, winning scales | dict(J3, arms per maximum, type, centres) |
| `arm_widths(OD, junction, cfg)` | OD, one junction | per-arm width and contrast |
| `continuation(junction, cfg)` | arms with widths | pairings and scores |
| `transparency(OD, noise, junction, cfg)` | OD, noise map, a crossing candidate | rho, likelihood ratio, ambiguous flag |
| `compare_models(OD, w, junction, cfg)` | OD, precision weights, an ambiguous junction | chosen hypothesis, residuals |
| `curvature_field(U, sigma, cfg)` (phase 4) | step 4's U | C, curvature map |
| `flicker(frames, valid, camera)` (phase 5) | registered burst | flow map |

**Changes to existing code.**
- `association_field` returns its last lobes `A+`, `A-` when asked (no change to C).
- `Config` (neuromimetic): `v2_junctions: bool = False`, and the V2 parameters (section 6.3); `iso_frac`
  for 4.6 (default 0.5, today's behaviour).
- `end_stopping` (step 7): with `v2_junctions`, junctions come from `junction_cells`; traces are still cut at
  their point nearest each junction to make edges; types and arm pairings come from V2.
- `proposals.build_network` (step 9): uses V2's arm pairings for crossings and through-nodes when present,
  the same-trace rule otherwise.
- Everything is off by default; the current pipeline's digests must not change with the flags off.

## 6. Evaluation

### 6.1 Instruments (all existing)

- **Per-junction scorecard** (`junction_gallery.judge`): found within the strict radius, right coarse type,
  render >= 95 %, per type. Run on the 5 development averages; the held-out images only at the end.
- **Probe battery** (`probes.py`), its junction sweeps: `fork` (forks matched and typed 3-way), `t`,
  `crossing_angle` (smallest angle typed a crossing), `crossing_depth`, `crossing_width` (unequal crossing),
  `end` (ends traced without a false junction), and `empty`.
- **Controls** (`controls.py`, 7 imaging seeds): empty average, two lines, a wide fork, a 45-degree crossing.
- **Tier 1 and tier 2 with the paired per-scene test and its guards** (`run_experiment.py`, `paired.py`).

### 6.2 Phases and gates

| phase | work | measured offline (no change to the pipeline) | gate to the next phase |
|---|---|---|---|
| 0 | dev per-junction scorecard of the current pipeline; dev junction gallery | baseline found / typed per type on the 5 dev averages | done |
| 1 | V2a + V2b | junction recall and precision against the truth: junction cells alone, step 7 alone, and their union | 3-way recall up by at least 15 points at equal or better precision; false junctions on `empty`, `end` and background not up |
| 2 | V2c + V2d + V2e | typing accuracy on matched junctions; histogram of rho at truth crossings and forks; share ambiguous | fork / crossing AUC of rho at least 0.75 where min/max arm contrast >= 0.5; typing up by at least 10 points |
| 3 | integrate behind `v2_junctions`; 4.6 surround fix | tier 1, then tier 2 paired | composite confirmed or non-inferior with the per-junction gains held; no guard veto |
| 4 | curvature cells; grouping by label spreading | edges per truth vessel (fragmentation); centreline F1 | fragmentation down, centreline F1 not down |
| 5 | vesselscene burst rendering; motion channel | on synthetic bursts: lumen coverage of the flicker mask, background offset; on real bursts: overlap with the Frangi mask, OD on empty background | background offset not up (unlike E24); coverage of wide-vessel lumen up |

Each phase states its prediction before it runs, as the overnight loop did (`program.md`), and logs the outcome
in `splinefit/LOG.md` and `results.tsv`.

### 6.3 Parameters

| parameter | start | swept on dev |
|---|---|---|
| beta (end-stopping) | 1.0 | 0.5-2 |
| tau_end, tau_pass | t_low (1.5), t_high (3) | yes |
| tau_J | 3 | yes |
| arm merge angle | 22.5 deg | no |
| continuation sigma_theta, sigma_w | 20 deg, ln 1.5 | yes |
| transparency kappa | the image's fitted kappa, or 0.87 before the fit | no |
| ambiguity band | likelihood ratio 1/3-3 | yes |

All tuning uses the development scenes. More development scenes (healthy_s009 and on, as `program.md` allows)
should be generated before phase 3: with 5 scenes, the paired test has little power.

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| End-stopped cells fire on background texture: the texture "web" reaches C ~ 5 at its 99th percentile | gate ends by `C > t_high` at the line and by the surround; measure false junctions on `empty` and background first (phase 1 gate) |
| Every small vessel ending on a wide one makes a junction cell; neighbouring ones merge into compounds | that is also the truth's behaviour; measured by the compound rows of the scorecard |
| Transparency fails when one vessel is much fainter (separation goes to 0) | the ambiguity band sends those to model comparison |
| Arm widths and contrasts near a junction are biased by the overlapping lumens | measure them from R to R + 2 w outside the centre |
| Runtime | V2 reuses step 5's lobes; model comparison only for ambiguous junctions; budget +30 % of the proposer's time |
| Overfitting 5 development scenes | pre-registered gates; more dev scenes; held-out only at the end |
| Motion: registration error makes edges flicker | score the lumen centre; model edge flicker from gradient x registration error |

## 8. Alternatives considered

- **A trained recurrent network** (hGRU-style horizontal connections; Linsley et al. 2018) on vesselscene's
  unlimited labelled scenes. Rejected for now: the work is to be done by hand, and a hand-built model with a few
  tuned parameters should transfer to real frames more predictably than one with millions of learned weights.
  It remains the obvious comparison once V2 exists.
- **Local model comparison at every junction.** Simplest and most principled, but slow, and it needs a good
  arm set to compare; V2 provides the arms and settles the clear cases cheaply.
- **A denser scale grid for contrast-to-noise.** Tested: no change in mask coverage (77 % either way).
- **An OD-amplitude background mask** (E24). Tested: biases the background; the motion channel replaces it.

## 9. Dependencies

- vesselscene renders one frame per scene today. Phase 5 needs a burst renderer: frames 0..N-1 of one burst
  (`imaging.filling(frame=i)` already draws a frame's filling from the burst seed and frame index).
- `render.JunctionModel` and `fit.make_model` for model comparison (exist).

## 10. Open questions

1. **Which kappa** for the transparency rule: vesselscene's reference 0.87, or the image's fitted value (0.94 on
   development healthy 7)? Phase 2 measures both.
2. **Compound junctions:** truth compounds gather 5-6 lines in a region. Should a compound be one junction cell
   with many arms, or several nearby 3-way cells merged? The truth's own definition decides; phase 1 reports both.
3. **Flow direction from bursts** would allow bifurcation against confluence. Worth a phase of its own once
   phase 5 works.
4. **How far V2 gets toward the 95 % bar.** Detection and typing are two of its three tests; the third (the
   render at the junction) depends on the fit.

## Appendix A. The visual system and this pipeline

| brain | pipeline today | this spec |
|---|---|---|
| photoreceptors, horizontal cells (light adaptation) | step 1 | motion cue for the mask (4.9) |
| centre-surround ganglion cells, gain control | step 2 | - |
| V1 simple cells in orientation columns, spatial-frequency channels | step 3 | - |
| normalisation, surround suppression | step 4 | crossing fix (4.6) |
| V1 horizontal connections (association field) | step 5 | curvature (4.7) |
| end-stopped cells | symbolic events in step 7 | V2a (4.2) |
| V2 angle and junction cells | arm-counting rules | V2b (4.3) |
| good continuation | same-trace rule in step 9 | V2c (4.4) |
| transparency at X-junctions | kappa in the render only | V2d (4.5) |
| predictive coding | the spline fit | V2e (4.5b), at junctions |
| incremental grouping (curve tracing) | greedy parallel tracing | 4.8 |
| medial-axis shape code (V1 late responses, IT) | the fitted splines | - |
| common fate (V1 direction selectivity, MT) | none | 4.9 |

## Appendix B. References

- Adelson, E. H. and Anandan, P. (1990). Ordinal characteristics of transparency.
- Dobbins, A., Zucker, S. W. and Cynader, M. S. (1987). Endstopped neurons in the visual cortex as a substrate
  for calculating curvature. Nature.
- Field, D. J., Hayes, A. and Hess, R. F. (1993). Contour integration by the human visual system: evidence for a
  local "association field". Vision Research.
- Heitger, F., Rosenthaler, L., von der Heydt, R., Peterhans, E. and Kübler, O. (1992). Simulation of neural
  contour mechanisms: from simple to end-stopped cells. Vision Research.
- Ito, M. and Komatsu, H. (2004). Representation of angles embedded within contour stimuli in area V2 of macaque
  monkeys. Journal of Neuroscience.
- Jolicoeur, P., Ullman, S. and Mackay, M. (1986). Curve tracing: a possible basic operation in the perception of
  spatial relations. Memory and Cognition.
- Linsley, D., Kim, J., Veerabadran, V., Windolf, C. and Serre, T. (2018). Learning long-range spatial
  dependencies with horizontal gated recurrent units. NeurIPS.
- Metelli, F. (1974). The perception of transparency. Scientific American.
- Parent, P. and Zucker, S. W. (1989). Trace inference, curvature consistency, and curve detection. IEEE PAMI.
- Pasupathy, A. and Connor, C. E. (2001). Shape representation in area V4: position-specific tuning for boundary
  conformation. Journal of Neurophysiology.
- Rao, R. P. N. and Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience.
- Roelfsema, P. R., Lamme, V. A. F. and Spekreijse, H. (1998). Object-based attention in the primary visual
  cortex of the macaque monkey. Nature.
