# The neuromimetic annotator: notes

`neuromimetic.py` implements DESIGN.md: a deterministic annotator of vesselscene stills built from what the
early visual system is thought to do with contours, plus a render-and-compare refinement borrowed from
diffusion models. These notes give each stage's biology and maths, the final parameters, the dev results
with the ablations and the curvature (Hessian) baseline, what each change did, what failed, and what is
still open.

**Dev only.** Everything here was tuned and measured on the three dev scenes (healthy_s004_320x512,
healthy_s000_480x768, pathologic_s000_480x768; average and frame, so 6 images). The numbers are therefore
optimistic, and a difference of about 0.01 is within the noise of six images. The held-out test scenes were
never opened.

## Answer in short

The question was whether contrast detection, V1 orientation processing, contour integration and
diffusion-style refinement can annotate the vessel graph better than curvature (Hessian) analysis.

**Against the Hessian baseline, on dev: clearly yes.** `baseline_hessian` is the curvature pipeline, tuned
on the same dev scenes. The full annotator beats it on every headline metric, in both the average and the
frame:

| dev, mean of 3 scenes | centreline F1 | junction F1 | junction F1 strict | coarse type acc. (majority ref.) | coarse type balanced | crossing R / P | edge cover | s / image |
|---|---|---|---|---|---|---|---|---|
| neuromimetic `annotate`, average | **0.805** | **0.694** | **0.651** | **0.549** (0.522) | **0.581** | 0.38 / 0.57 | **0.649** | 6.5 |
| Hessian baseline, average | 0.678 | 0.539 | 0.487 | 0.375 (0.595) | 0.393 | 0.06 / 0.24 | 0.528 | 0.8 |
| vesselmap (LIMBUS), average * | 0.684 | 0.607 | – | 0.320 | – | 0.40 / 0.16 | 0.610 | 564 |
| neuromimetic `annotate`, frame | **0.801** | **0.608** | **0.587** | **0.560** (0.483) | **0.616** | 0.33 / 0.46 | **0.651** | 5.9 |
| Hessian baseline, frame | 0.653 | 0.541 | 0.462 | 0.424 (0.503) | 0.379 | 0.06 / 0.33 | 0.552 | 1.2 |
| vesselmap (LIMBUS), frame * | 0.715 | 0.600 | – | 0.323 | – | 0.40 / 0.25 | 0.617 | 442 |

\* vesselmap was run by the orchestrator with an earlier version of the harness, before the strict and
balanced scores existed. Its crossing precision and edge cover were computed slightly differently.

Two numbers stand out:
- **Crossings.** The Hessian finds 6 % of the crossings; the orientation score finds 33-38 %. In the score,
  a crossing's two vessels live in different orientation layers, so neither breaks the other.
- **Coarse typing: competent by balanced accuracy, but beating the majority label depends on a dev-fitted
  prior.**
  - Balanced accuracy (mean per-class recall; a constant label scores 0.33) is 0.58 on averages and 0.62 on
    frames. Without the compound-radius prior it is 0.51 / 0.56.
  - Plain coarse accuracy beats the image's majority-label reference only because of the `compound_r = 16`
    prior (stage 7), which was tuned on the same dev truth. With the prior, it is 0.549 against 0.522 on
    averages and 0.560 against 0.483 on frames. The margin on averages is about the noise level.
  - With the prior switched off (`compound_r = ∞`, same proposals), it is 0.405 and 0.437: below the
    reference on both kinds.
  - Even with the prior, it is below the reference on 2 of 6 images: both pathologic stills, 0.469 against
    0.594 and 0.545 against 0.727. Before the review fixes it was 3 of 6, healthy_s004 average included.
  - The Hessian baseline stays below the majority reference.

**Which ideas did the work.**
- **Most of the gain comes from the V1 stage and the readout.**
  - Separate spatial-frequency channels of an orientation score (a thin vessel crossing a wide vein
    survives).
  - Phase gating by the odd partner (no echoes along vein walls).
  - Tracing along the tangent in SE(2), so a line runs straight through a crossing in its own layer.
  - End-stopping (graph) logic built on that.
- **Contour integration and the surround buy robustness to noise, not peak quality.**
  - With both switched off (`annotate_v1_only`), the averaged stills score better: composite 0.700 against
    0.671.
  - But the single frames collapse: centreline precision drops from 0.950 to 0.691, and the composite from
    0.664 to 0.594.
  - The association field is the stage that matters for frames. Without it, frame precision is 0.875 and
    the composite 0.634. Without the surround, the composite is 0.662, as good as the full annotator.
  - With each variant at its own best threshold, all four lay within 0.01 on dev (see Ablations; measured
    before the review fixes).
- **The diffusion-model lesson helped least.**
  - Rendering in OD and pruning by MDL are about neutral overall: they raise balanced typing and lower
    recall slightly.
  - Re-proposing from the residual (DESIGN's full loop, `annotate_repropose`) lowered every score I tried.
  - The OD knot at junction centres, the proposed crossing/fork cue, did not separate the types, because the
    additive render is not accurate at junctions.

## The eight stages

Every stage is one function of `neuromimetic.py`. Its docstring states the biological counterpart and the
maths, and all parameters are fields of the frozen `Config`.

### 1. Photoreceptors and horizontal cells: `photoreceptors`

- **Biology.** Cones respond roughly to log intensity (Weber). Horizontal cells pool a wide surround and
  feed the pooled mean back, so the output is log intensity relative to the local background (light
  adaptation).
- **Maths.**
  - `L = ln I`. NaN and saturated pixels (frame glare at 4095) are filled from the nearest valid pixel.
  - A provisional background `B0` is the robust upper envelope: a grey closing of L with a 60 px disc,
    computed on a 4× downsampled grid, then smoothed.
  - Then twice in a row:
    - `OD = B - L`.
    - The vesselness mask is `M = dilate(Z > 2 or G_1*OD > 3 RMS, 2 px)`, where Z is stage 2's band CNR
      measured against the current background pixels.
    - B is refitted **only on the mask's negative**, by masked normalised convolution
      `B = sum_k a^k G_(8*2^k)*(L w) / sum_k a^k G_(8*2^k)*w`, with `w = valid & ~M` and `a = 0.03`. The
      coarser scales fill holes wider than the kernel smoothly.
  - The target is `OD = B - L` in nepers, vessels positive. Overlapping vessels add their densities
    (Beer-Lambert). This OD is the only residual target anywhere in the annotator (the user's residual rule).
- **Why the 60 px envelope.** The first version used a 28 px closing radius. It filled the 80 px vessel of
  the pathologic scene into the background (B contained the vessel, and OD showed only a faint ghost of it).
  The envelope disc must be wider than the widest half lumen.
- **Known defect: B is partly fitted on faint vessels** (found in review, not fixed). The mask
  `Z > 2 or S > 3 RMS` misses many observable vessels, so their pixels count as background, and B is pulled
  towards them. Their OD is attenuated. An audit against the truth (truth used only for the audit) found:
  - Share of the truth's centreline px in the pixels B is fitted on: 0.12 / 0.05 for healthy_s000
    (average / frame), 0.28 / 0.21 for healthy_s004 and 0.32 / 0.26 for pathologic_s000.
  - Share of those background pixels lying within ±3 px of a truth vessel: 0.04-0.21.
  - At those centreline px, the median OD is 0.56-0.70 of the OD from a refit that also leaves out the
    truth lumens. Example: 0.0174 against 0.0298 on healthy_s004 average.
  - This is one cause of the faint-line recall problem and of my band CNR being about 0.23 of the truth's
    (stage 8, Open problems). The vessel pixels also enter the background RMS that every CNR is divided by.
- **The fix I tried, `bg_trace_k`** (a Config option, off). Stage 6 feeds stage 1 back once: the traced
  lumens (±k × the readout width) join the vesselness mask, B is refitted, and stages 3-6 run again.
  - It removes most of the leak. At k = 0.5, the share of truth centreline px in the fit set falls from
    0.24 to 0.08 on averages and from 0.17 to 0.07 on frames. Faint vessels that are not traced still leak.
  - It did not raise the dev scores, and it costs about 4 s more per image. Composite = mean of centreline F1,
    strict junction F1, balanced coarse type accuracy and edge cover; average / frame:

    | variant | t_high / t_low | composite | clR | clP | balanced coarse type |
    |---|---|---|---|---|---|
    | off (final) | 3.0 / 1.5 | 0.671 / 0.664 | 0.706 / 0.701 | 0.948 / 0.950 | 0.581 / 0.616 |
    | k = 0.5 * | 3.0 / 1.5 | 0.684 / 0.643 | 0.724 / 0.720 | 0.926 / 0.898 | 0.574 / 0.505 |
    | k = 1.0 * | 3.0 / 1.5 | 0.665 / 0.622 | 0.757 / 0.767 | 0.867 / 0.802 | 0.520 / 0.477 |
    | k = 0.5 | 3.5 / 1.7 | 0.653 / 0.645 | 0.685 / 0.683 | 0.937 / 0.919 | 0.539 / 0.581 |
    | k = 1.0 | 3.5 / 1.7 | 0.681 / 0.632 | 0.742 / 0.730 | 0.905 / 0.867 | 0.588 / 0.512 |
    | k = 0.5, lumens left out of the B fit only | 3.0 / 1.5 | 0.667 / 0.648 | 0.714 / 0.704 | 0.944 / 0.938 | 0.555 / 0.558 |
    | k = 1.0, lumens left out of the B fit only | 3.0 / 1.5 | 0.676 / 0.651 | 0.690 / 0.675 | 0.933 / 0.913 | 0.629 / 0.630 |

    \* measured before the stage 7 review fixes (log 26). For comparison, the final row before those fixes
    was 0.680 / 0.656.
  - Recall rises, but precision falls, most on frames. A smaller background RMS raises every CNR, which
    acts like lower thresholds. Raising the thresholds to restore precision loses the recall again. When the lumens leave only the B
    fit and the noise pixels stay, the recall gain mostly vanishes. The attenuation is real, but on dev,
    correcting it does not improve the annotation.

### 2. OFF-centre ganglion cells with contrast gain control: `ganglion_cells`

- **Biology.** Centre-surround (difference of Gaussians) cells whose gain is divided by the local contrast,
  so their output is a contrast-to-noise ratio rather than a contrast.
- **Maths.**
  - For each band `s ∈ {1, 2, 4, 8, 16, 32}`: `D_s = (G_s - G_2s) * OD`, divided by the robust local RMS of
    the same band over the background pixels. The RMS is the Winsorised (at 3× the global robust RMS)
    masked normalised convolution of `D_s²`.
  - `Z = max_s D_s / RMS_s`. This is the band-matched CNR the truth uses to decide what is observable.
- **Used for.** The mask iterations of stage 1 (Z), and the visibility floor of stage 8 (the RMS maps). It is
  a side branch: V1 (stage 3) does not read Z. It filters stage 1's OD itself and normalises its own
  responses by their local RMS.

### 3. V1 simple cells: `simple_cells`

- **Biology.** Simple cells have elongated receptive fields. Even-symmetric cells have an excitatory centre
  stripe between inhibitory flanks (line detectors); odd-symmetric cells have one excitatory and one
  inhibitory half (edge detectors). They come at all orientations and in separate spatial-frequency
  channels. A quadrature (even/odd) pair gives local energy and phase: a line is where the even cell
  dominates its odd partner.
- **Maths.**
  - With t the tangent and n the normal, the even filter is `-sigma² d²/dn²` of an anisotropic Gaussian
    (std sigma across, `elong·sigma` along). The odd partner is `sigma d/dn` of the same Gaussian.
  - Both are applied exactly in the Fourier domain (reflect-padded rFFT, analytic transfer functions), so
    there is no kernel truncation and all 16 orientations are equivalent.
  - Each scale's channels are divided by their robust local RMS over background pixels, pooled over
    orientations. The kernel's noise gain and the local texture level both cancel, so the response is a CNR.
  - The line evidence per scale is `z = z_even - 0.7 |z_odd|` (phase gating). At the wall of a wide,
    flat-floored vein the small-scale even response is an edge echo with a large odd partner, so it is
    rejected. At a vessel centre the odd response vanishes by symmetry.
  - Nine scales `sigma = 1.5 … 24 px` are grouped into three channels: fine 1.5-3, medium 4.2-6, coarse
    8.5-24. Each channel is computed on a grid downsampled 1, 2 and 4 times respectively, so every scale is
    at least 2.1 grid px. Elongation is 3, 2.5 and 2.
  - Per channel, `U(x, y, theta) = max over its scales`, and `S` is the argmax scale.
- **Why channels.** The first version took one max over all scales. A thin vessel crossing a wide vein then
  vanished: the vein's broad orientation tuning swamped it. In the fine channel the vein's flat floor gives
  nothing, and the thin vessel is a clean ridge.
- **Why the coarse filters are short.** Long coarse filters draw star-shaped rays from the dark knot where
  wide vessels meet.

### 4. Non-classical surround: `surround`

- **Biology.** Cross-orientation suppression by the pooled activity of all orientations at the location (in
  cortex largely divisive normalisation), and iso-orientation surround suppression. A contour on an empty
  background is salient; inside texture it is not.
- **Maths.** Both suppressions are **subtractive** here. DESIGN's divisive normalisation was not implemented.
  - `iso` = the mean of the lower half of the orientation tuning `max(U, 0)` over theta. For a line it is
    about 0.04 of the peak, for a blob about the peak. The update is `U1 = U - 1.0 iso`, so several
    orientation peaks per pixel (crossings) survive.
  - Flank energy is `U+` of the same orientation averaged over a strip on each side, at distances
    `2.5 sigma + 1` to `+max(6, 1.5 sigma)`. Only the **weaker** flank counts: texture surrounds a pixel on
    both sides, while a neighbour in a vessel bundle is on one side only.
  - `U2 = U1 - 0.3 min(F_left, F_right)`.

### 5. Horizontal connections, the association field: `association_field`

- **Biology.** Long-range horizontal connections between co-linear (co-circular) cells (Field, Hayes and
  Hess 1993). Bipole cells (Grossberg) need both lobes driven, so gaps are completed and line ends are not
  extended. This is the same computation as the steady state of Williams and Jacobs' stochastic completion
  field.
- **Maths.**
  - Pooling over neighbouring orientation layers: `P_k = max(C_k, 0.8 C_k±1)` (the walk may turn by one
    layer).
  - Two lobes per orientation: `A±_k = sum_d w(d) P_k(p ± d t_k)` with `w(d) = exp(-d²/2l²)`,
    `l = 8 sqrt(max(1, sigma/2))` px, laterally spread 0.7 px and normalised.
  - The bipole is `B_k = sqrt(A+ A-)`.
  - Three fixed steps of `C ← (U + B(C)) / 2`, starting from `C0 = max(U, 0)`. On a continuous line C = U.
    Across a gap C rises towards half the line. A texture blob without collinear support falls to U/2.
- **Option.** `assoc_fill` keeps `max(U, C)` (facilitation only). It was tested and not adopted
  (Ablations).

### 6. Readout: `readout` (and `merge_channels`)

- **Candidates.** C_k must be a maximum along the normal n_k (bilinear neighbours at ±1 px) and a maximum
  over orientation (`C_k ≥ C_k-1`, `C_k > C_k+1`, so several peaks per pixel are allowed). C_k must exceed
  `t_low = 1.5`, and the pixel must be at least 3 px from the frame and from invalid pixels.
- **Tracking.** Seeds are the candidates above `t_high = 3.0`, strongest first, with ties broken by
  position.
  - A trace steps 1 px along its tangent in both directions. The next sample is the best candidate among
    positions 0 and ±1 px across the predicted one, in layers k and k±1 (SE(2) connectivity: hysteresis
    along the contour). The score is C, less 15 % per px of lateral offset and 10 % per layer of turn.
  - So a trace stays in its own orientation layer through a crossing, and takes the straighter branch at a
    fork.
  - It looks up to 3 px ahead, which bridges 1-2 px gaps.
  - It passes through another trace's claim for at most 4 samples (a shallow crossing). If it keeps running
    along that trace (a T or a merge), it stops at the first contact.
  - Each sample claims ±max(1, 0.5 sigma) px across itself in layers k and k±1, which is its lumen.
- **Channel merge.** Traces are taken strongest first. A point is dropped where an accepted trace runs
  parallel (orientation bin ±1, 11.25° bins) within 3 px. It is also dropped where a coarser channel's
  trace runs parallel (same bin) within 0.6 of its width; these are the wall echoes of a wide vein. A thin
  vessel crossing that vein at a shallow angle is kept.

### 7. End-stopped cells, graph assembly: `end_stopping`

- **Biology.** End-stopped (hypercomplex) cells signal line terminations. A line ending on another is a T
  (or a Y); two lines continuing through a point are an X.
- **Maths.**
  - Collinear end-to-end gaps up to 14 px are bridged (mutual best pairs, turn < 40°; short laterally
    offset gaps allowed).
  - Vessel widths are measured as the full width at half maximum of OD cross-sections (`od_widths`).
  - **Events.**
    - T: an end within 6 px plus half the other vessel's width of another trace, inside a 45° cone ahead or
      already in its lumen.
    - X: two traces intersect.
  - **Junctions.** Events cluster by complete linkage (`_cluster_events`). Two events join when closer than
    `4 + (w_thin,a + w_thin,b)/2`, the size of their lumen overlaps. Two events on one trace join when closer
    than `4 + (w_cross,a + w_cross,b)/2`: a thin vessel crossing two adjacent veins is one compound.
    Distances above 2 lambda never join.
  - **Where an event sits.** An X sits at the intersection. A T sits where the end's own line (its outward
    direction) meets the other centreline, if that is within the attachment reach (6 px plus half the other
    width); otherwise it sits at the nearest point of the other centreline. With the nearest point, the two
    halves of an oblique crossing broken at a wide vessel landed `w / tan(angle)` apart along it and were
    emitted as two pseudo-Ts (review fix, log 26).
  - **Arms** are the participating traces' stretches leaving the junction disc that are at least 6 px long.
  - **Types.**
    - 3 arms: 'pseudo-T'. This is the commonest 3-way type in the dev truth (92 against 12 forks); a still
      shows no flow direction.
    - 4 arms that pair into two straight lines (turn < 40°, at least 15° apart; a trace passing through
      pairs its own arms): 'crossing'.
    - Otherwise: 'compound'.
    - A junction region of radius ≥ 16 px is typed 'compound' regardless. This is a dev prior: large
      regions gather lines too faint to trace; it raised plain and balanced coarse accuracy.
  - **Splitting.** T-ends are extended to their attachment point when the gap is ≤ 12 px and outside the
    other lumen. Every trace is cut at its point nearest each of its junctions (where the truth's edges end),
    so polylines run junction to junction. A free piece (one end at a junction) is kept only if its stretch
    beyond the junction disc is at least `arm_min`. This is the same test that counts an arm, so a junction's
    edges are its arms. Before the review fix, the test measured from the cut at the centre, and stubs lying
    mostly inside the disc were output: some 'pseudo-T' junctions had 4 or more edges.

### 8. Iterative refinement, the diffusion-model lesson: `refine`

- **Idea.** Like DDIM, a fixed number of deterministic rounds of render, compare to the target, correct.
- **Each round.**
  - **Render.** Every trace gets a contrast, half width and blur along it from cross-sections of the cleaned
    OD (`fit_profiles`). Every 6 px, the median cross-section over ±max(6, w) px of the trace is fitted by a
    blurred box `a [Phi((r-u)/s) - Phi((-r-u)/s)] + b`: grid over (r, s), (a, b) by least squares, with
    r ≤ 0.75 w + 2 so a thin vessel cannot borrow a wide neighbour's profile. The fits are interpolated
    along the trace, so one vessel renders without seams.
  - The graph is rendered **additively in OD** (`_tube`, `_Render`). Each pixel takes the parameters of its
    nearest centreline point; tube ends are butt ends, so the pieces of a vessel cut at a junction join
    without overlap.
  - **Residual** = `OD - render`. The target is always stage 1's OD, the image cleaned with the background
    fitted on the vesselness mask's negative. It is never a vesselness or orientation map, and never a
    mask.
  - **MDL pruning** (`_mdl_prune`). Only edges with a free end are candidates: whole isolated traces,
    spurs, false arms. An edge between two junctions belongs to a vessel that continues. An edge must pass
    two tests:
    - The gain `(sum p r_-e / sigma²)² / sum p² / sigma² / corr_px` must reach
      `2 (1 + L / 20)`. Here p is the edge's tube used as a template and r_-e is the residual without the
      edge; its amplitude is re-fitted against what the others leave (explaining away). Junction discs
      (0.7 R) carry no weight, and sigma² is the local RMS² of the high-passed OD over background pixels.
    - The band CNR of its OD profile against stage 2's band RMS along it must reach 0.5. On my scale this is
      about the truth's CNR 2, since my edge CNR is about 0.23 of the truth's. Part of that factor is the
      stage 1 background leak: attenuated faint OD and vessel pixels in the RMS.
    - Edges are removed greedily, weakest first; the gains of overlapping edges are updated, and the removed
      ranges are cut out of their traces.
  - **Re-proposal**, optional (`repropose`; `annotate_repropose`). Stages 3-6 run again on
    `max(residual, 0)` outside the junction discs, at 1.3× the thresholds. A new trace is kept if it runs at
    least 20 px outside the lumens already rendered and is not a duplicate or wall echo of the graph's
    vessels. New edges pay 20× the description cost in the next prune. The re-proposed flag is carried per
    point, so it survives a gap join with an older trace. Before the review fix, a join dropped it and the
    joined trace escaped the 20× cost.
- **After the last round.** The graph is assembled again. The **knot cue** (mean residual at each junction
  centre over the thinner arm's contrast) is stored on each junction but not used (see What failed).
- **Final setting.** Two fixed rounds of render, prune and re-assemble (`rounds = 2`), without re-proposal.
  - The second round prunes the free-end edges that the first round's removals expose once the graph is
    re-assembled. On healthy_s004 average, round 1 removes 9 edges (284 px), round 2 removes 1 (23 px), and
    a third round removes nothing (before the stage 7 review fixes: 15 edges / 338 px, then 2 / 81 px).
  - Until the review, the code looped `rounds + 1` times, so this setting was mislabelled 'one round'. The
    loop now runs `rounds` times and the default is 2: the output is unchanged.
  - With re-proposal (`annotate_repropose`), the re-proposal runs between the two rounds.
  - Every re-proposal setting lowered the dev scores (log 20, 25). DESIGN's 2-3 rounds were tried as
    re-proposal rounds, and two re-proposal rounds were worse than one.

## Final parameters (`Config`)

| stage | parameters |
|---|---|
| 1 | env_radius 60, bg_sigma 8, bg_iters 2, mask_z 2, mask_dilate 2, saturation 4095, bg_trace_k 0 (off) |
| 2 | bands 1, 2, 4, 8, 16, 32; rms_sigma 24 |
| 3 | 16 orientations; scales 1.5, 2.1, 3, 4.2, 6, 8.5, 12, 17, 24; channels (0-2), (3-4), (5-8); grid factors 1, 2, 4; elong 3, 2.5, 2; odd_alpha 0.7 |
| 4 | cross_k 1.0, flank_k 0.3 |
| 5 | assoc_len 8, assoc_steps 3, assoc_gain 1, assoc_fill off |
| 6 | t_high 3.0, t_low 1.5, min_len 10, border 3, look 3, pass 4, claim_k 0.5, merge bins ±1 / 0 |
| 7 | gap_att 6, cone 45°, ext_max 12, gap_join 14, join_turn 40°, r0 4, link_k 1, share_k 1, arm_min 6, cross_turn 40°, compound_r 16, OD widths |
| 8 | rounds 2 (render / prune passes), repropose off (between the passes when on: re_k 1.3, re_mdl 20, re_novel 2), jmask_k 0.7, corr_px 6, mdl_k 2, mdl_len 20, cnr_min 0.5 |

## Dev results and ablations

All runs used `python -m experiments.neuromimetic.harness --annotator experiments.neuromimetic.neuromimetic:<fn>
--scenes <3 dev scenes> --kinds average frame --repeat 2` (the current harness). They were re-run after the
review fixes (log 26). Every run reported `deterministic = True`. Each row is the mean of the 3 dev scenes.
- "comp." is my summary, the mean of centreline F1, strict junction F1, balanced coarse type accuracy and
  edge cover.
- "maj." is the majority-label reference for coarse typing.
- Times are the faster of the two runs, on 2 threads of a loaded shared machine.
- The Hessian rows are from the earlier run; `baseline_hessian` did not change.

| annotator | kind | clR | clP | clF | jR | jP | jF | jF strict | coarse type (maj.) | coarse bal. | exact type | xR | xP | cover | pcs/edge | s | comp. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `annotate` (1-8) | average | 0.706 | 0.948 | 0.805 | 0.601 | 0.822 | 0.694 | 0.651 | 0.549 (0.522) | 0.581 | 0.510 | 0.375 | 0.573 | 0.649 | 1.11 | 6.5 | 0.671 |
| | frame | 0.701 | 0.950 | 0.801 | 0.505 | 0.807 | 0.608 | 0.587 | 0.560 (0.483) | 0.616 | 0.505 | 0.333 | 0.462 | 0.651 | 1.10 | 5.9 | 0.664 |
| `annotate_no_verify` (1-7) | average | 0.728 | 0.946 | 0.818 | 0.621 | 0.804 | 0.700 | 0.651 | 0.534 (0.514) | 0.561 | 0.506 | 0.429 | 0.553 | 0.663 | 1.12 | 4.3 | 0.673 |
| | frame | 0.715 | 0.940 | 0.807 | 0.538 | 0.782 | 0.629 | 0.602 | 0.523 (0.468) | 0.560 | 0.487 | 0.333 | 0.412 | 0.653 | 1.12 | 4.1 | 0.655 |
| `annotate_no_association` | average | 0.723 | 0.935 | 0.810 | 0.674 | 0.808 | 0.734 | 0.687 | 0.607 (0.506) | 0.643 | 0.559 | 0.359 | 0.572 | 0.663 | 1.14 | 6.1 | 0.701 |
| | frame | 0.733 | 0.875 | 0.792 | 0.641 | 0.731 | 0.670 | 0.615 | 0.437 (0.493) | 0.481 | 0.402 | 0.292 | 0.403 | 0.648 | 1.14 | 6.4 | 0.634 |
| `annotate_no_surround` | average | 0.707 | 0.947 | 0.804 | 0.625 | 0.817 | 0.707 | 0.681 | 0.541 (0.536) | 0.570 | 0.512 | 0.350 | 0.482 | 0.650 | 1.10 | 5.7 | 0.676 |
| | frame | 0.728 | 0.928 | 0.810 | 0.551 | 0.809 | 0.636 | 0.596 | 0.526 (0.491) | 0.578 | 0.486 | 0.306 | 0.386 | 0.666 | 1.11 | 5.9 | 0.662 |
| `annotate_v1_only` (1-3, 6-7) | average | 0.762 | 0.908 | 0.823 | 0.736 | 0.801 | 0.767 | 0.715 | 0.580 (0.499) | 0.571 | 0.556 | 0.408 | 0.560 | 0.692 | 1.14 | 3.6 | 0.700 |
| | frame | 0.778 | 0.691 | 0.723 | 0.783 | 0.468 | 0.552 | 0.528 | 0.487 (0.452) | 0.485 | 0.457 | 0.306 | 0.311 | 0.640 | 1.22 | 4.3 | 0.594 |
| `annotate_repropose` (8 with re-proposal) | average | 0.744 | 0.826 | 0.779 | 0.761 | 0.541 | 0.631 | 0.579 | 0.513 (0.513) | 0.540 | 0.479 | 0.324 | 0.445 | 0.600 | 1.27 | 13.3 | 0.624 |
| | frame | 0.743 | 0.891 | 0.805 | 0.654 | 0.627 | 0.630 | 0.590 | 0.473 (0.480) | 0.514 | 0.425 | 0.347 | 0.375 | 0.636 | 1.18 | 10.9 | 0.636 |
| `baseline_hessian` | average | 0.584 | 0.815 | 0.678 | 0.432 | 0.774 | 0.539 | 0.487 | 0.375 (0.595) | 0.393 | 0.352 | 0.062 | 0.241 | 0.528 | 1.15 | 0.8 | 0.521 |
| | frame | 0.655 | 0.665 | 0.653 | 0.524 | 0.618 | 0.541 | 0.462 | 0.424 (0.503) | 0.379 | 0.424 | 0.056 | 0.333 | 0.552 | 1.26 | 1.2 | 0.512 |

Before the review fixes, the composite was 0.680 / 0.656 for `annotate` (average / frame), 0.677 / 0.663 for
stages 1-7, 0.692 / 0.647 without association, 0.672 / 0.647 without surround, 0.708 / 0.602 for V1 only and
0.611 / 0.619 with re-proposal.

**Full annotator per scene**:

| scene | kind | clR | clP | clF | jF strict | coarse type (maj.) | coarse bal. | xR / xP | cover | s |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy_s000_480x768 | average | 0.871 | 0.970 | 0.918 | 0.749 | 0.611 (0.431) | 0.640 | 0.43 / 0.68 | 0.802 | 8.7 |
| healthy_s000_480x768 | frame | 0.856 | 0.963 | 0.906 | 0.733 | 0.618 (0.236) | 0.632 | 0.63 / 0.71 | 0.788 | 7.8 |
| healthy_s004_320x512 | average | 0.688 | 0.967 | 0.804 | 0.718 | 0.568 (0.541) | 0.613 | 0.39 / 0.54 | 0.637 | 4.1 |
| healthy_s004_320x512 | frame | 0.707 | 0.960 | 0.814 | 0.614 | 0.516 (0.484) | 0.633 | 0.25 / 0.55 | 0.658 | 3.4 |
| pathologic_s000_480x768 | average | 0.560 | 0.906 | 0.692 | 0.486 | 0.469 (0.594) | 0.490 | 0.31 / 0.50 | 0.508 | 6.7 |
| pathologic_s000_480x768 | frame | 0.541 | 0.926 | 0.683 | 0.413 | 0.545 (0.727) | 0.583 | 0.13 / 0.13 | 0.506 | 6.4 |

### Ablations at each variant's own best thresholds

This sweep was run before the review fixes (log 26). Only V1 only was re-swept after them: its
dev-best is now 3.5 / 2.0 (see Isolating the orientation score). The ablations above
share the thresholds of the full annotator. But the surround and the association field
lower C, so at shared thresholds an ablation also moves the operating point. To separate the two effects,
each variant was swept over `(t_high, t_low)` with stage 8 off: 2.0/1.0, 2.5/1.3, 3.0/1.5, 3.5/1.7,
4.0/2.0, 4.5/2.2, 5.0/2.5 and 6.0/3.0. The composite is the mean over all 6 images.

| stages 1-7 variant | best t_high / t_low | comp. | clF | jF strict | coarse bal. | cover | comp. at 2.5 / 3.0 / 3.5 / 4.0 |
|---|---|---|---|---|---|---|---|
| surround + association (= `annotate_no_verify`) | 3.0 / 1.5 | 0.670 | 0.814 | 0.646 | 0.555 | 0.663 | 0.662 / 0.670 / 0.651 / 0.620 |
| no surround | 3.0 / 1.5 | 0.671 | 0.819 | 0.651 | 0.538 | 0.674 | 0.657 / 0.671 / 0.664 / 0.635 |
| no association | 3.0 / 1.5 | 0.663 | 0.791 | 0.653 | 0.546 | 0.663 | 0.615 / 0.663 / 0.658 / 0.639 |
| V1 only | 3.5 / 1.7 | 0.672 | 0.804 | 0.659 | 0.548 | 0.677 | 0.595 / 0.655 / 0.672 / 0.663 |
| facilitation-only association + cross_k 0.5 | 3.5 / 1.7 | 0.664 | 0.807 | 0.658 | 0.518 | 0.673 | – / – / 0.664 / 0.659 |

What the ablations say:
- **At their own best, the four variants are within 0.01 on dev.** At matched recall (about 0.72), the
  contextual stages give the higher precision: clF 0.814 against 0.803 for V1-only at 4.0/2.0, strict jF
  0.646 against 0.633.
- **What the contextual stages really change is robustness.** V1-only is among the best variants on averaged
  stills and the worst on single frames. Its frame precision is 0.69: noise lumps pass a threshold tuned
  for averages. The full pipeline holds a precision of about 0.95 on both kinds at one threshold. Without
  the association field, frame precision is 0.875.
- **Stage 8 (pruning) against stages 1-7, after the review fixes.**
  - Balanced coarse typing is +0.02 on averages and +0.06 on frames.
  - Centreline recall is −0.02 / −0.01, and edge cover −0.01 / 0.00.
  - Strict junction F1 is unchanged on averages and −0.015 on frames.
  - Overall composite: 0.671 / 0.664 against 0.673 / 0.655, which is neutral.
- **Re-proposal (`annotate_repropose`)** adds junctions faster than true lines (junction precision 0.82 →
  0.54 on averages), and costs typing.

## Isolating the orientation score

**Question.** Does the advantage over curvature (Hessian) analysis come from keeping a full orientation
score `U(x, y, theta)`, or from the rest of the pipeline? The rest is the stage 1 background, the SE(2)
readout, the stage 7 graph logic with its compound prior, the refinement and the tuning. `baseline_hessian`
differs from `annotate` in all of these at once, so it cannot answer this.

**Design: swap only stage 3** (`Config.stage3 = 'hessian'`, `_hessian_layers`; the default
`'orientation'` is unchanged).
- **Same inputs and grids.** The Hessian runs on the same cleaned OD, channels, downsampled grids and scales
  (fine 1.5-3, medium 4.2-6, coarse 8.5-24 px).
- **Ridge strength.** At each scale s: the Gaussian Hessian of OD, with vesselmap's own filters
  (`scipy.ndimage.gaussian_filter`, reflect, truncate 3.5). `_hessian_eig` reimplements
  `vesselmap.ridges.hessian_eig` and gives bit-identical output on a dev image at s = 1.5, 2.1 and 3.
  Vessels are bright in OD, so l1 <= l2 and l1 is strongly negative across a vessel. The strength is
  `rho = s² max(0, -l1 - |l2|)`, `ridge_maps`' measure with aniso = 1.
- **CNR units.** rho is divided by the robust local RMS over background pixels of the signed response it
  clips. That response is the directional second derivative `-s² ∂²/∂n² (G_s * OD)` at the 16 normals n_k.
  `-s² l1` is its maximum over directions, and it is simple_cells' even filter at elongation 1.
  - The RMS comes from the same `_local_rms_stack` as simple_cells: Winsorised at 3x the global robust RMS,
    pooled over orientations, window `max(24, 4 s)` px.
  - So rho is noise-normalised the same way as the orientation score (same RMS estimator). It is not the
    same false-alarm scale: rho is clipped and suppressed by the anisotropy term, so in noise it is mostly
    0. The operating points differ, which is why the thresholds were re-swept.
- **One orientation per pixel and scale** (not strictly per pixel: see the max over scales below). The
  normalised strength goes into the layers of the Hessian's
  along-vessel direction (perpendicular to l1's eigenvector), split linearly between the two nearest of the
  16 bins. Every other layer is 0.
- **Max over scales.** Per channel, the per-layer max over its scales, the same code as simple_cells.
- **No phase gating.** There is no odd partner.
- **Everything downstream is unchanged:** surround, association field, readout, merge, end_stopping and
  refine.
- **Thresholds.** Only `t_high` / `t_low` were re-tuned.
- **Checks of the construction.**
  - `hessian_bins = 'nearest'` puts the whole strength into the nearest bin. With the linear split, a line
    half-way between two bins has half its strength in each layer.
  - `hessian_scale = 'pixel'` keeps one scale per pixel and channel, so a channel's U also has only one
    orientation per pixel. With the per-layer max, a pixel can carry the orientations of several scales:
    up to 6 non-zero layers in the fine channel, 4 in the medium and 8 in the coarse, on healthy_s004
    frame.
  - `v1_only` with `stage3 = 'hessian'` is the context-free counterpart of `annotate_v1_only` (stages 1-3,
    6-7).

**Tuning.**
- **Scores.** Composite = the mean of centreline F1, strict junction F1, balanced coarse type accuracy and
  edge cover. Per kind, each score is averaged over the 3 dev scenes; the composite is then averaged over
  average and frame.
- **Speed-up.** The sweep cached stages 1-5 per image (the thresholds enter only at the readout) and ran
  readout, merge, stage 7 and stage 8 per threshold pair. For `annotate_hessian_score`,
  `annotate_hessian_score_nearest`, `annotate_v1_only_own` and `annotate_v1_only`, the cached path's
  output digests equal the entry points' on all 6 images.
- **Grid.** The required grid is t_high 2.0-4.0 × t_low 1.0-2.0. The Hessian variant's best lay on its low
  corner (2.0 / 1.0: 0.539). The grid was therefore extended down until the best was interior.

Grid for `stage3 = 'hessian'`: composite, with (average / frame) in brackets.

| t_high \ t_low | 0.4 | 0.5 | 0.6 | 0.75 | 0.85 | 1 | 1.2 | 1.5 | 1.7 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 0.563 (0.548 / 0.579) | 0.564 (0.555 / 0.573) | 0.582 (0.582 / 0.581) | - | - | - | - | - | - | - |
| 0.85 | - | 0.565 (0.554 / 0.576) | 0.582 (0.582 / 0.582) | - | - | - | - | - | - | - |
| 1 | 0.581 (0.580 / 0.581) | 0.570 (0.565 / 0.575) | **0.585 (0.582 / 0.588)** | 0.567 (0.585 / 0.549) | 0.545 (0.559 / 0.531) | - | - | - | - | - |
| 1.25 | - | 0.574 (0.582 / 0.566) | 0.584 (0.593 / 0.574) | 0.566 (0.587 / 0.545) | 0.545 (0.560 / 0.530) | 0.551 (0.569 / 0.534) | 0.538 (0.545 / 0.531) | - | - | - |
| 1.5 | - | 0.579 (0.588 / 0.571) | 0.577 (0.590 / 0.563) | 0.564 (0.577 / 0.550) | 0.552 (0.564 / 0.541) | 0.550 (0.569 / 0.531) | 0.540 (0.549 / 0.531) | - | - | - |
| 1.75 | - | 0.577 (0.593 / 0.561) | 0.576 (0.592 / 0.560) | 0.570 (0.579 / 0.561) | 0.552 (0.567 / 0.537) | 0.551 (0.570 / 0.533) | 0.532 (0.544 / 0.521) | - | - | - |
| 2 | - | 0.569 (0.587 / 0.552) | 0.561 (0.570 / 0.552) | 0.559 (0.565 / 0.553) | 0.540 (0.556 / 0.524) | 0.539 (0.555 / 0.524) | 0.517 (0.526 / 0.508) | 0.491 (0.502 / 0.481) | 0.486 (0.477 / 0.496) | - |
| 2.5 | - | - | - | - | - | 0.520 (0.526 / 0.514) | 0.508 (0.513 / 0.503) | 0.485 (0.496 / 0.474) | 0.472 (0.474 / 0.469) | 0.417 (0.470 / 0.364) |
| 3 | - | - | - | - | - | 0.470 (0.486 / 0.454) | 0.465 (0.469 / 0.461) | 0.438 (0.450 / 0.425) | 0.417 (0.445 / 0.389) | 0.448 (0.472 / 0.424) |
| 3.5 | - | - | - | - | - | 0.450 (0.488 / 0.412) | 0.446 (0.477 / 0.416) | 0.440 (0.457 / 0.424) | 0.414 (0.446 / 0.381) | 0.404 (0.491 / 0.317) |
| 4 | - | - | - | - | - | 0.454 (0.468 / 0.439) | 0.444 (0.455 / 0.433) | 0.421 (0.444 / 0.399) | 0.426 (0.426 / 0.426) | 0.413 (0.471 / 0.355) |

- **Dev-best: 1.0 / 0.6, composite 0.585.**
- **The surface is flat near the best.** The composite is 0.570-0.585 for t_high 1.0-1.75 at t_low 0.5-0.6.
  The best thresholds are low because the Hessian's CNR is lower than the orientation score's, for two reasons:
  - Its filters are isotropic, so they do not integrate along the vessel. On a synthetic line at sigma 2.1 the peak
    is about 0.7 of the elongated (3x) even filter's.
  - The linear split puts as little as half the strength into a layer.
- **Missing scores.** In 2 cells (3.5 / 1.7, 3.5 / 2.0), healthy_s004 frame matched no junction, so its
  balanced type accuracy is undefined. It is left out of that cell's mean.

**The three checks**, each at the dev-best of its own sweep:

| variant | dev-best t_high / t_low | composite (average / frame) |
|---|---|---|
| `hessian_bins = 'nearest'` (`annotate_hessian_score_nearest`) | 1.25 / 0.5 | 0.613 (0.626 / 0.601) |
| `hessian_scale = 'pixel'` | 1.0 / 0.85 | 0.579 (0.578 / 0.580) |
| stages 1-3 + 6-7 with `stage3 = 'hessian'` | 2.0 / 0.75 | 0.558 (0.580 / 0.535) |

- **Grids.** Each check was swept over the union of the required grid and the extension: t_high 1.0-2.0 ×
  t_low 0.5-1.2 with t_low < t_high. For nearest, t_low 0.3 and 0.4 were added at t_high 1.0-1.5.
- **Each surface is flat near its best.** The values below are composite against t_high:
  - nearest: 0.599-0.613 for t_high 1.0-2.0 at t_low 0.4-0.6, and 0.582 / 0.546 / 0.523 / 0.501 at
    t_high 2.5 / 3 / 3.5 / 4 with t_low 1.0;
  - pixel: 0.570-0.579 for t_high 1.0-1.75 at t_low 0.5-0.85, and 0.503 / 0.492 / 0.473 / 0.416 at
    t_high 2.5 / 3 / 3.5 / 4 with t_low 1.0;
  - context-free Hessian: 0.546-0.558 for t_high 1.5-2.0 at t_low 0.75-1.2, and 0.545 / 0.545 / 0.515 /
    0.486 at t_high 2.5 / 3 / 3.5 / 4 with t_low 1.0.

**`annotate_v1_only` at its own thresholds, re-run after the review fixes.** Same grid, plus an edge check
at t_low 2.2 / 2.5 and t_high 4.5:

| t_high \ t_low | 1 | 1.2 | 1.5 | 1.7 | 2 | 2.2 | 2.5 |
|---|---|---|---|---|---|---|---|
| 2 | 0.517 (0.641 / 0.392) | 0.529 (0.657 / 0.400) | 0.564 (0.677 / 0.452) | 0.594 (0.683 / 0.505) | - | - | - |
| 2.5 | 0.562 (0.660 / 0.463) | 0.586 (0.683 / 0.490) | 0.601 (0.689 / 0.514) | 0.619 (0.690 / 0.549) | 0.646 (0.698 / 0.593) | - | - |
| 3 | 0.605 (0.679 / 0.532) | 0.623 (0.690 / 0.556) | 0.647 (0.700 / 0.594) | 0.658 (0.703 / 0.613) | 0.669 (0.709 / 0.629) | 0.664 (0.704 / 0.625) | 0.669 (0.694 / 0.644) |
| 3.5 | 0.645 (0.685 / 0.605) | 0.651 (0.689 / 0.614) | 0.669 (0.700 / 0.638) | 0.670 (0.699 / 0.642) | **0.674 (0.695 / 0.654)** | 0.666 (0.693 / 0.638) | 0.667 (0.688 / 0.645) |
| 4 | 0.644 (0.681 / 0.608) | 0.646 (0.677 / 0.615) | 0.658 (0.679 / 0.637) | 0.661 (0.682 / 0.640) | 0.660 (0.674 / 0.646) | 0.654 (0.675 / 0.632) | 0.657 (0.680 / 0.634) |
| 4.5 | - | - | - | - | 0.646 (0.659 / 0.634) | 0.641 (0.657 / 0.625) | 0.647 (0.670 / 0.624) |

- **The dev-best is now 3.5 / 2.0 (composite 0.674), not 3.5 / 1.7.** 3.5 / 1.7 (0.670) and 3.0 / 2.0 or
  3.0 / 2.5 (0.669) are within the noise.
- `V1_T_HIGH, V1_T_LOW = 3.5, 2.0`; the entry point is `annotate_v1_only_own`.

**Dev table.**
- Each row comes from `python -m experiments.neuromimetic.harness --annotator <fn> --scenes <3 dev scenes>
  --kinds average frame --repeat 2`, and every row is deterministic. Each row is the mean of the 3 dev scenes.
- `annotate` and `annotate_v1_only` reproduce the table above exactly.
- s is the faster run, on 2 threads, with up to two other single-thread jobs on the 4-core machine.

| annotator | stage 3 | t_high / t_low | kind | clR | clP | clF | jF | jF strict | coarse type (maj.) | coarse bal. | xR | xP | cover | purity | s | comp. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `annotate` | orientation score | 3.0 / 1.5 | average | 0.706 | 0.948 | 0.805 | 0.694 | 0.651 | 0.549 (0.522) | 0.581 | 0.375 | 0.573 | 0.649 | 0.884 | 6.1 | 0.671 |
| | | | frame | 0.701 | 0.950 | 0.801 | 0.608 | 0.587 | 0.560 (0.483) | 0.616 | 0.333 | 0.462 | 0.651 | 0.848 | 5.9 | 0.664 |
| `annotate_hessian_score` | Hessian, linear split | 1.0 / 0.6 | average | 0.639 | 0.817 | 0.714 | 0.632 | 0.560 | 0.506 (0.537) | 0.488 | 0.195 | 0.531 | 0.568 | 0.885 | 7.8 | 0.582 |
| | | | frame | 0.667 | 0.801 | 0.724 | 0.608 | 0.521 | 0.511 (0.453) | 0.530 | 0.153 | 0.583 | 0.580 | 0.873 | 9.0 | 0.588 |
| `annotate_hessian_score_nearest` | Hessian, nearest bin | 1.25 / 0.5 | average | 0.658 | 0.790 | 0.715 | 0.693 | 0.644 | 0.562 (0.514) | 0.569 | 0.216 | 0.678 | 0.576 | 0.903 | 8.4 | 0.626 |
| | | | frame | 0.680 | 0.760 | 0.715 | 0.645 | 0.586 | 0.504 (0.476) | 0.514 | 0.181 | 0.667 | 0.590 | 0.887 | 10.6 | 0.601 |
| `annotate_v1_only` (1-3, 6-7) | orientation score | 3.0 / 1.5 | average | 0.762 | 0.908 | 0.823 | 0.767 | 0.715 | 0.580 (0.499) | 0.571 | 0.408 | 0.560 | 0.692 | 0.898 | 3.5 | 0.700 |
| | | | frame | 0.778 | 0.691 | 0.723 | 0.552 | 0.528 | 0.487 (0.452) | 0.485 | 0.306 | 0.311 | 0.640 | 0.909 | 4.2 | 0.594 |
| `annotate_v1_only_own` (1-3, 6-7) | orientation score | 3.5 / 2.0 | average | 0.734 | 0.945 | 0.821 | 0.722 | 0.679 | 0.605 (0.499) | 0.590 | 0.389 | 0.574 | 0.687 | 0.876 | 3.1 | 0.695 |
| | | | frame | 0.724 | 0.870 | 0.783 | 0.663 | 0.620 | 0.481 (0.472) | 0.556 | 0.292 | 0.424 | 0.658 | 0.870 | 3.3 | 0.654 |
| `baseline_hessian` | (own pipeline) | tuned | average | 0.584 | 0.815 | 0.678 | 0.539 | 0.487 | 0.375 (0.595) | 0.393 | 0.062 | 0.241 | 0.528 | 0.788 | 0.8 | 0.521 |
| | | | frame | 0.655 | 0.665 | 0.653 | 0.541 | 0.462 | 0.424 (0.503) | 0.379 | 0.056 | 0.333 | 0.552 | 0.808 | 1.2 | 0.512 |

The context-free Hessian check (stages 1-3, 6-7 with `stage3 = 'hessian'`, 2.0 / 0.75) comes from the
cached sweep, not from an entry point:
- average: clR 0.700, clP 0.764, clF 0.728, strict jF 0.560, coarse balanced 0.463, cover 0.570;
- frame: clR 0.694, clP 0.658, clF 0.671, strict jF 0.506, coarse balanced 0.423, cover 0.540;
- composite 0.558.

**What it says** (dev only, 6 images; differences of about 0.01 are noise). Composites below are the mean of
average and frame.
- **About a third of the gap over the Hessian baseline is the orientation score; two thirds are the
  pipeline** (measured against the stronger Hessian variant, nearest bin; the review's correction).
  - `annotate` against `baseline_hessian`: 0.668 against 0.516, a gap of 0.152.
  - Swapping only stage 3 for the Hessian, re-tuned, gives 0.613 with the nearest-bin assignment
    (`annotate_hessian_score_nearest`) and 0.585 with the linear split (`annotate_hessian_score`). Keeping the
    full orientation score is therefore worth 0.055 against the stronger variant (36 % of the gap), and 0.083
    against the weaker one.
  - The rest of the pipeline, fed with Hessian orientations, is worth 0.097 (or 0.069) over the baseline.
    This covers the background, the SE(2) tracker, the graph logic and compound prior, and the refinement.
  - **This is an upper bound on the orientation score's share.** Only t_high / t_low were re-tuned for the
    Hessian. Every downstream parameter (flank_k, assoc_gain, assoc_len, claim_k, min_len, mdl_k, cnr_min,
    compound_r, ...) was tuned on the same dev scenes together with the orientation score.
- **The orientation score matters more without context.** Without stages 4, 5 and 8, at each variant's own
  thresholds:
  - orientation score (`annotate_v1_only_own`) 0.674, against 0.558 for the Hessian: a gap of 0.116;
  - the contextual stages lift the Hessian variant by only 0.027 (0.558 to 0.585, linear split; the
    context-free check was not run with nearest bins);
  - for the orientation score they are neutral on the composite (0.674 against 0.668).
- **Where the orientation score wins:**
  - **Crossings.** Crossing recall is 0.375 / 0.333 with the orientation score, 0.195 / 0.153 with the
    Hessian in the same pipeline (0.216 / 0.181 nearest-bin), and 0.06 for the baseline. Most of
    the README's crossing advantage needs the orientation score. Of the 0.30 gap in crossing recall
    between `annotate` and the baseline, about 0.12 (0.14 nearest-bin) survives with one Hessian
    orientation per pixel; that part comes from the pipeline. The other 0.16-0.18 needs a second
    orientation layer at the crossing.
  - **Line precision and recall together.** Centreline precision is 0.95 against 0.80, and recall is 0.70
    against 0.65.
  - **Line F1 and edge cover.** 0.80 against 0.71-0.72, and 0.65 against 0.57-0.59. Even and odd gating,
    elongated filters and several orientations per pixel keep texture and wall echoes out at a threshold
    that still reaches faint vessels.
- **Where it does not.**
  - With the nearest-bin assignment, strict junction F1 is close to the full annotator: 0.644 / 0.586
    against 0.651 / 0.587. Junction detection itself comes mostly from stage 7's end-stopping logic.
  - Coarse balanced typing is 0.569 / 0.514 against 0.581 / 0.616, with fewer crossings to type.
- **Construction checks.**
  - The linear split costs 0.028 against the nearest bin (about a third of the linear variant's 0.083 gap);
    the half-strength layers lower junction F1 most. The binning is therefore part of the linear variant's
    deficit, and the nearest-bin variant is the fair headline.
  - One scale per pixel ('pixel', 0.579, linear split) is no different from the per-layer max (0.585). The
    strictest form, one scale and the nearest bin, was not measured.
  - What remains with the nearest bin (0.055 on the composite, and half the crossing recall) is what one
    orientation per pixel and scale, with no elongation and no phase gating, gives in this pipeline.
- **What this does not separate.** Stage 3's design has three ingredients: several orientations per pixel
  (the score itself), elongated filters along the vessel, and odd-symmetric phase gating. This ablation
  swaps all three at once. An elongation-1 orientation score, and one without the odd partner, would split
  them.
- **V1 only at its own thresholds**, after the review fixes:
  - It equals the full annotator on dev: 0.674 against 0.668 (average 0.695 against 0.671, frame 0.654
    against 0.664).
  - Frame centreline precision is still lower, 0.870 against 0.950. Frame junction precision is the same
    (0.80 against 0.81, not in the table).
  - The README's "context is what makes it robust on frames" therefore holds for line precision. On the
    dev composite it is within noise once V1 only gets a higher t_low.

**Sanity checks.**
- **The default output is unchanged.** `annotate`'s digests on the 6 dev images are the same before and
  after this change: healthy_s000 03f8801cce4079a3 / ca282112a4c3cbdc, healthy_s004
  b23a0aa41267883a / 7b3ff07e13136f95, pathologic_s000 017517a5b3de1241 / c22cd2fa125471f6 (average /
  frame).
- **Layer counts.** In the Hessian stack, each scale has at most 2 non-zero layers per pixel, and at most 1
  with 'nearest'.
  - Where rho > 0 there are 2, or 1 where the tangent falls exactly on a bin (a few pixels with a tangent
    of exactly 0 deg). Where rho = 0 there are none, and no layer is negative.
  - The layers of a pixel sum to its normalised strength.
  - Checked on all 9 scales of all 6 dev images (`_hessian_layers`). After the max over a channel's scales
    ('layer' mode), a pixel can carry up to 6 / 4 / 8 non-zero layers (fine / medium / coarse; see Design).
- **Determinism.** All five neuromimetic rows and the baseline are deterministic over 2 runs on all 6
  images.

## Development log (what each change did)

Composite = mean(centreline F1, strict junction F1, balanced coarse type accuracy, edge cover), on the dev
images available at the time (4 images until log 7, then 6).

1. **v0** (one max over all scales, stages 1-7), s004 average only: clR 0.56, clP 0.93, jR 0.33, coarse type
   0.45. The thin vessels crossing the wide vein were all missed.
2. **Three spatial-frequency channels**, read out separately and merged: on 4 images clF 0.80, jF 0.53.
3. **Uniform elongation 4** was worse; odd_alpha 0.35 was a wash. Thresholds 4/2 → 3.5/1.7: clF 0.81.
4. **Stage 7 rework**: complete-linkage events by the thinner widths; cut at the nearest point instead of
   extending to the junction centre; T-ends extended to their attachment. jR 0.41 → 0.61, jF 0.57 → 0.70,
   but coarse typing fell to 0.48 (compounds split).
5. **Merging events or junctions more aggressively, and counting traces through a region** traded junction
   recall for typing one for one. OD-measured widths lowered junction precision then (they were kept, since
   later stages need them).
6. **First MDL pruning** gave small gains at mdl_k 2 and cost 10 % recall at mdl_k 20.
7. **The pathologic scene arrived (6 images from here).** The tangent tracker replaced skeletons of SE(2)
   components: same clF, coarse typing 0.41 → 0.45, crossing recall 0.32 → 0.39.
8. **Compound radius prior**: coarse typing 0.45 → 0.55.
9. **Lumen claims in the tracker**: junction precision 0.71 → 0.77.
10. **Per-channel elongation 3 / 2.5 / 2**: clP 0.84 → 0.89, junction precision 0.77 → 0.86 (fewer star
    rays).
11. **Background envelope 28 → 60 px** plus band 32 and sigma 24: the 80 px vessel stays in OD. Healthy
    scenes were equal or better; pathologic junction recall dropped (thin vessels on the huge vessel had
    been easier when it was "background"). Composite 0.623 → 0.637.
12. **Coarse channels on downsampled grids**: stages 3-5 three times faster; composite 0.637 against 0.646
    at full resolution.
13. **Raster-based channel merge**: 63 s → 0.2 s; merge tolerances ±1 / 0 bins were best.
14. **CNR floor calibration**: my edge CNR is about 0.23 of the truth's, so a floor of 2 removed a third of
    the true lines. Edges far from the truth have more evidence than true ones (real but offset or
    unobservable structure); pruning cannot fix them.
15. **T-extension limited to short gaps outside the other lumen**: removed parallelogram artefacts at huge
    vessels.
16. **Stage 8 render rebuilt.**
    - Per-trace smooth profiles replaced the per-piece fits, whose seams were being re-proposed.
    - The profile width is bounded by the readout scale.
    - Only free-end edges are pruned.
    - Offset gaps are joined.
17. **Junction discs weightless in the gains; re-proposal from `max(residual, 0)` with a novelty test.**
18. **MDL gain with a re-fitted amplitude.** The earlier gain `sum(2 r p + p²)` was negative for 89 % of the
    good edges it pruned. Prune-only became neutral.
19. **Knot cue** (residual at the junction centre over a_min): medians −2.4 at truth crossings, −1.8 at
    compounds, −1.4 at 3-ways. The render over-predicts every junction, so the cue does not type. Not used.
20. **Re-proposal at 3.5/1.7**: one re-proposal round composite 0.644-0.649, two re-proposal rounds
    0.629-0.642, against 0.651 without.
    A 20× cost on re-proposed edges barely changed this.
21. **Lower thresholds with the full loop** (3.2/1.5): worse then. Grid factors (1, 1, 2) were no better than
    (1, 2, 4).
22. **Runtime**: 480×768 is 4-16 s depending on load. 1200×1920 (a mirrored dev frame) takes 70 s with the
    final setting (52 s stages 1-6, 18 s stage 7-8), and took 118 s with the re-proposal loop on.
23. **Crowding cue** (a nearby junction implies compound): nearest-junction distances are alike for all truth
    types; it lowered typing. Not used.
24. **First final harness (at 3.5/1.7)** showed V1-only beating the full pipeline on averages. That led to the
    per-variant threshold sweep (Ablations): at each variant's best they are within 0.01, and the context
    helps frames.
25. **At 3.0/1.5**: stages 1-7 0.670, plus prune 0.671 (balanced typing 0.591), plus re-proposal (re_k 1.3 /
    1.5 / 2.0 / 2.5) 0.615 / 0.634 / 0.652 / 0.649. **Final**: 3.0/1.5 with render and prune, no
    re-proposal.
26. **Review fixes** (after the first final run). Composite, average / frame, same scorer:
    - **Before:** 0.680 / 0.656.
    - **Stage 7, `_split`:** a free piece is kept only if it reaches `arm_min` beyond the junction disc, the
      arm test. Composite 0.675 / 0.662.
      - Junctions whose edge count differs from their arm count: 36 → 26 over the 6 images.
      - 'pseudo-T' junctions with 4 or more edge ends: 15 → 6.
      - The remaining mismatches I inspected are short pieces between two nearby junctions.
    - **Stage 7, T events at the end's ray hit:** composite 0.671 / 0.664.
      - Anti-collinear pseudo-T pairs on one trace within 2 lambda (split crossings): 5 → 2. On
        healthy_s004 average, the truth crossing at (185, 135) is now one 'crossing', 2.5 px off, instead of
        two pseudo-Ts 13.6 px apart.
      - Matched junctions lie closer to the truth on the healthy averages: median 2.8 → 2.4 px and
        3.7 → 2.6 px.
      - Some T events now merge, so junction F1 falls by 0.01-0.02.
    - **Both stage 7 fixes together:**
      - Junction F1: 0.714 / 0.616 → 0.694 / 0.608.
      - Coarse typing: 0.545 / 0.515 → 0.549 / 0.560 (balanced 0.594 / 0.575 → 0.581 / 0.616).
      - All of this is within the noise of six images.
    - **Stage 8:**
      - `rounds` now counts the render/prune passes (2). Before, the loop ran `rounds + 1` times; the output
        is unchanged.
      - The re-proposed flag is carried per point through gap joins (`annotate_repropose` only).
    - **Stage 1:** the background leak was measured, and `bg_trace_k` was tried and not adopted (stage 1).

## What failed

- **One max over scales** (v0) lost every thin vessel crossing a wide vein. This is the single biggest fix:
  separate frequency channels.
- **A 28 px background envelope** absorbed the widest pathologic vessel into the background.
- **Skeletons of SE(2) connected components** as the readout. They were comparable on lines, but dropped
  short internal paths and typed junctions worse than tracing along the tangent.
- **Long coarse filters** (elongation 3 at sigma 8.5-24) drew star rays from wide-vessel knots.
- **Stage 8 MDL as first written.**
  - Per-piece profile fits put seams across wide vessels, which were then re-proposed as vessels.
  - A gain of `sum(2 r p + p²)` was negative wherever tubes overlapped.
  - Pruning interior edges fragmented veins.
  - A CNR floor on the truth's scale (2) removed a third of the true lines.
- **Re-proposal from the residual**, at every threshold and cost tried. Near rendered vessels the positive
  residual is dominated by model misfit (profile shape, centreline offsets, junction knots), not by missed
  vessels. The new traces mostly add false T arms: junction precision 0.82 → 0.54.
- **The knot cue.** In principle a crossing shows an additive dark knot and a fork a union. On dev the
  residual at junction centres is dominated by render error, because the additive render over-predicts
  every junction by about twice the thinner vessel's contrast. It does not separate the types.
- **Merging nearby junctions, counting all traces through a region, or a crowding rule** for compounds.
  These only traded junction recall for typing.
- **Facilitation-only association and weaker cross-orientation suppression** (cross_k 0.5). Within noise of
  the default.
- **Per-point coverage loop in the channel merge.** It cost 63 s of a 101 s refinement and was replaced by
  orientation-binned coverage rasters (0.2 s).

## Open problems

- **Faint lines.**
  - Truth-CNR 3-7 vessels are the bulk of the missed centreline. My band CNR is about 0.23 of the truth's.
  - One cause is the background leak of stage 1. Faint vessels missed by the mask enter the B fit, so their
    OD comes out at 0.56-0.70 of a leak-free refit, and they also enter the background RMS. Removing the leak
    for the traced vessels (`bg_trace_k`) did not raise the dev scores (stage 1). The untraced faint
    vessels, the ones that matter for recall, would need a better vesselness mask.
  - A texture "web" in the orientation score (99th percentile of background C about 5) sets the working
    threshold: in truth terms it is roughly CNR 5-7.
  - The truth also counts thin vessels lying on a 60-100 px vessel by their own contrast. Most of these are
    missed, which is why pathologic recall is 0.54-0.56.
  - Better noise models inside dark vessels (log-domain shot noise rises as 1/sqrt(I)) and longer, curved
    integration (a real co-circular association field rather than ±1 orientation layer) are the obvious
    next steps.
- **Typing is detection-limited.**
  - Most truth compounds have 5-6 lines and I trace 3-4 of them; most truth crossings typed 3-way are
    missing their fourth arm.
  - The radius prior is a crutch, tuned on six images, and the majority-label margin rests on it. Without it,
    coarse typing is below the majority reference in the mean of both kinds (0.405 / 0.437 against
    0.522 / 0.483). With it, typing is still below the reference on 2 of 6 images: both pathologic stills,
    where compounds are 59-73 % of the matched junctions.
- **A junction-aware render.** The additive tube render is right for a crossing at different depths and
  wrong for a fork (a union). A proper version would render both hypotheses per junction (sum against
  union, with depth attenuation) and compare their residuals. That would make the knot cue usable as a
  crossing-vs-fork test, and would make pruning and re-proposal near junctions trustworthy. As things stand
  it is the reason stage 8 does not help.
- **Bifurcation against confluence** cannot be decided from a still; 3-way junctions are labelled
  'pseudo-T'.
- **Overfitting risk.** Everything was tuned on three scenes. The differences between ablations (±0.01)
  are within the noise of that sample. The radius prior and the thresholds are the most likely to move on
  held-out scenes.
- **Speed.** 4-16 s per 480×768 image and about 70 s per 1200×1920 image on 2 threads. Stages 3-6 are
  three quarters of it. With re-proposal they run again on the residual; caching per channel, or running
  the fine channel only, would halve that.
