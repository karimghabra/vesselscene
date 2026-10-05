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
| neuromimetic `annotate`, average | **0.807** | **0.714** | **0.667** | **0.545** (0.530) | **0.594** | 0.36 / 0.55 | **0.654** | 6.9 |
| Hessian baseline, average | 0.678 | 0.539 | 0.487 | 0.375 (0.595) | 0.393 | 0.06 / 0.24 | 0.528 | 0.8 |
| vesselmap (LIMBUS), average * | 0.684 | 0.607 | – | 0.320 | – | 0.40 / 0.16 | 0.610 | 564 |
| neuromimetic `annotate`, frame | **0.801** | **0.616** | **0.594** | **0.515** (0.459) | **0.575** | 0.31 / 0.46 | **0.653** | 6.5 |
| Hessian baseline, frame | 0.653 | 0.541 | 0.462 | 0.424 (0.503) | 0.379 | 0.06 / 0.33 | 0.552 | 1.2 |
| vesselmap (LIMBUS), frame * | 0.715 | 0.600 | – | 0.323 | – | 0.40 / 0.25 | 0.617 | 442 |

\* vesselmap was run by the orchestrator with an earlier version of the harness, before the strict and
balanced scores existed. Its crossing precision and edge cover were computed slightly differently.

Two numbers stand out:
- **Crossings.** The Hessian finds 6 % of the crossings; the orientation score finds 31-36 %. In the score,
  a crossing's two vessels live in different orientation layers, so neither breaks the other.
- **Coarse typing.** The annotator types junctions above the image's majority-label reference (0.545
  against 0.530 on averages, 0.515 against 0.459 on frames). Its balanced accuracy is 0.58-0.59, against
  0.33 for a constant label. The Hessian baseline stays below the majority reference.

**Which ideas did the work.**
- **Most of the gain comes from the V1 stage and the readout.**
  - Separate spatial-frequency channels of an orientation score (a thin vessel crossing a wide vein
    survives).
  - Phase gating by the odd partner (no echoes along vein walls).
  - Tracing along the tangent in SE(2), so a line runs straight through a crossing in its own layer.
  - End-stopping (graph) logic built on that.
- **Contour integration and the surround buy robustness to noise, not peak quality.** With both switched off
  (`annotate_v1_only`), the averaged stills score as well or better: composite 0.708 against 0.680. But the
  single frames collapse: centreline precision drops from 0.945 to 0.683, and the composite from 0.656 to
  0.602. Only the variants that keep both contextual stages (`annotate`, and stages 1-7) are good on both
  kinds. With each variant at its own best threshold, all four lie within 0.01 on dev (see Ablations).
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

### 2. OFF-centre ganglion cells with contrast gain control: `ganglion_cells`

- **Biology.** Centre-surround (difference of Gaussians) cells whose gain is divided by the local contrast,
  so their output is a contrast-to-noise ratio rather than a contrast.
- **Maths.**
  - For each band `s ∈ {1, 2, 4, 8, 16, 32}`: `D_s = (G_s - G_2s) * OD`, divided by the robust local RMS of
    the same band over the background pixels. The RMS is the Winsorised (at 3× the global robust RMS)
    masked normalised convolution of `D_s²`.
  - `Z = max_s D_s / RMS_s`. This is the band-matched CNR the truth uses to decide what is observable.
- **Used for.** The mask iterations of stage 1, and the visibility floor of stage 8 (the RMS maps).

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

- **Biology.** Cross-orientation normalisation by the pooled activity of all orientations at the location,
  and iso-orientation surround suppression. A contour on an empty background is salient; inside texture it
  is not.
- **Maths.**
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
    so polylines run junction to junction.

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
      about the truth's CNR 2, since my edge CNR is about 0.23 of the truth's.
    - Edges are removed greedily, weakest first; the gains of overlapping edges are updated, and the removed
      ranges are cut out of their traces.
  - **Re-proposal**, optional (`repropose`; `annotate_repropose`). Stages 3-6 run again on
    `max(residual, 0)` outside the junction discs, at 1.3× the thresholds. A new trace is kept if it runs at
    least 20 px outside the lumens already rendered and is not a duplicate or wall echo of the graph's
    vessels. New edges pay 20× the description cost in the next prune.
- **After the last round.** The graph is assembled again. The **knot cue** (mean residual at each junction
  centre over the thinner arm's contrast) is stored on each junction but not used (see What failed).
- **Final setting.** One fixed round of render, prune and re-assemble, without re-proposal. Every
  re-proposal setting lowered the dev scores (log 20, 25). DESIGN's 2-3 rounds were tried, and two rounds
  were worse than one.

## Final parameters (`Config`)

| stage | parameters |
|---|---|
| 1 | env_radius 60, bg_sigma 8, bg_iters 2, mask_z 2, mask_dilate 2, saturation 4095 |
| 2 | bands 1, 2, 4, 8, 16, 32; rms_sigma 24 |
| 3 | 16 orientations; scales 1.5, 2.1, 3, 4.2, 6, 8.5, 12, 17, 24; channels (0-2), (3-4), (5-8); grid factors 1, 2, 4; elong 3, 2.5, 2; odd_alpha 0.7 |
| 4 | cross_k 1.0, flank_k 0.3 |
| 5 | assoc_len 8, assoc_steps 3, assoc_gain 1, assoc_fill off |
| 6 | t_high 3.0, t_low 1.5, min_len 10, border 3, look 3, pass 4, claim_k 0.5, merge bins ±1 / 0 |
| 7 | gap_att 6, cone 45°, ext_max 12, gap_join 14, join_turn 40°, r0 4, link_k 1, share_k 1, arm_min 6, cross_turn 40°, compound_r 16, OD widths |
| 8 | rounds 1, repropose off (re_k 1.3, re_mdl 20, re_novel 2 when on), jmask_k 0.7, corr_px 6, mdl_k 2, mdl_len 20, cnr_min 0.5 |

## Dev results and ablations

All runs used `python -m experiments.neuromimetic.harness --annotator experiments.neuromimetic.neuromimetic:<fn>
--scenes <3 dev scenes> --kinds average frame --repeat 2` (the current harness). Every run reported
`deterministic = True`. Each row is the mean of the 3 dev scenes.
- "comp." is my summary, the mean of centreline F1, strict junction F1, balanced coarse type accuracy and
  edge cover.
- "maj." is the majority-label reference for coarse typing.
- Times are the faster of the two runs, on 2 threads of a loaded shared machine.

| annotator | kind | clR | clP | clF | jR | jP | jF | jF strict | coarse type (maj.) | coarse bal. | exact type | xR | xP | cover | pcs/edge | s | comp. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `annotate` (1-8) | average | 0.710 | 0.948 | 0.807 | 0.616 | 0.852 | 0.714 | 0.667 | 0.545 (0.530) | 0.594 | 0.507 | 0.357 | 0.554 | 0.654 | 1.10 | 6.9 | 0.680 |
| | frame | 0.704 | 0.945 | 0.801 | 0.501 | 0.834 | 0.616 | 0.594 | 0.515 (0.459) | 0.575 | 0.466 | 0.306 | 0.463 | 0.653 | 1.09 | 6.5 | 0.656 |
| `annotate_no_verify` (1-7) | average | 0.736 | 0.942 | 0.821 | 0.636 | 0.831 | 0.719 | 0.666 | 0.518 (0.511) | 0.551 | 0.490 | 0.411 | 0.551 | 0.669 | 1.12 | 4.5 | 0.677 |
| | frame | 0.720 | 0.933 | 0.807 | 0.543 | 0.822 | 0.647 | 0.627 | 0.527 (0.449) | 0.560 | 0.484 | 0.319 | 0.406 | 0.658 | 1.12 | 4.4 | 0.663 |
| `annotate_no_association` | average | 0.724 | 0.932 | 0.810 | 0.669 | 0.840 | 0.742 | 0.681 | 0.572 (0.515) | 0.612 | 0.541 | 0.367 | 0.559 | 0.666 | 1.13 | 6.0 | 0.692 |
| | frame | 0.732 | 0.874 | 0.791 | 0.640 | 0.764 | 0.687 | 0.636 | 0.479 (0.497) | 0.513 | 0.438 | 0.306 | 0.432 | 0.648 | 1.14 | 6.6 | 0.647 |
| `annotate_no_surround` | average | 0.710 | 0.946 | 0.805 | 0.643 | 0.852 | 0.732 | 0.696 | 0.490 (0.518) | 0.536 | 0.457 | 0.340 | 0.487 | 0.653 | 1.10 | 6.1 | 0.672 |
| | frame | 0.728 | 0.926 | 0.809 | 0.540 | 0.830 | 0.638 | 0.607 | 0.471 (0.478) | 0.504 | 0.446 | 0.306 | 0.382 | 0.667 | 1.10 | 6.1 | 0.647 |
| `annotate_v1_only` (1-3, 6-7) | average | 0.772 | 0.902 | 0.827 | 0.728 | 0.821 | 0.771 | 0.723 | 0.586 (0.498) | 0.584 | 0.555 | 0.408 | 0.544 | 0.698 | 1.14 | 3.8 | 0.708 |
| | frame | 0.788 | 0.683 | 0.722 | 0.780 | 0.465 | 0.554 | 0.542 | 0.483 (0.439) | 0.497 | 0.446 | 0.278 | 0.288 | 0.648 | 1.24 | 4.5 | 0.602 |
| `annotate_repropose` (8 with re-proposal) | average | 0.747 | 0.819 | 0.777 | 0.757 | 0.520 | 0.615 | 0.557 | 0.484 (0.510) | 0.516 | 0.446 | 0.349 | 0.407 | 0.594 | 1.28 | 13.9 | 0.611 |
| | frame | 0.744 | 0.882 | 0.801 | 0.613 | 0.634 | 0.613 | 0.572 | 0.423 (0.486) | 0.460 | 0.379 | 0.264 | 0.355 | 0.641 | 1.17 | 11.2 | 0.619 |
| `baseline_hessian` | average | 0.584 | 0.815 | 0.678 | 0.432 | 0.774 | 0.539 | 0.487 | 0.375 (0.595) | 0.393 | 0.352 | 0.062 | 0.241 | 0.528 | 1.15 | 0.8 | 0.521 |
| | frame | 0.655 | 0.665 | 0.653 | 0.524 | 0.618 | 0.541 | 0.462 | 0.424 (0.503) | 0.379 | 0.424 | 0.056 | 0.333 | 0.552 | 1.26 | 1.2 | 0.512 |

**Full annotator per scene**:

| scene | kind | clR | clP | clF | jF strict | coarse type (maj.) | coarse bal. | xR / xP | cover | s |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy_s000_480x768 | average | 0.879 | 0.968 | 0.921 | 0.783 | 0.600 (0.413) | 0.630 | 0.43 / 0.71 | 0.806 | 9.8 |
| healthy_s000_480x768 | frame | 0.861 | 0.957 | 0.907 | 0.731 | 0.604 (0.226) | 0.620 | 0.58 / 0.78 | 0.791 | 8.3 |
| healthy_s004_320x512 | average | 0.688 | 0.966 | 0.804 | 0.738 | 0.474 (0.553) | 0.547 | 0.33 / 0.50 | 0.634 | 3.7 |
| healthy_s004_320x512 | frame | 0.709 | 0.958 | 0.815 | 0.627 | 0.531 (0.469) | 0.628 | 0.21 / 0.50 | 0.659 | 3.6 |
| pathologic_s000_480x768 | average | 0.563 | 0.910 | 0.695 | 0.481 | 0.562 (0.625) | 0.605 | 0.31 / 0.46 | 0.522 | 7.4 |
| pathologic_s000_480x768 | frame | 0.541 | 0.920 | 0.681 | 0.422 | 0.409 (0.682) | 0.478 | 0.13 / 0.11 | 0.509 | 7.7 |

### Ablations at each variant's own best thresholds

The ablations above share the thresholds of the full annotator. But the surround and the association field
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
- **What the contextual stages really change is robustness.** V1-only is the best variant on averaged
  stills and the worst on single frames. Its frame precision is 0.68: noise lumps pass a threshold tuned
  for averages. The full pipeline holds a precision of about 0.945 on both kinds at one threshold.
- **Stage 8 (pruning) against stages 1-7.**
  - Balanced coarse typing is +0.04 on averages and +0.015 on frames.
  - Centreline recall is −0.02 to −0.03, and edge cover −0.01.
  - Strict junction F1 is unchanged on averages and −0.03 on frames.
  - Overall composite: 0.668 against 0.670, which is neutral.
- **Re-proposal (`annotate_repropose`)** adds junctions faster than true lines (junction precision 0.85 →
  0.52 on averages), and costs typing.

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
20. **Re-proposal at 3.5/1.7**: 1 round composite 0.644-0.649, 2 rounds 0.629-0.642, against 0.651 without.
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
  vessels. The new traces mostly add false T arms: junction precision 0.85 → 0.52.
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
  - The radius prior is a crutch, tuned on six images. Coarse typing is above the majority reference on
    average, but below it on the pathologic scene, where compounds are 62-68 % of junctions.
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
