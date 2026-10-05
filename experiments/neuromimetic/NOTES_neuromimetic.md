# The neuromimetic annotator: notes

`neuromimetic.py` implements DESIGN.md: a deterministic annotator of vesselscene stills built from what the
early visual system is thought to do with contours, plus a render-and-compare refinement borrowed from
diffusion models. These notes give each stage's biology and maths, the final parameters, the dev results
with the ablations and the curvature (Hessian) baseline, what each change did, what failed, and what is
still open.

**Dev only.** Everything below was tuned and measured on the three dev scenes (healthy_s004_320x512,
healthy_s000_480x768, pathologic_s000_480x768; average and frame, so 6 images). The numbers are therefore
optimistic. The held-out test scenes were never opened.

RESULTS_PLACEHOLDER

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

### 6. Readout: `readout` (and `merge_channels`)

- **Candidates.** C_k must be a maximum along the normal n_k (bilinear neighbours at ±1 px) and a maximum
  over orientation (`C_k ≥ C_k-1`, `C_k > C_k+1`, so several peaks per pixel are allowed). C_k must exceed
  `t_low = 1.7`, and the pixel must be at least 3 px from the frame and from invalid pixels.
- **Tracking.** Seeds are the candidates above `t_high = 3.5`, strongest first, with ties broken by
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
- **Replaced.** The first readout was skeletons of the SE(2) connected components (`scipy.ndimage.label`
  over (theta, y, x), theta circular). Both gave the same centreline scores, but the tracker typed junctions
  better and found more crossings (log 7).

### 7. End-stopped cells, graph assembly: `end_stopping`

- **Biology.** End-stopped (hypercomplex) cells signal line terminations. A line ending on another is a T
  (or a Y); two lines continuing through a point are an X.
- **Maths.**
  - Collinear end-to-end gaps up to 14 px are bridged (mutual best pairs, turn < 40°, short offset gaps
    allowed).
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
      regions gather lines too faint to trace (log 8, 19).
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
      `mdl_k (1 + L / mdl_len)`. Here p is the edge's tube used as a template and r_-e is the residual
      without the edge; its amplitude is re-fitted against what the others leave (explaining away).
      Junction discs (0.7 R) carry no weight, and sigma² is the local RMS² of the high-passed OD over
      background pixels.
    - The band CNR of its OD profile against stage 2's band RMS along it must reach `cnr_min`.
    - Edges are removed greedily, weakest first, and the gains of the overlapping edges are updated.
  - **Re-proposal.** Stages 3-6 run again on `max(residual, 0)` outside the junction discs, at 1.3× the
    thresholds. A new trace is kept if it runs at least 20 px outside the lumens already rendered and is not
    a duplicate or wall echo of the graph's vessels. New edges pay 20× the description cost in the next
    prune.
- **After the last round.** The graph is assembled again. The **knot cue** (mean residual at each junction
  centre over the thinner arm's contrast) is stored on each junction but not used (see What failed).

Fixed numbers: one round (`rounds = 1`). DESIGN suggested 2-3; two rounds were worse on dev (log 20).

## Final parameters (`Config`)

| stage | parameters |
|---|---|
| 1 | env_radius 60, bg_sigma 8, bg_iters 2, mask_z 2, mask_dilate 2, saturation 4095 |
| 2 | bands 1, 2, 4, 8, 16, 32; rms_sigma 24 |
| 3 | 16 orientations; scales 1.5, 2.1, 3, 4.2, 6, 8.5, 12, 17, 24; channels (0-2), (3-4), (5-8); grid factors 1, 2, 4; elong 3, 2.5, 2; odd_alpha 0.7 |
| 4 | cross_k 1.0, flank_k 0.3 |
| 5 | assoc_len 8, assoc_steps 3, assoc_gain 1 |
| 6 | t_high 3.5, t_low 1.7, min_len 10, border 3, look 3, pass 4, claim_k 0.5, merge bins ±1 / 0 |
| 7 | gap_att 6, cone 45°, ext_max 12, gap_join 14, join_turn 40°, r0 4, link_k 1, share_k 1, arm_min 6, cross_turn 40°, compound_r 16, OD widths |
| 8 | rounds 1, re_k 1.3, re_mdl 20, re_novel 2, jmask_k 0.7, corr_px 6, mdl_k 2, mdl_len 20, cnr_min 0.5 |

ABLATION_PLACEHOLDER

## Development log (what each change did)

Composite = mean(centreline F1, strict junction F1, balanced coarse type accuracy, edge cover), on the dev
images available at the time (4 images until log 7, then 6).

LOG_PLACEHOLDER

## What failed

- **One max over scales** (v0) lost every thin vessel crossing a wide vein. This is the single biggest fix:
  separate frequency channels.
- **A 28 px background envelope** absorbed the widest pathologic vessel into the background.
- **Skeletons of SE(2) components** as the readout. They were comparable on lines, but dropped short
  internal paths and typed junctions worse than tracing along the tangent.
- **Long coarse filters** (elongation 3 at sigma 8.5-24) drew star rays from wide-vessel knots.
- **Stage 8 MDL as first written.**
  - Per-piece profile fits put seams across wide vessels, which were then re-proposed as vessels.
  - A gain of `sum(2 r p + p²)` was negative wherever tubes overlapped. 89 % of the good edges it pruned
    had negative gain.
  - Pruning interior edges fragmented veins.
  - A CNR floor of 2 removed a third of the true lines: my edge CNR is about 0.23 of the truth's.
- **Two or more re-proposal rounds.** Each round adds false T arms faster than true lines (junction
  precision 0.87 → 0.59 at two rounds before the novelty and cost rules, still slightly worse after).
- **The knot cue.** In principle a crossing shows an additive dark knot and a fork a union. On dev the
  residual at junction centres is dominated by render error: the additive render over-predicts every
  junction by about twice the thinner vessel's contrast (median knot −2.4 at truth crossings, −1.8 at
  compounds, −1.4 at 3-ways). It does not separate the types, so it is stored but not used.
- **Merging nearby junctions into compounds, or counting every trace through a region as arms.** These
  traded junction recall for type accuracy one for one.
- **Lower detection thresholds plus pruning.** Precision fell faster than recall rose; pruning cannot
  separate a real-but-unobservable vessel from an observable one.
- **Per-point coverage loop in the channel merge.** It cost 63 s of a 101 s refinement and was replaced by
  orientation-binned coverage rasters (0.2 s).

## Open problems

OPEN_PLACEHOLDER
