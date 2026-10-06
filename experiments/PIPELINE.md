# From image to graph: every manipulation, step by step, with images

This report follows one image through the whole pipeline, from pixels to the exported graph, and shows what
every step produces. It is meant as a recipe to re-implement by hand: each step gives the operation, its
formula and parameters, a figure of its output, and what to look for in the figure. The last section lists
what to change, with the evidence for each change.

**The image** is a development scene, healthy 7, averaged still, 480 x 768 px. Development scenes were used
for tuning, so its scores are better than on held-out images; [REPORT.md](REPORT.md) has the held-out numbers.
Each figure shows the full frame and, below or beside it, a 192 px zoom (the yellow box). The zoom is the
window with the most kinds of truth junction: 15 junctions, 7 of them crossings, 4 three-way and 4 compound.

**How the figures were made.** `python -m experiments.splinefit.pipeline_figures` calls the pipeline's own
functions in order, with the default parameters. It checks that its stages 1-6 give the same 207 merged
traces as `neuromimetic.propose`. The truth is read only to choose the zoom, in step 13, and in the last
section.

| step | model | manipulation | output |
|---|---|---|---|
| 1 | photoreceptors, horizontal cells | log; background fitted outside a vessel mask | OD map (the fit's target) |
| 2 | OFF-centre ganglion cells | difference of Gaussians per band, over the local RMS | CNR map Z |
| 3 | V1 simple cells | even / odd oriented filter pairs, 16 orientations x 9 scales in 3 channels | orientation score U(x, y, theta) |
| 4 | non-classical surround | subtract the isotropic part and the weaker flank | texture suppressed |
| 5 | association field | bipole diffusion along the orientation | gaps completed: C(x, y, theta) |
| 6 | readout | non-maximum suppression, hysteresis tracing in (x, y, theta), channel merge, widths | traces |
| 7 | end-stopped cells | T and X events, clustered into typed junctions; traces cut at them | graph |
| 8 | render, compare, prune | blurred-box profiles, additive render, MDL and visibility tests | pruned graph |
| 9 | spline network | chains through crossings and through-nodes become B-splines | initial network |
| 10 | render model | cylinder chord, blur, union at nodes, kappa at crossings, halo | differentiable render |
| 11 | loss | precision-weighted squared residual plus priors | one scalar |
| 12 | fit schedule | profiles, everything, prune, retarget, profiles | fitted network |
| 13 | export | edges cut at nodes, typed junctions, crossings | the graph |

## Steps 1-8: proposing the graph

### Step 1. Optical density and the background

**Input:** the averaged still I and its valid-pixel map (inside the aperture, not saturated).

1. Take the log: `L = ln I`. Fill invalid pixels from the nearest valid pixel.
2. Make a provisional background `B0`: a grey closing of L with a disc of radius 60 px, wider than any vessel.
   The closing fills every dark structure narrower than the disc from its surround. It runs on a grid
   downsampled 4x and is smoothed afterwards.
3. Repeat twice:
   - `OD = B - L`;
   - vessel mask M = (band CNR Z > 2, step 2) or (OD smoothed at 1 px > 3 x its local RMS), dilated by 2 px;
   - refit B on the pixels outside M only, by masked normalised convolution:
     `B = G_8 * (L w) / G_8 * w`, with `w = valid and not M` and a Gaussian of sigma 8 px. Holes are filled
     from coarser scales.
4. **Output:** `OD = B - L` in nepers. Vessels are positive. By Beer-Lambert, overlapping vessels add their
   densities.

This is your residual rule. The fit's target is the OD cleaned by a background fitted on the mask's
negative. It is never the mask, and never a vesselness map.

![step 1](splinefit/results/pipeline/step1_background.jpg)

*Top row: I, L and the provisional background B0. Bottom row: the mask M, the refitted background B and the
OD. The zoom shows the thin vessels crossing below the wide one.*

**What to look for:**
- B has no vessels in it: wherever M covers a vessel, B is interpolated from around it. The facets in B are
  the edges of masked regions.
- Where M misses part of a vessel, usually a faint or wide one, B dips into it and the OD loses that
  vessel's density. This is the background leak; improvement 1 below shows how large it is.

### Step 2. Contrast-to-noise per band

For each band s in {1, 2, 4, 8, 16, 32} px:
- take `D_s = G_s * OD - G_2s * OD`, an OFF-centre difference of Gaussians (a dark vessel in I is a bright
  ridge of OD);
- divide it by its robust local RMS over the background pixels: D_s squared, Winsorised at (3 x the global
  robust RMS) squared, then a masked Gaussian mean with sigma = max(24, 4s) px.

Then `Z = max over s of D_s / RMS_s`. Z feeds step 1's mask (Z > 2) and step 8's visibility test. The
orientation stages filter the OD itself, not Z.

![step 2](splinefit/results/pipeline/step2_cnr.jpg)

*The band s = 2 and its local RMS (top); the band s = 8 and Z (bottom).*

**What to look for:** the RMS is high where the background is textured. Z is a contrast-to-noise ratio, so
the same contrast counts for less there.

### Step 3. Oriented filters: the simple cells

**Orientations:** theta_k = k pi / 16, k = 0 ... 15. The tangent is t = (cos theta, sin theta), the normal
is n.

**Filters:** at scale sigma, take an anisotropic Gaussian with std sigma across and e sigma along. e is 3,
2.5 and 2 in the fine, medium and coarse channels.
- even filter: `-sigma^2 d^2G/dn^2`, a centre-on line detector;
- odd filter: `sigma dG/dn`, an edge detector.

Both are applied exactly by FFT with reflect padding, so the 16 orientations are equivalent.

**Channels:**

| channel | scales (px) | grid |
|---|---|---|
| fine | 1.5, 2.1, 3.0 | full |
| medium | 4.2, 6.0 | downsampled 2x |
| coarse | 8.5, 12, 17, 24 | downsampled 4x |

**Normalisation:** each response is divided by the robust local RMS of its scale over the background,
pooled over orientations. The kernel's noise gain cancels, so z is a CNR.

**Line evidence:** `z = z_even - 0.7 |z_odd|`. This is phase gating. On a vessel's centre the odd response
is zero by symmetry. On the wall of a wide vessel it is large, so wall echoes are rejected.

**Per channel:** U(theta) is the max over the channel's scales, and S records which scale won.

![step 3 filters](splinefit/results/pipeline/step3_filters.jpg)

*The filter pair at two scales and four of the 16 orientations: even (top) and odd (bottom).*

![step 3 orientation](splinefit/results/pipeline/step3_orientation.jpg)

*The score per channel: the max over orientations, coloured by the winning orientation (key in the lower
left of the first zoom).*

**What to look for:**
- At a crossing, each vessel keeps its own colour through the crossing point: two orientations are present
  at one pixel. This is what lets step 6 trace straight through a crossing.
- The fine channel sees the thin vessels crossing the wide one; the coarse channel sees only the wide ones.
  That is why the channels are kept apart rather than taking one max over all scales.

### Step 4. The surround: texture suppression

Let `P = max(U, 0)`.
1. **Isotropic part.** `iso` = the mean of the lower half of P over theta. A blob responds at every
   orientation; a line responds at one, and a crossing at two. Set `U1 = U - 1.0 iso`.
2. **Flanks.** The flank energy F_left and F_right is the mean of P_theta in a strip on each side of the
   line, from d1 = 2.5 sigma + 1 to d1 + max(6, 1.5 sigma) px. Set `U2 = U1 - 0.3 min(F_left, F_right)`.

Only the weaker flank counts. Texture surrounds a pixel on both sides; a neighbouring vessel in a bundle is
on one side only.

### Step 5. The association field: contour completion

1. Pool P over neighbouring orientation layers: `max(P_k, 0.8 P_k+-1)`. This lets a contour turn by one
   layer (co-circularity).
2. Sum the two lobes, ahead and behind: `A+-_k(p) = sum_d w(d) P_k(p +- d t_k)`, with
   `w(d) = exp(-d^2 / 2 l^2)`, `l = 8 sqrt(max(1, sigma / 2))` and d = 1 ... 3l. The lobes are spread
   0.7 px sideways and normalised.
3. Take the bipole `B_k = sqrt(A+_k A-_k)`. It needs support on both sides, so it fills a gap but does not
   extend a line past its end.
4. Run three steps of `C <- (U + B(C)) / 2`, starting from `C0 = max(U, 0)`.

![steps 4 and 5](splinefit/results/pipeline/step45_context.jpg)

*Fine channel. Top: the score after steps 3, 4 and 5, full brightness at the seed threshold 3. Bottom: what
step 6 will see: white above 3 (seeds), grey above 1.5 (where tracing may continue).*

**What to look for:**
- Step 4 removes the texture specks on the left of the frame.
- Step 5 bridges the dips where two vessels cross (zoom) and continues faint vessels in grey.

### Step 6. Readout: tracing the vessels

**Per channel:**
1. **Candidates.** A sample (theta_k, x, y) is a candidate if all of these hold:
   - C_k is a local maximum along the normal (bilinear neighbours at +-1 px);
   - C_k is at least its neighbours in theta (C_k >= C_k-1 and C_k > C_k+1);
   - C_k > t_low = 1.5;
   - it is at least 3 px from the frame and from invalid pixels.
2. **Seeds.** Candidates above t_high = 3, strongest first.
3. **Tracing.** From a seed, step 1 px along the tangent in both directions. The next sample is the best
   candidate among positions 0 and +-1 px across the predicted one, in layers k and k+-1. Each candidate's
   C is reduced by 15 % per px of lateral offset and 10 % per layer of turn. Further rules:
   - look up to 3 px ahead to cross small gaps;
   - pass through another trace's claimed pixels for at most 4 px (a shallow crossing);
   - stop where nothing continues, or where the trace starts running along another trace (a T: the end is
     put at the first contact).
4. **Claims.** Every accepted sample claims +-max(1, 0.5 sigma) px across it, in layers k and k+-1, so a
   lumen holds one trace.
5. **Clean-up.** Drop traces shorter than 10 px; smooth the rest and resample at 1 px.

**Merging channels.** Strongest first, drop a point where an accepted trace runs parallel (within one
orientation bin of 11.25 deg) within 3 px. Keep the uncovered runs of at least 10 px. Here 217 fine, 138
medium and 73 coarse traces merge into 207.

**Widths** come from the OD, not from a filter. Along each trace, sample the cross-section along the normal
over +-max(6, 1.5 w0) px (at most 60), every 0.5 px. The baseline is the median of the outer fifth on each
side. The width is the full width at half maximum, smoothed by a running median of 7 along the trace.

![step 6 traces](splinefit/results/pipeline/step6_traces.jpg)

*Left: the traces of each channel. Right: after merging, one colour per trace.*

![step 6 width](splinefit/results/pipeline/step6_width.jpg)

*One cross-section of the widest long trace and its full width at half maximum.*

**What to look for:**
- The channels overlap: the same vessel is traced at several scales, and the merge keeps one trace per
  vessel.
- The merged traces break up around the crossing cluster in the zoom. Pieces of one vessel come from
  different channels or stop at another trace's claim. Every later step inherits this fragmentation.

### Step 7. Junctions: the end-stopped cells

1. **Bridge gaps.** Join collinear ends at most 14 px apart that turn by less than 40 deg (mutual best
   pairs).
2. **Attach ends.** Extend a trace that ends within 12 px of another trace's centreline, outside its lumen,
   to the attachment point.
3. **Find events.**
   - T: an end lies within 6 px + w/2 of another trace, either in a 45 deg cone ahead of the end or inside
     the other trace's lumen.
   - X: two traces intersect.
4. **Cluster events** by complete linkage. Two events join when they are closer than r0 + (w_a + w_b) / 2,
   with r0 = 4 px, which is the size of the lumen overlap. The junction centre is the mean of its events;
   its radius is the farthest reach.
5. **Type by arms.** An arm is a trace leaving the junction disc, at least 6 px long.
   - 3 arms: a 3-way junction.
   - 4 arms that pair into two straight lines (turn < 40 deg) at clearly different orientations: a
     crossing.
   - Otherwise (4 unpaired arms, or 5 or more): compound.
   - A region of radius 16 px or more is compound whatever its arms. This is a prior fitted on dev.
6. **Cut.** Cut every trace at its point nearest each of its junctions, so that every edge runs from
   junction to junction.

Here this gives 93 junctions and 230 edges.

![step 7](splinefit/results/pipeline/step7_junctions.jpg)

*Edges, one colour each; junction markers by type (cyan 3-way, magenta crossing, yellow compound); white
discs are the junction regions.*

**What to look for:**
- Along the wide vessels, every small vessel that ends on them makes a junction. Neighbouring ones merge
  into compound regions: the chains of yellow squares.
- The type depends on how many traced arms leave the disc. A missing arm (a branch the tracing lost) turns
  a 3-way into nothing, or a crossing into a 3-way.

### Step 8. Render, compare, prune

1. **Profiles.** Every few px along each trace, take the median OD cross-section and fit a blurred box,
   `a box(u; r, s) + b`, with r <= 0.75 w + 2.
2. **Render.** Add up the profiled edges.
3. **Weights.** Each pixel's weight is `1 / sigma^2`, where sigma^2 is the local variance of `OD - G_8 * OD`
   over the background (sigma 32 px). The weight is zero in the junction discs (0.7 x the radius), where an
   additive render is known to be wrong.
4. **Greedy MDL.** Repeatedly remove the free-end edge that fails worst on one of two tests, until every
   such edge passes both:
   - gain >= 2 (1 + L / 20), where the gain is the weighted squared residual the edge removes (its contrast
     refitted against what the other edges leave) divided by 6, the texture's correlation area in px^2;
   - the CNR of its profile in its own band is at least 0.5.

Two rounds are run. Here the first round removes 19 of the 230 edges.

![step 8](splinefit/results/pipeline/step8_prune.jpg)

*The additive render, the residual (red: OD not explained; blue: over-predicted) and the first round's
prune (blue kept, red removed).*

**What to look for:**
- The dark ticks across the wide vessel are cut points. The two pieces of a vessel cut at a junction
  overlap there and add up. The prune gives them no weight, but this is why the final render must treat
  nodes differently (step 10).
- The red residual along the wide vessels' edges is OD the boxes do not explain.

## Steps 9-12: the whole-image spline fit

### Step 9. The spline network

The proposal becomes a network of clamped cubic B-splines (vesselmap's `VesselNetwork`):

- **Crossings.** A junction typed crossing with four incident ends is not a node. Its ends pair into the two
  straightest lines, same trace first, and each line passes through as one continuous edge.
- **Joints.** A junction with two ends that turn by less than 60 deg is a pass-through.
- **Nodes.** Every other junction with at least three ends is a node.
  - Two ends of the same trace that turn by less than 40 deg become one edge through the node: the parent
    vessel continues, and the branch ends on it.
  - The other ends end at the node.
- **Free ends** become degree-1 nodes.
- **Edges.** Each chain of pieces linked through crossings, joints and through-nodes is one edge. Its
  control-point spacing adapts to calibre and curvature.
- **Profiles.** Each edge has r, s and a profile splines. The box widths are converted to a cylinder chord:
  half width x 1.2, and contrast divided by the cylinder's blurred peak.

![step 9](splinefit/results/pipeline/step9_network.jpg)

*The initial network (left) and the fitted, pruned network (right). Dots are control points (zoom only);
white rings are nodes of degree 3 or more.*

### Step 10. The render model

Each edge e contributes `a_e P(d; r_e, s_e) T_e` at a pixel at distance d from its centreline:
- P is the chord of a cylinder, `sqrt(1 - (d / r)^2)`, blurred by a Gaussian of std s in closed form;
- T_e fades the end caps with the same blur.

The contributions are summed, except in small windows around nodes and crossings. Those are corrected in a
sharp domain (blur 0.5 px), then blurred to the local blur:

- **node (fork):** `D = max_m cap_m - sum_m butt_m`. The lumens meeting there are one blood volume, so the
  render takes their union with round caps, not their sum.
- **crossing:** `D = -(1 - kappa)(sum_m c_m - max_m c_m)`. The two vessels lie at different depths and
  their ODs nearly add. kappa is fitted per image, starting from 0.87. A crossing is detected geometrically:
  two centrelines within 1 px of each other, at least 15 deg apart.
- **halo:** the whole render passes `(1 - h) X + h G(s_h) * X`, with h and s_h fitted (starting at 0.25 and
  6 px).

![step 10](splinefit/results/pipeline/step10_render_model.jpg)

*One cross-section: one vessel at three blurs; a fork, where the union is lower than the sum; a crossing,
where the OD is nearly additive; the halo.*

### Step 11. The loss

```
loss = sum over pixels of w (OD - R(theta))^2
       + 600 x bending energy of the centrelines   (sum |second difference of control points|^2 / h^3)
       + 20  x smoothness of log r, log s, log a along each edge
       + 300 x calibre prior                        (log r may not stray beyond x1.6 of its edge mean)
       + cusp, even-spacing and attachment priors  (a node an edge passes through stays on its centreline)
       + anchor (std 4 px) to the start positions  (joint stage only)
```

The weight w is step 8's `1 / sigma^2`, and zero off valid data. sigma includes the background texture,
which on an averaged still is much larger than the pixel noise. The optimiser is Adam with a cosine-decayed learning
rate.

### Step 12. The fit schedule

1. **Profiles.** 30 iterations of the profiles and optics (r, s, a, halo, kappa), geometry frozen.
2. **Joint fit.** 100 iterations of everything, positions included, anchored to where the stage started.
3. **MDL prune.**
   - An edge must explain more weighted error than its description costs:
     `n_params x log(pixels) x 0.25 / 2`, with the gain divided by the 6 px^2 correlation area.
   - A faint wide edge must also be seen against background on both flanks; otherwise it is illumination
     roll-off.
   - Here the prune removes 7 edges and leaves 85.
4. **Retarget.** Refit the background outside a better mask: stage 1's mask OR the fitted render's support
   (R > 0.5 x the local noise, dilated 3 px). Then `OD = B - L` again.
5. **Final fit.** 60 iterations of the profiles and optics on the new target, geometry frozen.

![step 12](splinefit/results/pipeline/step12_fit.jpg)

*Top: the stage-1 target, the precision weight (log scale) and the fitted render. Bottom: the retargeted
target, what the retarget changed, and the final residual on the same scale as step 8.*

**What to look for:**
- The retarget adds OD (red) around junctions and along the wide vessels: OD that stage 1's background had
  absorbed.
- The residual is far smaller than step 8's. Within the vessel band, the render holds 98.8 % of the
  image's OD under the true background (the scorer's `explained`: 1 - sum (OD_obs - R)^2 / sum OD_obs^2).

### Step 13. Export and the result

Each fitted edge is cut at every node it passes through, so each polyline runs junction to junction. Nodes
of degree 3 or more keep the proposal's junction type when they had one. Crossings are found geometrically
in the fitted render. The output also carries the render and the target.

![step 13](splinefit/results/pipeline/step13_output.jpg)

*The truth (left) and the output (right). Rings on the right are the truth junctions; filled markers are the
output's.*

On this development image:

| metric | value |
|---|---|
| line F1 | 0.85 |
| strict junction F1 | 0.72 |
| balanced junction-type accuracy | 0.48 |
| edge cover | 0.73 |
| composite | 0.781 |

On the 6 held-out averages the same pipeline scores line F1 0.840, junction F1 0.672 and type accuracy
0.564 (REPORT.md, section 4).

## Where it falls short, and what to change

The 95 % bar is per junction, not per image. Each truth junction asks three things:
- was a junction found within the strict radius;
- did it get the right coarse type;
- does the render hold 95 % of vesselscene's clean OD in its disc?

Over all 500 observable junctions of the 6 held-out averages (`junction_gallery.py`):

| junction type | n | found | right type | render >= 0.95 | all three |
|---|---|---|---|---|---|
| 3-way | 176 | 38 % | 24 % | 52 % | 11 % |
| crossing | 135 | 73 % | 36 % | 53 % | 23 % |
| compound | 189 | 74 % | 42 % | 54 % | 30 % |
| **all** | 500 | **61 %** | **34 %** | **53 %** | **21 %** |

For comparison, all three pass on 9 % of the junctions for the proposals and on 10 % for vesselmap.

![3-way](splinefit/results/heldout/junctions/junctions_3-way.jpg)

*Eight 3-way junctions drawn at random from the held-out averages. Each row shows the image, the truth
graph, the fitted graph, vesselscene's render, the fitted render and their difference, 64 px around the
junction.*

![crossings](splinefit/results/heldout/junctions/junctions_crossing.jpg)

*The same for eight crossings.*

![compounds](splinefit/results/heldout/junctions/junctions_compound.jpg)

*The same for eight compound junctions.*

### The changes, in order

1. **Fit the background outside a mask made from the OD, not the vesselness mask.** On the pathologic
   development scene, step 1's mask covers 46 % of the true lumen and 37 % of the wide vessels' lumen.
   Everything it misses is absorbed by the background and lost from the target. Hysteresis on the OD itself
   (seeds above 3 x the noise, grown above 1.5 x) covers 70 % and 66 %. On the OD smoothed by 3 px it covers
   75 % and 71 %. The mask stays a background tool only: as a fitting term (E23) it lowered positions.

   ![mask](splinefit/results/pipeline/improve_mask.jpg)

   *Red: true lumen the mask misses, which the background absorbs. Green: covered. Grey: mask on the blur
   flanks.*

2. **Search for side branches along every traced vessel.** Only 38 % of 3-way junctions are found, against
   73-74 % for crossings and compounds. Step 7 forms a 3-way junction only when a trace ends on its parent
   within reach, so the junction is lost whenever the branch's trace stops short, never reaches the parent,
   or is pruned. Walk along each traced vessel instead and test, on each side, for a line leaving it in the
   orientation score of step 5. Accept it against the residual of the fit, not against a fixed threshold.
3. **Type junctions by comparing local models.** Only 34 % get the right type. Today the type is counted from
   traced arms (step 7), and a lost arm changes it. Instead, render the competing hypotheses in the junction
   disc: fork, crossing, T, or two nearby forks. Fit each locally to the OD and keep the one with the
   smallest residual plus description cost. The step-10 render already distinguishes them: a fork is a
   union, and a crossing is nearly additive.
4. **Choose the continuing vessel by turn and width.** At a node, the parent vessel is the pair of arms with
   the smallest turn and the most similar widths. The daughter widths should satisfy Murray's law,
   `r_parent^3 = r_1^3 + r_2^3`. This replaces the current same-trace rule, which only links pieces of one
   trace.
5. **Keep false-alarm control in the proposer.** E22 linked pieces across gaps and traces. On faint texture,
   a link made two false pieces one edge that then survived the prune. Continuity has to come from the
   tracing (steps 5-6), with each piece judged on its own evidence.
6. **Measure per junction.** The image-level scores average over hundreds of junctions and hide the 21 %.
   Track the table above for every change.

## Reproduce

```
export OMP_NUM_THREADS=2
python -m experiments.splinefit.pipeline_figures --out experiments/splinefit/results/pipeline \
    --scene healthy_s007_480x768 --mask-scene pathologic_s000_480x768      # dev scenes ($SPLINEFIT_DEV)
SPLINEFIT_ALLOW_HELDOUT=1 python -m experiments.splinefit.junction_gallery <6 held-out scene dirs> \
    --out experiments/splinefit/results/heldout/junctions                   # the scorecard and galleries
python -m experiments.splinefit.report_html experiments/PIPELINE.md pipeline.html
```

The figure run takes about 3 minutes on 2 threads and is deterministic.
