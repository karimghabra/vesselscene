# A neuromimetic, deterministic vessel annotator: design

The goal is an annotation of a vesselscene still that matches its **observable image graph**: the vessel lines,
the junctions where they meet, and the type of each junction (a fork or a crossing look alike in 2-D). The
annotator reads only the image. It is scored against `truth.observable_<kind>.json` with `harness.py`.

## Why not curvature alone

The current approach (vesselmap in LIMBUS) proposes vessels from the eigenvalues of the Hessian (local
curvature of the log image), then fits splines with a differentiable renderer. The Hessian collapses each
pixel to **one** orientation. At a crossing or a fork both eigenvalues are large, and the anisotropy term
`-l1 - |l2|` that rejects blobs cancels the response exactly there. Lines therefore break at junctions, and
junction structure has to be rebuilt afterwards from the broken pieces. The same single-orientation readout
breaks dotted capillaries at red-cell gaps and makes texture lumps look like short vessels.

## The stages and their biological counterparts

| stage | biology | algorithm |
|---|---|---|
| 1. photoreceptors and horizontal cells | log response (Weber), adaptation to the local mean, the surround subtracted | `L = log I`; a background `B` fitted **only on the pixels outside the vesselness mask** (the mask's negative), by masked normalised convolution; the target `OD = B - L` (nepers, vessels positive) |
| 2. OFF-centre ganglion cells, contrast gain control | centre-surround, divided by local contrast | multi-scale centre-surround responses of OD divided by the local RMS of the same band over background pixels: a band-matched CNR, the same quantity the truth uses to decide visibility |
| 3. V1 simple cells | oriented, elongated receptive fields | even-symmetric oriented filters (second derivative across, Gaussian along, 3x elongation) at 16 orientations and several scales. The full stack `U(x, y, theta)` is kept: an **orientation score**. At a crossing the two vessels live in different orientation layers and do not collide |
| 4. non-classical surround | iso-orientation surround suppression, cross-orientation normalisation | flanking inhibition by orientation energy in an annulus (texture is suppressed, isolated contours are not) and divisive normalisation across orientations that keeps several peaks per pixel |
| 5. horizontal connections, association field | collinear facilitation (Field, Hayes and Hess 1993), bipole cells | a deterministic diffusion-convection in `(x, y, theta)`: Williams and Jacobs' stochastic completion field, computed as a fixed number of PDE-like steps. Bipole gating (both lobes need support) bridges gaps without extrapolating past line ends |
| 6. readout | | non-maximum suppression across space and across orientation (several peaks allowed), hysteresis in SE(2) connectivity, tracing along the tangent, so a trace passes straight through a crossing in its own orientation layer |
| 7. end-stopped cells | line terminations | a curve ending on another is a 3-way junction (T or Y); two curves passing through each other at different orientations make a crossing; a cluster with more arms makes a compound |
| 8. iterative refinement | (the lesson from diffusion models: refine step by step towards a prior, deterministically, as DDIM does) | render the candidate graph in OD (densities add where vessels overlap), take the residual against the **cleaned OD target**, prune edges that do not pay for themselves (MDL), propose what the residual still shows, and repeat a fixed number of times |

## The residual rule

When residuals are computed from a render of the candidate graph, the target is the image's own optical
density, never a vesselness map. The vesselness mask is used only through its negative: the pixels outside
it estimate the background, and that background cleans the original image into `OD = B - L`. A vesselness
response is distorted at bifurcations and crossings: the Hessian or orientation response drops or spreads
there. Fitting a render to it gives wrong optical densities at exactly the places where the junction type is
decided. In OD, two vessels crossing add their densities (Beer-Lambert), so a crossing shows a darker knot
than a fork, and that difference is a cue for typing.

## Determinism

Fixed filter banks, fixed iteration counts, no random numbers, stable sorts with coordinate tie-breaks,
and per-pixel operations whose results do not depend on thread scheduling. The harness runs every image
twice and compares a digest of the output.

## How it is measured

`harness.py` scores each annotator on held-out scenes (seeds not used while developing), average and
frame:
- centreline recall and precision (3 px);
- junction recall and precision;
- junction type accuracy, exact and coarse (3-way / crossing / compound);
- crossing recall and precision;
- edge cover (how much of each true image-graph edge one traced edge covers);
- runtime and determinism.

The comparisons are a plain Hessian ridge and skeleton annotator (curvature analysis) and, where time
allows, vesselmap itself.
