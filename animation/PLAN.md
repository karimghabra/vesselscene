# Animation plan: how the recurrent mid-level vision spec annotates a vessel video

A 3Blue1Brown-style explainer of *Tech spec: recurrent mid-level vision for the
vessel annotator* (revision 2, 11 October 2026). It covers the math, the physiology
of the image, and the neuroscience behind each layer. The video follows the spec
from a burst of conjunctival video to an annotated vessel graph with typed
junctions.

## Principles

1. **Computed, not drawn.** Every image, tuning curve, map and plot on screen is
   computed from the spec's own equations by `mica_anim/neuro.py`, on toy stimuli
   (chord-law vessels, white noise in OD, 16 orientations). When the probe is
   dragged, the tuning curve is recomputed live. This is what makes the 3b1b style
   honest: the picture *is* the math.
2. **Toy numbers and spec numbers are kept apart.** Values computed on screen are
   labelled as the toy model's. Values measured in MICA or on vesselscene are
   quoted from the spec and marked "measured (spec 4.x)". Where the toy disagrees
   with the spec, the video says so instead of hiding it (the proof of concept
   does this for the crossing fix).
3. **One question per chapter.** Each chapter opens with a concrete puzzle, such
   as "a fork and a crossing look the same in 2-D, so how do you tell them
   apart?". It builds the intuition before the equation. The equation then
   appears colour-coded, with the same colours as the picture.
4. **Fixed colour meanings** (`mica_anim/style.py`): orientation = hue,
   excitation = blue, inhibition = red, feedback = gold, line ends = orange,
   junctions = pink, continuation = green, attention label = yellow.
5. **A recurring map.** The architecture diagram of spec 4.1 is built in
   chapter 2. Each later chapter opens by lighting its own box on that map, so
   the viewer always knows where they are in the pipeline.
6. **Real imagery from vesselscene.** The cold open, the grouping chapter and
   the finale use scenes rendered by this repository's generator. A burst's
   frames carry their red-cell flicker, and the ground truth provides the
   reference annotation.

## Tools

| tool | role |
|---|---|
| Manim Community 0.22 (the maintained fork of 3b1b's own engine) | all animation, Cairo renderer, headless |
| LaTeX (texlive) | every equation, in Computer Modern as in 3b1b |
| numpy / scipy | the spec's computations, live under the animation |
| vesselscene (this repo) | physiological stills and frames with exact truth (CPU: ~4 min per 640x360 scene) |
| ffmpeg | chapter assembly, previews |

ManimGL, 3b1b's personal version, was considered and not used. It needs an OpenGL
context, and in a headless cloud container that makes it fragile. Manim CE gives
the same look and renders reliably here.

## Storyboard

The approximate durations below assume captions. Chapters marked **core** carry
the junction story, which is the spec's main goal, and form the short cut.

| # | chapter | spec | the visual idea | core | ~min |
|---|---|---|---|---|---|
| 0 | **Cold open: from flicker to graph** | 1, 2.1 | vesselscene frames flicker with red cells, then average into the still; the annotation draws itself (vessels in identity colours, junction rings by type). Then the problem: 61 % of junctions found, 34 % typed, 38 % of 3-ways found. "Junctions are where it fails." | ● | 1.5 |
| 1 | **The physics of the picture** | 4.5, 4.12, 6.4 | 3-D tube, light through it: Beer-Lambert gives OD = −ln(I/B). Slicing the cylinder gives the chord law, 2√(r²−x²), which blur turns into the measured profile. Two tubes at different depths: ODs add (κ). A fork: lumens merge, a union and not a sum. Murray's law r³ = r₁³ + r₂³. | ● | 2.5 |
| 2 | **Two modes of vision** | 1, 4.1, App. A | Timeline 0–200 ms: a feedforward wave climbs retina → V1 → V2, then recurrent and feedback arrows light up (Lamme & Roelfsema). Today's pipeline is only the first wave plus symbolic rules. The spec's architecture diagram assembles itself; it is our map. | ● | 1.5 |
| 3 | Retina and ganglion cells; noise units | 4.0 | Centre-surround (difference of Gaussians) as a 3-D Mexican hat. Dividing by the background's spread makes "8" mean the same thing in a dim and a bright frame. Drive and rate: max(0, drive). | | 1.0 |
| 4 | **One point, sixteen orientations** *(proof of concept, built)* | 4.0, 2.4, 4.6 | The probe's simple cell rotates through 16 orientations and the polar tuning curve grows spoke by spoke. Dragging the probe updates the curve live: a vessel gives two lobes, a crossing four, a blob a weak ring. Next comes the surround floor, the crossing's lost half shown with sorted bars, and the lowest-quarter fix swept over crossing angle against its gate. Variant: divisive normalisation. | ● | 3.0 |
| 5 | Size channels and scale selection | 4.6b | A coarse cell 100 px long overhangs a thin line's end and still fires 40 px past it. Then end-zone inhibition with a centre pool: on a continuous vessel the two pools cancel, and past an end the coarse cell goes quiet. Spec numbers: 15 → 2 noise units. | | 1.5 |
| 6 | Bends: arc cells | 4.8 | Rings of radius 60/30/15 px: long straight cells keep 75/56/44 %. The arc formula p + (sin κs/κ) t + ((1−cos κs)/κ) n is driven by a κ slider, so the pooling path bends to fit. Curvature competes like width, and the winner is a population code for 1/r. | | 2.0 |
| 7 | **Horizontal connections** | 4.7 | Field-Hayes-Hess Gabor field: the viewer finds the hidden path. The bipole is √(A⁺·A⁻). The recurrent equation's terms appear one by one (excite, parallel, rival), and 8 iterations run live: the texture "web" fades and the vessel stays. Zoomed pixel grid: a 1-px lobe slides off the crest of a vessel at 5.6°. | ● | 2.5 |
| 8 | **Where lines end: end-stopped cells** | 4.2 | Lobes behind and ahead ride along a line. At the end A⁺ collapses and E = max(0, min(C, A⁻) − βA⁺) fires, giving 32 direction channels drawn as arrows. Real vessels dip, giving spurious ends. The divisive form [A⁻ − βA⁺]₊/(A⁻ + A⁺ + c) is plotted against dip depth: 1 at a real end, about 0.1 at a 20 % dip. | ● | 2.0 |
| 9 | **Junction cells** | 4.3, 2.2 | Old step 7: a branch's trace stops short, so the junction is lost. New: a disc ρ = r₀ + w/2 gathers rays from end cells and from passing lines, the arms are sorted, and J₃ = the 3rd-largest. This is compared side by side with the Harris corner's second eigenvalue. Non-maximum suppression, then the typing table animates T, Y, X and compound. | ● | 2.5 |
| 10 | **Good continuation is co-circularity** | 4.4, 4.10b | Two curved vessel ends 23° apart across a 12 px gap are rejected by the straight-turn rule. Then the geometry: chord c, headings, φₐ and φ_b. Drag the ends and a circle passes through both exactly when φₐ + φ_b = 0, with r = d / (2 sin(turn/2)). An S-bend and a kink fail. Then pairing arms at a junction: crossing against fork. | ● | 3.0 |
| 11 | **Fork or crossing? Transparency** | 4.5, 4.5b | Two absorbers in series compared with one merged lumen. A plot of ρ = a_c/(a₁+a₂) against the contrast ratio shows two prediction lines that converge as one vessel fades. A noise band gives the likelihood ratio, and the ambiguous band (1/3–3) goes to model comparison: each hypothesis is rendered and the lowest residual plus description cost wins (predictive coding at one junction). | ● | 2.5 |
| 12 | Border ownership, grouping cells, evidence fusion | 4.9, 4.9b | A dark strip owns both borders: √(left·right) peaks on the medial axis at h = half-width, while a lone glare edge gives 0. Fusion: a 2-D cue plane where a union of thresholds makes an L-shaped acceptance region and a sum makes a diagonal, so two weak cues can agree. | | 2.0 |
| 13 | Sheets: readout per orientation | 4.0, 4.9c | 16 sheets stacked in 3-D. Collapsing them to the best orientation erases the weaker vessel's crest at a crossing (41 px lost at 30°, unequal). Tracing within a sheet keeps it. | | 1.5 |
| 14 | **Attentive grouping** | 4.10, 4.10b | A yellow label spreads along one curve (Roelfsema's curve tracing). The gain lowers the threshold to t_low/(1+γ), so weak stretches pass. At a crossing the label goes straight through, and at a fork it follows the parent. Inhibition of return, then the next seed. On a vesselscene image the vessels light up one by one in identity colours, and facing ends are linked. | ● | 3.0 |
| 15 | Top-down feedback | 4.11 | The graph is rendered and subtracted from the OD: err = max(OD − R, 0)/noise. A gain applies only near junctions and free ends, and a missing arm reappears. The earlier whole-image re-proposal is contrasted with it: precision fell from 0.82. Two loops. | | 2.0 |
| 16 | **Population codes** | 4.12 | Each orientation cell is a vector at doubled angle in the complex plane. The vectors add head to tail, and halving the angle gives θ = ½ arg Σ C_k e^{2iθ_k}, sub-degree where the best of 16 gives 11.25° steps. Sub-pixel position comes from a parabola through 3 samples, and width from the weighted mean of log scale. These start values seed the spline fit. | ● | 2.5 |
| 17 | Motion: flicker and flow | 4.14 | Red-cell columns and plasma gaps flow, and the flicker map uses the robust temporal spread minus camera noise. Along one vessel, the space-time image D(s, t) shows oriented streaks, and spatiotemporal energy filters (Adelson-Bergen) read them as direction and speed. At a 3-way junction the arrows separate a bifurcation from a confluence, which a still cannot do. | | 2.5 |
| 18 | **The whole loop, and how we'd know it works** | 4.1, 6.2, 6.4, 1.1 | The map runs end to end: feedforward, then the recurrent loops. The two gates follow, synthetic against truth and real without truth, along with the inverse crime (the same tube model renders and fits). The build status map shows each layer as built, off, designed or not started. Close on the annotated image. | ● | 2.0 |

## The four cuts

Each chapter is written once, with three depths (`mica_anim/chapter.py`):
level 1 (short) shows one idea, level 2 (medium) adds the equation and mechanism, and level 3 (detailed) adds
the spec's measurements, variants and caveats. `versions.py` assembles the cuts:

| cut | content | length |
|---|---|---|
| very short | `s99_trailer.py`: one beat per idea | ~1.5 min |
| short | chapters 0, 1, 2, 4, 8, 9, 10, 11, 14, 18 at level 1 | ~5 min |
| medium | the core chapters (●) at level 2 | ~15 min |
| detailed | all chapters at level 3, with the architecture map before each and the references at the end | ~40 min |

## Proof of concept (built)

`scenes/s04_hypercolumn.py` (about 90 s) is chapter 4. It exercises every
technique the full video needs: images computed from the spec's equations, a
live-updating plot driven by a dragged probe, colour-coded LaTeX, transforms
between representations (polar curve → sorted bars), and a parameter sweep that
traces a curve. It reproduces the spec's measured step-4 effect on its toy
hypercolumn:

| pixel | toy (this animation) | spec, measured |
|---|---|---|
| vessel | −5 % | −4 % (13.5 → 13.0) |
| crossing (90°) | −47 % | −44 % (14.7 → 8.3) |
| blob | −36 % | −48 % (2.7 → 1.4) |

## Build order

1. Foundation: `mica_anim/` style, widgets, the computation core, chapter levels,
   the architecture map, vesselscene data (done).
2. All chapters at three levels, reviewed at 480p.
3. The trailer and end cards.
4. Final renders at 1080p30, then each cut joined with ffmpeg (`python versions.py render all`).

## Layout

```
animation/
  PLAN.md            this plan
  README.md          how to render
  requirements.txt   manim, numpy, scipy
  mica_anim/         shared code: style.py (palette, colormaps), widgets.py
                     (polar plots, receptive fields, bars), neuro.py (the spec's
                     computations on toy stimuli), later scenes_data.py (vesselscene renders)
  scenes/            one file per chapter: s00_cold_open.py ... s18_whole_loop.py
  render.sh          render all chapters and concatenate
```

Rendered video is not committed (it goes to `media/`, which git ignores). It is
delivered as files.
