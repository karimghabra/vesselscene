# MICA: retiring the old code and building the new layers one at a time

This is the working plan for the MICA code (`C:\Users\ihave\OneDrive\Documents\MICA`). It covers three things:
- which parts of the current code stay, which get fixed, and which are retired;
- the order in which the new neuronal layers of `SPEC_V2.md` are built;
- how each layer is developed on its own, with a figure that can be opened and compared after every change.

Companion documents: `PIPELINE.md` explains the current pipeline step by step; `SPEC_V2.md` gives the design of each
new layer (the biology, the computation, the gates). "SPEC 4.3" means section 4.3 of `SPEC_V2.md`.

## Words used here

| term | meaning |
|---|---|
| OD (optical density) | ln(background / image): how much light a pixel absorbs compared with the tissue behind it. 0 on clean background, positive on vessels. `fit['OD']` |
| noise units | a value divided by the local noise level of the background, so 3 means "three times the noise". Every threshold in the new layers is in these units, so the same number works on a bright frame and a dim one |
| layer | one stage of the network: a function that takes maps and returns maps (for example the end-stopped cells) |
| stack | a layer's output kept per orientation (16 maps, one per 11.25 degrees), rather than collapsed to the best orientation at each pixel |
| channel | one of `neuro.CHANNELS`' three size ranges (fine, medium, coarse), each computed on its own grid |
| lobes | the two regions ahead of and behind a pixel along its orientation, which the association field reads (`A+`, `A-`) |
| probe site | a fixed pixel (a crossing, a fork, a texture spot...) shown in every layer's figure, so the same place can be followed through the network |
| zoom window | a fixed square crop of the image, shown in every layer's figure |
| split-half | run the chain on the average of the odd frames and on the average of the even frames, and compare the two outputs. A real vessel shows up in both; noise coincidences don't. Needs no truth |
| Dice | the overlap of two masks, 2 x shared area / (area A + area B); 1 = identical |
| precision, recall, F1 | precision: of the detections, the share that are real. Recall: of the real things, the share detected. F1 combines the two |
| gate | the measured condition a layer must pass before it is switched on by default and the code it replaces is retired |
| vesselscene | the synthetic scene generator; its scenes come with exact truth (centrelines, widths, junction types) |
| development scenes | the vesselscene scenes used for tuning; held-out scenes are scored only at the end |

## 1. In short

- The old code keeps running while each new layer is built next to it, behind a switch. Old code is retired only after
  its replacement passes a stated test, and it moves to `legacy/` rather than being deleted.
- Layers are built one at a time, in an order where each one is useful and measurable on its own.
- Each layer comes with four things:
  - its function, with no plotting inside;
  - its figure function;
  - one `dev.py` cell;
  - a row of numbers.
- Every figure uses the same probe sites, zoom windows and colour scales, and every version is kept, so this week's
  figure can sit next to last week's.
- Two pictures are rebuilt after every run, to open whenever you want to see where things stand:
  - a gallery page (`output/layers/index.html`);
  - a probe sheet: one image following every probe site through every layer.
- Every layer is tested twice:
  - on vesselscene, against the truth;
  - on the real burst, without truth: split-half agreement, the OD left on the background, and your hand-annotated
    junctions.

## 2. Where the code stands

| file and function | what happens to it | phase |
|---|---|---|
| `ingest_stabilize.py` | kept; delete the stray `from django.core.files import images` | 0 |
| `view_burst.py` | kept | – |
| `background.py` | kept; `fit_background` gains `extra=` vessel evidence for its mask (layer 19) | 7 |
| `neuro.receptive_field`, `oriented_scores`, `surround`, `complete` | kept as layers 02-04. Fixes: FFT padding, `scale`. Changes: return per-channel stacks and step 5's lobes; `oriented_scores` also returns the odd (edge) responses; `surround` gets the crossing fix | 0-1 |
| `neuro.neuro_fields` | split into per-channel stacks; its collapse to one orientation per pixel survives for display only | 1 |
| `neuro.centrelines` | to `legacy/` (or made a call to `ridge_tops`) | 0 |
| `neuro.track` | rewritten to read the stacks (layer 06); then replaced by attentive grouping (layer 14) | 1, 5 |
| `neuro.width_mask`, `neuro.neuro_mask` | retired: widths come from the fit | 6 |
| `vessels._bridge_ends`, `_end_points`, `_walk_back` | retired: end-stopped and junction cells (layers 07, 08) | 2 |
| `vessels._dense_additions`, then `vessel_seeds` | retired: grouping cells (layer 11) | 3 |
| `temporal.flicker_maps`, `_in_background_spreads`, `flicker_artifacts` | kept: layer 12 puts the coherence in noise units with `_in_background_spreads` (it already divides by the spread on the background) and leaves out the reflection and dust spots `flicker_artifacts` finds | 4 |
| `temporal.flicker_mask`, `combine_masks` | retired | 4 |
| `manipulation.py` | to `legacy/` except `hessian_eigenvalues` and `vesselness_at_scale`, which `vessels.py` and `flicker_mask` still call; those go with their last caller | 0, 4 |
| `figures.py`, `neuro_figures.py` | kept; `workflow_figure` to `legacy/`; the neuro figures are redrawn on the figure template of section 3 | 0 |
| `dev.py` | the single-frame cell to `legacy/`; one new cell per layer; the union cells (`merged`, `with_flicker`) retired | 0, 5 |

## 3. How a layer is developed

### 3.1 The loop, for every layer

1. **Write the prediction first.** Two or three sentences in `LAYERS_LOG.md`:
   - what the figure should show (for example "at the fork site: one line end and one passing line, three arms");
   - which numbers should move, and by how much.

   Writing it before the run keeps the look at the figure honest.

2. **Write the layer** as a plain function in its module: arrays in, arrays or a dict out, every parameter an argument
   with a default, no plotting, no global state. It sits behind a switch in `dev.py`; off means the old path.

3. **Write its figure function** in `layer_figures.py`, on the template of 3.3.
4. **Add one `dev.py` cell** (template in 3.6): run the layer, print its numbers, save its figure, show it.
5. **Run it fast first** (crop mode, 3.5) until the figure looks right, then on the full image.
6. **Look.** Open the gallery; compare with the previous version and with the layer upstream.
7. **Measure.** `checks.py` on the real burst and the vesselscene adapter on the development scenes; both append to
   `output/metrics.tsv`.

8. **Decide** at the layer's gate:
   - switch it on;
   - keep tuning;
   - or record the failure and leave it off.

   Write the outcome under the prediction in `LAYERS_LOG.md`.

9. **Commit** the code, `LAYERS_LOG.md` and `metrics.tsv`. Figures stay out of git, since any commit can regenerate
   them, but they are kept on disk, stamped with their commit.

### 3.2 Probe sites and zoom windows

The figures can be compared across layers and versions only if they look at the same places.

- **Real burst:** pick the sites once on the registered average with `sites.pick(average)`. It shows the image, asks
  for a click on each named site in turn (`plt.ginput`), and writes `output/sites.json`.
  - Coordinates are in the average's frame. They stay valid as long as the registration reference stays the same;
    re-pick if it changes.
- **vesselscene:** `sites_from_truth(scene)` picks the same set automatically from the truth graph.

| site | what it tests |
|---|---|
| `thin` | a thin vessel, mid-segment |
| `wide` | the widest vessel, mid-segment: flat floor, weak small-scale response |
| `faint` | a faint vessel that today's mask misses or breaks |
| `curved` | a tightly curved stretch |
| `crossing` | a crossing of two vessels at different depths |
| `fork` | a Y-shaped fork |
| `tee` | a T-shaped branch (a thin branch off a wide parent) |
| `end` | a vessel end (a real one, or where the vessel leaves the field) |
| `texture` | a dark texture spot away from vessels |
| `edge` | a one-sided edge: the aperture border, or a glare edge |
| `empty` | clean background |

Each site has a 96 x 96 px crop. There are also three zoom windows of 256 x 256 px:
- a dense region;
- a region with several junctions;
- the widest vessel next to an edge.

The truth is drawn on every vesselscene figure: centrelines in green, junctions as green rings labelled with their
type. On real frames, your hand-annotated junctions are drawn the same way.

### 3.3 The figure template

Every layer's figure has the same four parts, in the same places:

| part | where | content |
|---|---|---|
| A. mechanism | top left | what the layer computes, on a clean example: its kernel or receptive field, its rule as a curve, or a synthetic line, end or crossing passed through it |
| B. probe sites | top right | one small panel per relevant site: the crop of the layer's input and output, plus a quantitative plot (a tuning curve over the 16 orientations, a profile across the vessel, arms drawn as arrows...) |
| C. whole image | bottom | the zoom windows: input and output side by side, at fixed colour scales |
| D. numbers | footer | the layer's numbers from this run, and their change since the previous version |

Fixed conventions:
- **Colour scales are fixed per quantity, never auto-scaled**, so a change in the figure is a change in the data:
  - OD 0-0.6;
  - line evidence 0-12 noise units;
  - junction evidence 0-8 noise units;
  - orientation as hue, cyclic over 180 degrees;
  - width as a sequential map in pixels.
- **Colours mean the same thing in every figure:**
  - truth: green;
  - detections: magenta;
  - junction types: crossings blue, 3-way orange, compound grey;
  - ends: red arrows pointing to the free side;
  - rejected candidates: dashed.
- **Title:** the layer, the date, the commit, and any switch or parameter that differs from its default.
- **One figure function for both datasets:** it takes the truth as an optional argument and draws it when given.

### 3.4 Output folder, versions and the gallery

```
output/
  sites.json            probe sites and zoom windows (real burst)
  junctions.json        your hand-annotated junctions (position, type)
  metrics.tsv           one row per layer per run: date, commit, layer, dataset, switches, numbers
  cache/                layer outputs, keyed by a hash of their inputs and parameters
  layers/
    index.html          the gallery
    probe_sheet.png     every probe site through every layer
    01_retina/
      01_retina_2026-10-14_a1b2c3d.png
      latest.png
    02_simple/
    ...
```

- **Saving.** `layer_figures.save(fig, '07_end_stopped', numbers)` does four things:
  - writes the figure under a name stamped with the date and commit;
  - updates `latest.png`;
  - appends the numbers to `metrics.tsv`;
  - rebuilds the gallery.

  Nothing is overwritten except `latest.png`.
- **The gallery** (`gallery.py`) is a static HTML page that opens in any browser:
  - one row per layer, in network order;
  - the latest figure with the previous version beside it, and the numbers with their change;
  - a menu on each row to pick any older version;
  - layers not yet built show their card from section 5.
- **The probe sheet** is one image: a column per probe site, a row per layer. Each cell is the site's crop of that
  layer's display map: the strongest response over orientations, or the layer's own natural map.
  - Reading down a column follows one crossing, or one texture spot, through the whole network.
  - This is the picture to check when something downstream changes unexpectedly.

Each layer therefore also provides `display(output) -> 2-D map` for the probe sheet.

### 3.5 Keeping it fast

`neuro_fields` takes about 2 minutes on the full image, and a dozen layers on top would make every look-and-tweak slow.

- **Cache:** each layer's output is saved in `output/cache/` under a hash of its inputs and parameters. Re-running a
  downstream layer reuses everything upstream.
- **Crop mode:** run the layers on the site crops and zoom windows only, each padded by the layers' reach (the
  longest filter plus the association lobes: about 3 x elongation x the channel's largest scale).
  - Seconds instead of minutes.
  - Use it while working on a figure, then run the full image for the numbers.
- **Memory:** a full stack is 16 maps per channel. Keep stacks in `float32`, keep only the ones the next layer
  reads, and drop the rest once the figure is drawn.

### 3.6 The `dev.py` cell, for every layer

```python
# %% layer 07, end-stopped cells (v2.end_stopped, SPEC 4.2): a line that ends
# inside the cell's field, with support behind it and none ahead. figure:
# output/layers/07_end_stopped. switch: USE_END_STOPPED
ends = {c: v2.end_stopped(s['C'], s['A_plus'], s['A_minus'], beta=1.0, t_low=1.5)
        for c, s in stacks.items()}
numbers = checks.end_stopped(ends, stacks, sites)
print(checks.format_numbers('07_end_stopped', numbers))
fig = layer_figures.end_stopped_figure(fit['OD'], stacks, ends, sites, numbers)
layer_figures.save(fig, '07_end_stopped', numbers)
plt.show()
```

### 3.7 What is measured

Each layer card (section 5) lists its own numbers. They come from two places.

**Real burst, without truth** (`checks.py`):
- **split-half agreement** of the layer's output, odd-frame average against even-frame average:
  - Dice, for masks;
  - the share of detections within 3 px of one in the other half, for points (ends, junctions);
  - the correlation, for maps;
- **the OD left on the background**: the median OD on pixels the output calls background. It should stay at 0;
- **your hand-annotated junctions**: found within a few pixels, and typed right;
- **size of the output**: mask area, pieces, centreline length, junction counts;
- **run time**.

**vesselscene, against the truth.** An adapter in the vesselscene repository (`experiments/mica/adapter.py`) runs the
MICA functions on a scene's average. It converts their outputs into what the existing scorers take: centreline
raster, junction list with types, traces. It reports:
- centreline precision, recall and F1;
- the per-junction scorecard (found, typed, by type), as `junction_gallery.judge`;
- the probe battery's junction sweeps (`fork`, `t`, `crossing_angle`, `crossing_depth`, `crossing_width`, `end`,
  `empty`);
- the layer-specific numbers listed on each card.

The adapter imports MICA from the path in the `MICA_PATH` environment variable. So there is one implementation,
yours, and vesselscene only scores it.

## 4. Phase 0: housekeeping, baseline and figure tools (no change in behaviour)

1. **Put MICA under git** (a private GitHub repository). Each layer becomes one or a few commits, and its figures and
   numbers are stamped with the commit.

2. **Move dead code to `legacy/`** (section 2).
3. **Fix:**
   - delete the Django import in `ingest_stabilize.py`;
   - in `oriented_scores`, pad the FFT by the filter's reach, `pad = ceil(3 * elongation * max(scales) * scale /
     factor)`, instead of 32. Today a coarse filter wraps around the image border;
   - decide what `scale` means (the pixel size relative to the report's pixels), pass it through `neuro_fields` to
     every layer, and make the header comment match;
   - put `density_gate` (in `vessel_seeds`) and `density_floor` (in `flicker_mask`) in noise units rather than
     absolute OD, so they don't depend on the frame's brightness. Both are retired later, but they set the baseline,
     and the baseline should be fair.

4. **Build the tools of section 3:**
   - `sites.py` (pick and load), and `sites_from_truth` on the vesselscene side;
   - `layer_figures.save` and `gallery.py`;
   - `checks.py`;
   - the cache and crop mode.

5. **Baseline figures of the existing layers.** Redraw layers 01-04 and today's tracker on the template, from the
   current figure functions (`profiles_figure`, `receptive_field_figure`, `vessel_profile_figure`, `tuning_figure`,
   `stages_figure`). Record their numbers on the real burst and on the development scenes. Everything later is
   compared with these.

6. **Hand-annotate 50-100 junctions** on the real average, with position and type (crossing, 3-way, compound), using
   the same click tool. Save them to `output/junctions.json`. This is the only real-data truth the plan needs.

**Done when:**
- the gallery shows the baseline layers for both datasets;
- `metrics.tsv` holds the baseline;
- the outputs equal today's, apart from the bug fixes, whose effect is recorded on its own.

### Baseline cards

**01 retina: background, OD and noise.** `background.fit_background(image, valid, ...)`.
- **Figure:**
  - A: the fit's steps on one cross-section (upper envelope, first guess, refit), as `profiles_figure`;
  - B: at every site, the OD crop and the background mask;
  - C: the zooms of the average, the background and the OD; a histogram of the OD on background pixels.
- **Look for:**
  - the background mask covering every vessel's full width (generous on purpose);
  - the OD on background centred on 0;
  - no bright or dark halo along wide vessels.
- **Numbers:** median and 90th-percentile OD on background pixels; mask area; split-half correlation of the OD.

**02 simple cells.** `neuro.oriented_scores(...)`.
- **Figure:**
  - A: the receptive fields at 4 orientations in each channel, as `receptive_field_figure`;
  - B: tuning curves (line evidence over the 16 orientations) at every site; the profile across `thin` and `wide`, as
    `vessel_profile_figure`;
  - C: the strongest line evidence over orientations, in the zooms.
- **Look for:**
  - one clear peak at vessel sites;
  - two peaks at `crossing`;
  - a broad bump at `texture`;
  - a strong response along `edge`. Simple cells cannot tell an edge from a vessel's wall; that is layer 11's job.
- **Numbers:** line evidence at each site; its 99th percentile on background.

**03 surround and 04 association field.** As `tuning_figure` and `stages_figure`, redrawn on the template: tuning
curves before and after, at every site; the response along `thin` and `curved` (dips); the zooms. Their full cards
are in phases 1 and 5, where they change.

## 5. The layers

### 5.1 Build order and network position

Folders are numbered by position in the network, so the gallery reads from the retina upward. They are built in the
order of the phases.

| folder | layer | SPEC | phase | replaces |
|---|---|---|---|---|
| 01_retina | background fit, OD, noise | PIPELINE step 1 | exists | – |
| 02_simple | simple cells (line and edge) | step 3 | exists | – |
| 03_surround | surround suppression, with the crossing fix | step 4, 4.6 | exists; fixed in 1 | – |
| 04_association | association field, then the recurrent field | step 5, 4.7 | exists; replaced in 5 | `complete`'s three excitatory iterations |
| 05_curvature | curvature cells | 4.8 | 5 | – |
| 06_readout | ridge tops per orientation, and a tracer that stays in its layer | step 6 | 1 | the collapse in `neuro_fields` before `track` |
| 07_end_stopped | end-stopped cells | 4.2 | 2 | `_end_points`, `_walk_back` |
| 08_junctions | junction cells | 4.3 | 2 | `_bridge_ends`; junctions from skeleton ends |
| 09_continuation | good continuation at junctions | 4.4 | 2 | – |
| 10_transparency | fork or crossing, from the OD | 4.5 | 2 | – |
| 11_grouping | border ownership and grouping cells | 4.9 | 3 | `_dense_additions`, then `vessel_seeds` |
| 12_flicker | flicker as line evidence | 4.14 | 4 | `flicker_mask`, `combine_masks` |
| 13_fusion | evidence summed in noise units, one threshold | – | 5 | the unions `merged` (`seeds` or `neuro_vessels`) and `with_flicker` |
| 14_attention | attentive grouping, one vessel at a time | 4.10 | 5 | the layer-06 tracer |
| 15_graph_fit | graph and spline fit | steps 9-12 | 6 | `width_mask`, `neuro_mask` |
| 16_model_compare | local model comparison at ambiguous junctions | 4.5b | 6 | – |
| 17_feedback | top-down gain near ends and junctions | 4.11 | 7 | – |
| 18_population | population codes as start values for the fit | 4.12 | 7 | – |
| 19_surface | background completed only behind vessel evidence | 4.13 | 7 | the OD threshold in the background mask |
| 20_direction | flow direction along each vessel | 4.14 | 7 | – |

Two changes from the first version of this plan:
- **The surround's crossing fix (SPEC 4.6) moves into phase 1.** Like the readout, it is about keeping crossings, and
  the readout cannot keep a crossing the surround has already removed.
- **Local model comparison (SPEC 4.5b) moves from phase 2 to phase 6.** It renders hypotheses, and MICA has no
  renderer until the fit arrives. Until then, ambiguous junctions are flagged and left untyped.

Each card below gives:
- what the layer computes;
- its code;
- its figure (parts A-C of the template; part D is always its numbers);
- what to look for in the figure;
- its numbers;
- its gate.

### 5.2 Phase 1: keep crossings (layers 03, 06)

**Why first:** it is cheap, and every later layer needs crossings to survive. Today the surround removes 44 % of a
crossing's response, and `neuro_fields` keeps one orientation per pixel. So at a crossing one of the two vessels is
lost before tracking starts.

**Groundwork.**
- `neuro_stacks(od, background, scale, channels)` returns, per channel, `U`, `S`, `U2`, `C`, `A_plus`, `A_minus`,
  `even`, `odd` and `factor`, plus `full_size(stack)` for display.
- `complete(..., return_lobes=True)` and `oriented_scores(..., return_parts=True)` provide the extra outputs.
- `neuro_fields` becomes the collapse of `neuro_stacks`. Check that its output is identical, array for array.

#### 03 surround: the crossing fix (SPEC 4.6)

- **Computes:** step 4's orientation-blind suppression subtracts the mean of the weakest half of the 16 orientations.
  At a crossing, the second vessel's lobe falls into that half. Variants:
  - the mean of the weakest quarter;
  - the mean of the weakest half after removing the layers next to the two largest peaks;
  - divisive normalisation, `P / (1 + a * iso + b * flank)`, which scales a response down instead of removing a
    fixed amount.
- **Code:** `neuro.surround(U, S, factor, iso_weight, flank_weight, iso_mode='half', divisive=False, a=1, b=0.3)`;
  `iso_mode` is 'half' (today), 'quarter' or 'peaks'.
- **Figure:**
  - A: the tuning curve at `crossing`, with the orientations that enter the suppression shaded; one panel per
    variant;
  - B: bars of the share of response lost at `thin`, `crossing`, `texture` and `empty`, per variant;
  - C: the zooms, today's output against the chosen variant's.
- **Look for:**
  - both lobes at `crossing` surviving;
  - `texture` losing as much as today;
  - no new response along `edge`.
- **Numbers:** the loss at crossings and at blobs: on vesselscene over all truth crossings and background pixels; on
  the real burst at the sites and at your hand-annotated crossings.
- **Gate (SPEC 4.6):** crossings lose at most 15 % (today 44 %), and blobs still lose at least 40 % (today 48 %).

#### 06 readout: ridge tops per orientation, and a tracer that stays in its layer

- **Computes:**
  - A ridge top in layer k is a pixel whose `C[k]` is a maximum across the line, along the normal of orientation k.
    This is found in every layer separately, so a crossing pixel can be a ridge top in two layers.
  - The tracer keeps `track`'s rules: start at tops above `high`, continue to tops above `low` within `reach`, within
    `turn` and inside the `cone`. The difference: each step stays in its own layer, or one layer either side.
  - So two vessels through a crossing are traced separately, each in its own layer.
  - Traces from the three channels are merged as in step 6: duplicates within half a width are removed and the
    stronger is kept.
- **Code:** `readout.ridge_tops(C, floor) -> tops (16, h, w)`; `readout.trace(C, tops, S, high=8, low=2, reach=2,
  turn_steps=1, cone=45, min_length=15) -> traces` (points, with the layer and scale at each).
- **Figure:**
  - A: at `crossing`, the 16 layers as small crops with their ridge tops, next to the old collapsed readout (one
    layer per pixel);
  - B: at `crossing`, `fork` and `tee`, one colour per trace; the old `track` beside the new;
  - C: the zooms, traces over the OD, old and new.
- **Look for:**
  - two traces through every crossing, each continuing straight;
  - no doubled traces along one vessel. A vessel's response spreads over neighbouring layers, and one layer either
    side counts as the same vessel;
  - no more texture traced than before.
- **Numbers:**
  - centreline length and number of traces;
  - traces through each hand-annotated crossing (should be 2);
  - split-half agreement of the centrelines;
  - vesselscene: centreline precision, recall and F1, and crossing recall (both lines traced through, within the
    strict radius).
- **Gate:** centreline F1 and crossing recall on vesselscene not worse than `track`; split-half agreement not worse.
- **Retires:** the collapsed `track`. The tracer itself is replaced in phase 5.

### 5.3 Phase 2: junctions (layers 07-10)

**Why:** this is the biggest single failure. Over the six held-out vesselscene images, only 38 % of 3-way junctions
are found.
Junctions are found where traces happen to meet, so a branch whose trace stops short is lost. `vessels._bridge_ends`
is a hand-made attempt at the same problem.

#### 07 end-stopped cells (SPEC 4.2)

- **Computes:** a line that ends inside the cell's field, with support behind it and none ahead:
  `E_fwd[k] = max(0, min(C[k], A-[k]) - beta * A+[k])`, and `E_bwd` the mirror.
  - This gives 32 directions (16 orientations x 2 ends); each points out of the end, to its free side.
  - Variant: complex-cell energy `Q = sqrt(even^2 + odd^2)` as input, which marks an end whatever its phase (SPEC
    4.6). It is compared on the `end` probe.
- **Code:** `v2.end_stopped(C, A_plus, A_minus, beta=1.0, t_low=1.5) -> E (32, h, w)`.
- **Figure:**
  - A: a synthetic bar with an end: C, the lobes ahead and behind, and E along the bar (zero mid-bar, a peak at the
    end);
  - B: at `end`, `thin`, `fork` and `tee`, E over the 32 directions as a polar plot;
  - C: the zooms, the strongest E as a heat map over the OD, with red arrows at its maxima pointing to the free side.
- **Look for:**
  - an arrow at every vessel end, and at the end of a branch's trace near a fork;
  - none along the middle of vessels, at gaps the association field bridged, or at crossings.
- **Numbers:** ends per mm of centreline; split-half agreement of end positions; vesselscene: recall of truth ends,
  and false ends along vessels per mm.
- **Gate:** layer 08's. This layer feeds it and is judged with it.

#### 08 junction cells (SPEC 4.3)

- **Computes:** around each pixel p, the arms that meet there, within a disc of radius `r0 + w / 2`:
  - line ends pointing into p, from E;
  - lines passing through p: the bipole `sqrt(A+ x A-)` at the local orientation peaks.

  Then:
  - `J3` = the third-strongest arm. A junction needs three arms, and its weakest essential arm is its evidence;
  - junctions = the local maxima of `J3` above `tau_J`;
  - the type, from the arms:
    - 3 arms with one passing line: T;
    - 3 ends: Y;
    - 2 passing lines at least 15 degrees apart: crossing candidate;
    - otherwise: compound.
- **Code:** `v2.junction_cells(C, E, S, r0=4, tau_end=1.5, tau_pass=3, tau_J=3)` returns `J3`, the centres, their arms
  and types.
- **Figure:**
  - A: a junction cell's anatomy on a synthetic fork, T and crossing: the disc, the rays, the arms found;
  - B: at `fork`, `tee`, `crossing`, `end` and `texture`, the OD crop with the disc, the arms as arrows (solid for
    passing lines, dashed for ends, length by strength), `J3` and the type;
  - C: the zooms, `J3` as a heat map, junction centres coloured by type, truth or hand-annotated junctions as green
    rings; a histogram of `J3` at truth junctions and elsewhere.
- **Look for:**
  - 3 arms at forks and tees, 4 at crossings;
  - a junction at forks whose branch trace stopped short (the reason this layer exists);
  - nothing at `end`, `texture` or `curved`.
- **Numbers:**
  - junction recall and precision per type, on vesselscene (junction cells alone, today's junctions from traces, and
    their union) and on your hand-annotated junctions;
  - arm direction error against vesselscene's `junction_arm_dir`;
  - false junctions on `empty`, `end` and the background;
  - split-half agreement of junction positions.
- **Gate (SPEC phase 1):** 3-way recall up by at least 15 points at equal or better precision; false junctions on
  `empty`, `end` and the background not up.
- **Retires:** `vessels._bridge_ends`, `_end_points`, `_walk_back`.

#### 09 good continuation (SPEC 4.4)

- **Computes:** which arms of a junction belong to one vessel.
  - `cont(a, b) = exp(-turn^2 / (2 * 20deg^2)) * exp(-ln(w_a / w_b)^2 / (2 * ln(1.5)^2))`: high when b carries a
    straight on at a similar width.
  - A crossing pairs its 4 arms into the two best lines.
  - A fork's parent is its best pair, and the third arm is the branch.
- **Code:** `v2.arm_widths(od, junction)`; `v2.continuation(junction, sigma_turn=20, sigma_w=ln(1.5))`.
- **Figure:**
  - A: `cont` as a map over turn angle and width ratio, with the 0.3 line marked;
  - B: at `crossing`, `fork` and `tee`, the arms coloured by pairing, widths as bars, scores written; at forks,
    Murray's ratio (`r_parent^3` against `r_1^3 + r_2^3`) as a diagnostic;
  - C: the typing confusion matrix on matched junctions (vesselscene; hand-annotated junctions on real frames).
- **Look for:** the crossing's arms paired straight across; the fork's parent being the wider, straighter pair.
- **Numbers:** pairing accuracy (vesselscene: paired arms belong to the same truth vessel); typing accuracy.
- **Gate:** together with layer 10 (SPEC phase 2): typing up by at least 10 points.

#### 10 transparency: fork or crossing (SPEC 4.5)

- **Computes:** two vessels at different depths add their OD where they cross; at a fork the lumens merge, so the
  centre is about as dark as the darker arm.
  - `rho = centre OD / (a1 + a2)`, with a1, a2 the two lines' arm ODs.
  - A crossing predicts `(max + kappa * min) / (a1 + a2)`; a fork predicts `max / (a1 + a2)`.
  - Decided by a likelihood ratio using the local noise. Between 1/3 and 3 the junction is ambiguous.
- **Code:** `v2.transparency(od, noise, junction, kappa=0.87) -> rho, likelihood_ratio, ambiguous`.
- **Figure:**
  - A: where the numbers are measured at `crossing`: each arm's stretch from R to R + 2w, the centre disc, a1, a2 and
    the centre value;
  - B: rho against the arms' min/max contrast for every 4-arm junction, with the fork and crossing predictions drawn;
    points coloured by truth type (vesselscene) or by decision (real);
  - C: histograms of rho at truth crossings and forks, and the ROC curve with its area.
- **Look for:** two separate clouds where the arms' contrasts are similar; overlap only where one vessel is much
  fainter, and those flagged ambiguous.
- **Numbers:** the area under the ROC where min/max contrast is at least 0.5; the share ambiguous.
- **Gate (SPEC phase 2):** area at least 0.75 where min/max is at least 0.5; with layer 09, typing up by at least 10
  points.

### 5.4 Phase 3: wide vessels and one-sided edges (layer 11)

#### 11 border ownership and grouping cells (SPEC 4.9)

- **Computes:** a vessel owns both its borders; an edge (the aperture, glare, a lone wall) owns one.
  - `left[k, h]`: an edge darkening toward p, at distance h on one side; `right[k, h]`: its mirror on the other side.
  - `G[k, h] = sqrt(left * right)`, zero for a single edge.
  - `h*` = the half width at the best G.
  - One feedback step gives a border shared by two strips to the darker strip.
- **Code:** `grouping.grouping_cells(odd, scales, half_widths) -> G, h_star, ownership`. It reads the odd responses
  from `oriented_scores(..., return_parts=True)`.
- **Figure:**
  - A: a synthetic strip and a single edge: left, right and G across h (G is zero for the edge);
  - B: at `wide`, `thin`, `edge` and `texture`, the profile across, and G as a map over position and h;
  - C: the zooms: G's maximum over orientations and widths; h* as colour on the medial axis; edges that pass step 4
    but have G = 0, outlined dashed.
- **Look for:**
  - a strong G along the centre of `wide`, where the line evidence is weak;
  - zero along `edge`;
  - h* matching the visible width.
- **Numbers:**
  - wide-vessel lumen coverage against truth (vesselscene), compared with `vessel_seeds`;
  - the response along the aperture and glare edges (real);
  - h* against the truth width (vesselscene) and against `width_mask` (real);
  - split-half agreement.
- **Gate:** lumen coverage at least `vessel_seeds`'; no more one-sided edges than today.
- **Retires:** `_dense_additions` first, then `vessel_seeds` and, with it, the Frangi part of `vessels.py`.

### 5.5 Phase 4: motion as evidence (layer 12)

**Prerequisite on the vesselscene side:** rendering bursts (the frames, as well as the average) for the development
scenes, so flicker can be scored against truth.

#### 12 flicker as line evidence (SPEC 4.14)

- **Computes:** in a registered burst, red cells and gaps move along vessels, so the lumen flickers while the tissue
  doesn't.
  - Flicker in noise units: `flicker_maps`' coherence, divided by its spread on the background
    (`_in_background_spreads`), with the reflection and dust spots of `flicker_artifacts` left out.
  - Fed into the line evidence as an extra drive, `oriented_scores(..., drive=flicker, drive_weight)`. A faint
    flowing vessel then gets the same oriented processing as a dark one, instead of being thresholded on its own and
    added by a union.
- **Code:** `motion.flicker_cnr(flicker_stats, background, noise) -> map`; `oriented_scores(..., drive=None,
  drive_weight=0)`.
- **Figure:**
  - A: the density over time at three pixels: the lumen of `faint`, its wall, and `empty`;
  - B: at `faint`, `thin` and `edge`, tuning curves with and without the drive;
  - C: the zooms: flicker in noise units, and the line evidence with and without the drive; flicker on the odd frames
    against flicker on the even frames, as a scatter.
- **Look for:**
  - flicker inside lumens, not on walls. Wall flicker is registration error; if present, subtract a model of it,
    the image gradient x the residual registration error (SPEC 4.14), or score the lumen centre only;
  - faint vessels gaining line evidence;
  - nothing new at `texture` or `empty`.
- **Numbers:**
  - split-half correlation of flicker;
  - lumen coverage (synthetic bursts);
  - the OD on empty background when flicker joins the background mask;
  - the ratio of wall flicker to lumen flicker.
- **Gate:** flicker-driven detections agree across split halves; empty-background OD not up.
- **Retires:** `flicker_mask`, `combine_masks`. `manipulation.py` then has no caller left, and goes.

### 5.6 Phase 5: recurrence and one decision (layers 04, 05, 13, 14)

#### 04 recurrent field, replacing the association field (SPEC 4.7)

- **Computes:** T = 8 iterations. Each one:
  - adds collinear support (the bipole, as now);
  - subtracts parallel neighbours (the weaker flank);
  - subtracts rival orientations within 2 steps.

  Texture's web of short parallel pieces dies out and vessels persist. A crossing's two lobes, more than 2 steps
  apart, are untouched by the rival inhibition.
- **Code:** `recurrent.recurrent_field(U4, sigma, T=8, g_E=1, g_P=0.3, g_R=0.5) -> C, (A_plus, A_minus),
  change_per_iteration`. With T = 0 it gives today's `complete`.
- **Figure:**
  - A: C at iterations 0, 1, 2, 4 and 8, on one zoom;
  - B: the change per iteration (it should fall below 1 % by T); tuning curves over the iterations at `crossing` (two
    lobes kept) and `texture` (decays);
  - C: the zooms, today's C against the new C; the response along `thin` and `curved` (dips).
- **Look for:** texture fading over the iterations; no vessel breaking; crossings keeping both lobes.
- **Numbers:**
  - the 99th percentile of C on the background (today about 5);
  - the lobes at crossings;
  - the dips along traced vessels;
  - centreline precision and recall downstream;
  - convergence.
- **Gate (SPEC phase 4):** texture response down by at least 30 %, with crossings' lobes and centreline recall kept.
- **Then:** every downstream layer (06-12) is re-run on the new C, and the gallery and probe sheet show the change.

#### 05 curvature cells (SPEC 4.8)

- **Computes:** the lobes bent along circles of curvature 0, +-1/40, +-1/20 and +-1/10 per px. The best wins, and the
  winning curvature is the curvature map.
- **Code:** `recurrent.curvature_field(U4, sigma, curvatures)`, or `curvature=True` in `recurrent_field`.
- **Figure:**
  - A: the bent lobe kernels;
  - B: at `curved` and `thin`, the response along the vessel, with straight and with bent lobes;
  - C: the zooms, curvature as signed colour on the ridges.
- **Look for:** no dips left on curved vessels; the curvature's sign matching the bend.
- **Numbers:** dips along curved vessels; curvature against truth (vesselscene).
- **Gate:** with layer 04 (SPEC phase 4).

#### 13 evidence fusion

- **Computes:** every cue in noise units, summed with weights before one threshold, instead of thresholding each cue
  and taking a union.
  - A union keeps every cue's false positives. A sum lets two weak cues agree, and lets one strong false cue be
    outvoted.
  - The cues at this point: line evidence C (with the flicker drive inside it, or flicker as a separate term, as
    phase 4 decided) and two-border evidence G.
  - The fused map is what the readout and attentive grouping read, and part of the background mask's evidence.
- **Code:** `fusion.fuse(evidence, weights) -> F`, then a threshold with hysteresis.
- **Figure:**
  - A: a toy example: two weak cues adding up, one strong false cue not passing;
  - B: at every site, a stacked bar of each cue's contribution;
  - C: the zooms: each cue, the fused map, and which cue dominates; today's `merged` beside the new mask.
- **Look for:** fewer isolated fragments than the union; faint vessels that need two cues present.
- **Numbers:** pieces, area and split-half Dice, against `merged` and `with_flicker`; vesselscene mask precision and
  recall.
- **Gate:** fewer pieces and better split-half agreement than the unions; vesselscene recall not down.
- **Retires:** `merged = seeds | neuro_vessels`, `with_flicker`, and the union cells in `dev.py`.

#### 14 attentive grouping (SPEC 4.10)

- **Computes:** one vessel at a time, strongest first.
  - A label spreads along the vessel's own orientation and boosts the evidence just ahead of it. So it carries
    through weak stretches when both sides belong to the vessel.
  - At a junction, it continues only into the arm paired with the arriving arm (layer 09): straight through
    crossings, along the parent at forks. The branches it meets are queued as later seeds.
  - A grouped vessel's lumen is removed from later seeds (inhibition of return).
- **Code:** `attend.attend_group(C, junctions, gamma=0.3, t_low=1.5) -> vessels` (ordered, each with its identity).
- **Figure:**
  - A: the label spreading for the first vessel, in snapshots, through `crossing`;
  - B: at `crossing` and `fork`, vessels coloured by identity;
  - C: the zooms, every vessel by identity, with its order number.
- **Look for:**
  - one colour straight through each crossing;
  - the parent's colour continuing through a fork, and a new colour for the branch;
  - no vessel jumping to a parallel neighbour.
- **Numbers:** pieces per truth vessel (fragmentation, about 3 today), centreline F1 and fork recall (vesselscene);
  vessels found, mean length and split-half agreement (real).
- **Gate (SPEC phase 5):** fragmentation down, centreline F1 not down, fork recall up.
- **Retires:** the layer-06 tracer.

### 5.7 Phase 6: from masks to a graph and a fit (layers 15, 16)

#### 15 graph and spline fit

- **Computes:**
  - a graph: the grouped vessels are its edges, and the junction cells (typed, with their pairings) its nodes;
  - then the spline fit of `experiments/splinefit` on the real average. It renders the graph, compares the render
    with the OD (weighted, in noise units), adjusts positions, widths, blur and contrast, and prunes what doesn't pay
    for itself.
- **Code:** `graph.build(vessels, junctions) -> network`; the fit is called through the adapter in the other
  direction, with vesselscene's fitter on MICA's data.
- **Figure:**
  - A: the initial network over the OD;
  - B: at every junction site, the OD, the render and the residual (OD - render) / noise, as in the junction gallery;
  - C: the zooms, the fitted render and the residual; the width along `thin` and `wide`, fit against `width_mask`.
- **Look for:**
  - a residual near zero on vessels and background;
  - no systematic dark ring (render too thin) or bright halo (render too wide);
  - junction tiles that look right.
- **Numbers:**
  - residual RMS on vessels and on background, in noise units;
  - fitted widths against `width_mask` on clean vessels;
  - vesselscene per-junction scorecard (found, typed, render at least 95 %).
- **Gate:** fitted widths agree with `width_mask` on clean vessels; the residual and the background offset are
  acceptable.
- **Retires:** `neuro.width_mask`, `neuro.neuro_mask`. The vessel output becomes the fitted graph and its render.

#### 16 local model comparison (SPEC 4.5b)

- **Computes:** for each ambiguous junction from layer 10:
  - render the competing hypotheses in a small window: crossing, fork with each pair as parent, T;
  - fit their profiles briefly;
  - keep the one with the lowest residual plus description cost.
- **Code:** `v2.compare_models(od, weights, junction) -> choice, residuals`.
- **Figure:** for up to six ambiguous junctions: the OD, each hypothesis's render and residual, the cost bars, and the
  choice marked.
- **Numbers:** typing accuracy on ambiguous junctions; time per junction (about 0.2-0.5 s).
- **Gate:** typing on ambiguous junctions up, with no loss elsewhere.

### 5.8 Phase 7: feedback and refinement (layers 17-20)

#### 17 top-down feedback (SPEC 4.11)

- **Computes:** where the fitted render leaves OD unexplained (in noise units) near a junction or ahead of a free end,
  the gain of the line evidence is raised there, and layers 04-14 are run again. Nowhere else.
  - Two loops, each run once: inside the proposer, and from the fit back to the proposer.
- **Code:** `feedback.feedback_gain(od, render, noise, junctions, ends, gamma_fb=0.5) -> gain`;
  `recurrent_field(..., gain=gain)`.
- **Figure:**
  - A: the error map, the "near" mask, and the gain;
  - B: at `fork`, `tee` and `end`, before and after;
  - C: the zooms, with the junctions and ends that feedback added marked.
- **Look for:** missed branches recovered near forks; no new junctions in texture.
- **Numbers:** 3-way recall at equal junction precision. The earlier attempt lowered precision from 0.82.
- **Gate (SPEC phase 6):** 3-way recall up at equal precision.

#### 18 population codes (SPEC 4.12)

- **Computes:** start values for the fit, read from the whole population rather than the winning cell:
  - position: sub-pixel, from the profile across the ridge;
  - tangent: the population vector over the 16 orientations;
  - width: from the scale population, or layer 11's h*;
  - contrast: from the OD.
- **Code:** `decode.decode_parameters(C, S, od, vessels)`, which returns the position, tangent, r, s, a and curvature
  at each point.
- **Figure:**
  - B: along `thin`, `wide` and `curved`, the decoded width and tangent against the fitted values;
  - C: decoded values against truth, as scatters (vesselscene).
- **Numbers:** start-value errors against truth; the fitted geometry.
- **Gate (SPEC phase 7):** start errors down; the fitted geometry not worse.

#### 19 surface completion (SPEC 4.13)

- **Computes:** the background is interpolated only behind evidence that a vessel is there. The evidence is the union
  of:
  - today's mask;
  - the grouping cells' strips (layer 11);
  - the grouped vessels' lumens (layer 14);
  - flicker (layer 12).

  Each is a cue that a dark texture blob doesn't produce. An OD threshold alone failed (E24): it took dark texture
  and biased the background.
- **Code:** `background.fit_background(..., extra=evidence)`.
- **Figure:**
  - A: the background under `wide`, before and after;
  - B: at `texture`, `empty` and `wide`, the background mask and the OD;
  - C: histograms of the OD on the background, before and after.
- **Numbers:** the OD kept in the true lumen and the OD on empty background, which must stay at 0 (vesselscene); the
  OD on the background and split-half agreement (real).
- **Gate (SPEC phase 6):** empty-background OD not up.

#### 20 flow direction (SPEC 4.14)

- **Computes:** along each grouped vessel, the OD sampled along its centreline in every frame gives a space-time
  image. Moving red cells make slanted streaks in it.
  - Two motion-energy filters, tuned downstream and upstream, give the direction; the streaks' slope gives the speed.
  - At a 3-way junction: one arm in and two out is a bifurcation; two in and one out is a confluence.
- **Code:** `motion.flow_direction(frames, vessel) -> direction, speed`.
- **Figure:**
  - A: the space-time image along `thin`, with its streaks and the two energy responses;
  - C: the zooms, arrows along the vessels, and junctions labelled bifurcation or confluence.
- **Numbers:** direction accuracy (synthetic bursts); consistency at junctions (real: every 3-way junction should be
  one in and two out, or two in and one out).
- **Gate (SPEC phase 8):** direction right on most perfused vessels.

## 6. Checkpoints: when to look

- **After every change to a layer:** its own figure, against the previous version, in the gallery.
- **After a layer is switched on:** the probe sheet, to see whether anything downstream changed.
- **When a layer changes an earlier layer's input** (the recurrent field in phase 5, feedback in phase 7): re-run
  every layer downstream and read the gallery top to bottom.
- **At the end of every phase:**
  - the whole gallery, for both datasets;
  - the phase's gate, on truth and on the real burst;
  - the code the phase replaces moved to `legacy/`;
  - a short entry in `LAYERS_LOG.md`.

## 7. Who builds what

- **In MICA (you):** the layers, their figures and `dev.py` cells, `checks.py`, and the sites and gallery tools.
- **In vesselscene (I can do this):**
  - the adapter that runs the MICA functions on the development scenes and scores them;
  - `sites_from_truth`;
  - burst rendering for phase 4;
  - the scorecards.

  Once MICA is on GitHub and added to a session, I can also review each layer against the spec and run the
  vesselscene scoring on each commit.

## 8. End state

- **Kept:**
  - `ingest_stabilize`, `view_burst`;
  - `background.py`, with surface completion;
  - `neuro.py`'s simple cells and surround;
  - `temporal.flicker_maps` and its helpers;
  - the figure modules.
- **New:**
  - the readout;
  - V2: end-stopped cells, junction cells, continuation, transparency, model comparison;
  - grouping cells;
  - motion;
  - the recurrent field and curvature;
  - fusion and attentive grouping;
  - the graph and fit, feedback, and decoding.
- **Retired** (in `legacy/` first, deleted once the end state's gates hold): `manipulation.py`, `vessels.py`,
  `flicker_mask`, `combine_masks`, `neuro.track`, `width_mask`, `neuro_mask`, and the union cells.
- **Two outputs, with separate jobs:**
  - **the background mask:** generous by design, built from vessel evidence (layer 19), it keeps vessels out of the
    background fit;
  - **the vessels:** the fitted graph (typed junctions, vessels with identity, widths along each) and its render, not
    a union of thresholded masks.
