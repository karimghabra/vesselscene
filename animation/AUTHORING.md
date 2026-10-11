# Authoring a chapter

Read [PLAN.md](PLAN.md) for the storyboard and `scenes/s04_hypercolumn.py`. That
file is the reference chapter, and every other chapter follows its pattern.

## The four cuts

| cut | chapters | level | per chapter |
|---|---|---|---|
| very short | `s99_trailer.py` only | – | ~90 s total |
| short | 0, 1, 2, 4, 8, 9, 10, 11, 14, 18 | 1 | 20–40 s |
| medium | 0, 1, 2, 4, 7, 8, 9, 10, 11, 14, 16, 18 | 2 | 60–110 s |
| detailed | 0–18, all | 3 | 120–200 s |

`versions.py` lists them. A chapter may assume the chapters before it **in the
same cut**. For example, chapter 8 in the short cut cannot rely on chapter 7,
the association field, because chapter 7 is not in the short cut. In that case it
must introduce the lobes A⁺ and A⁻ itself, in one sentence and one picture.
Chapters outside the short and medium cuts need only be good at level 3, but they
must still render at every level.

## Structure

```python
class EndStopped(ChapterScene):
    NUMBER, TITLE, SPEC = 8, "Where lines end", "spec 4.2"
    MAP_KEYS = ("endstop",)              # boxes lit on the architecture map (archmap.py)

    def construct(self):
        self.intro()                     # map (level 3) + chapter card
        ...                              # level 1: the one idea
        if self.at(2): ...               # level 2: + the equation, the mechanism
        if self.at(3): ...               # level 3: + spec numbers, variants, caveats
        self.outro()

variants(EndStopped, module_globals=globals())   # -> EndStoppedShort / Medium / Detailed
```

File names: `scenes/sNN_<topic>.py` (two-digit chapter number). One base
class per chapter.

## Rules

1. **Computed, not drawn.** Images, maps, tuning curves and plots come from
   the spec's equations, run in numpy on toy stimuli: `mica_anim/neuro.py`, or
   your own module `mica_anim/<topic>.py`. Cache anything slow with
   `functools.lru_cache`. Keep the computation under about 20 s per scene.
2. **Honest numbers.** Label values computed on screen as the toy model's. Quote
   spec measurements with their section ("measured, spec 4.6b: 15 → 2"). If
   the toy disagrees with the spec, say so on screen. Never present a toy number
   as a measurement.
3. **One question per chapter.** Open with the puzzle in a caption, show the
   picture, and only then show the equation. Build equations term by term with
   `MathTex` substrings, colouring each term like the thing it stands for in
   the picture.
4. **Colours** (`mica_anim/style.py`) mean the same thing everywhere:
   - orientation: hue (`orientation_color`)
   - `EXC` (blue): excitation
   - `INH` (red): inhibition
   - `FF` (blue): feedforward
   - `FB` (gold): feedback or recurrence
   - `END` (orange): line ends
   - `JUNC` (pink): junctions
   - `CONT` (green): continuation
   - `ATTN` (yellow): the attention label
   - `OD_COL`: OD in equations
   - `MUTED`: axes and guides

   Junction types in vesselscene imagery use `vsdata.JUNCTION_COLORS`.
5. **Layout.** The frame is x ∈ [−7.1, 7.1], y ∈ [−4, 4]. Captions sit at the
   bottom (`self.say`), and all content must stay above y = −3.05. Leave
   breathing room and never overlap text. Use `Tex`/`MathTex`, never `Text`.
6. **Captions** (`self.say(text, wait=...)`) carry the narration. Keep each one
   under about 25 words and hold it for its reading time (`wait="read"`, or
   animations that last as long). Plain words, no jargon without a picture.
7. **3Blue1Brown motion.** Use smooth transforms between representations
   (`TransformFromCopy`, `ReplacementTransform`, `TransformMatchingTex`), and
   `ValueTracker` + `always_redraw` for anything that changes continuously
   (sliders, sweeps, a dragged probe). Use `Indicate`/`Circumscribe` sparingly for
   emphasis. One thing moves at a time unless the motions are the same idea.
8. **Shared code is read-only for chapter authors.** That covers `chapter.py`,
   `style.py`, `widgets.py`, `neuro.py`, `archmap.py` and `vsdata.py`. Put new
   helpers in your own module or scene file. If you need a shared change,
   report it instead of making it.

## Shared toolkit

- `style`
  - `PixelFrame(shape, height, centre)`: `.point(x, y)` maps image pixels (y down)
    to scene points, `.pixel(point)` inverts it, `.unit` is scene units per pixel,
    and `.screen_angle(theta)` gives an image angle in scene terms.
  - `image(array, height, ramp, vmin, vmax)` makes a pixelated `ImageMobject`;
    the ramps are `"grey"`, `"magma"` and `"diverging"`.
  - `bilinear(stack, x, y)` samples a stack.
- `widgets`
  - `PolarTuning(radius, vmax)`, with `.curve(values)`, `.spoke(k, v)`,
    `.spokes(values)` and `.ring(v)`.
  - `receptive_field(length, sigma)`.
  - `bar_chart(values, ...)`.
- `neuro`
  - `vessel_od` (segments, sum or union), `ring_od`, `blob_od`, `add_noise` and
    `to_intensity`.
  - `line_kernel` and `simple_cells`, which give 16 orientations, `THETAS`.
  - `noise_units`, `floor`, `surround` and `tuning_curve`.
- `archmap.ArchitectureMap()`: `.boxes[key]` and `.highlight(keys)`.
- `vsdata`: vesselscene imagery and truth.
  - `load()` returns `.average`, `.od`, `.burst`, `.truth` and `.view(img)`.
  - `runs_overlay`, `junctions_overlay` and `identity_color` draw the truth.
  - `available()` reports whether the data is there.

## Render and check

```bash
V=/tmp/claude-0/-home-user-vesselscene/c284b648-6b71-53f2-bbc7-6f9ccdde7e61/scratchpad/venv
cd animation
$V/bin/manim -ql --media_dir <your scratch dir> scenes/s08_end_stopped.py EndStoppedShort EndStoppedMedium EndStoppedDetailed
```

Then extract frames with ffmpeg (`ffmpeg -ss T -i file.mp4 -frames:v 1 f.png`, or
tile several with `hstack`/`vstack`) and look at them. Check for overlaps,
clipping, unreadable text, and wrong orientations (image y points down, scene y
points up).
