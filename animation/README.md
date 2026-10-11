# Animation: the recurrent mid-level vision spec, explained

A 3Blue1Brown-style explainer of how the recurrent mid-level vision spec turns a
burst of conjunctival video into an annotated vessel graph. It covers the math,
the physiology of the image, and the neuroscience behind each layer. Every image,
map and curve on screen is computed from the spec's equations on toy stimuli or
on vesselscene renders. Numbers marked *measured* are quoted from the spec.

## Four cuts

| cut | content | length |
|---|---|---|
| very short | a standalone trailer (`scenes/s99_trailer.py`) | ~1.3 min |
| short | chapters 0, 1, 2, 4, 8, 9, 10, 11, 14 and 18, one idea each | ~6 min |
| medium | the core chapters, with their equations | ~17 min |
| detailed | all 19 chapters, with the spec's measurements, the architecture map before each chapter, and the references | ~45 min |

The chapters are in `scenes/`, one file per chapter. Each chapter is written once
at three depths, and `versions.py` lists which variants each cut uses. See
[PLAN.md](PLAN.md) for the storyboard and [AUTHORING.md](AUTHORING.md) for the rules
the chapters follow.

| # | chapter | spec |
|---|---|---|
| 0 | Cold open: a burst of video, the target graph, the junction problem | 1, 2 |
| 1 | The physics of the picture: Beer-Lambert, the chord law, sum against union | 4.5, 6.4 |
| 2 | Two modes of vision; the architecture map | 1, 4.1, App. A |
| 3 | Retina, ganglion cells and noise units | 4.0 |
| 4 | One point, sixteen orientations: the hypercolumn and the surround | 2.4, 4.6 |
| 5 | Coarse cells that overshoot: scale selection | 4.6b |
| 6 | Bends: arc cells | 4.8 |
| 7 | Horizontal connections: the recurrent association field | 4.7 |
| 8 | Where lines end: end-stopped cells | 4.2 |
| 9 | Junction cells | 4.3, 2.2 |
| 10 | Good continuation is co-circularity | 4.4, 4.10b |
| 11 | Fork or crossing? Transparency and model comparison | 4.5, 4.5b |
| 12 | A vessel owns both of its borders; evidence fusion | 4.9, 4.9b |
| 13 | Sixteen sheets: reading out per orientation | 4.0, 4.9c |
| 14 | Attention traces one vessel at a time | 4.10, 4.10b |
| 15 | Top-down feedback | 4.11 |
| 16 | Population codes | 4.12 |
| 17 | Motion: the blood flows | 4.14 |
| 18 | The whole loop, and how we'd know it works | 4.1, 6.2, 6.4, 1.1 |

## Setup

Manim needs Cairo/Pango headers and LaTeX:

```bash
sudo apt-get install libpango1.0-dev pkg-config texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-science dvisvgm cm-super ffmpeg
pip install -r animation/requirements.txt
```

The vesselscene imagery lives in `data_cache/`, which git ignores. Making it
needs this package (`pip install -e .` from the repository root) and PyTorch:

```bash
python animation/tools/make_vs_data.py --seed 0 --shape 540 960 --frames 16   # ~30-60 min on a CPU
```

## Render

```bash
cd animation
manim -ql scenes/s09_junction_cells.py JunctionCellsMedium   # one chapter variant, 480p preview
python versions.py list                                      # what each cut contains
python versions.py render all --skip-existing -j 3           # 1080p30, then join each cut
```

Each cut lands in `media/cuts/<cut>.mp4`. `media/` is git-ignored.
