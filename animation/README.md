# Animation: the recurrent mid-level vision spec, explained

A 3Blue1Brown-style explainer of how the recurrent mid-level vision spec turns a
burst of conjunctival video into an annotated vessel graph. See [PLAN.md](PLAN.md)
for the storyboard.

## Setup

Manim needs Cairo/Pango headers and LaTeX:

```bash
sudo apt-get install libpango1.0-dev pkg-config texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-science dvisvgm cm-super ffmpeg
pip install -r animation/requirements.txt
```

Chapters that show vesselscene imagery also need this package (`pip install -e .`
from the repository root) and PyTorch.

## Render

```bash
cd animation
manim -ql scenes/s04_hypercolumn.py Hypercolumn    # 480p15 preview
manim -qh scenes/s04_hypercolumn.py Hypercolumn    # 1080p60
```

Output goes to `animation/media/`, which git ignores.
