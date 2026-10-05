"""Spline-network fitting of whole vesselscene stills (README.md when written).

The neuromimetic annotator (experiments.neuromimetic) PROPOSES vessels and junctions; this package turns the
proposal into a spline network (centreline B-splines with radius, blur and contrast profiles, nodes at forks
and confluences, crossings as overlaps without nodes) and fits its differentiable, junction-aware render
jointly to the whole image's optical density (OD = B - log I, the background B fitted on the vesselness
mask's negative; never a vesselness map).

Modules: proposals (neuromimetic proposal -> initial network), render (the junction-aware renderer, a
subclass of LIMBUS vesselmap's NetworkModel), fit (joint optimisation and MDL pruning), pipeline (the
harness entry point annotate(image, valid)).
"""
from __future__ import annotations

import os
import sys

THREADS = 2


def use_vesselmap():
    """Put LIMBUS (vesselmap, read-only) on sys.path and return the vesselmap package."""
    from experiments.neuromimetic import limbus_root
    root = limbus_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    import vesselmap
    return vesselmap


def set_threads(n: int = THREADS):
    """Pin every numeric library to n threads (the machine is shared) and make torch deterministic."""
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    import cv2
    import torch
    cv2.setNumThreads(n)
    torch.set_num_threads(n)
    torch.use_deterministic_algorithms(True)
