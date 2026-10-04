"""Side-by-side figures: each annotator's tracing over the image, with the observable truth beneath it.

    python -m experiments.neuromimetic.figures SCENE_DIR --kind average --crop Y,X,H,W --zoom 2 \
        --annotators experiments.neuromimetic.baseline_hessian:annotate experiments.neuromimetic.neuromimetic:annotate \
        -o fig.png

Each panel: the image (grey), the observable truth's lines (thin green) and junctions (small green rings),
the annotator's polylines (one colour per polyline) and junctions (magenta X = crossing, cyan circle = 3-way,
yellow square = compound).  The first panel is the truth alone.
"""
from __future__ import annotations

import argparse
import importlib

import cv2
import numpy as np

from . import harness as H

JCOL = {"crossing": (255, 0, 255), "compound": (0, 255, 255)}


def _grey(img, valid):
    v = img[valid]
    lo, hi = np.percentile(v, [1, 99.5])
    g = np.clip((np.where(valid, img, lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _pts(P, crop, z):
    y0, x0 = crop[0], crop[1]
    Q = (np.asarray(P, float).reshape(-1, 2) - [x0, y0]) * z
    return np.rint(Q * 4).astype(np.int32)                  # 2 bits of sub-pixel precision for cv2


def panel(case, out, crop, z, title):
    y0, x0, h, w = crop
    base = _grey(case["image"], case["valid"])[y0:y0 + h, x0:x0 + w]
    im = cv2.resize(base, (w * z, h * z), interpolation=cv2.INTER_NEAREST)
    obs = case["obs"]
    for r in obs["runs"]:
        for P in H.T.observed_pieces(r):
            if len(P) > 1:
                cv2.polylines(im, [_pts(P, crop, z)], False, (0, 170, 0), 1, cv2.LINE_AA, shift=2)
    for j in obs["junctions"]:
        c = _pts([[j["x"], j["y"]]], crop, z)[0]
        cv2.circle(im, tuple(int(v) for v in c), int(4 * z * 4), (0, 170, 0), 2, cv2.LINE_AA, shift=2)
    if out is not None:
        rng = np.random.default_rng(0)                        # fixed colours (figure only)
        for P in out.get("polylines", []):
            if len(P) > 1:
                col = tuple(int(c) for c in rng.integers(60, 256, 3))
                cv2.polylines(im, [_pts(P, crop, z)], False, col, 2, cv2.LINE_AA, shift=2)
        for j in out.get("junctions", []):
            c = _pts([[j[0], j[1]]], crop, z)[0] // 4
            t = j[2] if len(j) > 2 else "branch"
            if t == "crossing":
                cv2.drawMarker(im, tuple(int(v) for v in c), JCOL[t], cv2.MARKER_TILTED_CROSS, 6 * z, 2)
            elif t == "compound":
                cv2.drawMarker(im, tuple(int(v) for v in c), JCOL[t], cv2.MARKER_SQUARE, 6 * z, 2)
            else:
                cv2.circle(im, tuple(int(v) for v in c), 3 * z, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (w * z, 22), (0, 0, 0), -1)
    cv2.putText(im, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--kind", default="average")
    ap.add_argument("--crop", default=None, help="Y,X,H,W (default: the whole image)")
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--annotators", nargs="+", default=[])
    ap.add_argument("--cols", type=int, default=0, help="panels per row (default: all in one row)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    case = H.load_case(a.scene, a.kind)
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else (0, 0) + case["image"].shape
    panels = [panel(case, None, crop, a.zoom, "observable truth")]
    for ann in a.annotators:
        mod, fn = ann.split(":")
        out = getattr(importlib.import_module(mod), fn)(case["image"].copy(), case["valid"].copy())
        panels.append(panel(case, out, crop, a.zoom, ann.split(".")[-1]))
    cols = a.cols or len(panels)
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[i:i + cols], 1) for i in range(0, len(panels), cols)]
    cv2.imwrite(a.out, np.concatenate(rows, 0))


if __name__ == "__main__":
    main()
