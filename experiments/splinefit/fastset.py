r"""The fast tier: a fixed set of 256 x 256 px crops of the dev images (fastset.json), chosen once from the truth.

One crop per (dev scene, kind): for each scene, every window on a 32 px grid (plus the windows flush with the
right and bottom borders) that is at least MIN_VALID valid is ranked by the number of observable junctions of
that kind's truth at least score.CROP_BORDER_PX inside it (the junctions the crop scores), ties broken by the
observable lumen area inside it, then by (y0, x0).  The average takes the best window; the frame the best one
overlapping the average's by at most MAX_IOU (intersection over union), so the two kinds of a scene show
different places.  Scenes are the dev folders in sorted order, so the set is deterministic; it is written
once to fastset.json (scene folder name, kind, crop = [x0, y0, w, h], the junction count) and then frozen
with the scorer.  The fast-tier score is the mean composite over the crops (run_experiment.py).

    python -m experiments.splinefit.fastset [--dev DIR] [--write]

DIR defaults to $SPLINEFIT_DEV or the session's dev scenes (DEV_DIR).  Held-out scenes are refused.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CROP = 256
STRIDE = 32
MIN_VALID = 0.97
MAX_IOU = 0.25
KINDS = ("average", "frame")
HERE = os.path.dirname(os.path.abspath(__file__))
FASTSET_JSON = os.path.join(HERE, "fastset.json")
DEV_DIR = os.environ.get("SPLINEFIT_DEV", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/"
                                          "scratchpad/scenes/dev")


def check_not_heldout(path: str):
    """The development loop never reads held-out scenes (only the final evaluation may, with
    SPLINEFIT_ALLOW_HELDOUT=1)."""
    if "heldout" in os.path.abspath(path) and os.environ.get("SPLINEFIT_ALLOW_HELDOUT") != "1":
        raise PermissionError(f"refusing a held-out path in the development loop: {path}")


def dev_scenes(dev: str = DEV_DIR) -> list:
    """The dev scene folders (those with a truth.json), sorted."""
    check_not_heldout(dev)
    return [os.path.join(dev, d) for d in sorted(os.listdir(dev))
            if os.path.isfile(os.path.join(dev, d, "truth.json"))]


def _starts(n: int) -> list:
    s = list(range(0, n - CROP + 1, STRIDE))
    return sorted(set(s + [n - CROP]))


def _iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / float(a[2] * a[3] + b[2] * b[3] - i)


def rank_windows(folder: str, kind: str) -> list:
    """Every candidate window of one image as (junctions, lumen px, y0, x0), best first."""
    import tifffile
    from vesselscene import truth as T
    from .score import CROP_BORDER_PX
    img = tifffile.imread(os.path.join(folder, f"still_{kind}.tif")).astype(np.float32)
    valid = np.isfinite(img)
    obs = T.load_observable(os.path.join(folder, f"observable_{kind}.json"))
    with np.load(os.path.join(folder, "labels.npz")) as z:
        lumen = z[f"observable_lumen_{kind}"].astype(bool)
    J = np.array([[j["x"], j["y"]] for j in obs["junctions"]], float).reshape(-1, 2)
    Hh, Ww = img.shape
    cv = np.pad(np.cumsum(np.cumsum(valid, 0), 1), ((1, 0), (1, 0)))
    cl = np.pad(np.cumsum(np.cumsum(lumen, 0), 1), ((1, 0), (1, 0)))
    box = lambda c, y, x: int(c[y + CROP, x + CROP] - c[y, x + CROP] - c[y + CROP, x] + c[y, x])
    out = []
    for y0 in _starts(Hh):
        for x0 in _starts(Ww):
            if box(cv, y0, x0) < MIN_VALID * CROP * CROP:
                continue
            x, y = J[:, 0] - x0, J[:, 1] - y0
            d = np.minimum.reduce([x + 0.5, CROP - 0.5 - x, y + 0.5, CROP - 0.5 - y]) if len(J) else np.zeros(0)
            out.append((int((d >= CROP_BORDER_PX).sum()), box(cl, y0, x0), y0, x0))
    out.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    return out


def select(dev: str = DEV_DIR) -> list:
    """The fast set (module docstring)."""
    rows = []
    for folder in dev_scenes(dev):
        name = os.path.basename(folder)
        chosen = None
        for kind in KINDS:
            ranked = rank_windows(folder, kind)
            pick = ranked[0]
            if chosen is not None:
                for cand in ranked:
                    if _iou((cand[3], cand[2], CROP, CROP), chosen) <= MAX_IOU:
                        pick = cand
                        break
            crop = [pick[3], pick[2], CROP, CROP]
            chosen = chosen or crop
            rows.append(dict(scene=name, kind=kind, crop=crop, junctions=pick[0], lumen_px=pick[1]))
    return rows


def load(path: str = FASTSET_JSON) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["crops"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", default=DEV_DIR)
    ap.add_argument("--write", action="store_true", help=f"write {FASTSET_JSON} (refused if it exists)")
    a = ap.parse_args(argv)
    rows = select(a.dev)
    for r in rows:
        print(json.dumps(r))
    if a.write:
        if os.path.exists(FASTSET_JSON):
            raise SystemExit(f"{FASTSET_JSON} exists: the fast set is frozen")
        with open(FASTSET_JSON, "w", encoding="utf-8") as fh:
            json.dump(dict(note="fast tier: crops of the dev scenes (fastset.py); frozen with score.py",
                           size=CROP, stride=STRIDE, min_valid=MIN_VALID, max_iou=MAX_IOU, crops=rows), fh, indent=1)


if __name__ == "__main__":
    main()
