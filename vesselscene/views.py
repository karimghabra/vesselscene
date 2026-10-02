"""Side-by-side views of synthetic scenes and real stills (for looking, not
for scoring): every image shown as F = ln I - background (stats.flatten) on
the same grey scale, -0.35 .. +0.08 Np, NaN white.

    figure_average(scene_dir, real_paths, path)   the synthetic average still next to real stills:
                                                  full frames and two zoomed crops at the same places
    figure_frame(scene_dir, real_frame, path)     the synthetic single frame, its own average (same
                                                  anatomy: "fuller") and a real raw frame
    figure_truth(scene_dir, path)                 the truth overlay on a zoomed crop, next to the plain still
    load_scene(folder)                            a saved scene back as a scene.Scene (no fills / formed)
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from . import realism as RM
from . import scene as SC
from .graph import VesselGraph

VMIN, VMAX = -0.35, 0.08
DATA_ROOT = str(RM.find_data_root() or "")      # $LIMBUS_DATA, or a sibling LIMBUS checkout ../limbus
DEFAULT_REAL = (os.path.join(DATA_ROOT, "stabilization", "nonrigid", "burst_2026-09-16_15-50-52", "mean_stabilized.tif"),
                os.path.join(DATA_ROOT, "stabilization", "translation", "burst_2026-09-17_13-51-26",
                             "mean_stabilized.tif"))
DEFAULT_REAL_FRAME_BURST = os.path.join(DATA_ROOT, "reference_data", "burst_2026-09-16_15-50-52")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white", "axes.titlesize": 9})
    return plt


def _read(path):
    import tifffile
    return tifffile.imread(path).astype(np.float64)


def _F(img, frame=False):
    """F (flattened log, Np) and valid; a raw frame's saturated pixels are invalid."""
    from . import stats as st
    I = np.asarray(img, np.float64)
    valid = np.isfinite(I) & (I > 0)
    if frame:
        valid &= I < 4095
    F = st.flatten(I, valid)[0]
    return np.where(valid, F, np.nan), valid


def _show(ax, F, title):
    import matplotlib as mpl
    cm = mpl.colormaps["gray"].copy()
    cm.set_bad("white")
    ax.imshow(np.ma.masked_invalid(np.clip(F, VMIN, VMAX)), cmap=cm, vmin=VMIN, vmax=VMAX,
              interpolation="antialiased")
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _crop_boxes(shape, fracs, size):
    H, W = shape
    ch, cw = size
    out = []
    for fy, fx in fracs:
        y0 = int(np.clip(fy * H - ch / 2, 0, max(H - ch, 0)))
        x0 = int(np.clip(fx * W - cw / 2, 0, max(W - cw, 0)))
        out.append((y0, x0))
    return out


def figure_average(scene_dir: str, real_paths=DEFAULT_REAL, path: str | None = None,
                   crops=((0.35, 0.3), (0.65, 0.7)), crop_size=(300, 480), dpi=100) -> str:
    """Synthetic average still next to real stills: full frames on the same
    grey scale, and two 300 x 480 crops at the same fractional positions
    (1:1 pixels)."""
    from matplotlib.patches import Rectangle
    plt = _plt()
    syn = _read(os.path.join(scene_dir, "still_average.tif"))
    imgs = [("synthetic average (" + os.path.basename(os.path.normpath(scene_dir)) + ")", syn)]
    for p in real_paths:
        imgs.append(("real " + os.path.basename(os.path.dirname(p)).replace("burst_2026-", "") + " "
                     + os.path.basename(os.path.dirname(os.path.dirname(p))), _read(p)))
    n = len(imgs)
    fig = plt.figure(figsize=(6.2 * n, 4.1 + 2 * 3.1))
    gs = fig.add_gridspec(3, n, height_ratios=[1200 / 1920, 300 / 480, 300 / 480], hspace=0.12, wspace=0.03)
    for j, (name, I) in enumerate(imgs):
        F, _ = _F(I)
        ax = fig.add_subplot(gs[0, j])
        _show(ax, F, f"{name}\n{F.shape[0]}x{F.shape[1]} px, F on {VMIN}..{VMAX:+} Np")
        boxes = _crop_boxes(F.shape, crops, crop_size)
        for i, (y0, x0) in enumerate(boxes):
            ax.add_patch(Rectangle((x0, y0), crop_size[1], crop_size[0], fill=False, ec="#eb6834", lw=1.2))
            ax.text(x0 + 6, y0 + 6, str(i + 1), color="#eb6834", va="top", fontweight="bold")
            axc = fig.add_subplot(gs[1 + i, j])
            _show(axc, F[y0:y0 + crop_size[0], x0:x0 + crop_size[1]], f"crop {i + 1} (1:1), y{y0} x{x0}")
    path = path or os.path.join(scene_dir, "compare_average.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def default_real_frame(burst_dir: str = DEFAULT_REAL_FRAME_BURST) -> str | None:
    fs = sorted(glob.glob(os.path.join(burst_dir, "frame_*.tif")))
    return fs[len(fs) // 2] if fs else None


def figure_frame(scene_dir: str, real_frame: str | None = None, path: str | None = None, crop=(0.5, 0.5),
                 crop_size=(300, 480), dpi=100) -> str:
    """The single frame: synthetic frame and a real raw frame (full, and a
    1:1 crop), plus the synthetic average's crop at the same place (the same
    anatomy averaged over the burst: fuller capillaries, 20x less noise)."""
    plt = _plt()
    real_frame = real_frame or default_real_frame()
    fr = _read(os.path.join(scene_dir, "still_frame.tif"))
    av = _read(os.path.join(scene_dir, "still_average.tif"))
    cols = [("synthetic single frame", fr, True), ("synthetic average, same anatomy", av, False)]
    if real_frame and os.path.exists(real_frame):
        cols.append(("real raw frame " + os.path.basename(os.path.dirname(real_frame)).replace("burst_2026-", "")
                     + "/" + os.path.basename(real_frame), _read(real_frame), True))
    n = len(cols)
    fig = plt.figure(figsize=(6.2 * n, 4.1 + 3.1))
    gs = fig.add_gridspec(2, n, height_ratios=[1200 / 1920, crop_size[0] / crop_size[1]], hspace=0.12, wspace=0.03)
    for j, (name, I, is_frame) in enumerate(cols):
        F, _ = _F(I, frame=is_frame)
        _show(fig.add_subplot(gs[0, j]), F, f"{name}\n{F.shape[0]}x{F.shape[1]} px, F on {VMIN}..{VMAX:+} Np")
        (y0, x0), = _crop_boxes(F.shape, [crop], crop_size)
        _show(fig.add_subplot(gs[1, j]), F[y0:y0 + crop_size[0], x0:x0 + crop_size[1]],
              f"crop (1:1), y{y0} x{x0}")
    path = path or os.path.join(scene_dir, "compare_frame.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def load_scene(folder: str) -> SC.Scene:
    """A saved scene (save_scene) back as a Scene: graph, params, images,
    labels, crossings, visibility, junctions (junctions_all, observable,
    vis_profiles where saved).  The fills and form_image extras are not
    saved."""
    import tifffile
    with open(os.path.join(folder, "scene.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    g = VesselGraph.load(os.path.join(folder, "truth.json"))
    sc = SC.Scene(graph=g, params=params)
    for kind in SC.KINDS:
        p = os.path.join(folder, f"still_{kind}.tif")
        if os.path.exists(p):
            sc.images[kind] = tifffile.imread(p).astype(np.float32)
    p = os.path.join(folder, "labels.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            sc.labels = {k: z[k] for k in z.files}
    p = os.path.join(folder, "crossings.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            sc.crossings = json.load(fh)
    for kind in SC.KINDS:                                   # the image-junction truth (v2)
        p = os.path.join(folder, f"junctions_{kind}.json")
        if os.path.exists(p):
            from . import junctions as J
            sc.junctions[kind] = J.load_junctions(p)
        p = os.path.join(folder, f"junctions_all_{kind}.json")
        if os.path.exists(p):
            from . import junctions as J
            sc.junctions_all[kind] = J.load_junctions(p)
        from . import truth as TR                           # the observable and complete truth
        p = os.path.join(folder, f"observable_{kind}.json")
        if os.path.exists(p):
            sc.observable[kind] = TR.load_observable(p)
        p = os.path.join(folder, f"complete_visibility_{kind}.npz")
        if os.path.exists(p):
            sc.vis_profiles[kind] = TR.load_profiles(p)
    p = os.path.join(folder, "vessels.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                vid = int(row["vid"])
                for kind in SC.KINDS:
                    if f"tier_{kind}" in row and row[f"tier_{kind}"]:
                        sc.visibility.setdefault(vid, {})[kind] = dict(
                            tier=row[f"tier_{kind}"], max_cnr=float(row[f"max_cnr_{kind}"] or 0),
                            visible_px=float(row[f"visible_px_{kind}"] or 0),
                            visible_run_px=float(row.get(f"visible_run_px_{kind}") or 0),
                            frame_px=float(row[f"frame_px_{kind}"] or 0))
    return sc


def figure_truth(scene_dir: str, path: str | None = None, crop=None, scale: float = 2.0, dpi=100) -> str:
    """The truth on a zoomed crop (default: the 300 x 480 px around the frame
    centre, enlarged 2x): the plain average still, and the same with every
    vessel coloured by class, the crossings marked with the vessel on top."""
    import cv2
    plt = _plt()
    sc = load_scene(scene_dir)
    H, W = sc.params["shape"]
    crop = crop or (H // 2 - 150, W // 2 - 240, 300, 480)
    y0, x0, h, w = crop
    kind = "average" if "average" in sc.images else next(iter(sc.images))
    plain = SC.flat_view(sc.images[kind])[y0:y0 + h, x0:x0 + w]
    plain = cv2.resize(plain, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ov = SC.draw_overlay(sc, None, kind=kind, by="class", crop=crop, scale=scale)
    ovi = SC.draw_overlay(sc, None, kind=kind, by="id", crop=crop, scale=scale)
    fig, axs = plt.subplots(1, 3, figsize=(3 * 6.4, 4.4))
    for ax, im, t in zip(axs, (cv2.cvtColor(plain, cv2.COLOR_GRAY2RGB), ov[..., ::-1], ovi[..., ::-1]),
                         (f"synthetic {kind}, crop y{y0} x{x0} {h}x{w} px (x{scale:g})",
                          "truth by class (ring = crossing, white-edged stroke = vessel on top)",
                          "truth by vessel id (one colour per digraph edge)")):
        ax.imshow(im)
        ax.set_title(t, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    path = path or os.path.join(scene_dir, "compare_truth.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
