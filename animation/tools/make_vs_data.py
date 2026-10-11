"""Render the vesselscene imagery the animation uses, into animation/data_cache/.

One healthy scene: the burst average with its observable truth (vessel
polylines, typed junctions, flow direction), and a burst of frames, each with
its own red-cell filling and camera noise.

    python animation/tools/make_vs_data.py --seed 0 --shape 540 960 --frames 16

On the CPU this takes about half an hour at 960 x 540.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import vesselscene.scene as VS

OUT = Path(__file__).resolve().parents[1] / "data_cache"


def _med(x):
    a = np.asarray(x if x is not None else np.nan, float).ravel()
    return float(np.nanmedian(a)) if a.size and np.isfinite(a).any() else float("nan")


def truth_of(sc, kind="average"):
    """The observable truth reduced to what the animation draws."""
    obs = sc.observable[kind]
    g = sc.graph
    runs = []
    for r in obs["runs"]:
        v = g.vessels.get(r["vid"])
        runs.append(dict(vid=int(r["vid"]), cls=str(r["cls"]), xy=np.asarray(r["xy"], float).round(2).tolist(),
                         width_px=_med(r.get("width_px")), contrast=_med(r.get("contrast")),
                         s0_px=float(r["s0_px"]), s1_px=float(r["s1_px"]),
                         flow=None if v is None else dict(u=int(v.u), v=int(v.v), q=float(v.flow))))
    juncs = [dict(id=int(j["id"]), x=float(j["x"]), y=float(j["y"]), radius=float(j["radius"]),
                  type=str(j["type"]), type_observable=str(j["type_observable"]), n_lines=int(j["n_lines"]),
                  lines=[dict(vid=int(l["vid"]), flow=l.get("flow"), angle_deg=float(l["angle_deg"]),
                              width_px=_med(l.get("width_px"))) for l in j["lines"]])
             for j in obs["junctions"]]
    return dict(shape=list(obs["shape"]), runs=runs, junctions=juncs, summary=obs.get("summary"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="healthy")
    ap.add_argument("--shape", type=int, nargs=2, default=(540, 960))
    ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = f"{a.preset}_s{a.seed:03d}_{a.shape[1]}x{a.shape[0]}"
    t = time.time()
    sc = VS.make_scene(a.seed, a.preset, shape=tuple(a.shape), device="cpu", kinds=("average",),
                       log=lambda *m: print(*m, flush=True))
    print(f"average + truth: {time.time() - t:.0f} s", flush=True)
    np.savez_compressed(OUT / f"{tag}_average.npz", image=sc.images["average"], od=sc.od_clean["average"],
                        centreline=sc.labels["centreline"])
    (OUT / f"{tag}_truth.json").write_text(json.dumps(truth_of(sc)))

    # the burst: each frame with its own filling (frame index) and its own camera noise
    filling, form = VS.IM.filling, VS.IM.form_image
    frames = []
    for i in range(a.frames):
        t = time.time()
        VS.IM.filling = lambda g, kind="average", frame=None, burst_seed=None, _i=i, **p: \
            filling(g, kind=kind, frame=_i, burst_seed=burst_seed, **p)
        VS.IM.form_image = lambda od, ip, rng, _i=i, **kw: form(od, ip, np.random.default_rng(10_000 + _i), **kw)
        try:
            fr = VS.make_scene(a.seed, a.preset, shape=tuple(a.shape), device="cpu", kinds=("frame",),
                               graph=sc.graph, labels=False, frame_focus_offset=0.0)
        finally:
            VS.IM.filling, VS.IM.form_image = filling, form
        frames.append(fr.images["frame"].astype(np.float32))
        print(f"frame {i}: {time.time() - t:.0f} s", flush=True)
        tmp = OUT / f"{tag}_burst.tmp.npz"            # written whole, then swapped in: readers never see half a file
        np.savez_compressed(tmp, frames=np.stack(frames))
        os.replace(tmp, OUT / f"{tag}_burst.npz")


if __name__ == "__main__":
    main()
