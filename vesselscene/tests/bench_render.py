"""Benchmark of the closed-form renderer on a stress graph (not a test).

    python -m vesselscene.tests.bench_render [n_vessels] [k]

The stress graph is NOT anatomy: random trees of tortuous vessels (random
curvature processes, some tight coils), radii from capillaries to large
dilated vessels, depths 0-100 d_c, forks with short links between them, on a
1920 x 1200 frame.  It exists to load the renderer the way a dense scene
would: many pieces, many junctions, many crossings."""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from vesselscene.graph import VesselGraph
from vesselscene.render import Optics, build_plan, evaluate_plan, render_od


def _path(rng, p0, heading, length, kappa_amp, step=0.25):
    """Tortuous path (d_c): heading driven by smoothed random curvature."""
    n = max(4, int(length / step))
    raw = rng.normal(0, 1, n + 40)
    ker = np.exp(-0.5 * (np.arange(-20, 21) / 6.0) ** 2)
    kap = np.convolve(raw, ker / np.sqrt((ker ** 2).sum()), "same")[20:20 + n] * kappa_amp
    h = heading + np.cumsum(kap) * step
    d = step * np.stack([np.cos(h), np.sin(h)], 1)
    return np.vstack([p0, p0 + np.cumsum(d, 0)]), h[-1]


def stress_graph(n_vessels=1500, k=3.0, W=1920, H=1200, seed=0) -> VesselGraph:
    rng = np.random.default_rng(seed)
    fw, fh = W / k, H / k
    g = VesselGraph(meta=dict(frame=(0, 0, fw, fh), k=k))
    while len(g.vessels) < n_vessels:
        # a tree: root radius log-uniform 0.4 .. 8 d_c, depth 0 .. 100 d_c
        r0 = float(np.exp(rng.uniform(np.log(0.4), np.log(8.0))))
        z = float(rng.uniform(0, 100))
        p = np.array([rng.uniform(-0.1, 1.1) * fw, rng.uniform(-0.1, 1.1) * fh])
        root = g.add_node(p, depth=z, kind="inlet", pressure=1.0)
        stack = [(root, p, rng.uniform(0, 2 * np.pi), r0, 0)]
        while stack and len(g.vessels) < n_vessels:
            nid, p, hd, r, gen = stack.pop()
            length = float(np.clip(rng.lognormal(np.log(12 * max(r, 0.5) + 10), 0.6), 3.0, 400.0))
            if rng.random() < 0.15:
                length = float(rng.uniform(2.0, 6.0)) * max(r, 0.5) + 2.0   # a short link: forks close together
            amp = 0.35 / max(r, 0.5) ** 0.7 * (3.0 if rng.random() < 0.1 else 1.0)
            path, hend = _path(rng, p, hd, length, amp)
            fork = gen < 6 and rng.random() < 0.7 and r > 0.35
            end = g.add_node(path[-1], depth=z, kind="fork" if fork else "outlet")
            g.add_vessel(nid, end, path=path, radius=r, depth=z, hct=float(rng.uniform(0.5, 1.0)))
            if fork:
                for side in (-1, 1):
                    rc = r * float(rng.uniform(0.6, 0.85))
                    stack.append((end, path[-1], hend + side * rng.uniform(0.3, 1.2), rc, gen + 1))
    return g


if __name__ == "__main__":
    import torch
    nv = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    t0 = time.time()
    g = stress_graph(nv, k)
    print(f"stress graph: {len(g.vessels)} vessels, {len(g.nodes)} nodes, built in {time.time() - t0:.1f} s")
    opt = Optics()
    shape = (1200, 1920)
    render_od(g, k, (64, 64), opt)                     # compile the kernels
    for rep in range(2):
        t0 = time.time()
        plan = build_plan(g, k, shape, opt)
        t1 = time.time()
        od = evaluate_plan(plan, shape)
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"plan (CPU) {t1 - t0:.2f} s   evaluate (GPU) {t2 - t1:.2f} s   {plan.stats}")
    t0 = time.time()
    full = render_od(g, k, shape, opt)
    print(f"render_od (with halo pad) {time.time() - t0:.2f} s; OD max {full.max():.3f}, "
          f"median of OD > 0.005: {np.median(full[full > 0.005]):.3f}")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
