"""Comparison annotator: the user's own mapper, LIMBUS vesselmap ``build_map``, behind the harness contract.

vesselmap (``/home/user/karimghabra/limbus/vesselmap``) proposes centrelines from multi-scale Hessian ridges of
the residual log image, band by band from coarse to fine, turns them into B-spline edges, joins them (gap
bridging, snapping free ends onto vessels, straight-through pairs at 4-way nodes joined into one edge that
passes over), fits every spline and the background jointly with a differentiable renderer, and prunes edges
that do not pay for themselves (BIC / MDL).  This module only converts the input and the output:

  * input: the still in DN -> vesselmap's [0, 1] intensity (12-bit: / 4095; NaN stays NaN, so ``prepare``
    marks it and a 3 px margin invalid; saturated glare at 4095 -> 1.0 is masked by ``prepare`` too);
  * polylines: every edge's centreline, sampled every 0.7 px in arclength (``net.sample``);
  * junctions: nodes where at least 3 vessel segments meet (``net.degrees``: an edge passing through a node
    counts two).  3 -> 'pseudo-T' (the most frequent 3-way class of the truth; a still shows no flow
    direction), 4 or more -> 'crossing' when the arms pair into two near-collinear through lines, else
    'compound' (``baseline_hessian.type_from_arms``, the same rule as the Hessian baseline);
  * plus ``net.crossings()``: vesselmap does not make crossings nodes (an edge passes over another), so the
    points where two edges overlap without sharing a node are added as 'crossing'.

build_map takes minutes per image, so each network (``VesselNetwork.to_dict``, with the build time and the
MapConfig overrides) is cached under ``_cache/vesselmap/<sha256 of the image bytes>[:16].json`` (keyed by the
image content only) and later runs reuse it; the export above runs on the cached network every time.  vesselmap is not
bit-reproducible (torch reductions on several threads), so a cached result is what makes a rerun identical;
the harness's determinism check is only meaningful with the cache empty.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

LIMBUS = "/home/user/karimghabra/limbus"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "vesselmap")
FULL_SCALE = 4095.0
# MapConfig overrides: none (the defaults), or the README's faster discovery setting (about 40 % less time at
# slightly lower recall) with VESSELMAP_CONFIG=fast
FAST = dict(reps_per_band=2, penalty_scale=3.0)
CONFIG = dict(FAST) if os.environ.get("VESSELMAP_CONFIG") == "fast" else {}
ARM_LOOK_PX = 8.0           # arm direction from the edge's last 8 px at the node
CROSS_DEDUP_PX = 6.0        # a crossing point this close to a listed junction is the same junction


def _key(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image, np.float32).tobytes()).hexdigest()[:16]


def _limbus():
    if LIMBUS not in sys.path:
        sys.path.insert(0, LIMBUS)


def _run_vesselmap(image: np.ndarray) -> dict:
    """build_map on the still; returns the network (``VesselNetwork.to_dict``) and the build time."""
    _limbus()
    import torch
    torch.set_num_threads(2)
    from vesselmap import build_map
    from vesselmap.fit import MapConfig

    inten = np.clip(np.asarray(image, np.float32) / FULL_SCALE, 0.0, 1.0)     # NaN stays NaN
    t0 = time.perf_counter()
    net = build_map(inten, MapConfig(verbose=True, **CONFIG))
    return dict(seconds=round(time.perf_counter() - t0, 1), config=dict(CONFIG), net=net.to_dict())


def export(net) -> dict:
    """A VesselNetwork as harness polylines and typed junctions (module docstring)."""
    from .baseline_hessian import type_from_arms
    polylines = [net.sample(e, 0.7)["xy"] for e in sorted(net.edges)]
    junctions = []
    deg = net.degrees()
    for nid in sorted(net.nodes):
        if deg[nid] < 3:
            continue
        dirs = [-net.end_tangent(e, end, ARM_LOOK_PX) for e, end in net.incident(nid)]
        for e in net.passing(nid):                    # an edge through the node gives two opposite arms
            smp = net.sample(e, 0.7)
            j = int(np.argmin(np.hypot(*(smp["xy"] - net.nodes[nid].xy).T)))
            dirs += [smp["tan"][j], -smp["tan"][j]]
        n = net.nodes[nid]
        junctions.append((float(n.x), float(n.y), type_from_arms(np.array(dirs))))
    for c in net.crossings():
        if all(np.hypot(c["x"] - j[0], c["y"] - j[1]) > CROSS_DEDUP_PX for j in junctions):
            junctions.append((float(c["x"]), float(c["y"]), "crossing"))
    return dict(polylines=polylines, junctions=junctions)


def annotate(image: np.ndarray, valid: np.ndarray) -> dict:
    """vesselmap's map of one still, as polylines and typed junctions (the network cached by image content)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(image) + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
    else:
        res = _run_vesselmap(image)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh)
        os.replace(tmp, path)
    _limbus()
    from vesselmap.network import VesselNetwork
    out = export(VesselNetwork.from_dict(res["net"]))
    out["build_seconds"] = res["seconds"]
    return out
