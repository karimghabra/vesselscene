"""Comparison annotator: LIMBUS vesselmap ``build_map`` (the current curvature-based mapper) behind the
harness contract.

vesselmap (``$LIMBUS_DATA/vesselmap`` or ``../limbus/vesselmap``) proposes centrelines from multi-scale
Hessian ridges of the residual log image, band by band from coarse to fine, turns them into B-spline edges,
joins them (gap bridging, snapping free ends onto vessels, straight-through pairs at 4-way nodes joined into
one edge that passes over), fits every spline and the background jointly with a differentiable renderer, and
prunes edges that do not pay for themselves (BIC / MDL).  This module only converts the input and the output:

  * input: the still in DN -> vesselmap's [0, 1] intensity (12-bit: / 4095; NaN stays NaN, so ``prepare``
    marks it and a 3 px margin invalid; saturated glare at 4095 -> 1.0 is masked by ``prepare`` too);
  * polylines: every edge's centreline, sampled every 0.7 px in arclength (``net.sample``);
  * junctions: nodes where at least 3 vessel segments meet (``net.degrees``: an edge passing through a node
    counts two).  3 -> 'pseudo-T' (the most frequent 3-way class of the truth; a still shows no flow
    direction), 4 or more -> 'crossing' when the arms pair into two near-collinear through lines, else
    'compound' (``baseline_hessian.type_from_arms``, the same rule as the Hessian baseline, without its node
    merge).  Like that rule, it scores below the constant majority label on dev (harness typing references);
  * plus ``net.crossings()``: vesselmap does not make crossings nodes (an edge passes over another), so the
    points where two edges overlap without sharing a node are added as 'crossing', but only true X crossings:
    the two edges' tangents there must be transversal (``|cos| < CROSS_COS``), the point must be farther than
    ``CROSS_END_WIDTHS`` vessel widths (and 8 px) from either edge's end nodes (nearer, it is a T or a touch),
    and an edge pair gives at most one crossing (the most transversal; a pair that overlaps in several places
    is a duplicated, parallel or touching pair).  Unfiltered, about two thirds of the points were such
    artefacts (on the dev scenes 26 of 36 points on healthy_s004 average came from overlapping pairs), and the
    permissive junction radii let them match truth junctions of other types.

The harness cuts the polylines at the exported junctions before its edge scores, so an edge that passes over
a crossing counts like two edges cut there (the Hessian baseline's convention).

build_map takes minutes per image, so each network (``VesselNetwork.to_dict``, with the build time and the
MapConfig overrides) is cached under ``_cache/vesselmap/<key>.json``; the key hashes the image bytes, the
MapConfig overrides and the vesselmap sources (``_key``), so a changed config or vesselmap gets its own entry.
Later runs reuse it; the export above runs on the cached network every time, and the output carries
``build_seconds``, ``vesselmap_config`` and ``from_cache`` (the harness copies them into its row; its
``seconds`` of a cached run is the export time).  vesselmap is not bit-reproducible (torch reductions on several
threads), so a cached result is what makes a rerun identical: a run served from the cache reports
``from_cache`` and the harness then records determinism as not tested.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from . import limbus_root

LIMBUS = limbus_root()                  # $LIMBUS_DATA or ../limbus
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "vesselmap")
FULL_SCALE = 4095.0
# MapConfig overrides: none (the defaults), or the README's faster discovery setting (about 40 % less time at
# slightly lower recall) with VESSELMAP_CONFIG=fast
FAST = dict(reps_per_band=2, penalty_scale=3.0)
CONFIG = dict(FAST) if os.environ.get("VESSELMAP_CONFIG") == "fast" else {}
ARM_LOOK_PX = 8.0           # arm direction from the edge's last 8 px at the node
CROSS_DEDUP_PX = 6.0        # a crossing point this close to a listed junction is the same junction
CROSS_COS = 0.7             # an X crossing: the two tangents within ~45 deg of perpendicular ...
CROSS_END_WIDTHS = 1.5      # ... and the point this many vessel widths (and 8 px) from either edge's end nodes
_VERSION = []


def vesselmap_version() -> str:
    """sha256 of the vesselmap package sources (the clone is read-only, but a changed checkout re-keys)."""
    if not _VERSION:
        h = hashlib.sha256()
        src = os.path.join(LIMBUS, "vesselmap")
        for name in sorted(os.listdir(src)):
            if name.endswith(".py"):
                with open(os.path.join(src, name), "rb") as fh:
                    h.update(name.encode() + fh.read())
        _VERSION.append(h.hexdigest()[:16])
    return _VERSION[0]


def _key(image: np.ndarray, config: dict | None = None) -> str:
    """Cache key: the image bytes, the MapConfig overrides and the vesselmap sources."""
    h = hashlib.sha256(np.ascontiguousarray(image, np.float32).tobytes())
    h.update(json.dumps(CONFIG if config is None else config, sort_keys=True).encode())
    h.update(vesselmap_version().encode())
    return h.hexdigest()[:16]


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
    return dict(seconds=round(time.perf_counter() - t0, 1), config=dict(CONFIG), vesselmap=vesselmap_version(),
                net=net.to_dict())


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
    best = {}                                         # X crossings only, at most one per edge pair
    for c in net.crossings():
        p = np.array([c["x"], c["y"]])
        cos, far = [], True
        for e in c["edges"]:
            smp = net.sample(e, 0.7)
            j = int(np.argmin(np.hypot(*(smp["xy"] - p).T)))
            cos.append(smp["tan"][j])
            reach = max(8.0, CROSS_END_WIDTHS * 2.0 * float(smp["r"][j]))
            far &= all(np.hypot(*(net.nodes[n].xy - p)) > reach for n in (net.edges[e].u, net.edges[e].v))
        a = abs(float(cos[0] @ cos[1]))
        if far and a < CROSS_COS and (c["edges"] not in best or a < best[c["edges"]][0]):
            best[c["edges"]] = (a, float(c["x"]), float(c["y"]))
    for _, x, y in (best[k] for k in sorted(best)):
        if all(np.hypot(x - j[0], y - j[1]) > CROSS_DEDUP_PX for j in junctions):
            junctions.append((x, y, "crossing"))
    return dict(polylines=polylines, junctions=junctions)


def annotate(image: np.ndarray, valid: np.ndarray) -> dict:
    """vesselmap's map of one still, as polylines and typed junctions (the network cached, ``_key``)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(image) + ".json")
    cached = os.path.exists(path)
    if cached:
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
    assert res["config"] == CONFIG and res.get("vesselmap") == vesselmap_version(), (path, res["config"])
    out = export(VesselNetwork.from_dict(res["net"]))
    out.update(build_seconds=res["seconds"], vesselmap_config=res["config"], from_cache=cached)
    return out
