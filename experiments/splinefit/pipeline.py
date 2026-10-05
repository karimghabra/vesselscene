"""The spline-fit pipeline: annotate(image, valid) -> the fitted network, its render and its target.

1. The neuromimetic annotator (experiments.neuromimetic.neuromimetic.run with a debug dict) PROPOSES: the
   cleaned target OD = B - log I (stage 1, background fitted on the vesselness mask's negative), traced
   vessels cut at typed junctions, OD profiles along them, and the texture variance map of its stage 8.
2. proposals.build_network turns the proposal into a spline network (nodes at forks / compounds / free ends,
   crossings pass through without a node). The coarse channel (coarse.py; Config.coarse) adds free DEEP
   edges read out from a target whose background also excludes coarse ridges; that coarse mask is used
   again by the retarget (3), so the final fit sees the deep vessels' OD. Joint fit and MDL stay on stage 1's
   target (a cleaner target early drags the centrelines: E1).
3. fit.fit_network fits the junction-aware render (render.JunctionModel) jointly to the whole image's OD,
   precision-weighted, MDL-prunes, re-cleans the target with the render's support (retarget) and refits.
4. The result is exported in the scorer's contract (experiments/splinefit/score.py): every edge cut at the
   nodes it passes through (one polyline per edge, junction to junction), nodes with a kind (a degree >= 3
   node keeps the proposal's junction type when it had one, else 'branch' / 'compound' by degree; 'end',
   'border', 'joint' otherwise), the crossings found geometrically in the fitted render (two edges'
   centrelines within 1 px away from a shared node), od_render (the fitted vessel OD, halo included, no
   background) and od_target (the OD the final stage explained: the retargeted OD, plus the smooth background
   correction when fit_background is on).

Only image and valid are read. Any H, W (a crop) works. Config exposes every iteration count (Config.fit).
fit_from(image, valid, init) runs the same fit from a given contract network (the evaluator's oracle-init
hook); annotate_additive is the lesion with vesselmap's plain additive render.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import set_threads, use_vesselmap
from .fit import FitConfig, fit_network, precision
from .proposals import build_network

use_vesselmap()


@dataclass(frozen=True)
class Config:
    fit: FitConfig = field(default_factory=FitConfig)
    nm_verify: bool = True          # use the neuromimetic stage-8 (pruned) graph as the proposal
    coarse: bool = True             # coarse channel: ridges out of B, deep edges proposed (coarse.py)
    step: float = 0.5               # px, sample spacing of the exported edges


DEFAULT = Config()


def _propose(image, valid, cfg: Config):
    from experiments.neuromimetic import neuromimetic as N
    dbg = {}
    N.run(image, valid, replace(N.DEFAULT, verify=cfg.nm_verify), debug=dbg)
    s1 = dbg["s1"]
    sig2 = dbg.get("sig2")
    return dbg, s1, sig2


def export(model, net, node_type: dict, step: float = 0.5) -> dict:
    """The contract network of a fitted model / network (module docstring, 4)."""
    seg = net.to_segments()
    deg = seg.degrees()
    H, W = net.shape
    nodes = []
    for nid in sorted(seg.nodes):
        d = deg[nid]
        if d >= 3:
            t = node_type.get(nid)
            typed = t in ("pseudo-T", "compound", "bifurcation", "confluence") and not (d >= 4 and t == "pseudo-T")
            kind = t if typed else ("branch" if d == 3 else "compound")
        elif d == 2:
            kind = "joint"
        else:
            kind = seg.node_kind(nid, deg=1)
            kind = "end" if kind == "endpoint" else kind
        n = seg.nodes[nid]
        nodes.append(dict(id=int(nid), x=float(n.x), y=float(n.y), kind=kind))
    edges, by_vessel = [], {}
    for eid in sorted(seg.edges):
        e = seg.edges[eid]
        smp = seg.sample(eid, step)
        edges.append(dict(id=int(eid), u=int(e.u), v=int(e.v), xy=smp["xy"], r=smp["r"], s=smp["s"], a=smp["a"]))
        by_vessel.setdefault(int(e.info.get("vessel", eid)), []).append(len(edges) - 1)
    crossings = []
    for s in getattr(model, "site_info", []):
        if s["kind"] != "cross" or len(s["edges"]) != 2:
            continue
        p = np.array([s["x"], s["y"]])
        ids = []
        for v in s["edges"]:
            cands = by_vessel.get(int(v), [])
            if not cands:
                break
            dmin = [float(np.min(np.hypot(*(edges[i]["xy"] - p).T))) for i in cands]
            ids.append(edges[cands[int(np.argmin(dmin))]]["id"])
        if len(ids) == 2:
            crossings.append(dict(x=float(p[0]), y=float(p[1]), e1=int(ids[0]), e2=int(ids[1])))
    junctions = [(n["x"], n["y"], n["kind"]) for n in nodes if deg[n["id"]] >= 3]
    junctions += [(c["x"], c["y"], "crossing") for c in crossings]
    return dict(nodes=nodes, edges=edges, crossings=crossings), [ed["xy"] for ed in edges], junctions


def annotate_cfg(image: np.ndarray, valid: np.ndarray, cfg: Config = DEFAULT, debug: dict | None = None) -> dict:
    set_threads()
    t0 = time.time()
    image = np.asarray(image, np.float32)
    valid = np.asarray(valid, bool)
    dbg, s1, sig2 = _propose(image, valid, cfg)
    s1c = s1                        # the retarget's stage 1 (the coarse ridges in its mask)
    if cfg.coarse:
        from .coarse import coarse_stage1
        s1c = coarse_stage1(s1)
    OD = s1["OD"].astype(np.float32)
    w = precision(OD, s1["ok"], s1["bg"], sig2)
    t1 = time.time()
    net, info = build_network(dbg, OD.shape)
    if cfg.coarse:
        from .coarse import deep_edges
        deep_edges(net, s1c)
    log = []
    if net.edges:
        model, net, OD = fit_network(net, OD, w, cfg.fit, log, s1=s1c)
    else:
        model = None
    t2 = time.time()
    if model is not None and model.eids:
        with torch.no_grad():
            R = model.optical_density().numpy().astype(np.float32)
            bg = model.background().numpy().astype(np.float32)
        network, polylines, junctions = export(model, net, info["node_type"], cfg.step)
        hw, hs = (float(v.detach()) for v in model.halo())
        kappa = float(model.kappa().detach())
    else:
        R, bg = np.zeros_like(OD), np.zeros_like(OD)
        network, polylines, junctions = dict(nodes=[], edges=[], crossings=[]), [], []
        hw, hs, kappa = float("nan"), float("nan"), float("nan")
    out = dict(polylines=polylines, junctions=junctions, network=network, od_render=R,
               od_target=(OD + bg).astype(np.float32), halo=(hw, hs), kappa=kappa,
               propose_seconds=t1 - t0, fit_seconds=t2 - t1, build_seconds=time.time() - t0)
    if debug is not None:
        debug.update(proposal=dbg, net=net, model=model, log=log, weight=w, od_stage1=OD, bg=bg, info=info)
    return out


def network_from_contract(network: dict, shape):
    """A contract network (score.py) as a vesselmap VesselNetwork for the fitter (edges refitted on their
    spline bases from the 0.5 px samples) and its node kinds by new node id."""
    from vesselmap.network import VesselNetwork
    net = VesselNetwork(shape)
    ids, kinds = {}, {}
    for n in sorted(network.get("nodes", []), key=lambda n: n["id"]):
        ids[n["id"]] = net.add_node(float(n["x"]), float(n["y"]))
        kinds[ids[n["id"]]] = n.get("kind")
    for e in network.get("edges", []):
        xy = np.asarray(e["xy"], float).reshape(-1, 2)
        if len(xy) < 2 or float(np.hypot(*np.diff(xy, axis=0).T).sum()) < 1.0:
            continue
        r, s_, a = (np.asarray(e[k], float).reshape(-1) for k in ("r", "s", "a"))
        wd = float(np.median(r) + np.median(s_))
        net.add_edge_dense(xy, r, s_, a, u=ids.get(e["u"]), v=ids.get(e["v"]), info=dict(band=[0.0, max(1.5, wd)]))
    for n in [n for n in net.nodes if not net.incident(n)]:
        del net.nodes[n]
    net._touch()
    return net, kinds


def fit_from(image: np.ndarray, valid: np.ndarray, init: dict, cfg: Config = DEFAULT) -> dict:
    """The fit (stage 1 target, precision, fit_network, export) started from a given contract network instead
    of the neuromimetic proposal: the evaluator's ORACLE-INIT hook (init = the true network; positive
    control). The target and weights still come from the image alone."""
    from experiments.neuromimetic import neuromimetic as N
    set_threads()
    t0 = time.time()
    s1 = N.photoreceptors(np.asarray(image, np.float32), np.asarray(valid, bool), N.DEFAULT)
    OD = s1["OD"].astype(np.float32)
    w = precision(OD, s1["ok"], s1["bg"])
    net, kinds = network_from_contract(init, OD.shape)
    log = []
    model, net, OD = fit_network(net, OD, w, cfg.fit, log, s1=s1)
    with torch.no_grad():
        R = model.optical_density().numpy().astype(np.float32)
        bg = model.background().numpy().astype(np.float32)
    network, polylines, junctions = export(model, net, kinds, cfg.step)
    hw, hs = (float(v.detach()) for v in model.halo())
    return dict(polylines=polylines, junctions=junctions, network=network, od_render=R,
                od_target=(OD + bg).astype(np.float32), halo=(hw, hs), kappa=float(model.kappa().detach()),
                build_seconds=time.time() - t0, fit_log=log)


def annotate(image, valid):
    """The spline-fit pipeline with the default Config (junction-aware render)."""
    return annotate_cfg(image, valid, DEFAULT)


def annotate_additive(image, valid):
    """Lesion: the same pipeline with vesselmap's plain additive render (no union at nodes, kappa = 1)."""
    return annotate_cfg(image, valid, replace(DEFAULT, fit=replace(DEFAULT.fit, junctions=False)))
