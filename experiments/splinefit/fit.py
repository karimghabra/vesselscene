"""Joint optimisation of the whole spline network against the image's own optical density, and MDL pruning.

Predictive coding in one loop: the render (render.JunctionModel) is the top-down prediction, the residual
against the target is the prediction error, and every parameter (all centreline control points, the shared
node positions, the r / s / a profile splines, the smooth background correction, the halo and the crossing
additivity kappa) descends the same precision-weighted error at once, so neighbouring vessels explain the
image away from each other instead of each being fitted alone.

Target (the residual rule): OD = B - log I of neuromimetic stage 1, B fitted only on the vesselness mask's
NEGATIVE (never a vesselness or orientation map, never a mask as target). The model is vesselmap's log-domain
form with logI := -OD, so pred = bg - OD_render and the residual is (OD + bg) - OD_render; bg, a smooth
regularised correction grid, is frozen at zero by default (fit_background: fitted jointly with the vessels it
is not a background estimated on the mask's negative, and on dev it traded centreline accuracy for
explained OD). Stage 1's mask is a CNR threshold, so faint and wide lumens leak into its B (an oracle-init
fit on healthy_s004 lost 31 % of contrast and under-predicted the junctions by 28 % against the oracle
target). retarget therefore re-fits B on the negative of a better mask, stage 1's OR the support of the
fitted render (the model's own prediction of where blood is), and the final stage fits that target.

Precision weighting: w = 1 / sigma^2 with sigma the local RMS of the high-passed OD over the background pixels
(neuromimetic refine's sig2: texture included, which dominates the pixel noise of an average still by ~20x);
invalid pixels weigh 0. Texture is spatially correlated over about corr_px pixels, so the NLL gains of the MDL
test are divided by corr_px (one independent observation per correlation area).

Schedule (fixed iteration counts, FitConfig): profiles + optics with the geometry frozen, then everything
jointly (anchored to where the stage started, vesselmap's tracking prior), then MDL pruning (an edge must
explain more NLL than its description costs: vesselmap's n_params x log(pixels) x mdl_scale / 2; the larger of
the gains with and without the node-site cores, where the additive gain approximation of a union is wrong),
topology clean-up, retarget, and a final fit of the profiles and optics with the geometry frozen (a
re-cleaned target exposes OD the network does not explain, e.g. missed vessels, and free centrelines slide
into it). The optimiser loop is adapted from LIMBUS vesselmap.fit.optimize (same author), with the kappa
parameter group added.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import use_vesselmap
from .render import JunctionModel, KAPPA0

use_vesselmap()
from vesselmap.fit import n_params                          # noqa: E402
from vesselmap.network import VesselNetwork                 # noqa: E402


@dataclass(frozen=True)
class FitConfig:
    iters_prof: int = 30            # profiles, background, optics; geometry frozen
    iters_joint: int = 100          # everything, positions anchored
    iters_final: int = 60           # everything after pruning
    rebuild_every: int = 25         # pixel association / junction sites refreshed
    prune_rounds: int = 1
    lr_pos: float = 0.15
    lr_prof: float = 0.04
    lr_bg: float = 0.002
    lr_kappa: float = 0.02
    lam_bend: float = 600.0
    lam_prof: float = 20.0
    lam_bg: float = 2e4
    lam_calibre: float = 300.0
    anchor_px: float = 4.0          # positional prior of the anchored stages (std, px)
    anchor_logprof: float = 1e3     # ... and of log r, log s (std; 1e3 = free)
    bg_spacing: float = 64.0
    fit_background: bool = False    # True: a smooth bg correction fitted jointly (off: B only from the mask's
                                    # negative)
    halo0: tuple = (0.25, 6.0)      # initial halo weight, sigma (px); fitted
    kappa0: float = KAPPA0
    junctions: bool = True          # False: vesselmap's plain additive render (the lesion)
    corr_px: float = 6.0            # px^2, correlation area of the OD texture
    mdl_scale: float = 0.25         # x vesselmap's BIC cost (the proposals were already MDL-vetted upstream)
    min_gain_per_px: float = 0.2    # after the corr_px scaling
    wide_test_w: float = 8.0        # px, r + s from which an edge must also beat a smooth background
    wide_bg_sigma: float = 25.0     # px, the smooth background's scale in that test
    prune_free_only: bool = False   # True: only edges with a free end may be pruned
    fit_pos: bool = True            # False: geometry stays at the proposal (profiles, optics only)
    retarget: bool = True           # re-clean the target with the render's support before the final fit
    final_pos: bool = False         # the final fit (on the re-cleaned target) moves no centreline
    bg_thr: float = 0.5             # x local noise sigma: render support
    bg_dil: int = 3                 # px
    bg_sigma: float = 16.0          # px, masked normalised convolution of the re-fitted background


def precision(OD: np.ndarray, ok: np.ndarray, bgmask: np.ndarray, sig2: np.ndarray | None = None) -> np.ndarray:
    """Precision weights 1 / sigma^2 (0 off valid data). sig2: neuromimetic refine's local variance map."""
    if sig2 is None:
        from experiments.neuromimetic.neuromimetic import _gblur, _local_rms
        hp = OD - _gblur(OD, 8.0)
        sig2 = np.maximum(_local_rms(hp, bgmask, 32.0), 1e-4) ** 2
    w = np.where(ok, 1.0 / np.maximum(sig2, 1e-8), 0.0)
    return w.astype(np.float32)


def _zero_grid(shape, spacing):
    H, W = shape
    return np.zeros((int(math.ceil((H - 1) / spacing)) + 1, int(math.ceil((W - 1) / spacing)) + 1), np.float32)


def make_model(net: VesselNetwork, OD: np.ndarray, w: np.ndarray, cfg: FitConfig, kappa: float | None = None,
               anchored: bool = False) -> JunctionModel:
    if net.background is None:
        net.background, net.bg_spacing = _zero_grid(OD.shape, cfg.bg_spacing), cfg.bg_spacing
    net.meta.setdefault("optics", dict(halo_weight=cfg.halo0[0], halo_sigma=cfg.halo0[1]))
    return JunctionModel(net, -OD.astype(np.float32), w, stride=1, bg_spacing=cfg.bg_spacing,
                         ref=net if anchored else None, fit_background=cfg.fit_background,
                         junctions=cfg.junctions, kappa=cfg.kappa0 if kappa is None else kappa)


def priors(cfg: FitConfig) -> dict:
    return dict(lam_bend=cfg.lam_bend, lam_prof=cfg.lam_prof, lam_bg=cfg.lam_bg, lam_calibre=cfg.lam_calibre)


def optimize(model: JunctionModel, iters: int, cfg: FitConfig, fit_pos=True, anchored=False, log=None):
    """Adam with a cosine-decayed learning rate (vesselmap.fit.optimize), fixed iteration count."""
    if iters <= 0 or not model.eids:
        return []
    groups = [dict(params=[model.raw_r, model.raw_s, model.raw_a], lr=cfg.lr_prof),
              dict(params=[model.raw_hw, model.raw_hs], lr=cfg.lr_prof * 0.5)]
    if fit_pos:
        groups.append(dict(params=[model.node_xy, model.inner], lr=cfg.lr_pos))
    if model.bg.requires_grad:
        groups.append(dict(params=[model.bg], lr=cfg.lr_bg))
    if model.raw_kappa.requires_grad and model.junctions:
        groups.append(dict(params=[model.raw_kappa], lr=cfg.lr_kappa))
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    track = dict(sigma_pos=cfg.anchor_px, sigma_logprof=cfg.anchor_logprof, sigma_amp=1e3) if anchored else None
    hist = []
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters)) * 0.9 + 0.1
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        opt.zero_grad(set_to_none=True)
        loss, nll, _, _ = model.loss(track=track, priors=priors(cfg))
        loss.backward()
        if not fit_pos:
            model.node_xy.grad = None
            model.inner.grad = None
        opt.step()
        model.project()
        hist.append((float(loss.detach()), float(nll.detach())))
        if cfg.rebuild_every and (it + 1) % cfg.rebuild_every == 0 and it + 1 < iters:
            model.rebuild()
    model.rebuild()
    if log is not None:
        log.append(dict(iters=iters, fit_pos=fit_pos, loss0=hist[0][0], loss1=hist[-1][0], nll0=hist[0][1],
                        nll1=hist[-1][1]))
    return hist


@torch.no_grad()
def site_mask(model: JunctionModel, kinds=("node",)) -> np.ndarray:
    """Pixels in the core of the model's junction sites of the given kinds: discs of radius r_max + 2 s_max
    (the members' widest lumen and blur there) around each node of the site, where the lumens overlap."""
    H, W = model.H, model.W
    m = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in getattr(model, "site_info", []):
        if s["kind"] not in kinds:
            continue
        r = s["core"]
        for cx, cy in s["centres"]:
            x0, x1 = max(0, int(cx - r - 1)), min(W, int(cx + r + 2))
            y0, y1 = max(0, int(cy - r - 1)), min(H, int(cy + r + 2))
            m[y0:y1, x0:x1] |= (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= r * r
    return m


@torch.no_grad()
def edge_gains(model: JunctionModel, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """NLL decrease each edge is responsible for (vesselmap NetworkModel._entry_gains: removing the edge's
    entries, halo included, all else fixed), with the residual of the full junction-aware prediction and
    the pixels in `exclude` left out. Returns (gains, pixel counts) per model edge."""
    entries = model.vessel_entries()
    pred = model.predict()
    g = model._entry_gains(entries, pred)
    keep = torch.ones_like(g, dtype=torch.bool)
    if exclude is not None:
        keep = ~torch.as_tensor(exclude.reshape(-1))[model.e_pix]
    ne = len(model.eids)
    gains = torch.zeros(ne).index_add(0, model.e_edge[keep], g[keep])
    cnt = torch.zeros(ne).index_add(0, model.e_edge, torch.ones_like(g))
    return gains.numpy().astype(float), cnt.numpy().astype(float)


def clean_topology(net: VesselNetwork):
    """After removals: nodes no edge ends at but edges still pass through are dropped from those edges'
    through lists (a branch-less through node is no junction); unused nodes deleted; smooth degree-2 joints
    fused (vesselmap merge_joints)."""
    for n in list(net.nodes):
        if not net.incident(n):
            for eid in net.passing(n):
                e = net.edges[eid]
                e.info["through"] = [m for m in e.through if m != n]
                if not e.info["through"]:
                    e.info.pop("through")
            net._touch()
            if not net.passing(n):
                del net.nodes[n]
                net._touch()
    # a through node with exactly one branch and no other passing edge stays (a T on the vessel)
    net.merge_joints()


def prune(net: VesselNetwork, model: JunctionModel, cfg: FitConfig, log=None) -> list:
    """MDL: remove edges whose gain (per corr_px) does not pay for their description. Returns removed ids."""
    if not model.eids:
        return []
    # two estimates: without the node-site cores (where removing an arm's additive entries is not what the
    # union render does) and with them (a short wide edge may lie entirely inside a core); an edge is kept
    # when either pays for it
    g_out, counts = edge_gains(model, exclude=site_mask(model, ("node",)))
    g_all, _ = edge_gains(model)
    gains = np.maximum(g_out, g_all)
    # wide edges must also beat the background (vesselmap score_and_prune): their gain re-evaluated with the
    # low-frequency part of their contribution given to a smooth background, so a broad dark lump of
    # background texture is not kept as a wide, faint 'vessel'
    wide = [k for k, eid in enumerate(model.eids) if eid in net.edges and
            float(np.mean(net.edges[eid].r) + np.mean(net.edges[eid].s)) >= cfg.wide_test_w]
    if wide:
        gp = model.edge_gains_bg_orthogonal(wide, sigma_bg=cfg.wide_bg_sigma)
        for k in wide:
            gains[k] = min(gains[k], gp[k])
    gains = gains / cfg.corr_px
    deg = net.degrees()
    removed = []
    for k, eid in enumerate(model.eids):
        if eid not in net.edges:
            continue
        e = net.edges[eid]
        L = max(net.length(eid), 1.0)
        pen = cfg.mdl_scale * 0.5 * n_params(net, eid, L) * math.log(max(counts[k] / cfg.corr_px, 2.0))
        e.info.update(gain=float(gains[k]), penalty=float(pen))
        free = min(deg.get(e.u, 0), deg.get(e.v, 0)) <= 1
        if cfg.prune_free_only and not free:
            continue
        if gains[k] < pen or gains[k] / L < cfg.min_gain_per_px:
            removed.append(eid)
    for eid in removed:
        net.remove_edge(eid)
    clean_topology(net)
    if log is not None:
        log.append(dict(prune=len(removed), left=len(net.edges)))
    return removed


def retarget(s1: dict, R: np.ndarray, sig: np.ndarray, cfg: FitConfig) -> np.ndarray:
    """The background re-fitted on the negative of a better vesselness mask: stage 1's mask OR the support of
    the fitted render (R > bg_thr x the local noise sigma, dilated by bg_dil px), B = the masked normalised
    Gaussian mean of log I over the remaining valid pixels (neuromimetic _masked_mean, holes filled from
    coarser scales), OD = B - log I. The render is the top-down prediction of where blood is, so the faint
    and wide lumens that leaked into stage 1's background (its mask is a CNR threshold) are excluded."""
    import cv2
    from experiments.neuromimetic.neuromimetic import _masked_mean
    from scipy import ndimage as ndi
    L, ok = s1["L"], s1["ok"]
    m = R > cfg.bg_thr * sig
    k = 2 * cfg.bg_dil + 1
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    m |= s1["mask"]
    B = _masked_mean(L, ok & ~m, cfg.bg_sigma)
    OD = np.where(ok, B - L, 0.0).astype(np.float32)
    if not ok.all():
        iy, ix = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        OD = OD[iy, ix]
    return OD


def fit_network(net: VesselNetwork, OD: np.ndarray, w: np.ndarray, cfg: FitConfig = FitConfig(), log=None,
                s1: dict | None = None):
    """The whole schedule (module docstring). With s1 (neuromimetic stage 1) and cfg.retarget, the target is
    re-cleaned from the render before the final fit (retarget). Returns (model, net, OD): the final model on
    the fitted network and the target it was fitted to."""
    log = [] if log is None else log
    net.meta["optics"] = dict(halo_weight=cfg.halo0[0], halo_sigma=cfg.halo0[1])
    model = make_model(net, OD, w, cfg, anchored=True)
    optimize(model, cfg.iters_prof, cfg, fit_pos=False, anchored=True, log=log)
    model.write_back()
    kappa = float(model.kappa().detach())
    for rnd in range(cfg.prune_rounds):
        model = make_model(net, OD, w, cfg, kappa, anchored=True)
        optimize(model, cfg.iters_joint, cfg, fit_pos=cfg.fit_pos, anchored=True, log=log)
        model.write_back()
        kappa = float(model.kappa().detach())
        prune(net, model, cfg, log)
    if cfg.retarget and s1 is not None:
        with torch.no_grad():
            R = model.optical_density().numpy()
        sig = np.sqrt(1.0 / np.maximum(w, 1e-12))
        OD = retarget(s1, R, sig, cfg)
        net.background = None                          # the new B replaces the smooth correction
        log.append(dict(retarget=True))
    model = make_model(net, OD, w, cfg, kappa, anchored=True)
    optimize(model, cfg.iters_final, cfg, fit_pos=cfg.fit_pos and cfg.final_pos, anchored=True, log=log)
    model.write_back()
    return model, net, OD
