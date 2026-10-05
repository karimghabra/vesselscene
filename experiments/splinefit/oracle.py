r"""ORACLE diagnostics (truth-initialised; never part of the pipeline): the positive control and the ceiling.

Both functions follow evaluate_fit's oracle hook, ``fn(image, valid, init) -> pipeline output``, with init
the TRUE complete network as a contract dict (score.true_network_complete, plus 'halo' = the scene's
(weight, sigma) when the caller knows it).  They read the truth only through init; the target and the
precision weights still come from the image alone (neuromimetic stage 1, as in the pipeline).

  fit      ORACLE INIT (positive control): pipeline.fit_from, the pipeline's own fit schedule (fit.fit_network:
           profiles, joint fit, MDL prune, retarget, final fit) started from the true network instead of the
           neuromimetic proposal.  What the fit does to a perfect start: a fitter that is right should keep
           pos, width and graph near the truth's while explaining more; whatever it loses is the fit's (or its
           target's) fault, not the proposal's.
  render   ORACLE RENDER (the render model's ceiling, no fitting): the true network rendered by the
           junction-aware renderer (render.JunctionModel: lumens unioned at nodes, crossings added with
           kappa = KAPPA0) with the scene's halo.  Compared with evaluate_fit's true_complete row (the same
           network through vesselmap's plain additive render) it isolates the junction model: junction_bias
           should drop toward 0 and explained_junction rise; what remains below 1 is the chord law, the red
           cells, glare and the image noise, i.e. what no spline render of this family can explain.
  render_additive  the same with junctions=False (sham of `render`: must reproduce true_complete's render up
           to the faithful spline refit).

    python -m experiments.splinefit.oracle SCENE_DIR [--kind average|frame] [--crop x0 y0 w h]

prints the three rows' main scores next to true_complete's (a quick look; evaluate_fit makes the tables).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import time                                         # noqa: E402

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

from . import set_threads, use_vesselmap            # noqa: E402
from .render import KAPPA0, JunctionModel           # noqa: E402

use_vesselmap()


def fit(image: np.ndarray, valid: np.ndarray, init: dict) -> dict:
    """ORACLE INIT: the pipeline's fit started from the true network (module docstring)."""
    from .pipeline import fit_from
    return fit_from(image, valid, init)


def _render_true(image, valid, init: dict, junctions: bool) -> dict:
    from experiments.neuromimetic import neuromimetic as N
    from vesselmap.network import R_MIN, S_MIN
    from vesselmap.render import inv_softplus
    from . import score as S
    set_threads()
    t0 = time.time()
    image = np.asarray(image, np.float32)
    s1 = N.photoreceptors(image, np.asarray(valid, bool), N.DEFAULT)
    OD = s1["OD"].astype(np.float32)
    halo = init.get("halo")
    net = S.to_vesselmap(init, OD.shape, halo)        # faithful spline refit of the true samples
    net.meta.setdefault("optics", dict(halo_weight=0.25, halo_sigma=5.0))
    if not net.edges:
        R = np.zeros_like(OD)
    else:
        gh, gw = (int(np.ceil((n - 1) / 64.0)) + 1 for n in OD.shape)
        net.background, net.bg_spacing = np.zeros((gh, gw), np.float32), 64.0
        with torch.no_grad():
            model = JunctionModel(net, -OD, np.ones_like(OD), stride=1, fit_background=False,
                                  junctions=junctions, kappa=KAPPA0, fit_kappa=False)
            # no width / blur caps on a true network (score.render_network does the same)
            cat = lambda key: np.concatenate([getattr(net.edges[e], key) for e in model.eids])
            model.raw_r_max = torch.full_like(model.raw_r_max, float("inf"))
            model.raw_s_max = torch.full_like(model.raw_s_max, float("inf"))
            model.raw_r.data = torch.tensor(inv_softplus(np.maximum(cat("r"), R_MIN + 1e-3) - R_MIN),
                                            dtype=torch.float32)
            model.raw_s.data = torch.tensor(inv_softplus(np.maximum(cat("s"), S_MIN + 1e-3) - S_MIN),
                                            dtype=torch.float32)
            model.rebuild()
            R = model.optical_density().numpy().astype(np.float32)
    o = net.meta["optics"]
    network = {k: v for k, v in init.items() if k in ("nodes", "edges", "crossings")}
    return dict(network=network, od_render=R, od_target=OD, halo=(float(o["halo_weight"]), float(o["halo_sigma"])),
                kappa=KAPPA0 if junctions else 1.0, build_seconds=time.time() - t0)


def render(image: np.ndarray, valid: np.ndarray, init: dict) -> dict:
    """ORACLE RENDER: the true network through the junction-aware renderer, no fitting (module docstring)."""
    return _render_true(image, valid, init, junctions=True)


def render_additive(image: np.ndarray, valid: np.ndarray, init: dict) -> dict:
    """Sham of render: the same with the plain additive render (junctions=False)."""
    return _render_true(image, valid, init, junctions=False)


def main(argv=None):
    from . import score as S
    from .fastset import check_not_heldout
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene")
    ap.add_argument("--kind", default="average", choices=("average", "frame"))
    ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--rows", nargs="+", default=["render_additive", "render", "fit"])
    a = ap.parse_args(argv)
    check_not_heldout(a.scene)
    case = S.load_truth(a.scene, a.kind, a.crop)
    img, val = case["image"], case["valid"]
    tn = S.true_network_complete(a.scene, a.kind, img.shape, a.crop)
    vn = tn.pop("vesselnetwork")
    o = vn.meta["optics"]
    tn["halo"] = (o["halo_weight"], o["halo_sigma"])
    keys = ("composite", "graph_score", "pos", "width", "explained", "explained_junction", "junction_bias")
    s = S.score(case, dict(network=tn, od_render=S.render_network(vn, img.shape), halo=tn["halo"]))
    print("true_complete  ", " ".join(f"{k}={s[k]:.3f}" for k in keys))
    for name in a.rows:
        t = time.time()
        out = globals()[name](img.copy(), val.copy(), tn)
        s = S.score(case, out)
        print(f"{name:15s}", " ".join(f"{k}={s[k]:.3f}" for k in keys), f"t={time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
