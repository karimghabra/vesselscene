"""Unit tests of the frozen scorer (score.py) and the experiment runner on a dev crop.

    python -m pytest experiments/splinefit/tests -q

Known answers: the true observable network scored against itself gives pos = width = 1 and a graph score
near 1; shifted 2 px along its normals, pos ~ 1/3; widths x 1.5, width ~ 0.5; an empty network, graph 0; the
clean image OD as the render explains the oracle target to the noise (chi2 ~ 1-2); the oracle's additive
render of the true network (sham) reproduces vesselmap's, and its junction-aware render lowers the junctions'
over-prediction.  Skipped without the dev
scenes ($SPLINEFIT_DEV).
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import pytest

from experiments.splinefit import fastset as FS
from experiments.splinefit import score as S

SCENE = os.path.join(FS.DEV_DIR, "healthy_s004_320x512")
CROP = [64, 0, 256, 256]
pytestmark = pytest.mark.skipif(not os.path.isdir(SCENE), reason="dev scenes not found")


@pytest.fixture(scope="module")
def case():
    return S.load_truth(SCENE, "average", CROP)


@pytest.fixture(scope="module")
def net(case):
    return S.true_network_observable(case)


def _normal_shift(net, d):
    out = copy.deepcopy(net)
    for e in out["edges"]:
        if len(e["xy"]) < 2:
            continue
        t = np.gradient(e["xy"], axis=0)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        e["xy"] = e["xy"] + d * n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return out


def test_self(case, net):
    s = S.score(case, dict(network=net))
    assert s["pos"] > 0.99 and s["width"] > 0.99
    assert s["graph_score"] > 0.95
    assert s["geom_matched"] > 0.95


def test_shift_2px(case, net):
    g = S.score_geometry(case, _normal_shift(net, 2.0))
    assert abs(g["pos"] - 1.0 / 3.0) < 0.05


def test_width_x15(case, net):
    wide = copy.deepcopy(net)
    for e in wide["edges"]:
        e["r"] = e["r"] * 1.5
    g = S.score_geometry(case, wide)
    assert abs(g["width"] - 0.5) < 0.02


def test_empty(case):
    s = S.score(case, dict(network=dict(nodes=[], edges=[], crossings=[])))
    assert s["graph_score"] == 0.0 and s["pos"] == 0.0 and s["width"] == 0.0 and s["composite"] == 0.0


def test_oracle_target(case):
    r = S.score_render(case, case["od_img"], None)
    assert r["explained"] > 0.99
    assert 0.8 < r["chi2"] < 2.0                  # robust sigma: heavier tails (texture, glare) lift it


def test_crop_truth(case):
    h, w = case["image"].shape
    assert (h, w) == (CROP[3], CROP[2])
    for j in case["obs"]["junctions"]:
        assert S.CROP_BORDER_PX <= j["x"] + 0.5 <= w - S.CROP_BORDER_PX
        assert S.CROP_BORDER_PX <= j["y"] + 0.5 <= h - S.CROP_BORDER_PX
    xy = case["samples"]["xy"]
    assert len(xy) and xy.min() >= -0.5 and xy[:, 0].max() <= w - 0.5 and xy[:, 1].max() <= h - 0.5


def slow_pipeline(image, valid):
    """A pipeline that never finishes in time (the runner's wall-clock limit)."""
    while True:
        time.sleep(0.05)


def test_runner_timeout():
    from experiments.splinefit import run_experiment as RE
    res = RE.run(1, "experiments.splinefit.tests.test_score:slow_pipeline", limit=3.0, verbose=False)
    assert res["status"] == "crash" and "Timeout" in res["error"]


@pytest.fixture(scope="module")
def true_complete(case):
    tn = S.true_network_complete(SCENE, "average", case["image"].shape, CROP)
    vn = tn.pop("vesselnetwork")
    o = vn.meta["optics"]
    tn["halo"] = (o["halo_weight"], o["halo_sigma"])
    return tn, vn


def test_oracle_render_sham(case, true_complete):
    """oracle.render_additive (JunctionModel, junctions off) reproduces vesselmap's render of the truth; the
    junction-aware oracle.render lowers the junctions' over-prediction (a positive junction_bias)."""
    from experiments.splinefit import oracle as O
    tn, vn = true_complete
    ref = S.render_network(vn, case["image"].shape)
    add = O.render_additive(case["image"], case["valid"], dict(tn))
    assert np.abs(add["od_render"] - ref).max() < 0.02 * max(float(ref.max()), 1e-3)
    sa = S.score(case, add)
    sj = S.score(case, O.render(case["image"], case["valid"], dict(tn)))
    assert sj["junction_bias"] < sa["junction_bias"]
    assert sj["explained_junction"] >= sa["explained_junction"] - 1e-3
