"""Known answers of the probe battery's helpers (probes.py): Murray's angles, the transect line count, the
battery's geometry folders, and a probe's score for the null and empty outputs."""
from __future__ import annotations

import math

import numpy as np

from experiments.splinefit import probes as P


def test_murray_symmetric_fork_is_37_5_deg():
    r0 = 2.0
    r1 = r0 * 2 ** (-1 / 3)
    a1, a2 = P.murray_angles(r0, r1, r1)
    assert abs(a1 - a2) < 1e-9 and abs(a1 - math.degrees(math.acos(2 ** (-1 / 3)))) < 1e-9


def test_n_lines_counts_distinct_crossings_of_the_transect():
    a = np.array([[0.0, 60.0], [128.0, 60.0]])
    b = np.array([[0.0, 66.0], [128.0, 66.0]])
    c = np.array([[0.0, 66.5], [128.0, 66.5]])            # within 1 px of b: one line
    assert P.n_lines([a, b, c], (64.0, 64.0), 90.0, 10.0) == 2
    assert P.n_lines([a], (64.0, 64.0), 90.0, 3.0) == 0     # outside the transect's half length


def test_geometries_share_folders_between_kinds():
    geo = P.geometries()
    by = {st["name"]: st["folder"] for st in P.BATTERY}
    assert by["fork_thin_frame"] == by["fork_thin"] and by["empty_frame"] == by["empty_average"]
    assert len(set(by.values())) == len(geo)
    assert all(st["kind"] in ("average", "frame") for st in P.BATTERY)


def test_empty_probe_score_by_fitted_length():
    st = dict(name="empty_average", sweep="empty", x="average", kind="average")
    case = dict(obs=dict(runs=[], junctions=[], dont_care_junctions=[]),
                samples=dict(r=np.zeros(0), s=np.zeros(0), a=np.zeros(0)))
    assert P.score_probe(st, case, P.nothing(np.zeros((8, 8)), np.ones((8, 8), bool)))["probe_composite"] == 1.0
    edge = dict(u=0, v=1, xy=np.array([[0.0, 0.0], [50.0, 0.0]]), r=np.ones(2), s=np.ones(2), a=np.ones(2))
    out = dict(network=dict(nodes=[], edges=[edge], crossings=[]))
    assert abs(P.score_probe(st, case, out)["probe_composite"] - 0.5) < 1e-9
