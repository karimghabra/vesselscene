"""The loop's arithmetic (run_experiment.summarise): the tier-1 composite is the mean of the fast-crop
composite and the probe score, and the digest covers the probe battery."""
from __future__ import annotations

from experiments.splinefit import run_experiment as RE


def test_tier1_composite_is_half_crops_half_probes():
    rows = [dict(composite=0.6, graph_score=0.5, pos=0.7, width=0.8, explained=0.9, explained_junction=0.9,
                 digest="a"),
            dict(composite=0.8, graph_score=0.7, pos=0.7, width=0.8, explained=0.9, explained_junction=0.9,
                 digest="b")]
    s = RE.summarise(rows, dict(probe_score=0.9, digest="p"))
    assert abs(s["fast_composite"] - 0.7) < 1e-12 and abs(s["composite"] - 0.8) < 1e-12
    s2 = RE.summarise(rows, None)
    assert abs(s2["composite"] - 0.7) < 1e-12 and "probe_score" not in s2
    assert RE.summarise(rows, dict(probe_score=0.9, digest="q"))["digest"] != s["digest"]
