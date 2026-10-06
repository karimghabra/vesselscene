"""Pipeline variants re-tested under the average-only evaluator (re-baseline R2; LOG.md, R2).

    python -m experiments.splinefit.run_experiment --tier 1 --pipeline experiments.splinefit.r2_variants:annotate_e7

Both swap the proposal stage only (pipeline._propose); the fit is unchanged.  Stage 8 runs with zero prune
rounds, so the graph is assembled and its OD profiles fitted (build_network needs them) but nothing is pruned.
  E7      the stage-7 graph as the proposal, judged by the joint fit's MDL alone (batch 2's E7);
  V1OWN   the same, without surround suppression and the association field, at V1-only's own dev-best
          thresholds (neuromimetic ABLATIONS['v1_only_own'] with rounds=0 in place of verify=False).
"""
from dataclasses import replace

from experiments.neuromimetic import neuromimetic as N
from experiments.splinefit import pipeline as P

NM = dict(E7=replace(N.DEFAULT, rounds=0),
          V1OWN=replace(N.DEFAULT, surround=False, association=False, rounds=0, t_high=N.V1_T_HIGH,
                        t_low=N.V1_T_LOW))


def _with(nm_cfg):
    def propose(image, valid, cfg):
        dbg = {}
        N.run(image, valid, nm_cfg, debug=dbg)
        return dbg, dbg["s1"], dbg.get("sig2")

    def annotate(image, valid):
        orig = P._propose
        P._propose = propose
        try:
            return P.annotate_cfg(image, valid, P.DEFAULT)
        finally:
            P._propose = orig
    return annotate


annotate_e7 = _with(NM["E7"])
annotate_v1own = _with(NM["V1OWN"])
