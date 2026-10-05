# vesselnet lives in LIMBUS

vesselnet learns the vessel graph of real averaged slit-lamp frames from vesselscene scenes, then finishes it by
`vesselmap`'s render-and-residual optimisation. Its code, plan and results log live in one place:

**[LIMBUS, branch `claude/vesselnet-plan`, folder `vesselnet/`](https://github.com/karimghabra/limbus/tree/claude/vesselnet-plan/vesselnet)**
([PLAN.md](https://github.com/karimghabra/limbus/blob/claude/vesselnet-plan/vesselnet/PLAN.md),
[RESULTS.md](https://github.com/karimghabra/limbus/blob/claude/vesselnet-plan/vesselnet/RESULTS.md)).

LIMBUS holds `vesselmap`, which the optimisation uses, the stabilization that makes the real averaged frames, and
the reference bursts. This repository provides the synthetic scenes and their truth. Check both out side by side
(PLAN.md §0). Until iteration 0 this folder held a mirror of the plan and the CPU pilot, which is in the git
history of this branch.
