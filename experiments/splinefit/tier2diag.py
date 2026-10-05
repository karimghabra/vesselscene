r"""Where the whole-image residual sits: explained by vessel-width class and per junction disc, and how much of
the true OD the pipeline's own target keeps (a DIAGNOSTIC outside the frozen metric; batch-5 review finding).

The frozen tier-2 'explained' pools every band pixel of a whole image, so on healthy images it sits near its
ceiling (0.99) while a compound knot is under-predicted and a faint wide vessel is not rendered at all; on the
pathologic image the pipeline's target (stage 1's background, retargeted) has already lost about half of the
wide vessels' OD, so the fit cannot explain what its target does not contain. This breaks the pooled numbers
down (truth is read only here, to know where to look):

    width classes   every band pixel is assigned to its nearest true observable sample and classed by the
                    true radius r: thin < 2 px, mid 2-4, wide 4-8, vwide >= 8. Per class: explained (pooled
                    over the class's pixels, against OD_obs, as score.py), the class's share of the band's
                    squared residual, and target_keep = sum(od_target * OD_img) / sum(OD_img^2) (the fraction
                    of the true OD the pipeline's own target keeps: 1 = no background leak) with
                    target_fidelity (score.py's formula) on the class.
    junction discs  explained per observable junction disc (score.py's discs, radius >= 4 px): the median, the
                    10th percentile, the fraction of discs below 0.5, and the pooled value.

Usage. Record whole-image outputs while tier 2 runs (the wrapper returns annotate's output unchanged, so the
digests and the metric are those of the default pipeline):

    DIAG2_RUN=e15 python -m experiments.splinefit.run_experiment --tier 2 --repeat 2 \
        --pipeline experiments.splinefit.tier2diag:annotate_saving
    python -m experiments.splinefit.tier2diag --run e15 [--compare e14]

or run the pipeline on a few dev images directly (`--compute SCENE KIND ...`). Outputs:
$SPLINEFIT_WORK/diag2/<run>/<sha>.npz (the render and target) and <run>/diag.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import hashlib                                      # noqa: E402
import json                                         # noqa: E402

import numpy as np                                  # noqa: E402

WORK = os.environ.get("SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/"
                                        "scratchpad/work/splinefit")
CLASSES = (("thin", 0.0, 2.0), ("mid", 2.0, 4.0), ("wide", 4.0, 8.0), ("vwide", 8.0, np.inf))


def _key(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image, np.float32).tobytes()).hexdigest()[:16]


def _dir(run: str) -> str:
    return os.path.join(WORK, "diag2", run)


def annotate_saving(image, valid):
    """pipeline.annotate, with its render and target saved under diag2/$DIAG2_RUN/ (output unchanged)."""
    from .pipeline import annotate
    out = annotate(image, valid)
    d = _dir(os.environ.get("DIAG2_RUN", "last"))
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, _key(image) + ".npz"), od_render=out["od_render"],
                        od_target=out["od_target"])
    return out


def _explained(t, R, m):
    m = m & np.isfinite(t) & np.isfinite(R)
    den = float((t[m].astype(np.float64) ** 2).sum())
    return 1.0 - float(((t[m] - R[m]).astype(np.float64) ** 2).sum()) / den if den > 0 else float("nan")


def case_diag(case: dict, R: np.ndarray, T: np.ndarray) -> dict:
    """The width-class and junction-disc breakdown of one case (module docstring)."""
    from scipy.spatial import cKDTree
    from . import score as S
    t, img, band = case["od_obs"], case["od_img"], case["band"]
    smp = case["samples"]
    yy, xx = np.nonzero(band)
    cls = np.full(band.shape, -1, int)
    if len(smp["xy"]) and len(yy):
        _, i = cKDTree(smp["xy"]).query(np.stack([xx, yy], 1))
        r = smp["r"][i]
        for k, (_, lo, hi) in enumerate(CLASSES):
            sel = (r >= lo) & (r < hi)
            cls[yy[sel], xx[sel]] = k
    res2 = np.where(band & np.isfinite(t), (t - R) ** 2, 0.0)
    tot = float(res2.sum()) or 1.0
    out = dict(classes={})
    for k, (name, _, _) in enumerate(CLASSES):
        m = cls == k
        n = int(m.sum())
        ii = float((img[m].astype(np.float64) ** 2).sum())
        out["classes"][name] = dict(
            px=n, explained=_explained(t, R, m), sq_share=float(res2[m].sum()) / tot,
            target_keep=float((T[m] * img[m]).astype(np.float64).sum()) / ii if ii > 0 else float("nan"),
            target_fidelity=_explained(img, T, m))
    ex = []
    for x, y, rr in case["discs"]:
        m = S.disc_mask(t.shape, np.array([[x, y, max(rr, 4.0)]])) & band
        if m.any():
            ex.append(_explained(t, R, m))
    ex = np.array([e for e in ex if np.isfinite(e)])
    out["discs"] = dict(n=int(len(ex)), median=float(np.median(ex)) if len(ex) else float("nan"),
                        p10=float(np.percentile(ex, 10)) if len(ex) else float("nan"),
                        frac_below_half=float((ex < 0.5).mean()) if len(ex) else float("nan"),
                        pooled=_explained(t, R, case["jdisc"]))
    out["target_keep"] = float((T[band] * img[band]).astype(np.float64).sum()) / \
        max(float((img[band].astype(np.float64) ** 2).sum()), 1e-12)
    out["target_fidelity"] = _explained(img, T, band)
    out["explained"] = _explained(t, R, band)
    return out


def analyse(run: str, cases=None) -> list:
    from . import score as S
    from .run_experiment import tier_cases
    rows = []
    for folder, kind, crop in (cases or tier_cases(2)):
        case = S.load_truth(folder, kind, crop, reference_target=False)
        f = os.path.join(_dir(run), _key(case["image"]) + ".npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            R, T = z["od_render"].astype(np.float32), z["od_target"].astype(np.float32)
        d = case_diag(case, R, T)
        d.update(scene=os.path.basename(folder), kind=kind)
        rows.append(d)
    with open(os.path.join(_dir(run), "diag.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def table(rows: list, ref: list | None = None) -> str:
    """Markdown: per image, explained / target_keep per width class and the disc stats (ref: deltas)."""
    refd = {(r["scene"], r["kind"]): r for r in (ref or [])}
    hdr = ["image", "explained"] + [f"ex_{c[0]}" for c in CLASSES] + [f"keep_{c[0]}" for c in CLASSES] + \
          ["target_keep", "disc_med", "disc_p10", "disc<0.5"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        q = refd.get((r["scene"], r["kind"]))

        def f(v, v0=None):
            if v is None or not np.isfinite(v):
                return "-"
            return f"{v:.3f}" + (f" ({v - v0:+.3f})" if v0 is not None and np.isfinite(v0) else "")
        vals = [f(r["explained"], q and q["explained"])]
        vals += [f(r["classes"][c[0]]["explained"], q and q["classes"][c[0]]["explained"]) for c in CLASSES]
        vals += [f(r["classes"][c[0]]["target_keep"], q and q["classes"][c[0]]["target_keep"]) for c in CLASSES]
        vals += [f(r["target_keep"], q and q["target_keep"]), f(r["discs"]["median"], q and q["discs"]["median"]),
                 f(r["discs"]["p10"], q and q["discs"]["p10"]),
                 f(r["discs"]["frac_below_half"], q and q["discs"]["frac_below_half"])]
        lines.append(f"| {r['scene']} {r['kind']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="last")
    ap.add_argument("--compare", help="another run's diag.json to print deltas against")
    ap.add_argument("--compute", nargs="*", help="SCENE KIND pairs: run the pipeline on these dev images first")
    a = ap.parse_args(argv)
    from . import fastset as FS
    from . import score as S
    from . import set_threads
    set_threads(2)
    cases = None
    if a.compute:
        os.environ["DIAG2_RUN"] = a.run
        cases = []
        for sc, kind in zip(a.compute[::2], a.compute[1::2]):
            folder = os.path.join(FS.DEV_DIR, sc)
            FS.check_not_heldout(folder)
            case = S.load_truth(folder, kind, None, reference_target=False)
            if not os.path.exists(os.path.join(_dir(a.run), _key(case["image"]) + ".npz")):
                annotate_saving(case["image"].copy(), case["valid"].copy())
            cases.append((folder, kind, None))
    rows = analyse(a.run, cases)
    ref = None
    if a.compare:
        with open(os.path.join(_dir(a.compare), "diag.json"), encoding="utf-8") as fh:
            ref = json.load(fh)
    print(table(rows, ref))


if __name__ == "__main__":
    main()
