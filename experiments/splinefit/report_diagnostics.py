r"""Diagnostics behind the render comparison and the junction findings of experiments/REPORT.md (sections 4 and
6), on the held-out averaged stills.  A report tool: it reads the truth and the cached outputs of
report_figures run (never a pipeline path).

    python -m experiments.splinefit.report_diagnostics SCENE_DIR [SCENE_DIR ...] [--cache DIR] [--kind average]

Per image, against OD_img (vesselscene's clean vessel OD, score.formed_od) and OD_obs (the scorer's
oracle-background OD):
  junctions   per observable truth junction, the mean over its disc (score's discs, within the band) of the
              render minus OD_obs, the render minus the fit's own target, and the target minus OD_obs: the
              number of junctions below zero; and the net bias sum(R - OD_obs) / sum(OD_obs) inside the
              junction discs and over the rest of the band, with the target's loss sum(target - OD_img) /
              sum(OD_img) on the same two regions;
  wide        each pixel with OD_img > 0.03 takes the radius of its nearest truth centreline sample (in frame,
              not merged; observable or not); over pixels whose vessel has r >= 8 px, the share of OD_img the
              fit's render, its target and vesselmap's render hold (1 - sum max(OD_img - X, 0) / sum OD_img);
              and the share of all OD the fit misses that lies on non-observable vessels;
  steps       neighbouring-pixel steps of the render that exceed OD_img's by more than STEP OD, more than
              BORDER px from the image border (sharper walls and square edge ends): count and the largest;
  texture     the sigma-2 high-pass s.d. of OD_img inside OD_img > 0.35 (the large vessels);
  widths      for pathologic scenes, the contiguous width above half maximum of the large vessel at the right
              (x >= 520) on rows 150, 240 and 330, in OD_img and in the render.
Held-out scenes need SPLINEFIT_ALLOW_HELDOUT=1.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import score as S
from .report_figures import _load

STEP, BORDER, WIDE_R = 0.03, 24, 8.0


def _held(T, X, m):
    return 1.0 - float(np.clip(T - X, 0, None)[m].sum() / max(T[m].sum(), 1e-12))


def _run_width(row, x0):
    """Contiguous run above half maximum around the row's maximum (x >= x0)."""
    seg = row[x0:]
    c = int(np.argmax(seg))
    half = seg[c] / 2
    lo = c
    while lo > 0 and seg[lo - 1] > half:
        lo -= 1
    hi = c
    while hi < len(seg) - 1 and seg[hi + 1] > half:
        hi += 1
    return hi - lo + 1


def diagnose(folder: str, kind: str, cache: str) -> dict:
    from experiments.splinefit import fastset as FS
    FS.check_not_heldout(folder)
    case = S.load_truth(folder, kind, reference_target=False)
    res = _load(folder, kind, cache)
    fit = res["fit"]
    T, t, band, val = case["od_img"], case["od_obs"], case["band"], case["valid"]
    R = np.asarray(fit["od_render"], np.float32)
    G = np.asarray(fit["od_target"], np.float32)
    out = {}
    # junctions
    jd = case["jdisc"]
    rest = band & ~jd
    d_obs, d_tgt, t_obs = [], [], []
    for d in case["discs"]:
        m = S.disc_mask(T.shape, d[None]) & band
        if m.any():
            d_obs.append(float((R - t)[m].mean()))
            d_tgt.append(float((R - G)[m].mean()))
            t_obs.append(float((G - t)[m].mean()))
    out["junctions"] = dict(n=len(d_obs), render_below_obs=int(np.sum(np.array(d_obs) < 0)),
                            render_below_target=int(np.sum(np.array(d_tgt) < 0)),
                            target_below_obs=int(np.sum(np.array(t_obs) < 0)),
                            mean_target_minus_obs=float(np.mean(t_obs)), mean_render_minus_target=float(np.mean(d_tgt)),
                            worst=sorted(d_obs)[:3],
                            bias_disc=float((R - t)[jd].sum() / t[jd].sum()), bias_rest=float((R - t)[rest].sum() / t[rest].sum()),
                            loss_disc=float((G - T)[jd].sum() / T[jd].sum()), loss_rest=float((G - T)[rest].sum() / T[rest].sum()))
    # wide vessels
    P = case["prof"]
    keep = P["in_frame"] & (P["merged_into"] < 0)
    xy = np.stack([P["x_px"], P["y_px"]], 1)[keep]
    r = P["radius_px"][keep]
    obs = (P["observable"] & P["line"])[keep]
    H, W = T.shape
    Y, X = np.mgrid[0:H, 0:W]
    px = (T > 0.03) & val
    _, i = cKDTree(xy).query(np.stack([X[px], Y[px]], 1))
    wide = np.zeros_like(px)
    wide[px] = r[i] >= WIDE_R
    miss = np.clip(T - R, 0, None)[px]
    out["wide"] = dict(render=_held(T, R, wide), target=_held(T, G, wide),
                       non_observable_share=float(miss[~obs[i]].sum() / max(miss.sum(), 1e-12)))
    if res.get("vesselmap") is not None:
        out["wide"]["vesselmap"] = _held(T, np.asarray(res["vesselmap"]["od_render"], np.float32), wide)
    # steps
    best = []
    for ax in (0, 1):
        g, gT = np.abs(np.diff(R, axis=ax)), np.abs(np.diff(T, axis=ax))
        m = g - gT > STEP
        m[:BORDER] = m[-BORDER:] = False
        m[:, :BORDER] = m[:, -BORDER:] = False
        best += [(float(g[y, x] - gT[y, x]), int(x), int(y)) for y, x in zip(*np.nonzero(m))]
    best.sort(reverse=True)
    out["steps"] = dict(n=len(best), largest=best[:3])
    hp = T - ndi.gaussian_filter(T, 2.0)
    out["texture_sd"] = float(hp[T > 0.35].std())
    if "pathologic" in os.path.basename(os.path.normpath(folder)):
        out["widths"] = {y: (_run_width(T[y], 520), _run_width(R[y], 520)) for y in (150, 240, 330)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--kind", default="average")
    ap.add_argument("--cache", default=os.path.join(os.environ.get(
        "SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/scratchpad/"
        "work/splinefit"), "report_cache"))
    a = ap.parse_args(argv)
    tot = dict(n=0, render_below_obs=0, render_below_target=0, target_below_obs=0)
    for folder in a.scenes:
        d = diagnose(folder, a.kind, a.cache)
        j, w = d["junctions"], d["wide"]
        for k in tot:
            tot[k] += j[k]
        print(f"{os.path.basename(os.path.normpath(folder))} {a.kind}")
        print(f"  junctions: {j['render_below_obs']}/{j['n']} render < OD_obs, {j['render_below_target']}/{j['n']} "
              f"render < own target, {j['target_below_obs']}/{j['n']} target < OD_obs; mean target - OD_obs "
              f"{j['mean_target_minus_obs']:+.4f}, render - target {j['mean_render_minus_target']:+.4f}; "
              f"worst {[round(v, 2) for v in j['worst']]}")
        print(f"  net bias (R - OD_obs)/OD_obs: discs {j['bias_disc']:+.3f}, rest of band {j['bias_rest']:+.3f}; "
              f"target loss (target - OD_img)/OD_img: discs {j['loss_disc']:+.3f}, rest {j['loss_rest']:+.3f}")
        vm = f", vesselmap {w['vesselmap']:.3f}" if "vesselmap" in w else ""
        print(f"  wide (r >= {WIDE_R:g} px) OD held: render {w['render']:.3f}, target {w['target']:.3f}{vm}; "
              f"missed OD on non-observable vessels {w['non_observable_share']:.2f}")
        print(f"  steps > {STEP} OD beyond OD_img's: {d['steps']['n']}; largest "
              f"{[(round(v, 3), x, y) for v, x, y in d['steps']['largest']]}; texture s.d. {d['texture_sd']:.4f}")
        if "widths" in d:
            print(f"  large-vessel half-max widths (OD_img, render) on rows 150/240/330: {d['widths']}")
    print(f"all: {tot['render_below_obs']}/{tot['n']} junctions render < OD_obs, {tot['render_below_target']}/{tot['n']} "
          f"render < own target, {tot['target_below_obs']}/{tot['n']} target < OD_obs")


if __name__ == "__main__":
    main()
