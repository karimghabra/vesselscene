"""Command line of vesselscene.

    python -m vesselscene scene --seed N [N ...] --preset healthy|pathologic|random -o OUT
        one scene per seed (OUT itself for one seed, OUT/<preset>_sNNN for several): the average
        still and the single frame, the truth and the overlay (scene.save_scene)
    python -m vesselscene score OUT_DIR [OUT_DIR ...] [--out FOLDER]
        the realism scorecard of the scenes' average stills against the 9 real reference stills and
        of their single frames against the real single frames (data/real_stats.json); the statistics
        of each image are cached next to it (stats_<kind>.json)
    python -m vesselscene compare OUT_DIR [--real STILL.tif ...] [--real-frame FRAME.tif]
        side-by-side figures: the synthetic average next to real stills (full frame and zoomed crops,
        same grey scale), the synthetic single frame next to a real raw frame, the truth overlay
    python -m vesselscene junctions OUT_DIR [OUT_DIR ...] [--kinds average,frame] [--zoom Y,X,H,W] [--force]
        the image-junction truth of saved scenes recomputed from their files (junctions.inputs_from_saved:
        an approximation, the noise estimated from the still and no red-cell filling; save_scene writes the
        exact one): junctions_<kind>_recomputed.json and .png (and a zoomed overlay), and the counts.
        --force replaces junctions_<kind>.json instead, with the labels.npz junction rasters and the
        scene.json counts (for scenes saved without the junction truth)
    python -m vesselscene truth OUT_DIR [OUT_DIR ...] [--kind average|frame|both] [--cnr 3] [--min-run 11.9]
                                [--out FOLDER] [--zoom Y,X,H,W] [--force]
        the observable truth (truth.py) of saved scenes at other thresholds, without re-rendering: the
        per-sample CNR from complete_visibility_<kind>.npz (exact; written for older scenes from the saved OD
        and an estimated noise): observable_<kind>[_cnrC_runR].json, .graphml, .png, _rasters.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _cmd_scene(a):
    from . import scene as SC
    seeds = a.seed
    shape = tuple(int(v) for v in a.shape.split(","))
    kinds = tuple(k for k in a.kinds.split(",") if k)
    for seed in seeds:
        out = a.out if len(seeds) == 1 else os.path.join(a.out, f"{a.preset}_s{seed:03d}")
        t0 = time.perf_counter()
        sc = SC.make_scene(seed, a.preset, shape=shape, k=a.k, kinds=kinds, device=a.device,
                           labels=not a.no_labels, junctions=not a.no_junctions,
                           log=lambda *m: print("  ", *m, flush=True))
        files = SC.save_scene(sc, out, overlay=not a.no_labels)
        vis = {}
        for rec in sc.visibility.values():
            for kind, r in rec.items():
                vis.setdefault(kind, {}).setdefault(r["tier"], 0)
                vis[kind][r["tier"]] += 1
        print(f"scene seed {seed} ({a.preset}) -> {out}  {time.perf_counter() - t0:.1f} s  "
              f"{len(sc.graph.vessels)} vessels, {len(sc.crossings)} crossings in frame")
        print("   timings (s):", json.dumps(sc.timings))
        for kind, d in vis.items():
            print(f"   visibility {kind}:", json.dumps(dict(sorted(d.items()))))
        if sc.junctions:
            from . import junctions as J
            for kind, js in sc.junctions.items():
                s_ = J.summary(js)
                print(f"   junctions {kind}: {s_['n']} ({s_['n_shown']} shown), types {json.dumps(s_['types'])}")
    return 0


def _cmd_junctions(a):
    """The image-junction truth of saved scenes, recomputed from their files (junctions.inputs_from_saved: the
    background noise is estimated from the still and the red-cell filling is not saved, so the closed form runs
    at the vessels' full haematocrit, scaled along each vessel by the contrast the saved OD shows).  It is an
    approximation of the exact truth save_scene wrote (v2 correctness review, defect 5: overwriting it changed
    77 of the frame's type_visible and left the rasters stale), so it goes to junctions_<kind>_recomputed.json
    unless --force, which replaces junctions_<kind>.json together with the labels.npz rasters and the
    scene.json counts."""
    from . import junctions as J
    kinds = tuple(k for k in a.kinds.split(",") if k)
    for d in a.dirs:
        new = {}
        for kind in kinds:
            if not os.path.exists(os.path.join(d, f"still_{kind}.tif")):
                continue
            t0 = time.perf_counter()
            kw, img = J.inputs_from_saved(d, kind)
            js = J.image_junctions(**kw)
            new[kind] = js
            tag = "" if a.force else "_recomputed"
            J.save_junctions(js, os.path.join(d, f"junctions_{kind}{tag}.json"))
            name = os.path.basename(os.path.normpath(d))
            J.draw_junctions(img, js, os.path.join(d, f"junctions_{kind}{tag}.png"),
                             title=f"{name} {kind}: junctions" + (" (recomputed)" if tag else ""))
            if a.zoom:
                y0, x0, h, w = (int(v) for v in a.zoom.split(","))
                J.draw_junctions(img, js, os.path.join(d, f"junctions_{kind}{tag}_zoom.png"), crop=(y0, x0, h, w),
                                 scale=3.0, title=f"{name} {kind}")
            s_ = J.summary(js)
            print(f"{name} {kind}: {s_['n']} junctions ({s_['n_shown']} shown, ambiguous {s_['ambiguous_shown']}), "
                  f"{time.perf_counter() - t0:.1f} s; types {json.dumps(s_['types'])}", flush=True)
        if a.force and new:
            _replace_junction_labels(d, new)
    return 0


def _cmd_truth(a):
    """The observable truth of saved scenes, recomputed (other thresholds) without re-rendering: the per-sample
    CNR comes from complete_visibility_<kind>.npz (written by save_scene: exact); for a scene saved before it, from
    the saved OD and an estimate of the noise (junctions.inputs_from_saved), and that profile file is written
    (with junctions_all_<kind>.json) so that every later threshold uses the same CNR.  Writes
    observable_<kind><tag>.json, .graphml, .png and _rasters.npz, tag '' (the default thresholds where the scene
    has no observable truth yet, or --force), '_recomputed' (default thresholds) or '_cnr<C>_run<R>'.
    --force with the default output folder also replaces the labels.npz rasters and the scene.json counts."""
    from . import junctions as J
    from . import truth as TR
    kinds = ("average", "frame") if a.kind in (None, "both") else (a.kind,)
    default = abs(a.cnr - TR.CNR_OBSERVE) < 1e-9 and abs(a.min_run - TR.MIN_RUN_PX) < 1e-9
    for d in a.dirs:
        name = os.path.basename(os.path.normpath(d))
        out = d if not a.out else (a.out if len(a.dirs) == 1 else os.path.join(a.out, name))
        os.makedirs(out, exist_ok=True)
        new = {}
        for kind in kinds:
            if not os.path.exists(os.path.join(d, f"still_{kind}.tif")):
                continue
            t0 = time.perf_counter()
            ctx = TR.folder_context(d, kind)
            if ctx.junctions_all is None:                   # the complete junctions, under the permissive rules
                ctx.junctions_all = J.junctions_from_samples(ctx.S, ctx.graph, crossings=ctx.crossings,
                                                             keep_hidden=True)
                J.save_junctions(ctx.junctions_all, os.path.join(out, f"junctions_all_{kind}.json"))
            obs, prof = TR.observable_truth(ctx, kind, a.cnr, a.min_run, return_profiles=True)
            pp = os.path.join(d, f"complete_visibility_{kind}.npz")
            if not os.path.exists(pp) or out != d:
                prof_d = prof if default else TR.observable_truth(ctx, kind, return_profiles=True)[1]
                po = os.path.join(out, f"complete_visibility_{kind}.npz")
                if not os.path.exists(po):
                    TR.save_profiles(prof_d, po)
            exists = os.path.exists(os.path.join(out, f"observable_{kind}.json"))
            tag = ("" if (a.force or not exists) else "_recomputed") if default else f"_cnr{a.cnr:g}_run{a.min_run:g}"
            base = os.path.join(out, f"observable_{kind}{tag}")
            TR.save_observable(obs, base + ".json")
            TR.save_graphml(obs, base + ".graphml")
            ras = TR.observable_rasters(obs, prof)
            np.savez_compressed(base + "_rasters.npz", **ras)
            if ctx.image is not None:
                TR.draw_observable(ctx.image, obs, base + ".png", prof=prof,
                                   title=f"{name} {kind} (noise {ctx.noise_source})")
                if a.zoom:
                    y0, x0, h, w = (int(v) for v in a.zoom.split(","))
                    TR.draw_observable(ctx.image, obs, base + "_zoom.png", crop=(y0, x0, h, w), scale=3.0, prof=prof,
                                       title=f"{name} {kind}")
            if tag == "" and out == d:
                new[kind] = (obs, ras)
            s_ = obs["summary"]
            src = "saved profiles" if ctx.cnr_from_profiles else f"noise {ctx.noise_source}"
            print(f"{name} {kind}: CNR >= {a.cnr:g}, runs >= {a.min_run:g} px ({src}): "
                  f"{s_['n_runs']} runs on {s_['n_vessels']} vessels ({s_['run_length_px']:.0f} px), "
                  f"{s_['n_junctions']} junctions {json.dumps(s_['types_observable'])}, "
                  f"{s_['n_hidden_junctions']} hidden; {time.perf_counter() - t0:.1f} s -> {base}.json", flush=True)
            problems = TR.check_observable(obs, prof)
            if problems:
                print(f"   {len(problems)} consistency problems, e.g. {problems[:3]}")
        if a.force and new:
            _replace_truth_labels(d, new)
    return 0


def _replace_truth_labels(d: str, new: dict):
    """truth --force: the observable rasters of labels.npz (observable_<key>_<kind>) and the scene.json counts."""
    p = os.path.join(d, "labels.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            lab = {key: z[key] for key in z.files}
        for kind, (obs, ras) in new.items():
            for key, v in ras.items():
                lab[f"observable_{key}_{kind}"] = v
        np.savez_compressed(p, **lab)
    p = os.path.join(d, "scene.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            meta = json.load(fh)
        counts = meta.setdefault("counts", {}).get("observable") or {}
        counts.update({kind: obs["summary"] for kind, (obs, _) in new.items()})
        meta["counts"]["observable"] = counts
        meta["truth_rules"] = next(iter(new.values()))[0]["rules"]
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)


def _replace_junction_labels(d: str, new: dict):
    """--force: the junction rasters of labels.npz (junction_<key>, the frame's junction_<key>_frame) and the
    junction counts and rules of scene.json, from the recomputed junctions."""
    from . import junctions as J
    p = os.path.join(d, "labels.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            lab = {key: z[key] for key in z.files}
        for kind, js in new.items():
            shape = lab[f"od_{kind}"].shape if f"od_{kind}" in lab else lab["dominant"].shape
            suf = "" if kind == "average" else "_frame"
            for key, v in J.junction_rasters(js, shape).items():
                lab[f"junction_{key}{suf}"] = v.astype(np.float16) if key.startswith(("heat_", "arm_dir")) else v
        np.savez_compressed(p, **lab)
    p = os.path.join(d, "scene.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            meta = json.load(fh)
        counts = (meta.setdefault("counts", {}).get("junctions") or {})
        counts.update({kind: J.summary(js) for kind, js in new.items()})
        meta["counts"]["junctions"] = counts
        meta["junction_rules"] = J.rules()
        meta["junctions_recomputed"] = sorted(new)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)


def _analyse_cached(path: str, kind: str, force: bool = False) -> dict:
    """realism.record of stats.analyse on one image, cached in
    stats_<kind>.json next to it (keyed by the file's size and mtime)."""
    import tifffile
    from . import realism as RM
    from . import stats as st
    cache = os.path.join(os.path.dirname(path), f"stats_{kind}.json")
    stamp = [os.path.getsize(path), os.path.getmtime(path), st.STATS_VERSION]
    if not force and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("stamp") == stamp:
            return RM._from_json(d["record"])
    I = tifffile.imread(path).astype(np.float64)
    valid = np.isfinite(I) & (I > 0)
    if kind == "frame":
        valid &= I < RM.SATURATED_DN
    res = st.analyse(I, valid=valid, seed=RM.STILL_SEED if kind == "average" else RM.FRAME_SEED)
    rec = RM.record(res, name=os.path.basename(os.path.dirname(path)), kind=kind)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(dict(stamp=stamp, record=RM._jsonable(rec)), fh)
    return RM._from_json(json.loads(json.dumps(RM._jsonable(rec))))


def score_dirs(dirs, kinds=("average", "frame"), out=None, force=False, label=None, log=print) -> dict:
    """Scorecards of the scenes in dirs against the real reference (stills for
    'average', single frames for 'frame').  Writes scorecard_<kind>.md / .csv
    and fig_scorecard_<kind>.png into out.  Returns {kind: Scorecard}."""
    from . import realism as RM
    out = out or os.path.dirname(os.path.abspath(os.path.normpath(dirs[0])))
    os.makedirs(out, exist_ok=True)
    cards = {}
    for kind in kinds:
        recs = []
        for d in dirs:
            p = os.path.join(d, f"still_{kind}.tif")
            if not os.path.exists(p):
                continue
            t0 = time.perf_counter()
            recs.append(_analyse_cached(p, kind, force))
            log(f"  {kind} {d}: {time.perf_counter() - t0:.1f} s")
        if not recs:
            continue
        real = RM.load_real("stills" if kind == "average" else "frames")
        card = RM.scorecard(recs, real)
        cards[kind] = card
        tag = f"_{label}" if label else ""
        with open(os.path.join(out, f"scorecard_{kind}{tag}.md"), "w", encoding="utf-8") as fh:
            fh.write(card.markdown(sort="z"))
        card.to_csv(os.path.join(out, f"scorecard_{kind}{tag}.csv"))
        RM.figure_scorecard(card, os.path.join(out, f"fig_scorecard_{kind}{tag}.png"),
                            title=f"{len(recs)} synthetic {kind} image(s) vs real {real['kind']}")
        s = card.summary()
        log(f"{kind}: " + "; ".join(f"{k} {a_}/{b}" for k, (a_, b) in s.items()))
        worst = sorted(card.scored(), key=lambda r: -abs(np.nan_to_num(r["z"], nan=0.0, posinf=99, neginf=99)))[:12]
        for r in worst:
            log(f"   z {r['z']:+6.1f}  {'pass' if r['passed'] else 'FAIL'}  {r['label']}  "
                f"(real {r['real_mean']:.4g}, synthetic {r['syn_mean']:.4g})")
    return cards


def _cmd_score(a):
    kinds = tuple(k for k in a.kinds.split(",") if k)
    score_dirs(a.dirs, kinds, a.out, a.force, a.label)
    return 0


def _cmd_compare(a):
    from . import views as V
    reals = a.real or list(V.DEFAULT_REAL)
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    tag = os.path.basename(os.path.normpath(a.dir))
    paths = [V.figure_average(a.dir, reals, os.path.join(out, f"compare_average_{tag}.png"))]
    if os.path.exists(os.path.join(a.dir, "still_frame.tif")):
        paths.append(V.figure_frame(a.dir, a.real_frame, os.path.join(out, f"compare_frame_{tag}.png")))
    if os.path.exists(os.path.join(a.dir, "truth.json")):
        paths.append(V.figure_truth(a.dir, os.path.join(out, f"compare_truth_{tag}.png")))
    for p in paths:
        print(p)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vesselscene", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scene", help="generate scenes")
    s.add_argument("--seed", type=int, nargs="+", required=True)
    s.add_argument("--preset", choices=("healthy", "pathologic", "random"), default="healthy")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--k", type=float, default=None, help="px per capillary diameter (default: the preset's)")
    s.add_argument("--shape", default="1200,1920", help="H,W in px")
    s.add_argument("--kinds", default="average,frame")
    s.add_argument("--device", default="cuda")
    s.add_argument("--no-labels", action="store_true")
    s.add_argument("--no-junctions", action="store_true", help="skip the image-junction truth (10-30 s per image)")
    s.set_defaults(fn=_cmd_scene)
    c = sub.add_parser("score", help="realism scorecard against the real reference")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--kinds", default="average,frame")
    c.add_argument("--out", default=None, help="folder for the scorecards (default: parent of the first dir)")
    c.add_argument("--label", default=None, help="suffix of the scorecard files")
    c.add_argument("--force", action="store_true", help="recompute cached statistics")
    c.set_defaults(fn=_cmd_score)
    v = sub.add_parser("compare", help="side-by-side figures with real stills")
    v.add_argument("dir")
    v.add_argument("--real", action="append", help="real mean_stabilized.tif (repeatable; default two stills)")
    v.add_argument("--real-frame", default=None, help="a real raw frame_*.tif")
    v.add_argument("--out", default=None)
    v.set_defaults(fn=_cmd_compare)
    j = sub.add_parser("junctions", help="image-junction truth of saved scenes")
    j.add_argument("dirs", nargs="+")
    j.add_argument("--kinds", default="average,frame")
    j.add_argument("--zoom", default=None, help="Y,X,H,W px: also a 3x zoomed overlay of this crop")
    j.add_argument("--force", action="store_true",
                   help="replace junctions_<kind>.json, the labels.npz rasters and the scene.json counts")
    j.set_defaults(fn=_cmd_junctions)
    t = sub.add_parser("truth", help="observable truth of saved scenes at other thresholds (no re-render)")
    t.add_argument("dirs", nargs="+")
    t.add_argument("--kind", choices=("average", "frame", "both"), default="both")
    t.add_argument("--cnr", type=float, default=3.0, help="CNR_OBSERVE (default 3)")
    t.add_argument("--min-run", type=float, default=11.9, help="MIN_RUN_PX (default 1 lambda = 11.9 px)")
    t.add_argument("--out", default=None, help="write here instead of the scene folder")
    t.add_argument("--zoom", default=None, help="Y,X,H,W px: also a 3x zoomed overlay of this crop")
    t.add_argument("--force", action="store_true",
                   help="write the standard names (observable_<kind>.json ...), and in the scene folder also the "
                        "labels.npz rasters and scene.json counts")
    t.set_defaults(fn=_cmd_truth)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
