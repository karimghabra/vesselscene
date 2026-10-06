r"""Psychophysics for the spline fitter: a frozen battery of small canonical stimuli with exact truth, and the
fitter's tuning curves and thresholds on it.

Why.  A whole-image score says how good a pipeline is, not what it fails at.  As a psychophysicist measures
a visual system with gratings and sweeps one stimulus parameter at a time, this module measures the fitter
with hand-built vessel scenes whose answer is known, each sweep varying one property (vessel width, blur,
contrast, fork geometry, crossing angle / depth / width, wall gap between parallel vessels, vessel ends,
red-cell gaps in a frame), plus a negative control (an empty background: nothing should be fitted).  Per
sweep it reports the metric parts against the swept parameter (a tuning curve) and a threshold (the
smallest width still detected, the smallest crossing angle still typed a crossing, the smallest gap still
resolved, ...).

Stimuli.  Each geometry is a hand-built VesselGraph (as in vesselscene/tests/test_junctions.py and
test_truth.py; coordinates in px here, d_c = px / K in the graph) rendered by vesselscene.scene.make_scene
on a SHAPE = 128 x 128 px frame with fixed calibrated optics (scene.CALIBRATED_OPTICS on a flat focal
surface at Z_FOCUS, defocus gain DEFOCUS, burst focus spread FOCUS_SPREAD; the frame at the burst's mean
focus) and the 'healthy' imaging preset of one SEED for every stimulus: the same texture and camera,
common random numbers across a sweep, so a tuning curve's steps come from the stimulus, not from a new
noise draw.  Blur is swept by moving the focus (focus_offset), not the vessel, so the depth-dependent
bypass of the contrast stays fixed; contrast by the haematocrit.  make_scene computes the observable truth
(truth.py) and the image junctions (junctions.py) exactly as for the dev scenes, and save_scene writes the
usual folder, so score.py scores a probe like any image (crop-free, the whole probe).  Each stimulus is one
(geometry, kind): mostly the average still, a few frames (their red-cell gaps and 20x noise).  Only the
stimuli of the kinds the loop fits (fastset.KINDS: the average still, re-baseline R2) are run and scored; the
frame stimuli stay in the battery and its manifest, unscored.

The battery is FROZEN with the evaluator: ``generate()`` writes it once into PROBE_DIR (refusing to
overwrite) with a manifest (battery.json: every stimulus's parameters and the sha256 of its image and truth
files), committed as probe_battery.json (the data, 9 MB, is not committed: generation is deterministic and
``load_battery()`` regenerates a missing battery); ``load_battery()`` checks every hash against it.  The truth is read here only to generate and to score, never
in a pipeline path; a pipeline sees image and valid only.

Scoring (``score_probe``).  score.score(case, out) (audit=True) on the whole probe, and

    probe_composite = score.py's composite on a probe with observable junctions;
                      on a junction-free probe (an isolated vessel, a parallel pair, an end) the graph part
                      (graph_probe) is mean(centreline_f1, edge_cover, junction_absent), junction_absent = 1 / (1 + the
                      number of fitted junctions that match neither a truth nor a don't-care junction):
                      score.py's junction terms are 0 / NaN by construction there (no truth junction to
                      find), which would cap these probes and hide false junctions;
                      on the empty probe (and any probe whose truth has no observable vessel)
                      1 - min(1, fitted length / EMPTY_LEN_PX).

    probe_score = mean probe_composite over the battery (higher is better).

Per row also: the stimulus's measured truth (medians of the true samples' r, s, a), the fitted length,
edges and junction types, the crossings fitted, and for the parallel pairs the number of distinct fitted
centrelines crossing the pair's mid transect (``n_lines_fit``, resolved = 2) next to the truth's.

Thresholds (``tuning``): width / blur / contrast: detected = centreline recall >= 0.5; crossing: typed =
crossing_recall 1; gap: resolved = n_lines_fit == 2; fork / T: matched and coarse type right.

Controls.  ``run_probes(..., oracle=True)`` scores the TRUE observable network of each probe (score.py's
true_network_observable; ORACLE, truth-reading, a self-check of the battery's ceiling) and
``nothing`` is the null pipeline (the floor).

    python -m experiments.splinefit.probes generate            # once (frozen; regenerated on demand)
    python -m experiments.splinefit.probes run [--pipeline mod:fn] [--out JSON] [--plot PNG] [--only SWEEP]
    python -m experiments.splinefit.probes oracle | nothing    # the ceiling / floor controls

API: run_probes(annotate, deadline=None) -> dict(probe_score, rows, tuning, digest, seconds) (run_experiment
tier 1), run(pipeline) -> (rows, summary), tuning(rows), report(res) (markdown), plot(res, path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

WORK = os.environ.get("SPLINEFIT_WORK", "/tmp/claude-0/-home-user-vesselscene/8df61651-ad5a-5e7a-847b-43ea09c109c3/"
                                        "scratchpad/work/splinefit")
PROBE_DIR = os.environ.get("SPLINEFIT_PROBES", os.path.join(WORK, "probes"))
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_battery.json")   # committed, frozen
VERSION = "probes-v1"
SEED = 11                    # one imaging draw for every stimulus (common random numbers)
SHAPE = (128, 128)           # px (H, W)
K = 4.0                      # px per d_c
Z_FOCUS = 10.0               # d_c, flat focal surface
DEFOCUS = 0.06               # px of blur per d_c of defocus
FOCUS_SPREAD = 3.0           # d_c, the burst's axial focus jitter (the average's extra blur)
Z0 = 10.0                    # d_c, the default vessel depth (in focus)
C = (64.0, 64.0)             # px, the stimulus centre
EMPTY_LEN_PX = 100.0         # the empty probe: 1 - min(1, fitted length / this)
DETECT_RECALL = 0.5          # 'detected': centreline recall at least this


# ======================================================================== geometry builders (px)
def _graph():
    from vesselscene.graph import VesselGraph
    return VesselGraph(dict(k=K))


def _node(g, xy, z=Z0):
    return g.add_node(np.asarray(xy, float) / K, depth=float(np.atleast_1d(z)[0]))


def _vessel(g, a, b, r_px, z=Z0, hct=1.0, cls="venule", side="V", za=None):
    """A straight vessel a -> b (px; flow a -> b), radius r_px, depth z (scalar or profile, d_c)."""
    na = _node(g, a, z if za is None else za)
    nb = _node(g, b, np.atleast_1d(z)[-1])
    return g.add_vessel(na, nb, radius=float(r_px) / K, depth=z, hct=hct, cls=cls, side=side)


def _dir(deg):
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)])


def _through(deg, half=130.0, centre=C, off=0.0):
    """The ends of a line through centre (+ off px along its normal) at deg, reaching past the frame."""
    d, n = _dir(deg), _dir(deg + 90.0)
    c = np.asarray(centre, float) + off * n
    return c - half * d, c + half * d


def build_line(r_px=3.0, hct=1.0, deg=17.0, z=Z0, cls="venule"):
    g = _graph()
    a, b = _through(deg)
    _vessel(g, a, b, r_px, z, hct, cls=cls, side="C" if cls == "capillary" else "V")
    return g


def murray_angles(r0, r1, r2):
    """Murray's optimal branch angles (deg) of daughters r1, r2 off the parent r0 (r0^3 = r1^3 + r2^3)."""
    c1 = (r0 ** 4 + r1 ** 4 - r2 ** 4) / (2 * r0 ** 2 * r1 ** 2)
    c2 = (r0 ** 4 + r2 ** 4 - r1 ** 4) / (2 * r0 ** 2 * r2 ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c1)))), math.degrees(math.acos(max(-1.0, min(1.0, c2))))


def build_fork(r0=6.0, r1=None, r2=None, th1=None, th2=None, base=8.0, z=Z0, confluence=False):
    """A parent from the left into a node at the centre, two daughters out to the right: radii by Murray's
    law (symmetric by default), angles by Murray unless given.  confluence: the flow reversed."""
    r1 = r0 * 2 ** (-1 / 3) if r1 is None else r1
    r2 = (r0 ** 3 - r1 ** 3) ** (1 / 3) if r2 is None else r2
    m1, m2 = murray_angles(r0, r1, r2)
    th1, th2 = (m1 if th1 is None else th1), (m2 if th2 is None else th2)
    g = _graph()
    n = _node(g, C, z)
    p = _node(g, np.asarray(C) - 130.0 * _dir(base), z)
    d1 = _node(g, np.asarray(C) + 130.0 * _dir(base - th1), z)
    d2 = _node(g, np.asarray(C) + 130.0 * _dir(base + th2), z)
    for (u, v), r in (((p, n), r0), ((n, d1), r1), ((n, d2), r2)):
        if confluence:
            u, v = v, u
        g.add_vessel(u, v, radius=r / K, depth=z, cls="venule", side="V")
    return g


def build_crossing(angle=45.0, r_a=3.0, r_b=3.0, z_a=Z0, z_b=Z0 + 20.0, base=10.0):
    """Two straight vessels through the centre, angle apart; a (base deg, depth z_a) and b (z_b)."""
    g = _graph()
    a0, a1 = _through(base)
    b0, b1 = _through(base + angle)
    _vessel(g, a0, a1, r_a, z_a, cls="venule", side="V")
    _vessel(g, b0, b1, r_b, z_b, cls="arteriole", side="A")
    return g


def build_t_branch(r0=5.0, r_b=1.5, base=8.0, z=Z0):
    """A true T: a wide vessel through a node at the centre (Murray: the continuation r1^3 = r0^3 - r_b^3),
    a thin branch leaving it at 90 deg."""
    r1 = (r0 ** 3 - r_b ** 3) ** (1 / 3)
    return build_fork(r0, r1, r_b, th1=0.0, th2=90.0, base=base, z=z)


def build_pseudo_t(r_w=5.0, r_t=1.5, gap=3.0, base=8.0, z_w=Z0, z_t=Z0 + 15.0):
    """A pseudo-T: a thin vessel (deeper) coming down at 90 deg and ending gap px (wall to wall) short of a
    wide one, with no node: the image shows a T where there is no fork."""
    g = _graph()
    a0, a1 = _through(base)
    _vessel(g, a0, a1, r_w, z_w, cls="venule", side="V")
    d = _dir(base + 90.0)
    end = np.asarray(C) - (r_w + gap + r_t) * d
    _vessel(g, np.asarray(C) - 130.0 * d, end, r_t, z_t, cls="arteriole", side="A")
    return g


def build_parallel(r_px=2.0, gap=1.0, deg=17.0, z=Z0):
    """Two equal parallel vessels, wall gap `gap` px (centre distance 2 r + gap)."""
    g = _graph()
    h = r_px + 0.5 * gap
    for off, side in ((-h, "V"), (h, "A")):
        a, b = _through(deg, off=off)
        _vessel(g, a, b, r_px, z, cls="venule" if side == "V" else "arteriole", side=side)
    return g


def build_dive(r_px=2.5, z_end=120.0, deg=12.0, end_px=40.0):
    """A vessel from the left diving from Z0 to z_end d_c towards its end (end_px past the centre, inside
    the frame): it fades."""
    g = _graph()
    d = _dir(deg)
    end = np.asarray(C) + end_px * d
    z = np.r_[np.full(5, Z0), np.linspace(Z0, z_end, 6)[1:]]
    _vessel(g, np.asarray(C) - 130.0 * d, end, r_px, z)
    return g


def build_blind_end(r_px=2.0, deg=12.0, end_px=24.0):
    """A vessel from the left ending abruptly (end_px past the centre) inside the frame at constant depth (a
    blind end)."""
    g = _graph()
    d = _dir(deg)
    _vessel(g, np.asarray(C) - 130.0 * d, np.asarray(C) + end_px * d, r_px, Z0)
    return g


def build_empty():
    """The negative control: one thin vessel 30 px outside the frame (vesselscene needs one in its padded
    domain; its OD in the frame is < 1e-5); the frame shows only background (texture, illumination, noise)."""
    g = _graph()
    _vessel(g, (-300.0, -30.0), (500.0, -30.0), 1.0, Z0)
    return g


BUILDERS = dict(line=build_line, fork=build_fork, crossing=build_crossing, t_branch=build_t_branch,
                pseudo_t=build_pseudo_t, parallel=build_parallel, dive=build_dive, blind_end=build_blind_end,
                empty=build_empty)


# ======================================================================== the battery
def _st(name, sweep, x, geom, kind="average", focus=0.0, **args):
    """One stimulus: its name, the sweep it belongs to and its value x there, the geometry (builder, args),
    the image kind and the focus offset (d_c; blur sweep)."""
    return dict(name=name, sweep=sweep, x=x, geom=geom, args=args, kind=kind, focus=float(focus))


SWEEPS = dict(                # x axis label, threshold rule
    width=("diameter px", "smallest diameter detected"),
    blur=("focus offset d_c", "largest defocus detected with width err < 0.25"),
    contrast=("haematocrit", "smallest haematocrit detected"),
    corner=("-", "thin, defocused, low contrast detected"),
    fork=("geometry", "forks matched and typed 3-way"),
    crossing_angle=("angle deg", "smallest angle typed a crossing"),
    crossing_depth=("depth gap d_c", "smallest depth gap typed a crossing"),
    crossing_width=("geometry", "unequal crossing typed a crossing"),
    t=("geometry", "T junctions matched and typed 3-way"),
    gap=("wall gap px", "smallest wall gap resolved (2 lines)"),
    end=("geometry", "ends traced without a false junction"),
    rbc=("kind", "capillary detected"),
    empty=("kind", "nothing fitted"))

BATTERY = [
    # isolated vessel: width (blur low, contrast high), blur (focus moved), contrast (haematocrit)
    *[_st(f"line_d{2 * r:g}", "width", 2 * r, "line", r_px=r) for r in (1.25, 1.5, 3.0, 6.0)],
    *[_st(f"line_d6_f{f:g}", "blur", f, "line", focus=f, r_px=3.0) for f in (25.0, 50.0)],
    *[_st(f"line_d6_h{h:g}", "contrast", h, "line", r_px=3.0, hct=h) for h in (0.5, 0.25)],
    _st("line_d5_f25_h0.5", "corner", "d5 f25 h0.5", "line", focus=25.0, r_px=2.5, hct=0.5),
    # Y-forks (Murray radii and angles)
    _st("fork_thin", "fork", "thin r0=2", "fork", r0=2.0),
    _st("fork_wide", "fork", "wide r0=6", "fork", r0=6.0),
    _st("fork_asym", "fork", "asym 6>5.5+3.7", "fork", r0=6.0, r1=5.5),
    _st("fork_acute", "fork", "acute 2x15deg", "fork", r0=4.0, th1=15.0, th2=15.0),
    _st("fork_thin_frame", "fork", "thin frame", "fork", "frame", r0=2.0),
    # X-crossings: angle (equal widths, depth gap 20 d_c), depth gap at 45 deg, unequal widths, a frame
    *[_st(f"cross_a{a:g}", "crossing_angle", a, "crossing", angle=float(a)) for a in (20, 30, 45, 90)],
    *[_st(f"cross_a45_dz{dz:g}", "crossing_depth", dz, "crossing", angle=45.0, z_b=Z0 + dz) for dz in (5.0, 50.0)],
    _st("cross_a45_wide_over_thin", "crossing_width", "6 over 1.5", "crossing", angle=45.0, r_a=6.0, r_b=1.5),
    _st("cross_a45_frame", "crossing_angle", "45 frame", "crossing", "frame", angle=45.0),
    # T: a true T-branch (a node) and a pseudo-T (a thin vessel ending at a wide one, no node)
    _st("t_branch", "t", "T branch", "t_branch"),
    _st("pseudo_t", "t", "pseudo-T", "pseudo_t"),
    # parallel pairs at shrinking wall gaps
    *[_st(f"parallel_g{gp:g}", "gap", gp, "parallel", r_px=2.0, gap=float(gp)) for gp in (1.0, 2.0, 3.0, 5.0)],
    # vessel ends
    _st("end_dive", "end", "dive", "dive"),
    _st("end_blind", "end", "blind end", "blind_end"),
    # a capillary: red-cell gaps in the frame, the burst mean in the average
    _st("capillary_frame", "rbc", "frame", "line", "frame", r_px=2.4, cls="capillary", deg=-14.0),
    _st("capillary_average", "rbc", "average", "line", r_px=2.4, cls="capillary", deg=-14.0),
    # negative control
    _st("empty_average", "empty", "average", "empty"),
    _st("empty_frame", "empty", "frame", "empty", "frame"),
]


def _geom_key(st: dict) -> str:
    """One folder per (geometry, args, focus): both kinds of a geometry share it."""
    a = ",".join(f"{k}={v}" for k, v in sorted(st["args"].items()))
    return f"{st['geom']}({a})@{st['focus']:g}"


def geometries() -> dict:
    """{folder name: (geom, args, focus)} of the battery, in order (the first stimulus using it names it)."""
    out, seen = {}, {}
    for st in BATTERY:
        key = _geom_key(st)
        if key not in seen:
            nm = st["name"].replace("_frame", "") if st["kind"] == "frame" else st["name"]
            while nm in out:
                nm += "_"
            seen[key] = nm
            out[nm] = (st["geom"], st["args"], st["focus"])
        st["folder"] = seen[key]
    return out


def optics(focus: float = 0.0):
    from vesselscene import render as R
    from vesselscene import scene as SC
    o = replace(R.Optics(), **SC.CALIBRATED_OPTICS)
    return replace(o, z_focus=Z_FOCUS, defocus_px_per_dc=DEFOCUS, focus_spread=FOCUS_SPREAD, focus_offset=focus)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(out_dir: str = PROBE_DIR, force: bool = False, log=print) -> dict:
    """Render every geometry of the battery into out_dir/<folder> (scene.save_scene, both kinds) and write
    the manifest battery.json; refuses to overwrite an existing battery unless force."""
    from experiments.splinefit import fastset as FS
    from vesselscene import scene as SC
    FS.check_not_heldout(out_dir)
    man_p = os.path.join(out_dir, "battery.json")
    if os.path.exists(man_p) and not force:
        raise FileExistsError(f"{man_p} exists: the battery is frozen (force=True to regenerate)")
    os.makedirs(out_dir, exist_ok=True)
    geo = geometries()
    meta = {}
    for nm, (geom, args, focus) in geo.items():
        t0 = time.perf_counter()
        g = BUILDERS[geom](**args)
        sc = SC.make_scene(SEED, "healthy", shape=SHAPE, graph=g, optics=optics(focus), device="cpu",
                           frame_focus_offset=0.0)
        folder = os.path.join(out_dir, nm)
        SC.save_scene(sc, folder, overlay=False, vesselmap_export=False)
        glare = {k: int(sc.params["formed"][k]["glare_spots"]) for k in sc.params["formed"]}
        obs = {k: dict(n_junctions=len(o["junctions"]),
                       types=sorted(j.get("type_observable", j.get("type")) for j in o["junctions"]),
                       n_dont_care=len(o.get("dont_care_junctions", [])),
                       observable_px=round(float(o["summary"]["observable_px"]), 1),
                       n_runs=int(o["summary"]["n_runs"])) for k, o in sc.observable.items()}
        meta[nm] = dict(geom=geom, args=args, focus=focus, glare=glare, observable=obs)
        log(f"probe {nm}: {time.perf_counter() - t0:.1f} s  {json.dumps(obs)}  glare {glare}")
    stimuli = []
    for st in BATTERY:
        folder = os.path.join(out_dir, st["folder"])
        stimuli.append(dict({k: v for k, v in st.items()},
                            sha_image=_sha(os.path.join(folder, f"still_{st['kind']}.tif")),
                            sha_truth=_sha(os.path.join(folder, f"observable_{st['kind']}.json"))))
    man = dict(version=VERSION, seed=SEED, shape=list(SHAPE), k=K, z_focus=Z_FOCUS, defocus=DEFOCUS,
               focus_spread=FOCUS_SPREAD, note="psychophysics probe battery (probes.py); frozen with score.py",
               geometries=meta, stimuli=stimuli)
    with open(man_p, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=float)
    return man


def load_battery(probe_dir: str = PROBE_DIR, verify: bool = True) -> list:
    """The frozen battery's stimuli (manifest order), with absolute folders; verify: the image and truth
    files must hash as in the committed manifest (MANIFEST, probe_battery.json).  A missing battery is
    regenerated first (generate is deterministic: the same files, the same hashes, ~30 s)."""
    if not os.path.exists(os.path.join(probe_dir, "battery.json")):
        generate(probe_dir)
    with open(os.path.join(probe_dir, "battery.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    if verify and os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
            ref = json.load(fh)
        key = [(st["name"], st["sha_image"], st["sha_truth"]) for st in ref["stimuli"]]
        if [(st["name"], st["sha_image"], st["sha_truth"]) for st in man["stimuli"]] != key:
            raise RuntimeError(f"{probe_dir}/battery.json differs from the frozen manifest {MANIFEST}")
    out = []
    for st in man["stimuli"]:
        st = dict(st, path=os.path.join(probe_dir, st["folder"]))
        if verify:
            for key, fn in (("sha_image", f"still_{st['kind']}.tif"), ("sha_truth", f"observable_{st['kind']}.json")):
                if _sha(os.path.join(st["path"], fn)) != st[key]:
                    raise RuntimeError(f"probe {st['name']}: {fn} changed since the battery was frozen")
        out.append(st)
    return out


# ======================================================================== scoring
def _seg_hits(P: np.ndarray, p: np.ndarray, d: np.ndarray, half: float) -> list:
    """Signed positions along the transect (point p, unit direction d, half length half) where polyline P
    crosses it."""
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) < 2:
        return []
    n = np.array([-d[1], d[0]])
    s = (P - p) @ n                                     # side of the transect line
    t = (P - p) @ d
    out = []
    for i in np.flatnonzero(np.sign(s[:-1]) * np.sign(s[1:]) <= 0):
        if s[i] == s[i + 1]:
            continue
        u = s[i] / (s[i] - s[i + 1])
        ti = t[i] + u * (t[i + 1] - t[i])
        if abs(ti) <= half:
            out.append(float(ti))
    return out


def n_lines(polylines, centre, normal_deg: float, half: float, merge_px: float = 1.0) -> int:
    """Distinct lines crossing the transect through centre along normal_deg (within half px): crossing points
    closer than merge_px count once."""
    p, d = np.asarray(centre, float), _dir(normal_deg)
    t = sorted(x for P in polylines for x in _seg_hits(P, p, d, half))
    k, last = 0, None
    for x in t:
        if last is None or x - last > merge_px:
            k += 1
        last = x
    return k


def _length(network: dict | None) -> float:
    L = 0.0
    for e in (network or {}).get("edges", []):
        xy = np.asarray(e["xy"], float).reshape(-1, 2)
        if len(xy) > 1:
            L += float(np.hypot(*np.diff(xy, axis=0).T).sum())
    return L


def _fitted_junctions(out: dict) -> list:
    from experiments.splinefit import score as S
    if out.get("network") is not None:
        return S.network_graph(out["network"])[1]
    return [tuple(j) for j in out.get("junctions", [])]


def _polylines(out: dict) -> list:
    from experiments.splinefit import score as S
    if out.get("network") is not None:
        return S.network_graph(out["network"])[0]
    return [np.asarray(P, float).reshape(-1, 2) for P in out.get("polylines", [])]


SCORE_KEYS = ("composite", "graph_score", "centreline_recall", "centreline_precision", "centreline_f1",
              "junction_recall_strict", "junction_precision_strict", "junction_f1_strict",
              "junction_type_accuracy_coarse", "junction_type_balanced_coarse", "crossing_recall",
              "crossing_precision", "edge_cover", "pieces_per_edge", "geom_matched", "offset_px", "width_rel_err",
              "blur_err_px", "contrast_rel_err", "width_bias", "blur_bias_px", "contrast_bias", "pos", "width",
              "explained", "explained_junction", "junction_bias", "chi2", "explained_true", "target_fidelity",
              "render_agreement", "n_junctions_fit", "n_junctions_truth")


def score_probe(st: dict, case: dict, out: dict) -> dict:
    """One probe's row (module docstring, 'Scoring')."""
    from experiments.neuromimetic import harness as H
    from experiments.splinefit import score as S
    obs = case["obs"]
    net = out.get("network")
    juncs = _fitted_junctions(out)
    row = dict(name=st["name"], sweep=st["sweep"], x=st["x"], kind=st["kind"])
    L = _length(net) if net is not None else sum(float(np.hypot(*np.diff(np.asarray(P, float).reshape(-1, 2),
                                                                         axis=0).T).sum())
                                                 for P in out.get("polylines", []) if len(P) > 1)
    row.update(fitted_len_px=L, n_edges=len((net or {}).get("edges", [])) if net is not None
               else len(out.get("polylines", [])),
               n_crossings_fit=len((net or {}).get("crossings", [])),
               junction_types_fit=sorted(str(j[2]) if len(j) > 2 else "?" for j in juncs),
               junction_types_truth=sorted(j.get("type_observable", "?") for j in obs["junctions"]))
    sm = case["samples"]
    row.update(r_true_px=float(np.median(sm["r"])) if len(sm["r"]) else float("nan"),
               s_true_px=float(np.median(sm["s"])) if len(sm["s"]) else float("nan"),
               a_true=float(np.median(sm["a"])) if len(sm["a"]) else float("nan"),
               observable_px=float(sum(len(r["xy"]) for r in obs["runs"])))
    truth_empty = st["sweep"] == "empty" or row["observable_px"] < S.LAMBDA_PX
    row["truth_empty"] = truth_empty
    if truth_empty:
        row["probe_composite"] = 1.0 - min(1.0, L / EMPTY_LEN_PX)
        return row
    s = S.score(case, out, audit=True)
    for k in SCORE_KEYS:
        v = s.get(k, float("nan"))
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    if obs["junctions"]:
        row["graph_probe"] = row["graph_score"]
        row["probe_composite"] = row["composite"]
    else:
        _, _, fp = H.match_junctions(obs, juncs, full=True)
        row["n_false_junctions"] = len(fp)
        row["junction_absent"] = 1.0 / (1.0 + len(fp))
        g = [row["centreline_f1"], row["edge_cover"], row["junction_absent"]]
        row["graph_probe"] = float(np.mean([v if np.isfinite(v) else 0.0 for v in g]))
        e = row["explained"]
        e = min(1.0, max(0.0, e)) if np.isfinite(e) else 0.0
        row["probe_composite"] = 0.5 * row["graph_probe"] + 0.5 * float(np.mean([row["pos"], row["width"], e]))
    if st["sweep"] == "gap":                              # the parallel pair: distinct lines at the mid transect
        deg = float(st["args"].get("deg", 17.0)) + 90.0
        half = 2.0 * float(st["args"]["r_px"]) + float(st["args"]["gap"]) + 4.0
        row["n_lines_fit"] = n_lines(_polylines(out), C, deg, half)
        row["n_lines_truth"] = n_lines([r["xy"] for r in obs["runs"]], C, deg, half)
    return row


def oracle_out(case: dict) -> dict:
    """ORACLE (truth-reading; the battery's ceiling self-check): the true observable network as the output."""
    from experiments.splinefit import score as S
    if not case["obs"]["runs"]:
        return dict(network=dict(nodes=[], edges=[], crossings=[]))
    return dict(network=S.true_network_observable(case))


def nothing(image, valid) -> dict:
    """The null pipeline (the floor): nothing fitted."""
    return dict(polylines=[], junctions=[], network=dict(nodes=[], edges=[], crossings=[]),
                od_render=np.zeros(np.shape(image), np.float32))


def run_probes(annotate=None, deadline: float | None = None, oracle: bool = False, only=None,
               probe_dir: str = PROBE_DIR, verbose: bool = False) -> dict:
    """The pipeline annotate(image, valid) on every probe (or the ORACLE): dict(probe_score, rows, tuning,
    digest, seconds).  deadline: a time.monotonic() value; past it the run raises TimeoutError (a crash)."""
    from experiments.splinefit import score as S
    from experiments.splinefit.fastset import KINDS
    t_start = time.monotonic()
    rows, h = [], hashlib.sha256()
    for st in load_battery(probe_dir):
        if st["kind"] not in KINDS or (only and st["sweep"] not in only and st["name"] not in only):
            continue
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"probe battery past the deadline at {st['name']}")
        case = S.load_truth(st["path"], st["kind"])
        t0 = time.perf_counter()
        out = oracle_out(case) if oracle else annotate(case["image"].copy(), case["valid"].copy())
        secs = time.perf_counter() - t0
        row = score_probe(st, case, out)
        row["seconds"] = secs
        row["digest"] = S.digest(out)
        h.update(row["digest"].encode())
        rows.append(row)
        if verbose:
            print(f"probe {st['name']:28s} {st['kind']:7s} composite {row['probe_composite']:.3f}  "
                  f"len {row['fitted_len_px']:6.1f}  junctions {row['junction_types_fit']}  "
                  f"truth {row['junction_types_truth']}  {secs:.1f} s", flush=True)
    score = float(np.mean([r["probe_composite"] for r in rows])) if rows else float("nan")
    return dict(probe_score=score, rows=rows, tuning=tuning(rows), digest=h.hexdigest()[:16],
                seconds=time.monotonic() - t_start)


def run(pipeline, **kw):
    """run_probes for a pipeline given as a callable or 'module:function': (rows, summary)."""
    if isinstance(pipeline, str):
        import importlib
        mod, fn = pipeline.split(":")
        pipeline = getattr(importlib.import_module(mod), fn)
    res = run_probes(pipeline, **kw)
    return res["rows"], dict(probe_score=res["probe_score"], digest=res["digest"], seconds=res["seconds"],
                             **res["tuning"])


# ======================================================================== tuning curves and thresholds
CURVE_KEYS = ("probe_composite", "graph_probe", "graph_score", "centreline_recall", "junction_f1_strict",
              "crossing_recall",
              "pos", "width", "explained", "explained_junction", "junction_bias", "width_bias", "blur_bias_px",
              "contrast_bias", "fitted_len_px", "n_lines_fit")


def _detected(r: dict) -> bool:
    return np.isfinite(r.get("centreline_recall", np.nan)) and r["centreline_recall"] >= DETECT_RECALL


def _typed_3way(r: dict) -> bool:
    return (r.get("junction_recall_strict", 0) or 0) >= 1.0 and (r.get("junction_type_accuracy_coarse", 0) or 0) >= 1.0


def tuning(rows: list) -> dict:
    """Per sweep: the curve (x, the CURVE_KEYS) and the threshold (SWEEPS)."""
    by = {}
    for r in rows:
        by.setdefault(r["sweep"], []).append(r)
    # the default isolated vessel (d 6, in focus, hct 1) anchors the blur and contrast sweeps
    base = next((r for r in rows if r["name"] == "line_d6"), None)
    curves, thr = {}, {}
    for sw, rs in by.items():
        rs = list(rs)
        if sw in ("blur", "contrast") and base is not None:
            rs = [dict(base, x=0.0 if sw == "blur" else 1.0)] + rs
        curves[sw] = [dict(name=r["name"], x=r["x"], kind=r["kind"],
                           **{k: r[k] for k in CURVE_KEYS + ("r_true_px", "s_true_px", "a_true") if k in r})
                      for r in rs]
        num = [r for r in rs if isinstance(r["x"], (int, float))]
        if sw == "width":
            ok = [r["x"] for r in num if _detected(r)]
            thr[sw] = min(ok) if ok else None
        elif sw == "blur":
            ok = [r["x"] for r in num if _detected(r) and r.get("width_rel_err", 9) < 0.25]
            thr[sw] = max(ok) if ok else None
        elif sw == "contrast":
            ok = [r["x"] for r in num if _detected(r)]
            thr[sw] = min(ok) if ok else None
        elif sw in ("crossing_angle", "crossing_depth"):
            ok = [r["x"] for r in num if (r.get("crossing_recall") or 0) >= 1.0]
            thr[sw] = min(ok) if ok else None
        elif sw == "gap":
            ok = [r["x"] for r in num if r.get("n_lines_fit") == 2]
            thr[sw] = min(ok) if ok else None
            thr["gap_truth"] = min([r["x"] for r in num if r.get("n_lines_truth") == 2] or [None],
                                   key=lambda v: (v is None, v))
        elif sw in ("fork", "t"):
            thr[sw] = f"{sum(_typed_3way(r) for r in rs)}/{len(rs)}"
        elif sw == "crossing_width":
            thr[sw] = f"{sum((r.get('crossing_recall') or 0) >= 1.0 for r in rs)}/{len(rs)}"
        elif sw == "end":
            thr[sw] = f"{sum(_detected(r) and r.get('n_false_junctions', 1) == 0 for r in rs)}/{len(rs)}"
        elif sw in ("corner", "rbc"):
            thr[sw] = f"{sum(_detected(r) for r in rs)}/{len(rs)}"
        elif sw == "empty":
            thr[sw] = f"{sum(r['fitted_len_px'] == 0 for r in rs)}/{len(rs)} empty; " \
                      f"max len {max(r['fitted_len_px'] for r in rs):.0f} px"
    sweep_mean = {sw: float(np.mean([r["probe_composite"] for r in rs])) for sw, rs in by.items()}
    return dict(curves=curves, thresholds=thr, sweep_mean=sweep_mean)


def _f(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "-" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def report(res: dict, title: str = "probe battery") -> str:
    """Markdown: the probe score, per sweep its mean, threshold and tuning-curve table."""
    tu = res["tuning"]
    cols = ("probe_composite", "graph_probe", "centreline_recall", "junction_f1_strict", "crossing_recall", "pos",
            "width", "explained", "explained_junction", "junction_bias", "fitted_len_px", "n_lines_fit")
    lines = [f"## {title}", "", f"probe_score {res['probe_score']:.4f}  digest {res['digest']}  "
                                f"{res['seconds']:.0f} s  ({len(res['rows'])} probes)", ""]
    lines += ["| sweep | mean | threshold |", "|---|---|---|"]
    for sw in tu["curves"]:
        lines.append(f"| {sw} | {tu['sweep_mean'].get(sw, float('nan')):.3f} | {tu['thresholds'].get(sw)} "
                     f"({SWEEPS.get(sw, ('', ''))[1]}) |")
    if "gap_truth" in tu["thresholds"]:
        lines.append(f"| gap (truth) | | {tu['thresholds']['gap_truth']} (smallest gap the observable truth "
                     "resolves) |")
    for sw, cv in tu["curves"].items():
        lines += ["", f"**{sw}** (x: {SWEEPS.get(sw, ('x',))[0]})", "",
                  "| name | x | kind | r/s/a true | " + " | ".join(cols) + " |",
                  "|" + "---|" * (4 + len(cols))]
        for r in cv:
            tr = "/".join(_f(r.get(k), 2) for k in ("r_true_px", "s_true_px", "a_true"))
            lines.append(f"| {r['name']} | {_f(r['x'])} | {r['kind']} | {tr} | "
                         + " | ".join(_f(r.get(k)) for k in cols) + " |")
    rows = {r["name"]: r for r in res["rows"]}
    lines += ["", "**junctions** (fitted vs truth types)", "", "| name | fitted | truth |", "|---|---|---|"]
    for nm, r in rows.items():
        if r.get("junction_types_fit") or r.get("junction_types_truth"):
            lines.append(f"| {nm} | {', '.join(r['junction_types_fit'])} | {', '.join(r['junction_types_truth'])} |")
    return "\n".join(lines) + "\n"


def plot(res: dict, path: str):
    """The tuning curves of the numeric sweeps (one panel each) as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = {sw: [r for r in c if isinstance(r["x"], (int, float))] for sw, c in res["tuning"]["curves"].items()}
    cv = {sw: c for sw, c in cv.items() if len(c) >= 2}
    keys = ("probe_composite", "graph_probe", "pos", "width", "explained", "explained_junction")
    n = max(1, len(cv))
    nc = min(3, n)
    nr = (n + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.9 * nr), squeeze=False)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for ax, (sw, c) in zip(axs.flat, cv.items()):
        c = sorted(c, key=lambda r: r["x"])
        x = [r["x"] for r in c]
        for k in keys:
            ax.plot(x, [r.get(k, np.nan) for r in c], marker="o", ms=3, lw=1.2, label=k)
        ax.set_title(sw, fontsize=9)
        ax.set_xlabel(SWEEPS.get(sw, ("x",))[0], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"probe_score {res['probe_score']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ======================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("generate", "run", "oracle", "nothing"))
    ap.add_argument("--pipeline", default="experiments.splinefit.pipeline:annotate")
    ap.add_argument("--out", help="JSON of the rows and tuning (default $SPLINEFIT_WORK/runs/probes_<cmd>.json)")
    ap.add_argument("--md", help="markdown report path (default: next to --out)")
    ap.add_argument("--plot", help="tuning-curve PNG (default: next to --out)")
    ap.add_argument("--only", nargs="*", help="sweeps or probe names")
    ap.add_argument("--force", action="store_true", help="generate: overwrite the frozen battery")
    a = ap.parse_args(argv)
    from experiments.splinefit import set_threads
    set_threads(2)
    if a.cmd == "generate":
        generate(force=a.force)
        return 0
    if a.cmd == "oracle":
        res = run_probes(oracle=True, only=a.only, verbose=True)
    else:
        if a.cmd == "nothing":
            fn = nothing
        else:
            import importlib
            mod, f = a.pipeline.split(":")
            fn = getattr(importlib.import_module(mod), f)
        res = run_probes(fn, only=a.only, verbose=True)
    out = a.out or os.path.join(WORK, "runs", f"probes_{a.cmd}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    md = report(res, f"probes: {a.cmd} {a.pipeline if a.cmd == 'run' else ''}")
    with open(a.md or os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    try:
        plot(res, a.plot or os.path.splitext(out)[0] + ".png")
    except Exception as exc:                                  # a plot is a convenience, not a result
        print(f"plot failed: {exc}", file=sys.stderr)
    print(md)
    print(f"probe_score: {res['probe_score']:.6f}")
    print(f"digest: {res['digest']}")
    print(f"json: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
