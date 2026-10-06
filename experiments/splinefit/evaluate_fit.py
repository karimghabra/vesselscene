r"""Reference rows of the fit scorer (score.py) on the dev images: what the fitter has to beat, and the ceiling.

    python -m experiments.splinefit.evaluate_fit [--tier 2|1] [--rows proposals vesselmap true_observable
        true_complete] [--pipeline [NAME=]module:function ...] [--oracle-fit [NAME=]module:function ...]
        [--out DIR] [--table-only]

Rows (tier 2: every dev scene's average still, fastset.KINDS; tier 1: the fast crops; vesselmap only on
tier 2):

  proposals        the neuromimetic annotator alone (experiments.neuromimetic.neuromimetic.run, stages 1-8) as
                   a network: its edges (cut at every junction) with the profiles of its fit_profiles (a
                   blurred box, peak a, half width r, blur s) converted to vesselmap's blurred cylinder chord
                   (``box_to_cyl``: half width x BOX_TO_CYL, contrast / the blurred chord's peak; the conversion
                   of proposals.py, copied so this row stays fixed while the fitter changes); every junction
                   a node of its type, free ends 'end' nodes; od_target its stage-1 OD; rendered by the scorer
                   (vesselmap's additive renderer).  A pipeline: ``proposals_annotate``, also usable with
                   run_experiment --pipeline experiments.splinefit.evaluate_fit:proposals_annotate;
  vesselmap        LIMBUS vesselmap's cached dev networks (experiments.neuromimetic.baseline_vesselmap, never
                   rebuilt here: an image without a cached network is skipped), its junctions typed by that
                   module's export (nodes and X crossings), r, s, a by net.sample, rendered by vesselmap with
                   its own fitted halo;
  true_observable  SELF-CHECK: the observable image graph with the true r, s, a (score.true_network_observable),
                   rendered by vesselmap: pos = width = 1 by construction, graph near 1;
  true_complete    ORACLE: the complete true network (score.true_network_complete, scene.to_vesselnetwork)
                   rendered by vesselmap with the scene's halo: the additive renderer's ceiling on explained
                   (its junction_bias is the over-prediction at forks), and the graph score of a network that
                   also holds the invisible vessels;
  pipelines        --pipeline [NAME=]module:function: any pipeline annotate(image, valid) (row NAME, default 'fit'
                   for experiments.splinefit.pipeline:annotate, else the function's name), e.g. the fitter;
  oracle rows      ORACLE hooks (--oracle-fit [NAME=]module:function): fn(image, valid, init) -> pipeline output,
                   init = the true_complete contract network plus 'halo' (the scene's (weight, sigma)); row
                   NAME, default 'oracle_init' for a function named fit, else 'oracle_<function>'.
                   experiments.splinefit.oracle: fit (the fitter started from the truth, the positive control),
                   render (the truth through the junction-aware render, the render model's ceiling),
                   render_additive (its sham).

Writes <out>/<row>.json (score rows, audit=True: render_agreement for pipeline renders) and <out>/table.md
(means per kind over every <row>.json in <out>, so rows run in separate invocations share one table;
--table-only rewrites it).  Truth is read only to score and for the clearly labelled self-check and oracle
rows.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                     # noqa: E402
import json                                         # noqa: E402
import math                                         # noqa: E402
import time                                         # noqa: E402

import numpy as np                                  # noqa: E402

from . import set_threads, use_vesselmap            # noqa: E402
from . import score as S                            # noqa: E402
from .run_experiment import WORK, _scalars, load_pipeline, tier_cases   # noqa: E402

ROWS = ("proposals", "vesselmap", "true_observable", "true_complete")
BOX_TO_CYL = 1.2            # a blurred box of half width r ~ a cylinder chord of half width 1.2 r (FWHM / area)
TABLE = ("composite", "graph_score", "pos", "width", "explained", "explained_junction", "junction_bias", "chi2",
         "explained_ref", "explained_true", "centreline_f1", "junction_f1_strict", "junction_type_balanced_coarse",
         "edge_cover", "geom_matched", "offset_px", "width_rel_err", "blur_err_px", "contrast_rel_err",
         "target_fidelity", "render_agreement", "seconds")
ORDER = ("proposals", "vesselmap", "fit", "true_observable", "true_complete", "oracle_render_additive",
         "oracle_render", "oracle_init")     # table order (other rows after these, by name)


# ======================================================================== proposals only
def _cyl_peak(r, s):
    """Peak of vesselmap's blurred cylinder chord (unit centre chord)."""
    from scipy.special import erf
    use_vesselmap()
    from vesselmap.render import PROFILE_C, PROFILE_W
    c, w = PROFILE_C.numpy().astype(float), PROFILE_W.numpy().astype(float)
    r, s = np.asarray(r, float)[..., None], np.asarray(s, float)[..., None]
    return (erf(r * c / (math.sqrt(2.0) * s)) * w).sum(-1)


def box_to_cyl(a, r, s):
    """Blurred-box profile (peak OD a, half width r, blur s) -> vesselmap cylinder parameters (r, s, a)."""
    rc = np.maximum(BOX_TO_CYL * np.asarray(r, float), 0.4)
    s = np.maximum(np.asarray(s, float), 0.65)
    ac = np.maximum(np.asarray(a, float), 1e-3) / np.maximum(_cyl_peak(rc, s), 0.05)
    return rc, s, ac


def proposals_annotate(image, valid):
    """The neuromimetic annotator's own graph and profiles as a contract network (module docstring)."""
    from experiments.neuromimetic import neuromimetic as NM
    dbg = {}
    res = NM.run(image, valid, NM.DEFAULT, dbg)
    js = dbg["junctions"]
    nodes = [dict(id=i, x=float(J["x"]), y=float(J["y"]), kind=J["type"]) for i, J in enumerate(js)]
    edges = []
    for e in dbg["edges"]:
        xy = np.asarray(e["xy"], float)
        if len(xy) < 2:
            continue
        uv = []
        for end, ji in enumerate(e["j"]):
            if ji is None:                             # a free end: its own node
                p = xy[0] if end == 0 else xy[-1]
                nodes.append(dict(id=len(nodes), x=float(p[0]), y=float(p[1]), kind="end"))
                uv.append(len(nodes) - 1)
            else:
                uv.append(int(ji))
        r, s, a = box_to_cyl(e["pa"], e["pr"], e["ps"])
        edges.append(dict(u=uv[0], v=uv[1], xy=xy, r=r, s=s, a=a))
    return dict(res, network=dict(nodes=nodes, edges=edges, crossings=[]), od_target=dbg["s1"]["OD"])


# ======================================================================== vesselmap (cached)
def vesselmap_output(image) -> dict | None:
    """The cached vesselmap network of a dev image as a contract output with its own render (or None)."""
    from experiments.neuromimetic import baseline_vesselmap as BV
    path = os.path.join(BV.CACHE_DIR, BV._key(image) + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        res = json.load(fh)
    use_vesselmap()
    from vesselmap.network import VesselNetwork
    net = VesselNetwork.from_dict(res["net"])
    ex = BV.export(net)
    deg = net.degrees()
    hubs = [nid for nid in sorted(net.nodes) if deg[nid] >= 3]
    kinds = {nid: j[2] for nid, j in zip(hubs, ex["junctions"])}      # export lists the hubs first, in order
    cr = [dict(x=j[0], y=j[1], e1=None, e2=None) for j in ex["junctions"][len(hubs):]]
    out = dict(network=S.vesselmap_to_contract(net, kinds, cr),
               od_render=S.render_network(net, image.shape), build_seconds=res["seconds"])
    return out


# ======================================================================== rows
def _row(case, out, name, secs) -> dict:
    s = _scalars(S.score(case, out, audit=True))
    s.update(row=name, scene=os.path.basename(case["folder"]), kind=case["kind"], crop=case.get("crop"),
             seconds=secs, digest=S.digest(out))
    for k in ("kappa", "propose_seconds", "fit_seconds"):
        if isinstance(out.get(k), (int, float)):
            s[k] = float(out[k])
    if out.get("network") is not None:
        s["n_edges"] = len(out["network"].get("edges", []))
        s["n_crossings"] = len(out["network"].get("crossings", []))
    return s


def _named(specs, default_name) -> list:
    """[NAME=]module:function specs -> [(name, spec)]."""
    out = []
    for sp in specs or []:
        name, _, fn = sp.rpartition("=")
        out.append((name or default_name(fn), fn))
    return out


def _pipeline_name(spec: str) -> str:
    return "fit" if spec == "experiments.splinefit.pipeline:annotate" else spec.split(":")[1]


def _oracle_name(spec: str) -> str:
    fn = spec.split(":")[1]
    return "oracle_init" if fn == "fit" else f"oracle_{fn}"


def evaluate(tier: int, rows, oracle_fit, out_dir: str, pipelines=None) -> dict:
    set_threads(2)
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(oracle_fit, str):
        oracle_fit = [oracle_fit]
    pipes = {n: load_pipeline(f) for n, f in _named(pipelines, _pipeline_name)}
    oracles = {n: load_pipeline(f) for n, f in _named(oracle_fit, _oracle_name)}
    names = list(rows) + list(pipes) + list(oracles)
    res = {n: [] for n in names}
    for folder, kind, crop in tier_cases(tier):
        case = S.load_truth(folder, kind, crop)
        img, val = case["image"], case["valid"]
        tn = None
        for n in names:
            t0 = time.perf_counter()
            if n in pipes:
                out = pipes[n](img.copy(), val.copy())
            elif n == "proposals":
                out = proposals_annotate(img.copy(), val.copy())
            elif n == "vesselmap":
                if crop is not None:
                    continue
                out = vesselmap_output(img)
                if out is None:
                    print(f"vesselmap: no cached network for {os.path.basename(folder)} {kind}: skipped")
                    continue
            elif n == "true_observable":
                out = dict(network=S.true_network_observable(case))
            elif n == "true_complete" or n in oracles:
                if tn is None:
                    tn = S.true_network_complete(folder, kind, img.shape, crop)
                    vn = tn.pop("vesselnetwork")
                    o = vn.meta["optics"]
                    tn["halo"] = (float(o["halo_weight"]), float(o["halo_sigma"]))
                if n == "true_complete":
                    out = dict(network={k: tn[k] for k in ("nodes", "edges", "crossings")},
                               od_render=S.render_network(vn, img.shape), halo=tn["halo"])
                else:
                    out = oracles[n](img.copy(), val.copy(), dict(tn))
            secs = time.perf_counter() - t0
            r = _row(case, out, n, secs)
            res[n].append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                              if k in ("row", "scene", "kind", "composite", "graph_score", "pos", "width",
                                       "explained", "explained_junction", "junction_bias", "seconds")}), flush=True)
            # each row's JSON is rewritten after every image, so a long run leaves its partial rows
            with open(os.path.join(out_dir, f"{n}.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(row=n, tier=tier, rows=res[n]), fh, indent=1, default=float)
    md = write_table(out_dir)
    print(md)
    return res


def load_rows(out_dir: str) -> dict:
    """Every <row>.json in out_dir, in table order."""
    res = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json"):
            with open(os.path.join(out_dir, f), encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "rows" in d and "row" in d:
                res[d["row"]] = d["rows"]
    rank = {n: i for i, n in enumerate(ORDER)}
    return dict(sorted(res.items(), key=lambda kv: (rank.get(kv[0], len(ORDER)), kv[0])))


def write_table(out_dir: str) -> str:
    md = table(load_rows(out_dir))
    with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return md


def table(res: dict) -> str:
    """Markdown: per kind, the mean of each TABLE score per row (n images in brackets)."""
    from .fastset import KINDS
    out = []
    for kind in KINDS:
        out.append(f"\n**{kind}** (mean over images)\n")
        out.append("| row | n | " + " | ".join(TABLE) + " |")
        out.append("|---" * (len(TABLE) + 2) + "|")
        for n, rr in res.items():
            rr = [r for r in rr if r["kind"] == kind]
            if not rr:
                continue
            cells = []
            for k in TABLE:
                v = [r.get(k) for r in rr if isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k))]
                cells.append(f"{np.mean(v):.3f}" if v else "-")
            out.append(f"| {n} | {len(rr)} | " + " | ".join(cells) + " |")
    # per image (rows may cover different images, e.g. vesselmap only where a cached network exists)
    imgs = sorted({(r["scene"], r["kind"], tuple(r["crop"]) if r.get("crop") else None)
                   for rr in res.values() for r in rr}, key=lambda t: (t[1], t[0], t[2] or ()))
    for key in ("composite", "graph_score", "explained_junction"):
        out.append(f"\n**{key} per image**\n")
        out.append("| row | " + " | ".join(f"{sc} {k[0]}" + (f" {list(c)[:2]}" if c else "") for sc, k, c in imgs)
                   + " |")
        out.append("|---" * (len(imgs) + 1) + "|")
        for n, rr in res.items():
            by = {(r["scene"], r["kind"], tuple(r["crop"]) if r.get("crop") else None): r.get(key) for r in rr}
            out.append(f"| {n} | " + " | ".join(f"{by[i]:.3f}" if isinstance(by.get(i), (int, float)) else "-"
                                                for i in imgs) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", type=int, choices=(1, 2), default=2)
    ap.add_argument("--rows", nargs="*", default=list(ROWS), choices=ROWS)
    ap.add_argument("--pipeline", nargs="+", help="[NAME=]module:function annotate(image, valid) -> output")
    ap.add_argument("--oracle-fit", nargs="+", help="[NAME=]module:function fit(image, valid, init_network) -> output")
    ap.add_argument("--out")
    ap.add_argument("--table-only", action="store_true", help="only rewrite <out>/table.md from its row JSONs")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(WORK, "reference", f"tier{a.tier}")
    if a.table_only:
        print(write_table(out))
        return
    evaluate(a.tier, a.rows, a.oracle_fit, out, a.pipeline)


if __name__ == "__main__":
    main()
