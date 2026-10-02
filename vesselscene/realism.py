"""The real reference, the realism scorecard, and comparison figures.

A synthetic scene is judged against the real stills only through the
annotation-free statistics of vesselscene.stats, measured the same way on
both.  The real side is data/real_stats.json, built once by
`build_real_stats()` (python -m vesselscene.realism build) from

    stills  the 9 reference stabilized average stills of the plan, one per
            burst (nonrigid where it exists); excluded: 12-57-20 (a 100-row
            strip) and 15-40-55 (stability 0.43, its statistics are
            registration artefacts); flagged but kept: 15-39-40 (14 frames,
            glare).  About 6 independent sites, one participant, two days.
    frames  single raw frames (3 per burst, at 25/50/75 % of the burst) from
            3 bursts, for single-frame synthetic scenes: noisier and "less
            full" than the averages.  Saturated pixels (4095 DN) are invalid.

Rule (plan section 1): a scalar statistic passes when the synthetic MEAN
lies inside the real [min, max] AND |z| <= 2, z = (synthetic mean - real
mean) / real SD (between stills, ddof=1).  A distribution or spectrum
passes when the mean synthetic-to-real distance is at most the 95th
percentile of the leave-one-out real-to-real distances.  With only 9 real
stills a single left-out real still falls outside the others' [min, max]
for about 2 statistics in 9 by chance alone (probability 2/9 each), so the
rule is meant for the mean of >= 20 synthetic scenes, not for one image.

Note on contrast: the camera applies a gamma (0.5, 0.8 or 1.0 depending on
the burst; stored per record under "camera"), so F = ln I - B of a real
still is gamma times the optical density; contrast statistics of the stills
include that factor.
"""
from __future__ import annotations

import datetime
import glob
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stats as st

DATA_PATH = Path(__file__).with_name("data") / "real_stats.json"
FORMAT = "vesselscene real_stats v1"

REFERENCE_STILLS = (
    ("09-16_15-22-26", "nonrigid"), ("09-16_15-30-33", "translation"), ("09-16_15-31-37", "nonrigid"),
    ("09-16_15-31-50", "nonrigid"), ("09-16_15-39-40", "translation"), ("09-16_15-50-30", "translation"),
    ("09-16_15-50-36", "translation"), ("09-16_15-50-52", "nonrigid"), ("09-17_13-51-26", "translation"),
)
EXCLUDED_STILLS = {
    "09-16_12-57-20": "a 100-row strip (1920x100 ROI)",
    "09-16_15-40-55": "stability index 0.43: its statistics are registration artefacts (FWHM p10 1.6 px)",
}
FLAGGED_STILLS = {"09-16_15-39-40": "averaged from 14 usable frames and has glare; kept"}
SITES = {"09-16_15-50-30": "A", "09-16_15-50-36": "A", "09-16_15-50-52": "A",
         "09-16_15-31-37": "B", "09-16_15-31-50": "B"}
FRAME_BURSTS = ("09-16_15-50-52", "09-16_15-31-37", "09-16_15-22-26")
FRAME_FRACTIONS = (0.25, 0.5, 0.75)
STILL_SEED, FRAME_SEED = 1, 1
SATURATED_DN = 4095

# plot tokens (validated categorical order of the dataviz reference palette)
REAL, SYN, SYN2, SYN3 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SCALE_COLORS = ("#e34948", "#eda100", "#1baf7a", "#2a78d6", "#4a3aa7")   # sigma* = 1..16 (as fig2)
VMIN, VMAX = -0.35, 0.08          # common display range of F (Np)


# ------------------------------------------------------------------ data
def find_data_root() -> Path | None:
    """The LIMBUS checkout holding stabilization/ and reference_data/ (the
    real stills; not part of this repository): $LIMBUS_DATA, else a sibling
    folder 'limbus' of this repository.  None when absent: whatever needs
    the real images then skips or asks for a path."""
    here = Path(__file__).resolve().parents[1]
    cands = [os.environ.get("LIMBUS_DATA"), here, here.parent / "limbus"]
    for c in cands:
        if c and (Path(c) / "stabilization").is_dir():
            return Path(c)
    return None


def still_path(root, burst, method) -> Path:
    return Path(root) / "stabilization" / method / f"burst_2026-{burst}" / "mean_stabilized.tif"


def _camera(root, burst) -> dict:
    """Camera gamma / gain / exposure of a burst from its recording manifest."""
    for sub in ("recordings", "reference_data"):
        p = Path(root) / sub / f"burst_2026-{burst}" / "manifest.json"
        if p.is_file():
            cs = json.loads(p.read_text()).get("camera_state_at_start", {})
            return {k: cs.get(k) for k in ("gamma", "gain", "exposure_us", "gain_auto", "exposure_auto")}
    return {}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def _from_json(rec):
    """A record read back from JSON: null -> NaN in numbers, lists -> arrays."""
    nan = float("nan")
    rec["scalars"] = {k: nan if v is None else float(v) for k, v in rec["scalars"].items()}
    rec["quantiles"] = {k: None if v is None else np.asarray(v, float) for k, v in rec["quantiles"].items()}
    for name in ("psd", "pairs"):
        if rec.get(name):
            rec[name] = {k: np.asarray(v, float) if isinstance(v, list) else v for k, v in rec[name].items()}
    for name in ("bg_band", "orient_hist"):
        rec[name] = np.asarray(rec[name], float)
    return rec


def record(res, **meta) -> dict:
    """The comparable content of an analyse() result: scalars, the curves
    (spectrum, pair correlations, background bands, orientation histogram)
    and the quantile functions of the distributions.  Everything the
    scorecard and the figures need, small enough for JSON."""
    if "scalars" in res:                       # already a record
        return res
    ps, pp = res.get("psd"), res.get("pairs")
    rec = dict(meta)
    rec.update(
        shape=list(res["shape"]),
        scalars=st.scalars(res),
        psd=None if ps is None else dict(f=np.asarray(ps["f"]), psd=np.asarray(ps["psd"]), tile=ps["tile"]),
        pairs=None if pp is None else dict(v=pp["v"], g_par=pp["g_par"], g_perp=pp["g_perp"]),
        bg_band=[res["bg_band"][s]["rms"] for s in st.SCALES],
        orient_hist=np.asarray(res["orient_hist"]),
        quantiles=st.quantile_functions(res),
    )
    return rec


def summarise(records) -> dict:
    """{key: dict(mean, sd, min, max, n)} over the records' scalars (NaN
    ignored; sd with ddof=1)."""
    keys = sorted({k for r in records for k in r["scalars"]})
    out = {}
    for k in keys:
        v = np.array([r["scalars"].get(k, np.nan) for r in records], float)
        v = v[np.isfinite(v)]
        out[k] = dict(mean=float(v.mean()) if len(v) else np.nan, sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                      min=float(v.min()) if len(v) else np.nan, max=float(v.max()) if len(v) else np.nan, n=int(len(v)))
    return out


def build_real_stats(root=None, out=DATA_PATH, frames=True, verbose=True) -> dict:
    """Measure the reference stills (and single frames) and write the JSON.
    About 1-2 minutes."""
    import tifffile
    root = Path(root) if root else find_data_root()
    if root is None:
        raise FileNotFoundError("no LIMBUS data root with stabilization/ (set LIMBUS_DATA)")
    t_all = time.time()
    stills = {}
    for burst, method in REFERENCE_STILLS:
        p = still_path(root, burst, method)
        I = tifffile.imread(p).astype(np.float64)
        t0 = time.time()
        res = st.analyse(I, seed=STILL_SEED)
        dt = time.time() - t0
        n_frames = None
        mj = p.parent / "metrics.json"
        if mj.is_file():
            n_frames = json.loads(mj.read_text()).get("frames", {}).get("registered")
        stills[burst] = record(res, name=burst, kind="still", method=method, site=SITES.get(burst, burst),
                               source=str(p.relative_to(root)).replace("\\", "/"), seed=STILL_SEED,
                               seconds=round(dt, 2), n_frames=n_frames, camera=_camera(root, burst),
                               flag=FLAGGED_STILLS.get(burst))
        if verbose:
            print(f"still {burst} {method} {I.shape} {dt:.1f}s", flush=True)
    doc = dict(format=FORMAT, stats_version=st.STATS_VERSION,
               created=datetime.datetime.now().isoformat(timespec="seconds"),
               q_levels=st.Q_LEVELS, distributions=list(st.DISTRIBUTIONS),
               rule="scalar: synthetic mean in real [min, max] and |z| <= 2 (z vs between-still SD); "
                    "distribution/spectrum: mean synthetic distance <= p95 of leave-one-out real-real distances",
               stills=dict(reference=[b for b, _ in REFERENCE_STILLS], excluded=EXCLUDED_STILLS,
                           flagged=FLAGGED_STILLS, sites=SITES, records=stills,
                           summary=summarise(list(stills.values()))))
    if frames:
        frecs = {}
        for burst in FRAME_BURSTS:
            fs = sorted(glob.glob(str(Path(root) / "reference_data" / f"burst_2026-{burst}" / "frame_*.tif")))
            if not fs:
                if verbose:
                    print(f"frames of {burst}: none found", flush=True)
                continue
            for fr in FRAME_FRACTIONS:
                f = fs[int(round(fr * (len(fs) - 1)))]
                I = tifffile.imread(f).astype(np.float64)
                valid = (I > 0) & (I < SATURATED_DN)
                t0 = time.time()
                res = st.analyse(I, valid=valid, seed=FRAME_SEED)
                dt = time.time() - t0
                name = f"{burst}/{Path(f).stem}"
                frecs[name] = record(res, name=name, kind="frame", site=SITES.get(burst, burst),
                                     source=str(Path(f).relative_to(root)).replace("\\", "/"), seed=FRAME_SEED,
                                     seconds=round(dt, 2), saturated_frac=float((I >= SATURATED_DN).mean()),
                                     camera=_camera(root, burst))
                if verbose:
                    print(f"frame {name} {I.shape} {dt:.1f}s", flush=True)
        doc["frames"] = dict(bursts=list(FRAME_BURSTS), fractions=list(FRAME_FRACTIONS), records=frecs,
                             summary=summarise(list(frecs.values())),
                             note="single raw frames (unregistered, gamma-encoded DN); SD is between frames "
                                  "of 3 bursts, so it mixes frame-to-frame and site-to-site variation")
    doc["seconds_total"] = round(time.time() - t_all, 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(doc), indent=1))
    if verbose:
        print(f"wrote {out} in {doc['seconds_total']} s")
    return doc


def load_real(kind="stills", exclude=(), path=DATA_PATH) -> dict:
    """The real reference: dict(kind, names, records {name: record},
    summary {key: {mean, sd, min, max, n}}, q_levels).  kind = 'stills'
    (average stills) or 'frames' (single raw frames); `exclude` drops
    records by name (the summary is recomputed), e.g. to score a real still
    against the others."""
    doc = json.loads(Path(path).read_text())
    sec = doc[kind]
    unknown = set(exclude) - set(sec["records"])
    if unknown:
        raise KeyError(f"not in the {kind} reference: {sorted(unknown)}")
    recs = {k: _from_json(v) for k, v in sec["records"].items() if k not in set(exclude)}
    summ = summarise(list(recs.values())) if exclude else {
        k: {s: float("nan") if v is None else v for s, v in d.items()} for k, d in sec["summary"].items()}
    return dict(kind=kind, names=list(recs), records=recs, summary=summ, q_levels=np.asarray(doc["q_levels"]),
                stats_version=doc["stats_version"], excluded=list(exclude))


# ------------------------------------------------------------------ scorecard
@dataclass
class Scorecard:
    """Result of `scorecard`: one row per scalar statistic, one per
    distribution / spectrum band.  Row fields: key, label, tier,
    real_mean, real_sd, real_min, real_max, n_real, syn_mean, syn_sd,
    syn_min, syn_max, n_syn, z, inside, passed (None = not scoreable)."""
    rows: list
    dist_rows: list
    reference: str
    n_synthetic: int
    z_max: float = 2.0
    notes: list = field(default_factory=list)

    def scored(self, dist=False):
        rr = self.dist_rows if dist else self.rows
        return [r for r in rr if r["passed"] is not None]

    @property
    def n_pass(self):
        return sum(r["passed"] for r in self.scored())

    @property
    def n_scored(self):
        return len(self.scored())

    def failures(self):
        return [r for r in self.scored() if not r["passed"]]

    def n_far(self, z_lim=4.0):
        """Number of scored statistics with |z| > z_lim.  A left-out real
        still has 0-3 of these (8 for the 14-frame still 15-39-40); the old
        vesselmap synthetic scenes have 17-23.  The pass count alone
        separates them less well (real 58-89 % vs 45-56 %) because with 9
        reference stills a real still misses the others' [min, max] often."""
        return int(sum(bool(np.isnan(r["z"]) or abs(r["z"]) > z_lim) for r in self.scored()))

    def summary(self) -> dict:
        out = {"scalars": (self.n_pass, self.n_scored), "|z|>4": (self.n_far(), self.n_scored)}
        for t in (1, 2, 3):
            s = [r for r in self.scored() if r["tier"] == t]
            out[f"tier {t}"] = (sum(r["passed"] for r in s), len(s))
        d = self.scored(dist=True)
        out["distributions+spectra"] = (sum(r["passed"] for r in d), len(d))
        return out

    def markdown(self, only_failures=False, sort="tier") -> str:
        f = _fmt
        L = [f"Realism scorecard: {self.n_synthetic} synthetic image(s) vs {self.reference}", ""]
        s = self.summary()
        L.append("Passed: " + "; ".join(f"{k} {a}/{b}" for k, (a, b) in s.items()))
        L += [f"Rule: synthetic mean inside real [min, max] and |z| <= {self.z_max:g}.", ""]
        L.append("| tier | statistic | real mean +- sd [min, max] | synthetic mean [min, max] | z | pass |")
        L.append("|---|---|---|---|---|---|")
        rows = self.rows if not only_failures else self.failures()
        if sort == "z":
            rows = sorted(rows, key=lambda r: -abs(np.nan_to_num(r["z"], nan=0.0, posinf=1e9, neginf=1e9)))
        else:
            rows = sorted(rows, key=lambda r: (r["tier"], -abs(np.nan_to_num(r["z"], nan=0.0, posinf=1e9,
                                                                             neginf=1e9))))
        for r in rows:
            ok = {True: "pass", False: "**FAIL**", None: "n/a"}[r["passed"]]
            L.append(f"| {r['tier']} | {r['label']} (`{r['key']}`) | {f(r['real_mean'])} +- {f(r['real_sd'])} "
                     f"[{f(r['real_min'])}, {f(r['real_max'])}] | {f(r['syn_mean'])} [{f(r['syn_min'])}, "
                     f"{f(r['syn_max'])}] | {r['z']:+.1f} | {ok} |")
        if self.dist_rows:
            L += ["", "| distribution / spectrum band | real-real distance median [p95] | synthetic distance "
                      "(mean) | ratio to p95 | bias | pass |", "|---|---|---|---|---|---|"]
            for r in self.dist_rows:
                ok = {True: "pass", False: "**FAIL**", None: "n/a"}[r["passed"]]
                L.append(f"| {r['label']} | {f(r['rr_med'])} [{f(r['rr_p95'])}] | {f(r['syn_mean'])} | "
                         f"{r['ratio']:.2f} | {f(r['bias'])} | {ok} |")
        L += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(L)

    def to_csv(self, path):
        import csv
        keys = ["key", "label", "tier", "real_mean", "real_sd", "real_min", "real_max", "n_real", "syn_mean",
                "syn_sd", "syn_min", "syn_max", "n_syn", "z", "inside", "passed"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def _fmt(x):
    if x is None or not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


PSD_BANDS = (("PSD 1/128-1/16 c/px", 1 / 128, 1 / 16), ("PSD 1/16-1/4 c/px", 1 / 16, 1 / 4),
             ("PSD 1/4-1/2 c/px", 1 / 4, 0.51))


def _loo_mean(arrs, i):
    return np.nanmean([a for j, a in enumerate(arrs) if j != i], 0)


def scorecard(stats_list, real=None, keys=None, z_max=2.0) -> Scorecard:
    """Score synthetic images against the real reference.

    stats_list: analyse() results or records (or one of them).  real:
    load_real(...) (default: the 9 reference stills).  keys: scalar keys to
    score (default: every key of stats.LABELS).  Returns a Scorecard; use
    .markdown() for the table and figure_scorecard() for the chart."""
    if isinstance(stats_list, dict):
        stats_list = [stats_list]
    syn = [record(r) for r in stats_list]
    real = real if real is not None else load_real()
    recs = list(real["records"].values())
    rows = []
    for key in keys or list(st.LABELS):
        label, tier = st.LABELS.get(key, (key, 0))
        rv = np.array([r["scalars"].get(key, np.nan) for r in recs], float)
        sv = np.array([r["scalars"].get(key, np.nan) for r in syn], float)
        rv, svf = rv[np.isfinite(rv)], sv[np.isfinite(sv)]
        row = dict(key=key, label=label, tier=tier, n_real=len(rv), n_syn=len(svf))
        nan = float("nan")
        row.update(real_mean=rv.mean() if len(rv) else nan, real_sd=rv.std(ddof=1) if len(rv) > 1 else nan,
                   real_min=rv.min() if len(rv) else nan, real_max=rv.max() if len(rv) else nan,
                   syn_mean=svf.mean() if len(svf) else nan, syn_sd=svf.std(ddof=1) if len(svf) > 1 else nan,
                   syn_min=svf.min() if len(svf) else nan, syn_max=svf.max() if len(svf) else nan)
        if len(rv) < 3:
            row.update(z=nan, inside=None, passed=None)            # not defined on the real stills
        elif not len(svf):
            row.update(z=nan, inside=False, passed=False)          # defined on real, undefined on synthetic
        else:
            d = row["syn_mean"] - row["real_mean"]
            sd = row["real_sd"]
            z = d / sd if sd > 0 else (0.0 if d == 0 else math.copysign(math.inf, d))
            inside = bool(row["real_min"] <= row["syn_mean"] <= row["real_max"])
            row.update(z=float(z), inside=inside, passed=bool(inside and abs(z) <= z_max))
        rows.append(row)
    # distributions: W1 between quantile functions (mean |dQ| over the levels)
    dist_rows = []
    for name in st.DISTRIBUTIONS:
        Qr = [r["quantiles"].get(name) for r in recs]
        Qr = [q for q in Qr if q is not None]
        Qs = [r["quantiles"].get(name) for r in syn]
        Qs = [q for q in Qs if q is not None]
        row = dict(label=f"W1 of {name}", key=f"dist:{name}")
        if len(Qr) < 3:
            row.update(rr_med=np.nan, rr_p95=np.nan, syn_mean=np.nan, ratio=np.nan, bias=np.nan, passed=None)
        else:
            rr = [float(np.mean(np.abs(q - _loo_mean(Qr, i)))) for i, q in enumerate(Qr)]
            bar = np.mean(Qr, 0)
            sd = [float(np.mean(np.abs(q - bar))) for q in Qs]
            p95 = float(np.quantile(rr, 0.95))
            m = float(np.mean(sd)) if sd else np.nan
            row.update(rr_med=float(np.median(rr)), rr_p95=p95, syn_mean=m, ratio=m / p95 if p95 > 0 else np.nan,
                       bias=float(np.mean([np.mean(q - bar) for q in Qs])) if Qs else np.nan,
                       passed=bool(np.isfinite(m) and m <= p95))
        dist_rows.append(row)
    # spectra: RMS of the log10 PSD difference per band
    Pr = [np.log10(r["psd"]["psd"]) for r in recs if r.get("psd")]
    Ps = [np.log10(r["psd"]["psd"]) for r in syn if r.get("psd")]
    if len(Pr) >= 3:
        f = np.asarray(next(r["psd"]["f"] for r in recs if r.get("psd")), float)
        for label, lo, hi in PSD_BANDS:
            m = (f >= lo) & (f < hi)
            rr = [float(np.sqrt(np.nanmean((p[m] - _loo_mean(Pr, i)[m]) ** 2))) for i, p in enumerate(Pr)]
            mu = np.nanmean(Pr, 0)
            sd = [float(np.sqrt(np.nanmean((p[m] - mu[m]) ** 2))) for p in Ps if len(p) == len(mu)]
            p95 = float(np.quantile(rr, 0.95))
            mm = float(np.mean(sd)) if sd else np.nan
            dist_rows.append(dict(label=f"{label}, RMS log10 ratio (dex)", key=f"psd:{lo:g}-{hi:g}",
                                  rr_med=float(np.median(rr)), rr_p95=p95, syn_mean=mm,
                                  ratio=mm / p95 if p95 > 0 else np.nan,
                                  bias=float(np.mean([np.nanmean(p[m] - mu[m]) for p in Ps if len(p) == len(mu)]))
                                  if sd else np.nan, passed=bool(np.isfinite(mm) and mm <= p95)))
    ref = f"{len(recs)} real {real['kind']} ({', '.join(real['names'])})"
    notes = []
    shapes = {tuple(r["shape"]) for r in syn}
    rshapes = {tuple(r["shape"]) for r in recs}
    if not shapes <= rshapes:
        notes.append(f"synthetic image sizes {sorted(shapes)} differ from the real ones {sorted(rshapes)}: "
                     "field-of-view dependent statistics (FWHM p90, anisotropy, loop counts) are biased.")
    if len(syn) < 20:
        notes.append(f"only {len(syn)} synthetic image(s); the plan asks for >= 20 per configuration.")
    return Scorecard(rows=rows, dist_rows=dist_rows, reference=ref, n_synthetic=len(syn), z_max=z_max, notes=notes)


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "figure.facecolor": "white", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False})
    return plt


def _maps(img, valid, res):
    if res is not None and "maps" in res:
        return res
    return st.analyse(img, valid=valid, seed=STILL_SEED, keep_maps=True)


def _show_F(ax, F, title, valid=None):
    if valid is not None:
        F = np.where(valid, F, np.nan)
    ax.imshow(np.clip(F, VMIN, VMAX), cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="antialiased")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def figure_side_by_side(real_img, syn_img, path, real_valid=None, syn_valid=None, real_res=None, syn_res=None,
                        crops=((0.5, 0.3), (0.5, 0.7)), crop_size=(200, 320), titles=("real", "synthetic"),
                        dpi=110):
    """Real and synthetic images side by side on the SAME flattened grey
    scale (F = ln I - background, VMIN..VMAX Np): the full field, two
    zoomed crops (centres as fractions (fy, fx) of each image, size in px,
    boxes drawn on the full view), the multi-scale ridge skeleton coloured
    by best scale sigma* (as fig2), and the faint-mesh skeleton with the
    loops it closes.  real_res / syn_res: analyse(..., keep_maps=True)
    results to reuse (computed if absent).  Returns path."""
    plt = _style()
    from matplotlib.patches import Rectangle
    rr = _maps(real_img, real_valid, real_res)
    rs = _maps(syn_img, syn_valid, syn_res)
    H, W = rr["maps"]["F"].shape
    ch, cw = crop_size
    hr = np.array([H / W] + [ch / cw] * len(crops) + [H / W, H / W])
    # size the figure so every row's box has exactly its image aspect (no dead space)
    fw, left, right, wsp, gap, top_in, bot_in = 15.0, 0.01, 0.99, 0.03, 0.62, 0.3, 0.75
    colw = fw * (right - left) / (2 + wsp)
    rows_in = colw * hr
    fh = rows_in.sum() + gap * (len(hr) - 1) + top_in + bot_in
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(len(hr), 2, height_ratios=hr, hspace=gap / rows_in.mean(), wspace=wsp,
                          left=left, right=right, top=1 - top_in / fh, bottom=bot_in / fh)
    for j, (res, name) in enumerate(((rr, titles[0]), (rs, titles[1]))):
        mp = res["maps"]
        F, valid = mp["F"], mp["valid"]
        h, w = F.shape
        ax = fig.add_subplot(gs[0, j])
        _show_F(ax, F, f"{name}\n{h}x{w} px, F = ln I - background on {VMIN:g}..{VMAX:+g} Np (same grey scale)",
                valid)
        boxes = []
        for (fy, fx) in crops:
            y0 = int(np.clip(fy * h - ch / 2, 0, max(h - ch, 0)))
            x0 = int(np.clip(fx * w - cw / 2, 0, max(w - cw, 0)))
            boxes.append((y0, x0))
        for i, (y0, x0) in enumerate(boxes):
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), cw, ch, fill=False, ec=SYN, lw=1.2))
            ax.text(x0 + 4, y0 + 4, str(i + 1), color=SYN, fontsize=9, va="top", fontweight="bold")
            axc = fig.add_subplot(gs[1 + i, j])
            _show_F(axc, F[y0:y0 + ch, x0:x0 + cw], f"{name}: crop {i + 1} at y{y0} x{x0}, {ch}x{cw} px",
                    valid[y0:y0 + ch, x0:x0 + cw])
        # skeleton marker: 1.5 image px, but never below 1.3 output px (smaller squares are dropped)
        ms = max(1.5 * colw * 72 / w, 1.3 * 72 / dpi) ** 2
        axs = fig.add_subplot(gs[1 + len(crops), j])
        _show_F(axs, F, "ridge skeleton coloured by best scale sigma*; circles = detected crossings\n"
                        f"vessel frac {res['vessel_frac']:.2f}, skeleton {res['skel_density']:.1f} px/1e3 px2, "
                        f"lambda {res['lam']:.1f} px, {res['cross_density']:.1f} crossings/1e5 px2", valid)
        ys, xs = np.nonzero(mp["S"])
        kk = mp["k"][ys, xs]
        for i, c in enumerate(SCALE_COLORS):
            m = kk == i
            axs.scatter(xs[m], ys[m], s=ms, c=c, marker="s", linewidths=0, rasterized=True)
        cy = np.asarray(mp["cross_yx"]).reshape(-1, 2)
        axs.scatter(cy[:, 1], cy[:, 0], s=40, facecolors="none", edgecolors=INK, linewidths=0.9)
        axf = fig.add_subplot(gs[2 + len(crops), j])
        _show_F(axf, F, "faint-mesh skeleton (blue main, orange faint-only); closed loops shaded\n"
                        f"{res['faint_loop_density']:.2f} loops/1e5 px2 (main only {res['loop_density']:.2f}), "
                        f"faint-only {res['faint_extra_density']:.1f} px/1e3 px2", valid)
        lm = mp["faint_loops"]
        axf.imshow(np.ma.masked_where(~lm, np.ones_like(lm, float)), cmap="Blues", vmin=0, vmax=2.5, alpha=0.35,
                   interpolation="nearest")
        ys, xs = np.nonzero(mp["faint_S"] & ~mp["faint_extra"])
        axf.scatter(xs, ys, s=ms, c=REAL, marker="s", linewidths=0, rasterized=True)
        ys, xs = np.nonzero(mp["faint_extra"])
        axf.scatter(xs, ys, s=ms, c=SYN, marker="s", linewidths=0, rasterized=True)
        for a in (axs, axf):
            a.set_xlim(-0.5, w - 0.5)
            a.set_ylim(h - 0.5, -0.5)
    handles = [plt.Line2D([], [], color=c, marker="s", ls="", ms=6, label=f"sigma*={s:g} px")
               for c, s in zip(SCALE_COLORS, st.SCALES)]
    handles += [plt.Line2D([], [], color=INK, marker="o", mfc="none", ls="", ms=7, label="detected crossing"),
                plt.Line2D([], [], color=REAL, marker="s", ls="", ms=6, label="faint mesh: main skeleton"),
                plt.Line2D([], [], color=SYN, marker="s", ls="", ms=6, label="faint mesh: faint-only")]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.05 / fh))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_sets(syn, labels=None):
    if isinstance(syn, dict) and "shape" not in syn and "scalars" not in syn:
        return {k: [record(r) for r in (v if isinstance(v, list) else [v])] for k, v in syn.items()}
    syn = syn if isinstance(syn, list) else [syn]
    return {labels or "synthetic": [record(r) for r in syn]}


def figure_spectra(real, syn, path, labels=None, dpi=120):
    """Radially averaged power spectra of F: every real still (thin grey),
    the real geometric mean, and each synthetic set's geometric mean
    (left); log10 ratio to the real mean with the real min-max band
    (right).  real: load_real(...); syn: list of analyse results/records,
    or {label: list}.  Returns path."""
    plt = _style()
    sets = _as_sets(syn, labels)
    recs = [r for r in real["records"].values() if r.get("psd")]
    f = np.asarray(recs[0]["psd"]["f"], float)
    lr = np.log10([r["psd"]["psd"] for r in recs])
    mu = np.nanmean(lr, 0)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for p in lr:
        axs[0].loglog(f, 10 ** p, color=MUTED, lw=0.7, alpha=0.7)
    axs[0].loglog(f, 10 ** mu, color=REAL, lw=2.2, label=f"real {real['kind']}: geometric mean (grey: {len(recs)})")
    axs[1].fill_between(f, np.nanmin(lr - mu, 0), np.nanmax(lr - mu, 0), color=REAL, alpha=0.15, lw=0,
                        label="real, min-max")
    for c, (name, rs) in zip((SYN, SYN2, SYN3), sets.items()):
        ls = np.log10([r["psd"]["psd"] for r in rs if r.get("psd") and len(r["psd"]["psd"]) == len(f)])
        if not len(ls):
            continue
        axs[0].loglog(f, 10 ** np.nanmean(ls, 0), color=c, lw=2, label=f"{name} ({len(ls)})")
        axs[1].semilogx(f, np.nanmean(ls, 0) - mu, color=c, lw=2, label=name)
    axs[1].axhline(0, color=INK2, lw=0.8)
    for lo in (1 / 128, 1 / 16, 1 / 4):
        axs[1].axvline(lo, color=GRID, lw=1)
    axs[0].set_xlabel("spatial frequency (cycles / px)")
    axs[0].set_ylabel("PSD of F (Np$^2$ px$^2$)")
    axs[0].set_title("Radially averaged power spectrum of F", loc="left")
    axs[0].legend(loc="lower left", fontsize=8)
    axs[1].set_xlabel("spatial frequency (cycles / px)")
    axs[1].set_ylabel("log10 PSD / real mean (dex)")
    axs[1].set_title("Log ratio to the real mean (bands: 1/128, 1/16, 1/4)", loc="left")
    axs[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_scorecard(cards, path, top=30, zclip=12.0, title=None, dpi=120):
    """z-score bar chart (as fig4): the `top` scalar statistics with the
    largest |z| over the given scorecards, bars clipped at +-zclip with the
    value written, |z| <= 2 shaded, and a cross at the end of every bar that
    fails the rule (outside the real [min, max] or |z| > 2).  A second panel
    shows distribution and spectrum distances as a ratio to the real-real
    95th percentile (fail > 1).  cards: Scorecard or {label: Scorecard}."""
    plt = _style()
    if isinstance(cards, Scorecard):
        cards = {"synthetic": cards}
    names = list(cards)
    colors = (SYN, SYN2, SYN3, REAL)
    by = {n: {r["key"]: r for r in c.rows if r["passed"] is not None} for n, c in cards.items()}
    keys = sorted({k for d in by.values() for k in d},
                  key=lambda k: -max(abs(np.nan_to_num(by[n][k]["z"], nan=0.0, posinf=1e3, neginf=1e3))
                                     for n in names if k in by[n]))[:top][::-1]
    nd = max(len(c.dist_rows) for c in cards.values())
    fig = plt.figure(figsize=(14, max(6.0, 0.3 * len(keys) + 1.6)))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1] if nd else [1, 0.001], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(keys))
    hgt = 0.8 / len(names)
    ax.axvspan(-2, 2, color=REAL, alpha=0.08, lw=0)
    for i, n in enumerate(names):
        off = ((len(names) - 1) / 2 - i) * hgt              # first card on top
        for yi, k in zip(y, keys):
            r = by[n].get(k)
            if r is None or not np.isfinite(r["z"]) and not np.isinf(r["z"]):
                continue
            z = r["z"]
            zc = float(np.clip(z, -zclip, zclip))
            ax.barh(yi + off, zc, height=hgt * 0.9, color=colors[i % 4], label=n if yi == y[0] else None)
            end = zc + math.copysign(0.25, zc if zc else 1)
            txt = ("x " if not r["passed"] else "") + (f"{z:+.0f}" if abs(z) > zclip else "")
            if txt:
                ax.text(end, yi + off, txt.strip(), va="center", ha="left" if zc >= 0 else "right",
                        fontsize=7.5, color=INK)
    lab = {k: next(by[n][k]["label"] for n in names if k in by[n]) for k in keys}
    ax.set_yticks(y)
    ax.set_yticklabels([lab[k] for k in keys], fontsize=8)
    ax.set_xlim(-zclip - 3, zclip + 4)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.axvline(0, color=INK2, lw=0.8)
    ref = next(iter(cards.values())).reference.split(" (")[0]
    ax.set_xlabel(f"z = (synthetic mean - real mean) / real SD over {ref}   (shaded |z| <= 2; x = fails the rule)")
    s = "; ".join(f"{n}: {c.n_pass}/{c.n_scored} scalars, {c.summary()['distributions+spectra'][0]}/"
                  f"{c.summary()['distributions+spectra'][1]} distributions pass" for n, c in cards.items())
    fig.suptitle(title or f"Realism scorecard vs {ref}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.985 - 0.35 / fig.get_figheight(), s, ha="left", va="top", fontsize=9, color=INK2)
    ax.set_title(f"Scalar statistics: the {len(keys)} largest |z|", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    if nd:
        ax2 = fig.add_subplot(gs[0, 1])
        labels = [r["label"] for r in next(iter(cards.values())).dist_rows][::-1]
        y2 = np.arange(len(labels))
        for i, (n, c) in enumerate(cards.items()):
            off = ((len(names) - 1) / 2 - i) * hgt
            d = {r["label"]: r for r in c.dist_rows}
            for yi, lb in zip(y2, labels):
                r = d.get(lb)
                if r is None or r["passed"] is None or not np.isfinite(r["ratio"]):
                    ax2.text(0.08, yi + off, "n/a (too few samples)", va="center", fontsize=7, color=MUTED)
                    continue
                v = min(r["ratio"], 6.0)
                ax2.barh(yi + off, v, height=hgt * 0.9, color=colors[i % 4])
                if not r["passed"] or r["ratio"] > 6:
                    ax2.text(v + 0.08, yi + off, ("x " if not r["passed"] else "") +
                             (f"{r['ratio']:.0f}" if r["ratio"] > 6 else ""), va="center", fontsize=7.5, color=INK)
        ax2.axvspan(0, 1, color=REAL, alpha=0.08, lw=0)
        ax2.axvline(1, color=INK2, lw=0.8)
        ax2.set_yticks(y2)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlim(0, 7)
        ax2.set_xlabel("synthetic distance / real-real p95 (pass <= 1)")
        ax2.set_title("Distributions and spectra", loc="left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="vesselscene realism reference")
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--root", default=None, help="LIMBUS checkout with stabilization/ and reference_data/")
    ap.add_argument("--out", default=str(DATA_PATH))
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    build_real_stats(a.root, a.out, frames=not a.no_frames)
