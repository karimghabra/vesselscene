"""Chapter 3: retina, ganglion cells and noise units.

The front end of the pipeline (spec 4.0, appendix A): step 1 turns the image
into OD against a background interpolated under the vessels; step 2's
centre-surround ganglion cells (a difference of Gaussians) keep vessels and
drop slow gradients; a cell's drive is excitation minus inhibition, its rate
max(0, drive); and every drive is divided by its background spread, so a
threshold of 8 means the same thing in a dim frame and a bright one.

Render:  manim -ql scenes/s03_retina.py RetinaShort RetinaMedium RetinaDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import neuro as N  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import EXC, INH, MUTED, OD_COL, colormap, image  # noqa: E402

SIG_C, SIG_S = 1.5, 4.5                 # ganglion centre and surround widths (px): a toy choice, ratio 3
SIGNED = np.array([[0.95, 0.40, 0.36], [0.0, 0.0, 0.0], [0.36, 0.62, 0.95]])   # red (inhibition) - black - blue (excitation)
PICK = ManimColor("#5CE1E6")            # pixels a threshold picks
LIGHT = ManimColor("#8FE07E")           # I, the camera's light (as in chapter 1)
SCLERA = ManimColor("#EAE2CF")          # B, the background brightness (as in chapter 1)
THRESH = 8.0                            # noise units
DIM = 0.4                               # the dim frame's contrast, relative to the bright one

SHAPE = (96, 96)
PAD = 24                                # stimuli are made on a padded canvas, filtered, then cropped (no border effects)


def _u(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), -np.sin(a)])


def dog(s, mode="nearest"):
    """A ganglion cell layer: centre Gaussian minus surround Gaussian (both unit sum)."""
    return gaussian_filter(s, SIG_C, mode=mode) - gaussian_filter(s, SIG_S, mode=mode)


def crop(a):
    return a[PAD:-PAD, PAD:-PAD]


@lru_cache(maxsize=None)
def scene():
    """The toy scene on the padded canvas: vessels (OD), the tissue's texture
    (the same in every frame), a vessel mask, and the background pixels used
    for the spread (outside the mask, beyond the cells' surround)."""
    H, W = SHAPE[0] + 2 * PAD, SHAPE[1] + 2 * PAD
    o = np.array([PAD, PAD], float)
    c = o + np.array([44.0, 52.0])
    segs = [(c - 120 * _u(20), c + 120 * _u(20), 3.5, 0.15),
            (c, c + 110 * _u(100), 2.2, 0.08),
            (o + np.array([70.0, 130.0]), o + np.array([110.0, -40.0]), 2.6, 0.10)]
    od_v = N.vessel_od((H, W), segs, blur=1.0, combine="max")
    t = gaussian_filter(np.random.default_rng(5).normal(0, 1, (H, W)), 2.0)
    tex = t * 0.004 / t.std()
    mask = binary_dilation(od_v > 0.01, iterations=2)
    bg = ~binary_dilation(od_v > 0.005, iterations=10)
    x, y = N.grid((H, W))
    return dict(od_v=od_v, tex=tex, mask=mask, bg=bg, x=x - PAD, y=y - PAD)


def frame_od(scale=1.0, seed=11, gradient=0.0):
    """A frame's OD: scale x (vessels + texture + this frame's sensor noise), plus
    an optional slow gradient (uneven light the background did not remove)."""
    S = scene()
    n = np.random.default_rng(seed).normal(0, 0.004, S["od_v"].shape)
    return scale * (S["od_v"] + S["tex"] + n) + gradient * (S["x"] / 95.0)


@lru_cache(maxsize=None)
def retina_step():
    """Step 1 on the toy: the camera's frame under uneven light, the vessel mask,
    the background B interpolated under it (normalised convolution of the pixels
    outside), and OD = -ln(I / B)."""
    S = scene()
    x, y = S["x"], S["y"]
    B = 0.50 + 0.38 * np.clip(x / 95.0, -0.3, 1.3) - 0.10 * ((y - 48) / 60.0) ** 2
    rng = np.random.default_rng(3)
    I = B * np.exp(-(S["od_v"] + S["tex"])) * (1 + rng.normal(0, 0.004, B.shape))
    keep = (~S["mask"]).astype(float)
    B_est = gaussian_filter(I * keep, 8.0) / np.maximum(gaussian_filter(keep, 8.0), 1e-6)
    od = -np.log(I / B_est)
    err = np.abs(crop(od) - crop(S["od_v"] + S["tex"]))
    print(f"[s03] step 1: |OD error| median {np.median(err):.4f}, 99th pct {np.percentile(err, 99):.4f}")
    return dict(I=crop(I), B=crop(B_est), od=crop(od), mask=crop(S["mask"]))


@lru_cache(maxsize=None)
def signal_1d():
    """A 1-D stimulus in OD: a vessel, a uniform stretch, a slow gradient."""
    x = np.arange(-40.0, 221.0)
    v = 0.15 * np.sqrt(np.clip(1 - ((x - 40) / 3.5) ** 2, 0, None))
    v = gaussian_filter(v, 1.0)
    ramp = 0.18 * 0.5 * (1 - np.cos(np.pi * np.clip((x - 105) / 70.0, 0, 1)))
    s = v + ramp
    E = gaussian_filter(s, SIG_C, mode="nearest")
    Ih = gaussian_filter(s, SIG_S, mode="nearest")
    keep = (x >= 0) & (x <= 180)
    return x[keep], s[keep], E[keep], Ih[keep]


@lru_cache(maxsize=None)
def frames_2d():
    """Drive and rate of the bright frame with a gradient (for the 2-D view), and
    the dim / bright pair in raw and noise units."""
    S = scene()
    bg = S["bg"]
    g = frame_od(1.0, 11, gradient=0.20)
    d_g = dog(g)
    out = dict(inp=crop(g), drive=crop(d_g), rate=crop(np.maximum(d_g, 0)))
    for name, scale, seed in (("bright", 1.0, 11), ("dim", DIM, 12)):
        f = frame_od(scale, seed)
        d = dog(f)
        sd = float(d[bg].std())
        out[name] = dict(od=crop(f), raw=crop(np.maximum(d, 0)), sd=sd, nu=crop(np.maximum(d, 0) / sd))
    sb = out["bright"]["sd"]
    for name in ("bright", "dim"):
        out[name]["raw_pick"] = out[name]["raw"] > THRESH * sb          # one fixed raw threshold (set on the bright frame)
        out[name]["nu_pick"] = out[name]["nu"] > THRESH
    pb, pd = out["bright"]["nu_pick"], out["dim"]["nu_pick"]
    out["jaccard"] = float((pb & pd).sum() / max((pb | pd).sum(), 1))
    print(f"[s03] spreads: bright {sb:.5f}, dim {out['dim']['sd']:.5f} (ratio {out['dim']['sd'] / sb:.3f}); "
          f"raw picks {out['bright']['raw_pick'].sum()} / {out['dim']['raw_pick'].sum()}; "
          f"noise-unit picks {pb.sum()} / {pd.sum()}, overlap {out['jaccard']:.3f}; "
          f"peak NU {out['bright']['nu'].max():.1f} / {out['dim']['nu'].max():.1f}; "
          f"background false picks {(pb & crop(bg)).sum()} / {(pd & crop(bg)).sum()}")
    return out


def mexican_hat_ridges(centre, n_rows=27, half=13.0, sx=0.17, depth=0.09, skew=0.04, zh=1.5):
    """The 2-D difference of Gaussians as ridge lines: one profile per row of the
    kernel, offset upward with depth, drawn back to front with a black fill under
    each so nearer rows hide farther ones.  Segments are blue where the kernel is
    positive (excitation) and red where it is negative (inhibition)."""
    xs = np.linspace(-half, half, 161)
    ys = np.linspace(-half, half, n_rows)
    def k(x, y):
        r2 = x ** 2 + y ** 2
        return (np.exp(-r2 / (2 * SIG_C ** 2)) / (2 * np.pi * SIG_C ** 2)
                - np.exp(-r2 / (2 * SIG_S ** 2)) / (2 * np.pi * SIG_S ** 2))
    peak = k(0.0, 0.0)
    rows = VGroup()
    for y in ys:                                   # y = -half is the back row
        z = k(xs, y) / peak
        pts = np.array([centre + np.array([x * sx + y * skew, -y * depth + zz * zh, 0]) for x, zz in zip(xs, z)])
        low = centre[1] - y * depth - 0.25
        fill = Polygon(*pts, pts[-1] * [1, 0, 1] + [0, low, 0], pts[0] * [1, 0, 1] + [0, low, 0],
                       stroke_width=0, fill_color=BLACK, fill_opacity=1.0)
        g = VGroup(fill)
        sign = np.where(z > 0.02, 1, np.where(z < -0.012, -1, 0))
        start = 0
        for i in range(1, len(xs) + 1):
            if i == len(xs) or sign[i] != sign[start]:
                seg = pts[max(start - 1, 0):i + 1] if i < len(xs) else pts[max(start - 1, 0):i]
                col = EXC if sign[start] > 0 else (INH if sign[start] < 0 else GREY_B)
                if len(seg) >= 2:
                    g.add(VMobject(stroke_color=col, stroke_width=2.2 if sign[start] else 1.4,
                                   stroke_opacity=1.0 if sign[start] else 0.6).set_points_as_corners(seg))
                start = i
        rows.add(g)
    return rows


@lru_cache(maxsize=None)
def uneven_frame(win=12.0, lo=0.25):
    """One frame whose contrast falls from 100 % on the right to `lo` on the left:
    noise units with one spread for the whole frame, and with the spread of the
    background pixels nearby (a Gaussian window of `win` px); and each map's
    value along the main vessel, with the bright frame's for reference."""
    S = scene()
    bg = S["bg"]
    gain = np.clip(lo + (1 - lo) * S["x"] / 95.0, lo, 1.0)
    f = gain * (S["od_v"] + S["tex"] + np.random.default_rng(13).normal(0, 0.004, S["od_v"].shape))
    d = dog(f)
    r = np.maximum(d, 0)
    sd_glob = float(d[bg].std())
    w = gaussian_filter(bg.astype(float), win)
    sd_loc = np.sqrt(gaussian_filter(np.where(bg, d, 0.0) ** 2, win) / np.maximum(w, 1e-6))
    out = dict(glob=crop(r / sd_glob), loc=crop(r / sd_loc), ref=frames_2d()["bright"]["nu"])
    # along the main vessel (crop coordinates): the ridge value, max over +-2 px across it
    c, t_dir = np.array([44.0, 52.0]), _u(20)
    n_dir = np.array([-t_dir[1], t_dir[0]])
    ts = np.arange(-44.0, 52.0, 1.0)
    xs = c[0] + ts * t_dir[0]
    for k in ("glob", "loc", "ref"):
        m = out[k]
        vals = []
        for t in ts:
            best = 0.0
            for o in np.arange(-2.0, 2.01, 0.5):
                q = c + t * t_dir + o * n_dir
                best = max(best, float(bilinear_2d(m, q[0], q[1])))
            vals.append(best)
        out[k + "_line"] = np.array(vals)
    out["xs"] = xs
    print(f"[s03] uneven frame (contrast {lo:.2f} -> 1): along the vessel, global NU {out['glob_line'][:10].mean():.1f} (left) "
          f"to {out['glob_line'][-10:].mean():.1f} (right); nearby {out['loc_line'][:10].mean():.1f} to {out['loc_line'][-10:].mean():.1f}; "
          f"bright frame {out['ref_line'][:10].mean():.1f} to {out['ref_line'][-10:].mean():.1f}")
    return out


def bilinear_2d(a, x, y):
    h, w = a.shape
    x = float(np.clip(x, 0, w - 1.001)); y = float(np.clip(y, 0, h - 1.001))
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
            + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def signed_image(arr, height, vmax):
    return image(arr, height, SIGNED, vmin=-vmax, vmax=vmax)


def outline_image(pick, height, color=PICK):
    """The boundary pixels of a picked set, as an RGBA overlay."""
    edge = pick & ~binary_erosion(pick)
    rgba = np.zeros(pick.shape + (4,), np.uint8)
    r, g, b = (np.array(color.to_rgb()) * 255).astype(np.uint8)
    rgba[edge] = (r, g, b, 255)
    im = ImageMobject(rgba)
    from manim.constants import RESAMPLING_ALGORITHMS
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def masked_view(I, mask, height):
    rgba = colormap(I, "grey", 0.0, 1.0)
    rgba[mask] = (40, 70, 120, 255)
    im = ImageMobject(rgba)
    from manim.constants import RESAMPLING_ALGORITHMS
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def framed(im, title, scale=0.6):
    fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
    t = Tex(title).scale(scale).next_to(fr, UP, buff=0.15)
    return fr, t


class Retina(ChapterScene):
    NUMBER, TITLE, SPEC = 3, "Retina, ganglion cells and noise units", "spec 4.0, appendix A"
    MAP_KEYS = ("retina", "ganglion")

    def construct(self):
        self.intro()
        if self.at(3):
            self.retina_step()
        elif self.at(2):
            self.opening()
        self.dog_profile()
        self.slide()
        self.drive_and_rate()
        self.in_2d()
        if self.at(2):
            self.noise_units()
        if self.at(3):
            self.nearby()
            self.recap()
        self.outro()

    # ---------------------------------------------------------------- the opening (levels 1-2)
    def opening(self):
        st = retina_step()
        im = image(st["I"], 3.4, "grey", vmin=0.2, vmax=1.0).move_to(UP * 0.4)
        fr, t = framed(im, r"a frame under uneven light")
        self.play(FadeIn(im), Create(fr), FadeIn(t), run_time=0.9)
        self.say(r"Light drifts slowly across a frame, and from frame to frame. Vessels change fast. What should the first cells pass on?",
                 wait="read")
        self.play(FadeOut(Group(im, fr, t)), run_time=0.6)

    # ---------------------------------------------------------------- step 1, the retina (level 3)
    def retina_step(self):
        st = retina_step()
        h = 2.7
        xs = (-4.7, 0.0, 4.7)
        cam = image(st["I"], h, "grey", vmin=0.2, vmax=1.0).move_to([xs[0], 0.55, 0])
        fr0, t0 = framed(cam, r"camera: $I$")
        t0[0][-1].set_color(LIGHT)
        self.play(FadeIn(cam), Create(fr0), FadeIn(t0), run_time=0.9)
        self.say(r"Step 1, the retina. The camera's frame is lit unevenly: brighter on the right, darker at the edges.", wait="read")

        holes = masked_view(st["I"], st["mask"], h).move_to([xs[1], 0.55, 0])
        fr1, t1 = framed(holes, r"pixels outside a vessel mask")
        a1 = Arrow([xs[0] + h / 2 + 0.1, 0.55, 0], [xs[1] - h / 2 - 0.1, 0.55, 0], buff=0.05, stroke_width=3, color=MUTED)
        self.say(r"Mask the vessels out, and keep only the pixels around them.")
        self.play(GrowArrow(a1), TransformFromCopy(cam, holes), Create(fr1), FadeIn(t1), run_time=1.3)
        self.wait(1.2)
        Bim = image(st["B"], h, "grey", vmin=0.2, vmax=1.0).move_to(holes)
        t1b = Tex(r"background $B$, interpolated under it").scale(0.6).move_to(t1)
        t1b[0][10].set_color(SCLERA)
        self.say(r"Interpolate the background $B$ under the vessels from those pixels: the light the vessels would have let through.")
        self.play(FadeIn(Bim), FadeOut(holes), ReplacementTransform(t1, t1b), run_time=1.4)
        self.wait(1.5)
        odim = image(st["od"], h, "magma", vmin=-0.02, vmax=0.2).move_to([xs[2], 0.55, 0])
        fr2, t2 = framed(odim, r"$\mathrm{OD} = -\ln(I/B)$")
        t2[0][0:2].set_color(OD_COL)
        a2 = Arrow([xs[1] + h / 2 + 0.1, 0.55, 0], [xs[2] - h / 2 - 0.1, 0.55, 0], buff=0.05, stroke_width=3, color=MUTED)
        self.say(r"Then $\mathrm{OD} = -\ln(I/B)$, as in chapter 1: the uneven light is divided out, and blood is bright.")
        self.play(GrowArrow(a2), FadeIn(odim), Create(fr2), FadeIn(t2), run_time=1.3)
        self.wait(2.0)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- the difference of Gaussians
    def dog_profile(self):
        ax = Axes(x_range=[-14, 14, 7], y_range=[-0.1, 0.3, 0.1], x_length=5.6, y_length=3.0,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5, font_size=20),
                  x_axis_config=dict(numbers_to_include=[-14, -7, 0, 7, 14]), y_axis_config=dict(include_ticks=False))
        ax.move_to([-3.3, 0.2, 0])
        xl = Tex(r"distance from the cell (px)", color=GREY_B).scale(0.5).next_to(ax.x_axis, DOWN, buff=0.4)
        gc = lambda x: np.exp(-0.5 * (x / SIG_C) ** 2) / (np.sqrt(2 * np.pi) * SIG_C)
        gs = lambda x: np.exp(-0.5 * (x / SIG_S) ** 2) / (np.sqrt(2 * np.pi) * SIG_S)
        cc = ax.plot(gc, x_range=[-14, 14, 0.1], color=EXC, stroke_width=4)
        cs = ax.plot(gs, x_range=[-14, 14, 0.1], color=INH, stroke_width=4)
        cd = ax.plot(lambda x: gc(x) - gs(x), x_range=[-14, 14, 0.1], color=WHITE, stroke_width=5)
        lc = Tex(r"centre: excitation", color=EXC).scale(0.55).next_to(ax.c2p(1.8, 0.25), RIGHT, buff=0.1)
        ls = Tex(r"surround: inhibition", color=INH).scale(0.55).next_to(ax.c2p(5.0, 0.09), RIGHT, buff=0.1)
        eq = MathTex(r"\mathrm{DoG}", "=", r"G_{\sigma_c}", "-", r"G_{\sigma_s}").scale(0.95).move_to([0, 3.25, 0])
        eq[2].set_color(EXC); eq[4].set_color(INH)

        self.say(r"Step 2: a ganglion cell is excited by light in a small centre and inhibited by a wider surround.")
        self.play(Create(ax), FadeIn(xl), run_time=0.8)
        self.play(Create(cc), FadeIn(lc), run_time=1.0)
        self.play(Create(cs), FadeIn(ls), run_time=1.0)
        self.say(r"Centre minus surround is a difference of Gaussians: a Mexican hat that sums to zero.")
        self.play(Write(eq), run_time=1.0)
        self.play(TransformFromCopy(VGroup(cc, cs), cd), cc.animate.set_stroke(opacity=0.35), cs.animate.set_stroke(opacity=0.35),
                  run_time=1.4)
        self.wait(0.8 if self.short else 1.4)

        # in 2-D: the hat, computed, drawn as ridge lines (back to front, each hiding what is behind it)
        hat = mexican_hat_ridges(centre=np.array([3.55, -0.1, 0]))
        th = Tex(r"in 2-D: a Mexican hat (computed)").scale(0.55).move_to([3.55, 2.55, 0])
        key = VGroup(Tex(r"blue: excites", color=EXC).scale(0.5), Tex(r"red: inhibits", color=INH).scale(0.5)
                     ).arrange(RIGHT, buff=0.4).move_to([3.55, -1.85, 0])
        self.say(r"In two dimensions it is a disc of excitation inside a ring of inhibition: a centre-surround receptive field.")
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.08) for r in hat], lag_ratio=0.06), FadeIn(th), FadeIn(key), run_time=2.0)
        self.wait(2.0 if self.short else 2.6)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- slide the cell along a 1-D stimulus
    def slide(self):
        x, s, E, Ih = signal_1d()
        d = E - Ih
        top = Axes(x_range=[0, 180, 30], y_range=[0, 0.2, 0.1], x_length=9.0, y_length=1.5,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5, font_size=20),
                   x_axis_config=dict(include_ticks=False), y_axis_config=dict(numbers_to_include=[0.1, 0.2]))
        top.move_to([-1.55, 1.75, 0])
        bot = Axes(x_range=[0, 180, 30], y_range=[-0.03, 0.07, 0.03], x_length=9.0, y_length=1.7,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5, font_size=20),
                   x_axis_config=dict(include_ticks=False), y_axis_config=dict(include_ticks=False))
        bot.move_to([-1.55, -1.75, 0])
        tl = MathTex(r"\mathrm{OD}", color=OD_COL).scale(0.55).next_to(top.y_axis, UP, buff=0.08)
        bl = Tex(r"drive", color=WHITE).scale(0.55).next_to(bot.y_axis, UP, buff=0.08)
        sig = VMobject(stroke_color=OD_COL, stroke_width=3.5).set_points_as_corners([top.c2p(a, b) for a, b in zip(x, s)])
        regions = VGroup(*[Tex(t, color=GREY_B).scale(0.5).move_to(top.c2p(px, 0.2) + UP * 0.22)
                           for t, px in ((r"a vessel", 40), (r"uniform", 80), (r"a slow gradient", 145))])
        self.say(r"Slide one ganglion cell along a profile: a vessel, a uniform stretch, then a slow gradient of light.")
        self.play(Create(top), FadeIn(tl), Create(sig), FadeIn(regions), run_time=1.3)
        self.play(Create(bot), FadeIn(bl), run_time=0.7)

        p = ValueTracker(0.0)
        y_hat = 0.42                      # baseline of the receptive-field glyph, between the two plots
        kx = np.linspace(-14, 14, 120)
        kv = np.exp(-0.5 * (kx / SIG_C) ** 2) / SIG_C - np.exp(-0.5 * (kx / SIG_S) ** 2) / SIG_S
        kv = kv / kv.max() * 0.5
        unit = top.x_length / 180.0

        def glyph():
            c = top.c2p(p.get_value(), 0)[0]
            pos = VMobject(stroke_width=0, fill_color=EXC, fill_opacity=0.8)
            sel = kv > 0
            pts = [[c + a * unit, y_hat + b, 0] for a, b in zip(kx[sel], kv[sel])]
            pos.set_points_as_corners([[c + kx[sel][0] * unit, y_hat, 0]] + pts + [[c + kx[sel][-1] * unit, y_hat, 0]])
            g = VGroup(pos)
            for side in (kx < 0, kx > 0):
                m = (kv < 0) & side
                q = [[c + a * unit, y_hat + b, 0] for a, b in zip(kx[m], kv[m])]
                neg = VMobject(stroke_width=0, fill_color=INH, fill_opacity=0.8)
                neg.set_points_as_corners([[c + kx[m][0] * unit, y_hat, 0]] + q + [[c + kx[m][-1] * unit, y_hat, 0]])
                g.add(neg)
            g.add(DashedLine([c, y_hat + 0.5, 0], top.c2p(p.get_value(), float(np.interp(p.get_value(), x, s))),
                             stroke_color=GREY_B, stroke_width=1.2, dash_length=0.05))
            return g

        def trace():
            keep = x <= p.get_value()
            if keep.sum() < 2:
                return VMobject()
            return VMobject(stroke_color=WHITE, stroke_width=3.5).set_points_as_corners(
                [bot.c2p(a, b) for a, b in zip(x[keep], d[keep])])

        base_y = -0.35
        bw = 0.42
        hs = 1.7 / 0.2

        def bars():
            e = float(np.interp(p.get_value(), x, E))
            i = float(np.interp(p.get_value(), x, Ih))
            be = Rectangle(width=bw, height=max(e * hs, 1e-3), stroke_width=0, fill_color=EXC, fill_opacity=0.9)
            be.move_to([4.9, base_y, 0], aligned_edge=DOWN)
            bi = Rectangle(width=bw, height=max(i * hs, 1e-3), stroke_width=0, fill_color=INH, fill_opacity=0.9)
            bi.move_to([5.6, base_y, 0], aligned_edge=DOWN)
            dv = e - i if abs(e - i) >= 5e-4 else 0.0
            val = MathTex(rf"\text{{drive}} = {dv:+.3f}" if dv else r"\text{drive} = 0.000").scale(0.55).move_to([5.25, base_y - 0.95, 0])
            return VGroup(be, bi, val)

        bars_base = Line([4.5, base_y, 0], [6.0, base_y, 0], stroke_color=MUTED, stroke_width=1.5)
        blabs = VGroup(Tex(r"exc.", color=EXC).scale(0.5).move_to([4.9, base_y - 0.3, 0]),
                       Tex(r"inh.", color=INH).scale(0.5).move_to([5.6, base_y - 0.3, 0]))
        eq = MathTex(r"\text{drive}", "=", r"\text{excitation}", "-", r"\text{inhibition}").scale(0.6)
        eq[2].set_color(EXC); eq[4].set_color(INH)
        eq.move_to([5.25, 2.2, 0])
        if eq.width > 3.4:
            eq.scale_to_fit_width(3.4)
        live = [always_redraw(glyph), always_redraw(trace), always_redraw(bars)]
        self.say(r"Its drive is excitation minus inhibition: what the centre sees, minus what the surround sees.")
        self.play(FadeIn(bars_base), FadeIn(blabs), Write(eq), *[FadeIn(m) for m in live], run_time=1.0)
        if self.short:
            self.play(p.animate.set_value(178.0), run_time=6.0, rate_func=linear)
            self.say(r"Only the vessel drives it. Uniform light and a slow gradient balance out: the cell ignores the lighting.",
                     wait="read")
        else:
            self.wait(1.6)
            self.play(p.animate.set_value(40.0), run_time=2.2, rate_func=smooth)
            self.say(r"On the vessel the centre sees more blood than the surround: a strong positive drive.", wait=1.2)
            self.play(p.animate.set_value(85.0), run_time=2.2, rate_func=smooth)
            self.say(r"On a uniform stretch, centre and surround see the same: zero.", wait=0.8)
            self.play(p.animate.set_value(178.0), run_time=3.6, rate_func=smooth)
            self.say(r"On a slow gradient they still balance, since the surround is symmetric: almost zero. The cell ignores the lighting.",
                     wait="read")
        print(f"[s03] 1-D drive: vessel {d[(x > 30) & (x < 50)].max():.4f}, flank {d[(x > 25) & (x < 55)].min():.4f}, "
              f"uniform {np.abs(d[(x > 65) & (x < 95)]).max():.2e}, gradient {np.abs(d[x > 105]).max():.4f}")
        for m in live:
            m.clear_updaters()
        self.slide_state = dict(top=top, bot=bot, x=x, d=d, trace=live[1], bars=VGroup(live[2], bars_base, blabs), eq=eq,
                                glyph=live[0], sig=sig, regions=regions, labels=VGroup(tl, bl))

    # ---------------------------------------------------------------- drive and rate
    def drive_and_rate(self):
        st = self.slide_state
        bot, x, d = st["bot"], st["x"], st["d"]
        self.play(FadeOut(st["bars"]), FadeOut(st["glyph"]), run_time=0.6)
        # the drive's negative flanks, marked
        neg = (d < -0.004) & (x < 70)
        flank = VGroup(*[Dot(bot.c2p(a, b), radius=0.025, color=INH) for a, b in zip(x[neg][::2], d[neg][::2])])
        self.say(r"The drive is like a membrane potential relative to threshold: beside the vessel, where the surround sees blood, it goes negative.")
        self.play(FadeIn(flank, lag_ratio=0.1), run_time=0.9)
        self.wait(0.6 if self.short else 1.6)

        hx = Axes(x_range=[-0.03, 0.06, 0.03], y_range=[0, 0.06, 0.03], x_length=2.6, y_length=1.9,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5),
                  x_axis_config=dict(include_ticks=False), y_axis_config=dict(include_ticks=False))
        hx.move_to([4.9, -1.05, 0])
        hl = VGroup(Tex(r"drive", color=GREY_B).scale(0.45).next_to(hx.x_axis, DOWN, buff=0.12),
                    Tex(r"rate", color=GREY_B).scale(0.45).next_to(hx.y_axis, UP, buff=0.08))
        zero = DashedLine(hx.c2p(0, 0), hx.c2p(0, 0.06), stroke_color=MUTED, stroke_width=1, dash_length=0.05)
        hinge = VMobject(stroke_color=YELLOW_B, stroke_width=4).set_points_as_corners([hx.c2p(-0.03, 0), hx.c2p(0, 0), hx.c2p(0.06, 0.06)])
        req = MathTex(r"\text{rate}", "=", r"\max(0,\ ", r"\text{drive}", ")").scale(0.62).move_to([4.9, 0.6, 0])
        req[0].set_color(YELLOW_B)
        rate = VMobject(stroke_color=YELLOW_B, stroke_width=4).set_points_as_corners([bot.c2p(a, max(b, 0)) for a, b in zip(x, d)])
        self.say(r"A firing rate cannot be negative: $\text{rate} = \max(0, \text{drive})$, the threshold-linear form.")
        self.play(FadeOut(st["eq"]), Create(hx), FadeIn(hl), Create(zero), Write(req), run_time=1.0)
        self.play(Create(hinge), run_time=0.8)
        self.play(TransformFromCopy(st["trace"], rate), FadeOut(flank), st["trace"].animate.set_stroke(opacity=0.3), run_time=1.4)
        if self.at(3):
            ref = Tex(r"Carandini and Ferster 2000", color=GREY_B).scale(0.42).next_to(hx, DOWN, buff=0.4)
            self.play(FadeIn(ref))
            self.add(ref)
        self.wait(1.0 if self.short else 2.0)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- the same in 2-D
    def in_2d(self):
        f = frames_2d()
        h = 2.8
        vm = float(np.abs(f["drive"]).max()) * 0.9
        ims = [image(f["inp"], h, "magma", vmin=-0.02, vmax=0.32), signed_image(f["drive"], h, vm),
               image(f["rate"], h, "magma", vmin=0.0, vmax=vm)]
        titles = [r"input OD, with a slow gradient", r"drive (blue $+$, red $-$)", r"rate $= \max(0,$ drive$)$"]
        grp = []
        for im, t, x in zip(ims, titles, (-4.7, 0.0, 4.7)):
            im.move_to([x, 0.45, 0])
            fr, tt = framed(im, t, 0.55)
            grp.append((im, fr, tt))
        arrows = VGroup(*[Arrow([a + h / 2 + 0.1, 0.45, 0], [b - h / 2 - 0.1, 0.45, 0], buff=0.05, stroke_width=3, color=MUTED)
                          for a, b in ((-4.7, 0.0), (0.0, 4.7))])
        if self.short:
            self.say(r"In 2-D: the drive keeps the vessels and drops the gradient; the rate clips the negative flanks.")
            self.play(*[FadeIn(g[0]) for g in grp], *[Create(g[1]) for g in grp], *[FadeIn(g[2]) for g in grp],
                      *[GrowArrow(a) for a in arrows], run_time=1.2)
            self.wait(3.4)
            self.clear(run_time=0.7)
            return
        self.say(r"In 2-D: a gradient of light across the frame, under a few vessels.")
        self.play(FadeIn(grp[0][0]), Create(grp[0][1]), FadeIn(grp[0][2]), run_time=0.9)
        self.wait(0.6 if self.short else 1.4)
        self.say(r"The ganglion cells' drive keeps the vessels, with their negative flanks, and drops the gradient.")
        self.play(GrowArrow(arrows[0]), FadeIn(grp[1][0]), Create(grp[1][1]), FadeIn(grp[1][2]), run_time=1.1)
        self.wait(1.2 if self.short else 2.0)
        self.say(r"The rate clips the flanks: what is left is the vessels.")
        self.play(GrowArrow(arrows[1]), FadeIn(grp[2][0]), Create(grp[2][1]), FadeIn(grp[2][2]), run_time=1.1)
        self.wait(2.0)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- noise units (level 2)
    def noise_units(self):
        f = frames_2d()
        h = 2.45
        cols = {}
        vm_raw = float(f["bright"]["raw"].max()) * 0.9
        for name, x, title in (("dim", -3.7, rf"a dim, hazy frame: {100 * DIM:.0f}\% contrast"), ("bright", 3.7, r"a bright frame")):
            cam = image(np.exp(-f[name]["od"]) * (0.6 if name == "dim" else 0.9), h, "grey", vmin=0.45, vmax=0.95)
            cam.move_to([x, 1.65, 0])
            fr, t = framed(cam, title, 0.55)
            raw = image(f[name]["raw"], h, "magma", vmin=0.0, vmax=vm_raw).move_to([x, -1.3, 0])
            frr = SurroundingRectangle(raw, buff=0, color=MUTED, stroke_width=1)
            cols[name] = dict(cam=cam, fr=fr, t=t, raw=raw, frr=frr, x=x)
        self.say(r"The same vessels in a dim, hazy frame and in a bright one. In the dim frame every contrast is weaker.")
        self.play(*[FadeIn(c["cam"]) for c in cols.values()], *[Create(c["fr"]) for c in cols.values()],
                  *[FadeIn(c["t"]) for c in cols.values()], run_time=1.0)
        self.wait(1.6)

        mid = np.array([0.0, -1.3, 0])
        lab_raw = Tex(r"raw rate", color=GREY_A).scale(0.6).move_to(mid + UP * 1.0)
        nb, nd = int(f["bright"]["raw_pick"].sum()), int(f["dim"]["raw_pick"].sum())
        self.say(rf"So are the raw responses. One fixed threshold picks {nb} pixels in the bright frame, only {nd} in the dim one.")
        self.play(*[FadeIn(c["raw"]) for c in cols.values()], *[Create(c["frr"]) for c in cols.values()], FadeIn(lab_raw), run_time=1.0)
        picks_raw, counts_raw = Group(), VGroup()
        for name, c in cols.items():
            o = outline_image(f[name]["raw_pick"], h).move_to(c["raw"])
            picks_raw.add(o)
            counts_raw.add(MathTex(rf"{int(f[name]['raw_pick'].sum())}\ \text{{px}}", color=PICK).scale(0.5)
                           .next_to(c["frr"], LEFT if name == "dim" else RIGHT, buff=0.15))
        rt = MathTex(r"\text{rate} > \text{a fixed } T", color=PICK).scale(0.55).move_to(mid + DOWN * 0.05)
        self.play(FadeIn(picks_raw), FadeIn(counts_raw), FadeIn(rt), run_time=0.9)
        self.wait(2.2)

        # divide by the spread of each frame's own background
        eq = MathTex(r"\frac{\text{drive}}{\sigma_{\text{bg}}}").scale(0.8).move_to(mid + UP * 0.25)
        sds = VGroup(MathTex(rf"\sigma_{{\text{{bg}}}} = {f['dim']['sd']:.4f}", color=GREY_B).scale(0.45),
                     MathTex(rf"\sigma_{{\text{{bg}}}} = {f['bright']['sd']:.4f}", color=GREY_B).scale(0.45))
        for sd, name in zip(sds, ("dim", "bright")):
            sd.next_to(cols[name]["frr"], DOWN, buff=0.08)
        lab_nu = Tex(r"noise units", color=GREY_A).scale(0.6).move_to(lab_raw)
        self.say(r"Now divide every drive by how much that cell fluctuates over its frame's background: noise units.")
        self.play(FadeOut(picks_raw), FadeOut(counts_raw), FadeOut(rt), run_time=0.5)
        self.play(Write(eq), ReplacementTransform(lab_raw, lab_nu), run_time=1.0)
        # the spreads, shown as what is divided
        self.play(FadeIn(sds), run_time=0.6)
        vm_nu = float(f["bright"]["nu"].max()) * 0.9
        nus = {}
        for name, c in cols.items():
            nus[name] = image(f[name]["nu"], h, "magma", vmin=0.0, vmax=vm_nu).move_to(c["raw"])
        self.play(*[FadeIn(nus[n]) for n in cols], *[FadeOut(c["raw"]) for c in cols.values()], run_time=1.4)
        self.say(r"In noise units the two maps match: 8 means eight times the background's variability in either frame.")
        self.wait(1.0)
        picks_nu, counts_nu = Group(), VGroup()
        for name, c in cols.items():
            picks_nu.add(outline_image(f[name]["nu_pick"], h).move_to(c["raw"]))
            counts_nu.add(MathTex(rf"{int(f[name]['nu_pick'].sum())}\ \text{{px}}", color=PICK).scale(0.5)
                          .next_to(c["frr"], LEFT if name == "dim" else RIGHT, buff=0.15))
        nt = MathTex(r"> 8", color=PICK).scale(0.6).next_to(eq, DOWN, buff=0.25)
        agree = Tex(rf"toy: {100 * f['jaccard']:.0f}\% the same pixels", color=PICK).scale(0.5).next_to(nt, DOWN, buff=0.25)
        self.say(r"A threshold at 8 picks the same pixels in both. That is why every threshold in the spec is written in noise units.")
        self.play(FadeIn(nt), FadeIn(picks_nu), FadeIn(counts_nu), run_time=1.0)
        self.play(FadeIn(agree), run_time=0.6)
        self.wait(2.6)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- why "nearby" (level 3)
    def nearby(self):
        u = uneven_frame()
        h = 2.35
        vm = float(u["ref"].max()) * 0.9
        maps = Group()
        for k, x, title in (("glob", -3.6, r"one spread for the whole frame"), ("loc", 3.6, r"the spread of background nearby")):
            im = image(u[k], h, "magma", vmin=0.0, vmax=vm).move_to([x, 1.55, 0])
            fr, t = framed(im, title, 0.55)
            maps.add(Group(im, fr, t))
        arr = Arrow(LEFT * 0.5, RIGHT * 0.5, buff=0, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.2)
        grad = VGroup(Tex(r"contrast across the frame", color=GREY_B).scale(0.48),
                      VGroup(Tex(r"25\%", color=GREY_B).scale(0.48), arr, Tex(r"100\%", color=GREY_B).scale(0.48)
                             ).arrange(RIGHT, buff=0.12)).arrange(DOWN, buff=0.12).move_to([0, 1.55, 0])
        ax = Axes(x_range=[0, 95, 95], y_range=[0, 50, 10], x_length=8.0, y_length=1.85,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5, font_size=20),
                  x_axis_config=dict(include_ticks=False), y_axis_config=dict(numbers_to_include=[10, 20, 30, 40, 50]))
        ax.move_to([0, -1.55, 0])
        yl = Tex(r"noise units along the main vessel", color=GREY_B).scale(0.48).next_to(ax, UP, buff=0.12)
        xl = Tex(r"left $\rightarrow$ right", color=GREY_B).scale(0.45).next_to(ax.x_axis, DOWN, buff=0.12)
        thr = DashedLine(ax.c2p(0, THRESH), ax.c2p(95, THRESH), stroke_color=PICK, stroke_width=1.5, dash_length=0.08)
        thl = MathTex(r"8", color=PICK).scale(0.5).next_to(ax.c2p(95, THRESH), RIGHT, buff=0.1)

        def line(vals, color, width=3.5):
            return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
                [ax.c2p(x, v) for x, v in zip(u["xs"], vals)])

        lg, ll, lr = line(u["glob_line"], INH), line(u["loc_line"], PICK), line(u["ref_line"], WHITE, 2.0)
        lr = DashedVMobject(lr, num_dashes=60)
        key = VGroup(Tex(r"one spread", color=INH).scale(0.45), Tex(r"spread nearby", color=PICK).scale(0.45),
                     Tex(r"the bright frame", color=WHITE).scale(0.45)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        key.next_to(ax, RIGHT, buff=0.3).shift(UP * 0.4)
        self.say(r"Why ``nearby''? A single frame can be dim on one side and bright on the other. Here its contrast rises from left to right.")
        self.play(FadeIn(grad), *[FadeIn(m[0]) for m in maps], *[Create(m[1]) for m in maps], *[FadeIn(m[2]) for m in maps],
                  run_time=1.2)
        self.wait(1.4)
        self.play(Create(ax), FadeIn(yl), FadeIn(xl), Create(thr), FadeIn(thl), run_time=0.9)
        self.say(r"With one spread for the whole frame, the same vessel reads low on the dim side and high on the bright side.")
        self.play(Create(lg), FadeIn(key[0]), run_time=1.4)
        self.wait(1.2)
        self.say(r"With the spread of the background nearby, it reads the same all along, as in the bright frame.")
        self.play(Create(ll), FadeIn(key[1]), run_time=1.4)
        self.play(Create(lr), FadeIn(key[2]), run_time=1.0)
        self.wait(2.4)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- the conventions, as the spec writes them (level 3)
    def recap(self):
        title = Tex(r"Conventions (spec 4.0)").scale(0.8).move_to([0, 2.9, 0])
        rows = [
            (r"\textbf{drive}", r"excitation minus inhibition, like a membrane potential\\relative to threshold; it can be negative"),
            (r"\textbf{rate}", r"$\max(0,\ \text{drive})$: the threshold-linear form\\of the linear-nonlinear model (Carandini and Ferster 2000)"),
            (r"\textbf{noise units}", r"every drive divided by how much that cell type fluctuates\\over background pixels nearby; kept by every later layer"),
        ]
        grid = VGroup()
        for k, v in rows:
            kt = Tex(k, color=YELLOW_B if "rate" in k else GREY_A).scale(0.6)
            vt = Tex(v, color=GREY_B, tex_environment="flushleft").scale(0.55)
            grid.add(VGroup(kt, vt))
        for g in grid:
            g[1].next_to(g[0], RIGHT, buff=0.5)
        grid.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        for g in grid:
            g[0].align_to(grid, LEFT)
        # keys in one column, values in another
        kx = max(g[0].width for g in grid)
        for g in grid:
            g[1].next_to(g[0].get_left() + RIGHT * kx, RIGHT, buff=0.5).align_to(g[0], UP).shift(UP * 0.05)
        grid.move_to([0, 0.2, 0])
        self.say(r"So, the front end in three conventions that every later chapter keeps.")
        self.play(FadeIn(title), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.15) for g in grid], lag_ratio=0.5), run_time=2.2)
        self.wait(2.0)
        self.say(r"Appendix A: the brain's centre-surround ganglion cells and gain control are the pipeline's step 2.", wait="read")


variants(Retina, module_globals=globals())
