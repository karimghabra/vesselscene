"""Chapter 1: the physics of the picture.

What the camera measures, and why a crossing will look different from a fork
(spec 4.5, 4.12, 6.4).  Green light crosses blood on its way from the sclera
to the camera (Beer-Lambert: I = B exp(-OD), OD proportional to the path), the
path across a round vessel is a chord (2 sqrt(r^2 - x^2)), the optics blur it,
and where two vessels meet the OD either adds (a crossing, two absorbers in
series) or is a union (a fork, one merged lumen).

Render:  manim -ql scenes/s01_physics.py PhysicsShort PhysicsMedium PhysicsDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from manim.constants import RESAMPLING_ALGORITHMS
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import neuro as N  # noqa: E402
from mica_anim import vsdata as VS  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import BLOOD, MUTED, OD_COL, PixelFrame, colormap, image  # noqa: E402

LIGHT = ManimColor("#8FE07E")      # the green light, and I in equations
SCLERA = ManimColor("#EAE2CF")     # the white of the eye, and B in equations
KAP = TEAL_B                       # kappa, the crossing's transparency coefficient
VMAX = 0.30                        # OD colour scale of the top-view images

# ------------------------------------------------------------------ top-view toy stimuli (computed)
SHAPE = (96, 96)
CENTRE = np.array([48.0, 48.0])
A_PER_PX = 0.03                    # toy contrast: peak OD 0.03 per px of radius (r = 5 px -> 0.15, the spec's test vessels)
R1, R2 = 5.0, 4.0                  # crossing: the vessel along the cut, the one crossing it
RP, RB = 5.0, 3.5                  # fork: parent and branch radius ...
RD = (RP ** 3 - RB ** 3) ** (1 / 3)  # ... and the daughter by Murray's law (4.35 px)
BRANCH_DEG = 65.0
KAPPA = 0.90                       # toy kappa, inside the spec's 0.87-0.94


def _u(deg):
    """Image direction (y down) of a screen angle in degrees."""
    a = np.deg2rad(deg)
    return np.array([np.cos(a), -np.sin(a)])


@lru_cache(maxsize=None)
def crossing_layers():
    """Each crossing vessel's own sharp OD (chord law), in OD units."""
    c = CENTRE
    l1 = N.vessel_od(SHAPE, [(c - 90 * _u(0), c + 90 * _u(0), R1, A_PER_PX * R1)], blur=0)
    l2 = N.vessel_od(SHAPE, [(c - 90 * _u(BRANCH_DEG), c + 90 * _u(BRANCH_DEG), R2, A_PER_PX * R2)], blur=0)
    return l1, l2


@lru_cache(maxsize=None)
def crossing_sum():
    """Two vessels at different depths: ODs add (neuro.vessel_od, combine='sum')."""
    c = CENTRE
    return N.vessel_od(SHAPE, [(c - 90 * _u(0), c + 90 * _u(0), R1, A_PER_PX * R1),
                               (c - 90 * _u(BRANCH_DEG), c + 90 * _u(BRANCH_DEG), R2, A_PER_PX * R2)],
                       blur=1.0, combine="sum")


def crossing_kappa(kappa, noise=0.0):
    """max + kappa * min per pixel, then the same 1 px blur (kappa = 1 is the sum)."""
    l1, l2 = crossing_layers()
    od = gaussian_filter(np.maximum(l1, l2) + kappa * np.minimum(l1, l2), 1.0)
    return N.add_noise(od, noise, seed=3) if noise else od


@lru_cache(maxsize=None)
def fork_union():
    """A fork: parent, daughter (Murray) and branch merged at one depth (combine='max')."""
    c = CENTRE
    return N.vessel_od(SHAPE, [(c - 90 * _u(0), c, RP, A_PER_PX * RP),
                               (c, c + 90 * _u(0), RD, A_PER_PX * RD),
                               (c, c + 90 * _u(BRANCH_DEG), RB, A_PER_PX * RB)],
                       blur=1.0, combine="max")


@lru_cache(maxsize=None)
def real_crops():
    """vesselscene's still around a crossing (junction 85) and a fork (junction 90),
    flattened log intensity (vessels dark), or None without the data."""
    if not VS.available():
        return None
    S = VS.load()
    v = S.view(S.average)
    return v[274 - 56:274 + 56, 63 - 56:63 + 56], v[285 - 40:285 + 40, 392 - 40:392 + 40]


def photo(arr, height):
    """A photograph-like ImageMobject (smooth resampling, unlike the pixelated maps)."""
    im = ImageMobject(colormap(arr, "grey", 0.0, 1.0))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    im.height = height
    return im


def line_plot(ax, ys, color=OD_COL, width=3.5):
    """A polyline of ys (one value per pixel) on axes ax."""
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners([ax.c2p(i, float(v)) for i, v in enumerate(ys)])
    return m


@lru_cache(maxsize=None)
def real_numbers():
    """The spec 4.5 recipe on vesselscene's OD (the truth's junctions 85 and 90):
    a_c = max OD within 2 px of the centre; each arm's median OD from R to R + 2w."""
    if not VS.available():
        return None
    S = VS.load()
    out = {}
    yy, xx = np.mgrid[0:S.od.shape[0], 0:S.od.shape[1]]
    for j in S.truth["junctions"]:
        if j["id"] not in (85, 90):
            continue
        x, y, R = j["x"], j["y"], j["radius"]
        ac = float(S.od[np.hypot(xx - x, yy - y) <= 2].max())
        arms = {}
        for ln in j["lines"]:
            a = np.deg2rad(ln["angle_deg"])
            s = np.arange(R, R + 2 * ln["width_px"], 1.0)
            v = S.od[np.round(y + s * np.sin(a)).astype(int), np.round(x + s * np.cos(a)).astype(int)]
            arms.setdefault(ln["vid"], []).append(float(np.median(v)))
        out[j["id"]] = (ac, {k: float(np.mean(v)) for k, v in arms.items()})
    return out


# ------------------------------------------------------------------ the side view: a slice through the tissue
class Slice:
    """Sclera below, blood above it, light rising to the camera's pixels, and the
    OD profile drawn over the pixels on the same x.  Vessel 1 is cut across (a
    disc); vessel 2 runs along the slice (a band) at height y2.  Absorption is
    exaggerated (MU) so the dimming shows."""
    CX, HALF, STEP = -3.45, 2.8, 0.175
    Y_SCL, Y_CAM = -2.62, 0.46
    Y1, R1, R2 = -1.70, 0.78, 0.34
    Y2_CROSS = -0.32
    MU = 0.8
    Y_BASE = 1.12

    def __init__(self):
        self.y2 = ValueTracker(self.Y2_CROSS)
        self.on2 = ValueTracker(0.0)
        self.sigma = ValueTracker(0.0)
        n = int(2 * self.HALF / self.STEP)
        n -= 1 - n % 2
        self.ray_x = (np.arange(n) - (n - 1) / 2) * self.STEP
        self.grid = np.arange(-self.HALF - 1.6, self.HALF + 1.6 + 1e-9, 0.004)

    # -------------------------------------------------------------- physics
    def half_chord(self, x):
        return np.sqrt(np.clip(self.R1 ** 2 - np.asarray(x, float) ** 2, 0.0, None))

    def od_sharp(self, x):
        """MU x the length of blood along the vertical ray at x: the union of the
        disc's chord and the band (they only overlap once merged)."""
        h = self.half_chord(x)
        y2, on2 = self.y2.get_value(), self.on2.get_value()
        ov = np.clip(np.minimum(self.Y1 + h, y2 + self.R2) - np.maximum(self.Y1 - h, y2 - self.R2), 0.0, None) * (h > 0)
        return self.MU * (2 * h + on2 * 2 * self.R2 - ov)

    def od_seen(self, x):
        s = self.sigma.get_value()
        if s < 2e-3:
            return self.od_sharp(x)
        g = self.grid
        u = np.arange(-4 * s, 4 * s + 1e-9, 0.004)
        k = np.exp(-0.5 * (u / s) ** 2)
        b = np.convolve(np.pad(self.od_sharp(g), len(u), mode="edge"), k / k.sum(), "same")[len(u):-len(u)]
        return np.interp(x, g, b)

    def pt(self, x, od):
        return np.array([self.CX + x, self.Y_BASE + od, 0.0])

    # -------------------------------------------------------------- mobjects
    def profile(self, color=OD_COL, width=4.0, sharp=False, lo=None, hi=None):
        xs = np.linspace(-self.HALF if lo is None else lo, self.HALF if hi is None else hi, 500)
        ys = self.od_sharp(xs) if sharp else self.od_seen(xs)
        m = VMobject(stroke_color=color, stroke_width=width)
        m.set_points_as_corners([self.pt(x, y) for x, y in zip(xs, ys)])
        return m

    def ray_pieces(self, x):
        """(y_from, y_to, transmittance so far) for the gaps between blood on the ray at x."""
        iv = []
        h = float(self.half_chord(x))
        if h > 0:
            iv.append([self.Y1 - h, self.Y1 + h, 1.0])
        on2 = self.on2.get_value()
        if on2 > 1e-3:
            y2 = self.y2.get_value()
            iv.append([y2 - self.R2, y2 + self.R2, on2])
        iv.sort()
        merged = []
        for lo, hi, w in iv:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
                merged[-1][2] = max(merged[-1][2], w)
            else:
                merged.append([lo, hi, w])
        out, cur, T = [], self.Y_SCL, 1.0
        for lo, hi, w in merged:
            out.append((cur, lo, T))
            T *= np.exp(-self.MU * (hi - lo) * w)
            cur = hi
        out.append((cur, self.Y_CAM, T))
        return out

    def ray(self, x, color=LIGHT, width=2.0, opacity=0.85):
        g = VGroup()
        for a, b, T in self.ray_pieces(x):
            if b - a > 0.01:
                g.add(Line([self.CX + x, a, 0], [self.CX + x, b, 0], stroke_color=color, stroke_width=width,
                           stroke_opacity=opacity * T))
        return g

    def rays(self):
        return VGroup(*[self.ray(x) for x in self.ray_x])

    def pixels(self):
        vals = 0.92 * np.exp(-self.od_seen(self.ray_x))
        g = VGroup()
        for x, v in zip(self.ray_x, vals):
            c = ManimColor.from_rgb((v, v, v))
            g.add(Square(self.STEP, stroke_width=0.6, stroke_color=BLACK, fill_color=c, fill_opacity=1.0)
                  .move_to([self.CX + x, self.Y_CAM + self.STEP / 2, 0]))
        return g

    def sclera(self):
        r = Rectangle(width=2 * self.HALF + 0.4, height=0.34, stroke_width=0, fill_color=SCLERA, fill_opacity=0.95)
        r.move_to([self.CX, self.Y_SCL - 0.17, 0])
        lab = Tex(r"sclera: brightness $B$", color=BLACK).scale(0.45).move_to(r).align_to(r, RIGHT).shift(LEFT * 0.12)
        return VGroup(r, lab)

    def disc(self):
        return Circle(radius=self.R1, stroke_width=0, fill_color=BLOOD, fill_opacity=1.0).move_to([self.CX, self.Y1, 0])

    def band(self):
        r = Rectangle(width=2 * self.HALF + 0.3, height=2 * self.R2, stroke_width=0, fill_color=BLOOD,
                      fill_opacity=self.on2.get_value())
        return r.move_to([self.CX, self.y2.get_value(), 0])

    def baseline(self):
        return Line(self.pt(-self.HALF, 0), self.pt(self.HALF, 0), stroke_color=MUTED, stroke_width=1.5)


# ------------------------------------------------------------------ the chapter
class Physics(ChapterScene):
    NUMBER, TITLE, SPEC = 1, "The physics of the picture", "spec 4.5, 4.12, 6.4"
    MAP_KEYS = ("retina",)

    def construct(self):
        self.intro()
        self.puzzle()
        sv = self.light_through_blood()
        self.chord_law(sv)
        if self.at(2):
            self.blur(sv)
        self.crossing_and_fork(sv)
        cols = self.top_view()
        if self.at(2):
            self.kappa(cols)
        if self.at(3):
            self.noise(cols)
            self.murray()
            self.inverse_crime()
        self.outro()

    # ---------------------------------------------------------------- 0. the puzzle, on vesselscene's still
    def puzzle(self):
        crops = real_crops()
        if crops is None:                                   # no vesselscene data: the toy's own camera view
            crops = (N.to_intensity(crossing_sum(), 1.0) * 1.1 - 0.15, N.to_intensity(fork_union(), 1.0) * 1.1 - 0.15)
        rings = [(0.82, VS.JUNCTION_COLORS["crossing"]), (0.5, VS.JUNCTION_COLORS["bifurcation"])]
        self.crops = Group()
        for arr, x, name, (rr, col) in zip(crops, (-3.45, 3.45), ("a crossing", "a fork"), rings):
            im = photo(arr, 3.5).move_to([x, 0.45, 0])
            fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
            t = Tex(name).scale(0.7).next_to(fr, UP, buff=0.15)
            ring = DashedVMobject(Circle(radius=rr, color=col, stroke_width=3), num_dashes=24).move_to(im)
            self.crops.add(Group(im, fr, t, ring))
        self.play(*[FadeIn(g[0]) for g in self.crops], *[Create(g[1]) for g in self.crops],
                  *[FadeIn(g[2], shift=DOWN * 0.1) for g in self.crops], run_time=0.9)
        self.say(r"In the camera's still, a crossing has a dark centre. A fork has none. Why?")
        self.play(*[Create(g[3]) for g in self.crops], run_time=0.7)
        self.wait(1.4 if self.short else 2.8)
        if self.at(3) and real_numbers() is not None:
            rn = real_numbers()
            (cc, carms), (fc, farms) = rn[85], rn[90]
            ca = sorted(carms.values())
            print(f"[s01] vesselscene OD: crossing 85 centre {cc:.3f}, arms {carms}; fork 90 centre {fc:.3f}, arms {farms}")
            nums = VGroup(
                MathTex(rf"\mathrm{{OD}}\text{{: centre }} {cc:.2f},\ \text{{arms }} {ca[0]:.2f} \text{{ and }} {ca[1]:.2f}",
                        color=GREY_B).scale(0.55),
                MathTex(rf"\mathrm{{OD}}\text{{: centre }} {fc:.2f},\ \text{{widest arm }} {max(farms.values()):.2f}",
                        color=GREY_B).scale(0.55))
            for n, g in zip(nums, self.crops):
                n.next_to(g[1], DOWN, buff=0.15)
            self.play(FadeIn(nums))
            self.say(r"vesselscene's own OD agrees: the crossing's centre is about the sum of its arms, the fork's is not.",
                     wait="read")
            self.crops.add(nums)
        self.play(FadeOut(self.crops), run_time=0.6)

    # ---------------------------------------------------------------- 1. light through blood
    def light_through_blood(self):
        sv = Slice()
        scl, disc = sv.sclera(), sv.disc()
        rays0 = sv.rays()
        lab1 = Tex(r"$\leftarrow$ a vessel, cut across", color=GREY_B).scale(0.55).move_to([-0.35, sv.Y1, 0], aligned_edge=LEFT)
        cam = Tex(r"$\leftarrow$ camera pixels: $I$", color=GREY_B).scale(0.55)
        cam[0][-1].set_color(LIGHT)
        cam.move_to([-0.35, sv.Y_CAM + sv.STEP / 2, 0], aligned_edge=LEFT)
        if self.short:
            self.say(r"Slice across a vessel. Green light from the white sclera crosses the blood to the camera.")
        else:
            self.say(r"Slice across a vessel. Green light, scattered back by the white sclera, crosses the blood on its way to the camera.")
        self.play(FadeIn(scl), GrowFromCenter(disc), FadeIn(lab1), run_time=0.8)
        self.play(LaggedStart(*[Create(r) for r in rays0], lag_ratio=0.03), run_time=1.2 if self.short else 1.4)
        rays = always_redraw(sv.rays)
        self.remove(*rays0)
        self.add(rays)
        pix = always_redraw(sv.pixels)
        self.play(FadeIn(pix, lag_ratio=0.03), FadeIn(cam), run_time=0.8)
        self.wait(1.0 if self.short else 2.0)

        if self.at(2):
            # Beer-Lambert, step by step
            spot = np.array([3.75, 3.15, 0])
            d1 = MathTex("d", "I", "=", r"-\mu", r"\,I", r"\,dL").scale(1.0).move_to(spot)
            d1[1].set_color(LIGHT); d1[4].set_color(LIGHT)
            self.say(r"Blood absorbs green light. Each thin layer $dL$ of it removes the same fraction $\mu$ of the light that reaches it.")
            self.play(Write(d1))
            self.wait(2.2)
            d2 = MathTex("I", "=", "B", r"\,e^{-", r"\mu", "L", "}").scale(1.0).move_to(spot)
            d2[0].set_color(LIGHT); d2[2].set_color(SCLERA)
            self.say(r"So after a path $L$ through blood, the light $B$ has decayed exponentially: Beer--Lambert.")
            self.play(TransformMatchingTex(d1, d2), run_time=1.3)
            self.wait(1.8)
            d3 = MathTex(r"-\ln(", "I", "/", "B", ")", "=", r"\mu", "L").scale(1.0).move_to(spot)
            d3[1].set_color(LIGHT); d3[3].set_color(SCLERA)
            self.say(r"Take logarithms: the log of the dimming is linear in the path.")
            self.play(TransformMatchingTex(d2, d3), run_time=1.3)
            self.wait(1.2)
            d4 = MathTex(r"\mathrm{OD}", r"\equiv", r"-\ln(", "I", "/", "B", ")", "=", r"\mu", "L").scale(1.0).move_to(spot)
            d4[0].set_color(OD_COL); d4[3].set_color(LIGHT); d4[5].set_color(SCLERA)
            self.say(r"That is the optical density, $\mathrm{OD} = -\ln(I/B)$: the path through blood, times $\mu$.")
            self.play(TransformMatchingTex(d3, d4), run_time=1.3)
            self.play(Circumscribe(d4[0], color=OD_COL, fade_out=True), run_time=1.2)
            self.wait(1.2)
            self.eq_top = d4
        else:
            e1 = MathTex("I", "=", "B", r"\,e^{-", r"\mathrm{OD}", "}").scale(1.05).move_to([3.75, 3.15, 0])
            e1[0].set_color(LIGHT); e1[2].set_color(SCLERA); e1[4].set_color(OD_COL)
            e2 = MathTex(r"\mathrm{OD}", "=", r"\mu", "L").scale(0.9)
            e2[0].set_color(OD_COL)
            e2l = Tex(r"($L$: path through blood)", color=GREY_B).scale(0.55)
            e2g = VGroup(e2, e2l).arrange(RIGHT, buff=0.25).next_to(e1, DOWN, buff=0.35)
            self.say(r"Blood absorbs it: the camera sees $I = B\,e^{-\mathrm{OD}}$, and OD grows with the path through blood.")
            self.play(Write(e1), run_time=1.0)
            self.play(FadeIn(e2g, shift=UP * 0.1), run_time=0.6)
            self.wait(1.8)
            self.eq_top = VGroup(e1, e2g)
        self.sv_static = VGroup(scl, disc)
        self.sv_live = [rays, pix]
        self.sv_labels = VGroup(lab1, cam)
        return sv

    # ---------------------------------------------------------------- 2. the chord law
    def chord_law(self, sv):
        base = sv.baseline()
        odl = MathTex(r"\mathrm{OD}(x)", color=OD_COL).scale(0.7).move_to(sv.pt(-sv.HALF, 0) + np.array([0.55, 1.95, 0]))
        chord_eq = MathTex("L", "=", r"2\sqrt{r^2 - x^2}").scale(1.0).move_to([3.75, 1.75, 0])
        self.say(r"Across a round vessel the path is a chord: $L = 2\sqrt{r^2-x^2}$ at an offset $x$ from its centre.")
        self.play(Create(base), FadeIn(odl), Write(chord_eq), run_time=1.0)

        xt = ValueTracker(-sv.R1 * 0.999)
        C = np.array([sv.CX, sv.Y1, 0])

        def tri():
            x = xt.get_value()
            h = float(sv.half_chord(x))
            top = C + np.array([x, h, 0])
            g = VGroup(
                Line(C + np.array([x, -h, 0]), top, stroke_color=WHITE, stroke_width=5),
                DashedLine(C, top, stroke_color=GREY_A, stroke_width=2, dash_length=0.06),
                Line(C, C + np.array([x, 0, 0]), stroke_color=GREY_A, stroke_width=2.5),
                Dot(C, radius=0.035, color=GREY_A))
            return g

        def hi_ray():
            return sv.ray(xt.get_value(), color=LIGHT, width=4.0, opacity=1.0)

        def dot():
            x = xt.get_value()
            return Dot(sv.pt(x, float(sv.od_sharp(x))), radius=0.06, color=OD_COL)

        def guide():
            x = xt.get_value()
            return DashedLine(sv.pt(x, 0) + DOWN * 0.15, sv.pt(x, float(sv.od_sharp(x))), stroke_color=GREY_B,
                              stroke_width=1.5, dash_length=0.05)

        def partial():
            return sv.profile(sharp=True, lo=-sv.R1, hi=max(xt.get_value(), -sv.R1 + 1e-3))

        r_lab = MathTex("r", color=GREY_A).scale(0.6)
        x_lab = MathTex("x", color=GREY_A).scale(0.6)
        r_lab.add_updater(lambda m: m.move_to(C + 0.5 * np.array([xt.get_value(), sv.half_chord(xt.get_value()), 0])
                                              + np.array([-0.14 if xt.get_value() > 0 else 0.14, 0.1, 0])))
        x_lab.add_updater(lambda m: m.move_to(C + np.array([xt.get_value() / 2, -0.16, 0])))
        readout = always_redraw(lambda: MathTex(
            rf"x = {xt.get_value() / sv.R1:+.2f}\,r \;\Rightarrow\; L = {2 * sv.half_chord(xt.get_value()) / sv.R1:.2f}\,r"
        ).scale(0.6).move_to([3.75, 0.95, 0]))
        live = [always_redraw(tri), always_redraw(hi_ray), always_redraw(partial), always_redraw(guide), always_redraw(dot)]
        self.play(*[FadeIn(m) for m in live], FadeIn(r_lab), FadeIn(x_lab), FadeIn(readout), run_time=0.5)
        if self.short:
            self.play(xt.animate.set_value(sv.R1 * 0.999), run_time=3.0, rate_func=smooth)
        else:
            self.play(xt.animate.set_value(0.0), run_time=3.0, rate_func=smooth)
            self.wait(0.5)
            self.play(xt.animate.set_value(sv.R1 * 0.999), run_time=2.6, rate_func=smooth)
        full = sv.profile()
        prof = always_redraw(sv.profile)
        self.add(full)
        for m in (r_lab, x_lab):
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in (*live, r_lab, x_lab, readout)], run_time=0.6)
        self.remove(full)
        self.add(prof)
        if not self.short:
            self.say(r"So a vessel's OD profile is a half-ellipse: as wide as the vessel, as high as $2\mu r$.", wait="read")
        self.chord_eq = chord_eq
        self.sv_live += [prof]
        self.sv_static.add(base, odl)

    # ---------------------------------------------------------------- 2b. blur (level 2)
    def blur(self, sv):
        ghost = sv.profile(color=OD_COL, width=2.0, sharp=True)
        ghost = DashedVMobject(ghost, num_dashes=90).set_stroke(opacity=0.6)
        eq = MathTex(r"\mathrm{OD}_{\text{seen}}", "=", r"G_\sigma", "*", r"\mathrm{OD}_{\text{chord}}").scale(0.8)
        eq[0].set_color(OD_COL); eq[4].set_color(OD_COL)
        eq.move_to([4.45, 0.8, 0])
        # the Gaussian kernel, drawn live under the equation
        ax_c = np.array([4.45, -0.35, 0])

        def kernel():
            s = sv.sigma.get_value()
            u = np.linspace(-1.2, 1.2, 160)
            if s < 2e-3:
                return Line(ax_c, ax_c + UP * 0.8, stroke_color=GREY_A, stroke_width=3)
            g = np.exp(-0.5 * (u / s) ** 2)
            m = VMobject(stroke_color=GREY_A, stroke_width=3)
            m.set_points_as_corners([ax_c + np.array([x, 0.8 * y, 0]) for x, y in zip(u, g)])
            return m

        kbase = Line(ax_c + LEFT * 1.25, ax_c + RIGHT * 1.25, stroke_color=MUTED, stroke_width=1.2)
        klab = always_redraw(lambda: MathTex(rf"G_\sigma,\ \sigma = {sv.sigma.get_value() / sv.R1:.2f}\,r").scale(0.55)
                             .next_to(kbase, DOWN, buff=0.12))
        peak = always_redraw(lambda: MathTex(
            rf"\text{{peak: }} {100 * sv.od_seen(np.array([0.0]))[0] / (2 * sv.MU * sv.R1):.0f}\%"
            r"\text{ of } 2\mu r", color=GREY_B).scale(0.55).next_to(klab, DOWN, buff=0.15))
        kern = always_redraw(kernel)
        self.say(r"The optics blur the picture: what we measure is the chord profile convolved with a Gaussian of width $\sigma$.")
        self.play(FadeIn(ghost), Write(eq), FadeIn(kbase), FadeIn(kern), FadeIn(klab), FadeIn(peak), run_time=1.2)
        self.play(sv.sigma.animate.set_value(0.40), run_time=3.0, rate_func=smooth)
        self.say(r"Blur lowers the peak and spreads the edges. The fit will have to measure radius, blur and contrast together.",
                 wait="read")
        self.play(FadeOut(VGroup(eq, kbase, kern, klab, peak)), FadeOut(ghost), run_time=0.6)

    # ---------------------------------------------------------------- 3. crossing and fork, in the slice
    def crossing_and_fork(self, sv):
        band = always_redraw(sv.band)
        self.add(band)
        self.bring_to_front(*self.sv_live)
        lab2 = Tex(r"$\leftarrow$ a second vessel, higher up", color=GREY_B).scale(0.55)
        lab2.move_to([-0.35, sv.Y2_CROSS, 0], aligned_edge=LEFT)
        self.play(FadeOut(self.chord_eq), FadeOut(self.eq_top), run_time=0.4)

        if self.at(2):
            s_eq = MathTex(r"e^{-\mathrm{OD}_1}", r"\,e^{-\mathrm{OD}_2}", "=", r"e^{-(\mathrm{OD}_1+\mathrm{OD}_2)}").scale(0.9)
        else:
            s_eq = MathTex(r"\mathrm{OD}", "=", r"\mathrm{OD}_1", "+", r"\mathrm{OD}_2").scale(1.0)
            s_eq[0].set_color(OD_COL)
        s_eq.move_to([3.75, 2.9, 0])
        s_tit = Tex(r"a crossing: two absorbers in series", color=GREY_A).scale(0.65).next_to(s_eq, UP, buff=0.3)

        def centre_chords():
            h = float(sv.half_chord(0.0))
            y2 = sv.y2.get_value()
            return VGroup(Line([sv.CX, sv.Y1 - h, 0], [sv.CX, sv.Y1 + h, 0], stroke_color=WHITE, stroke_width=5),
                          Line([sv.CX, y2 - sv.R2, 0], [sv.CX, y2 + sv.R2, 0], stroke_color=WHITE, stroke_width=5))

        chords = always_redraw(centre_chords)
        top_sum = float(sv.MU * (2 * sv.R1 + 2 * sv.R2)) if sv.sigma.get_value() < 2e-3 else None
        if top_sum is None:                      # blurred: the sum's peak, computed with the band in place
            on = sv.on2.get_value()
            sv.on2.set_value(1.0)
            top_sum = float(sv.od_seen(np.array([0.0]))[0])
            sv.on2.set_value(on)
        sum_line = DashedLine(sv.pt(-1.5, top_sum), sv.pt(1.5, top_sum), stroke_color=GREY_B, stroke_width=1.5, dash_length=0.08)
        sum_lab = MathTex(r"\mathrm{OD}_1 + \mathrm{OD}_2", color=GREY_B).scale(0.5).next_to(sum_line, RIGHT, buff=0.1)
        if self.short:
            self.say(r"A crossing: a second vessel, at another depth. Light crosses both, so their ODs add.")
        else:
            self.say(r"A crossing: the second vessel runs at another depth. Light passes both absorbers in series, so their ODs add.")
        self.play(sv.on2.animate.set_value(1.0), FadeIn(lab2), run_time=1.0)
        self.play(FadeIn(s_tit), Write(s_eq), FadeIn(chords), Create(sum_line), FadeIn(sum_lab), run_time=1.1)
        self.wait(1.4 if self.short else 2.8)

        f_eq = MathTex(r"\mathrm{OD}", "=", r"\max(", r"\mathrm{OD}_1", ",", r"\mathrm{OD}_2", ")").scale(1.0)
        f_eq[0].set_color(OD_COL)
        f_eq.move_to(s_eq)
        f_tit = Tex(r"a fork: one merged lumen", color=GREY_A).scale(0.65).move_to(s_tit)
        if self.short:
            self.say(r"A fork: the lumens merge into one volume. Light crosses the blood once: a union, not a sum.")
        else:
            self.say(r"A fork: the vessels meet at one depth and their lumens merge. Light crosses the blood once: a union, not a sum.")
        self.play(FadeOut(lab2), FadeOut(self.sv_labels[0]), run_time=0.4)
        self.play(sv.y2.animate.set_value(sv.Y1), run_time=2.0 if self.short else 2.6, rate_func=smooth)
        top_max = float(sv.od_seen(np.array([0.0]))[0])
        max_line = DashedLine(sv.pt(-1.5, top_max), sv.pt(1.5, top_max), stroke_color=OD_COL, stroke_width=1.5, dash_length=0.08)
        max_lab = MathTex(r"\max", color=OD_COL).scale(0.5).next_to(max_line, RIGHT, buff=0.1)
        self.play(FadeOut(s_eq), FadeIn(f_eq), ReplacementTransform(s_tit, f_tit), Create(max_line), FadeIn(max_lab), run_time=0.9)
        self.wait(1.3 if self.short else 2.6)
        print(f"[s01] slice: centre OD sum {top_sum:.3f}, union {top_max:.3f} (illustrative units)")
        for m in self.sv_live + [band, chords]:
            m.clear_updaters()
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- 4. from above: computed OD images
    def top_view(self):
        cols = {}
        specs = (("crossing", -3.45, crossing_sum(), r"crossing: ODs add"),
                 ("fork", 3.45, fork_union(), r"fork: a union (max)"))
        anims_a, anims_b = [], []
        for name, x, od, title in specs:
            pf = PixelFrame(SHAPE, 3.0, np.array([x, 1.45, 0]))
            im = image(od, pf.height, "magma", vmin=0.0, vmax=VMAX).move_to(pf.centre)
            fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
            tt = Tex(title).scale(0.65).next_to(fr, UP, buff=0.15)
            cut = DashedLine(pf.point(-0.5, 48), pf.point(95.5, 48), stroke_color=WHITE, stroke_width=1.5, dash_length=0.07)
            ax = Axes(x_range=[0, 95, 95], y_range=[0, 0.3, 0.1], x_length=pf.width, y_length=1.9,
                      axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5, font_size=20),
                      y_axis_config=dict(numbers_to_include=[0.1, 0.2, 0.3], decimal_number_config=dict(num_decimal_places=1)),
                      x_axis_config=dict(include_ticks=False))
            ax.move_to([x, -1.8, 0])
            ylab = MathTex(r"\mathrm{OD}", color=OD_COL).scale(0.5).next_to(ax.y_axis, UP, buff=0.08)
            row = od[48]
            prof = line_plot(ax, row)
            cols[name] = dict(pf=pf, im=im, fr=fr, tt=tt, cut=cut, ax=ax, ylab=ylab, prof=prof, od=od)
            anims_a += [FadeIn(im), Create(fr), FadeIn(tt)]
            anims_b += [Create(cut), Create(ax), FadeIn(ylab)]
        if not self.short:
            self.say(r"Seen from above, computed: the OD of a crossing and of a fork, and a profile through each centre.")
        self.play(*anims_a, run_time=0.7 if self.short else 0.8)
        self.play(*anims_b, run_time=0.5 if self.short else 0.6)
        self.play(*[Create(c["prof"]) for c in cols.values()], run_time=0.9 if self.short else 1.1)

        # labels on the profiles: the crossing's centre is the sum, the fork's is the parent's
        c, f = cols["crossing"], cols["fork"]
        cpk = float(c["od"][48].max())
        arm = float(c["od"][48, 10])
        fpk = float(f["od"][48].max())
        print(f"[s01] top view: crossing centre {cpk:.3f}, arm {arm:.3f}, other arm {A_PER_PX * R2:.3f}; "
              f"fork centre {fpk:.3f}, parent {f['od'][48, 10]:.3f}, daughter {f['od'][48, 85]:.3f}")
        lc = MathTex(rf"{cpk:.2f} \approx {A_PER_PX * R1:.2f} + {A_PER_PX * R2:.2f}").scale(0.5)
        lc.next_to(c["ax"].c2p(48, cpk), RIGHT, buff=0.18)
        lf = MathTex(rf"{fpk:.2f} = \text{{parent}}").scale(0.5).next_to(f["ax"].c2p(60, fpk), UP, buff=0.12)
        self.play(FadeIn(lc), FadeIn(lf), run_time=0.6)
        if self.short:
            self.say(r"From above: a crossing's centre is the sum of its arms, a fork's only its largest arm.", wait=3.8)
        else:
            self.say(r"The crossing's centre carries the sum of its arms: the dark centre of the still. The fork's carries only its largest arm.",
                     wait="read")
        cols["labels"] = VGroup(lc, lf)
        return cols

    # ---------------------------------------------------------------- 5. kappa (level 2)
    def kappa(self, cols):
        f = cols["fork"]
        self.play(*[FadeOut(f[k]) for k in ("im", "fr", "tt", "cut", "ax", "ylab", "prof")], FadeOut(cols["labels"]), run_time=0.7)
        c = cols["crossing"]
        k = ValueTracker(1.0)
        img = c["im"]
        img.add_updater(lambda m: setattr(m, "pixel_array", colormap(crossing_kappa(k.get_value()), "magma", 0.0, VMAX)))
        prof = always_redraw(lambda: line_plot(c["ax"], crossing_kappa(k.get_value())[48]))
        self.remove(c["prof"])
        self.add(prof)

        # why: a side-view pictogram, light scattered around the deeper vessel on its way to one pixel
        o = np.array([2.3, 2.05, 0])
        sc = Line(o + np.array([-1.3, -1.1, 0]), o + np.array([1.3, -1.1, 0]), stroke_color=SCLERA, stroke_width=6)
        deep = Circle(radius=0.3, stroke_width=0, fill_color=BLOOD, fill_opacity=1).move_to(o + DOWN * 0.5)
        shallow = Rectangle(width=2.6, height=0.32, stroke_width=0, fill_color=BLOOD, fill_opacity=1).move_to(o + UP * 0.4)
        pix = Square(0.2, stroke_color=GREY_A, stroke_width=1.5, fill_color=GREY_D, fill_opacity=1).move_to(o + UP * 1.2)
        straight = VGroup(Line(o + DOWN * 1.08, o + DOWN * 0.8, stroke_color=LIGHT, stroke_width=3),
                          Line(o + DOWN * 0.2, o + UP * 0.24, stroke_color=LIGHT, stroke_width=3, stroke_opacity=0.55),
                          Line(o + UP * 0.56, o + UP * 1.1, stroke_color=LIGHT, stroke_width=3, stroke_opacity=0.3))
        bend = VMobject(stroke_color=LIGHT, stroke_width=3, stroke_opacity=0.9)
        bend.set_points_smoothly([o + np.array([0.95, -1.08, 0]), o + np.array([0.62, -0.5, 0]),
                                  o + np.array([0.3, 0.2, 0]), o + np.array([0.12, 0.75, 0]), o + np.array([0.03, 1.1, 0])])
        why = Tex(r"part of the light reaching\\the pixel was scattered\\around the deeper vessel", color=GREY_B).scale(0.5)
        why.next_to(o + RIGHT * 1.3, RIGHT, buff=0.3).shift(DOWN * 0.15)
        picto = VGroup(sc, bend, deep, shallow, straight, pix)

        eq = MathTex(r"\mathrm{OD}_c", r"\approx", r"\max(a_1,a_2)", "+", r"\kappa", r"\,\min(a_1,a_2)").scale(0.8)
        eq[0].set_color(OD_COL); eq[4].set_color(KAP)
        eq.move_to([3.75, 0.45, 0])
        nl = NumberLine(x_range=[0, 1, 0.5], length=4.6, color=MUTED, include_numbers=True, font_size=22,
                        decimal_number_config=dict(num_decimal_places=1)).move_to([3.75, -0.85, 0])
        band = Rectangle(width=nl.n2p(0.94)[0] - nl.n2p(0.87)[0], height=0.22, stroke_width=0, fill_color=KAP,
                         fill_opacity=0.5).move_to((nl.n2p(0.87) + nl.n2p(0.94)) / 2)
        bl = Tex(r"crossings (spec 4.5)", color=KAP).scale(0.45).next_to(band, UP, buff=0.35).shift(LEFT * 0.35)
        l0 = Tex(r"union: a fork", color=GREY_B).scale(0.45).next_to(nl.n2p(0), DOWN, buff=0.42)
        l1 = Tex(r"full sum", color=GREY_B).scale(0.45).next_to(nl.n2p(1), DOWN, buff=0.42)
        mark = always_redraw(lambda: Triangle(fill_color=KAP, fill_opacity=1, stroke_width=0).scale(0.1).rotate(PI)
                             .move_to(nl.n2p(k.get_value()) + UP * 0.2))
        kval = always_redraw(lambda: MathTex(rf"\kappa = {k.get_value():.2f}", color=KAP).scale(0.6)
                             .next_to(nl, UP, buff=0.5).align_to(nl, LEFT))
        cval = always_redraw(lambda: MathTex(rf"\text{{toy centre: }}\mathrm{{OD}}_c = {crossing_kappa(k.get_value())[48].max():.3f}",
                                             color=OD_COL).scale(0.6).next_to(nl, DOWN, buff=0.95))
        self.say(r"Not quite a full sum: some light is scattered around the deeper vessel, so it counts a little less.")
        self.play(FadeIn(picto, lag_ratio=0.15), FadeIn(why), run_time=1.6)
        self.wait(1.4)
        self.say(r"The spec writes the crossing's centre as $\max + \kappa\,\min$, with $\kappa \approx 0.87$--$0.94$ (spec 4.5).")
        self.play(Write(eq), run_time=1.2)
        self.play(Create(nl), FadeIn(band), FadeIn(bl), FadeIn(l0), FadeIn(l1), FadeIn(mark), FadeIn(kval), FadeIn(cval))
        self.wait(0.8)
        self.say(r"$\kappa = 0$ would be a union, like a fork. Real crossings sit near 1, so their centres stay distinctly darker.")
        self.play(k.animate.set_value(0.0), run_time=2.6, rate_func=smooth)
        self.wait(0.6)
        self.play(k.animate.set_value(KAPPA), run_time=2.4, rate_func=smooth)
        self.wait(1.2)
        img.clear_updaters()
        prof.clear_updaters()
        cols["kappa_panel"] = VGroup(picto, why, eq, nl, band, bl, l0, l1, mark, kval, cval)
        cols["prof_live"] = prof

    # ---------------------------------------------------------------- 6. noise (level 3)
    def noise(self, cols):
        c = cols["crossing"]
        k_panel = cols["kappa_panel"]
        for m in k_panel:
            m.clear_updaters()
        self.play(FadeOut(k_panel), run_time=0.6)
        noisy = crossing_kappa(KAPPA, noise=0.01)
        im2 = image(noisy, c["pf"].height, "magma", vmin=0.0, vmax=VMAX).move_to(c["pf"].centre)
        prof2 = line_plot(c["ax"], noisy[48], width=2.5)
        clean_row = crossing_kappa(KAPPA)[48]
        sd = 0.01
        up = [c["ax"].c2p(i, v + 2 * sd) for i, v in enumerate(clean_row)]
        dn = [c["ax"].c2p(i, v - 2 * sd) for i, v in enumerate(clean_row)]
        bandp = Polygon(*up, *dn[::-1], stroke_width=0, fill_color=GREY_B, fill_opacity=0.25)
        bg = crossing_kappa(KAPPA) < 0.005
        sd_meas = float(noisy[bg].std())
        extra = float(crossing_kappa(KAPPA)[48].max() - crossing_kappa(KAPPA)[48, 10])
        print(f"[s01] noise: measured background sd {sd_meas:.4f}, crossing extra {extra:.3f} = {extra / sd_meas:.1f} sd")
        txt = VGroup(
            MathTex(r"\text{white noise: } \sigma = 0.01\ \mathrm{OD}", color=GREY_A),
            MathTex(r"\text{a } 0.15\ \mathrm{OD} \text{ vessel: } 15\,\sigma", color=GREY_A),
            MathTex(rf"\text{{this crossing's extra: }} {extra:.2f} \approx {extra / 0.01:.0f}\,\sigma", color=GREY_A),
            MathTex(r"\text{a faint } 0.03\ \mathrm{OD} \text{ vessel: } 3\,\sigma", color=WHITE),
        ).scale(0.62).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, 0.9, 0])
        note = Tex(r"the spec's synthetic test vessels (spec 1.1):\\6 px wide, peak 0.15 OD, white noise 0.01 OD", color=GREY_B
                   ).scale(0.48).next_to(txt, DOWN, buff=0.45).align_to(txt, LEFT)
        self.say(r"Every frame also carries noise. The spec's test vessels peak at 0.15 OD under white noise of 0.01 OD.")
        self.play(FadeIn(im2), FadeOut(c["im"]), FadeIn(bandp), ReplacementTransform(cols["prof_live"], prof2), run_time=1.2)
        self.play(FadeIn(txt[0]), FadeIn(note))
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in txt[1:]], lag_ratio=0.6), run_time=2.0)
        self.say(r"A strong crossing stands far above the noise. When one vessel is faint, sum and union differ by only a few $\sigma$.",
                 wait="read")
        self.play(*[FadeOut(m) for m in (im2, c["fr"], c["tt"], c["cut"], c["ax"], c["ylab"], prof2, bandp, txt, note)], run_time=0.7)

    # ---------------------------------------------------------------- 7. Murray's law (level 3)
    def murray(self):
        pf = PixelFrame(SHAPE, 3.4, np.array([-3.6, 0.55, 0]))
        im = image(fork_union(), pf.height, "magma", vmin=0.0, vmax=VMAX).move_to(pf.centre)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        tt = Tex(r"a fork").scale(0.65).next_to(fr, UP, buff=0.15)
        c = CENTRE
        cols = {"r_p": ManimColor("#E74C3C"), "r_1": OD_COL, "r_2": ManimColor("#F5B7B1")}
        marks = VGroup()
        for lab, d_deg, r in (("r_p", 180.0, RP), ("r_1", 0.0, RD), ("r_2", BRANCH_DEG, RB)):
            d = _u(d_deg)
            n = np.array([-d[1], d[0]])                 # across the arm
            p = c + 30 * d
            seg = Line(pf.point(*(p - r * n)), pf.point(*(p + r * n)), stroke_color=cols[lab], stroke_width=3.5)
            t = MathTex(lab, color=cols[lab]).scale(0.6)
            t.next_to(seg, DOWN if lab != "r_2" else RIGHT, buff=0.1)
            marks.add(VGroup(seg, t))
        self.say(r"One more piece of physics, at forks: the parent's flow is shared by the two daughters.")
        self.play(FadeIn(im), Create(fr), FadeIn(tt), run_time=0.9)
        self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.3), run_time=1.2)

        eq = MathTex(r"r_p^3", "=", r"r_1^3", "+", r"r_2^3").scale(1.1).move_to([3.4, 2.75, 0])
        eq[0].set_color(cols["r_p"]); eq[2].set_color(cols["r_1"]); eq[4].set_color(cols["r_2"])
        sub = Tex(r"Murray's law: flow $\propto r^3$", color=GREY_B).scale(0.55).next_to(eq, DOWN, buff=0.2)
        num = MathTex(rf"{RP:.1f}^3 = {RP ** 3:.0f}", r"\;=\;", rf"{RD:.2f}^3 + {RB:.1f}^3 = {RD ** 3:.0f} + {RB ** 3:.0f}").scale(0.6)
        num.next_to(sub, DOWN, buff=0.3)
        self.say(r"Murray's law: flow scales as $r^3$, so the parent's $r^3$ is the sum of its daughters'. The toy fork is built that way.")
        self.play(Write(eq), FadeIn(sub), run_time=1.3)
        self.play(FadeIn(num), run_time=0.8)

        # r^3 as bars: the parent's against the daughters' stacked
        u = 1.9 / RP ** 3
        base_y = -2.55
        bp = Rectangle(width=0.7, height=u * RP ** 3, stroke_width=0, fill_color=cols["r_p"], fill_opacity=0.9)
        bp.move_to([2.3, base_y, 0], aligned_edge=DOWN)
        b1 = Rectangle(width=0.7, height=u * RD ** 3, stroke_width=0, fill_color=cols["r_1"], fill_opacity=0.9).move_to([4.4, base_y, 0], aligned_edge=DOWN)
        b2 = Rectangle(width=0.7, height=u * RB ** 3, stroke_width=0, fill_color=cols["r_2"], fill_opacity=0.9).next_to(b1, UP, buff=0)
        floor_l = Line([1.6, base_y, 0], [5.1, base_y, 0], stroke_color=MUTED, stroke_width=1.5)
        lp = MathTex(r"r_p^3").scale(0.55).next_to(bp, LEFT, buff=0.12)
        l12 = MathTex(r"r_1^3 + r_2^3").scale(0.55).next_to(b1, RIGHT, buff=0.12)
        self.play(Create(floor_l), GrowFromEdge(bp, DOWN), run_time=0.8)
        self.play(GrowFromEdge(b1, DOWN), run_time=0.6)
        self.play(GrowFromEdge(b2, DOWN), FadeIn(lp), FadeIn(l12), run_time=0.6)
        note = Tex(r"a soft check, reported but not used to decide (spec 4.4);\\and a plausibility test on real images (spec 6.4)",
                   color=GREY_B).scale(0.48).move_to([3.4, 0.35, 0])
        self.say(r"It is only a soft check: real forks scatter around it, so the spec reports it and never decides with it.")
        self.play(FadeIn(note))
        self.wait(2.0)
        self.clear(run_time=0.7)

    # ---------------------------------------------------------------- 8. the inverse crime (level 3)
    def inverse_crime(self):
        title = Tex(r"the tube model").scale(0.75).move_to([0, 3.35, 0])
        xs = np.linspace(-3.0, 3.0, 241)

        def plot_icon(fn, w=1.6, h=0.75):
            ys = fn(xs)
            m = VMobject(stroke_color=OD_COL, stroke_width=3)
            m.set_points_as_corners([[x / 3.0 * w / 2, y * h - h / 2, 0] for x, y in zip(xs, ys)])
            return VGroup(Line([-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], stroke_color=MUTED, stroke_width=1.2), m)

        def chord_fn(x):
            return np.sqrt(np.clip(1 - x ** 2, 0, None))

        def blur_fn(x):
            dx = x[1] - x[0]
            u = np.arange(-4 * 0.4, 4 * 0.4 + 1e-9, dx)
            g = np.exp(-0.5 * (u / 0.4) ** 2)
            return np.convolve(chord_fn(x), g / g.sum(), "same")

        i_x = image(crossing_kappa(KAPPA), 0.95, "magma", vmin=0, vmax=VMAX)
        i_y = image(fork_union(), 0.95, "magma", vmin=0, vmax=VMAX)
        items = [(plot_icon(chord_fn), r"chord-law cylinders"), (plot_icon(blur_fn), r"blur"),
                 (i_y, r"a union at forks"), (i_x, r"$\kappa$ at crossings")]
        grp = Group()
        for (ic, lab), x in zip(items, (-4.8, -1.6, 1.6, 4.8)):
            ic.move_to([x, 2.35, 0])
            t = Tex(lab, color=GREY_A).scale(0.5).next_to(ic, DOWN, buff=0.15)
            grp.add(Group(ic, t))
        self.say(r"Everything in this chapter is one model of the picture: tubes of blood, blurred, a union at forks, $\kappa$ at crossings.")
        self.play(FadeIn(title), LaggedStart(*[FadeIn(g, shift=UP * 0.15) for g in grp], lag_ratio=0.25), run_time=2.0)
        self.wait(1.0)

        crops = real_crops()
        syn = photo(crops[0], 2.1) if crops is not None else image(crossing_kappa(KAPPA), 2.1, "magma", 0, VMAX)
        syn.move_to([-3.6, -0.75, 0])
        syn_fr = SurroundingRectangle(syn, buff=0, color=MUTED, stroke_width=1)
        syn_t = Tex(r"vesselscene renders\\with exactly this model", color=GREY_A).scale(0.5).next_to(syn_fr, UP, buff=0.12)
        fit = Tex(r"the fit inverts\\the same model", color=GREY_A).scale(0.55).move_to([0.0, -0.75, 0])
        a1 = Arrow(syn_fr.get_right(), fit.get_left(), buff=0.15, stroke_width=3, color=MUTED)
        ok = Tex(r"exactly right:\\scores overstate", color=GREEN_C).scale(0.55).move_to([3.6, -0.35, 0])
        real = Tex(r"real vessels follow it\\only approximately", color=YELLOW_B).scale(0.55).move_to([3.6, -1.45, 0])
        a2 = Arrow(fit.get_right(), ok.get_left(), buff=0.15, stroke_width=3, color=MUTED)
        a3 = Arrow(fit.get_right(), real.get_left(), buff=0.15, stroke_width=3, color=MUTED)
        self.say(r"The inverse crime (spec 6.4): vesselscene renders with exactly this model, so on synthetic data the fit's model is exactly right.")
        self.play(FadeIn(syn), Create(syn_fr), FadeIn(syn_t), run_time=1.0)
        self.play(GrowArrow(a1), FadeIn(fit), run_time=0.9)
        self.play(GrowArrow(a2), FadeIn(ok), run_time=0.9)
        self.wait(1.8)
        self.say(r"Real vessels follow it only approximately. So every change must also pass a second gate, on real images.")
        self.play(GrowArrow(a3), FadeIn(real), run_time=0.9)
        ref = Tex(r"Kaipio and Somersalo 2007", color=GREY_B).scale(0.42).next_to(real, DOWN, buff=0.25)
        self.play(FadeIn(ref))
        self.wait(2.2)


variants(Physics, module_globals=globals())
