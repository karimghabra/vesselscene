"""Chapter 11: fork or crossing? Transparency (spec 4.5, 4.5b).

Four arms pair into two straight lines: two vessels at different depths, or
one merged lumen?  Two absorbers in series add their ODs; a fork's lumens
merge into a union.  The test reads the OD at the centre against the arms':
rho = a_c / (a1 + a2).  Where the two predictions converge (one vessel much
fainter), the junction goes to local model comparison.

Render:  manim -ql scenes/s11_transparency.py TransparencyShort
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import transparency as T  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (FB, JUNC, MUTED, OD_COL, PixelFrame, image,  # noqa: E402
                             orientation_color)
from mica_anim.vsdata import JUNCTION_COLORS  # noqa: E402

XC = JUNCTION_COLORS["crossing"]        # crossings: blue
FK = JUNCTION_COLORS["bifurcation"]     # forks: pink
AMB = FB                                # ambiguous -> feedback (V2e), gold

# the level-1 pair: same arms, same angles
A1, A2, W1, W2, D1, D2, SEED = 0.15, 0.10, 6.0, 4.5, 20.0, 100.0, 21
# the ambiguous junction of level 3
AMB_ARGS = dict(a1=0.15, a2=0.15 * 0.09, w1=6.5, w2=4.5, deg1=25.0, deg2=100.0, seed=50)


def v3(v):
    return np.array([v[0], v[1], 0.0])


def camera_view(od, height):
    """What the camera records, I = B exp(-OD), vessels dark."""
    return image(0.85 * np.exp(-od), height, "grey", vmin=0.6, vmax=0.88)


def colour_tex(m, mapping):
    for i, c in mapping.items():
        m[i].set_color(c)
    return m


class Transparency(ChapterScene):
    NUMBER, TITLE, SPEC = 11, "Fork or crossing?", "spec 4.5, 4.5b"
    MAP_KEYS = ("transp", "compare")

    def construct(self):
        self.intro()
        self.level1()
        if not self.at(2):
            self.outro()
            return
        self.level2()
        if not self.at(3):
            self.outro()
            return
        self.model_comparison()
        self.knot_and_gate()
        self.outro()

    # ------------------------------------------------------------------ level 1
    def level1(self):
        H = 2.9
        L1c = orientation_color(np.deg2rad(-D1))
        L2c = orientation_color(np.deg2rad(-D2))
        self.L1c, self.L2c = L1c, L2c
        ods, geos, meas = {}, {}, {}
        for kind in ("crossing", "fork"):
            ods[kind], geos[kind] = T.junction(kind, A1, A2, W1, W2, D1, D2, seed=SEED)
            meas[kind] = T.measure(ods[kind], geos[kind])
        self.ods, self.geos, self.meas = ods, geos, meas
        pfs = {"crossing": PixelFrame(ods["crossing"].shape, H, LEFT * 5.05 + UP * 1.45),
               "fork": PixelFrame(ods["fork"].shape, H, LEFT * 1.75 + UP * 1.45)}
        imgs, borders, titles = {}, {}, {}
        for kind, col, sub in (("crossing", XC, r"two depths"), ("fork", FK, r"one merged lumen")):
            pf = pfs[kind]
            imgs[kind] = camera_view(ods[kind], H).move_to(pf.centre)
            borders[kind] = SurroundingRectangle(imgs[kind], buff=0, color=col, stroke_width=2)
            titles[kind] = VGroup(Tex(kind, color=col).scale(0.62), Tex(sub, color=GREY_B).scale(0.48)
                                  ).arrange(DOWN, buff=0.06).next_to(borders[kind], UP, buff=0.1)
        self.say(r"Four arms pair into two straight lines. Two vessels crossing at different depths, "
                 r"or a fork, where they merge?")
        self.play(*[FadeIn(imgs[k]) for k in imgs], *[Create(borders[k]) for k in borders], run_time=1.0)
        self.play(*[FadeIn(titles[k]) for k in titles], run_time=0.8)
        note = Tex(r"toy: the camera's view, same arms and angles", color=GREY_B).scale(0.48)
        note.next_to(VGroup(borders["crossing"], borders["fork"]), DOWN, buff=0.12)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.2)

        # profiles along the stronger vessel, through the centre
        ax = Axes(x_range=[-24, 24, 8], y_range=[0, 0.3, 0.1], x_length=5.4, y_length=2.75,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=22),
                  x_axis_config=dict(numbers_to_include=[-16, 0, 16]),
                  y_axis_config=dict(numbers_to_include=[0.1, 0.2, 0.3], decimal_number_config=dict(num_decimal_places=1))
                  ).move_to(RIGHT * 3.95 + UP * 1.3)
        xl = Tex(r"along the stronger vessel (px)").scale(0.48).next_to(ax.x_axis, DOWN, buff=0.32)
        yl = Tex(r"OD").scale(0.5).next_to(ax.y_axis, UP, buff=0.08)
        cuts, curves = VGroup(), {}
        for kind, col in (("crossing", XC), ("fork", FK)):
            pf, g = pfs[kind], geos[kind]
            p0, p1 = g["c"] - 24 * g["u1"], g["c"] + 24 * g["u1"]
            cuts.add(DashedLine(pf.point(*p0), pf.point(*p1), color=WHITE, stroke_width=2, dash_length=0.06))
            s, prof = T.profile(ods[kind], g, 1, half=24.0)
            curves[kind] = ax.plot_line_graph(s, prof, line_color=col, add_vertex_dots=False, stroke_width=3.5)
        self.say(r"Read the OD along the stronger vessel. On the arms the two are the same; at the centre they are not.")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(cuts), run_time=1.0)
        self.play(Create(curves["fork"]), run_time=1.0)
        self.play(Create(curves["crossing"]), run_time=1.0)
        lab_x = Tex(r"crossing: the ODs add", color=XC).scale(0.5).move_to(ax.c2p(3.5, 0.285), aligned_edge=LEFT)
        lab_f = Tex(r"fork: a union, not a sum", color=FK).scale(0.5).move_to(ax.c2p(3.5, 0.245), aligned_edge=LEFT)
        self.play(FadeIn(lab_x), FadeIn(lab_f), run_time=0.6)
        self.wait(2.4)

        # rho
        rho_eq = MathTex(r"\rho", "=", r"{a_c", r"\over", r"a_1", "+", r"a_2}").scale(0.85)
        rho_eq[2].set_color(JUNC); rho_eq[4].set_color(L1c); rho_eq[6].set_color(L2c)
        rho_eq.move_to(LEFT * 3.4 + DOWN * 1.65)
        rho_txt = VGroup(Tex(r"$a_c$: the centre's OD", color=JUNC).scale(0.48),
                         Tex(r"$a_1, a_2$: each line's arms", color=GREY_A).scale(0.48)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(rho_eq, RIGHT, buff=0.35)
        nl = NumberLine(x_range=[0.5, 1.1, 0.1], length=5.0, color=MUTED, include_numbers=True, font_size=22,
                        decimal_number_config=dict(num_decimal_places=1)).move_to(RIGHT * 3.95 + DOWN * 1.85)
        nl_l = MathTex(r"\rho", color=GREY_A).scale(0.6).next_to(nl, LEFT, buff=0.2)
        dots = VGroup()
        for kind, col in (("crossing", XC), ("fork", FK)):
            r = meas[kind]["rho"]
            d = Dot(nl.n2p(r), radius=0.09, color=col)
            t = MathTex(f"{r:.2f}", color=col).scale(0.55).next_to(d, UP, buff=0.14)
            dots.add(VGroup(d, t))
        self.say(r"So compare the centre with the arms: $\rho = a_c / (a_1 + a_2)$. "
                 r"The crossing's centre holds both vessels; the fork's, only the stronger.")
        self.play(Write(rho_eq), FadeIn(rho_txt), run_time=1.2)
        self.play(Create(nl), FadeIn(nl_l), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.4), run_time=1.2)
        self.say(rf"Here $\rho$ reads {meas['crossing']['rho']:.2f} against {meas['fork']['rho']:.2f} (toy): "
                 r"the image itself says which it is.", wait="read")
        self.l1 = Group(*imgs.values(), *borders.values(), *titles.values(), note, ax, xl, yl, cuts,
                        *curves.values(), lab_x, lab_f, rho_eq, rho_txt, nl, nl_l, dots)
        self.l1_parts = dict(imgs=imgs, borders=borders, titles=titles, pfs=pfs, rho_eq=rho_eq)

    # ------------------------------------------------------------------ level 2
    def level2(self):
        self.play(FadeOut(self.l1), run_time=0.8)
        L1c, L2c = self.L1c, self.L2c

        # Metelli: two absorbers in series
        f1 = Rectangle(width=5.0, height=1.0, stroke_width=0, fill_color=GREY_B, fill_opacity=0.45).rotate(20 * DEGREES)
        f2 = Rectangle(width=5.0, height=0.8, stroke_width=0, fill_color=GREY_B, fill_opacity=0.45).rotate(100 * DEGREES)
        filt = VGroup(f1, f2).move_to(UP * 0.6)
        ft = Tex(r"two filters in series: the overlap is darker than either", color=GREY_A).scale(0.55)
        ft.next_to(filt, DOWN, buff=0.3)
        bg = Rectangle(width=6.5, height=4.4, stroke_width=0, fill_color=WHITE, fill_opacity=0.85).move_to(filt)
        f1.set_fill(BLACK, 0.45); f2.set_fill(BLACK, 0.45)
        self.say(r"At an X-junction the visual system tests just this: is one surface seen through another? "
                 r"(Metelli; Adelson and Anandan 1990)")
        self.play(FadeIn(bg), run_time=0.5)
        self.play(FadeIn(f1, shift=RIGHT * 0.3), run_time=0.7)
        self.play(FadeIn(f2, shift=LEFT * 0.3), FadeIn(ft), run_time=0.7)
        self.wait(2.2)
        self.say(r"Blood is such an absorber: vessels at two depths add their ODs. At a fork the lumens merge "
                 r"into one volume, a union.", wait="read")
        self.play(FadeOut(VGroup(bg, filt, ft)), run_time=0.6)

        # the measurement, on the OD itself
        od, geo, ms = self.ods["crossing"], self.geos["crossing"], self.meas["crossing"]
        pf = PixelFrame(od.shape, 3.7, LEFT * 4.5 + UP * 0.55)
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        bd = SurroundingRectangle(img, buff=0, color=XC, stroke_width=2)
        ttl = Tex(r"the crossing, as OD", color=GREY_A).scale(0.55).next_to(bd, UP, buff=0.1)
        strips = VGroup()
        for k, col in ((1, L1c), (2, L2c)):
            pts, R = T.arm_samples(geo, k)
            n = len(pts) // 2
            for half in (pts[:n], pts[n:]):
                strips.add(Line(pf.point(*half[0]), pf.point(*half[-1]), stroke_color=col, stroke_width=7,
                                stroke_opacity=0.9).set_stroke(BLACK, width=10, background=True))
        disc = Circle(radius=2.0 * pf.unit, stroke_color=JUNC, stroke_width=3).move_to(pf.point(*geo["c"]))
        Rc = DashedVMobject(Circle(radius=ms["R"] * pf.unit, stroke_color=WHITE, stroke_width=1.5), num_dashes=24
                            ).move_to(pf.point(*geo["c"]))
        e_a1 = MathTex(r"a_1", r"= \operatorname{median}\ \mathrm{OD}", r"\text{ on line 1's arms, } R \ldots R + 2w").scale(0.6)
        e_a2 = MathTex(r"a_2", r"= \operatorname{median}\ \mathrm{OD}", r"\text{ on line 2's arms}").scale(0.6)
        e_ac = MathTex(r"a_c", r"= \max\ \mathrm{OD}", r"\text{ within 2 px of the centre}").scale(0.6)
        e_rho = MathTex(r"\rho", "=", r"{a_c", r"\over", r"a_1", "+", r"a_2}").scale(0.8)
        e_a1[0].set_color(L1c); e_a2[0].set_color(L2c); e_ac[0].set_color(JUNC)
        e_rho[2].set_color(JUNC); e_rho[4].set_color(L1c); e_rho[6].set_color(L2c)
        eqs = VGroup(e_a1, e_a2, e_ac, e_rho).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.75 + UP * 1.55)
        vals = MathTex(rf"\text{{toy: }} a_1 = {ms['a1']:.3f},\ a_2 = {ms['a2']:.3f},\ a_c = {ms['ac']:.3f},\ "
                       rf"\rho = {ms['rho']:.2f}", color=GREY_B).scale(0.5).next_to(eqs, DOWN, buff=0.25).align_to(eqs, LEFT)
        self.say(r"Measured on the OD, never on a render: the median OD along each line's arms, outside the junction, "
                 r"and the maximum at its centre.")
        self.play(FadeIn(img), Create(bd), FadeIn(ttl), run_time=0.8)
        self.play(Create(Rc), run_time=0.5)
        self.play(LaggedStart(*[Create(s) for s in strips[:2]], lag_ratio=0.3), Write(e_a1), run_time=1.2)
        self.play(LaggedStart(*[Create(s) for s in strips[2:]], lag_ratio=0.3), Write(e_a2), run_time=1.2)
        self.play(Create(disc), Write(e_ac), run_time=1.0)
        self.play(Write(e_rho), run_time=1.0)
        self.play(FadeIn(vals), run_time=0.6)
        self.wait(1.4)

        # the two predictions
        p_x = MathTex(r"\text{crossing: }", r"\rho \approx", r"{\max(a_1,a_2) + \kappa \min(a_1,a_2)", r"\over",
                      r"a_1 + a_2}").scale(0.62)
        p_f = MathTex(r"\text{fork: }", r"\rho \approx", r"{\max(a_1,a_2)", r"\over", r"a_1 + a_2}").scale(0.62)
        p_x[0].set_color(XC); p_f[0].set_color(FK)
        preds = VGroup(p_x, p_f).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.75 + DOWN * 1.35)
        kap = Tex(r"$\kappa \approx 0.87$--$0.94$: how fully a crossing adds (spec 4.5)", color=GREY_B).scale(0.46)
        kap.next_to(preds, DOWN, buff=0.15).align_to(preds, LEFT)
        self.say(r"Each hypothesis predicts $\rho$. A crossing: the stronger plus $\kappa$ times the weaker. "
                 r"A fork: the stronger alone.")
        self.play(Write(p_x), run_time=1.4)
        self.play(Write(p_f), FadeIn(kap), run_time=1.2)
        self.wait(1.8)

        # in terms of m
        q_x = MathTex(r"\text{crossing: }", r"\rho \approx", r"{1 + \kappa m", r"\over", r"1 + m}").scale(0.68)
        q_f = MathTex(r"\text{fork: }", r"\rho \approx", r"{1", r"\over", r"1 + m}").scale(0.68)
        q_x[0].set_color(XC); q_f[0].set_color(FK)
        mdef = MathTex(r"m = {\min(a_1,a_2) \over \max(a_1,a_2)}", color=GREY_A).scale(0.62)
        qs = VGroup(mdef, q_x, q_f).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        self.say(r"Divide through by the stronger arm: both depend only on the contrast ratio $m$.")
        self.play(FadeOut(VGroup(e_a1, e_a2, e_ac, vals, kap)), run_time=0.6)
        self.play(e_rho.animate.scale(0.8).move_to(RIGHT * 2.9 + UP * 3.2), run_time=0.8)
        qs.move_to(RIGHT * 2.9 + UP * 1.0)
        self.play(FadeIn(mdef), FadeTransform(p_x, q_x), FadeTransform(p_f, q_f), run_time=1.4)
        self.wait(1.4)

        # the plot: rho against m; the equations become the curves' labels
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0.45, 1.15, 0.1], x_length=6.0, y_length=3.8,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=22),
                  x_axis_config=dict(numbers_to_include=[0, 0.25, 0.5, 0.75, 1.0], decimal_number_config=dict(num_decimal_places=2)),
                  y_axis_config=dict(numbers_to_include=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                                     decimal_number_config=dict(num_decimal_places=1))
                  ).move_to(RIGHT * 2.95 + UP * 0.45)
        xl = MathTex(r"m", r"\text{ (fainter arm / stronger arm)}").scale(0.55).next_to(ax.x_axis, DOWN, buff=0.38)
        yl = MathTex(r"\rho").scale(0.7).next_to(ax.y_axis, LEFT, buff=0.45).shift(UP * 0.4)
        cx = ax.plot(lambda m: T.curve_cross(m), x_range=[0.0, 1.0], color=XC, stroke_width=4)
        cf = ax.plot(lambda m: T.curve_fork(m), x_range=[0.0, 1.0], color=FK, stroke_width=4)
        lx = MathTex(r"\text{crossing: }", r"{1 + \kappa m", r"\over", r"1 + m}").scale(0.55)
        lf = MathTex(r"\text{fork: }", r"{1", r"\over", r"1 + m}").scale(0.55)
        lx[0].set_color(XC); lf[0].set_color(FK)
        lx.move_to(ax.c2p(0.8, 1.1))
        lf.move_to(ax.c2p(0.4, 0.6))
        self.play(FadeOut(VGroup(strips, Rc, disc, e_rho, mdef)), run_time=0.6)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.9)
        self.play(Create(cx), Create(cf), FadeTransform(q_x, lx), FadeTransform(q_f, lf), run_time=1.6)
        self.wait(0.6)

        # a live junction: fade the weaker vessel
        mt = ValueTracker(ms["m"])

        def live_img():
            m = mt.get_value()
            o, _ = T.junction("crossing", A1, A1 * m, W1, W2, D1, D2, seed=SEED)
            return image(o, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)

        live = always_redraw(live_img)
        dx = always_redraw(lambda: Dot(ax.c2p(mt.get_value(), T.curve_cross(mt.get_value())), radius=0.09, color=XC))
        dfk = always_redraw(lambda: Dot(ax.c2p(mt.get_value(), T.curve_fork(mt.get_value())), radius=0.09, color=FK))
        gap = always_redraw(lambda: Line(ax.c2p(mt.get_value(), T.curve_fork(mt.get_value())),
                                         ax.c2p(mt.get_value(), T.curve_cross(mt.get_value())),
                                         stroke_color=WHITE, stroke_width=2))
        sep = always_redraw(lambda: MathTex(
            rf"\text{{separation}} = {T.curve_cross(mt.get_value()) - T.curve_fork(mt.get_value()):.2f}",
            color=GREY_A).scale(0.5).next_to(ax.c2p(mt.get_value(), (T.curve_cross(mt.get_value()) + T.curve_fork(mt.get_value())) / 2),
                                              LEFT if mt.get_value() > 0.55 else RIGHT, buff=0.15).shift(
                DOWN * max(0.0, 0.3 - mt.get_value()) * 2.2))
        mlab = always_redraw(lambda: MathTex(rf"m = {mt.get_value():.2f}", color=GREY_A).scale(0.55).next_to(bd, DOWN, buff=0.15))
        self.add(live)
        self.remove(img)
        self.play(FadeIn(dx), FadeIn(dfk), Create(gap), FadeIn(sep), FadeIn(mlab), FadeOut(lx), run_time=0.8)
        self.say(r"Fade the weaker vessel: the two predictions converge. The separation is $\kappa m/(1+m)$, "
                 r"nothing when one vessel is much fainter.")
        self.play(mt.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        self.play(mt.animate.set_value(0.04), run_time=4.0, rate_func=smooth)
        self.wait(1.2)
        for m_ in (live, dx, dfk, gap, sep, mlab):
            m_.clear_updaters()
        self.play(FadeOut(Group(live, dx, dfk, gap, sep, mlab, bd, ttl)), FadeIn(lx), run_time=0.7)

        # the toy population: points, noise bands, the ambiguous zone
        mc = T.monte_carlo()
        grid = np.linspace(0.02, 1.0, 60)
        bands = VGroup()
        for kind, col in (("crossing", XC), ("fork", FK)):
            mu, sd = T.band_mean_sd(kind, grid)
            top = [ax.c2p(m, min(v, 1.15)) for m, v in zip(grid, mu + 2 * sd)]
            bot = [ax.c2p(m, v) for m, v in zip(grid[::-1], (mu - 2 * sd)[::-1])]
            bands.add(Polygon(*top, *bot, stroke_width=0, fill_color=col, fill_opacity=0.2))
        pts = VGroup(*[Dot(ax.c2p(m, r), radius=0.035, color=XC if c else FK, fill_opacity=0.9)
                       for c, m, r in zip(mc["is_cross"], mc["m"], mc["rho"]) if r <= 1.15])
        self.say(r"Now 240 toy junctions, crossings and forks with noise, at many contrast ratios.")
        self.play(LaggedStart(*[FadeIn(p, scale=0.5) for p in pts], lag_ratio=0.01), run_time=2.4)
        self.wait(0.6)
        self.say(r"Each hypothesis has a noise band (toy: $\pm 2$ SD). Noise lifts the 2 px maximum, so both clouds "
                 r"sit a little above their lines.")
        self.play(FadeIn(bands), run_time=1.0)
        self.wait(2.4)

        # the ambiguous zone
        m_star, band = T.ambiguous_zone()
        zone = VGroup(Polygon(ax.c2p(0, 0.45), ax.c2p(m_star, 0.45), ax.c2p(m_star, 1.15), ax.c2p(0, 1.15),
                              stroke_width=0, fill_color=AMB, fill_opacity=0.3))
        if len(band) > 2:
            up = [ax.c2p(m, y1) for m, y0, y1 in band]
            lo = [ax.c2p(m, y0) for m, y0, y1 in band[::-1]]
            zone.add(Polygon(*up, *lo, stroke_width=0, fill_color=AMB, fill_opacity=0.5))
        amb = T.ambiguous(mc["rho"], mc["m"])
        lr_eq = MathTex(r"\frac{1}{3} <", r"\frac{P(\rho \mid \text{crossing})}{P(\rho \mid \text{fork})}", r"< 3",
                        color=AMB).scale(0.55)
        lr_t = Tex(r"or the lines too close: ambiguous", color=AMB).scale(0.5)
        lr = VGroup(lr_eq, lr_t).arrange(DOWN, buff=0.12).move_to(LEFT * 4.5 + UP * 1.5)
        share = Tex(rf"toy: {amb.mean() * 100:.0f}\% ambiguous, all at $m < {mc['m'][amb].max():.2f}$", color=GREY_A).scale(0.5)
        share.next_to(lr, DOWN, buff=0.35)
        llr = T.log_lr(mc["rho"], mc["m"])
        acc = ((llr > 0) == (mc["is_cross"] == 1))[~amb].mean()
        acc_t = Tex(rf"the rest: {acc * 100:.1f}\% typed right", color=GREY_A).scale(0.5).next_to(share, DOWN, buff=0.12)
        self.say(r"A likelihood ratio from the local noise decides. Between 1/3 and 3, or where the lines are too close, "
                 r"the junction is ambiguous.")
        self.play(FadeIn(zone), FadeIn(lr), run_time=1.2)
        self.play(FadeIn(share), FadeIn(acc_t), run_time=0.8)
        self.wait(1.6)
        arrow = Arrow(acc_t.get_bottom() + DOWN * 0.3, acc_t.get_bottom() + DOWN * 1.2, color=AMB, buff=0,
                      stroke_width=4, max_tip_length_to_length_ratio=0.3)
        v2e = Tex(r"to V2e: model comparison", color=AMB).scale(0.55).next_to(arrow, DOWN, buff=0.1)
        self.say(r"Those go up a level, to local model comparison: top-down feedback at one junction.")
        self.play(GrowArrow(arrow), FadeIn(v2e), run_time=0.8)
        self.wait(1.8)
        self.l2 = VGroup(ax, xl, yl, cx, cf, lx, lf, bands, pts, zone, lr, share, acc_t, arrow, v2e)

    # ------------------------------------------------------------------ level 3: V2e
    def model_comparison(self):
        self.play(FadeOut(self.l2), run_time=0.8)
        od, geo = T.junction("crossing", **AMB_ARGS)
        fits, ms = T.compare(od, geo)
        names = ["crossing", "fork, parent 1", "fork, parent 2"]
        totals = {k: fits[k]["total"] for k in names}
        best = min(totals, key=totals.get)
        sig = T.NOISE
        H = 1.95
        xs = [-5.35, -2.95, -0.55, 1.85]
        y_top, y_bot = 1.75, -0.65
        win = fits["crossing"]["window"]

        def framed(arr, x, y, ramp, vmin, vmax, col=MUTED):
            im = image(arr, H, ramp, vmin=vmin, vmax=vmax).move_to(RIGHT * x + UP * y)
            return Group(im, SurroundingRectangle(im, buff=0, color=col, stroke_width=1.2))

        data = framed(od, xs[0], y_top, "magma", -0.02, 0.2)
        data_t = Tex(r"the OD (toy)").scale(0.5).next_to(data, UP, buff=0.08)
        r_t = MathTex(rf"\rho = {ms['rho']:.2f},\ m = {ms['m']:.2f}", color=AMB).scale(0.5).next_to(data, DOWN, buff=0.1)
        amb_t = Tex(r"ambiguous", color=AMB).scale(0.5).next_to(r_t, DOWN, buff=0.06)
        self.say(r"An ambiguous toy junction: a faint vessel crosses a strong one, or joins it. $\rho$ cannot tell.")
        self.play(FadeIn(data), FadeIn(data_t), run_time=0.8)
        self.play(FadeIn(r_t), FadeIn(amb_t), run_time=0.6)
        self.wait(1.6)

        cols = {"crossing": XC, "fork, parent 1": FK, "fork, parent 2": FK}
        labels = {"crossing": r"crossing ($\kappa$)", "fork, parent 1": r"fork, parent: line 1",
                  "fork, parent 2": r"fork, parent: line 2"}
        renders, resids, heads = Group(), Group(), VGroup()
        for k, x in zip(names, xs[1:]):
            renders.add(framed(fits[k]["model"], x, y_top, "magma", -0.02, 0.2, cols[k]))
            heads.add(Tex(labels[k], color=cols[k]).scale(0.5).next_to(renders[-1], UP, buff=0.08))
            rs = gaussian_filter(np.where(win, fits[k]["resid"], 0.0), 1.0)
            resids.add(framed(rs, x, y_bot, "diverging", -1.2 * sig, 1.2 * sig, cols[k]))
        row_r = Tex(r"each hypothesis, rendered and fitted", color=GREY_B).scale(0.46)
        row_r.next_to(renders, DOWN, buff=0.08)
        row_e = Tex(r"residual, OD $-$ render (smoothed 1 px)", color=GREY_B).scale(0.46).next_to(resids, DOWN, buff=0.08)
        self.say(r"Build each hypothesis from the junction's arms: the crossing, and a fork with each line as the parent. "
                 r"Union at nodes, $\kappa$ at the crossing.")
        self.play(LaggedStart(*[FadeIn(r) for r in renders], lag_ratio=0.25), FadeIn(heads), run_time=1.6)
        self.play(FadeIn(row_r), run_time=0.4)
        self.wait(1.4)
        self.say(r"Fit each one's profiles for 20 iterations, positions fixed. The forks leave the centre unexplained.")
        self.play(LaggedStart(*[FadeIn(r) for r in resids], lag_ratio=0.25), FadeIn(row_e), run_time=1.6)
        ring = Circle(radius=0.32, stroke_color=WHITE, stroke_width=2).move_to(resids[1][0].get_center())
        ring2 = ring.copy().move_to(resids[2][0].get_center())
        self.play(Create(ring), Create(ring2), run_time=0.8)
        self.wait(1.6)

        # costs
        bx0, bw = 3.35, 3.2
        bars = VGroup()
        vmax = max(max(fits[k]["chi"] - min(fits[q]["chi"] for q in names) + fits[k]["desc"] for k in names), 1.0)
        chi0 = min(fits[q]["chi"] for q in names)
        for i, k in enumerate(names):
            y = y_top - 0.15 - 0.72 * i
            dchi = fits[k]["chi"] - chi0
            dd = fits[k]["desc"]
            r1 = Rectangle(width=max(bw * dchi / vmax, 1e-3), height=0.3, stroke_width=0, fill_color=OD_COL, fill_opacity=0.85)
            r2 = Rectangle(width=max(bw * dd / vmax, 1e-3), height=0.3, stroke_width=0, fill_color=GREY_B, fill_opacity=0.85)
            r1.move_to([bx0, y, 0], aligned_edge=LEFT)
            r2.next_to(r1, RIGHT, buff=0)
            t = MathTex(f"{dchi + dd:.1f}", color=cols[k]).scale(0.5).next_to(r2, RIGHT, buff=0.1)
            lab = Tex(labels[k], color=cols[k]).scale(0.42).next_to(r1, UP, buff=0.05, aligned_edge=LEFT).align_to(r1, LEFT)
            bars.add(VGroup(r1, r2, t, lab))
        key = VGroup(Tex(r"weighted residual (above the lowest)", color=OD_COL).scale(0.42),
                     Tex(r"+ description cost", color=GREY_B).scale(0.42)).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        key.next_to(bars, DOWN, buff=0.18).align_to(bars, LEFT)
        dc = MathTex(r"\text{description cost} = 0.25 \cdot 0.5 \cdot n_\text{params} \cdot \ln\max(\text{pixels}/6,\ 2)",
                     r"\text{ per edge (spec 4.5b)}", color=GREY_A).scale(0.48)
        dc.move_to(RIGHT * 0.3 + DOWN * 2.2)
        self.say(r"Keep the hypothesis with the lowest weighted residual plus description cost: the fit's own prune cost per edge.")
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in bars], lag_ratio=0.3), FadeIn(key), run_time=1.6)
        self.play(FadeIn(dc), run_time=0.6)
        win_i = names.index(best)
        hl = SurroundingRectangle(Group(renders[win_i], resids[win_i], heads[win_i]), color=cols[best], buff=0.08,
                                  corner_radius=0.06, stroke_width=3)
        self.play(Create(hl), Indicate(bars[win_i], color=cols[best], scale_factor=1.05), run_time=1.0)
        self.wait(1.2)
        pop = T.compare_population()
        n_ok = int((pop[:, 0] == pop[:, 1]).sum())
        res_t = Tex(rf"toy: on all {len(pop)} ambiguous toy junctions, model comparison types {n_ok} right", color=GREY_A
                    ).scale(0.5).move_to(RIGHT * 0.3 + DOWN * 2.56)
        sp = Tex(r"spec 4.5b: about 0.2--0.5 s per ambiguous junction",
                 color=GREY_B).scale(0.46).next_to(res_t, DOWN, buff=0.07)
        self.say(r"Predictive coding (Rao and Ballard 1999): the renderer acts as the higher area, and its prediction error "
                 r"picks the interpretation, at one junction only.")
        self.play(FadeIn(res_t), FadeIn(sp), run_time=0.8)
        self.wait(2.6)
        self.l3 = Group(data, data_t, r_t, amb_t, renders, resids, heads, row_r, row_e, ring, ring2, bars, key, dc, hl,
                        res_t, sp)

    # ------------------------------------------------------------------ level 3: the knot cue, the gate
    def knot_and_gate(self):
        self.play(FadeOut(self.l3), run_time=0.8)
        head = Tex(r"What failed before (spec 2.3): the `knot' cue").scale(0.62).move_to(LEFT * 3.4 + UP * 3.2)
        sub = Tex(r"the residual of the additive render at junction centres,\\in thinner-arm contrasts (medians, measured)",
                  color=GREY_B).scale(0.48).next_to(head, DOWN, buff=0.15)
        nl = NumberLine(x_range=[-3, 0.5, 0.5], length=5.8, color=MUTED, include_numbers=True, font_size=22,
                        decimal_number_config=dict(num_decimal_places=1)).move_to(LEFT * 3.4 + UP * 0.9)
        zero = DashedLine(nl.n2p(0) + UP * 0.9, nl.n2p(0) + DOWN * 0.3, color=WHITE, stroke_width=1.5)
        meds = [(-2.4, "crossings", XC), (-1.8, "compounds", JUNCTION_COLORS["compound"]), (-1.4, "3-ways", FK)]
        mk = VGroup()
        for i, (v, name, col) in enumerate(meds):
            d = Triangle(color=col, fill_opacity=1, stroke_width=0).scale(0.1).rotate(PI).move_to(nl.n2p(v) + UP * 0.18)
            t = Tex(rf"{name} ${v}$", color=col).scale(0.46).next_to(d, UP, buff=0.08 + 0.32 * (i % 2))
            mk.add(VGroup(d, t))
        why = Tex(r"all below zero: the additive render over-predicts every junction,\\so the cue cannot separate them",
                  color=GREY_A).scale(0.48).next_to(nl, DOWN, buff=0.45)
        fix = Tex(r"the spec measures transparency on the OD itself", color=JUNC).scale(0.5).next_to(why, DOWN, buff=0.2)
        self.say(r"An earlier cue read the residual of the additive render at junction centres. It did not separate types.")
        self.play(FadeIn(head), FadeIn(sub), Create(nl), Create(zero), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(m, shift=DOWN * 0.2) for m in mk], lag_ratio=0.3), run_time=1.2)
        self.say(r"The additive render over-predicts every junction, so every residual is negative. "
                 r"Transparency is read from the OD, with no render involved.")
        self.play(FadeIn(why), run_time=0.6)
        self.play(FadeIn(fix), run_time=0.6)
        self.wait(1.8)

        # the gate, and the toy's AUC by contrast ratio
        mc = T.monte_carlo()
        bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.01)]
        aucs = []
        for lo, hi in bins:
            s = (mc["m"] >= lo) & (mc["m"] < hi)
            aucs.append(T.auc(mc["rho"][s & (mc["is_cross"] == 1)], mc["rho"][s & (mc["is_cross"] == 0)]))
        ax = Axes(x_range=[0, 4, 1], y_range=[0.5, 1.0, 0.1], x_length=4.2, y_length=2.3,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=20),
                  y_axis_config=dict(numbers_to_include=[0.5, 0.75, 1.0], decimal_number_config=dict(num_decimal_places=2)),
                  x_axis_config=dict(include_ticks=False)).move_to(RIGHT * 3.75 + UP * 0.45)
        bars = VGroup()
        for i, (a, (lo, hi)) in enumerate(zip(aucs, bins)):
            h = ax.c2p(0, a)[1] - ax.c2p(0, 0.5)[1]
            r = Rectangle(width=0.7, height=max(h, 1e-3), stroke_width=0, fill_color=XC if hi > 0.5 else GREY_B, fill_opacity=0.85)
            r.move_to(ax.c2p(i + 0.5, 0.5), aligned_edge=DOWN)
            t = MathTex(f"{a:.2f}").scale(0.45).next_to(r, UP, buff=0.06)
            lab = MathTex([r"< 0.15", r"0.15\text{--}0.3", r"0.3\text{--}0.5", r"\geq 0.5"][i]).scale(0.42)
            lab.next_to(ax.c2p(i + 0.5, 0.5), DOWN, buff=0.12)
            bars.add(VGroup(r, t, lab))
        mlab = MathTex(r"m", color=GREY_B).scale(0.5).next_to(ax.c2p(4, 0.5), RIGHT, buff=0.15).shift(DOWN * 0.22)
        gate = DashedLine(ax.c2p(0, 0.75), ax.c2p(4, 0.75), color=FB, stroke_width=2.5)
        gate_l = Tex(r"gate", color=FB).scale(0.45).next_to(gate, RIGHT, buff=0.08)
        ghead = Tex(r"the phase-2 gate (spec 6.2)").scale(0.62).move_to(RIGHT * 3.9 + UP * 3.2)
        gtxt = Tex(r"fork / crossing AUC of $\rho \geq 0.75$ where $m \geq 0.5$;\\typing up by at least 10 points",
                   color=GREY_A).scale(0.48).next_to(ghead, DOWN, buff=0.12)
        yl = Tex(r"toy AUC of $\rho$", color=GREY_B).scale(0.45).next_to(ax.y_axis, UP, buff=0.12).shift(RIGHT * 0.3)
        self.say(r"The phase-2 gate: an AUC of at least 0.75 where the fainter arm is at least half the stronger.")
        self.play(FadeIn(ghead), FadeIn(gtxt), run_time=0.8)
        self.play(Create(ax), FadeIn(yl), FadeIn(mlab), Create(gate), FadeIn(gate_l), run_time=0.8)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.2),
                  *[FadeIn(b[1:]) for b in bars], run_time=1.4)
        honest = Tex(r"The toy passes easily: the same chord law\\draws and measures it. The gate is measured\\"
                     r"on vesselscene's scenes, and real images differ.", color=GREY_A).scale(0.46)
        honest.next_to(ax, DOWN, buff=0.5)
        self.say(r"The toy passes easily, as it should: one model both draws and measures it. The gate is for real scenes.")
        self.play(FadeIn(honest), run_time=0.8)
        self.wait(1.8)
        oq = Tex(r"Open question (spec 10): which $\kappa$, vesselscene's reference 0.87,\\or the image's fitted value (0.94 on development healthy 7)?",
                 color=GREY_A).scale(0.48).move_to(LEFT * 3.4 + DOWN * 2.35)
        self.say(r"And an open question: which $\kappa$ to predict with, the reference 0.87 or each image's fitted value.")
        self.play(FadeIn(oq), run_time=0.8)
        self.wait(2.4)


variants(Transparency, module_globals=globals())
