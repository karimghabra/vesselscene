"""Chapter 16: population codes (spec 4.12).

Sixteen orientation cells see a vessel at 37 degrees.  The best one says 33.75:
today only the winners are used.  Read as a population (each cell an arrow at
twice its angle, added head to tail), they say 37.  Then why the angle is
doubled, a sweep of the vessel's angle (winner's sawtooth against the
population vector), sub-pixel position from a parabola, and (detailed) width,
blur and contrast from their own populations, used to start the spline fit.

Render:  manim -ql scenes/s16_population_codes.py PopulationCodesMedium
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import popcode as P  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import EXC, MUTED, OD_COL, PixelFrame, colormap, image, orientation_color  # noqa: E402

TRUE = 37.0                 # the vessel's screen angle (degrees)
NEAR0 = 4.0                 # a nearly horizontal vessel, for the doubling
SIZE_COL = TEAL_C           # size channels (no fixed meaning in style.py; used only here)


def hue(alpha_deg):
    """Orientation hue of a SCREEN angle (style's hue is keyed on image angles)."""
    return orientation_color(np.deg2rad(-alpha_deg) % np.pi)


def cdir(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


def cv(z):
    """A complex number as a scene vector."""
    return np.array([float(np.real(z)), float(np.imag(z)), 0.0])


def arrow(a, b, color, width=5.0, tip=0.13):
    return Arrow(a, b, buff=0, stroke_width=width, tip_length=tip, max_tip_length_to_length_ratio=0.4,
                 max_stroke_width_to_length_ratio=14, color=color)


def deg(x, nd=1):
    return rf"{x:.{nd}f}^\circ"


class PopulationCodes(ChapterScene):
    NUMBER, TITLE, SPEC = 16, "Population codes", "spec 4.12"
    MAP_KEYS = ("popcode", "fit")

    def construct(self):
        self.intro()
        self.population_vector()
        if self.at(2):
            self.why_doubled()
            self.angle_sweep()
            self.subpixel_position()
        if self.at(3):
            self.other_parameters()
            self.start_the_fit()
        self.outro()

    # ================================================================== level 1
    def ring(self, centre, radius, mult=2.0, width=5, n=72):
        """The orientations 0..180 drawn at mult x their angle, in their hues."""
        g = VGroup()
        for i in range(n):
            a0, a1 = 180.0 * i / n, 180.0 * (i + 1) / n
            g.add(Arc(radius=radius, start_angle=np.deg2rad(mult * a0), angle=np.deg2rad(mult * (a1 - a0)),
                      arc_center=centre, stroke_width=width, stroke_color=hue(0.5 * (a0 + a1))))
        return g

    def ring_labels(self, centre, radius, mult=2.0, with_180=False):
        g = VGroup()
        for a in ((0, 45, 90, 135, 180) if with_180 else (0, 45, 90, 135)):
            t = MathTex(rf"{a}^\circ", color=hue(a % 180)).scale(0.45)
            off = cdir(mult * a + 90) * (0.17 if a != 180 else -0.17)
            t.move_to(centre + cdir(mult * a) * (radius + 0.28) + off)
            g.add(t)
        return g

    def population_vector(self):
        C = P.rates(TRUE)
        od = P.patch(TRUE)[0]
        jw = int(np.argmax(C))
        pop = P.decode(C)

        # ------------------------------------------------ the puzzle: a vessel, sixteen cells, one winner
        pf = PixelFrame(P.SHAPE, 4.3, LEFT * 4.6 + UP * 0.55)
        self.pf = pf
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"a vessel at $37^\circ$ (toy)").scale(0.6).next_to(border, UP, buff=0.15)
        probe = Dot(pf.point(48, 48), radius=0.06, color=YELLOW, z_index=5)
        self.play(FadeIn(img), Create(border), FadeIn(head))
        self.play(FadeIn(probe, scale=2))

        ax = Axes(x_range=[-10, 180, 45], y_range=[0, 80, 20], x_length=6.0, y_length=3.2,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5),
                  ).move_to(RIGHT * 3.4 + UP * 0.2)
        xt = VGroup(*[MathTex(rf"{a}^\circ").scale(0.45).next_to(ax.c2p(a, 0), DOWN, buff=0.12) for a in (0, 45, 90, 135, 180)])
        yt = VGroup(*[MathTex(f"{v}").scale(0.42).next_to(ax.c2p(-10, v), LEFT, buff=0.1) for v in (20, 40, 60, 80)])
        xl = Tex(r"preferred orientation $\theta_k$").scale(0.52).next_to(xt, DOWN, buff=0.12)
        yl = Tex(r"rate $C_k$ (noise units)").scale(0.52).rotate(PI / 2).next_to(yt, LEFT, buff=0.15)
        bars = VGroup(*[Line(ax.c2p(P.PHI[j], 0), ax.c2p(P.PHI[j], max(C[j], 0.15)), stroke_width=15,
                             stroke_color=hue(P.PHI[j]), cap_style=CapStyleType.BUTT) for j in range(16)])
        self.say(r"Sixteen orientation cells at the yellow point. Each fires by how well its orientation fits.")
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl))
        self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.08), run_time=2.0)

        top = ax.c2p(0, 80)[1] + 0.3
        win_lbl = Tex(r"best: $33.75^\circ$", color=hue(P.PHI[jw])).scale(0.55)
        win_lbl.move_to([ax.c2p(P.PHI[jw], 0)[0], top, 0], aligned_edge=RIGHT).shift(LEFT * 0.08)
        true_ln = DashedLine(ax.c2p(TRUE, 0), ax.c2p(TRUE, 80), color=WHITE, stroke_width=2, dash_length=0.06)
        true_lbl = Tex(r"vessel: $37^\circ$").scale(0.55)
        true_lbl.move_to([ax.c2p(TRUE, 0)[0], top, 0], aligned_edge=LEFT).shift(RIGHT * 0.08)
        win_line = Line(probe.get_center() - 2.0 * cdir(P.PHI[jw]), probe.get_center() + 2.0 * cdir(P.PHI[jw]),
                        stroke_width=3, color=hue(P.PHI[jw]))
        self.say(r"Today only the winner is used: the tangent is the best cell's angle, in steps of $11.25^\circ$.")
        self.play(Indicate(bars[jw], color=WHITE, scale_factor=1.08), FadeIn(win_lbl))
        self.play(Create(win_line), run_time=1.0)
        self.play(Create(true_ln), FadeIn(true_lbl))
        self.wait(1.0)
        self.say(r"The winner says $33.75^\circ$; the vessel is at $37^\circ$. But look at the losers.", wait=1.0)
        nb = VGroup(bars[2], bars[4])
        nbl = VGroup(MathTex(f"{C[2]:.0f}", color=hue(P.PHI[2])).scale(0.5).next_to(bars[2], UP, buff=0.08),
                     MathTex(f"{C[4]:.0f}", color=hue(P.PHI[4])).scale(0.5).next_to(bars[4], UP, buff=0.08))
        self.say(r"The $45^\circ$ cell fires far more than the $22.5^\circ$ cell: the vessel leans toward $45^\circ$.")
        self.play(Indicate(nb, color=WHITE, scale_factor=1.06), FadeIn(nbl))
        self.wait(1.0)

        # ------------------------------------------------ every cell an arrow at twice its angle
        O = RIGHT * 0.45 + DOWN * 1.35
        z = P.pop_vector(C)
        SC = 3.45 / abs(z)
        R0 = 1.35
        re_ax = Line(O + LEFT * 1.75, O + RIGHT * 1.75, stroke_color=MUTED, stroke_width=1.2)
        im_ax = Line(O + DOWN * 1.6, O + UP * 3.65, stroke_color=MUTED, stroke_width=1.2)
        axl = VGroup(MathTex(r"\mathrm{Re}", color=MUTED).scale(0.45).next_to(re_ax, RIGHT, buff=0.08),
                     MathTex(r"\mathrm{Im}", color=MUTED).scale(0.45).next_to(im_ax, UP, buff=0.06))
        ring = self.ring(O, R0)
        rlab = self.ring_labels(O, R0)
        eq = MathTex(r"\theta", "=", r"\tfrac{1}{2}", r"\arg", r"\sum_k", r"C_k", r"\,e^{2i\theta_k}").scale(0.9)
        eq.to_corner(UR, buff=0.45)
        eq[6].set_color_by_gradient(*[hue(a) for a in (0, 45, 90, 135)])
        sig = [j for j in range(16) if C[j] > 0.8]
        arrows = {j: arrow(O, O + SC * C[j] * cdir(2 * P.PHI[j]), hue(P.PHI[j])) for j in sig}

        # the chart moves aside, small, to feed the plane
        mini = VGroup(ax, xt, yt, bars, true_ln)
        mini_t = Tex(r"the 16 rates", color=GREY_B).scale(0.5)
        self.say(r"Make each cell an arrow: as long as its rate $C_k$, pointing at \emph{twice} its angle.")
        self.play(FadeOut(VGroup(xl, yl, win_lbl, true_lbl, nbl)), run_time=0.6)
        self.play(mini.animate.scale(0.5).move_to(RIGHT * 5.0 + DOWN * 1.35), run_time=1.2)
        mini_t.next_to(mini, UP, buff=0.12)
        self.play(FadeIn(mini_t), Create(re_ax), Create(im_ax), FadeIn(axl), Create(ring), FadeIn(rlab), run_time=1.4)
        # the winner first, with its angle
        self.play(ReplacementTransform(bars[jw].copy(), arrows[jw]), bars[jw].animate.set_opacity(0.35), run_time=1.3)
        arc1 = Arc(radius=0.45, start_angle=0, angle=np.deg2rad(2 * P.PHI[jw]), arc_center=O, stroke_width=2.5,
                   color=hue(P.PHI[jw]))
        arc1_l = MathTex(r"2\theta_k", color=hue(P.PHI[jw])).scale(0.5).move_to(O + cdir(P.PHI[jw]) * 0.72 + DOWN * 0.04)
        self.play(Create(arc1), FadeIn(arc1_l), FadeIn(eq[5]), FadeIn(eq[6]))
        self.wait(0.8)
        self.play(FadeOut(arc1), FadeOut(arc1_l),
                  LaggedStart(*[AnimationGroup(ReplacementTransform(bars[j].copy(), arrows[j]), bars[j].animate.set_opacity(0.35))
                                for j in sig if j != jw], lag_ratio=0.12), run_time=2.6)
        self.wait(0.5)

        # ------------------------------------------------ head to tail
        self.say(r"Now let them vote: add the arrows head to tail.")
        order = sorted(sig, key=lambda j: (2 * P.PHI[j] + 90) % 360)        # from the most clockwise round
        tip = O.copy()
        for j in order:
            a = arrows[j]
            self.play(a.animate.shift(tip - a.get_start()), run_time=0.45 if C[j] > 10 else 0.25)
            tip = tip + SC * C[j] * cdir(2 * P.PHI[j])
        res = arrow(O, O + SC * cv(z), WHITE, width=7, tip=0.2)
        self.play(GrowArrow(res), FadeIn(eq[4]), run_time=1.2)
        self.say(r"The sum points at twice the vessel's angle. Halve it.")
        ang2 = np.rad2deg(np.angle(z))
        arc2 = Arc(radius=0.55, start_angle=0, angle=np.deg2rad(ang2), arc_center=O, stroke_width=2.5, color=WHITE)
        arc2_l = MathTex(r"2\theta =", deg(ang2), color=WHITE).scale(0.5).next_to(O + SC * cv(z), UP, buff=0.12)
        self.play(Create(arc2), FadeIn(arc2_l))
        half = DashedLine(O, O + 2.55 * cdir(ang2 / 2), color=WHITE, stroke_width=3, dash_length=0.08)
        arc3 = Arc(radius=1.0, start_angle=0, angle=np.deg2rad(ang2 / 2), arc_center=O, stroke_width=3, color=WHITE)
        arc3_l = MathTex(r"\theta =", deg(pop), color=WHITE).scale(0.55).next_to(half.get_end(), RIGHT, buff=0.12)
        arc_c = arc2.copy()
        self.play(Create(half), Transform(arc_c, arc3), FadeIn(arc3_l), FadeIn(eq[0:4]), run_time=1.5)
        self.wait(0.6)

        # ------------------------------------------------ back onto the vessel
        pop_line = Line(probe.get_center() - 2.0 * cdir(pop), probe.get_center() + 2.0 * cdir(pop), stroke_width=3, color=WHITE)
        self.play(TransformFromCopy(half, pop_line), run_time=1.4)
        rd = VGroup(
            Tex(r"vessel ", r"$37^\circ$").scale(0.6),
            Tex(r"best cell ", r"$33.75^\circ$", color=hue(P.PHI[jw])).scale(0.6),
            Tex(r"population ", rf"${pop:.1f}^\circ$").scale(0.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(border, DOWN, buff=0.18).align_to(border, LEFT)
        rd_note = Tex(r"toy hypercolumn,", r" noise 0.01 OD", color=GREY_B).scale(0.45).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        rd_note.next_to(rd, RIGHT, buff=0.35).align_to(rd, DOWN)
        self.play(FadeIn(rd, lag_ratio=0.3), FadeIn(rd_note))
        self.say(r"Every cell votes, weighted by its rate: the population reads the angle to a fraction of a degree.", wait="read")
        self.pv = dict(img=img, border=border, head=head, probe=probe, O=O, SC=SC, R0=R0, re_ax=re_ax, im_ax=im_ax,
                       axl=axl, ring=ring, rlab=rlab, eq=eq, mini=VGroup(mini, mini_t),
                       clutter=VGroup(*arrows.values(), res, arc2, arc2_l, arc_c, half, arc3_l, pop_line, win_line, rd, rd_note))

    # ================================================================== level 2
    def why_doubled(self):
        v = self.pv
        O, SC, R0 = v["O"], v["SC"], v["R0"]
        C4 = P.rates(NEAR0)
        od4 = P.patch(NEAR0)[0]
        self.play(FadeOut(v["clutter"]), FadeOut(v["mini"]), run_time=0.8)
        img4 = image(od4, self.pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(self.pf.centre)
        head4 = Tex(r"a vessel at $4^\circ$ (toy)").scale(0.6).move_to(v["head"])
        self.say(r"Why twice the angle? Take a vessel at $4^\circ$, and try it \emph{without} doubling.")
        self.play(FadeOut(v["img"]), FadeIn(img4), ReplacementTransform(v["head"], head4), run_time=1.0)
        v["img"], v["head"] = img4, head4

        m = ValueTracker(2.0)
        ring = always_redraw(lambda: self.ring(O, R0, mult=m.get_value()))
        rlab = always_redraw(lambda: self.ring_labels(O, R0, mult=m.get_value(), with_180=True))
        self.remove(v["ring"], v["rlab"])
        self.add(ring, rlab)
        eq1 = MathTex(r"\theta", "=", r"\arg", r"\sum_k", r"C_k", r"\,e^{i\theta_k}").scale(0.9).move_to(v["eq"], aligned_edge=RIGHT)
        eq1[5].set_color_by_gradient(*[hue(a) for a in (0, 45, 90, 135)])
        eq2 = v["eq"].copy()
        self.play(m.animate.set_value(1.0), TransformMatchingTex(v["eq"], eq1), run_time=2.2)
        sig = [j for j in range(16) if C4[j] > 0.8]

        def arrows_now():
            mm = m.get_value()
            return VGroup(*[arrow(O, O + SC * C4[j] * cdir(mm * P.PHI[j]), hue(P.PHI[j])) for j in sig])

        def resultant_now():
            zz = P.pop_vector(C4, m.get_value())
            return arrow(O, O + SC * cv(zz), WHITE, width=7, tip=0.2)

        def decoded_now():
            mm = m.get_value()
            return (np.rad2deg(np.angle(P.pop_vector(C4, mm))) / mm) % (360.0 / mm) % 180.0

        arr = always_redraw(arrows_now)
        res = always_redraw(resultant_now)
        self.play(FadeIn(arr), run_time=1.0)
        self.play(FadeIn(res))
        val = DecimalNumber(decoded_now(), num_decimal_places=1, unit=r"^\circ").scale(0.65)
        val.add_updater(lambda d: d.set_value(decoded_now()))
        lbl = VGroup(Tex(r"decoded:").scale(0.6), val).arrange(RIGHT, buff=0.15)
        lbl.next_to(self.pf.point(48, 95), DOWN, buff=0.35).align_to(v["border"], LEFT)
        truth_l = Tex(r"vessel: $4^\circ$").scale(0.6).next_to(lbl, DOWN, buff=0.12, aligned_edge=LEFT)
        dline = always_redraw(lambda: Line(v["probe"].get_center() - 2.0 * cdir(decoded_now()),
                                           v["probe"].get_center() + 2.0 * cdir(decoded_now()), stroke_width=3, color=WHITE))
        self.play(FadeIn(lbl), FadeIn(truth_l), Create(dline))
        self.say(r"The cell at $168.75^\circ$ sees almost the same line as the cell at $0^\circ$, yet their arrows point apart. The sum goes wrong.", wait=2.5)
        self.say(r"Doubling wraps the half-circle of orientations once around the circle: $0^\circ$ and $180^\circ$ land on the same point.")
        self.play(m.animate.set_value(2.0), TransformMatchingTex(eq1, eq2), run_time=4.0)
        self.wait(1.0)
        self.say(r"Now the near-horizontal cells all point the same way, and the sum, halved, is the vessel's angle.", wait="read")
        val.clear_updaters()
        for mob in (ring, rlab, arr, res, dline):
            mob.clear_updaters()
        self.play(*[FadeOut(mob) for mob in (arr, res, dline, lbl, truth_l, ring, rlab, v["re_ax"], v["im_ax"], v["axl"],
                                              v["img"], v["border"], v["head"], v["probe"], eq2)], run_time=0.8)

    def angle_sweep(self):
        sw = P.sweep()
        A = sw["angles"]
        pf = PixelFrame(P.SHAPE, 2.7, LEFT * 5.2 + UP * 1.85)
        border = SurroundingRectangle(Rectangle(width=pf.width, height=pf.height).move_to(pf.centre), buff=0,
                                      color=MUTED, stroke_width=1)
        a = ValueTracker(A[0])

        def idx():
            return int(np.clip(round((a.get_value() - A[0]) / (A[1] - A[0])), 0, len(A) - 1))

        img = image(sw["od"][0], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)

        def swap(mob):
            mob.pixel_array = colormap(sw["od"][idx()], "magma", -0.02, 0.3)

        img.add_updater(swap)
        c = pf.point(48, 48)
        wline = always_redraw(lambda: Line(c - 1.25 * cdir(sw["win"][idx()]), c + 1.25 * cdir(sw["win"][idx()]),
                                           stroke_width=3, color=hue(sw["win"][idx()])))
        pline = always_redraw(lambda: Line(c - 1.25 * cdir(sw["pop"][idx()]), c + 1.25 * cdir(sw["pop"][idx()]),
                                           stroke_width=2.5, color=WHITE))
        # mini complex plane under the patch
        O = LEFT * 5.2 + DOWN * 2.05
        R0 = 0.62
        SC = 1.15 / np.abs([P.pop_vector(C) for C in sw["C"]]).max()
        axes0 = VGroup(Line(O + LEFT * 0.95, O + RIGHT * 1.35, stroke_color=MUTED, stroke_width=1),
                       Line(O + DOWN * 0.9, O + UP * 1.3, stroke_color=MUTED, stroke_width=1))
        ring = self.ring(O, R0, width=4)

        def plane():
            Cn = sw["C"][idx()]
            g = VGroup(*[arrow(O, O + SC * Cn[j] * cdir(2 * P.PHI[j]), hue(P.PHI[j]), width=3.5, tip=0.08)
                         for j in range(16) if Cn[j] > 2.0])
            g.add(arrow(O, O + SC * cv(P.pop_vector(Cn)), WHITE, width=5, tip=0.13))
            return g

        pl = always_redraw(plane)

        ax = Axes(x_range=[0, 45, 15], y_range=[-8, 8, 4], x_length=7.2, y_length=4.4,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5)).move_to(RIGHT * 2.95 + UP * 0.45)
        xt = VGroup(*[MathTex(rf"{v}^\circ").scale(0.45).next_to(ax.c2p(v, -8), DOWN, buff=0.1) for v in (0, 15, 30, 45)])
        yt = VGroup(*[MathTex(rf"{v}^\circ").scale(0.42).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (-8, -4, 0, 4, 8)])
        xl = Tex(r"the vessel's angle").scale(0.52).next_to(xt, DOWN, buff=0.1)
        yl = Tex(r"error").scale(0.52).rotate(PI / 2).next_to(yt, LEFT, buff=0.12)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(45, 0), color=MUTED, stroke_width=1, dash_length=0.05)
        bands = VGroup(*[DashedLine(ax.c2p(0, y), ax.c2p(45, y), color=MUTED, stroke_width=1, stroke_opacity=0.5, dash_length=0.04)
                         for y in (-5.625, 5.625)])
        key = VGroup(Tex(r"best of 16 (in its cell's hue)").scale(0.5), Tex(r"population vector", color=WHITE).scale(0.5)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(ax, UP, buff=0.12).align_to(ax, RIGHT)
        key_sw = VGroup(*[Line(ORIGIN, RIGHT * 0.35, stroke_width=4, color=c) for c in (hue(22.5), WHITE)])
        for s, k in zip(key_sw, key):
            s.next_to(k, LEFT, buff=0.12)

        def traces():
            n = idx() + 1
            g = VGroup()
            if n < 2:
                return g
            x, ew, ep, w = A[:n], sw["err_win"][:n], sw["err_pop"][:n], sw["win"][:n]
            cuts = np.r_[0, np.where(np.diff(w) != 0)[0] + 1, n]
            for i0, i1 in zip(cuts[:-1], cuts[1:]):
                if i1 - i0 >= 2:
                    g.add(VMobject(stroke_color=hue(w[i0]), stroke_width=4).set_points_as_corners(
                        [ax.c2p(xx, yy) for xx, yy in zip(x[i0:i1], ew[i0:i1])]))
            g.add(VMobject(stroke_color=WHITE, stroke_width=4).set_points_as_corners([ax.c2p(xx, yy) for xx, yy in zip(x, ep)]))
            return g

        tr = always_redraw(traces)
        self.say(r"Now turn the vessel slowly from $0^\circ$ to $45^\circ$ and watch both readouts.")
        self.play(FadeIn(img), Create(border), FadeIn(wline), FadeIn(pline), Create(axes0), Create(ring), FadeIn(pl),
                  Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), Create(zero), FadeIn(bands),
                  FadeIn(key), FadeIn(key_sw), run_time=1.5)
        self.add(tr)
        self.play(a.animate.set_value(A[-1]), run_time=8.0, rate_func=linear)
        mx_w, mx_p = np.abs(sw["err_win"]).max(), np.abs(sw["err_pop"]).max()
        self.say(rf"The winner jumps cell to cell: a sawtooth, off by up to ${mx_w:.1f}^\circ$. The population vector glides, within ${mx_p:.1f}^\circ$ (toy).", wait=2.5)
        for mob in (img, wline, pline, pl, tr):
            mob.clear_updaters()
        self.clear()

    def subpixel_position(self):
        rp = P.ridge_profile()
        half = 7
        crop = rp["od"][32 - half:32 + half + 1, 32 - half:32 + half + 1]
        pf = PixelFrame(crop.shape, 4.3, LEFT * 4.4 + UP * 0.75)
        img = image(crop, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"a horizontal vessel, pixel by pixel").scale(0.6).next_to(border, UP, buff=0.15)
        cx = half                                   # the sampled column in crop pixels
        rows = [half - p for p in rp["pos"]]        # crop rows of the samples (screen-up offsets)
        sq = VGroup(*[Square(side_length=pf.unit, stroke_color=WHITE, stroke_width=2).move_to(pf.point(cx, r)) for r in rows])
        true_y = pf.point(cx, half - rp["truth"])[1]
        true_ln = DashedLine([img.get_left()[0], true_y, 0], [img.get_right()[0], true_y, 0], color=WHITE,
                             stroke_width=2, dash_length=0.07)
        true_l = Tex(r"vessel centre", color=WHITE).scale(0.5).next_to(true_ln, RIGHT, buff=0.1)
        self.say(r"Position has the same problem. The ridge top is a whole pixel; the vessel's centre falls between pixels.")
        self.play(FadeIn(img), Create(border), FadeIn(head))
        self.play(Create(true_ln), FadeIn(true_l))
        self.play(LaggedStart(*[Create(s) for s in sq], lag_ratio=0.1), run_time=1.2)

        Cr = np.maximum(rp["C"], 0.0)                 # rates
        ax = Axes(x_range=[0, 9, 1], y_range=[0, 80, 20], x_length=5.8, y_length=3.4,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5)).move_to(RIGHT * 3.6 + UP * 0.3)

        def u(p):
            return p + 4.5

        xt = VGroup(*[MathTex(f"{v:+d}" if v else "0").scale(0.42).next_to(ax.c2p(u(v), 0), DOWN, buff=0.1) for v in range(-4, 5)])
        yt = VGroup(*[MathTex(f"{v}").scale(0.4).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (20, 40, 60, 80)])
        xl = Tex(r"position across the vessel (px, up)").scale(0.5).next_to(xt, DOWN, buff=0.1)
        yl = Tex(r"rate $C$").scale(0.5).rotate(PI / 2).next_to(yt, LEFT, buff=0.12)
        dots = VGroup(*[Dot(ax.c2p(u(p), c), radius=0.06, color=GREY_A) for p, c in zip(rp["pos"], Cr)])
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl))
        self.play(*[TransformFromCopy(s, d) for s, d in zip(sq, dots)], run_time=1.6)
        i0 = rp["i0"]
        top3 = VGroup(dots[i0 - 1], dots[i0], dots[i0 + 1])
        self.play(top3.animate.set_color(YELLOW).scale(1.3))
        p3, c3 = rp["pos"][i0 - 1:i0 + 2], Cr[i0 - 1:i0 + 2]
        coef = np.polyfit(p3, c3, 2)
        par = ax.plot(lambda x: np.polyval(coef, x - 4.5), x_range=[u(-1.7), u(1.7)], color=YELLOW, stroke_width=3)
        vx = rp["vertex"]
        vline = DashedLine(ax.c2p(u(vx), 0), ax.c2p(u(vx), np.polyval(coef, vx)), color=YELLOW, stroke_width=2.5, dash_length=0.06)
        self.say(r"Fit a parabola through the top three samples. Its vertex lands between the pixels.")
        self.play(Create(par), run_time=1.3)
        self.play(Create(vline))
        eqp = MathTex(r"\delta", "=", r"\frac{1}{2}\,", r"\frac{C_{+1} - C_{-1}}{2C_0 - C_{+1} - C_{-1}}").scale(0.75)
        eqp.next_to(ax, UP, buff=0.25).align_to(ax, RIGHT)
        eqp[0].set_color(YELLOW)
        self.play(Write(eqp))
        rd = VGroup(Tex(r"ridge top: $0$ px").scale(0.55),
                    Tex(rf"vertex: ${vx:+.2f}$ px", color=YELLOW).scale(0.55),
                    Tex(rf"vessel centre: ${rp['truth']:+.2f}$ px").scale(0.55)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rd.next_to(border, DOWN, buff=0.15).align_to(border, LEFT)
        self.play(FadeIn(rd, lag_ratio=0.3))
        self.say(r"Three neighbouring samples give the centre to a few hundredths of a pixel (toy, noise 0.01 OD).", wait="read")
        if not self.at(3):
            self.say(r"Today only the winners are used. Spec 4.12 reads every parameter from its whole population, to start the spline fit.", wait="read")
        self.spos = Group(img, border, head, sq, true_ln, true_l, ax, xt, yt, xl, yl, dots, par, vline, eqp, rd)

    # ================================================================== level 3
    def other_parameters(self):
        self.clear()
        od = P.fit_scene()
        fr = P.fit_runs()
        nd = 4
        m = fr["population"]["measures"][nd]
        d = P.decode_rsa(m["prof"], m["x"])
        mw = fr["winner"]["measures"][nd]
        dw = P.decode_rsa(mw["prof"], mw["x"])
        r_t, s_t, a_t = P.TRUE_RSA

        pf = PixelFrame(P.FIT_SHAPE, 2.3, LEFT * 4.55 + UP * 2.45)
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        q = m["pos"]
        nrm = m["normal"]
        cut = Line(pf.point(*(q - 12 * nrm)), pf.point(*(q + 12 * nrm)), color=YELLOW, stroke_width=2.5)
        dot = Dot(pf.point(*q), radius=0.05, color=YELLOW)
        self.say(r"Width, blur and contrast are population codes too. Take one point of a curved toy vessel, and the profile across it.")
        self.play(FadeIn(img), Create(border))
        self.play(FadeIn(dot), Create(cut))

        # the profile across (OD), with the walls and the centre
        pax = Axes(x_range=[-12, 12, 4], y_range=[-0.02, 0.16, 0.04], x_length=4.3, y_length=2.2,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(LEFT * 4.45 + DOWN * 0.95)
        pxl = Tex(r"across the vessel (px)").scale(0.45).next_to(pax, DOWN, buff=0.1)
        pyl = MathTex(r"\mathrm{OD}", color=OD_COL).scale(0.5).next_to(pax.c2p(0, 0.15), LEFT, buff=0.12)
        sel = np.abs(m["x"]) <= 12
        prof = pax.plot_line_graph(m["x"][sel][::4], m["prof"][sel][::4], line_color=OD_COL, add_vertex_dots=False, stroke_width=2.5)
        self.play(TransformFromCopy(cut, prof), Create(pax), FadeIn(pxl), FadeIn(pyl), run_time=1.4)

        # ---- the size channels: 9 scales, a parabola in log scale
        sp = d["sp"]
        ls = np.log(P.SCALES) - np.log(P.SCALES[0]) + 0.15     # log scale, shifted so the y axis sits at the left
        sax = Axes(x_range=[0, ls[-1] + 0.15, 1], y_range=[0, 1.1, 0.5], x_length=3.9, y_length=2.4,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(RIGHT * 0.35 + UP * 1.55)
        rn = sp["resp"] / sp["resp"].max()
        sbars = VGroup(*[Line(sax.c2p(l, 0), sax.c2p(l, v), stroke_width=13, color=SIZE_COL, cap_style=CapStyleType.BUTT)
                         for l, v in zip(ls, rn)])
        sxt = VGroup(*[MathTex(f"{P.SCALES[i]:.1f}").scale(0.38).next_to(sax.c2p(ls[i], 0), DOWN, buff=0.08) for i in (0, 2, 4, 6, 8)])
        sxl = Tex(r"size channel $\sigma$ (px, log)").scale(0.45).next_to(sxt, DOWN, buff=0.08)
        stt = Tex(r"9 size channels at the centre", color=SIZE_COL).scale(0.5).next_to(sax, UP, buff=0.12)
        self.say(r"Nine size channels fire at the centre. The best one gives the width in coarse steps; the population interpolates.")
        self.play(Create(sax), FadeIn(sxt), FadeIn(sxl), FadeIn(stt))
        self.play(LaggedStart(*[Create(b_) for b_ in sbars], lag_ratio=0.1), run_time=1.4)
        i = sp["top"]
        coef = np.polyfit(ls[i - 1:i + 2], rn[i - 1:i + 2], 2)
        spar = sax.plot(lambda x: np.polyval(coef, x), x_range=[ls[i - 1] - 0.12, ls[i + 1] + 0.12], color=YELLOW, stroke_width=3)
        lsx = sp["log_sigma"] - np.log(P.SCALES[0]) + 0.15
        sv = DashedLine(sax.c2p(lsx, 0), sax.c2p(lsx, np.polyval(coef, lsx)), color=YELLOW,
                        stroke_width=2.5, dash_length=0.05)
        sv_l = MathTex(rf"\sigma^* = {np.exp(sp['log_sigma']):.2f}\,\text{{px}}", color=YELLOW).scale(0.45).next_to(sax.c2p(ls[0], 1.0), RIGHT, buff=0.15)
        self.play(sbars[i - 1:i + 2].animate.set_color(YELLOW), Create(spar))
        self.play(Create(sv), FadeIn(sv_l))
        w_all = np.maximum(sp["resp"], 0.0)
        r_all = float(np.exp((w_all * np.log(sp["pref"])).sum() / w_all.sum()))
        self.say(rf"The spec's weighted mean of log scale over all nine gives {r_all:.1f} px here (truth {P.TRUE_RSA[0]}): the broad tuning pulls it to the middle. The toy reads the peak instead.", wait="read")

        # ---- blur: fine edge cells at the walls against the line cell at the centre; the lookup table
        xw = m["x"][sel]
        walls = [xw[np.argmax(np.gradient(m["prof"][sel]))], xw[np.argmin(np.gradient(m["prof"][sel]))]]
        wm = VGroup(*[Dot(pax.c2p(w, float(np.interp(w, m["x"], m["prof"]))), radius=0.06, color=ORANGE) for w in walls])
        cm = Dot(pax.c2p(0, float(np.interp(0, m["x"], m["prof"]))), radius=0.07, color=SIZE_COL)
        ratio_t = MathTex(r"\frac{\text{edge}}{\text{line}}", "=", f"{d['ratio']:.2f}").scale(0.55)
        ratio_t[0].set_color(ORANGE)
        ratio_t.next_to(pax, RIGHT, buff=0.25).shift(UP * 0.3)
        ss, tbl = P.blur_curve(d["r"])
        lax = Axes(x_range=[0.4, 2.6, 0.5], y_range=[0.3, 0.75, 0.15], x_length=3.4, y_length=2.4,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(RIGHT * 4.95 + UP * 1.55)
        lxt = VGroup(*[MathTex(f"{v:.1f}").scale(0.38).next_to(lax.c2p(v, 0.3), DOWN, buff=0.08) for v in (0.5, 1.0, 1.5, 2.0, 2.5)])
        lyt = VGroup(*[MathTex(f"{v:.2f}").scale(0.36).next_to(lax.c2p(0.4, v), LEFT, buff=0.06) for v in (0.45, 0.6, 0.75)])
        lxl = Tex(r"blur $s$ (px)").scale(0.45).next_to(lxt, DOWN, buff=0.08)
        ltt = Tex(rf"lookup: blurred cylinder, $r = {d['r']:.2f}$").scale(0.45).next_to(lax, UP, buff=0.12)
        ok = (tbl > 0.3) & (tbl < 0.75)
        lcur = lax.plot_line_graph(ss[ok], tbl[ok], line_color=WHITE, add_vertex_dots=False, stroke_width=2.5)
        hl = DashedLine(lax.c2p(0.4, d["ratio"]), lax.c2p(d["s"], d["ratio"]), color=ORANGE, stroke_width=2, dash_length=0.05)
        vl = DashedLine(lax.c2p(d["s"], d["ratio"]), lax.c2p(d["s"], 0.3), color=ORANGE, stroke_width=2, dash_length=0.05)
        self.say(r"Blur: fine edge cells fire at the walls, the line cell at the centre. Their ratio, through a table of the blurred-cylinder model, gives $s$ (and refines $r$).")
        self.play(FadeIn(wm, scale=1.5), FadeIn(cm, scale=1.5))
        self.play(FadeIn(ratio_t))
        self.play(Create(lax), FadeIn(lxt), FadeIn(lyt), FadeIn(lxl), FadeIn(ltt), Create(lcur), run_time=1.2)
        self.play(TransformFromCopy(ratio_t[2], hl), run_time=1.0)
        self.play(Create(vl))
        self.wait(1.0)

        # ---- contrast: an unnormalised pathway
        aeq = MathTex(r"a", "=", r"\frac{\mathrm{OD}_\text{centre}}{g(r, s)}", "=",
                      rf"\frac{{{d['od_c']:.3f}}}{{{d['g']:.2f}}}", "=", f"{d['a']:.3f}").scale(0.6)
        aeq[2].set_color(OD_COL)
        aeq.move_to(RIGHT * 2.6 + DOWN * 1.25)
        self.say(r"Contrast needs an unnormalised pathway: the noise-normalised layers discard absolute contrast on purpose. The OD at the centre, corrected for $r$ and $s$.")
        self.play(Write(aeq), run_time=1.5)
        self.wait(1.0)
        res = VGroup(
            Tex(r"this point (toy):").scale(0.5),
            MathTex(rf"r = {d['r']:.2f}\ ({r_t})", rf"\quad s = {d['s']:.2f}\ ({s_t})", rf"\quad a = {d['a']:.3f}\ ({a_t})").scale(0.5),
            Tex(rf"winners only: $r = {dw['sp']['r_win']:.2f}$, $s$ unknown, $a = {dw['od_c']:.3f}$ (raw OD)", color=GREY_B).scale(0.45),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(RIGHT * 2.6 + DOWN * 2.45)
        trn = Tex(r"(truth in brackets)", color=GREY_B).scale(0.4).next_to(res[0], RIGHT, buff=0.2)
        self.play(FadeIn(res, lag_ratio=0.2), FadeIn(trn))
        self.say(r"And curvature: the arc cells' winning $\kappa$ (chapter 6). Together, a start value for every spline parameter.", wait="read")

    def start_the_fit(self):
        self.clear()
        od = P.fit_scene()
        fr = P.fit_runs()
        runs = [("winner", hue(33.75), r"start from winners"), ("population", WHITE, r"start from populations")]
        pf = PixelFrame(P.FIT_SHAPE, 3.3, LEFT * 3.45 + UP * 1.75)
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"the spline network on the toy vessel's OD").scale(0.55).next_to(border, UP, buff=0.12)
        self.say(r"Use: these are start values for the spline network, instead of step 8's box fits. The fit then measures the OD itself.")
        self.play(FadeIn(img), Create(border), FadeIn(head))

        it = ValueTracker(0.0)

        def state(mode):
            h = fr[mode]["hist"]
            t = it.get_value()
            i = int(np.clip(np.floor(t), 0, len(h) - 2))
            f = float(np.clip(t - i, 0, 1))
            nodes = h[i][0] * (1 - f) + h[i + 1][0] * f
            rsa = np.array(h[i][1:]) * (1 - f) + np.array(h[i + 1][1:]) * f
            return nodes, rsa

        # zoom on one node
        nd = 4
        zc = fr["population"]["hist"][-1][0][nd]
        half = 4
        zf = PixelFrame((2 * half + 1, 2 * half + 1), 2.5, LEFT * 5.0 + DOWN * 1.55)
        x0, y0 = int(round(zc[0])) - half, int(round(zc[1])) - half
        zimg = image(od[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1], zf.height, "magma", vmin=-0.02, vmax=0.3).move_to(zf.centre)
        zb = SurroundingRectangle(zimg, buff=0, color=YELLOW, stroke_width=1.5)
        zsrc = Rectangle(width=(2 * half + 1) * pf.unit, height=(2 * half + 1) * pf.unit, color=YELLOW, stroke_width=1.5
                         ).move_to(pf.point(x0 + half, y0 + half))

        def to_zoom(p):
            return zf.point(p[0] - x0, p[1] - y0)

        def draw(mode, col, frame, conv, width):
            def f():
                nodes, (r, s, a) = state(mode)
                pts = P.node_curve(nodes, n=160, ext=0.0)
                g = VGroup()
                if frame is pf:
                    g.add(VMobject(stroke_color=col, stroke_width=width).set_points_as_corners([pf.point(*p) for p in pts]))
                    g.add(*[Dot(pf.point(*p), radius=0.04, color=col) for p in nodes])
                else:
                    near = pts[np.hypot(pts[:, 0] - zc[0], pts[:, 1] - zc[1]) < half + 0.4]
                    if len(near) > 2:
                        tng = np.gradient(near, axis=0)
                        tng /= np.linalg.norm(tng, axis=1)[:, None]
                        nrm = np.c_[-tng[:, 1], tng[:, 0]]
                        g.add(VMobject(stroke_color=col, stroke_width=width).set_points_as_corners([to_zoom(p) for p in near]))
                        for sgn in (-1, 1):
                            g.add(DashedVMobject(VMobject(stroke_color=col, stroke_width=1.8).set_points_as_corners(
                                [to_zoom(p) for p in near + sgn * r * nrm]), num_dashes=14))
                    g.add(Dot(to_zoom(nodes[nd]), radius=0.07, color=col))
                return g
            return always_redraw(f)

        live = [draw(mode, col, pf, None, 2.2) for mode, col, _ in runs]
        zlive = [draw(mode, col, zf, None, 3) for mode, col, _ in runs]
        zt = Tex(r"one node, zoomed", color=YELLOW).scale(0.45).next_to(zb, UP, buff=0.08)

        # convergence plot
        n = len(fr["population"]["geo"]) - 1
        cax = Axes(x_range=[0, n, 2], y_range=[0, 0.3, 0.1], x_length=4.6, y_length=2.6,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(RIGHT * 3.75 + UP * 1.75)
        cxt = VGroup(*[MathTex(f"{v}").scale(0.4).next_to(cax.c2p(v, 0), DOWN, buff=0.08) for v in range(0, n + 1, 2)])
        cyt = VGroup(*[MathTex(f"{v:.1f}").scale(0.4).next_to(cax.c2p(0, v), LEFT, buff=0.08) for v in (0.1, 0.2, 0.3)])
        cxl = Tex(r"fit iteration").scale(0.45).next_to(cxt, DOWN, buff=0.08)
        cyl = Tex(r"centreline error (px)").scale(0.45).rotate(PI / 2).next_to(cyt, LEFT, buff=0.1)

        def conv_curve(mode, col):
            def f():
                g = np.array(fr[mode]["geo"])
                t = it.get_value()
                k = int(np.floor(t))
                xs = list(range(0, k + 1))
                ys = list(g[:k + 1])
                if t > k and k + 1 < len(g):
                    xs.append(t)
                    ys.append(g[k] + (g[k + 1] - g[k]) * (t - k))
                vm = VMobject(stroke_color=col, stroke_width=3.5)
                pts = [cax.c2p(x, min(y, 0.3)) for x, y in zip(xs, ys)]
                if len(pts) == 1:
                    pts = pts * 2
                vm.set_points_as_corners(pts)
                return VGroup(vm, Dot(pts[-1], radius=0.05, color=col))
            return always_redraw(f)

        key = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.35, color=col, stroke_width=4), Tex(lbl, color=col).scale(0.45)
                              ).arrange(RIGHT, buff=0.12) for _, col, lbl in runs]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        key.next_to(cax, UP, buff=0.1).align_to(cax, RIGHT)
        self.play(*[FadeIn(l) for l in live], FadeIn(zimg), Create(zb), Create(zsrc), FadeIn(zt), *[FadeIn(z) for z in zlive],
                  Create(cax), FadeIn(cxt), FadeIn(cyt), FadeIn(cxl), FadeIn(cyl), FadeIn(key), run_time=1.5)
        cl = [conv_curve(mode, col) for mode, col, _ in runs]
        self.add(*cl)

        # the table of start and fitted values
        def row(*cells, col=WHITE, scale=0.45):
            return [Tex(c, color=col).scale(scale) for c in cells]

        gw, gp = fr["winner"]["geo"], fr["population"]["geo"]
        rw, rp_ = fr["winner"]["rsa"], fr["population"]["rsa"]
        r_t, s_t, a_t = P.TRUE_RSA
        rows = [row("", "winners", "populations", "after the fit", "truth", col=GREY_B),
                row("centreline (px)", f"{gw[0]:.2f}", f"{gp[0]:.2f}", f"{gw[-1]:.2f} / {gp[-1]:.2f}", "0"),
                row("half width $r$", f"{rw[0][0]:.2f}", f"{rp_[0][0]:.2f}", f"{rw[-1][0]:.2f} / {rp_[-1][0]:.2f}", f"{r_t}"),
                row("blur $s$", f"{rw[0][1]:.2f} (guess)", f"{rp_[0][1]:.2f}", f"{rw[-1][1]:.2f} / {rp_[-1][1]:.2f}", f"{s_t}"),
                row("contrast $a$", f"{rw[0][2]:.3f}", f"{rp_[0][2]:.3f}", f"{rw[-1][2]:.3f} / {rp_[-1][2]:.3f}", f"{a_t}")]
        xs = [-1.55, 0.55, 2.15, 4.0, 5.75]
        tbl = VGroup()
        for k, rr in enumerate(rows):
            for j, c in enumerate(rr):
                c.move_to([xs[j], -0.75 - 0.36 * k, 0], aligned_edge=LEFT if j == 0 else ORIGIN)
                tbl.add(c)
        tbl_h = Tex(r"toy fit: Levenberg-Marquardt on the OD, from two starts", color=GREY_B).scale(0.45).move_to([2.2, -0.35, 0])
        hdr_cols = [hue(33.75), WHITE]
        rows[0][1].set_color(hdr_cols[0])
        rows[0][2].set_color(hdr_cols[1])
        self.play(FadeIn(tbl_h), *[FadeIn(c) for c in rows[0]], *[FadeIn(r_[0]) for r_ in rows[1:]],
                  *[FadeIn(r_[1]) for r_ in rows[1:]], *[FadeIn(r_[2]) for r_ in rows[1:]], *[FadeIn(r_[4]) for r_ in rows[1:]])
        self.say(r"Two starts for the same fit. The population start begins closer, in every parameter.", wait=1.5)
        self.say(r"Run the fit: the nodes snap onto the ridge and the width settles.")
        self.play(it.animate.set_value(float(n)), run_time=6.0, rate_func=smooth)
        self.play(*[FadeIn(r_[3]) for r_ in rows[1:]])
        self.say(r"Both end in the same place: the fit stays the final estimator. The gate (spec 6.2, phase 7): initial errors down, fitted geometry not down.", wait="read")
        self.say(r"Status: designed, not started (phase 7). The numbers above are the toy's.", wait=1.5)
        for mob in (*live, *zlive, *cl):
            mob.clear_updaters()


variants(PopulationCodes, module_globals=globals())
