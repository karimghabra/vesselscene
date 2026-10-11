"""Chapter 15: top-down feedback (spec 4.11, 2.3).

The graph predicts the image; what it does not explain (the residual) is fed
back as a gain on existing evidence, only near junction cells and free ends.
On a toy fork whose third arm is just under the seed threshold, the local gain
lifts the arm's root over threshold and tracing carries it out; the earlier
whole-image re-proposal also turns texture into false pieces and junctions.

Render:  manim -ql scenes/s15_feedback.py FeedbackDetailed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import feedback as F  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import CONT, FB, INH, JUNC, MUTED, OD_COL, PixelFrame, image  # noqa: E402
from mica_anim.vsdata import identity_color  # noqa: E402

PW = 5.4                                   # panel width (scene units)
LEFT_C = LEFT * 3.45 + UP * 1.8
RIGHT_C = RIGHT * 3.45 + UP * 1.8


def overlay(mask, color, height, alpha=0.95):
    im = ImageMobject(F.mask_rgba(mask, color, alpha))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


class Feedback(ChapterScene):
    NUMBER, TITLE, SPEC = 15, "Top-down feedback", "spec 4.11, 2.3"
    MAP_KEYS = ("feedback", "render", "assoc")

    def construct(self):
        self.intro()
        sc = F.scene()
        runs = F.runs()
        h = PW * F.SHAPE[0] / F.SHAPE[1]
        pl = PixelFrame(F.SHAPE, h, LEFT_C)
        pr = PixelFrame(F.SHAPE, h, RIGHT_C)
        self.pl, self.pr = pl, pr

        # ------------------------------------------------------------ the puzzle
        od_l = image(sc["od"], h, "magma", vmin=-0.02, vmax=0.3).move_to(pl.centre)
        bl = SurroundingRectangle(od_l, buff=0, color=MUTED, stroke_width=1)
        hl = Tex(r"the OD (toy fork)").scale(0.55).next_to(bl, UP, buff=0.12)
        trace_p = Line(pl.point(*(F.PARENT[0] + [22, -2.4])), pl.point(*F.J), color=identity_color(1), stroke_width=3)
        trace_s = Line(pl.point(*F.J), pl.point(*(F.J + (F.STRONG[1] - F.J) * 0.83)), color=identity_color(2), stroke_width=3)
        jring = Circle(radius=F.R_J * pl.unit, color=JUNC, stroke_width=3).move_to(pl.point(*F.J))
        jring_d = DashedVMobject(jring, num_dashes=16)
        self.play(FadeIn(od_l), Create(bl), FadeIn(hl))
        self.say(r"A fork. The proposer traced two of its arms. The third is faint: just under the threshold for a new trace.")
        self.play(Create(trace_p), Create(trace_s), run_time=1.4)
        self.play(Create(jring_d))
        self.say(r"A junction cell fires at the fork, weakly. Can the graph help find the missing arm, without inventing vessels?", wait="read")
        graph = VGroup(trace_p, trace_s)

        # ------------------------------------------------------------ predict, subtract
        R_r = image(sc["R"], h, "magma", vmin=-0.02, vmax=0.3).move_to(pr.centre)
        br = SurroundingRectangle(R_r, buff=0, color=MUTED, stroke_width=1)
        hr = Tex(r"the graph's prediction: its render $R$").scale(0.55).next_to(br, UP, buff=0.12)
        self.say(r"Predict: render the graph. This is what the higher level expects the image to be.")
        gcopy = graph.copy()
        self.play(gcopy.animate.shift(pr.centre - pl.centre), Create(br), FadeIn(hr), run_time=1.2)
        self.play(FadeIn(R_r), FadeOut(gcopy), run_time=0.8)
        self.wait(1.0)
        from scipy.ndimage import gaussian_filter
        errb = gaussian_filter(F.residual(), 1.0)
        err_r = image(errb, h, "magma", vmin=0.0, vmax=4.0).move_to(pr.centre)
        he = Tex(r"what the graph does not explain").scale(0.55).next_to(br, UP, buff=0.12)
        eq_err = MathTex(r"\mathrm{err}", "=", r"\max(", r"\mathrm{OD}", "-", "R", r",\,0)", "/", r"\text{noise}").scale(0.75)
        eq_err[0].set_color(FB)
        eq_err[3].set_color(OD_COL)
        eq_err.next_to(br, DOWN, buff=0.3)
        self.say(r"Subtract, and keep the positive part in noise units: the residual shows the missing arm.")
        self.play(FadeOut(R_r), FadeIn(err_r), ReplacementTransform(hr, he), run_time=1.4)
        if self.at(2):
            self.play(Write(eq_err))
        self.wait(1.0)
        self.say(r"But also everything else the graph leaves out: dark texture, and noise.", wait="read")

        # ------------------------------------------------------------ the gain, near the junction only
        near = Circle(radius=2 * F.R_J * pr.unit, color=FB, stroke_width=2.5, fill_color=FB, fill_opacity=0.12).move_to(pr.point(*F.J))
        near_l = Tex(r"near: within $2R_j$ of the junction cell", color=FB).scale(0.5).next_to(near, UP, buff=0.05).shift(LEFT * 0.4)
        eq_g = MathTex(r"U_0'", "=", r"U_0", r"\,(1 +", r"\gamma_\text{fb}", r"\,\mathrm{blur}(", r"\mathrm{err}", r",1)",
                       r"\cdot", r"\text{near}", ")").scale(0.75)
        eq_g[6].set_color(FB)
        eq_g[9].set_color(FB)
        eq_g[0].set_color(FB)
        self.say(r"Feed it back as a \emph{gain} on the evidence already there, and only near the junction cell.")
        self.play(GrowFromCenter(near), FadeIn(near_l))
        near_lft = near.copy().move_to(pl.point(*F.J))
        self.play(TransformFromCopy(near, near_lft), run_time=1.0)
        if self.at(2):
            eq_g.next_to(eq_err, DOWN, buff=0.25)
            self.play(Write(eq_g), run_time=1.6)
            self.say(r"The gain multiplies the evidence $U_0$ that is already there. Where nothing is predicted missing, near $= 0$ and nothing changes.", wait="read")
            self.say(r"$\gamma_\text{fb} = 0.5$: a residual of 2 noise units doubles the evidence, inside the disc only.", wait="read")
        self.wait(0.6)

        # ------------------------------------------------------------ the arm's evidence: below, then above threshold
        bp = F.branch_profile()
        ax = Axes(x_range=[0, 110, 20], y_range=[0, 25, 8], x_length=5.0, y_length=1.9,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(LEFT * 3.25 + DOWN * 1.2)
        xtk = VGroup(*[MathTex(f"{v}").scale(0.38).next_to(ax.c2p(v, 0), DOWN, buff=0.06) for v in (0, 20, 40, 60, 80, 100)])
        xl = Tex(r"along the faint arm, from the fork (px)").scale(0.45).next_to(xtk, DOWN, buff=0.06)
        yl = Tex(r"line evidence").scale(0.45).rotate(PI / 2).next_to(ax, LEFT, buff=0.1)
        th = DashedLine(ax.c2p(0, F.T_HIGH), ax.c2p(110, F.T_HIGH), color=WHITE, stroke_width=2, dash_length=0.06)
        th_l = Tex(r"seed: 16", color=WHITE).scale(0.42).next_to(ax.c2p(110, F.T_HIGH), RIGHT, buff=0.06)
        tl = DashedLine(ax.c2p(0, F.T_LOW), ax.c2p(110, F.T_LOW), color=GREY_B, stroke_width=1.5, dash_length=0.05)
        tl_l = Tex(r"grow: 8", color=GREY_B).scale(0.42).next_to(ax.c2p(110, F.T_LOW), RIGHT, buff=0.06)
        zone = Rectangle(width=ax.c2p(2 * F.R_J, 0)[0] - ax.c2p(0, 0)[0], height=1.9, stroke_width=0, fill_color=FB,
                         fill_opacity=0.12).move_to(ax.c2p(F.R_J, 12.5))
        ok = bp["d"] >= 0
        c0 = ax.plot_line_graph(bp["d"][ok], np.minimum(bp["none"][ok], 25), line_color=GREY_A, add_vertex_dots=False, stroke_width=2.5)
        c1 = ax.plot_line_graph(bp["d"][ok], np.minimum(bp["local"][ok], 25), line_color=FB, add_vertex_dots=False, stroke_width=3)
        if self.at(2):
            self.play(FadeOut(eq_err), FadeOut(eq_g))
        self.say(r"Along the faint arm, the evidence never reaches the seed threshold, so no trace starts there.")
        self.play(Create(ax), FadeIn(xtk), FadeIn(xl), FadeIn(yl), Create(th), FadeIn(th_l), Create(tl), FadeIn(tl_l))
        self.play(Create(c0), run_time=1.5)
        self.wait(0.6)
        self.say(r"With the gain, its root, inside the disc, crosses the threshold: a seed. Tracing then carries it outward.")
        self.play(FadeIn(zone), TransformFromCopy(c0, c1), run_time=1.6)

        loc = runs["local"]
        found = overlay(loc["mask"] & ~runs["known"], FB, h).move_to(pl.centre)
        self.play(FadeIn(found), run_time=1.6)
        jring_s = Circle(radius=F.R_J * pl.unit, color=JUNC, stroke_width=4).move_to(pl.point(*F.J))
        typ = Tex(r"3-way", color=JUNC).scale(0.55).next_to(near_lft, UP, buff=0.06)
        self.play(ReplacementTransform(jring_d, jring_s), FadeIn(typ))
        n_true = sum(p["true"] for p in loc["pieces"])
        n_false = sum(not p["true"] for p in loc["pieces"])
        self.say(rf"The arm is found and the fork becomes a 3-way junction. Nothing else was added (toy: {n_true} arm, {n_false} false pieces).", wait="read")

        if not self.at(2):
            self.outro()
            return

        # ------------------------------------------------------------ level 2: the earlier attempt, globally
        self.play(*[FadeOut(m) for m in (ax, xtk, xl, yl, th, th_l, tl, tl_l, zone, c0, c1)], run_time=0.7)
        glob = runs["global"]
        od_r = image(sc["od"], h, "magma", vmin=-0.02, vmax=0.3).move_to(pr.centre)
        hg = Tex(r"gain and seeds everywhere (the earlier re-proposal)").scale(0.55).next_to(br, UP, buff=0.12)
        hl2 = Tex(r"gain near the junction only (spec 4.11)").scale(0.55).move_to(hl)
        self.say(r"Compare the earlier attempt: re-propose from the residual across the whole image.")
        self.play(FadeOut(err_r), FadeOut(near), FadeOut(near_l), FadeIn(od_r), ReplacementTransform(he, hg),
                  ReplacementTransform(hl, hl2), run_time=1.2)
        graph_r = graph.copy().shift(pr.centre - pl.centre)
        self.play(FadeIn(graph_r))
        true_m = np.zeros(F.SHAPE, bool)
        false_m = np.zeros(F.SHAPE, bool)
        for p in glob["pieces"]:
            if p["true"]:
                true_m |= p["mask"]
            else:
                false_m |= p["mask"]
        g_true = overlay(true_m & ~runs["known"], FB, h).move_to(pr.centre)
        g_false = overlay(false_m, INH, h).move_to(pr.centre)
        self.play(FadeIn(g_true), run_time=1.0)
        self.play(FadeIn(g_false), run_time=1.4)
        fj = VGroup()
        for p in glob["pieces"]:
            if p["touches"] and not p["true"]:
                fj.add(Cross(scale_factor=0.12, stroke_color=INH, stroke_width=4).move_to(pr.point(*p["centre"])))
        n_false_g = sum(not p["true"] for p in glob["pieces"])
        n_fj = len(fj)
        self.play(LaggedStart(*[GrowFromCenter(x) for x in fj], lag_ratio=0.3))
        self.say(rf"It finds the arm too, and turns texture into {n_false_g} false pieces, {n_fj} of them touching a vessel: false junctions (toy).", wait="read")
        cmp = VGroup(
            Tex(r"near only: ", rf"{sum(p['true'] for p in loc['pieces'])} arm, {n_false} false", color=FB).scale(0.55),
            Tex(r"everywhere: ", rf"{sum(p['true'] for p in glob['pieces'])} arm, {n_false_g} false, {n_fj} false junctions", color=INH).scale(0.55),
        )
        cmp[0].next_to(bl, DOWN, buff=0.25)
        cmp[1].next_to(br, DOWN, buff=0.25)
        self.play(FadeIn(cmp, lag_ratio=0.3))
        self.say(r"So feedback is a gain on existing evidence, not a source of new seeds, and only where the graph predicts something is missing.", wait="read")

        if not self.at(3):
            self.outro()
            return

        # ------------------------------------------------------------ level 3: the spec's numbers, free ends, two loops
        meas = VGroup(Tex(r"measured (spec 2.3, 4.11): the whole-image re-proposal", color=GREY_A).scale(0.55),
                      Tex(r"added junctions faster than true lines and lowered every score:", color=GREY_A).scale(0.55),
                      Tex(r"junction precision fell from 0.82.", color=GREY_A).scale(0.55)).arrange(DOWN, buff=0.1)
        meas.next_to(cmp, DOWN, buff=0.35).set_x(0)
        self.say(r"The toy only illustrates it. On real scenes the spec measured the same failure.")
        self.play(FadeIn(meas, lag_ratio=0.3))
        self.wait(3.0)
        self.clear()

        # near also covers free ends: a cone ahead of a trace that stops
        self.say(r"Where does the graph predict something missing? Near a junction cell, within $2R_j$; and ahead of a trace's free end, within $3w$.")
        nearfull = MathTex(r"\text{near}", "=", r"\big\{\, |p - j| \le 2R_j \,\big\}", r"\;\cup\;",
                           r"\big\{\, p \text{ within } 3w \text{ ahead of a free end} \,\big\}").scale(0.7).to_edge(UP, buff=0.5)
        nearfull[0].set_color(FB)
        eq_g2 = MathTex(r"U_0'", "=", r"U_0", r"\,(1 +", r"\gamma_\text{fb}", r"\,\mathrm{blur}(", r"\mathrm{err}", r",1)",
                        r"\cdot", r"\text{near}", ")", r",\quad \gamma_\text{fb} = 0.5").scale(0.7).next_to(nearfull, DOWN, buff=0.3)
        eq_g2[0].set_color(FB)
        eq_g2[6].set_color(FB)
        eq_g2[9].set_color(FB)
        self.play(Write(nearfull))
        self.play(Write(eq_g2))
        # small sketch: a junction disc, and a free end with its cone
        o1, o2 = LEFT * 3.2 + DOWN * 0.6, RIGHT * 2.6 + DOWN * 0.6
        arms = VGroup(Line(o1 + LEFT * 1.8 + DOWN * 0.3, o1, color=identity_color(1), stroke_width=4),
                      Line(o1, o1 + RIGHT * 1.6 + UP * 0.8, color=identity_color(2), stroke_width=4))
        disc = Circle(radius=0.75, color=FB, fill_color=FB, fill_opacity=0.15, stroke_width=2).move_to(o1)
        jr = Circle(radius=0.37, color=JUNC, stroke_width=3).move_to(o1)
        tr = Line(o2 + LEFT * 2.2, o2, color=identity_color(3), stroke_width=4)
        endd = Dot(o2, radius=0.07, color=ORANGE)
        cone = Polygon(o2 + UP * 0.25, o2 + RIGHT * 1.6 + UP * 0.25, o2 + RIGHT * 1.6 + DOWN * 0.25, o2 + DOWN * 0.25,
                       color=FB, fill_color=FB, fill_opacity=0.15, stroke_width=2)
        lbl1 = Tex(r"$2R_j$ around a junction cell", color=FB).scale(0.5).next_to(disc, DOWN, buff=0.15)
        lbl2 = Tex(r"$3w$ ahead of a free end", color=FB).scale(0.5).next_to(cone, DOWN, buff=0.15)
        self.play(Create(arms), Create(jr), Create(tr), FadeIn(endd))
        self.play(GrowFromCenter(disc), FadeIn(cone), FadeIn(lbl1), FadeIn(lbl2))
        self.wait(3.0)
        self.clear()

        # the biology: predictions down, errors up
        hi = RoundedRectangle(width=4.2, height=0.9, corner_radius=0.15, stroke_color=WHITE, stroke_width=1.6,
                              fill_color=WHITE, fill_opacity=0.05).move_to(UP * 2.0 + LEFT * 2.8)
        lo = hi.copy().move_to(DOWN * 0.9 + LEFT * 2.8)
        hi_t = Tex(r"higher level: the graph").scale(0.55).move_to(hi)
        lo_t = Tex(r"V1: the line evidence $U_0$").scale(0.55).move_to(lo)
        down = Arrow(hi.get_bottom() + LEFT * 0.8, lo.get_top() + LEFT * 0.8, buff=0.08, color=FB, stroke_width=5)
        up = Arrow(lo.get_top() + RIGHT * 0.8, hi.get_bottom() + RIGHT * 0.8, buff=0.08, color=WHITE, stroke_width=5)
        down_l = Tex(r"prediction $R$", color=FB).scale(0.5).next_to(down, LEFT, buff=0.12)
        up_l = Tex(r"error $\mathrm{OD} - R$").scale(0.5).next_to(up, RIGHT, buff=0.12)
        def ref(title, line, cite):
            return VGroup(Tex(title, color=FB).scale(0.55), Tex(line, color=GREY_A).scale(0.48),
                          Tex(cite, color=GREY_B).scale(0.44)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        refs = VGroup(ref(r"\textbf{predictive coding}", r"the error drives the lower area", r"(Rao and Ballard 1999)"),
                      ref(r"\textbf{biased competition}", r"feedback favours what is expected", r"(Desimone and Duncan 1995)"),
                      ref(r"\textbf{the recurrent phase of V1}", r"carries this feedback", r"(Lamme and Roelfsema 2000)"),
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.4 + UP * 0.7)
        self.say(r"This is the brain's arrangement: higher areas send predictions down, and what they fail to predict comes back up.")
        self.play(FadeIn(hi), FadeIn(hi_t), FadeIn(lo), FadeIn(lo_t))
        self.play(GrowArrow(down), FadeIn(down_l))
        self.play(GrowArrow(up), FadeIn(up_l))
        self.play(FadeIn(refs, lag_ratio=0.15), run_time=2.0)
        self.wait(3.0)
        self.clear()

        # the two loops
        self.say(r"In the pipeline, two loops, each a fixed number of passes so the result stays deterministic.")

        def box(txt, pos, w=2.7):
            r = RoundedRectangle(width=w, height=0.7, corner_radius=0.12, stroke_color=WHITE, stroke_width=1.6,
                                 fill_color=WHITE, fill_opacity=0.05)
            return VGroup(r, Tex(txt).scale(0.5)).move_to(pos)
        b_prop = box(r"proposer: 4.7--4.10", LEFT * 3.6 + UP * 1.7, 3.0)
        b_rend = box(r"step 8: render, prune", RIGHT * 2.6 + UP * 1.7, 3.0)
        b_fit = box(r"spline fit (9--12)", RIGHT * 2.6 + DOWN * 0.9, 3.0)
        b_frend = box(r"fitted render", LEFT * 3.6 + DOWN * 0.9, 3.0)
        a1 = Arrow(b_prop.get_right(), b_rend.get_left(), buff=0.1, color=MUTED, stroke_width=2.5)
        g1 = CurvedArrow(b_rend.get_top() + LEFT * 0.4, b_prop.get_top() + RIGHT * 0.4, angle=0.9, color=FB, stroke_width=3)
        g1_l = Tex(r"loop 1: the gain from step 8's render, then 4.7--4.10 again", color=FB).scale(0.45).next_to(g1, UP, buff=0.06)
        a2 = Arrow(b_rend.get_bottom(), b_fit.get_top(), buff=0.1, color=MUTED, stroke_width=2.5)
        a3 = Arrow(b_fit.get_left(), b_frend.get_right(), buff=0.1, color=MUTED, stroke_width=2.5)
        g2 = Arrow(b_frend.get_top(), b_prop.get_bottom(), buff=0.1, color=FB, stroke_width=3)
        g2_l = VGroup(Tex(r"loop 2: the gain from the fitted render,", color=FB).scale(0.45),
                      Tex(r"the proposer once more, refit (one iteration)", color=FB).scale(0.45)).arrange(DOWN, buff=0.06)
        g2_l.next_to(VGroup(b_frend, b_fit), DOWN, buff=0.3)
        self.play(FadeIn(b_prop), FadeIn(b_rend), GrowArrow(a1))
        self.play(Create(g1), FadeIn(g1_l), run_time=1.3)
        self.wait(1.0)
        self.play(GrowArrow(a2), FadeIn(b_fit), GrowArrow(a3), FadeIn(b_frend))
        self.play(GrowArrow(g2), FadeIn(g2_l), run_time=1.2)
        self.wait(2.0)
        self.say(r"Gate (spec 6.2, phase 6): 3-way junction recall up, at equal junction precision. Status: not started.", wait="read")
        self.outro()


variants(Feedback, module_globals=globals())
