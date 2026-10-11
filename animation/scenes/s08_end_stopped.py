"""Chapter 8: where lines end.

End-stopped cells (spec 4.2): a line here and behind, none ahead.  Level 1 slides a
probe with its two lobes to the end of a line while live meters show C, A-, A+ and E;
level 2 adds the equation, the 32 direction channels and the end map of a toy scene;
level 3 adds the build (MICA layer 07), the real-image problem and the divisive variant.

Render:  manim -ql scenes/s08_end_stopped.py EndStoppedShort EndStoppedMedium EndStoppedDetailed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import v2cells as V  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (CONT, END, EXC, INH, MUTED, PixelFrame, bilinear, image,  # noqa: E402
                             orientation_color)
from mica_anim.v2draw import END_RAMP, Meters, end_arrow, tiny  # noqa: E402

C_COL = WHITE                    # "a line here": the line cell's own response


class EndStopped(ChapterScene):
    NUMBER, TITLE, SPEC = 8, "Where lines end", "spec 4.2"
    MAP_KEYS = ("endstop",)

    def construct(self):
        self.intro()
        self.probe_along_a_line()
        if self.at(2):
            self.channels_and_toy_scene()
        if self.at(3):
            self.as_built()
            self.divisive()
        self.outro()

    # ================================================================ level 1: the probe
    def probe_along_a_line(self):
        S = V.single_line()
        pf = PixelFrame(V.LINE_SHAPE, 3.6, np.array([-2.75, 0.75, 0.0]))
        u = S["u"]
        n = np.array([-u[1], u[0]])
        P = S["P"][S["k"]]
        od = image(S["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        frame = SurroundingRectangle(od, buff=0, color=MUTED, stroke_width=1)
        head = Tex("a vessel that ends (optical density)").scale(0.55).next_to(frame, UP, buff=0.12)
        self.play(FadeIn(od), Create(frame), FadeIn(head), run_time=1.0)
        self.say(r"Where does a vessel end? A cell that sees a line responds all along it, up to its last pixel.")

        s = ValueTracker(-62.0)

        def q():
            return S["end"] + s.get_value() * u

        cache = {}

        def vals():
            key = round(s.get_value(), 4)
            if key not in cache:
                cache[key] = V.probe_values(S, *q())
            return cache[key]

        probe = always_redraw(lambda: VGroup(Circle(radius=0.085, stroke_color=WHITE, stroke_width=2.5),
                                             Dot(radius=0.025, color=WHITE)).move_to(pf.point(*q())))

        # --- meters
        M = Meters([r"C", r"A^-", r"A^+", r"E"], [C_COL, EXC, INH, END], vmax=72.0, height=2.5, spacing=1.0)
        M.move_to(np.array([4.4, -0.2, 0.0]))
        words = VGroup(*[tiny(w, c, 0.42).next_to(lb, DOWN, buff=0.1)
                         for w, c, lb in zip(["here", "behind", "ahead", "end"], [GREY_A, EXC, INH, END], M.labels)])
        units = tiny(r"toy cells, noise units").next_to(words, DOWN, buff=0.18)
        bars = [always_redraw(lambda i=i: M.bar(i, vals()[i])) for i in range(4)]
        nums = [M.number(i, lambda i=i: vals()[i]) for i in range(4)]

        self.play(FadeIn(probe, scale=2.0), FadeIn(M.base), FadeIn(M.labels[0]), FadeIn(words[0]), FadeIn(units))
        self.add(bars[0], nums[0])
        self.wait(1.2)

        # --- the lobes (the one-sentence association field, for cuts without chapter 7)
        ds, ws = V.lobe_samples(1.0)
        wmax = ws.max()

        def lobe(sign, color):
            def build():
                c0 = q()
                g = VGroup()
                outline_l, outline_r = [], []
                for d, w in zip(ds, ws):
                    c = c0 + sign * d * u
                    hw = 0.7 + 1.5 * w / wmax
                    v = float(bilinear(P[None], c[0], c[1])[0]) / 60.0
                    a, b = c - 0.5 * u * sign, c + 0.5 * u * sign
                    quad = [pf.point(*(a + hw * n)), pf.point(*(b + hw * n)), pf.point(*(b - hw * n)), pf.point(*(a - hw * n))]
                    g.add(Polygon(*quad, stroke_width=0, fill_color=color, fill_opacity=0.10 + 0.8 * np.clip(v, 0, 1)))
                    outline_l.append(pf.point(*(c + hw * n)))
                    outline_r.append(pf.point(*(c - hw * n)))
                g.add(Polygon(*(outline_l + outline_r[::-1]), stroke_color=color, stroke_width=1.5, fill_opacity=0))
                return g
            return always_redraw(build)

        behind_l, ahead_l = lobe(-1, EXC), lobe(+1, INH)
        self.say(r"Horizontal connections give each cell two lobes: $A^-$ pools the line behind it, $A^+$ the line ahead.")
        self.play(FadeIn(behind_l), FadeIn(ahead_l), FadeIn(M.labels[1:3]), FadeIn(words[1:3]), run_time=1.0)
        self.add(bars[1], bars[2], nums[1], nums[2])
        self.wait(2.0)
        self.say(r"An end-stopped cell fires where there is line here and behind, but none ahead.")
        self.play(FadeIn(M.labels[3]), FadeIn(words[3]))
        self.add(bars[3], nums[3])
        self.wait(1.4)

        # --- slide to the end
        self.say(r"Near the end, the lobe ahead runs off the line: $A^+$ collapses, and $E$ fires.")
        self.play(s.animate.set_value(-16.0), run_time=2.2, rate_func=smooth)
        self.play(s.animate.set_value(0.0), run_time=2.6, rate_func=smooth)
        arrow = end_arrow(pf, *S["end"], V.PHIS[S["j"]], length=0.8, width=6).shift(pf.unit * 2.5 * V.scene_vec(V.PHIS[S["j"]]))
        self.play(GrowArrow(arrow), Indicate(M.labels[3], color=END), run_time=1.0)
        self.say(r"An end, with its free side pointing out of the line.", wait=0.8)
        if self.at(2):
            self.say(r"Past the end there is no line here: $C$, and with it $E$, falls silent.")
            self.play(s.animate.set_value(10.0), run_time=1.8, rate_func=smooth)
            self.play(s.animate.set_value(0.0), run_time=1.4, rate_func=smooth)
        self.say(r"Such cells exist in V1 and V2 (end-stopped, or hypercomplex): the complement of the bipole, which needs line on both sides.",
                 wait="read")

        self._line = dict(pf=pf, od=od, frame=frame, head=head, probe=probe, lobes=VGroup(behind_l, ahead_l), M=M,
                          words=words, units=units, bars=bars, nums=nums, arrow=arrow)
        if self.at(2):
            self.equation()

    # ================================================================ level 2: the equation
    def equation(self):
        L = self._line
        M = L["M"]
        eq = MathTex(r"E", r"=", r"\max\big(0,\;", r"\min(", r"C", r",\,", r"A^-", r")", r"-", r"\beta", r"A^+", r"\big)").scale(0.72)
        eq.move_to(np.array([3.95, 3.15, 0.0]))
        if eq.get_right()[0] > 6.6:
            eq.shift(LEFT * (eq.get_right()[0] - 6.6))
        for i, c in ((0, END), (4, C_COL), (6, EXC), (9, INH), (10, INH)):
            eq[i].set_color(c)
        self.say(r"In symbols: the smaller of line here and line behind, minus line ahead, and never below zero.")
        self.play(TransformFromCopy(M.labels[3], eq[0]), FadeIn(eq[1]), run_time=0.9)
        self.play(FadeIn(eq[2]), FadeIn(eq[3]), TransformFromCopy(M.labels[0], eq[4]), FadeIn(eq[5]),
                  TransformFromCopy(M.labels[1], eq[6]), FadeIn(eq[7]), run_time=1.2)
        b1 = Brace(eq[3:8], DOWN, buff=0.08, color=GREY_B)
        t1 = tiny(r"line here and behind", GREY_A, 0.45).next_to(b1, DOWN, buff=0.06)
        self.play(GrowFromCenter(b1), FadeIn(t1), run_time=0.7)
        self.play(FadeIn(eq[8]), FadeIn(eq[9]), TransformFromCopy(M.labels[2], eq[10]), FadeIn(eq[11]), run_time=1.0)
        b2 = Brace(eq[8:11], DOWN, buff=0.08, color=INH)
        t2 = tiny(r"none ahead", INH, 0.45).next_to(b2, DOWN, buff=0.06)
        if t2.get_left()[0] < t1.get_right()[0] + 0.12:                  # stagger, never touch
            t2.shift(DOWN * (t2.height + 0.08))
        self.play(GrowFromCenter(b2), FadeIn(t2), run_time=0.7)
        self.wait(1.0)
        gate = MathTex(r"\beta = 1,\quad \text{only where } C > t_{\mathrm{low}}", color=GREY_B).scale(0.55)
        gate.next_to(eq, UP, buff=0.12).align_to(eq, RIGHT)
        self.say(r"$\beta = 1$ to start, and only where a line is present: $C$ above the low threshold.", wait=0.6)
        self.play(FadeIn(gate))
        self.wait(1.4)
        self._eq = VGroup(eq, b1, t1, b2, t2, gate)

    # ================================================================ level 2: 32 channels, a toy scene
    def channels_and_toy_scene(self):
        L = self._line
        for b in L["bars"]:
            b.clear_updaters()
        for nm in L["nums"]:
            nm.clear_updaters()
        L["probe"].clear_updaters()
        for lb in L["lobes"]:
            lb.clear_updaters()
        self.clear(run_time=0.8)

        # --- the wheel: 16 orientations x 2 free directions
        c = np.array([-3.7, 0.3, 0.0])
        R = 1.65
        lines = VGroup()
        for k, th in enumerate(V.N.THETAS):
            d = np.array([np.cos(-th), np.sin(-th), 0.0])
            lines.add(Line(c - d * R, c + d * R, stroke_color=orientation_color(th), stroke_width=4))
        self.say(r"One end cell per orientation would not say which way the line stops. So each orientation has two.")
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.08), run_time=1.6)
        arrows = VGroup()
        for j, phi in enumerate(V.PHIS):
            v = V.scene_vec(phi)
            arrows.add(Arrow(c + v * 0.25, c + v * R, buff=0, color=orientation_color(phi % np.pi), stroke_width=4,
                             max_tip_length_to_length_ratio=0.14, tip_length=0.16))
        self.play(*[ReplacementTransform(lines[k], VGroup(arrows[k], arrows[k + 16])) for k in range(16)], run_time=1.4)
        n32 = MathTex(r"16 \times 2 = 32", r"\text{ direction channels}").scale(0.75)
        sub = VGroup(MathTex(r"E_{\mathrm{fwd}}[k]:\ \text{free side } +t_k", color=GREY_A).scale(0.6),
                     MathTex(r"E_{\mathrm{bwd}}[k]:\ \text{free side } -t_k \ (A^+ \leftrightarrow A^-)", color=GREY_A).scale(0.6)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(n32, DOWN, buff=0.35, aligned_edge=LEFT)
        VGroup(n32, sub).move_to(np.array([3.4, 0.3, 0.0]))
        self.say(r"16 orientations, 2 directions each: 32 channels, so an end's free direction covers the full $360^\circ$.")
        self.play(Write(n32), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.1))
        ours = arrows[V.LINE_J]
        self.play(ours.animate.set_color(END).set_stroke(width=7), Indicate(ours, color=END, scale_factor=1.15), run_time=1.0)
        ours_l = tiny(r"the end we just found", END, 0.5).next_to(ours.get_end(), UP + RIGHT, buff=0.05)
        self.play(FadeIn(ours_l))
        self.wait(1.2)
        self.clear(run_time=0.7)

        # --- the toy scene
        T = V.toy_scene()
        pf = PixelFrame(V.SCENE_SHAPE, 4.5, np.array([0.0, 0.55, 0.0]))
        od = image(T["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        Em = T["E"].max(axis=0)
        em = image(Em, pf.height, END_RAMP, vmin=0, vmax=0.8 * Em.max()).move_to(pf.centre)
        frame = SurroundingRectangle(od, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"a toy scene (optical density)").scale(0.55).next_to(frame, UP, buff=0.12)
        head2 = Tex(r"end-stopped cells: the strongest of 32 channels at each pixel").scale(0.55).move_to(head)
        self.play(FadeIn(od), Create(frame), FadeIn(head))
        self.say(r"A toy scene: a T, a Y, a crossing, and five lines that simply stop.", wait="read")
        self.play(FadeIn(em), FadeTransform(head, head2), run_time=1.4)
        self.say(r"All 32 channels at once. Each bright spot is a line that ends there.", wait="read")
        self.wait(1.0)
        ends = T["ends"]
        def crowded(x, y):
            return any(0 < np.hypot(x - a, y - b) < 9 for a, b, _, _ in ends)
        arrs = VGroup(*[end_arrow(pf, x, y, V.PHIS[j], length=0.05, back=0.4, width=4) if crowded(x, y)
                        else end_arrow(pf, x, y, V.PHIS[j], length=0.42, width=4) for x, y, j, v in ends])
        self.play(od.animate.set_opacity(1.0), FadeOut(em), run_time=0.8)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.12), run_time=2.0)
        note = tiny(rf"toy: {len(ends)} ends above {V.TOY_TAU_END:.0f} noise units", MUTED, 0.45).next_to(frame, DOWN, buff=0.12).align_to(frame, RIGHT)
        self.play(FadeIn(note))
        if len(ends) == 9:
            self.say(r"Nine ends: five free, one where the T's branch meets its parent, three meeting at the Y. The crossing has none.", wait="read")
        else:
            self.say(rf"{len(ends)} ends, free ones and ends meeting at the T and the Y. The crossing has none.", wait="read")
        hl = VGroup(*[Circle(radius=17 * pf.unit, stroke_color=GREY_A, stroke_width=2).move_to(pf.point(*p))
                      for p in (V.T_C, V.Y_C)])
        self.play(Create(hl), run_time=0.8)
        self.say(r"So a junction is a place where ends point in and lines pass through. That is the next layer.", wait="read")
        self._scene = Group(od, frame, head2, arrs, note, hl)

    # ================================================================ level 3: as built
    def as_built(self):
        self.clear(run_time=0.8)
        title = Tex(r"As built: MICA layer 07 (spec 4.2)").scale(0.7).to_edge(UP, buff=0.35)
        self.play(FadeIn(title))
        lh = Tex(r"first version: rules", color=GREY_A).scale(0.62)
        rules = VGroup(*[Tex(t).scale(0.55) for t in (r"a second-lobe test", r"a gate at $2\times$ the median",
                                                         r"a boolean veto", r"spur pruning")]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        left = VGroup(lh, rules).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(np.array([-4.0, 0.9, 0.0]))
        self.say(r"The first build patched defects with rules. They removed spurious ends, but broke junctions.")
        self.play(FadeIn(lh), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rules], lag_ratio=0.2), run_time=1.6)
        strikes = VGroup(*[Line(r.get_left() + LEFT * 0.05, r.get_right() + RIGHT * 0.05, stroke_color=INH, stroke_width=3)
                           for r in rules])
        broke = Tex(r"spurious ends gone, junctions broken", color=INH).scale(0.5).next_to(rules, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.15), FadeIn(broke), run_time=1.4)
        self.wait(1.0)

        rh = Tex(r"now: inhibition between cells", color=GREY_A).scale(0.62)
        items = VGroup(
            self._item(self._glyph_blob(), [r"untuned suppression: a blob-like response", r"drives every orientation, so it is suppressed in all"]),
            self._item(self._glyph_wta(), [r"winner-take-all among the nearest rivals:", r"across the vessel, and neighbouring orientations"]),
            self._item(self._glyph_end_bipole(), [r"an end cell inhibits the bipole of its own", r"orientation: a line that ends here does not pass"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        right = VGroup(rh, items).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(np.array([2.75, 0.55, 0.0]))
        self.say(r"They were replaced by inhibition between cells, the way cortex does it.")
        self.play(FadeIn(rh))
        for it in items:
            self.play(FadeIn(it, shift=LEFT * 0.2), run_time=0.8)
            self.wait(0.9)
        toy = tiny(r"(this chapter's toy uses the winner-take-all gate)").next_to(items, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(toy))
        self.say(r"Correct on a synthetic line end. But on the real test image these cells find 1,399 ends (measured, spec 4.2): too many.", wait="read")
        self.clear(run_time=0.8)

        # --- why: a real vessel's strength varies along it (computed)
        W = V.varying_vessel()
        pf = PixelFrame(W["od"].shape, 1.95, np.array([-2.85, 1.55, 0.0]))
        od1 = image(W["od"], pf.height, "magma", vmin=-0.02, vmax=0.22).move_to(pf.centre)
        fr1 = SurroundingRectangle(od1, buff=0, color=MUTED, stroke_width=1)
        h1 = Tex(r"a toy vessel whose strength varies along it (55--145\%)").scale(0.5).next_to(fr1, UP, buff=0.1)
        self.play(FadeIn(od1), Create(fr1), FadeIn(h1), run_time=0.8)
        self.say(r"Why: a real vessel's strength varies along it, and the subtractive form fires a little wherever it weakens ahead.")
        sub_arr = VGroup(*[end_arrow(pf, x, y, V.PHIS[j], length=0.45, width=4) for x, y, j, v in W["sub_ends"]])
        self.play(LaggedStart(*[GrowArrow(a) for a in sub_arr], lag_ratio=0.25), run_time=1.6)
        lab1 = Tex(rf"subtractive: {len(W['sub_ends'])} ends (1 real)", color=END).scale(0.55).next_to(fr1, RIGHT, buff=0.3)
        self.play(FadeIn(lab1))
        self.wait(1.6)

        meas = VGroup(
            Tex(r"measured on the real test image (spec 4.2):", color=GREY_B).scale(0.5),
            MathTex(r"\beta = 1:\ 1{,}399 \text{ ends}").scale(0.62),
            MathTex(r"\text{larger } \beta:\ 185 \text{ ends, but}").scale(0.62),
            Tex(r"the synthetic T is mistyped,").scale(0.52),
            MathTex(r"\text{real crossings } 227 \to 496").scale(0.62),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(np.array([-2.9, -1.45, 0.0]))
        self.say(r"Stronger inhibition from ahead cuts the ends to 185, but mistypes the synthetic T and doubles the real crossings.")
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in meas], lag_ratio=0.3), run_time=2.0)
        why = Tex(r"$\beta$ also sets the end-to-bipole inhibition:\\fewer ends, so more passing lines survive", color=GREY_A).scale(0.5)
        why.move_to(np.array([3.3, -1.45, 0.0]))
        self.play(FadeIn(why))
        self.say(r"Probably because $\beta$ also sets how strongly ends silence the bipole. The setting that keeps synthetic cases right is kept.", wait="read")
        self._vary = dict(pf=pf, od=od1, fr=fr1, h=h1, sub=sub_arr, lab=lab1, meas=meas, why=why)

    def _item(self, glyph, lines):
        t = VGroup(*[Tex(l).scale(0.46) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        if glyph.width / glyph.height > 1.2:
            glyph.scale_to_fit_width(0.62)
        else:
            glyph.scale_to_fit_height(0.46)
        slot = Rectangle(width=0.7, height=0.5, stroke_opacity=0, fill_opacity=0)
        glyph.move_to(slot)
        t.next_to(slot, RIGHT, buff=0.2)
        return VGroup(slot, glyph, t)

    def _glyph_blob(self):
        g = VGroup(*[Line(ORIGIN, 0.35 * np.array([np.cos(a), np.sin(a), 0]), stroke_color=orientation_color(a % PI), stroke_width=3)
                     for a in np.linspace(0, TAU, 12, endpoint=False)])
        g.add(MathTex(r"-", color=INH).scale(0.9).next_to(g, RIGHT, buff=0.05))
        return g

    def _glyph_wta(self):
        bars = VGroup(*[Line(DOWN * 0.3, UP * 0.3, stroke_color=WHITE, stroke_width=4, stroke_opacity=o).shift(RIGHT * x)
                        for x, o in ((-0.2, 0.3), (0.0, 1.0), (0.2, 0.3))])
        return bars

    def _glyph_end_bipole(self):
        a = Arrow(LEFT * 0.45, LEFT * 0.08, buff=0, color=END, stroke_width=4, max_tip_length_to_length_ratio=0.4)
        bar = Line(UP * 0.12, DOWN * 0.12, stroke_color=INH, stroke_width=4).shift(RIGHT * 0.0)
        inh = Line(LEFT * 0.08, RIGHT * 0.0, stroke_color=INH, stroke_width=3)
        bip = Line(RIGHT * 0.05, RIGHT * 0.5, stroke_color=CONT, stroke_width=4)
        return VGroup(a, inh, bar, bip)

    # ================================================================ level 3: the divisive variant
    def divisive(self):
        W = V.varying_vessel()
        D = self._vary
        self.play(FadeOut(D["meas"]), FadeOut(D["why"]), run_time=0.6)
        eq = MathTex(r"E", r"=", r"{\big[\,", r"A^-", r"-", r"\beta", r"A^+", r"\,\big]_+", r"\over", r"A^-", r"+", r"A^+", r"+", r"c}").scale(0.85)
        for i, c in ((0, END), (3, EXC), (5, INH), (6, INH), (9, EXC), (11, INH)):
            eq[i].set_color(c)
        eq.move_to(np.array([3.6, -0.15, 0.0]))
        tag = Tex(r"divisive end-stopping (next variant, not yet tested)", color=GREY_B).scale(0.48).next_to(eq, UP, buff=0.2)
        self.say(r"The next variant divides instead: shunting inhibition by the local line activity, so only the relative drop ahead counts.")
        self.play(FadeIn(tag), Write(eq[:8]), run_time=1.4)
        self.play(Write(eq[8:]), run_time=1.2)
        self.wait(0.8)
        pf2 = PixelFrame(W["od"].shape, 1.95, np.array([-2.85, -1.0, 0.0]))
        od2 = image(W["od"], pf2.height, "magma", vmin=-0.02, vmax=0.22).move_to(pf2.centre)
        fr2 = SurroundingRectangle(od2, buff=0, color=MUTED, stroke_width=1)
        div_arr = VGroup(*[end_arrow(pf2, x, y, V.PHIS[j], length=0.45, width=4) for x, y, j, v in W["div_ends"]])
        lab2 = Tex(rf"divisive: {len(W['div_ends'])} end", color=END).scale(0.55).next_to(fr2, RIGHT, buff=0.3)
        self.play(FadeIn(od2), Create(fr2))
        self.play(LaggedStart(*[GrowArrow(a) for a in div_arr], lag_ratio=0.25), FadeIn(lab2), run_time=1.2)
        cnote = tiny(rf"toy: $c = {V.C_DIV:.0f}$, gate $C > {V.DIV_GATE:.0f}$ noise units, threshold {V.DIV_THR}").next_to(fr2, DOWN, buff=0.1).align_to(fr2, LEFT)
        self.play(FadeIn(cnote))
        self.say(r"On the same toy vessel, the divisive form fires once, at the real end.", wait="read")
        self.clear(run_time=0.8)

        # --- the response against dip depth, subtractive and divisive (computed)
        depth = V.DIP_DEPTHS * 100
        strong, weak = 64.0, 13.0
        s_sub, s_div = V.dip_response(strength=strong)
        w_sub, w_div = V.dip_response(strength=weak)
        cfg = dict(color=MUTED, include_tip=False, font_size=22)
        ax1 = Axes(x_range=[0, 100, 25], y_range=[0, 35, 10], x_length=5.2, y_length=2.7, axis_config=cfg,
                   x_axis_config=dict(numbers_to_include=[0, 25, 50, 75, 100]),
                   y_axis_config=dict(numbers_to_include=[10, 20, 30])).move_to(np.array([-3.35, 0.15, 0.0]))
        ax2 = Axes(x_range=[0, 100, 25], y_range=[0, 1, 0.25], x_length=5.2, y_length=2.7, axis_config=cfg,
                   x_axis_config=dict(numbers_to_include=[0, 25, 50, 75, 100]),
                   y_axis_config=dict(numbers_to_include=[0.5, 1.0], decimal_number_config=dict(num_decimal_places=1))
                   ).move_to(np.array([3.55, 0.15, 0.0]))
        t1 = Tex(r"subtractive (noise units)").scale(0.6).next_to(ax1, UP, buff=0.25)
        t2 = Tex(r"divisive (a ratio)").scale(0.6).next_to(ax2, UP, buff=0.25)
        xl1 = Tex(r"dip depth (\%) \quad 100 = a real end").scale(0.5).next_to(ax1.x_axis, DOWN, buff=0.4)
        xl2 = xl1.copy().next_to(ax2.x_axis, DOWN, buff=0.4)
        self.say(r"Against dip depth, computed on toy vessels: a strong one and a weak one. 100\% is a real end.")
        self.play(Create(ax1), Create(ax2), FadeIn(t1), FadeIn(t2), FadeIn(xl1), FadeIn(xl2), run_time=1.2)

        def curve(ax, y, col, dashed=False):
            g = ax.plot_line_graph(depth, y, line_color=col, add_vertex_dots=False, stroke_width=4)["line_graph"]
            return DashedVMobject(g, num_dashes=40) if dashed else g

        weak_col = interpolate_color(END, WHITE, 0.45)
        cs1, cw1 = curve(ax1, s_sub, END), curve(ax1, w_sub, weak_col, True)
        cs2, cw2 = curve(ax2, s_div, END), curve(ax2, w_div, weak_col, True)
        key = VGroup(Tex(r"strong vessel", color=END).scale(0.5), Tex(r"weak vessel", color=weak_col).scale(0.5)
                     ).arrange(RIGHT, buff=0.4).next_to(t1, UP, buff=0.12)
        key.move_to([0.1, 2.5, 0])
        thr1 = DashedLine(ax1.c2p(0, V.TOY_TAU_END), ax1.c2p(100, V.TOY_TAU_END), color=GREY_B, stroke_width=2)
        thr2 = DashedLine(ax2.c2p(0, V.DIV_THR), ax2.c2p(100, V.DIV_THR), color=GREY_B, stroke_width=2)
        tl1 = tiny(r"threshold").next_to(ax1.c2p(100, V.TOY_TAU_END), UP + LEFT, buff=0.05)
        tl2 = tiny(r"threshold").next_to(ax2.c2p(0, V.DIV_THR), UP + RIGHT, buff=0.05)
        self.play(Create(cs1), Create(cw1), Create(cs2), Create(cw2), FadeIn(key), run_time=2.2)
        self.play(Create(thr1), Create(thr2), FadeIn(tl1), FadeIn(tl2))
        i20 = int(np.argmin(np.abs(depth - 20)))
        d1 = Dot(ax1.c2p(20, s_sub[i20]), color=END, radius=0.06)
        d2 = Dot(ax1.c2p(100, w_sub[-1]), color=weak_col, radius=0.06)
        l1 = MathTex(f"{s_sub[i20]:.1f}", color=END).scale(0.55).next_to(d1, UP + LEFT, buff=0.05)
        l2 = MathTex(f"{w_sub[-1]:.1f}", color=weak_col).scale(0.55).next_to(d2, DOWN + LEFT, buff=0.05)
        self.say(r"Subtractive: a 20\% dip on a strong vessel beats a real end on a weak one. No threshold separates them.")
        self.play(FadeIn(d1, scale=2), FadeIn(l1), FadeIn(d2, scale=2), FadeIn(l2))
        self.wait(2.2)
        e1 = Dot(ax2.c2p(20, s_div[i20]), color=END, radius=0.06)
        e2 = Dot(ax2.c2p(100, w_div[-1]), color=weak_col, radius=0.06)
        m1 = MathTex(f"{s_div[i20]:.2f}", color=END).scale(0.55).next_to(e1, UP, buff=0.08)
        m2 = MathTex(f"{w_div[-1]:.2f}", color=weak_col).scale(0.55).next_to(e2, RIGHT, buff=0.1)
        self.say(rf"Divisive: the dip gives under 0.1 on any vessel, a real end {w_div[-1]:.1f} or more. One threshold separates them.")
        self.play(FadeIn(e1, scale=2), FadeIn(m1), FadeIn(e2, scale=2), FadeIn(m2))
        self.wait(2.0)

        # --- against contrast: a live vessel strength
        st = ValueTracker(weak)
        live1 = always_redraw(lambda: ax1.plot_line_graph(depth, V.dip_response(strength=st.get_value())[0], line_color=WHITE,
                                                          add_vertex_dots=False, stroke_width=3))
        live2 = always_redraw(lambda: ax2.plot_line_graph(depth, V.dip_response(strength=st.get_value())[1], line_color=WHITE,
                                                          add_vertex_dots=False, stroke_width=3))
        rd = always_redraw(lambda: MathTex(rf"\text{{vessel strength }} C_0 = {st.get_value():.0f}").scale(0.6)
                           .move_to(np.array([0.1, -2.55, 0.0])))
        self.say(r"Now sweep the vessel's contrast: the subtractive response grows with it; the divisive one hardly moves.")
        self.play(FadeIn(live1), FadeIn(live2), FadeIn(rd))
        self.play(st.animate.set_value(66.0), run_time=3.5, rate_func=smooth)
        self.play(st.animate.set_value(weak), run_time=2.0, rate_func=smooth)
        for m in (live1, live2, rd):
            m.clear_updaters()
        lo_end, hi_end = w_div[-1], s_div[-1]
        lo_dip, hi_dip = min(w_div[i20], s_div[i20]), max(w_div[i20], s_div[i20])
        honest = Tex(rf"toy: real end {lo_end:.2f}--{hi_end:.2f} (spec: about 1), 20\% dip {lo_dip:.2f}--{hi_dip:.2f} (spec: about 0.1)",
                     color=GREY_B).scale(0.5)
        honest.move_to(np.array([0.1, -2.55, 0.0]))
        self.play(FadeOut(rd), FadeIn(honest))
        self.say(r"The toy agrees with the spec's prediction. In MICA, divisive end-stopping is designed but not yet tested (spec 4.2).",
                 wait="read")


variants(EndStopped, module_globals=globals())
