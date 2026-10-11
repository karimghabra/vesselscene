"""Chapter 5: coarse cells that overshoot (spec 4.6b).

A coarse size channel's cell is ~100 px long, so past a thin vessel's end it
keeps firing, and the per-orientation max over channels lets that tail stand
for the vessel.  End-zone inhibition with a centre pool silences the tail and
leaves a continuous vessel (and a wide one) untouched.  Everything plotted is
computed by mica_anim/scalesel.py on toy lines.

Render:  manim -ql scenes/s05_scale_selection.py ScaleSelectionDetailed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import scalesel as S  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import ATTN, END, EXC, INH, MUTED, PixelFrame, image  # noqa: E402
from mica_anim.widgets import receptive_field  # noqa: E402

FINE_COL = TEAL_B
COARSE_COL = PURPLE_A
CROP = (36, 84)                    # rows of the computed image shown on screen (the line is at row 60)
PF = PixelFrame((CROP[1] - CROP[0], S.SHAPE[1]), 2.2, np.array([0.45, 2.3, 0.0]))
YMAX = 50.0


def px(col, row=S.ROW):
    """Scene point of a computed-image pixel (column, row)."""
    return PF.point(col, row - CROP[0])


def od_image(name):
    return image(S.compute(name)["od"][CROP[0]:CROP[1]], PF.height, "magma", vmin=-0.02, vmax=0.17).move_to(PF.centre)


def make_axes(ymax=YMAX, step=20):
    """Response along the line's axis, against the column of the cell's centre;
    columns line up with the image above."""
    ax = Axes(x_range=[0, S.SHAPE[1], 40], y_range=[0, ymax, step], x_length=PF.width, y_length=3.15,
              axis_config=dict(color=MUTED, include_tip=False, font_size=22, stroke_width=1.5),
              x_axis_config=dict(include_ticks=False),
              y_axis_config=dict(numbers_to_include=list(range(0, int(ymax) + 1, step))))
    ax.shift(np.array([px(0)[0] - PF.unit / 2 - ax.c2p(0, 0)[0], -2.6 - ax.c2p(0, 0)[1], 0.0]))
    yl = Tex("noise units").scale(0.5).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.42)
    xl = Tex(r"centre of the cell (column) $\rightarrow$", color=MUTED).scale(0.45)
    xl.next_to(ax.c2p(S.SHAPE[1], 0), UP + LEFT, buff=0.08)
    return ax, VGroup(yl, xl)


def prof_curve(ax, prof, a, b, color, width=3.5, dashed=False, opacity=1.0):
    a, b = int(round(a)), int(round(b))
    if b - a < 1:
        return VMobject()
    xs = np.arange(a, b + 1)
    pts = [ax.c2p(x + 0.5, min(prof[x], ax.y_range[1])) for x in xs]
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity).set_points_as_corners(pts)
    return DashedVMobject(m, num_dashes=max(int((b - a) / 3), 2)) if dashed else m


def interp(prof, x):
    return float(np.interp(x, np.arange(len(prof)), prof))


def zone_boxes(xc, height_px=11):
    """The coarse cell's two end zones (red) and its centre pool (blue), on the image row."""
    h = height_px * PF.unit
    def box(lo, hi, col):
        r = Rectangle(width=(hi - lo) * PF.unit, height=h, stroke_color=col, stroke_width=2.5,
                      fill_color=col, fill_opacity=0.18)
        return r.move_to(px(xc + (lo + hi) / 2))
    return VGroup(box(-S.ZONE_OUT, -S.ZONE_IN, INH), box(-S.CENTRE, S.CENTRE, EXC), box(S.ZONE_IN, S.ZONE_OUT, INH))


def fine_bars(name, step=4, opacity_scale=40.0):
    """The fine cells (same orientation) along the line: one short bar every 4 px, bright where they fire."""
    F = S.compute(name)["F"][S.ROW]
    g = VGroup()
    for c in range(2, S.SHAPE[1], step):
        o = float(np.clip(F[c] / opacity_scale, 0, 1))
        g.add(Line(px(c - 1.4), px(c + 1.4), stroke_color=FINE_COL, stroke_width=5, stroke_opacity=0.15 + 0.85 * o))
    return g


def tex_table(rows, widths, scale=0.5, header_color=GREY_B, row_buff=0.2):
    """Rows of Tex cells, left-aligned in columns of the given widths; a rule under the header."""
    lines = VGroup()
    for i, row in enumerate(rows):
        cells = VGroup()
        x = 0.0
        for txt, w in zip(row, widths):
            c = Tex(txt, color=header_color if i == 0 else WHITE).scale(scale)
            c.move_to(np.array([x, 0, 0]), aligned_edge=LEFT)
            cells.add(c)
            x += w
        lines.add(cells)
    lines.arrange(DOWN, aligned_edge=LEFT, buff=row_buff)
    for row in lines:                                  # re-align columns after arrange
        x = row[0].get_left()[0]
        for c, w in zip(row, widths):
            c.move_to(np.array([x, row.get_y(), 0]), aligned_edge=LEFT)
            x += w
    rule = Line(lines.get_left(), lines.get_left() + RIGHT * sum(widths), stroke_color=MUTED, stroke_width=1)
    rule.next_to(lines[0], DOWN, buff=row_buff / 2).align_to(lines, LEFT)
    return VGroup(lines, rule)


class ScaleSelection(ChapterScene):
    NUMBER, TITLE, SPEC = 5, "Coarse cells that overshoot", "spec 4.6b"
    MAP_KEYS = ("scale",)

    def construct(self):
        self.intro()
        thin = S.compute("thin_end")
        cur = {"d": thin}                  # the stimulus the pool labels read
        R = S.ROW
        F, G, G2, I = thin["F"][R], thin["G"][R], thin["G2"][R], thin["I"][R]
        reach0 = S.reach(np.maximum(F, G), start=60)
        reach1 = S.reach(np.maximum(F, G2), start=60)
        g20, g20b = G[S.END + 20], G2[S.END + 20]

        # ------------------------------------------------------------ 1. a thin line that ends
        img = od_image("thin_end")
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        od_lbl = Tex("OD").scale(0.55).next_to(border, LEFT, buff=0.2)
        end_line = DashedLine(px(S.END, CROP[0]) + UP * 0.28, px(S.END, CROP[0]) + DOWN * 6.2, color=END,
                              stroke_width=2, dash_length=0.08)
        end_lbl = Tex(rf"end (column {S.END})", color=END).scale(0.5).next_to(end_line.get_start(), UP, buff=0.06)
        self.play(FadeIn(img), Create(border), FadeIn(od_lbl))
        self.say(r"A thin vessel, 3 px wide, ends at column 120. Two size channels look at it, at the same orientation.")
        self.play(Create(end_line), FadeIn(end_lbl), run_time=1.0)
        end_line.set_z_index(-1)

        ax, ax_lbls = make_axes()
        self.play(Create(ax), FadeIn(ax_lbls), run_time=1.0)

        # fine cell slides
        xc = ValueTracker(20.0)
        fine_rf = receptive_field(length=PF.unit * 4 * S.FINE[0] * S.FINE[1], sigma=PF.unit * S.FINE[0])
        fine_rf.add_updater(lambda m: m.move_to(px(xc.get_value())))
        fine_trace = always_redraw(lambda: prof_curve(ax, F, 20, xc.get_value(), FINE_COL))
        fine_dot = always_redraw(lambda: Dot(ax.c2p(xc.get_value() + 0.5, interp(F, xc.get_value())), radius=0.05, color=FINE_COL))
        fine_key = Tex(r"fine cell, 18 px long", color=FINE_COL).scale(0.5)
        coarse_key = Tex(r"coarse cell, $\sim$100 px long", color=COARSE_COL).scale(0.5)
        keys = VGroup(fine_key, coarse_key).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        keys.move_to(ax.c2p(172, 42), aligned_edge=LEFT)
        self.add(fine_rf, fine_trace, fine_dot)
        self.play(FadeIn(fine_rf), FadeIn(fine_key))
        self.say(r"A fine cell is 18 px long. Slide it along the line: it fires on the vessel and falls silent at its end.")
        self.play(xc.animate.set_value(220), run_time=3.5 if self.at(2) else 1.8, rate_func=linear)
        fine_trace.clear_updaters(); fine_dot.clear_updaters()
        self.play(FadeOut(fine_rf), FadeOut(fine_dot), run_time=0.5)
        fine_rf.clear_updaters()

        # coarse cell slides
        xc.set_value(20.0)
        coarse_rf = receptive_field(length=PF.unit * 4 * S.ALONG, sigma=PF.unit * S.COARSE[0])
        coarse_rf.add_updater(lambda m: m.move_to(px(xc.get_value())))
        coarse_trace = always_redraw(lambda: prof_curve(ax, G, 20, xc.get_value(), COARSE_COL))
        coarse_dot = always_redraw(lambda: Dot(ax.c2p(xc.get_value() + 0.5, interp(G, xc.get_value())), radius=0.05, color=COARSE_COL))
        self.add(coarse_rf, coarse_trace, coarse_dot)
        self.play(FadeIn(coarse_rf), FadeIn(coarse_key))
        self.say(r"A coarse cell is about 100 px long. It sums along its whole length, so it keeps firing well past the end.")
        self.play(xc.animate.set_value(S.END - 10), run_time=3.0 if self.at(2) else 2.0, rate_func=linear)
        self.play(xc.animate.set_value(S.END + 20), run_time=2.2 if self.at(2) else 1.5, rate_func=linear)
        tail_dot = Dot(ax.c2p(S.END + 20.5, g20), radius=0.06, color=END)
        tail_lbl = Tex(rf"{g20:.1f} noise units, 20 px past the end", color=END).scale(0.5)
        tail_lbl.next_to(tail_dot, UR, buff=0.08)
        self.play(FadeIn(tail_dot, scale=2), FadeIn(tail_lbl))
        self.wait(1.2 if self.at(2) else 0.6)
        self.play(xc.animate.set_value(190), run_time=2.0 if self.at(2) else 1.2, rate_func=linear)
        coarse_trace.clear_updaters(); coarse_dot.clear_updaters()
        self.play(FadeOut(coarse_dot), run_time=0.3)
        if self.at(3):
            self.say(rf"This toy: {g20:.1f} noise units 20 px past the end. Measured on a synthetic line (spec 4.6b): 12 units even 40 px past it.", wait="read")

        # the max over channels: grouping overshoots
        grp_y = PF.centre[1] + PF.height / 2 + 0.17
        def grp_bar(reach):
            return Line(np.array([px(0)[0] - PF.unit / 2, grp_y, 0]), np.array([px(reach)[0], grp_y, 0]),
                        stroke_color=ATTN, stroke_width=6)
        grp = grp_bar(reach0)
        grp_lbl = Tex(r"grouped", color=ATTN).scale(0.5).next_to(grp, LEFT, buff=0.15)
        self.say(r"Per orientation, the stronger channel speaks for the line. Past the end that is the coarse tail, so grouping runs on.")
        self.play(FadeOut(tail_lbl), FadeOut(tail_dot), Create(grp), FadeIn(grp_lbl), run_time=1.5)
        over_lbl = Tex(rf"overshoot {reach0 - S.END} px (toy)", color=END).scale(0.45)
        over_lbl.next_to(grp.get_end(), RIGHT, buff=0.15)
        self.play(FadeIn(over_lbl))
        self.wait(1.5)
        if self.at(3):
            self.say(r"In MICA the grouped thin line ran to the image border, and Y arms ran on past their junctions.", wait="read")

        # ------------------------------------------------------------ 2. end zones and a centre pool
        self.play(FadeOut(end_lbl), FadeOut(over_lbl), run_time=0.5)
        self.say(r"The fix: let the fine cells veto. Those in the coarse cell's two end zones inhibit it; those at its centre cancel that.")
        xc.set_value(40.0)
        coarse_rf.clear_updaters()
        self.play(coarse_rf.animate.move_to(px(40.0)), run_time=1.0)
        outline = coarse_rf[0]
        self.play(*[FadeOut(m) for m in coarse_rf[1:]], outline.animate.set_stroke(COARSE_COL, opacity=0.6),
                  img.animate.set_opacity(0.35), run_time=0.8)
        outline.add_updater(lambda m: m.move_to(px(xc.get_value())))
        bars = fine_bars("thin_end")
        fine_lbl = Tex(r"fine cells", color=FINE_COL).scale(0.5).next_to(border, LEFT, buff=0.2).shift(DOWN * 0.45)
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.02), FadeIn(fine_lbl), run_time=1.5)
        zones = always_redraw(lambda: zone_boxes(xc.get_value()))
        self.play(FadeIn(zones))

        def pool_label(sym, key, col, side):
            def f():
                v = interp(cur["d"][key][R], xc.get_value())
                t = MathTex(sym, "=", f"{v:.0f}").scale(0.5).set_color(col)
                z = zones[side]
                return t.move_to(np.array([z.get_x(), PF.centre[1] - PF.height / 2 - 0.2, 0.0]))
            return always_redraw(f)
        pools = VGroup(pool_label(r"E^-", "Em", INH, 0), pool_label(r"C", "Cz", EXC, 1), pool_label(r"E^+", "Ep", INH, 2))
        self.play(FadeIn(pools))

        # live inhibition and the selected coarse response, traced as the cell moves
        sel_trace = always_redraw(lambda: prof_curve(ax, G2, 20, xc.get_value(), WHITE, width=4))
        inh_trace = always_redraw(lambda: prof_curve(ax, I, 20, xc.get_value(), INH, width=3))
        sel_key = Tex(r"coarse, after the veto", color=WHITE).scale(0.5)
        inh_key = Tex(r"inhibition $[\max(E^+\!,E^-)-C]_+$", color=INH).scale(0.5)
        keys2 = VGroup(sel_key, inh_key).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(keys, DOWN, aligned_edge=LEFT, buff=0.12)
        self.play(coarse_trace.animate.set_stroke(opacity=0.45), fine_trace.animate.set_stroke(opacity=0.45),
                  FadeIn(keys2), run_time=0.8)
        self.add(sel_trace, inh_trace)
        self.play(xc.animate.set_value(20.0), run_time=0.4)
        rw = "read" if self.at(2) else 2.0
        self.play(xc.animate.set_value(70.0), run_time=2.0 if self.at(2) else 1.2, rate_func=linear)
        self.say(r"On a continuous vessel the end zones and the centre see the same fine cells. The pools are equal: nothing is lost.", wait=rw)
        self.play(xc.animate.set_value(S.END + 20), run_time=4.0 if self.at(2) else 2.5, rate_func=linear)
        self.say(r"Past the end only an end zone fires; the centre is empty. The inhibition wins and the coarse cell goes quiet.", wait=rw)
        self.play(xc.animate.set_value(190), run_time=2.0 if self.at(2) else 1.2, rate_func=linear)
        sel_trace.clear_updaters(); inh_trace.clear_updaters()

        grp1 = grp_bar(reach1)
        self.play(Transform(grp, grp1), run_time=1.5)
        stop_lbl = Tex(rf"now stops at column {reach1} (toy)", color=ATTN).scale(0.45).next_to(grp1.get_end(), RIGHT, buff=0.15)
        self.play(FadeIn(stop_lbl))
        if self.at(2):
            res0 = MathTex(r"\text{20 px past the end: }", f"{g20:.1f}", r"\to", f"{g20b:.0f}", r"\text{ (toy)}").scale(0.55)
            res0[1].set_color(COARSE_COL)
            res0.next_to(keys2, DOWN, aligned_edge=LEFT, buff=0.2)
            res = VGroup(res0)
            self.play(FadeIn(res0))
            self.wait(1.5)
            if self.at(3):
                meas = Tex(r"measured (spec 4.6b): $15 \to 2$", color=GREY_B).scale(0.5).next_to(res0, DOWN, aligned_edge=LEFT, buff=0.12)
                res.add(meas)
                self.play(FadeIn(meas))
                self.wait(1.5)
        else:
            res = VGroup()
            self.wait(1.0)

        # the wide vessel
        if self.at(2):
            self.play(FadeOut(stop_lbl), FadeOut(res), FadeOut(pools), run_time=0.5)
            wide = S.compute("wide_end")
            Gw, G2w = wide["G"][R], wide["G2"][R]
            img_w = od_image("wide_end").set_opacity(0.35)
            bars_w = fine_bars("wide_end")
            ax_w, ax_w_lbls = make_axes(ymax=140, step=40)
            self.say(r"A wide vessel has a flat top and edges for walls: fine cells barely fire on it, so its coarse response is untouched.")
            self.play(FadeOut(VGroup(sel_trace, inh_trace, coarse_trace, fine_trace)), FadeOut(grp), FadeOut(grp_lbl),
                      FadeOut(keys2), FadeOut(keys), xc.animate.set_value(50.0), run_time=0.8)
            cur["d"] = wide
            self.play(FadeIn(img_w), FadeOut(img), Transform(bars, bars_w), ReplacementTransform(ax, ax_w),
                      ReplacementTransform(ax_lbls, ax_w_lbls), FadeIn(pools), run_time=1.2)
            ax, ax_lbls, img = ax_w, ax_w_lbls, img_w
            wb = prof_curve(ax, Gw, 50, 100, COARSE_COL, dashed=True, opacity=0.9)
            wk = VGroup(Tex(r"coarse, before", color=COARSE_COL).scale(0.5), Tex(r"coarse, after the veto", color=WHITE).scale(0.5)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(ax.c2p(172, 125), aligned_edge=LEFT)
            self.play(Create(wb), FadeIn(wk), run_time=1.0)
            w_trace = always_redraw(lambda: prof_curve(ax, G2w, 50, xc.get_value(), WHITE, width=4))
            self.add(w_trace)
            self.play(xc.animate.set_value(80.0), run_time=3.0, rate_func=linear)
            w_trace.clear_updaters()
            note = VGroup(Tex(rf"fine cells on the wide vessel: {wide['Cz'][R, 80]:.0f} noise units", color=FINE_COL).scale(0.5),
                          Tex(rf"({S.compute('thin_end')['Cz'][R, 80]:.0f} on the thin line)", color=FINE_COL).scale(0.5)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(ax.c2p(112, 70), aligned_edge=LEFT)
            self.play(FadeIn(note))
            self.wait(2.0)
            self.play(FadeOut(VGroup(wb, wk, w_trace)), run_time=0.5)
        else:
            note = VGroup()

        if not self.at(2):
            self.outro()
            return

        # ------------------------------------------------------------ 3. the rule, written down
        self.play(FadeOut(note), FadeOut(ax), FadeOut(ax_lbls), FadeOut(end_line), run_time=0.6)
        eq1 = MathTex(r"I_k", "=", r"\Big[", r"\max\big(E^+_k,\,E^-_k\big)", "-", r"C_k", r"\Big]_+").scale(0.85)
        eq2 = MathTex(r"G'_k", "=", r"\max\!\big(0,\;", r"G_k", "-", r"w\,", r"I_k", r"\big)").scale(0.85)
        eq1[0].set_color(INH); eq1[3].set_color(INH); eq1[5].set_color(EXC)
        eq2[0].set_color(WHITE); eq2[3].set_color(COARSE_COL); eq2[6].set_color(INH)
        eqs = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(DOWN * 1.0 + LEFT * 1.6)
        lbl = VGroup(Tex(r"$E^\pm_k$: mean fine rate in an end zone", color=INH).scale(0.5),
                     Tex(r"$C_k$: mean fine rate at the centre", color=EXC).scale(0.5),
                     Tex(r"$G_k$: the coarse cell, same orientation $k$", color=COARSE_COL).scale(0.5),
                     Tex(r"toy form: spec 4.6b states it in words", color=GREY_B).scale(0.45)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(eqs, RIGHT, buff=0.8)
        self.say(r"Written down: inhibition is the end zones' excess over the centre, subtracted from the coarse cell of the same orientation.")
        self.play(Write(eq1[:2]), run_time=0.6)
        self.play(Write(eq1[2:4]), run_time=0.9)
        self.play(Write(eq1[4:]), run_time=0.9)
        self.play(TransformFromCopy(eq1[0], eq2[6]), Write(eq2[:6]), Write(eq2[7]), run_time=1.4)
        self.play(FadeIn(lbl, lag_ratio=0.2), run_time=1.5)
        self.wait(2.0)
        self.say(r"It runs after the surround and before the association field (\texttt{v1.scale\_inhibition}).", wait="read")

        if not self.at(3):
            self.outro()
            return

        # ------------------------------------------------------------ 4. measured
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self._cap], run_time=0.8)
        for m in (outline, zones, *pools):
            m.clear_updaters()
        hdr = Tex(r"measured on MICA's synthetic scenes (spec 4.6b)", color=GREY_B).scale(0.6).to_edge(UP, buff=0.5)
        tbl = tex_table([
            [r"synthetic scene", r"20 px past the end", r"grouped vessel"],
            [r"thin line, end at column 120", r"$15 \to 2$", r"ran to the border $\to$ stops at column 104"],
            [r"faint thin line", r"$5 \to 1$", r"ended at column 183 $\to$ 146"],
            [r"Y stem", r"$4 \to 3$", r"overshoot 34 px $\to$ 5 px"],
            [r"this toy: thin line, end at 120", rf"${g20:.1f} \to {g20b:.0f}$", rf"column {reach0} $\to$ {reach1}"],
        ], widths=[4.1, 2.6, 5.4], scale=0.52, row_buff=0.24)
        tbl.next_to(hdr, DOWN, buff=0.35)
        tbl[0][-1].set_color(GREY_A)
        self.play(FadeIn(hdr), FadeIn(tbl[1]), FadeIn(tbl[0][0]))
        self.say(r"Measured in MICA: the coarse tail drops to a trace, and grouped lines stop near their ends.")
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in tbl[0][1:4]], lag_ratio=0.5), run_time=2.5)
        self.wait(1.5)
        self.play(FadeIn(tbl[0][4], shift=RIGHT * 0.2))
        self.say(r"The toy row is this animation's own line, computed above; it agrees in kind, not in the numbers.", wait="read")
        extra = VGroup(
            Tex(r"T junction: the false second junction cell is gone; a crossing is unchanged.").scale(0.5),
            Tex(r"real test image: centreline pieces $755 \to 679$, junction cells $444 \to 425$,").scale(0.5),
            Tex(r"grouped vessels $557 \to 544$; the fine channel speaks for 96\,\% of tracked cells,").scale(0.5),
            Tex(r"the coarse one alone for 0.8\,\% (real wide vessels, median OD 0.49)").scale(0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(tbl, DOWN, buff=0.5).align_to(tbl, LEFT)
        self.play(FadeIn(extra[0]))
        self.wait(1.5)
        self.say(r"On the real test image, fewer, longer pieces: the coarse channel now speaks only for truly wide vessels.")
        self.play(FadeIn(extra[1:], lag_ratio=0.4), run_time=1.6)
        self.wait(2.5)

        # ------------------------------------------------------------ 5. the cost: a taper
        self.play(FadeOut(Group(hdr, tbl, extra)), run_time=0.8)
        tap = S.compute("taper")
        Gt, G2t = tap["G"][R], tap["G2"][R]
        img_t = od_image("taper")
        border_t = SurroundingRectangle(img_t, buff=0, color=MUTED, stroke_width=1)
        ax_t, ax_t_lbls = make_axes(ymax=140, step=40)
        self.say(r"The cost: where a wide vessel tapers into a thin one, the thin part's fine cells sit in the wide cell's end zone.")
        self.play(FadeIn(img_t), Create(border_t), FadeIn(od_lbl), Create(ax_t), FadeIn(ax_t_lbls))
        before = prof_curve(ax_t, Gt, 20, 220, COARSE_COL, dashed=True, opacity=0.9)
        after = prof_curve(ax_t, G2t, 20, 220, WHITE, width=4)
        k1 = Tex(r"coarse, before", color=COARSE_COL).scale(0.5)
        k2 = Tex(r"coarse, after the veto", color=WHITE).scale(0.5)
        kk = VGroup(k1, k2).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(ax_t.c2p(172, 128), aligned_edge=LEFT)
        self.play(Create(before), FadeIn(k1), run_time=1.5)
        self.play(TransformFromCopy(before, after), FadeIn(k2), run_time=1.5)
        xd = 100
        dip = Arrow(ax_t.c2p(xd + 0.5, Gt[xd]) + UP * 0.05, ax_t.c2p(xd + 0.5, G2t[xd]), buff=0.02, color=INH, stroke_width=3,
                    max_tip_length_to_length_ratio=0.3)
        dip_lbl = Tex(rf"toy: {Gt[xd]:.0f} $\to$ {G2t[xd]:.0f}, 20 px before the taper", color=INH).scale(0.5)
        dip_lbl.move_to(ax_t.c2p(140, 98), aligned_edge=LEFT)
        self.play(GrowArrow(dip), FadeIn(dip_lbl))
        meas_t = VGroup(Tex(r"measured (spec 4.6b): $59 \to 33$ near", color=GREY_B).scale(0.5),
                        Tex(r"the taper, and a false Y there", color=GREY_B).scale(0.5)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        meas_t.next_to(dip_lbl, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeIn(meas_t))
        self.wait(2.5)

        # ------------------------------------------------------------ 6. the rejected first version
        self.play(FadeOut(Group(img_t, border_t, ax_t, ax_t_lbls, before, after, kk, dip, dip_lbl, meas_t)), run_time=0.8)
        wp = S.plain_weight_needed(20)
        plain = np.maximum(G - wp * thin["axis_mean"][R], 0.0)
        img2 = od_image("thin_end")
        ax2, ax2_lbls = make_axes()
        self.say(r"Why the centre pool? The first version let fine cells inhibit coarse ones all along the axis (spec 8).")
        self.play(FadeIn(img2), Create(border_t), Create(ax2), FadeIn(ax2_lbls), FadeIn(end_line))
        c0 = prof_curve(ax2, G, 20, 220, COARSE_COL, dashed=True, opacity=0.9)
        c1 = prof_curve(ax2, G2, 20, 220, WHITE, width=4)
        c2 = prof_curve(ax2, plain, 20, 220, INH, width=4)
        k0 = Tex(r"coarse, before", color=COARSE_COL).scale(0.5)
        k1 = Tex(r"end zones $-$ centre pool", color=WHITE).scale(0.5)
        k2 = Tex(r"plain: fine inhibits coarse along the axis", color=INH).scale(0.5)
        kk = VGroup(k0, k1, k2).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(ax2.c2p(150, 44), aligned_edge=LEFT)
        self.play(Create(c0), FadeIn(k0), run_time=1.0)
        self.play(Create(c1), FadeIn(k1), run_time=1.0)
        self.play(Create(c2), FadeIn(k2), run_time=1.2)
        cut = 100 * (1 - np.median(plain[30:100]) / np.median(G[30:100]))
        cut_lbl = VGroup(Tex(r"toy: strong enough to silence the tail,", color=INH).scale(0.5),
                         Tex(rf"it cuts the line itself by {cut:.0f}\,\%", color=INH).scale(0.5)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        cut_meas = VGroup(Tex(r"measured (spec 8): legitimate overlaps", color=GREY_B).scale(0.48),
                          Tex(r"cut by a third to a half", color=GREY_B).scale(0.48),
                          Tex(r"(a T's parent $53\to36$, a taper $59\to31$)", color=GREY_B).scale(0.48)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        VGroup(cut_lbl, cut_meas).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(ax2.c2p(146, 24), aligned_edge=LEFT)
        self.say(r"Strong enough to stop the overshoot, it also cuts the coarse cell where both channels rightly respond.")
        self.play(FadeIn(cut_lbl))
        self.play(FadeIn(cut_meas))
        self.wait(2.5)
        self.say(r"Biologically: a competition between spatial-frequency channels for the same contour. No specific circuit is claimed.", wait="read")
        self.outro()


variants(ScaleSelection, module_globals=globals())
