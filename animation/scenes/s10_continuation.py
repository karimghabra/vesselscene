"""Chapter 10: good continuation is co-circularity (spec 4.4, 4.10b).

Two vessel ends face each other across a gap: one vessel or two?  The first
rule scored the turn between their directions and marked smooth bends down;
co-circularity asks whether one circle passes through both ends, tangent to
both, which holds exactly when phi_a + phi_b = 0.

Render:  manim -ql scenes/s10_continuation.py ContinuationShort
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *
from manim.constants import RESAMPLING_ALGORITHMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import continuation as C  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (CONT, END, FB, INH, JUNC, MUTED, OD_COL, PixelFrame, image,  # noqa: E402
                             orientation_color)

PHI_A = TEAL_B             # phi_a: its wedge and its symbol
PHI_B = PURPLE_A           # phi_b


def v3(v):
    return np.array([v[0], v[1], 0.0])


def score_color(v):
    """Red (0) through amber to green (1)."""
    v = float(np.clip(v, 0, 1))
    if v < 0.5:
        return interpolate_color(INH, GOLD_C, v / 0.5)
    return interpolate_color(GOLD_C, CONT, (v - 0.5) / 0.5)


def heading(p, h, length, color=END, width=5, dashed=False):
    a = Arrow(p, p + v3(h) * length, buff=0, color=color, stroke_width=width, tip_length=0.16,
              max_tip_length_to_length_ratio=0.35, max_stroke_width_to_length_ratio=20)
    if dashed:
        line = DashedLine(p, p + v3(h) * (length - 0.12), dash_length=0.06, color=color, stroke_width=width * 0.7)
        tip = a.tip.copy()
        a = VGroup(line, tip)
    a.set_stroke(BLACK, width=width + 4, background=True)
    return a


def wedge(centre, v_from, v_to, radius, color, fill=0.35, width=2.5):
    a0 = np.arctan2(v_from[1], v_from[0])
    sweep = C.angle(v_from, v_to)
    if abs(sweep) < 1e-4:
        return VGroup()
    sec = AnnularSector(inner_radius=0, outer_radius=radius, angle=sweep, start_angle=a0, fill_color=color,
                        fill_opacity=fill, stroke_width=0, arc_center=centre)
    arc = Arc(radius=radius, start_angle=a0, angle=sweep, arc_center=centre, stroke_color=color, stroke_width=width)
    return VGroup(sec, arc)


def clipped_circle(P, centre, r, rect, color, width, n=400):
    """The circle (math frame) mapped to the scene, keeping the longest run
    of points inside rect."""
    th = np.linspace(-np.pi / 2, 1.5 * np.pi, n)
    pts = [P(np.asarray(centre) + r * np.array([np.cos(t), np.sin(t)])) for t in th]
    x0, x1 = rect.get_left()[0], rect.get_right()[0]
    y0, y1 = rect.get_bottom()[1], rect.get_top()[1]
    inside = [x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in pts]
    best, cur = [], []
    for p, ok in zip(pts, inside):
        if ok:
            cur.append(p)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    m = VMobject(stroke_color=color, stroke_width=width)
    if len(best) > 2:
        m.set_points_smoothly(best[::2] if len(best) > 40 else best)
    return m


class Meter(VGroup):
    """A score bar 0..1 with its value."""

    def __init__(self, label, value, color=None, width=3.0, height=0.2, label_scale=0.55):
        super().__init__()
        self.w, self.h = width, height
        self.track = Rectangle(width=width, height=height, stroke_color=MUTED, stroke_width=1.2)
        self.lbl = Tex(label).scale(label_scale).next_to(self.track, UP, buff=0.12, aligned_edge=LEFT)
        self.add(self.track, self.lbl)
        self.fixed = color
        self.set_value(value)

    def bar(self, value):
        v = float(np.clip(value, 0, 1))
        col = self.fixed if self.fixed is not None else score_color(v)
        r = Rectangle(width=max(self.w * v, 1e-3), height=self.h, stroke_width=0, fill_color=col, fill_opacity=0.9)
        r.move_to(self.track.get_left(), aligned_edge=LEFT)
        num = MathTex(f"{v:.2f}", color=col).scale(0.6).next_to(self.track, RIGHT, buff=0.15)
        return VGroup(r, num)

    def set_value(self, value):
        if hasattr(self, "fill"):
            self.remove(self.fill)
        self.fill = self.bar(value)
        self.add(self.fill)
        return self


def mini_junction(arms, R, pair_list, centre, scale, colors=None, arm_len=1.6):
    """A schematic junction: disc of radius R (px), arm stubs, and the
    co-circular arc of each listed pair inside the disc."""
    g = VGroup()
    c = np.asarray(centre, float)
    disc = DashedVMobject(Circle(radius=R * scale, stroke_color=MUTED, stroke_width=1.5), num_dashes=24).move_to(c)
    g.add(disc)
    for k, deg in enumerate(arms):
        u = v3(C.unit(deg))
        col = colors[k] if colors else WHITE
        g.add(Line(c + u * R * scale, c + u * (R + arm_len / scale) * scale, stroke_color=col, stroke_width=5))
    for (i, j), col in pair_list:
        ui, uj = C.unit(arms[i]), C.unit(arms[j])
        a, b = R * ui, R * uj
        kap, sb, _ = C.circle_through(a, -ui, b)
        pts = C.arc(a, -ui, kap, np.linspace(0, sb, 40))
        m = VMobject(stroke_color=col, stroke_width=4)
        m.set_points_smoothly([c + v3(p) * scale for p in pts])
        g.add(m)
    return g


class Continuation(ChapterScene):
    NUMBER, TITLE, SPEC = 10, "Good continuation is co-circularity", "spec 4.4, 4.10b"
    MAP_KEYS = ("contin",)

    def construct(self):
        self.intro()
        full, od_full, g = C.bend_toy()
        toy, od, _ = C.bend_crop()
        pf = PixelFrame(toy.shape, 4.4, LEFT * 3.0 + UP * 1.0)          # the zoomed main view
        P = lambda p: toy.scene(pf, p)                      # math frame (px, y up) -> scene
        u = pf.unit
        a, b, ha, hb = g["a"], g["b"], g["ha"], g["hb"]
        A, B = P(a), P(b)

        # ---------------------------------------------------------- 1. the puzzle
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.2).move_to(pf.centre)
        border = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        lbl = Tex(r"toy OD: a vessel bent at $r = 30$ px, cut by a 12 px gap", color=GREY_B).scale(0.5)
        lbl.next_to(border, DOWN, buff=0.12)
        tpf = PixelFrame(full.shape, 1.45, RIGHT * 4.3 + UP * 3.0)          # thumbnail: the whole bend
        thumb = image(od_full, tpf.height, "magma", vmin=-0.02, vmax=0.2).move_to(tpf.centre)
        tb_border = SurroundingRectangle(thumb, buff=0, color=MUTED, stroke_width=1)
        zoom = Rectangle(width=toy.shape[1] * tpf.unit, height=toy.shape[0] * tpf.unit, stroke_color=WHITE,
                         stroke_width=1.5).move_to(tpf.point(15 + (toy.shape[1] - 1) / 2, (toy.shape[0] - 1) / 2))
        links = VGroup(Line(zoom.get_corner(UL), border.get_corner(UR), stroke_color=WHITE, stroke_width=0.8, stroke_opacity=0.35),
                       Line(zoom.get_corner(DL), border.get_corner(DR), stroke_color=WHITE, stroke_width=0.8, stroke_opacity=0.35))
        self.play(FadeIn(thumb), Create(tb_border), run_time=0.8)
        self.play(Create(zoom), Create(links), FadeIn(img), Create(border), FadeIn(lbl), run_time=1.0)
        self.say(r"Two vessel ends face each other across a 12 px gap. Are they one vessel, or two?", wait="read")

        dots = VGroup(Dot(A, radius=0.07, color=END, z_index=4), Dot(B, radius=0.07, color=END, z_index=4))
        L_h = 1.0
        arr_a, arr_b = heading(A, ha, L_h), heading(B, hb, L_h)
        ta = MathTex("a", color=END).scale(0.7).move_to(A + DOWN * 1.05)
        tb = MathTex("b", color=END).scale(0.7).move_to(B + DOWN * 1.05)
        for t in (ta, tb):
            t.set_stroke(BLACK, width=5, background=True)
        self.play(FadeIn(dots, scale=2), GrowArrow(arr_a), GrowArrow(arr_b), FadeIn(ta), FadeIn(tb), FadeOut(links), run_time=1.0)

        # ---------------------------------------------------------- 2. the straight rule
        px1 = RIGHT * 4.3
        h1 = Tex(r"straight rule: how far does it turn?").scale(0.58).move_to(px1 + UP * 1.75)
        h2 = Tex(r"co-circular: one circle through both?").scale(0.58).move_to(px1 + DOWN * 0.15)
        m1 = Meter(r"$\exp\!\big(-\mathrm{turn}^2 / 2\cdot 20^\circ{}^2\big)$" if self.at(2) else r"score", 0.0, color=INH)
        m1.next_to(h1, DOWN, buff=0.25)
        ext = DashedLine(A, A + v3(ha) * (12 + 5) * u, dash_length=0.09, color=WHITE, stroke_width=2.5)
        exp_b = heading(B, -ha, L_h, color=WHITE, width=4, dashed=True)
        turn_w = wedge(B, -ha, hb, 0.95, INH, fill=0.55)
        turn_l = MathTex(r"23^\circ", color=INH).scale(0.6).move_to(B + LEFT * 1.25 + DOWN * 0.32)
        self.say(r"The first rule continues one end in a straight line, and asks how far the other end turns from it: $23^\circ$.")
        self.play(Create(ext), run_time=1.0)
        self.play(FadeIn(exp_b), run_time=0.6)
        self.play(FadeIn(turn_w), FadeIn(turn_l), FadeIn(h1), run_time=0.8)
        s_straight = C.straight_score(ha, hb)
        self.play(FadeIn(m1.track), FadeIn(m1.lbl), run_time=0.5)
        m1v = ValueTracker(0.0)
        m1.remove(m1.fill)
        fill1 = always_redraw(lambda: m1.bar(m1v.get_value()))
        self.add(fill1)
        self.play(m1v.animate.set_value(s_straight), run_time=1.0)
        self.say(r"A perfectly smooth bend gets half marks. Real facing ends are typically $23^\circ$ apart (spec 4.4); "
                 r"this rule joined only 50--54 of $\sim$165 breaks (4.10b).", wait="read")

        # ---------------------------------------------------------- 3. co-circularity
        circle = clipped_circle(P, g["centre"], g["r"], border, CONT, 3.5)
        tcircle = clipped_circle(lambda p: full.scene(tpf, p), g["centre"], g["r"], tb_border, CONT, 2.0)
        chord = Line(A, B, stroke_color=WHITE, stroke_width=2)
        wa = wedge(A, b - a, ha, 0.95, PHI_A, fill=0.6)
        wb = wedge(B, a - b, hb, 0.95, PHI_B, fill=0.6)
        la = MathTex(r"\varphi_a", color=PHI_A).scale(0.62).move_to(A + RIGHT * 0.5 + UP * 0.52)
        lb = MathTex(r"\varphi_b", color=PHI_B).scale(0.62).move_to(B + LEFT * 0.5 + UP * 0.52)
        for t in (la, lb):
            t.set_stroke(BLACK, width=4, background=True)
        m2 = Meter(r"$\exp\!\big(-(\varphi_a+\varphi_b)^2 / 2\cdot 20^\circ{}^2\big)$" if self.at(2) else r"score", 0.0,
                   color=CONT)
        m2.next_to(h2, DOWN, buff=0.25)
        self.say(r"Good continuation is \emph{co-circularity}: is there one circle through both ends, tangent to both?")
        self.play(FadeOut(VGroup(ext, exp_b, turn_w, turn_l)), m1.lbl.animate.set_opacity(0.5),
                  h1.animate.set_opacity(0.5), run_time=0.6)
        self.play(Create(circle), Create(tcircle), run_time=2.2, rate_func=smooth)
        self.play(Create(chord), FadeIn(wa), FadeIn(wb), FadeIn(la), FadeIn(lb), FadeIn(h2), run_time=1.0)
        sum_t = MathTex(r"\varphi_a", "+", r"\varphi_b", "=", r"0^\circ").scale(0.65)
        sum_t[0].set_color(PHI_A); sum_t[2].set_color(PHI_B); sum_t[4].set_color(CONT)
        sum_t.next_to(m2, DOWN, buff=0.3).align_to(m2.track, LEFT)
        self.play(FadeIn(m2.track), FadeIn(m2.lbl), run_time=0.5)
        m2v = ValueTracker(0.0)
        m2.remove(m2.fill)
        fill2 = always_redraw(lambda: m2.bar(m2v.get_value()))
        self.add(fill2)
        s_cc = C.cocircular_score(a, ha, b, hb)
        self.play(m2v.animate.set_value(s_cc), Write(sum_t), run_time=1.0)
        self.say(r"There is one. Its angles to the chord are equal and opposite, "
                 r"$\varphi_a + \varphi_b = 0$: full marks.", wait="read")
        if self.short:
            self.outro()
            return

        # ================================================================ level 2: the geometry
        fill1.clear_updaters(); fill2.clear_updaters()
        self.say(r"Contour integration follows good continuation (Field, Hayes and Hess 1993); "
                 r"for curves, co-circularity (Parent and Zucker 1989).")
        self.play(FadeOut(Group(img, border, lbl, h1, h2, m1.track, m1.lbl, fill1, m2.track, m2.lbl, fill2, sum_t,
                                 thumb, tb_border, zoom, tcircle)), run_time=0.8)
        self.wait(2.6)
        self.say(r"Name the parts: the chord $c$ between the ends, and each heading's angle to it.")

        # the equations, term by term, on the right
        e1 = MathTex(r"c", "=", "b", "-", "a", r",\quad d = |c|").scale(0.7)
        e1[2].set_color(END); e1[4].set_color(END)
        e2 = MathTex(r"\varphi_a", "=", r"\angle(c,\,", "h_a", r")", r",\quad", r"\varphi_b", "=", r"\angle(-c,\,", "h_b", ")").scale(0.7)
        e2[0].set_color(PHI_A); e2[6].set_color(PHI_B); e2[3].set_color(END); e2[9].set_color(END)
        e3 = MathTex(r"\text{one circle}", r"\iff", r"\varphi_a", "+", r"\varphi_b", "=", "0").scale(0.7)
        e3[0].set_color(CONT); e3[2].set_color(PHI_A); e3[4].set_color(PHI_B)
        e4 = MathTex(r"\mathrm{turn} = |", r"\varphi_a", "-", r"\varphi_b", r"|,\quad", r"r", "=",
                     r"\frac{d}{2\sin(\mathrm{turn}/2)}").scale(0.7)
        e4[1].set_color(PHI_A); e4[3].set_color(PHI_B); e4[5].set_color(CONT)
        eqs = VGroup(e1, e2, e3, e4).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        if eqs.width > 5.3:
            eqs.scale_to_fit_width(5.3)
        eqs.move_to(RIGHT * 4.0 + UP * 1.75)
        hc = Tex(r"$h_a, h_b$: headings, out of each vessel into the gap", color=GREY_B).scale(0.48)
        hc.next_to(eqs, DOWN, buff=0.38).align_to(eqs, LEFT)
        c_lbl = MathTex("c").scale(0.55).next_to(chord, DOWN, buff=0.08)
        self.play(Write(e1), FadeIn(c_lbl), Indicate(chord, color=WHITE), run_time=1.2)
        self.play(Write(e2), FadeIn(hc), Indicate(wa, color=PHI_A), Indicate(wb, color=PHI_B), run_time=1.6)
        self.play(Write(e3), Indicate(circle, color=CONT, scale_factor=1.02), run_time=1.2)
        self.say(r"Straight: both angles are 0. A bend: equal and opposite. The turn and the radius follow from them.")
        self.play(Write(e4), run_time=1.4)
        geo = C.cocircular(a, ha, b, hb)
        chk = MathTex(rf"\text{{toy check: }} d = {geo['d']:.1f}\text{{ px}},\ "
                      rf"\mathrm{{turn}} = {np.rad2deg(geo['turn']):.1f}^\circ,\ r = {geo['r']:.1f}\text{{ px}}",
                      color=GREY_A).scale(0.55)
        if chk.width > 5.3:
            chk.scale_to_fit_width(5.3)
        chk.next_to(hc, DOWN, buff=0.3).align_to(eqs, LEFT)
        self.play(FadeIn(chk, shift=UP * 0.1))
        self.wait(1.6)

        # ---------------------------------------------------------- the score, live
        self.play(FadeOut(VGroup(e1, e2, e4, hc, chk, c_lbl)), e3.animate.move_to(RIGHT * 4.0 + UP * 2.85), run_time=0.8)
        e5 = MathTex(r"\mathrm{cont}", "=", r"e^{-(\varphi_a+\varphi_b)^2/2\sigma^2}", r"\cdot", r"p(r)", r"\cdot",
                     r"e^{-\ln^2(w_a/w_b)/2\ln^2 1.5}").scale(0.62)
        e5[0].set_color(CONT); e5[4].set_color(FB)
        e5.next_to(e3, DOWN, buff=0.35)
        if e5.width > 5.6:
            e5.scale_to_fit_width(5.6)
        e5.next_to(e3, DOWN, buff=0.35)
        e5n = Tex(r"$\sigma = 20^\circ$; $p(r)$: a soft penalty for $r <$ 15 px\\"
                  r"(the tightest arc cell); the width term as before", color=GREY_B).scale(0.46).next_to(e5, DOWN, buff=0.18)
        self.say(r"The score keeps the $20^\circ$ tolerance, now on the asymmetry $\varphi_a + \varphi_b$, "
                 r"and a soft penalty for circles tighter than 15 px.")
        self.play(Write(e5), FadeIn(e5n), run_time=1.6)
        self.wait(1.0)

        # live geometry: a stub that turns with h_a
        rot = ValueTracker(0.0)

        def h_a():
            return C.rot(ha, rot.get_value())

        def piece(k):
            m = ImageMobject(C.piece_rgba(toy, k))
            m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            m.height = pf.height
            return m.move_to(pf.centre)

        piece_a, stub_b = piece(0), piece(1)
        base_a = piece_a.points.copy()

        def turn_piece(m):                                       # the vessel behind a turns with its heading
            m.points = (base_a - A) @ rotation_matrix(np.deg2rad(rot.get_value()), OUT).T + A

        def live_circle():
            kap, sb, _ = C.circle_through(a, h_a(), b)
            pts = C.arc(a, h_a(), kap, np.linspace(-10, sb + 10, 80))
            m = VMobject(stroke_color=CONT, stroke_width=3.5)
            m.set_points_smoothly([P(p) for p in pts])
            return m

        def need_arrow():
            _, _, nb = C.circle_through(a, h_a(), b)
            return heading(B, nb, L_h, color=CONT, width=4, dashed=True)

        def mismatch():
            _, _, nb = C.circle_through(a, h_a(), b)
            s = C.cocircular(a, h_a(), b, hb)["sum"]
            return wedge(B, nb, hb, 0.95, score_color(C.turn_term(s)), fill=0.6)

        live_arr_a = always_redraw(lambda: heading(A, h_a(), L_h))
        live_wa = always_redraw(lambda: wedge(A, b - a, h_a(), 0.95, PHI_A, fill=0.6))
        live_c = always_redraw(live_circle)
        live_need = always_redraw(need_arrow)
        live_mis = always_redraw(mismatch)

        def readout():
            gg = C.cocircular(a, h_a(), b, hb)
            s = C.cocircular_score(a, h_a(), b, hb)
            r1 = MathTex(r"\varphi_a", rf"= {np.rad2deg(gg['phi_a']):+.1f}^\circ").scale(0.62)
            r1[0].set_color(PHI_A)
            r2 = MathTex(r"\varphi_b", rf"= {np.rad2deg(gg['phi_b']):+.1f}^\circ").scale(0.62)
            r2[0].set_color(PHI_B)
            r3 = MathTex(r"\varphi_a + \varphi_b", rf"= {np.rad2deg(gg['sum']):+.1f}^\circ").scale(0.62)
            r3[1].set_color(score_color(C.turn_term(gg["sum"])))
            col = VGroup(r1, r2, r3).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            mt = Meter(r"cont", s, width=2.1, label_scale=0.55)
            out = VGroup(col, mt).arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
            return out.move_to(RIGHT * 3.85 + DOWN * 1.35)

        live_read = always_redraw(readout)
        self.play(FadeOut(VGroup(arr_a, wa, la, circle)), FadeIn(stub_b), FadeIn(piece_a), run_time=0.8)
        self.bring_to_back(stub_b, piece_a)
        piece_a.add_updater(turn_piece)
        self.add(live_c, live_wa, live_arr_a, live_need, live_mis)
        self.play(FadeIn(live_read), run_time=0.6)
        self.say(r"Turn one end. The circle tangent at $a$ through $b$ now arrives off $b$'s heading, "
                 r"by exactly $\varphi_a + \varphi_b$.")
        self.play(rot.animate.set_value(22.0), run_time=2.6, rate_func=smooth)
        self.play(rot.animate.set_value(-24.0), run_time=3.2, rate_func=smooth)
        self.say(r"The score falls with that mismatch, and is 1 again only where one circle fits both ends.")
        self.play(rot.animate.set_value(0.0), run_time=2.4, rate_func=smooth)
        self.wait(1.0)
        live = [live_c, live_wa, live_arr_a, live_need, live_mis, live_read]
        for m in live + [piece_a]:
            m.clear_updaters()
        self.play(FadeOut(Group(*live, piece_a, stub_b, dots, arr_b, wb, lb, ta, tb, chord, e3, e5, e5n)), run_time=0.8)

        # ---------------------------------------------------------- S-bend and kink
        self.cases()

        # ---------------------------------------------------------- arms at a junction
        self.pairing()

        if not self.at(3):
            self.outro()
            return

        # ================================================================ level 3
        self.heading_fit()
        self.widths()
        self.linking()
        self.outro()

    # ------------------------------------------------------------------ level 2: three cases
    def cases(self):
        kinds = [("bend", r"a bend, $r = 30$ px"), ("s", r"an S-bend"), ("kink", r"a kink")]
        cols = Group()
        xs = [-1.55, 1.65, 4.85]
        cells = []
        for (kind, name), x in zip(kinds, xs):
            toy, od, g = C.case_toy(kind)
            pf = PixelFrame(toy.shape, 2.25, RIGHT * x + UP * 2.35)
            P = lambda p, toy=toy, pf=pf: toy.scene(pf, p)
            img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.2).move_to(pf.centre)
            bd = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
            A, B = P(g["a"]), P(g["b"])
            kap, sb, nb = C.circle_through(g["a"], g["ha"], g["b"])
            pts = C.arc(g["a"], g["ha"], kap, np.linspace(-5, sb + 5, 50))
            circ = VMobject(stroke_color=CONT, stroke_width=2.5).set_points_smoothly([P(p) for p in pts])
            arr = VGroup(*[Arrow(p, p + v3(h) * 0.3, buff=0, color=END, stroke_width=3, tip_length=0.1,
                                 max_tip_length_to_length_ratio=0.4, max_stroke_width_to_length_ratio=20)
                           for p, h in ((A, g["ha"]), (B, g["hb"]))])
            arr.set_stroke(BLACK, width=6, background=True)
            sc = C.case_scores(kind)
            mis = wedge(B, nb, g["hb"], 0.36, score_color(C.turn_term(np.deg2rad(sc["sum"]))), fill=0.6)
            ov = VGroup(circ, mis, arr)
            t = Tex(name).scale(0.55).next_to(bd, UP, buff=0.1)
            cols.add(Group(img, bd, ov, t))
            cells.append(sc)
        rows = [("one circle through both?", None), (r"straight, $20^\circ$", "straight20"), (r"straight, $40^\circ$", "straight40"),
                ("co-circular", "cocirc")]
        ys = [0.3, -0.38, -1.03, -1.68]
        table = VGroup()
        heads = VGroup()
        for (name, key), y in zip(rows, ys):
            heads.add(Tex(name).scale(0.55).move_to([-3.35, y, 0], aligned_edge=RIGHT))
            row = VGroup()
            for k, (sc, x) in enumerate(zip(cells, xs)):
                if key is None:
                    m = MathTex(r"\checkmark" if k == 0 else r"\times", color=CONT if k == 0 else INH).scale(0.75)
                else:
                    v = sc[key]
                    m = MathTex(f"{v:.2f}", color=score_color(v)).scale(0.7)
                row.add(m.move_to([x, y, 0]))
            table.add(row)
        geo = VGroup(*[MathTex(rf"\mathrm{{turn}}\ {sc['turn']:.0f}^\circ,\ \ \varphi_a{{+}}\varphi_b\ {abs(sc['sum']):.0f}^\circ",
                               color=GREY_B).scale(0.48).move_to([x, 0.93, 0]) for sc, x in zip(cells, xs)])
        self.say(r"Three pairs of ends. The bend: a turn but no asymmetry. The S-bend: no turn, both tilted the same way. "
                 r"The kink: both.")
        self.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.25), run_time=1.8)
        self.play(FadeIn(geo), run_time=0.8)
        self.wait(2.0)
        self.play(FadeIn(heads[0]), FadeIn(table[0]), run_time=0.6)
        self.say(r"The straight rule marks the bend down and passes the S-bend. Widened to $40^\circ$, it passes the kink too.")
        self.play(FadeIn(heads[1]), FadeIn(table[1]), run_time=0.8)
        self.play(FadeIn(heads[2]), FadeIn(table[2]), run_time=0.8)
        self.wait(1.4)
        self.say(r"Co-circularity passes only the bend. Right for the kink; for a real S-bend it is a risk the spec leaves open (section 10).")
        self.play(FadeIn(heads[3]), FadeIn(table[3]), run_time=0.8)
        box = SurroundingRectangle(VGroup(heads[3], table[3]), color=CONT, buff=0.12, corner_radius=0.08)
        self.play(Create(box), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(Group(cols, geo, heads, table, box)), run_time=0.8)

    # ------------------------------------------------------------------ level 2: pairing arms at a junction
    def pairing(self):
        for kind in ("crossing", "fork"):
            J = C.junction_toy(kind)
            arms, widths = J["arms"], J["widths"]
            R = 4.0 + max(widths) / 2                  # the junction disc, r0 + w / 2 (spec 4.3)
            pf = PixelFrame(J["shape"], 3.6, LEFT * 4.75 + UP * 0.7)
            img = image(J["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
            bd = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
            ctr = pf.point(*J["origin"])
            disc = DashedVMobject(Circle(radius=R * pf.unit, stroke_color=WHITE, stroke_width=1.5), num_dashes=20).move_to(ctr)
            cols = [orientation_color(np.deg2rad(-d)) for d in arms]
            nums = VGroup(*[MathTex(str(k + 1), color=cols[k]).scale(0.55).move_to(ctr + v3(C.unit(d)) * 1.5)
                            for k, d in enumerate(arms)])
            for n_ in nums:
                n_.set_stroke(BLACK, width=4, background=True)
            title = Tex("a crossing: 4 arms" if kind == "crossing" else "a fork: 3 arms").scale(0.6).next_to(bd, UP, buff=0.1)
            pairs = C.junction_pairs(arms, widths, R)
            grp = Group(img, bd, disc, nums, title)
            if kind == "crossing":
                self.say(r"At a junction the same score pairs arms. Four arms can pair into two lines in three ways.")
            else:
                self.say(r"At a fork, the pair that continues best is the parent; the third arm is the branch.")
            self.play(FadeIn(img), Create(bd), FadeIn(title), run_time=0.8)
            self.play(Create(disc), FadeIn(nums), run_time=0.6)
            minis = VGroup()
            mx = [-0.7, 2.05, 4.8]
            my = 1.2
            if kind == "crossing":
                options = C.pairings4(pairs)
                for k, (pp, prod) in enumerate(options):
                    m = mini_junction(arms, R, [(pp[0], CONT), (pp[1], CONT)], RIGHT * mx[k] + UP * my,
                                      0.115, colors=cols, arm_len=0.5)
                    s1, s2 = pairs[pp[0]][0], pairs[pp[1]][0]
                    t = MathTex(rf"{s1:.2f}\times{s2:.2f}", "=", f"{prod:.2f}").scale(0.58)
                    t[2].set_color(score_color(prod))
                    lab = MathTex(rf"({pp[0][0] + 1}{pp[0][1] + 1})({pp[1][0] + 1}{pp[1][1] + 1})", color=GREY_B).scale(0.55)
                    minis.add(VGroup(m, lab.next_to(m, UP, buff=0.15), t.next_to(m, DOWN, buff=0.2)))
                best = int(np.argmax([p for _, p in options]))
                verdict = Tex(r"the largest product, both pairs above 0.3: a crossing", color=JUNC).scale(0.62)
            else:
                plist = sorted(pairs.items(), key=lambda kv: kv[0])
                for k, ((i, j), (sv, r, sm)) in enumerate(plist):
                    m = mini_junction(arms, R, [((i, j), CONT)], RIGHT * mx[k] + UP * my, 0.115,
                                      colors=cols, arm_len=0.5)
                    t = MathTex(r"\mathrm{cont} =", f"{sv:.2f}").scale(0.58)
                    t[1].set_color(score_color(sv))
                    lab = MathTex(rf"({i + 1}{j + 1})", color=GREY_B).scale(0.55)
                    minis.add(VGroup(m, lab.next_to(m, UP, buff=0.15), t.next_to(m, DOWN, buff=0.2)))
                best = int(np.argmax([sv for _, (sv, _, _) in plist]))
                bi, bj = plist[best][0]
                third = [k for k in range(3) if k not in (bi, bj)][0]
                verdict = Tex(rf"the parent: arms {bi + 1} and {bj + 1}; the branch: arm {third + 1}", color=JUNC).scale(0.62)
            self.play(LaggedStart(*[FadeIn(m) for m in minis], lag_ratio=0.3), run_time=1.8)
            self.wait(1.0 if kind == "crossing" else 0.6)
            hl = SurroundingRectangle(minis[best], color=JUNC, buff=0.12, corner_radius=0.08)
            verdict.move_to(RIGHT * 2.05 + DOWN * 0.95)
            self.play(Create(hl), FadeIn(verdict), *[m.animate.fade(0.65) for k, m in enumerate(minis) if k != best],
                      run_time=1.0)
            if kind == "crossing":
                note = Tex(r"Both arms aim at one centre, so every pair is co-circular ($\varphi_a + \varphi_b = 0$).\\"
                           r"The circle differs: $r = R\tan(\alpha/2)$, with $\alpha$ the angle between the arms,\\"
                           r"and $p(r)$ penalises the tight ones.", color=GREY_A).scale(0.52)
                note.move_to(RIGHT * 2.05 + DOWN * 2.0)
                self.say(r"Here every pair is co-circular: both arms aim at the centre. The circle differs, "
                         r"tight for a sharp turn, and $p(r)$ penalises it.")
                self.play(FadeIn(note), run_time=0.8)
                self.wait(2.6)
                grp.add(note)
            else:
                murray = None
                if self.at(3):
                    w = widths
                    murray = MathTex(r"\text{Murray, a diagnostic only: }", rf"{w[0]:.1f}^3 = {w[0] ** 3:.0f}", r"\ \text{vs}\ ",
                                     rf"{w[1]:.1f}^3 + {w[2]:.1f}^3 = {w[1] ** 3 + w[2] ** 3:.0f}", color=GREY_A).scale(0.55)
                    murray.move_to(RIGHT * 2.05 + DOWN * 1.75)
                    note2 = Tex(r"widths in px, toy fork; phase 2 reports it, the pairing does not use it",
                                color=GREY_B).scale(0.46).next_to(murray, DOWN, buff=0.15)
                    self.say(r"Murray's law, $r_p^3 \approx r_1^3 + r_2^3$, is reported beside the pairing as a check, "
                             r"not used to decide (spec 4.4).")
                    self.play(FadeIn(murray), FadeIn(note2), run_time=0.8)
                    self.wait(2.0)
                    grp.add(murray, note2)
                else:
                    self.wait(1.8)
            self.play(FadeOut(Group(grp, minis, hl, verdict)), run_time=0.8)

    # ------------------------------------------------------------------ level 3: headings
    def heading_fit(self):
        toy, od, g = C.bend_toy()
        pf = PixelFrame(toy.shape, 4.8, LEFT * 3.0 + UP * 0.75)
        P = lambda p: toy.scene(pf, p)
        a, ha = g["a"], g["ha"]
        img = image(od, pf.height, "magma", vmin=-0.02, vmax=0.2).move_to(pf.centre)
        bd = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
        self.say(r"Each end needs a heading. Averaging the direction over its last stretch lags on a curve.")
        self.play(FadeIn(img), Create(bd), run_time=0.8)
        # the last 4 widths (24 px) of vessel a, its centreline
        tail = C.end_piece(a, ha, 24.0, -1 / 30.0)
        tail_m = VMobject(stroke_color=WHITE, stroke_width=3).set_points_smoothly([P(p) for p in tail[::4]])
        mean_dir = (tail[-1] - tail[0]) / np.linalg.norm(tail[-1] - tail[0])
        lag = np.rad2deg(abs(C.angle(mean_dir, ha)))
        A = P(a)
        avg = heading(A, mean_dir, 1.3, color=INH, width=5)
        ctr, rad = C.fit_circle(tail)
        n = a - ctr
        t_fit = np.array([-n[1], n[0]]) / np.linalg.norm(n)
        if t_fit @ ha < 0:
            t_fit = -t_fit
        fit_err = np.rad2deg(abs(C.angle(t_fit, ha)))
        fit_c = DashedVMobject(clipped_circle(P, ctr, rad, bd, CONT, 2.5), num_dashes=50)
        fit_h = heading(A, t_fit, 1.3, color=CONT, width=5)
        txt = VGroup(
            Tex(r"the last 4 widths (24 px) of end $a$", color=WHITE).scale(0.55),
            Tex(rf"mean direction: lags by $L/2r = {lag:.0f}^\circ$", color=INH).scale(0.55),
            Tex(rf"tangent of a fitted circle: off by ${fit_err:.1f}^\circ$", color=CONT).scale(0.55),
            Tex(r"toy, on the exact centreline", color=GREY_B).scale(0.48),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(RIGHT * 4.3 + UP * 1.2)
        self.play(Create(tail_m), FadeIn(txt[0]), run_time=1.0)
        self.play(GrowArrow(avg), FadeIn(txt[1]), run_time=1.0)
        self.wait(1.0)
        self.say(r"So each heading is the tangent of a circle fitted to the end's last 1-4 widths (spec 4.4).")
        self.play(Create(fit_c), run_time=1.4)
        self.play(GrowArrow(fit_h), FadeIn(txt[2]), FadeIn(txt[3]), run_time=1.0)
        self.wait(2.2)
        self.play(FadeOut(Group(img, bd, tail_m, avg, fit_c, fit_h, txt)), run_time=0.8)

    # ------------------------------------------------------------------ level 3: widths, Murray
    def widths(self):
        prof = C.width_profiles()
        xs, p21, (hw, lvl, xl, xr) = prof[21.0]
        ax = Axes(x_range=[-24, 24, 8], y_range=[0, 0.16, 0.05], x_length=5.6, y_length=2.6,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=22),
                  x_axis_config=dict(numbers_to_include=[-16, 0, 16]),
                  y_axis_config=dict(numbers_to_include=[0.05, 0.1, 0.15], decimal_number_config=dict(num_decimal_places=2))
                  ).move_to(LEFT * 3.3 + UP * 0.9)
        keep = np.abs(xs) <= 24
        curve = ax.plot_line_graph(xs[keep], p21[keep], line_color=OD_COL, add_vertex_dots=False, stroke_width=4)
        half = DashedLine(ax.c2p(-24, lvl), ax.c2p(24, lvl), color=FB, dash_length=0.08, stroke_width=2)
        br = DoubleArrow(ax.c2p(xl, lvl), ax.c2p(xr, lvl), buff=0, color=FB, stroke_width=3, tip_length=0.12)
        xlab = Tex(r"across the vessel (px)").scale(0.5).next_to(ax.x_axis, DOWN, buff=0.35)
        ylab = Tex(r"OD").scale(0.5).next_to(ax.y_axis, UP, buff=0.1)
        hh = Tex(r"half height,\\against the local baseline", color=FB).scale(0.5).next_to(ax.c2p(13, lvl), UR, buff=0.1)
        hh.shift(RIGHT * 0.35)
        rows = VGroup(
            Tex(r"vessel", r"\quad 21 px", r"\quad 5 px").scale(0.55),
            Tex(r"half-height width", r"\quad 20", r"\quad 5.5").scale(0.55),
            Tex(r"winning filter scale", r"\quad 30", r"\quad 7.5").scale(0.55),
        )
        for r in rows:
            r[0].set_width(r[0].width)
        tab = VGroup(*rows).arrange(DOWN, aligned_edge=RIGHT, buff=0.18).move_to(RIGHT * 3.7 + UP * 1.35)
        rows[1].set_color(CONT); rows[2].set_color(INH)
        src = Tex(r"measured, spec 4.4 (synthetic vessels)", color=GREY_B).scale(0.46).next_to(tab, DOWN, buff=0.2)
        self.say(r"Widths: the OD's width at half height, not the winning filter's scale, which reads about 1.45 times too wide.")
        self.play(Create(ax), FadeIn(xlab), FadeIn(ylab), run_time=0.8)
        self.play(Create(curve), run_time=1.0)
        self.play(Create(half), GrowFromCenter(br), FadeIn(hh), run_time=0.8)
        self.play(FadeIn(tab), FadeIn(src), run_time=0.8)
        self.wait(1.6)
        wt = C.width_term(1.0, np.exp(0.36))
        geo = C.cocircular(*C.gap_pair("bend"))
        tt = C.turn_term(geo["turn"])
        cmp = VGroup(
            MathTex(r"\text{width term, median ratio } 1.4\times:", rf"\ {wt:.2f}").scale(0.56),
            MathTex(rf"\text{{straight turn term at }} {np.rad2deg(geo['turn']):.0f}^\circ:", rf"\ {tt:.2f}").scale(0.56),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.16).next_to(src, DOWN, buff=0.45).align_to(tab, RIGHT)
        cmp[0][1].set_color(score_color(wt)); cmp[1][1].set_color(score_color(tt))
        self.say(r"Facing ends of real vessels differ by about $1.4\times$ (spec 4.4): the width term is not what limits joining. "
                 r"The turn is.")
        self.play(FadeIn(cmp, shift=UP * 0.1), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(VGroup(ax, curve, half, br, xlab, ylab, hh, tab, src, cmp)), run_time=0.8)

    # ------------------------------------------------------------------ level 3: linking facing ends
    def linking(self):
        head = Tex(r"Linking facing label ends (spec 4.10b, measured on one real image)").scale(0.62).to_edge(UP, buff=0.4)
        facts = Tex(r"1,088 vessel ends; $\sim$333 face another roughly in line (median gap 12 px)\\"
                    r"$\Rightarrow$ $\sim$165 breaks inside one vessel (an upper bound)", color=GREY_A).scale(0.55)
        facts.next_to(head, DOWN, buff=0.25)
        data = [("straight rule, as built", 54, INH, r"50--54"),
                ("without the width term", 88, GOLD_C, "88"),
                (r"tolerance doubled to $40^\circ$", 157, GOLD_C, r"157, but kinks pass too"),
                ("co-circular", None, CONT, r"designed, not yet measured")]
        x0, wmax, y0 = -2.2, 6.0, 0.8
        bars = VGroup()
        ref = DashedLine([x0 + wmax * 165 / 170, y0 + 0.4, 0], [x0 + wmax * 165 / 170, y0 - 0.6 * 3 - 0.25, 0],
                         color=MUTED, dash_length=0.07)
        ref_l = Tex(r"$\sim$165", color=GREY_B).scale(0.5).next_to(ref, UP, buff=0.05)
        for k, (name, n, col, txt) in enumerate(data):
            y = y0 - 0.6 * k
            lab = Tex(name).scale(0.55).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
            if n is None:
                bar = DashedVMobject(Rectangle(width=wmax * 165 / 170, height=0.34, stroke_color=col), num_dashes=40)
            else:
                bar = Rectangle(width=wmax * n / 170, height=0.34, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([x0, y, 0], aligned_edge=LEFT)
            t = Tex(txt, color=col).scale(0.5)
            if n is not None and n < 120:
                t.next_to(bar, RIGHT, buff=0.12)
            else:
                t.move_to(bar).shift(RIGHT * 0.1)
                if n is not None:
                    t.set_color(BLACK)
            bars.add(VGroup(lab, bar, t))
        self.say(r"Linking the ends of grouped vessels (4.10b): about 165 breaks lie inside one vessel.")
        self.play(FadeIn(head), FadeIn(facts), run_time=0.8)
        self.play(Create(ref), FadeIn(ref_l), run_time=0.6)
        self.wait(1.4)
        self.say(r"The straight rule joined 50-54. Without the width term, 88; with the tolerance doubled, 157, "
                 r"but kinks and unrelated vessels pass as readily.")
        for k in range(3):
            self.play(FadeIn(bars[k][0]), GrowFromEdge(bars[k][1], LEFT), FadeIn(bars[k][2]), run_time=0.8)
        self.wait(1.6)
        self.say(r"Co-circularity replaces the turn term; it is designed, and the number is still to be measured.")
        self.play(FadeIn(bars[3]), run_time=0.8)
        self.wait(1.6)
        caveat = VGroup(
            Tex(r"Caveats, from the toy's geometry: a symmetric kink is co-circular too;", color=GREY_A),
            Tex(r"and at a junction (disc $R$ = 7 px, equal widths), a crossing shallower", color=GREY_A),
            Tex(r"than $50^\circ$ ties with a `bounce'. Both need each end's own curvature,", color=GREY_A),
            Tex(r"from the arc cells (4.8), which are not built yet.", color=GREY_A),
        ).scale(0.46).arrange(DOWN, buff=0.08).move_to(DOWN * 2.2)
        self.say(r"Open question (spec 10): co-circularity rejects an S-bend inside a gap. How often do real vessels inflect within 12 px?")
        self.play(FadeIn(caveat), run_time=0.8)
        self.wait(2.5)


variants(Continuation, module_globals=globals())
