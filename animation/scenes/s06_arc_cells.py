"""Chapter 6: bends, and arc cells (spec 4.8).

Long straight cells lose a bent vessel; short cells fit the bend but are weak
alone; pooling short cells along circles (arc cells) keeps the bend, and the
winning curvature is a population code for 1/r.  Everything is computed by
mica_anim/arcs.py on the spec's reference test (rings of 6 px vessels).

Render:  manim -ql scenes/s06_arc_cells.py ArcCellsDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import arcs as A  # noqa: E402
from mica_anim import neuro as N  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import EXC, INH, MUTED, PixelFrame, image, orientation_color  # noqa: E402
from mica_anim.widgets import receptive_field  # noqa: E402

KAP_COL = PURPLE_A                   # curvature, wherever it appears
LONG_COL = GREY_B
K_PROBE = 2                          # the probe's orientation: theta = 22.5 deg
R_PROBE = 15
ZOOM = (258, 53, 64)                 # crop of the reference image around the r = 15 ring: x0, y0, size


def probe_point(r=R_PROBE, k=K_PROBE):
    th = N.THETAS[k]
    phi = th - np.pi / 2             # the ring's tangent at angle phi is phi + 90 deg
    c = np.array(A.RINGS[r])
    return c + r * np.array([np.cos(phi), np.sin(phi)])


@lru_cache(maxsize=None)
def kappa_curve():
    """The arc cell at the probe, orientation K_PROBE, for curvatures -0.1..0.1, in the
    noise units of the straight arc cell."""
    ref = A.reference()
    kap = np.linspace(-0.1, 0.1, 101)
    resp = A.response_vs_kappa(ref["short"], probe_point(), N.THETAS[K_PROBE], kap)
    sd0 = A.arc_cell(ref["short"], K_PROBE, 0.0)[ref["bg"]].std()
    return kap, resp / sd0


def tex_table(rows, widths, scale=0.5, header_color=GREY_B, row_buff=0.2):
    lines = VGroup()
    for i, row in enumerate(rows):
        cells = VGroup(*[Tex(t, color=header_color if i == 0 else WHITE).scale(scale) for t in row])
        lines.add(cells)
    for row in lines:
        x = 0.0
        for c, w in zip(row, widths):
            c.move_to(np.array([x, 0, 0]), aligned_edge=LEFT)
            x += w
    lines.arrange(DOWN, aligned_edge=LEFT, buff=row_buff)
    for row in lines:
        x = lines.get_left()[0]
        y = row.get_y()
        for c, w in zip(row, widths):
            c.move_to(np.array([x, y, 0]), aligned_edge=LEFT)
            x += w
    rule = Line(LEFT, RIGHT, stroke_color=MUTED, stroke_width=1).set_width(sum(widths))
    rule.next_to(lines[0], DOWN, buff=row_buff / 2).align_to(lines, LEFT)
    return VGroup(lines, rule)


class ArcCells(ChapterScene):
    NUMBER, TITLE, SPEC = 6, "Bends: arc cells", "spec 4.8"
    MAP_KEYS = ("arc",)

    def construct(self):
        self.intro()
        ref = A.reference()
        tbl = A.table()

        # ------------------------------------------------------------ 1. rings, and what straight cells make of them
        pf = PixelFrame(A.SHAPE, 3.9, np.array([-2.25, 1.25, 0.0]))
        od = image(ref["od"], pf.height, "magma", vmin=-0.02, vmax=0.17).move_to(pf.centre)
        border = SurroundingRectangle(od, buff=0, color=MUTED, stroke_width=1)
        names = VGroup(*[Tex(rf"$r = {r}$ px").scale(0.5).move_to(pf.point(A.RINGS[r][0], A.SHAPE[0]) + DOWN * 0.25) for r in A.RADII],
                       Tex(r"straight").scale(0.5).move_to(pf.point(A.STRAIGHT[0][0], A.SHAPE[0]) + DOWN * 0.25))
        head = Tex(r"rings of 6 px vessels, in OD").scale(0.6).next_to(border, UP, buff=0.12)
        self.play(FadeIn(od), Create(border), FadeIn(head))
        self.play(FadeIn(names, lag_ratio=0.2))
        self.say(r"Vessels bend. The spec's reference test: rings of radius 60, 30 and 15 px, and a straight vessel.")

        long_map = image(np.clip(ref["long"], 0, None).max(axis=0), pf.height, "magma", vmin=30, vmax=88).move_to(pf.centre)
        head2 = Tex(r"straight cells, elongation 3: best orientation").scale(0.6).move_to(head)
        self.play(FadeIn(long_map), ReplacementTransform(head, head2), run_time=1.5)
        self.say(r"Through long straight cells, the tighter the ring, the dimmer it gets.", wait=1.0)

        # bars: % of the straight vessel's response
        base_y, bh = -1.6, 3.2
        xs = {60: 3.55, 30: 4.75, 15: 5.95}
        axis = Line(np.array([2.85, base_y, 0]), np.array([6.75, base_y, 0]), stroke_color=MUTED, stroke_width=1.5)
        hundred = DashedLine(np.array([2.85, base_y + bh, 0]), np.array([6.75, base_y + bh, 0]), stroke_color=MUTED, stroke_width=1)
        hlbl = Tex(r"100\,\% = straight", color=MUTED).scale(0.45).next_to(hundred, UP, buff=0.06).align_to(hundred, RIGHT)
        bars, vals, spec_bars = VGroup(), VGroup(), VGroup()
        spec_long = {60: 75, 30: 56, 15: 44}
        for i, r in enumerate(A.RADII):
            v = tbl["long"][1 + i]
            b = Rectangle(width=0.42, height=bh * v / 100, stroke_width=0, fill_color=WHITE, fill_opacity=0.85)
            b.move_to(np.array([xs[r] - 0.23, base_y, 0]), aligned_edge=DOWN)
            sb = Rectangle(width=0.42, height=bh * spec_long[r] / 100, stroke_width=0, fill_color=GREY_D, fill_opacity=1.0)
            sb.move_to(np.array([xs[r] + 0.23, base_y, 0]), aligned_edge=DOWN)
            bars.add(b); spec_bars.add(sb)
            vals.add(VGroup(Tex(f"{v:.0f}", color=WHITE).scale(0.45).next_to(b, UP, buff=0.06),
                            Tex(f"{spec_long[r]}", color=GREY_B).scale(0.45).next_to(sb, UP, buff=0.06)))
        rl = VGroup(*[Tex(rf"$r={r}$").scale(0.45).move_to(np.array([xs[r], base_y - 0.22, 0])) for r in A.RADII])
        key = VGroup(Tex(r"this toy", color=WHITE).scale(0.45), Tex(r"measured (spec 4.8)", color=GREY_B).scale(0.45)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(np.array([2.95, base_y + bh + 0.55, 0]), aligned_edge=LEFT)
        title_b = Tex(r"response on the ring, \% of straight").scale(0.5).move_to(np.array([4.8, base_y + bh + 1.15, 0]))
        self.play(Create(axis), Create(hundred), FadeIn(hlbl), FadeIn(rl), FadeIn(title_b))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.3), FadeIn(key[0]), run_time=1.5)
        self.play(FadeIn(VGroup(*[v[0] for v in vals])))
        if self.at(3):
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in spec_bars], lag_ratio=0.3), FadeIn(key[1]),
                      FadeIn(VGroup(*[v[1] for v in vals])), run_time=1.5)
            self.say(r"Measured in MICA the loss is larger than in this toy: 75, 56 and 44\,\% of a straight vessel.", wait="read")
        self.say(r"And bends cost twice: the association field's straight lobes then take another 19--26\,\% (spec 4.8). Tracks break at bends.",
                 wait="read" if self.at(2) else 1.5)

        # ------------------------------------------------------------ 2. zoom into the tightest ring
        x0, y0, sz = ZOOM
        zf = PixelFrame((sz, sz), 5.2, np.array([-3.55, 0.3, 0.0]))

        def zp(x, y):
            return zf.point(x - x0, y - y0)

        crop = ref["od"][y0:y0 + sz, x0:x0 + sz]
        zim = image(crop, zf.height, "magma", vmin=-0.02, vmax=0.17).move_to(zf.centre)
        zbd = SurroundingRectangle(zim, buff=0, color=MUTED, stroke_width=1)
        src = Rectangle(width=sz * pf.unit, height=sz * pf.unit, stroke_color=WHITE, stroke_width=1.5)
        src.move_to(pf.point(x0 + sz / 2 - 0.5, y0 + sz / 2 - 0.5))
        self.play(Create(src), run_time=0.6)
        others = Group(od, long_map, border, names, head2, axis, hundred, hlbl, rl, title_b, bars, key[0],
                       *[v[0] for v in vals], *([spec_bars, key[1], *[v[1] for v in vals]] if self.at(3) else []))
        self.play(FadeOut(others), ReplacementTransform(src, zbd), FadeIn(zim), run_time=1.5)
        zhead = Tex(r"the $r = 15$ px ring, in OD").scale(0.6).next_to(zbd, UP, buff=0.12)
        self.play(FadeIn(zhead))

        p = probe_point()
        th = N.THETAS[K_PROBE]
        P = zp(*p)
        dot = Dot(P, radius=0.06, color=YELLOW, z_index=6)
        long_rf = receptive_field(length=zf.unit * 4 * A.SIGMA * A.E_LONG, sigma=zf.unit * A.SIGMA,
                                  theta_scene=zf.screen_angle(th)).move_to(P)
        long_rf.set_z_index(4)
        self.play(FadeIn(dot, scale=2))
        self.say(r"A long straight cell lies along the bend at one point. The vessel curves away under both its ends.")
        self.play(FadeIn(long_rf), run_time=1.0)
        self.wait(1.0)

        # the plot: the arc cell at this point against curvature
        kap, resp = kappa_curve()
        ax0 = Axes(x_range=[0, 0.2, 1 / 30], y_range=[0, 100, 20], x_length=5.0, y_length=3.6,
                   axis_config=dict(color=MUTED, include_tip=False, font_size=22, stroke_width=1.5),
                   y_axis_config=dict(numbers_to_include=[20, 40, 60, 80, 100])).move_to(np.array([3.45, -0.15, 0]))

        class _K:                                   # the axes in kappa: x = kappa + 0.1
            def __init__(self, a): self.a = a
            def c2p(self, k, v): return self.a.c2p(k + 0.1, v)
            def plot_line_graph(self, ks, vs, **kw): return self.a.plot_line_graph(np.asarray(ks) + 0.1, vs, **kw)
        kax = _K(ax0)
        ax = ax0
        x_nums = VGroup(*[MathTex(t).scale(0.5).next_to(kax.c2p(v, 0), DOWN, buff=0.14)
                          for v, t in ((-1 / 15, r"-\tfrac{1}{15}"), (-1 / 30, r"-\tfrac{1}{30}"), (0, "0"),
                                       (1 / 30, r"\tfrac{1}{30}"), (1 / 15, r"\tfrac{1}{15}"))])
        xl = MathTex(r"\kappa", color=KAP_COL).scale(0.65).next_to(kax.c2p(0.1, 0), RIGHT, buff=0.15)
        yl = Tex(r"response here (noise units)").scale(0.5).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.42)
        long_here = float(A.sample(ref["long"], p[None])[:, 0].max())
        straight_lvl = tbl["arcs"][0]
        ln_long = DashedLine(kax.c2p(-0.1, long_here), kax.c2p(0.1, long_here), color=LONG_COL, stroke_width=2)
        ln_long_l = Tex(r"long straight cell, here", color=LONG_COL).scale(0.45).next_to(kax.c2p(-0.1, long_here), UP, buff=0.06).align_to(kax.c2p(-0.1, 0), LEFT).shift(RIGHT * 0.1)
        ln_str = DashedLine(kax.c2p(-0.1, straight_lvl), kax.c2p(0.1, straight_lvl), color=WHITE, stroke_width=1.5, stroke_opacity=0.6)
        ln_str_l = Tex(r"on a straight vessel", color=GREY_A).scale(0.45).next_to(kax.c2p(-0.1, straight_lvl), UP, buff=0.06).align_to(kax.c2p(-0.1, 0), LEFT).shift(RIGHT * 0.1)
        self.play(Create(ax), FadeIn(x_nums), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.play(Create(ln_long), FadeIn(ln_long_l), Create(ln_str), FadeIn(ln_str_l))
        self.say(rf"Here it gives {long_here:.0f} noise units; on a straight vessel the same cells give {straight_lvl:.0f} (toy).", wait="read")

        # short cells
        short_rf = receptive_field(length=zf.unit * 4 * A.SIGMA * A.E_SHORT, sigma=zf.unit * A.SIGMA,
                                   theta_scene=zf.screen_angle(th)).move_to(P).set_z_index(4)
        self.say(rf"A short cell, elongation 1.2, fits the bend, but alone it is weak: {tbl['short'][0]:.0f} units on a straight vessel (toy).")
        self.play(ReplacementTransform(long_rf, short_rf), run_time=1.2)
        self.wait(1.5)

        # the arc: a necklace of short cells along a circle
        kt = ValueTracker(0.0)
        s_beads = np.arange(-15.0, 15.01, 3.0)
        w_beads = np.exp(-s_beads ** 2 / (2 * A.SIGMA_S ** 2))

        def necklace():
            kv = kt.get_value()
            pts = A.path(p, th, kv, np.linspace(-17, 17, 60))
            curve = VMobject(stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.55)
            curve.set_points_smoothly([zp(*q) for q in pts])
            g = VGroup(curve)
            bp = A.path(p, th, kv, s_beads)
            for (x, y), s, w in zip(bp, s_beads, w_beads):
                loc = th + kv * s
                e = Ellipse(width=0.21, height=0.085, stroke_color=BLACK, stroke_width=1.0,
                            fill_color=orientation_color(loc), fill_opacity=0.3 + 0.7 * w)
                g.add(e.rotate(zf.screen_angle(loc)).move_to(zp(x, y)))
            return g.set_z_index(5)

        neck = always_redraw(necklace)
        self.say(r"So pool short cells along a path: the Gaussian-weighted mean of their drive. Straight, that is the long cell again.")
        self.play(FadeOut(short_rf), FadeIn(neck), zim.animate.set_opacity(0.55), run_time=1.2)
        resp_at = lambda v: float(np.interp(v, kap, resp))
        kdot = always_redraw(lambda: Dot(kax.c2p(kt.get_value(), resp_at(kt.get_value())), radius=0.07, color=KAP_COL, z_index=6))
        trace_lo = ValueTracker(0.0)
        trace = always_redraw(lambda: kax.plot_line_graph(
            kap[(kap >= trace_lo.get_value() - 1e-9) & (kap <= kt.get_value() + 1e-9)],
            resp[(kap >= trace_lo.get_value() - 1e-9) & (kap <= kt.get_value() + 1e-9)],
            line_color=KAP_COL, add_vertex_dots=False, stroke_width=4) if kt.get_value() > trace_lo.get_value() + 0.002 else VGroup())
        k_num = DecimalNumber(0.0, num_decimal_places=3, include_sign=True, color=KAP_COL).scale(0.55)
        k_lbl = MathTex(r"\kappa =", color=KAP_COL).scale(0.55)
        k_unit = MathTex(r"\mathrm{px}^{-1}", color=KAP_COL).scale(0.55)
        kread = VGroup(k_lbl, k_num, k_unit).arrange(RIGHT, buff=0.1).next_to(ax, UP, buff=0.2).align_to(ax, LEFT).shift(RIGHT * 0.3)
        k_num.add_updater(lambda m: m.set_value(kt.get_value()))
        self.add(kdot, kread)
        self.wait(1.0)
        self.say(r"Now give the path a curvature $\kappa$. Each short cell is read at the local tangent, so the colours turn along it.")
        self.play(kt.animate.set_value(-0.1), run_time=2.0)
        trace_lo.set_value(-0.1)
        self.add(trace)
        self.play(kt.animate.set_value(0.1), run_time=6.0, rate_func=linear)
        self.play(kt.animate.set_value(1 / R_PROBE), run_time=1.5)
        peak = DashedLine(kax.c2p(1 / R_PROBE, 0), kax.c2p(1 / R_PROBE, resp_at(1 / R_PROBE)), color=YELLOW, stroke_width=2)
        peak_l = MathTex(r"\kappa = 1/r", color=YELLOW).scale(0.55).next_to(kax.c2p(1 / R_PROBE, 0), UP, buff=0.12).shift(RIGHT * 0.55)
        self.play(Create(peak), FadeIn(peak_l))
        self.say(r"The response peaks where the path hugs the ring: at $\kappa = 1/r$. An arc cell keeps the bend.", wait="read")
        trace.clear_updaters()

        if self.at(2):
            # ------------------------------------------------------------ 3. the formula
            eq1 = MathTex(r"p(s)", "=", "p", "+", r"\frac{\sin \kappa s}{\kappa}\,", r"t_k", "+", r"\frac{1-\cos\kappa s}{\kappa}\,", r"n_k").scale(0.62)
            eq2 = MathTex(r"A_{k,\kappa}(p)", "=", r"\frac{\sum_s w(s)\,", r"U_{\theta_k+\kappa s}", r"\big(p(s)\big)", r"}{\sum_s w(s)}",
                          r",\quad", r"w(s) = e^{-s^2/2\sigma_s^2}").scale(0.62)
            for e, idx in ((eq1, (4, 7)), ):
                for i in idx:
                    e[i].set_color(KAP_COL)
            eq1[5].set_color(orientation_color(th)); eq1[8].set_color(orientation_color(th))
            eq2[0].set_color(KAP_COL); eq2[3].set_color(EXC)
            eqs = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            eqs.scale(min(1.0, 6.6 / eqs.width)).move_to(np.array([3.6, 2.8, 0]))
            # make room: the plot moves down a little
            self.play(VGroup(ax, x_nums, xl, yl, ln_long, ln_str, ln_long_l, ln_str_l, peak, peak_l).animate.shift(DOWN * 0.55),
                      FadeOut(kread), run_time=0.6)
            kdot.clear_updaters(); k_num.clear_updaters()
            kdot2 = always_redraw(lambda: Dot(kax.c2p(kt.get_value(), resp_at(kt.get_value())), radius=0.07, color=KAP_COL, z_index=6))
            self.remove(kdot); self.add(kdot2)
            trace_static = kax.plot_line_graph(kap, resp, line_color=KAP_COL, add_vertex_dots=False, stroke_width=4)
            self.remove(trace); self.add(trace_static)
            self.say(r"The path: start at $p$, along the tangent $t_k$, turning toward the normal $n_k$ at rate $\kappa$.")
            self.play(Write(eq1[:4]), run_time=0.8)
            self.play(Write(eq1[4:6]), run_time=0.9)
            self.play(Write(eq1[6:]), run_time=0.9)
            self.wait(1.0)
            self.say(r"Each sample reads the short cells' sheet at its local tangent, $\theta_k + \kappa s$ (interpolated), with Gaussian weights.")
            self.play(Write(eq2[:2]), run_time=0.7)
            self.play(Write(eq2[2:6]), run_time=1.4)
            self.play(Write(eq2[6:]), run_time=1.0)
            sig = Tex(rf"$\sigma_s = \sigma\sqrt{{e_\mathrm{{long}}^2 - 1.2^2}} = {A.SIGMA_S:.1f}$ px: straight, it is today's long cell",
                      color=GREY_B).scale(0.46).next_to(eqs, DOWN, buff=0.14).align_to(eqs, LEFT)
            self.play(FadeIn(sig))
            self.wait(2.0)
            formula = VGroup(eqs, sig)
        else:
            formula = VGroup()

        if not self.at(3):
            self.outro()
            return

        # ------------------------------------------------------------ 4. curvature competes like width: a population code
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self._cap], run_time=0.8)
        for m in (neck, kdot2):
            m.clear_updaters()
        comp = MathTex(r"A_k", "=", r"\max_{\kappa}", r"A_{k,\kappa}", r",\qquad", r"K_k", "=", r"\arg\max_{\kappa}", r"A_{k,\kappa}").scale(0.75)
        comp[3].set_color(KAP_COL); comp[8].set_color(KAP_COL); comp[5].set_color(YELLOW)
        comp.to_edge(UP, buff=0.4)
        kap_list = MathTex(r"\kappa \in \{0,\ \pm\tfrac{1}{60},\ \pm\tfrac{1}{30},\ \pm\tfrac{1}{15}\}\ \mathrm{px}^{-1}", color=KAP_COL).scale(0.55)
        kap_list.next_to(comp, DOWN, buff=0.15)
        self.say(r"Curvature competes like width: per orientation, the best of seven curvatures wins, and its value is kept in a map $K$.")
        self.play(Write(comp), run_time=1.5)
        self.play(FadeIn(kap_list))

        # four small population plots: one point on each ring and on the straight vessel
        order = [6, 4, 2, 0, 1, 3, 5]                       # kappa ascending: -1/15 .. 1/15
        cases = [(60, "r = 60"), (30, "r = 30"), (15, "r = 15"), ("straight", "straight")]
        panels = VGroup()
        for i, (case, label) in enumerate(cases):
            pts = A.centreline(case, 360)
            q = pts[45]
            col = A.sample(ref["best"], q[None])[:, 0]
            kb = int(np.argmax(col))
            vals = np.array([A.sample(ref["arcs"][j], q[None])[kb, 0] for j in order])
            win = int(np.argmax(vals))
            bw, bh = 0.24, 1.9
            g = VGroup()
            for j, v in enumerate(vals):
                b = Rectangle(width=bw * 0.8, height=max(bh * v / 90, 0.01), stroke_width=0,
                              fill_color=YELLOW if j == win else KAP_COL, fill_opacity=0.95 if j == win else 0.55)
                b.move_to(np.array([j * bw, 0, 0]), aligned_edge=DOWN)
                g.add(b)
            base = Line(np.array([-bw / 2, 0, 0]), np.array([6.5 * bw, 0, 0]), stroke_color=MUTED, stroke_width=1.5)
            ticks = VGroup(*[MathTex(t).scale(0.34).next_to(np.array([j * bw, 0, 0]), DOWN, buff=0.1)
                             for j, t in ((0, r"-\frac{1}{15}"), (3, "0"), (6, r"\frac{1}{15}"))])
            ttl = Tex(label).scale(0.55).next_to(base, UP, buff=bh + 0.35)
            wl = A.KAPPAS[order[win]]
            wtxt = MathTex(r"K = " + (r"0" if wl == 0 else (r"-" if wl < 0 else "") + rf"\tfrac{{1}}{{{int(round(1 / abs(wl)))}}}"),
                           color=YELLOW).scale(0.5).next_to(ticks, DOWN, buff=0.12)
            panels.add(VGroup(base, g, ticks, ttl, wtxt))
        panels.arrange(RIGHT, buff=0.95).move_to(DOWN * 0.35)
        self.play(LaggedStart(*[AnimationGroup(Create(pn[0]), FadeIn(pn[3]), FadeIn(pn[2]),
                                               LaggedStart(*[GrowFromEdge(b, DOWN) for b in pn[1]], lag_ratio=0.1))
                                for pn in panels], lag_ratio=0.35), run_time=3.5)
        self.say(r"Read at one point of each ring, the seven arc cells peak at $1/r$: curvature comes out as a population code.")
        self.play(LaggedStart(*[FadeIn(pn[4], shift=UP * 0.1) for pn in panels], lag_ratio=0.25), run_time=1.5)
        wins = [A.winning_kappa(r) for r in A.RADII]
        ok = all(abs(w - 1 / r) < 1e-9 for w, r in zip(wins, A.RADII))
        note = Tex(r"toy: along every ring the median winning $|\kappa|$ is exactly $1/r$" if ok else
                   r"toy: the median winning $|\kappa|$ is " + ", ".join(f"{w:.3f}" for w in wins), color=GREY_B).scale(0.5)
        note.next_to(panels, DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2.0)

        # ------------------------------------------------------------ 5. the reference table, toy beside the spec
        self.play(FadeOut(VGroup(comp, kap_list, panels, note)), run_time=0.8)
        hdr = VGroup(Tex(r"median response along the centreline, fine channel", color=GREY_A).scale(0.55),
                     Tex(r"str.: noise units; rings: \% of str.; bg: background, 99th percentile", color=GREY_B).scale(0.46)
                     ).arrange(DOWN, buff=0.1).to_edge(UP, buff=0.35)

        def row(name, key, spec):
            t = tbl[key]
            return [name, f"{t[0]:.0f}", f"{t[1]:.0f}", f"{t[2]:.0f}", f"{t[3]:.0f}", f"{t[4]:.1f}",
                    spec[0], spec[1], spec[2], spec[3], spec[4]]
        rows = [
            [r"cells", r"str.", r"60", r"30", r"15", r"bg", r"str.", r"60", r"30", r"15", r"bg"],
            row(r"straight, elongation 3", "long", ("101", "75", "56", "44", "2.8")),
            row(r"short, elongation 1.2", "short", ("84", "91", "88", "82", "2.7")),
            row(r"short, pooled straight", "pooled", ("130", "85", "70", "51", "2.9")),
            row(r"short, pooled on 7 arcs", "arcs", ("130", "91", "88", "87", "3.2")),
        ]
        T = tex_table(rows, widths=[3.7] + [0.75] * 5 + [0.15 + 0.75] + [0.75] * 4, scale=0.5, row_buff=0.28)
        T.move_to(UP * 0.85)
        lines, rule = T
        grp_toy = Tex(r"this toy", color=WHITE).scale(0.5).next_to(VGroup(lines[0][1], lines[0][5]), UP, buff=0.18)
        grp_spec = Tex(r"measured (spec 4.8)", color=GREY_B).scale(0.5).next_to(VGroup(lines[0][6], lines[0][10]), UP, buff=0.18)
        for r in lines[1:]:
            for c in r[6:]:
                c.set_color(GREY_B)
        sep = DashedLine(lines[0][6].get_left() + LEFT * 0.2 + UP * 0.25, lines[-1][6].get_left() + LEFT * 0.2 + DOWN * 0.2,
                         stroke_color=MUTED, stroke_width=1)
        last = lines[-1]
        hl = SurroundingRectangle(last, buff=0.08, color=YELLOW, stroke_width=1.5)
        self.say(r"The reference test, toy beside measured. Straight cells fade around the bends; short cells pooled on arcs do not.")
        self.play(FadeIn(hdr), FadeIn(lines[0]), Create(rule), FadeIn(grp_toy), FadeIn(grp_spec), Create(sep))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in lines[1:]], lag_ratio=0.45), run_time=3.0)
        self.play(Create(hl))
        self.wait(2.5)
        floor_toy = 100 * (tbl["arcs"][4] / tbl["pooled"][4] - 1)
        cost = VGroup(
            Tex(rf"cost: the best of 7$\times$ more cells raises the noise floor (bg):").scale(0.52),
            Tex(rf"toy $+{floor_toy:.0f}\,\%$ over the straight pool; measured about $+10\,\%$").scale(0.52),
            Tex(r"the toy loses less at bends than MICA, but the four rows rank the same", color=GREY_B).scale(0.48),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(T, DOWN, buff=0.45).align_to(T, LEFT)
        self.say(r"The price is a slightly higher noise floor: picking the best of seven templates also picks the luckiest noise.")
        self.play(FadeIn(cost[:2]))
        self.wait(1.5)
        self.play(FadeIn(cost[2]))
        self.wait(2.0)

        # ------------------------------------------------------------ 6. status and checks
        self.play(FadeOut(VGroup(hdr, T, grp_toy, grp_spec, sep, hl, cost)), run_time=0.8)
        st = Tex(r"status: designed, not built (MICA layer 05)", color=GREY_A).scale(0.7).to_edge(UP, buff=0.6)
        checks = VGroup(
            Tex(r"forks: one arc can fit a parent and its branch as a single bend"),
            Tex(r"crossings: two orientations must survive"),
            Tex(r"line ends: no overshoot (against the scale selection of 4.6b)"),
            Tex(r"the noise floor, re-checked in noise units"),
            Tex(r"on the real image: do free ends cluster at bends?"),
        ).scale(0.55).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        boxes = VGroup(*[Square(0.22, stroke_color=GREY_B, stroke_width=1.5).next_to(c, LEFT, buff=0.2) for c in checks])
        chk = VGroup(checks, boxes).next_to(st, DOWN, buff=0.55)
        steps = Tex(r"introduced in two steps: curvatures $=\{0\}$ first (a control), then all seven", color=GREY_B).scale(0.5)
        steps.next_to(chk, DOWN, buff=0.45)
        self.say(r"Arc cells are designed, not built. Before switching them on, the spec lists what must be checked.")
        self.play(FadeIn(st))
        self.play(LaggedStart(*[FadeIn(VGroup(b, c), shift=RIGHT * 0.15) for b, c in zip(boxes, checks)], lag_ratio=0.3), run_time=2.5)
        self.play(FadeIn(steps))
        self.wait(2.0)
        self.say(r"Biology: V1 cells are mostly straight; curvature tuning comes from combining them (Dobbins, Zucker and Cynader 1987; V2, V4).",
                 wait="read")
        self.outro()


variants(ArcCells, module_globals=globals())
