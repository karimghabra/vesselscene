"""Chapter 4: one point, sixteen orientations.

A hypercolumn's tuning curve, and why step 4's surround costs crossings
(spec 4.0, 2.4, 4.6).  The template for every chapter: one construct(),
gated by self.at(level), and Short / Medium / Detailed variants at the end.

Render:  manim -qm scenes/s04_hypercolumn.py HypercolumnMedium
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import neuro as N  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (EXC, FB, INH, MUTED, OD_COL, PixelFrame, bilinear,  # noqa: E402
                             image, orientation_color)
from mica_anim.widgets import PolarTuning, bar_chart, receptive_field  # noqa: E402

SHAPE = (128, 128)
CROSS = np.array([80.0, 48.0])
A_DEG, B_DEG = 150.0, 60.0          # the two vessels' image orientations: 90 degrees apart
SIGMA, ELONG = 2.0, 4.0             # simple cell: 2 px across, 8 px along


def _unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


@lru_cache(maxsize=None)
def stimulus():
    """A vessel crossing another, and a blob of dark texture, in OD; the
    camera's view of it; and the 16 simple-cell layers in noise units."""
    ua, ub = _unit(A_DEG), _unit(B_DEG)
    clean = N.vessel_od(SHAPE, [(CROSS - 90 * ua, CROSS + 90 * ua, 3.0, 0.15),
                                (CROSS - 40 * ub, CROSS + 40 * ub, 3.0, 0.15)], blur=1.0)
    blob_c = (26.0, 104.0)
    clean = clean + N.blob_od(SHAPE, blob_c, 4.5, 0.06)
    od = N.add_noise(clean, 0.01, seed=1)
    tex = gaussian_filter(np.random.default_rng(7).normal(0, 1, SHAPE), 14)
    B = 0.82 * (1 + 0.6 * tex / np.abs(tex).max() * 0.12)
    I = N.to_intensity(od, B)
    x, y = N.grid(SHAPE)
    bg = (clean < 0.005) & (np.hypot(x - blob_c[0], y - blob_c[1]) > 16)
    U = N.noise_units(N.simple_cells(od, SIGMA, ELONG), bg)
    probes = {"vessel": CROSS + 34 * ua, "crossing": CROSS, "blob": np.array(blob_c)}
    return dict(od=od, I=I, U=U, probes=probes)


@lru_cache(maxsize=None)
def angle_sweep(step=2.5):
    """Tuning curves at the centre of two equal vessels crossing at each angle."""
    c = np.array([64.0, 64.0])
    angles = np.arange(20.0, 90.0 + 1e-9, step)
    curves = []
    for d in angles:
        ua, ub = _unit(150.0), _unit(150.0 - d)
        clean = N.vessel_od(SHAPE, [(c - 90 * ua, c + 90 * ua, 3.0, 0.15), (c - 90 * ub, c + 90 * ub, 3.0, 0.15)], blur=1.0)
        U = N.noise_units(N.simple_cells(N.add_noise(clean, 0.01, 1), SIGMA, ELONG), clean < 0.005)
        curves.append(N.tuning_curve(U, c))
    return angles, np.array(curves)


class Hypercolumn(ChapterScene):
    NUMBER, TITLE, SPEC = 4, "One point, sixteen orientations", "spec 4.0, 2.4, 4.6"
    MAP_KEYS = ("simple", "surround")

    def construct(self):
        self.intro()
        st = stimulus()
        pf = PixelFrame(SHAPE, 5.6, LEFT * 3.5 + UP * 0.25)

        # ---------------------------------------------------------- 1. the stimulus, in OD
        cam = image(st["I"], pf.height, "grey", vmin=0.55, vmax=0.95).move_to(pf.centre)
        odim = image(st["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(cam, buff=0, color=MUTED, stroke_width=1)
        head = Tex("a crossing, as the camera sees it").scale(0.6).next_to(border, UP, buff=0.15)
        head2 = Tex(r"the same, as optical density").scale(0.6).move_to(head)
        self.play(FadeIn(cam), Create(border), FadeIn(head))
        self.say(r"Two vessels crossing, and a blob of dark texture. Converted to OD, blood is bright.")
        self.play(FadeIn(odim), FadeOut(cam), ReplacementTransform(head, head2), run_time=1.3)
        self.wait(0.8 if self.short else 1.5)

        # ---------------------------------------------------------- 2. one simple cell, then sixteen
        polar = PolarTuning(radius=2.0, vmax=16.0).move_to(RIGHT * 3.6 + DOWN * 0.35)
        self.play(FadeIn(polar.axes))
        p0 = st["probes"]["vessel"]
        probe = Dot(pf.point(*p0), radius=0.06, color=YELLOW, z_index=5)
        rf = receptive_field(length=pf.unit * 4 * SIGMA * ELONG, sigma=pf.unit * SIGMA).move_to(probe)
        self.play(FadeIn(probe, scale=2), FadeIn(rf))

        cell_eq = MathTex(r"U_k(p)", "=", r"\sum_q", r"w_k(q-p)", r"\,", r"\mathrm{OD}(q)").scale(0.8)
        cell_eq.next_to(polar, UP, buff=0.2)
        cell_eq[3].set_color(EXC)
        cell_eq[5].set_color(OD_COL)
        self.say(r"A V1 simple cell weighs the OD around $p$: an excitatory stripe (blue) between inhibitory flanks (red).")
        if self.at(2):
            self.play(Write(cell_eq))
            self.wait(1.5)
        else:
            self.wait(1.5)

        tc = N.tuning_curve(st["U"], p0)
        theta_val = DecimalNumber(0.0, num_decimal_places=1, unit=r"^\circ").scale(0.7)
        u_val = DecimalNumber(tc[0], num_decimal_places=1).scale(0.7)
        readout = VGroup(VGroup(MathTex(r"\theta_k =").scale(0.7), theta_val).arrange(RIGHT, buff=0.12),
                         VGroup(MathTex(r"U_k =").scale(0.7), u_val).arrange(RIGHT, buff=0.12)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        readout.next_to(polar, LEFT, buff=0.1).shift(DOWN * 1.4)
        if self.at(3):
            self.say(r"Sixteen cells at one point, one per orientation: a \emph{hypercolumn}. (Noise units: 8 = eight times the background's spread.)")
        else:
            self.say(r"Sixteen cells at one point, one per orientation: a \emph{hypercolumn}.")
        self.play(FadeIn(readout))

        step = 0.38 if self.at(2) else 0.22
        spokes = VGroup(polar.spoke(0, tc[0]))
        self.play(Create(spokes[0]), rf[0].animate.set_stroke(orientation_color(N.THETAS[0])), run_time=0.4)
        for k in range(1, N.N_ORIENT):
            sk = polar.spoke(k, tc[k])
            spokes.add(sk)
            self.play(Rotate(rf, angle=-PI / N.N_ORIENT, about_point=probe.get_center()),
                      Create(sk), rf[0].animate.set_stroke(orientation_color(N.THETAS[k])),
                      ChangeDecimalToValue(theta_val, np.rad2deg(N.THETAS[k])),
                      ChangeDecimalToValue(u_val, tc[k]), run_time=step, rate_func=smooth)
        curve = polar.curve(tc, WHITE, width=3)
        self.play(Create(curve), spokes.animate.set_stroke(opacity=0.45), run_time=1.0)
        self.say(r"Read through one pixel, the 16 values are its \emph{tuning curve}: two lobes, along the vessel.", wait=1.5)

        # ---------------------------------------------------------- 3. drag the probe around
        self.play(*[FadeOut(m) for m in (rf, readout, spokes, curve)], *([FadeOut(cell_eq)] if self.at(2) else []))

        def tuning_here():
            return bilinear(st["U"], *pf.pixel(probe.get_center()))

        def rosette():
            vals = np.clip(tuning_here(), 0, None)
            c = probe.get_center()
            return VGroup(*[Line(c - polar.direction(k) * 0.016 * v, c + polar.direction(k) * 0.016 * v,
                                 stroke_color=orientation_color(N.THETAS[k]), stroke_width=2.5) for k, v in enumerate(vals)])

        lc = always_redraw(lambda: polar.curve(tuning_here(), WHITE, width=3, fill_opacity=0.08))
        ls = always_redraw(lambda: polar.spokes(tuning_here(), width=3).set_stroke(opacity=0.5))
        ro = always_redraw(rosette)
        where = Tex("on a vessel").scale(0.7).next_to(polar, UP, buff=0.25)
        self.add(ls, lc, ro)
        self.play(FadeIn(where))
        if self.at(2):
            self.say(r"Drag the point along the vessel: the curve hardly changes.")
            self.play(probe.animate.move_to(pf.point(*(CROSS + 62 * _unit(A_DEG)))), run_time=2.5)
            self.play(probe.animate.move_to(pf.point(*(CROSS + 16 * _unit(A_DEG)))), run_time=2.0)
        self.say(r"At the crossing, two vessels: two pairs of lobes.")
        self.play(probe.animate.move_to(pf.point(*CROSS)),
                  Transform(where, Tex("at a crossing").scale(0.7).move_to(where)), run_time=2.0)
        self.wait(1.5)
        self.say(r"On a round blob of dark texture: weak, and about the same at every orientation.")
        self.play(probe.animate.move_to(pf.point(*st["probes"]["blob"])),
                  Transform(where, Tex("on a blob of texture").scale(0.7).move_to(where)), run_time=2.5)
        self.wait(1.5)

        # ---------------------------------------------------------- 4. the surround, on three pixels
        for m in (lc, ls, ro):
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in (lc, ls, ro, where, probe, odim, border, head2, polar)], run_time=0.8)

        names, xs = ["vessel", "crossing", "blob"], [-4.5, 0.0, 4.5]
        panels, greys, titles, curves = [], [], [], {}
        for name, x in zip(names, xs):
            P = PolarTuning(radius=1.65, vmax=16.0).move_to(RIGHT * x + DOWN * 0.45)
            curves[name] = N.tuning_curve(st["U"], st["probes"][name])
            panels.append(P)
            greys.append(P.curve(curves[name], GREY_B, width=3))
            titles.append(Tex(name).scale(0.7).next_to(P, UP, buff=0.12))
        self.play(*[FadeIn(P.axes) for P in panels], *[Create(g) for g in greys], *[FadeIn(t) for t in titles])

        floor_eq = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 8", r"\text{ of } 16", r"\big)").scale(0.8)
        sub_eq = MathTex(r"U'_k", "=", r"\max\!\big(0,\;", "U_k", "-", r"\text{floor}", r"\big)").scale(0.8)
        eqs = VGroup(floor_eq, sub_eq).arrange(RIGHT, buff=0.9).to_edge(UP, buff=0.45)
        floor_eq[0].set_color(FB); sub_eq[5].set_color(FB); sub_eq[0].set_color(INH)
        self.say(r"Step 4, the surround: subtract the part of the response that is the same at every orientation.")
        self.play(Write(eqs) if self.at(2) else FadeIn(eqs))
        self.wait(1.0)

        rings, reds, losses = [], [], []
        for P, name in zip(panels, names):
            t = curves[name]
            rings.append(P.ring(N.floor(t, "half"), FB))
            reds.append(P.curve(N.surround(t, "half"), INH, width=3.5))
            loss = 1 - N.surround(t, "half").max() / t.max()
            losses.append(MathTex(rf"-{100 * loss:.0f}\%", color=INH).scale(0.8).next_to(P, DOWN, buff=0.12))
        self.play(*[Create(r) for r in rings], run_time=1.0)
        self.play(*[TransformFromCopy(g, r) for g, r in zip(greys, reds)], run_time=1.4)
        self.play(*[FadeIn(l, shift=UP * 0.2) for l in losses])
        self.say(r"It removes most of a blob and little of a vessel, as intended. But a crossing loses half of its response.", wait=2.0)

        if self.at(3):
            spec_tbl = MathTex(r"\text{measured (spec 2.4): vessel } {-4\%},\ \text{crossing } {-44\%},\ \text{blob } {-48\%}",
                               color=GREY_B).scale(0.55).next_to(VGroup(*losses), DOWN, buff=0.15)
            self.play(FadeIn(spec_tbl))
            self.wait(2.0)
            self.play(FadeOut(spec_tbl))

        if not self.at(2):
            self.outro()
            return

        # ---------------------------------------------------------- 5. why: sorted bars
        self.play(*[FadeOut(m) for m in (*panels, *greys, *reds, *rings, *losses, *titles)], run_time=0.8)
        charts = VGroup()
        for name, x in (("vessel", -3.4), ("crossing", 3.4)):
            t = curves[name]
            order = np.argsort(t)
            ch = bar_chart(t[order], width=5.0, height=3.0, vmax=16.0,
                           colors=[orientation_color(N.THETAS[k]) for k in order]).move_to(RIGHT * x + DOWN * 1.0)
            br = Brace(ch[1][:8], DOWN, buff=0.08, color=FB)
            brl = Tex(r"lowest 8", color=FB).scale(0.55).next_to(br, DOWN, buff=0.05)
            y = ch[0].get_y() + 3.0 * N.floor(t, "half") / 16.0
            fl = DashedLine([ch[0].get_left()[0], y, 0], [ch[0].get_right()[0], y, 0], color=FB)
            ttl = Tex(f"{name}, sorted").scale(0.65).set_y(1.6).set_x(x)
            charts.add(VGroup(ch, br, brl, fl, ttl))
        self.play(*[FadeIn(c[4]) for c in charts], *[Create(c[0]) for c in charts], run_time=1.5)
        self.say(r"Sort the 16 responses. One vessel leaves half the orientations silent, so the floor stays low.")
        self.play(GrowFromCenter(charts[0][1]), FadeIn(charts[0][2]), Create(charts[0][3]))
        self.wait(1.5)
        self.say(r"Two vessels leave none silent: the second lobe lifts the lower half, and the floor with it.")
        self.play(GrowFromCenter(charts[1][1]), FadeIn(charts[1][2]), Create(charts[1][3]))
        self.wait(2.0)

        if not self.at(3):
            self.outro()
            return

        # ---------------------------------------------------------- 6. the candidate fix, swept over crossing angle
        self.play(FadeOut(charts), FadeOut(eqs), run_time=0.8)
        old = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 8", r"\text{ of } 16", r"\big)").scale(0.8)
        fix = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 4", r"\text{ of } 16", r"\big)").scale(0.8)
        old[0].set_color(FB); fix[0].set_color(FB); fix[3].set_color(YELLOW)
        VGroup(old, fix).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UL, buff=0.45)
        self.say(r"Candidate fix (spec 4.6): take the floor from the lowest quarter instead. Sweep the crossing angle.")
        self.play(Write(old))
        self.play(TransformFromCopy(old, fix))

        angles, sweep = angle_sweep()
        loss_h = np.array([1 - N.surround(c, "half").max() / c.max() for c in sweep]) * 100
        loss_q = np.array([1 - N.surround(c, "quarter").max() / c.max() for c in sweep]) * 100
        ax = Axes(x_range=[20, 90, 10], y_range=[0, 60, 15], x_length=6.0, y_length=3.0,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=24),
                  x_axis_config=dict(numbers_to_include=[30, 45, 60, 75, 90]),
                  y_axis_config=dict(numbers_to_include=[15, 30, 45, 60])).move_to(RIGHT * 2.9 + DOWN * 0.25)
        xl = Tex(r"crossing angle ($^\circ$)").scale(0.55).next_to(ax.x_axis, DOWN, buff=0.35)
        yl = Tex(r"crossing's loss (\%)").scale(0.55).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.35)
        gate = DashedLine(ax.c2p(20, 15), ax.c2p(90, 15), color=GREEN_C)
        gate_l = Tex(r"gate: $\le 15\%$", color=GREEN_C).scale(0.55).next_to(ax.c2p(90, 15), RIGHT, buff=0.1)
        P = PolarTuning(radius=1.5, vmax=18.0).move_to(LEFT * 4.2 + DOWN * 0.55)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(P.axes))
        self.play(Create(gate), FadeIn(gate_l))

        ang = ValueTracker(angles[0])

        def curve_at(a):
            i = np.interp(a, angles, np.arange(len(angles)))
            i0 = int(np.floor(i)); i1 = min(i0 + 1, len(angles) - 1); f = i - i0
            return sweep[i0] * (1 - f) + sweep[i1] * f

        live = always_redraw(lambda: P.curve(curve_at(ang.get_value()), GREY_B, width=3))
        live_h = always_redraw(lambda: P.curve(N.surround(curve_at(ang.get_value()), "half"), INH, width=3))
        live_q = always_redraw(lambda: P.curve(N.surround(curve_at(ang.get_value()), "quarter"), YELLOW, width=3))
        ang_lbl = always_redraw(lambda: MathTex(rf"{ang.get_value():.0f}^\circ").scale(0.7).next_to(P, UP, buff=0.1))

        def trace(losses, color):
            def f():
                keep = angles <= ang.get_value() + 1e-9
                if keep.sum() < 2:
                    return VGroup()
                return ax.plot_line_graph(angles[keep], losses[keep], line_color=color, add_vertex_dots=False, stroke_width=4)
            return always_redraw(f)

        key = VGroup(Tex(r"lowest 8 (today)", color=INH).scale(0.55), Tex(r"lowest 4 (fix)", color=YELLOW).scale(0.55)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        self.add(live, live_h, live_q, ang_lbl, trace(loss_h, INH), trace(loss_q, YELLOW))
        self.play(FadeIn(key))
        self.play(ang.animate.set_value(angles[-1]), run_time=7.0, rate_func=linear)
        self.wait(0.5)
        self.say(r"In this toy hypercolumn the fix helps at shallow angles but fails the gate near $90^\circ$.", wait=2.0)
        self.say(r"So every change is gated (spec 4.6): crossings may lose at most 15\%, while blobs must still lose 40\%.", wait=2.5)
        self.outro()


variants(Hypercolumn, module_globals=globals())
