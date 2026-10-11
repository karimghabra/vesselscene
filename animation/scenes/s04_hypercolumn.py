"""Chapter 4 (proof of concept): a hypercolumn, and why the surround hurts crossings.

Spec sections 4.0 (sheets and stacks, noise units), 2.4 (the surround
penalises crossings) and 4.6 (the crossing fix and its gate).

Render:  manim -qm scenes/s04_hypercolumn.py Hypercolumn
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
from mica_anim.style import (BLOOD, EXC, FB, INH, MUTED, PixelFrame, bilinear,  # noqa: E402
                             image, orientation_color)
from mica_anim.widgets import PolarTuning, bar_chart, receptive_field  # noqa: E402

SHAPE = (128, 128)
CROSS = np.array([80.0, 48.0])
A_DEG, B_DEG = 150.0, 60.0          # the two vessels' image orientations: 90 degrees apart
SIGMA, ELONG = 2.0, 4.0             # simple cell: 2 px across, 8 px along
OD_COL = ManimColor("#FF7B6B")      # OD in equations (lighter than BLOOD, readable on black)


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


def caption(text, scale=0.56):
    return Tex(text, color=GREY_A).scale(scale).to_edge(DOWN, buff=0.3)


class Hypercolumn(Scene):
    def say(self, text, wait=0.0, **kw):
        new = caption(text, **kw)
        old = getattr(self, "_cap", None)
        self.play(*( [FadeOut(old, shift=UP * 0.15)] if old else []), FadeIn(new, shift=UP * 0.15), run_time=0.7)
        self._cap = new
        if wait:
            self.wait(wait)

    def construct(self):
        st = stimulus()
        pf = PixelFrame(SHAPE, 5.6, LEFT * 3.5 + UP * 0.25)

        # ---------------------------------------------------------- 1. the image, as OD
        cam = image(st["I"], pf.height, "grey", vmin=0.55, vmax=0.95).move_to(pf.centre)
        odim = image(st["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        border = SurroundingRectangle(cam, buff=0, color=MUTED, stroke_width=1)
        head = Tex("what the camera records").scale(0.6).next_to(border, UP, buff=0.15)

        title = Tex(r"One point, sixteen orientations").scale(1.0)
        self.play(Write(title), run_time=1.2)
        self.wait(0.6)
        self.play(FadeOut(title, shift=UP * 0.3))

        self.play(FadeIn(cam), Create(border), FadeIn(head))
        eq1 = MathTex("I", "=", "B", r"\,e^{-", r"\mathrm{OD}", "}").scale(1.1).move_to(RIGHT * 3.6 + UP * 1.6)
        eq1[4].set_color(OD_COL)
        self.say(r"Green light through blood: the image is the background $B$ dimmed by the blood's optical density.")
        self.play(Write(eq1))
        self.wait(1.0)
        eq2 = MathTex(r"\mathrm{OD}", "=", r"-\ln", r"{I", r"\over", r"B}").scale(1.1).move_to(eq1)
        eq2[0].set_color(OD_COL)
        self.play(TransformMatchingShapes(eq1, eq2))
        head2 = Tex(r"optical density, OD").scale(0.6).move_to(head)
        self.play(FadeIn(odim), FadeOut(cam), ReplacementTransform(head, head2), run_time=1.5)
        self.say(r"OD is proportional to the path length through blood: vessels light up, and a crossing adds two of them.", wait=1.5)

        # ---------------------------------------------------------- 2. one simple cell, then sixteen
        polar = PolarTuning(radius=2.0, vmax=16.0).move_to(RIGHT * 3.6 + DOWN * 0.35)
        self.play(eq2.animate.scale(0.7).to_corner(UR, buff=0.4), FadeIn(polar.axes))

        p0 = st["probes"]["vessel"]
        probe = Dot(pf.point(*p0), radius=0.06, color=YELLOW, z_index=5)
        rf = receptive_field(length=pf.unit * 4 * SIGMA * ELONG, sigma=pf.unit * SIGMA).move_to(probe)
        self.play(FadeIn(probe, scale=2), FadeIn(rf))

        cell_eq = MathTex(r"U_k(p)", "=", r"\sum_q", r"w_k(q-p)", r"\,", r"\mathrm{OD}(q)").scale(0.8)
        cell_eq.next_to(polar, UP, buff=0.2)
        cell_eq[3].set_color(EXC)
        cell_eq[5].set_color(OD_COL)
        self.say(r"A V1 simple cell weighs the OD around $p$: an excitatory stripe (blue) between inhibitory flanks (red).")
        self.play(Write(cell_eq))
        self.wait(1.0)

        tc = N.tuning_curve(st["U"], p0)
        theta_lbl = MathTex(r"\theta_k =").scale(0.7)
        theta_val = DecimalNumber(0.0, num_decimal_places=1, unit=r"^\circ").scale(0.7)
        u_lbl = MathTex(r"U_k =").scale(0.7)
        u_val = DecimalNumber(tc[0], num_decimal_places=1).scale(0.7)
        readout = VGroup(VGroup(theta_lbl, theta_val).arrange(RIGHT, buff=0.12),
                         VGroup(u_lbl, u_val).arrange(RIGHT, buff=0.12)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        readout.next_to(polar, LEFT, buff=0.1).shift(DOWN * 1.4)
        self.say(r"Sixteen cells at one point, one per orientation: a \emph{hypercolumn}. (Noise units: 8 = eight times the background's spread.)")
        self.play(FadeIn(readout))

        spokes = VGroup()
        s0 = polar.spoke(0, tc[0])
        spokes.add(s0)
        rf_col = orientation_color(N.THETAS[0])
        self.play(Create(s0), rf[0].animate.set_stroke(rf_col), run_time=0.5)
        for k in range(1, N.N_ORIENT):
            sk = polar.spoke(k, tc[k])
            spokes.add(sk)
            self.play(Rotate(rf, angle=-PI / N.N_ORIENT, about_point=probe.get_center()),
                      Create(sk), rf[0].animate.set_stroke(orientation_color(N.THETAS[k])),
                      ChangeDecimalToValue(theta_val, np.rad2deg(N.THETAS[k])),
                      ChangeDecimalToValue(u_val, tc[k]),
                      run_time=0.38, rate_func=smooth)
        curve = polar.curve(tc, WHITE, width=3)
        self.play(Create(curve), spokes.animate.set_stroke(opacity=0.45), run_time=1.2)
        self.say(r"Read through one pixel, the 16 values are its \emph{tuning curve}: two lobes, along the vessel.", wait=1.5)

        # ---------------------------------------------------------- 3. drag the probe around
        self.play(FadeOut(rf), FadeOut(readout), FadeOut(spokes), FadeOut(curve), FadeOut(cell_eq))

        def live_curve():
            return polar.curve(bilinear(st["U"], *pf.pixel(probe.get_center())), WHITE, width=3, fill_opacity=0.08)

        def live_spokes():
            return polar.spokes(bilinear(st["U"], *pf.pixel(probe.get_center())), width=3).set_stroke(opacity=0.5)

        def rosette():
            """The same tuning curve, drawn small at the probe itself."""
            vals = np.clip(bilinear(st["U"], *pf.pixel(probe.get_center())), 0, None)
            c = probe.get_center()
            g = VGroup()
            for k, v in enumerate(vals):
                d = polar.direction(k) * 0.016 * v
                g.add(Line(c - d, c + d, stroke_color=orientation_color(N.THETAS[k]), stroke_width=2.5))
            return g

        lc, ls, ro = always_redraw(live_curve), always_redraw(live_spokes), always_redraw(rosette)
        where = Tex("on a vessel").scale(0.7).next_to(polar, UP, buff=0.25)
        self.add(ls, lc, ro)
        self.play(FadeIn(where))
        self.say(r"Drag the point along the vessel: the curve hardly changes.")
        along = CROSS + 62 * _unit(A_DEG)
        self.play(probe.animate.move_to(pf.point(*along)), run_time=2.5)
        self.play(probe.animate.move_to(pf.point(*(CROSS + 16 * _unit(A_DEG)))), run_time=2.0)
        self.say(r"At the crossing, two vessels: two pairs of lobes.")
        self.play(probe.animate.move_to(pf.point(*CROSS)),
                  Transform(where, Tex("at a crossing").scale(0.7).move_to(where)), run_time=2.0)
        self.wait(1.5)
        self.say(r"On a round blob of dark texture: weak, and roughly the same at every orientation.")
        self.play(probe.animate.move_to(pf.point(*st["probes"]["blob"])),
                  Transform(where, Tex("on a blob of texture").scale(0.7).move_to(where)), run_time=3.0)
        self.wait(1.5)

        # ---------------------------------------------------------- 4. the surround, on three pixels
        lc.clear_updaters(); ls.clear_updaters(); ro.clear_updaters()
        self.play(*[FadeOut(m) for m in (lc, ls, ro, where, probe, odim, border, head2, polar, eq2)], run_time=1.0)

        names = ["vessel", "crossing", "blob"]
        xs = [-4.5, 0.0, 4.5]
        panels, grey_curves, titles, curves = [], [], [], {}
        for name, x in zip(names, xs):
            P = PolarTuning(radius=1.65, vmax=16.0).move_to(RIGHT * x + DOWN * 0.45)
            t = N.tuning_curve(st["U"], st["probes"][name])
            curves[name] = t
            g = P.curve(t, GREY_B, width=3)
            ttl = Tex(name).scale(0.7).next_to(P, UP, buff=0.12)
            panels.append(P); grey_curves.append(g); titles.append(ttl)
        self.play(*[FadeIn(P.axes) for P in panels], *[Create(g) for g in grey_curves], *[FadeIn(t) for t in titles])

        floor_eq = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 8", r"\text{ of } 16", r"\big)").scale(0.8)
        sub_eq = MathTex(r"U'_k", "=", r"\max\!\big(0,\;", "U_k", "-", r"\text{floor}", r"\big)").scale(0.8)
        eqs = VGroup(floor_eq, sub_eq).arrange(RIGHT, buff=0.9).to_edge(UP, buff=0.45)
        floor_eq[0].set_color(FB); sub_eq[5].set_color(FB); sub_eq[0].set_color(INH)
        self.say(r"Step 4, the surround: subtract the part of the response that is the same at every orientation.")
        self.play(Write(eqs))
        self.wait(1.0)

        rings, reds, losses = [], [], []
        for P, name in zip(panels, names):
            t = curves[name]
            f = N.floor(t, "half")
            rings.append(P.ring(f, FB))
            reds.append(P.curve(N.surround(t, "half"), INH, width=3.5))
            loss = 1 - N.surround(t, "half").max() / t.max()
            losses.append(MathTex(rf"-{100 * loss:.0f}\%", color=INH).scale(0.8).next_to(P, DOWN, buff=0.12))
        self.play(*[Create(r) for r in rings], run_time=1.2)
        self.play(*[TransformFromCopy(g, r) for g, r in zip(grey_curves, reds)], run_time=1.6)
        self.play(*[FadeIn(l, shift=UP * 0.2) for l in losses])
        self.say(r"It removes most of a blob and little of a vessel, as intended. But a crossing loses half of its response.", wait=2.0)

        # ---------------------------------------------------------- 5. why: sorted bars
        self.play(*[FadeOut(m) for m in (*panels, *grey_curves, *reds, *rings, *losses, *titles)], run_time=0.8)
        charts = VGroup()
        for name, x in (("vessel", -3.4), ("crossing", 3.4)):
            t = curves[name]
            order = np.argsort(t)
            cols = [orientation_color(N.THETAS[k]) for k in order]
            ch = bar_chart(t[order], width=5.0, height=3.0, vmax=16.0, colors=cols).move_to(RIGHT * x + DOWN * 0.5, aligned_edge=ORIGIN)
            ch.shift(DOWN * 0.5)
            low = ch[1][:8]
            br = Brace(low, DOWN, buff=0.08, color=FB)
            brl = Tex(r"lowest 8", color=FB).scale(0.55).next_to(br, DOWN, buff=0.05)
            f = N.floor(t, "half")
            y = ch[0].get_y() + 3.0 * f / 16.0
            fl = DashedLine(ch[0].get_left() + UP * (y - ch[0].get_y()), ch[0].get_right() + UP * (y - ch[0].get_y()), color=FB)
            ttl = Tex(f"{name}, sorted").scale(0.65).next_to(ch, UP, buff=0.25).set_y(1.6)
            charts.add(VGroup(ch, br, brl, fl, ttl))
        self.play(*[FadeIn(c[4]) for c in charts], *[Create(c[0]) for c in charts], run_time=1.5)
        self.say(r"Sort the 16 responses. One vessel leaves half the orientations silent, so the floor stays low.")
        self.play(GrowFromCenter(charts[0][1]), FadeIn(charts[0][2]), Create(charts[0][3]))
        self.wait(1.5)
        self.say(r"Two vessels leave none silent: the second lobe lifts the lower half, and the floor with it.")
        self.play(GrowFromCenter(charts[1][1]), FadeIn(charts[1][2]), Create(charts[1][3]))
        self.wait(2.0)

        # ---------------------------------------------------------- 6. the candidate fix, swept over crossing angle
        self.play(FadeOut(charts), FadeOut(eqs), run_time=0.8)
        fix = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 4", r"\text{ of } 16", r"\big)").scale(0.8)
        fix[0].set_color(FB); fix[3].set_color(YELLOW)
        old = MathTex(r"\text{floor}", "=", r"\operatorname{mean}\big(", r"\text{lowest } 8", r"\text{ of } 16", r"\big)").scale(0.8)
        old[0].set_color(FB)
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
        gate_l = Tex(r"gate: $\le 15\%$", color=GREEN_C).scale(0.55).next_to(gate, UP, buff=0.05).align_to(gate, LEFT)
        P = PolarTuning(radius=1.5, vmax=18.0).move_to(LEFT * 4.2 + DOWN * 0.55)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(P.axes))
        self.play(Create(gate), FadeIn(gate_l))

        ang = ValueTracker(angles[0])

        def curve_at(a):
            i = np.interp(a, angles, np.arange(len(angles)))
            i0 = int(np.floor(i)); i1 = min(i0 + 1, len(angles) - 1); f = i - i0
            return sweep[i0] * (1 - f) + sweep[i1] * f

        live = always_redraw(lambda: P.curve(curve_at(ang.get_value()), GREY_B, width=3))
        live_red = always_redraw(lambda: P.curve(N.surround(curve_at(ang.get_value()), "half"), INH, width=3))
        live_q = always_redraw(lambda: P.curve(N.surround(curve_at(ang.get_value()), "quarter"), YELLOW, width=3))
        ang_lbl = always_redraw(lambda: MathTex(rf"{ang.get_value():.0f}^\circ").scale(0.7).next_to(P, UP, buff=0.1))

        def trace(losses, color):
            return always_redraw(lambda: ax.plot_line_graph(
                angles[angles <= ang.get_value() + 1e-9], losses[angles <= ang.get_value() + 1e-9],
                line_color=color, add_vertex_dots=False, stroke_width=4) if ang.get_value() > angles[0] + 1e-9 else VGroup())

        th, tq = trace(loss_h, INH), trace(loss_q, YELLOW)
        key = VGroup(Tex(r"lowest 8 (today)", color=INH).scale(0.55), Tex(r"lowest 4 (fix)", color=YELLOW).scale(0.55)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        key.next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        self.add(live, live_red, live_q, ang_lbl, th, tq)
        self.play(FadeIn(key))
        self.play(ang.animate.set_value(angles[-1]), run_time=7.0, rate_func=linear)
        self.wait(0.5)
        self.say(r"In this toy hypercolumn the fix helps at shallow angles but fails the gate near $90^\circ$: why every fix is gated.", wait=2.5)
        self.say(r"Spec 4.6: accept only if crossings lose at most 15\% while blobs still lose at least 40\%.", wait=2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
