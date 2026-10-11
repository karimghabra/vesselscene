"""The very short cut: the whole spec in about 75 seconds.

One beat per idea: the goal on a vesselscene burst, the junction problem,
the two modes of vision, V1's hypercolumn, V2's junction cells, fork
against crossing, co-circular continuation and attentive grouping,
feedback, and the annotated result.  Every map is computed (neuro.py and
the chapter-4 stimulus); the annotation drawn on the vesselscene image is
its observable truth, the target.

Render:  manim -ql scenes/s99_trailer.py Trailer
"""
from __future__ import annotations

import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mica_anim import neuro as N  # noqa: E402
from mica_anim import vsdata as D  # noqa: E402
from mica_anim.archmap import FEEDFORWARD, ArchitectureMap  # noqa: E402
from mica_anim.chapter import ChapterScene  # noqa: E402
from mica_anim.style import (ATTN, CONT, FB, FF, JUNC, MUTED, PixelFrame,  # noqa: E402
                             bilinear, image, orientation_color)
from mica_anim.widgets import PolarTuning, bar_chart  # noqa: E402
from s04_hypercolumn import A_DEG, CROSS, ELONG, SIGMA, _unit, stimulus  # noqa: E402

TOY = (96, 96)


@lru_cache(maxsize=None)
def t_junction():
    """A parent vessel and a branch that ends on it, in OD, with the arm
    strength around the junction for 32 outward directions."""
    c = np.array([48.0, 40.0])
    parent = (c + [-60, 8], c + [60, -8], 3.0, 0.15)
    branch = (c, c + 52 * _unit(70.0), 2.4, 0.12)
    od = N.add_noise(N.vessel_od(TOY, [parent, branch], blur=1.0, combine="max"), 0.01, seed=3)
    U = N.noise_units(N.simple_cells(od, SIGMA, ELONG), N.vessel_od(TOY, [parent, branch], blur=1.0) < 0.005)
    phis = np.arange(32) * 2 * np.pi / 32
    strength = []
    for phi in phis:
        k = int(round((phi % np.pi) / (np.pi / 16))) % 16
        vals = [bilinear(U[k:k + 1], c[0] + d * np.cos(phi), c[1] + d * np.sin(phi))[0] for d in np.arange(6, 18)]
        strength.append(np.mean(np.clip(vals, 0, None)))
    strength = np.array(strength)
    peaks = [i for i in range(32) if strength[i] >= strength[i - 1] and strength[i] >= strength[(i + 1) % 32]]
    peaks = sorted(peaks, key=lambda i: -strength[i])
    return dict(od=od, c=c, phis=phis, strength=strength, peaks=peaks)


@lru_cache(maxsize=None)
def fork_and_crossing():
    """The same two vessels drawn as a crossing (ODs add) and as a fork
    (lumens merge: the union), and the OD along vessel A through the centre."""
    c = np.array([48.0, 48.0])
    ua, ub = _unit(160.0), _unit(105.0)
    a = (c - 60 * ua, c + 60 * ua, 3.2, 0.13)
    b_cross = (c - 60 * ub, c + 60 * ub, 3.0, 0.11)
    b_fork = (c, c + 60 * ub, 3.0, 0.11)
    cross = N.vessel_od(TOY, [a, b_cross], blur=1.2, combine="sum")
    fork = N.vessel_od(TOY, [a, b_fork], blur=1.2, combine="max")
    s = np.linspace(-36, 36, 145)
    prof = lambda im: np.array([bilinear(im[None], *(c + t * ua))[0] for t in s])
    return dict(cross=N.add_noise(cross, 0.006, 4), fork=N.add_noise(fork, 0.006, 5), s=s,
                p_cross=prof(cross), p_fork=prof(fork), c=c, ua=ua)


@lru_cache(maxsize=None)
def labels():
    """Attentive grouping on the chapter-4 crossing: each label spreads in
    its own vessel's orientation sheets (plus or minus one), so it passes
    straight through the crossing.  Returns geodesic distance maps."""
    st = stimulus()
    U = st["U"]

    def spread(theta_deg, seed):
        k = int(round(np.deg2rad(theta_deg % 180) / (np.pi / 16))) % 16
        sheet = np.max(U[[(k - 1) % 16, k, (k + 1) % 16]], axis=0)
        ok = sheet > 4.0
        h, w = ok.shape
        dist = np.full(ok.shape, np.inf)
        q = deque()
        sx, sy = int(round(seed[0])), int(round(seed[1]))
        dist[sy, sx] = 0
        q.append((sy, sx))
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and ok[yy, xx] and dist[yy, xx] == np.inf:
                        dist[yy, xx] = dist[y, x] + 1
                        q.append((yy, xx))
        return dist

    da = spread(150.0, CROSS + 40 * _unit(150.0))
    db = spread(60.0, CROSS - 30 * _unit(60.0))
    return dict(od=st["od"], da=da, db=db)


def label_image(dist, d, color, height):
    """The label as a translucent overlay: pixels reached by distance d."""
    rgba = np.zeros(dist.shape + (4,), np.uint8)
    on = dist <= d
    rgba[on, :3] = (np.array(color.to_rgb()) * 255).astype(np.uint8)
    rgba[on, 3] = 200
    im = ImageMobject(rgba)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


class Trailer(ChapterScene):
    LEVEL = 1

    def construct(self):
        S = D.load()
        pf = PixelFrame(S.shape, 6.3, UP * 0.3)
        avg = image(S.view(S.average), pf.height, "grey").move_to(pf.centre)

        # ------------------------------------------------ 1. a burst of video -> a graph
        if S.burst is not None and len(S.burst) > 1:
            frames = [image(v, pf.height, "grey").move_to(pf.centre) for v in S.burst_views()]
            tag = Tex(r"a burst of video: red cells flowing", color=GREY_A).scale(0.55).next_to(avg, UP, buff=0.12)
            self.add(tag)
            for i in range(24):
                f = frames[i % len(frames)]
                self.add(f)
                self.wait(1 / 8)
                self.remove(f)
            self.add(frames[0])
            tag2 = Tex(r"averaged: the still", color=GREY_A).scale(0.55).move_to(tag)
            self.play(FadeIn(avg), FadeOut(frames[0]), ReplacementTransform(tag, tag2), run_time=1.2)
        else:
            tag2 = Tex(r"the still", color=GREY_A).scale(0.55).next_to(avg, UP, buff=0.12)
            self.play(FadeIn(avg), FadeIn(tag2))
        self.say(r"The surface vessels of the eye. The goal: every vessel, and every junction with its type.")
        runs = D.runs_overlay(S.truth, pf, min_len_px=25, width=2.0)
        rings = D.junctions_overlay(S.truth, pf, width=2.0)
        self.play(LaggedStart(*[Create(r) for r in runs], lag_ratio=0.015), run_time=3.0)
        legend = VGroup(*[VGroup(Circle(radius=0.08, stroke_color=col, stroke_width=3), Tex(name, color=GREY_A).scale(0.45)).arrange(RIGHT, buff=0.1)
                          for name, col in (("fork", D.JUNCTION_COLORS["bifurcation"]), ("crossing", D.JUNCTION_COLORS["crossing"]),
                                            ("compound", D.JUNCTION_COLORS["compound"]))]).arrange(RIGHT, buff=0.4)
        target = Tex(r"(the target: vesselscene's truth)", color=GREY_B).scale(0.45)
        VGroup(legend, target).arrange(RIGHT, buff=0.6).move_to(tag2)
        self.play(LaggedStart(*[FadeIn(r, scale=1.6) for r in rings], lag_ratio=0.01), FadeOut(tag2), FadeIn(legend), FadeIn(target), run_time=2.0)
        self.wait(2.0)

        # ------------------------------------------------ 2. the problem
        crops = {"crossing": (609, 104), "fork": (392, 285)}
        insets, boxes = Group(), VGroup()
        half = 34
        view = S.view(S.average)
        for name, (x, y) in crops.items():
            boxes.add(Square(side_length=2 * half * pf.unit, stroke_color=YELLOW, stroke_width=3).move_to(pf.point(x, y)))
            sub = view[y - half:y + half, x - half:x + half]
            im = image(sub, 3.3, "grey")
            insets.add(Group(im, SurroundingRectangle(im, buff=0, color=YELLOW, stroke_width=2),
                             Tex(name).scale(0.6).next_to(im, UP, buff=0.12)))
        insets.arrange(RIGHT, buff=1.2).shift(UP * 0.2)
        self.play(FadeOut(runs), FadeOut(rings), FadeOut(legend), FadeOut(target), Create(boxes), run_time=1.0)
        self.say(r"Today's annotator finds 61\% of junctions and types 34\% right (measured, spec 2.1). A fork and a crossing look alike in 2-D.")
        self.play(avg.animate.set_opacity(0.15), TransformFromCopy(boxes[0], insets[0][1]), TransformFromCopy(boxes[1], insets[1][1]),
                  FadeIn(insets[0][0]), FadeIn(insets[1][0]), FadeIn(insets[0][2]), FadeIn(insets[1][2]), FadeOut(boxes), run_time=1.5)
        self.wait(3.5)
        self.play(FadeOut(insets), FadeOut(avg), run_time=0.8)

        # ------------------------------------------------ 3. two modes of vision
        m = ArchitectureMap().scale(0.82).shift(UP * 0.35)
        self.say(r"The brain's answer: a fast feedforward sweep, then recurrent processing that groups, interprets and feeds back.")
        self.play(FadeIn(m), run_time=0.8)
        self.play(*[Indicate(m.boxes[k][0], color=FF, scale_factor=1.06) for k in FEEDFORWARD], run_time=1.2)
        rec = [k for k in m.boxes if k not in FEEDFORWARD and k != "motion"]
        self.play(LaggedStart(*[Indicate(m.boxes[k][0], color=WHITE, scale_factor=1.06) for k in rec], lag_ratio=0.08), run_time=1.6)
        self.play(*[ShowPassingFlash(a.copy().set_stroke(FB, width=8), time_width=0.6) for a in m.feedback], run_time=1.4)
        self.wait(1.2)
        self.play(FadeOut(m), run_time=0.6)

        # ------------------------------------------------ 4. V1: a hypercolumn
        st = stimulus()
        hf = PixelFrame(st["od"].shape, 5.2, LEFT * 3.3 + UP * 0.3)
        odim = image(st["od"], hf.height, "magma", vmin=-0.02, vmax=0.3).move_to(hf.centre)
        polar = PolarTuning(radius=2.0, vmax=16.0).move_to(RIGHT * 3.3 + UP * 0.1)
        probe = Dot(hf.point(*st["probes"]["vessel"]), radius=0.07, color=YELLOW, z_index=5)
        live = always_redraw(lambda: polar.curve(bilinear(st["U"], *hf.pixel(probe.get_center())), WHITE, width=3, fill_opacity=0.1))
        spokes = always_redraw(lambda: polar.spokes(bilinear(st["U"], *hf.pixel(probe.get_center())), width=3).set_stroke(opacity=0.55))
        self.say(r"V1: at every point, sixteen cells tuned to orientation. A vessel gives two lobes; a crossing, four.")
        self.play(FadeIn(odim), FadeIn(polar.axes), FadeIn(probe))
        self.add(spokes, live)
        self.wait(1.2)
        self.play(probe.animate.move_to(hf.point(*CROSS)), run_time=2.0)
        self.wait(1.5)
        live.clear_updaters(); spokes.clear_updaters()
        self.play(*[FadeOut(x) for x in (odim, polar, probe, live, spokes)], run_time=0.6)

        # ------------------------------------------------ 5. V2: junction cells
        T = t_junction()
        jf = PixelFrame(TOY, 5.0, LEFT * 3.3 + UP * 0.3)
        tim = image(T["od"], jf.height, "magma", vmin=-0.02, vmax=0.2).move_to(jf.centre)
        cpt = jf.point(*T["c"])
        disc = DashedVMobject(Circle(radius=17 * jf.unit, stroke_color=WHITE, stroke_width=2).move_to(cpt), num_dashes=30)
        smax = T["strength"].max()
        arms = VGroup(*[Line(cpt, cpt + np.array([np.cos(p), -np.sin(p), 0]) * 1.9 * v / smax,
                             stroke_color=orientation_color(p), stroke_width=5)
                        for p, v in zip(T["phis"], T["strength"]) if v > 0.15 * smax])
        self.say(r"V2: a junction cell gathers the arms meeting at a point. A junction needs a third arm: its evidence is the third strongest.")
        self.play(FadeIn(tim), Create(disc))
        self.play(LaggedStart(*[GrowFromPoint(a, cpt) for a in arms], lag_ratio=0.03), run_time=1.5)
        top = T["peaks"][:5]
        vals = T["strength"][top]
        cols = [GREY_B] * len(top)
        ch = bar_chart(vals, width=3.6, height=2.6, vmax=smax, colors=cols).move_to(RIGHT * 3.4 + DOWN * 0.3)
        third = ch[1][2]
        j3 = MathTex(r"J_3", color=JUNC).scale(0.9).next_to(third, UP, buff=0.12)
        lab = Tex(r"arms, strongest first", color=GREY_B).scale(0.5).next_to(ch, DOWN, buff=0.15)
        self.play(FadeIn(ch), FadeIn(lab))
        self.play(third.animate.set_fill(JUNC), Write(j3))
        ring = Circle(radius=12 * jf.unit, stroke_color=JUNC, stroke_width=5).move_to(cpt)
        typ = Tex(r"3-way", color=JUNC).scale(0.7).next_to(ring, UP + LEFT, buff=0.05)
        self.play(Create(ring), FadeIn(typ))
        self.wait(2.0)
        self.play(*[FadeOut(x) for x in (tim, disc, arms, ch, j3, lab, ring, typ)], run_time=0.6)

        # ------------------------------------------------ 6. fork or crossing
        F = fork_and_crossing()
        g = Group()
        for key, name in (("cross", "crossing: the ODs add"), ("fork", "fork: the lumens merge")):
            ff = PixelFrame(TOY, 3.4, ORIGIN)
            im = image(F[key], ff.height, "magma", vmin=-0.02, vmax=0.26)
            g.add(Group(im, Tex(name).scale(0.55).next_to(im, UP, buff=0.12)))
        g.arrange(RIGHT, buff=0.6).to_edge(LEFT, buff=0.6).shift(UP * 0.6)
        ax = Axes(x_range=[-36, 36, 12], y_range=[0, 0.26, 0.1], x_length=4.2, y_length=2.6,
                  axis_config=dict(color=MUTED, include_tip=False, include_ticks=False)).to_edge(RIGHT, buff=0.5).shift(UP * 0.4)
        axl = Tex(r"OD along one vessel", color=GREY_B).scale(0.5).next_to(ax, DOWN, buff=0.12)
        pc = ax.plot_line_graph(F["s"], F["p_cross"], line_color=BLUE_B, add_vertex_dots=False, stroke_width=4)
        pk = ax.plot_line_graph(F["s"], F["p_fork"], line_color=JUNC, add_vertex_dots=False, stroke_width=4)
        self.say(r"Vessels crossing at different depths add their blood; at a fork the lumens merge. The centre's darkness tells them apart.")
        self.play(FadeIn(g), Create(ax), FadeIn(axl))
        self.play(Create(pk), run_time=1.2)
        self.play(Create(pc), run_time=1.2)
        bump = Tex(r"sum", color=BLUE_B).scale(0.55).next_to(ax.c2p(0, F["p_cross"].max()), UP, buff=0.08)
        self.play(FadeIn(bump))
        self.wait(2.5)
        self.play(*[FadeOut(x) for x in (g, ax, axl, pc, pk, bump)], run_time=0.6)

        # ------------------------------------------------ 7. co-circularity, then attention
        R, gap = 2.2, 0.55
        circ = Circle(radius=R, stroke_color=MUTED, stroke_width=2).move_to(LEFT * 3.4 + DOWN * 0.1)
        dashed = DashedVMobject(circ, num_dashes=60)
        a0, a1 = 200 * DEGREES, 340 * DEGREES - gap
        b0 = 340 * DEGREES + gap
        arc_a = Arc(radius=R, start_angle=a0, angle=a1 - a0, arc_center=circ.get_center(), stroke_color=WHITE, stroke_width=7)
        arc_b = Arc(radius=R, start_angle=b0, angle=40 * DEGREES, arc_center=circ.get_center(), stroke_color=WHITE, stroke_width=7)
        pa, pb = arc_a.get_end(), arc_b.get_start()
        chord = DashedLine(pa, pb, color=CONT)
        rule = MathTex(r"\varphi_a + \varphi_b = 0", color=CONT).scale(0.9).next_to(circ, UP, buff=0.15)
        self.say(r"Grouping: two ends continue each other when they lie on one circle. Attention then traces one vessel at a time.")
        self.play(Create(arc_a), Create(arc_b))
        ha = Arrow(pa, pa + 0.9 * normalize(rotate_vector(pa - circ.get_center(), PI / 2)), buff=0, color=CONT, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        hb = Arrow(pb, pb + 0.9 * normalize(rotate_vector(pb - circ.get_center(), -PI / 2)), buff=0, color=CONT, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        self.play(GrowArrow(ha), GrowArrow(hb))
        self.play(Create(dashed), Create(chord), Write(rule), run_time=1.4)
        self.wait(1.0)
        L = labels()
        lf = PixelFrame(L["od"].shape, 4.8, RIGHT * 3.4 + UP * 0.1)
        lod = image(L["od"], lf.height, "magma", vmin=-0.02, vmax=0.3).move_to(lf.centre)
        self.play(FadeIn(lod))
        d = ValueTracker(0.0)
        la = always_redraw(lambda: label_image(L["da"], d.get_value(), ATTN, lf.height).move_to(lf.centre))
        self.add(la)
        self.play(d.animate.set_value(90), run_time=2.5, rate_func=linear)
        la.clear_updaters()
        d2 = ValueTracker(0.0)
        lb = always_redraw(lambda: label_image(L["db"], d2.get_value(), TEAL_B, lf.height).move_to(lf.centre))
        self.add(lb)
        self.play(d2.animate.set_value(75), run_time=1.8, rate_func=linear)
        lb.clear_updaters()
        self.wait(1.5)
        self.play(*[FadeOut(x) for x in (arc_a, arc_b, ha, hb, dashed, chord, rule, lod, la, lb)], run_time=0.6)

        # ------------------------------------------------ 8. feedback, and the result
        avg2 = image(S.view(S.average), pf.height, "grey").move_to(pf.centre)
        runs2 = D.runs_overlay(S.truth, pf, min_len_px=25, width=2.0)
        rings2 = D.junctions_overlay(S.truth, pf, width=2.0)
        self.say(r"The render's prediction feeds back to find missing arms. The result: every vessel with an identity, every junction typed.")
        self.play(FadeIn(avg2), run_time=0.8)
        self.play(LaggedStart(*[Create(r) for r in runs2], lag_ratio=0.01), LaggedStart(*[FadeIn(r) for r in rings2], lag_ratio=0.01), run_time=2.5)
        self.wait(2.5)
        self.unsay()
        title = Tex(r"Recurrent mid-level vision for the vessel annotator").scale(0.95)
        sub = Tex(r"the very short version", color=FB).scale(0.6)
        tg = VGroup(title, sub).arrange(DOWN, buff=0.25)
        back = BackgroundRectangle(tg, color=BLACK, fill_opacity=0.75, buff=0.35)
        self.play(avg2.animate.set_opacity(0.35), runs2.animate.set_stroke(opacity=0.35), rings2.animate.set_stroke(opacity=0.35),
                  FadeIn(back), Write(title), FadeIn(sub), run_time=1.5)
        self.wait(3.0)
        self.outro()
