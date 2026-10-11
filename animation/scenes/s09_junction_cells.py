"""Chapter 9: junction cells.

V2 junction cells (spec 4.3; the problem, 2.2): at every pixel, the arms meeting there are
gathered from a small disc, from end cells whose free side points in and from lines
passing through; J3, the third-largest arm, is the junction's evidence.  Level 1: the disc,
the sorted arms, typed rings on a toy scene.  Level 2: the old way against the new, the J3
map with non-maximum suppression, the corner analogy.  Level 3: the typing table, the build
(MICA layer 08) and its numbers, the remaining rule, the phase-1 gate.

Render:  manim -ql scenes/s09_junction_cells.py JunctionCellsShort JunctionCellsMedium JunctionCellsDetailed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import v2cells as V  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import CONT, END, INH, JUNC, MUTED, PixelFrame, image  # noqa: E402
from mica_anim.v2draw import TYPE_COLOR, TYPE_LABEL, arm_glyph, junction_ring, overlay_image, tiny  # noqa: E402
from mica_anim.vsdata import JUNCTION_COLORS  # noqa: E402

T_TRACK = 20.0                     # the toy tracer follows a crest while its cell stays above 20 noise units
CROP = (22, 38, 58, 58)            # x0, y0, w, h of the zoom on the T, in scene-image pixels


def _arm_dir(phi):
    return V.scene_vec(phi)


class JunctionCells(ChapterScene):
    NUMBER, TITLE, SPEC = 9, "Junction cells", "spec 4.3, 2.2"
    MAP_KEYS = ("junction",)

    def construct(self):
        self.intro()
        self.T = V.toy_scene()
        self.J = self.T["J"]
        self.pf = PixelFrame(V.SCENE_SHAPE, 4.5, np.array([0.0, 0.55, 0.0]))
        self.scene_and_old_way()
        self.the_disc()
        self.typed_rings()
        if self.at(2):
            self.j3_map()
            self.corner_analogy()
        if self.at(3):
            self.typing_table()
            self.as_built()
        self.outro()

    # ================================================================ the scene, and why tracing loses junctions
    def scene_and_old_way(self):
        T, pf = self.T, self.pf
        od = image(T["od"], pf.height, "magma", vmin=-0.02, vmax=0.3).move_to(pf.centre)
        frame = SurroundingRectangle(od, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"the toy scene (optical density)").scale(0.55).next_to(frame, UP, buff=0.12)
        self.play(FadeIn(od), Create(frame), FadeIn(head), run_time=1.0)
        self.say(r"A junction is where three or more vessel arms meet. Can a cell see one without tracing the vessels?", wait="read")
        self.scene = Group(od, frame, head)
        self.od = od

        # zoom on the T
        x0, y0, w, h = CROP
        zf = PixelFrame((h, w), 4.6, np.array([-3.55, 0.45, 0.0]))
        crop_rect = Rectangle(width=w * pf.unit, height=h * pf.unit, stroke_color=GREY_A, stroke_width=2)
        crop_rect.move_to(pf.point(x0 + (w - 1) / 2, y0 + (h - 1) / 2))
        zim = image(T["od"][y0:y0 + h, x0:x0 + w], zf.height, "magma", vmin=-0.02, vmax=0.3).move_to(zf.centre)
        zframe = SurroundingRectangle(zim, buff=0, color=GREY_A, stroke_width=1.5)
        self.play(Create(crop_rect), run_time=0.5)
        self.play(FadeOut(self.scene), ReplacementTransform(crop_rect, zframe), FadeIn(zim), run_time=1.2)

        def zp(x, y):
            return zf.point(x - x0, y - y0)

        def inside(pts):
            return np.array([q for q in pts if x0 <= q[0] <= x0 + w - 1 and y0 <= q[1] <= y0 + h - 1])

        self.zf, self.zp, self.zim, self.zframe = zf, zp, zim, zframe

        # the old way: traces (computed by a toy tracer); the branch's stops short
        branch = inside(V.trace_along(T, V._at(V.T_C, -1, 38), V.T_C, T_TRACK))
        parent = inside(V.trace_along(T, V._at(V.T_C, -9, 58), V._at(V.T_C, 7, 60), T_TRACK))

        def poly(pts):
            m = VMobject(stroke_color=WHITE, stroke_width=4)
            m.set_points_as_corners([zp(*q) for q in pts])
            return m

        tb, tp = poly(branch), poly(parent)
        foot = np.array(V.T_C)
        gap = DashedLine(zp(*branch[-1]), zp(*foot), color=INH, stroke_width=4, dash_length=0.06)
        cross = VGroup(Line(UL, DR), Line(UR, DL)).set_stroke(INH, 5).scale(0.13).move_to(zp(*foot) + UP * 0.55 + RIGHT * 0.55)
        nolab = tiny(r"no junction", INH, 0.5).next_to(cross, RIGHT, buff=0.1)
        trl = tiny(r"traces (toy tracer)", GREY_A, 0.48).next_to(zframe, RIGHT, buff=0.25).align_to(zframe, UP)
        if self.short:
            self.say(r"Today a junction exists only where one trace reaches another. This branch's trace stops short: the junction is lost.")
        else:
            self.say(r"Today a junction exists only where one trace reaches another. This branch fades by its parent; its trace stops short.")
        self.play(Create(tp), FadeIn(trl), run_time=1.0)
        self.play(Create(tb), run_time=1.0)
        self.play(Create(gap), FadeIn(cross, scale=1.5), FadeIn(nolab), run_time=0.8)
        old = VGroup(tb, tp, gap, cross, nolab, trl)
        if self.at(2):
            step7 = tiny(r"old step 7: a junction only where a trace ends on another trace", GREY_A, 0.48)
            step7.next_to(zframe, DOWN, buff=0.15).align_to(zframe, LEFT)
            self.play(FadeIn(step7))
            old.add(step7)
            self.say(r"So the junction is lost. The same happens when a trace is pruned, or never reaches its parent (spec 2.2).", wait="read")
        else:
            self.wait(1.5)
        self.play(FadeOut(old), run_time=0.6)

    # ================================================================ the disc, the arms, J3
    def the_disc(self):
        T, J = self.T, self.J
        zf, zp, zim, zframe = self.zf, self.zp, self.zim, self.zframe
        jx, jy, j3, kind, arms = next(j for j in T["junctions"] if j[3] == "T")
        p = zp(jx, jy)
        rho = V.rho()
        disc = Circle(radius=rho * zf.unit, stroke_color=JUNC, stroke_width=2.5).move_to(p)
        disc.set_fill(JUNC, opacity=0.08)
        dot = Dot(p, radius=0.05, color=WHITE)
        plab = MathTex("p").scale(0.6).next_to(dot, DOWN + LEFT, buff=0.04)
        self.say(r"A junction cell at $p$ gathers arms from a small disc: ends that point into $p$, and lines that pass through it.")
        self.play(FadeIn(dot, scale=2), FadeIn(plab), Create(disc), run_time=0.9)

        # ends pointing into p: the ray from the end cell (orange)
        rays, passes = VGroup(), VGroup()
        for phi, s, k in arms:
            if k == "end":
                j = (int(round(phi / (np.pi / 16))) + 16) % 32
                qx, qy, _ = V.ray_source(T["E"], jx, jy, j)
                q = zp(qx, qy)
                out = q + 0.55 * V.scene_vec(V.PHIS[(j + 16) % 32])      # back along the branch
                rays.add(VGroup(Arrow(out, p, buff=0.06, color=END, stroke_width=5, max_tip_length_to_length_ratio=0.2),
                                Dot(q, radius=0.045, color=END)))
        pass_phis = [phi for phi, s, k in arms if k == "pass"]
        if pass_phis:
            d = _arm_dir(pass_phis[0])
            L = (rho + 9) * zf.unit
            passes.add(DoubleArrow(p - d * L, p + d * L, buff=0, color=CONT, stroke_width=5, tip_length=0.18,
                                   max_tip_length_to_length_ratio=0.1))
        kl = VGroup(VGroup(Arrow(LEFT * 0.35, ORIGIN, buff=0, color=END, stroke_width=4, max_tip_length_to_length_ratio=0.4),
                           tiny(r"an end pointing in", END, 0.48)).arrange(RIGHT, buff=0.12),
                    VGroup(Line(LEFT * 0.35, ORIGIN, stroke_color=CONT, stroke_width=4),
                           tiny(r"a line passing through", CONT, 0.48)).arrange(RIGHT, buff=0.12)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(zframe, DOWN, buff=0.15).align_to(zframe, LEFT)
        self.play(LaggedStart(GrowArrow(rays[0][0]) if len(rays) else Wait(0.1), *[FadeIn(r[1]) for r in rays], lag_ratio=0.3),
                  FadeIn(kl[0]), run_time=1.0)
        self.play(GrowFromCenter(passes), FadeIn(kl[1]), run_time=1.0)
        if self.at(2):
            dl = np.array([-0.7071, -0.7071, 0.0])
            rad = Line(p, p + dl * rho * zf.unit, stroke_color=JUNC, stroke_width=3)
            rtx = MathTex(r"\rho = r_0 + w/2", color=JUNC).scale(0.6)
            rtx.next_to(rad.get_end(), DOWN + LEFT, buff=0.05)
            rl = VGroup(rad, BackgroundRectangle(rtx, fill_opacity=0.7, buff=0.04), rtx)
            self.say(r"The disc's radius is $\rho = r_0 + w/2$: 4 px beyond the vessel's half-width, so a branch may stop a little short.")
            self.play(Create(rad), FadeIn(rl[1:]))
            self.wait(1.5)
            self.extra_disc = rl
        else:
            self.extra_disc = VGroup()
        self.wait(0.6)

        # --- the arms, sorted, as live bars
        base_y = -1.75
        names = [r"J_1", r"J_2", r"J_3", r"J_4", r"J_5"]
        xs = np.linspace(1.6, 5.6, 5)
        vmax, hmax, bw = 40.0, 3.4, 0.55
        base = Line([xs[0] - 0.6, base_y, 0], [xs[-1] + 0.6, base_y, 0], stroke_color=MUTED, stroke_width=2)
        labels = VGroup(*[MathTex(n, color=(JUNC if i == 2 else GREY_A)).scale(0.7).move_to([x, base_y - 0.32, 0])
                          for i, (n, x) in enumerate(zip(names, xs))])
        tag = tiny(r"arm strengths at $p$, sorted (toy, noise units)", GREY_B, 0.48).move_to([3.6, base_y - 0.8, 0])
        probe = ValueTracker(0.0)
        par_dir = V.unit(pass_phis[0]) if pass_phis else np.array([0.0, 1.0])
        if par_dir[1] < 0:
            par_dir = -par_dir                                            # walk down the parent

        def here():
            q = np.array([jx, jy]) + probe.get_value() * par_dir
            return int(round(q[0])), int(round(q[1]))

        def bars():
            hx, hy = here()
            a = V.arms_at(J, hx, hy)
            g = VGroup()
            for i in range(5):
                v = a[i][1] if i < len(a) else 0.0
                col = (CONT if a[i][2] == "pass" else END) if i < len(a) else GREY_B
                hgt = max(hmax * min(v, vmax) / vmax, 1e-3)
                r = Rectangle(width=bw, height=hgt, stroke_width=0, fill_color=col, fill_opacity=0.9)
                r.move_to([xs[i], base_y + hgt / 2, 0])
                g.add(r)
            j3v = a[2][1] if len(a) > 2 else 0.0
            ring = SurroundingRectangle(g[2], buff=0.05, color=JUNC, stroke_width=3) if j3v > 0 else \
                Line([xs[2] - 0.35, base_y + 0.04, 0], [xs[2] + 0.35, base_y + 0.04, 0], stroke_color=JUNC, stroke_width=5)
            val = MathTex(rf"J_3 = {j3v:.0f}", color=JUNC).scale(0.7).move_to([xs[2], base_y + hmax + 0.35, 0])
            return VGroup(g, ring, val)

        def rosette():
            hx, hy = here()
            c = zp(hx, hy)
            g = VGroup()
            for phi, s, k in V.arms_at(J, hx, hy):
                d = _arm_dir(phi)
                L = 0.025 * s + 0.15
                if k == "pass":
                    g.add(Line(c, c + d * L, stroke_color=CONT, stroke_width=5))
                else:
                    g.add(Arrow(c + d * L, c, buff=0, color=END, stroke_width=5, max_tip_length_to_length_ratio=0.3))
            g.add(Dot(c, radius=0.05, color=WHITE))
            return g

        live_bars = always_redraw(bars)
        self.say(r"Sort the arms. A junction needs three, so its evidence is the third: $J_3$.")
        self.play(Create(base), FadeIn(labels), FadeIn(tag), FadeIn(live_bars), run_time=1.2)
        self.wait(1.6)
        live_ros = always_redraw(rosette)
        moving_disc = always_redraw(lambda: Circle(radius=rho * zf.unit, stroke_color=JUNC, stroke_width=2.5)
                                    .set_fill(JUNC, opacity=0.08).move_to(zp(*here())))
        self.play(FadeOut(rays), FadeOut(passes), FadeOut(dot), FadeOut(plab), FadeOut(disc), FadeOut(self.extra_disc),
                  FadeIn(live_ros), FadeIn(moving_disc), run_time=0.8)
        self.add(moving_disc)
        self.say(r"Slide $p$ along the parent: only the line passes here, two arms, so $J_3 = 0$.")
        self.play(probe.animate.set_value(14.0), run_time=2.2 if self.short else 3.0, rate_func=smooth)
        self.wait(0.5 if self.short else 0.8)
        self.play(probe.animate.set_value(0.0), run_time=1.3 if self.short else 2.0, rate_func=smooth)
        self.wait(0.3)
        for m in (live_bars, live_ros, moving_disc):
            m.clear_updaters()
        self.zoom = Group(zim, zframe, kl, live_ros, moving_disc)
        self.barsg = VGroup(base, labels, tag, live_bars)

    # ================================================================ typed rings on the scene
    def typed_rings(self):
        T, pf = self.T, self.pf
        self.play(FadeOut(self.zoom), FadeOut(self.barsg), run_time=0.7)
        self.play(FadeIn(self.scene), run_time=0.8)
        rings, labels, glyphs = VGroup(), VGroup(), VGroup()
        for x, y, v, kind, arms in T["junctions"]:
            ring = junction_ring(pf, x, y, kind, radius_px=11)
            rings.add(ring)
            lab = Tex(TYPE_LABEL[kind], color=TYPE_COLOR[kind]).scale(0.55)
            side = {"T": LEFT, "Y": DOWN + RIGHT, "crossing": UP}.get(kind, DOWN)
            lab.next_to(ring, side, buff=0.08)
            labels.add(VGroup(BackgroundRectangle(lab, fill_opacity=0.7, buff=0.05), lab))
            g = arm_glyph(["pass" if a[2] == "pass" else "end" for a in arms], size=0.32,
                          phis=[np.arctan2(_arm_dir(a[0])[1], _arm_dir(a[0])[0]) for a in arms])
            glyphs.add(g)
        self.say(r"Peaks of $J_3$ are junctions, and their arms give the type: a line and an end, a T; three ends, a Y; two lines, a crossing.")
        if self.short:
            self.play(LaggedStart(*[AnimationGroup(Create(r), FadeIn(l)) for r, l in zip(rings, labels)], lag_ratio=0.35), run_time=1.8)
            self.wait(1.6)
        else:
            for r, l in zip(rings, labels):
                self.play(Create(r), FadeIn(l, shift=UP * 0.1), run_time=0.7)
            self.wait(1.0)
        # the arms of each, as small glyphs under the frame
        row = VGroup()
        for g, (x, y, v, kind, arms) in zip(glyphs, T["junctions"]):
            row.add(VGroup(g, Tex(TYPE_LABEL[kind], color=TYPE_COLOR[kind]).scale(0.45)).arrange(DOWN, buff=0.1))
        order = sorted(range(len(row)), key=lambda i: T["junctions"][i][0])
        row = VGroup(*[row[i] for i in order]).arrange(RIGHT, buff=1.2).next_to(self.scene[1], DOWN, buff=0.18)
        if self.at(2):
            self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in row], lag_ratio=0.2), run_time=1.0)
        else:
            row = VGroup()
        tring = rings[[k for k, j in enumerate(T["junctions"]) if j[3] == "T"][0]]
        self.say(r"Including the T that tracing lost: its arms come from the field around the point, not from traces that reached it.")
        self.play(Indicate(tring, color=JUNC, scale_factor=1.3), run_time=1.2)
        self.wait(1.0 if self.short else 1.6)
        self.rings, self.ring_labels, self.glyph_row = rings, labels, row

    # ================================================================ level 2: the J3 map, NMS
    def j3_map(self):
        T, pf = self.T, self.pf
        self.play(FadeOut(self.glyph_row), FadeOut(self.ring_labels), FadeOut(self.rings), run_time=0.6)
        J3 = T["J"]["J3"]
        jm = overlay_image(J3, T["od"], pf.height, "magma", 0, 24, under_max=0.3).move_to(pf.centre)
        head = self.scene[2]
        head2 = Tex(r"$J_3$ at every pixel (toy)").scale(0.55).move_to(head)
        self.say(r"$J_3$ at every pixel: zero along every line (two arms), bright only where three arms meet.")
        self.play(FadeIn(jm), FadeTransform(head, head2), run_time=1.4)
        self.wait(1.6)
        peaks = VGroup()
        for x, y, v, kind, arms in T["junctions"]:
            c = pf.point(x, y)
            peaks.add(VGroup(Circle(radius=V.R0 * pf.unit, stroke_color=WHITE, stroke_width=2).move_to(c),
                             Dot(c, radius=0.035, color=WHITE)))
        nms = tiny(rf"local maxima above $\tau_J$, at least $r_0 = {V.R0:.0f}$ px apart (toy $\tau_J = {V.TOY_TAU_J:.0f}$)", GREY_B, 0.48)
        nms.next_to(self.scene[1], DOWN, buff=0.15)
        self.say(r"Non-maximum suppression keeps one peak per junction: local maxima above a threshold, at least $r_0$ apart.")
        self.play(LaggedStart(*[GrowFromCenter(p) for p in peaks], lag_ratio=0.3), FadeIn(nms), run_time=1.4)
        self.wait(1.8)
        self.play(FadeOut(jm), FadeOut(peaks), FadeOut(nms), FadeTransform(head2, head), run_time=0.8)

    # ================================================================ level 2: like a corner detector
    def corner_analogy(self):
        T = self.T
        self.clear(run_time=0.8)
        img, l1, l2 = V.corner_maps()
        J1, J3 = T["J"]["vals"][0], T["J"]["J3"]
        H = 1.95
        cols = [-2.3, 2.3]
        rows = [1.75, -1.0]
        panels = Group()
        specs = [(l1, img, 1.0, 0, np.percentile(l1, 99.7), rows[0], cols[0], r"$\lambda_1$: the stronger direction"),
                 (l2, img, 1.0, 0, np.percentile(l2, 99.9), rows[0], cols[1], r"$\lambda_2$: the weaker, needs two"),
                 (J1, T["od"], 0.3, 0, 40, rows[1], cols[0], r"$J_1$: the strongest arm"),
                 (J3, T["od"], 0.3, 0, 24, rows[1], cols[1], r"$J_3$: the third, needs three")]
        for arr, under, umax, lo, hi, y, x, lab in specs:
            im = overlay_image(arr, under, H, "magma", lo, hi, under_max=umax).move_to([x, y, 0])
            fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
            t = Tex(lab).scale(0.48).next_to(fr, UP, buff=0.08)
            panels.add(Group(im, fr, t))
        rl = VGroup(Tex(r"corners\\(structure\\tensor)", color=GREY_A).scale(0.5),
                    Tex(r"junctions\\(arms)", color=GREY_A).scale(0.5))
        rl[0].next_to(panels[0], LEFT, buff=0.3)
        rl[1].next_to(panels[2], LEFT, buff=0.3)
        self.say(r"A corner detector reads the structure tensor's 2nd eigenvalue: an edge has one direction, a corner two.")
        self.play(FadeIn(panels[0]), FadeIn(rl[0]), run_time=0.9)
        self.play(FadeIn(panels[1]), run_time=0.9)
        self.wait(1.6)
        self.say(r"A junction cell reads the 3rd-largest arm: a line has two arms, a junction three.")
        self.play(FadeIn(panels[2]), FadeIn(rl[1]), run_time=0.9)
        self.play(FadeIn(panels[3]), run_time=0.9)
        self.wait(2.2)

    # ================================================================ level 3: the typing table
    def typing_table(self):
        self.clear(run_time=0.8)
        title = Tex(r"Typing, first pass (spec 4.3)").scale(0.7).to_edge(UP, buff=0.3)
        rows_spec = [
            ("3", ["pass", "pass", "end"], [PI / 2, -PI / 2, 0.0], r"one passing line + one end", "T"),
            ("3", ["end", "end", "end"], [-PI / 2, PI / 2 + 0.7, PI / 2 - 0.7], r"three ends", "Y"),
            ("4", ["pass"] * 4, [0.6, 0.6 + PI, -0.6, PI - 0.6], r"two passing lines, $\ge 15^\circ$ apart", "crossing"),
            ("4", ["end"] * 4, [0.6, 0.6 + PI, -0.6, PI - 0.6], r"four ends that pair into two lines", "crossing (broken)"),
            ("4", ["pass", "pass", "end", "end"], [PI / 2, -PI / 2, 0.3, -0.5], r"otherwise", "compound"),
            ("5+", ["pass", "pass", "end", "end", "end"], [PI / 2, -PI / 2, 0.2, -0.9, PI - 0.4], r"", "compound"),
        ]
        type_text = {"T": r"3-way (T)", "Y": r"3-way (Y)", "crossing": r"crossing candidate",
                     "crossing (broken)": r"crossing candidate (broken at the centre)", "compound": r"compound"}
        hdr = VGroup(Tex(r"arms", color=GREY_B).scale(0.5), Tex(r"pattern", color=GREY_B).scale(0.5), Tex(r"type", color=GREY_B).scale(0.5))
        X = [-5.6, -4.4, -1.9, 1.5]
        y0, dy = 2.55, 0.7
        hdr[0].move_to([X[0], y0, 0]); hdr[1].move_to([X[2] - 1.0, y0, 0]); hdr[2].move_to([X[3], y0, 0], aligned_edge=LEFT)
        rule = Line([-6.2, y0 - 0.25, 0], [6.6, y0 - 0.25, 0], stroke_color=MUTED, stroke_width=1)
        self.play(FadeIn(title), FadeIn(hdr), Create(rule))
        self.say(r"The arms decide the type. Three arms: a line and an end make a T, three ends a Y.")
        rows = VGroup()
        for i, (n, kinds, phis, pat, typ) in enumerate(rows_spec):
            y = y0 - 0.65 - i * dy
            g = arm_glyph(kinds, size=0.27, phis=phis).move_to([X[1], y, 0])
            r = VGroup(MathTex(n).scale(0.65).move_to([X[0], y, 0]), g,
                       Tex(pat).scale(0.5).move_to([X[2], y, 0], aligned_edge=LEFT).shift(LEFT * 1.4) if pat else VGroup(),
                       Tex(type_text[typ], color=TYPE_COLOR[typ]).scale(0.52).move_to([X[3], y, 0], aligned_edge=LEFT))
            rows.add(r)
        for i, r in enumerate(rows):
            if i == 2:
                self.say(r"Four arms: two passing lines make a crossing; four ends that pair by good continuation, a crossing broken at its centre.")
            if i == 4:
                self.say(r"Anything else is a compound junction, and so is any point with five arms or more.")
            self.play(FadeIn(r, shift=RIGHT * 0.15), run_time=0.7)
            self.wait(0.9 if i in (1, 3, 5) else 0.4)
        flag = VGroup(tiny(r"refined later by good continuation (4.4) and transparency (4.5)", GREY_B, 0.46),
                      tiny(r"step 7's compound radius prior ($\ge 16$ px) stays, as a separate flag", GREY_B, 0.46)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        flag.next_to(rows, DOWN, buff=0.25).align_to(rows, LEFT)
        assert flag.get_bottom()[1] > -3.0
        self.play(FadeIn(flag))
        self.wait(2.0)

    # ================================================================ level 3: as built
    def as_built(self):
        T = self.T
        self.clear(run_time=0.8)
        title = Tex(r"As built: MICA layer 08 (spec 4.3)").scale(0.7).to_edge(UP, buff=0.35)
        self.play(FadeIn(title))
        ok = VGroup(*[VGroup(Tex(t).scale(0.55), MathTex(r"\checkmark", color=GREEN_C).scale(0.7)).arrange(RIGHT, buff=0.15)
                      for t in (r"synthetic T", r"synthetic Y", r"synthetic crossing")]).arrange(RIGHT, buff=0.6)
        ok.move_to([0, 2.55, 0])
        okl = tiny(r"typed correctly (MICA's synthetic scenes; this chapter's toy too)", GREY_B, 0.46).next_to(ok, DOWN, buff=0.1)
        self.say(r"Built in MICA, the junction cells type the synthetic T, Y and crossing correctly, as this toy does.")
        self.play(FadeIn(ok), FadeIn(okl))
        self.wait(1.5)

        # the real test image: 444 junction cells, by type
        counts = [("crossing", 227), ("3-way", 162), ("compound", 55)]
        cols = [JUNCTION_COLORS["crossing"], JUNC, JUNCTION_COLORS["compound"]]
        total = sum(c for _, c in counts)
        W, H = 10.0, 0.55
        x = -W / 2
        bar = VGroup()
        labs = VGroup()
        for (name, c), col in zip(counts, cols):
            w = W * c / total
            r = Rectangle(width=w, height=H, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x + w / 2, 0.85, 0])
            bar.add(r)
            labs.add(Tex(rf"{c} {name}", color=col).scale(0.5).next_to(r, DOWN, buff=0.1))
            x += w
        head = VGroup(Tex(r"the real test image: 444 junction cells; 425 after scale selection").scale(0.55),
                      tiny(r"measured, spec 4.3", GREY_B, 0.46)).arrange(DOWN, buff=0.08)
        head.next_to(bar, UP, buff=0.18)
        self.say(r"On the real test image: 444 junction cells, 227 crossings, 162 three-way, 55 compound; 425 after scale selection.")
        self.play(FadeIn(head), LaggedStart(*[GrowFromEdge(r, LEFT) for r in bar], lag_ratio=0.3), run_time=1.6)
        self.play(FadeIn(labs))
        self.wait(1.6)
        ten = VGroup(Tex(r"10 lie more than 15 px from any centreline: branches whose traces stopped", color=GREY_A).scale(0.5),
                     Tex(r"short (what this layer exists to find), or noise. Not yet checked against truth.", color=GREY_A).scale(0.5)
                     ).arrange(DOWN, buff=0.1)
        ten.next_to(labs, DOWN, buff=0.3)
        self.play(FadeIn(ten), run_time=0.8)
        self.say(r"Ten lie more than 15 px from any centreline: branches whose traces stopped short, which this layer exists to find, or noise.",
                 wait="read")
        self.wait(1.0)
        self.clear(run_time=0.8)

        # one rule remains: the third arm at least 15 % of the strongest
        rule = MathTex(r"J_3", r"\ \ge\ ", r"0.15", r"\,", r"J_1", r"\quad\longrightarrow\quad",
                       r"{J_3", r"\over", r"J_1", r"+", r"c}").scale(0.85).move_to([0, 2.2, 0])
        rule[0].set_color(JUNC); rule[6].set_color(JUNC)
        rl = VGroup(tiny(r"the one rule left (as built)", GREY_B, 0.48).next_to(rule[:5], UP, buff=0.25),
                    tiny(r"candidate divisive form", GREY_B, 0.48).next_to(rule[6:], UP, buff=0.25))
        self.say(r"Besides firing thresholds, one rule remains: the third arm must be at least 15\% of the strongest. A divisive form could replace it.")
        self.play(Write(rule[:5]), FadeIn(rl[0]))
        self.play(Write(rule[5:]), FadeIn(rl[1]))
        # the toy's ratios at its three junctions
        tbl = VGroup()
        for x, y, v, kind, arms in sorted(T["junctions"], key=lambda j: j[0]):
            r = v / arms[0][1]
            tbl.add(VGroup(Tex(TYPE_LABEL[kind], color=TYPE_COLOR[kind]).scale(0.52),
                           MathTex(rf"J_3/J_1 = {r:.2f}").scale(0.6)).arrange(DOWN, buff=0.12))
        tbl.arrange(RIGHT, buff=1.0).move_to([0, 0.75, 0])
        tl = tiny(r"this chapter's toy junctions", GREY_B, 0.46).next_to(tbl, DOWN, buff=0.15)
        self.play(FadeIn(tbl), FadeIn(tl))
        self.wait(2.0)

        # the phase-1 gate
        gate = VGroup(
            Tex(r"phase-1 gate (spec 6.2)", color=GREY_A).scale(0.6),
            Tex(r"3-way recall up by at least 15 points, at equal or better precision").scale(0.52),
            Tex(r"false junctions on the \emph{empty}, \emph{end} and background probes: not up").scale(0.52),
            Tex(r"not yet measured against truth", color=INH).scale(0.55),
        ).arrange(DOWN, buff=0.18).move_to([0, -1.55, 0])
        box = SurroundingRectangle(gate, buff=0.2, color=MUTED, stroke_width=1.5, corner_radius=0.1)
        self.say(r"The gate for this phase: 3-way recall up 15 points with no more false junctions. It has not yet been measured against truth.",
                 wait="read")
        self.play(Create(box), FadeIn(gate), run_time=1.2)
        self.wait(1.5)


variants(JunctionCells, module_globals=globals())
