"""Chapter 7: horizontal connections (spec 4.7).

Level 1: Field, Hayes and Hess's path among distractors; the association field;
         the bipole sqrt(A+ A-).
Level 2: today's step 5, then the spec's recurrent field built term by term and
         run live on a toy image (8 iterations, convergence).
Level 3: built and switched off: 1-px lobes slide off the crest of misaligned
         vessels; the spec's measurements beside the toy's; lobes with width.

Everything is computed by mica_anim/horizontal.py.

Render:  manim -ql scenes/s07_association_field.py AssociationFieldMedium
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import horizontal as H  # noqa: E402
from mica_anim import neuro as N  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import CONT, EXC, FB, INH, MUTED, PixelFrame, colormap, image, orientation_color  # noqa: E402

VES_COL, DASH_COL, WEB_COL = WHITE, TEAL_B, PURPLE_A


def seg(P, theta_img, length, **kw):
    """A line element at scene point P with image orientation theta (y down)."""
    a = -theta_img
    d = np.array([np.cos(a), np.sin(a), 0.0]) * length / 2
    return Line(P - d, P + d, **kw)


def tex_table(rows, widths, scale=0.5, header_color=GREY_B, row_buff=0.2):
    lines = VGroup()
    for i, row in enumerate(rows):
        lines.add(VGroup(*[Tex(t, color=header_color if i == 0 else WHITE).scale(scale) for t in row]))
    for row in lines:
        x = 0.0
        for c, w in zip(row, widths):
            c.move_to(np.array([x, 0, 0]), aligned_edge=LEFT)
            x += w
    lines.arrange(DOWN, aligned_edge=LEFT, buff=row_buff)
    for row in lines:
        x = lines.get_left()[0]
        for c, w in zip(row, widths):
            c.move_to(np.array([x, row.get_y(), 0]), aligned_edge=LEFT)
            x += w
    rule = Line(LEFT, RIGHT, stroke_color=MUTED, stroke_width=1).set_width(sum(widths))
    rule.next_to(lines[0], DOWN, buff=row_buff / 2).align_to(lines, LEFT)
    return VGroup(lines, rule)


def assoc_glyph(P, theta, unit_len, spacing_scene, color=EXC):
    """The association field around an element: co-circular short segments in a lobe ahead and
    a lobe behind, brighter near the axis (Field, Hayes and Hess 1993)."""
    g = VGroup()
    for sgn in (+1, -1):
        for r in (1.0, 1.7, 2.4, 3.1):
            for phi_deg in range(-30, 31, 10):
                phi = np.deg2rad(phi_deg)
                ang = theta + phi                           # image angle of the position
                q = np.array([np.cos(ang), np.sin(ang)]) * r * sgn
                w = np.exp(-phi_deg ** 2 / (2 * 22.0 ** 2)) * np.exp(-r ** 2 / (2 * 2.2 ** 2))
                pos = P + np.array([q[0], -q[1], 0.0]) * spacing_scene
                g.add(seg(pos, theta + 2 * phi, unit_len, stroke_color=color, stroke_width=4.5,
                          stroke_opacity=0.25 + 0.75 * w))
    return g


class AssociationField(ChapterScene):
    NUMBER, TITLE, SPEC = 7, "Horizontal connections", "spec 4.7"
    MAP_KEYS = ("assoc",)

    def construct(self):
        self.intro()
        self.level1()
        if self.at(2):
            self.level2()
        if self.at(3):
            self.level3()
        self.outro()

    # ================================================================== level 1: a path among distractors
    def level1(self):
        F = H.fhh()
        bip = np.sqrt(F["Ap"] * F["Am"])
        h_px, w_px = int(F["H"]), int(F["W"])
        pf = PixelFrame((h_px, w_px), 5.5, np.array([0.0, 0.5, 0.0]))
        field = image(H.gabor_image(F), pf.height, "grey", vmin=0, vmax=1).move_to(pf.centre)
        field.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
        self.play(FadeIn(field), run_time=1.0)
        self.say(r"A field of short line elements at random orientations. A few of them line up into a path. Can you find it?")
        self.wait(3.5 if self.at(2) else 2.5)

        sp = F["spacing"] * pf.unit
        pts = [pf.point(*q) for q in F["path_pts"]]
        path_curve = VMobject(stroke_color=CONT, stroke_width=10, stroke_opacity=0.25).set_points_smoothly(pts)
        path_els = VGroup(*[seg(pf.point(*F["pos"][i]), F["ori"][i], 0.62 * sp, stroke_color=CONT, stroke_width=5)
                            for i in np.nonzero(F["path"])[0]])
        self.play(Create(path_curve), LaggedStart(*[Create(e) for e in path_els], lag_ratio=0.12), run_time=2.0)
        self.say(r"There. We see it at once because collinear neighbours support each other (Field, Hayes and Hess 1993).")
        self.wait(3.0)

        # the association field around one path element
        idx = np.nonzero(F["path"])[0]
        i0 = idx[len(idx) // 2]
        P0, th0 = pf.point(*F["pos"][i0]), F["ori"][i0]
        glyph = assoc_glyph(P0, th0, 0.42 * sp, sp)
        self.play(field.animate.set_opacity(0.3), path_curve.animate.set_stroke(opacity=0.12),
                  path_els.animate.set_stroke(opacity=0.35), run_time=0.8)
        self.say(r"The association field: a cell is excited by neighbours that continue its line, in a lobe ahead and a lobe behind.")
        self.play(LaggedStart(*[Create(m) for m in glyph], lag_ratio=0.01), run_time=2.0)
        t_dir = np.array([np.cos(-th0), np.sin(-th0), 0.0])
        lab_p = MathTex(r"A^+", color=EXC).scale(0.85).move_to(P0 + t_dir * 3.75 * sp)
        lab_m = MathTex(r"A^-", color=EXC).scale(0.85).move_to(P0 - t_dir * 3.75 * sp)
        for lab in (lab_p, lab_m):
            lab.add_background_rectangle(color=BLACK, opacity=0.6, buff=0.05)
        self.play(FadeIn(lab_p), FadeIn(lab_m))
        self.wait(2.5)

        bip_eq = MathTex(r"\text{bipole}", "=", r"\sqrt{\,", r"A^+", r"\cdot", r"A^-", r"\,}").scale(0.85)
        bip_eq[3].set_color(EXC); bip_eq[5].set_color(EXC)
        box = BackgroundRectangle(bip_eq, color=BLACK, fill_opacity=0.85, buff=0.15)
        eq_g = VGroup(box, bip_eq).move_to(np.array([0.0, -2.55, 0]))
        self.say(r"A \emph{bipole} cell needs support on both sides: the geometric mean is zero if either lobe is empty.")
        self.play(FadeIn(box), Write(bip_eq), run_time=1.5)
        self.wait(3.0)

        # the bipole computed for every element
        vmax = np.percentile(bip[F["path"]], 70)
        els = VGroup(*[seg(pf.point(*F["pos"][i]), F["ori"][i], 0.62 * sp, stroke_color=WHITE, stroke_width=4,
                           stroke_opacity=float(np.clip(0.06 + 0.94 * (bip[i] / vmax) ** 1.5, 0, 1)))
                       for i in range(len(F["pos"]))])
        self.say(r"Computed for every element: the path lights up; its two ends, supported on one side only, stay dim.")
        self.play(FadeOut(glyph), FadeOut(lab_p), FadeOut(lab_m), FadeOut(path_els), FadeOut(path_curve),
                  field.animate.set_opacity(0.12), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(e) for e in els], lag_ratio=0.004), run_time=2.0)
        ends = [idx[0], idx[-1]]
        rings = VGroup(*[Circle(radius=0.45 * sp, stroke_color=CONT, stroke_width=2).move_to(pf.point(*F["pos"][i])) for i in ends])
        self.play(Create(rings), run_time=0.8)
        self.wait(2.0 if self.at(2) else 1.0)
        self.play(*[FadeOut(m) for m in (field, els, rings, eq_g)], run_time=0.8)

    # ================================================================== level 2: the recurrent field, run live
    def level2(self):
        sc = H.toy_scene()
        run = H.toy_run()
        U0, Cs = run["U0"], run["Cs"]

        # today's step 5
        today = MathTex(r"C", r"\;\leftarrow\;", r"\tfrac{1}{2}\big(", r"U^0", "+", r"\text{bipole}(C)", r"\big)").scale(0.85)
        today[5].set_color(EXC)
        today_l = Tex(r"today's step 5: three iterations, excitation only", color=GREY_B).scale(0.55)
        tg = VGroup(today, today_l).arrange(DOWN, buff=0.25).move_to(UP * 2.2)
        self.say(r"Today's step 5 is three iterations of this: excitation only. Nothing pushes back on clutter.")
        self.play(Write(today), FadeIn(today_l), run_time=1.5)
        self.wait(1.5)

        # the spec's recurrent field, term by term
        eq = MathTex(r"C_k", r"\;\leftarrow\;", r"\max\!\Big(0,\ ", r"U^0_k", r"+", r"g_E\,\text{excite}_k", r"-",
                     r"g_P\,\text{parallel}_k", r"-", r"g_R\,\big[\text{rival}_k - C_k\big]_+", r"\Big)", r"\Big/\,(1+g_E)").scale(0.72)
        eq[5].set_color(EXC); eq[7].set_color(INH); eq[9].set_color(INH)
        eq.move_to(UP * 2.2)
        self.say(r"The spec's recurrent field keeps that excitation and adds two kinds of inhibition, inside the loop.")
        self.play(FadeOut(tg, shift=UP * 0.3), run_time=0.7)
        self.play(Write(eq[:4]), Write(eq[10:]), run_time=1.4)

        # pictograms under the terms
        def cell_bar(P, theta=0.0, L=0.5, col=WHITE):
            return seg(P, theta, L, stroke_color=col, stroke_width=6)
        y_pic = 0.45
        px_e, px_p, px_r = -4.3, 0.2, 4.5
        Pe = np.array([px_e, y_pic, 0])
        pic_e = VGroup(cell_bar(Pe),
                       *[Dot(Pe + RIGHT * (0.38 + 0.13 * j), radius=0.035, color=EXC).set_opacity(1 - j / 9) for j in range(8)],
                       *[Dot(Pe + LEFT * (0.38 + 0.13 * j), radius=0.035, color=EXC).set_opacity(1 - j / 9) for j in range(8)])
        lab_e = MathTex(r"\sqrt{\text{ahead}\cdot\text{behind}}", color=EXC).scale(0.5).next_to(pic_e, DOWN, buff=0.25)
        sub_e = Tex(r"of $P_m = \max(C_k,\,0.8\,C_{k\pm1})$", color=GREY_B).scale(0.42).next_to(lab_e, DOWN, buff=0.12)
        Pp = np.array([px_p, y_pic, 0])
        pic_p = VGroup(cell_bar(Pp),
                       Rectangle(width=0.9, height=0.13, stroke_width=0, fill_color=INH, fill_opacity=0.6).move_to(Pp + UP * 0.3),
                       Rectangle(width=0.9, height=0.13, stroke_width=0, fill_color=INH, fill_opacity=0.6).move_to(Pp + DOWN * 0.3))
        lab_p = MathTex(r"\min(F_\text{left},\,F_\text{right})", color=INH).scale(0.5).next_to(pic_p, DOWN, buff=0.25)
        sub_p = Tex(r"the weaker flank: parallel neighbours", color=GREY_B).scale(0.42).next_to(lab_p, DOWN, buff=0.12)
        Pr = np.array([px_r, y_pic, 0])
        spokes = VGroup()
        for j in range(N.N_ORIENT):
            d = abs(((j + 8) % 16) - 8)
            col, op, w = (WHITE, 1.0, 5) if d == 0 else ((INH, 0.9, 4) if d <= 2 else (MUTED, 0.3, 2))
            spokes.add(seg(Pr, N.THETAS[j], 0.75, stroke_color=col, stroke_opacity=op, stroke_width=w))
        lab_r = MathTex(r"\max(C_{k\pm1},\,C_{k\pm2})", color=INH).scale(0.5).next_to(spokes, DOWN, buff=0.25)
        sub_r = Tex(r"nearby orientations, same place", color=GREY_B).scale(0.42).next_to(lab_r, DOWN, buff=0.12)

        def link(term, pic, col):
            return Line(term.get_bottom() + DOWN * 0.1, pic.get_top() + UP * 0.12, stroke_color=col, stroke_width=1.5, stroke_opacity=0.5)
        ln_e, ln_p, ln_r = link(eq[5], pic_e, EXC), link(eq[7], pic_p, INH), link(eq[9], spokes, INH)

        self.say(r"Excitation: collinear support on both sides, the bipole of step 5, read from the orientation-pooled field.")
        self.play(Write(eq[4:6]), FadeIn(pic_e, lag_ratio=0.1), Create(ln_e), run_time=1.5)
        self.play(FadeIn(lab_e), FadeIn(sub_e))
        self.wait(1.2)
        self.say(r"Parallel inhibition: the weaker of the two flanks. A texture of parallel pieces has both; a lone vessel has neither.")
        self.play(Write(eq[6:8]), FadeIn(pic_p, lag_ratio=0.2), Create(ln_p), run_time=1.5)
        self.play(FadeIn(lab_p), FadeIn(sub_p))
        self.wait(1.5)
        self.say(r"Rival inhibition: the nearest orientations compete, within two steps only, so a crossing's two lobes both survive.")
        self.play(Write(eq[8:10]), FadeIn(spokes, lag_ratio=0.05), Create(ln_r), run_time=1.5)
        self.play(FadeIn(lab_r), FadeIn(sub_r))
        self.wait(1.5)
        params = MathTex(r"T = 8,\quad g_E = 1,\quad g_P = 0.3,\quad g_R = 0.5", color=GREY_A).scale(0.6)
        params_l = Tex(r"(spec 6.3)", color=GREY_B).scale(0.45)
        pg = VGroup(params, params_l).arrange(RIGHT, buff=0.2).move_to(DOWN * 1.75)
        self.play(FadeIn(pg))
        self.wait(1.5)
        pics = VGroup(pic_e, lab_e, sub_e, pic_p, lab_p, sub_p, spokes, lab_r, sub_r, pg, ln_e, ln_p, ln_r)
        self.play(FadeOut(pics), eq.animate.scale(0.82).to_edge(UP, buff=0.3), run_time=1.0)

        # ---------------- the live run on a toy image
        pf = PixelFrame(H.SCENE_SHAPE, 4.1, np.array([-3.35, 0.15, 0.0]))
        od_img = image(sc["od"], pf.height, "magma", vmin=-0.02, vmax=0.17).move_to(pf.centre)
        border = SurroundingRectangle(od_img, buff=0, color=MUTED, stroke_width=1)
        head = Tex(r"a toy image, in OD").scale(0.55).next_to(border, UP, buff=0.12)
        self.say(r"A toy image: two vessels crossing, scattered dashes, and a texture `web' of short parallel pieces.")
        self.play(FadeIn(od_img), Create(border), FadeIn(head))
        self.wait(1.5)

        VMAX = 45.0
        maps = [image(C.max(axis=0), pf.height, "magma", vmin=0, vmax=VMAX).move_to(pf.centre) for C in Cs]
        head0 = Tex(r"$U^0$: line evidence, best orientation").scale(0.55).move_to(head)
        self.play(FadeIn(maps[0]), FadeOut(od_img), ReplacementTransform(head, head0), run_time=1.2)
        self.say(r"Step 4's output, $U^0$: thin crests along every line. Now run the loop, and watch the clutter.")

        # traces: medians per class, relative to U0; convergence
        cm = H.class_medians(Cs, sc["masks"])
        conv = H.change_per_iteration(Cs)
        ax1 = Axes(x_range=[0, 8, 1], y_range=[0.5, 1.05, 0.25], x_length=4.6, y_length=1.9,
                   axis_config=dict(color=MUTED, include_tip=False, font_size=20, stroke_width=1.5),
                   x_axis_config=dict(numbers_to_include=[0, 4, 8]),
                   y_axis_config=dict(numbers_to_include=[0.5, 0.75, 1.0], decimal_number_config=dict(num_decimal_places=2))
                   ).move_to(np.array([3.55, 1.1, 0]))
        ax1_t = Tex(r"median $C$ / $U^0$ (toy)").scale(0.48).next_to(ax1, UP, buff=0.1)
        ax2 = Axes(x_range=[0, 8, 1], y_range=[0, 3, 1], x_length=4.6, y_length=1.7,
                   axis_config=dict(color=MUTED, include_tip=False, font_size=20, stroke_width=1.5),
                   x_axis_config=dict(numbers_to_include=[0, 4, 8])).move_to(np.array([3.55, -1.75, 0]))
        ylab2 = VGroup(*[Tex(t).scale(0.4).next_to(ax2.c2p(0, v + 3), LEFT, buff=0.1)
                         for v, t in ((0, r"100\%"), (-1, r"10\%"), (-2, r"1\%"), (-3, r"0.1\%"))])
        ax2_t = Tex(r"change per iteration (log)").scale(0.48).next_to(ax2, UP, buff=0.1)
        gate = DashedLine(ax2.c2p(0, 1), ax2.c2p(8, 1), color=CONT, stroke_width=2)
        gate_l = Tex(r"1\,\%", color=CONT).scale(0.42).next_to(ax2.c2p(8, 1), RIGHT, buff=0.08)
        it_lbl = Tex(r"iteration").scale(0.4).next_to(ax2.c2p(8, 0), RIGHT, buff=0.15)
        self.play(Create(ax1), FadeIn(ax1_t), Create(ax2), FadeIn(ylab2), FadeIn(ax2_t), Create(gate), FadeIn(gate_l), FadeIn(it_lbl),
                  run_time=1.2)
        cols = {"vessel": VES_COL, "dashes": DASH_COL, "web": WEB_COL}
        dots = VGroup(*[Dot(ax1.c2p(0, 1.0), radius=0.05, color=c) for c in cols.values()])
        self.add(dots)

        counter = MathTex(r"t = 0").scale(0.6).next_to(border, DOWN, buff=0.15).align_to(border, LEFT)
        head_t = Tex(r"$C$ after the loop, best orientation").scale(0.55).move_to(head0)
        self.play(FadeIn(counter), ReplacementTransform(head0, head_t), run_time=0.6)
        lines1 = VGroup()
        lines2 = VGroup()
        prev_map = maps[0]
        for t in range(1, len(Cs)):
            new_c = MathTex(rf"t = {t}").scale(0.6).move_to(counter, aligned_edge=LEFT)
            segs = [Line(ax1.c2p(t - 1, cm[k][t - 1]), ax1.c2p(t, cm[k][t]), stroke_color=cols[k], stroke_width=3)
                    for k in cols]
            anims = [FadeIn(maps[t]), FadeOut(prev_map), Transform(counter, new_c)]
            anims += [Create(s) for s in segs]
            anims += [d.animate.move_to(ax1.c2p(t, cm[k][t])) for d, k in zip(dots, cols)]
            cv = np.log10(max(conv[t - 1], 1e-3)) + 3
            cdot = Dot(ax2.c2p(t, cv), radius=0.05, color=FB)
            anims.append(FadeIn(cdot, scale=1.5))
            if t > 1:
                pv = np.log10(max(conv[t - 2], 1e-3)) + 3
                l2 = Line(ax2.c2p(t - 1, pv), ax2.c2p(t, cv), stroke_color=FB, stroke_width=3)
                anims.append(Create(l2)); lines2.add(l2)
            lines1.add(*segs); lines2.add(cdot)
            self.play(*anims, run_time=0.9 if t <= 3 else 0.6)
            prev_map = maps[t]
            if t == 2:
                self.say(r"Each pass: excitation along lines, inhibition from flanks and rival orientations. The change shrinks every time.")
        labels = VGroup(*[Tex(name, color=cols[k]).scale(0.45).next_to(ax1.c2p(8, cm[k][-1]), RIGHT, buff=0.12)
                          for k, name in (("vessel", r"vessels"), ("dashes", r"dashes"), ("web", r"web"))])
        order = sorted(range(3), key=lambda i: -labels[i].get_y())
        for a, b in zip(order[:-1], order[1:]):          # keep the labels apart
            if labels[a].get_y() - labels[b].get_y() < 0.3:
                labels[b].set_y(labels[a].get_y() - 0.3)
        self.play(FadeIn(labels))
        ct = H.class_medians(run["today"], sc["masks"])
        hollow = VGroup(*[Circle(radius=0.07, stroke_color=cols[k], stroke_width=2.5).move_to(ax1.c2p(3, ct[k][-1])) for k in cols])
        hkey = VGroup(Circle(radius=0.06, stroke_color=GREY_A, stroke_width=2), Tex(r"today's step 5, after its 3 iterations", color=GREY_A).scale(0.4)
                      ).arrange(RIGHT, buff=0.1).move_to(ax1.c2p(4.3, 0.56))
        below = int(np.argmax(conv < 0.01)) + 1
        self.say(rf"Toy, after 8 iterations: vessels keep {100 * cm['vessel'][-1]:.0f}\,\%, dashes {100 * cm['dashes'][-1]:.0f}\,\%, "
                 rf"the web {100 * cm['web'][-1]:.0f}\,\%. The change falls below 1\,\% at iteration {below}.", wait="read")
        self.say(rf"For comparison, today's excitation-only step 5 (circles) already takes dashes and the web to {100 * ct['dashes'][-1]:.0f}\,\%: "
                 r"in this toy the new inhibition adds most on the web.")
        self.play(FadeIn(hollow, scale=1.5), FadeIn(hkey))
        self.wait(3.5)
        self.say(r"The clutter fades and the vessels stay; at the crossing, both orientations survive the rivalry.")
        before = maps[0].copy()
        t0_lbl = MathTex(r"t = 0").scale(0.6).move_to(counter, aligned_edge=LEFT)
        t8_lbl = counter.copy()
        for _ in range(2):
            self.play(FadeIn(before), Transform(counter, t0_lbl), run_time=0.5)
            self.wait(0.7)
            self.play(FadeOut(before), Transform(counter, t8_lbl), run_time=0.5)
            self.wait(0.7)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self._cap], run_time=0.8)

    # ================================================================== level 3: switched off; the lobes are too thin
    def level3(self):
        st = Tex(r"status: built in MICA, switched off", color=GREY_A).scale(0.75)
        st2 = Tex(r"it failed its gate on the real test image (spec 4.7)", color=GREY_B).scale(0.55)
        sg = VGroup(st, st2).arrange(DOWN, buff=0.2).move_to(UP * 0.3)
        self.say(r"That is the design. Built in MICA, it failed its gate on the real test image, and it is switched off.")
        self.play(FadeIn(st), FadeIn(st2, shift=UP * 0.1))
        self.wait(1.5)
        self.say(r"The finding: the lobes are too thin. Step 5's lobes are 1-px rays; the crest along a vessel is 1--2 px wide.",
                 wait="read")
        self.play(FadeOut(sg), run_time=0.6)

        # ---------------- zoomed pixel grid: a vessel at 5.6 degrees, the 0-degree cell's ray
        ang = 5.625
        od, c, u = H.vessel_at(ang)
        U0 = H.line_evidence(od)
        P = H.pooled(U0)[0]                                   # the 0-degree sheet, orientation-pooled
        p0 = c - 8 * u
        x0, y0 = int(round(p0[0])) - 3, int(round(p0[1])) - 4
        nx, ny = 22, 9
        cs = 0.5                                              # scene units per pixel
        grid_c = np.array([-1.2, 1.15, 0.0])

        def gp(x, y):                                         # image pixel (x, y) -> scene point
            return grid_c + np.array([(x - x0 - (nx - 1) / 2) * cs, -(y - y0 - (ny - 1) / 2) * cs, 0.0])
        vals = P[y0:y0 + ny, x0:x0 + nx]
        rgba = colormap(vals, "magma", 0, 42)
        squares = VGroup()
        for j in range(ny):
            for i in range(nx):
                col = ManimColor.from_rgb(tuple(rgba[j, i, :3] / 255))
                squares.add(Square(cs, stroke_color=GREY_E, stroke_width=1, fill_color=col, fill_opacity=1).move_to(gp(x0 + i, y0 + j)))
        g_head = Tex(r"the $0^\circ$ sheet, pixel by pixel: a vessel at $5.6^\circ$ (toy)").scale(0.55).next_to(squares, UP, buff=0.15)
        self.play(FadeIn(squares, lag_ratio=0.002), FadeIn(g_head), run_time=1.5)
        crest = DashedLine(gp(*(p0 - 3 * u)), gp(*(p0 + 19 * u)), color=WHITE, stroke_width=2, dash_length=0.1)
        crest_l = Tex(r"vessel centreline", color=WHITE).scale(0.45).next_to(gp(*(p0 + 19 * u)), RIGHT, buff=0.12)
        self.play(Create(crest), FadeIn(crest_l))

        cell = Dot(gp(*p0), radius=0.13, color=YELLOW, z_index=5).set_stroke(BLACK, width=3, background=True)
        ds = np.arange(2, 13)
        ray_pts = [p0 + np.array([d, 0.0]) for d in ds]
        from scipy.ndimage import map_coordinates
        ray_vals = np.array([map_coordinates(P, [[q[1]], [q[0]]], order=1)[0] for q in ray_pts])
        crest_vals = np.array([map_coordinates(P, [[(p0 + d * u)[1]], [(p0 + d * u)[0]]], order=1)[0] for d in ds])
        ray = Line(gp(*p0), gp(*(p0 + np.array([12.5, 0.0]))), stroke_color=EXC, stroke_width=4)
        rdots = VGroup(*[Dot(gp(*q), radius=0.06, color=EXC) for q in ray_pts])
        self.say(r"A cell on the crest at $0^\circ$ looks ahead along a 1-px ray. The vessel drifts off it: one pixel every ten.")
        self.play(FadeIn(cell, scale=2))
        self.play(Create(ray), LaggedStart(*[FadeIn(d, scale=1.5) for d in rdots], lag_ratio=0.1), run_time=1.5)

        # the samples under the ray, as bars, against the crest
        base_y = -2.55
        bw = 0.42
        bars = VGroup()
        cbars = VGroup()
        for j, (d, v, cv) in enumerate(zip(ds, ray_vals, crest_vals)):
            x = gp(*(p0 + np.array([d, 0.0])))[0]
            cbars.add(Rectangle(width=bw, height=1.15 * cv / 42, stroke_color=GREY_B, stroke_width=1, fill_opacity=0).move_to(
                np.array([x, base_y, 0]), aligned_edge=DOWN))
            bars.add(Rectangle(width=bw * 0.7, height=max(1.15 * v / 42, 0.01), stroke_width=0, fill_color=EXC, fill_opacity=0.9).move_to(
                np.array([x, base_y, 0]), aligned_edge=DOWN))
        base = Line(np.array([cbars.get_left()[0] - 0.1, base_y, 0]), np.array([cbars.get_right()[0] + 0.1, base_y, 0]),
                    stroke_color=MUTED, stroke_width=1.5)
        b_lbl = VGroup(Tex(r"under the ray", color=EXC).scale(0.45), Tex(r"on the crest", color=GREY_B).scale(0.45)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(base, RIGHT, buff=0.25).shift(UP * 0.5)
        d_lbl = Tex(r"distance ahead: 2 to 12 px", color=MUTED).scale(0.42).next_to(base, DOWN, buff=0.08)
        self.play(Create(base), FadeIn(cbars), FadeIn(b_lbl[1]), FadeIn(d_lbl))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), FadeIn(b_lbl[0]), run_time=1.8)
        drop = 100 * (1 - ray_vals[-1] / crest_vals[-1])
        self.say(rf"Under the ray the support falls away: {drop:.0f}\,\% lost by 12 px (toy). So the field lowers the vessel instead of keeping it.",
                 wait="read")

        # the fix: a lobe with width
        band = Polygon(gp(*(p0 + np.array([2, -2.0]))), gp(*(p0 + np.array([12.5, -2.0]))), gp(*(p0 + np.array([12.5, 2.0]))),
                       gp(*(p0 + np.array([2, 2.0]))), stroke_color=EXC, stroke_width=1.5, fill_color=EXC, fill_opacity=0.18)
        band_l = VGroup(Tex(r"a lobe with width:", color=EXC), Tex(r"a Gaussian 1--2 px", color=EXC), Tex(r"across the ray", color=EXC)
                        ).scale(0.45).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(squares, RIGHT, buff=0.2)
        self.say(r"Horizontal connections in cortex spread over positions and orientations (Bosking et al. 1997). Give the lobes width.")
        self.play(FadeIn(band), FadeIn(band_l))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self._cap], run_time=0.8)

        # ---------------- the spec's measurements beside the toy's
        rows = [[r"vessel angle", r"nearest cell", r"measured (spec 4.7)", r"this toy"]]
        for a, near, meas in ((0, r"$0^\circ$", r"$0\,\%$"), (11.25, r"$11.25^\circ$", r"$-5\,\%$"), (70, r"$67.5^\circ$", r"$-8\,\%$"),
                              (30, r"$33.75^\circ$", r"$-11\,\%$"), (5.625, r"$0^\circ$", r"$-15\,\%$")):
            name = rf"${a:g}^\circ$" if a != 5.625 else r"$5.6^\circ$ (between two cells)"
            rows.append([name, near, meas, rf"${100 * H.change_along(a):+.0f}\,\%$"])
        T = tex_table(rows, widths=[4.0, 2.2, 3.0, 1.6], scale=0.55, row_buff=0.24)
        hdr = Tex(r"median change along a straight synthetic vessel, after the field", color=GREY_A).scale(0.55)
        hdr.to_edge(UP, buff=0.45)
        T.next_to(hdr, DOWN, buff=0.35)
        for r in T[0][1:]:
            r[3].set_color(GREY_A)
        self.say(r"Measured on straight synthetic vessels: only a vessel on a cell's axis is spared. The toy shows the same order, larger.")
        self.play(FadeIn(hdr), FadeIn(T[0][0]), Create(T[1]))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in T[0][1:]], lag_ratio=0.35), run_time=2.5)
        self.wait(2.0)
        om = H.other_measures()
        more = tex_table([
            [r"also measured (spec 4.7)", r"", r""],
            [r"an isolated 14 px dash", r"$-18\,\%$", rf"${100 * om['dash']:+.0f}\,\%$"],
            [r"the centre of an 8 px gap", r"$+9\,\%$", rf"${100 * om['gap8']:+.0f}\,\%$"],
            [r"the centre of an 18 px gap", r"$+34\,\%$", rf"${100 * om['gap18']:+.0f}\,\%$"],
            [r"the last 10 px before an end", r"about $-10\,\%$", rf"${100 * om['end']:+.0f}\,\%$"],
        ], widths=[6.2, 3.0, 1.6], scale=0.5, row_buff=0.16)
        more.next_to(T, DOWN, buff=0.4).align_to(T, LEFT)
        for r in more[0][1:]:
            r[2].set_color(GREY_A)
        more = VGroup(VGroup(more[0][0], more[1], more[0][1]), more[0][2:])
        self.say(r"And an isolated dash loses almost as much as a misaligned vessel, so the stage barely separates contours from clutter.")
        self.play(FadeIn(more[0]))
        self.wait(1.5)
        self.play(FadeIn(more[1:], lag_ratio=0.4), run_time=1.2)
        self.wait(2.5)
        self.play(FadeOut(VGroup(hdr, T, more)), run_time=0.8)

        # ---------------- lobes with width remove the dips (toy, feedforward reading)
        angs, sup = H.support_vs_angle(widths=(0.0, 1.5))
        ax = Axes(x_range=[0, 22.5, 5.625], y_range=[0.5, 1.3, 0.25], x_length=7.0, y_length=3.4,
                  axis_config=dict(color=MUTED, include_tip=False, font_size=22, stroke_width=1.5),
                  y_axis_config=dict(numbers_to_include=[0.5, 0.75, 1.0, 1.25], decimal_number_config=dict(num_decimal_places=2))
                  ).move_to(np.array([-0.6, -0.2, 0]))
        xn = VGroup(*[MathTex(t).scale(0.45).next_to(ax.c2p(v, 0.5), DOWN, buff=0.12)
                      for v, t in ((0, r"0^\circ"), (5.625, r"5.6^\circ"), (11.25, r"11.25^\circ"), (16.875, r"16.9^\circ"), (22.5, r"22.5^\circ"))])
        xl = Tex(r"vessel angle (cells every $11.25^\circ$)").scale(0.48).next_to(xn, DOWN, buff=0.12)
        yl = Tex(r"lobe reading / crest").scale(0.48).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.55)
        ttl = Tex(r"support along a straight vessel, relative to $0^\circ$ (toy, one pass)").scale(0.55).next_to(ax, UP, buff=0.3)
        cells = VGroup(*[DashedLine(ax.c2p(v, 0.5), ax.c2p(v, 1.3), color=MUTED, stroke_width=1, stroke_opacity=0.5) for v in (0, 11.25, 22.5)])
        thin = ax.plot_line_graph(angs, sup[0.0], line_color=WHITE, add_vertex_dots=True, stroke_width=3,
                                  vertex_dot_radius=0.04)
        wide = ax.plot_line_graph(angs, sup[1.5], line_color=EXC, add_vertex_dots=True, stroke_width=4,
                                  vertex_dot_radius=0.04, vertex_dot_style=dict(color=EXC))
        k_thin = Tex(r"1-px rays (step 5)", color=WHITE).scale(0.48)
        k_wide = Tex(r"lobes 1.5 px across", color=EXC).scale(0.48)
        keys = VGroup(k_thin, k_wide).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(ax, RIGHT, buff=0.3).shift(UP * 0.6)
        self.say(r"In the toy, one pass of each kind of lobe along vessels at every angle: rays dip between the cells' axes; wide lobes do not.")
        self.play(Create(ax), FadeIn(xn), FadeIn(xl), FadeIn(yl), FadeIn(ttl), Create(cells), run_time=1.2)
        self.play(Create(thin), FadeIn(k_thin), run_time=2.0)
        self.play(Create(wide), FadeIn(k_wide), run_time=2.0)
        self.wait(1.5)
        fix = VGroup(Tex(r"the spec's fix, before"), Tex(r"the field is re-tested:"),
                     Tex(r"lobes with width,", color=EXC), Tex(r"following each cell's", color=EXC), Tex(r"winning arc (chapter 6)", color=EXC)
                     ).scale(0.5).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(keys, DOWN, buff=0.5).align_to(keys, LEFT)
        self.say(r"So, before the recurrent field is re-tested, its lobes get width and follow each cell's winning arc.")
        self.play(FadeIn(fix, lag_ratio=0.15), run_time=1.5)
        self.wait(3.5)


variants(AssociationField, module_globals=globals())
