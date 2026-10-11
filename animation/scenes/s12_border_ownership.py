"""Chapter 12: a vessel owns both of its borders.

Border-ownership and grouping cells (spec 4.9), and evidence fusion (4.9b).
A dark strip owns both of its borders; a single edge (a glare border, the
aperture) owns one.  A grouping cell takes the geometric mean of the two
border cells a distance h either side of it: it peaks on the midline when h
is the half width, and is zero for a single edge.  Then fusion: line and
two-border evidence summed before one threshold, not a union of masks.

Render:  manim -ql scenes/s12_border_ownership.py BorderOwnershipDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import grouping as GR  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from manim.constants import RESAMPLING_ALGORITHMS  # noqa: E402
from mica_anim.style import FB, INH, MUTED, OD_COL, PixelFrame, image  # noqa: E402

LEFT_OWN = TEAL_C          # a border owned by the side to its left  (the odd cell's positive lobe)
RIGHT_OWN = PURPLE_B       # a border owned by the side to its right (its negative lobe)
G_COL = FB                 # grouping cells: they feed back to the border cells
FAINT_COL = ManimColor("#8FD3FF")
X = GR.X1
PLOT = (X >= -20.0) & (X <= 20.0)
XS = X[PLOT][::3]


@lru_cache(maxsize=None)
def profiles():
    """The two 1-D stimuli, their odd-cell responses (scaled so the vessel's
    lobes reach 1) and the vessel's best half width."""
    od_v = GR.vessel_profile(5.0, 0.12)
    od_s = GR.step_profile(0.08, 1.0)
    odd_v, odd_s = GR.odd_profile(od_v), GR.odd_profile(od_s)
    s = odd_v.max()
    odd_v, odd_s = odd_v / s, odd_s / s
    hs = np.arange(0.2, 12.0001, 0.05)
    g0 = np.array([GR.grouping_profile(odd_v, h)[np.argmin(np.abs(X))] for h in hs])
    return dict(od_v=od_v, od_s=od_s, odd_v=odd_v, odd_s=odd_s, hs=hs, g0=g0,
                hstar=float(hs[np.argmax(g0)]), hhw=float(GR.half_height_halfwidth(od_v)))


def strip_image(od, rows=24, noise=0.0, seed=0):
    """The camera's view of a 1-D profile, extended along the vessel."""
    xs = np.arange(-20.0, 20.01, 0.5)
    o = np.interp(xs, X, od)
    img = np.tile(0.85 * np.exp(-o), (rows, 1))
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img


def graph(ax, xs, ys, color, width=3.0, fill=0.0):
    pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners(pts)
    if fill:
        base = [ax.c2p(xs[-1], 0), ax.c2p(xs[0], 0)]
        f = VMobject(stroke_width=0, fill_color=color, fill_opacity=fill)
        f.set_points_as_corners(pts + base + [pts[0]])
        return VGroup(f, m)
    return m


def label(text, scale=0.45, color=GREY_A):
    return Tex(text, color=color).scale(scale)


class BorderOwnership(ChapterScene):
    NUMBER, TITLE, SPEC = 12, "A vessel owns both of its borders", "spec 4.9, 4.9b"
    MAP_KEYS = ("border", "fusion")

    def construct(self):
        self.intro()
        cols = self.opener()
        self.profiles(cols)
        if self.at(2):
            self.sweep(cols)
        self.clear()
        self.maps()
        if self.at(2):
            self.medial()
        if self.at(3):
            self.caveat()
        if self.at(2):
            self.fusion()
        self.outro()

    # ------------------------------------------------------------------ opener
    def opener(self):
        P = profiles()
        cols = {}
        for name, x, od, title in (("v", -3.55, P["od_v"], r"a vessel: a dark strip"),
                                   ("s", 3.55, P["od_s"], r"a single edge: a glare border")):
            img = image(strip_image(od, rows=60, noise=0.006, seed=1 if name == "v" else 2), 2.3, "grey",
                        vmin=0.66, vmax=0.88)
            img.stretch_to_fit_width(5.6).move_to([x, 0.55, 0])
            frame = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1)
            ttl = Tex(title).scale(0.62).next_to(img, UP, buff=0.18)
            cols[name] = dict(x=x, img=img, frame=frame, title=ttl)
        self.play(*[FadeIn(c["img"]) for c in cols.values()], *[Create(c["frame"]) for c in cols.values()],
                  *[FadeIn(c["title"]) for c in cols.values()], run_time=1.2)
        if self.at(3):
            self.say(r"In V2, border-ownership cells signal which side of an edge belongs to the figure"
                     r" (Zhou, Friedman and von der Heydt 2000).", wait=1.5)

        # ownership arrows: each border points to the side that owns it
        def arrows(c, xs_dirs):
            img = c["img"]
            u = img.width / 40.0
            g = VGroup()
            for xpx, d, col in xs_dirs:
                for yy in (-0.65, 0.0, 0.65):
                    p = np.array([c["x"] + xpx * u, img.get_y() + yy, 0])
                    g.add(Arrow(p - RIGHT * d * 0.12, p + RIGHT * d * 0.4, buff=0, stroke_width=6,
                                max_tip_length_to_length_ratio=0.42, color=col))
            return g

        h = P["hhw"]
        cols["v"]["arrows"] = arrows(cols["v"], [(-h, 1, RIGHT_OWN), (h, -1, LEFT_OWN)])
        cols["s"]["arrows"] = arrows(cols["s"], [(0.0, 1, RIGHT_OWN)])
        self.play(LaggedStart(*[FadeIn(a, shift=RIGHT * 0.1) for a in cols["v"]["arrows"]]), run_time=1.2)
        self.say(r"A vessel is a dark strip: it owns both of its borders.", wait=0.6)
        self.play(LaggedStart(*[FadeIn(a, shift=RIGHT * 0.1) for a in cols["s"]["arrows"]]), run_time=0.8)
        self.say(r"A single edge, such as a glare border or the aperture's rim, owns only one."
                 r" Yet line cells answer to both.", wait="read")

        # squash the views into strips above the plots
        anims = []
        for c in cols.values():
            tgt_y, tgt_h = 2.83, 0.48
            sy = tgt_h / c["img"].height
            for m in (c["img"], c["frame"]):
                anims.append(m.animate.stretch(sy, 1).move_to([c["x"], tgt_y, 0]))
            ar = c["arrows"]
            keep = VGroup(*[a for i, a in enumerate(ar) if i % 3 == 1])
            for i, a in enumerate(ar):
                if i % 3 != 1:
                    anims.append(FadeOut(a, run_time=0.5))
            anims.append(keep.animate.scale(0.75).set_y(tgt_y))
            c["arrows"] = keep
            anims.append(c["title"].animate.move_to([c["x"], 3.45, 0]))
        self.play(*anims, run_time=1.2)
        return cols

    # ---------------------------------------------------------------- 1-D cells
    def profiles(self, cols):
        P = profiles()
        for name, c in cols.items():
            od = P["od_" + name]
            odd = P["odd_" + name]
            ax_od = Axes(x_range=[-20, 20, 10], y_range=[0, 0.14, 0.05], x_length=5.6, y_length=1.15,
                         tips=False, axis_config=dict(color=MUTED, stroke_width=1.5, include_ticks=False)
                         ).move_to([c["x"], 1.72, 0])
            ax_c = Axes(x_range=[-20, 20, 10], y_range=[-1.15, 1.15, 0.5], x_length=5.6, y_length=2.75,
                        tips=False, axis_config=dict(color=MUTED, stroke_width=1.5, include_ticks=False)
                        ).move_to([c["x"], -1.25, 0])
            c.update(ax_od=ax_od, ax_c=ax_c, od=od, odd=odd)
            c["od_curve"] = graph(ax_od, XS, np.interp(XS, X, od), OD_COL, 3, fill=0.25)
            c["od_lbl"] = label(r"OD", 0.45, OD_COL).move_to(ax_od.c2p(-17.5, 0.11))
            c["c_lbl"] = label(r"edge cells", 0.45).move_to(ax_c.c2p(-15.5, 1.02))
        self.play(*[Create(c["ax_od"]) for c in cols.values()], *[Create(c["ax_c"]) for c in cols.values()],
                  run_time=0.8)
        self.play(*[Create(c["od_curve"]) for c in cols.values()], *[FadeIn(c["od_lbl"]) for c in cols.values()],
                  run_time=1.2)

        # the odd (edge) cell, split by ownership
        for c in cols.values():
            ax, odd = c["ax_c"], c["odd"]
            c["odd_curve"] = graph(ax, XS, np.interp(XS, X, odd), WHITE, 2.5)
            c["pos"] = graph(ax, XS, np.maximum(np.interp(XS, X, odd), 0), LEFT_OWN, 3, fill=0.25)
            c["neg"] = graph(ax, XS, np.minimum(np.interp(XS, X, odd), 0), RIGHT_OWN, 3, fill=0.25)
        self.say(r"An edge cell responds where the OD steps. Its sign says which side is darker: the side that owns the border.")
        self.play(*[Create(c["odd_curve"]) for c in cols.values()], *[FadeIn(c["c_lbl"]) for c in cols.values()],
                  run_time=1.4)
        self.play(*[FadeOut(c["odd_curve"]) for c in cols.values()],
                  *[FadeIn(c["pos"]) for c in cols.values()], *[FadeIn(c["neg"]) for c in cols.values()],
                  run_time=1.0)
        if self.at(2):
            kv = cols["v"]
            own_l = label(r"owned to its left", 0.42, LEFT_OWN).move_to(kv["ax_c"].c2p(12.0, 0.75))
            own_r = label(r"owned to its right", 0.42, RIGHT_OWN).move_to(kv["ax_c"].c2p(-12.0, -0.75))
            self.play(FadeIn(own_l), FadeIn(own_r))
            self.wait(1.2)
            self.play(FadeOut(own_l), FadeOut(own_r))
        else:
            self.wait(0.6)

        # the grouping cell: the two borders, shifted onto p
        h = ValueTracker(0.0)
        self.h = h
        for c in cols.values():
            ax, odd = c["ax_c"], c["odd"]

            def left_c(ax=ax, odd=odd):
                L, _ = GR.shifted(odd, h.get_value())
                return graph(ax, XS, np.interp(XS, X, L), LEFT_OWN, 3, fill=0.25)

            def right_c(ax=ax, odd=odd):
                _, R = GR.shifted(odd, h.get_value())
                return graph(ax, XS, np.interp(XS, X, R), RIGHT_OWN, 3, fill=0.25)

            def g_c(ax=ax, odd=odd):
                g = GR.grouping_profile(odd, h.get_value())
                return graph(ax, XS, np.interp(XS, X, g), G_COL, 4, fill=0.35)

            def glyph(c=c):
                img = c["img"]
                u = img.width / 40.0
                hv = h.get_value() * u
                y = img.get_y()
                p = np.array([c["x"], y, 0])
                return VGroup(Line(p + LEFT * hv, p + RIGHT * hv, stroke_color=G_COL, stroke_width=3),
                              Line(p + LEFT * hv + UP * 0.2, p + LEFT * hv + DOWN * 0.2, stroke_color=G_COL, stroke_width=3),
                              Line(p + RIGHT * hv + UP * 0.2, p + RIGHT * hv + DOWN * 0.2, stroke_color=G_COL, stroke_width=3),
                              Dot(p, radius=0.06, color=G_COL))
            c["left"], c["right"], c["g"], c["glyph"] = (always_redraw(left_c), always_redraw(right_c),
                                                         always_redraw(g_c), always_redraw(glyph))
            c["mid"] = DashedLine(ax.c2p(0, -1.1), ax.c2p(0, 1.1), stroke_color=MUTED, stroke_width=1.5, dash_length=0.06)
        self.say(r"A grouping cell at $p$ reads a border at $p+h$ owned toward $p$, and the mirror border at $p-h$, sign flipped.")
        flipped = {}
        for name, c in cols.items():
            _, R0 = GR.shifted(c["odd"], 0.0)
            flipped[name] = graph(c["ax_c"], XS, np.interp(XS, X, R0), RIGHT_OWN, 3, fill=0.25)
        self.play(*[ReplacementTransform(c["neg"], flipped[n]) for n, c in cols.items()],
                  *[FadeIn(c["glyph"]) for c in cols.values()], *[Create(c["mid"]) for c in cols.values()],
                  run_time=1.2)
        for n, c in cols.items():                 # swap the static lobes for the live, shifting copies
            self.remove(c["pos"], flipped[n])
            self.add(c["left"], c["right"])
        hv = P["hstar"]
        self.play(h.animate.set_value(hv), run_time=2.5)
        self.add(*[c["g"] for c in cols.values()])
        g_lbl = MathTex(r"G", r"=", r"\sqrt{\text{left}\cdot\text{right}}").scale(0.62)
        g_lbl[0].set_color(G_COL)
        g_lbl[2][2:6].set_color(LEFT_OWN)
        g_lbl[2][7:12].set_color(RIGHT_OWN)
        g_lbl.move_to(cols["v"]["ax_c"].c2p(-12.0, -0.65))
        zero = MathTex(r"G = 0", color=G_COL).scale(0.62).move_to(cols["s"]["ax_c"].c2p(-11.0, 0.4))
        self.play(FadeIn(g_lbl), FadeIn(zero))
        self.say(r"Their geometric mean peaks on the vessel's midline. For the single edge it is exactly zero:"
                 r" there is no second border.", wait="read")
        for c in cols.values():
            for k in ("left", "right", "g", "glyph"):
                c[k].clear_updaters()
        self.g_lbl, self.zero = g_lbl, zero

    # ------------------------------------------------------- equation and sweep
    def sweep(self, cols):
        P = profiles()
        s = cols["s"]
        self.play(*[FadeOut(s[k]) for k in ("img", "frame", "title", "arrows", "ax_od", "ax_c", "od_curve", "od_lbl",
                                            "c_lbl", "left", "right", "g", "glyph", "mid")], FadeOut(self.zero), run_time=0.8)
        v = cols["v"]
        eq = VGroup(
            MathTex(r"\text{left}_{k,h}(p)", r"=", r"\max\!\big(0,\ \text{odd}_k(p + h\,n_k)\big)"),
            MathTex(r"\text{right}_{k,h}(p)", r"=", r"\max\!\big(0,\ -\text{odd}_k(p - h\,n_k)\big)"),
            MathTex(r"G_{k,h}(p)", r"=", r"\sqrt{\text{left}_{k,h}(p)\cdot\text{right}_{k,h}(p)}"),
            MathTex(r"h^*(p,k)", r"=", r"\arg\max_h\, G_{k,h}(p)"),
        ).scale(0.6).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eq[0][0].set_color(LEFT_OWN)
        eq[1][0].set_color(RIGHT_OWN)
        eq[2][0].set_color(G_COL)
        eq[2][2][2:12].set_color(LEFT_OWN)
        eq[2][2][13:24].set_color(RIGHT_OWN)
        eq[3][0].set_color(G_COL)
        eq.move_to([3.65, 2.35, 0])
        self.say(r"Per orientation $k$: the border cell owned toward $p$ at $p + h\,n_k$, the mirror one at $p - h\,n_k$.")
        self.play(Write(eq[0]), run_time=1.2)
        self.play(Write(eq[1]), run_time=1.2)
        self.say(r"The grouping cell is their geometric mean: large only if both borders are there.")
        self.play(ReplacementTransform(self.g_lbl, eq[2]), run_time=1.2)
        self.wait(0.8)

        # G at the midline against h
        ax = Axes(x_range=[0, 12, 2], y_range=[0, 1.1, 0.5], x_length=4.6, y_length=1.9, tips=False,
                  axis_config=dict(color=MUTED, stroke_width=1.5, font_size=22),
                  x_axis_config=dict(numbers_to_include=[2, 4, 6, 8, 10, 12])).move_to([3.75, -0.85, 0])
        xl = label(r"$h$ (px)", 0.45).next_to(ax.x_axis, DOWN, buff=0.32)
        yl = label(r"$G$ at the midline", 0.45, G_COL).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        h = self.h
        odd = v["odd"]
        live = [always_redraw(lambda: graph(v["ax_c"], XS, np.interp(XS, X, GR.shifted(odd, h.get_value())[0]), LEFT_OWN, 3, 0.25)),
                always_redraw(lambda: graph(v["ax_c"], XS, np.interp(XS, X, GR.shifted(odd, h.get_value())[1]), RIGHT_OWN, 3, 0.25)),
                always_redraw(lambda: graph(v["ax_c"], XS, np.interp(XS, X, GR.grouping_profile(odd, h.get_value())), G_COL, 4, 0.35))]
        u = v["img"].width / 40.0

        def glyph():
            hv = h.get_value() * u
            p = np.array([v["x"], v["img"].get_y(), 0])
            return VGroup(Line(p + LEFT * hv, p + RIGHT * hv, stroke_color=G_COL, stroke_width=3),
                          Line(p + LEFT * hv + UP * 0.2, p + LEFT * hv + DOWN * 0.2, stroke_color=G_COL, stroke_width=3),
                          Line(p + RIGHT * hv + UP * 0.2, p + RIGHT * hv + DOWN * 0.2, stroke_color=G_COL, stroke_width=3),
                          Dot(p, radius=0.06, color=G_COL))
        self.remove(v["left"], v["right"], v["g"], v["glyph"])
        gl = always_redraw(glyph)
        self.add(*live, gl)

        hmax = ValueTracker(0.0)
        hmax.add_updater(lambda m: m.set_value(max(m.get_value(), h.get_value())))
        self.add(hmax)

        def trace():
            hv = hmax.get_value()
            keep = P["hs"] <= hv + 1e-9
            if keep.sum() < 2:
                return VGroup()
            return graph(ax, P["hs"][keep][::4], P["g0"][keep][::4], G_COL, 3)
        tr = always_redraw(trace)
        dot = always_redraw(lambda: Dot(ax.c2p(h.get_value(), np.interp(h.get_value(), P["hs"], P["g0"])), radius=0.06, color=G_COL))
        hval = always_redraw(lambda: MathTex(rf"h = {h.get_value():.1f}\ \text{{px}}").scale(0.5)
                             .next_to(v["img"], DOWN, buff=0.05).align_to(v["img"], RIGHT).shift(LEFT * 0.1 + UP * 0.03))
        self.say(r"Now sweep $h$, the cell's half width. At the midline, $G$ is largest when $h$ matches the vessel's half width.")
        self.play(h.animate.set_value(0.4), run_time=1.0)
        self.add(tr, dot, hval)
        self.play(h.animate.set_value(11.5), run_time=5.0, rate_func=linear)
        self.play(h.animate.set_value(P["hstar"]), run_time=2.0)
        hs_line = DashedLine(ax.c2p(P["hhw"], 0), ax.c2p(P["hhw"], 1.05), stroke_color=OD_COL, stroke_width=2)
        hs_lbl = VGroup(MathTex(rf"h^* = {P['hstar']:.1f}\ \text{{px}}", color=G_COL).scale(0.5),
                        Tex(rf"half width at half height: {P['hhw']:.1f} px", color=OD_COL).scale(0.42)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        hs_lbl.move_to(ax.c2p(6.2, 0.98), aligned_edge=LEFT)
        self.play(Create(hs_line), FadeIn(hs_lbl), Write(eq[3]))
        if self.at(3):
            self.say(r"In the model of Craft, Sch\"utze, Niebur and von der Heydt (2007), grouping cells peak on the"
                     r" figure's medial axis at a scale set by its width,", wait="read")
            self.say(r"close to the medial-axis responses seen in V1 (Lee, Mumford, Romero and Lamme 1998).", wait="read")
            # one feedback step: the grouping cell scales the two border cells it read
            img = v["img"]
            p = np.array([v["x"], img.get_top()[1] + 0.02, 0])
            ends = [np.array([v["x"] + sg * P["hhw"] * u, img.get_top()[1] + 0.02, 0]) for sg in (-1, 1)]
            fb = VGroup(*[CurvedArrow(p + UP * 0.03, e + UP * 0.03, angle=(-1.4 if e[0] < p[0] else 1.4), color=G_COL,
                                      stroke_width=3.5, tip_length=0.13) for e in ends])
            fb.shift(UP * 0.02)
            self.say(r"One feedback step: each border cell is scaled by the grouping cell it supports,")
            self.play(Create(fb), run_time=1.0)
            self.play(*[Indicate(a, color=G_COL, scale_factor=1.5) for a in v["arrows"]], run_time=1.2)
            self.say(r"so an edge shared by two strips is given to the one whose interior is darker.", wait="read")
            self.play(FadeOut(fb))
        else:
            self.say(r"So the peak's position is the vessel's midline, and the winning $h$ is its half width.", wait="read")
        for m in (*live, gl, tr, dot, hval, hmax):
            m.clear_updaters()

    # ------------------------------------------------------------- 2-D maps
    def maps(self):
        sc, M = GR.border_scene(), GR.border_maps()
        C, _ = GR.best(M["C"])
        G, _ = GR.best(M["G"])
        I = 0.85 * np.exp(-sc["od"])
        H = 2.6
        xs = (-4.62, 0.0, 4.62)
        cam = image(I, H, "grey", vmin=0.62, vmax=0.87).move_to([xs[0], 1.0, 0])
        cmap = image(C, H, "magma", vmin=2, vmax=36).move_to([xs[1], 1.0, 0])
        gmap = image(G, H, "magma", vmin=4, vmax=36).move_to([xs[2], 1.0, 0])
        frames = VGroup(*[SurroundingRectangle(m, buff=0, color=MUTED, stroke_width=1) for m in (cam, cmap, gmap)])
        titles = VGroup(Tex(r"camera view").scale(0.55),
                        Tex(r"line cells $C$").scale(0.55),
                        Tex(r"grouping cells $G$", color=G_COL).scale(0.55))
        for t, m in zip(titles, (cam, cmap, gmap)):
            t.next_to(m, UP, buff=0.14)
        self.cam, self.cam_frame = cam, frames[0]
        pf = PixelFrame(GR.BO_SHAPE, H, cam.get_center())
        self.pf = pf

        def tag(text, xy, col=WHITE):
            t = Tex(text, color=col).scale(0.4)
            t.move_to(pf.point(*xy))
            return VGroup(BackgroundRectangle(t, fill_opacity=0.65, buff=0.04), t)
        tags = VGroup(tag(r"glare edge", (60, 112)), tag(r"wide, flat floor", (166, 9)),
                      tag(r"thin", (182, 45)), tag(r"faint", (22, 30), FAINT_COL))
        self.say(r"In two dimensions, with noise: a thin vessel, a wide flat-floored one, a faint one and a glare edge.")
        self.play(FadeIn(cam), Create(frames[0]), FadeIn(titles[0]), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.3), run_time=1.2)
        self.wait(0.8)
        self.say(r"Line cells light the midlines, but also a false line along the edge, and only the walls of the wide vessel.")
        self.play(FadeIn(cmap), Create(frames[1]), FadeIn(titles[1]), run_time=1.2)
        self.wait(1.6)
        self.say(r"Grouping cells light strips: the edge almost vanishes, and the wide vessel becomes one strong strip.")
        self.play(FadeIn(gmap), Create(frames[2]), FadeIn(titles[2]), run_time=1.2)
        self.wait(1.0)

        m = sc["masks"]
        med = {k: (np.median(C[m[k]]), np.median(G[m[k]])) for k in ("thin", "wide", "edge")}
        cells = [Tex(""), Tex(r"thin"), Tex(r"wide, flat"), Tex(r"glare edge"),
                 MathTex(r"C"), *[MathTex(f"{med[k][0]:.0f}") for k in ("thin", "wide", "edge")],
                 MathTex(r"G", color=G_COL), *[MathTex(f"{med[k][1]:.0f}", color=G_COL) for k in ("thin", "wide", "edge")]]
        tbl = VGroup(*[c.scale(0.55) for c in cells]).arrange_in_grid(rows=3, cols=4, buff=(0.45, 0.14))
        note = Tex(r"toy: medians in noise units", color=GREY_B).scale(0.42)
        grp = VGroup(tbl, note).arrange(DOWN, buff=0.12).move_to([0.0, -1.55, 0])
        self.play(FadeIn(grp))
        self.wait(2.2 if self.at(2) else 1.5)
        self.maps_mobs = dict(cmap=cmap, gmap=gmap, frames=frames, titles=titles, tags=tags, grp=grp)

    def medial(self):
        mm = self.maps_mobs
        self.play(*[FadeOut(m) for m in (mm["cmap"], mm["gmap"], mm["frames"][1], mm["frames"][2], mm["titles"],
                                         mm["tags"], mm["grp"])], run_time=0.8)
        cam, fr = self.cam, self.cam_frame
        H = 4.3
        tgt = [-1.75, 0.45, 0]
        self.play(cam.animate.scale(H / cam.height).move_to(tgt), fr.animate.scale(H / fr.height).move_to(tgt), run_time=1.2)
        pf = PixelFrame(GR.BO_SHAPE, H, cam.get_center())
        pts = GR.medial_axis(11.0)
        circles = VGroup(*[Circle(radius=h * pf.unit, stroke_color=G_COL, stroke_width=2.2).move_to(pf.point(x, y))
                           for x, y, h, g in sorted(pts, key=lambda p: (p[0], p[1]))])
        dots = VGroup(*[Dot(pf.point(x, y), radius=0.035, color=G_COL) for x, y, h, g in pts])
        self.say(r"Where $G$ peaks is the midline; the $h$ that wins is the half width. Circles of radius $h^*$ rebuild each strip.")
        self.play(LaggedStart(*[GrowFromCenter(c) for c in circles], lag_ratio=0.05), FadeIn(dots), run_time=2.5)
        thin = [p for p in pts if p[2] < 6]
        wide = [p for p in pts if p[2] >= 8]
        lt = VGroup(Tex(rf"thin: $h^* = {np.median([p[2] for p in thin]):.0f}$ px", color=G_COL),
                    Tex(r"(half width at half height: 3.1 px)", color=GREY_B)).scale(0.48)
        lw = VGroup(Tex(rf"wide, flat: $h^* = {np.median([p[2] for p in wide]):.0f}$ px", color=G_COL),
                    Tex(r"(half width: 11 px)", color=GREY_B)).scale(0.48)
        for t in (lt, lw):
            t.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        right = cam.get_right()[0] + 0.3
        lt.move_to([right, pf.point(0, 22)[1], 0], aligned_edge=LEFT)
        lw.move_to([right, pf.point(0, 100)[1], 0], aligned_edge=LEFT)
        pt_t = pf.point(196, 25.5)
        pt_w = pf.point(141, 100) + RIGHT * 11 * pf.unit
        lines = VGroup(Line(lt.get_left() + LEFT * 0.08, pt_t, stroke_color=G_COL, stroke_width=1.5),
                       Line(lw.get_left() + LEFT * 0.08, pt_w, stroke_color=G_COL, stroke_width=1.5))
        lt, lw = VGroup(lt, lines[0]), VGroup(lw, lines[1])
        self.play(FadeIn(lt), FadeIn(lw))
        self.say(r"A width map, for free (toy). And the wide vessel, weak for line cells, has two strong owned borders.",
                 wait="read")
        self.play(FadeOut(Group(circles, dots, lt, lw, cam, fr)), run_time=0.8)

    def caveat(self):
        true, hstar = GR.width_bias()
        ax = Axes(x_range=[0, 18, 3], y_range=[0.8, 1.5, 0.1], x_length=6.2, y_length=3.6, tips=False,
                  axis_config=dict(color=MUTED, stroke_width=1.5, font_size=22),
                  x_axis_config=dict(numbers_to_include=[3, 6, 9, 12, 15, 18]),
                  y_axis_config=dict(numbers_to_include=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                                     decimal_number_config=dict(num_decimal_places=1))).move_to([-1.6, 0.35, 0])
        xl = label(r"vessel width at half height (px)", 0.48).next_to(ax.x_axis, DOWN, buff=0.35)
        yl = label(r"$h^*$ / true half width", 0.48).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45)
        one = DashedLine(ax.c2p(0, 1.0), ax.c2p(18, 1.0), stroke_color=MUTED, stroke_width=1.5)
        third = DashedLine(ax.c2p(0, 4 / 3), ax.c2p(18, 4 / 3), stroke_color=INH, stroke_width=2)
        third_l = Tex(r"reported (spec 10): about a third\\too large on thin vessels, not confirmed",
                      color=INH).scale(0.42).next_to(ax.c2p(18, 4 / 3), RIGHT, buff=0.15)
        curve = graph(ax, 2 * true, hstar / true, G_COL, 4)
        dots = VGroup(*[Dot(ax.c2p(2 * t, h / t), radius=0.05, color=G_COL) for t, h in zip(true, hstar)])
        toy_l = Tex(r"toy: edge cells $\sigma = 1.5$ px", color=G_COL).scale(0.45).next_to(ax.c2p(9, 1.12), UP, buff=0.1)
        self.say(r"A reported caveat (spec 10): on real thin vessels, the grouping cells' half width runs about a third too large.")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(one), run_time=1.0)
        self.play(Create(third), FadeIn(third_l))
        self.play(Create(curve), FadeIn(dots), FadeIn(toy_l), run_time=1.5)
        self.say(r"The toy shows one possible cause: when a vessel is no wider than the edge cell, the cell's own blur pushes"
                 r" the two border peaks apart.", wait="read")
        self.clear()

    # ------------------------------------------------------------- 4.9b fusion
    def fusion(self):
        cp = GR.cue_plane()
        sc, M = GR.border_scene(), GR.border_maps()
        ttl = Tex(r"Evidence fusion (spec 4.9b)").scale(0.7).to_edge(UP, buff=0.3)
        self.say(r"Two cues: line evidence $C$ and two-border evidence $G$. Threshold each and take a union of masks,"
                 r" or add them first?")
        self.play(FadeIn(ttl))
        lim = 20.0
        ax = Axes(x_range=[0, lim, 5], y_range=[0, lim, 5], x_length=4.7, y_length=4.7, tips=False,
                  axis_config=dict(color=MUTED, stroke_width=1.5, font_size=22),
                  x_axis_config=dict(numbers_to_include=[5, 10, 15, 20]),
                  y_axis_config=dict(numbers_to_include=[5, 10, 15, 20])).move_to([-2.9, 0.35, 0]).scale(0.95)
        xl = label(r"line evidence $C$ (noise units)", 0.45).next_to(ax.x_axis, DOWN, buff=0.32)
        yl = label(r"two-border evidence $G$", 0.45, G_COL).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.4)

        def pts(arr, col, r, op):
            a = arr[(arr[:, 0] < lim) & (arr[:, 1] < lim)]
            return VGroup(*[Dot(ax.c2p(c, g), radius=r, color=col, fill_opacity=op) for c, g in a])
        bgd = pts(cp["bg"], GREY_B, 0.022, 0.55)
        fd = pts(cp["faint"], FAINT_COL, 0.026, 0.9)
        key = VGroup(VGroup(Dot(radius=0.05, color=GREY_B), label(r"background pixels")).arrange(RIGHT, buff=0.12),
                     VGroup(Dot(radius=0.05, color=FAINT_COL), label(r"a faint vessel (OD 0.01, toy)", 0.45, FAINT_COL)
                            ).arrange(RIGHT, buff=0.12)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        key.next_to(ax, UP, buff=0.12).align_to(ax, LEFT).shift(RIGHT * 0.2)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.play(FadeIn(bgd, lag_ratio=0.002), FadeIn(key[0]), run_time=1.2)
        self.play(FadeIn(fd, lag_ratio=0.004), FadeIn(key[1]), run_time=1.2)
        self.wait(0.6)

        tu, ts = cp["t_u"], cp["t_s"]
        L = Polygon(ax.c2p(tu, 0), ax.c2p(lim, 0), ax.c2p(lim, lim), ax.c2p(0, lim), ax.c2p(0, tu), ax.c2p(tu, tu),
                    stroke_color=GREY_A, stroke_width=2, fill_color=GREY_A, fill_opacity=0.16)
        S = Polygon(ax.c2p(ts, 0), ax.c2p(lim, 0), ax.c2p(lim, lim), ax.c2p(0, lim), ax.c2p(0, ts),
                    stroke_color=G_COL, stroke_width=2, fill_color=G_COL, fill_opacity=0.16)
        k = M["F"].argmax(axis=0)
        Ck = np.take_along_axis(M["C"], k[None], 0)[0]
        Gk = np.take_along_axis(M["G"], k[None], 0)[0]
        union_mask = (Ck > tu) | (Gk > tu)
        sum_mask = Ck + Gk > ts
        fm = sc["masks"]["faint"]

        def mask_img(mask):
            rgb = np.zeros(mask.shape + (3,))
            rgb[mask] = 0.6
            rgb[mask & GR.border_scene()["masks"]["faint"]] = np.array([0.56, 0.83, 1.0])
            rgba = np.concatenate([rgb, np.ones(mask.shape + (1,))], -1)
            im = ImageMobject((rgba * 255).astype(np.uint8))
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            im.height = 2.05
            return im
        ui = mask_img(union_mask).move_to([3.6, 1.45, 0])
        si = mask_img(sum_mask).move_to([3.6, -1.55, 0])
        uf = SurroundingRectangle(ui, buff=0, color=GREY_A, stroke_width=1.5)
        sf = SurroundingRectangle(si, buff=0, color=G_COL, stroke_width=1.5)
        ut = Tex(rf"union: $C > {tu:.1f}$ or $G > {tu:.1f}$").scale(0.48).next_to(ui, UP, buff=0.1)
        st = Tex(rf"sum: $C + G > {ts:.1f}$", color=G_COL).scale(0.48).next_to(si, UP, buff=0.1)
        ur = Tex(rf"finds {100 * cp['hit_union']:.0f}\% of the faint vessel").scale(0.45).next_to(ui, DOWN, buff=0.08)
        sr = Tex(rf"finds {100 * cp['hit_sum']:.0f}\% of the faint vessel", color=G_COL).scale(0.45).next_to(si, DOWN, buff=0.08)

        self.say(r"A union of thresholds accepts an L-shaped region: each cue's own false positives, but not two weak cues that agree.")
        self.play(FadeIn(L), run_time=1.2)
        self.play(FadeIn(ui), Create(uf), FadeIn(ut), FadeIn(ur), run_time=1.0)
        self.wait(1.2)
        self.say(r"A sum accepts a diagonal half-plane: two weak cues can agree. Both rules pass"
                 rf" {100 * cp['fa_sum']:.1f}\% of background pixels (toy).")
        self.play(L.animate.set_fill(opacity=0.0).set_stroke(GREY_A, width=2.5), FadeIn(S), run_time=1.2)
        self.play(FadeIn(si), Create(sf), FadeIn(st), FadeIn(sr), run_time=1.0)
        self.wait(2.2)
        if not self.at(3):
            self.say(r"So the spec sums the two cues per orientation, in noise units, before one threshold.", wait="read")
            return

        # measured on the real image
        self.play(*[FadeOut(m) for m in (ui, si, uf, sf, ut, st, ur, sr)], run_time=0.7)
        cells = [Tex(""), Tex(r"line alone"), Tex(r"fused", color=G_COL),
                 Tex(r"centreline"), Tex(r"71,475 px"), Tex(r"88,514 px", color=G_COL),
                 Tex(r"pieces"), Tex(r"241"), Tex(r"755", color=G_COL),
                 Tex(r"mask"), Tex(r"21.6\,\%"), Tex(r"28.9\,\%", color=G_COL)]
        tbl = VGroup(*[c.scale(0.52) for c in cells]).arrange_in_grid(rows=4, cols=3, buff=(0.4, 0.14),
                                                                         col_alignments="lrr")
        head = Tex(r"measured, real test image (spec 4.9b)", color=GREY_B).scale(0.48)
        tb = VGroup(head, tbl).arrange(DOWN, buff=0.18).move_to([3.55, 1.6, 0])
        self.say(r"On the real test image (spec 4.9b), fusion adds 16,907 px of centreline on faint vessels:"
                 r" median OD 0.04 under them, not 0.10.")
        self.play(FadeIn(tb), run_time=1.0)
        self.wait(2.0)
        pieces = SurroundingRectangle(VGroup(*cells[6:9]), color=INH, buff=0.07)
        self.say(r"But the additions are fragmented: short stretches that start and connect to nothing. Pieces: 241 to 755.")
        self.play(Create(pieces))
        self.wait(2.0)
        rule = VGroup(Tex(r"candidate (spec 4.9b):").scale(0.5),
                      Tex(r"start a trace: line evidence $C$ alone").scale(0.5),
                      Tex(r"continue it: fused $C + G$", color=G_COL).scale(0.5)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rule.move_to([3.55, -1.25, 0])
        box = SurroundingRectangle(rule, color=MUTED, buff=0.18, corner_radius=0.08)
        self.say(r"Candidate: seed from line evidence only; fused evidence lowers only the continuation threshold."
                 r" A weak cue extends an object, never starts one.")
        self.play(FadeIn(rule), Create(box), run_time=1.0)
        self.wait(3.0)


variants(BorderOwnership, module_globals=globals())
