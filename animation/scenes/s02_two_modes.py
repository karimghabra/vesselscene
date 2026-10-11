"""Chapter 2: two modes of vision (spec 1, 4.1, appendix A).

A feedforward sweep climbs the visual areas in about 100 ms, extracting local
features everywhere; then recurrent processing (horizontal and feedback
connections) groups them into objects, assigns figure and ground and reads
junctions (Lamme and Roelfsema 2000).  Today's annotator has the first sweep
and hand-written rules; the spec adds the second.  The chapter ends by
building the spec's architecture, the map the rest of the video returns to.

Render:  manim -ql scenes/s02_two_modes.py TwoModesShort TwoModesMedium TwoModesDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from scipy.ndimage import gaussian_filter, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import bookends as BK  # noqa: E402
from mica_anim import neuro as N  # noqa: E402
from mica_anim.archmap import EDGES, FEEDFORWARD  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (ATTN, CONT, END, EXC, FB, FF, INH, JUNC, MUTED, PixelFrame,  # noqa: E402
                             orientation_color)
from mica_anim.widgets import PolarTuning  # noqa: E402

# ------------------------------------------------------------------ the timeline (schematic)
# area, feedforward latency (ms), onset of the recurrent (late) component (ms); approximate,
# after Lamme and Roelfsema (2000): the sweep reaches every area within about 100 ms
AREAS = (("retina", 20, None), ("LGN", 32, 125), ("V1", 45, 108), ("V2", 60, 102), ("V4", 78, 100))
T_END = 200.0
X0, X1 = -4.85, 1.25                  # time axis, scene x
Y_AX = -2.3                           # time axis, scene y
LANE0, LANE_DY, LANE_H = -2.02, 0.95, 0.6
BOX_X = -6.05

REGION = (340, 40, 200)               # x0, y0, size (px): a window of the vesselscene still
PANEL_C, PANEL_H = np.array([4.5, 0.12, 0.0]), 4.45


def tx(t):
    return X0 + (X1 - X0) * t / T_END


def lane_y(i):
    return LANE0 + LANE_DY * i


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def responses(t, lat, t_rec):
    """A schematic population response: a feedforward transient at the
    area's latency settling to a sustained level, and a late, recurrent
    component (figure-ground modulation) from t_rec on."""
    x = np.clip((np.asarray(t, float) - lat) / 9.0, 0.0, None)
    ff = x * np.exp(1.0 - x) + 0.28 * (1.0 - np.exp(-x / 2.0))
    rec = np.zeros_like(ff) if t_rec is None else 0.42 * smoothstep((np.asarray(t, float) - t_rec) / 35.0)
    return ff, rec


@lru_cache(maxsize=None)
def features():
    """V1's local features on the window: 16 simple-cell orientations at two
    sizes on the OD (-ln I/B, B a smooth upper envelope), sampled every 9 px;
    each tick is the best orientation where the response clears 18 % of the
    window's maximum."""
    S = BK.scene()
    x0, y0, sz = REGION
    m = 24
    A = np.asarray(S.average[y0 - m if y0 >= m else 0:y0 + sz + m, x0 - m:x0 + sz + m], float)
    oy = m if y0 >= m else y0
    A = np.where(np.isfinite(A), A, np.nanmedian(A))
    B = gaussian_filter(maximum_filter(A, 25), 10)
    od = np.log(B / A)
    best = None
    for sig in (1.8, 3.6):
        U = N.simple_cells(od, sig, 3.0)
        U = U / U.max()
        best = U if best is None else np.maximum(best, U)
    best = best[:, oy:oy + sz, m:m + sz]
    M, K = best.max(0), best.argmax(0)
    ticks = []
    for yy in range(4, sz, 9):
        for xx in range(4, sz, 9):
            if M[yy, xx] > 0.18:
                ticks.append((xx, yy, N.THETAS[K[yy, xx]], float(M[yy, xx])))
    return ticks


@lru_cache(maxsize=None)
def toy_tuning():
    """A hypercolumn's tuning curve on a toy vessel (16 orientations)."""
    od = N.vessel_od((64, 64), [((6.0, 44.0), (58.0, 20.0), 3.0, 0.15)], blur=1.0)
    U = N.simple_cells(N.add_noise(od, 0.01, 2), 2.0, 4.0)
    U = N.noise_units(U, od < 0.005)
    return N.tuning_curve(U, (32, 32))


# appendix A, the rows this chapter shows: brain | pipeline today | this spec
TABLE = (("end", r"end-stopped cells", r"symbolic events in step 7", r"V2a: end-stopped cells (4.2)"),
         ("junc", r"V2 angle and junction cells", r"arm-counting rules", r"V2b: junction cells (4.3)"),
         ("cont", r"good continuation", r"same-trace rule (step 9)", r"V2c: co-circularity (4.4)"),
         ("transp", r"transparency at X-junctions", r"$\kappa$ in the render only", r"V2d: transparency (4.5)"),
         ("attn", r"incremental grouping", r"greedy parallel tracing", r"attentive grouping (4.10)"),
         ("fb", r"feedback, predictive coding", r"the fit, which never feeds back", r"at junctions (4.5b); to V1 (4.11)"))

# spec 1: the four groups of the recurrent phase, on the map
GROUPS = ((r"V2 junction computations, phases 1--3", JUNC,
           ("endstop", "junction", "contin", "transp", "compare", "surround", "scale")),
          (r"Recurrent grouping, phases 4--5", CONT, ("assoc", "arc", "border", "fusion", "readout", "attend")),
          (r"Top-down feedback and parameters, phases 6--7", FB, ("render", "fit", "popcode", "feedback")),
          (r"Motion, phase 8", TEAL_B, ("motion",)))
RATES = ("retina", "ganglion", "simple", "arc", "surround", "scale", "assoc", "endstop", "junction", "contin",
         "transp", "border", "fusion")
LINES = ("readout", "attend", "graph", "render", "popcode", "fit", "compare")


def glyph(kind):
    """Small icons in the fixed colours, one per row of the table."""
    if kind == "end":
        return VGroup(Line(LEFT * 0.22, RIGHT * 0.12, stroke_color=WHITE, stroke_width=4),
                      Dot(RIGHT * 0.14, radius=0.07, color=END))
    if kind == "junc":
        g = VGroup(*[Line(ORIGIN, 0.22 * np.array([np.cos(a), np.sin(a), 0]), stroke_color=WHITE, stroke_width=3.5)
                     for a in (PI / 2, PI * 7 / 6, -PI / 6)])
        return g.add(Circle(radius=0.1, stroke_color=JUNC, stroke_width=3))
    if kind == "cont":
        a = Arc(radius=0.35, start_angle=PI * 0.62, angle=0.45, stroke_color=WHITE, stroke_width=3.5)
        b = Arc(radius=0.35, start_angle=PI * 0.2, angle=0.45, stroke_color=WHITE, stroke_width=3.5)
        gap = DashedVMobject(Arc(radius=0.35, start_angle=PI * 0.2 + 0.45, angle=PI * 0.62 - PI * 0.2 - 0.45,
                                 stroke_color=CONT, stroke_width=3.5), num_dashes=3)
        return VGroup(a, b, gap).move_to(ORIGIN)
    if kind == "transp":
        r1 = Rectangle(width=0.55, height=0.12, stroke_width=0, fill_color=GREY_B, fill_opacity=0.6).rotate(0.6)
        r2 = Rectangle(width=0.55, height=0.12, stroke_width=0, fill_color=GREY_B, fill_opacity=0.6).rotate(-0.6)
        return VGroup(r1, r2)
    if kind == "attn":
        c = VMobject().set_points_smoothly([LEFT * 0.25 + DOWN * 0.1, LEFT * 0.08 + UP * 0.1, RIGHT * 0.08 + DOWN * 0.08,
                                            RIGHT * 0.25 + UP * 0.1])
        return VGroup(c.copy().set_stroke(GREY_B, 3), c.set_stroke(ATTN, 4).pointwise_become_partial(c, 0, 0.6))
    if kind == "fb":
        return CurvedArrow(RIGHT * 0.2 + UP * 0.1, LEFT * 0.2 + UP * 0.1, angle=PI * 0.9, color=FB, stroke_width=3,
                           tip_length=0.1).move_to(ORIGIN)
    raise KeyError(kind)


class TwoModes(BK.PacedCaptions, ChapterScene):
    NUMBER, TITLE, SPEC = 2, "Two modes of vision", "spec 1, 4.1, appendix A"
    MAP_KEYS = ()

    def construct(self):
        self.intro()
        self.timeline()
        if self.at(2):
            self.appendix_a()
            self.principle()
        m = self.build_map()
        if self.at(3):
            self.four_groups(m)
            self.conventions(m)
        self.outro()

    # =========================================================== 1. two modes, in time
    def timeline(self):
        T = ValueTracker(0.0)
        # ---- areas, lanes, axis
        boxes, cells, links = VGroup(), VGroup(), VGroup()
        for i, (name, lat, trec) in enumerate(AREAS):
            y = lane_y(i) + 0.3
            r = RoundedRectangle(width=1.35, height=0.62, corner_radius=0.08, stroke_color=GREY_B, stroke_width=1.6,
                                 fill_color=FF, fill_opacity=0.0).move_to([BOX_X, y, 0])
            lab = Tex(name, color=GREY_A).scale(0.48).move_to(r.get_left() + RIGHT * 0.38)
            ds = VGroup(*[Dot(radius=0.035, color=GREY_B) for _ in range(3)]).arrange(RIGHT, buff=0.12)
            ds.move_to(r.get_right() + LEFT * 0.32)
            boxes.add(VGroup(r, lab))
            cells.add(ds)
            if trec is not None:
                lk = VGroup(*[ArcBetweenPoints(ds[k].get_center() + UP * 0.04, ds[k + 1].get_center() + UP * 0.04, angle=-1.6,
                                               stroke_color=FB, stroke_width=2.2) for k in range(2)])
                links.add(lk)
            else:
                links.add(VGroup())
        ff_arrows = VGroup(*[Arrow(boxes[i][0].get_top() + LEFT * 0.38, boxes[i + 1][0].get_bottom() + LEFT * 0.38,
                                   buff=0.02, stroke_width=3, tip_length=0.1, max_tip_length_to_length_ratio=0.5, color=FF)
                             for i in range(4)])
        fb_arrows = VGroup(*[Arrow(boxes[i + 1][0].get_bottom() + RIGHT * 0.38, boxes[i][0].get_top() + RIGHT * 0.38,
                                   buff=0.02, stroke_width=3, tip_length=0.1, max_tip_length_to_length_ratio=0.5, color=FB)
                             for i in range(1, 4)])
        axis = Line([X0, Y_AX, 0], [X1 + 0.1, Y_AX, 0], stroke_color=MUTED, stroke_width=1.5)
        ticks = VGroup()
        for t in (0, 50, 100, 150, 200):
            ticks.add(Line([tx(t), Y_AX, 0], [tx(t), Y_AX - 0.08, 0], stroke_color=MUTED, stroke_width=1.5))
            ticks.add(Tex(f"{t}" + (r" ms" if t == 200 else ""), color=GREY_B).scale(0.42).next_to([tx(t), Y_AX - 0.08, 0],
                                                                                              DOWN, buff=0.06))
        bases = VGroup(*[Line([X0, lane_y(i), 0], [X1, lane_y(i), 0], stroke_color=MUTED, stroke_width=0.8,
                              stroke_opacity=0.5) for i in range(5)])
        src = Tex(r"schematic, after Lamme and Roelfsema (2000); latencies approximate",
                  color=GREY_B).scale(0.42).move_to([X0 - 1.85, 3.62, 0], aligned_edge=LEFT)
        ff_lab = Tex(r"feedforward sweep", color=FF).scale(0.58).move_to([tx(50), 3.1, 0])
        rc_lab = Tex(r"recurrent processing", color=FB).scale(0.58).move_to([tx(150), 3.1, 0])
        split = DashedLine([tx(100), Y_AX, 0], [tx(100), 3.3, 0], stroke_color=MUTED, stroke_width=1, dash_length=0.06,
                           stroke_opacity=0.5)

        cursor = always_redraw(lambda: Line([tx(T.get_value()), Y_AX, 0], [tx(T.get_value()), lane_y(4) + LANE_H + 0.1, 0],
                                            stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.6))

        def lane(i):
            name, lat, trec = AREAS[i]

            def f():
                tn = T.get_value()
                g = VGroup()
                if tn <= lat:
                    return g
                ts = np.linspace(0, tn, max(int(tn / 1.5), 3))
                ff, rec = responses(ts, lat, trec)
                y0 = lane_y(i)
                pf = [np.array([tx(t), y0 + LANE_H * v, 0]) for t, v in zip(ts, ff)]
                g.add(VMobject().set_points_as_corners(pf).set_stroke(FF, 2.5))
                if trec is not None and tn > trec:
                    keep = ts >= trec
                    top = [np.array([tx(t), y0 + LANE_H * (a + b), 0]) for t, a, b in zip(ts[keep], ff[keep], rec[keep])]
                    bot = [np.array([tx(t), y0 + LANE_H * a, 0]) for t, a in zip(ts[keep], ff[keep])]
                    if len(top) > 1:
                        fill = Polygon(*top, *bot[::-1], stroke_width=0, fill_color=FB, fill_opacity=0.3)
                        g.add(fill, VMobject().set_points_as_corners(top).set_stroke(FB, 2.5))
                return g
            return always_redraw(f)

        lanes = VGroup(*[lane(i) for i in range(5)])

        def light_boxes(mob):
            tn = T.get_value()
            for i, (name, lat, trec) in enumerate(AREAS):
                on = tn >= lat
                rec = trec is not None and tn >= trec
                mob[i][0].set_stroke(FB if rec else (FF if on else GREY_B), width=2.4 if on else 1.6)
                mob[i][0].set_fill(FB if rec else FF, opacity=0.16 if on else 0.0)
                cells[i].set_color(FF if on else GREY_B)
            for i in range(4):
                lat_up = AREAS[i + 1][1]
                ff_arrows[i].set_opacity(1.0 if tn >= lat_up - 4 else 0.0)
            for k, i in enumerate(range(1, 4)):
                trec_up = AREAS[i + 1][2]
                fb_arrows[k].set_opacity(1.0 if tn >= trec_up else 0.0)
            for i, (name, lat, trec) in enumerate(AREAS):
                links[i].set_stroke(opacity=1.0 if (trec is not None and tn >= trec) else 0.0)

        ff_arrows.set_opacity(0)
        fb_arrows.set_opacity(0)
        links.set_stroke(opacity=0)
        boxes.add_updater(light_boxes)

        # ---- the image panel
        x0, y0, sz = REGION
        crop = BK.average_rgba()[y0:y0 + sz, x0:x0 + sz]
        panel = BK.photo(crop, PANEL_H).move_to(PANEL_C)
        frame = SurroundingRectangle(panel, buff=0, color=MUTED, stroke_width=1)
        ppf = PixelFrame((sz, sz), PANEL_H, PANEL_C)
        feats = VGroup()
        for (x, y, th, v) in features():
            d = np.array([np.cos(-th), np.sin(-th), 0.0]) * 4.2 * ppf.unit
            c = ppf.point(x, y)
            feats.add(Line(c - d, c + d, stroke_color=orientation_color(th), stroke_width=2 + 2.5 * v))
        runs = BK.clipped_runs(x0, y0, sz, sz, ppf, width=3.5)
        rings = BK.rings_in(x0, y0, sz, sz, ppf, width=3.0)
        veil = Rectangle(width=PANEL_H, height=PANEL_H, stroke_width=0, fill_color=BLACK, fill_opacity=0.55).move_to(PANEL_C)
        p_t1 = Tex(r"V1: local features, everywhere", color=FF).scale(0.5).next_to(frame, UP, buff=0.15)
        p_t2 = Tex(r"objects, figure and ground, junctions", color=FB).scale(0.5).next_to(frame, UP, buff=0.15)

        # ---- play
        self.play(LaggedStart(FadeIn(boxes), FadeIn(cells), Create(axis), FadeIn(ticks), FadeIn(bases), FadeIn(src),
                              lag_ratio=0.15), FadeIn(panel), Create(frame), run_time=1.6)
        self.add(lanes, cursor, ff_arrows, fb_arrows, links)
        if not self.short:
            self.say(r"Show the eye an image, and follow each visual area through the first 200 milliseconds.")
        self.play(FadeIn(ff_lab), FadeIn(split), run_time=0.6)
        self.say(r"A feedforward wave climbs from the retina to V4 in about 100 ms, extracting local features everywhere.")
        self.play(T.animate.set_value(45), run_time=1.6, rate_func=linear)
        self.play(T.animate.set_value(100), FadeIn(p_t1), LaggedStart(*[Create(f) for f in feats], lag_ratio=0.01),
                  run_time=2.4, rate_func=linear)
        self.wait(0.3 if self.short else 1.2)
        if self.short:
            self.say(r"Then recurrent processing groups them into objects, assigns figure and ground, and reads junctions.")
        else:
            self.say(r"Then recurrent processing: horizontal and feedback connections group the features into objects$\ldots$")
        self.play(FadeIn(rc_lab), T.animate.set_value(150), FadeOut(feats), FadeIn(veil),
                  LaggedStart(*[Create(r) for r in runs], lag_ratio=0.08), ReplacementTransform(p_t1, p_t2),
                  run_time=3.0, rate_func=linear)
        if not self.short:
            self.say(r"$\ldots$assign figure and ground, and interpret junctions (Lamme and Roelfsema 2000).")
        self.play(T.animate.set_value(200), LaggedStart(*[GrowFromCenter(c) for c in rings], lag_ratio=0.1),
                  run_time=2.4, rate_func=linear)
        boxes.clear_updaters()
        if not self.short:
            self.wait(1.0)

        # ---- what the annotator has
        today = Tex(r"the annotator today, plus rules", color=FF).scale(0.45).next_to(ff_lab, DOWN, buff=0.1)
        spec = Tex(r"what this spec adds", color=FB).scale(0.45).next_to(rc_lab, DOWN, buff=0.1)
        hl_ff = Rectangle(width=tx(100) - X0, height=lane_y(4) + LANE_H + 0.1 - Y_AX, stroke_width=0, fill_color=FF,
                          fill_opacity=0.07).move_to([(X0 + tx(100)) / 2, (Y_AX + lane_y(4) + LANE_H + 0.1) / 2, 0])
        hl_rc = hl_ff.copy().set_fill(FB, 0.07).move_to([(tx(100) + X1) / 2, hl_ff.get_y(), 0])
        self.say(r"The annotator has only the first sweep, plus hand-written rules. This spec adds the second.")
        self.play(FadeIn(hl_ff), FadeIn(today), run_time=0.9)
        self.play(FadeIn(hl_rc), FadeIn(spec), run_time=0.9)
        self.wait(0.6 if self.short else 2.4)
        self.clear()

    # =========================================================== L2: appendix A
    def appendix_a(self):
        cols_x = (-6.05, -1.8, 2.35)
        widths = (4.0, 3.75, 4.6)
        head = VGroup(Tex(r"brain", color=WHITE), Tex(r"pipeline today", color=GREY_B), Tex(r"this spec", color=FB)).scale(0.72)
        for h, x in zip(head, cols_x):
            h.move_to([x, 3.0, 0], aligned_edge=LEFT)
        rule = Line([-6.85, 2.62, 0], [6.85, 2.62, 0], stroke_color=MUTED, stroke_width=1)
        src = Tex(r"spec, appendix A (six of its rows)", color=GREY_B).scale(0.42).move_to([6.85, 3.62, 0], aligned_edge=RIGHT)
        ys = (2.0, 1.1, 0.2, -0.7, -1.6, -2.5)
        rows = []
        for (kind, brain, now, new), y in zip(TABLE, ys):
            cells = []
            for k, (txt, x, w) in enumerate(zip((brain, now, new), cols_x, widths)):
                col = (GREY_A, GREY_B, FB)[k]
                t = Tex(txt, color=col).scale(0.6)
                if t.width > w:
                    t.scale_to_fit_width(w)
                cells.append(t.move_to([x, y, 0], aligned_edge=LEFT))
            g = glyph(kind).scale(1.1).move_to([-6.6, y, 0])
            rows.append((g, *cells))
        self.play(FadeIn(head), Create(rule), FadeIn(src), run_time=0.9)
        self.say(r"The spec's appendix A sets the brain beside the pipeline: each mid-level mechanism, what does its job today$\ldots$")
        for i, (g, b, now, new) in enumerate(rows[:2]):
            self.play(FadeIn(g, scale=0.6), FadeIn(b, shift=RIGHT * 0.1), run_time=0.5)
            self.play(FadeIn(now, shift=RIGHT * 0.1), run_time=0.45)
        self.say(r"$\ldots$and what replaces it. Symbolic events become end-stopped cells; arm counting becomes junction cells.")
        for g, b, now, new in rows[:2]:
            self.play(TransformFromCopy(now, new), run_time=0.7)
        self.wait(0.8)
        self.say(r"The same rule-to-population swap for continuation, transparency and grouping; and the fit, which never fed back, now does.")
        for g, b, now, new in rows[2:]:
            self.play(FadeIn(g, scale=0.6), FadeIn(b, shift=RIGHT * 0.1), FadeIn(now, shift=RIGHT * 0.1), run_time=0.5)
            self.play(TransformFromCopy(now, new), run_time=0.55)
        self.wait(1.8)
        self.clear()

    # =========================================================== L2: the principle
    def principle(self):
        words = VGroup(Tex(r"deterministic"), Tex(r"hand-built"), Tex(r"interpretable")).scale(0.85)
        words.arrange(RIGHT, buff=0.9).move_to(UP * 2.7)
        dots = VGroup(*[Dot(radius=0.05, color=FB).move_to((words[i].get_right() + words[i + 1].get_left()) / 2)
                        for i in range(2)])
        tc = toy_tuning()
        P = PolarTuning(radius=1.35, vmax=float(tc.max()) * 1.1, rings=()).move_to(LEFT * 3.4 + DOWN * 0.15)
        spokes = P.spokes(tc, width=3.5)
        curve = P.curve(tc, WHITE, width=2.5)
        p_lab = Tex(r"every map is a population of model cells\\with a stated tuning (here: 16 orientations)",
                    color=GREY_A).scale(0.48).next_to(P, DOWN, buff=0.3)
        # a trained network, as a glyph: three layers, every weight a free number
        layers = [VGroup(*[Dot(radius=0.06, color=GREY_B) for _ in range(n)]).arrange(DOWN, buff=0.28) for n in (5, 7, 5)]
        net = VGroup(*layers).arrange(RIGHT, buff=1.0).move_to(RIGHT * 3.4 + DOWN * 0.15)
        rng = np.random.default_rng(4)
        wires = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1.2, stroke_color=GREY_B,
                              stroke_opacity=float(rng.uniform(0.15, 0.6)))
                         for la, lb in zip(layers[:-1], layers[1:]) for a in la for b in lb])
        cross = VGroup(Line(UL, DR), Line(UR, DL)).scale(1.0).set_stroke(INH, 6).move_to(net)
        n_lab = Tex(r"no trained network\\(spec 3, 8)", color=GREY_A).scale(0.48).next_to(net, DOWN, buff=0.3).match_y(p_lab)
        self.say(r"And one principle runs through it: deterministic, hand-built, interpretable.")
        self.play(LaggedStart(*[FadeIn(w, shift=UP * 0.1) for w in words], lag_ratio=0.3), FadeIn(dots), run_time=1.4)
        self.say(r"Every map is a population of model neurons with a stated tuning, the kind you can draw like this$\ldots$")
        self.play(FadeIn(P.axes), LaggedStart(*[Create(s) for s in spokes], lag_ratio=0.08), run_time=1.6)
        self.play(Create(curve), FadeIn(p_lab), run_time=0.9)
        self.say(r"$\ldots$and no trained network: the architecture is built by hand, its few dozen parameters set or swept on development scenes.")
        self.play(FadeIn(net), Create(wires), run_time=1.2)
        self.play(Create(cross), FadeIn(n_lab), run_time=0.8)
        self.wait(1.8)
        self.clear()

    # =========================================================== 2. the map assembles
    def build_map(self):
        m = BK.fitted_map()
        rec_order = (("assoc",), ("endstop", "border"), ("junction", "fusion"), ("contin", "readout"),
                     ("transp", "attend"), ("compare", "graph"), ("popcode", "render"), ("fit", "motion"))
        arrow_of = {b: [i for i, (a, bb) in enumerate(EDGES) if bb == b] for _, b in EDGES}
        if self.short:
            self.say(r"The spec's architecture: a feedforward row, the sweep the annotator has today$\ldots$")
        else:
            self.say(r"The spec turns this into an architecture. First the feedforward row: the sweep the annotator has today.")
        self.play(FadeIn(m.labels[0], shift=RIGHT * 0.2), run_time=0.5)
        anims = []
        for k, key in enumerate(FEEDFORWARD):
            a = [GrowArrow(m.arrows[i]) for i in arrow_of.get(key, [])]
            anims.append(AnimationGroup(FadeIn(m.boxes[key], shift=RIGHT * 0.15), *a))
        self.play(LaggedStart(*anims, lag_ratio=0.35), run_time=2.2 if self.short else 2.6)
        if self.short:
            self.say(r"$\ldots$a recurrent stack of V2 and grouping layers$\ldots$")
        else:
            self.say(r"Then the recurrent stack: association field, V2's junction cells, border ownership, grouping, the graph.")
        self.play(FadeIn(m.labels[1], shift=RIGHT * 0.2), run_time=0.5)
        anims = []
        for keys in rec_order:
            grp = []
            for key in keys:
                grp.append(FadeIn(m.boxes[key], shift=DOWN * 0.1))
                grp += [GrowArrow(m.arrows[i]) for i in arrow_of.get(key, [])]
            anims.append(AnimationGroup(*grp))
        self.play(LaggedStart(*anims, lag_ratio=0.3), run_time=3.2 if self.short else 4.0)
        if self.short:
            self.say(r"$\ldots$and gold feedback loops. This is our map for the rest of the video.")
            self.play(*[Create(a) for a in m.feedback], FadeIn(m.labels[2]), run_time=1.6)
            self.wait(1.0)
        else:
            self.say(r"And gold feedback loops: the render's and the fit's prediction errors go back down to the early layers.")
            self.play(*[Create(a) for a in m.feedback], FadeIn(m.labels[2]), run_time=1.6)
            self.wait(0.8)
            self.say(r"This is our map. Each chapter from here on lights its own boxes.", wait="read")
        return m

    # =========================================================== L3: the four groups
    def four_groups(self, m):
        tag = None
        lines = (r"Phases 1--3: V2's junction computations (ends, junction cells, continuation, transparency, comparison), plus V1 fixes.",
                 r"Phases 4--5: recurrent grouping. A recurrent association field, arc cells, border ownership, attentive grouping.",
                 r"Phases 6--7: top-down feedback near ends and junctions, and population codes to start the fit.",
                 r"Phase 8: motion. In a real burst, flowing red cells mark the lumen and give the direction of flow.")
        self.say(r"The spec adds the recurrent phase in four groups, each gated before the next one starts (spec 1, 6.2).", wait="read")
        for (name, col, keys), line in zip(GROUPS, lines):
            new = Tex(name, color=col).scale(0.55).move_to([6.9, 3.78, 0], aligned_edge=RIGHT)
            anims = [BK.light(m, keys, color=col, dim=0.25)]
            anims.append(FadeIn(new) if tag is None else ReplacementTransform(tag, new))
            self.play(*anims, run_time=1.0)
            tag = new
            self.say(line, wait="read")
        self.play(BK.restore(m), FadeOut(tag), run_time=0.9)

    # =========================================================== L3: conventions (4.0)
    def conventions(self, m):
        self.say(r"Three conventions hold everywhere on the map (spec 4.0).", wait=1.2)
        self.play(FadeOut(m), run_time=0.8)

        # ---- drive and rate
        ax = Axes(x_range=[-6, 10, 2], y_range=[0, 10, 5], x_length=4.6, y_length=2.6,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5)).move_to(LEFT * 4.1 + DOWN * 0.55)
        relu = VGroup(Line(ax.c2p(-6, 0), ax.c2p(0, 0)), Line(ax.c2p(0, 0), ax.c2p(10, 10))).set_stroke(WHITE, 3)
        xl = Tex(r"drive", color=GREY_A).scale(0.48).next_to(ax.x_axis, DOWN, buff=0.12)
        yl = Tex(r"rate", color=GREY_A).scale(0.48).next_to(ax.y_axis, UP, buff=0.1)
        zero = Tex(r"0", color=GREY_B).scale(0.42).next_to(ax.c2p(0, 0), DOWN, buff=0.12)
        eq1 = MathTex(r"\text{drive}", "=", r"\text{excitation}", "-", r"\text{inhibition}").scale(0.58)
        eq1[2].set_color(EXC)
        eq1[4].set_color(INH)
        eq2 = MathTex(r"\text{rate}", "=", r"\max(0,\ \text{drive})").scale(0.58)
        eqs = VGroup(eq1, eq2).arrange(DOWN, buff=0.18).next_to(ax, UP, buff=0.45)
        title1 = Tex(r"drive and rate", color=WHITE).scale(0.6).next_to(eqs, UP, buff=0.3)
        d = ValueTracker(-5.0)
        probe = always_redraw(lambda: Dot(ax.c2p(d.get_value(), max(d.get_value(), 0)), radius=0.07, color=YELLOW))
        self.play(FadeIn(title1), Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(zero), run_time=0.9)
        self.say(r"A cell's drive is excitation minus inhibition, and can be negative; its rate is max(0, drive).")
        self.play(Write(eq1), run_time=1.0)
        self.play(Write(eq2), Create(relu), run_time=1.0)
        self.add(probe)
        self.play(d.animate.set_value(9.0), run_time=2.2, rate_func=smooth)
        self.wait(0.6)

        # ---- noise units
        rng = np.random.default_rng(11)
        sample = rng.normal(0.0, 1.0, 20000)
        hist, edges = np.histogram(sample, bins=np.arange(-4.0, 4.01, 0.5))
        ax2 = Axes(x_range=[-4, 10, 2], y_range=[0, 1, 1], x_length=4.8, y_length=2.6,
                   axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.5),
                   x_axis_config=dict(numbers_to_include=[0, 2, 4, 6, 8], font_size=22),
                   y_axis_config=dict(stroke_opacity=0)).move_to(RIGHT * 2.6 + DOWN * 0.55)
        bars = VGroup()
        for h, e0 in zip(hist / hist.max(), edges[:-1]):
            bars.add(Rectangle(width=ax2.x_length / 14 * 0.5 * 0.92, height=max(h * 2.4, 1e-3), stroke_width=0,
                               fill_color=GREY_B, fill_opacity=0.7).move_to(ax2.c2p(e0 + 0.25, 0), aligned_edge=DOWN))
        mark = DashedLine(ax2.c2p(8, 0), ax2.c2p(8, 0.95), stroke_color=YELLOW, stroke_width=2.5)
        mark_l = Tex(r"a vessel: 8", color=YELLOW).scale(0.48).next_to(mark, UP, buff=0.08)
        bl = Tex(r"background", color=GREY_B).scale(0.45).next_to(bars, UP, buff=0.1)
        xl2 = Tex(r"drive $\div$ the cell's spread on background", color=GREY_A).scale(0.45).next_to(ax2.x_axis, DOWN, buff=0.45)
        title2 = Tex(r"noise units", color=WHITE).scale(0.6).match_y(title1).match_x(ax2)
        self.say(r"Every drive is divided by that cell's spread on the background nearby, so 8 means eight times the noise$\ldots$")
        self.play(FadeIn(title2), Create(ax2), FadeIn(xl2), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05),
                  FadeIn(bl), run_time=1.4)
        self.play(Create(mark), FadeIn(mark_l), run_time=0.8)
        self.say(r"$\ldots$and a threshold means the same in a dim frame and a bright one.", wait="read")
        panels = VGroup(title1, ax, relu, xl, yl, zero, eqs, title2, ax2, bars, mark, mark_l, bl, xl2)
        self.remove(probe)
        self.play(FadeOut(panels), run_time=0.7)
        self.play(FadeIn(m), run_time=0.8)

        # ---- lines exist only from the readout on
        paint = {k: FF for k in RATES}
        paint.update({k: WHITE for k in LINES})
        self.say(r"And lines exist only from the readout on. Before it, every layer is rates in sheets, 16 per size channel.")
        self.play(self.paint(m, paint), run_time=1.2)
        sheets = VGroup()
        for k in range(16):
            sh = Polygon([-0.5, -0.2, 0], [0.3, -0.2, 0], [0.5, 0.2, 0], [-0.3, 0.2, 0], stroke_color=orientation_color(N.THETAS[k]),
                         stroke_width=1.2, fill_color=orientation_color(N.THETAS[k]), fill_opacity=0.18)
            sheets.add(sh.shift(UP * 0.045 * k))
        sheets.move_to([6.15, 2.0, 0])
        s_lab = Tex(r"rates in sheets", color=FF).scale(0.45).next_to(sheets, DOWN, buff=0.08)
        line_g = VGroup(VMobject().set_points_smoothly([[-0.55, -0.15, 0], [-0.1, 0.12, 0], [0.5, 0.0, 0]]).set_stroke(WHITE, 3),
                        Line([-0.05, 0.1, 0], [0.15, -0.35, 0], stroke_color=WHITE, stroke_width=3),
                        Circle(radius=0.11, stroke_color=JUNC, stroke_width=2.5).move_to([-0.05, 0.1, 0]))
        line_g.move_to([6.15, 0.55, 0])
        l_lab = Tex(r"lines, traces,\\junction lists", color=WHITE).scale(0.45).next_to(line_g, DOWN, buff=0.1)
        self.play(FadeIn(sheets, lag_ratio=0.05), FadeIn(s_lab), run_time=1.0)
        self.play(FadeIn(line_g), FadeIn(l_lab), run_time=0.8)
        self.wait(2.4)

    def paint(self, m, mapping, dim=0.2):
        """Colour boxes by a key -> colour mapping; the rest dimmed."""
        anims = []
        for k, box in m.boxes.items():
            if k in mapping:
                c = mapping[k]
                anims.append(box[0].animate.set_stroke(c, width=2.6, opacity=1.0).set_fill(c, opacity=0.2))
                anims.append(box[1].animate.set_opacity(1.0))
            else:
                anims.append(box[0].animate.set_stroke(opacity=dim).set_fill(opacity=0.02))
                anims.append(box[1].animate.set_opacity(dim + 0.1))
        anims += [BK.fade_arrow(a, 0.5) for a in m.feedback]
        return AnimationGroup(*anims)


variants(TwoModes, module_globals=globals())
