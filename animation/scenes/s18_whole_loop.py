"""Chapter 18: the whole loop, and how we'd know it works (spec 4.1, 6.2, 6.4, 1.1).

The map runs end to end: a pulse climbs the feedforward row, circles in the
recurrent loops, reaches the graph and the render, and returns along the gold
feedback arrows.  Then the two gates every change must pass, the inverse
crime, the build status painted on the map, why the network is hand-built,
and the open questions.  It closes on the annotated vesselscene still: the
target the loop is built to draw.

Render:  manim -ql scenes/s18_whole_loop.py WholeLoopShort WholeLoopMedium WholeLoopDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import bookends as BK  # noqa: E402
from mica_anim import neuro as N  # noqa: E402
from mica_anim.archmap import EDGES, FEEDFORWARD  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import FB, FF, INH, JUNC, MUTED, OD_COL, PixelFrame, image  # noqa: E402

IMG_H, IMG_C = 6.2, UP * 0.22

# spec 1.1 / 5.1: build status as of 11 October 2026, with MICA's layer numbers
EXISTING = {"retina": "", "ganglion": "", "simple": ""}
BUILT = {"surround": "03", "readout": "06", "endstop": "07", "junction": "08", "contin": "09", "transp": "10",
         "border": "11", "fusion": "13", "attend": "14", "scale": "new"}
OFF = {"assoc": "04"}
DESIGNED = {"arc": "05"}
NOT_STARTED = {"motion": "12", "graph": "15", "fit": "15", "compare": "16", "popcode": "17--20", "render": "17--20"}


@lru_cache(maxsize=None)
def residual_tiles():
    """Two toy residual maps in noise units: white noise (the model explains
    the image down to its noise), and noise plus a faint unexplained vessel
    (structured residual: a model mismatch)."""
    rng = np.random.default_rng(5)
    white = rng.normal(0, 1, (48, 48))
    od = N.vessel_od((48, 48), [((4.0, 40.0), (44.0, 10.0), 2.5, 1.0)], blur=1.0)
    return white, white + 1.6 * od


@lru_cache(maxsize=None)
def profile_curve():
    """OD across one vessel of the scene (vesselscene's clean OD): a bump on
    a background that should stay at 0."""
    S = BK.scene()
    od = np.asarray(S.od, float)
    return od[184, 75:175]


class WholeLoop(BK.PacedCaptions, ChapterScene):
    NUMBER, TITLE, SPEC = 18, r"The whole loop, and how we'd know it works", "spec 4.1, 6.2, 6.4, 1.1"
    MAP_KEYS = ()

    def construct(self):
        self.intro()
        m = self.run_loop()
        if self.at(2):
            self.clear()
            self.gates()
            self.inverse_crime()
            m = None
        if self.at(3):
            m = self.status()
            self.play(FadeOut(m), run_time=0.8)
            m = None
            self.hand_built()
            self.open_questions()
        self.finale(m)
        self.outro()

    # ================================================================ 1. the loop runs
    def run_loop(self):
        m = BK.fitted_map()
        for k, box in m.boxes.items():
            box[0].set_stroke(opacity=0.45)
            box[1].set_opacity(0.6)
        m.arrows.set_opacity(0.45)
        for a in m.feedback:
            a.set_stroke(opacity=0.45)
            a.tip.set_fill(opacity=0.45)
        self.play(FadeIn(m), run_time=1.0)
        c = {k: b.get_center() for k, b in m.boxes.items()}

        lit = set()

        def lighter(dt):
            for p in pulses:
                q = p.get_center()
                for k, b in m.boxes.items():
                    if k in lit:
                        continue
                    r = b[0]
                    if abs(q[0] - r.get_x()) < r.width / 2 + 0.02 and abs(q[1] - r.get_y()) < r.height / 2 + 0.02:
                        col = FF if k in FEEDFORWARD else WHITE
                        r.set_stroke(col, width=3.0, opacity=1.0).set_fill(col, opacity=0.28)
                        b[1].set_opacity(1.0)
                        lit.add(k)

        def path(*keys, via=None):
            pts = [c[keys[0]]]
            for a, b in zip(keys[:-1], keys[1:]):
                if (a, b) in EDGES:
                    arr = BK.edge_arrow(m, a, b)
                    pts += [arr.get_start(), arr.get_end()]
                pts.append(c[b])
            v = VMobject()
            v.set_points_as_corners(pts)
            return v

        pulses = []

        def pulse(col=WHITE):
            p = BK.glow_dot(col, radius=0.06)
            pulses.append(p)
            return p

        def trail(p, col):
            return TracedPath(p.get_center, stroke_color=col, stroke_width=4, dissipating_time=0.35, stroke_opacity=[0, 1])

        self.add_updater(lighter)
        m.add_updater(lambda mob: None)          # redraw the map every frame, so boxes light as the pulse passes
        # ---- the feedforward sweep
        p0 = pulse(FF).move_to(c["retina"])
        t0 = trail(p0, FF)
        self.add(t0, p0)
        self.say(r"Run the whole loop. First the feedforward sweep, from OD to scale selection$\ldots$")
        self.play(MoveAlongPath(p0, path(*FEEDFORWARD)), run_time=2.6 if self.short else 3.0, rate_func=linear)
        self.play(MoveAlongPath(p0, path("scale", "assoc")), run_time=0.8, rate_func=linear)
        # ---- the association field iterates
        loop = m.boxes["assoc"][0].copy().scale(1.25)
        self.say(r"$\ldots$then the recurrent loops: the association field iterates, V2 reads ends and junctions, grouping collects vessels$\ldots$")
        self.play(MoveAlongPath(p0, loop), run_time=1.2, rate_func=linear)
        self.play(MoveAlongPath(p0, Line(p0.get_center(), c["assoc"])), run_time=0.2)
        # ---- V2 and grouping, in parallel
        pl, pm, pr = pulse(), pulse(), pulse()
        for p in (pl, pm, pr):
            p.move_to(c["assoc"])
        tl, tm, tr = trail(pl, WHITE), trail(pm, WHITE), trail(pr, WHITE)
        self.remove(p0, t0)
        pulses.remove(p0)
        self.add(tl, tm, tr, pl, pm, pr)
        self.play(MoveAlongPath(pl, path("assoc", "endstop", "junction", "contin", "transp", "compare")),
                  MoveAlongPath(pm, path("assoc", "endstop", "junction", "contin", "attend")),
                  MoveAlongPath(pr, path("assoc", "border", "fusion", "readout", "attend")),
                  run_time=2.8 if self.short else 3.2, rate_func=linear)
        self.remove(pl, tl, pr, tr)
        pulses.remove(pl)
        pulses.remove(pr)
        # ---- the graph, render and fit
        pf_ = pulse()
        pf_.move_to(c["graph"])
        tf = trail(pf_, WHITE)
        self.play(MoveAlongPath(pm, path("attend", "graph")), run_time=0.6, rate_func=linear)
        self.add(tf, pf_)
        self.say(r"$\ldots$into a graph, which is rendered and fitted$\ldots$")
        self.play(MoveAlongPath(pm, path("graph", "render")), MoveAlongPath(pf_, path("graph", "popcode", "fit")),
                  run_time=1.6, rate_func=linear)
        self.wait(0.3)
        # ---- feedback
        arcs = []
        for a in m.feedback:
            arc = a.copy()
            arc.pop_tips()
            arcs.append(arc)
        gr, gf = pulse(FB), pulse(FB)
        gr.move_to(arcs[0].get_start())
        gf.move_to(arcs[1].get_start())
        tgr, tgf = trail(gr, FB), trail(gf, FB)
        self.remove(pm, tm, pf_, tf)
        pulses.remove(pm)
        pulses.remove(pf_)
        self.add(tgr, tgf, gr, gf)
        self.say(r"$\ldots$and their prediction errors return along the feedback loops, to guide the next pass.")
        self.play(*[BK.fade_arrow(a, 1.0) for a in m.feedback],
                  MoveAlongPath(gr, arcs[0]), MoveAlongPath(gf, arcs[1]), run_time=2.0, rate_func=smooth)
        self.remove_updater(lighter)
        m.clear_updaters()
        self.play(m.boxes["assoc"][0].animate.set_stroke(FB, width=4).set_fill(FB, opacity=0.3),
                  Flash(c["assoc"], color=FB, line_length=0.25, flash_radius=0.9), FadeOut(gr), FadeOut(gf), run_time=0.9)
        self.remove(tgr, tgf)
        self.wait(1.2 if self.short else 1.6)
        return m

    # ================================================================ L2: the two gates
    def gates(self):
        S = BK.scene()
        # ---- synthetic gate
        h1 = Tex(r"Synthetic gate", color=WHITE).scale(0.7)
        h1s = Tex(r"against truth", color=GREY_B).scale(0.5)
        hd1 = VGroup(h1, h1s).arrange(DOWN, buff=0.1).move_to([-3.75, 3.2, 0])
        th_h = 2.45
        pf = PixelFrame(S.shape, th_h, np.array([-3.75, 1.2, 0]))
        thumb = BK.photo(BK.average_rgba(), th_h).move_to(pf.centre)
        t_runs = BK.truth_runs(pf, width=1.4)
        t_rings = BK.truth_rings(pf, width=1.4, rmin=4, rmax=8)
        dimr = Rectangle(width=pf.width, height=pf.height, stroke_width=0, fill_color=BLACK, fill_opacity=0.5).move_to(pf.centre)
        syn = VGroup(Tex(r"vesselscene's development scenes,", color=GREY_A),
                     Tex(r"exact truth for every vessel and junction", color=GREY_A),
                     Tex(r"the change must improve its target:", color=GREY_B),
                     Tex(r"paired test with guards (6.1, 6.2)", color=GREY_B),
                     Tex(r"held-out scenes only at the end", color=GREY_B)).scale(0.48)
        syn.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(thumb, DOWN, buff=0.3)
        sep = DashedLine([-0.95, 3.5, 0], [-0.95, -2.85, 0], stroke_color=MUTED, stroke_width=1, dash_length=0.08)

        # ---- real gate
        h2 = Tex(r"Real gate", color=WHITE).scale(0.7)
        h2s = Tex(r"little or no truth", color=GREY_B).scale(0.5)
        hd2 = VGroup(h2, h2s).arrange(DOWN, buff=0.1).move_to([3.0, 3.2, 0])
        ys = (2.05, 1.08, 0.11, -0.86, -1.83)
        ix, tx_ = 0.0, 1.05
        icons, texts = [], []
        # 1 split-half: odd frames against even frames (computed from the burst)
        fr = BK.burst()
        x0, y0, sz = 380, 200, 120
        halves = Group()
        for par in (1, 0):
            if len(fr) >= 2:
                arr = BK.display_rgba(BK.half_mean(par))[y0:y0 + sz, x0:x0 + sz]
            else:
                arr = BK.average_rgba()[y0:y0 + sz, x0:x0 + sz]
            halves.add(BK.photo(arr, 0.74))
        halves.arrange(RIGHT, buff=0.08)
        icons.append(halves)
        texts.append(VGroup(Tex(r"split-half agreement: the chain on the", color=GREY_A),
                            Tex(r"odd-frame average against the even-frame one", color=GREY_A)))
        # 2 OD on the background stays 0
        row = profile_curve()
        ax = Axes(x_range=[0, len(row), len(row)], y_range=[0, row.max() * 1.15, 1], x_length=1.5, y_length=0.66,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2))
        prof = ax.plot_line_graph(np.arange(len(row)), row, line_color=OD_COL, add_vertex_dots=False, stroke_width=2.5)
        icons.append(VGroup(ax, prof))
        texts.append(VGroup(Tex(r"the OD left on the background", color=GREY_A),
                            Tex(r"must stay at 0", color=GREY_A)))
        # 3 whitened residual: noise-like, about 1 everywhere
        white, struct = residual_tiles()
        tiles = Group(image(white, 0.74, "diverging", vmin=-3, vmax=3), image(struct, 0.74, "diverging", vmin=-3, vmax=3))
        tiles.arrange(RIGHT, buff=0.08)
        icons.append(tiles)
        texts.append(VGroup(Tex(r"the fit's whitened residual $\approx 1$ everywhere;", color=GREY_A),
                            Tex(r"structure in it marks a model mismatch", color=GREY_B)))
        # 4 physiological plausibility
        mur = MathTex(r"r^3 = r_1^3 + r_2^3", color=WHITE).scale(0.56)
        icons.append(mur)
        texts.append(VGroup(Tex(r"plausible: widths continuous along a vessel,", color=GREY_A),
                            Tex(r"Murray's law at forks, flow conserved", color=GREY_A)))
        # 5 hand-annotated windows
        win = BK.photo(BK.average_rgba()[150:270, 160:280], 0.72)
        wbox = DashedVMobject(Square(side_length=0.72, stroke_color=YELLOW, stroke_width=2), num_dashes=16).move_to(win)
        icons.append(Group(win, wbox))
        texts.append(VGroup(Tex(r"windows of a real still annotated by hand:", color=GREY_A),
                            Tex(r"recall of junctions and ends", color=GREY_A)))
        rows = Group()
        for ic, tt, y in zip(icons, texts, ys):
            tt.scale(0.48).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            if ic.width > 1.6:
                ic.scale_to_fit_width(1.6)
            ic.move_to([ix, y, 0])
            tt.move_to([tx_, y, 0], aligned_edge=LEFT)
            rows.add(Group(ic, tt))

        self.say(r"How would we know it works? Every change must pass two gates (spec 6.4). The first is synthetic$\ldots$")
        self.play(FadeIn(hd1), FadeIn(thumb), run_time=0.9)
        self.play(FadeIn(dimr), LaggedStart(*[Create(r) for r in t_runs], lag_ratio=0.01),
                  *[FadeIn(g) for g in t_rings.values()], run_time=1.6)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.08) for s in syn], lag_ratio=0.2), run_time=1.4)
        self.say(r"$\ldots$on vesselscene's development scenes, against exact truth: the change must improve its target.", wait=1.0)
        self.play(Create(sep), FadeIn(hd2), run_time=0.8)
        self.say(r"The second is on real bursts, where there is little or no truth: checks that need no annotation$\ldots$")
        for r in rows[:4]:
            self.play(FadeIn(r[0], scale=0.8), FadeIn(r[1], shift=RIGHT * 0.1), run_time=0.7)
            self.wait(0.9)
        self.say(r"$\ldots$and a few windows of a real still, annotated by hand, every junction and every end.")
        self.play(FadeIn(rows[4][0], scale=0.8), FadeIn(rows[4][1], shift=RIGHT * 0.1), run_time=0.7)
        self.wait(1.0)
        self.say(r"A change that wins the first gate and loses the second is not accepted.", wait="read")
        self.clear()

    # ================================================================ L2: the inverse crime
    def inverse_crime(self):
        # the tube model: a cylinder and the chord law it gives
        x = np.linspace(-1.6, 1.6, 161)
        prof = N.chord(np.abs(x), 1.0, 1.0)
        ax = Axes(x_range=[-1.6, 1.6, 1], y_range=[0, 1.2, 1], x_length=2.6, y_length=1.2,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2), y_axis_config=dict(stroke_opacity=0))
        curve = ax.plot_line_graph(x, prof, line_color=OD_COL, add_vertex_dots=False, stroke_width=3)
        circ = Circle(radius=0.38, stroke_color=OD_COL, stroke_width=2.5, fill_color=OD_COL, fill_opacity=0.25)
        circ.next_to(ax, DOWN, buff=0.12)
        model = VGroup(ax, curve, circ)
        m_lab = Tex(r"the tube model", color=WHITE).scale(0.58)
        m_sub = Tex(r"chord-law cylinders, blur,\\union at nodes, $\kappa$ at crossings", color=GREY_B).scale(0.45)
        model_g = VGroup(m_lab, model, m_sub).arrange(DOWN, buff=0.18).move_to([-4.0, 0.55, 0])
        # the synthetic image
        crop = BK.average_rgba()[180:330, 380:530]
        syn = BK.photo(crop, 2.3)
        s_lab = Tex(r"a vesselscene image", color=WHITE).scale(0.58).next_to(syn, UP, buff=0.18)
        syn_g = Group(s_lab, syn).move_to([3.7, 0.75, 0])
        top = CurvedArrow(model.get_right() + RIGHT * 0.25 + UP * 0.45, syn.get_left() + UP * 0.55 + LEFT * 0.15, angle=-0.6,
                          color=GREY_A, stroke_width=3, tip_length=0.18)
        top_l = Tex(r"vesselscene renders with it", color=GREY_A).scale(0.5).next_to(top, UP, buff=0.1)
        bot = CurvedArrow(syn.get_left() + DOWN * 0.55 + LEFT * 0.15, model.get_right() + RIGHT * 0.25 + DOWN * 0.45, angle=-0.6,
                          color=GREY_A, stroke_width=3, tip_length=0.18)
        bot_l = Tex(r"the fit inverts it", color=GREY_A).scale(0.5).next_to(bot, DOWN, buff=0.1)
        crime = Tex(r"the inverse crime", color=INH).scale(0.7).move_to([-0.1, 0.55, 0])
        crime_s = Tex(r"(Kaipio and Somersalo 2007)", color=GREY_B).scale(0.45).next_to(crime, DOWN, buff=0.1)
        self.say(r"There is a trap in the synthetic gate. vesselscene draws its vessels with the same tube model the fit inverts.")
        self.play(FadeIn(model_g), run_time=1.0)
        self.play(Create(top), FadeIn(top_l), FadeIn(syn_g), run_time=1.3)
        self.play(Create(bot), FadeIn(bot_l), run_time=1.2)
        self.say(r"On synthetic data the fit's model is exactly right: the inverse crime. Its scores there overstate real performance.")
        self.play(Write(crime), FadeIn(crime_s), run_time=1.2)
        self.wait(1.6)
        real = DashedVMobject(RoundedRectangle(width=3.5, height=1.0, corner_radius=0.1, stroke_color=GREY_B, stroke_width=2),
                              num_dashes=40)
        real_t = Tex(r"real vessels follow the model\\only approximately", color=GREY_A).scale(0.45).move_to(real)
        real_g = VGroup(real, real_t).move_to([3.7, -2.25, 0])
        q = DashedLine(real_g.get_left(), model_g.get_corner(DR) + LEFT * 0.4 + DOWN * 0.1, stroke_color=GREY_B,
                       stroke_width=2, dash_length=0.1)
        self.say(r"Real profiles, blur and junctions follow it only approximately. Hence the second gate, and renders with a deliberately different model.")
        self.play(FadeIn(real_g), Create(q), run_time=1.2)
        self.wait(2.4)
        if self.at(3):
            back = CurvedArrow(real_g.get_top() + UP * 0.05, syn.get_bottom() + DOWN * 0.08, angle=0.5, color=FB,
                               stroke_width=3, tip_length=0.16)
            back_l = Tex(r"what the real residuals show is missing\\goes into the generator", color=FB).scale(0.45)
            back_l.next_to(back, LEFT, buff=0.15).shift(UP * 0.05)
            self.say(r"Where the gates disagree, the real residuals show what vesselscene lacks; that is added to the generator, not tuned around in the annotator.")
            self.play(Create(back), FadeIn(back_l), run_time=1.2)
            self.wait(3.0)
        self.clear()

    # ================================================================ L3: build status
    def status(self):
        m = BK.fitted_map(scale=0.74, bottom=-2.95)
        m.shift(LEFT * (m.get_right()[0] - 3.95))
        self.say(r"Where the build stands (spec 1.1, 5.1): the layers are being built in MICA, in their own numbered order.")
        self.play(FadeIn(m), run_time=0.9)
        anims, tags = [], VGroup()
        for k, box in m.boxes.items():
            r, t = box
            if k in EXISTING:
                anims.append(r.animate.set_stroke(FF, width=2, opacity=1).set_fill(FF, opacity=0.14))
            elif k in BUILT:
                anims.append(r.animate.set_stroke(WHITE, width=3, opacity=1).set_fill(WHITE, opacity=0.3))
            elif k in OFF:
                anims.append(r.animate.set_stroke(WHITE, width=3, opacity=0.8).set_fill(GREY_D, opacity=0.6))
            elif k in DESIGNED:
                anims.append(r.animate.set_stroke(opacity=0).set_fill(opacity=0.0))
            else:
                anims.append(r.animate.set_stroke(GREY_B, width=1.2, opacity=0.3).set_fill(opacity=0.0))
                anims.append(t.animate.set_opacity(0.35))
            num = {**BUILT, **OFF, **DESIGNED, **NOT_STARTED}.get(k, "")
            if num:
                tg = Tex(num, color=GREY_A if k not in NOT_STARTED else GREY_C).scale(0.38)
                tg.next_to(r.get_corner(UR), UP, buff=0.03).shift(LEFT * 0.12)
                tags.add(tg)
        dashed = VGroup(*[DashedVMobject(m.boxes[k][0].copy().set_stroke(WHITE, 2, opacity=1), num_dashes=26) for k in DESIGNED])
        off_tags = VGroup(*[Tex(r"off", color=INH).scale(0.36).next_to(m.boxes[k][0], LEFT, buff=0.08) for k in OFF])
        anims.append(m.arrows.animate.set_opacity(0.35))
        anims += [BK.fade_arrow(a, 0.3) for a in m.feedback]

        # legend
        def swatch(stroke, fill, op, dashed_=False, sop=1.0, width=2.5):
            r = RoundedRectangle(width=0.5, height=0.26, corner_radius=0.05, stroke_color=stroke, stroke_width=width,
                                 stroke_opacity=sop, fill_color=fill, fill_opacity=op)
            return DashedVMobject(r, num_dashes=12) if dashed_ else r
        items = [(swatch(FF, FF, 0.14, width=2), r"today's steps"),
                 (swatch(WHITE, WHITE, 0.3, width=3), r"built"),
                 (swatch(WHITE, GREY_D, 0.6, width=3, sop=0.8), r"built, switched off"),
                 (swatch(WHITE, BLACK, 0.0, dashed_=True, width=2), r"designed, not built"),
                 (swatch(GREY_B, BLACK, 0.0, sop=0.3, width=1.2), r"not started")]
        legend = VGroup(*[VGroup(s, Tex(t, color=GREY_A).scale(0.46)).arrange(RIGHT, buff=0.18) for s, t in items])
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.55, 1.55, 0])
        note = Tex(r"no gate from phase 1 on\\has been measured\\against truth yet", color=GREY_A).scale(0.46)
        note_box = SurroundingRectangle(note, buff=0.15, color=FB, stroke_width=2, corner_radius=0.08)
        note_g = VGroup(note_box, note).move_to([5.55, -1.0, 0])
        note_s = Tex(r"spec 6.2", color=GREY_B).scale(0.4).next_to(note_g, DOWN, buff=0.1)
        self.play(*anims, FadeIn(legend, lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(tags), Create(dashed), FadeIn(off_tags), run_time=1.0)
        self.say(r"Built: the surround fix, readout, end-stopped and junction cells, continuation, transparency, grouping cells, fusion, attentive grouping, scale selection.",
                 wait="read")
        self.say(r"Built but switched off: the recurrent field (layer 04) failed its gate on the real test image. Arc cells are designed, not built.")
        self.play(Indicate(m.boxes["assoc"], color=INH, scale_factor=1.08), run_time=1.2)
        self.wait(5.0)
        self.say(r"Not started: flicker, the graph and fit, model comparison, and the feedback and population-code layers.", wait="read")
        self.say(r"And no gate from phase 1 on has been measured against truth yet.")
        self.play(Create(note_box), FadeIn(note), FadeIn(note_s), run_time=1.0)
        self.wait(2.2)
        self.play(FadeOut(VGroup(legend, note_g, note_s, tags, dashed, off_tags)), run_time=0.6)
        return m

    # ================================================================ L3: why hand-built
    def hand_built(self):
        # a few dozen physical parameters
        cats = ((r"widths (px)", 9), (r"angles ($^\circ$)", 8), (r"curvatures", 7), (r"thresholds (noise units)", 10))
        left = VGroup()
        for name, n in cats:
            knobs = VGroup()
            for i in range(n):
                k = VGroup(Circle(radius=0.11, stroke_color=WHITE, stroke_width=2),
                           Line(ORIGIN, 0.1 * np.array([np.cos(1.0 + 0.7 * i), np.sin(1.0 + 0.7 * i), 0]),
                                stroke_color=FB, stroke_width=2.5))
                knobs.add(k)
            knobs.arrange(RIGHT, buff=0.1)
            left.add(VGroup(Tex(name, color=GREY_A).scale(0.46), knobs).arrange(DOWN, aligned_edge=LEFT, buff=0.1))
        left.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.6, -0.2, 0])
        l_head = Tex(r"hand-built: a few dozen physical parameters", color=WHITE).scale(0.56).next_to(left, UP, buff=0.4)
        # millions of learned weights (not to scale)
        rng = np.random.default_rng(8)
        pts = rng.uniform([-2.2, -1.6], [2.2, 1.6], (2600, 2))
        cloud = VGroup(*[Dot([x + 3.6, y - 0.2, 0], radius=0.012, color=GREY_B, fill_opacity=float(o))
                         for (x, y), o in zip(pts, rng.uniform(0.25, 0.9, len(pts)))])
        r_head = Tex(r"trained: millions of learned weights", color=WHITE).scale(0.56).move_to([3.6, l_head.get_y(), 0])
        r_note = Tex(r"(not to scale)", color=GREY_B).scale(0.42).next_to(cloud, DOWN, buff=0.15)
        self.say(r"Why hand-built (spec 8)? A few dozen physical parameters, in pixels, degrees and noise units$\ldots$")
        self.play(FadeIn(l_head), LaggedStart(*[FadeIn(g, shift=UP * 0.1) for g in left], lag_ratio=0.25), run_time=1.6)
        self.say(r"$\ldots$should transfer to real frames more predictably than millions of weights learned on synthetic scenes.")
        self.play(FadeIn(r_head), FadeIn(cloud, lag_ratio=0.0005), FadeIn(r_note), run_time=1.6)
        self.wait(1.4)
        self.say(r"The middle ground (4.15): tune those few dozen by gradient on vesselscene's dense labels, the architecture still built by hand$\ldots$")
        rng = np.random.default_rng(3)
        turns = [Rotate(k[1], angle=float(rng.uniform(-1.2, 1.2)), about_point=k[0].get_center())
                 for g in left for k in g[1]]
        self.play(LaggedStart(*turns, lag_ratio=0.03), run_time=2.4)
        self.say(r"$\ldots$and accepted only through the same two gates. A trained recurrent network remains the obvious comparison once phases 1 to 5 exist.",
                 wait="read")
        self.clear()

    # ================================================================ L3: open questions
    def open_questions(self):
        qs = (r"Which $\kappa$ for transparency: vesselscene's reference 0.87, or the image's fitted 0.94?",
              r"How much of the recurrence is needed? Which loop carries the gain?",
              r"Why did the recurrent field fail its gate, and would lobes that follow the arc fix it?",
              r"How far does this get toward the 95\% bar, with the fit still to come?")
        cards = VGroup()
        for q in qs:
            t = Tex(r"\parbox{5.0cm}{\raggedright " + q + "}", color=GREY_A).scale(0.6)
            mark = Tex(r"?", color=FB).scale(1.3)
            body = VGroup(mark, t).arrange(RIGHT, buff=0.3)
            box = RoundedRectangle(width=6.3, height=1.75, corner_radius=0.12, stroke_color=MUTED, stroke_width=1.5)
            cards.add(VGroup(box, body.move_to(box)))
        cards.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.35)).move_to(UP * 0.35)
        head = Tex(r"Open questions (spec 10)", color=WHITE).scale(0.65).next_to(cards, UP, buff=0.35)
        self.say(r"And questions the spec leaves open. Each is to be settled by a measurement, not by taste.")
        self.play(FadeIn(head), LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cards], lag_ratio=0.3), run_time=2.2)
        self.wait(5.5)
        self.clear()

    # ================================================================ the finale
    def finale(self, m):
        pf = BK.scene_frame(IMG_H, IMG_C)
        if m is None:
            m = BK.fitted_map()
            self.play(FadeIn(m), run_time=0.8)
        g = m.boxes["graph"]
        others = [b for k, b in m.boxes.items() if k != "graph"] + [m.arrows, m.feedback, m.labels]
        self.play(*[FadeOut(o) for o in others], g.animate.set_stroke(WHITE, width=3, opacity=1).set_fill(WHITE, 0.15),
                  run_time=0.9)
        img = BK.photo(BK.average_rgba(), IMG_H).move_to(IMG_C)
        dim = Rectangle(width=pf.width, height=pf.height, stroke_width=0, fill_color=BLACK, fill_opacity=0.5).move_to(IMG_C)
        frame = Rectangle(width=pf.width, height=pf.height, stroke_color=WHITE, stroke_width=1.5).move_to(IMG_C)
        small = Group(img, dim)
        small.scale(g.width / pf.width).move_to(g)
        runs = BK.truth_runs(pf, width=2.4)
        rings = BK.truth_rings(pf, width=2.6)
        tag = Tex(r"the target: vesselscene's observable truth", color=GREY_B).scale(0.45)
        tag.move_to([-pf.width / 2, (IMG_C[1] + IMG_H / 2 + 4.0) / 2, 0], aligned_edge=LEFT)
        self.say(r"What the whole loop is built to draw: every vessel with its identity, every junction with its type.")
        self.add(small, g)
        self.play(small.animate.scale(pf.width / small.width).move_to(IMG_C), ReplacementTransform(g[0], frame),
                  FadeOut(g[1], scale=3.0), FadeIn(tag), run_time=1.4)
        self.play(FadeOut(frame), run_time=0.4)
        self.play(LaggedStart(*[Create(r) for r in runs], lag_ratio=0.012), run_time=2.6)
        self.play(LaggedStart(*[GrowFromCenter(c) for gname in BK.GROUPS for c in rings[gname]], lag_ratio=0.006),
                  run_time=1.4)
        self.say(r"A fast sweep, then recurrence: the second mode of vision, put to work on vessels.", wait="read")
        self.unsay()
        self.wait(1.2)


variants(WholeLoop, module_globals=globals())
