"""Chapter 0: cold open, from flicker to graph (spec 1, 2.1, 2.2).

A burst of vesselscene frames flickers, averages into the still, and the
annotation the video is about draws itself over it.  Then the problem (the
measured scorecard of today's pipeline), the question (fork or crossing?)
and the video's title.

Render:  manim -ql scenes/s00_cold_open.py ColdOpenShort ColdOpenMedium ColdOpenDetailed
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
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import END, FB, FF, JUNC, MUTED, PixelFrame, image  # noqa: E402

# measured, spec 2.1: today's pipeline, the 500 junctions of the six held-out averages
#            n,  found, right type, render >= 95 %, all three  (percent)
SCORECARD = {"3-way": (176, 38, 24, 52, 11),
             "crossing": (135, 73, 36, 53, 23),
             "compound": (189, 74, 42, 54, 30),
             "all": (500, 61, 34, 53, 21)}
TOWER_OF = {"fork": "3-way", "crossing": "crossing", "compound": "compound"}

CROSSING_ID, FORK_ID = 85, 128          # two junctions of the scene that look alike
ZOOM = (651, 324, 60)                   # x0, y0, size (px): small vessels whose red cells move between frames
ZOOM_C = np.array([3.95, 1.6, 0.0])     # where their close-up sits
CROP = 50                               # half-size of their windows, px
IMG_H, IMG_C = 6.2, UP * 0.22           # the still, above the caption line
FLICKER_FPS = 8.0


def crop_window(jid):
    j = BK.junction(jid)
    h, w = BK.scene().shape
    x0 = int(np.clip(round(j["x"]) - CROP, 0, w - 2 * CROP))
    y0 = int(np.clip(round(j["y"]) - CROP, 0, h - 2 * CROP))
    return x0, y0


def clipped_runs(x0, y0, size, pf, vids=None):
    """The truth runs inside a window (of the given vessels), each cut to the
    window, in identity colours."""
    g = VGroup()
    for r in BK.scene().truth["runs"]:
        if vids is not None and r["vid"] not in vids:
            continue
        xy = np.asarray(r["xy"], float) - [x0, y0]
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= size - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= size - 1)
        if inside.sum() < 4:
            continue
        idx = np.flatnonzero(inside)
        for piece in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(piece) < 4:
                continue
            m = VMobject()
            m.set_points_smoothly([pf.point(x, y) for x, y in xy[piece][::2]])
            g.add(m.set_stroke(BK.D.identity_color(r["vid"]), width=5, opacity=0.95))
    return g


@lru_cache(maxsize=None)
def stopped_branch():
    """Spec 2.2 on a toy: a parent vessel and a branch whose last stretch
    before the parent is faint (a trace stops there).  OD, union at the node."""
    shape = (120, 220)
    p0, p1 = (0.0, 38.0), (219.0, 38.0)
    node = (104.0, 38.0)
    knee = (113.0, 60.0)
    foot = (150.0, 119.0)
    od = N.vessel_od(shape, [(p0, p1, 4.5, 0.17), (foot, knee, 3.0, 0.13), (knee, node, 3.0, 0.035)],
                     blur=1.2, combine="max")
    I = N.to_intensity(N.add_noise(od, 0.008, seed=3), 0.85)
    return dict(I=I, shape=shape, p0=p0, p1=p1, node=node, knee=knee, foot=foot)


@lru_cache(maxsize=None)
def missing_arm():
    """Spec 2.2's second failure on a toy: a crossing (ODs add) whose lower
    arm is too faint to trace, so three traces leave its disc."""
    shape = (120, 220)
    c = (112.0, 58.0)
    top, bottom = (84.0, 0.0), (140.0, 119.0)
    od = N.vessel_od(shape, [((0.0, c[1]), (219.0, c[1]), 4.5, 0.16), (top, c, 3.0, 0.13), (c, bottom, 3.0, 0.03)],
                     blur=1.2, combine="sum")
    I = N.to_intensity(N.add_noise(od, 0.008, seed=4), 0.85)
    return dict(I=I, shape=shape, c=c, top=(top[0] + 1.5, 3.0), bottom=bottom, r=16.0)


class ColdOpen(BK.PacedCaptions, ChapterScene):
    NUMBER, TITLE, SPEC = None, "", "spec 1, 2.1, 2.2"
    MAP_KEYS = ()

    # ------------------------------------------------------------------ helpers
    def header(self, left, right=None):
        """A row above the still: left-aligned text, right-aligned extra."""
        half = IMG_H * 16 / 9 / 2
        top = IMG_C[1] + IMG_H / 2
        y = (top + 4.0) / 2
        g = VGroup(left.move_to([-half, y, 0], aligned_edge=LEFT))
        if right is not None:
            g.add(right.move_to([half, y, 0], aligned_edge=RIGHT))
        return g

    def legend(self, counts):
        items = VGroup()
        for g, name in (("fork", "3-way (fork)"), ("crossing", "crossing"), ("compound", "compound")):
            c = Circle(radius=0.09, stroke_color=BK.group_color(g), stroke_width=3)
            t = Tex(f"{name} {counts[g]}", color=GREY_A).scale(0.45)
            items.add(VGroup(c, t).arrange(RIGHT, buff=0.1))
        return items.arrange(RIGHT, buff=0.35)

    # ---------------------------------------------------------------- construct
    def construct(self):
        self.intro()
        S = BK.scene()
        pf = BK.scene_frame(IMG_H, IMG_C)

        # ======================================================= 1. the burst flickers
        frames = BK.burst()
        n = len(frames)
        rgba = [BK.frame_rgba(i) for i in range(n)]
        means = [BK.running_mean_rgba(k) for k in range(1, n + 1)]
        full_h = config.frame_height
        flick = BK.photo(rgba[0], full_h).move_to(ORIGIN)
        st = dict(t=0.0, mode="flicker", fade=0.0, k=1, inset=None)
        shrink = ValueTracker(0.0)
        zx, zy, zs = ZOOM
        z_frames = [a[zy:zy + zs, zx:zx + zs] for a in rgba]
        z_means = [a[zy:zy + zs, zx:zx + zs] for a in means]

        def drive(dt):
            st["t"] += dt
            if st["mode"] == "flicker":
                arr = rgba[int(st["t"] * FLICKER_FPS) % n]
            else:
                arr = means[min(st["k"], n) - 1]
            if st["fade"] < 1.0:
                arr = arr.copy()
                arr[..., 3] = (arr[..., 3] * st["fade"]).astype(np.uint8)
            flick.pixel_array = arr
            if st["inset"] is not None:
                st["inset"].pixel_array = (z_frames[int(st["t"] * FLICKER_FPS) % n] if st["mode"] == "flicker"
                                           else z_means[min(st["k"], n) - 1])
            s = shrink.get_value()
            flick.height = full_h + (IMG_H - full_h) * s
            flick.move_to(IMG_C * s)

        fade = ValueTracker(0.0)
        fade.add_updater(lambda m: st.__setitem__("fade", m.get_value()))
        self.add(flick, fade)
        self.add_updater(drive)
        self.play(fade.animate.set_value(1.0), run_time=0.8, rate_func=linear)
        fade.clear_updaters()
        st["fade"] = 1.0
        self.wait(1.3 if self.short else 2.6)

        tag = self.header(Tex(r"a burst of video: %d frames" % n, color=GREY_A).scale(0.5),
                          Tex(r"synthetic: rendered by vesselscene", color=GREY_B).scale(0.45))
        self.play(shrink.animate.set_value(1.0), FadeIn(tag, rate_func=lambda t: smooth(np.clip((t - 0.65) / 0.35, 0, 1))),
                  run_time=1.3)
        if self.short:
            self.say(r"A burst of video of the white of the eye: red cells flow, so every frame flickers.", wait="read")
        else:
            self.say(r"A burst of video of the white of the eye. Red cells flow through the vessels, so every frame flickers.",
                     wait=3.2)
        if self.at(3):
            inset = BK.photo(z_frames[0], 2.9).move_to(ZOOM_C)
            inset.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            src = Square(side_length=zs * pf.unit, stroke_color=YELLOW, stroke_width=2).move_to(
                pf.point(zx + zs / 2 - 0.5, zy + zs / 2 - 0.5))
            ibox = SurroundingRectangle(inset, buff=0, color=YELLOW, stroke_width=2)
            links = VGroup(Line(src.get_corner(UL), ibox.get_corner(DL)), Line(src.get_corner(UR), ibox.get_corner(DR)))
            links.set_stroke(YELLOW, 1.5, opacity=0.7)
            st["inset"] = inset
            self.play(Create(src), run_time=0.5)
            self.play(FadeIn(inset), Create(ibox), Create(links), run_time=0.8)
            self.say(r"Zoom in: clumps of red cells and plasma gaps travel along the capillaries, so their darkness changes frame to frame.",
                     wait=3.0)

        # ======================================================= 2. average the burst
        count = Integer(1).scale(0.5)
        count_l = VGroup(Tex(r"frames averaged:", color=GREY_A).scale(0.5), count).arrange(RIGHT, buff=0.12)
        count_l.move_to(tag[0], aligned_edge=LEFT)
        k_tr = ValueTracker(1.0)
        if self.short:
            self.say(r"Averaging the burst cancels the flicker and the noise.")
            count_l = tag[0]
        else:
            st["mode"], st["k"] = "mean", 1
            k_tr.add_updater(lambda m: st.__setitem__("k", int(round(m.get_value()))))
            self.add(k_tr)
            self.play(FadeOut(tag[0]), FadeIn(count_l), run_time=0.5)
            count.add_updater(lambda m: m.set_value(st["k"]))
            self.say(r"Averaging the burst cancels the flicker and the noise, and leaves one steady still.")
            self.play(k_tr.animate.set_value(n), run_time=2.0 if not self.at(3) else 2.8,
                      rate_func=rate_functions.ease_in_out_sine)
            k_tr.clear_updaters()
            count.clear_updaters()
        if st["inset"] is not None:
            self.wait(1.0)
            st["inset"] = None
            self.play(FadeOut(Group(inset, src, ibox, links)), run_time=0.6)
        avg = BK.photo(BK.average_rgba(), IMG_H).move_to(IMG_C)
        avg_tag = Tex(r"the stabilised average (vesselscene)", color=GREY_A).scale(0.5).move_to(tag[0], aligned_edge=LEFT)
        self.play(FadeIn(avg), FadeOut(count_l), FadeIn(avg_tag), run_time=1.2)
        self.remove_updater(drive)
        self.remove(flick, k_tr)
        self.wait(0.3 if self.short else 0.8)

        # ======================================================= 3. the annotation draws itself
        dim = Rectangle(width=pf.width, height=pf.height, stroke_width=0, fill_color=BLACK,
                        fill_opacity=0.5).move_to(IMG_C)
        runs = BK.truth_runs(pf, width=2.4)
        rings = BK.truth_rings(pf, width=2.6)
        counts = {g: len(rings[g]) for g in BK.GROUPS}
        legend = self.legend(counts)
        target = self.header(VGroup(Tex(r"the target: vesselscene's observable truth,", color=GREY_A),
                                    Tex(r"what a careful annotator would draw", color=GREY_B)
                                    ).scale(0.45).arrange(DOWN, aligned_edge=LEFT, buff=0.06), legend)
        self.play(FadeIn(dim), FadeOut(avg_tag), FadeOut(tag[1]), FadeIn(target[0]), run_time=0.8)
        if self.short:
            self.say(r"The goal: a graph. Every vessel with an identity, every junction with a type.")
            self.play(LaggedStart(*[Create(r) for r in runs], lag_ratio=0.012), run_time=2.3)
            self.play(LaggedStart(*[FadeIn(it) for it in legend], lag_ratio=0.3),
                      LaggedStart(*[GrowFromCenter(c) for g in BK.GROUPS for c in rings[g]], lag_ratio=0.006), run_time=1.1)
            self.wait(0.3)
        else:
            self.say(r"The goal: from a burst of video to a graph. Every vessel with an identity, its own colour here$\ldots$")
            self.play(LaggedStart(*[Create(r) for r in runs], lag_ratio=0.012), run_time=4.0)
            self.say(r"$\ldots$and every junction with a type: forks, crossings, and compounds where several meet.")
            for g, item in zip(BK.GROUPS, legend):
                self.play(FadeIn(item), LaggedStart(*[GrowFromCenter(c) for c in rings[g]], lag_ratio=0.02), run_time=0.9)
            self.wait(1.8)

        # ======================================================= 4. the problem: 500 junctions as dots
        towers, labels, dots_of = self.build_towers()
        head = VGroup(Tex(r"Today's pipeline on 500 junctions of six held-out images", color=GREY_A).scale(0.55),
                      Tex(r"measured, spec 2.1", color=GREY_B).scale(0.45)).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        head.move_to(UP * 3.45)
        if self.short:
            self.say(r"Score today's pipeline on 500 junctions of six held-out images.")
        else:
            self.say(r"Score today's pipeline on six held-out images like this one: 500 junctions, one dot each.")
        moves = []
        for g in BK.GROUPS:
            for c, d in zip(rings[g], dots_of[TOWER_OF[g]]):
                moves.append(ReplacementTransform(c, d))
        moved = {TOWER_OF[g]: len(rings[g]) for g in BK.GROUPS}
        self.play(FadeOut(avg), FadeOut(dim), FadeOut(runs), FadeOut(target[0]), FadeOut(legend),
                  LaggedStart(*moves, lag_ratio=0.004), run_time=1.5 if self.short else 1.8)
        rest = [d for name in SCORECARD if name != "all" for d in dots_of[name][moved[name]:]]
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in rest], lag_ratio=0.002), FadeIn(labels), FadeIn(head),
                  run_time=1.0 if self.short else 1.6)

        state_key = self.state_legend().next_to(head, DOWN, buff=0.28)
        pct_found, pct_typed = VGroup(), VGroup()
        for name, tw in towers.items():
            f = Integer(0, unit=r"\%", color=GREY_B).scale(0.55)
            f.next_to(tw, UP, buff=0.2).shift(LEFT * 0.45)
            t = Integer(0, unit=r"\%", color=self.tower_color(name)).scale(0.55)
            t.next_to(tw, UP, buff=0.2).shift(RIGHT * 0.45)
            pct_found.add(f)
            pct_typed.add(t)
        big = self.big_stats()
        if self.short:
            self.say(r"It finds 61\% of them, and gives 34\% the right type.")
        else:
            self.say(r"It finds 61\% of them$\ldots$")
        found_anims, typed_anims = [], []
        for name in towers:
            n_, fd, ty = SCORECARD[name][:3]
            k_f, k_t = round(n_ * fd / 100), round(n_ * ty / 100)
            found_anims += [d.animate.set_fill(GREY_B, 0.9).set_stroke(GREY_B, opacity=1.0) for d in dots_of[name][:k_f]]
            col = self.tower_color(name)
            typed_anims += [d.animate.set_fill(col, 1.0).set_stroke(col, opacity=1.0) for d in dots_of[name][:k_t]]
        self.play(FadeIn(state_key[:2]), FadeIn(pct_found), run_time=0.4)
        self.play(LaggedStart(*found_anims, lag_ratio=0.003),
                  *[ChangeDecimalToValue(p, SCORECARD[nm][1]) for p, nm in zip(pct_found, towers)],
                  FadeIn(big[0]), run_time=1.2 if self.short else 1.8)
        if not self.short:
            self.say(r"$\ldots$and gives 34\% the right type.")
        self.play(FadeIn(state_key[2]), FadeIn(pct_typed), run_time=0.4)
        self.play(LaggedStart(*typed_anims, lag_ratio=0.004),
                  *[ChangeDecimalToValue(p, SCORECARD[nm][2]) for p, nm in zip(pct_typed, towers)],
                  FadeIn(big[1]), run_time=1.2 if self.short else 1.8)
        self.wait(0.3 if self.short else 0.8)
        self.say(r"Worst are the 3-way junctions: only 38\% are found at all.")
        self.play(Circumscribe(VGroup(towers["3-way"], pct_found[0]), color=JUNC, buff=0.12), FadeIn(big[2]),
                  run_time=1.2 if self.short else 1.6)
        if not self.short:
            self.wait(1.8)
        waffle = VGroup(*towers.values(), labels, head, state_key, pct_found, pct_typed, big)

        # ======================================================= L3: the scorecard and why
        if self.at(3):
            self.scorecard(waffle)
            self.why_traces()
        else:
            self.play(FadeOut(waffle), run_time=0.8)

        # ======================================================= L2: what today's pipeline is
        if self.at(2):
            self.todays_pipeline()

        # ======================================================= 5. the question
        avg = BK.photo(BK.average_rgba(), IMG_H).move_to(IMG_C)
        dim = Rectangle(width=pf.width, height=pf.height, stroke_width=0, fill_color=BLACK,
                        fill_opacity=0.35).move_to(IMG_C)
        marks = VGroup()
        for jid, g in ((CROSSING_ID, "crossing"), (FORK_ID, "fork")):
            j = BK.junction(jid)
            marks.add(Circle(radius=(CROP - 4) * pf.unit, stroke_color=BK.group_color(g), stroke_width=4)
                      .move_to(pf.point(j["x"], j["y"])))
        self.play(FadeIn(avg), FadeIn(dim), run_time=1.0)
        if self.at(2):
            self.say(r"Junctions are where it fails. And in 2-D, a fork and a crossing look alike.")
            self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.4), run_time=1.4)
            self.zoom_pair(pf, avg, dim, marks)
        else:
            self.say(r"Junctions are where it fails. A fork and a crossing look alike: how would a visual system tell them apart?")
            self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.4), run_time=1.4)

        # ======================================================= 6. the title
        self.title_card(pf)
        self.outro()

    # ------------------------------------------------------------------ the towers
    def tower_color(self, name):
        return {"3-way": BK.group_color("fork"), "crossing": BK.group_color("crossing"),
                "compound": BK.group_color("compound")}[name]

    def build_towers(self, s=0.19, r=0.064, cols=10, bottom=-2.1, xs=(-5.05, -2.6, -0.15)):
        towers, dots_of = {}, {}
        labels = VGroup()
        for (name, x) in zip(("3-way", "crossing", "compound"), xs):
            n = SCORECARD[name][0]
            col = self.tower_color(name)
            dots = VGroup()
            for i in range(n):
                row, c = divmod(i, cols)
                p = np.array([x + (c - (cols - 1) / 2) * s, bottom + row * s, 0.0])
                dots.add(Circle(radius=r, stroke_color=col, stroke_width=1.6, stroke_opacity=0.55,
                                fill_color=col, fill_opacity=0.0).move_to(p))
            towers[name] = dots
            dots_of[name] = list(dots)
            lab = VGroup(Tex(name, color=col).scale(0.5), Tex(f"{n}", color=GREY_B).scale(0.45)).arrange(RIGHT, buff=0.15)
            lab.move_to([x, bottom - 0.45, 0])
            labels.add(lab)
        return towers, labels, dots_of

    def state_legend(self):
        def dot(fill, stroke, op):
            return Circle(radius=0.07, stroke_color=stroke, stroke_width=1.8, fill_color=fill, fill_opacity=op)
        items = [VGroup(dot(BLACK, GREY_B, 0.0), Tex(r"missed", color=GREY_B).scale(0.45)),
                 VGroup(dot(GREY_B, GREY_B, 0.9), Tex(r"found", color=GREY_B).scale(0.45)),
                 VGroup(VGroup(*[dot(self.tower_color(n), self.tower_color(n), 1.0) for n in ("3-way", "crossing", "compound")]
                               ).arrange(RIGHT, buff=0.04), Tex(r"found, right type", color=GREY_B).scale(0.45))]
        g = VGroup(*[VGroup(*it).arrange(RIGHT, buff=0.12) if not isinstance(it, VGroup) else it.arrange(RIGHT, buff=0.12)
                     for it in items])
        return g.arrange(RIGHT, buff=0.5)

    def big_stats(self):
        def stat(value, words, color):
            v = Tex(value, color=color).scale(1.45)
            w = Tex(words, color=GREY_A).scale(0.5)
            return VGroup(v, w).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
        a = stat(r"61\%", r"of all 500 found", GREY_B)
        b = stat(r"34\%", r"typed right", WHITE)
        c = stat(r"38\%", r"of 3-way found", JUNC)
        g = VGroup(a, b, c).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(RIGHT * 4.55 + DOWN * 0.3)
        return g

    # ------------------------------------------------------------- L3: scorecard
    def scorecard(self, waffle):
        cols = (r"found", r"right type", r"render $\geq 95\%$", r"all three")
        xs = (-2.05, 0.6, 3.25, 5.9)
        bw, bh = 2.2, 0.36
        rows = ("3-way", "crossing", "compound", "all")
        ys = (1.25, 0.25, -0.75, -1.95)
        head = VGroup(Tex(r"The per-junction scorecard of today's pipeline", color=GREY_A).scale(0.55),
                      Tex(r"measured, spec 2.1", color=GREY_B).scale(0.45)).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        head.move_to(UP * 3.3)
        note = Tex(r"found: within the strict radius of 23.8 px of a true junction", color=GREY_B).scale(0.42)
        note.next_to(head, DOWN, buff=0.12)
        head.add(note)
        col_t = VGroup(*[Tex(c, color=GREY_A).scale(0.5).move_to([x, 2.2, 0]) for c, x in zip(cols, xs)])
        row_t = VGroup()
        frames, bars, nums = VGroup(), [VGroup() for _ in cols], [VGroup() for _ in cols]
        for name, y in zip(rows, ys):
            col = WHITE if name == "all" else self.tower_color(name)
            n = SCORECARD[name][0]
            row_t.add(VGroup(Tex(name, color=col).scale(0.55), Tex(f"$n = {n}$", color=GREY_B).scale(0.45))
                      .arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to([-5.6, y, 0], aligned_edge=LEFT))
            for c, x in enumerate(xs):
                v = SCORECARD[name][1 + c]
                fr = Rectangle(width=bw, height=bh, stroke_color=MUTED, stroke_width=1, stroke_opacity=0.6)
                fr.move_to([x, y, 0])
                b = Rectangle(width=max(bw * v / 100, 1e-3), height=bh, stroke_width=0, fill_color=col,
                              fill_opacity=0.85 if c < 3 else 1.0)
                b.move_to(fr.get_left(), aligned_edge=LEFT)
                t = Tex(f"{v}\\%", color=WHITE).scale(0.45).next_to(b, RIGHT, buff=0.08)
                if t.get_right()[0] > fr.get_right()[0]:
                    t.next_to(b.get_right(), LEFT, buff=0.08).set_color(BLACK)
                frames.add(fr)
                bars[c].add(b)
                nums[c].add(t)
        sep = DashedLine([-5.7, -1.45, 0], [7.0, -1.45, 0], stroke_color=MUTED, stroke_width=1, dash_length=0.08)
        self.say(r"The full scorecard (spec 2.1): found, typed right, rendered to 95\%, and all three at once.")
        self.play(FadeOut(waffle, run_time=0.6), FadeIn(head), FadeIn(col_t), FadeIn(row_t), Create(frames), Create(sep),
                  run_time=1.4)
        for c in range(4):
            self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars[c]], lag_ratio=0.15),
                      LaggedStart(*[FadeIn(t) for t in nums[c]], lag_ratio=0.15), run_time=0.9)
        self.wait(0.8)
        hl = SurroundingRectangle(VGroup(frames[3], nums[3][0]), color=JUNC, buff=0.1, stroke_width=3)
        self.say(r"Only 21\% of all junctions pass all three tests. For 3-way junctions, 11\%.")
        self.play(Create(hl), run_time=0.8)
        self.wait(3.0)
        self.play(FadeOut(VGroup(head, col_t, row_t, frames, sep, hl, *bars, *nums)), run_time=0.8)

    # ---------------------------------------------------------------- L3: why
    def why_traces(self):
        T = stopped_branch()
        IH, IC = 4.0, np.array([-3.05, 0.45, 0.0])
        pf = PixelFrame(T["shape"], IH, IC)
        cam = image(T["I"], IH, "grey", vmin=0.68, vmax=0.88).move_to(pf.centre)
        border = SurroundingRectangle(cam, buff=0, color=MUTED, stroke_width=1)
        title = VGroup(Tex(r"Why: junctions are built from traces", color=GREY_A).scale(0.6),
                       Tex(r"spec 2.2, on toy vessels", color=GREY_B).scale(0.45)).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        title.move_to(UP * 3.3)
        self.play(FadeIn(cam), Create(border), FadeIn(title), run_time=1.0)
        self.say(r"Step 7 makes a junction only where a trace ends on another trace, and types it by counting traces.")

        def pt(p):
            return pf.point(*p)
        parent = Line(pt((4, T["p0"][1])), pt((215, T["p1"][1])), stroke_color=WHITE, stroke_width=4)
        stop = (T["knee"][0] + 1.0, T["knee"][1] + 2.0)
        branch = Line(pt((T["foot"][0] - 1.0, 116.0)), pt(stop), stroke_color=WHITE, stroke_width=4)
        end = Dot(pt(stop), radius=0.08, color=END)
        self.play(Create(parent), run_time=1.0)
        self.play(Create(branch), run_time=1.0)
        self.play(FadeIn(end, scale=2.0), run_time=0.4)
        end_l = Tex(r"the trace stops:\\the branch is faint here", color=END).scale(0.55)
        end_l.move_to([1.15, -0.05, 0], aligned_edge=LEFT)
        end_a = Arrow(end_l.get_left(), end.get_center(), buff=0.12, color=END, stroke_width=2.5, tip_length=0.14)
        self.play(FadeIn(end_l), GrowArrow(end_a), run_time=0.7)
        self.say(r"Here the branch fades just before its parent, so its trace stops short and never reaches it.", wait=1.2)
        ring = DashedVMobject(Circle(radius=14 * pf.unit, stroke_color=JUNC, stroke_width=3), num_dashes=22)
        ring.move_to(pt(T["node"]))
        cross = VGroup(Line(UL, DR), Line(UR, DL)).scale(0.12).set_stroke(JUNC, 4).next_to(ring, UP, buff=0.04)
        lost = Tex(r"no trace ends on a trace:\\the junction is lost", color=JUNC).scale(0.55)
        lost.move_to([1.15, 1.75, 0], aligned_edge=LEFT)
        lost_a = Arrow(lost.get_left(), ring.get_right(), buff=0.1, color=JUNC, stroke_width=2.5, tip_length=0.14)
        self.play(Create(ring), run_time=0.7)
        self.play(FadeIn(lost), GrowArrow(lost_a), Create(cross), run_time=0.9)
        self.wait(1.6)

        # ---- the second way: a crossing whose fourth arm is missing is typed 3-way
        X = missing_arm()
        pf2 = PixelFrame(X["shape"], IH, IC)
        cam2 = image(X["I"], IH, "grey", vmin=0.68, vmax=0.88).move_to(pf2.centre)
        self.play(FadeOut(VGroup(parent, branch, end, end_l, end_a, ring, cross, lost, lost_a)), run_time=0.6)
        self.play(FadeIn(cam2), run_time=0.8)
        self.remove(cam)
        self.say(r"Typing is limited by detection too. Here one arm of a crossing is too faint to trace$\ldots$")

        def pt2(p):
            return pf2.point(*p)
        h = Line(pt2((4, X["c"][1])), pt2((215, X["c"][1])), stroke_color=WHITE, stroke_width=4)
        up = Line(pt2(X["top"]), pt2(X["c"]), stroke_color=WHITE, stroke_width=4)
        self.play(Create(h), Create(up), run_time=1.2)
        disc = Circle(radius=X["r"] * pf2.unit, stroke_color=JUNC, stroke_width=3).move_to(pt2(X["c"]))
        arms = VGroup()
        c = np.asarray(pt2(X["c"]))
        for p in (pt2((X["c"][0] - X["r"] - 12, X["c"][1])), pt2((X["c"][0] + X["r"] + 12, X["c"][1])), pt2(X["top"])):
            d = np.asarray(p) - c
            d = d / np.linalg.norm(d)
            arms.add(Arrow(c + d * X["r"] * pf2.unit * 0.95, c + d * (X["r"] + 14) * pf2.unit, buff=0, color=JUNC,
                           stroke_width=3, tip_length=0.12, max_tip_length_to_length_ratio=0.5))
        typed = VGroup(Tex(r"3 traces leave the disc:", color=JUNC), Tex(r"typed 3-way", color=JUNC)).scale(0.55)
        typed.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([1.15, 1.75, 0], aligned_edge=LEFT)
        truth = VGroup(Tex(r"truth: a crossing, four arms", color=BK.group_color("crossing")).scale(0.55),
                       Tex(r"(its lower arm is faint)", color=GREY_B).scale(0.5)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        truth.move_to([1.15, 0.45, 0], aligned_edge=LEFT)
        faint = DashedLine(pt2(X["c"]), pt2(X["bottom"]), stroke_color=BK.group_color("crossing"), stroke_width=3,
                           dash_length=0.08)
        self.play(Create(disc), LaggedStart(*[GrowArrow(a) for a in arms], lag_ratio=0.2), run_time=1.0)
        self.play(FadeIn(typed), run_time=0.6)
        self.say(r"$\ldots$so three traces leave the disc and it is typed 3-way. Most crossings typed 3-way are missing their fourth arm.")
        self.play(Create(faint), FadeIn(truth), run_time=1.0)
        self.wait(1.2)
        more = VGroup(Tex(r"Compounds in the truth have 5--6 lines; tracing finds 3--4.", color=GREY_B),
                      Tex(r"Fork against crossing is never tested on the image itself.", color=GREY_B)).scale(0.47)
        more.arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([-6.45, -2.35, 0], aligned_edge=LEFT)
        self.say(r"Compounds lose arms the same way, and nothing checks the image at the centre: fork against crossing is never tested.")
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in more], lag_ratio=0.4), run_time=1.2)
        self.wait(2.4)
        self.play(FadeOut(Group(cam2, border, title, h, up, disc, arms, typed, truth, faint, more)), run_time=0.8)

    # --------------------------------------------------- L2: today's pipeline
    def todays_pipeline(self):
        steps = [(r"1--2", r"retina,\\ganglion", "nn"), (r"3", r"simple\\cells", "nn"), (r"4", r"surround", "nn"),
                 (r"5", r"3 fixed\\linking steps", "nn"), (r"6", r"tracing", "rule"),
                 (r"7", r"junctions\\from traces", "rule"), (r"9", r"same-trace\\rule", "rule"),
                 (r"10--12", r"spline\\fit", "fit")]
        boxes = VGroup()
        for num, label, kind in steps:
            col = FF if kind == "nn" else (GREY_B if kind == "rule" else WHITE)
            r = RoundedRectangle(width=1.56, height=1.35, corner_radius=0.1, stroke_color=col, stroke_width=1.8,
                                 fill_color=col, fill_opacity=0.08)
            if kind == "rule":
                r = DashedVMobject(r, num_dashes=36).set_stroke(col, 1.8)
            t = VGroup(Tex(num, color=col).scale(0.5), Tex(label, color=GREY_A).scale(0.5)).arrange(DOWN, buff=0.1)
            if t.width > 1.42:
                t.scale_to_fit_width(1.42)
            boxes.add(VGroup(r, t.move_to(r)))
        boxes.arrange(RIGHT, buff=0.17).move_to(UP * 0.55)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.02, stroke_width=2, tip_length=0.1,
                                max_tip_length_to_length_ratio=0.5, color=MUTED) for a, b in zip(boxes[:-1], boxes[1:])])
        b1 = Brace(boxes[:4], UP, buff=0.12, color=FF)
        l1 = Tex(r"neural layers: V1, first sweep only", color=FF).scale(0.6).next_to(b1, UP, buff=0.1)
        b2 = Brace(boxes[4:7], UP, buff=0.12, color=GREY_B)
        l2 = Tex(r"hand-written rules", color=GREY_B).scale(0.6).next_to(b2, UP, buff=0.1)
        head = Tex(r"The annotator today (spec 1)", color=GREY_A).scale(0.65).move_to(UP * 3.2)
        ca = CurvedArrow(boxes[7].get_bottom() + DOWN * 0.05, boxes[1].get_bottom() + DOWN * 0.05, angle=-0.7,
                         color=FB, stroke_width=3, tip_length=0.18)
        tip = ca.pop_tips()
        mid = ca.point_from_proportion(0.5)
        fb = VGroup(DashedVMobject(ca, num_dashes=40).set_stroke(FB, 3), tip)
        x_mark = VGroup(Line(UL, DR), Line(UR, DL)).scale(0.2).set_stroke(RED_C, 6).move_to(mid)
        nf = Tex(r"nothing feeds back", color=FB).scale(0.6).next_to(x_mark, DOWN, buff=0.15)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in boxes], lag_ratio=0.12),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), run_time=1.8)
        self.say(r"Today's neural layers stop at V1's first, feedforward sweep$\ldots$")
        self.play(GrowFromCenter(b1), FadeIn(l1), run_time=0.9)
        self.wait(0.6)
        self.say(r"$\ldots$everything above them is hand-written rules$\ldots$")
        self.play(GrowFromCenter(b2), FadeIn(l2), run_time=0.9)
        self.wait(0.6)
        self.say(r"$\ldots$and nothing feeds back: the fit never changes what the early layers see.")
        self.play(Create(fb), run_time=1.2)
        self.play(GrowFromCenter(x_mark), FadeIn(nf), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(VGroup(boxes, arrows, b1, l1, b2, l2, head, fb, x_mark, nf)), run_time=0.8)

    # ------------------------------------------------------- L2: the two crops
    def zoom_pair(self, pf, avg, dim, marks):
        size = 2 * CROP
        big_h = 4.3
        centres = (LEFT * 3.35 + UP * 0.45, RIGHT * 3.35 + UP * 0.45)
        crops, frames, overlays = Group(), VGroup(), VGroup()
        full = BK.average_rgba()
        for jid, c in zip((CROSSING_ID, FORK_ID), centres):
            x0, y0 = crop_window(jid)
            small = BK.photo(full[y0:y0 + size, x0:x0 + size], size * pf.unit)
            small.move_to(pf.point(x0 + size / 2 - 0.5, y0 + size / 2 - 0.5))
            crops.add(small)
            frames.add(Square(side_length=big_h, stroke_color=GREY_B, stroke_width=1.5).move_to(c))
            cpf = PixelFrame((size, size), big_h, c)
            overlays.add(clipped_runs(x0, y0, size, cpf, {l["vid"] for l in BK.junction(jid)["lines"]}))
        self.add(crops)
        self.play(FadeOut(avg), FadeOut(dim), FadeOut(marks),
                  *[cr.animate.scale_to_fit_height(big_h).move_to(c) for cr, c in zip(crops, centres)],
                  *[FadeIn(f) for f in frames], run_time=1.6)
        qs = VGroup(*[Tex(r"?", color=WHITE).scale(1.1).next_to(f, UP, buff=0.12) for f in frames])
        self.say(r"Zoom in on two of them: dark lines meeting at a point. Which is a crossing, and which a fork?")
        self.play(FadeIn(qs, shift=DOWN * 0.1), run_time=0.6)
        self.wait(1.6)
        names = VGroup(
            Tex(r"crossing: two vessels, one over the other", color=BK.group_color("crossing")).scale(0.5),
            Tex(r"fork: two vessels join into one", color=BK.group_color("fork")).scale(0.5))
        for nm, f in zip(names, frames):
            nm.next_to(f, UP, buff=0.15)
        self.say(r"The truth: on the left, two vessels pass at different depths; on the right, two vessels merge.")
        self.play(*[ReplacementTransform(q, nm) for q, nm in zip(qs, names)],
                  *[LaggedStart(*[Create(p) for p in ov], lag_ratio=0.2) for ov in overlays], run_time=1.8)
        self.wait(1.6)
        self.say(r"How would a visual system tell them apart? That is the question this whole spec answers.", wait="read")

    # ------------------------------------------------------------------- title
    def title_card(self, pf):
        self.unsay()
        others = list(self.mobjects)
        bg = BK.photo(BK.average_rgba(), IMG_H).move_to(IMG_C)
        runs = BK.truth_runs(pf, width=2.2)
        rings = BK.truth_rings(pf, width=2.2)
        veil = Rectangle(width=config.frame_width + 0.2, height=config.frame_height + 0.2, stroke_width=0,
                         fill_color=BLACK, fill_opacity=0.72)
        art = Group(bg, runs, *rings.values())
        title = Tex(r"Recurrent mid-level vision\\for the vessel annotator").scale(1.25)
        rule = Line(LEFT * 2.6, RIGHT * 2.6, stroke_color=FB, stroke_width=2.5)
        sub = Tex(r"how a brain-inspired pipeline reads a vessel video", color=GREY_A).scale(0.65)
        block = VGroup(title, rule, sub).arrange(DOWN, buff=0.32).move_to(UP * 0.15)
        self.play(*[FadeOut(m) for m in others], FadeIn(art), FadeIn(veil), run_time=1.0 if self.short else 1.2)
        if self.short:
            self.play(Write(title), GrowFromCenter(rule), FadeIn(sub, shift=UP * 0.1, rate_func=lambda t: smooth(np.clip(2 * t - 1, 0, 1))),
                      run_time=1.4)
            self.wait(1.8)
        else:
            self.play(Write(title), GrowFromCenter(rule), run_time=1.6)
            self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.8)
            self.wait(3.0)


variants(ColdOpen, module_globals=globals())
