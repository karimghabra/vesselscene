"""Chapter 14: attention traces one vessel at a time.

Incremental grouping with attention (spec 4.10) and linking of facing label
ends (4.10b).  Curve tracing is serial: an attentional label spreads along one
curve (Jolicoeur, Ullman and Mackay 1986; Roelfsema, Lamme and Spekreijse
1998).  On a computed toy, a label spreads from the strongest seed, goes
straight through a crossing, follows the parent at a fork, carries through a
weak stretch thanks to its gain, and is then inhibited from returning; the
next seed starts a new identity.

Render:  manim -ql scenes/s14_attention.py AttentiveGroupingShort
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from manim.constants import RESAMPLING_ALGORITHMS
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import attention as AT  # noqa: E402
from mica_anim import vsdata as VD  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import ATTN, CONT, INH, JUNC, MUTED, PixelFrame, image  # noqa: E402

IDS = (1, 4, 3)          # identity colours of the toy's three vessels (vsdata.identity_color)


def ident(i):
    return VD.identity_color(IDS[i % len(IDS)] if i < len(IDS) else i + 7)


def smooth_xy(xy, s=1.5):
    if len(xy) < 5:
        return xy
    return np.stack([gaussian_filter1d(xy[:, 0], s, mode="nearest"), gaussian_filter1d(xy[:, 1], s, mode="nearest")], 1)


def front_points(v, fid):
    """A front's points (smoothed, image px) and their arrival steps."""
    pts = v.paths[fid]
    xy = np.array([(p[0], p[1]) for p in pts], float)
    st = np.array([p[3] for p in pts], float)
    return smooth_xy(xy), st


def tube(points, width, color, opacity=0.9, glow=True):
    """A label or vessel drawn as a tube of the given scene width, with a soft glow."""
    if len(points) < 2:
        points = [points[0], points[0] + RIGHT * 1e-3]
    core = VMobject(stroke_color=color, stroke_width=100 * width, stroke_opacity=opacity, joint_type=LineJointType.ROUND)
    core.set_points_as_corners(points)
    if not glow:
        return core
    halo = VMobject(stroke_color=color, stroke_width=100 * width * 2.2, stroke_opacity=0.18, joint_type=LineJointType.ROUND)
    halo.set_points_as_corners(points)
    return VGroup(halo, core)


def rgba_image(mask, color, alpha, height):
    rgb = np.array(color.to_rgb())
    arr = np.zeros(mask.shape + (4,), np.uint8)
    arr[..., :3] = (rgb * 255).astype(np.uint8)
    arr[..., 3] = (mask * alpha * 255).astype(np.uint8)
    im = ImageMobject(arr)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def toy_view(od, peak=0.15):
    """A toy's OD as brightness (blood is bright, as in chapter 4), contrast-
    stretched (OD^0.45) so that a weak stretch stays visible."""
    return 0.1 + 0.62 * np.clip(np.maximum(od, 0) / peak, 0, 1) ** 0.45


# ---------------------------------------------------------------- the puzzle
def puzzle_curves():
    P = [(-5.6, -1.9), (-4.6, 0.6), (-3.2, 2.1), (-1.6, 1.3), (-1.9, -0.9), (-0.4, -2.0), (1.0, -0.6), (0.2, 1.4),
         (1.8, 2.3), (3.3, 1.0), (2.6, -1.2), (4.0, -2.1), (5.4, -0.8)]
    Q = [(-5.4, 1.6), (-3.6, 0.9), (-3.0, -1.6), (-1.0, -1.2), (-0.2, 0.6), (-1.0, 2.2), (0.9, 2.4), (2.1, 0.4),
         (1.7, -2.0), (3.4, -2.3), (4.4, 0.1), (3.6, 2.2), (5.5, 2.0)]
    return [np.array([x, y + 0.35, 0.0]) for x, y in P], [np.array([x, y + 0.35, 0.0]) for x, y in Q]


class AttentiveGrouping(ChapterScene):
    NUMBER, TITLE, SPEC = 14, "Attention traces one vessel at a time", "spec 4.10, 4.10b"
    MAP_KEYS = ("attend",)

    def construct(self):
        self.intro()
        self.puzzle()
        self.toy_scene()
        if self.at(2):
            self.gain()
            self.junction_rule()
            self.gap()
        self.real()
        if self.at(3):
            self.as_built()
            self.breaks()
            self.alternative()
        self.outro()

    # ------------------------------------------------------------- 1. puzzle
    def puzzle(self):
        P, Q = puzzle_curves()
        cp = VMobject(stroke_color=GREY_B, stroke_width=5).set_points_smoothly(P)
        cq = VMobject(stroke_color=GREY_B, stroke_width=5).set_points_smoothly(Q)
        a = Dot(P[0], radius=0.11, color=WHITE, z_index=4)
        b = Dot(P[-1], radius=0.11, color=WHITE, z_index=4)
        self.play(Create(cp), Create(cq), run_time=1.2 if self.short else 1.6)
        self.play(FadeIn(a, scale=2), FadeIn(b, scale=2), run_time=0.6)
        self.say(r"Are the two dots on the same curve? You cannot see it at a glance: you have to trace it.",
                 wait=1.0 if self.short else 2.0)

        # the label spreads along the curve at a constant speed
        lab = cp.copy().set_stroke(ATTN, width=9, opacity=0.95)
        glow = cp.copy().set_stroke(ATTN, width=26, opacity=0.18)
        bar_bg = Rectangle(width=2.6, height=0.14, stroke_color=MUTED, stroke_width=1).move_to([4.75, -2.7, 0])
        bar_t = Tex(r"time", color=GREY_B).scale(0.45).next_to(bar_bg, LEFT, buff=0.15)
        bar = Rectangle(width=2.6, height=0.14, stroke_width=0, fill_color=ATTN, fill_opacity=0.85)
        bar.move_to(bar_bg, aligned_edge=LEFT)
        bar.stretch_to_fit_width(0.001, about_edge=LEFT)
        self.add(bar_bg, bar_t)
        T = 3.4 if self.short else 4.2
        if self.short:
            self.say(r"In V1 a label spreads along the curve (Roelfsema et al.\ 1998). The longer the curve, the longer"
                     r" it takes (Jolicoeur et al.\ 1986).")
        else:
            self.say(r"In V1 a label spreads along the curve, enhancing the neurons on it and not those on the curves"
                     r" crossing it (Roelfsema et al.\ 1998).")
        self.play(Create(lab), Create(glow), bar.animate.stretch_to_fit_width(2.6, about_edge=LEFT), run_time=T,
                  rate_func=linear)
        self.play(Flash(b, color=ATTN, line_length=0.25, num_lines=12), b.animate.set_color(ATTN), run_time=0.7)
        if self.at(2):
            # a shorter stretch of the other curve takes less time
            c0, c1 = 0.42, 0.62
            seg = cq.copy().pointwise_become_partial(cq, c0, c1)
            lab2 = seg.copy().set_stroke(ATTN, width=9, opacity=0.95)
            d0 = Dot(cq.point_from_proportion(c0), radius=0.1, color=WHITE, z_index=4)
            d1 = Dot(cq.point_from_proportion(c1), radius=0.1, color=WHITE, z_index=4)
            bar2_bg = bar_bg.copy().shift(DOWN * 0.24)
            bar2 = bar.copy().set_fill(ATTN, 0.85).shift(DOWN * 0.24).stretch_to_fit_width(0.001, about_edge=LEFT)
            self.play(lab.animate.set_stroke(opacity=0.35), glow.animate.set_stroke(opacity=0.06), FadeIn(d0), FadeIn(d1),
                      FadeIn(bar2_bg))
            self.add(bar2)
            frac = (c1 - c0) / 1.0 * cq.get_arc_length() / cp.get_arc_length()
            self.play(Create(lab2), bar2.animate.stretch_to_fit_width(2.6 * frac, about_edge=LEFT),
                      run_time=T * frac, rate_func=linear)
            self.say(r"The longer the path along the curve, the longer it takes (Jolicoeur, Ullman and Mackay 1986):"
                     r" grouping a curve is serial.", wait=2.4)
        else:
            self.wait(1.6)
        self.clear()

    # ---------------------------------------------------------- 2. the toy
    def toy_scene(self):
        T = AT.toy()
        g = AT.toy_grouping()
        vs = g.vessels
        H = 5.0
        pf = PixelFrame(AT.TOY_SHAPE, H, np.array([0.0, 0.48, 0.0]))
        self.pf_toy = pf
        cam = image(toy_view(T["od"]), H, "grey", vmin=0.0, vmax=1.0).move_to(pf.centre)
        frame = SurroundingRectangle(cam, buff=0, color=MUTED, stroke_width=1)
        if self.short:
            self.say(r"On computed vessels: one at a time, a label (yellow) starts at the strongest point and spreads both ways.")
        else:
            self.say(r"Now vessels, computed: a crossing, a weak stretch and a fork. One vessel at a time,"
                     r" strongest seed first.")
        self.play(FadeIn(cam), Create(frame), run_time=1.0)
        cross, fork = T["junctions"]
        P = T["paths"]
        s0, s1 = T["weak_s"]
        sA = np.r_[0, np.cumsum(np.hypot(*np.diff(P["A"], axis=0).T))]
        wx = float(np.interp(0.5 * (s0 + s1), sA, P["A"][:, 0]))
        wy = float(np.interp(wx, P["A"][:, 0], P["A"][:, 1]))

        def tag(text, xy, d):
            t = Tex(text, color=GREY_A).scale(0.45)
            t.next_to(pf.point(*xy), d, buff=0.32)
            return VGroup(BackgroundRectangle(t, fill_opacity=0.6, buff=0.04), t)
        tags = VGroup(tag(r"crossing", (cross.x, cross.y), UR), tag(r"weak stretch", (wx, wy), UP),
                      tag(r"fork", (fork.x, fork.y), UP))
        if not self.short:
            self.play(LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.3), run_time=1.0)
            self.wait(2.0)
            self.play(FadeOut(tags), run_time=0.5)

        unit = pf.unit
        self.toy_mobs = Group(cam, frame)
        claim_overlays = []
        for i, v in enumerate(vs):
            col = ident(i)
            fr = [front_points(v, 0), front_points(v, 1)]
            nmax = max(fr[0][1].max(), fr[1][1].max())
            tt = ValueTracker(0.0)
            w = v.width * unit

            def label_mob(fr=fr, tt=tt, w=w):
                g = VGroup()
                for xy, st in fr:
                    n = int(np.searchsorted(st, tt.get_value(), side="right"))
                    if n < 2:
                        continue
                    pts = [pf.point(x, y) for x, y in xy[:n]]
                    g.add(tube(pts, w, ATTN, 0.85))
                    g.add(Dot(pts[-1], radius=w * 0.75, color=ATTN).set_opacity(0.95))
                return g
            lab = always_redraw(label_mob).set_z_index(4)
            k0, y0, x0 = v.seed
            seed = Dot(pf.point(x0, y0), radius=0.07, color=ATTN, z_index=5)
            if i == 0 and not self.short:
                self.say(r"A label (yellow) starts at the strongest point and spreads both ways along the vessel's"
                         r" own orientation sheet.")
                self.wait(1.6)
            elif i == 1:
                self.say(r"Attention moves on: the next strongest seed starts a new vessel, with a new identity."
                         if not self.short else r"Attention moves on: each next seed starts a new identity.")
            elif not self.short:
                self.say(r"Then the branch. Its label stops where it meets the parent, already grouped.")
            self.play(FadeIn(seed, scale=3), Flash(seed, color=ATTN, line_length=0.18), run_time=0.8)
            self.add(lab)
            if i == 0:
                # the label keeps moving while each event's caption is read
                ev = self._events(v, fr, cross, fork, (wx, wy))
                marks = [0.0] + [e[0] for e in ev] + [nmax]
                texts = [None] + [e[1] for e in ev]
                for j in range(len(marks) - 1):
                    if texts[j]:
                        self.say(texts[j])
                    seg = marks[j + 1] - marks[j]
                    need = 0.9 if texts[j] is None else (1.6 if self.short else 2.6)
                    self.play(tt.animate.set_value(marks[j + 1]), run_time=max(seg * 0.03, need), rate_func=linear)
            else:
                self.play(tt.animate.set_value(nmax), run_time=1.3 if self.short else 2.6, rate_func=linear)
            lab.clear_updaters()
            # the vessel = the labelled ridge: it takes its identity colour
            whole = np.concatenate([fr[0][0][::-1], fr[1][0][1:]]) if len(fr[1][0]) > 1 else fr[0][0]
            final = tube([pf.point(x, y) for x, y in whole], 0.55 * w, col, 0.95)
            # inhibition of return: its claim, as a red veil
            claim = self._claim_mask(g, v)
            veil = rgba_image(claim, INH, 0.45, pf.height).move_to(pf.centre).set_z_index(1)
            final.set_z_index(3)
            if i == 0:
                self.say(r"The labelled ridge is one vessel. Inhibition of return (red) keeps attention from coming back."
                         if not self.short else r"One vessel. Inhibition of return (red): attention does not come back.")
            self.play(FadeOut(lab), FadeIn(final), FadeIn(veil), FadeOut(seed), run_time=1.0)
            claim_overlays.append(veil)
            self.toy_mobs.add(final, veil)
            if i == 0:
                self.wait(0.8 if self.short else 1.8)
            if i == 0 and v.queued and not self.short:
                qx, qy = v.queued[0]
                ring = Circle(radius=0.14, stroke_color=ATTN, stroke_width=3).move_to(pf.point(qx, qy))
                qt = Tex(r"branch queued", color=ATTN).scale(0.42).next_to(ring, DOWN, buff=0.08)
                qt = VGroup(BackgroundRectangle(qt, fill_opacity=0.6, buff=0.03), qt)
                self.play(Create(ring), FadeIn(qt), run_time=0.7)
                self.wait(0.6 if self.short else 1.2)
                self.play(FadeOut(ring), FadeOut(qt), run_time=0.5)
            if i == 1 and self.at(3):
                self.say(r"It passes the first vessel's claim: inhibition of return holds only in that vessel's"
                         r" own orientation sheets.", wait="read")
        if self.at(3):
            self.say(r"Three vessels, three identities: through a crossing, past a weak stretch, and along the parent at a fork.",
                     wait="read")
        else:
            self.wait(0.5)
        self.play(*[FadeOut(v) for v in claim_overlays], run_time=0.6)

    def _events(self, v, fr, cross, fork, weak):
        """(step, caption) for the first vessel: crossing, weak stretch, fork, in the order they are reached."""
        out = []
        if self.short:
            texts = (r"Straight through the crossing.", r"Its gain carries it through the weak stretch.",
                     r"At the fork it keeps to the parent.")
        else:
            texts = (r"At the crossing it goes straight through: the other vessel lies in other orientation sheets.",
                     r"Attention is a gain on what the label reaches: the weak stretch passes.",
                     r"At the fork it keeps to the parent, the best continuation. The branch is queued.")
        for (cx, cy), text in zip(((cross.x, cross.y), weak, (fork.x, fork.y)), texts):
            best = None
            for xy, st in fr:
                d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
                j = int(np.argmin(d))
                if d[j] < 8 and (best is None or st[j] < best[0]):
                    best = (float(st[j]), text)
            if best:
                out.append(best)
        return sorted(out)

    def _claim_mask(self, g, v):
        """The pixels claimed by vessel v (any sheet), recomputed from its points."""
        h, w = g.od.shape
        m = np.zeros((h, w), bool)
        half = 0.625 * v.width
        for path in v.paths:
            for (x, y, k, _) in path:
                m[max(0, int(np.floor(y - half))):int(np.ceil(y + half)) + 1,
                  max(0, int(np.floor(x - half))):int(np.ceil(x + half)) + 1] = True
        return m

    # ------------------------------------------------------- 3. the gain
    def gain(self):
        self.play(FadeOut(self.toy_mobs), run_time=0.8)
        T = AT.toy()
        g = AT.toy_grouping()
        A = g.vessels[0]
        fid = int(np.argmax([A.paths[f][-1][0] for f in (0, 1)]))          # the front that runs right, through the dip
        raw = A.paths[fid]
        xy = smooth_xy(np.array([(p[0], p[1]) for p in raw], float))
        ks = [p[2] for p in raw]
        C = np.array([max(AT._bil(g.C[(k + d) % AT.K], x, y) for d in (-1, 0, 1)) for (x, y), k in zip(xy, ks)])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        g0 = AT.Grouper(T["od"], T["junctions"], gaps=False)
        v0 = g0.spread(g0.next_seed(), 1, gain=False)
        stop_x = max(v0.paths[f][-1][0] for f in (0, 1) if v0.stops[f] == "weak")    # where it stops without the gain
        i_stop = int(np.argmin(np.abs(xy[:, 0] - stop_x)))
        i_seed = 0

        eq1 = MathTex(r"C^a_k", r"=", r"C_k", r"\,\big(1 + ", r"\gamma", r"\,", r"\mathrm{spread}_k(a)", r"\big)")
        eq2 = MathTex(r"a_k(p)", r"\leftarrow", r"\max\!\big(a_k(p),\ ", r"[\,C^a_k(p) > t_\text{low}\,]",
                      r"\cdot", r"\mathrm{spread}_k(a)(p)", r"\big)")
        for e in (eq1, eq2):
            e.scale(0.72)
        eq1[0].set_color(ATTN); eq1[4].set_color(ATTN); eq1[6].set_color(ATTN)
        eq2[0].set_color(ATTN); eq2[5].set_color(ATTN)
        eqs = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(UP, buff=0.35)
        self.say(r"The label $a$ lives in the orientation sheets. Attention is a gain on the evidence $C$ where the label is"
                 r" about to spread.")
        self.play(Write(eq1[:3]), run_time=0.9)
        self.play(Write(eq1[3:]), run_time=1.2)
        self.wait(0.6 if not self.at(3) else 1.2)
        self.say(r"The label grows wherever the attended evidence clears the continuation threshold $t_\text{low}$.")
        self.play(Write(eq2), run_time=1.8)
        self.wait(0.6 if not self.at(3) else 1.2)

        ymax = 12.0
        ax = Axes(x_range=[float(s[i_seed]) - 5, float(s[-1]) + 2, 50], y_range=[0, ymax, 2], x_length=11.0, y_length=3.0,
                  tips=False, axis_config=dict(color=MUTED, stroke_width=1.5, font_size=22),
                  y_axis_config=dict(numbers_to_include=[2, 4, 6, 8, 10])).move_to([0.2, -1.05, 0])
        xl = Tex(r"along the vessel (px)", color=GREY_B).scale(0.45).next_to(ax.x_axis, DOWN, buff=0.12).align_to(ax, RIGHT)
        yl = Tex(r"$C$ on the ridge (noise units)", color=GREY_B).scale(0.45).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
        sl = s[i_seed:]
        cl = np.minimum(C[i_seed:], ymax)
        curve = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners([ax.c2p(a, b) for a, b in zip(sl, cl)])
        tl = DashedLine(ax.c2p(sl[0], AT.T_LOW), ax.c2p(sl[-1], AT.T_LOW), stroke_color=GREY_A, stroke_width=2)
        tl_l = MathTex(r"t_\text{low} = 2", color=GREY_A).scale(0.55).next_to(tl, RIGHT, buff=0.1).shift(UP * 0.12)
        off = Tex(r"(off the scale: $C \approx 60$)", color=GREY_B).scale(0.4).next_to(ax.c2p(sl[5], ymax), DOWN, buff=0.1).align_to(ax.c2p(sl[2], 0), LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.play(Create(curve), FadeIn(off), run_time=1.5)
        self.play(Create(tl), FadeIn(tl_l))
        self.say(r"Along the first vessel, $C$ dips below $t_\text{low}$ in the weak stretch. Without the gain the label"
                 r" stops there.")
        stop = Dot(ax.c2p(s[i_stop], min(C[i_stop], ymax)), radius=0.07, color=INH)
        stop_l = Tex(r"without the gain: stops", color=INH).scale(0.45).next_to(stop, LEFT, buff=0.15).shift(UP * 0.15)
        self.play(FadeIn(stop, scale=2), FadeIn(stop_l))
        self.wait(1.2)
        eq3 = MathTex(r"C_k\,(1+\gamma) > t_\text{low}", r"\iff", r"C_k > \frac{t_\text{low}}{1+\gamma}", r"= 1.54")
        eq3.scale(0.62)
        eq3[0][3:6].set_color(ATTN)
        eq3.next_to(eqs, DOWN, buff=0.18).align_to(eqs, LEFT)
        lbl3 = Tex(r"just ahead of the label, $\gamma = 0.3$ (as built, spec 4.10)", color=GREY_B).scale(0.45)
        lbl3.next_to(eq3, RIGHT, buff=0.3)
        self.say(r"Just ahead of the label, $\mathrm{spread} = 1$: the threshold there falls to $t_\text{low}/(1+\gamma)$.")
        self.play(Write(eq3), FadeIn(lbl3), run_time=1.6)

        # the label runs along the profile with its lowered threshold just ahead
        fr = ValueTracker(float(sl[0]))
        low = AT.T_LOW / (1 + AT.GAMMA)

        def filled():
            sv = fr.get_value()
            m = sl <= sv
            if m.sum() < 2:
                return VGroup()
            pts = [ax.c2p(a, b) for a, b in zip(sl[m], cl[m])]
            poly = VMobject(stroke_width=0, fill_color=ATTN, fill_opacity=0.3)
            poly.set_points_as_corners(pts + [ax.c2p(sl[m][-1], 0), ax.c2p(sl[m][0], 0), pts[0]])
            line = VMobject(stroke_color=ATTN, stroke_width=5).set_points_as_corners(pts)
            return VGroup(poly, line)

        def ahead():
            sv = fr.get_value()
            return VGroup(Line(ax.c2p(sv, low), ax.c2p(min(sv + 14, float(sl[-1])), low), stroke_color=ATTN, stroke_width=5),
                          Dot(ax.c2p(sv, float(np.interp(sv, sl, cl))), radius=0.07, color=ATTN))
        fl, ah = always_redraw(filled), always_redraw(ahead)
        self.add(fl, ah)
        self.say(r"With the gain, the stretch passes: its evidence is above $1.54$ though below $2$.")
        self.play(fr.animate.set_value(float(sl[-1]) - 1.0), run_time=5.0, rate_func=linear)
        fl.clear_updaters(); ah.clear_updaters()
        self.play(FadeOut(ah), run_time=0.3)
        if self.at(3):
            self.say(r"A vessel continues through stretches below the tracing threshold when both sides belong to it.",
                     wait="read")
        else:
            self.wait(1.0)
        self.clear()

    # ---------------------------------------------------- 4. junction rule
    def junction_rule(self):
        T = AT.toy()
        g = AT.toy_grouping()
        I = toy_view(T["od"])
        cross, fork = T["junctions"]
        panels = Group()
        half = 26
        for j, x0, title in ((cross, -3.5, r"crossing"), (fork, 3.5, r"fork")):
            cx, cy = int(round(j.x)), int(round(j.y))
            crop = I[cy - half:cy + half, cx - half:cx + half]
            H = 4.2
            pfz = PixelFrame(crop.shape, H, np.array([x0, 0.3, 0]))
            im = image(crop, H, "grey", vmin=0.0, vmax=1.0).move_to(pfz.centre)
            fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
            ttl = Tex(title).scale(0.6).next_to(im, UP, buff=0.12)
            ctr = pfz.point(j.x - (cx - half), j.y - (cy - half))
            disc = Circle(radius=j.radius * pfz.unit, stroke_color=JUNC, stroke_width=3).move_to(ctr)
            arms = VGroup()
            for ang, wd in j.arms:
                u = np.array([np.cos(np.radians(ang)), -np.sin(np.radians(ang)), 0])      # image angle -> scene
                arms.add(Line(ctr + u * j.radius * pfz.unit, ctr + u * (half - 3) * pfz.unit, stroke_color=WHITE,
                              stroke_width=2, stroke_opacity=0.6))
            panels.add(Group(im, fr, ttl, disc, arms))
            j._scene = (pfz, ctr)
        self.say(r"At a junction cell, the label enters only the arm paired with the one it arrived on (good continuation,"
                 r" spec 4.4).")
        self.play(*[FadeIn(p[0]) for p in panels], *[Create(p[1]) for p in panels], *[FadeIn(p[2]) for p in panels])
        self.play(*[Create(p[3]) for p in panels], *[Create(p[4]) for p in panels], run_time=1.0)

        def arm_vec(j, i):
            ang = j.arms[i][0]
            return np.array([np.cos(np.radians(ang)), -np.sin(np.radians(ang)), 0])
        anims, extra = [], VGroup()
        for j in (cross, fork):
            pfz, ctr = j._scene
            a = 1 if j.kind == "crossing" else 0          # arriving from the left
            b = j.pairs[a]
            r0, r1 = j.radius * pfz.unit, 22 * pfz.unit
            arrive = Arrow(ctr + arm_vec(j, a) * r1, ctr + arm_vec(j, a) * r0 * 0.6, buff=0, color=ATTN, stroke_width=6,
                           max_tip_length_to_length_ratio=0.18)
            leave = Arrow(ctr - arm_vec(j, a) * r0 * 0.2, ctr + arm_vec(j, b) * r1, buff=0, color=ATTN, stroke_width=6,
                          max_tip_length_to_length_ratio=0.18)
            pair = ArcBetweenPoints(ctr + arm_vec(j, a) * r0 * 1.6, ctr + arm_vec(j, b) * r0 * 1.6, angle=-0.6,
                                    stroke_color=CONT, stroke_width=3)
            blocks = VGroup()
            for i in range(len(j.arms)):
                if i in (a, b):
                    continue
                p = ctr + arm_vec(j, i) * (r0 + 0.35)
                blocks.add(VGroup(Line(p + UL * 0.09, p + DR * 0.09), Line(p + UR * 0.09, p + DL * 0.09)).set_stroke(INH, 4))
            extra.add(arrive, leave, pair, blocks)
            anims.append((arrive, leave, pair, blocks))
        for arrive, leave, pair, blocks in anims:
            self.play(GrowArrow(arrive), run_time=0.5)
            self.play(Create(pair), GrowArrow(leave), FadeIn(blocks), run_time=0.8)
        self.say(r"Straight through a crossing; along the parent at a fork, the pair that continues best."
                 r" A label that arrives on the branch stops there.", wait="read")
        self.play(FadeOut(panels), FadeOut(extra), run_time=0.8)

    # ---------------------------------------------------------- 5. gaps
    def gap(self):
        G = AT.gap_toy()
        g = G["g"]
        I = toy_view(G["od"], 0.06)
        H = 3.5
        pf = PixelFrame(AT.GAP_SHAPE, H, np.array([0.0, 0.55, 0]))
        im = image(I, H, "grey", vmin=0.0, vmax=1.0).move_to(pf.centre)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        self.say(r"Where the line breaks, the label looks further along its heading, and crosses if the vessel resumes.")
        self.play(FadeIn(im), Create(fr))
        v = max(g.vessels, key=lambda v: v.length)
        # the front that crosses the gap
        fid, idx = v.gaps[0]
        raw = np.array([(p[0], p[1]) for p in v.paths[fid]], float)
        w = v.width * pf.unit
        before = [pf.point(x, y) for x, y in smooth_xy(raw[:idx])]
        after = [pf.point(x, y) for x, y in smooth_xy(raw[idx:])]
        xy = raw
        other = front_points(v, 1 - fid)[0]
        oth = [pf.point(x, y) for x, y in other]
        k0, y0, x0 = v.seed
        seed = Dot(pf.point(x0, y0), radius=0.07, color=ATTN)
        self.play(FadeIn(seed, scale=3))
        lab1, lab0 = tube(before, w, ATTN, 0.85), tube(oth, w, ATTN, 0.85)
        self.play(Create(lab1), Create(lab0), run_time=1.6, rate_func=linear)
        # the cone: up to 3 widths, 20 degrees either side of the heading
        p_end = np.array(xy[idx - 1])
        head = np.array(xy[idx - 1]) - np.array(xy[max(idx - 6, 0)])
        head /= np.linalg.norm(head)
        reach = min(3 * v.width, 30.0)
        ang = np.arctan2(-head[1], head[0])
        c = pf.point(*p_end)
        cone = AnnularSector(inner_radius=0, outer_radius=reach * pf.unit, angle=np.radians(40), start_angle=ang - np.radians(20),
                             fill_color=ATTN, fill_opacity=0.3, stroke_color=ATTN, stroke_width=2.5).move_arc_center_to(c)
        cl = Tex(r"3 widths, $\pm 20^\circ$", color=ATTN).scale(0.55).next_to(cone, UP, buff=0.35)
        self.play(FadeIn(cone), FadeIn(cl), run_time=1.0)
        dx, dy = G["distractor"]
        dist = Circle(radius=0.12, stroke_color=INH, stroke_width=3).move_to(pf.point(dx, dy))
        dist_l = Tex(r"outside the cone: ignored", color=INH).scale(0.5).next_to(dist, LEFT, buff=0.15)
        self.play(Create(dist), FadeIn(dist_l))
        jump = DashedLine(before[-1], after[0], stroke_color=ATTN, stroke_width=4, dash_length=0.08)
        lab2 = tube(after, w, ATTN, 0.85)
        self.play(Create(jump), run_time=0.6)
        self.play(Create(lab2), run_time=1.4, rate_func=linear)
        self.say(r"As built (spec 4.10): up to 3 vessel widths ahead, in a $20^\circ$ cone. One vessel, one identity, across the gap.",
                 wait="read")
        self.clear()

    # ----------------------------------------------- 6. the real image (target)
    def real(self):
        self.clear()
        if not VD.available():
            return
        S = VD.load()
        view = S.view(S.average)
        H = 4.45
        pf = PixelFrame(view.shape, H, np.array([0.0, 0.45, 0.0]))
        im = image(view, H, "grey", vmin=0.0, vmax=1.0).move_to(pf.centre)
        im.set_opacity(0.85)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        runs = [r for r in S.truth["runs"] if len(r["xy"]) >= 25]
        by_vid = {}
        for r in runs:
            by_vid.setdefault(r["vid"], []).append(r)
        score = {vid: sum(len(r["xy"]) * float(np.nan_to_num(r["contrast"] or 0.0)) for r in rs) for vid, rs in by_vid.items()}
        order = sorted(by_vid, key=lambda vid: -score[vid])
        groups = []
        for vid in order:
            gp = VGroup(*[VD.run_path(r, pf).set_stroke(VD.identity_color(vid), width=2.6, opacity=0.95) for r in by_vid[vid]])
            groups.append(gp)
        ttl = Tex(r"the target on a real image: vesselscene's truth, one colour per vessel", color=GREY_A).scale(0.5)
        ttl.next_to(im, UP, buff=0.12)
        self.play(FadeIn(im), Create(fr), FadeIn(ttl), run_time=1.0)
        self.say(r"On a real image the goal is one identity per vessel, through every crossing and fork, found one vessel at a time.")
        n = len(groups)
        self.play(LaggedStart(*[Create(gp) for gp in groups], lag_ratio=0.06), run_time=3.6 if self.short else 5.0)
        self.wait(1.3 if self.short else 2.0)
        self.real_mobs = Group(im, fr, ttl, *groups)

    # ------------------------------------------------ 7. level 3: as built
    def as_built(self):
        self.play(FadeOut(self.real_mobs), run_time=0.8)
        head = Tex(r"measured on the real test image, fused evidence (spec 4.10)", color=GREY_B).scale(0.52)
        cells = [Tex(""), Tex(r"parallel readout"), Tex(r"attentive grouping", color=ATTN),
                 Tex(r"centreline pieces"), Tex(r"755"), Tex(r"112", color=ATTN),
                 Tex(r"free ends"), Tex(r"2,237"), Tex(r"811", color=ATTN),
                 Tex(r"vessels with an identity"), Tex(r"--"), Tex(r"557", color=ATTN)]
        tbl = VGroup(*[c.scale(0.58) for c in cells]).arrange_in_grid(rows=4, cols=3, buff=(0.55, 0.16), col_alignments="lrr")
        sub = VGroup(Tex(r"557 vessels: median 86 px, longest 1,450 px", color=ATTN),
                     Tex(r"82\% of the grouped length is in vessels over 100 px", color=ATTN)).scale(0.46).arrange(DOWN, buff=0.08)
        blk = VGroup(head, tbl, sub).arrange(DOWN, buff=0.2).move_to([0, 2.05, 0])
        self.say(r"As built (layer 14), on the real test image: 755 centreline pieces become 112, and 557 vessels get an identity.")
        self.play(FadeIn(head), FadeIn(tbl), run_time=1.0)
        self.play(FadeIn(sub))
        self.wait(2.0)
        # where the labels end
        ends = [("on a grouped vessel", 606), ("free", 354), ("at a fork", 88), ("at the border", 66)]
        bars = VGroup()
        mx = 606
        for i, (name, n) in enumerate(ends):
            y = -0.75 - i * 0.42
            lab = Tex(name, color=GREY_A).scale(0.48).move_to([-3.4, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=3.4 * n / mx, height=0.26, stroke_width=0, fill_color=ATTN if i == 0 else GREY_B,
                            fill_opacity=0.8).move_to([-3.2, y, 0], aligned_edge=LEFT)
            num = Tex(f"{n}", color=GREY_A).scale(0.48).next_to(bar, RIGHT, buff=0.1)
            bars.add(VGroup(lab, bar, num))
        bh = Tex(r"where the labels end (spec 4.10)", color=GREY_B).scale(0.48).next_to(bars, UP, buff=0.18)
        ctrl = VGroup(Tex(r"controls:", color=GREY_B), Tex(r"without junction cells: 455 vessels, median 107 px"),
                      Tex(r"without gain and gap crossing: 176 pieces,"), Tex(r"about 8,000 px less centreline")
                      ).scale(0.48).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([3.6, -1.35, 0])
        self.say(r"Most labels end on a vessel grouped earlier. Without the gain and the gap crossing, the vessels fragment again.")
        self.play(FadeIn(bh), LaggedStart(*[AnimationGroup(FadeIn(b[0]), GrowFromEdge(b[1], LEFT), FadeIn(b[2]))
                                            for b in bars], lag_ratio=0.15), run_time=1.5)
        self.play(FadeIn(ctrl))
        self.wait(2.5)
        self.clear()

    # -------------------------------------------- 8. level 3: identity breaks
    def breaks(self):
        if VD.available():
            self._breaks_crop()
        self._breaks_cause()

    def _breaks_crop(self):
        x0, y0, w, h = 0, 60, 300, 180
        g = AT.real_crop(x0, y0, w, h)
        S = VD.load()
        view = S.view(S.average)[y0:y0 + h, x0:x0 + w]
        H = 3.35
        pfs = []
        mobs = Group()
        for cx, title in ((-3.55, r"this toy's grouping, one pass (computed)"), (3.55, r"target: vesselscene's truth")):
            pf = PixelFrame(view.shape, H, np.array([cx, 0.75, 0]))
            im = image(view, H, "grey", vmin=0, vmax=1).move_to(pf.centre).set_opacity(0.85)
            fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
            t = Tex(title, color=GREY_A).scale(0.48).next_to(im, UP, buff=0.12)
            pfs.append(pf)
            mobs.add(im, fr, t)
        comp = VGroup()
        for i, v in enumerate(g.vessels):
            xy, _ = v.polyline()
            xy = smooth_xy(xy)
            comp.add(VMobject(stroke_color=VD.identity_color(i + 1), stroke_width=2.6).set_points_as_corners(
                [pfs[0].point(x, y) for x, y in xy]))
        tru = VGroup()
        for r in S.truth["runs"]:
            xy = np.asarray(r["xy"], float) - [x0, y0]
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            if inside.sum() < 20:
                continue
            xy = xy[inside][::3]
            m = VMobject(stroke_color=VD.identity_color(r["vid"]), stroke_width=2.6)
            m.set_points_smoothly([pfs[1].point(x, y) for x, y in xy]) if len(xy) > 2 else m.set_points_as_corners([pfs[1].point(x, y) for x, y in xy])
            tru.add(m)
        self.say(r"Identity breaks. By pixels the grouping is far less fragmented, but one real vessel is often split"
                 r" into several identities.")
        self.play(FadeIn(mobs), run_time=1.0)
        self.play(LaggedStart(*[Create(c) for c in comp], lag_ratio=0.08), LaggedStart(*[Create(c) for c in tru], lag_ratio=0.08),
                  run_time=3.0)
        n_t = len({r["vid"] for r in S.truth["runs"]
                   if ((np.asarray(r["xy"])[:, 0] >= x0) & (np.asarray(r["xy"])[:, 0] < x0 + w)
                       & (np.asarray(r["xy"])[:, 1] >= y0) & (np.asarray(r["xy"])[:, 1] < y0 + h)).sum() >= 20})
        cnt = Tex(rf"toy, this crop: {len(g.vessels)} identities, against {n_t} true vessels", color=GREY_B).scale(0.45)
        cnt.next_to(mobs[0], DOWN, buff=0.15)
        self.play(FadeIn(cnt))
        self.wait(1.5)

        self.play(FadeOut(mobs), FadeOut(comp), FadeOut(tru), FadeOut(cnt), run_time=0.8)

    def _breaks_cause(self):
        # the cause: the claim sticks out past a vessel's end
        self.say(r"The cause (spec 4.10): the claimed square sticks out past a vessel's end, so a later label coming toward"
                 r" that end stops just short of it.")
        cause = self._cause_diagram()
        self.play(FadeIn(cause[0]), run_time=0.8)
        self.play(Create(cause[1]), run_time=1.5, rate_func=linear)
        self.play(FadeIn(cause[2][0]), LaggedStart(*[Create(sq) for sq in cause[2][1]], lag_ratio=0.2), run_time=1.2)
        self.play(Create(cause[3]), run_time=1.5, rate_func=linear)
        self.play(FadeIn(cause[4]))
        self.wait(2.0)
        # linking facing ends (4.10b)
        self.say(r"Remedies: claim only beside the path, merge when a label meets an earlier end in line, or link facing ends afterwards (4.10b).")
        self.play(Create(cause[5]), run_time=1.0)
        self.play(cause[3].animate.set_stroke(color=ident(0)), FadeOut(cause[4]), run_time=1.0)
        self.wait(1.5)
        link = VGroup(Tex(r"linking facing ends, measured (spec 4.10b):", color=GREY_B),
                      Tex(r"about 165 breaks lie inside one vessel (an upper bound)"),
                      Tex(r"50--54 joined: 494--490 vessels after joining", color=CONT),
                      Tex(r"the limit is the straight turn term: facing ends lie $23^\circ$ apart,"),
                      Tex(r"which co-circularity (chapter 10) accepts, unlike a looser turn")
                      ).scale(0.47).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0, -1.85, 0])
        self.say(r"Each end gets at most one partner, by good continuation. Joined so far: 50 to 54 of about 165 breaks.")
        self.play(FadeIn(link), run_time=1.0)
        self.wait(3.0)
        self.clear()

    def _cause_diagram(self):
        """A straight vessel with an earlier label ending at its middle, that
        label's claimed squares, and a later label stopping short of them.
        The squares and the stop are computed with the grouper's own claim."""
        shape = (40, 170)
        from mica_anim import neuro as N
        x, y = N.grid(shape)
        d, _ = N.dist_to_segment(x, y, (-10.0, 20.0), (180.0, 20.0))
        od = N.chord(d, 3.5, 0.12)
        g = AT.Grouper(od, (), gaps=False)
        k0 = int(np.argmin(np.abs(AT.THETAS - 0.0)))
        # the earlier vessel: its label ended at x = 80 (stopped for some other reason, e.g. at a junction)
        pts = [(float(xx), 20.0, k0, i) for i, xx in enumerate(range(6, 81))]
        v1 = AT.Vessel(1, (k0, 20, 6), 6.0, [pts[:1], pts], [], [], ["end", "stopped"])
        g.claim(v1)
        # the later label, from the right toward that end, computed by the same tracer
        fpts, why = g._front(160.0, 20.0, k0, np.array([-1.0, 0.0]), 6.0, 0, True, [], [], 0)
        H = 2.3
        pf = PixelFrame(shape, H, np.array([0.0, 0.85, 0]))
        im = image(toy_view(od, 0.12), H, "grey", vmin=0.0, vmax=1.0).move_to(pf.centre)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        w = 6.0 * pf.unit
        l1 = tube([pf.point(p[0], p[1]) for p in pts], 0.55 * w, ident(0), 0.95)
        half = 0.625 * 6.0
        claim = np.zeros(shape, bool)
        for p in pts:
            claim[int(np.floor(p[1] - half)):int(np.ceil(p[1] + half)) + 1, int(np.floor(p[0] - half)):int(np.ceil(p[0] + half)) + 1] = True
        veil = rgba_image(claim, INH, 0.35, H).move_to(pf.centre)
        squares = VGroup(*[Square(side_length=2 * half * pf.unit, stroke_color=INH, stroke_width=2, fill_opacity=0)
                           .move_to(pf.point(p[0], p[1])) for p in pts[-13::4]])
        l2 = tube([pf.point(p[0], p[1]) for p in fpts], 0.55 * w, ident(1), 0.95)
        gap = fpts[-1][0] - pts[-1][0]
        note = Tex(rf"the later label stops {gap:.0f} px short: two identities, one vessel (computed)", color=INH).scale(0.5)
        note.next_to(fr, DOWN, buff=0.18)
        e0, e1 = pf.point(pts[-1][0], 20), pf.point(fpts[-1][0], 20)
        arc = ArcBetweenPoints(e0 + UP * 0.1, e1 + UP * 0.1, angle=-2.6, stroke_color=CONT, stroke_width=5)
        ends = VGroup(Dot(e0, radius=0.06, color=CONT), Dot(e1, radius=0.06, color=CONT))
        ll = Tex(r"continuation cell between facing ends (4.10b)", color=CONT).scale(0.45).next_to(arc, UP, buff=0.12)
        link = VGroup(arc, ends, ll)
        return [Group(im, fr), l1, Group(veil, squares), l2, note, link]

    # ------------------------------------------- 9. level 3: the alternative
    def alternative(self):
        txt = VGroup(Tex(r"Open question (spec 10): parallel readout or serial grouping?"),
                     Tex(r"The parallel readout keeps continuations better by eye;", color=GREY_A),
                     Tex(r"serial grouping gives identities with fewer pieces, but splits vessels.", color=GREY_A),
                     Tex(r"Alternative: keep the parallel traces, and assign identities", color=ATTN),
                     Tex(r"as a labelling pass over them, using the junction pairings.", color=ATTN)
                     ).scale(0.58).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, 0.6, 0])
        self.say(r"Open alternative: assign identities as a labelling pass over the parallel readout's traces,"
                 r" using the junction pairings, instead of re-tracing.")
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in txt], lag_ratio=0.25), run_time=2.0)
        self.wait(3.0)


variants(AttentiveGrouping, module_globals=globals())
