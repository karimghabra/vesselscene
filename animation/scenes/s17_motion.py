"""Chapter 17: motion, the blood flows (spec 4.14).

On vesselscene's burst (frames rendered one by one with imaging.filling's
frame index, see tools/make_vs_data.py): the lumen flickers while the tissue
stays still.  The flicker map (robust temporal spread minus camera noise, in
contrast-to-noise units) finds vessels without using their darkness.  Along
one vessel the space-time image D(s, t) shows oriented streaks; two families
of spatiotemporal energy filters read their direction and speed; with flow on
every arm, a 3-way junction is a bifurcation or a confluence.

Render:  manim -ql scenes/s17_motion.py MotionDetailed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import motion as M  # noqa: E402
from mica_anim import vsdata  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import (CONT, END, FB, INH, JUNC, MUTED, OD_COL, PixelFrame, colormap,  # noqa: E402
                             image)

RUN_VID = 106            # the vessel whose space-time image is shown (an arteriole, ~480 px in frame)
JUNCTIONS = ((95, 152, 23), (44, 109, 17))   # a bifurcation, then a confluence: the first of each whose arms all read
FLOW_COL = TEAL_B        # flicker / flow (no fixed meaning in style.py; used for motion only)


class Motion(ChapterScene):
    NUMBER, TITLE, SPEC = 17, "Motion: the blood flows", "spec 4.14"
    MAP_KEYS = ("motion",)

    def construct(self):
        self.intro()
        if not vsdata.available() or vsdata.load().burst is None:
            self.say(r"(The vesselscene burst is not rendered yet: run tools/make\_vs\_data.py.)", wait=2.0)
            self.outro()
            return
        self.S = vsdata.load()
        self.views = M.views()
        self.n = len(self.views) - 1
        self.flicker_frames()
        self.flicker_map()
        if self.at(2):
            self.direction()
            self.junction()
        if self.at(3):
            self.uses_and_confounds()
        self.outro()

    # ------------------------------------------------------------------ the burst flickers
    def flicker_frames(self):
        x0, y0 = M.best_window(320, 180)
        crop = lambda a: a[y0:y0 + 180, x0:x0 + 320]
        PW = 6.4
        h = PW * 180 / 320
        lc, rc = LEFT * 3.5 + UP * 0.6, RIGHT * 3.5 + UP * 0.6
        fr = image(crop(self.views[1]), h, "grey", 0, 1).move_to(lc)
        av = image(crop(self.views[0]), h, "grey", 0, 1).move_to(rc)
        bl = SurroundingRectangle(fr, buff=0, color=MUTED, stroke_width=1)
        br = SurroundingRectangle(av, buff=0, color=MUTED, stroke_width=1)
        t = ValueTracker(0.0)
        lab = Tex(r"one frame").scale(0.55).next_to(bl, UP, buff=0.12)
        cnt = always_redraw(lambda: Tex(rf"frame {int(t.get_value()) % self.n + 1} of {self.n} (slowed $6\times$)", color=GREY_B)
                            .scale(0.45).next_to(bl, DOWN, buff=0.12))
        labr = Tex(rf"the average of the burst").scale(0.55).next_to(br, UP, buff=0.12)

        def swap(m):
            m.pixel_array = colormap(crop(self.views[1 + int(t.get_value()) % self.n]), "grey", 0, 1)

        fr.add_updater(swap)
        self.say(r"In a burst, the blood moves. Red cells and plasma gaps travel along the vessels, frame after frame.")
        self.play(FadeIn(fr), Create(bl), FadeIn(lab), FadeIn(cnt))
        self.play(t.animate.set_value(2 * self.n - 0.01), run_time=2 * self.n / 5.4, rate_func=linear)
        self.say(r"The lumen flickers; the tissue around it stays still. Averaged, the flicker is gone.")
        self.play(FadeIn(av), Create(br), FadeIn(labr))
        t.set_value(0)
        self.play(t.animate.set_value(self.n - 0.01), run_time=self.n / 5.4, rate_func=linear)
        fr.clear_updaters()
        self.say(r"So flicker marks vessels without asking how dark they are: a cue the OD mask (E24) did not have.", wait="read")
        cnt.clear_updaters()
        self.clear()

    # ------------------------------------------------------------------ the flicker map
    def flicker_map(self):
        f = M.flicker()
        S = self.S
        PW = 6.6
        h = PW * S.shape[0] / S.shape[1]
        pl = PixelFrame(S.shape, h, LEFT * 3.5 + UP * 1.55)
        pr = PixelFrame(S.shape, h, RIGHT * 3.5 + UP * 1.55)
        av = image(self.views[0], h, "grey", 0, 1).move_to(pl.centre)
        fl = f["flow"]
        fl2 = fl[: fl.shape[0] // 2 * 2, : fl.shape[1] // 2 * 2].reshape(fl.shape[0] // 2, 2, fl.shape[1] // 2, 2).mean(axis=(1, 3))
        fm = image(fl2, h, "magma", 0.0, 4.0).move_to(pr.centre)        # shown as 2 x 2 means (the panel is ~400 px wide)
        bl = SurroundingRectangle(av, buff=0, color=MUTED, stroke_width=1)
        br = SurroundingRectangle(fm, buff=0, color=MUTED, stroke_width=1)
        hl = Tex(r"the average (vesselscene, healthy)").scale(0.5).next_to(bl, UP, buff=0.1)
        hr = Tex(rf"flicker, from {self.n} frames", color=FLOW_COL).scale(0.5).next_to(br, UP, buff=0.1)
        self.play(FadeIn(av), Create(bl), FadeIn(hl))

        eqs = VGroup(
            MathTex(r"L_t = \ln F_t", r",\quad", r"D_t = L_t - \operatorname{median}_t L_t").scale(0.62),
            MathTex(r"\Phi", "=", r"1.4826\,\operatorname{median}_t |D_t|", r",\quad", r"\Phi_2", "=",
                    r"\max\!\big(\Phi^2 - \text{noise\_var}(I),\,0\big)").scale(0.62),
            MathTex(r"\text{flow}", "=", r"\frac{\sqrt{\Phi_2}}{\text{local\_rms}(\sqrt{\Phi_2},\ \text{background},\ 48)}").scale(0.62),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to(DOWN * 1.65)
        eqs[1][0].set_color(FLOW_COL)
        eqs[2][0].set_color(FLOW_COL)
        if self.at(2):
            self.say(r"Per pixel: each frame's departure from the burst's median, in log units.")
            self.play(Write(eqs[0]))
            self.say(r"Its robust spread over time (the median absolute deviation), minus what the camera's noise alone would give.")
            self.play(Write(eqs[1]), run_time=1.6)
            self.wait(1.0)
            self.say(r"Divided by the spread of the same quantity over nearby background: flicker in contrast-to-noise units.")
            self.play(Write(eqs[2]), run_time=1.4)
            self.wait(1.0)
            self.play(FadeOut(eqs), run_time=0.6)
        else:
            self.say(r"Per pixel: the robust spread of the frames over time, minus the camera's own noise, in noise units.")
        self.play(FadeIn(fm), Create(br), FadeIn(hr), run_time=1.5)
        self.say(r"Computed on this burst: the vessels light up, the tissue does not.", wait="read")
        if self.at(3):
            c0, c1 = f["coef"]
            note = VGroup(Tex(r"camera noise: the median $\Phi^2$ over background pixels, binned by brightness,", color=GREY_B),
                          Tex(rf"fitted with $c_0 + c_1/I$ (read and shot noise): $c_1 = {c1:.2f}$ DN", color=GREY_B),
                          Tex(r"background: the truth's, as a stand-in for step 1's mask", color=GREY_B)
                          ).scale(0.5).arrange(DOWN, buff=0.08).move_to(DOWN * 1.2)
            self.play(FadeIn(note))
            self.wait(3.5)
            self.play(FadeOut(note))

        # the mask: flow > 3 with hysteresis to 1.5
        m = M.hysteresis(f["flow"])
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[..., :3] = (np.array(FLOW_COL.to_rgb()) * 255).astype(np.uint8)
        rgba[..., 3] = (m * 230).astype(np.uint8)
        ov = ImageMobject(rgba)
        ov.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        ov.height = h
        ov.move_to(pl.centre)
        cov = M.coverage()
        self.say(r"Threshold it at 3, with hysteresis down to 1.5, and lay it on the average.")
        self.play(FadeIn(ov), run_time=1.5)
        rows = VGroup(
            Tex(r"truth centrelines inside the flicker mask (this burst):").scale(0.5),
            Tex(rf"capillaries {100 * cov['capillaries']:.0f}\,\%, arterioles and venules {100 * cov['arterioles and venules']:.0f}\,\%,").scale(0.5),
            Tex(rf"wide episcleral vessels {100 * cov['episcleral (wide)']:.0f}\,\%;"
                rf" background pixels: {100 * cov['background']:.1f}\,\%").scale(0.5),
        ).arrange(DOWN, buff=0.1).move_to(DOWN * 1.45)
        self.play(FadeIn(rows, lag_ratio=0.2))
        self.say(r"Thin vessels flicker most: their red cells pass in single file, with gaps. Wide vessels are full; only clumps of cells move.", wait="read")
        self.clear()

    # ------------------------------------------------------------------ direction: the space-time image
    def direction(self):
        S = self.S
        r = M.run_by_vid(RUN_VID)
        xy = np.asarray(r["xy"], float)
        s, D = M.spacetime(xy)
        dd = M.run_direction(r)
        # the vessel on a crop of the average
        pad = 30
        x0, x1 = int(max(xy[:, 0].min() - pad, 0)), int(min(xy[:, 0].max() + pad, S.shape[1]))
        y0, y1 = int(max(xy[:, 1].min() - pad, 0)), int(min(xy[:, 1].max() + pad, S.shape[0]))
        crop = self.views[0][y0:y1, x0:x1]
        hc = 3.3
        pf = PixelFrame(crop.shape, hc, LEFT * 4.6 + UP * 1.5)
        if pf.width > 4.6:
            pf = PixelFrame(crop.shape, hc * 4.6 / pf.width, LEFT * 4.6 + UP * 1.5)
        av = image(crop, pf.height, "grey", 0, 1).move_to(pf.centre)
        b = SurroundingRectangle(av, buff=0, color=MUTED, stroke_width=1)
        path = VMobject(stroke_color=FLOW_COL, stroke_width=3).set_points_smoothly(
            [pf.point(x - x0, y - y0) for x, y in xy[::6]])
        self.say(r"Which way does it flow? Follow one vessel, and sample the OD along its centreline in every frame.")
        self.play(FadeIn(av), Create(b))
        self.play(Create(path), run_time=1.5)

        # D(s, t): s across, t down
        W, H = 8.6, 2.6
        T = D.shape[0]
        rep = max(int(round(H / T * D.shape[1] / W)), 1)
        from scipy.ndimage import gaussian_filter1d
        Dd = gaussian_filter1d(D, 1.0, axis=1)                  # display only: 1 px along s
        img = np.repeat(Dd, rep, axis=0)
        st = image(img, H, "diverging", -0.12, 0.12)
        st.stretch_to_fit_width(W).stretch_to_fit_height(H)
        st.move_to(RIGHT * 0.0 + DOWN * 1.45)
        sb = SurroundingRectangle(st, buff=0, color=MUTED, stroke_width=1)
        sx = Tex(r"position along the vessel $s$ (px) \;$\rightarrow$ downstream (truth)").scale(0.45).next_to(sb, DOWN, buff=0.08)
        sy = Tex(r"frame $t$").scale(0.45).rotate(PI / 2).next_to(sb, LEFT, buff=0.1)
        sh = Tex(r"$D(s, t)$: more blood (red), less (blue)").scale(0.5).next_to(sb, UP, buff=0.1).align_to(sb, RIGHT)
        self.play(Create(sb), FadeIn(sx), FadeIn(sy), FadeIn(sh))
        # rows appear frame by frame, with a probe running along the vessel
        rows = []
        for t in range(T):
            row = image(np.repeat(Dd[t:t + 1], rep, axis=0), H / T, "diverging", -0.12, 0.12)
            row.stretch_to_fit_width(W).stretch_to_fit_height(H / T)
            row.move_to(st.get_top() + DOWN * (t + 0.5) * H / T)
            rows.append(row)
        dot = Dot(path.get_start(), radius=0.06, color=YELLOW)
        self.add(dot)
        for t, row in enumerate(rows):
            dot.move_to(path.get_start())
            self.play(MoveAlongPath(dot, path), FadeIn(row), run_time=0.45 if t < 3 else 0.18, rate_func=linear)
        self.remove(dot)
        self.add(st)
        self.remove(*rows)
        self.say(r"Moving red cells draw oriented streaks: a clump at $s$ in one frame is further downstream in the next.", wait="read")

        # the two energy filters
        v0 = max(abs(dd["v_xcorr"]), 4.0)
        kd = M.gabor_pair(v0, +1, T)[0]
        ku = M.gabor_pair(v0, -1, T)[0]
        px_w = W / D.shape[1]
        def kim(k):
            im = image(np.repeat(k, rep, axis=0), 1.0, "diverging", -1, 1)
            im.stretch_to_fit_width(k.shape[1] * px_w).stretch_to_fit_height(H * k.shape[0] / T)
            return im
        gd, gu = kim(kd), kim(ku)
        self.play(FadeOut(sh))
        sc_k = 1.8 / max(gd.height, 1e-6)
        gd.scale(sc_k).move_to(RIGHT * 0.2 + UP * 2.05)
        gu.scale(sc_k).move_to(RIGHT * 2.1 + UP * 2.05)
        # the direction each pair is tuned to, drawn with the D(s, t) image's own scales (scaled like the kernels)
        dx = v0 * px_w * sc_k * 2.0
        dy = 2.0 * H / T * sc_k
        tilt_d = DashedLine(gd.get_center() + np.array([-dx / 2, dy / 2, 0]), gd.get_center() + np.array([dx / 2, -dy / 2, 0]),
                            color=WHITE, stroke_width=2, dash_length=0.05)
        tilt_u = DashedLine(gu.get_center() + np.array([dx / 2, dy / 2, 0]), gu.get_center() + np.array([-dx / 2, -dy / 2, 0]),
                            color=WHITE, stroke_width=2, dash_length=0.05)
        gdl = Tex(r"tuned\\downstream", color=FLOW_COL).scale(0.4).next_to(gd, DOWN, buff=0.06)
        gul = Tex(r"tuned\\upstream", color=INH).scale(0.4).next_to(gu, DOWN, buff=0.06)
        self.say(r"Two families of spatiotemporal filters, tilted like the streaks: one for downstream motion, one for upstream (Adelson and Bergen 1985).")
        self.play(FadeIn(gd), FadeIn(gu), FadeIn(gdl), FadeIn(gul))
        self.play(Create(tilt_d), Create(tilt_u))
        self.wait(1.0)

        # the energies against speed
        sp, Ep, Em = dd["speeds"], dd["Ep"], dd["Em"]
        mx = max(Ep.max(), Em.max())
        ax = Axes(x_range=[0, 25, 5], y_range=[0, 1.05, 0.5], x_length=2.9, y_length=1.7,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(RIGHT * 5.3 + UP * 2.15)
        axt = VGroup(*[MathTex(f"{v}").scale(0.36).next_to(ax.c2p(v, 0), DOWN, buff=0.05) for v in (0, 10, 20)])
        axl = Tex(r"speed (px/frame)").scale(0.4).next_to(axt, DOWN, buff=0.04)
        ayl = Tex(r"energy").scale(0.4).rotate(PI / 2).next_to(ax, LEFT, buff=0.08)
        cd = ax.plot_line_graph(sp, Ep / mx, line_color=FLOW_COL, add_vertex_dots=False, stroke_width=3)
        cu = ax.plot_line_graph(sp, Em / mx, line_color=INH, add_vertex_dots=False, stroke_width=3)
        self.play(Create(ax), FadeIn(axt), FadeIn(axl), FadeIn(ayl))
        self.play(Create(cd), Create(cu), run_time=1.5)
        vbest = float(sp[np.argmax(Ep if Ep.max() >= Em.max() else Em)])
        mm_s = vbest * M.FPS * M.UM_PER_PX / 1000.0
        downstream = Ep.max() >= Em.max()
        res = Tex(rf"{'downstream' if downstream else 'upstream'}: {vbest:.0f} px/frame $\approx$ {mm_s:.2f} mm/s",
                  color=FLOW_COL if downstream else INH).scale(0.42).next_to(axl, DOWN, buff=0.1).align_to(ax, RIGHT)
        self.play(FadeIn(res))
        truth_ok = downstream          # the truth's flow runs u -> v, i.e. towards increasing s
        self.say(rf"Downstream energy wins: the flow runs this way at about {mm_s:.1f} mm/s"
                 rf"{', as in the truth' if truth_ok else ', against the truth'} (vesselscene's red cells: about 0.5 mm/s).", wait="read")
        self.clear()

    # ------------------------------------------------------------------ junctions: bifurcation or confluence
    def junction_panel(self, j, arms, centre):
        S = self.S
        half = 64
        cx, cy = int(round(j["x"])), int(round(j["y"]))
        x0, y0 = int(np.clip(cx - half, 0, S.shape[1] - 2 * half)), int(np.clip(cy - half, 0, S.shape[0] - 2 * half))
        crop = self.views[0][y0:y0 + 2 * half, x0:x0 + 2 * half]
        pf = PixelFrame(crop.shape, 3.6, centre)
        av = image(crop, pf.height, "grey", 0, 1).move_to(pf.centre)
        b = SurroundingRectangle(av, buff=0, color=MUTED, stroke_width=1)
        ring = Circle(radius=max(j["radius"], 7.0) * pf.unit, color=JUNC, stroke_width=3).move_to(pf.point(cx - x0, cy - y0))
        arrows = VGroup()
        for a in arms:
            xy = a["xy"]
            if len(xy) < 8 or a["measured"] is None:
                continue
            d = np.hypot(xy[:, 0] - j["x"], xy[:, 1] - j["y"])
            pick = lambda dist: xy[int(np.argmin(np.abs(d - dist)))]
            p_far = pick(min(d.max(), half - 8))
            p_near = pick(max(d.min(), j["radius"] + 8))
            P_far, P_near = pf.point(p_far[0] - x0, p_far[1] - y0), pf.point(p_near[0] - x0, p_near[1] - y0)
            start, end = (P_far, P_near) if a["measured"] == "in" else (P_near, P_far)
            arrows.add(Arrow(start, end, buff=0.0, color=FLOW_COL, stroke_width=6, max_tip_length_to_length_ratio=0.25))
        n_in = sum(a["measured"] == "in" for a in arms)
        n_out = sum(a["measured"] == "out" for a in arms)
        kind = "confluence" if (n_in, n_out) == (2, 1) else ("bifurcation" if (n_in, n_out) == (1, 2) else "unclear")
        agree = sum(a["measured"] == a["truth"] for a in arms)
        verdict = Tex(rf"{n_in} in, {n_out} out $\Rightarrow$ \emph{{{kind}}}", color=JUNC).scale(0.55).next_to(b, DOWN, buff=0.15)
        tru = Tex(rf"truth: {j['type_observable']}; arms right: {agree} of {len(arms)}", color=GREY_B).scale(0.45)
        tru.next_to(verdict, DOWN, buff=0.08)
        return dict(img=av, box=b, ring=ring, arrows=arrows, verdict=verdict, truth=tru, kind=kind)

    def junction(self):
        picks = []
        for group in JUNCTIONS:
            best = None
            for jid in group:
                j, arms = M.junction_arms(jid)
                if all(a["measured"] is not None for a in arms) and all(a["measured"] == a["truth"] for a in arms):
                    best = (j, arms)
                    break
                if best is None and all(a["measured"] is not None for a in arms):
                    best = (j, arms)
            if best is not None:
                picks.append(best)
        rule = VGroup(Tex(r"one arm in, two out: \emph{bifurcation}").scale(0.55),
                      Tex(r"two arms in, one out: \emph{confluence}").scale(0.55)).arrange(RIGHT, buff=0.8).to_edge(UP, buff=0.35)
        self.say(r"In a still, a Y is a Y: the blood could split there, or merge. Only the flow on its arms can tell.")
        self.play(FadeIn(rule, lag_ratio=0.3))
        centres = [LEFT * 3.5 + UP * 0.75, RIGHT * 3.5 + UP * 0.75] if len(picks) > 1 else [UP * 0.75]
        panels = [self.junction_panel(j, arms, c) for (j, arms), c in zip(picks, centres)]
        self.play(*[FadeIn(p["img"]) for p in panels], *[Create(p["box"]) for p in panels])
        self.play(*[Create(p["ring"]) for p in panels])
        self.say(r"Measure the flow on each arm, as before, and count the arms going in and out.")
        for p in panels:
            self.play(LaggedStart(*[GrowArrow(a) for a in p["arrows"]], lag_ratio=0.4), run_time=1.8)
            self.play(FadeIn(p["verdict"]), FadeIn(p["truth"]))
        kinds = " and a ".join(p["kind"] for p in panels)
        self.say(rf"The same Y shape, opposite flows: a {kinds}, which no still image could tell apart.", wait="read")
        self.clear()

    # ------------------------------------------------------------------ uses, confounds, status
    def uses_and_confounds(self):
        x, p, n = M.edge_vs_centre()
        self.say(r"Uses, and what can go wrong.")
        uses = VGroup(
            Tex(r"\textbf{Uses}").scale(0.6),
            Tex(r"step 1 and 4.13: add flow $> 3$ (hysteresis 1.5) to the background mask").scale(0.5),
            Tex(r"steps 3--5: an extra drive into the line evidence").scale(0.5),
            Tex(r"junctions: bifurcation or confluence, from the arms' flow").scale(0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_corner(UL, buff=0.5)
        self.play(FadeIn(uses, lag_ratio=0.25), run_time=1.6)
        self.wait(2.0)
        conf = VGroup(
            Tex(r"\textbf{Confounds}").scale(0.6),
            Tex(r"registration error makes every high-contrast edge flicker, walls included:").scale(0.5),
            Tex(r"score the lumen centre only, or subtract gradient $\times$ registration error").scale(0.5),
            Tex(r"capillaries that are not perfused do not flicker:").scale(0.5),
            Tex(r"motion can only add vessels to the mask, never remove them").scale(0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(uses, DOWN, buff=0.4).align_to(uses, LEFT)
        for k in (2, 4):
            conf[k].shift(RIGHT * 0.3)
        self.play(FadeIn(conf, lag_ratio=0.25), run_time=1.6)
        # the burst's own profile across wide vessels
        ytop = float(max(1.0, np.ceil(p.max() * 1.2 * 2) / 2))
        ax = Axes(x_range=[0, 3.2, 0.8], y_range=[0, ytop, 0.5], x_length=4.0, y_length=2.0,
                  axis_config=dict(color=MUTED, include_tip=False, stroke_width=1.2)).move_to(RIGHT * 4.3 + DOWN * 0.7)
        x = x + 1.6                                             # half widths from the left wall region; centre at 1.6
        xt = VGroup(*[Tex(t_).scale(0.38).next_to(ax.c2p(v, 0), DOWN, buff=0.06)
                      for t_, v in (("wall", 0.6), ("centre", 1.6), ("wall", 2.6))])
        xl = Tex(r"across 29 wide vessels (mean profile)".replace("29", str(n))).scale(0.4).next_to(xt, DOWN, buff=0.06)
        yl = Tex(r"flow").scale(0.4).rotate(PI / 2).next_to(ax, LEFT, buff=0.06)
        ytk = VGroup(*[MathTex(f"{v:.1f}").scale(0.36).next_to(ax.c2p(0, v), LEFT, buff=0.05) for v in np.arange(0.5, ytop + 0.01, 0.5)])
        yl.next_to(ytk, LEFT, buff=0.06)
        ttl = Tex(r"flicker across wide vessels, this burst", color=FLOW_COL).scale(0.45).next_to(ax, UP, buff=0.1)
        cur = ax.plot_line_graph(x, p, line_color=FLOW_COL, add_vertex_dots=False, stroke_width=3)
        walls = VGroup(*[DashedLine(ax.c2p(w, 0), ax.c2p(w, ytop), color=MUTED, stroke_width=1.2, dash_length=0.04) for w in (0.6, 2.6)])
        self.play(Create(ax), FadeIn(xt), FadeIn(xl), FadeIn(yl), FadeIn(ytk), FadeIn(ttl), Create(walls), Create(cur))
        self.say(rf"On this burst the walls do not flicker ({n} wide vessels, mean profile): vesselscene's frames are perfectly registered. Real bursts will.", wait="read")
        self.clear()
        st = VGroup(
            Tex(r"Status (spec 1.1): phase 8, not started.").scale(0.6),
            Tex(r"Spec 9: vesselscene needs a burst renderer. The burst shown here was rendered").scale(0.5),
            Tex(r"frame by frame for this video (imaging.filling at each frame's index).").scale(0.5),
            Tex(r"Gate (spec 6.2): background offset not up; wide-vessel lumen coverage up;").scale(0.5),
            Tex(r"flow direction right on most perfused vessels.").scale(0.5),
        ).arrange(DOWN, buff=0.14).move_to(UP * 0.5)
        st[3].shift(DOWN * 0.2)
        st[4].shift(DOWN * 0.2)
        self.play(FadeIn(st, lag_ratio=0.3), run_time=1.5)
        self.say(r"Motion is the one cue that needs no darkness at all, and the only one that knows which way the blood goes.", wait="read")


variants(Motion, module_globals=globals())
