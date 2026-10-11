"""Chapter 13: sixteen sheets, reading out per orientation.

Sheets and stacks (spec 4.0) and the readout per orientation (4.9c).  A
layer's output is kept per orientation: 16 maps per size channel.  Read
through one pixel they are a tuning curve; read across one sheet they show
where that orientation is active.  Collapsing the stack to the best
orientation per pixel before tracing cuts the weaker vessel at a crossing;
reading the ridges in each sheet keeps it.

Render:  manim -ql scenes/s13_sheets.py SheetsDetailed
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import *
from manim.constants import RESAMPLING_ALGORITHMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim import grouping as GR  # noqa: E402
from mica_anim import neuro as N  # noqa: E402
from mica_anim.chapter import ChapterScene, variants  # noqa: E402
from mica_anim.style import INH, JUNC, MUTED, PixelFrame, image, orientation_color  # noqa: E402
from mica_anim.widgets import PolarTuning, bar_chart  # noqa: E402

VMAX = 40.0
SHEAR, SQUASH, DY = 0.55, 0.34, 0.235       # the stack's pseudo-3-D view


def rgba(arr):
    im = ImageMobject(arr)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return im


def tinted(values, theta, vmax=VMAX, alpha_floor=0.0, gamma=0.6):
    """A sheet as an RGBA image: its orientation's hue, brightness and opacity from the response."""
    v = np.clip(values / vmax, 0, 1) ** gamma
    col = np.array(orientation_color(theta).to_rgb())
    out = np.zeros(values.shape + (4,))
    out[..., :3] = col * (0.25 + 0.75 * v[..., None])
    out[..., 3] = np.maximum(v, alpha_floor)
    return (out * 255).astype(np.uint8)


def od_view(od, peak=0.15):
    return 0.08 + 0.7 * np.clip(np.maximum(od, 0) / peak, 0, 1) ** 0.45


def ridge_overlay(mask, color=WHITE, alpha=1.0, colors=None):
    out = np.zeros(mask.shape + (4,))
    if colors is None:
        out[mask, :3] = np.array(color.to_rgb())
    else:
        out[mask, :3] = colors[mask]
    out[mask, 3] = alpha
    return (out * 255).astype(np.uint8)


def thick(mask):
    """Ridges drawn two pixels wide, so they read at video size."""
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, structure=np.ones((2, 2), bool))


def sheet_mobject(arr, side, centre):
    """A square image sheared and squashed into a sheet of the stack."""
    im = rgba(arr)
    im.height = side
    im.apply_matrix([[1, SHEAR], [0, SQUASH]])
    im.move_to(centre)
    return im


def sheet_outline(side, centre, color):
    h = side / 2
    corners = [np.array([x + SHEAR * y, SQUASH * y, 0]) for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))]
    return Polygon(*corners, stroke_color=color, stroke_width=1.2, stroke_opacity=0.7).move_to(centre)


def sheet_point(px, py, shape, side, centre):
    """Where image pixel (px, py) of a sheet lies in the scene."""
    h, w = shape
    x = (px + 0.5) / w * side - side / 2
    y = side / 2 - (py + 0.5) / h * side
    return np.array(centre) + np.array([x + SHEAR * y, SQUASH * y, 0])


class Sheets(ChapterScene):
    NUMBER, TITLE, SPEC = 13, "Sixteen sheets: reading out per orientation", "spec 4.0, 4.9c"
    MAP_KEYS = ("readout",)

    def construct(self):
        self.intro()
        self.stack()
        if self.at(2):
            self.two_reads()
        self.collapse()
        if self.at(2):
            self.table()
        if self.at(3):
            self.as_built()
            self.real_numbers()
        self.outro()

    # ------------------------------------------------------------- the stack
    def stack(self):
        st = GR.crossing_stack(30.0, 0.15, 0.03)
        self.st = st
        C = st["C"]
        H = 3.4
        pf = PixelFrame(GR.RO_SHAPE, H, np.array([-4.3, 0.45, 0]))
        odim = image(od_view(st["od"]), H, "grey", vmin=0, vmax=1).move_to(pf.centre)
        odfr = SurroundingRectangle(odim, buff=0, color=MUTED, stroke_width=1)
        ttl = Tex(r"a faint vessel (0.03 OD) across\\a strong one (0.15 OD), at $30^\circ$").scale(0.5)
        ttl.next_to(odim, UP, buff=0.15)
        self.say(r"A crossing: a faint vessel across a strong one, at $30^\circ$ (computed toy, as optical density).")
        self.play(FadeIn(odim), Create(odfr), FadeIn(ttl), run_time=1.0)
        self.wait(0.6 if self.short else 1.2)

        # the sixteen sheets, laid out flat
        side, gap = 1.36, 0.1
        grid, frames, labels = Group(), VGroup(), VGroup()
        for k in range(N.N_ORIENT):
            r, c = divmod(k, 4)
            centre = np.array([1.45 + c * (side + gap), 2.55 - r * (side + gap), 0])
            im = rgba(tinted(C[k], N.THETAS[k], alpha_floor=1.0))
            im.height = side
            im.move_to(centre)
            grid.add(im)
            frames.add(SurroundingRectangle(im, buff=0, color=orientation_color(N.THETAS[k]), stroke_width=1.5))
            labels.add(MathTex(rf"{np.degrees(N.THETAS[k]):.1f}^\circ", color=orientation_color(N.THETAS[k]))
                       .scale(0.32).move_to(im.get_corner(UL) + np.array([0.3, -0.12, 0])))
        self.say(r"A layer's output is kept per orientation: 16 maps, the \emph{sheets}, one per preferred orientation (spec 4.0).")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(g_, scale=0.9), Create(f_), FadeIn(l_)) for g_, f_, l_ in zip(grid, frames, labels)],
                              lag_ratio=0.1), run_time=2.6 if self.short else 3.4)
        self.say(r"Each vessel lights the sheets near its own orientation: the strong one brightly, the faint one dimly.",
                 wait=1.2 if self.short else "read")
        self.mobs = dict(odim=odim, odfr=odfr, ttl=ttl, grid=grid, frames=frames, labels=labels, pf=pf)

    def _pile_centre(self, k):
        return np.array([2.9, -2.35, 0]) + UP * (DY * k + self.side * SQUASH / 2)

    # ------------------------------------------------- two ways to read it
    def two_reads(self):
        m, C = self.mobs, self.st["C"]
        self.side = side = 2.9
        # stack the sheets into a pile
        pile, outlines = Group(), VGroup()
        for k in range(N.N_ORIENT):
            pile.add(sheet_mobject(tinted(C[k], N.THETAS[k], alpha_floor=0.04), side, self._pile_centre(k)))
            outlines.add(sheet_outline(side, self._pile_centre(k), orientation_color(N.THETAS[k])))
        self.say(r"Stacked, they form the layer: the same pixel in every sheet lies on one vertical line.")
        self.play(FadeOut(m["odim"]), FadeOut(m["odfr"]), FadeOut(m["ttl"]), FadeOut(m["labels"]), run_time=0.6)
        anims = []
        for k in range(N.N_ORIENT):
            g_, f_ = m["grid"][k], m["frames"][k]
            tgt = self._pile_centre(k)
            sc = side / g_.height
            anims.append(g_.animate.scale(sc).apply_matrix([[1, SHEAR], [0, SQUASH]]).move_to(tgt))
            anims.append(Transform(f_, outlines[k]))
        self.play(LaggedStart(*anims, lag_ratio=0.03), run_time=2.0)
        self.play(*[FadeOut(g_) for g_ in m["grid"]], FadeIn(pile), run_time=0.6)
        self.remove(*m["frames"], *m["grid"])
        self.add(outlines)
        brace = Brace(VGroup(outlines[0], outlines[-1]), RIGHT, buff=0.1, color=MUTED)
        bl = Tex(r"16 sheets", color=GREY_A).scale(0.45).next_to(brace, RIGHT, buff=0.1)
        self.play(GrowFromCenter(brace), FadeIn(bl), run_time=0.6)

        # through one pixel: a probe on the strong vessel, near the crossing
        p = GR.RO_CENTRE + 22 * self.st["us"]
        pts = [sheet_point(p[0], p[1], GR.RO_SHAPE, side, self._pile_centre(k)) for k in range(N.N_ORIENT)]
        probe = VGroup(Line(pts[0] + DOWN * 0.25, pts[-1] + UP * 0.25, stroke_color=WHITE, stroke_width=2),
                       *[Dot(q, radius=0.045, color=WHITE) for q in pts])
        vals = C[:, int(round(p[1])), int(round(p[0]))]
        polar = PolarTuning(radius=1.55, vmax=float(vals.max()) * 1.05, rings=()).move_to([-3.6, 0.35, 0])
        spokes = polar.spokes(vals, width=4)
        curve = polar.curve(vals, WHITE, width=3)
        pt = Tex(r"read through one pixel:\\its tuning curve (chapter 4)").scale(0.5).next_to(polar, UP, buff=0.25)
        self.say(r"Read through one pixel, the 16 values are that hypercolumn's tuning curve.")
        self.play(Create(probe), run_time=1.0)
        self.play(FadeIn(polar.axes), FadeIn(pt), LaggedStart(*[TransformFromCopy(probe[1 + k], spokes[k]) for k in range(16)],
                                                              lag_ratio=0.05), run_time=2.0)
        self.play(Create(curve))
        self.wait(1.4)
        self.play(FadeOut(VGroup(polar.axes, spokes, curve, pt, probe)), run_time=0.6)

        # across one sheet: pull the faint vessel's sheet out and flatten it
        k_f = int(np.argmin(np.abs(((N.THETAS - self.st["theta_f"]) + np.pi / 2) % np.pi - np.pi / 2)))
        flat = rgba(tinted(C[k_f], N.THETAS[k_f], alpha_floor=1.0))
        flat.height = 3.2
        flat.move_to([-3.6, 0.35, 0])
        ffr = SurroundingRectangle(flat, buff=0, color=orientation_color(N.THETAS[k_f]), stroke_width=1.5)
        ft = Tex(rf"read across one sheet ($\theta = {np.degrees(N.THETAS[k_f]):.0f}^\circ$):\\where that orientation is active").scale(0.5)
        ft.next_to(flat, UP, buff=0.2)
        self.say(r"Read across one sheet, they show where that orientation is active: here, the faint vessel's sheet.")
        self.play(Indicate(outlines[k_f], color=WHITE, scale_factor=1.05), run_time=0.8)
        sc = pile[k_f].copy()
        self.add(sc)
        self.play(sc.animate.move_to(flat.get_center()), TransformFromCopy(outlines[k_f], ffr), run_time=1.2)
        self.play(FadeOut(sc), FadeIn(flat), FadeIn(ft), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(flat), FadeOut(ffr), FadeOut(ft), FadeOut(brace), FadeOut(bl), run_time=0.6)
        m.update(pile=pile, outlines=outlines)

    # ------------------------------------------------------ the collapse
    def collapse(self):
        m, st = self.mobs, self.st
        C = st["C"]
        ms = GR.missing_stretch(st)
        k_f = ms["k_f"]
        H = 3.55
        L, R = np.array([-3.6, 0.55, 0]), np.array([3.6, 0.55, 0])
        c0, n = 30, 60                                   # zoom: a 60 px window around the crossing
        crop = (slice(c0, c0 + n), slice(c0, c0 + n))
        pfL0, pfR0 = PixelFrame((n, n), H, L), PixelFrame((n, n), H, R)

        class _PF:                                       # image pixel -> scene, through the crop
            def __init__(self, pf):
                self.pf, self.unit = pf, pf.unit

            def point(self, x, y):
                return self.pf.point(x - c0, y - c0)
        pfL, pfR = _PF(pfL0), _PF(pfR0)
        # the collapsed map: hue of the best orientation, brightness of its value
        M, kb = ms["M"], ms["kb"]
        v = np.clip(M / VMAX, 0, 1) ** 0.6
        hues = np.stack([np.array(orientation_color(t).to_rgb()) for t in N.THETAS])
        cols = hues[kb]
        arr = np.concatenate([cols * 0.75 * v[..., None], np.ones(M.shape + (1,))], -1)[crop]
        cmap = rgba((arr * 255).astype(np.uint8))
        cmap.height = H
        cmap.move_to(L)
        cfr = SurroundingRectangle(cmap, buff=0, color=MUTED, stroke_width=1)
        ct = Tex(r"collapsed: the best orientation per pixel\\(the old step 6; zoomed on the crossing)").scale(0.5).next_to(cmap, UP, buff=0.15)
        self.say(r"The old step 6 collapsed the stack to the best orientation per pixel before tracing.")
        if self.at(2):
            src, outl = m["pile"], m["outlines"]
            bottom = self._pile_centre(0)
            self.play(*[sh.animate.move_to(bottom) for sh in src], *[o.animate.move_to(bottom) for o in outl], run_time=1.4)
            self.play(FadeOut(src), FadeOut(outl), FadeIn(cmap, shift=LEFT * 0.3), FadeIn(cfr), FadeIn(ct), run_time=1.2)
        else:
            self.play(FadeOut(m["odim"]), FadeOut(m["odfr"]), FadeOut(m["ttl"]), FadeOut(m["labels"]), run_time=0.5)
            self.play(*[g_.animate.move_to(R).set_opacity(0.0) for g_ in m["grid"]],
                      *[f_.animate.move_to(R).set_opacity(0.0) for f_ in m["frames"]], run_time=1.2)
            self.remove(*m["grid"], *m["frames"])
            self.play(FadeIn(cmap, shift=LEFT * 0.3), FadeIn(cfr), FadeIn(ct), run_time=1.0)
        # ridge pixels by the orientation they report: the faint vessel's (one sheet either side), the strong one's, other
        k_s = int(np.argmin(np.abs(((N.THETAS - st["theta_s"]) + np.pi / 2) % np.pi - np.pi / 2)))
        cls = np.full(M.shape + (3,), 0.45)
        cls[GR._kdist(kb, k_f) <= 1] = hues[k_f]
        cls[GR._kdist(kb, k_s) <= 1] = hues[k_s]
        rc = rgba(ridge_overlay(thick(ms["Rc"])[crop], colors=cls[crop]))
        rc.height = H
        rc.move_to(L)
        self.play(FadeIn(rc), run_time=0.8)
        # the gap in the faint vessel's track
        g = ms["gap_c"]
        p0 = GR.RO_CENTRE + g.min() * st["uf"]
        p1 = GR.RO_CENTRE + g.max() * st["uf"]
        nrm = np.array([-st["uf"][1], st["uf"][0]])
        a0, a1 = pfL.point(*(p0 - 6 * nrm)), pfL.point(*(p1 - 6 * nrm))
        gap_line = VGroup(Line(a0, a1, stroke_color=INH, stroke_width=4),
                          Line(a0, pfL.point(*(p0 - 2 * nrm)), stroke_color=INH, stroke_width=3),
                          Line(a1, pfL.point(*(p1 - 2 * nrm)), stroke_color=INH, stroke_width=3))
        gl = Tex(rf"no crest of its own: {ms['collapsed']:.1f} px (toy)", color=INH).scale(0.48)
        gl.next_to(cmap, DOWN, buff=0.12)
        self.say(r"Near the crossing the strong vessel wins every pixel: the faint one has no crest at its own orientation, and its track is cut.")
        self.play(Create(gap_line), FadeIn(gl), run_time=1.0)
        self.wait(1.0 if self.short else 1.6)
        extra = VGroup()
        if self.at(2):
            rings = VGroup(*[Circle(radius=0.28, stroke_color=JUNC, stroke_width=3).move_to(pfL.point(*(GR.RO_CENTRE + t * st["uf"])))
                             for t in (g.min() - 1.0, g.max() + 1.0)])
            tl = VGroup(*[Tex("T", color=JUNC).scale(0.6).next_to(r, nd, buff=0.04) for r, nd in zip(rings, (RIGHT, LEFT))])
            self.say(r"So each such vessel is cut in two, and reads as two T-junctions.")
            self.play(Create(rings), FadeIn(tl))
            self.wait(1.5)
            extra.add(rings, tl)
            # why: at a pixel of the faint vessel near the crossing, the strong vessel's lobe wins
            q = GR.RO_CENTRE + 5.0 * st["uf"]
            vals = C[:, int(round(q[1])), int(round(q[0]))]
            ring = Circle(radius=0.12, stroke_color=WHITE, stroke_width=3).move_to(pfL.point(*q))
            ch = bar_chart(vals, width=4.2, height=2.3, vmax=float(vals.max()) * 1.1,
                           colors=[orientation_color(t) for t in N.THETAS]).move_to(R + DOWN * 0.3 + LEFT * 0.5)
            kbest = int(np.argmax(vals))
            bt = Tex(r"the 16 values at that pixel, 5 px from the crossing").scale(0.48).next_to(ch, UP, buff=0.3)
            b1 = Tex(r"best: toward the\\strong vessel", color=orientation_color(N.THETAS[kbest])).scale(0.42)
            b2 = Tex(r"the faint vessel's\\own sheet: discarded", color=orientation_color(N.THETAS[k_f])).scale(0.42)
            top1, top2 = ch[1][kbest].get_top(), ch[1][k_f].get_top()
            b1.next_to(ch, RIGHT, buff=0.15).set_y(top1[1])
            b2.move_to([ch.get_left()[0] + 1.6, top2[1] + 0.15, 0])
            ar1 = Arrow(b1.get_left() + LEFT * 0.02, top1 + RIGHT * 0.05, buff=0.04, stroke_width=2,
                        color=orientation_color(N.THETAS[kbest]), max_tip_length_to_length_ratio=0.2)
            ar2 = Arrow(b2.get_right() + RIGHT * 0.02, top2 + LEFT * 0.05, buff=0.04, stroke_width=2,
                        color=orientation_color(N.THETAS[k_f]), max_tip_length_to_length_ratio=0.2)
            b1, b2 = VGroup(b1, ar1), VGroup(b2, ar2)
            self.say(r"Why: on the faint vessel near the crossing, the strong vessel's lobe is the larger, so the best"
                     r" orientation is the strong one's.")
            self.play(Create(ring), FadeIn(bt), Create(ch), run_time=1.2)
            self.play(FadeIn(b1), FadeIn(b2), Indicate(ch[1][k_f], color=WHITE), run_time=1.2)
            self.wait(2.2)
            self.play(FadeOut(VGroup(ch, bt, b1, b2)), FadeOut(ring), run_time=0.6)

        # per orientation: the faint vessel's own sheet keeps its crest
        sh = rgba(tinted(C[k_f][crop], N.THETAS[k_f], alpha_floor=0.0))
        sh.height = H
        sh.move_to(R)
        bgR = Square(side_length=H, stroke_width=0, fill_color=BLACK, fill_opacity=1).move_to(R)
        sfr = SurroundingRectangle(sh, buff=0, color=orientation_color(N.THETAS[k_f]), stroke_width=1.5)
        stt = Tex(rf"per orientation: the faint vessel's sheet\\($\theta = {np.degrees(N.THETAS[k_f]):.2f}^\circ$, and one either side)").scale(0.5)
        stt.next_to(sh, UP, buff=0.15)
        own = ms["Rs"][[(k_f - 1) % 16, k_f, (k_f + 1) % 16]].any(axis=0)
        rs = rgba(ridge_overlay(thick(own)[crop], colors=np.broadcast_to(hues[k_f], own[crop].shape + (3,))))
        rs.height = H
        rs.move_to(R)
        self.say(r"Read per orientation (layer 06): ridge tops in each sheet, across that sheet's own normal.")
        self.play(FadeIn(bgR), FadeIn(sh), Create(sfr), FadeIn(stt), run_time=1.2)
        self.play(FadeIn(rs), run_time=0.8)
        ok = Tex(rf"its crest runs through: missing {ms['sheets']:.0f} px (toy)", color=GREEN_C).scale(0.48)
        ok.next_to(sh, DOWN, buff=0.12)
        hl = Circle(radius=0.9, stroke_color=GREEN_C, stroke_width=3).move_to(pfR.point(*GR.RO_CENTRE))
        self.say(r"In its own sheet the faint vessel keeps a crest through the crossing, and its track is not cut.",
                 wait=0.5)
        self.play(Create(hl), FadeIn(ok))
        self.wait(1.5 if self.short else 3.0)
        self.coll = Group(cmap, cfr, ct, rc, gap_line, gl, extra, bgR, sh, sfr, stt, rs, ok, hl)

    # ---------------------------------------------------- toy vs measured
    def table(self):
        self.play(FadeOut(self.coll), run_time=0.8)
        rows = GR.missing_table()
        spec = (1, 5, 7, 41)
        cells = [Tex(r"crossing (synthetic)"), Tex(r"toy: collapsed"), Tex(r"toy: per sheet", color=GREEN_C),
                 Tex(r"measured (spec 4.9c)", color=GREY_B)]
        for (name, c, s), sp in zip(rows, spec):
            cells += [Tex(name.replace("°", r"$^\circ$")), Tex(f"{c:.1f} px"), Tex(f"{s:.0f} px", color=GREEN_C),
                      Tex(f"{sp} px", color=GREY_B)]
        tbl = VGroup(*[c.scale(0.55) for c in cells]).arrange_in_grid(rows=5, cols=4, buff=(0.5, 0.2), col_alignments="lrrr")
        head = Tex(r"the stretch of the second vessel with no track of its own").scale(0.58)
        blk = VGroup(head, tbl).arrange(DOWN, buff=0.3).move_to([0, 0.75, 0])
        line = Line(tbl.get_left(), tbl.get_right(), stroke_color=MUTED, stroke_width=1).next_to(cells[0], DOWN, buff=0.1)
        line.set_x(tbl.get_x())
        self.say(r"The missing stretch, on the spec's four synthetic crossings: worst when one vessel is faint and the angle shallow.")
        self.play(FadeIn(head), FadeIn(tbl), Create(line), run_time=1.2)
        self.wait(3.0 if self.at(3) else 2.0)
        note = Tex(r"The toy shows the same order, with shorter stretches: the spec's numbers come from Karim's earlier readout.",
                   color=GREY_B).scale(0.45).next_to(tbl, DOWN, buff=0.3)
        self.say(r"Measured on Karim's earlier readout; this toy's filters are shorter, so its stretches are shorter, in the same order.",
                 wait="read")
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear()

    # ------------------------------------------------- level 3: as built
    def as_built(self):
        card = VGroup(Tex(r"Conventions (spec 4.0):", color=GREY_B),
                      Tex(r"widths (and curvatures) compete inside each hypercolumn;"),
                      Tex(r"the winner's value is kept in a map,"),
                      Tex(r"so later layers still see 16 sheets per channel."),
                      Tex(r"Lines exist only from the readout on: before it, only rates.", color=GREEN_C)
                      ).scale(0.55).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, 0.6, 0])
        self.say(r"So every layer up to the readout keeps all 16 sheets; lines, traces and junction lists are decoded from them.")
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in card], lag_ratio=0.25), run_time=1.6)
        self.wait(3.5)
        self.play(FadeOut(card), run_time=0.6)
        S = GR.sine_trace()
        H = 2.4
        pf = PixelFrame(GR.SINE_SHAPE, H, np.array([-1.3, 1.95, 0]))
        im = image(od_view(S["od"], 0.12), H, "grey", vmin=0, vmax=1).move_to(pf.centre)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        rules = VGroup(Tex(r"readout per orientation (layer 06):", color=GREY_B),
                       Tex(r"ridge tops in each sheet"),
                       Tex(r"start above $t_\text{high} = 8$"),
                       Tex(r"continue above $t_\text{low} = 2$"),
                       Tex(r"steps within 2 px along the line"),
                       Tex(r"in the same sheet, or one either side")).scale(0.48).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        rules.next_to(im, RIGHT, buff=0.35)
        self.say(r"Tracing within a sheet, or one either side, follows curves: a step may move one sheet.")
        self.play(FadeIn(im), Create(fr), FadeIn(rules, lag_ratio=0.1), run_time=1.2)
        xy, ks = S["xy"], S["k"]
        segs = VGroup()
        for i in range(len(xy) - 1):
            segs.add(Line(pf.point(*xy[i]), pf.point(*xy[i + 1]), stroke_color=orientation_color(N.THETAS[ks[i]]), stroke_width=5))
        self.play(Create(segs), run_time=3.5, rate_func=linear)
        self.wait(1.0)
        # decoded orientation against the tangent
        ax = Axes(x_range=[0, 240, 60], y_range=[-40, 40, 20], x_length=8.0, y_length=1.9, tips=False,
                  axis_config=dict(color=MUTED, stroke_width=1.5, font_size=20),
                  y_axis_config=dict(numbers_to_include=[-40, -20, 0, 20, 40])).move_to([0.2, -1.05, 0])
        yl = Tex(r"orientation ($^\circ$)", color=GREY_B).scale(0.42).next_to(ax, LEFT, buff=0.15).rotate(PI / 2)
        xl = Tex(r"$x$ (px)", color=GREY_B).scale(0.42).next_to(ax.x_axis, RIGHT, buff=0.1)

        def wrap(a):
            return (np.degrees(a) + 90) % 180 - 90
        order = np.argsort(xy[:, 0])
        xs = xy[order, 0]
        true = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners([ax.c2p(x, wrap(t)) for x, t in zip(xs, S["true"][order])])
        dots = VGroup(*[Dot(ax.c2p(x, wrap(d)), radius=0.03, color=orientation_color(N.THETAS[k]))
                        for x, d, k in zip(xs[::2], S["decoded"][order][::2], ks[order][::2])])
        key = VGroup(Tex(r"tangent", color=WHITE), Tex(r"decoded from the sheets", color=GREY_A)).scale(0.42).arrange(RIGHT, buff=0.4)
        key.next_to(ax, UP, buff=0.05).align_to(ax, RIGHT)
        self.play(Create(ax), FadeIn(yl), FadeIn(xl), FadeIn(key), run_time=0.8)
        self.play(Create(true), FadeIn(dots, lag_ratio=0.01), run_time=1.5)
        res = Tex(rf"toy: {np.median(S['err']):.1f}$^\circ$ median, {S['err'].max():.1f}$^\circ$ at worst"
                  r" (spec 4.9c: about $2^\circ$)", color=GREY_A).scale(0.45)
        res.next_to(ax, DOWN, buff=0.2)
        self.say(r"On a synthetic sine, the reported orientation matches the tangent within about $2^\circ$ (spec 4.9c).")
        self.play(FadeIn(res))
        self.wait(4.0)
        self.clear()

    def real_numbers(self):
        st = self.st
        H = 3.3
        pf = PixelFrame(GR.RO_SHAPE, H, np.array([-4.4, 0.35, 0]))
        im = image(od_view(st["od"]), H, "grey", vmin=0, vmax=1).move_to(pf.centre)
        fr = SurroundingRectangle(im, buff=0, color=MUTED, stroke_width=1)
        tracks = VGroup()
        for xy, ks in GR.crossing_tracks():
            for i in range(len(xy) - 1):
                tracks.add(Line(pf.point(*xy[i]), pf.point(*xy[i + 1]), stroke_color=orientation_color(N.THETAS[ks[i]]),
                                stroke_width=4))
        ring = Circle(radius=0.45, stroke_color=JUNC, stroke_width=3).move_to(pf.point(*GR.RO_CENTRE))
        tt = Tex(r"traced per orientation (toy):\\two tracks through the crossing").scale(0.48).next_to(im, UP, buff=0.15)
        self.say(r"Traced per orientation, the crossing keeps both tracks, so the junction candidate there sees two lines.")
        self.play(FadeIn(im), Create(fr), FadeIn(tt), run_time=0.8)
        self.play(Create(tracks), run_time=2.5, rate_func=linear)
        self.play(Create(ring))
        self.wait(2.5)

        cells = [Tex(""), Tex(r"old tracker"), Tex(r"per orientation", color=GREEN_C),
                 Tex(r"centreline"), Tex(r"54,764 px"), Tex(r"71,475 px", color=GREEN_C),
                 Tex(r"mask"), Tex(r"23 pieces"), Tex(r"15 pieces", color=GREEN_C)]
        tbl = VGroup(*[c.scale(0.56) for c in cells]).arrange_in_grid(rows=3, cols=3, buff=(0.45, 0.2), col_alignments="lrr")
        head = Tex(r"measured on the real test image (spec 4.9c)", color=GREY_B).scale(0.5)
        more = VGroup(Tex(r"94\% of junction candidates carry two tracks", color=JUNC),
                      Tex(r"1\% of plain vessel points are doubled"),
                      Tex(r"open: small fans at some line ends,", color=GREY_B),
                      Tex(r"lines 2--3 px thick in places", color=GREY_B)
                      ).scale(0.5).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        blk = VGroup(head, tbl, more).arrange(DOWN, buff=0.3).move_to([2.6, 0.45, 0])
        self.say(r"On the real test image: 71,475 centreline pixels instead of 54,764, and the mask in 15 pieces instead of 23.")
        self.play(FadeIn(head), FadeIn(tbl), run_time=1.0)
        self.wait(2.5)
        self.say(r"Crossings now carry two tracks: 94\% of junction candidates do. The price: 1\% of plain vessel points doubled.")
        self.play(FadeIn(more, lag_ratio=0.2), run_time=1.2)
        self.wait(4.0)


variants(Sheets, module_globals=globals())
