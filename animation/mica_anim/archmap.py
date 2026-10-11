"""The spec's architecture (section 4.1) as a map the video keeps returning to.

Boxes are keyed so chapters can light their own:

  feedforward: retina, ganglion, simple, arc, surround, scale
  recurrent:   assoc; endstop, junction, contin, transp, compare (V2 stack);
               border, fusion, readout; attend; graph, render, popcode, fit
  motion:      motion
"""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, WHITE, AnimationGroup, Arrow, CurvedArrow, GREY_B,
                   RoundedRectangle, Tex, VGroup)

from .style import FB, FF, MUTED

# key: (label, centre (x, y), width)
BOXES = {
    "retina":   (r"1 retina: OD",            (-5.55, 3.05), 1.95),
    "ganglion": (r"2 ganglion cells",        (-3.35, 3.05), 1.95),
    "simple":   (r"3 simple cells",          (-1.15, 3.05), 1.95),
    "arc":      (r"3b arc cells",            (1.05, 3.05), 1.95),
    "surround": (r"4 surround",              (3.25, 3.05), 1.95),
    "scale":    (r"4b scale selection",      (5.45, 3.05), 1.95),
    "assoc":    (r"5$'$ association field",  (0.0, 1.75), 3.3),
    "endstop":  (r"V2a end-stopped",         (-4.3, 0.75), 2.3),
    "junction": (r"V2b junction cells",      (-4.3, -0.15), 2.3),
    "contin":   (r"V2c continuation",        (-4.3, -1.05), 2.3),
    "transp":   (r"V2d transparency",        (-4.3, -1.95), 2.3),
    "compare":  (r"V2e model comparison",    (-4.3, -2.85), 2.3),
    "border":   (r"border ownership",        (4.3, 0.75), 2.5),
    "fusion":   (r"evidence fusion",         (4.3, -0.15), 2.5),
    "readout":  (r"readout per orientation", (4.3, -1.05), 2.5),
    "attend":   (r"attentive grouping",      (0.0, -0.45), 2.4),
    "graph":    (r"graph: typed junctions",  (0.0, -1.5), 2.4),
    "render":   (r"render",                  (1.45, -2.55), 1.5),
    "popcode":  (r"population codes",        (-1.45, -2.55), 1.9),
    "fit":      (r"spline fit",              (-1.45, -3.45), 1.9),
    "motion":   (r"motion (bursts)",         (4.3, -2.85), 2.3),
}
FEEDFORWARD = ("retina", "ganglion", "simple", "arc", "surround", "scale")
EDGES = [("retina", "ganglion"), ("ganglion", "simple"), ("simple", "arc"), ("arc", "surround"),
         ("surround", "scale"), ("scale", "assoc"), ("assoc", "endstop"), ("assoc", "border"),
         ("endstop", "junction"), ("junction", "contin"), ("contin", "transp"), ("transp", "compare"),
         ("border", "fusion"), ("fusion", "readout"), ("contin", "attend"), ("readout", "attend"),
         ("attend", "graph"), ("graph", "render"), ("graph", "popcode"), ("popcode", "fit")]
FEEDBACK = [("render", "assoc"), ("fit", "assoc")]


class ArchitectureMap(VGroup):
    def __init__(self, label_scale=0.42, **kw):
        super().__init__(**kw)
        self.boxes = {}
        for key, (label, (x, y), w) in BOXES.items():
            rect = RoundedRectangle(width=w, height=0.58, corner_radius=0.1, stroke_color=FF if key in FEEDFORWARD else WHITE,
                                    stroke_width=1.6, fill_color=FF if key in FEEDFORWARD else WHITE, fill_opacity=0.06)
            txt = Tex(label).scale(label_scale)
            if txt.width > w - 0.15:
                txt.scale_to_fit_width(w - 0.15)
            box = VGroup(rect, txt).move_to(np.array([x, y, 0.0]))
            self.boxes[key] = box
        self.arrows = VGroup()
        for a, b in EDGES:
            self.arrows.add(self._arrow(a, b))
        self.feedback = VGroup(
            CurvedArrow(self.boxes["render"].get_right() + RIGHT * 0.02, self.boxes["assoc"].get_right() + DOWN * 0.1,
                        angle=1.2, color=FB, stroke_width=2.5, tip_length=0.15),
            CurvedArrow(self.boxes["fit"].get_left() + LEFT * 0.02, self.boxes["assoc"].get_left() + DOWN * 0.1,
                        angle=-1.35, color=FB, stroke_width=2.5, tip_length=0.15),
        )
        self.labels = VGroup(
            Tex(r"FEEDFORWARD SWEEP", color=FF).scale(0.42).move_to([-5.55, 3.62, 0], aligned_edge=LEFT).shift(LEFT * 0.95),
            Tex(r"RECURRENT PROCESSING", color=GREY_B).scale(0.42).move_to([-6.55, 2.25, 0], aligned_edge=LEFT),
            Tex(r"top-down feedback", color=FB).scale(0.42).move_to([2.55, -3.35, 0]),
        )
        self.add(self.arrows, self.feedback, *self.boxes.values(), self.labels)

    def _arrow(self, a, b):
        A, B = self.boxes[a], self.boxes[b]
        pa, pb = A.get_center(), B.get_center()
        d = pb - pa
        if abs(d[1]) < 0.05:                               # same row
            start, end = (A.get_right(), B.get_left()) if d[0] > 0 else (A.get_left(), B.get_right())
        elif abs(d[0]) < 0.05:                             # same column
            start, end = (A.get_bottom(), B.get_top()) if d[1] < 0 else (A.get_top(), B.get_bottom())
        elif a == "scale":                                 # the long drop into the recurrent phase
            start, end = A.get_bottom(), B.get_right()
        elif abs(d[0]) > 1.5 * abs(d[1]):                  # mostly sideways: facing sides
            start, end = (A.get_right(), B.get_left()) if d[0] > 0 else (A.get_left(), B.get_right())
        else:                                              # mostly down: bottom to top
            start, end = A.get_bottom(), B.get_top()
        return Arrow(start, end, buff=0.04, stroke_width=2, max_tip_length_to_length_ratio=0.25,
                     tip_length=0.12, color=MUTED)

    def highlight(self, keys, color=FB, dim=0.25):
        """Light the given boxes and dim the rest ("feedback" keeps the gold loops lit)."""
        keys = set(keys)
        anims = []
        for k, box in self.boxes.items():
            if k in keys:
                anims.append(box[0].animate.set_stroke(color, width=3.5).set_fill(color, opacity=0.22))
            else:
                anims.append(box[0].animate.set_stroke(opacity=dim).set_fill(opacity=0.02))
                anims.append(box[1].animate.set_opacity(dim))
        anims.append(self.arrows.animate.set_opacity(dim))
        if not keys & {"feedback", "render", "fit"}:
            anims.append(self.feedback.animate.set_opacity(dim))
        return AnimationGroup(*anims)
