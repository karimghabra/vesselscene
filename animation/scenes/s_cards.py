"""End cards for the short, medium and detailed cuts (the cold open carries the title).

The detailed cut's card rolls the spec's references (appendix B).
"""
from __future__ import annotations

import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_anim.style import FB, MUTED  # noqa: E402

REFERENCES = [
    r"Adelson and Anandan (1990). Ordinal characteristics of transparency.",
    r"Adelson and Bergen (1985). Spatiotemporal energy models for the perception of motion. \emph{JOSA A}.",
    r"Bosking, Zhang, Schofield and Fitzpatrick (1997). Horizontal connections in tree shrew striate cortex. \emph{J. Neurosci.}",
    r"Carandini and Ferster (2000). Membrane potential and firing rate in cat V1. \emph{J. Neurosci.}",
    r"Carandini and Heeger (2012). Normalization as a canonical neural computation. \emph{Nat. Rev. Neurosci.}",
    r"Craft, Sch\"utze, Niebur and von der Heydt (2007). A neural model of figure-ground organization. \emph{J. Neurophysiol.}",
    r"Desimone and Duncan (1995). Neural mechanisms of selective visual attention. \emph{Annu. Rev. Neurosci.}",
    r"Dobbins, Zucker and Cynader (1987). Endstopped neurons as a substrate for calculating curvature. \emph{Nature}.",
    r"Field, Hayes and Hess (1993). Contour integration: a local ``association field''. \emph{Vision Res.}",
    r"Heeger (1992). Normalization of cell responses in cat striate cortex. \emph{Vis. Neurosci.}",
    r"Heitger, Rosenthaler, von der Heydt, Peterhans and K\"ubler (1992). From simple to end-stopped cells. \emph{Vision Res.}",
    r"Ito and Komatsu (2004). Representation of angles within contour stimuli in V2. \emph{J. Neurosci.}",
    r"Jolicoeur, Ullman and Mackay (1986). Curve tracing. \emph{Memory and Cognition}.",
    r"Kaipio and Somersalo (2007). Statistical inverse problems: inverse crimes. \emph{J. Comput. Appl. Math.}",
    r"Lamme and Roelfsema (2000). The distinct modes of vision offered by feedforward and recurrent processing. \emph{TINS}.",
    r"Lee, Mumford, Romero and Lamme (1998). The role of the primary visual cortex in higher level vision. \emph{Vision Res.}",
    r"Linsley, Kim, Veerabadran, Windolf and Serre (2018). Horizontal gated recurrent units. \emph{NeurIPS}.",
    r"Metelli (1974). The perception of transparency. \emph{Scientific American}.",
    r"Parent and Zucker (1989). Trace inference, curvature consistency, and curve detection. \emph{IEEE PAMI}.",
    r"Pasupathy and Connor (2001). Shape representation in area V4. \emph{J. Neurophysiol.}",
    r"Pouget, Dayan and Zemel (2000). Information processing with population codes. \emph{Nat. Rev. Neurosci.}",
    r"Rao and Ballard (1999). Predictive coding in the visual cortex. \emph{Nat. Neurosci.}",
    r"Roelfsema (2006). Cortical algorithms for perceptual grouping. \emph{Annu. Rev. Neurosci.}",
    r"Roelfsema, Lamme and Spekreijse (1998). Object-based attention in V1 of the macaque. \emph{Nature}.",
    r"Zhou, Friedman and von der Heydt (2000). Coding of border ownership in monkey visual cortex. \emph{J. Neurosci.}",
]


def _line(text, scale, color, max_width=13.0):
    """One line of text that LaTeX may not wrap."""
    t = Tex(r"\mbox{" + text + "}", color=color).scale(scale)
    return t.scale_to_fit_width(max_width) if t.width > max_width else t


class _End(Scene):
    LEVEL = 1

    def construct(self):
        title = Tex(r"Recurrent mid-level vision for the vessel annotator").scale(0.9)
        rule = Line(LEFT * 2.2, RIGHT * 2.2, stroke_color=FB, stroke_width=2)
        lines = VGroup(
            _line(r"From the technical spec, revision 2 (11 October 2026)", 0.6, GREY_A),
            _line(r"Every image, map and curve is computed from the spec's equations on toy stimuli;", 0.5, GREY_B),
            _line(r"numbers marked \emph{measured} are quoted from the spec.", 0.5, GREY_B),
            _line(r"Vessel imagery and its truth: vesselscene (synthetic conjunctival stills and bursts).", 0.5, GREY_B),
            _line(r"Animated with Manim Community.", 0.5, GREY_B),
        ).arrange(DOWN, buff=0.22)
        g = VGroup(title, rule, lines).arrange(DOWN, buff=0.35)
        if self.LEVEL < 3:
            self.play(Write(title), GrowFromCenter(rule), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in lines], lag_ratio=0.25), run_time=1.6)
            self.wait(3.0)
            self.play(FadeOut(g), run_time=1.0)
            return
        # detailed: the credits, then the references roll
        self.play(Write(title), GrowFromCenter(rule), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in lines], lag_ratio=0.25), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(g), run_time=0.8)
        head = Tex(r"References", color=FB).scale(0.8).to_edge(UP, buff=0.4)
        refs = VGroup(*[_line(r, 0.42, GREY_A) for r in REFERENCES]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        refs.next_to(head, DOWN, buff=0.5).align_to(LEFT * 6.3, LEFT)
        self.play(FadeIn(head))
        shift = refs.get_top()[1] - refs.get_bottom()[1] + 0.5
        refs.shift(DOWN * 4.5)
        mask_top = Rectangle(width=16, height=1.6, fill_color=BLACK, fill_opacity=1, stroke_width=0).to_edge(UP, buff=0)
        self.add(refs, mask_top, head)
        self.play(refs.animate.shift(UP * (shift + 1.0)), run_time=18, rate_func=linear)
        self.play(FadeOut(head), run_time=0.6)


class EndShort(_End):
    LEVEL = 1


class EndMedium(_End):
    LEVEL = 2


class EndDetailed(_End):
    LEVEL = 3
