"""The base scene every chapter derives from.

A chapter is written once, with three depths:

  LEVEL 1  short     one idea, about 20-40 s; used in the short cut
  LEVEL 2  medium    the idea and its equation, about 60-100 s
  LEVEL 3  detailed  everything, with the spec's numbers, about 2-3 min

and the variants are subclasses that only set LEVEL:

    class EndStopped(ChapterScene):
        NUMBER, TITLE, SPEC, MAP_KEYS = 8, "Where lines end", "spec 4.2", ("endstop",)
        def construct(self):
            self.intro()
            ...                      # always shown
            if self.at(2): ...       # medium and detailed
            if self.at(3): ...       # detailed only
            self.outro()

    class EndStoppedShort(EndStopped): LEVEL = 1
    class EndStoppedMedium(EndStopped): LEVEL = 2
    class EndStoppedDetailed(EndStopped): LEVEL = 3
"""
from __future__ import annotations

import re

from manim import (DOWN, GREY_A, GREY_B, LEFT, RIGHT, UP, FadeIn, FadeOut, Line, Scene, Tex,
                   VGroup, Write)

from .style import FB, MUTED

CAPTION_SCALE = 0.56
CAPTION_Y_TOP = -3.05      # keep chapter content above this line; captions live below it
WORDS_PER_SECOND = 3.2     # reading speed used by reading_time()


def reading_time(text: str) -> float:
    words = len(re.sub(r"\$[^$]*\$", " x ", text).split())
    return words / WORDS_PER_SECOND


def caption_mobject(text: str):
    return Tex(text, color=GREY_A).scale(CAPTION_SCALE).to_edge(DOWN, buff=0.3)


class ChapterScene(Scene):
    LEVEL = 3
    NUMBER: int | None = None
    TITLE = ""
    SPEC = ""                 # e.g. "spec 4.2"
    MAP_KEYS: tuple = ()      # boxes of the architecture map this chapter explains

    def setup(self):
        self._cap = None

    # ------------------------------------------------------------------ levels
    def at(self, level: int) -> bool:
        return self.LEVEL >= level

    @property
    def short(self) -> bool:
        return self.LEVEL == 1

    # ---------------------------------------------------------------- captions
    def say(self, text: str, wait: float | str = 0.0):
        """Replace the caption.  wait='read' holds it for its reading time
        (minus the 0.7 s the swap takes); a number waits that long."""
        new = caption_mobject(text)
        anims = [FadeIn(new, shift=UP * 0.15)]
        if self._cap is not None:
            anims.insert(0, FadeOut(self._cap, shift=UP * 0.15))
        self.play(*anims, run_time=0.7)
        self._cap = new
        if wait == "read":
            wait = max(reading_time(text) - 0.7, 0.5)
        if wait:
            self.wait(wait)

    def unsay(self):
        if self._cap is not None:
            self.play(FadeOut(self._cap), run_time=0.5)
            self._cap = None

    # ------------------------------------------------------------ intro, outro
    def intro(self):
        """Chapter card.  Detailed: the architecture map with this chapter's
        boxes lit, then the card.  Medium: the card.  Short: a brief card."""
        if self.LEVEL >= 3 and self.MAP_KEYS:
            from .archmap import ArchitectureMap
            m = ArchitectureMap().scale(0.92)
            self.play(FadeIn(m), run_time=0.8)
            self.play(m.highlight(self.MAP_KEYS), run_time=1.0)
            self.wait(1.4)
            self.play(FadeOut(m), run_time=0.6)
        if self.NUMBER is None and not self.TITLE:
            return
        title = Tex(self.TITLE).scale(1.1 if self.LEVEL > 1 else 0.95)
        parts = [title]
        if self.LEVEL > 2 and self.NUMBER is not None:      # numbers only in the detailed cut, which has every chapter
            num = Tex(f"{self.NUMBER}", color=MUTED).scale(1.6)
            parts.insert(0, num)
        g = VGroup(*parts).arrange(DOWN, buff=0.3)
        rule = Line(LEFT * 1.6, RIGHT * 1.6, stroke_color=FB, stroke_width=2).next_to(g, DOWN, buff=0.25)
        g.add(rule)
        if self.SPEC and self.LEVEL > 1:
            g.add(Tex(self.SPEC, color=GREY_B).scale(0.55).next_to(rule, DOWN, buff=0.18))
        self.play(Write(title), *[FadeIn(p) for p in g if p is not title], run_time=1.1)
        self.wait(1.0 if self.LEVEL > 1 else 0.5)
        self.play(FadeOut(g, shift=UP * 0.3), run_time=0.6)

    def clear(self, run_time: float = 0.8, keep_caption: bool = True):
        """Fade out everything on screen (but the caption)."""
        ms = [m for m in self.mobjects if not (keep_caption and m is self._cap)]
        if ms:
            self.play(*[FadeOut(m) for m in ms], run_time=run_time)

    def outro(self):
        self._cap = None
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.2)


def variants(cls, name: str | None = None, module_globals: dict | None = None):
    """Make <Name>Short, <Name>Medium and <Name>Detailed subclasses of cls in
    the calling module, so `manim file.py EndStoppedMedium` works."""
    name = name or cls.__name__
    out = {}
    for lv, suffix in ((1, "Short"), (2, "Medium"), (3, "Detailed")):
        out[name + suffix] = type(name + suffix, (cls,), {"LEVEL": lv, "__module__": cls.__module__})
    if module_globals is not None:
        module_globals.update(out)
    return out
