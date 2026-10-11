"""Original music for the animation: composed by rule, played by FluidSynth.

Each cue is written for one chapter, to its exact length:

  harmony   a mood's key, tempo and looping diatonic progression; the bar
            before the last is the dominant, the last the tonic, held, so
            every cue cadences home as its chapter ends
  piano     bass on the downbeat, rolling eighth-note arpeggios, sustain
            pedal per chord
  strings   the chord held, with slow swells (expression)
  melody    phrases of four bars: chord tones on strong beats, steps
            between; every other phrase rests, to leave room for the picture
  celesta   a few high chord tones at the brightest moments
  arc       an intensity curve over the cue decides which layers play

The 'wonder' mood opens its melody with the video's motif (a rising
5-6-8, answered 7-5-2 over the dominant).  Instruments: FluidR3_GM
(MIT licence), the General MIDI soundfont Debian ships as fluid-soundfont-gm.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
RATE = 44100
TPB = 480                                     # ticks per beat
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}

# channel: (GM program, reverb send, pan)
PIANO_MEL, PIANO_ACC, STRINGS, CELLO, CELESTA, PIZZ = 0, 1, 2, 3, 4, 5
CHANNELS = {PIANO_MEL: (0, 70, 64), PIANO_ACC: (0, 70, 60), STRINGS: (49, 100, 70),
            CELLO: (42, 90, 52), CELESTA: (8, 110, 84), PIZZ: (45, 80, 72)}


@dataclass
class Mood:
    tonic: int                    # MIDI note of the tonic, in octave 4
    mode: str
    bpm: float
    prog: list                    # [(degree 1-7, extension, bass degree or None)]
    pattern: str = "roll"         # arpeggio pattern
    pizz: bool = False
    cadence: int = 5              # degree of the penultimate chord (5: authentic, 4: plagal)
    motif: bool = False


MOODS = {
    # D major, slow: I  V/3  vi7  IV    (the video's own key)
    "wonder": Mood(62, "major", 68, [(1, "add9", None), (5, "", 7), (6, "7", None), (4, "maj7", None)], motif=True),
    # C major: vi7  IVmaj7  Imaj7  V
    "physics": Mood(60, "major", 74, [(6, "7", None), (4, "maj7", None), (1, "maj7", None), (5, "", None)]),
    # E minor: i(add9)  VImaj7  III  VII
    "cells": Mood(64, "minor", 70, [(1, "add9", None), (6, "maj7", None), (3, "", None), (7, "", None)], pattern="wave"),
    # F major: I  iii  IV  V
    "geometry": Mood(65, "major", 78, [(1, "add9", None), (3, "7", None), (4, "maj7", None), (5, "", None)], pattern="climb"),
    # A major: I  vi7  ii7  V
    "grouping": Mood(57, "major", 72, [(1, "", None), (6, "7", None), (2, "7", None), (5, "", None)], pattern="wave"),
    # B minor, moving: i  VI  III  VII, with pizzicato
    "motion": Mood(59, "minor", 92, [(1, "", None), (6, "", None), (3, "", None), (7, "", None)], pattern="pulse", pizz=True),
    # E-flat major, slowest: IVmaj7  I  V  vi, plagal close
    "reflection": Mood(63, "major", 62, [(4, "maj7", None), (1, "", None), (5, "", None), (6, "7", None)], cadence=4),
}

PATTERNS = {                                  # indices into the chord's arpeggio pool, one per eighth; None = rest
    "roll": [0, 2, 3, 4, 5, 4, 3, 2],
    "wave": [0, 3, 2, 4, 3, 5, 4, 2],
    "climb": [0, 1, 2, 3, 4, 5, 6, 4],
    "pulse": [0, 3, 0, 4, 0, 3, 0, 5],
    "sparse": [0, None, 3, None, 4, None, 3, None],
}
MELODY_RHYTHMS = [[4, 2, 2], [2, 2, 4], [6, 2], [3, 1, 4], [8], [2, 2, 2, 2], [-2, 2, 4], [4, -2, 2]]


def degree_pitch(m: Mood, degree: int) -> int:
    """Semitones above the tonic of a 1-based scale degree (any octave)."""
    sc = SCALES[m.mode]
    d = degree - 1
    return sc[d % 7] + 12 * (d // 7)


def chord(m: Mood, degree: int, ext: str) -> list[int]:
    """The chord's pitch classes above the tonic, root first."""
    steps = [0, 2, 4] + ([6] if ext in ("7", "maj7") else []) + ([8] if ext == "add9" else [])
    return [degree_pitch(m, degree + s) for s in steps]


@dataclass
class Score:
    bpm: float
    events: list = field(default_factory=list)       # (tick, channel, kind, a, b)

    def note(self, ch, pitch, t0, dur, vel):
        self.events.append((int(t0), ch, "on", int(pitch), int(np.clip(vel, 1, 127))))
        self.events.append((int(t0 + dur), ch, "off", int(pitch), 0))

    def cc(self, ch, t, num, val):
        self.events.append((int(t), ch, "cc", num, int(np.clip(val, 0, 127))))

    def write(self, path):
        mf = mido.MidiFile(ticks_per_beat=TPB)
        tr = mido.MidiTrack()
        mf.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.bpm), time=0))
        head = []
        for ch, (prog, rev, pan) in CHANNELS.items():
            head += [mido.Message("program_change", channel=ch, program=prog, time=0),
                     mido.Message("control_change", channel=ch, control=91, value=rev, time=0),
                     mido.Message("control_change", channel=ch, control=93, value=0, time=0),
                     mido.Message("control_change", channel=ch, control=10, value=pan, time=0),
                     mido.Message("control_change", channel=ch, control=7, value=100, time=0)]
        tr.extend(head)
        order = {"off": 0, "cc": 1, "on": 2}
        last = 0
        for t, ch, kind, a, b in sorted(self.events, key=lambda e: (e[0], order[e[2]])):
            t = max(t, 0)
            dt = t - last
            last = t
            if kind == "on":
                tr.append(mido.Message("note_on", channel=ch, note=a, velocity=b, time=dt))
            elif kind == "off":
                tr.append(mido.Message("note_off", channel=ch, note=a, velocity=0, time=dt))
            else:
                tr.append(mido.Message("control_change", channel=ch, control=a, value=b, time=dt))
        mf.save(path)


def intensity_curve(n_bars: int, keys=None):
    """Intensity per bar in [0, 1].  keys: [(fraction of the cue, level)];
    the default rises to a crest at 60 % and settles for the cadence."""
    keys = keys or [(0.0, 0.35), (0.25, 0.6), (0.6, 0.8), (0.85, 0.55), (1.0, 0.35)]
    xs, ys = zip(*keys)
    return np.interp((np.arange(n_bars) + 0.5) / n_bars, xs, ys)


def compose(mood: str, seconds: float, seed: int = 0, keys=None, min_hold: float = 2.6) -> tuple[Score, float]:
    """A cue of the given mood lasting `seconds`.  Returns the score and the
    time (s) at which the final chord starts."""
    m = MOODS[mood]
    rng = np.random.default_rng(seed)
    beat = 60.0 / m.bpm
    bar_s = 4 * beat
    n_bars = max(2, int((seconds - min_hold) // bar_s))
    hold = seconds - n_bars * bar_s                          # the final chord fills the rest
    level = intensity_curve(n_bars, keys)
    sc = Score(m.bpm)
    bar_t = 4 * TPB
    eighth = TPB // 2
    jitter = lambda: rng.integers(-8, 9)

    prog = [m.prog[i % len(m.prog)] for i in range(n_bars)]
    if n_bars >= 2:
        prog[-1] = (m.cadence, "", None)                     # the approach to the final tonic
    prev_mel = m.tonic + 12 + degree_pitch(m, 5)
    motif_done = False
    for b, (deg, ext, bass_deg) in enumerate(prog):
        t0 = b * bar_t
        L = level[b]
        tones = chord(m, deg, ext)
        root = m.tonic + tones[0]
        # bass: the root (or the inversion's bass) in octave 2-3, held through the bar
        bass = m.tonic - 24 + (degree_pitch(m, bass_deg) if bass_deg else tones[0])
        while bass < 36:
            bass += 12
        sc.cc(PIANO_ACC, t0 - 6, 64, 0)
        sc.cc(PIANO_ACC, t0 + 4, 64, 127)
        sc.note(PIANO_ACC, bass, t0 + jitter() + 6, bar_t - 30, 46 + 14 * L + rng.integers(-3, 4))
        # arpeggio pool: chord tones from about G3 upward
        pool = sorted({m.tonic - 12 + p + 12 * o for p in tones for o in range(3)})
        pool = [p for p in pool if 55 <= p <= 81][:8]
        pat = PATTERNS["sparse" if L < 0.3 else m.pattern]
        for i, idx in enumerate(pat):
            if idx is None or idx >= len(pool):
                continue
            accent = 6 if i in (0, 4) else 0
            sc.note(PIANO_ACC, pool[idx], t0 + i * eighth + jitter(), int(eighth * 1.6), 42 + 14 * L + accent + rng.integers(-4, 5))
        # strings: from a level of 0.4, the chord held with a swell
        if L >= 0.4:
            voicing = sorted({m.tonic - 12 + p for p in tones[:3]} | {m.tonic + tones[1]})
            for p in voicing:
                sc.note(STRINGS, p, t0 + 10, bar_t + 20, 52 + 20 * L)
            for k, e in enumerate([70, 88, 100, 92, 80]):
                sc.cc(STRINGS, t0 + k * bar_t // 5, 11, e * (0.75 + 0.35 * L))
            if L >= 0.65:
                sc.note(CELLO, bass if bass >= 43 else bass + 12, t0 + 12, bar_t + 10, 48 + 20 * L)
        if m.pizz and L >= 0.45:
            for i in (0, 2, 4, 6):
                sc.note(PIZZ, pool[(i // 2) % len(pool)] - 12, t0 + i * eighth + jitter(), eighth, 50 + 15 * L)
        # melody: four-bar phrases, every other phrase sung when the level allows
        phrase, pos = divmod(b, 4)
        if L >= 0.55 and phrase % 2 == (0 if mood in ("wonder", "reflection") else 1) and b < n_bars - 1:
            if m.motif and not motif_done and pos in (0, 1) and deg in (1, 5):
                # the motif: 5 6 8 over the tonic, answered 7 5 2 over the dominant
                degs, rhy = ([5, 6, 8], [2, 2, 4]) if pos == 0 else ([7, 5, 2], [2, 2, 4])
                tt = t0
                for dg, r in zip(degs, rhy):
                    p = m.tonic + 12 + degree_pitch(m, dg)
                    sc.note(PIANO_MEL, p, tt + jitter(), r * eighth - 20, 64 + 10 * L)
                    tt += r * eighth
                    prev_mel = p
                motif_done = pos == 1
            else:
                rhythm = MELODY_RHYTHMS[rng.integers(len(MELODY_RHYTHMS))] if pos < 3 else [6, -2]
                chord_pcs = {(m.tonic + p) % 12 for p in tones}
                scale_pcs = {(m.tonic + p) % 12 for p in SCALES[m.mode]}
                tt = t0
                for r in rhythm:
                    if r < 0:
                        tt += -r * eighth
                        continue
                    strong = (tt - t0) % (2 * TPB) == 0
                    pcs = chord_pcs if strong or pos == 3 else scale_pcs
                    cands = [p for p in range(67, 88) if p % 12 in pcs and abs(p - prev_mel) <= (5 if strong else 2) and p != prev_mel]
                    if not cands:
                        cands = [p for p in range(67, 88) if p % 12 in chord_pcs]
                    w = np.array([1.0 / (1 + abs(p - prev_mel)) for p in cands])
                    p = int(rng.choice(cands, p=w / w.sum()))
                    sc.note(PIANO_MEL, p, tt + jitter(), r * eighth - 20, 58 + 14 * L + rng.integers(-4, 5))
                    prev_mel = p
                    tt += r * eighth
        # celesta: a sparkle on the crest
        if L >= 0.75 and pos == 2:
            p = m.tonic + 12 + tones[rng.integers(len(tones))]
            while p > 88:
                p -= 12
            sc.note(CELESTA, p, t0 + 2 * TPB + jitter(), 2 * TPB, 44)

    # the final chord: tonic, rolled on the piano, held in the strings, to the end
    tf = n_bars * bar_t
    hold_t = int(hold / beat * TPB)
    tones = chord(m, 1, "add9" if m.mode == "major" else "")
    sc.cc(PIANO_ACC, tf - 6, 64, 0)
    sc.cc(PIANO_ACC, tf + 4, 64, 127)
    low = m.tonic - 24 + tones[0]
    low += 12 if low < 36 else 0
    final = [low, m.tonic - 12 + tones[0], m.tonic - 12 + tones[2], m.tonic + tones[1], m.tonic + tones[0] + 12]
    for i, p in enumerate(final):
        sc.note(PIANO_ACC, p, tf + i * 50, hold_t, 52 - 2 * i)
    for p in (m.tonic - 12 + tones[0], m.tonic - 12 + tones[2], m.tonic + tones[1]):
        sc.note(STRINGS, p, tf + 10, hold_t, 52)
    for k in range(6):                                       # the strings fade with the chord
        sc.cc(STRINGS, tf + k * hold_t // 6, 11, 90 - 13 * k)
    sc.cc(PIANO_ACC, tf + hold_t, 64, 0)
    return sc, n_bars * bar_s


def render(score: Score, seconds: float, tail: float = 2.0, fade_in: float = 0.6, fade_out: float = 1.2) -> np.ndarray:
    """Play a score through FluidSynth: float32 stereo at RATE, `seconds`
    long plus a faded reverb tail."""
    with tempfile.TemporaryDirectory() as d:
        mid, wav = Path(d) / "cue.mid", Path(d) / "cue.wav"
        score.write(mid)
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.45", "-r", str(RATE), "-R", "1", "-C", "0",
                        "-o", "synth.reverb.room-size=0.75", "-o", "synth.reverb.damp=0.35",
                        "-o", "synth.reverb.width=0.9", "-o", "synth.reverb.level=0.7",
                        "-F", str(wav), SF2, str(mid)], check=True, capture_output=True)
        rate, x = wavfile.read(wav)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    n = int((seconds + tail) * RATE)
    x = np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n]
    t = np.arange(n) / RATE
    env = np.clip(t / fade_in, 0, 1) * np.clip((seconds + tail - t) / (fade_out + tail), 0, 1)
    return x * env[:, None]
