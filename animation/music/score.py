"""The soundtrack of each cut: one composed cue per chapter, joined, and laid
under the rendered video.

    python music/score.py short            # media/cuts/short_music.wav and short_music.mp4
    python music/score.py all

Each cue is composed to its chapter's exact rendered length and cadences
on its tonic as the chapter ends; the next cue fades in over its predecessor's
reverb tail.  Moods are assigned so that neighbouring chapters contrast.
The mix is high-passed at 60 Hz and normalised to -18 LUFS (true peak
-1.5 dBTP).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import versions as V  # noqa: E402
from music.compose import RATE, compose, render  # noqa: E402

CHAPTER_MOOD = {0: "wonder", 1: "physics", 2: "wonder", 3: "reflection", 4: "cells", 5: "geometry", 6: "grouping",
                7: "reflection", 8: "geometry", 9: "physics", 10: "grouping", 11: "cells", 12: "geometry",
                13: "reflection", 14: "grouping", 15: "cells", 16: "physics", 17: "motion", 18: "wonder"}
TRAILER_ARC = [(0.0, 0.42), (0.19, 0.5), (0.3, 0.62), (0.42, 0.66), (0.7, 0.72), (0.83, 0.85), (0.93, 0.9), (1.0, 0.5)]
LUFS, TRUE_PEAK = -18.0, -1.5


def duration(path: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(path)]))


def cues(cut: str) -> list[tuple[str, float, str, int, list | None]]:
    """(scene, seconds, mood, seed, arc) for each scene of a cut, in order."""
    out = []
    names = {base: n for n, (_, base) in V.CHAPTERS.items()}
    for file, scene in V.scenes(cut):
        sec = duration(HERE / "media" / "videos" / Path(file).stem / "1080p30" / f"{scene}.mp4")
        if scene == "Trailer":
            out.append((scene, sec, "wonder", 99, TRAILER_ARC))
        elif scene.startswith("End"):
            out.append((scene, sec, "reflection", 7, [(0, 0.45), (0.5, 0.6), (1, 0.4)]))
        else:
            base = next(b for b in names if scene.startswith(b) and scene[len(b):] in V.LEVEL_SUFFIX.values())
            n = names[base]
            lvl = next(k for k, s in V.LEVEL_SUFFIX.items() if scene.endswith(s))
            out.append((scene, sec, CHAPTER_MOOD[n], 100 * n + lvl, None))
    return out


def loudnorm(src: Path, dst: Path):
    """Two-pass EBU R128 normalisation to LUFS / TRUE_PEAK."""
    flt = f"highpass=f=60,loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA=11"     # no sub-bass rumble under the picture
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-af", flt + ":print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    m = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
    flt += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
            f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", flt, "-ar", str(RATE), str(dst)], check=True)


def soundtrack(cut: str) -> Path:
    plan = cues(cut)
    total = sum(s for _, s, *_ in plan)
    mix = np.zeros((int((total + 3.0) * RATE), 2), np.float32)
    t = 0.0
    for scene, sec, mood, seed, arc in plan:
        score, _ = compose(mood, sec, seed=seed, keys=arc)
        x = render(score, sec)
        x /= max(np.sqrt(np.mean(x[x != 0] ** 2)), 1e-6) * 8.0          # every cue at about the same level
        i = int(round(t * RATE))
        mix[i:i + len(x)] += x[: len(mix) - i]
        print(f"  {scene:28s} {sec:6.1f} s  {mood}", flush=True)
        t += sec
    mix = mix[: int(total * RATE)]
    mix /= max(np.abs(mix).max(), 1e-6) / 0.9
    raw = HERE / "media" / "cuts" / f"{cut}_music_raw.wav"
    out = HERE / "media" / "cuts" / f"{cut}_music.wav"
    wavfile.write(raw, RATE, (mix * 32767).astype(np.int16))
    loudnorm(raw, out)
    raw.unlink()
    return out


def mux(cut: str, music: Path) -> Path:
    """The full-quality cut with its soundtrack (AAC 160 kb/s)."""
    video = HERE / "media" / "cuts" / f"{cut}.mp4"
    out = HERE / "media" / "cuts" / f"{cut}_music.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(music), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
    return out


def main():
    cuts = list(V.CUTS) if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]
    for cut in cuts:
        print(cut, flush=True)
        print(" ", mux(cut, soundtrack(cut)), flush=True)


if __name__ == "__main__":
    main()
