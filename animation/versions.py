"""The four cuts of the video, and how to render and assemble them.

    python versions.py list                      # chapters and scenes per cut
    python versions.py render short              # render the cut's scenes, then join them
    python versions.py render detailed --preview # 480p15, quick
    python versions.py render all -j 3           # every cut, 3 scenes at a time

Final quality is 1920x1080 at 30 fps.  Each chapter is a scene variant
(<Base>Short / Medium / Detailed); a cut is its title card, its chapters
in order, and the end card, joined with ffmpeg without re-encoding.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHAPTERS = {
    0: ("s00_cold_open.py", "ColdOpen"),
    1: ("s01_physics.py", "Physics"),
    2: ("s02_two_modes.py", "TwoModes"),
    3: ("s03_retina.py", "Retina"),
    4: ("s04_hypercolumn.py", "Hypercolumn"),
    5: ("s05_scale_selection.py", "ScaleSelection"),
    6: ("s06_arc_cells.py", "ArcCells"),
    7: ("s07_association_field.py", "AssociationField"),
    8: ("s08_end_stopped.py", "EndStopped"),
    9: ("s09_junction_cells.py", "JunctionCells"),
    10: ("s10_continuation.py", "Continuation"),
    11: ("s11_transparency.py", "Transparency"),
    12: ("s12_border_ownership.py", "BorderOwnership"),
    13: ("s13_sheets.py", "Sheets"),
    14: ("s14_attention.py", "AttentiveGrouping"),
    15: ("s15_feedback.py", "Feedback"),
    16: ("s16_population_codes.py", "PopulationCodes"),
    17: ("s17_motion.py", "Motion"),
    18: ("s18_whole_loop.py", "WholeLoop"),
}
LEVEL_SUFFIX = {1: "Short", 2: "Medium", 3: "Detailed"}
CUTS = {
    "very_short": dict(level=None, chapters=[]),
    "short": dict(level=1, chapters=[0, 1, 2, 4, 8, 9, 10, 11, 14, 18]),
    "medium": dict(level=2, chapters=[0, 1, 2, 4, 7, 8, 9, 10, 11, 14, 16, 18]),
    "detailed": dict(level=3, chapters=list(range(19))),
}
CARD_NAME = {"very_short": "VeryShort", "short": "Short", "medium": "Medium", "detailed": "Detailed"}


def scenes(cut: str) -> list[tuple[str, str]]:
    """(file, scene class) in playing order."""
    c = CUTS[cut]
    if c["level"] is None:
        return [("s99_trailer.py", "Trailer")]
    out = [("s_cards.py", f"Title{CARD_NAME[cut]}")]
    out += [(CHAPTERS[n][0], CHAPTERS[n][1] + LEVEL_SUFFIX[c["level"]]) for n in c["chapters"]]
    out.append(("s_cards.py", "EndCard"))
    return out


def render_one(file: str, scene: str, media: Path, preview: bool, manim: str) -> Path:
    q = ["-ql"] if preview else ["-r", "1920,1080", "--fps", "30"]
    cmd = [manim, *q, "--media_dir", str(media), "--disable_caching", str(HERE / "scenes" / file), scene]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError(f"{file} {scene} failed")
    folder = "480p15" if preview else "1080p30"
    return media / "videos" / Path(file).stem / folder / f"{scene}.mp4"


def render_cut(cut: str, preview: bool, jobs: int, manim: str) -> Path:
    media = HERE / "media"
    todo = scenes(cut)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        paths = list(ex.map(lambda fs: render_one(*fs, media, preview, manim), todo))
    out_dir = media / "cuts"
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / f"{cut}{'_preview' if preview else ''}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    out = out_dir / f"{cut}{'_preview' if preview else ''}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "render"])
    ap.add_argument("cut", nargs="?", default="all", choices=["all", *CUTS])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--manim", default=os.environ.get("MANIM", "manim"))
    a = ap.parse_args()
    cuts = list(CUTS) if a.cut == "all" else [a.cut]
    for cut in cuts:
        if a.action == "list":
            print(cut)
            for f, s in scenes(cut):
                print(f"  {f:28s} {s}")
        else:
            print(render_cut(cut, a.preview, a.jobs, a.manim))


if __name__ == "__main__":
    main()
