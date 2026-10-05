"""Neuromimetic vessel annotation: an experiment on vesselscene scenes (README.md)."""
from __future__ import annotations

import os
from pathlib import Path


def limbus_root() -> str:
    """The LIMBUS checkout that holds vesselmap (the comparison annotators import it): $LIMBUS_DATA, as for
    vesselscene's realism checks, else a sibling checkout ../limbus of this repository."""
    repo = Path(__file__).resolve().parents[2]
    for c in (os.environ.get("LIMBUS_DATA"), repo.parent / "limbus"):
        if c and (Path(c) / "vesselmap").is_dir():
            return str(c)
    raise ImportError("vesselmap not found: set LIMBUS_DATA to a LIMBUS checkout or clone it next to this "
                      "repository as ../limbus")
