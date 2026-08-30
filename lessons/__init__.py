"""PHASE 6 -- the learning interface.

    python -m lessons.run --list          table of contents
    python -m lessons.run                 where you are, and the next lesson
    python -m lessons.run --lesson N      lesson N, fields 1-9
    python -m lessons.run --lesson N --run    ... and execute the prediction check
    python -m lessons.run --lesson N --challenge     show the challenge (field 10)
    python -m lessons.run --lesson N --reveal        show the solution hint (gated)
    python -m lessons.run --next          mark the current lesson done, advance

Progress is stored in .lessons_progress.json (git-ignored, per user).
Challenge solutions are never printed unless you pass --reveal.
"""
from __future__ import annotations

import pathlib
import sys

_KATAS = str(pathlib.Path(__file__).resolve().parent.parent / "katas")
if _KATAS not in sys.path:
    sys.path.insert(0, _KATAS)

from lessons.content import LESSONS, Lesson  # noqa: E402

__all__ = ["LESSONS", "Lesson"]
