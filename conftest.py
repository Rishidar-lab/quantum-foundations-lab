"""Repo-root conftest: put `tests/` and `katas/` on sys.path so that

  * test files can do `from kataloader import load_kata`
  * kata modules that build on an earlier kata can do `from _deps import kata`

regardless of pytest's import mode.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent
for _p in (str(_ROOT), str(_ROOT / "tests"), str(_ROOT / "katas")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
