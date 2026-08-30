"""Load a sibling kata module whose filename starts with a digit.

Used by katas that build directly on an earlier node (e.g. F20 needs F19's
partial trace). Keeps the dependency inside the `katas/` package rather than
reaching into the test harness.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

_DIR = pathlib.Path(__file__).resolve().parent


def kata(stem: str) -> ModuleType:
    """Return the module object for ``katas/<stem>.py`` (cached)."""
    name = f"katadep_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
