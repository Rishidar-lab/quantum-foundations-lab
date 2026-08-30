"""Import a kata module whose filename starts with a digit (e.g. ``01_hilbert_space.py``).

Python identifiers cannot start with a digit, so ``import katas.01_hilbert_space``
is a SyntaxError. Tests call ``load_kata("01_hilbert_space")`` instead.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

_KATAS_DIR = pathlib.Path(__file__).resolve().parent.parent / "katas"


def load_kata(stem: str) -> ModuleType:
    """Load ``katas/<stem>.py`` and return the module object.

    The module is cached in ``sys.modules`` under ``kata_<stem>``.
    """
    mod_name = f"kata_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = _KATAS_DIR / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"kata file not found: {path}")

    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
