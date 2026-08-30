"""PHASE 2 — Mathematical foundation.

For every quantum concept the katas implement, `foundations.crosscheck` provides
FOUR linked representations:

    MATHEMATICS            -- a one-line statement of the identity / rule
        |
    NUMPY LINEAR ALGEBRA   -- the object built explicitly from complex matrices
        |
    QISKIT                 -- the SAME object obtained independently via
                              qiskit.quantum_info (Operator / Statevector /
                              DensityMatrix / SparsePauliOp / partial_trace)
        |
    PHYSICAL PREDICTION    -- a measurable consequence, checked numerically

`tests/test_foundations_crosscheck.py` asserts NumPy == Qiskit (to 1e-10 where
the result is exact) and that each physical prediction holds.

Run `python -m foundations` for a human-readable table.
"""
from __future__ import annotations

import pathlib
import sys

# make the digit-prefixed kata modules importable when run outside pytest
_KATAS = str(pathlib.Path(__file__).resolve().parent.parent / "katas")
if _KATAS not in sys.path:
    sys.path.insert(0, _KATAS)

from foundations.crosscheck import CrossCheck, all_crosschecks  # noqa: E402

__all__ = ["CrossCheck", "all_crosschecks"]
