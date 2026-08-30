"""Kata 22 — Bell-pair preparation circuit  (dependency_map.md node F22).

CORE node — the standard entangling circuit that produces the shared resource.

|Phi+> = CNOT_{0->1} . (H (x) I) . |00>.  H puts q0 in |+>; CNOT then correlates
q1. The two component gates are exactly F13 (Hadamard) and F14 (CNOT).

Implement:
  - bell_pair_unitary()   -> 4x4 unitary  CNOT_{0->1} @ (H (x) I)
  - prepare_bell_pair()   -> bell_pair_unitary() @ |00>   (== |Phi+>)

Properties verified by tests/test_22_bell_pair_prep.py:
  - prepare_bell_pair() == (|00> + |11>) / sqrt(2)
  - bell_pair_unitary is unitary
  - both single-qubit reductions of the output are I/2 ; entanglement entropy 1
  - applied to |01>, |10>, |11> it yields the other three Bell states
    (matching kata 21's indexing)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from _deps import kata as _load

_k13 = _load("13_hadamard")
_k14 = _load("14_cnot_controlled_u")

Matrix = NDArray[np.complexfloating]


def bell_pair_unitary() -> Matrix:
    """Return the 4x4 unitary CNOT_{0->1} @ (H (x) I)."""
    return _k14.cnot(2, 0, 1) @ np.kron(_k13.hadamard(), np.eye(2))


def prepare_bell_pair() -> NDArray[np.complexfloating]:
    """Return the prepared state |Phi+> = bell_pair_unitary() @ |00>."""
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    return bell_pair_unitary() @ ket00
