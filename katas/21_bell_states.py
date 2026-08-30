"""Kata 21 — Bell states / Bell basis  (dependency_map.md node F21).

CORE node — the maximally entangled two-qubit basis.

Indexed so the teleportation decoder (H (x) I) . CNOT maps |beta(b0,b1)> to the
computational state |b0 b1>, equivalently |beta(b0,b1)> = CNOT . (H (x) I) |b0 b1>:
    beta(0,0) = |Phi+> = (|00>+|11>)/sqrt2      beta(0,1) = |Psi+> = (|01>+|10>)/sqrt2
    beta(1,0) = |Phi-> = (|00>-|11>)/sqrt2      beta(1,1) = |Psi-> = (|01>-|10>)/sqrt2

Implement:
  - phi_plus()               -> (|00>+|11>)/sqrt(2)
  - bell_state(b0, b1)       -> the indexed Bell state (length-4 vector)
  - bell_basis()             -> 4x4 matrix whose columns are bell_state(0,0..1,1)

Properties verified by tests/test_21_bell_states.py:
  - phi_plus explicit value; all four states orthonormal; bell_basis unitary
  - each Bell state is maximally entangled (entanglement entropy == 1)
  - (H (x) I) . CNOT decodes bell_state(b0,b1) -> |b0 b1>
  - local-Pauli orbit: (X (x) I)|Phi+> == |Psi+>, (Z (x) I)|Phi+> == |Phi->
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.complexfloating]

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def phi_plus() -> Vector:
    """Return the Bell state (|00> + |11>) / sqrt(2)."""
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def bell_state(b0: int, b1: int) -> Vector:
    """Return the Bell state beta(b0, b1) = CNOT . (H (x) I) |b0 b1>."""
    comp = np.zeros(4, dtype=complex)
    comp[(b0 << 1) | b1] = 1.0
    return _CNOT @ np.kron(_H, np.eye(2)) @ comp


def bell_basis() -> NDArray[np.complexfloating]:
    """Return the 4x4 change-of-basis matrix with Bell states as columns."""
    return np.column_stack([bell_state(b0, b1) for b0 in (0, 1) for b1 in (0, 1)])
