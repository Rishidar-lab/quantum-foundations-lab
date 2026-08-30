"""Kata 11 — Pauli operators  (dependency_map.md node F11).

CORE node.

The single-qubit Pauli operators I, X, Z (and Y = i X Z) in the computational
basis. Each of X, Y, Z is Hermitian AND unitary, hence involutive (squares to
I), traceless, with spectrum {+1, -1}. X is the bit flip (X|b> = |b^1|), Z the
phase flip (Z|b> = (-1)^b |b>).

Implement the four constructors below (each returns a fresh 2x2 complex array):
  - pauli_i()
  - pauli_x()
  - pauli_y()
  - pauli_z()

Properties verified by tests/test_11_pauli_operators.py:
  - each is 2x2; X, Y, Z are Hermitian and unitary; I is identity
  - involution: X@X == Y@Y == Z@Z == I
  - traceless: tr(X) == tr(Y) == tr(Z) == 0 ; eigenvalues are {+1, -1}
  - anticommutation: X@Z == -Z@X, X@Y == -Y@X, Y@Z == -Z@Y
  - binding relation: X @ Z == -1j * Y
  - basis action: X|0> == |1>, X|1> == |0>, Z|0> == |0>, Z|1> == -|1>
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.complexfloating]


def pauli_i() -> Matrix:
    """Return the 2x2 identity operator."""
    return np.array([[1, 0], [0, 1]], dtype=complex)


def pauli_x() -> Matrix:
    """Return the Pauli X (bit-flip) operator in the computational basis."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> Matrix:
    """Return the Pauli Y operator in the computational basis."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> Matrix:
    """Return the Pauli Z (phase-flip) operator in the computational basis."""
    return np.array([[1, 0], [0, -1]], dtype=complex)
