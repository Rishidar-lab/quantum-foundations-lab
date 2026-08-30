"""Kata 13 — Hadamard gate  (dependency_map.md node F13).

CORE node — new object (first non-Pauli single-qubit gate).

H = (1/sqrt(2)) [[1, 1], [1, -1]].  Hermitian AND unitary, so H@H = I. It maps
the computational (Z) basis to the X basis: H|0> = |+>, H|1> = |->, and
conjugates the Paulis as H X H = Z, H Z H = X, H Y H = -Y.

Implement:
  - hadamard() -> 2x2 complex array

Properties verified by tests/test_13_hadamard.py:
  - shape (2,2); Hermitian; unitary; involution H@H == I; det(H) == -1
  - H|0> == |+> == (|0>+|1>)/sqrt(2) ;  H|1> == |-> == (|0>-|1>)/sqrt(2)
  - H|+> == |0> ;  H|-> == |1>
  - H X H == Z ;  H Z H == X ;  H Y H == -Y
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hadamard() -> NDArray[np.complexfloating]:
    """Return the single-qubit Hadamard gate as a 2x2 complex array."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
