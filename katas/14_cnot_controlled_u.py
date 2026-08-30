"""Kata 14 — Controlled-unitary construction / CNOT  (dependency_map.md node F14).

CORE node — new object: two-qubit entangling gates.

A controlled-U on (control c, target t) is  |0><0|_c (x) I_t  +  |1><1|_c (x) U_t,
extended linearly. CNOT = controlled-X; CZ = controlled-Z. Embedding into an
n-qubit register uses the big-endian convention of dependency_map.md section 0
(q0 = most significant).

Implement:
  - controlled_unitary(u, n_qubits, control, target) -> 2**n x 2**n complex array
  - cnot(n_qubits=2, control=0, target=1)  -> controlled-X
  - cz(n_qubits=2, control=0, target=1)    -> controlled-Z

Properties verified by tests/test_14_cnot_controlled_u.py:
  - cnot(2,0,1) == [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
  - CNOT unitary; CNOT@CNOT == I; basis-permutation table (big-endian control)
  - CNOT (H (x) I) |00> == (|00>+|11>)/sqrt(2)   (entangling)
  - (H(x)H) cnot(0->1) (H(x)H) == cnot(1->0)
  - controlled_unitary(I) == I ; CZ is symmetric: cz(0->1) == cz(1->0)
  - 3-qubit embedding: cnot(3, control=1, target=2) |010> == |011>
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.complexfloating]


def _bit(index: int, qubit: int, n_qubits: int) -> int:
    """Value of `qubit` (0 = MSB) in the big-endian integer `index`."""
    return (index >> (n_qubits - 1 - qubit)) & 1


def _set_bit(index: int, qubit: int, value: int, n_qubits: int) -> int:
    pos = n_qubits - 1 - qubit
    return (index & ~(1 << pos)) | (value << pos)


def controlled_unitary(u: Matrix, n_qubits: int, control: int, target: int) -> Matrix:
    """Return the controlled-`u` gate embedded in a big-endian n-qubit register."""
    u = np.asarray(u, dtype=complex)
    if control == target:
        raise ValueError("control and target must differ")
    dim = 2**n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        if _bit(col, control, n_qubits) == 0:
            out[col, col] = 1.0
            continue
        tin = _bit(col, target, n_qubits)
        for tout in (0, 1):
            amp = u[tout, tin]
            if amp != 0.0:
                row = _set_bit(col, target, tout, n_qubits)
                out[row, col] += amp
    return out


def cnot(n_qubits: int = 2, control: int = 0, target: int = 1) -> Matrix:
    """Return the CNOT (controlled-X) gate."""
    return controlled_unitary(np.array([[0, 1], [1, 0]], dtype=complex), n_qubits, control, target)


def cz(n_qubits: int = 2, control: int = 0, target: int = 1) -> Matrix:
    """Return the CZ (controlled-Z) gate."""
    return controlled_unitary(np.array([[1, 0], [0, -1]], dtype=complex), n_qubits, control, target)
