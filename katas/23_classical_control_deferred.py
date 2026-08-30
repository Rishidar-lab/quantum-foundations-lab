"""Kata 23 — Classical control / deferred measurement  (dependency_map.md node F23).

CORE node — conditioning a gate on a classical bit, and the principle that lets
you push measurements to the end of a circuit.

A classically-controlled gate applies a fixed unitary iff a classical bit is 1.
Principle of deferred measurement: replacing each classically-controlled gate by
the corresponding quantum-controlled gate and moving the measurement to the end
leaves all output statistics unchanged.

Implement:
  - apply_conditional(state, classical_bit, gate, target, n_qubits)
        -> state with `gate` applied to `target` iff classical_bit == 1
  - bob_correction(q2_state, m0, m1)
        -> Z**m0 @ X**m1 @ q2_state   (X applied first, then Z)
  - defer(gate, control, target, n_qubits)
        -> the coherent controlled-`gate` that reproduces "measure control in the
           computational basis, then apply `gate` to target iff outcome 1"

Properties verified by tests/test_23_classical_control_deferred.py:
  - apply_conditional with bit 0 is identity; with bit 1 applies the gate
  - bob_correction inverts X**m1 Z**m0 : recovers |psi> for all four (m0,m1)
  - deferred-measurement equivalence: for a control qubit already in the
    computational basis, measure-then-conditional == defer()-then-measure,
    branch by branch and on the averaged density matrix
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from _deps import kata as _load

_k14 = _load("14_cnot_controlled_u")

Array = NDArray[np.complexfloating]

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def apply_conditional(
    state: Array, classical_bit: int, gate: Array, target: int, n_qubits: int
) -> Array:
    """Apply `gate` (2x2) to `target` iff `classical_bit` == 1; else return state."""
    psi = np.asarray(state, dtype=complex)
    if not classical_bit:
        return psi.copy()
    op = np.array([[1]], dtype=complex)
    for q in range(n_qubits):
        op = np.kron(op, np.asarray(gate, dtype=complex) if q == target else np.eye(2, dtype=complex))
    return op @ psi


def bob_correction(q2_state: Array, m0: int, m1: int) -> Array:
    """Return Z**m0 @ X**m1 @ q2_state (X first, then Z)."""
    v = np.asarray(q2_state, dtype=complex)
    return np.linalg.matrix_power(_Z, m0) @ np.linalg.matrix_power(_X, m1) @ v


def defer(gate: Array, control: int, target: int, n_qubits: int) -> Array:
    """Return the coherent controlled-`gate` equivalent to measure-then-conditional."""
    return _k14.controlled_unitary(gate, n_qubits, control, target)
