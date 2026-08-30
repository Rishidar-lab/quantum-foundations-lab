"""Kata 17 — Partial / subsystem measurement  (dependency_map.md node F17).

CORE node — projective measurement of a subset of qubits.

For outcome m on a subset A of qubits, the projector is P_m^A (x) I_B (big-endian
embedding). Then p(m) = <Psi| (P_m^A (x) I_B) |Psi> and the conditional state of
the whole register is (P_m^A (x) I_B)|Psi> / sqrt(p(m)).

Implement:
  - subsystem_projector(outcome_bits, measured_qubits, n_qubits) -> 2**n x 2**n
  - partial_measurement_probabilities(state, measured_qubits, n_qubits)
        -> dict {outcome_bits_tuple: probability}
  - partial_collapse(state, measured_qubits, outcome_bits, n_qubits)
        -> normalized post-measurement state on the full register

Properties verified by tests/test_17_partial_measurement.py:
  - Bell state on (q0,q1), measure q0: p(0) == p(1) == 0.5
  - teleportation pre-measurement state, measure (q0,q1): all four outcomes
    have probability 1/4 for EVERY input |psi>
  - conditional state of q2 after that collapse == X^{m1} Z^{m0} |psi> (up to norm)
  - probabilities sum to 1; marginal on q0 alone is (0.5, 0.5)
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]

_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)


def subsystem_projector(
    outcome_bits: Sequence[int], measured_qubits: Sequence[int], n_qubits: int
) -> Array:
    """Return P_m on `measured_qubits` tensored with identity elsewhere (big-endian)."""
    bit_of = dict(zip(measured_qubits, outcome_bits, strict=True))
    op = np.array([[1]], dtype=complex)
    for q in range(n_qubits):
        if q in bit_of:
            op = np.kron(op, _P1 if bit_of[q] else _P0)
        else:
            op = np.kron(op, np.eye(2, dtype=complex))
    return op


def partial_measurement_probabilities(
    state: Array, measured_qubits: Sequence[int], n_qubits: int
) -> dict[tuple[int, ...], float]:
    """Return {outcome_bits: probability} over the measured qubits."""
    psi = np.asarray(state, dtype=complex)
    measured = list(measured_qubits)
    out: dict[tuple[int, ...], float] = {}
    for combo in itertools.product((0, 1), repeat=len(measured)):
        proj = subsystem_projector(combo, measured, n_qubits)
        out[combo] = float(np.vdot(psi, proj @ psi).real)
    return out


def partial_collapse(
    state: Array,
    measured_qubits: Sequence[int],
    outcome_bits: Sequence[int],
    n_qubits: int,
) -> Array:
    """Return the normalized post-measurement state on the full register."""
    psi = np.asarray(state, dtype=complex)
    proj = subsystem_projector(outcome_bits, measured_qubits, n_qubits)
    out = proj @ psi
    norm = np.linalg.norm(out)
    if norm == 0.0:
        raise ValueError("outcome has zero probability")
    return out / norm
