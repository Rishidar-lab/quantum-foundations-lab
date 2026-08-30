"""Kata 24 — Teleportation identity  (dependency_map.md node F24).

CORE node — the algebraic core of the protocol.

For any |psi> = a|0> + b|1>, after applying  H1, CNOT(1->2), CNOT(0->1), H0  to
|psi>_q0 (x) |00>_{q1 q2}, the state is
  (1/2) sum_{m0,m1} |m0 m1>_{q0 q1} (x) (X**m1 Z**m0 |psi>)_{q2}.
Each of Alice's four equally likely outcomes leaves q2 in a known Pauli image of
|psi>, invertible by Z**m0 X**m1.

Implement:
  - pre_measurement_state(psi) -> length-8 vector (state described above)
  - teleportation_branches(psi) -> {(m0, m1): (probability, q2_state_unit)}

Properties verified by tests/test_24_teleportation_identity.py:
  - pre_measurement_state matches the explicit H/CNOT composition
  - each branch amplitude block equals (1/2) * X**m1 Z**m0 |psi>
  - every branch probability == 1/4 ; probabilities sum to 1
  - each branch q2 state == X**m1 Z**m0 |psi> (unit-normalized, up to phase)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from _deps import kata as _load

_k13 = _load("13_hadamard")
_k14 = _load("14_cnot_controlled_u")

Array = NDArray[np.complexfloating]


def pre_measurement_state(psi: Array) -> Array:
    """Return the 8-vector after H1, CNOT(1->2), CNOT(0->1), H0 on |psi>|00>."""
    psi = np.asarray(psi, dtype=complex)
    h = _k13.hadamard()
    eye = np.eye(2, dtype=complex)
    state = np.kron(psi, np.array([1, 0, 0, 0], dtype=complex))  # |psi> (x) |00>
    state = np.kron(np.kron(eye, h), eye) @ state                # H on q1
    state = _k14.cnot(3, control=1, target=2) @ state
    state = _k14.cnot(3, control=0, target=1) @ state
    state = np.kron(np.kron(h, eye), eye) @ state                # H on q0
    return state


def teleportation_branches(psi: Array) -> dict[tuple[int, int], tuple[float, Array]]:
    """Return {(m0, m1): (probability, normalized q2 state)} for the four outcomes."""
    full = pre_measurement_state(psi)
    branches: dict[tuple[int, int], tuple[float, Array]] = {}
    for m0 in (0, 1):
        for m1 in (0, 1):
            idx = (m0 << 2) | (m1 << 1)
            block = full[idx : idx + 2]
            prob = float(np.vdot(block, block).real)
            branches[(m0, m1)] = (prob, block / np.linalg.norm(block))
    return branches
