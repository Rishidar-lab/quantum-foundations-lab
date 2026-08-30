"""Kata 15 — Measurement postulate / Born rule  (dependency_map.md node F15).

CORE node — Postulate 3 (general form).

A measurement is a family {M_m} of operators with  sum_m M_m^dagger M_m == I.
On state |psi>: outcome m occurs with probability  p(m) = <psi| M_m^dagger M_m |psi>,
and the post-measurement state is  M_m |psi> / sqrt(p(m)).

Implement:
  - is_valid_measurement(ops, atol=1e-9) -> bool         (completeness relation)
  - born_probabilities(state, ops)        -> np.ndarray of real probabilities
  - collapse(state, ops, outcome)         -> post-measurement state (unit norm)

Properties verified by tests/test_15_measurement_postulate.py:
  - is_valid_measurement: True for computational projectors, False for a
    non-complete set
  - born_probabilities: real, all >= 0, sum to 1
  - for projective computational measurement, p(m) == |amplitude_m|**2
  - collapse returns a normalized vector; collapsing again on the same
    projective outcome is idempotent (prob 1)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]


def is_valid_measurement(ops: Sequence[Array], atol: float = 1e-9) -> bool:
    """Return True iff sum_m ops[m]^dagger ops[m] == I."""
    ops = [np.asarray(m, dtype=complex) for m in ops]
    dim = ops[0].shape[0]
    total = sum(m.conj().T @ m for m in ops)
    return bool(np.allclose(total, np.eye(dim), atol=atol))


def born_probabilities(state: Array, ops: Sequence[Array]) -> NDArray[np.floating]:
    """Return the vector of outcome probabilities p(m) = <psi| M_m^dagger M_m |psi>."""
    psi = np.asarray(state, dtype=complex)
    probs = []
    for m in ops:
        m = np.asarray(m, dtype=complex)
        probs.append(np.vdot(psi, m.conj().T @ m @ psi).real)
    return np.array(probs, dtype=float)


def collapse(state: Array, ops: Sequence[Array], outcome: int) -> Array:
    """Return the post-measurement state M_outcome |psi> / sqrt(p(outcome))."""
    psi = np.asarray(state, dtype=complex)
    m = np.asarray(ops[outcome], dtype=complex)
    out = m @ psi
    norm = np.linalg.norm(out)
    if norm == 0.0:
        raise ValueError("outcome has zero probability")
    return out / norm
