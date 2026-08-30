"""Kata 16 — Projective measurement  (dependency_map.md node F16).

CORE node — the projective special case of Postulate 3.

Given a Hermitian observable A = sum_m lambda_m P_m with orthogonal projectors
(P_m = P_m^dagger = P_m^2, sum_m P_m = I, P_m P_n = delta_mn P_m):
  p(m) = <psi| P_m |psi> = || P_m |psi> ||^2 ,   post-state = P_m |psi> / sqrt(p(m)) ,
  <A> = sum_m lambda_m p(m) = <psi| A |psi> .

Implement:
  - spectral_projectors(observable, atol=1e-9) -> list[(eigenvalue, projector)]
      grouped by distinct eigenvalue, projectors summing to I
  - expectation_value(state, observable) -> float (real part)
  - born_probs(state, projectors) -> np.ndarray
  - collapse(state, projector) -> normalized post-measurement state

Properties verified by tests/test_16_projective_measurement.py:
  - spectral_projectors: each Hermitian & idempotent; mutually orthogonal;
    sum == I; sum(lambda_m P_m) == observable
  - born_probs real, >= 0, sum to 1; equal <psi|P_m|psi>
  - expectation_value == <psi|A|psi> == sum lambda_m p(m)
  - collapse normalized; repeated projective measurement is deterministic
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]


def spectral_projectors(observable: Array, atol: float = 1e-9):
    """Return [(eigenvalue, projector), ...] from the spectral decomposition of A."""
    a = np.asarray(observable, dtype=complex)
    vals, vecs = np.linalg.eigh(a)
    pairs = []
    i = 0
    n = len(vals)
    while i < n:
        j = i
        while j < n and abs(vals[j] - vals[i]) <= atol * max(1.0, abs(vals[i])):
            j += 1
        block = vecs[:, i:j]
        pairs.append((float(vals[i].real), block @ block.conj().T))
        i = j
    return pairs


def expectation_value(state: Array, observable: Array) -> float:
    """Return the real expectation value <psi|A|psi>."""
    psi = np.asarray(state, dtype=complex)
    a = np.asarray(observable, dtype=complex)
    return float(np.vdot(psi, a @ psi).real)


def born_probs(state: Array, projectors: Sequence[Array]) -> NDArray[np.floating]:
    """Return p(m) = <psi|P_m|psi> for each projector."""
    psi = np.asarray(state, dtype=complex)
    return np.array(
        [np.vdot(psi, np.asarray(p, dtype=complex) @ psi).real for p in projectors],
        dtype=float,
    )


def collapse(state: Array, projector: Array) -> Array:
    """Return P|psi> / || P|psi> ||."""
    psi = np.asarray(state, dtype=complex)
    out = np.asarray(projector, dtype=complex) @ psi
    norm = np.linalg.norm(out)
    if norm == 0.0:
        raise ValueError("projector annihilates the state (zero probability)")
    return out / norm
