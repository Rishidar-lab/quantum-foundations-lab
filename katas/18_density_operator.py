"""Kata 18 — Density operator  (dependency_map.md node F18).

CORE node — mixed states and the operator form of a quantum state.

rho = sum_i p_i |psi_i><psi_i| : Hermitian, positive semidefinite, trace 1.
Pure iff tr(rho^2) == 1 iff rho^2 == rho. Expectation <A> = tr(rho A).
Evolution rho -> U rho U^dagger.

Implement:
  - density_from_state(psi)            -> |psi><psi|
  - density_from_ensemble(states, ps)  -> sum_i p_i |psi_i><psi_i|
  - is_density_matrix(rho, atol=1e-9)  -> bool  (Hermitian, PSD, trace 1)
  - purity(rho)                        -> float  tr(rho^2)
  - expectation(rho, observable)       -> complex  tr(rho A)

Properties verified by tests/test_18_density_operator.py:
  - density_from_state: Hermitian, trace 1, PSD, purity 1, rho^2 == rho
  - density_from_ensemble of non-identical states: trace 1, PSD, purity < 1
  - is_density_matrix rejects trace != 1 and non-PSD matrices
  - expectation(|psi><psi|, A) == <psi|A|psi>
  - unitary evolution preserves trace and eigenvalues (hence purity)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]


def density_from_state(psi: Array) -> Array:
    """Return the pure-state density matrix |psi><psi|."""
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


def density_from_ensemble(states: Sequence[Array], probs: Sequence[float]) -> Array:
    """Return sum_i probs[i] * |states[i]><states[i]|."""
    states = [np.asarray(s, dtype=complex) for s in states]
    return sum(p * np.outer(s, s.conj()) for s, p in zip(states, probs, strict=True))


def is_density_matrix(rho: Array, atol: float = 1e-9) -> bool:
    """Return True iff rho is Hermitian, positive semidefinite, and has trace 1."""
    rho = np.asarray(rho, dtype=complex)
    if not np.allclose(rho, rho.conj().T, atol=atol):
        return False
    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        return False
    return bool(np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2)) >= -atol)


def purity(rho: Array) -> float:
    """Return tr(rho^2) as a real float."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.trace(rho @ rho).real)


def expectation(rho: Array, observable: Array) -> complex:
    """Return tr(rho @ observable)."""
    rho = np.asarray(rho, dtype=complex)
    return complex(np.trace(rho @ np.asarray(observable, dtype=complex)))
