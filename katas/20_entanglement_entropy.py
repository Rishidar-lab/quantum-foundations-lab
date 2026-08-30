"""Kata 20 — Entanglement entropy  (dependency_map.md node F20).

CORE node — quantifying entanglement of a pure bipartite state.

S(rho) = -tr(rho log2 rho) = -sum_i lambda_i log2(lambda_i), with 0 log 0 = 0.
A pure state factorizes across a cut iff its reduced state is pure (S == 0).
For a maximally entangled 2-qubit state the reduction is I/2 and S == 1 bit.

Implement:
  - von_neumann_entropy(rho, base=2)                    -> float
  - entanglement_entropy(psi, part_qubits, n_qubits)   -> float
        (von Neumann entropy of the reduced state on `part_qubits`)
  - is_product_state(psi, part_qubits, n_qubits, atol=1e-9) -> bool

Properties verified by tests/test_20_entanglement_entropy.py:
  - S(I/d) == log2(d) ; S(pure) == 0
  - Bell state: entanglement_entropy == 1.0 ; reduced purity == 0.5
  - S(rho_A) == S(rho_B) for a pure global state
  - product state |a> (x) |b>: entanglement_entropy == 0, is_product_state True
  - 0 <= entanglement_entropy <= 1 for two qubits
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from _deps import kata as _load

_k19 = _load("19_partial_trace")

Array = NDArray[np.complexfloating]


def von_neumann_entropy(rho: Array, base: float = 2.0) -> float:
    """Return the von Neumann entropy -sum lambda_i log_base(lambda_i)."""
    rho = np.asarray(rho, dtype=complex)
    eigs = np.linalg.eigvalsh((rho + rho.conj().T) / 2).real
    eigs = eigs[eigs > 1e-12]
    return float(-np.sum(eigs * (np.log(eigs) / np.log(base))))


def _reduced(psi: Array, part_qubits: Sequence[int], n_qubits: int) -> Array:
    psi = np.asarray(psi, dtype=complex)
    rho = np.outer(psi, psi.conj())
    traced = [q for q in range(n_qubits) if q not in set(part_qubits)]
    return _k19.partial_trace(rho, traced, n_qubits)


def entanglement_entropy(
    psi: Array, part_qubits: Sequence[int], n_qubits: int
) -> float:
    """Return the entanglement entropy across the (part_qubits | rest) cut."""
    return von_neumann_entropy(_reduced(psi, part_qubits, n_qubits))


def is_product_state(
    psi: Array, part_qubits: Sequence[int], n_qubits: int, atol: float = 1e-9
) -> bool:
    """Return True iff |psi> factorizes across the cut (reduced state is pure)."""
    red = _reduced(psi, part_qubits, n_qubits)
    return bool(np.isclose(np.trace(red @ red).real, 1.0, atol=atol))
