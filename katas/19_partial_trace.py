"""Kata 19 — Partial trace / reduced density operator  (dependency_map.md node F19).

CORE node — the map that gives a subsystem's state.

rho_A = tr_B(rho_AB) is the unique linear map with
    tr_B(|a1><a2| (x) |b1><b2|) = |a1><a2| * <b2|b1>.
All measurement statistics restricted to A are reproduced by rho_A.

Implement (qubit register, big-endian, dependency_map.md section 0):
  - partial_trace(rho, traced_qubits, n_qubits) -> reduced density matrix on
    the remaining qubits (order preserved, ascending)

Properties verified by tests/test_19_partial_trace.py:
  - tr(partial_trace(rho)) == tr(rho) == 1 ; result Hermitian & PSD
  - product state: partial_trace(rho_A (x) rho_B, trace B) == rho_A
  - Bell state: each single-qubit reduction == I/2
  - consistency: tr(rho_A @ M_A) == tr(rho_AB @ (M_A (x) I_B))
  - linearity in rho
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]


def partial_trace(rho: Array, traced_qubits: Sequence[int], n_qubits: int) -> Array:
    """Trace out `traced_qubits` from an n-qubit density matrix (big-endian)."""
    rho = np.asarray(rho, dtype=complex)
    traced = sorted(set(traced_qubits))
    keep = [q for q in range(n_qubits) if q not in traced]

    tensor = rho.reshape([2] * (2 * n_qubits))
    letters = [chr(ord("a") + i) for i in range(2 * n_qubits)]
    row = letters[:n_qubits]
    col = letters[n_qubits:]
    for q in traced:
        col[q] = row[q]  # identify row/col index -> contraction (trace)

    subs = (
        "".join(row)
        + "".join(col)
        + "->"
        + "".join(row[q] for q in keep)
        + "".join(col[q] for q in keep)
    )
    k = len(keep)
    return np.einsum(subs, tensor).reshape(2**k, 2**k)
