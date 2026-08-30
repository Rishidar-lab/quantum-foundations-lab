"""Kata 09 — Multi-qubit computational basis  (dependency_map.md node F9).

WARMUP node: pure indexing / bookkeeping, no QM derivation.

Convention fixed by dependency_map.md section 0: **big-endian**,
    |q0 q1 ... q_{n-1}>  =  |q0> (x) |q1> (x) ... (x) |q_{n-1}>
so q0 is the MOST significant bit and the integer index of |b0 b1 ... b_{n-1}> is
    index = sum_k b_k * 2**(n-1-k).

Implement:
  - bits_to_index(bits)       : (b0, ..., b_{n-1}) -> int         (big-endian)
  - index_to_bits(index, n)   : int -> (b0, ..., b_{n-1})         (inverse)
  - basis_vector(index, n)    : -> length-2**n complex vector, 1.0 at `index`

Properties verified by tests/test_09_computational_basis_nq.py:
  - basis_vector(i, n) has shape (2**n,) and a single unit entry at i
  - the 2**n basis vectors are orthonormal; sum_i |i><i| == I_(2**n)
  - big-endian contract: bits (1, 0, 1) <-> index 5 for n = 3
  - round trip: index_to_bits(bits_to_index(b), len(b)) == b
  - tensor consistency: basis_vector(i, n) == kron of the single-qubit |b_k>
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def bits_to_index(bits: Sequence[int]) -> int:
    """Map a big-endian bit tuple (q0 first, most significant) to its integer index."""
    index = 0
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"bit must be 0 or 1, got {b!r}")
        index = (index << 1) | int(b)
    return index


def index_to_bits(index: int, n_qubits: int) -> tuple[int, ...]:
    """Inverse of `bits_to_index` for an `n_qubits`-wide register."""
    if not 0 <= index < 2**n_qubits:
        raise ValueError(f"index {index} out of range for {n_qubits} qubits")
    return tuple((index >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits))


def basis_vector(index: int, n_qubits: int) -> NDArray[np.complexfloating]:
    """Return the computational basis vector |index> in the 2**n_qubits-dim space."""
    dim = 2**n_qubits
    if not 0 <= index < dim:
        raise ValueError(f"index {index} out of range for {n_qubits} qubits")
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v
