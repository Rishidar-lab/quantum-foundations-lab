"""Kata 10 — Outer products & projectors  (dependency_map.md node F10).

WARMUP node: linear-algebra bookkeeping, no QM derivation.

|a><b| is the rank-1 operator  v |-> <b|v> |a|  (physics inner product, so
<b|v> conjugates b). A projector satisfies P = P^dagger = P^2. For an
orthonormal basis {|i>}, sum_i |i><i| = I (resolution of identity) and
|i><i| |j><j| = delta_ij |i><i|.

Implement:
  - outer(a, b)          : -> |a><b|  as a (len(a) x len(b)) complex matrix
  - rank1_projector(v)   : -> |v><v| / <v|v>, the orthogonal projector onto span{v}

Properties verified by tests/test_10_projectors.py:
  - outer(a, b) @ v  ==  <b|v> * a
  - (outer(a, b))^dagger == outer(b, a)
  - rank1_projector(v): idempotent (P@P == P), Hermitian, trace == 1
  - P @ v == v ;  P @ w == 0 for w orthogonal to v
  - sum over a computational basis of rank1_projector(e_i) == I
  - rank1_projector(e_i) @ rank1_projector(e_j) == 0 for i != j
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complexfloating]


def outer(a: Array, b: Array) -> Array:
    """Return the outer-product operator |a><b| (conjugates b, physics convention)."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    return np.outer(a, b.conj())


def rank1_projector(v: Array) -> Array:
    """Return the orthogonal projector |v><v| / <v|v> onto span{v}."""
    v = np.asarray(v, dtype=complex)
    nrm2 = np.vdot(v, v).real
    if nrm2 == 0.0:
        raise ValueError("cannot project onto the zero vector")
    return np.outer(v, v.conj()) / nrm2
