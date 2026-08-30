"""Kata 01 — Complex Hilbert space  (dependency_map.md node F1).

The state space of a quantum system is a finite-dimensional complex Hilbert
space C^d carrying a Hermitian inner product. Physics ("bra-ket") convention:
the inner product <a|b> is *conjugate-linear in its first argument* and
*linear in its second argument*.

YOUR TASK: implement `inner_product` and `norm`. Any NumPy routine is fair game
(this node is deliberately a warm-up); the reasoning you supply is *which*
argument gets conjugated and why the norm is real.

Properties verified by tests/test_01_hilbert_space.py:
  - conjugate symmetry:        <a|b> == conj(<b|a>)
  - linearity in arg 2:        <a| x*b + y*c> == x*<a|b> + y*<a|c>
  - conjugate-linearity arg 1: <x*a + y*b| c> == conj(x)*<a|c> + conj(y)*<b|c>
  - positive definiteness:     <a|a> is real, >= 0, and == 0 iff a == 0
  - norm consistency:          norm(a) == sqrt(<a|a>)
  - Cauchy-Schwarz:            |<a|b>| <= norm(a) * norm(b)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.complexfloating]


def inner_product(a: Vector, b: Vector) -> complex:
    """Return the Hermitian inner product <a|b> (conjugate-linear in `a`).

    Parameters
    ----------
    a, b : 1-D complex arrays of equal length.

    Returns
    -------
    complex
        The scalar <a|b>.
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # np.vdot conjugates its FIRST argument -> conjugate-linear in `a`, linear in `b`.
    return complex(np.vdot(a, b))


def norm(a: Vector) -> float:
    """Return the induced norm ||a|| = sqrt(<a|a>) as a real, non-negative float."""
    a = np.asarray(a, dtype=complex)
    # <a|a> is real and non-negative; the tiny imaginary part from rounding is dropped.
    return float(np.sqrt(np.vdot(a, a).real))
