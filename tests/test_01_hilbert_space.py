"""Tests for dependency_map.md node F1 — Complex Hilbert space.

Exact node: every check is deterministic np.allclose(atol=1e-8) on scalars
computed two ways, plus the Cauchy-Schwarz inequality with a small slack.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("01_hilbert_space")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)
DIMS = (2, 3, 4, 8)


def rand_vec(d: int) -> np.ndarray:
    return RNG.standard_normal(d) + 1j * RNG.standard_normal(d)


@pytest.mark.parametrize("d", DIMS)
def test_conjugate_symmetry(d):
    a, b = rand_vec(d), rand_vec(d)
    assert np.allclose(
        kata.inner_product(a, b), np.conj(kata.inner_product(b, a)), atol=ATOL
    )


@pytest.mark.parametrize("d", DIMS)
def test_linearity_in_second_argument(d):
    a, b, c = rand_vec(d), rand_vec(d), rand_vec(d)
    x, y = (0.3 + 1.2j), (-1.1 + 0.4j)
    lhs = kata.inner_product(a, x * b + y * c)
    rhs = x * kata.inner_product(a, b) + y * kata.inner_product(a, c)
    assert np.allclose(lhs, rhs, atol=ATOL)


@pytest.mark.parametrize("d", DIMS)
def test_conjugate_linearity_in_first_argument(d):
    a, b, c = rand_vec(d), rand_vec(d), rand_vec(d)
    x, y = (0.3 + 1.2j), (-1.1 + 0.4j)
    lhs = kata.inner_product(x * a + y * b, c)
    rhs = np.conj(x) * kata.inner_product(a, c) + np.conj(y) * kata.inner_product(b, c)
    assert np.allclose(lhs, rhs, atol=ATOL)


@pytest.mark.parametrize("d", DIMS)
def test_positive_definite_nonzero(d):
    a = rand_vec(d)
    ip = kata.inner_product(a, a)
    assert abs(np.imag(ip)) < ATOL
    assert np.real(ip) > 0.0


def test_positive_definite_zero_vector():
    z = np.zeros(4, dtype=complex)
    assert np.allclose(kata.inner_product(z, z), 0.0, atol=ATOL)


@pytest.mark.parametrize("d", DIMS)
def test_norm_matches_sqrt_inner_product(d):
    a = rand_vec(d)
    assert np.allclose(
        kata.norm(a), np.sqrt(np.real(kata.inner_product(a, a))), atol=ATOL
    )


def test_norm_is_real_nonnegative_float():
    a = rand_vec(8)
    n = kata.norm(a)
    assert np.isreal(n)
    assert float(n) >= 0.0


@pytest.mark.parametrize("trial", range(25))
def test_cauchy_schwarz(trial):
    a, b = rand_vec(4), rand_vec(4)
    assert abs(kata.inner_product(a, b)) <= kata.norm(a) * kata.norm(b) + ATOL
