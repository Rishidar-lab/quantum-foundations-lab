"""Tests for dependency_map.md node F19 — Partial trace. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("19_partial_trace")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)


def rand_density(d):
    a = RNG.standard_normal((d, d)) + 1j * RNG.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_trace_preserved_and_valid():
    rho = rand_density(8)
    red = kata.partial_trace(rho, [1], 3)
    assert red.shape == (4, 4)
    assert np.isclose(np.trace(red), 1.0, atol=ATOL)
    assert np.allclose(red, red.conj().T, atol=ATOL)
    assert np.min(np.linalg.eigvalsh(red)) > -1e-9


def test_product_state_factorizes():
    rho_a = rand_density(2)
    rho_b = rand_density(2)
    joint = np.kron(rho_a, rho_b)
    assert np.allclose(kata.partial_trace(joint, [1], 2), rho_a, atol=ATOL)
    assert np.allclose(kata.partial_trace(joint, [0], 2), rho_b, atol=ATOL)


def test_bell_state_reductions_are_maximally_mixed():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())
    assert np.allclose(kata.partial_trace(rho, [1], 2), np.eye(2) / 2, atol=ATOL)
    assert np.allclose(kata.partial_trace(rho, [0], 2), np.eye(2) / 2, atol=ATOL)


def test_local_expectation_consistency():
    rho = rand_density(8)
    m_a = RNG.standard_normal((2, 2)) + 1j * RNG.standard_normal((2, 2))
    m_a = m_a + m_a.conj().T
    # keep qubit 0, trace out 1 and 2
    red0 = kata.partial_trace(rho, [1, 2], 3)
    full = m_a
    for _ in range(2):
        full = np.kron(full, np.eye(2))
    assert np.isclose(np.trace(red0 @ m_a), np.trace(rho @ full), atol=ATOL)


def test_linearity():
    r1, r2 = rand_density(4), rand_density(4)
    mix = 0.3 * r1 + 0.7 * r2
    lhs = kata.partial_trace(mix, [1], 2)
    rhs = 0.3 * kata.partial_trace(r1, [1], 2) + 0.7 * kata.partial_trace(r2, [1], 2)
    assert np.allclose(lhs, rhs, atol=ATOL)


@pytest.mark.parametrize("traced", [[0], [1], [2], [0, 2]])
def test_shapes(traced):
    rho = rand_density(8)
    red = kata.partial_trace(rho, traced, 3)
    k = 3 - len(traced)
    assert red.shape == (2**k, 2**k)
