"""Tests for dependency_map.md node F18 — Density operator. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("18_density_operator")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)


def rand_state(d):
    v = RNG.standard_normal(d) + 1j * RNG.standard_normal(d)
    return v / np.linalg.norm(v)


def rand_unitary(d):
    q, r = np.linalg.qr(RNG.standard_normal((d, d)) + 1j * RNG.standard_normal((d, d)))
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


@pytest.mark.parametrize("d", [2, 4, 8])
def test_pure_state_density(d):
    psi = rand_state(d)
    rho = kata.density_from_state(psi)
    assert np.allclose(rho, rho.conj().T, atol=ATOL)
    assert np.isclose(np.trace(rho), 1.0, atol=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) > -ATOL
    assert np.isclose(kata.purity(rho), 1.0, atol=ATOL)
    assert np.allclose(rho @ rho, rho, atol=ATOL)
    assert kata.is_density_matrix(rho) is True


def test_mixed_state_density():
    d = 2
    states = [rand_state(d) for _ in range(3)]
    probs = np.array([0.5, 0.3, 0.2])
    rho = kata.density_from_ensemble(states, probs)
    assert np.isclose(np.trace(rho), 1.0, atol=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) > -ATOL
    assert kata.purity(rho) < 1.0 - 1e-6
    assert kata.is_density_matrix(rho) is True


def test_is_density_matrix_rejects_bad_input():
    assert kata.is_density_matrix(2.0 * np.eye(2, dtype=complex)) is False  # trace 2
    bad = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=complex)  # trace 1 but not PSD
    assert kata.is_density_matrix(bad) is False


@pytest.mark.parametrize("d", [2, 4])
def test_expectation_matches_bra_ket(d):
    psi = rand_state(d)
    a = RNG.standard_normal((d, d)) + 1j * RNG.standard_normal((d, d))
    a = a + a.conj().T
    val = kata.expectation(kata.density_from_state(psi), a)
    assert np.isclose(val, np.vdot(psi, a @ psi), atol=ATOL)


@pytest.mark.parametrize("d", [2, 4])
def test_unitary_evolution_preserves_spectrum(d):
    rho = kata.density_from_ensemble([rand_state(d), rand_state(d)], [0.6, 0.4])
    u = rand_unitary(d)
    evolved = u @ rho @ u.conj().T
    assert np.isclose(np.trace(evolved), 1.0, atol=ATOL)
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(evolved)), np.sort(np.linalg.eigvalsh(rho)), atol=ATOL
    )
    assert np.isclose(kata.purity(evolved), kata.purity(rho), atol=ATOL)
