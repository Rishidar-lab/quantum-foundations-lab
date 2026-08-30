"""Tests for dependency_map.md node F22 — Bell-pair preparation. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("22_bell_pair_prep")
k13 = load_kata("13_hadamard")
k14 = load_kata("14_cnot_controlled_u")
k19 = load_kata("19_partial_trace")
k20 = load_kata("20_entanglement_entropy")
k21 = load_kata("21_bell_states")

ATOL = 1e-8


def comp(b0, b1):
    v = np.zeros(4, dtype=complex)
    v[(b0 << 1) | b1] = 1.0
    return v


def test_prepares_phi_plus():
    assert np.allclose(
        kata.prepare_bell_pair(), np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2), atol=ATOL
    )


def test_unitary_is_composition_of_h_and_cnot():
    expected = k14.cnot(2, 0, 1) @ np.kron(k13.hadamard(), np.eye(2))
    assert np.allclose(kata.bell_pair_unitary(), expected, atol=ATOL)
    u = kata.bell_pair_unitary()
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=ATOL)


def test_output_is_maximally_entangled():
    psi = kata.prepare_bell_pair()
    rho = np.outer(psi, psi.conj())
    assert np.allclose(k19.partial_trace(rho, [1], 2), np.eye(2) / 2, atol=ATOL)
    assert np.allclose(k19.partial_trace(rho, [0], 2), np.eye(2) / 2, atol=ATOL)
    assert np.isclose(k20.entanglement_entropy(psi, [0], 2), 1.0, atol=1e-7)


@pytest.mark.parametrize("b0,b1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_maps_computational_basis_to_bell_basis(b0, b1):
    out = kata.bell_pair_unitary() @ comp(b0, b1)
    assert np.allclose(out, k21.bell_state(b0, b1), atol=ATOL)
