"""Tests for dependency_map.md node F21 — Bell states / Bell basis. Exact, atol=1e-8."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("21_bell_states")
k11 = load_kata("11_pauli_operators")
k13 = load_kata("13_hadamard")
k14 = load_kata("14_cnot_controlled_u")
k20 = load_kata("20_entanglement_entropy")

ATOL = 1e-8


def comp(b0, b1):
    v = np.zeros(4, dtype=complex)
    v[(b0 << 1) | b1] = 1.0
    return v


def test_phi_plus_value():
    assert np.allclose(kata.phi_plus(), np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2), atol=ATOL)
    assert np.allclose(kata.bell_state(0, 0), kata.phi_plus(), atol=ATOL)


def test_orthonormal():
    states = [kata.bell_state(b0, b1) for b0, b1 in itertools.product((0, 1), repeat=2)]
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            assert np.allclose(np.vdot(a, b), 1.0 if i == j else 0.0, atol=ATOL)


def test_bell_basis_unitary():
    b = kata.bell_basis()
    assert b.shape == (4, 4)
    assert np.allclose(b.conj().T @ b, np.eye(4), atol=ATOL)


@pytest.mark.parametrize("b0,b1", list(itertools.product((0, 1), repeat=2)))
def test_maximally_entangled(b0, b1):
    s = k20.entanglement_entropy(kata.bell_state(b0, b1), [0], 2)
    assert np.isclose(s, 1.0, atol=1e-7)


@pytest.mark.parametrize("b0,b1", list(itertools.product((0, 1), repeat=2)))
def test_teleportation_decoder(b0, b1):
    decoder = np.kron(k13.hadamard(), np.eye(2)) @ k14.cnot(2, 0, 1)
    assert np.allclose(decoder @ kata.bell_state(b0, b1), comp(b0, b1), atol=ATOL)


def test_local_pauli_orbit():
    x_i = np.kron(k11.pauli_x(), np.eye(2))
    z_i = np.kron(k11.pauli_z(), np.eye(2))
    assert np.allclose(x_i @ kata.phi_plus(), kata.bell_state(0, 1), atol=ATOL)
    assert np.allclose(z_i @ kata.phi_plus(), kata.bell_state(1, 0), atol=ATOL)
