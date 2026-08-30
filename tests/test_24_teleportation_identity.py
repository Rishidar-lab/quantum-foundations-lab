"""Tests for dependency_map.md node F24 — Teleportation identity. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("24_teleportation_identity")
k11 = load_kata("11_pauli_operators")
k13 = load_kata("13_hadamard")
k14 = load_kata("14_cnot_controlled_u")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)
X, Z = k11.pauli_x(), k11.pauli_z()


def rand_qubit():
    v = RNG.standard_normal(2) + 1j * RNG.standard_normal(2)
    return v / np.linalg.norm(v)


def kron(*ms):
    out = np.array([1.0], dtype=complex)
    for m in ms:
        out = np.kron(out, m)
    return out


def pauli_image(psi, m0, m1):
    return np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ psi


def test_pre_measurement_state_matches_circuit():
    psi = rand_qubit()
    state = kron(psi, np.array([1, 0, 0, 0], dtype=complex))
    state = kron(np.eye(2), k13.hadamard(), np.eye(2)) @ state
    state = k14.cnot(3, 1, 2) @ state
    state = k14.cnot(3, 0, 1) @ state
    state = kron(k13.hadamard(), np.eye(2), np.eye(2)) @ state
    assert np.allclose(kata.pre_measurement_state(psi), state, atol=ATOL)


@pytest.mark.parametrize("m0,m1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_amplitude_block_is_half_pauli_image(m0, m1):
    psi = rand_qubit()
    full = kata.pre_measurement_state(psi)
    idx = (m0 << 2) | (m1 << 1)
    block = full[idx : idx + 2]
    assert np.allclose(block, 0.5 * pauli_image(psi, m0, m1), atol=ATOL)


def test_branch_probabilities_uniform_and_normalized():
    psi = rand_qubit()
    branches = kata.teleportation_branches(psi)
    assert set(branches) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    total = 0.0
    for (m0, m1), (p, _) in branches.items():
        assert np.isclose(p, 0.25, atol=ATOL)
        total += p
    assert np.isclose(total, 1.0, atol=ATOL)


@pytest.mark.parametrize("trial", range(10))
def test_branch_q2_states_are_pauli_images(trial):
    psi = rand_qubit()
    for (m0, m1), (_, q2) in kata.teleportation_branches(psi).items():
        assert np.isclose(np.linalg.norm(q2), 1.0, atol=ATOL)
        expected = pauli_image(psi, m0, m1)
        assert np.isclose(abs(np.vdot(expected, q2)), 1.0, atol=ATOL)
