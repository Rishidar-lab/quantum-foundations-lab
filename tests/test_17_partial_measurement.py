"""Tests for dependency_map.md node F17 — Partial / subsystem measurement.

Exact node: deterministic np.allclose(atol=1e-8). Carries the map's signature
check: Alice's four outcomes are equiprobable (1/4) for every input state.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("17_partial_measurement")
k11 = load_kata("11_pauli_operators")
k13 = load_kata("13_hadamard")
k14 = load_kata("14_cnot_controlled_u")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)
X, Z = k11.pauli_x(), k11.pauli_z()


def rand_qubit():
    v = RNG.standard_normal(2) + 1j * RNG.standard_normal(2)
    return v / np.linalg.norm(v)


def kron(*mats):
    out = np.array([1.0], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def teleportation_pre_measurement(psi):
    """|psi>_q0 (x) |00> --> after  H1, CNOT(1->2), CNOT(0->1), H0."""
    state = kron(psi, np.array([1, 0, 0, 0], dtype=complex))  # |psi>|00>
    state = kron(np.eye(2), k13.hadamard(), np.eye(2)) @ state
    state = k14.cnot(3, control=1, target=2) @ state
    state = k14.cnot(3, control=0, target=1) @ state
    state = kron(k13.hadamard(), np.eye(2), np.eye(2)) @ state
    return state


def test_subsystem_projector_is_projection():
    p = kata.subsystem_projector([1], [0], 3)
    assert np.allclose(p @ p, p, atol=ATOL)
    assert np.allclose(p, p.conj().T, atol=ATOL)
    # sum over the two outcomes of q0 == I_8
    p0 = kata.subsystem_projector([0], [0], 3)
    assert np.allclose(p + p0, np.eye(8), atol=ATOL)


def test_bell_state_marginal_is_uniform():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    probs = kata.partial_measurement_probabilities(bell, [0], 2)
    assert np.isclose(probs[(0,)], 0.5, atol=ATOL)
    assert np.isclose(probs[(1,)], 0.5, atol=ATOL)


@pytest.mark.parametrize("trial", range(15))
def test_alice_outcomes_equiprobable_for_any_input(trial):
    psi = rand_qubit()
    state = teleportation_pre_measurement(psi)
    probs = kata.partial_measurement_probabilities(state, [0, 1], 3)
    assert set(probs) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for v in probs.values():
        assert np.isclose(v, 0.25, atol=ATOL)
    assert np.isclose(sum(probs.values()), 1.0, atol=ATOL)


@pytest.mark.parametrize("m0,m1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_conditional_q2_state_is_pauli_image(m0, m1):
    psi = rand_qubit()
    state = teleportation_pre_measurement(psi)
    collapsed = kata.partial_collapse(state, [0, 1], [m0, m1], 3)
    # q2 marginal: project q0,q1 onto |m0 m1> and read the remaining 2-vector
    idx = (m0 << 2) | (m1 << 1)
    q2 = collapsed[idx : idx + 2]
    q2 = q2 / np.linalg.norm(q2)
    expected = np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ psi
    # compare up to global phase
    assert np.isclose(abs(np.vdot(expected, q2)), 1.0, atol=ATOL)


def test_marginal_on_q0_alone():
    psi = rand_qubit()
    state = teleportation_pre_measurement(psi)
    probs = kata.partial_measurement_probabilities(state, [0], 3)
    assert np.isclose(probs[(0,)], 0.5, atol=ATOL)
    assert np.isclose(probs[(1,)], 0.5, atol=ATOL)
