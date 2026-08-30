"""Tests for dependency_map.md node F23 — Classical control / deferred measurement.

Exact node: np.allclose(atol=1e-8), seeded RNG.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("23_classical_control_deferred")
k11 = load_kata("11_pauli_operators")
k14 = load_kata("14_cnot_controlled_u")
k17 = load_kata("17_partial_measurement")
k19 = load_kata("19_partial_trace")

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


def test_apply_conditional_identity_when_bit_zero():
    state = RNG.standard_normal(8) + 1j * RNG.standard_normal(8)
    state = state / np.linalg.norm(state)
    assert np.allclose(kata.apply_conditional(state, 0, X, 2, 3), state, atol=ATOL)


def test_apply_conditional_applies_gate_when_bit_one():
    state = RNG.standard_normal(8) + 1j * RNG.standard_normal(8)
    state = state / np.linalg.norm(state)
    expected = kron(np.eye(2), np.eye(2), X) @ state
    assert np.allclose(kata.apply_conditional(state, 1, X, 2, 3), expected, atol=ATOL)


@pytest.mark.parametrize("m0,m1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_bob_correction_inverts_pauli_image(m0, m1):
    psi = rand_qubit()
    scrambled = np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ psi
    recovered = kata.bob_correction(scrambled, m0, m1)
    assert np.allclose(recovered, psi, atol=ATOL)


def test_defer_matches_controlled_unitary():
    assert np.allclose(kata.defer(X, 0, 1, 2), k14.cnot(2, 0, 1), atol=ATOL)
    assert np.allclose(kata.defer(Z, 0, 2, 3), k14.cz(3, 0, 2), atol=ATOL)


@pytest.mark.parametrize("trial", range(8))
def test_deferred_measurement_equivalence(trial):
    # control qubit q0 already in computational basis (superposition of |0>,|1>
    # would still work, but the principle is stated for classical data).
    rest = RNG.standard_normal(4) + 1j * RNG.standard_normal(4)
    rest = rest / np.linalg.norm(rest)
    branch_states = {}
    avg_classical = np.zeros((8, 8), dtype=complex)
    avg_deferred = np.zeros((8, 8), dtype=complex)
    for c in (0, 1):
        state = kron(np.array([1.0, 0.0] if c == 0 else [0.0, 1.0], dtype=complex), rest)
        # path A: measure control (trivial here), then conditional X on q2
        classical = kata.apply_conditional(state, c, X, 2, 3)
        # path B: coherent controlled-X (q0 -> q2), then measure control
        deferred_full = kata.defer(X, 0, 2, 3) @ state
        proj = k17.subsystem_projector([c], [0], 3)
        deferred = proj @ deferred_full
        deferred = deferred / np.linalg.norm(deferred)
        assert np.allclose(
            np.abs(np.vdot(classical, deferred)), 1.0, atol=ATOL
        )  # equal up to global phase
        avg_classical += 0.5 * np.outer(classical, classical.conj())
        avg_deferred += 0.5 * np.outer(deferred, deferred.conj())
    assert np.allclose(avg_classical, avg_deferred, atol=ATOL)
