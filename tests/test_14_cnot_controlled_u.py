"""Tests for dependency_map.md node F14 — Controlled-unitary / CNOT. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("14_cnot_controlled_u")
k11 = load_kata("11_pauli_operators")
k13 = load_kata("13_hadamard")

ATOL = 1e-8


def basis(bits):
    v = np.array([1.0], dtype=complex)
    for b in bits:
        v = np.kron(v, np.array([0.0, 1.0] if b else [1.0, 0.0], dtype=complex))
    return v


def test_cnot_matrix_literal():
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    assert np.allclose(kata.cnot(2, 0, 1), expected, atol=ATOL)


def test_cnot_unitary_and_involution():
    c = kata.cnot(2, 0, 1)
    assert np.allclose(c.conj().T @ c, np.eye(4), atol=ATOL)
    assert np.allclose(c @ c, np.eye(4), atol=ATOL)


@pytest.mark.parametrize(
    "src,dst",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot_basis_permutation(src, dst):
    assert np.allclose(kata.cnot(2, 0, 1) @ basis(src), basis(dst), atol=ATOL)


def test_cnot_creates_bell_state():
    h_i = np.kron(k13.hadamard(), np.eye(2))
    out = kata.cnot(2, 0, 1) @ h_i @ basis((0, 0))
    assert np.allclose(out, np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2), atol=ATOL)


def test_hadamard_sandwich_flips_control_target():
    hh = np.kron(k13.hadamard(), k13.hadamard())
    assert np.allclose(hh @ kata.cnot(2, 0, 1) @ hh, kata.cnot(2, 1, 0), atol=ATOL)


def test_controlled_identity_is_identity():
    assert np.allclose(kata.controlled_unitary(k11.pauli_i(), 2, 0, 1), np.eye(4), atol=ATOL)


def test_cz_is_symmetric():
    assert np.allclose(kata.cz(2, 0, 1), kata.cz(2, 1, 0), atol=ATOL)
    assert np.allclose(kata.cz(2, 0, 1), np.diag([1, 1, 1, -1]).astype(complex), atol=ATOL)


def test_three_qubit_embedding():
    assert np.allclose(
        kata.cnot(3, control=1, target=2) @ basis((0, 1, 0)), basis((0, 1, 1)), atol=ATOL
    )
    assert np.allclose(
        kata.cnot(3, control=1, target=2) @ basis((0, 0, 0)), basis((0, 0, 0)), atol=ATOL
    )


def test_controlled_unitary_matches_projector_formula():
    u = k13.hadamard()
    p0 = np.diag([1, 0]).astype(complex)
    p1 = np.diag([0, 1]).astype(complex)
    expected = np.kron(p0, np.eye(2)) + np.kron(p1, u)
    assert np.allclose(kata.controlled_unitary(u, 2, 0, 1), expected, atol=ATOL)
