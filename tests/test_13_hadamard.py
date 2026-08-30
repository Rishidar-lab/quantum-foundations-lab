"""Tests for dependency_map.md node F13 — Hadamard gate. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np

from kataloader import load_kata

kata = load_kata("13_hadamard")
k11 = load_kata("11_pauli_operators")

ATOL = 1e-8
I2 = np.eye(2, dtype=complex)
E0 = np.array([1.0, 0.0], dtype=complex)
E1 = np.array([0.0, 1.0], dtype=complex)
PLUS = (E0 + E1) / np.sqrt(2)
MINUS = (E0 - E1) / np.sqrt(2)


def test_shape():
    assert kata.hadamard().shape == (2, 2)


def test_hermitian():
    h = kata.hadamard()
    assert np.allclose(h, h.conj().T, atol=ATOL)


def test_unitary():
    h = kata.hadamard()
    assert np.allclose(h.conj().T @ h, I2, atol=ATOL)


def test_involution():
    h = kata.hadamard()
    assert np.allclose(h @ h, I2, atol=ATOL)


def test_determinant_is_minus_one():
    assert np.isclose(np.linalg.det(kata.hadamard()), -1.0, atol=ATOL)


def test_maps_z_basis_to_x_basis():
    h = kata.hadamard()
    assert np.allclose(h @ E0, PLUS, atol=ATOL)
    assert np.allclose(h @ E1, MINUS, atol=ATOL)


def test_maps_x_basis_back_to_z_basis():
    h = kata.hadamard()
    assert np.allclose(h @ PLUS, E0, atol=ATOL)
    assert np.allclose(h @ MINUS, E1, atol=ATOL)


def test_conjugates_paulis():
    h = kata.hadamard()
    x, y, z = k11.pauli_x(), k11.pauli_y(), k11.pauli_z()
    assert np.allclose(h @ x @ h, z, atol=ATOL)
    assert np.allclose(h @ z @ h, x, atol=ATOL)
    assert np.allclose(h @ y @ h, -y, atol=ATOL)
