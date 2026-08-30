"""Tests for dependency_map.md node F11 — Pauli operators.

Exact node: deterministic np.allclose(atol=1e-8) on fixed 2x2 matrices.
Kata calls happen inside each test body so a stubbed kata reports as FAILED
(not ERROR) pre-implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("11_pauli_operators")

ATOL = 1e-8
I2 = np.eye(2, dtype=complex)
E0 = np.array([1.0, 0.0], dtype=complex)
E1 = np.array([0.0, 1.0], dtype=complex)

_CTOR = {
    "I": "pauli_i",
    "X": "pauli_x",
    "Y": "pauli_y",
    "Z": "pauli_z",
}


def P(name: str) -> np.ndarray:
    return getattr(kata, _CTOR[name])()


def test_shapes():
    for name in _CTOR:
        assert P(name).shape == (2, 2), name


def test_identity_is_identity():
    assert np.allclose(P("I"), I2, atol=ATOL)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_hermitian(name):
    m = P(name)
    assert np.allclose(m, m.conj().T, atol=ATOL)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_unitary(name):
    m = P(name)
    assert np.allclose(m.conj().T @ m, I2, atol=ATOL)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_involution(name):
    m = P(name)
    assert np.allclose(m @ m, I2, atol=ATOL)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_traceless(name):
    assert np.allclose(np.trace(P(name)), 0.0, atol=ATOL)


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_spectrum_is_plus_minus_one(name):
    eigs = np.sort(np.linalg.eigvalsh(P(name)))
    assert np.allclose(eigs, np.array([-1.0, 1.0]), atol=ATOL)


def test_anticommutation():
    x, y, z = P("X"), P("Y"), P("Z")
    assert np.allclose(x @ z, -(z @ x), atol=ATOL)
    assert np.allclose(x @ y, -(y @ x), atol=ATOL)
    assert np.allclose(y @ z, -(z @ y), atol=ATOL)


def test_binding_relation_xz_is_minus_i_y():
    assert np.allclose(P("X") @ P("Z"), -1j * P("Y"), atol=ATOL)


def test_basis_action():
    x, z = P("X"), P("Z")
    assert np.allclose(x @ E0, E1, atol=ATOL)
    assert np.allclose(x @ E1, E0, atol=ATOL)
    assert np.allclose(z @ E0, E0, atol=ATOL)
    assert np.allclose(z @ E1, -E1, atol=ATOL)
