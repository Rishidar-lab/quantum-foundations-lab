"""Tests for dependency_map.md node F20 — Entanglement entropy.

Exact node: np.allclose / np.isclose with atol=1e-7 (eigen-decomposition + logs).
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("20_entanglement_entropy")

ATOL = 1e-7
RNG = np.random.default_rng(20260830)

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def rand_qubit():
    v = RNG.standard_normal(2) + 1j * RNG.standard_normal(2)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_entropy_of_maximally_mixed(d):
    assert np.isclose(kata.von_neumann_entropy(np.eye(d) / d), np.log2(d), atol=ATOL)


def test_entropy_of_pure_state_is_zero():
    psi = rand_qubit()
    assert np.isclose(kata.von_neumann_entropy(np.outer(psi, psi.conj())), 0.0, atol=ATOL)


def test_bell_state_has_one_bit_of_entanglement():
    assert np.isclose(kata.entanglement_entropy(BELL, [0], 2), 1.0, atol=ATOL)
    assert np.isclose(kata.entanglement_entropy(BELL, [1], 2), 1.0, atol=ATOL)


def test_symmetry_of_bipartite_entropy():
    # random 3-qubit pure state, entropy across {0} vs {1,2}
    v = RNG.standard_normal(8) + 1j * RNG.standard_normal(8)
    v = v / np.linalg.norm(v)
    assert np.isclose(
        kata.entanglement_entropy(v, [0], 3),
        kata.entanglement_entropy(v, [1, 2], 3),
        atol=ATOL,
    )


def test_product_state():
    a, b = rand_qubit(), rand_qubit()
    psi = np.kron(a, b)
    assert np.isclose(kata.entanglement_entropy(psi, [0], 2), 0.0, atol=ATOL)
    assert kata.is_product_state(psi, [0], 2) is True
    assert kata.is_product_state(BELL, [0], 2) is False


@pytest.mark.parametrize("trial", range(10))
def test_two_qubit_entropy_bounded(trial):
    v = RNG.standard_normal(4) + 1j * RNG.standard_normal(4)
    v = v / np.linalg.norm(v)
    s = kata.entanglement_entropy(v, [0], 2)
    assert -ATOL <= s <= 1.0 + ATOL
