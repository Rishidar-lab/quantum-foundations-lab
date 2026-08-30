"""Tests for dependency_map.md node F9 — Multi-qubit computational basis.

Exact node: deterministic np.allclose(atol=1e-8) on integer/vector identities.
This file also carries the project's **endianness contract test** — every
expected ket elsewhere in the build depends on the big-endian convention it
pins here.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("09_computational_basis_nq")

ATOL = 1e-8
E0 = np.array([1.0, 0.0], dtype=complex)
E1 = np.array([0.0, 1.0], dtype=complex)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_basis_vector_shape_and_single_entry(n):
    for i in range(2**n):
        v = kata.basis_vector(i, n)
        assert v.shape == (2**n,)
        expected = np.zeros(2**n, dtype=complex)
        expected[i] = 1.0
        assert np.allclose(v, expected, atol=ATOL)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_orthonormal(n):
    vs = [kata.basis_vector(i, n) for i in range(2**n)]
    for i, vi in enumerate(vs):
        for j, vj in enumerate(vs):
            assert np.allclose(np.vdot(vi, vj), 1.0 if i == j else 0.0, atol=ATOL)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_resolution_of_identity(n):
    dim = 2**n
    total = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        v = kata.basis_vector(i, n)
        total += np.outer(v, v.conj())
    assert np.allclose(total, np.eye(dim), atol=ATOL)


def test_big_endian_contract():
    # q0 is the MOST significant bit: (1,0,1) -> 1*4 + 0*2 + 1*1 = 5
    assert kata.bits_to_index((1, 0, 1)) == 5
    assert kata.index_to_bits(5, 3) == (1, 0, 1)
    # and (1,0) -> 2, (0,1) -> 1  (not the reverse)
    assert kata.bits_to_index((1, 0)) == 2
    assert kata.bits_to_index((0, 1)) == 1


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_round_trip_bits_index(n):
    for bits in itertools.product((0, 1), repeat=n):
        idx = kata.bits_to_index(bits)
        assert 0 <= idx < 2**n
        assert kata.index_to_bits(idx, n) == tuple(bits)
    for idx in range(2**n):
        assert kata.bits_to_index(kata.index_to_bits(idx, n)) == idx


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tensor_consistency_with_kron(n):
    for bits in itertools.product((0, 1), repeat=n):
        expected = np.array([1.0], dtype=complex)
        for b in bits:  # kron in register order q0, q1, ... (big-endian)
            expected = np.kron(expected, E1 if b else E0)
        got = kata.basis_vector(kata.bits_to_index(bits), n)
        assert np.allclose(got, expected, atol=ATOL)
