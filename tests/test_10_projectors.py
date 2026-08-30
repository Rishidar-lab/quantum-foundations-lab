"""Tests for dependency_map.md node F10 — Outer products & projectors.

Exact node: deterministic np.allclose(atol=1e-8) on matrix identities, seeded RNG.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("10_projectors")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)


def rand_vec(d: int) -> np.ndarray:
    return RNG.standard_normal(d) + 1j * RNG.standard_normal(d)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_outer_shape_and_action(d):
    a, b, v = rand_vec(d), rand_vec(d), rand_vec(d)
    op = kata.outer(a, b)
    assert op.shape == (d, d)
    assert np.allclose(op @ v, np.vdot(b, v) * a, atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_outer_adjoint_is_swapped(d):
    a, b = rand_vec(d), rand_vec(d)
    assert np.allclose(kata.outer(a, b).conj().T, kata.outer(b, a), atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 4, 8])
def test_rank1_projector_is_projection(d):
    v = rand_vec(d)
    p = kata.rank1_projector(v)
    assert np.allclose(p @ p, p, atol=ATOL)
    assert np.allclose(p, p.conj().T, atol=ATOL)
    assert np.allclose(np.trace(p), 1.0, atol=ATOL)


@pytest.mark.parametrize("d", [2, 3, 4, 8])
def test_rank1_projector_fixes_and_kills(d):
    v = rand_vec(d)
    p = kata.rank1_projector(v)
    assert np.allclose(p @ v, v, atol=ATOL)
    # build a vector orthogonal to v via Gram-Schmidt on a random vector
    w = rand_vec(d)
    w = w - (np.vdot(v, w) / np.vdot(v, v)) * v
    assert abs(np.vdot(v, w)) < ATOL
    assert np.allclose(p @ w, np.zeros(d), atol=ATOL)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_resolution_of_identity_from_projectors(n):
    dim = 2**n
    eye = np.eye(dim)
    total = sum(kata.rank1_projector(eye[i].astype(complex)) for i in range(dim))
    assert np.allclose(total, eye, atol=ATOL)


@pytest.mark.parametrize("n", [2, 3])
def test_basis_projectors_orthogonal(n):
    dim = 2**n
    eye = np.eye(dim)
    projs = [kata.rank1_projector(eye[i].astype(complex)) for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            prod = projs[i] @ projs[j]
            expected = projs[i] if i == j else np.zeros((dim, dim))
            assert np.allclose(prod, expected, atol=ATOL)
