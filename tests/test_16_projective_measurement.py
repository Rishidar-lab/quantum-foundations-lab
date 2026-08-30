"""Tests for dependency_map.md node F16 — Projective measurement. Exact, atol=1e-8."""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("16_projective_measurement")
k11 = load_kata("11_pauli_operators")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)


def rand_state(d):
    v = RNG.standard_normal(d) + 1j * RNG.standard_normal(d)
    return v / np.linalg.norm(v)


def rand_hermitian(d):
    m = RNG.standard_normal((d, d)) + 1j * RNG.standard_normal((d, d))
    return m + m.conj().T


@pytest.mark.parametrize("obs_name", ["Z", "X", "hermitian4"])
def test_spectral_projectors_properties(obs_name):
    a = {"Z": k11.pauli_z(), "X": k11.pauli_x(), "hermitian4": rand_hermitian(4)}[obs_name]
    pairs = kata.spectral_projectors(a)
    projs = [p for _, p in pairs]
    dim = a.shape[0]
    for p in projs:
        assert np.allclose(p, p.conj().T, atol=ATOL)
        assert np.allclose(p @ p, p, atol=ATOL)
    assert np.allclose(sum(projs), np.eye(dim), atol=ATOL)
    for i in range(len(projs)):
        for j in range(len(projs)):
            if i != j:
                assert np.allclose(projs[i] @ projs[j], 0, atol=ATOL)
    assert np.allclose(sum(lam * p for lam, p in pairs), a, atol=ATOL)


@pytest.mark.parametrize("d", [2, 4])
def test_born_probs_distribution_and_formula(d):
    psi = rand_state(d)
    a = rand_hermitian(d)
    projs = [p for _, p in kata.spectral_projectors(a)]
    p = np.asarray(kata.born_probs(psi, projs))
    assert np.all(p >= -ATOL)
    assert np.isclose(p.sum(), 1.0, atol=ATOL)
    manual = np.array([np.vdot(psi, pr @ psi).real for pr in projs])
    assert np.allclose(p, manual, atol=ATOL)


@pytest.mark.parametrize("d", [2, 4])
def test_expectation_value_consistency(d):
    psi = rand_state(d)
    a = rand_hermitian(d)
    ev = kata.expectation_value(psi, a)
    assert np.isclose(ev, np.vdot(psi, a @ psi).real, atol=ATOL)
    pairs = kata.spectral_projectors(a)
    p = kata.born_probs(psi, [pr for _, pr in pairs])
    assert np.isclose(ev, sum(lam * pi for (lam, _), pi in zip(pairs, p)), atol=ATOL)


def test_collapse_normalized_and_deterministic_on_repeat():
    psi = rand_state(4)
    a = rand_hermitian(4)
    projs = [p for _, p in kata.spectral_projectors(a)]
    probs = np.asarray(kata.born_probs(psi, projs))
    m = int(np.argmax(probs))
    post = kata.collapse(psi, projs[m])
    assert np.isclose(np.linalg.norm(post), 1.0, atol=ATOL)
    again = np.asarray(kata.born_probs(post, projs))
    assert np.isclose(again[m], 1.0, atol=ATOL)
