"""Tests for dependency_map.md node F15 — Measurement postulate / Born rule.

Exact node: deterministic np.allclose(atol=1e-8), seeded RNG for input states.
"""
from __future__ import annotations

import numpy as np
import pytest

from kataloader import load_kata

kata = load_kata("15_measurement_postulate")

ATOL = 1e-8
RNG = np.random.default_rng(20260830)


def rand_state(d):
    v = RNG.standard_normal(d) + 1j * RNG.standard_normal(d)
    return v / np.linalg.norm(v)


def comp_projectors(d):
    return [np.outer(np.eye(d)[i], np.eye(d)[i]).astype(complex) for i in range(d)]


@pytest.mark.parametrize("d", [2, 4, 8])
def test_valid_measurement_true_for_projectors(d):
    assert kata.is_valid_measurement(comp_projectors(d)) is True


def test_valid_measurement_false_for_incomplete_set():
    d = 4
    ops = comp_projectors(d)[:-1]  # drop one -> sum != I
    assert kata.is_valid_measurement(ops) is False


@pytest.mark.parametrize("d", [2, 4, 8])
def test_born_probabilities_are_a_distribution(d):
    psi = rand_state(d)
    p = np.asarray(kata.born_probabilities(psi, comp_projectors(d)))
    assert np.all(p >= -ATOL)
    assert np.allclose(p.imag if np.iscomplexobj(p) else 0.0, 0.0, atol=ATOL)
    assert np.isclose(p.sum(), 1.0, atol=ATOL)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_born_equals_amplitude_squared(d):
    psi = rand_state(d)
    p = np.asarray(kata.born_probabilities(psi, comp_projectors(d)))
    assert np.allclose(p, np.abs(psi) ** 2, atol=ATOL)


@pytest.mark.parametrize("d", [2, 4])
def test_collapse_is_normalized(d):
    psi = rand_state(d)
    projs = comp_projectors(d)
    p = np.asarray(kata.born_probabilities(psi, projs))
    m = int(np.argmax(p))
    post = kata.collapse(psi, projs, m)
    assert np.isclose(np.linalg.norm(post), 1.0, atol=ATOL)


@pytest.mark.parametrize("d", [2, 4])
def test_collapse_is_idempotent_for_projective(d):
    psi = rand_state(d)
    projs = comp_projectors(d)
    p = np.asarray(kata.born_probabilities(psi, projs))
    m = int(np.argmax(p))
    post = kata.collapse(psi, projs, m)
    p2 = np.asarray(kata.born_probabilities(post, projs))
    assert np.isclose(p2[m], 1.0, atol=ATOL)
