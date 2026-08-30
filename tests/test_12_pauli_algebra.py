"""Tests for dependency_map.md node F12 — Pauli algebra / group relations.

Exact node: deterministic np.allclose(atol=1e-8). Every check exercises
kata 12's `pauli_mul`; the F11 matrices are used only as ground truth.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from kataloader import load_kata

kata12 = load_kata("12_pauli_algebra")
kata11 = load_kata("11_pauli_operators")

ATOL = 1e-8
LABELS = ("I", "X", "Y", "Z")
PHASES = (1 + 0j, -1 + 0j, 1j, -1j)
MAT = {
    "I": kata11.pauli_i(),
    "X": kata11.pauli_x(),
    "Y": kata11.pauli_y(),
    "Z": kata11.pauli_z(),
}


@pytest.mark.parametrize("a,b", list(itertools.product(LABELS, LABELS)))
def test_matches_matrix_product(a, b):
    phase, label = kata12.pauli_mul(a, b)
    assert label in LABELS
    assert any(np.isclose(phase, p, atol=ATOL) for p in PHASES)
    assert np.allclose(MAT[a] @ MAT[b], phase * MAT[label], atol=ATOL)


@pytest.mark.parametrize("p", LABELS)
def test_square_is_identity(p):
    phase, label = kata12.pauli_mul(p, p)
    assert label == "I"
    assert np.isclose(phase, 1.0, atol=ATOL)


@pytest.mark.parametrize("p", LABELS)
def test_identity_is_two_sided_unit(p):
    for phase, label in (kata12.pauli_mul("I", p), kata12.pauli_mul(p, "I")):
        assert label == p
        assert np.isclose(phase, 1.0, atol=ATOL)


@pytest.mark.parametrize("a,b,c", [("X", "Y", "Z"), ("Y", "Z", "X"), ("Z", "X", "Y")])
def test_cyclic_and_anticyclic(a, b, c):
    ph_ab, lab_ab = kata12.pauli_mul(a, b)
    assert lab_ab == c and np.isclose(ph_ab, 1j, atol=ATOL)
    ph_ba, lab_ba = kata12.pauli_mul(b, a)
    assert lab_ba == c and np.isclose(ph_ba, -1j, atol=ATOL)


@pytest.mark.parametrize("a,b", list(itertools.product(LABELS, LABELS)))
def test_commutation_from_rule(a, b):
    ph_ab, lab_ab = kata12.pauli_mul(a, b)
    ph_ba, lab_ba = kata12.pauli_mul(b, a)
    assert lab_ab == lab_ba
    if a == b or "I" in (a, b):
        assert np.isclose(ph_ab, ph_ba, atol=ATOL)      # commute
    else:
        assert np.isclose(ph_ab, -ph_ba, atol=ATOL)     # anticommute


def _key(phase: complex, label: str) -> tuple:
    return (round(phase.real, 9), round(phase.imag, 9), label)


def _group_mul(e1: tuple, e2: tuple) -> tuple:
    (ph1, l1), (ph2, l2) = e1, e2
    ph, lab = kata12.pauli_mul(l1, l2)
    return (ph1 * ph2 * ph, lab)


ELEMENTS = [(ph, lab) for ph in PHASES for lab in LABELS]  # 16-element Pauli group


def test_group_closure():
    allowed = {_key(ph, lab) for ph, lab in ELEMENTS}
    for e1 in ELEMENTS:
        for e2 in ELEMENTS:
            ph, lab = _group_mul(e1, e2)
            assert _key(ph, lab) in allowed


def test_group_associativity():
    rng = random.Random(20260830)
    for _ in range(300):
        e1, e2, e3 = (rng.choice(ELEMENTS) for _ in range(3))
        left = _group_mul(_group_mul(e1, e2), e3)
        right = _group_mul(e1, _group_mul(e2, e3))
        assert np.isclose(left[0], right[0], atol=ATOL)
        assert left[1] == right[1]


def test_every_element_has_an_inverse():
    identity = _key(1 + 0j, "I")
    for e1 in ELEMENTS:
        assert any(_key(*_group_mul(e1, e2)) == identity for e2 in ELEMENTS)
