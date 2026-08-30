"""PHASE 2 -- NumPy vs Qiskit agreement for every foundational concept.

Each concept is computed twice by independent code paths (hand-written linear
algebra vs qiskit.quantum_info) and the two must agree; each also carries a
physical prediction that must hold.
"""
from __future__ import annotations

import pytest

from foundations import all_crosschecks

CHECKS = all_crosschecks()


@pytest.mark.parametrize("cc", CHECKS, ids=lambda c: c.key)
def test_numpy_matches_qiskit(cc):
    assert cc.agrees, (
        f"{cc.topic}: NumPy and Qiskit disagree\n"
        f"  math : {cc.math}\n"
        f"  numpy: {cc.numpy_value}\n"
        f"  qiskit: {cc.qiskit_value}"
    )


@pytest.mark.parametrize("cc", CHECKS, ids=lambda c: c.key)
def test_physical_prediction_holds(cc):
    assert cc.prediction_ok, f"{cc.topic}: prediction failed -- {cc.prediction}"


def test_catalogue_covers_the_required_topics():
    required = {
        "complex_amplitudes", "ket_bra", "normalization", "inner_product",
        "orthogonality", "basis_states", "superposition", "born_rule", "matrices",
        "unitary", "hermitian", "pauli_xyz", "hadamard", "phase_gates", "cnot",
        "tensor_product", "basis_ordering", "phase_global_relative", "expectation",
        "bell_states", "entanglement", "reduced_density_matrix", "partial_trace",
        "purity", "fidelity", "teleportation",
    }
    assert {c.key for c in CHECKS} == required
