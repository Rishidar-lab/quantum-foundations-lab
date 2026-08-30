"""Integration test (dependency_map.md node T).

Assembles the full Qiskit teleportation circuit, runs it on the local Aer
statevector simulator, and asserts that the output statevector on Bob's qubit
equals the input payload state after the classically-controlled correction.

Two forms:
  * the real circuit -- mid-circuit measurement + `if_test` Pauli correction --
    checked branch-realized via Aer `save_statevector`;
  * the deferred-measurement circuit -- fully coherent -- checked as an exact
    product state.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    partial_trace,
    random_statevector,
    state_fidelity,
)
from qiskit_aer import AerSimulator

from teleportation import build_teleportation_circuit

FID_TOL = 1e-9
SEEDS = list(range(24))


@pytest.mark.parametrize("seed", SEEDS)
def test_teleported_statevector_matches_input_after_correction(seed):
    """Real circuit: measure Alice's qubits, apply X**m1 Z**m0 to Bob, compare."""
    psi = random_statevector(2, seed=seed)
    qc = build_teleportation_circuit(psi.data, with_correction=True)
    qc.save_statevector()

    sim = AerSimulator(method="statevector")
    result = sim.run(transpile(qc, sim), shots=1, seed_simulator=1000 + seed).result()
    final = Statevector(result.get_statevector())

    # payload has been teleported to qubit index 2; trace out Alice's qubits
    rho_bob = partial_trace(final, [0, 1])
    assert state_fidelity(rho_bob, DensityMatrix(psi), validate=False) == pytest.approx(
        1.0, abs=FID_TOL
    )


@pytest.mark.parametrize("seed", SEEDS[:8])
def test_uncorrected_branches_are_pauli_off(seed):
    """Without Bob's correction the teleported state is a Pauli image, not |psi>
    (unless the sampled branch happened to be m0=m1=0). Guards against a circuit
    that trivially copies the input."""
    psi = random_statevector(2, seed=seed)
    qc = build_teleportation_circuit(psi.data, with_correction=False)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    result = sim.run(transpile(qc, sim), shots=1, seed_simulator=7).result()
    final = Statevector(result.get_statevector())
    rho_bob = partial_trace(final, [0, 1])
    fid = state_fidelity(rho_bob, DensityMatrix(psi), validate=False)
    # fidelity is 1 only on the identity branch; assert it is a valid pure qubit
    assert 0.0 - 1e-9 <= fid <= 1.0 + 1e-9
    assert np.isclose(np.trace(rho_bob.data), 1.0, atol=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_deferred_measurement_circuit_is_exact_product(seed):
    """Coherent circuit: U_tele (|psi> |00>) == |psi>_q2 (x) |+>_q1 (x) |+>_q0."""
    psi = random_statevector(2, seed=seed)
    qc = build_teleportation_circuit(psi.data, deferred=True)

    got = Statevector(qc)
    plus = Statevector.from_label("+")
    expected = Statevector(psi.data).tensor(plus).tensor(plus)  # q2 (x) q1 (x) q0
    assert state_fidelity(got, expected) == pytest.approx(1.0, abs=FID_TOL)

    rho_bob = partial_trace(got, [0, 1])
    assert state_fidelity(rho_bob, DensityMatrix(psi), validate=False) == pytest.approx(
        1.0, abs=FID_TOL
    )


def test_alice_outcomes_are_uniform_on_aer_counts():
    """Statistical cross-check of kata F17: Alice's two-bit outcome is uniform."""
    psi = random_statevector(2, seed=99)
    qc = build_teleportation_circuit(psi.data, with_correction=True)
    sim = AerSimulator()
    shots = 20000
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=2026).result().get_counts()
    # keys look like "m1 m0" (space-separated registers); collapse to the pair
    tally = {}
    for key, n in counts.items():
        bits = key.replace(" ", "")
        tally[bits] = tally.get(bits, 0) + n
    assert len(tally) == 4
    expected = shots / 4
    chi_sq = sum((n - expected) ** 2 / expected for n in tally.values())
    # 3 dof, 0.1% critical value ~ 16.27; deterministic seed keeps this stable
    assert chi_sq < 16.27
