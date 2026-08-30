"""PHASE 2 -- targeted tests closing the weaknesses AUDIT.md found in the
original 356-test suite (W1-W5). Each test is written so it would FAIL against
the specific weak/tautological pattern it replaces.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

from kataloader import load_kata

k11 = load_kata("11_pauli_operators")
k19 = load_kata("19_partial_trace")
k21 = load_kata("21_bell_states")
k24 = load_kata("24_teleportation_identity")

from teleportation import build_teleportation_circuit  # noqa: E402

X, Z = k11.pauli_x(), k11.pauli_z()
RNG = np.random.default_rng(2026_0830)

# hand-written Bell-state literals -- NOT derived from CNOT.(H(x)I)
PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PSI_PLUS = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PHI_MINUS = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
LITERALS = {(0, 0): PHI_PLUS, (0, 1): PSI_PLUS, (1, 0): PHI_MINUS, (1, 1): PSI_MINUS}


def rand_qubit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(2) + 1j * np.random.default_rng(seed + 1).standard_normal(2)
    return v / np.linalg.norm(v)


# ---- W1: independent literal anchors for all four Bell states ---------------


@pytest.mark.parametrize("idx,vec", list(LITERALS.items()))
def test_bell_states_match_independent_literals(idx, vec):
    got = k21.bell_state(*idx)
    assert np.allclose(got, vec, atol=1e-12) or np.allclose(got, -vec, atol=1e-12)


def test_bell_states_match_qiskit_construction():
    """Qiskit builds beta(b0,b1) as: prep |q0=b0, q1=b1>, then h(0), cx(0,1).
    Compare (up to global phase, after reversing the 2-qubit order) to both the
    hand literal and the kata."""
    perm2 = [0, 2, 1, 3]  # reverse little-endian <-> big-endian for 2 qubits
    for (b0, b1), lit in LITERALS.items():
        qc = QuantumCircuit(2)
        if b0:
            qc.x(0)
        if b1:
            qc.x(1)
        qc.h(0)
        qc.cx(0, 1)
        q_big = np.asarray(Statevector(qc).data)[perm2]
        assert np.isclose(abs(np.vdot(lit, q_big)), 1.0, atol=1e-10), (b0, b1)
        assert np.isclose(abs(np.vdot(k21.bell_state(b0, b1), q_big)), 1.0, atol=1e-10), (b0, b1)


# ---- W2: teleportation identity from the CLOSED FORM, not a gate replay -----


@pytest.mark.parametrize("seed", range(12))
def test_pre_measurement_state_matches_closed_form(seed):
    """Build the pre-measurement 8-vector directly from the analytic identity
        (1/2) sum_{m0,m1} |m0 m1> (x) X^m1 Z^m0 |psi>
    and require the kata's gate-built state to equal it. This does NOT re-run
    H/CNOT, so it is independent of the wiring inside `pre_measurement_state`.
    """
    psi = rand_qubit(seed)
    closed = np.zeros(8, dtype=complex)
    for m0 in (0, 1):
        for m1 in (0, 1):
            ket = np.zeros(4, dtype=complex)
            ket[(m0 << 1) | m1] = 1.0
            q2 = np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ psi
            closed += 0.5 * np.kron(ket, q2)
    assert np.allclose(k24.pre_measurement_state(psi), closed, atol=1e-12)


def test_pre_measurement_state_matches_qiskit_independently():
    psi = rand_qubit(99)
    # Qiskit: |psi> (x) |Phi+> then cx(0,1), h(0)  -- built from Statevector, not the kata helper
    phi_plus = QuantumCircuit(3)
    phi_plus.initialize(psi, [0])
    phi_plus.h(1)
    phi_plus.cx(1, 2)
    phi_plus.cx(0, 1)
    phi_plus.h(0)
    q = np.asarray(Statevector(phi_plus).data)
    # reverse 3-qubit order to big-endian
    perm = [int(f"{i:03b}"[::-1], 2) for i in range(8)]
    assert np.allclose(k24.pre_measurement_state(psi), q[perm], atol=1e-10)


# ---- W3: the uncorrected branch really IS a specific Pauli image -----------


@pytest.mark.parametrize("seed", range(8))
def test_uncorrected_branch_is_exactly_a_pauli_image(seed):
    psi = rand_qubit(seed)
    qc = build_teleportation_circuit(psi, with_correction=False)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    res = sim.run(transpile(qc, sim), shots=1, seed_simulator=100 + seed).result()

    (bitkey,) = res.get_counts()  # "m1 m0" (space-joined registers)
    m1_str, m0_str = bitkey.split()
    m0, m1 = int(m0_str), int(m1_str)

    bob = np.asarray(partial_trace(Statevector(res.get_statevector()), [0, 1]).data)
    expected = np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ psi

    assert np.isclose(
        state_fidelity(bob, np.outer(expected, expected.conj()), validate=False), 1.0, atol=1e-9
    )
    if (m0, m1) != (0, 0):  # a non-trivial branch must NOT already be |psi>
        assert state_fidelity(bob, np.outer(psi, psi.conj()), validate=False) < 0.999


# ---- W4: partial trace with a genuinely complex reduced matrix ------------


def _reduced_by_definition(v: np.ndarray, keep: tuple[int, int], n: int) -> np.ndarray:
    """rho_keep from the raw definition: sum over the traced index of outer products
    of the fixed-slice amplitude vectors. Big-endian. Independent of kata 19."""
    traced = [q for q in range(n) if q not in keep]
    (t,) = traced
    out = np.zeros((4, 4), dtype=complex)
    for tb in (0, 1):
        slice_idx = []
        for kb0 in (0, 1):
            for kb1 in (0, 1):
                bits = [0, 0, 0]
                bits[keep[0]], bits[keep[1]], bits[t] = kb0, kb1, tb
                slice_idx.append(bits[0] * 4 + bits[1] * 2 + bits[2])
        amp = v[slice_idx]
        out += np.outer(amp, amp.conj())
    return out


def test_partial_trace_is_transpose_sensitive():
    """A random reduction has complex off-diagonals, so rho_A != rho_A^T.
    This test fails against a `partial_trace` that returns the transpose
    (the mutation `tools/mutation_audit.py` calls 'partial-trace--transposed-output').
    """
    v = RNG.standard_normal(8) + 1j * RNG.standard_normal(8)
    v = v / np.linalg.norm(v)
    rho = np.outer(v, v.conj())

    red = k19.partial_trace(rho, [2], 3)  # keep q0, q1 -- big-endian
    assert not np.allclose(red, red.T, atol=1e-6), "reduction is real-symmetric; pick another seed"

    ref = _reduced_by_definition(v, keep=(0, 1), n=3)
    assert np.allclose(red, ref, atol=1e-12)          # matches the raw definition
    assert not np.allclose(red, ref.T, atol=1e-6)     # ... and NOT its transpose


# ---- W5: seed sweep -- the same identities over many random draws ----------


@pytest.mark.parametrize("seed", range(40))
def test_born_rule_normalisation_seed_sweep(seed):
    psi = rand_qubit(seed * 7 + 1)
    p = np.abs(psi) ** 2
    assert np.isclose(p.sum(), 1.0, atol=1e-12)
    assert np.allclose(p, Statevector(psi).probabilities(), atol=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_teleportation_fidelity_seed_sweep(seed):
    psi = rand_qubit(seed * 13 + 3)
    qc = build_teleportation_circuit(psi, deferred=True)
    bob = partial_trace(Statevector(qc), [0, 1]).data
    assert np.isclose(
        state_fidelity(np.asarray(bob), np.outer(psi, psi.conj()), validate=False), 1.0, atol=1e-9
    )
