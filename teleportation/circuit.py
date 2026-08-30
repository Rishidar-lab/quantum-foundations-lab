"""The teleportation circuit, gate-for-gate from research/dependency_map.md section 1.

Register layout (Qiskit qubit indices):
    q[0] = payload  |psi>   (Alice)
    q[1] = Alice's half of the Bell pair
    q[2] = Bob's half of the Bell pair -> receives |psi>

Note on ordering: dependency_map.md uses big-endian |q0 q1 q2>; Qiskit
`Statevector` is little-endian (q0 is the least-significant tensor factor). The
circuit is identical; only the index bookkeeping in tests differs. The payload
always lives on qubit index 0 and is teleported to qubit index 2.
"""
from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation

# names of the two classical bits holding Alice's measurement outcomes
ALICE_BITS = ("m0", "m1")


def _bell_and_measure_basis(qc: QuantumCircuit, q) -> None:
    """Shared middle of every variant: Bell pair on (q1,q2) then Alice's rotation."""
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier(label="bell")
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="bell-meas")


def _append_correction(qc: QuantumCircuit, q, m0, m1) -> None:
    """Bob applies X**m1 then Z**m0 (kata F23 bob_correction)."""
    with qc.if_test((m1[0], 1)):
        qc.x(q[2])
    with qc.if_test((m0[0], 1)):
        qc.z(q[2])


def build_teleportation_circuit(
    state,
    *,
    deferred: bool = False,
    with_correction: bool = True,
) -> QuantumCircuit:
    """Return the full 3-qubit teleportation circuit that loads `state` on q[0].

    Parameters
    ----------
    state : length-2 complex array-like
        The unknown single-qubit state to teleport (loaded with StatePreparation).
    deferred : bool
        If True, replace the mid-circuit measurement + classically-controlled
        Pauli correction with the coherent equivalent CX(q1->q2), CZ(q0->q2)
        (principle of deferred measurement -- kata F23). No classical register.
    with_correction : bool
        Only meaningful when ``deferred=False``. If False, Alice's outcomes are
        measured but Bob applies no correction.
    """
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)

    q = QuantumRegister(3, "q")
    qc = QuantumCircuit(q, name="teleport")
    qc.append(StatePreparation(state), [q[0]])
    qc.barrier(label="prep")

    _bell_and_measure_basis(qc, q)

    if deferred:
        qc.cx(q[1], q[2])
        qc.cz(q[0], q[2])
        return qc

    m0 = ClassicalRegister(1, ALICE_BITS[0])
    m1 = ClassicalRegister(1, ALICE_BITS[1])
    qc.add_register(m0)
    qc.add_register(m1)
    qc.measure(q[0], m0[0])
    qc.measure(q[1], m1[0])
    qc.barrier(label="measure")

    if with_correction:
        _append_correction(qc, q, m0, m1)

    return qc


def build_verification_circuit(
    theta: float,
    phi: float,
    *,
    deferred: bool = False,
) -> QuantumCircuit:
    """Angle-parametrised teleportation + inverse preparation on q[2] + measure.

    Payload: |psi> = RZ(phi) RY(theta) |0>. After teleportation, q[2] holds
    |psi>; applying RY(-theta) RZ(-phi) returns it to |0>, so a computational
    measurement of q[2] yields '0' with probability 1 for perfect teleportation.
    P(q2 = 1) is then a direct teleportation-infidelity estimate.

    Uses only ry/rz/cx/(cz) + if_test -- no StatePreparation multiplexer -- so it
    transpiles cleanly for dynamic-circuit hardware and for Aer.
    """
    q = QuantumRegister(3, "q")
    qc = QuantumCircuit(q, name="teleport-verify")

    # payload preparation
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])
    qc.barrier(label="prep")

    _bell_and_measure_basis(qc, q)

    if deferred:
        qc.cx(q[1], q[2])
        qc.cz(q[0], q[2])
    else:
        m0 = ClassicalRegister(1, ALICE_BITS[0])
        m1 = ClassicalRegister(1, ALICE_BITS[1])
        qc.add_register(m0)
        qc.add_register(m1)
        qc.measure(q[0], m0[0])
        qc.measure(q[1], m1[0])
        qc.barrier(label="measure")
        _append_correction(qc, q, m0, m1)

    # inverse preparation on Bob's qubit
    qc.barrier(label="unprepare")
    qc.rz(-phi, q[2])
    qc.ry(-theta, q[2])

    out = ClassicalRegister(1, "out")
    qc.add_register(out)
    qc.measure(q[2], out[0])
    return qc
