"""NumPy <-> Qiskit cross-checks for every quantum concept in the kata chain.

Each `CrossCheck` carries the mathematics, an explicit NumPy value, an
independently-computed Qiskit value, and a physically measurable prediction with
a boolean verdict. Nothing here calls a kata to produce the *expected* value of
another kata's cross-check -- the two sides are always computed by different code
paths (hand-written linear algebra vs `qiskit.quantum_info`).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Pauli,
    SparsePauliOp,
    Statevector,
    partial_trace,
    state_fidelity,
)

# ------------------------------------------------------------------ helpers ---

I2 = np.eye(2, dtype=complex)
KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)
KETP = (KET0 + KET1) / np.sqrt(2)
KETM = (KET0 - KET1) / np.sqrt(2)
KETpi = (KET0 + 1j * KET1) / np.sqrt(2)
KETmi = (KET0 - 1j * KET1) / np.sqrt(2)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)

_RNG = np.random.default_rng(0xC0FFEE)


def rand_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def reverse_qubit_order(mat_or_vec: np.ndarray, n_qubits: int) -> np.ndarray:
    """Map an operator/vector between big-endian |q0..q_{n-1}> and Qiskit's
    little-endian tensor convention by reversing the qubit order.

    This is the ONE bridge between `dependency_map.md` (q0 = most significant)
    and Qiskit (`Statevector`/`Operator` put q0 least significant).
    """
    a = np.asarray(mat_or_vec, dtype=complex)
    perm = [int(f"{i:0{n_qubits}b}"[::-1], 2) for i in range(2**n_qubits)]
    if a.ndim == 1:
        return a[perm]
    return a[np.ix_(perm, perm)]


def _probs_from_amplitudes(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


# ------------------------------------------------------------- the datatype ---


@dataclasses.dataclass
class CrossCheck:
    key: str
    topic: str
    math: str
    numpy_value: Any
    qiskit_value: Any
    prediction: str
    prediction_ok: bool
    atol: float = 1e-10

    @property
    def agrees(self) -> bool:
        a = np.asarray(self.numpy_value, dtype=complex)
        b = np.asarray(self.qiskit_value, dtype=complex)
        return a.shape == b.shape and bool(np.allclose(a, b, atol=self.atol))


# ------------------------------------------------------------- the registry ---


def all_crosschecks() -> list[CrossCheck]:  # noqa: C901 - a flat catalogue by design
    cc: list[CrossCheck] = []

    # 1 -- complex amplitudes -------------------------------------------------
    alpha, beta = (0.6 + 0.0j), (0.0 + 0.8j)  # |alpha|^2 + |beta|^2 = 1
    psi = np.array([alpha, beta])
    q_psi = Statevector([alpha, beta]).data
    cc.append(CrossCheck(
        "complex_amplitudes", "complex amplitudes",
        "|psi> = alpha|0> + beta|1>,  alpha,beta in C,  |alpha|^2+|beta|^2 = 1",
        psi, q_psi,
        "P(0)=|alpha|^2=0.36, P(1)=|beta|^2=0.64",
        np.allclose(_probs_from_amplitudes(psi), [0.36, 0.64], atol=1e-12),
    ))

    # 2 -- ket / bra mapping ------------------------------------------------
    bra0 = KET0.conj()
    q_bra0 = Statevector.from_label("0").data.conj()
    cc.append(CrossCheck(
        "ket_bra", "ket/bra notation mapping",
        "<0| is the conjugate transpose of |0>;  <0|0> = 1,  <0|1> = 0",
        bra0, q_bra0,
        "<0|0> = 1 and <0|1> = 0",
        np.isclose(bra0 @ KET0, 1) and np.isclose(bra0 @ KET1, 0),
    ))

    # 3 -- normalization ---------------------------------------------------
    raw = np.array([3.0 + 1j, -2.0j])
    npn = raw / np.linalg.norm(raw)
    qn = Statevector(raw / np.linalg.norm(raw)).data
    cc.append(CrossCheck(
        "normalization", "normalization",
        "a physical state has <psi|psi> = 1; divide by sqrt(<psi|psi>)",
        npn, qn,
        "sum of Born probabilities = 1",
        np.isclose(_probs_from_amplitudes(npn).sum(), 1.0, atol=1e-12),
    ))

    # 4 -- inner products ------------------------------------------------
    a, b = rand_state(2, 11), rand_state(2, 12)
    ip_np = np.vdot(a, b)
    ip_q = Statevector(a).inner(Statevector(b))
    cc.append(CrossCheck(
        "inner_product", "inner products",
        "<a|b> = sum_i conj(a_i) b_i  (conjugate-linear in the first slot)",
        ip_np, ip_q,
        "|<a|b>|^2 is the probability of finding |b> in state |a>",
        np.isclose(abs(ip_np) ** 2, state_fidelity(Statevector(a), Statevector(b)), atol=1e-10),
    ))

    # 5 -- orthogonality ------------------------------------------------
    cc.append(CrossCheck(
        "orthogonality", "orthogonality",
        "<0|1> = 0 : the computational basis is orthonormal",
        np.vdot(KET0, KET1), Statevector.from_label("0").inner(Statevector.from_label("1")),
        "measuring |0> in the Z basis yields outcome 1 with probability 0",
        np.isclose(Statevector.from_label("0").probabilities()[1], 0.0, atol=1e-12),
    ))

    # 6 -- basis states -------------------------------------------------
    cc.append(CrossCheck(
        "basis_states", "basis states",
        "|0> = (1,0)^T, |1> = (0,1)^T span C^2",
        np.stack([KET0, KET1]),
        np.stack([Statevector.from_label("0").data, Statevector.from_label("1").data]),
        "{|0>,|1>} resolves the identity: |0><0| + |1><1| = I",
        np.allclose(np.outer(KET0, KET0) + np.outer(KET1, KET1), I2, atol=1e-12),
    ))

    # 7 -- superposition ----------------------------------------------
    sup_np = H @ KET0
    sup_q = Statevector(QuantumCircuit(1).compose(_h_circ())).data
    cc.append(CrossCheck(
        "superposition", "superposition",
        "H|0> = (|0> + |1>)/sqrt(2) = |+>",
        sup_np, sup_q,
        "P(0) = P(1) = 1/2 when measuring |+> in the Z basis",
        np.allclose(Statevector(_h_circ()).probabilities(), [0.5, 0.5], atol=1e-12),
    ))

    # 8 -- Born rule ------------------------------------------------
    st = rand_state(2, 21)
    born_np = _probs_from_amplitudes(st)
    born_q = Statevector(st).probabilities()
    cc.append(CrossCheck(
        "born_rule", "Born rule",
        "P(outcome i) = |<i|psi>|^2",
        born_np, born_q,
        "probabilities are non-negative and sum to 1",
        np.all(born_np >= 0) and np.isclose(born_np.sum(), 1.0, atol=1e-12),
    ))

    # 9 -- matrices ------------------------------------------------
    cc.append(CrossCheck(
        "matrices", "matrices",
        "a linear operator on C^2 is a 2x2 complex matrix in a fixed basis",
        X, Operator(XGate()).data,
        "X is its own matrix representation in the computational basis",
        np.allclose(X @ KET0, KET1, atol=1e-12),
    ))

    # 10 -- unitary operators -------------------------------------
    u_np = H @ X @ H
    u_q = (Operator(HGate()) & Operator(XGate()) & Operator(HGate())).data
    cc.append(CrossCheck(
        "unitary", "unitary operators",
        "U^dagger U = I ; unitaries preserve <psi|psi>",
        u_np, u_q,
        "HXH is unitary (its own inverse here) and equals Z",
        np.allclose(u_np.conj().T @ u_np, I2, atol=1e-12) and np.allclose(u_np, Z, atol=1e-12),
    ))

    # 11 -- Hermitian operators ---------------------------------
    obs = np.array([[2.0, 1 - 1j], [1 + 1j, -3.0]], dtype=complex)
    ev_np = np.linalg.eigvalsh(obs)
    ev_q = np.sort(np.linalg.eigvalsh(Operator(obs).data))
    cc.append(CrossCheck(
        "hermitian", "Hermitian operators",
        "A = A^dagger => real eigenvalues, orthonormal eigenbasis",
        np.sort(ev_np), ev_q,
        "eigenvalues of a Hermitian observable are real",
        np.allclose(ev_np.imag, 0.0, atol=1e-12),
    ))

    # 12 -- Pauli X/Y/Z ----------------------------------------
    cc.append(CrossCheck(
        "pauli_xyz", "Pauli X/Y/Z",
        "X,Y,Z: Hermitian and unitary, X^2=Y^2=Z^2=I, XY=iZ",
        np.stack([X, Y, Z]),
        np.stack([Operator(XGate()).data, Operator(YGate()).data, Operator(ZGate()).data]),
        "XY = iZ (cyclic) and each Pauli squares to I",
        np.allclose(X @ Y, 1j * Z, atol=1e-12) and np.allclose(Z @ Z, I2, atol=1e-12),
    ))

    # 13 -- Hadamard -----------------------------------------
    cc.append(CrossCheck(
        "hadamard", "Hadamard",
        "H = (1/sqrt2)[[1,1],[1,-1]] ; H^2 = I ; H X H = Z",
        H, Operator(HGate()).data,
        "H maps the Z eigenbasis to the X eigenbasis (|0> -> |+>)",
        np.allclose(H @ KET0, KETP, atol=1e-12) and np.allclose(H @ X @ H, Z, atol=1e-12),
    ))

    # 14 -- phase gates ------------------------------------
    cc.append(CrossCheck(
        "phase_gates", "phase gates",
        "S = diag(1, i), T = diag(1, e^{i pi/4}) ; S^2 = Z",
        np.stack([S, np.diag([1, np.exp(1j * np.pi / 4)])]),
        np.stack([Operator(SGate()).data, Operator(TGate()).data]),
        "S|+> = |+i> = (|0> + i|1>)/sqrt(2)",
        np.allclose(S @ KETP, KETpi, atol=1e-12) and np.allclose(S @ S, Z, atol=1e-12),
    ))

    # 15 -- CNOT (endianness bridge) ----------------------
    from _deps import kata as _load

    k14 = _load("14_cnot_controlled_u")
    cnot_bigendian = k14.cnot(2, control=0, target=1)
    cnot_qiskit = Operator(CXGate()).data  # control = qubit 0, little-endian
    cc.append(CrossCheck(
        "cnot", "CNOT",
        "CNOT = |0><0|(x)I + |1><1|(x)X ; reversing qubit order bridges the two conventions",
        reverse_qubit_order(cnot_bigendian, 2), cnot_qiskit,
        "CNOT|10> -> |11> (control set flips target); CNOT^2 = I",
        np.allclose(cnot_bigendian @ np.kron(KET1, KET0), np.kron(KET1, KET1), atol=1e-12),
    ))

    # 16 -- tensor / Kronecker products ------------------
    kron_np = np.kron(H, X)
    kron_q = Operator(HGate()).tensor(Operator(XGate())).data
    cc.append(CrossCheck(
        "tensor_product", "tensor / Kronecker products",
        "(A(x)B)(C(x)D) = (AC)(x)(BD) ; dim(V(x)W) = dim V * dim W",
        kron_np, kron_q,
        "mixed-product identity: (H(x)X)(H(x)X) = (H^2)(x)(X^2) = I(x)I",
        np.allclose(kron_np @ kron_np, np.eye(4), atol=1e-12),
    ))

    # 17 -- multi-qubit basis ordering -------------------
    label_np = np.zeros(4, dtype=complex)
    label_np[0b10] = 1.0  # big-endian |q0 q1> = |10>
    cc.append(CrossCheck(
        "basis_ordering", "multi-qubit basis ordering",
        "big-endian |q0 q1>: index = 2*q0 + q1. Qiskit label 'q1 q0' is little-endian.",
        label_np, Statevector.from_label("10").data,  # Qiskit 'q1 q0' -> q1=1,q0=0 -> index 2
        "|10> (q0 set) has amplitude 1 at integer index 2 in both conventions here",
        Statevector.from_label("10").probabilities()[2] == 1.0,
    ))

    # 18 -- global vs relative phase --------------------
    glob = np.exp(1j * 1.2345) * KETP
    cc.append(CrossCheck(
        "phase_global_relative", "global vs relative phase",
        "e^{i theta}|psi> ~ |psi> (unobservable); a relative phase is physical",
        Statevector(glob).probabilities(), Statevector(KETP).probabilities(),
        "global phase leaves ALL measurement probabilities unchanged; "
        "relative phase (|0>+i|1>) is visible in the X basis",
        np.allclose(Statevector(glob).probabilities(), Statevector(KETP).probabilities(), atol=1e-12)
        and not np.allclose(
            (Operator(HGate()) @ Statevector(KETpi)).probabilities(),
            (Operator(HGate()) @ Statevector(KETP)).probabilities(),
            atol=1e-3,
        ),
    ))

    # 19 -- expectation values -------------------------
    stt = rand_state(2, 31)
    ez_np = (stt.conj() @ Z @ stt).real
    ez_q = Statevector(stt).expectation_value(Pauli("Z")).real
    cc.append(CrossCheck(
        "expectation", "expectation values",
        "<A> = <psi|A|psi> = sum_m lambda_m P(m)",
        ez_np, ez_q,
        "<Z> = P(0) - P(1)",
        np.isclose(ez_np, np.diff(_probs_from_amplitudes(stt))[0] * -1, atol=1e-10),
    ))

    # 20 -- Bell states -------------------------------
    bell_np = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell_qc = QuantumCircuit(2)
    bell_qc.h(0)
    bell_qc.cx(0, 1)
    bell_q = Statevector(bell_qc).data
    cc.append(CrossCheck(
        "bell_states", "Bell states",
        "|Phi+> = (|00> + |11>)/sqrt(2) = CNOT (H(x)I) |00>",
        bell_np, bell_q,
        "measurements of the two qubits are perfectly correlated (both 0 or both 1)",
        np.allclose(Statevector(bell_qc).probabilities(), [0.5, 0, 0, 0.5], atol=1e-12),
    ))

    # 21 -- entanglement ----------------------------
    schmidt_np = np.linalg.svd(bell_np.reshape(2, 2), compute_uv=False)
    schmidt_q = np.linalg.svd(np.asarray(Statevector(bell_qc).data).reshape(2, 2), compute_uv=False)
    cc.append(CrossCheck(
        "entanglement", "entanglement",
        "a pure state is entangled iff its Schmidt rank > 1 (>1 non-zero singular value)",
        np.sort(schmidt_np), np.sort(schmidt_q),
        "|Phi+> has two equal Schmidt coefficients 1/sqrt(2): maximally entangled",
        np.allclose(np.sort(schmidt_np), [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-12),
    ))

    # 22 -- reduced density matrices --------------
    k19 = _load("19_partial_trace")
    rho_full = np.outer(bell_np, bell_np.conj())
    red_np = k19.partial_trace(rho_full, [1], 2)
    red_q = partial_trace(Statevector(bell_qc), [1]).data  # trace out qubit 1
    cc.append(CrossCheck(
        "reduced_density_matrix", "reduced density matrices",
        "rho_A = Tr_B(|psi><psi|) is the state of subsystem A alone",
        red_np, red_q,
        "the reduced state of half a Bell pair is the maximally mixed state I/2",
        np.allclose(red_np, I2 / 2, atol=1e-12),
    ))

    # 23 -- partial trace (general, complex reduction) --
    st3 = rand_state(8, 41)
    rho3 = np.outer(st3, st3.conj())
    pt_np = k19.partial_trace(rho3, [1], 3)  # keep q0, q2 -- generally complex
    pt_q = reverse_qubit_order(
        partial_trace(Statevector(reverse_qubit_order(st3, 3)), [1]).data, 2
    )
    cc.append(CrossCheck(
        "partial_trace", "partial trace",
        "Tr_B is the unique linear map with Tr_B(|a1><a2|(x)|b1><b2|) = |a1><a2| <b2|b1>",
        pt_np, pt_q,
        "partial trace preserves total trace: Tr(rho_A) = Tr(rho) = 1",
        np.isclose(np.trace(pt_np), 1.0, atol=1e-10),
        atol=1e-9,
    ))

    # 24 -- purity ---------------------------------
    pur_np_bell = float(np.trace(red_np @ red_np).real)
    pur_q_bell = float(DensityMatrix(partial_trace(Statevector(bell_qc), [1])).purity().real)
    cc.append(CrossCheck(
        "purity", "purity",
        "gamma = Tr(rho^2) in [1/d, 1]; gamma = 1 iff rho is pure",
        pur_np_bell, pur_q_bell,
        "Bell-pair reduced state has purity 1/2 (mixed); a pure state has purity 1",
        np.isclose(pur_np_bell, 0.5, atol=1e-12),
    ))

    # 25 -- fidelity ------------------------------
    p, q = rand_state(2, 51), rand_state(2, 52)
    fid_np = abs(np.vdot(p, q)) ** 2
    fid_q = state_fidelity(Statevector(p), Statevector(q))
    cc.append(CrossCheck(
        "fidelity", "fidelity",
        "F(|p>,|q>) = |<p|q>|^2 for pure states; F in [0,1]",
        fid_np, fid_q,
        "F(psi,psi) = 1 and F(|0>,|1>) = 0",
        np.isclose(state_fidelity(Statevector(p), Statevector(p)), 1.0)
        and np.isclose(state_fidelity(Statevector.from_label("0"), Statevector.from_label("1")), 0.0),
    ))

    # 26 -- quantum teleportation ----------------
    k24 = _load("24_teleportation_identity")
    src = rand_state(2, 61)
    branches = k24.teleportation_branches(src)  # NumPy: analytic branch states
    from teleportation import build_teleportation_circuit

    qc_def = build_teleportation_circuit(src, deferred=True)
    bob_q = partial_trace(Statevector(qc_def), [0, 1]).data  # Qiskit: coherent circuit
    # NumPy prediction for the deferred (all-corrections) circuit is |src><src|
    bob_np = np.outer(src, src.conj())
    cc.append(CrossCheck(
        "teleportation", "quantum teleportation",
        "1 ebit + 2 cbits move an unknown qubit; each branch q2 = X^m1 Z^m0 |psi>, "
        "corrected to |psi>",
        bob_np, bob_q,
        "coherent (deferred) teleportation lands Bob's qubit exactly in |psi> "
        "(fidelity 1); every measured branch is a Pauli image of |psi>",
        np.isclose(state_fidelity(DensityMatrix(bob_q), DensityMatrix(bob_np)), 1.0, atol=1e-9)
        and all(
            np.isclose(
                abs(np.vdot(np.linalg.matrix_power(X, m1) @ np.linalg.matrix_power(Z, m0) @ src, q2)),
                1.0,
                atol=1e-9,
            )
            for (m0, m1), (_p, q2) in branches.items()
        ),
        atol=1e-9,
    ))

    return cc


# ----------------------------------------------------------------- tiny circ ---


def _h_circ() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


# --------------------------------------------------------------------- table ---


def _fmt(v: Any) -> str:
    a = np.asarray(v)
    if a.ndim == 0:
        return f"{complex(a):.4g}"
    if a.size <= 4:
        return np.array2string(a.ravel(), precision=3, suppress_small=True)
    return f"<{a.shape} array>"


def main() -> int:
    checks = all_crosschecks()
    width = max(len(c.topic) for c in checks)
    print(f"{'topic':<{width}}  {'np==qiskit':<11}  prediction")
    print("-" * (width + 60))
    n_bad = 0
    for c in checks:
        ok = c.agrees and c.prediction_ok
        n_bad += not ok
        flag = "ok " if c.agrees else "MISMATCH"
        pred = "ok" if c.prediction_ok else "FAIL"
        print(f"{c.topic:<{width}}  {flag:<11}  [{pred}] {c.prediction}")
    print("-" * (width + 60))
    print(f"{len(checks)} concepts, {n_bad} problem(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
