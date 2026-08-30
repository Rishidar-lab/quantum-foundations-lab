"""Lesson content: F1 -> F24 -> T -> experimental labs.

Each `Lesson` has the ten required fields. Field 7 ("actual result") is a
callable so the runner shows *real* output, not a transcribed number. Field 10's
solution hint is only surfaced by `--reveal`.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from _deps import kata as _load


@dataclasses.dataclass
class Lesson:
    n: int
    node: str
    title: str
    concept: str
    why: str
    mathematics: str
    worked_example: str
    code: str
    prediction: str
    actual: Callable[[], str]
    interpretation: str
    misconception: str
    challenge: str
    solution_hint: str


# --- helpers used by the `actual` callables --------------------------------

_KET0 = np.array([1, 0], dtype=complex)
_KET1 = np.array([0, 1], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _fmt(x) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


LESSONS: list[Lesson] = [
    Lesson(
        1, "F1", "Complex amplitudes and the state vector",
        concept="A pure qubit state is a unit vector in C^2: |psi> = alpha|0> + beta|1>, "
        "with alpha, beta complex and |alpha|^2 + |beta|^2 = 1.",
        why="Everything downstream -- gates, measurement, entanglement -- is linear "
        "algebra on this vector. If the state model is wrong, nothing else can be right.",
        mathematics="C^2 with the Hermitian inner product <a|b> = sum_i conj(a_i) b_i. "
        "Norm ||psi|| = sqrt(<psi|psi>). Physical states have norm 1.",
        worked_example="alpha = 0.6, beta = 0.8i: |0.6|^2 + |0.8i|^2 = 0.36 + 0.64 = 1. Valid.",
        code="katas/01_hilbert_space.py :: inner_product, norm   (tested in test_01_hilbert_space.py)",
        prediction="norm([0.6, 0.8j]) should be exactly 1.0; norm([1, 1]) should be sqrt(2) ~ 1.4142.",
        actual=lambda: (
            f"norm([0.6, 0.8j]) = {_load('01_hilbert_space').norm(np.array([0.6, 0.8j])):.6f}\n"
            f"norm([1, 1])      = {_load('01_hilbert_space').norm(np.array([1, 1])):.6f}"
        ),
        interpretation="The inner product is the single primitive: norm, orthogonality, "
        "measurement probability and fidelity are all built from it.",
        misconception="'Amplitudes are probabilities.' They are not -- amplitudes are complex "
        "and can interfere (cancel). Probabilities are |amplitude|^2.",
        challenge="Write down a normalised state whose Z-basis measurement gives P(0) = 1/3, "
        "and confirm its norm with katas/01.",
        solution_hint="alpha = sqrt(1/3), beta = sqrt(2/3) (any phase on beta works). "
        "norm = sqrt(1/3 + 2/3) = 1.",
    ),
    Lesson(
        2, "F1", "Bras, inner products, orthogonality",
        concept="<a| is the conjugate transpose of |a>. <a|b> measures overlap; "
        "<a|b> = 0 means orthogonal (perfectly distinguishable).",
        why="Measurement outcomes correspond to an orthonormal basis; overlaps with that "
        "basis give the probabilities (Born rule, lesson 8).",
        mathematics="Conjugate symmetry <a|b> = conj(<b|a>); linear in the 2nd slot, "
        "conjugate-linear in the 1st; Cauchy-Schwarz |<a|b>| <= ||a|| ||b||.",
        worked_example="<0|1> = 1*0 + 0*1 = 0. <+|-> = (1*1 + 1*(-1))/2 = 0. Both orthogonal.",
        code="katas/01_hilbert_space.py :: inner_product ; foundations/crosscheck.py 'orthogonality'",
        prediction="<+|-> = 0 and |<+|0>|^2 = 1/2.",
        actual=lambda: (
            f"<+|-> = {np.vdot((_KET0+_KET1)/np.sqrt(2), (_KET0-_KET1)/np.sqrt(2)):.4f}\n"
            f"|<+|0>|^2 = {abs(np.vdot((_KET0+_KET1)/np.sqrt(2), _KET0))**2:.4f}"
        ),
        interpretation="Orthogonality is the mathematical face of 'distinguishable'. "
        "Non-orthogonal states cannot be told apart with certainty in one shot.",
        misconception="'<a|b> is a probability.' It is a complex number; |<a|b>|^2 is the probability.",
        challenge="Find a state |g> with |<g|0>|^2 = |<g|1>|^2 = 1/2 that is NOT |+> or |->.",
        solution_hint="Any |g> = (|0> + e^{i phi}|1>)/sqrt(2) with phi not in {0, pi}, e.g. |+i>.",
    ),
    Lesson(
        3, "F9", "Basis states and qubit ordering (endianness)",
        concept="n qubits have a 2^n-dimensional space with basis |b_0 ... b_{n-1}>. "
        "This project is BIG-ENDIAN: q0 is the most significant bit.",
        why="Every multi-qubit ket, gate matrix and partial trace depends on the ordering "
        "convention. Qiskit is little-endian; mixing them silently corrupts results.",
        mathematics="index(|b_0 ... b_{n-1}>) = sum_k b_k 2^(n-1-k). "
        "|q0 q1> = |q0> (x) |q1> (Kronecker product, q0 leftmost).",
        worked_example="Big-endian |101> -> 1*4 + 0*2 + 1*1 = index 5. In Qiskit's label "
        "convention the same bits would be written '101' but indexed the other way.",
        code="katas/09_computational_basis_nq.py ; the endianness contract test lives in test_09.",
        prediction="bits_to_index((1,0,1)) == 5 ; index_to_bits(5, 3) == (1,0,1).",
        actual=lambda: (
            f"bits_to_index((1,0,1)) = {_load('09_computational_basis_nq').bits_to_index((1,0,1))}\n"
            f"index_to_bits(5, 3)    = {_load('09_computational_basis_nq').index_to_bits(5, 3)}"
        ),
        interpretation="Pick one convention, write it down once, and assert it in a test. "
        "The bridge to Qiskit is `foundations.crosscheck.reverse_qubit_order`.",
        misconception="'Endianness is cosmetic.' A CNOT embedded with the wrong endianness is "
        "a different, wrong gate -- see tools/mutation_audit.py 'tensor-order' (43 tests fail).",
        challenge="Without running code: what integer index is big-endian |q0 q1 q2> = |011>? "
        "And which Qiskit statevector index holds that amplitude?",
        solution_hint="Big-endian |011> = 0*4 + 1*2 + 1*1 = 3. Reversing the bits for Qiskit "
        "little-endian gives '110' = 6.",
    ),
    Lesson(
        4, "F13", "Superposition and the Hadamard gate",
        concept="H = (1/sqrt2)[[1,1],[1,-1]] creates equal superposition: H|0> = |+>, H|1> = |->. "
        "H is Hermitian and unitary, so H^2 = I.",
        why="Teleportation needs the sender to put her qubit in the X basis; H is the Z<->X "
        "basis change. It also builds the Bell pair.",
        mathematics="H X H = Z, H Z H = X (H swaps the roles of bit-flip and phase-flip). "
        "det(H) = -1.",
        worked_example="H|0> = (|0>+|1>)/sqrt2. Apply H again: (H|0> + H|1>)/sqrt2 = "
        "((|+>) + (|->))/sqrt2 = |0>. Interference restored the original.",
        code="katas/13_hadamard.py ; cross-checked against qiskit HGate in foundations.",
        prediction="H @ H == I (to machine precision); H @ X @ H == Z.",
        actual=lambda: (
            f"H@H - I  max|.| = {np.abs(_H @ _H - np.eye(2)).max():.2e}\n"
            f"H@X@H == Z ?  {np.allclose(_H @ _X @ _H, _Z)}"
        ),
        interpretation="Superposition is not 'the qubit is 0 and 1 at once' -- it is a definite "
        "vector whose measurement in a given basis is probabilistic.",
        misconception="'H randomises the qubit.' H is deterministic and reversible; the "
        "randomness appears only at measurement.",
        challenge="Show, with katas/13 and katas/11, that H = (X + Z)/sqrt(2).",
        solution_hint="Add the X and Z matrices, divide by sqrt(2): [[1,1],[1,-1]]/sqrt(2) = H.",
    ),
    Lesson(
        5, "F11", "The Pauli operators",
        concept="I, X, Y, Z: the 2x2 Hermitian AND unitary matrices. X = bit flip, "
        "Z = phase flip, Y = iXZ. Each squares to I, is traceless, has eigenvalues +/-1.",
        why="They are the correction operators in teleportation, the generators of "
        "single-qubit rotations, and a basis for all 2x2 Hermitian operators (observables).",
        mathematics="X^2 = Y^2 = Z^2 = I ; {X,Z} = XZ + ZX = 0 (anticommute) ; XZ = -iY.",
        worked_example="X|0> = |1>, Z|1> = -|1>. So XZ|1> = X(-|1>) = -|0>, "
        "and ZX|1> = Z|0> = |0>. Indeed XZ = -ZX on |1>.",
        code="katas/11_pauli_operators.py ; katas/12 builds the multiplication table.",
        prediction="X@Z == -1j * Y ; X and Z anticommute.",
        actual=lambda: (
            f"X@Z + Z@X  max|.| = {np.abs(_X @ _Z + _Z @ _X).max():.2e}\n"
            f"X@Z == -iY ?  {np.allclose(_X @ _Z, -1j * np.array([[0,-1j],[1j,0]]))}"
        ),
        interpretation="Anticommutation is why a single stray Pauli error is detectable: "
        "it flips the sign of a stabiliser measurement.",
        misconception="'Y is exotic.' Y = iXZ is just a bit-and-phase flip with a phase. "
        "Teleportation's four corrections are I, X, Z, XZ (= -iY).",
        challenge="Using katas/12 `pauli_mul`, what is Z * Y? Predict the phase and label first.",
        solution_hint="ZY: not identity, not cyclic (Y,Z)->X is cyclic so (Z,Y) is anti-cyclic "
        "-> phase -i, label X. So ZY = -iX.",
    ),
    Lesson(
        6, "F12", "The Pauli group",
        concept="The 16 elements {+/-1, +/-i} x {I,X,Y,Z} are closed under multiplication -- "
        "they form a group. `pauli_mul(a,b)` returns (phase, label).",
        why="Teleportation corrections compose: on branch (m0,m1) Bob undoes X^m1 Z^m0. "
        "Getting the phase/order right needs the group law.",
        mathematics="P@P = I ; I is the unit ; distinct non-I Paulis: cyclic X->Y->Z->X "
        "gives +i, the reverse gives -i.",
        worked_example="(XZ)(XZ) = X(ZX)Z = X(-XZ)Z = -(XX)(ZZ) = -I. So (X^m1 Z^m0) is its "
        "own inverse up to a sign -- which is why Bob applies Z^m0 X^m1.",
        code="katas/12_pauli_algebra.py ; 46 tests incl. closure + associativity.",
        prediction="pauli_mul('X','Y') == (1j, 'Z') ; pauli_mul('Y','X') == (-1j, 'Z').",
        actual=lambda: (
            f"pauli_mul('X','Y') = {_load('12_pauli_algebra').pauli_mul('X','Y')}\n"
            f"pauli_mul('Y','X') = {_load('12_pauli_algebra').pauli_mul('Y','X')}"
        ),
        interpretation="A finite non-abelian group with 16 elements sits underneath every "
        "stabiliser code and the whole Clifford hierarchy.",
        misconception="'Paulis commute.' They anticommute pairwise (distinct, non-I). Only the "
        "phases +/-1,+/-i commute through.",
        challenge="Prove Bob's correction Z^m0 X^m1 exactly inverts X^m1 Z^m0 (no leftover phase) "
        "for all four (m0,m1).",
        solution_hint="Z^m0 X^m1 . X^m1 Z^m0 = Z^m0 (X^m1 X^m1) Z^m0 = Z^m0 Z^m0 = I, "
        "since each Pauli squares to exactly I.",
    ),
    Lesson(
        7, "F15", "Measurement and the Born rule",
        concept="A projective measurement is a set of orthogonal projectors {P_m}, sum P_m = I. "
        "Outcome m has probability <psi|P_m|psi>; the state collapses to P_m|psi>/sqrt(p_m).",
        why="It is the only non-unitary, irreversible, probabilistic step -- and the reason "
        "teleportation destroys the source (no cloning).",
        mathematics="p_m = ||P_m |psi>||^2 >= 0, sum_m p_m = 1. For the computational basis "
        "P_m = |m><m| and p_m = |<m|psi>|^2.",
        worked_example="|+> = (|0>+|1>)/sqrt2: p_0 = |<0|+>|^2 = 1/2, p_1 = 1/2.",
        code="katas/15_measurement_postulate.py, katas/16_projective_measurement.py",
        prediction="Born probs for |+> are [0.5, 0.5]; they sum to 1.",
        actual=lambda: (
            "born_probabilities(|+>, comp) = "
            + _fmt(_load('15_measurement_postulate').born_probabilities(
                (_KET0 + _KET1) / np.sqrt(2),
                [np.outer(_KET0, _KET0), np.outer(_KET1, _KET1)]))
        ),
        interpretation="'Collapse' is a modelling rule for what to condition on after learning "
        "an outcome; it is not a physical force.",
        misconception="'Measuring in the Z basis tells you what the state was.' It tells you an "
        "outcome sampled from |amplitude|^2; the pre-measurement amplitudes are not recovered.",
        challenge="A state has Born probs [0.25, 0.75] in the Z basis. Give two states with "
        "those probs that behave differently in the X basis.",
        solution_hint="sqrt(1/4)|0> + sqrt(3/4)|1>  vs  sqrt(1/4)|0> - sqrt(3/4)|1>. Same Z "
        "probs, opposite relative phase -> different X-basis probabilities.",
    ),
    Lesson(
        8, "F16", "Expectation values",
        concept="<A> = <psi|A|psi> = sum_m lambda_m p_m is the average eigenvalue you would "
        "see over many measurements of the observable A.",
        why="Hardware never hands you a state vector -- it hands you estimated expectation "
        "values (and counts). Tomography (lesson 18) reconstructs a state from three of them.",
        mathematics="For a Pauli, <Z> = p_0 - p_1 in [-1, 1]. The Bloch vector is "
        "r = (<X>, <Y>, <Z>) and rho = (I + r . sigma)/2.",
        worked_example="<0|Z|0> = 1, <1|Z|1> = -1, <+|Z|+> = 0, <+|X|+> = 1.",
        code="katas/16_projective_measurement.py :: expectation_value",
        prediction="<+|Z|+> = 0 and <+|X|+> = 1.",
        actual=lambda: (
            f"<+|Z|+> = {_load('16_projective_measurement').expectation_value((_KET0+_KET1)/np.sqrt(2), _Z):.4f}\n"
            f"<+|X|+> = {_load('16_projective_measurement').expectation_value((_KET0+_KET1)/np.sqrt(2), _X):.4f}"
        ),
        interpretation="Expectation values are linear in the state's density matrix, which is "
        "why they survive averaging over noise and shots.",
        misconception="'<Z> = 0 means the qubit is in |0> or |1> with equal chance and nothing "
        "else.' It only constrains the Z-basis probabilities; the state could be |+>, |-i>, ...",
        challenge="What is <Y> for |+i> = (|0> + i|1>)/sqrt2? Predict, then check with katas/16.",
        solution_hint="|+i> is the +1 eigenstate of Y, so <Y> = +1 exactly.",
    ),
    Lesson(
        9, "F8", "Tensor products and composite systems",
        concept="The state space of two qubits is C^2 (x) C^2 = C^4. Independent subsystems "
        "combine as |a> (x) |b>; operators as A (x) B.",
        why="A Bell pair is a two-qubit state that is NOT a tensor product -- that is exactly "
        "what makes it a resource for teleportation.",
        mathematics="(A (x) B)(C (x) D) = (AC) (x) (BD) ; dim(V (x) W) = dim V * dim W. "
        "np.kron implements it.",
        worked_example="|0> (x) |1> = (1,0) (x) (0,1) = (0,1,0,0) = |01>. "
        "(X (x) I)|01> = |11>.",
        code="np.kron everywhere; foundations/crosscheck.py 'tensor_product' vs Qiskit .tensor().",
        prediction="(H (x) X)(H (x) X) == I_4 (mixed-product identity, since H^2 = X^2 = I).",
        actual=lambda: (
            "max|.(H(x)X)^2 - I| = "
            f"{np.abs(np.kron(_H,_X) @ np.kron(_H,_X) - np.eye(4)).max():.2e}"
        ),
        interpretation="Separable (product) states carry no correlation; entangled states carry "
        "correlations stronger than any classical model allows.",
        misconception="'Every 4-vector factorises as |a>(x)|b>.' Most do not -- e.g. "
        "(|00>+|11>)/sqrt2 has no such factorisation (Schmidt rank 2).",
        challenge="Is (|00> + |01> + |10> + |11>)/2 entangled? Factorise it or prove you cannot.",
        solution_hint="It factorises: |+> (x) |+>. Not entangled.",
    ),
    Lesson(
        10, "F14", "CNOT and controlled operations",
        concept="controlled-U = |0><0|_c (x) I + |1><1|_c (x) U_t. CNOT = controlled-X: it "
        "flips the target iff the control is |1>, and it entangles.",
        why="CNOT builds the Bell pair and performs half of Alice's Bell-basis measurement.",
        mathematics="CNOT is unitary, CNOT^2 = I, and (H (x) H) CNOT_{0->1} (H (x) H) = "
        "CNOT_{1->0}. It is NOT a tensor product of 1-qubit gates.",
        worked_example="CNOT|10> = |11>, CNOT|11> = |10> (big-endian control = q0). "
        "CNOT (|+> (x) |0>) = (|00> + |11>)/sqrt2 -- a Bell state.",
        code="katas/14_cnot_controlled_u.py :: controlled_unitary (bit-surgery), cnot, cz",
        prediction="cnot(2,0,1) @ |+0> == (|00> + |11>)/sqrt2.",
        actual=lambda: (
            "cnot @ |+0> = "
            + _fmt(_load('14_cnot_controlled_u').cnot(2, 0, 1)
                   @ np.kron((_KET0 + _KET1) / np.sqrt(2), _KET0))
        ),
        interpretation="Entanglement is generated exactly when a controlled gate acts on a "
        "superposed control -- the branches become correlated with the target.",
        misconception="'CNOT copies the control onto the target.' It does for basis states, "
        "but on |+> it entangles instead of copying (no-cloning).",
        challenge="What does CNOT do to |-> (x) |->? Predict the output, then verify with katas/14.",
        solution_hint="CNOT acts as a phase kickback: it leaves |-,-> unchanged (|-> is a -1 "
        "eigenstate of X, so the control picks up no net effect). Output = |-> (x) |->.",
    ),
    Lesson(
        11, "F18", "Density matrices",
        concept="rho = sum_i p_i |psi_i><psi_i|: Hermitian, positive semidefinite, trace 1. "
        "It describes statistical mixtures AND subsystems of entangled states.",
        why="A qubit that is half of a Bell pair has NO state vector of its own -- only a "
        "density matrix. Noise (lesson 19) also produces mixed states.",
        mathematics="Pure iff Tr(rho^2) = 1 iff rho^2 = rho. <A> = Tr(rho A). "
        "Evolution: rho -> U rho U^dagger.",
        worked_example="|+><+| = [[1,1],[1,1]]/2, Tr = 1, Tr(rho^2) = 1 (pure). "
        "I/2 = [[1,0],[0,1]]/2, Tr(rho^2) = 1/2 (maximally mixed).",
        code="katas/18_density_operator.py :: density_from_state, purity, is_density_matrix",
        prediction="purity(|+><+|) = 1 ; purity(I/2) = 0.5.",
        actual=lambda: (
            f"purity(|+><+|) = {_load('18_density_operator').purity(np.outer((_KET0+_KET1)/np.sqrt(2), (_KET0+_KET1).conj()/np.sqrt(2))):.4f}\n"
            f"purity(I/2)    = {_load('18_density_operator').purity(np.eye(2)/2):.4f}"
        ),
        interpretation="The density matrix is the honest description when you have classical "
        "uncertainty, entanglement with an inaccessible partner, or both.",
        misconception="'A mixed state is a superposition.' A superposition is a definite pure "
        "vector; a mixture is a probabilistic ensemble and generally has purity < 1.",
        challenge="Give a pure state and a mixed state that have the SAME Z-basis probabilities "
        "[0.5, 0.5] but different purity.",
        solution_hint="Pure: |+> (purity 1). Mixed: I/2 (purity 0.5). Same diagonal, different "
        "off-diagonal (coherence).",
    ),
    Lesson(
        12, "F19", "Partial trace and reduced states",
        concept="rho_A = Tr_B(rho_AB) is the state of subsystem A alone -- it reproduces every "
        "measurement statistic restricted to A.",
        why="To ask 'what does Bob have?' after teleportation you trace out Alice's two qubits. "
        "The reduced state IS the teleported qubit.",
        mathematics="Tr_B(|a1><a2| (x) |b1><b2|) = |a1><a2| <b2|b1>. Linear; trace-preserving.",
        worked_example="For |Phi+> = (|00>+|11>)/sqrt2: rho = |Phi+><Phi+|, and "
        "Tr_B(rho) = (|0><0| + |1><1|)/2 = I/2.",
        code="katas/19_partial_trace.py (einsum index identification, big-endian)",
        prediction="Reducing either half of |Phi+> gives I/2.",
        actual=lambda: (
            "Tr_{q1}(|Phi+><Phi+|) = "
            + _fmt(_load('19_partial_trace').partial_trace(
                np.outer(np.array([1,0,0,1])/np.sqrt(2), np.array([1,0,0,1])/np.sqrt(2)), [1], 2))
        ),
        interpretation="Local ignorance from entanglement: the whole is pure and perfectly "
        "known, yet each part is maximally mixed.",
        misconception="'Tracing out a qubit is the same as measuring and forgetting it.' "
        "Measuring-and-forgetting also gives rho_A, but partial trace needs no measurement basis.",
        challenge="For a product state |a> (x) |b>, what is Tr_B? Predict, then check with katas/19.",
        solution_hint="Tr_B(|a><a| (x) |b><b|) = |a><a| * Tr(|b><b|) = |a><a| -- a's own pure state.",
    ),
    Lesson(
        13, "F20", "Entanglement entropy",
        concept="The entanglement across a cut = the von Neumann entropy S(rho_A) = "
        "-sum lambda_i log2 lambda_i of the reduced state. 0 = separable, 1 bit = "
        "maximally entangled (for a qubit).",
        why="It quantifies the RESOURCE teleportation consumes: exactly 1 ebit per qubit sent.",
        mathematics="For a pure global state S(rho_A) = S(rho_B) (Schmidt). "
        "S(I/d) = log2 d ; S(pure) = 0.",
        worked_example="Bell pair: rho_A = I/2, eigenvalues (1/2, 1/2), "
        "S = -(1/2)log2(1/2) - (1/2)log2(1/2) = 1 bit.",
        code="katas/20_entanglement_entropy.py :: von_neumann_entropy, entanglement_entropy",
        prediction="S for a Bell state = 1.0 ; S for |+>(x)|+> = 0.0.",
        actual=lambda: (
            f"S(Bell)     = {_load('20_entanglement_entropy').entanglement_entropy(np.array([1,0,0,1])/np.sqrt(2), [0], 2):.4f}\n"
            f"S(|+>(x)|+>) = {_load('20_entanglement_entropy').entanglement_entropy(np.kron((_KET0+_KET1)/np.sqrt(2),(_KET0+_KET1)/np.sqrt(2)), [0], 2):.4f}"
        ),
        interpretation="Entanglement entropy is symmetric across the cut and monotone under "
        "local operations -- you cannot manufacture it with local gates + classical comms.",
        misconception="'More correlated = more entangled.' Classical correlation (a shared coin "
        "flip) has S = 0 for the pure global description; only quantum correlation counts here.",
        challenge="A state has S = 0.5 across the (q0 | q1) cut. Is it pure? Is it entangled?",
        solution_hint="If the GLOBAL 2-qubit state is pure, S in (0,1) means partially entangled. "
        "S = 0.5 -> reduced eigenvalues about (0.89, 0.11).",
    ),
    Lesson(
        14, "F21", "Bell states and the Bell pair",
        concept="The four maximally entangled two-qubit states {Phi+, Phi-, Psi+, Psi-} form an "
        "orthonormal basis. |Phi+> = CNOT (H (x) I) |00>.",
        why="Teleportation = Alice measuring her two qubits in THIS basis; the outcome tells "
        "Bob which Pauli to undo.",
        mathematics="beta(b0,b1) = CNOT (H (x) I) |b0 b1>. The four are related by local Paulis: "
        "(X (x) I)|Phi+> = |Psi+>, (Z (x) I)|Phi+> = |Phi->.",
        worked_example="Phi+ = (|00>+|11>)/sqrt2. Apply Z to the first qubit: "
        "(|00> - |11>)/sqrt2 = Phi-.",
        code="katas/21_bell_states.py, katas/22_bell_pair_prep.py ; literals in test_foundations_weakness_fixes.py",
        prediction="bell_state(0,0) == [1,0,0,1]/sqrt2 ; all four are orthonormal.",
        actual=lambda: (
            "bell_state(0,0) = " + _fmt(_load('21_bell_states').bell_state(0, 0)) + "\n"
            "bell_basis is unitary ? "
            + str(np.allclose(_load('21_bell_states').bell_basis().conj().T
                              @ _load('21_bell_states').bell_basis(), np.eye(4)))
        ),
        interpretation="The Bell basis diagonalises the 'which local Pauli was applied' question "
        "-- that is precisely the information Alice sends Bob.",
        misconception="'Measuring the Bell pair sends a signal.' Alice's marginal outcome is "
        "50/50 regardless of Bob -- no information moves until the classical bits arrive.",
        challenge="Which Bell state is (Y (x) I)|Phi+> (up to phase)? Predict, verify with katas/21.",
        solution_hint="Y = iXZ, so (Y(x)I)|Phi+> = i (X(x)I)(Z(x)I)|Phi+> = i(X(x)I)|Phi-> "
        "= i|Psi-> (up to the global i).",
    ),
    Lesson(
        15, "F4", "Global vs relative phase",
        concept="e^{i theta}|psi> is physically identical to |psi> (global phase, unobservable). "
        "A RELATIVE phase between components, e.g. |0> + e^{i phi}|1>, is physical.",
        why="Teleportation's standard correction order (Z^m0 X^m1) recovers |psi> exactly. A "
        "swapped order recovers it up to a harmless global -1 on one branch -- still correct.",
        mathematics="Measurement probabilities depend on |amplitude|^2, killing any overall "
        "phase. Interference between components keeps the relative phase.",
        worked_example="|+> and -|+> give identical statistics everywhere. |0>+|1> and |0>-|1> "
        "(= |+> vs |->) are perfectly distinguishable in the X basis.",
        code="foundations/crosscheck.py 'phase_global_relative' ; mutation 'global-phase--negate-branch-state' is correctly INVISIBLE.",
        prediction="probs(|+>) == probs(-|+>) exactly; probs_X(|+>) != probs_X(|->).",
        actual=lambda: (
            f"probs(|+>) vs probs(-|+>) equal ? "
            f"{np.allclose(np.abs((_KET0+_KET1)/np.sqrt(2))**2, np.abs(-(_KET0+_KET1)/np.sqrt(2))**2)}\n"
            f"H|+> = {_fmt(_H @ (_KET0+_KET1)/np.sqrt(2))}  ;  H|-> = {_fmt(_H @ (_KET0-_KET1)/np.sqrt(2))}"
        ),
        interpretation="'Up to global phase' is not sloppiness -- it is the statement that states "
        "are rays, not vectors. Tests here compare with |<a|b>| or fidelity for that reason.",
        misconception="'Any phase is unobservable.' Only the OVERALL phase is. Relative phases "
        "drive every interference experiment.",
        challenge="The teleportation integration test asserts fidelity = 1, not statevector "
        "equality. Why is that the right choice?",
        solution_hint="The circuit can legitimately produce e^{i g}|psi> on Bob's qubit; fidelity "
        "|<psi|out>|^2 = 1 captures 'same physical state', raw allclose would spuriously fail.",
    ),
    Lesson(
        16, "F24", "The teleportation identity",
        concept="For any |psi>, after H1, CNOT(1->2), CNOT(0->1), H0 on |psi>(x)|Phi+>, the "
        "state is (1/2) sum_{m0,m1} |m0 m1> (x) (X^m1 Z^m0 |psi>).",
        why="This one algebraic rewrite IS teleportation: each of Alice's four equally likely "
        "outcomes leaves Bob holding a known Pauli image of |psi>.",
        mathematics="Expand |psi>(x)|Phi+> in the Bell basis of (q0,q1); the coefficient of "
        "|beta(m0,m1)> is (1/2) X^m1 Z^m0 |psi>. Probabilities are all 1/4 (no signalling).",
        worked_example="psi = a|0> + b|1>. Branch (0,0): a|0> + b|1> = psi. Branch (1,1): "
        "a|1> - b|0> = XZ psi. Bob undoes with Z^0 X^0 = I and Z^1 X^1 respectively.",
        code="katas/24_teleportation_identity.py :: pre_measurement_state, teleportation_branches ; "
        "closed-form check in test_foundations_weakness_fixes.py",
        prediction="All four branch probabilities equal 1/4 for a random psi.",
        actual=lambda: (
            "branch probs = "
            + str({k: round(v[0], 4) for k, v in
                   _load('24_teleportation_identity').teleportation_branches(
                       np.array([0.5 + 0.3j, 0.6 - 0.55j]) / np.linalg.norm([0.5 + 0.3j, 0.6 - 0.55j])
                   ).items()})
        ),
        interpretation="No cloning: the source qubit q0 ends up maximally mixed (I/2) -- all its "
        "information has moved to q2. Nothing was copied.",
        misconception="'Teleportation is faster-than-light.' The two classical bits travel at "
        "<= c; without them Bob's qubit is I/2 (useless).",
        challenge="Show Alice's outcome q0 alone is 50/50 for EVERY input psi (no-signalling).",
        solution_hint="Marginal P(m0) = sum_{m1} (1/4) ||X^m1 Z^m0 psi||^2 = sum_{m1} 1/4 = 1/2, "
        "since Paulis are unitary and ||psi|| = 1.",
    ),
    Lesson(
        17, "T", "Classical control, deferred measurement, the full circuit",
        concept="Alice measures (q0,q1) -> two classical bits; Bob applies X if m1, then Z if "
        "m0. The 'principle of deferred measurement' lets you replace this with coherent "
        "CX(q1->q2), CZ(q0->q2) for analysis.",
        why="This is the assembled protocol -- and the deferred form gives an exact statevector "
        "you can check without sampling.",
        mathematics="U_tele (|psi> (x) |00>) = |+>_q0 (x) |+>_q1 (x) |psi>_q2 (coherent form). "
        "Classical and coherent forms have identical output statistics.",
        worked_example="Branch (1,1): q2 = XZ psi. Bob applies X (m1=1): -> Z psi. Then Z "
        "(m0=1): -> psi. Recovered exactly.",
        code="teleportation/circuit.py :: build_teleportation_circuit ; katas/23_classical_control_deferred.py",
        prediction="Deferred circuit puts Bob's reduced state at fidelity 1 with the input.",
        actual=lambda: _deferred_teleport_fidelity(),
        interpretation="Dynamic circuits (mid-circuit measure + feed-forward) are real hardware "
        "features now; the deferred trick is an analysis and testing convenience.",
        misconception="'Deferred measurement changes the physics.' It changes only WHEN you "
        "measure; the marginal statistics on every wire are identical.",
        challenge="Run `python -m experiments.teleport_lab --shots 4096` and explain why F_ideal "
        "is exactly 1 but F_emp is not.",
        solution_hint="F_ideal is an exact statevector calculation; F_emp is a finite-shot "
        "tomography estimate with sampling noise ~ 1/sqrt(shots).",
    ),
    Lesson(
        18, "LAB", "Teleportation as an experiment: shots, CIs, fidelity",
        concept="A statevector PROBABILITY is exact. A finite-shot FREQUENCY is an estimate "
        "with a confidence interval. They are related, not identical.",
        why="Every real result -- simulator with shots or hardware -- is a frequency. Reporting "
        "it without an interval is reporting noise as signal.",
        mathematics="For k successes in n trials, the Wilson 95% interval is a well-behaved "
        "CI for the underlying probability p. Width ~ 1/sqrt(n).",
        worked_example="8192 shots, 8150 zeros -> p_hat = 0.9949, Wilson 95% ~ [0.992, 0.997].",
        code="experiments/teleport_lab.py, experiments/stats.py :: wilson_interval",
        prediction="Alice's 4 outcomes pass a chi-square uniformity test; F_ideal = 1 exactly.",
        actual=lambda: "run:  python -m experiments.teleport_lab --shots 4096   (prints the table)",
        interpretation="3-basis tomography can even return F_emp slightly > 1 -- an unphysical "
        "estimate that honest error bars must accommodate.",
        misconception="'The simulator agrees with theory exactly, so shots do too.' Shots have "
        "sampling error; only the underlying probability matches theory.",
        challenge="How many shots to pin teleportation fidelity to +/- 0.01 at 95% confidence?",
        solution_hint="Worst case p=0.5: n ~ (1.96 / 0.01)^2 * 0.25 ~ 9600 shots. Near p=1 it is "
        "far fewer.",
    ),
    Lesson(
        19, "LAB", "Noise and density matrices",
        concept="Real qubits decohere. Channels: bit/phase flip, depolarizing, amplitude "
        "damping (T1), phase damping (T2), readout error. Each is a CPTP map rho -> sum K rho K^dagger.",
        why="It explains WHY hardware fidelity < 1 and lets you predict the degradation before "
        "you spend queue time.",
        mathematics="Completeness sum_k K_k^dagger K_k = I. Metrics: fidelity, purity Tr(rho^2), "
        "Bloch-vector shrinkage <X>,<Y>,<Z>.",
        worked_example="Amplitude damping gamma=1 sends any state to |0><0|. Phase damping "
        "keeps the populations, kills the coherences.",
        code="experiments/channels.py (NumPy Kraus) cross-checked vs qiskit_aer.noise ; experiments/noise_lab.py",
        prediction="phase_flip leaves <Z> unchanged; depolarizing p=1 gives I/2.",
        actual=lambda: "run:  python -m experiments.noise_lab   (prints channel sweeps + teleportation degradation)",
        interpretation="Different channels degrade different Bloch components -- the shape of the "
        "degradation identifies the dominant error.",
        misconception="'Noise just adds randomness to the outcome.' It maps the state to a "
        "different (mixed) state; the whole distribution changes, coherently-derived terms most.",
        challenge="Predict the sign of the change in <Z> for |+> under (a) amplitude damping, "
        "(b) phase damping, (c) bit flip.",
        solution_hint="(a) <Z> increases toward +1 (relaxation to |0>). (b) <Z> unchanged. "
        "(c) <Z> unchanged for |+> (already <Z>=0), but coherence in X is unaffected while... "
        "actually bit-flip leaves |+> invariant. Check with noise_lab.",
    ),
    Lesson(
        20, "LAB", "The hardware pipeline",
        concept="explicit linear algebra -> ideal Qiskit -> shot sim -> noisy sim -> IBM "
        "hardware. Each stage is a check on the previous one.",
        why="The project must work fully offline; hardware is the last, optional, credential- "
        "and confirmation-gated stage. Never spend queue time by accident.",
        mathematics="The observable P(out=0) for build_verification_circuit is the teleportation "
        "fidelity: perfect teleportation returns Bob's qubit to |0> after un-preparation.",
        worked_example="stage 1 (NumPy) = 1.0 exact ; stage 2 (Statevector) = 1.0 exact ; "
        "stage 3 (8192 shots) = 1.000 CI [0.9995, 1.0] ; stage 4 (noisy) ~ 0.97.",
        code="jobs/pipeline.py (stages 1-4, offline), jobs/preflight.py, jobs/submit_ibm_job.py, jobs/compare_counts.py",
        prediction="stage 1 == stage 2 to 1e-9 ; stage 3's CI contains stage 2.",
        actual=lambda: "run:  python jobs/pipeline.py --state plus",
        interpretation="Cross-checking stages catches convention bugs (endianness), transpilation "
        "surprises, and -- at stage 5 -- real device error.",
        misconception="'If credentials are set, the tool should just run on hardware.' No: "
        "submission needs an explicit spend confirmation every time.",
        challenge="Run preflight, then pipeline for |+i>, and state what a real backend would "
        "have to achieve for compare_counts.py to PASS at tol 0.05.",
        solution_hint="P_hardware(out=0) within 0.05 of the noiseless-shot value (~1.0), i.e. "
        "device teleportation fidelity >= ~0.95 for that circuit.",
    ),
]


def _deferred_teleport_fidelity() -> str:
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

    from teleportation import build_teleportation_circuit

    psi = np.array([0.5 + 0.3j, 0.6 - 0.55j])
    psi = psi / np.linalg.norm(psi)
    qc = build_teleportation_circuit(psi, deferred=True)
    bob = partial_trace(Statevector(qc), [0, 1])
    f = state_fidelity(bob, DensityMatrix(np.outer(psi, psi.conj())), validate=False)
    return f"deferred teleportation: fidelity(Bob, input) = {f:.10f}"


def lesson_by_number(n: int) -> Lesson:
    for lesson in LESSONS:
        if lesson.n == n:
            return lesson
    raise KeyError(n)
