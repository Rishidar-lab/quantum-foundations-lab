# Dependency Map — Quantum Teleportation Protocol

Reverse-engineering target: 3-qubit teleportation circuit
(state prep → Bell-pair prep → CNOT+H "Bell measurement" → mid-circuit
measurement → classically-controlled Pauli correction).

This document is the **map**: gate decomposition, backward dependency walk to
QM postulates, per-node specs, and test strategy. No implementation.

---

## 0. Conventions & Build Decisions

- **Qubit register:** `q0, q1, q2`.
  - `q0` = payload; holds the unknown state `|ψ⟩ = α|0⟩ + β|1⟩`, `|α|²+|β|²=1`.
  - `q1` = Alice's half of the entangled resource.
  - `q2` = Bob's half of the entangled resource; teleportation destination.
- **Tensor ordering:** big-endian, `|q0⟩ ⊗ |q1⟩ ⊗ |q2⟩`, matching Nielsen &
  Chuang. Basis index `b(q0)·4 + b(q1)·2 + b(q2)`. **Decision for build:** if
  the eventual code uses Qiskit, its statevectors are little-endian
  (`q2⊗q1⊗q0`); every matrix and expected ket below must be bit-reversed or the
  simulator wrapped. Pick one convention in the test harness and assert it once.
- **Gate set (primitive):** `I, X, Z` (2×2), `H` (2×2), `CNOT` (4×4,
  control→target). `Y` is defined for completeness but is **off the critical
  path** — teleportation corrections use only `X` and `Z`.
- **Correction operator:** for Alice's outcome `(m0, m1)` (m0 from q0, m1 from
  q1), the pre-correction state of q2 is `X^{m1} Z^{m0} |ψ⟩`. Bob applies the
  inverse `Z^{m0} X^{m1}` — i.e. **X first (if m1=1), then Z (if m0=1)**. This
  recovers `|ψ⟩` *exactly*, with no residual global phase.
- **Two dependency branches** (see §2): the **execution branch** (what the
  circuit needs to *run*: F1–F17, F21–F24) and the **characterization branch**
  (what "the resource is entangled" *means* and how to measure it: F18–F20).
  The second is not required to build/verify the circuit numerically but is
  required to justify why it works and to test the Bell pair.

---

## 1. Target Decomposition (gate-by-gate, execution order)

### 1.1 Circuit table

| # | Op | Operand(s) | 8×8 embedding (big-endian) | Purpose |
|---|-----|-----------|----------------------------|---------|
| 0 | init | — | `|000⟩` | register reset |
| 1 | `U_ψ` | q0 | `U_ψ ⊗ I ⊗ I` | prepare unknown payload (arbitrary SU(2) for testing) |
| 2 | `H` | q1 | `I ⊗ H ⊗ I` | begin Bell-pair prep |
| 3 | `CNOT` | q1→q2 | `I ⊗ (|0⟩⟨0|⊗I + |1⟩⟨1|⊗X)` | finish Bell pair `|Φ⁺⟩_{q1q2}` |
| 4 | `CNOT` | q0→q1 | `(|0⟩⟨0|⊗I + |1⟩⟨1|⊗X) ⊗ I` | Bell-basis measurement, part 1 |
| 5 | `H` | q0 | `H ⊗ I ⊗ I` | Bell-basis measurement, part 2 |
| 6 | measure | q0 → `m0` | `P = |m0⟩⟨m0| ⊗ I ⊗ I` | project, emit classical bit |
| 7 | measure | q1 → `m1` | `P = I ⊗ |m1⟩⟨m1| ⊗ I` | project, emit classical bit |
| 8 | `X` if `m1` | q2 | `I ⊗ I ⊗ X^{m1}` | Pauli correction (bit) |
| 9 | `Z` if `m0` | q2 | `I ⊗ I ⊗ Z^{m0}` | Pauli correction (phase) |
| — | result | q2 | — | `q2 == |ψ⟩`; q0,q1 collapsed & unentangled |

Steps 2–3 = **Bell-pair preparation**. Steps 4–5 (+6–7) = **Bell-basis
measurement** of `(q0,q1)` expressed in the computational basis. Steps 8–9 =
**classically-controlled correction**.

### 1.2 Exact state transformation per step

Let `|ψ⟩ = α|0⟩ + β|1⟩`.

- **After 1:** `(α|0⟩ + β|1⟩)_{q0} ⊗ |0⟩_{q1} ⊗ |0⟩_{q2}`
  `= α|000⟩ + β|100⟩`.

- **After 2** (`H` on q1, `H|0⟩ = (|0⟩+|1⟩)/√2`):
  `(α|0⟩+β|1⟩) ⊗ (|0⟩+|1⟩)/√2 ⊗ |0⟩`
  `= (1/√2)(α|000⟩ + α|010⟩ + β|100⟩ + β|110⟩)`.

- **After 3** (`CNOT` q1→q2, flips q2 iff q1=1):
  `(α|0⟩+β|1⟩) ⊗ (|00⟩+|11⟩)/√2`
  `= (1/√2)(α|000⟩ + α|011⟩ + β|100⟩ + β|111⟩)`.
  Now `q1,q2` are in `|Φ⁺⟩`; `q0` still separable.

- **After 4** (`CNOT` q0→q1, flips q1 iff q0=1):
  `(1/√2)(α|000⟩ + α|011⟩ + β|110⟩ + β|101⟩)`.

- **After 5** (`H` on q0, `H|0⟩=(|0⟩+|1⟩)/√2`, `H|1⟩=(|0⟩−|1⟩)/√2`):

  ```
  (1/2) [ |00⟩(α|0⟩ + β|1⟩)      ← q0q1 = 00 : I |ψ⟩
        + |01⟩(α|1⟩ + β|0⟩)      ← q0q1 = 01 : X |ψ⟩
        + |10⟩(α|0⟩ − β|1⟩)      ← q0q1 = 10 : Z |ψ⟩
        + |11⟩(α|1⟩ − β|0⟩) ]    ← q0q1 = 11 : X Z |ψ⟩
  ```
  (kets = `q0q1`; parenthetical = `q2`). This is the **teleportation
  identity** in computational form. Equivalently `q2 = X^{m1} Z^{m0} |ψ⟩`.

- **After 6–7** (project `q0→m0`, `q1→m1`): each of the 4 outcomes occurs with
  probability `(1/2)² · ⟨ψ|(X^{m1}Z^{m0})†(X^{m1}Z^{m0})|ψ⟩ = 1/4`,
  **independent of α, β**. Post-measurement (renormalized):
  `|m0⟩_{q0} ⊗ |m1⟩_{q1} ⊗ (X^{m1} Z^{m0} |ψ⟩)_{q2}`.

- **After 8** (`X` on q2 iff `m1=1`): `q2 → X^{m1} · X^{m1} Z^{m0} |ψ⟩ = Z^{m0} |ψ⟩`.

- **After 9** (`Z` on q2 iff `m0=1`): `q2 → Z^{m0} · Z^{m0} |ψ⟩ = |ψ⟩`. ∎

### 1.3 Coherent (deferred-measurement) equivalent — for exact testing

Replace steps 6–9 with unitaries: `CNOT(q1→q2)` then `CZ(q0→q2)`. Then
`U_tele (|ψ⟩ ⊗ |00⟩) = |+⟩_{q0} ⊗ |+⟩_{q1} ⊗ |ψ⟩_{q2}` for all `|ψ⟩`. Fully
factorized output ⇒ a single `np.allclose` on the 8-vector settles end-to-end
correctness without sampling.

---

## 2. Reverse Dependency Graph

### 2a. Nested bullets (target → postulates)

Legend: `[P]` = QM postulate, `[M]` = linear-algebra primitive, `[D]` = derived
concept. Recursion stops at `[P]`/`[M]`.

- **Quantum Teleportation Protocol** `[D]`
  - **Payload state prep `U_ψ|0⟩`** `[D]`
    - Single-qubit unitary → Hadamard/general 1-qubit gate `[D]`
      - Unitary evolution `[P]`
      - Qubit state space `C²` `[P]`
  - **Bell-pair preparation (H then CNOT on |00⟩)** `[D]`
    - Hadamard gate `[D]`
      - Single-qubit unitary; 2×2 unitary matrix `[M]`
      - Superposition principle `[D]` → vector-space structure of state space `[P]`
      - Basis change (Z-basis ↔ X-basis) `[D]`
    - CNOT gate `[D]`
      - Controlled-unitary construction `|0⟩⟨0|⊗I + |1⟩⟨1|⊗U` `[D]`
        - Outer products / projectors `[M]`
        - Tensor product of operators `[P]` (composite-system postulate)
        - Pauli-X (the controlled operation) `[D]`
      - Multi-qubit computational basis `[D]`
        - Orthonormal basis of a Hilbert space `[M]`
        - Tensor product of state spaces `[P]`
    - Initial 2-qubit state `|00⟩` `[D]` → computational basis `[D]`, tensor product `[P]`
    - (produces) Entanglement / non-separability `[D]`
      - Reduced density operator `[D]`
        - Density operator `[D]` → outer product, trace, ensembles `[M]`
        - Partial trace `[M]`
        - Tensor product `[P]`
      - Separability criterion `[D]`
  - **Bell-basis measurement of (q0,q1) as CNOT(q0→q1) + H(q0) + comp. measure** `[D]`
    - CNOT gate `[D]` (as above)
    - Hadamard gate `[D]` (as above)
    - Projective measurement in the computational basis `[D]`
      - Measurement postulate / Born rule `[P]`
      - Spectral projectors, completeness `Σ Pₘ = I`, `PₘPₙ = δₘₙPₘ` `[M]`
      - State collapse + renormalization `[P]`
      - Partial (subsystem) measurement `Pₘ ⊗ I` `[D]`
        - Tensor product `[P]`
        - Born rule on composite system `[P]`
    - Bell basis `[D]`
      - Bell states as `(P ⊗ I)|Φ⁺⟩`, `P ∈ {I,X,Y,Z}` `[D]`
      - Maximal entanglement `[D]`
      - Orthonormal basis of `C² ⊗ C²` `[M]`
  - **Classically-controlled Pauli correction** `[D]`
    - Classical control flow / conditional operation `[D]`
      - Principle of deferred measurement `[D]`
      - Classical outcome register `[D]` ← measurement postulate `[P]`
    - Pauli `X`, `Z` gates `[D]`
      - 2×2 Hermitian + unitary matrices `[M]`
      - Unitary evolution `[P]`
    - Pauli algebra: `X²=Z²=I`, `ZX = −XZ`, `HXH=Z` `[D]`
  - **Correctness = teleportation identity** `[D]`
    - Decomposition of `|ψ⟩ ⊗ |Φ⁺⟩` in the Bell basis of `(q0,q1)` `[D]`
    - Linearity of quantum mechanics `[P]` (operators are linear)
    - Pauli group relations `[D]`
    - Global-phase irrelevance / projective state space `[P]`
  - **(context, not a build dependency)** No-cloning theorem `[D]`
    - Linearity of unitary evolution `[P]` ; why teleportation destroys the source

### 2b. Topologically sorted flat list (fundamentals first)

This order drives the test build — a node's tests may assume every earlier
node passes.

```
F1   Complex Hilbert space (finite-dim, inner product)                 [M]
F2   Linear operators & matrices (adjoint, unitary, Hermitian, spectral)[M]
F3   Postulate 1 — State space; qubit = C² (unit vector)               [P]
F4   Global phase / projective state space (rays)                      [P]
F5   Superposition principle                                           [D]
F6   Single-qubit computational basis {|0⟩, |1⟩}                        [D]
F7   Postulate 2 — Unitary evolution                                   [P]
F8   Postulate 4 — Composite systems / tensor product; Kronecker prod  [P]
F9   Multi-qubit computational basis (2ⁿ-dim)                          [D]
F10  Outer products & projectors; completeness relation               [M]
F11  Pauli operators I, X, Z  (Y: defined, off critical path)          [D]
F12  Pauli algebra / group relations                                  [D]
F13  Hadamard gate                                                    [D]
F14  Controlled-unitary construction / CNOT gate                      [D]
F15  Postulate 3 — Measurement (Born rule)                            [P]
F16  Projective measurement (projectors, collapse, renormalization)   [D]
F17  Partial / subsystem measurement (Pₘ ⊗ I)                          [D]
F18  Density operator (mixed states, purity)                          [D]
F19  Partial trace / reduced density operator                         [M]
F20  Separability vs entanglement; entanglement entropy S(ρ_A)        [D]
F21  Bell states / Bell basis                                         [D]
F22  Bell-pair preparation circuit (H⊗I then CNOT on |00⟩)            [D]
F23  Classical control / conditional Pauli; deferred-measurement princ.[D]
F24  Teleportation identity (Bell-basis decomposition of |ψ⟩|Φ⁺⟩)     [D]
T    Quantum Teleportation Protocol (TARGET)                          [D]
```

Execution-branch subset: F1–F17, F21–F24, T.
Characterization-branch subset: F1–F3, F8, F10, F18–F21.

---

## 3. Per-Node Spec

Each: **concept** · **math object in code** · **verifiable property (unit
test)** · **primary source**.

### F1 — Complex Hilbert space
A finite-dimensional complex vector space `H = C^d` equipped with a
sesquilinear inner product `⟨·|·⟩` (conjugate-linear in the first argument by
physics convention), inducing a norm `‖v‖ = √⟨v|v⟩`; completeness is automatic
in finite dimension. All quantum state spaces and operator spaces in this
target are `C^d` for `d ∈ {2,4,8}`. The inner product is what makes
"orthonormal", "adjoint", and "probability" meaningful downstream.
- **Math object:** `numpy.ndarray` complex vectors in `C^d`; inner product
  `np.vdot(a, b)` (conjugates `a`).
- **Test:** conjugate symmetry `⟨a|b⟩ = conj(⟨b|a⟩)`; linearity
  `⟨a|(x·b + y·c)⟩ = x⟨a|b⟩ + y⟨a|c⟩`; positivity `⟨a|a⟩ ≥ 0` with equality iff
  `a = 0`; Cauchy–Schwarz `|⟨a|b⟩| ≤ ‖a‖‖b‖`.
- **Source:** Nielsen & Chuang §2.1.1–2.1.4 ("Bases and linear independence",
  "Inner products").

### F2 — Linear operators & matrices
Linear maps `A: H → H`, represented as `d×d` complex matrices once a basis is
fixed. Key structure: adjoint `A†` (conjugate transpose, defined by
`⟨a|Ab⟩ = ⟨A†a|b⟩`); **Hermitian** `A = A†` (real spectrum, orthogonal
eigenbasis — models observables); **unitary** `U†U = UU† = I` (norm-preserving,
invertible — models evolution); spectral theorem for normal operators
`A = Σ λᵢ |eᵢ⟩⟨eᵢ|`.
- **Math object:** `d×d` complex `numpy.ndarray`; `A.conj().T` for adjoint;
  `np.linalg.eigh` for Hermitian.
- **Test:** `(A†)† = A`; `(AB)† = B†A†`; for supplied unitary `U`,
  `np.allclose(U.conj().T @ U, I)`; spectral reconstruction
  `Σ λᵢ Pᵢ ≈ A` for Hermitian `A`.
- **Source:** Nielsen & Chuang §2.1.3–2.1.6 ("Linear operators and matrices",
  "The adjoint and Hermitian operators", "Eigenvectors and eigenvalues").

### F3 — Postulate 1: State space (qubit = C²)
Every isolated quantum system has an associated Hilbert space (state space);
its state is a unit vector `|ψ⟩ ∈ H`, `⟨ψ|ψ⟩ = 1`. A **qubit** is the `d = 2`
case, `H = C²`, standard basis `|0⟩ = (1,0)ᵀ`, `|1⟩ = (0,1)ᵀ`; a general pure
state is `α|0⟩ + β|1⟩`, `|α|²+|β|² = 1`, parametrizable on the Bloch sphere.
- **Math object:** length-2 (resp. length-`2ⁿ`) complex unit vector.
- **Test:** normalization `np.isclose(np.vdot(psi, psi).real, 1.0)` and
  `np.isclose(np.vdot(psi, psi).imag, 0.0)`; random-state generator produces
  vectors passing this for many seeds.
- **Source:** Nielsen & Chuang §2.2.1 ("State space", Postulate 1); qubit §1.2.

### F4 — Global phase / projective state space
`|ψ⟩` and `e^{iθ}|ψ⟩` produce identical measurement statistics under every
observable, so physical states are *rays* (equivalence classes under U(1)), not
vectors. **Relative** phase within a superposition is physical; **global** phase
is not. Relevant here because the teleportation correction must be checked to
recover `|ψ⟩` either exactly or up to a *known* global phase — the standard
`Z^{m0}X^{m1}` ordering gives exact recovery, and a test that ignores global
phase must compare `|⟨ψ|out⟩|² ≈ 1` or density matrices, not raw vectors.
- **Math object:** comparison helpers `phase_agnostic_allclose(a, b)` via
  `abs(vdot(a,b)) ≈ ‖a‖‖b‖`, or `np.outer(a, a.conj())`.
- **Test:** `measure_probs(psi) ≈ measure_probs(exp(1j*θ)·psi)` for random θ;
  `|ψ⟩⟨ψ|` invariant under global phase.
- **Source:** Nielsen & Chuang §2.2.7 ("The phase").

### F5 — Superposition principle
Because the state space is a complex vector space (F3), any normalized linear
combination of valid states is a valid state. Physical consequence: a qubit can
occupy `(|0⟩+|1⟩)/√2` with no classical analogue; amplitudes interfere
(add/cancel) under subsequent linear evolution. This is the operative resource
in the Hadamard step and in the Bell pair.
- **Math object:** vector addition + scalar multiplication in `C^d`, then
  renormalize.
- **Test:** for orthonormal `|0⟩,|1⟩` and `|a|²+|b|²=1`, `a|0⟩+b|1⟩` is
  normalized; interference check `H(H|0⟩) = |0⟩` (amplitudes for `|1⟩` cancel).
- **Source:** Nielsen & Chuang §1.2 (qubit superposition); consequence of
  §2.2.1 Postulate 1.

### F6 — Single-qubit computational basis
The ordered orthonormal basis `{|0⟩, |1⟩}` of `C²`, eigenbasis of `Z`. "Measure
in the computational basis" and "prepare `|0⟩`" both refer to it; all bit-string
labels are defined relative to it.
- **Math object:** `e0 = np.array([1,0])`, `e1 = np.array([0,1])` (complex).
- **Test:** `⟨0|0⟩ = ⟨1|1⟩ = 1`, `⟨0|1⟩ = 0`; `|0⟩⟨0| + |1⟩⟨1| = I₂`;
  `Z|0⟩ = |0⟩`, `Z|1⟩ = −|1⟩`.
- **Source:** Nielsen & Chuang §1.2.

### F7 — Postulate 2: Unitary evolution
A closed system's evolution over a fixed interval is a unitary map
`|ψ'⟩ = U|ψ⟩`, `U†U = I`. Consequences used everywhere below: norm/inner-product
preservation (probabilities stay normalized), reversibility (`U⁻¹ = U†`),
composition of gates = matrix product in reverse temporal order. Every gate
(H, X, Z, CNOT) is an instance.
- **Math object:** `d×d` unitary matrix; circuit = ordered product
  `U_n … U_2 U_1`.
- **Test:** `np.allclose(U.conj().T @ U, I)` for each gate; `‖U|ψ⟩‖ = ‖|ψ⟩‖`
  for random `|ψ⟩`; `U @ U.conj().T ≈ I`.
- **Source:** Nielsen & Chuang §2.2.2 ("Evolution", Postulate 2).

### F8 — Postulate 4: Composite systems / tensor product
The state space of a composite system is the tensor product of component state
spaces: `n` qubits ⇒ `H = (C²)^{⊗n} = C^{2ⁿ}`. Joint states of independent
subsystems are product states `|a⟩⊗|b⟩`; operators compose as `A⊗B`, with the
mixed-product identity `(A⊗B)(C⊗D) = (AC)⊗(BD)` and
`dim(V⊗W) = dim V · dim W`. Realized in code by the Kronecker product.
- **Math object:** `np.kron` for vectors and matrices; gate embedding
  `kron(kron(G0, G1), G2)`.
- **Test:** `dim` multiplicativity; `(A⊗B)(C⊗D) ≈ (A@C)⊗(B@D)`;
  `(A⊗B)† ≈ A†⊗B†`; `kron(I₂,I₂) ≈ I₄`; product state norm
  `‖a⊗b‖ = ‖a‖‖b‖`.
- **Source:** Nielsen & Chuang §2.2.8 ("Composite systems", Postulate 4);
  tensor-product algebra §2.1.7.

### F9 — Multi-qubit computational basis
The `2ⁿ` orthonormal vectors `|b_{n-1} … b_0⟩ = |b_{n-1}⟩ ⊗ … ⊗ |b_0⟩`,
`bᵢ ∈ {0,1}`. Fixes the integer↔bitstring indexing of every statevector and the
row/column layout of every embedded gate. Convention (big-endian here) must be
stated once and enforced.
- **Math object:** `basis(i, n)` = `2ⁿ`-vector, 1 at index `i`; index `i`'s bits
  give `(q0,…,q_{n-1})` under the chosen endianness.
- **Test:** the `2ⁿ` basis vectors are orthonormal and sum-of-projectors `= I`;
  `basis(0b101, 3)` corresponds to `q0=1, q1=0, q2=1` (big-endian) — assert
  explicitly.
- **Source:** Nielsen & Chuang §1.2, §1.3.2.

### F10 — Outer products & projectors
`|a⟩⟨b|` is the rank-1 operator `v ↦ ⟨b|v⟩ |a⟩`. A **projector** `P = P† = P²`
projects onto a subspace; rank-1 `Pᵢ = |i⟩⟨i|`. Completeness `Σᵢ |i⟩⟨i| = I`
for any orthonormal basis; orthogonality `PᵢPⱼ = δᵢⱼ Pᵢ`. These build both the
controlled-unitary (F14) and the measurement projectors (F16–F17).
- **Math object:** `np.outer(a, b.conj())`.
- **Test:** `P @ P ≈ P`; `P.conj().T ≈ P`; `|0⟩⟨0| + |1⟩⟨1| ≈ I₂`;
  `(|0⟩⟨0|)(|1⟩⟨1|) ≈ 0`; `trace(|i⟩⟨i|) = 1`.
- **Source:** Nielsen & Chuang §2.1.4 ("The outer product representation",
  completeness relation).

### F11 — Pauli operators (I, X, Z; Y off critical path)
`X = [[0,1],[1,0]]` (bit flip, `X|b⟩ = |b⊕1⟩`), `Z = [[1,0],[0,−1]]` (phase
flip, `Z|b⟩ = (−1)^b|b⟩`), `I = diag(1,1)`. All are Hermitian **and** unitary,
trace-zero (except I), eigenvalues `±1`. `Y = iXZ = [[0,−i],[i,0]]` completes
the basis of `2×2` Hermitian matrices but is not used by the corrections.
- **Math object:** four fixed `2×2` complex arrays.
- **Test:** `X@X = Z@Z = I`; `X† = X`, `Z† = Z`; `X@Z = -Z@X` (anticommute);
  `X@Z ≈ -1j*Y`; eigvals of `X`, `Z` are `{+1,−1}`.
- **Source:** Nielsen & Chuang §1.3.1 (gate defs) and §2.1.3 / Box 2.2 ("Pauli
  matrices").

### F12 — Pauli algebra / group relations
The single-qubit Pauli group relations that make the teleportation correction
work: `X² = Z² = I` (each correction is its own inverse), `ZX = −XZ` (order of
the two corrections matters only up to a global −1, hence irrelevant to the
ray but fixed by the `Z^{m0}X^{m1}` convention for exact recovery),
`HXH = Z`, `HZH = X` (why H maps a computational measurement into an X-basis
measurement in the Bell-measurement step).
- **Math object:** products of the F11 matrices.
- **Test:** `Z@X + X@Z ≈ 0`; `X@X ≈ I`, `Z@Z ≈ I`; `H@X@H ≈ Z`; `H@Z@H ≈ X`;
  `(X@Z)@(X@Z) ≈ -I` (so `X^{m1}Z^{m0}` inverse is `Z^{m0}X^{m1}`, verified by
  `Z@X@X@Z ≈ I`).
- **Source:** Nielsen & Chuang §2.1.3, §4.2 ("Single qubit operations");
  exact exercise numbering **uncertain**.

### F13 — Hadamard gate
`H = (1/√2)[[1,1],[1,−1]]`. Unitary and Hermitian, `H² = I`. Maps
`|0⟩ ↦ |+⟩`, `|1⟩ ↦ |−⟩` and back; converts a computational-basis measurement
into an X-basis measurement. Used to create superposition for the Bell pair
(step 2) and to rotate Alice's qubit for the Bell-basis readout (step 5).
- **Math object:** fixed `2×2` real/complex array (factor `1/√2`).
- **Test:** `H@H ≈ I`; `H.conj().T @ H ≈ I`; `H@e0 ≈ (e0+e1)/√2`;
  `H@e1 ≈ (e0−e1)/√2`; `H@X@H ≈ Z`.
- **Source:** Nielsen & Chuang §1.3.1 ("Single qubit gates", Hadamard).

### F14 — Controlled-unitary construction / CNOT
A controlled-`U` on (control `c`, target `t`) is
`|0⟩⟨0|_c ⊗ I_t + |1⟩⟨1|_c ⊗ U_t` — apply `U` to the target iff the control is
`|1⟩`, extended linearly to superpositions (creating entanglement).
`CNOT = controlled-X`; matrix (control = first qubit, big-endian)
`[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]`. `CNOT² = I`, unitary, not a
tensor product of single-qubit gates. For non-adjacent qubits, embed via
projector-sum with `I` on spectators (steps 3, 4).
- **Math object:** `kron(|0⟩⟨0|, I) + kron(|1⟩⟨1|, U)`, then embed with `np.kron`
  against spectator identities; `CZ` similarly with `U = Z`.
- **Test:** `CNOT @ CNOT ≈ I`; unitarity; action on basis
  `CNOT|00⟩=|00⟩, |01⟩=|01⟩, |10⟩=|11⟩, |11⟩=|10⟩` (big-endian control);
  `CNOT` on `|+0⟩` gives `|Φ⁺⟩` (entangling); `(H⊗H) CNOT_{0→1} (H⊗H) ≈ CNOT_{1→0}`.
- **Source:** Nielsen & Chuang §1.3.2 ("Multiple qubit gates", CNOT); §4.3
  ("Controlled operations").

### F15 — Postulate 3: Measurement (Born rule)
Measurement is described by a collection `{Mₘ}` with `Σ Mₘ†Mₘ = I`; outcome `m`
occurs with probability `p(m) = ⟨ψ|Mₘ†Mₘ|ψ⟩`, and the post-measurement state is
`Mₘ|ψ⟩ / √p(m)`. This is the only non-unitary, probabilistic, irreversible step
in the protocol and the reason teleportation does not clone. For this circuit
the projective special case (F16) suffices.
- **Math object:** set of matrices `{Mₘ}`; probability via quadratic form.
- **Test:** `Σ Mₘ†Mₘ ≈ I`; `p(m) ≥ 0`; `Σₘ p(m) ≈ 1` for random `|ψ⟩`;
  post-state normalized.
- **Source:** Nielsen & Chuang §2.2.3 ("Quantum measurement", Postulate 3).

### F16 — Projective measurement
Special case `Mₘ = Pₘ`, orthogonal projectors from the spectral decomposition
of a Hermitian observable, `Σ Pₘ = I`, `PₘPₙ = δₘₙPₘ`. Then
`p(m) = ⟨ψ|Pₘ|ψ⟩ = ‖Pₘ|ψ⟩‖²` and collapse `|ψ⟩ ↦ Pₘ|ψ⟩ / √p(m)`. Repeat
measurement is deterministic (idempotent). Computational-basis measurement uses
`Pₘ = |m⟩⟨m|`.
- **Math object:** `Pₘ = np.outer(basis_m, basis_m.conj())`; probabilities
  `= np.abs(psi)**2` in that basis.
- **Test:** idempotent + orthogonal + complete (F10); Born probabilities equal
  `|amplitude|²`; second measurement returns same `m` with `p = 1`;
  `Σ p(m) ≈ 1`.
- **Source:** Nielsen & Chuang §2.2.5 ("Projective measurements").

### F17 — Partial / subsystem measurement
Measuring a subset of qubits: projector for outcome `m` on subsystem `A` is
`Pₘ^A ⊗ I_B`. Outcome probability `p(m) = ⟨Ψ|(Pₘ^A ⊗ I_B)|Ψ⟩`; conditional
state of the rest is `(Pₘ^A ⊗ I_B)|Ψ⟩ / √p(m)`, generally still entangled or
now steered into a definite branch. In teleportation, measuring `(q0,q1)`
collapses `q2` into one of the four `X^{m1}Z^{m0}|ψ⟩` branches, each with
`p = 1/4` **for every input** `|ψ⟩` — the flat distribution is the signature
that no information about `α,β` leaks to Alice.
- **Math object:** `kron(kron(Pm0, Pm1), I2)` acting on the 8-vector;
  renormalize the surviving branch; classical outcome sampled from `p`.
- **Test:** the four `p(m0,m1)` are each `≈ 0.25` independent of random `|ψ⟩`;
  `Σ p ≈ 1`; conditional `q2` (after tracing out q0,q1 which are now product)
  equals `X^{m1}Z^{m0}|ψ⟩` up to normalization; marginal on `q0` alone is
  `1/2,1/2`.
- **Source:** Nielsen & Chuang §2.2.5 (projective measurement on a subsystem);
  applied form in §1.3.7. Exact "partial measurement" heading **uncertain** —
  it is developed by example rather than as a named subsection.

### F18 — Density operator
`ρ = Σ pᵢ |ψᵢ⟩⟨ψᵢ|`: Hermitian, positive-semidefinite, `tr ρ = 1`. Represents
statistical ensembles and — crucially here — the state of a subsystem of an
entangled pure state, which has no state vector of its own. Pure iff `tr ρ² = 1`
iff `ρ² = ρ`; otherwise mixed. Expectation `⟨A⟩ = tr(ρA)`; evolution
`ρ ↦ UρU†`.
- **Math object:** `d×d` complex matrix; `rho = np.outer(psi, psi.conj())` for
  pure.
- **Test:** `ρ ≈ ρ†`; eigenvalues `≥ −tol`; `tr ρ ≈ 1`; for pure state
  `tr(ρ@ρ) ≈ 1` and `ρ@ρ ≈ ρ`; `tr(ρ @ A)` matches `⟨ψ|A|ψ⟩`.
- **Source:** Nielsen & Chuang §2.4.1–2.4.2 ("Ensembles of quantum states",
  "General properties of the density operator").

### F19 — Partial trace / reduced density operator
`ρ_A = tr_B(ρ_{AB})`, the unique linear map with
`tr_B(|a₁⟩⟨a₂| ⊗ |b₁⟩⟨b₂|) = |a₁⟩⟨a₂| · ⟨b₂|b₁⟩`. Gives the correct local
state: all measurement statistics on `A` alone are reproduced by `ρ_A`. The
tool that turns "is this entangled?" into a computation.
- **Math object:** reshape `ρ` to `(dA,dB,dA,dB)` and `np.einsum('ibjb->ij', …)`
  (trace over the B indices).
- **Test:** `tr(ρ_A) ≈ 1`, `ρ_A` Hermitian PSD; `tr_B(σ_A ⊗ τ_B) ≈ σ_A` when
  `tr τ_B = 1`; linearity; consistency
  `tr(ρ_A X_A) ≈ tr(ρ_{AB} (X_A ⊗ I_B))`.
- **Source:** Nielsen & Chuang §2.4.3 ("The reduced density operator").

### F20 — Separability vs entanglement; entanglement entropy
A pure bipartite state is **separable** iff it is a product `|a⟩⊗|b⟩` iff its
reduced density operator is pure (`tr ρ_A² = 1`, Schmidt rank 1); otherwise
**entangled**. The **entanglement entropy** is the von Neumann entropy of the
reduction, `S(ρ_A) = −tr(ρ_A log₂ ρ_A) = −Σ λᵢ log₂ λᵢ`, ranging `0` (product)
to `log₂(min dim)` (maximally entangled). For any Bell state `ρ_A = I/2`, so
`S = 1` bit — the exact resource consumed per teleportation.
- **Math object:** eigenvalues of `ρ_A` via `np.linalg.eigvalsh`, then Shannon
  entropy of the spectrum (base 2), guarding `0·log0 = 0`.
- **Test:** `S(|Φ⁺⟩ reduced) ≈ 1.0`; `S(product state) ≈ 0.0`;
  `S(ρ_A) ≈ S(ρ_B)` (equal for pure global state); `0 ≤ S ≤ 1` for 2 qubits;
  purity `tr ρ_A² ≈ 0.5` for Bell.
- **Source:** Nielsen & Chuang §2.5 (Schmidt decomposition) and §11.3 (von
  Neumann entropy); the identification "entanglement of a pure state = `S(ρ_A)`"
  appears in §12.5.1 / Box — **exact section uncertain**.

### F21 — Bell states / Bell basis
The four maximally entangled orthonormal states of two qubits:
`|Φ⁺⟩ = (|00⟩+|11⟩)/√2`, `|Φ⁻⟩ = (|00⟩−|11⟩)/√2`,
`|Ψ⁺⟩ = (|01⟩+|10⟩)/√2`, `|Ψ⁻⟩ = (|01⟩−|10⟩)/√2`. Compactly
`|β_{xy}⟩ ∝ (Z^x X^y ⊗ I)|Φ⁺⟩`, i.e. the Bell basis is the orbit of `|Φ⁺⟩`
under local Paulis. Teleportation = Alice measuring `(q0,q1)` in this basis;
outcome `(m0,m1)` ⇔ Bell state `|β_{m0 m1}⟩` ⇔ Pauli `X^{m1}Z^{m0}` on `q2`.
- **Math object:** four fixed length-4 unit vectors; `4×4` change-of-basis
  matrix `B` with Bell states as columns.
- **Test:** orthonormal set, `B†B ≈ I`; each is entangled (`S = 1`, F20);
  `(X⊗I)|Φ⁺⟩ ≈ |Ψ⁺⟩`, `(Z⊗I)|Φ⁺⟩ ≈ |Φ⁻⟩`, `(X@Z ⊗ I)|Φ⁺⟩ ≈ |Ψ⁻⟩` up to
  phase; `(H⊗I)·CNOT` maps each `|β_{m0m1}⟩` to `|m0 m1⟩`.
- **Source:** Nielsen & Chuang §1.3.6 ("Bell states" / EPR), Eqs. (1.23)–(1.25).

### F22 — Bell-pair preparation circuit
`|Φ⁺⟩ = CNOT_{0→1} · (H ⊗ I) · |00⟩`. Steps: `H` puts `q0` in `|+⟩`
(`(|00⟩+|10⟩)/√2`), then `CNOT` correlates `q1` (`(|00⟩+|11⟩)/√2`). Output is
entangled, `S = 1`, and is the shared resource (`q1,q2` in the target after
relabeling). The two component gates are exactly F13 and F14.
- **Math object:** `CNOT4 @ np.kron(H, I2) @ basis(0,2)`.
- **Test:** `np.allclose(output, [1,0,0,1]/√2)`; reduced `ρ_{q0}` and `ρ_{q1}`
  each `≈ I/2`; entanglement entropy `≈ 1`; circuit is a product of unitaries
  ⇒ overall unitary; `CNOT·(H⊗I)` applied to `|01⟩,|10⟩,|11⟩` yields the other
  three Bell states.
- **Source:** Nielsen & Chuang §1.3.6 (circuit generating Bell states,
  Fig. 1.12).

### F23 — Classical control / conditional Pauli; deferred measurement
A **classically-controlled gate** applies a fixed unitary to a qubit iff a
classical bit (a measurement outcome) equals 1 — a branch on classical data,
not a quantum-controlled unitary. The **principle of deferred measurement**:
measurements can be pushed to the end of a circuit if each classically-
controlled gate is replaced by the corresponding quantum-controlled gate; the
output statistics are identical. This licenses testing teleportation as a pure
unitary (§1.3, `CNOT(q1→q2)` + `CZ(q0→q2)`) instead of simulating
stochastic mid-circuit collapse.
- **Math object:** in coherent form, `CNOT` / `CZ` embeddings (F14); in
  stochastic form, `if bit: state = apply(P, state)` after F17 sampling.
- **Test:** deferred-measurement equivalence —
  `U_tele(|ψ⟩⊗|00⟩) ≈ |+⟩⊗|+⟩⊗|ψ⟩` for random `|ψ⟩` (coherent);
  the stochastic simulation's per-branch corrected `q2` matches `|ψ⟩` in all 4
  branches; averaged output density matrix equals the coherent one.
- **Source:** Nielsen & Chuang §4.4 ("Measurement", "Principle of deferred
  measurement").

### F24 — Teleportation identity
The algebraic core: for any `|ψ⟩ = α|0⟩+β|1⟩`,
`|ψ⟩_{q0} ⊗ |Φ⁺⟩_{q1q2} =
½ [ |β₀₀⟩_{q0q1}(α|0⟩+β|1⟩) + |β₀₁⟩(α|1⟩+β|0⟩)
  + |β₁₀⟩(α|0⟩−β|1⟩) + |β₁₁⟩(α|1⟩−β|0⟩) ]_{q2}`,
equivalently in the computational basis after `CNOT_{0→1}` then `H_{q0}` (see
§1.2 "After 5"). Each of Alice's four equally likely outcomes leaves `q2` in a
known Pauli image of `|ψ⟩`, invertible by `Z^{m0}X^{m1}`. Rests on linearity
(expand, regroup) and the Bell-basis/Pauli correspondence (F21).
- **Math object:** symbolic or numeric expansion; `4×4` Bell change-of-basis
  applied to the `(q0,q1)` factor of the 8-vector.
- **Test:** numerically expand LHS for random `α,β`, apply
  `(H⊗I⊗I)(CNOT_{0→1}⊗I)`, and check the amplitude block for each `(m0,m1)`
  equals `½ · X^{m1}Z^{m0}|ψ⟩`; sum of the four branch probabilities `≈ 1`;
  each branch norm `≈ ¼`.
- **Source:** Nielsen & Chuang §1.3.7 ("Example: quantum teleportation"),
  Eqs. (1.30)–(1.34). Section number is 1.3.7 in the 10th-anniversary edition —
  verify against your copy.

### T — Quantum Teleportation Protocol (target)
Transfer of an unknown qubit state from `q0` to `q2` using one shared Bell pair
(`q1,q2`) and two classical bits, without moving the physical qubit and without
ever learning `α,β`. Consumes 1 ebit + 2 cbits; the source copy is destroyed by
Alice's measurement (consistent with no-cloning). Circuit = F22 (resource) ∘
F14+F13 (Bell measurement) ∘ F17 (measure `q0,q1`) ∘ F23 (correct `q2`).
- **Math object:** full `8`-dim statevector simulation (stochastic) **or**
  coherent `8×8` `U_tele` (deferred measurement).
- **Test (exact):** coherent
  `U_tele(|ψ⟩⊗|00⟩) ≈ |+⟩⊗|+⟩⊗|ψ⟩`; or stochastic — for each of 4 branches, the
  corrected reduced `ρ_{q2} ≈ |ψ⟩⟨ψ|` (`np.allclose`), for many random `|ψ⟩`.
- **Test (statistical):** sample `N` shots; branch counts `≈ N/4` (χ² vs
  uniform, dof 3); prepare `|ψ⟩ = U_ψ|0⟩`, teleport, apply `U_ψ†` to `q2`,
  measure — observed `P(1) ≤ 3/√N`; or 3-basis tomography of `q2`, Bloch vector
  within `‖Δr‖ ≤ 3/√N` of the input.
- **Source:** Bennett et al., *Phys. Rev. Lett.* **70**, 1895 (1993) (original);
  Nielsen & Chuang §1.3.7; IBM Quantum Learning, "Basics of quantum
  information — Entanglement in action" (teleportation section); PennyLane
  Codebook, "Quantum Teleportation" node.

---

## 4. Test Strategy Notes

`exact` = deterministic `np.allclose` on a statevector, matrix, probability
vector, or density matrix (default tol `atol=1e-8`, loosen to `1e-6` for
compounded `1/√2` factors). `statistical` = requires sampling; use enough shots
that the `±3σ ≈ 3·√(p(1−p)/N)` band excludes the failure region, or a χ²
goodness-of-fit test at a fixed α (e.g. 0.01) with a fixed RNG seed to keep CI
deterministic.

| Node | Check type | Concrete method | Tolerance / N |
|------|-----------|-----------------|---------------|
| F1 Hilbert space | exact | inner-product axioms on random vectors | `1e-12` |
| F2 operators | exact | `(AB)†=B†A†`, unitarity, spectral reconstruction | `1e-10` |
| F3 state space | exact | `‖ψ‖=1` over many random seeds | `1e-12` |
| F4 global phase | exact | `probs(ψ)=probs(e^{iθ}ψ)`; `|ψ⟩⟨ψ|` invariant | `1e-10` |
| F5 superposition | exact | normalization of combos; `H·H|0⟩=|0⟩` interference | `1e-10` |
| F6 comp. basis (1q) | exact | orthonormality, completeness, `Z` eigrel | `1e-12` |
| F7 unitary evolution | exact | `U†U=I` each gate; norm preservation | `1e-10` |
| F8 tensor product | exact | mixed-product identity, dim, `(A⊗B)†` | `1e-10` |
| F9 comp. basis (nq) | exact | orthonormality; **endianness assertion** on a known index | `1e-12` |
| F10 projectors | exact | `P²=P`, `P†=P`, `ΣP=I`, `PᵢPⱼ=0` | `1e-10` |
| F11 Pauli X,Z | exact | `X²=Z²=I`, `XZ=−ZX`, adjoint, eigvals `±1` | `1e-12` |
| F12 Pauli algebra | exact | `HXH=Z`, `HZH=X`, `ZXXZ=I` | `1e-10` |
| F13 Hadamard | exact | `H²=I`, action on `|0⟩,|1⟩`, `HXH=Z` | `1e-10` |
| F14 CNOT / ctrl-U | exact | `CNOT²=I`, basis-permutation table, entangles `|+0⟩` | `1e-10` |
| F15 measurement postulate | exact | `ΣMₘ†Mₘ=I`, `Σp=1`, post-state normalized | `1e-10` |
| F16 projective meas. | exact | Born `p=|amp|²`, idempotent repeat, completeness | `1e-10` |
| F17 partial measurement | exact | four `p(m0,m1)=¼` **∀ random ψ**; `Σp=1`; conditional `q2` state | `1e-9` |
| F18 density operator | exact | Hermitian, PSD, `trρ=1`, pure⇒`trρ²=1` | `1e-10` |
| F19 partial trace | exact | `trρ_A=1`, product consistency, local-expectation match | `1e-10` |
| F20 entanglement entropy | exact | `S(Bell)=1`, `S(product)=0`, `S(ρ_A)=S(ρ_B)` | `1e-7` (log/eig) |
| F21 Bell states/basis | exact | orthonormality, `S=1` each, `(P⊗I)|Φ⁺⟩` orbit, `(H⊗I)CNOT` decode | `1e-9` |
| F22 Bell-pair prep | exact | output `=[1,0,0,1]/√2`; `ρ_{q0}=ρ_{q1}=I/2`; `S=1` | `1e-9` |
| F23 classical control / deferred meas. | exact | `U_tele(ψ⊗00)=+ ⊗ + ⊗ ψ`; stochastic branches all recover ψ; averaged ρ matches coherent | `1e-9` |
| F24 teleportation identity | exact | per-branch amplitude block `= ½·X^{m1}Z^{m0}ψ`; branch norms `¼` | `1e-9` |
| **T end-to-end (exact)** | exact | coherent `U_tele(ψ⊗00)=+⊗+⊗ψ`, **or** per-branch corrected `ρ_{q2}=|ψ⟩⟨ψ|`, random ψ ×100 | `1e-8` |
| **T end-to-end (stat.)** | statistical | (a) branch counts vs uniform: χ², dof 3, α=0.01, `N≥10⁴`, fixed seed; (b) `U_ψ†` on `q2` then measure: `P(1) ≤ 3/√N`, `N≥10⁴`; (c) `q2` tomography: `‖Δr_Bloch‖ ≤ 3/√N` | `N = 10⁴–10⁵` |

**Notes for the build order:**
- Everything except the two `T` statistical rows is `np.allclose`-exact —
  favor those; the statistical rows are integration smoke tests, not the
  primary correctness signal.
- Fix one RNG seed for all sampling tests so CI is reproducible; assert the
  χ² p-value, not a hand-tuned count band.
- Add one **endianness contract test** (F9) that fails loudly if a future
  Qiskit/PennyLane backend swaps qubit order — every expected ket in §1 and
  §3 depends on it.
- F18–F20 can be deferred if only circuit correctness (not resource
  characterization) is in scope; the `T` exact test does not depend on them
  when written in coherent form.
- No-cloning is not a testable node here; if a "sanity" test is wanted, check
  that after Alice's measurement `ρ_{q0}` is `I/2` (all payload information has
  left `q0`) — i.e. `q0` retains nothing to clone.
