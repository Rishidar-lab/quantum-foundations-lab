# PHASE 1 — Baseline Audit

Audited commit: `82e4e58` (root commit, 356/356 passing).
Auditor pass date: 2026-08-30.
Reproduce the adversarial results: `python tools/mutation_audit.py --json experiments/results/mutation_audit.json`

---

## 1. Module inventory

| Module | Lines | Role | Independent of Qiskit? |
|---|---:|---|---|
| `katas/01_hilbert_space.py` | 55 | inner product, norm | yes |
| `katas/09_computational_basis_nq.py` | 63 | big-endian index ↔ bits ↔ basis vector | yes |
| `katas/10_projectors.py` | 45 | outer product, rank-1 projector | yes |
| `katas/11_pauli_operators.py` | 50 | I, X, Y, Z as fixed 2×2 matrices | yes |
| `katas/12_pauli_algebra.py` | 60 | `pauli_mul` — the Pauli-group table | yes |
| `katas/13_hadamard.py` | 39 | Hadamard gate | yes |
| `katas/14_cnot_controlled_u.py` | 63 | `controlled_unitary` (bit-surgery embedding), `cnot`, `cz` | yes |
| `katas/15_measurement_postulate.py` | 60 | general measurement `{M_m}`, Born rule, collapse | yes |
| `katas/16_projective_measurement.py` | 70 | spectral projectors, `expectation_value`, projective collapse | yes |
| `katas/17_partial_measurement.py` | 75 | `subsystem_projector`, partial probs, partial collapse | yes |
| `katas/18_density_operator.py` | 60 | `density_from_state/ensemble`, `purity`, predicate, `expectation` | yes |
| `katas/19_partial_trace.py` | 52 | `partial_trace` via `einsum` index identification | yes |
| `katas/20_entanglement_entropy.py` | 55 | von Neumann entropy, entanglement entropy, product-state test | yes (uses F19) |
| `katas/21_bell_states.py` | 44 | `phi_plus`, `bell_state(b0,b1)`, `bell_basis` | yes |
| `katas/22_bell_pair_prep.py` | 40 | `bell_pair_unitary`, `prepare_bell_pair` | yes (uses F13/F14) |
| `katas/23_classical_control_deferred.py` | 55 | `apply_conditional`, `bob_correction`, `defer` | yes (uses F14) |
| `katas/24_teleportation_identity.py` | 50 | `pre_measurement_state`, `teleportation_branches` | yes (uses F13/F14) |
| `teleportation/circuit.py` | 138 | Qiskit assembly: `build_teleportation_circuit`, `build_verification_circuit` | **no — this is the Qiskit layer** |

`katas/_deps.py`, `tests/kataloader.py`, `conftest.py` are harness only.

---

## 2. Test classification

Categories: **M** mathematical invariant · **P** physics invariant · **B** behavioral/unit ·
**I** integration / cross-representation · **R** regression (fixed seed) · **C** implementation-coupled / tautological.

| Test file | cases | dominant categories | notes |
|---|---:|---|---|
| `test_01_hilbert_space` | 47 | M | inner-product axioms, Cauchy–Schwarz. All routes ultimately go through `np.vdot`; trusted primitive. |
| `test_09_computational_basis_nq` | 18 | M, B | `test_big_endian_contract` is the project's independent endianness anchor `(1,0,1)↔5`. |
| `test_10_projectors` | 19 | M | `P²=P`, `P†=P`, `ΣP=I`, mutual orthogonality, fixes/kills. Strong. |
| `test_11_pauli_operators` | 20 | M, B | structural pin: Hermitian ∧ unitary ∧ involutive ∧ traceless ∧ spec `{±1}` + `XZ=−iY`. |
| `test_12_pauli_algebra` | 46 | M, I | **anchor:** `pauli_mul` (symbolic) cross-checked against `k11` matmul for all 16 pairs. |
| `test_13_hadamard` | 8 | M | anchor: `HXH=Z`, `HZH=X`, `HYH=−Y` (uses `k11`). `H\|0⟩=\|+⟩` is near-definitional (soft C). |
| `test_14_cnot_controlled_u` | 12 | M, B | anchor: `controlled_unitary(H)` vs projector-sum `\|0⟩⟨0\|⊗I+\|1⟩⟨1\|⊗H` (independent of bit-surgery impl). |
| `test_15_measurement_postulate` | 14 | P, M | anchor: `born == np.abs(psi)**2` (independent of the `M†M` form). |
| `test_16_projective_measurement` | 8 | M, P | anchor: spectral reconstruction `ΣλP == A`. `born_probs` "manual" line is coupled (same formula). |
| `test_17_partial_measurement` | 22 | P, M | anchors: Alice's 4 outcomes `= ¼` ∀ψ (no-signalling); conditional q2 `= X^m1 Z^m0 ψ` (analytic). |
| `test_18_density_operator` | 9 | M, P | anchors: `tr(ρA) == ⟨ψ\|A\|ψ⟩`; spectrum-preserving evolution; `is_density_matrix` literal rejections. |
| `test_19_partial_trace` | 9 | M, P | anchors: product factorizes; `tr(ρ_A M) == tr(ρ_AB (M⊗I))`. Bell→I/2 alone is transpose-blind. |
| `test_20_entanglement_entropy` | 17 | P, M | anchors: `S(I/d)=log₂d`; `S(ρ_A)=S(ρ_B)`; product `S=0`. |
| `test_21_bell_states` | 12 | M, P, **C** | 1 tautology (`test_teleportation_decoder`); missing independent literals for Ψ⁺/Φ⁻/Ψ⁻. |
| `test_22_bell_pair_prep` | 7 | P, **C** | **2 of 4 tautological** (see §3). Real anchors: `prepare_bell_pair()==[1,0,0,1]/√2`, reductions I/2. |
| `test_23_classical_control_deferred` | 15 | P, I, **C** | 1 near-tautology (`defer` vs `cnot` both call `controlled_unitary`). Anchor: deferred-measurement equivalence (two operational paths). |
| `test_24_teleportation_identity` | 16 | P, M, **C** | 1 tautology (`pre_measurement_state_matches_circuit`). Anchors: amplitude block `= ½ X^m1 Z^m0 ψ`, branch probs `¼`. |
| `test_integration_teleportation` | 57 | I, R | anchors: Aer statevector fidelity `=1` (24 seeds); deferred circuit `= ψ⊗\|+⟩⊗\|+⟩`. 1 weak assertion (see §3). |

**Overall:** the suite is **predominantly M/P** with genuine independent anchors in most nodes. The soft spots are concentrated in F21–F24 (the teleportation-specific nodes), exactly where the math is most redundantly encoded.

---

## 3. Tautological / implementation-coupled tests (explicitly hunted)

| # | Test | Why it is tautological | Severity |
|---|---|---|---|
| T1 | `test_21::test_teleportation_decoder` | asserts `(H⊗I)·CNOT · bell_state(b) == \|b⟩`, and `bell_state(b) := CNOT·(H⊗I)\|b⟩`. Reduces to `I\|b⟩ == \|b⟩` — only re-tests `H²=I`, `CNOT²=I`. | medium |
| T2 | `test_22::test_unitary_is_composition_of_h_and_cnot` (assert 1) | `expected = k14.cnot(2,0,1) @ np.kron(k13.hadamard(), I)` is **character-identical** to the body of `bell_pair_unitary()`. | medium |
| T3 | `test_22::test_maps_computational_basis_to_bell_basis` | compares `bell_pair_unitary() @ \|b⟩` to `k21.bell_state(b)`; both evaluate `CNOT·(H⊗I)\|b⟩`. Compares a value to itself across two module names. | medium |
| T4 | `test_24::test_pre_measurement_state_matches_circuit` | the test re-runs `kron(I,H,I) → cnot(3,1,2) → cnot(3,0,1) → kron(H,I,I)` — the exact sequence inside `pre_measurement_state`. Catches only a divergent typo. | medium |
| T5 | `test_23::test_defer_matches_controlled_unitary` | `defer(X,…)` returns `controlled_unitary(X,…)`; `cnot(…)` returns `controlled_unitary(X,…)`. Tests `controlled_unitary` against itself. | low |
| T6 | `test_13::test_maps_z_basis_to_x_basis`, `test_11::test_basis_action` | the asserted output vector is literally a column of the gate matrix under test. Near-definitional. | low (structural invariants elsewhere compensate) |

**Structural observation (not a single test):** the teleportation pre-measurement circuit is constructed from the same gate recipe in **three** places — `katas/24` (`pre_measurement_state`), `tests/test_24` (helper), `tests/test_17` (helper). A *conceptual* wiring error (e.g. wrong CNOT direction in the Bell-measurement step) would be reproduced identically in all three and their NumPy assertions would still agree. The only wiring check that does **not** replay the recipe is the Qiskit integration test.

---

## 4. Adversarial / mutation audit

`tools/mutation_audit.py` applies each mutation, runs the targeted tests, reverts via `git checkout`, and refuses to run on a dirty tree. Full machine-readable output: `experiments/results/mutation_audit.json`.

| Concept | Mutation | Result | Detecting tests (failing) |
|---|---|---|---:|
| normalization | `norm()` drops the `sqrt` → returns `⟨a\|a⟩` | **CAUGHT** | 4 |
| normalization | projective `collapse` skips renormalization | **CAUGHT** | 1 |
| tensor-product ordering | `controlled_unitary` reads qubit bits little-endian | **CAUGHT** | 43 |
| Hadamard | `H[1,1]` sign flip → matrix non-unitary | **CAUGHT** | 13 |
| CNOT | conditions on the **target** bit instead of the control | **CAUGHT** | 9 |
| measurement probability | Born rule returns `√p` instead of `p` | **CAUGHT** | 6 |
| global phase | every teleportation branch state ×(−1) | **INVISIBLE (correct)** | 0 — physics: a pure global phase must not be observable, and the tests correctly use `\|⟨·\|·⟩\|` / fidelity |
| relative phase | Bob's `Z` correction replaced by `S = diag(1,i)` | **CAUGHT** | 12 |
| teleportation correction | Bob keys `X`→m0 and `Z`→m1 (swapped) | **CAUGHT** | 10 |
| partial trace | reduced density matrix returned **transposed** | **CAUGHT** | 2 |

**Verdict: PASS** — 9/9 error-mutations detected, the 1 physically-invisible mutation correctly slips through, tree restored clean, baseline still 356/356.

Notable detail: the **transposed partial trace** is caught by only **2** tests (`test_product_state_factorizes`, `test_local_expectation_consistency`) — the marquee `test_bell_state_reductions_are_maximally_mixed` does **not** catch it because `I/2` is real-symmetric (`ρ_Aᵀ = ρ_A`). A first draft of the mutation (`col[q]=row[q]` → `row[q]=col[q]`) was a **semantic no-op** (label identification is symmetric) and initially reported as SURVIVED; it was corrected to the transpose mutation. Lesson recorded: mutation testing needs its own sanity check that the mutant actually changes behaviour.

---

## 5. Weaknesses discovered in the original 356-test implementation

| ID | Weakness | Risk | Addressed in |
|---|---|---|---|
| W1 | F21/F22: 3 tautological tests; no independent numeric literals for `\|Ψ⁺⟩`, `\|Φ⁻⟩`, `\|Ψ⁻⟩` | med | Phase 2 (`foundations`, independent Bell literals) |
| W2 | F24 `pre_measurement_state` verified only against a copy of its own gate sequence in NumPy; independent check is Qiskit-only | med | Phase 2 (analytic Bell-basis construction) + Phase 3 |
| W3 | `test_integration::test_uncorrected_branches_are_pauli_off` asserts only `0≤fid≤1` and `tr=1` — a trivial-copy circuit passes it | med | Phase 2 (strengthened assertion) |
| W4 | Partial-trace marquee test (`Bell→I/2`) is invariant under transposition of the result | low (covered elsewhere) | Phase 2 (complex-reduction marquee test) |
| W5 | Single global seed `RNG(20260830)`; no multi-seed / property sweep — a bug invisible for that seed's draws could pass | low | Phase 2 (seed-swept cross-checks) |
| W6 | **No NumPy↔Qiskit cross-check for the primitive gates/states** — only the full circuit is cross-checked. Convention mismatches that cancel in the full circuit are undetected | med | Phase 2 (`tests/test_foundations_*`) |
| W7 | `atol=1e-8` is loose for compounded `1/√2`; a wrong-by-1e-10 impl passes | very low | documented; Phase 2 tightens to 1e-10 where exact |
| W8 | `test_16::test_born_probs_distribution_and_formula` "manual" comparison uses the implementation's own formula | low | Phase 2 |
| W9 | Integration docstring says "output statevector **==** input" but every test is phase-agnostic (fidelity / `\|⟨·⟩\|`). It is `==` only up to global phase | doc only | corrected wording Phase 2/3 |
| W10 | Zero adversarial/mutation coverage before this audit | — | resolved (`tools/mutation_audit.py`, 10 mutations) |

---

## 6. What is genuinely solid (do not "fix")

- **F1–F20 mathematical core.** Structural invariants (unitarity, Hermiticity, spectral theorem, `ΣP=I`, group axioms, `S(ρ_A)=S(ρ_B)`, `tr(ρ_A M)=tr(ρ_AB(M⊗I))`) pin these tightly and independently.
- **F12 Pauli-group table** — symbolic rule vs numerical matmul is a textbook-quality cross-check.
- **F17 no-signalling test** — `¼` for every input ψ is a real physics invariant, not a coincidence of the implementation.
- **Integration fidelity test** — Qiskit builds and simulates the circuit by an entirely separate code path; fidelity `=1` across 24 seeds is a strong end-to-end cross-check of the whole NumPy chain.
- **Phase-agnostic comparisons** — the suite correctly treats global phase as unobservable (mutation confirms).

---

## 7. Recommendations feeding Phases 2–7

1. **Phase 2** builds `foundations/` — for every primitive, an explicit NumPy object *and* a Qiskit object computed independently, asserted equal (closes W6). Adds independent Bell literals (W1), an analytic teleportation-identity construction (W2), a transpose-sensitive partial-trace marquee (W4), seed sweeps (W5).
2. **Phase 2** rewrites `test_uncorrected_branches_are_pauli_off` to assert the branch state *is* a specific Pauli image (closes W3).
3. Keep `tools/mutation_audit.py` as a first-class check — wire it into the `Makefile` (`make audit`) and run it in the reproducibility sweep (Phase 7).
4. Leave F1–F20 tests as-is; add cross-checks alongside, do not replace.
