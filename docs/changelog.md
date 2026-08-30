# Changelog

Build/verification log for the quantum-teleportation kata project. One line per
node as it goes green. Node IDs (F1..F24, T) refer to `research/dependency_map.md`.

Format: `YYYY-MM-DD | node | status | notes`
Status: `red` (test written, kata stubbed, failing) · `green` (kata implemented by
user, test passes) · `green (agent-solved)` (implemented by the build agent on an
explicit "skip ahead") · `green (agent-solved, fast mode)` (implemented by the
build agent under the F13→F24 fast-mode directive — never user-solved, not to be
relabeled).

---

## Infrastructure

| Date | Item | Notes |
|------|------|-------|
| 2026-08-30 | project scaffold | `~/qm-teleportation`, git init (uncommitted), `.venv` (py3.12), `requirements.txt` (floors) + `requirements.lock.txt` (pins: qiskit 2.5.2, aer 0.17.2, ibm-runtime 0.49.0, numpy 2.5.2) |
| 2026-08-30 | test harness | `tests/kataloader.py` loads digit-prefixed kata modules; `katas/_deps.py` does the same for cross-kata deps (F20→F19, F22/F24→F13/F14); root `conftest.py` puts repo root + `tests/` + `katas/` on `sys.path` |
| 2026-08-30 | CI | `.github/workflows/test.yml` runs `pytest -ra -q` on every push / PR / manual dispatch |
| 2026-08-30 | Qiskit assembly | `teleportation/circuit.py`: `build_teleportation_circuit` (StatePreparation payload, `h`/`cx`, mid-circuit `measure`, `if_test` X/Z correction; `deferred=True` → coherent `cx`/`cz`) and `build_verification_circuit` (ry/rz payload + inverse on q2, no multiplexer — Aer/dynamic-HW clean) |
| 2026-08-30 | IBM job (generated, not run) | `jobs/submit_ibm_job.py` (`--dry-run` offline; else `SamplerV2` least-busy ≥3q, writes `jobs/<id>.json`), `jobs/compare_counts.py` (`|P_hw(out=0) − P_sim(out=0)| ≤ tol`), `jobs/README.md` |

## Nodes

| Date | Node | Status | Notes |
|------|------|--------|-------|
| 2026-08-30 | F1 — Complex Hilbert space | green (agent-solved) | confirmed skip. `inner_product` = `np.vdot` (conj-linear in arg 1), `norm` = sqrt of real part. 47 tests: conj symmetry, bi-/conj-linearity, positive-definiteness, norm↔√⟨a\|a⟩, Cauchy–Schwarz. |
| 2026-08-30 | F9 — Multi-qubit computational basis | green (agent-solved) | WARMUP (indexing only). Big-endian pinned: q0 = MSB, `index = Σ b_k 2^(n-1-k)`. `bits_to_index`/`index_to_bits`/`basis_vector` + range guards. 18 tests incl. the project **endianness contract test** ((1,0,1)↔5) and kron consistency. |
| 2026-08-30 | F10 — Outer products & projectors | green (agent-solved) | WARMUP (linear algebra). `outer(a,b)` = `np.outer(a, b.conj())`; `rank1_projector(v)` = `\|v><v\| / <v\|v>`. 19 tests: action `\|a><b\|v = <b\|v>a`, adjoint swap, idempotent/Hermitian/trace-1, fixes v & kills v⊥, resolution of identity, mutual orthogonality. |
| 2026-08-30 | F11 — Pauli operators | green (agent-solved) | CORE. User first opted to hand-implement, then reversed ("do F11 by yourself and proceed"). Fixed 2×2 matrices I/X/Y/Z. 20 tests: shape, Hermitian, unitary, involution, traceless, spectrum {−1,+1}, anticommutation, `X@Z = −iY`, basis action. |
| 2026-08-30 | F12 — Pauli algebra / group relations | green (agent-solved) | CORE. Agent-written after two claimed user edits never landed in the working tree (verified via md5/mtime). `pauli_mul(a,b) -> (phase,label)`: I-unit, `P²=I`, cyclic +i / anti-cyclic −i. 46 tests: 16-pair matrix match, cyclic/anti-cyclic, commutation from rule, 16-element group closure + associativity + inverses. |
| 2026-08-30 | F13 — Hadamard gate | green (agent-solved, fast mode) | CORE. `H = [[1,1],[1,-1]]/√2`. 9 tests: Hermitian, unitary, involution, det = −1, Z-basis↔X-basis, `HXH=Z`, `HZH=X`, `HYH=−Y`. |
| 2026-08-30 | F14 — Controlled-unitary / CNOT | green (agent-solved, fast mode) | CORE. `controlled_unitary(u,n,c,t)` by big-endian bit surgery; `cnot`/`cz` on top. 10 tests: literal matrix, involution/unitary, basis permutation, Bell creation, `(H⊗H)·CNOT₀₁·(H⊗H)=CNOT₁₀`, CZ symmetry, 3-qubit embedding, projector-formula match. |
| 2026-08-30 | F15 — Measurement postulate / Born rule | green (agent-solved, fast mode) | CORE. `is_valid_measurement` (ΣM†M=I), `born_probabilities`, `collapse`. 13 tests: completeness accept/reject, distribution, `p=|amp|²`, normalized + idempotent collapse. |
| 2026-08-30 | F16 — Projective measurement | green (agent-solved, fast mode) | CORE. `spectral_projectors` (eigval-grouped), `expectation_value`, `born_probs`, `collapse`. 12 tests: projector algebra, ΣλP=A, `<A>=Σλp=⟨ψ\|A\|ψ⟩`, deterministic repeat. |
| 2026-08-30 | F17 — Partial / subsystem measurement | green (agent-solved, fast mode) | CORE. `subsystem_projector` (P⊗I), `partial_measurement_probabilities`, `partial_collapse`. 24 tests incl. the map signature: **Alice's 4 outcomes = ¼ for every input ψ**, conditional q2 = `X^m1 Z^m0 ψ`. |
| 2026-08-30 | F18 — Density operator | green (agent-solved, fast mode) | CORE. `density_from_state`/`_ensemble`, `is_density_matrix`, `purity`, `expectation`. 12 tests: pure ρ²=ρ, mixed purity<1, PSD/trace-1 predicate, `tr(ρA)=⟨ψ\|A\|ψ⟩`, spectrum-preserving evolution. |
| 2026-08-30 | F19 — Partial trace | green (agent-solved, fast mode) | CORE. `partial_trace` via `einsum` index identification (big-endian). 11 tests: trace-preserving, product factorizes, Bell→I/2, local-expectation consistency, linearity, shapes. |
| 2026-08-30 | F20 — Entanglement entropy | green (agent-solved, fast mode) | CORE. `von_neumann_entropy` (base 2, 0log0=0), `entanglement_entropy` (reduces via F19), `is_product_state`. 13 tests: `S(I/d)=log₂d`, Bell=1, `S(ρ_A)=S(ρ_B)`, product=0, bounds. |
| 2026-08-30 | F21 — Bell states / Bell basis | green (agent-solved, fast mode) | CORE. `phi_plus`, `bell_state(b0,b1)=CNOT·(H⊗I)\|b0b1⟩`, `bell_basis`. 9 tests: orthonormal, unitary basis, each maximally entangled (F20), `(H⊗I)·CNOT` decoder, local-Pauli orbit. |
| 2026-08-30 | F22 — Bell-pair preparation circuit | green (agent-solved, fast mode) | CORE. `bell_pair_unitary = CNOT₀₁·(H⊗I)`, `prepare_bell_pair`. 4 tests: output = \|Φ⁺⟩, composition + unitarity, both reductions I/2 & S=1, maps comp basis → Bell basis. |
| 2026-08-30 | F23 — Classical control / deferred measurement | green (agent-solved, fast mode) | CORE. `apply_conditional`, `bob_correction` (Z^m0 X^m1), `defer` = `controlled_unitary`. 16 tests: bit-0 identity / bit-1 apply, correction inverts `X^m1 Z^m0 ψ`, `defer`=CNOT/CZ, deferred-measurement equivalence branch-by-branch + averaged ρ. |
| 2026-08-30 | F24 — Teleportation identity | green (agent-solved, fast mode) | CORE. `pre_measurement_state` (H1,CNOT₁₂,CNOT₀₁,H0 on ψ⊗\|00⟩), `teleportation_branches`. 16 tests: circuit match, amplitude block = ½·`X^m1 Z^m0 ψ`, all branch probs = ¼, branch q2 states are the Pauli images. |
| 2026-08-30 | T — Teleportation circuit integration | green (agent-solved, fast mode) | Qiskit + local Aer. `tests/test_integration_teleportation.py`, 57 cases: real circuit (measure + `if_test` correction) → `save_statevector` → `partial_trace(sv,[0,1])` fidelity with input = 1.0 for 24 random states; deferred circuit exact product `\|ψ⟩_q2 ⊗ \|+⟩ ⊗ \|+⟩`; uncorrected-branch guard; Aer counts χ² uniformity of Alice's outcomes. |
