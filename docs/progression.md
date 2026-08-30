# Progression: from a complex number to a teleported qubit

This is the learning path through the repository. Work it with the runner:

```
python -m lessons.run --list
python -m lessons.run                 # your current lesson
python -m lessons.run --lesson 4 --run
python -m lessons.run --next          # advance when you've done the challenge
```

Every lesson has ten parts: **concept · why it exists · mathematics · tiny worked
example · code · prediction · actual result · interpretation · common
misconception · challenge**. Challenge solutions are only shown with `--reveal`.

## The dependency spine

The katas were built bottom-up from QM postulates (`research/dependency_map.md`).
The lessons walk the same spine, then turn it into experiments.

```
 postulates:  Hilbert space ─ unitary evolution ─ measurement ─ tensor product
                   │                │                 │              │
 F1 amplitudes ────┘                │                 │              │
 F1 inner products / bras           │                 │              │
 F9 basis states, endianness ───────┼─────────────────┼──────────────┘
 F13 Hadamard / superposition ──────┘                 │
 F11 Pauli operators                                  │
 F12 Pauli group                                      │
 F15 measurement, Born rule ──────────────────────────┘
 F16 projective measurement, expectation values
 F8  tensor products
 F14 CNOT / controlled-U
 F18 density matrices
 F19 partial trace          ┐
 F20 entanglement entropy   ├── "what does a subsystem look like?"
 F21/F22 Bell states/pair   ┘
 F4  global vs relative phase
 F17 partial measurement    ┐
 F24 teleportation identity ├── the protocol, as algebra
 F23 classical control      ┘
 T   full Qiskit circuit + local Aer  (the protocol, as a circuit)
        │
 LAB teleportation experiment   (shots, confidence intervals, tomography)
 LAB noise laboratory           (channels, density matrices, degradation)
 LAB hardware pipeline          (linear algebra → ideal → shots → noisy → IBM)
```

## Lesson ↔ node ↔ code map

| # | node | lesson | primary code | tests |
|--:|------|--------|--------------|-------|
| 1 | F1 | complex amplitudes, the state vector | `katas/01_hilbert_space.py` | `test_01`, `test_foundations_crosscheck` |
| 2 | F1 | bras, inner products, orthogonality | `katas/01_hilbert_space.py` | `test_01`, foundations |
| 3 | F9 | basis states & qubit ordering | `katas/09_computational_basis_nq.py` | `test_09` (endianness contract) |
| 4 | F13 | superposition & the Hadamard gate | `katas/13_hadamard.py` | `test_13`, foundations |
| 5 | F11 | the Pauli operators | `katas/11_pauli_operators.py` | `test_11`, foundations |
| 6 | F12 | the Pauli group | `katas/12_pauli_algebra.py` | `test_12` |
| 7 | F15 | measurement & the Born rule | `katas/15_measurement_postulate.py` | `test_15`, foundations |
| 8 | F16 | expectation values | `katas/16_projective_measurement.py` | `test_16`, foundations |
| 9 | F8 | tensor products & composite systems | `np.kron`, `foundations/crosscheck.py` | foundations |
| 10 | F14 | CNOT & controlled operations | `katas/14_cnot_controlled_u.py` | `test_14`, foundations |
| 11 | F18 | density matrices | `katas/18_density_operator.py` | `test_18`, foundations |
| 12 | F19 | partial trace & reduced states | `katas/19_partial_trace.py` | `test_19`, `test_foundations_weakness_fixes` |
| 13 | F20 | entanglement entropy | `katas/20_entanglement_entropy.py` | `test_20`, foundations |
| 14 | F21/F22 | Bell states & the Bell pair | `katas/21_*`, `katas/22_*` | `test_21`, `test_22`, `test_foundations_weakness_fixes` |
| 15 | F4 | global vs relative phase | `foundations/crosscheck.py` | foundations, `tools/mutation_audit.py` |
| 16 | F24 | the teleportation identity | `katas/24_teleportation_identity.py` | `test_24`, `test_foundations_weakness_fixes` |
| 17 | T | classical control, deferred measurement, the circuit | `teleportation/circuit.py`, `katas/23_*` | `test_23`, `test_integration_teleportation` |
| 18 | LAB | teleportation as an experiment | `experiments/teleport_lab.py` | `test_teleport_lab` |
| 19 | LAB | noise & density matrices | `experiments/noise_lab.py`, `experiments/channels.py` | `test_noise_lab` |
| 20 | LAB | the hardware pipeline | `jobs/pipeline.py`, `jobs/preflight.py` | `test_pipeline` |

## Where the rigor lives

- **`AUDIT.md`** — which tests are mathematical/physical invariants vs
  implementation-coupled, the tautologies found, and the mutation-testing results.
- **`tools/mutation_audit.py`** — 10 deliberately-wrong implementations; the suite
  catches 9 and correctly ignores the 1 that is physically invisible (global phase).
- **`foundations/`** — every concept computed twice (NumPy vs Qiskit) and asserted equal.
- **experiments** — every emitted number is labelled `SIMULATED`; frequencies always
  carry a confidence interval and are never equated with exact probabilities.
