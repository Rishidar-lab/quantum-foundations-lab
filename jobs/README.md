# IBM Quantum hardware jobs

These scripts are **generated, not run** by the build. CI never touches them.
They require IBM Quantum credentials.

## One-time credential setup

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", token="<YOUR_TOKEN>", set_as_default=True)
```

## Flow

1. **Offline check** (no credentials, no submission):
   ```
   python jobs/submit_ibm_job.py --dry-run --state plus
   ```
   Builds the verification circuit and prints the local Aer expectation
   `P(out=0)` (1.0 for perfect teleportation).

2. **Submit to hardware**:
   ```
   python jobs/submit_ibm_job.py --state plus --shots 4096
   ```
   Picks the least-busy >=3-qubit backend (or pass `--backend <name>`),
   transpiles to the backend ISA, submits via `SamplerV2`, and writes
   `jobs/<job_id>.json` (angles, backend, shots, Aer expectation).
   Add `--deferred` to submit the coherent CX/CZ variant (no dynamic circuit).

3. **Compare when the job finishes**:
   ```
   python jobs/compare_counts.py jobs/<job_id>.json --tol 0.05
   ```
   Retrieves the result, computes `P_hardware(out=0)`, and checks
   `|P_hw - P_sim| <= tol`. Exit 0 pass / 1 mismatch / 2 not finished.

## Circuit under test

`teleportation.build_verification_circuit(theta, phi)`:
payload `|psi> = RZ(phi) RY(theta)|0>` on q0 -> teleport to q2 ->
apply `RY(-theta) RZ(-phi)` to q2 -> measure q2 into register `out`.
Perfect teleportation returns q2 to `|0>`, so `P(out=1)` is a direct
infidelity estimate. Uses `ry/rz/cx/cz` + `if_test` only (no
`StatePreparation` multiplexer), so it transpiles cleanly for
dynamic-circuit hardware.

`jobs/*.json` are git-ignored (except this README and `.gitkeep`).
