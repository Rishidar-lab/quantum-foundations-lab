# IBM Quantum hardware jobs (PHASE 5)

These scripts are **generated, not run** by the build. CI never touches them.
Stages 1-4 of the pipeline run fully **offline**; only stage 5 needs credentials,
and a submission always requires an **explicit spend confirmation** -- having
credentials is never treated as permission to submit.

## Scripts

| script | needs credentials | spends queue time | what it does |
|---|---|---|---|
| `pipeline.py` | no | no | runs stages 1-4 (linear algebra → ideal → shots → noisy) and prints/saves a SIMULATED comparison |
| `preflight.py` | reads presence only | no | validates that a submission *could* succeed; submits nothing |
| `submit_ibm_job.py` | yes | **yes, after confirmation** | transpiles for a backend and submits via `SamplerV2` |
| `compare_counts.py` | yes | no | retrieves a finished job, builds the HARDWARE record |

## One-time credential setup

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", token="<YOUR_TOKEN>", set_as_default=True)
```
or `export QISKIT_IBM_TOKEN=...`. Credentials are **never** committed
(`.gitignore` covers `.qiskit/`, `*token*`, `.env`, ...).

## Flow

```
python jobs/pipeline.py --state plus --out experiments/results/pipeline_plus.json
python jobs/preflight.py
python jobs/submit_ibm_job.py --state plus --shots 4096          # prompts: retype backend name
#   (or --yes-spend-credits for non-interactive)
python jobs/compare_counts.py jobs/<job_id>.json --record jobs/hw_plus.json
#   then transcribe into docs/hardware_runs.md
```

## Circuit under test

`teleportation.build_verification_circuit(theta, phi)`:
payload `|psi> = RZ(phi) RY(theta)|0>` on q0 → teleport to q2 →
`RY(-theta) RZ(-phi)` on q2 → measure q2 into register `out`.
Perfect teleportation returns q2 to `|0>`, so `P(out = 0)` **is** the fidelity.
Uses `ry/rz/cx/cz` + `if_test` only (no `StatePreparation` multiplexer) so it
transpiles cleanly for dynamic-circuit hardware.

`jobs/*.json` (job metadata + hardware records) are git-ignored, except this
README and `.gitkeep`.
