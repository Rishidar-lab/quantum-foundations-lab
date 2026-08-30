# IBM Quantum hardware runs

This file is the **curated, committed log** of real hardware executions.
It starts empty. Nothing here is generated automatically, and no run is
fabricated. Raw job artifacts (`jobs/<job_id>.json`, records emitted by
`compare_counts.py`) are git-ignored; copy the fields you want to keep into a
row here by hand.

## How a row gets here

```
python jobs/preflight.py                              # 1. check config, submit nothing
python jobs/pipeline.py --state plus --out experiments/results/pipeline_plus.json   # 2. local stages 1-4
python jobs/submit_ibm_job.py --state plus --shots 4096                             # 3. prompts for spend confirmation
#   ... wait for the job to finish ...
python jobs/compare_counts.py jobs/<job_id>.json --record jobs/hw_plus.json         # 4. build the HARDWARE record
#   then transcribe the record into the table below
```

## Record schema (`compare_counts.py --record`)

| field | meaning |
|---|---|
| `label` / `source` | always `HARDWARE` / `IBM_QUANTUM_HARDWARE` |
| `job_id`, `backend` | IBM identifiers |
| `submitted_utc`, `compared_utc` | timestamps |
| `shots` | number of shots |
| `payload.{theta,phi,deferred}` | the teleported state and circuit variant |
| `transpilation` | ISA depth / op counts (from the submission log) |
| `hardware.raw_counts` | raw bitstring counts from the device |
| `hardware.p_out0` + `p_out0_ci95_wilson` | derived P(out=0) and its 95% CI |
| `simulator.p_out0_ideal_statevector` | exact prediction (SIMULATED) |
| `simulator.p_out0_noiseless_shots` | noiseless Aer estimate (SIMULATED) |
| `comparison.abs_error_hw_vs_ideal` | \|P_hw − P_ideal\| |
| `comparison.teleportation_fidelity_estimate_hw` | = `hardware.p_out0` |

`P(out = 0)` for `build_verification_circuit` is the teleportation fidelity:
1.0 means Bob's qubit returned exactly to `|0>` after un-preparation.

## Runs

_No hardware runs recorded yet._

| date (UTC) | backend | job id | state | shots | P_hw(out=0) | 95% CI | P_ideal | \|Δ\| | notes |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | (add rows after a real run) |
