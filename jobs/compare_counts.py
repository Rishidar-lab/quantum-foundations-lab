#!/usr/bin/env python
"""Compare a finished IBM hardware job to the local simulator and write a record.

    *** NOT run by CI. *** Requires IBM Quantum credentials to retrieve the job.

Usage
-----
    python jobs/compare_counts.py jobs/<job_id>.json [--tol 0.05] [--record docs/hardware_runs/<name>.json]

Reads the metadata written by submit_ibm_job.py, retrieves the job result, and
emits a HARDWARE-labelled record with:

    backend name / job id / shots / timestamp
    transpilation info (from the submission metadata)
    raw counts (hardware)
    derived probability P(out = 0) with a Wilson 95% CI
    simulator comparison  (ideal statevector, noiseless shots)
    error metric |P_hw - P_sim| and pass/fail vs --tol

Exit 0 agreement, 1 mismatch, 2 job unfinished.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _wilson(n0: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = n0 / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - h), min(1.0, c + h))


def _counts_from_pub(pub_result, shots: int) -> tuple[dict, int]:
    bit_array = pub_result.data.out  # 1-bit register named 'out'
    counts = {str(k): int(v) for k, v in bit_array.get_counts().items()}
    return counts, counts.get("0", 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta", type=pathlib.Path, help="jobs/<job_id>.json from submit_ibm_job.py")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--record", type=pathlib.Path, default=None, help="write the HARDWARE record JSON here")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    shots = meta["shots"]
    p_sim0 = meta["local_expectation"]["p_out0"]

    # ideal statevector prediction, offline
    from teleportation import build_verification_circuit
    from qiskit.quantum_info import Statevector

    qc = build_verification_circuit(meta["theta"], meta["phi"], deferred=meta.get("deferred", False))
    qc.remove_final_measurements()
    p_ideal0 = float(Statevector(qc).probabilities([2])[0])

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    job = service.job(meta["job_id"])
    if str(job.status()) not in ("JobStatus.DONE", "DONE"):
        print(f"job {meta['job_id']} not finished (status {job.status()})")
        return 2

    counts, n0 = _counts_from_pub(job.result()[0], shots)
    p_hw0 = n0 / shots
    ci = _wilson(n0, shots)
    delta = abs(p_hw0 - p_sim0)
    ok = delta <= args.tol

    record = {
        "label": "HARDWARE",
        "source": "IBM_QUANTUM_HARDWARE",
        "compared_utc": dt.datetime.now(dt.UTC).isoformat(),
        "job_id": meta["job_id"],
        "backend": meta["backend"],
        "submitted_utc": meta.get("submitted_utc"),
        "shots": shots,
        "payload": {"theta": meta["theta"], "phi": meta["phi"], "deferred": meta.get("deferred", False)},
        "transpilation": meta.get("isa", "see submission log"),
        "hardware": {
            "raw_counts": counts,
            "p_out0": p_hw0,
            "p_out0_ci95_wilson": list(ci),
        },
        "simulator": {
            "p_out0_ideal_statevector": p_ideal0,
            "p_out0_noiseless_shots": p_sim0,
            "label": "SIMULATED",
        },
        "comparison": {
            "abs_error_hw_vs_noiseless_shots": delta,
            "abs_error_hw_vs_ideal": abs(p_hw0 - p_ideal0),
            "tolerance": args.tol,
            "within_tolerance": ok,
            "teleportation_fidelity_estimate_hw": p_hw0,
        },
    }

    print(json.dumps(record, indent=2))
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2))
        print(f"\nwrote {args.record}", file=sys.stderr)

    print(f"\n{'PASS' if ok else 'FAIL'}: |P_hw - P_sim| = {delta:.4f} (tol {args.tol})", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
