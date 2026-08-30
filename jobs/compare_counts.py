#!/usr/bin/env python
"""Compare a finished IBM hardware job's counts to the local Aer expectation.

    *** NOT run by CI. *** Requires IBM Quantum credentials to retrieve the job.

Usage
-----
    python jobs/compare_counts.py jobs/<job_id>.json [--tol 0.05]

Reads the metadata written by submit_ibm_job.py, retrieves the job result,
extracts P(out = 0) (teleportation success probability on Bob's qubit), and
checks

    | P_hardware(out=0) - P_simulator(out=0) | <= tol

Exit status 0 on agreement, 1 on mismatch, 2 if the job is not finished.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _p_zero_from_bitarray(bit_array, shots: int) -> float:
    counts = bit_array.get_counts()  # 1-bit register -> {"0": n0, "1": n1}
    return counts.get("0", 0) / shots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta", type=pathlib.Path, help="jobs/<job_id>.json from submit_ibm_job.py")
    ap.add_argument("--tol", type=float, default=0.05, help="max allowed |P_hw - P_sim| (default 0.05)")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    shots = meta["shots"]
    p_sim0 = meta["local_expectation"]["p_out0"]

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    job = service.job(meta["job_id"])
    status = job.status()
    if str(status) not in ("JobStatus.DONE", "DONE"):
        print(f"job {meta['job_id']} not finished (status {status})")
        return 2

    pub_result = job.result()[0]
    p_hw0 = _p_zero_from_bitarray(pub_result.data.out, shots)
    delta = abs(p_hw0 - p_sim0)

    print(f"backend         : {meta['backend']}")
    print(f"payload         : RY({meta['theta']:.4f}) RZ({meta['phi']:.4f}) |0>")
    print(f"shots           : {shots}")
    print(f"P_sim(out=0)     : {p_sim0:.4f}")
    print(f"P_hardware(out=0): {p_hw0:.4f}")
    print(f"|delta|          : {delta:.4f}   (tol {args.tol})")

    if delta <= args.tol:
        print("PASS: hardware counts agree with simulator within tolerance.")
        return 0
    print("FAIL: hardware counts deviate beyond tolerance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
