#!/usr/bin/env python
"""PHASE 5 -- the full teleportation-verification pipeline, local stages.

    stage 1  explicit linear algebra     (NumPy kata chain)          exact
    stage 2  ideal Qiskit simulation     (Statevector)               exact
    stage 3  shot simulation             (AerSimulator, N shots)     estimate + CI
    stage 4  noisy simulation            (AerSimulator + noise)      estimate + CI
    stage 5  IBM hardware                 (filled by submit/compare)  <optional>

Stages 1-4 need NO credentials and run offline. The observable is
    P(out = 0)  for  build_verification_circuit(theta, phi)
which equals the teleportation fidelity (perfect teleportation -> 1.0).

    python jobs/pipeline.py --state plus
    python jobs/pipeline.py --theta 1.17 --phi 0.83 --shots 8192 --out experiments/results/pipeline_sim.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "katas"))

from qiskit import transpile  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

from experiments.stats import wilson_interval  # noqa: E402
from teleportation import build_verification_circuit  # noqa: E402

NAMED = {
    "zero": (0.0, 0.0),
    "one": (np.pi, 0.0),
    "plus": (np.pi / 2, 0.0),
    "minus": (np.pi / 2, np.pi),
    "plus_i": (np.pi / 2, np.pi / 2),
    "minus_i": (np.pi / 2, -np.pi / 2),
}


def _p_out_zero(counts: dict, shots: int) -> tuple[int, float]:
    n0 = sum(v for k, v in counts.items() if k.split()[0] == "0")
    return n0, n0 / shots


def stage1_linear_algebra(theta: float, phi: float) -> dict:
    """NumPy: teleport |psi>, undo prep on q2, read P(0). Exact."""
    from _deps import kata as load

    k11, k24 = load("11_pauli_operators"), load("24_teleportation_identity")
    X, Z = k11.pauli_x(), k11.pauli_z()
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
    ry_inv = np.array(
        [[np.cos(-theta / 2), -np.sin(-theta / 2)], [np.sin(-theta / 2), np.cos(-theta / 2)]],
        dtype=complex,
    )
    rz_inv = np.diag([np.exp(1j * phi / 2), np.exp(-1j * phi / 2)])  # RZ(-phi)
    p0 = 0.0
    for (m0, m1), (prob, q2) in k24.teleportation_branches(psi).items():
        corrected = np.linalg.matrix_power(Z, m0) @ np.linalg.matrix_power(X, m1) @ q2
        unprepared = ry_inv @ rz_inv @ corrected
        p0 += prob * abs(unprepared[0]) ** 2
    return {"stage": 1, "method": "explicit linear algebra (NumPy)", "kind": "SIMULATED", "p_out0": float(p0), "exact": True}


def stage2_ideal_statevector(theta: float, phi: float) -> dict:
    qc = build_verification_circuit(theta, phi, deferred=True)
    # drop the final measurement, read the amplitude of q2 == 0
    qc.remove_final_measurements()
    sv = Statevector(qc)
    probs = sv.probabilities([2])  # marginal on Bob's qubit (index 2)
    return {"stage": 2, "method": "ideal Qiskit Statevector", "kind": "SIMULATED", "p_out0": float(probs[0]), "exact": True}


def stage3_shot_sim(theta: float, phi: float, shots: int, seed: int) -> dict:
    qc = build_verification_circuit(theta, phi)
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()
    n0, f = _p_out_zero(counts, shots)
    ci = wilson_interval(n0, shots)
    return {
        "stage": 3, "method": "AerSimulator shot sampling", "kind": "SIMULATED",
        "shots": shots, "counts": counts, "p_out0": f,
        "ci95": [ci.low, ci.high], "exact": False,
    }


def stage4_noisy_sim(theta: float, phi: float, shots: int, seed: int, gate_p: float) -> dict:
    qc = build_verification_circuit(theta, phi)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 1), ["sx", "x", "rz", "ry", "h", "u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 2), ["cx", "cz"])
    sim = AerSimulator(noise_model=nm)
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()
    n0, f = _p_out_zero(counts, shots)
    ci = wilson_interval(n0, shots)
    return {
        "stage": 4, "method": "AerSimulator + depolarizing noise", "kind": "SIMULATED",
        "gate_depolarizing_p": gate_p, "shots": shots, "counts": counts, "p_out0": f,
        "ci95": [ci.low, ci.high], "exact": False,
    }


def run_pipeline(theta: float, phi: float, shots: int, seed: int, gate_p: float) -> dict:
    s1 = stage1_linear_algebra(theta, phi)
    s2 = stage2_ideal_statevector(theta, phi)
    s3 = stage3_shot_sim(theta, phi, shots, seed)
    s4 = stage4_noisy_sim(theta, phi, shots, seed, gate_p)
    return {
        "experiment": "teleportation_verification_pipeline",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "payload": {"theta": theta, "phi": phi},
        "observable": "P(out = 0) == teleportation fidelity",
        "stages": [s1, s2, s3, s4],
        "stage5_hardware": None,
        "consistency": {
            "s1_vs_s2_abs_diff": abs(s1["p_out0"] - s2["p_out0"]),
            "s3_ci_contains_s2": s3["ci95"][0] - 1e-9 <= s2["p_out0"] <= s3["ci95"][1] + 1e-9,
        },
        "labels": {"SIMULATED": "stages 1-4", "HARDWARE": "stage 5 (not run)"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", choices=sorted(NAMED))
    ap.add_argument("--theta", type=float, default=1.17)
    ap.add_argument("--phi", type=float, default=0.83)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gate-p", type=float, default=0.01, help="depolarizing rate for stage 4")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    theta, phi = NAMED[args.state] if args.state else (args.theta, args.phi)
    report = run_pipeline(theta, phi, args.shots, args.seed, args.gate_p)

    print(f"payload RY({theta:.4f}) RZ({phi:.4f}) |0>   observable: P(out=0) = fidelity\n")
    for s in report["stages"]:
        ci = f"  95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]" if "ci95" in s else "  (exact)"
        print(f"  stage {s['stage']}  {s['method']:<38} P(out=0) = {s['p_out0']:.4f}{ci}   [{s['kind']}]")
    print(f"  stage 5  {'IBM hardware':<38} (not run -- see jobs/submit_ibm_job.py)   [HARDWARE]")
    c = report["consistency"]
    print(f"\nconsistency: |s1-s2| = {c['s1_vs_s2_abs_diff']:.2e} ; s3 CI contains s2: {c['s3_ci_contains_s2']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
