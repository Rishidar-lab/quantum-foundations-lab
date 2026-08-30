"""PHASE 4 -- density matrices & noise laboratory.

Controlled decoherence experiments. For every channel the NumPy Kraus result is
cross-checked against the equivalent `qiskit_aer.noise` error (density-matrix
simulation), then a parameter sweep records how the state degrades:

    fidelity   F(rho_out, |psi>)          -- how close to the input
    purity     Tr(rho_out^2)              -- how mixed
    <X>,<Y>,<Z> expectation values        -- Bloch-vector shrinkage
    P(0), P(1) measurement distribution   -- what a Z-basis experiment would see

Plus a readout-error experiment and an end-to-end "teleportation under gate
noise" degradation curve.

Everything is SIMULATED and seeded. Run:
    python -m experiments.noise_lab --out experiments/results/noise_sim.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
    phase_damping_error,
)

from experiments import channels as ch
from teleportation import build_teleportation_circuit

X, Y, Z = ch.X, ch.Y, ch.Z
_ONE_QUBIT_GATES = ["id", "sx", "x", "rz", "ry", "rx", "h", "u", "u1", "u2", "u3"]
_TWO_QUBIT_GATES = ["cx", "cz"]


# ---- qiskit equivalents of the NumPy channels -----------------------------


def qiskit_error(channel: str, param: float):
    return {
        "bit_flip": lambda p: pauli_error([("X", p), ("I", 1 - p)]),
        "phase_flip": lambda p: pauli_error([("Z", p), ("I", 1 - p)]),
        "bit_phase_flip": lambda p: pauli_error([("Y", p), ("I", 1 - p)]),
        "depolarizing": lambda p: depolarizing_error(p, 1),
        "amplitude_damping": lambda g: amplitude_damping_error(g),
        "phase_damping": lambda g: phase_damping_error(g),
    }[channel](param)


# ---- state + metrics --------------------------------------------------


def _state(theta: float, phi: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def state_metrics(rho: np.ndarray, psi: np.ndarray) -> dict:
    rho = np.asarray(rho, dtype=complex)
    p = np.clip(np.real(np.diag(rho)), 0, 1)
    return {
        "fidelity": float(np.real(psi.conj() @ rho @ psi)),
        "purity": float(np.real(np.trace(rho @ rho))),
        "exp_x": float(np.real(np.trace(rho @ X))),
        "exp_y": float(np.real(np.trace(rho @ Y))),
        "exp_z": float(np.real(np.trace(rho @ Z))),
        "prob_0": float(p[0]),
        "prob_1": float(p[1]),
    }


# ---- NumPy vs Aer for a single channel application -----------------


def numpy_channel_rho(channel: str, param: float, psi: np.ndarray) -> np.ndarray:
    return ch.apply_channel(np.outer(psi, psi.conj()), ch.CHANNELS[channel](param))


def aer_channel_rho(channel: str, param: float, theta: float, phi: float) -> np.ndarray:
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.id(0)  # the noisy identity -- carries the channel
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(qiskit_error(channel, param), ["id"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    tqc = transpile(qc, sim, optimization_level=0)
    rho = sim.run(tqc).result().data(0)["density_matrix"]
    return np.asarray(DensityMatrix(rho).data)


# ---- experiments ---------------------------------------------------


def channel_sweep(channel: str, params: list[float], theta: float, phi: float) -> dict:
    psi = _state(theta, phi)
    points = []
    for p in params:
        rho = numpy_channel_rho(channel, p, psi)
        points.append({"param": float(p), **state_metrics(rho, psi)})
    return {
        "channel": channel,
        "input_state": {"theta": theta, "phi": phi},
        "param_name": "gamma" if "damping" in channel else "p",
        "points": points,
    }


def readout_experiment(p01: float, p10: float, theta: float, phi: float) -> dict:
    psi = _state(theta, phi)
    ideal = np.abs(psi) ** 2
    conf = ch.readout_confusion(p01, p10)
    distorted_np = ch.apply_readout(ideal, conf)

    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.measure(0, 0)
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    sim = AerSimulator(noise_model=nm)
    shots = 40000
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=7).result().get_counts()
    distorted_shots = np.array([counts.get("0", 0), counts.get("1", 0)]) / shots

    return {
        "experiment": "readout_error",
        "p01": p01,
        "p10": p10,
        "ideal_distribution": ideal.tolist(),
        "distorted_distribution_numpy": distorted_np.tolist(),
        "distorted_distribution_shots": distorted_shots.tolist(),
        "shots": shots,
        "tv_distance_numpy_vs_ideal": float(0.5 * np.abs(distorted_np - ideal).sum()),
        "tv_distance_shots_vs_numpy": float(0.5 * np.abs(distorted_shots - distorted_np).sum()),
    }


def teleportation_under_gate_noise(params: list[float], theta: float, phi: float) -> dict:
    psi = _state(theta, phi)
    target = np.outer(psi, psi.conj())
    points = []
    for p in params:
        qc = build_teleportation_circuit(psi, deferred=True)
        qc.save_density_matrix()
        nm = NoiseModel()
        if p > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), _ONE_QUBIT_GATES)
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), _TWO_QUBIT_GATES)
        sim = AerSimulator(method="density_matrix", noise_model=nm)
        tqc = transpile(qc, sim, optimization_level=0)
        rho_full = DensityMatrix(sim.run(tqc).result().data(0)["density_matrix"])
        rho_bob = np.asarray(partial_trace(rho_full, [0, 1]).data)
        points.append(
            {
                "gate_depolarizing_p": float(p),
                "bob_fidelity": float(state_fidelity(rho_bob, target, validate=False)),
                "bob_purity": float(np.real(np.trace(rho_bob @ rho_bob))),
            }
        )
    return {"experiment": "teleportation_under_gate_noise", "input_state": {"theta": theta, "phi": phi}, "points": points}


# ---- suite -------------------------------------------------------


DEFAULT_PARAMS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]


def run_suite(seed: int = 2026) -> dict:
    theta, phi = 1.1780972450961724, 0.7853981633974483  # a generic off-axis state
    report = {
        "experiment": "noise_laboratory",
        "source": "SIMULATED",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "seed": seed,
        "input_state": {"theta": theta, "phi": phi},
        "note": (
            "Kraus channels (NumPy) are cross-checked against qiskit_aer.noise in "
            "tests/test_noise_lab.py. Sweeps and the teleportation degradation curve "
            "are density-matrix simulations; no hardware data."
        ),
        "channel_sweeps": [channel_sweep(c, DEFAULT_PARAMS, theta, phi) for c in ch.CHANNELS],
        "readout": readout_experiment(0.03, 0.06, theta, phi),
        "teleportation_degradation": teleportation_under_gate_noise(
            [0.0, 0.005, 0.01, 0.02, 0.04, 0.08], theta, phi
        ),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report = run_suite(args.seed)

    print(f"{'channel':<20} {'p/gamma':>8} {'fidelity':>9} {'purity':>8} {'<Z>':>7}")
    print("-" * 56)
    for sweep in report["channel_sweeps"]:
        for pt in sweep["points"]:
            if pt["param"] in (0.0, 0.1, 0.5):
                print(
                    f"{sweep['channel']:<20} {pt['param']:8.3f} {pt['fidelity']:9.4f} "
                    f"{pt['purity']:8.4f} {pt['exp_z']:7.3f}"
                )
    print("-" * 56)
    r = report["readout"]
    print(f"readout(p01={r['p01']}, p10={r['p10']}): TV(distorted,ideal) = {r['tv_distance_numpy_vs_ideal']:.4f}")
    print("teleportation vs gate depolarizing p:")
    for pt in report["teleportation_degradation"]["points"]:
        print(f"    p={pt['gate_depolarizing_p']:.3f} -> F_bob={pt['bob_fidelity']:.4f}  purity={pt['bob_purity']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
