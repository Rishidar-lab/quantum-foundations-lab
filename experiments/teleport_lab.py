"""PHASE 3 -- quantum teleportation run as a genuine experiment.

For each input state we report, side by side and never conflated:

    STATEVECTOR PROBABILITY   exact value from the ideal simulator
    FINITE-SHOT FREQUENCY     estimate from N shots, with a 95% CI

Pipeline per state:
    1. construct the source |psi> (Bloch vector s)
    2. teleport it (real dynamic circuit AND coherent deferred circuit)
    3. reconstruct Bob's qubit by 3-basis tomography from shot data
    4. fidelity: ideal (statevector) vs empirical (tomography), with CI
    5. Alice's 2-bit outcome distribution: predicted 1/4 vs observed frequency
    6. everything serialised to a machine-readable dict, tagged SIMULATED

Run:  python -m experiments.teleport_lab --shots 8192 --out experiments/results/teleportation_sim.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

from experiments.stats import CHI2_0001, Interval, chi_square_uniform, wilson_interval
from teleportation import build_teleportation_circuit

# ---- source states ---------------------------------------------------------

NAMED: dict[str, tuple[float, float]] = {
    "|0>": (0.0, 0.0),
    "|1>": (np.pi, 0.0),
    "|+>": (np.pi / 2, 0.0),
    "|->": (np.pi / 2, np.pi),
    "|+i>": (np.pi / 2, np.pi / 2),
    "|-i>": (np.pi / 2, -np.pi / 2),
}
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def state_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def bloch_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def random_source(seed: int) -> tuple[str, float, float]:
    rng = np.random.default_rng(seed)
    theta = float(np.arccos(1 - 2 * rng.random()))  # uniform on the sphere
    phi = float(rng.uniform(0, 2 * np.pi))
    return f"random#{seed}", theta, phi


# ---- circuits ------------------------------------------------------------

_BASIS_ROTATION = {"X": ("h",), "Y": ("sdg", "h"), "Z": ()}


def _tomography_circuit(psi: np.ndarray, basis: str) -> QuantumCircuit:
    """Teleport `psi`, then measure Bob's qubit (index 2) in the given basis."""
    qc = build_teleportation_circuit(psi, with_correction=True)
    for gate in _BASIS_ROTATION[basis]:
        getattr(qc, gate)(qc.qubits[2])
    creg = ClassicalRegister(1, "bob")
    qc.add_register(creg)
    qc.measure(qc.qubits[2], creg[0])
    return qc


# ---- one experiment ----------------------------------------------------


@dataclasses.dataclass
class TeleportExperiment:
    label: str
    theta: float
    phi: float
    shots: int

    # statevector (exact)
    ideal_fidelity: float = 0.0
    bob_z_prob_statevector: tuple[float, float] = (0.0, 0.0)
    alice_prob_statevector: float = 0.25

    # finite-shot (estimates)
    bloch_measured: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bloch_true: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bloch_ci: dict = dataclasses.field(default_factory=dict)
    empirical_fidelity: float = 0.0
    empirical_fidelity_ci: tuple[float, float] = (0.0, 0.0)
    alice_freq: dict = dataclasses.field(default_factory=dict)
    alice_chi_square: tuple[float, int, float] = (0.0, 3, 0.0)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["source"] = "SIMULATED"
        d["simulator"] = "qiskit-aer AerSimulator (ideal, no noise model)"
        return d


def run_experiment(label: str, theta: float, phi: float, shots: int, seed: int) -> TeleportExperiment:
    psi = state_from_angles(theta, phi)
    s_true = bloch_from_angles(theta, phi)
    exp = TeleportExperiment(label=label, theta=theta, phi=phi, shots=shots)
    exp.bloch_true = tuple(float(x) for x in s_true)

    # --- STATEVECTOR PROBABILITIES (exact) -------------------------------
    deferred = build_teleportation_circuit(psi, deferred=True)
    bob_ideal = partial_trace(Statevector(deferred), [0, 1])
    exp.ideal_fidelity = float(
        state_fidelity(bob_ideal, DensityMatrix(np.outer(psi, psi.conj())), validate=False)
    )
    exp.bob_z_prob_statevector = (float(abs(psi[0]) ** 2), float(abs(psi[1]) ** 2))
    exp.alice_prob_statevector = 0.25

    # --- FINITE-SHOT FREQUENCIES (estimates) ---------------------------
    sim = AerSimulator()
    r_meas, r_ci = [], {}
    for k, basis in enumerate("XYZ"):
        qc = _tomography_circuit(psi, basis)
        counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed + k).result().get_counts()
        n0 = sum(v for key, v in counts.items() if key.split()[0] == "0")
        ci = wilson_interval(n0, shots)  # P(bob = 0) in this basis
        r = 2 * ci.point - 1  # <sigma_k> = P(0) - P(1)
        r_meas.append(r)
        r_ci[basis] = {"expectation": r, "ci_low": 2 * ci.low - 1, "ci_high": 2 * ci.high - 1}
    exp.bloch_measured = tuple(float(x) for x in r_meas)
    exp.bloch_ci = r_ci

    # reconstruct rho from the measured Bloch vector, fidelity with the source
    rho = 0.5 * (np.eye(2) + r_meas[0] * PAULI["X"] + r_meas[1] * PAULI["Y"] + r_meas[2] * PAULI["Z"])
    exp.empirical_fidelity = float(np.real(psi.conj() @ rho @ psi))
    # F = (1 + r . s)/2 ; propagate the per-component Wilson bounds
    lo = 0.5 * (1 + sum(min(r_ci[b]["ci_low"] * s, r_ci[b]["ci_high"] * s) for b, s in zip("XYZ", s_true)))
    hi = 0.5 * (1 + sum(max(r_ci[b]["ci_low"] * s, r_ci[b]["ci_high"] * s) for b, s in zip("XYZ", s_true)))
    exp.empirical_fidelity_ci = (float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)))

    # Alice's 2-bit outcome distribution
    qc_alice = build_teleportation_circuit(psi, with_correction=True)
    counts = sim.run(transpile(qc_alice, sim), shots=shots, seed_simulator=seed + 9).result().get_counts()
    tally: dict[str, int] = {}
    for key, v in counts.items():
        tally[key.replace(" ", "")] = tally.get(key.replace(" ", ""), 0) + v
    exp.alice_freq = {k: {"count": v, "freq": v / shots} for k, v in sorted(tally.items())}
    stat, dof = chi_square_uniform(tally, 4)
    exp.alice_chi_square = (stat, dof, CHI2_0001[dof])

    return exp


# ---- suite + serialisation ------------------------------------------


def run_suite(shots: int, seed: int, n_random: int = 4) -> list[TeleportExperiment]:
    out = []
    for i, (label, (theta, phi)) in enumerate(NAMED.items()):
        out.append(run_experiment(label, theta, phi, shots, seed + 100 * i))
    for j in range(n_random):
        label, theta, phi = random_source(seed + j)
        out.append(run_experiment(label, theta, phi, shots, seed + 100 * (len(NAMED) + j)))
    return out


def suite_report(results: list[TeleportExperiment], shots: int, seed: int) -> dict:
    return {
        "experiment": "quantum_teleportation",
        "source": "SIMULATED",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "shots": shots,
        "seed": seed,
        "note": (
            "ideal_fidelity is exact (statevector). empirical_fidelity is a "
            "3-basis-tomography estimate from finite shots; its CI is a 95% "
            "Wilson-propagated interval. The two are NOT claimed to be equal."
        ),
        "results": [r.as_dict() for r in results],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--random", type=int, default=4, help="number of random source states")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    results = run_suite(args.shots, args.seed, args.random)
    report = suite_report(results, args.shots, args.seed)

    hdr = f"{'state':<10} {'F_ideal':>8} {'F_emp':>8} {'F_emp 95% CI':>22}  {'chi2(Alice)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        lo, hi = r.empirical_fidelity_ci
        chi, dof, crit = r.alice_chi_square
        print(
            f"{r.label:<10} {r.ideal_fidelity:8.5f} {r.empirical_fidelity:8.4f} "
            f"  [{lo:6.4f}, {hi:6.4f}]     {chi:7.2f} < {crit:.2f}? {'y' if chi < crit else 'N'}"
        )
    print("-" * len(hdr))
    print("F_ideal: exact statevector.  F_emp: finite-shot tomography estimate (not identical).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
