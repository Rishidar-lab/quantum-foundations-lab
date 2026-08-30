#!/usr/bin/env python
"""Submit the teleportation verification circuit to IBM Quantum hardware.

    *** NOT run by CI and NOT run automatically by the build. ***

Requires IBM Quantum credentials. Save them once:

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", token="<TOKEN>", set_as_default=True)

or export QISKIT_IBM_TOKEN (and QISKIT_IBM_INSTANCE).

Nothing is submitted unless you EXPLICITLY confirm the spend, either by
retyping the backend name at the interactive prompt or by passing
`--yes-spend-credits` for non-interactive use. Credentials being present is
never treated as permission to submit.

Usage
-----
    python jobs/submit_ibm_job.py --dry-run                 # offline: build + Aer expectation
    python jobs/preflight.py                                # check config, submit nothing
    python jobs/submit_ibm_job.py --state plus --shots 4096 # will prompt for confirmation
    python jobs/submit_ibm_job.py --state plus --yes-spend-credits   # non-interactive submit
    python jobs/submit_ibm_job.py --theta 1.17 --phi 0.83 --backend ibm_torino
    python jobs/submit_ibm_job.py --deferred                # coherent variant, no dynamic circuit

On submission it writes jobs/<job_id>.json with the angles, backend, shots and
the local Aer expectation for P(out=0), consumed later by jobs/compare_counts.py.
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

from teleportation import build_verification_circuit  # noqa: E402

# named payload states as (RY theta, RZ phi) with |psi> = RZ(phi) RY(theta) |0>
NAMED_STATES = {
    "zero": (0.0, 0.0),
    "one": (np.pi, 0.0),
    "plus": (np.pi / 2, 0.0),
    "minus": (np.pi / 2, np.pi),
    "plus_i": (np.pi / 2, np.pi / 2),
}


def resolve_angles(args: argparse.Namespace) -> tuple[float, float]:
    if args.state is not None:
        return NAMED_STATES[args.state]
    return float(args.theta), float(args.phi)


def _p_out_zero(counts: dict[str, int], shots: int) -> float:
    """P(out = 0). 'out' is the last-added register -> leftmost space-joined token."""
    hits = sum(n for k, n in counts.items() if k.split()[0].lstrip("0x") in ("", "0"))
    return hits / shots


def local_expectation(theta: float, phi: float, deferred: bool, shots: int) -> dict:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = build_verification_circuit(theta, phi, deferred=deferred)
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=2026).result().get_counts()
    return {"counts": counts, "p_out0": _p_out_zero(counts, shots)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state", choices=sorted(NAMED_STATES), help="named payload state")
    ap.add_argument("--theta", type=float, default=1.17, help="RY angle if --state omitted")
    ap.add_argument("--phi", type=float, default=0.83, help="RZ angle if --state omitted")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", default=None, help="explicit backend; default = least busy >=3q")
    ap.add_argument("--deferred", action="store_true", help="coherent variant (no dynamic circuit)")
    ap.add_argument("--dry-run", action="store_true", help="offline: build + Aer expectation only")
    ap.add_argument(
        "--yes-spend-credits",
        action="store_true",
        help="skip the interactive confirmation and submit (spends IBM queue time)",
    )
    args = ap.parse_args()

    theta, phi = resolve_angles(args)
    qc = build_verification_circuit(theta, phi, deferred=args.deferred)
    expectation = local_expectation(theta, phi, args.deferred, args.shots)

    print(f"payload:    RY({theta:.4f}) RZ({phi:.4f}) |0>   (deferred={args.deferred})")
    print(f"logical circuit: depth {qc.depth()}, ops {dict(qc.count_ops())}")
    print(f"local Aer:  P(out=0) = {expectation['p_out0']:.4f}   (perfect teleportation = 1.0)")

    if args.dry_run:
        print("--dry-run: not contacting IBM Quantum, not submitting.")
        return 0

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = (
        service.backend(args.backend)
        if args.backend
        else service.least_busy(operational=True, simulator=False, min_num_qubits=3)
    )
    print(f"backend:    {backend.name}")

    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(qc)
    print(f"ISA circuit: depth {isa.depth()}, ops {dict(isa.count_ops())}")

    # ---- explicit spend confirmation -------------------------------------
    if not args.yes_spend_credits:
        if not sys.stdin.isatty():
            print(
                "\nrefusing to submit: this consumes IBM Quantum queue time.\n"
                "re-run with --yes-spend-credits (non-interactive) or from a terminal."
            )
            return 3
        print(f"\nabout to submit {args.shots} shots to '{backend.name}' (spends queue time).")
        typed = input(f"type the backend name '{backend.name}' to confirm, anything else to abort: ")
        if typed.strip() != backend.name:
            print("aborted, nothing submitted.")
            return 3

    job = SamplerV2(mode=backend).run([isa], shots=args.shots)
    job_id = job.job_id()
    meta = {
        "job_id": job_id,
        "submitted_utc": dt.datetime.now(dt.UTC).isoformat(),
        "backend": backend.name,
        "shots": args.shots,
        "theta": theta,
        "phi": phi,
        "deferred": args.deferred,
        "local_expectation": expectation,
    }
    out_path = HERE / f"{job_id}.json"
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"submitted job {job_id}")
    print(f"metadata -> {out_path}")
    print(f"compare when finished:  python jobs/compare_counts.py {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
