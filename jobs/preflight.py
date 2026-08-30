#!/usr/bin/env python
"""PHASE 5 -- IBM Quantum configuration preflight.

Validates that a hardware submission *could* succeed, WITHOUT submitting anything
and WITHOUT printing any secret. Exit 0 = ready, 1 = not ready.

    python jobs/preflight.py
    python jobs/preflight.py --backend ibm_torino

Checks, in order:
    1. qiskit-ibm-runtime importable
    2. credentials resolvable (saved account or QISKIT_IBM_TOKEN) -- presence only
    3. service reachable, at least one operational >=3-qubit backend
    4. the verification circuit transpiles onto the chosen/least-busy backend
Nothing here consumes queue time or credits.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

CHECK, CROSS = "  [ok]  ", "  [--]  "


def _has_saved_account() -> bool:
    for name in ("qiskit-ibm.json", "qiskit_ibm.json"):
        if (pathlib.Path.home() / ".qiskit" / name).exists():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    ok = True

    # 1 --------------------------------------------------------------
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401

        print(CHECK + "qiskit-ibm-runtime is importable")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(CROSS + f"qiskit-ibm-runtime import failed: {exc}")
        return 1

    # 2 --------------------------------------------------------------
    env_token = bool(os.environ.get("QISKIT_IBM_TOKEN"))
    saved = _has_saved_account()
    if env_token or saved:
        src = "QISKIT_IBM_TOKEN env var" if env_token else "saved account (~/.qiskit)"
        print(CHECK + f"credentials present via {src} (value not read/printed)")
    else:
        print(CROSS + "no credentials found (set QISKIT_IBM_TOKEN or QiskitRuntimeService.save_account)")
        print("        -- stages 1-4 of jobs/pipeline.py still run fully offline")
        return 1

    # 3 --------------------------------------------------------------
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backends = service.backends(operational=True, simulator=False, min_num_qubits=3)
        if not backends:
            print(CROSS + "no operational >=3-qubit backend visible to this account")
            return 1
        print(CHECK + f"service reachable; {len(backends)} eligible backend(s): "
              + ", ".join(b.name for b in backends[:5]) + (" ..." if len(backends) > 5 else ""))
        target = service.backend(args.backend) if args.backend else service.least_busy(
            operational=True, simulator=False, min_num_qubits=3
        )
        print(CHECK + f"target backend: {target.name} ({target.num_qubits} qubits)")
    except Exception as exc:  # pragma: no cover - network dependent
        print(CROSS + f"service/backends query failed: {exc}")
        return 1

    # 4 --------------------------------------------------------------
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        from teleportation import build_verification_circuit

        qc = build_verification_circuit(1.17, 0.83)
        isa = generate_preset_pass_manager(optimization_level=1, backend=target).run(qc)
        print(CHECK + f"verification circuit transpiles: depth {isa.depth()}, "
              f"ops {dict(isa.count_ops())}")
        if not getattr(target, "num_qubits", 0):
            ok = False
    except Exception as exc:  # pragma: no cover
        print(CROSS + f"transpilation onto {target.name} failed: {exc}")
        return 1

    print()
    if ok:
        print("PREFLIGHT: READY. Submit with:")
        print(f"    python jobs/submit_ibm_job.py --backend {target.name} --state plus --shots 4096")
        print("    (submission still requires an explicit spend confirmation)")
        return 0
    print("PREFLIGHT: not ready.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
