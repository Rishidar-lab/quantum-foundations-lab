#!/usr/bin/env python
"""Adversarial / mutation audit (PHASE 1).

Deliberately introduces representative *incorrect* implementations of core
quantum primitives, runs the relevant part of the test suite, and records
whether the suite catches the mutation. Every mutation is reverted afterwards
via ``git checkout`` -- the tool refuses to run on a dirty working tree so it
can always restore cleanly.

    python tools/mutation_audit.py            # human-readable table
    python tools/mutation_audit.py --json out.json

A mutation is "CAUGHT" if at least one targeted test fails while it is applied.
A mutation that is *expected* to be invisible (pure global phase) is reported
as "INVISIBLE (correct)" -- physics says it must not be detectable.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Mutation:
    id: str
    concept: str
    file: str
    old: str
    new: str
    targets: list[str]
    expectation: str = "caught"  # "caught" | "invisible"
    note: str = ""


MUTATIONS: list[Mutation] = [
    Mutation(
        id="normalization--drop-sqrt",
        concept="normalization",
        file="katas/01_hilbert_space.py",
        old="return float(np.sqrt(np.vdot(a, a).real))",
        new="return float(np.vdot(a, a).real)  # MUTANT: missing sqrt",
        targets=["tests/test_01_hilbert_space.py"],
        note="norm() returns <a|a> instead of sqrt(<a|a>)",
    ),
    Mutation(
        id="normalization--collapse-unnormalized",
        concept="normalization",
        file="katas/16_projective_measurement.py",
        old="    return out / norm",
        new="    return out  # MUTANT: post-measurement state not renormalized",
        targets=["tests/test_16_projective_measurement.py"],
        note="projective collapse skips renormalization",
    ),
    Mutation(
        id="tensor-order--little-endian-bit",
        concept="tensor-product ordering",
        file="katas/14_cnot_controlled_u.py",
        old="    return (index >> (n_qubits - 1 - qubit)) & 1",
        new="    return (index >> qubit) & 1  # MUTANT: little-endian bit index",
        targets=[
            "tests/test_14_cnot_controlled_u.py",
            "tests/test_17_partial_measurement.py",
            "tests/test_24_teleportation_identity.py",
            "tests/test_integration_teleportation.py",
        ],
        note="controlled-U embedding reads qubit bits in the wrong endianness",
    ),
    Mutation(
        id="hadamard--drop-minus-sign",
        concept="Hadamard",
        file="katas/13_hadamard.py",
        old="return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)",
        new="return np.array([[1, 1], [1, 1]], dtype=complex) / np.sqrt(2)  # MUTANT",
        targets=[
            "tests/test_13_hadamard.py",
            "tests/test_21_bell_states.py",
            "tests/test_22_bell_pair_prep.py",
            "tests/test_integration_teleportation.py",
        ],
        note="H[1,1] sign flipped -> matrix no longer unitary",
    ),
    Mutation(
        id="cnot--condition-on-target",
        concept="CNOT",
        file="katas/14_cnot_controlled_u.py",
        old="        if _bit(col, control, n_qubits) == 0:",
        new="        if _bit(col, target, n_qubits) == 0:  # MUTANT: swap control/target",
        targets=[
            "tests/test_14_cnot_controlled_u.py",
            "tests/test_integration_teleportation.py",
        ],
        note="controlled-U conditions on the target bit instead of the control",
    ),
    Mutation(
        id="born-rule--probability-not-squared",
        concept="measurement probability",
        file="katas/15_measurement_postulate.py",
        old="        probs.append(np.vdot(psi, m.conj().T @ m @ psi).real)",
        new="        probs.append(float(np.sqrt(abs(np.vdot(psi, m.conj().T @ m @ psi).real))))  # MUTANT",
        targets=["tests/test_15_measurement_postulate.py"],
        note="returns sqrt(p) instead of p = <psi|M#M|psi>",
    ),
    Mutation(
        id="global-phase--negate-branch-state",
        concept="global phase handling",
        file="katas/24_teleportation_identity.py",
        old="            branches[(m0, m1)] = (prob, block / np.linalg.norm(block))",
        new="            branches[(m0, m1)] = (prob, -block / np.linalg.norm(block))  # MUTANT: global phase",
        targets=["tests/test_24_teleportation_identity.py"],
        expectation="invisible",
        note="multiplies every branch state by -1; a pure global phase must stay invisible",
    ),
    Mutation(
        id="relative-phase--S-instead-of-Z",
        concept="global vs relative phase",
        file="teleportation/circuit.py",
        old="    with qc.if_test((m0[0], 1)):\n        qc.z(q[2])",
        new="    with qc.if_test((m0[0], 1)):\n        qc.s(q[2])  # MUTANT: S=diag(1,i) not Z=diag(1,-1)",
        targets=["tests/test_integration_teleportation.py"],
        note="Z correction replaced by S -> genuine relative-phase error, must be caught",
    ),
    Mutation(
        id="teleport-correction--swap-x-z-conditions",
        concept="teleportation correction",
        file="teleportation/circuit.py",
        old=(
            "    with qc.if_test((m1[0], 1)):\n"
            "        qc.x(q[2])\n"
            "    with qc.if_test((m0[0], 1)):\n"
            "        qc.z(q[2])"
        ),
        new=(
            "    with qc.if_test((m0[0], 1)):  # MUTANT: X now keyed to m0\n"
            "        qc.x(q[2])\n"
            "    with qc.if_test((m1[0], 1)):  # MUTANT: Z now keyed to m1\n"
            "        qc.z(q[2])"
        ),
        targets=["tests/test_integration_teleportation.py"],
        note="Bob keys X to m0 and Z to m1 (swapped) -> wrong branch corrected",
    ),
    Mutation(
        id="partial-trace--transposed-output",
        concept="partial trace / entanglement",
        file="katas/19_partial_trace.py",
        old='        + "".join(row[q] for q in keep)\n        + "".join(col[q] for q in keep)',
        new='        + "".join(col[q] for q in keep)  # MUTANT: output indices swapped\n        + "".join(row[q] for q in keep)',
        targets=[
            "tests/test_19_partial_trace.py",
            "tests/test_20_entanglement_entropy.py",
        ],
        note="reduced density matrix comes out transposed (rho_A^T); invisible for real-symmetric reductions like I/2",
    ),
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _pytest(targets: list[str]) -> tuple[int, int, int, str]:
    """Return (passed, failed+errored, returncode, summary_line)."""
    proc = _run(
        [sys.executable, "-m", "pytest", "--no-header", "--tb=no", "-o", "addopts=", *targets]
    )
    text = proc.stdout + proc.stderr

    def _count(word: str) -> int:
        m = re.search(rf"(\d+) {word}", text)
        return int(m.group(1)) if m else 0

    passed = _count("passed")
    bad = _count("failed") + _count("error")
    tail = next(
        (
            ln.strip()
            for ln in reversed(text.splitlines())
            if any(w in ln for w in ("passed", "failed", "error", "no tests ran"))
        ),
        (text.strip().splitlines() or [""])[-1],
    )
    return passed, bad, proc.returncode, tail


def _restore() -> None:
    _run(["git", "checkout", "--", "katas", "teleportation"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    dirty = _run(["git", "status", "--porcelain", "katas", "teleportation"]).stdout.strip()
    if dirty:
        print("refusing to run: katas/ or teleportation/ has uncommitted changes:\n" + dirty)
        return 2

    base_pass, base_fail, base_rc, base_tail = _pytest(["tests"])
    print(f"baseline: {base_tail}\n")
    if base_rc != 0:
        print("refusing to run: baseline suite is not green.")
        return 2

    results = []
    header = f"{'concept':<26} {'mutation id':<38} {'result':<20} targeted"
    print(header)
    print("-" * len(header))
    try:
        for mut in MUTATIONS:
            path = ROOT / mut.file
            src = path.read_text()
            if src.count(mut.old) != 1:
                raise SystemExit(f"anchor not unique in {mut.file}: {mut.id!r}")
            path.write_text(src.replace(mut.old, mut.new))
            try:
                passed, failed, rc, tail = _pytest(mut.targets)
            finally:
                _restore()

            caught = rc != 0 or failed > 0
            if mut.expectation == "invisible":
                verdict = "INVISIBLE (correct)" if not caught else "CAUGHT (over-strict?)"
                ok = not caught
            else:
                verdict = "CAUGHT" if caught else "*** SURVIVED ***"
                ok = caught

            results.append(
                {
                    "id": mut.id,
                    "concept": mut.concept,
                    "file": mut.file,
                    "expectation": mut.expectation,
                    "failed_tests": failed,
                    "passed_tests": passed,
                    "verdict": verdict,
                    "ok": ok,
                    "pytest_tail": tail,
                    "note": mut.note,
                }
            )
            print(f"{mut.concept:<26} {mut.id:<38} {verdict:<20} {failed} failing / {passed} passing")
    finally:
        _restore()

    post_pass, post_fail, post_rc, post_tail = _pytest(["tests"])
    print(f"\npost-restore: {post_tail}")
    clean = _run(["git", "status", "--porcelain", "katas", "teleportation"]).stdout.strip()
    print(f"working tree restored: {'yes' if not clean else 'NO -- ' + clean}")

    all_ok = all(r["ok"] for r in results) and post_rc == 0 and not clean
    print(f"\naudit verdict: {'PASS -- every mutation handled as expected' if all_ok else 'ATTENTION NEEDED'}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "baseline": base_tail,
                    "post_restore": post_tail,
                    "restored_clean": not clean,
                    "results": results,
                    "all_ok": all_ok,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
