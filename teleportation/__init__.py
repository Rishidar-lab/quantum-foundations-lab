"""Qiskit assembly of the quantum teleportation protocol.

The katas verify the QM primitives from first principles (see research/
dependency_map.md); this package assembles them into a runnable Qiskit circuit
and provides the local-Aer and IBM-hardware entry points.
"""
from teleportation.circuit import (
    ALICE_BITS,
    build_teleportation_circuit,
    build_verification_circuit,
)

__all__ = [
    "ALICE_BITS",
    "build_teleportation_circuit",
    "build_verification_circuit",
]
