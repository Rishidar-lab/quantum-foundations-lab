"""Kata 12 — Pauli algebra / group relations  (dependency_map.md node F12).

CORE node — a direct continuation of kata 11's four matrices, no new operators.

Every product of two Paulis is a phase times a Pauli:
    P(a) @ P(b) == phase * P(c),   phase in {1, -1, 1j, -1j},   c in {I, X, Y, Z}.
Rule: P@P = I; I is the unit; for distinct non-identity Paulis the product is
cyclic (X->Y->Z->X) with phase +i and anti-cyclic with phase -i. The 16 elements
{+-1, +-i} x {I, X, Y, Z} close into the Pauli group under this multiplication.

Implement:
  - pauli_mul(a, b) : labels a, b in {"I","X","Y","Z"} -> (phase, label)
                      such that  P(a) @ P(b) == phase * P(label).

Properties verified by tests/test_12_pauli_algebra.py:
  - matches the F11 matrix product for all 16 label pairs; phase in {1,-1,i,-i}
  - P@P == I ; I is a two-sided identity
  - cyclic  XY=iZ, YZ=iX, ZX=iY ;  anti-cyclic  YX=-iZ, ZY=-iX, XZ=-iY
  - commutation from the rule: equal / I-involving pairs commute, distinct
    non-I Paulis anticommute (same label, opposite phase)
  - the 16-element group {+-1,+-i} x {I,X,Y,Z} is closed under pauli_mul, and
    the induced multiplication is associative
"""
from __future__ import annotations


def pauli_mul(a: str, b: str) -> tuple[complex, str]:
    """Return (phase, label) with P(a) @ P(b) == phase * P(label).

    Parameters
    ----------
    a, b : one of "I", "X", "Y", "Z".

    Returns
    -------
    (phase, label)
        phase in {1, -1, 1j, -1j} as a complex number; label one of "I","X","Y","Z".
    """
    valid = ("I", "X", "Y", "Z")
    if a not in valid or b not in valid:
        raise ValueError(f"labels must be in {valid}, got {a!r}, {b!r}")

    # I is the identity element.
    if a == "I":
        return (1 + 0j, b)
    if b == "I":
        return (1 + 0j, a)
    # P @ P = I.
    if a == b:
        return (1 + 0j, "I")
    # Distinct non-identity Paulis: cyclic X -> Y -> Z -> X gives +i, reverse gives -i.
    cyclic = {("X", "Y"): "Z", ("Y", "Z"): "X", ("Z", "X"): "Y"}
    if (a, b) in cyclic:
        return (1j, cyclic[(a, b)])
    return (-1j, cyclic[(b, a)])
