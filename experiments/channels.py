"""Single-qubit quantum channels as explicit Kraus operator sets (NumPy).

A channel acts as  rho -> sum_k K_k rho K_k^dagger  with  sum_k K_k^dagger K_k = I.
`tests/test_noise_lab.py` cross-checks every one of these against the equivalent
`qiskit_aer.noise` error.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

Kraus = list[np.ndarray]


def bit_flip(p: float) -> Kraus:
    """With probability p apply X."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def phase_flip(p: float) -> Kraus:
    """With probability p apply Z."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]


def bit_phase_flip(p: float) -> Kraus:
    """With probability p apply Y."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Y]


def depolarizing(p: float) -> Kraus:
    """rho -> (1 - p) rho + p I/2.  Kraus form uses p/4 weights on X,Y,Z."""
    return [
        np.sqrt(1 - 3 * p / 4) * I2,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]


def amplitude_damping(gamma: float) -> Kraus:
    """Energy loss |1> -> |0> with probability gamma (T1 relaxation)."""
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def phase_damping(gamma: float) -> Kraus:
    """Loss of coherence without energy loss (T2 dephasing)."""
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex),
    ]


CHANNELS = {
    "bit_flip": bit_flip,
    "phase_flip": phase_flip,
    "bit_phase_flip": bit_phase_flip,
    "depolarizing": depolarizing,
    "amplitude_damping": amplitude_damping,
    "phase_damping": phase_damping,
}


def apply_channel(rho: np.ndarray, kraus: Kraus) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return sum(k @ rho @ k.conj().T for k in kraus)


def is_cptp(kraus: Kraus, atol: float = 1e-12) -> bool:
    """Completeness (trace preservation): sum_k K_k^dagger K_k == I."""
    total = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(total, I2, atol=atol))


def readout_confusion(p01: float, p10: float) -> np.ndarray:
    """Column-stochastic readout-error matrix M with M[j, i] = P(report j | prepared i)."""
    return np.array([[1 - p01, p10], [p01, 1 - p10]], dtype=float)


def apply_readout(probs: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    return confusion @ np.asarray(probs, dtype=float)
