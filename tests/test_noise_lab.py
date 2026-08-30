"""PHASE 4 -- tests for the noise laboratory.

Cross-checks the NumPy Kraus channels against qiskit_aer.noise, verifies
channel-specific physics, and checks the teleportation degradation curve.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import channels as ch
from experiments.noise_lab import (
    aer_channel_rho,
    channel_sweep,
    numpy_channel_rho,
    readout_experiment,
    run_suite,
    state_metrics,
    teleportation_under_gate_noise,
)

THETA, PHI = 1.1780972450961724, 0.7853981633974483
PSI = np.array([np.cos(THETA / 2), np.exp(1j * PHI) * np.sin(THETA / 2)], dtype=complex)


# ---- Kraus completeness -------------------------------------------------


@pytest.mark.parametrize("name", list(ch.CHANNELS))
@pytest.mark.parametrize("param", [0.0, 0.1, 0.37, 1.0])
def test_channels_are_cptp(name, param):
    assert ch.is_cptp(ch.CHANNELS[name](param))


# ---- NumPy Kraus == qiskit_aer.noise --------------------------------


@pytest.mark.parametrize("name", list(ch.CHANNELS))
@pytest.mark.parametrize("param", [0.05, 0.3])
def test_numpy_channel_matches_qiskit_aer(name, param):
    rho_np = numpy_channel_rho(name, param, PSI)
    rho_q = aer_channel_rho(name, param, THETA, PHI)
    assert np.allclose(rho_np, rho_q, atol=1e-6), name


# ---- channel-specific physics -------------------------------------


def test_phase_flip_preserves_z_expectation():
    for p in (0.0, 0.2, 0.5, 0.9):
        rho = numpy_channel_rho("phase_flip", p, PSI)
        assert np.isclose(state_metrics(rho, PSI)["exp_z"], (PSI.conj() @ ch.Z @ PSI).real, atol=1e-12)


def test_phase_damping_kills_coherence_keeps_populations():
    rho0 = np.outer(PSI, PSI.conj())
    rho = numpy_channel_rho("phase_damping", 1.0, PSI)
    assert np.allclose(np.diag(rho), np.diag(rho0), atol=1e-12)  # populations unchanged
    assert abs(rho[0, 1]) < 1e-12  # coherence gone


def test_amplitude_damping_full_relaxes_to_ground():
    rho = numpy_channel_rho("amplitude_damping", 1.0, PSI)
    assert np.allclose(rho, np.array([[1, 0], [0, 0]]), atol=1e-12)


def test_depolarizing_full_gives_maximally_mixed():
    rho = numpy_channel_rho("depolarizing", 1.0, PSI)
    assert np.allclose(rho, np.eye(2) / 2, atol=1e-12)


@pytest.mark.parametrize("name", list(ch.CHANNELS))
def test_fidelity_and_purity_decrease_monotonically(name):
    sweep = channel_sweep(name, [0.0, 0.1, 0.25, 0.5], THETA, PHI)
    fids = [pt["fidelity"] for pt in sweep["points"]]
    purs = [pt["purity"] for pt in sweep["points"]]
    assert fids[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-9 for a, b in zip(fids, fids[1:]))
    assert all(b <= a + 1e-9 for a, b in zip(purs, purs[1:]))


# ---- readout error ----------------------------------------------


def test_readout_numpy_confusion_matches_aer():
    r = readout_experiment(0.04, 0.07, THETA, PHI)
    d_np = np.array(r["distorted_distribution_numpy"])
    d_shots = np.array(r["distorted_distribution_shots"])
    assert r["tv_distance_shots_vs_numpy"] < 0.01  # 40k shots
    assert np.isclose(d_np.sum(), 1.0)


def test_readout_confusion_matrix_is_column_stochastic():
    m = ch.readout_confusion(0.1, 0.2)
    assert np.allclose(m.sum(axis=0), 1.0)


# ---- teleportation under gate noise ---------------------------


def test_teleportation_degradation_is_monotone_and_starts_at_one():
    res = teleportation_under_gate_noise([0.0, 0.01, 0.03, 0.06], THETA, PHI)
    fids = [pt["bob_fidelity"] for pt in res["points"]]
    assert fids[0] == pytest.approx(1.0, abs=1e-9)
    assert all(b <= a + 1e-6 for a, b in zip(fids, fids[1:]))
    assert fids[-1] < 0.99  # noise actually did something


# ---- report ---------------------------------------------------


def test_suite_report_is_serialisable_seeded_and_tagged():
    r1 = run_suite(seed=1)
    r2 = run_suite(seed=1)
    assert r1["channel_sweeps"] == r2["channel_sweeps"]  # deterministic
    blob = json.dumps(r1)
    assert '"source": "SIMULATED"' in blob
    assert {s["channel"] for s in r1["channel_sweeps"]} == set(ch.CHANNELS)
