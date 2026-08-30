"""PHASE 3 -- tests for the teleportation experiment lab.

Each named-state experiment is run once and cached (Aer simulations dominate the
runtime), then interrogated by the individual tests.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.teleport_lab import (
    NAMED,
    bloch_from_angles,
    run_experiment,
    run_suite,
    state_from_angles,
    suite_report,
)

SHOTS = 2000
SEED = 4242

_CACHE: dict[str, object] = {}


def exp(label: str):
    if label not in _CACHE:
        _CACHE[label] = run_experiment(label, *NAMED[label], SHOTS, SEED)
    return _CACHE[label]


@pytest.mark.parametrize("label", list(NAMED))
def test_ideal_fidelity_is_exactly_one(label):
    assert exp(label).ideal_fidelity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("label", list(NAMED))
def test_statevector_z_probs_match_amplitudes(label):
    psi = state_from_angles(*NAMED[label])
    assert exp(label).bob_z_prob_statevector == pytest.approx(
        (abs(psi[0]) ** 2, abs(psi[1]) ** 2), abs=1e-12
    )


@pytest.mark.parametrize("label", list(NAMED))
def test_empirical_fidelity_is_consistent_with_one(label):
    e = exp(label)
    lo, hi = e.empirical_fidelity_ci
    assert hi >= 0.999  # a noiseless CI must reach 1
    assert e.empirical_fidelity > 1.0 - 6 / np.sqrt(SHOTS)


@pytest.mark.parametrize("label", list(NAMED))
def test_measured_bloch_vector_tracks_truth(label):
    tol = 5 / np.sqrt(SHOTS)  # ~0.11 for 2000 shots
    assert np.allclose(exp(label).bloch_measured, bloch_from_angles(*NAMED[label]), atol=tol)


@pytest.mark.parametrize("label", list(NAMED))
def test_alice_outcomes_uniform_by_chi_square(label):
    stat, dof, crit = exp(label).alice_chi_square
    assert dof == 3
    assert stat < crit


def test_statevector_probability_and_shot_frequency_are_reported_separately():
    d = exp("|+>").as_dict()
    assert "bob_z_prob_statevector" in d and "alice_freq" in d
    any_entry = next(iter(d["alice_freq"].values()))
    assert set(any_entry) == {"count", "freq"}
    assert d["source"] == "SIMULATED"


def test_suite_report_is_json_serialisable_and_tagged():
    results = run_suite(shots=800, seed=1, n_random=1)
    report = suite_report(results, shots=800, seed=1)
    blob = json.dumps(report)
    assert '"source": "SIMULATED"' in blob
    assert len(report["results"]) == len(NAMED) + 1
    assert all(r["source"] == "SIMULATED" for r in report["results"])
