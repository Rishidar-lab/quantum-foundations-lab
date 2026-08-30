"""PHASE 5 -- tests for the offline stages (1-4) of the hardware pipeline.

The IBM hardware stage (5) is not exercised here; it needs credentials and is
gated behind an explicit spend confirmation.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "jobs_pipeline", pathlib.Path(__file__).resolve().parent.parent / "jobs" / "pipeline.py"
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


STATES = [("plus", *pipeline.NAMED["plus"]), ("minus_i", *pipeline.NAMED["minus_i"]), ("generic", 1.9, -2.1)]


@pytest.mark.parametrize("name,theta,phi", STATES)
def test_stage1_linear_algebra_matches_stage2_statevector(name, theta, phi):
    s1 = pipeline.stage1_linear_algebra(theta, phi)
    s2 = pipeline.stage2_ideal_statevector(theta, phi)
    assert s1["p_out0"] == pytest.approx(1.0, abs=1e-9)
    assert s1["p_out0"] == pytest.approx(s2["p_out0"], abs=1e-9)


@pytest.mark.parametrize("name,theta,phi", STATES)
def test_stage3_shot_ci_contains_ideal(name, theta, phi):
    s2 = pipeline.stage2_ideal_statevector(theta, phi)
    s3 = pipeline.stage3_shot_sim(theta, phi, shots=8000, seed=1)
    lo, hi = s3["ci95"]
    assert lo - 1e-9 <= s2["p_out0"] <= hi + 1e-9


def test_stage4_noise_degrades_fidelity():
    theta, phi = pipeline.NAMED["plus"]
    clean = pipeline.stage3_shot_sim(theta, phi, shots=8000, seed=2)
    noisy = pipeline.stage4_noisy_sim(theta, phi, shots=8000, seed=2, gate_p=0.03)
    assert noisy["p_out0"] < clean["p_out0"] - 0.01


def test_pipeline_record_shape_and_labels():
    rep = pipeline.run_pipeline(*pipeline.NAMED["plus"], shots=2000, seed=3, gate_p=0.01)
    assert rep["stage5_hardware"] is None
    assert [s["stage"] for s in rep["stages"]] == [1, 2, 3, 4]
    assert all(s["kind"] == "SIMULATED" for s in rep["stages"])
    assert rep["consistency"]["s1_vs_s2_abs_diff"] < 1e-9
    assert rep["consistency"]["s3_ci_contains_s2"] is True


def test_no_ibm_import_needed_for_offline_stages():
    import sys

    # the offline pipeline must not require qiskit_ibm_runtime
    had = sys.modules.pop("qiskit_ibm_runtime", None)
    try:
        pipeline.run_pipeline(*pipeline.NAMED["zero"], shots=500, seed=0, gate_p=0.0)
    finally:
        if had is not None:
            sys.modules["qiskit_ibm_runtime"] = had
