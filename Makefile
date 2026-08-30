# qm-teleportation -- offline by default. Only `make preflight` touches IBM.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install test test-fast foundations experiments audit lesson clean check-clean-env

help:
	@echo "install       create .venv and install locked dependencies"
	@echo "test          run the full test suite"
	@echo "test-fast     skip the slow experiment/integration tests"
	@echo "foundations   print the NumPy-vs-Qiskit cross-check table"
	@echo "experiments   regenerate experiments/results/*.json"
	@echo "audit         run the mutation / adversarial audit"
	@echo "lesson        start the guided lessons"
	@echo "clean         remove caches and generated results"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock.txt
	@echo "done. activate with:  . .venv/bin/activate"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -k "not teleport_lab and not integration_teleportation and not noise_lab"

foundations:
	$(PY) -m foundations

experiments:
	$(PY) -m experiments.teleport_lab --shots 8192 --out experiments/results/teleportation_sim.json
	$(PY) -m experiments.noise_lab --out experiments/results/noise_sim.json
	$(PY) jobs/pipeline.py --state plus --out experiments/results/pipeline_sim.json
	$(PY) tools/mutation_audit.py --json experiments/results/mutation_audit.json

audit:
	$(PY) tools/mutation_audit.py

lesson:
	$(PY) -m lessons.run --list

preflight:
	$(PY) jobs/preflight.py

clean:
	rm -rf .pytest_cache **/__pycache__ .ruff_cache
	rm -f experiments/results/*.json
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
