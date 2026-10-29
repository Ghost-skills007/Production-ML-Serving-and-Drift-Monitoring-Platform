# Convenience commands. Uses the local virtualenv when present.
PY ?= python
ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
else
  VENV_BIN := .venv/bin
endif

.PHONY: install venv train train-fast serve test lint fmt up down rebuild drift demo clean

venv:
	$(PY) -m venv .venv

install: venv
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements-dev.txt

# Train on the full UCI Adult dataset (needs network on first run).
train:
	$(VENV_BIN)/python -m pipeline.train

# Fast offline training on load_breast_cancer, for demos and smoke tests.
train-fast:
	$(VENV_BIN)/python -m pipeline.train --dataset breast_cancer

serve:
	$(VENV_BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(VENV_BIN)/pytest -q --cov

lint:
	$(VENV_BIN)/ruff check .

fmt:
	$(VENV_BIN)/ruff check . --fix

up:
	docker compose up --build -d

down:
	docker compose down

rebuild:
	docker compose up --build

# Run the drift job via the compose one-off profile.
drift:
	docker compose --profile jobs run --rm drift-job

# End-to-end demo: traffic sim (normal + shifted) then the drift job.
demo:
	$(VENV_BIN)/python scripts/simulate_traffic.py --normal 500 --shifted 500
	$(VENV_BIN)/python -m monitoring.run_drift_job

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
