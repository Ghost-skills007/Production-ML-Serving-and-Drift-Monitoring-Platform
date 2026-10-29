# Features, Tech Stack & Design Rationale

This document explains **what** each feature of `ml-serving-drift-platform`
does, **how** it is implemented (with the exact files), and — most importantly —
**why** we built it that way. It complements the [README](../README.md)
(how to use it) and [results.md](results.md) (the measured numbers).

> Guiding principle: this is a **portfolio-grade system for an entry-level AI
> Engineer profile**. Every choice was made to (a) work end-to-end with one
> command, (b) demonstrate a real production pattern, and (c) be honest — no
> fabricated metrics, no claimed-but-missing features.

---

## Table of contents

1. [The big picture: why this project exists](#1-the-big-picture-why-this-project-exists)
2. [Feature-by-feature](#2-feature-by-feature)
   - [Training pipeline](#21-training-pipeline)
   - [Model store & versioning](#22-model-store--versioning)
   - [Serving API](#23-serving-api-fastapi)
   - [Input validation](#24-input-validation)
   - [PostgreSQL prediction logging](#25-postgresql-prediction-logging)
   - [Drift detection](#26-drift-detection)
   - [Drift job](#27-drift-job)
   - [Streamlit dashboard](#28-streamlit-dashboard)
   - [Traffic simulation](#29-traffic-simulation)
   - [Metrics harness](#29-metrics-harness)
   - [Testing](#211-testing)
   - [CI/CD](#212-cicd)
   - [Docker](#213-docker)
   - [AWS deployment](#214-aws-deployment)
3. [Tech stack & why each tool was chosen](#3-tech-stack--why-each-tool-was-chosen)
4. [Design decisions we deliberately made](#4-design-decisions-we-deliberately-made)
5. [What we did NOT build (and why)](#5-what-we-did-not-build-and-why)

---

## 1. The big picture: why this project exists

Most beginner ML projects stop at a notebook that prints an accuracy score. The
gap that hiring managers actually care about is **what happens after the model
ships**: is it served reliably, are predictions logged, and can you tell when
the real world stops matching your training data?

This project deliberately combines two usually-separate artifacts into **one
system**:

- a **production-style model-serving REST API** deployable on AWS, and
- a **model-monitoring / data-drift dashboard**.

By wiring them together through a shared PostgreSQL database and a shared
training baseline, the repo demonstrates the full loop: **train → serve →
log → detect drift → alert**. That loop is the single most convincing signal an
ML-engineering candidate can show.

---

## 2. Feature-by-feature

### 2.1 Training pipeline
**Files:** [`pipeline/preprocess.py`](../pipeline/preprocess.py),
[`pipeline/datasets.py`](../pipeline/datasets.py),
[`pipeline/evaluate.py`](../pipeline/evaluate.py),
[`pipeline/train.py`](../pipeline/train.py)

**What it does**
- Loads a dataset (UCI Adult by default, breast-cancer for offline runs).
- Wraps cleaning, encoding and scaling in a **single scikit-learn `Pipeline`**:
  median-imputation + `StandardScaler` for numerics, mode-imputation +
  `OneHotEncoder(handle_unknown="ignore")` for categoricals.
- Trains a `RandomForestClassifier` (or `GradientBoostingClassifier`), evaluates
  on a **stratified 80/20 held-out split**, and reports accuracy, macro-F1,
  ROC-AUC and the confusion matrix.
- Saves `model.joblib`, `metadata.json` and `baseline.csv` into
  `models/<UTC-timestamp>/`.

**Why we built it this way**
- **One pipeline for preprocess + model** eliminates train/serve skew — the exact
  same transformation code runs at inference, so you can never accidentally scale
  at train time and forget to scale at serve time. This is the #1 avoidable bug in
  deployed ML.
- **`handle_unknown="ignore"`** means a category the model never saw (e.g. a new
  `native-country`) degrades gracefully instead of crashing the API.
- **Stratified split** matters for Adult because the `>50K` class is a minority
  (~24%); a random split could misrepresent the positive class.
- **Macro-F1 and ROC-AUC** are reported alongside accuracy because accuracy alone
  is misleading on imbalanced data.
- **Saving the training baseline alongside the model** is what makes drift
  detection reproducible later — the reference distribution travels *with* the
  exact model version that produced it.

### 2.2 Model store & versioning
**File:** [`app/model_store.py`](../app/model_store.py)

**What it does**
- A pluggable storage abstraction with two backends selected by
  `MODEL_STORE=local|s3`: `LocalStore` (disk) and `S3Store` (boto3).
- Every artifact lives under a **UTC-timestamp version** (`20261029T105411Z`).
- `"latest"` resolves to the newest version; pinning `MODEL_VERSION` to an older
  timestamp **is** the rollback mechanism.

**Why we built it this way**
- **Timestamp versions sort chronologically as strings**, so "latest" needs no
  separate pointer file — the filesystem / S3 listing *is* the source of truth.
  Simpler and impossible to desync.
- **The same layout on disk and in S3** means a model trained locally can be
  dropped into S3 and consumed by EC2 with zero code change. Backend parity is a
  core design rule.
- **Rollback = config change**, not a redeploy. This mirrors how real model
  registries work and is demonstrated by `POST /reload` and the tests.
- **Documented deserialization risk**: `joblib`/pickle runs code on load, so the
  module docstring explicitly states artifacts are only loaded from trusted
  provenance (local `MODEL_DIR` or the private S3 bucket). We call this out
  because ignoring it is a real security footgun.

### 2.3 Serving API (FastAPI)
**Files:** [`app/main.py`](../app/main.py),
[`app/routes/predict.py`](../app/routes/predict.py),
[`app/routes/health.py`](../app/routes/health.py),
[`app/routes/admin.py`](../app/routes/admin.py)

**What it does**
- `POST /predict`, `POST /predict/batch` (up to 1,000),
  `GET /health`, `GET /ready`, `GET /model`, `POST /reload`.
- Loads the model in the **lifespan hook**; the app factory stores the model,
  store and settings on `app.state`.

**Why we built it this way**
- **Separate `/health` vs `/ready`**: liveness (`/health`, is the process alive?)
  and readiness (`/ready`, can it serve traffic yet? returns `503` with no model)
  answer two different questions for orchestrators like Kubernetes/ECS. Fusing
  them causes restart storms during slow model loads.
- **A broken artifact fails startup loudly** in the lifespan rather than serving
  garbage — fail fast, fail visible.
- **`/reload` without restart** lets you ship a new model version with zero
  downtime and roll back instantly.
- **`/model` exposes training metrics** so a reviewer (or an alerting tool) can
  see *which* model and *how good* it is straight from the running service.

### 2.4 Input validation
**Files:** [`app/schemas.py`](../app/schemas.py),
[`app/predictor.py`](../app/predictor.py)

**What it does**
- Pydantic v2 validates the request envelope; the `Predictor` validates
  feature-level rules against the loaded model's metadata — missing fields,
  non-numeric values in numeric features, and unexpected features.
- All failures return **structured `422`** responses (`{"loc","msg","type"}`),
  never a `500`.

**Why we built it this way**
- The feature set is model-dependent, so validation is driven by the **model's
  own metadata**, not a hard-coded schema — one code path serves both Adult (14
  mixed features) and breast-cancer (30 numeric).
- **A single bad request must never crash the server.** Catching
  `InputValidationError` and mapping it to `422` is the difference between a
  toy API and one you'd put behind a load balancer.
- **Structured errors** are self-documenting: the caller learns *which* field, in
  *which* record, and *why*.

### 2.5 PostgreSQL prediction logging
**Files:** [`app/db.py`](../app/db.py), [`scripts/init_db.sql`](../scripts/init_db.sql)

**What it does**
- Three tables: `predictions`, `drift_results`, `drift_runs`, each indexed on
  `ts`.
- Every prediction is written via a FastAPI **background task** so logging never
  sits on the response path.
- If the DB is unreachable, `log_prediction` swallows the error, logs a warning
  and the API keeps serving.

**Why we built it this way**
- **Logging must be best-effort.** Losing an audit row is acceptable; making a
  user wait for — or fail because of — the monitoring database is not. This
  "monitoring can never break serving" rule is a deliberate resilience choice,
  and it's covered by a dedicated failure-tolerance test.
- **Features stored as JSON** (not columns) because the schema is model-specific
  and evolves; a wide fixed table would need a migration per dataset.
- **`init_db.sql` mounted into the Postgres container** means a fresh
  `docker compose up` has the schema with zero manual steps.

### 2.6 Drift detection
**File:** [`monitoring/drift.py`](../monitoring/drift.py)

**What it does**
- **PSI** over 10 quantile bins from the baseline (with epsilon to avoid
  `log(0)`), plus a **two-sample KS test** (`scipy.stats.ks_2samp`) for numerics.
- Categorical features use PSI on category frequencies.
- A numeric feature is *drifted* when `PSI > 0.2` **and** `KS p < 0.05`; overall
  drift fires when **> 20%** of monitored features drift. All thresholds configurable.

**Why we built it this way**
- **PSI alone is noisy** on small windows and **KS alone is hypersensitive** on
  large ones; requiring both is a well-known guard against false alarms.
- **Quantile (not equal-width) bins** keep each baseline bin populated, which
  makes PSI stable regardless of skew.
- **The "0.2 / 20%" numbers are conventions, not magic** — so we made every
  threshold a setting rather than hard-coding a rule we can't tune.
- **Subtle bug we fixed and kept:** the outer bin edges are widened to `±inf`.
  Without it, a large shift pushes every current value *past* the top baseline
  quantile, `np.histogram` silently drops them, and PSI reads ~0 — hiding real
  drift. Our drift-job test caught this; it's now a documented behaviour.

### 2.7 Drift job
**File:** [`monitoring/run_drift_job.py`](../monitoring/run_drift_job.py)

**What it does**
- Reads the most recent N logged predictions from PostgreSQL, rebuilds the
  feature frame, loads the model's stored baseline, runs `detect_drift`, and
  writes per-feature `drift_results` + a `drift_runs` summary. Prints a table.

**Why we built it this way**
- **It closes the loop** between serving and monitoring using the same database
  both sides read/write — no extra infrastructure.
- **Window-based (`--window`)** because drift is about *recent* traffic versus
  *historical* baseline; a full-table scan would dilute the signal and be slow.
- **Runs as a compose "job" profile**, so it's on-demand in the demo but is
  obviously schedulable (cron/EventBridge) in production.

### 2.8 Streamlit dashboard
**File:** [`monitoring/dashboard.py`](../monitoring/dashboard.py)

**What it does**
- Tabs: Overview (volume, versions), Latency (p50/p95 over time), Drift (per-feature
  PSI bars with 0.1/0.2 thresholds, KS p-values, history), Recent predictions, and a
  top-level Alerts banner. Sidebar time-window filter. Reads straight from Postgres.

**Why we built it this way**
- **Streamlit over a custom React app**: the point is to *show monitoring
  thinking*, not to win a frontend award. Streamlit gets an interactive,
  shareable dashboard in one file.
- **Reads directly from PostgreSQL**, the same source the API writes — the
  dashboard is a genuine view of production behaviour, not a mock.
- **The alert banner is always visible** because a monitoring tool you have to
  navigate to is a monitoring tool nobody looks at.

### 2.9 Traffic simulation
**File:** [`scripts/simulate_traffic.py`](../scripts/simulate_traffic.py)

**What it does**
- Sends N "normal" records (resampled from the stored baseline) then N "shifted"
  records (selected numerics `× 1.4 + 0.5·std`) through `/predict/batch`.

**Why we built it this way**
- **Drift is unprovable without drifted traffic.** This script makes the alert
  *fire* on command so the end-to-end story is demonstrable, not theoretical.
- **Sampling from the real baseline** means "normal" is genuinely in-distribution,
  so any subsequent alert is attributable to the injected shift, not noise.

### 2.10 Metrics harness
**File:** [`scripts/measure_metrics.py`](../scripts/measure_metrics.py)

**What it does**
- Emits `docs/results.md`: held-out accuracy/F1/ROC-AUC, live p50/p95 latency,
  logged-request count, drift false-alarms on untouched features, and
  injected-drift detection counts.

**Why we built it this way**
- **Anti-fabrication.** A hard rule of this project is that README numbers must
  come from real output. This script is the single source of truth; anything that
  couldn't run in an environment is written as **"not yet measured"** rather than
  a made-up figure.

### 2.11 Testing
**Files:** [`tests/`](../tests/) — `test_api.py`, `test_pipeline.py`,
`test_drift.py`, `test_model_store.py`, `test_drift_job.py`, `conftest.py`

**What it does**
- 44 tests, 92% coverage: endpoint contract + validation, health/ready/reload,
  preprocessing/training/eval, **KS/PSI math on known distributions**, **S3 via
  `moto`**, the end-to-end drift job on SQLite, and **DB-failure tolerance**.

**Why we built it this way**
- **Offline-first fixtures** (breast-cancer + SQLite + moto) so CI needs no
  network, no AWS and no Postgres — tests are fast (<15s) and deterministic.
- **Drift math is tested against analytically-known cases** (identical data →
  PSI ≈ 0; strong shift → PSI > 0.2). Testing the *statistic*, not just the code
  path, is what makes the detection trustworthy.
- **moto exercises the real S3 code path** without an account, proving versioning
  / rollback logic works on the S3 backend too.

### 2.12 CI/CD
**Files:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml),
[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)

**What it does**
- On every push/PR: install → `ruff check` → `pytest --cov` → build both images.
  A separate manual `workflow_dispatch` job SSHes to EC2 and redeploys.

**Why we built it this way**
- **The Docker job depends on the test job** — you never build/publish an image
  that failed tests.
- **Deployment is manual, not auto-on-merge.** For a portfolio project, a
  human-gated deploy is the safer default and shows judgment about what "CI/CD"
  should and shouldn't automate.

### 2.13 Docker
**Files:** [`Dockerfile`](../Dockerfile),
[`Dockerfile.dashboard`](../Dockerfile.dashboard),
[`docker-compose.yml`](../docker-compose.yml),
[`scripts/entrypoint.sh`](../scripts/entrypoint.sh)

**What it does**
- **Multi-stage** API image (builder venv → slim runtime), **non-root user**,
  `HEALTHCHECK` hitting `/health`. Compose starts api + postgres + dashboard;
  drift-job is behind a profile.

**Why we built it this way**
- **Multi-stage** keeps compilers/build tools out of the final image — smaller,
  leaner attack surface.
- **Non-root user** is a baseline container-security expectation reviewers look
  for.
- **The entrypoint auto-trains a demo model** if `models/` is empty, so
  `docker compose up --build` genuinely has **zero manual steps** and `/health`
  always reports a version — satisfying the acceptance checklist even on a fresh
  clone.

### 2.14 AWS deployment
**Files:** [`infra/deploy_ec2.md`](../infra/deploy_ec2.md),
[`infra/s3_policy.json`](../infra/s3_policy.json)

**What it does**
- A step-by-step guide: private/versioned/encrypted S3 bucket, an EC2 **IAM role**
  with least-privilege S3 access, instance launch, env config, `docker compose up`,
  verification, rollback and teardown.

**Why we built it this way**
- **IAM role, not access keys** — boto3 picks up temporary credentials on the
  instance; **no AWS secrets ever land in code or `.env`.**
- **Versioned + encrypted + public-access-blocked bucket** demonstrates three
  S3 best-practices most tutorials skip.
- **Least-privilege policy scoped to `models/*`** (List/Get/Put only, no Delete)
  shows security thinking that interviewers specifically probe for.

---

## 3. Tech stack & why each tool was chosen

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Language | **Python 3.11** | The lingua franca of ML; 3.11 for speed + modern typing. |
| Data | **pandas, NumPy** | Ubiquitous tabular tooling; every reviewer expects fluency here. |
| ML | **scikit-learn** | A `Pipeline` cleanly bundles preprocessing + model, killing train/serve skew. RandomForest needs no tuning to be strong on Adult. |
| Stats | **SciPy (`ks_2samp`)** | Battle-tested two-sample KS test rather than a hand-rolled, easy-to-get-wrong implementation. |
| Serialization | **joblib** | Optimised for large NumPy/array-bearing sklearn objects (vs plain pickle). |
| API | **FastAPI + Uvicorn** | Async, auto OpenAPI docs, Pydantic-native validation — the modern default for Python ML serving. |
| Validation | **Pydantic v2** | Rust-core, fast, and the shared contract between routes and the predictor. |
| DB | **PostgreSQL 16 + SQLAlchemy 2.0 + psycopg2** | Relational store for logs/metrics; `JSONB` fits evolving feature schemas; SQLAlchemy ORM keeps the code DB-portable (tests swap in SQLite). |
| Cloud | **AWS EC2 + S3 + boto3** | The most common ML-infra expectation on job specs; S3 for artifacts, EC2 for compute. |
| Dashboard | **Streamlit + Plotly** | One-file interactive monitoring; Plotly for the PSI threshold lines. |
| Testing | **pytest + httpx + moto** | httpx drives the ASGI app in-process; moto mocks S3 so the real code path runs offline. |
| Quality | **Ruff** | One fast tool for lint + format (replaces flake8 + isort + black). |
| Containers | **Docker + Compose** | The "runs with one command" guarantee. |
| CI/CD | **GitHub Actions** | Free, native to the repo host, no extra infra. |
| Config | **pydantic-settings + `.env.example`** | 12-factor: config from environment, no secrets in code. |

**Versions are pinned** in [`requirements.txt`](../requirements.txt) so the
project is reproducible a year from now, not "works on my machine today".

---

## 4. Design decisions we deliberately made

- **Two halves, one system.** Serving and monitoring share the database and the
  baseline on purpose; the integration is the whole point of the project.
- **Resilience over completeness.** Serving must survive a dead DB and an unknown
  category. We chose graceful degradation everywhere.
- **Reproducibility & honesty.** No fabricated metrics — a script generates them,
  and un-run things say "not yet measured". This is a feature, not a limitation.
- **Offline-first tests and demo.** CI and the out-of-the-box `docker compose up`
  need no network/AWS/Postgres, so a reviewer can verify everything in minutes.
- **Config over code.** Backends (local/S3), thresholds, and DB toggle are all
  environment-driven, so the same image serves many environments.

---

## 5. What we did NOT build (and why)

Being explicit about scope boundaries is itself a signal of engineering maturity:

- **Scheduled / streaming drift monitoring** — the job is on-demand; production
  would add a scheduler (cron, EventBridge, Airflow). Kept out to stay focused.
- **Ground-truth labels & automated retraining** — requires a real feedback loop
  and human labelling that a demo can't honestly simulate.
- **Load / concurrency testing** — no `locust`/`k6`; latency numbers are honest
  single-client round-trips, labelled as dev-machine figures.
- **Concept drift & multivariate drift** — we detect **input** drift (PSI/KS per
  feature); model-performance and joint-distribution drift are a deeper topic.
- **Kubernetes / SageMaker / managed DB** — a single EC2 + Docker Compose target
  the brief ("deployable to EC2 with model in S3") without over-engineering.

See [README — Limitations and future work](../README.md#limitations-and-future-work).
