# Tech Stack & Why We Chose It

A focused deep-dive on every technology in `ml-serving-drift-platform`: **what**
it is, **where** it is used in this repo, and **why** we picked it over the
alternatives. Versions come straight from
[`requirements.txt`](../requirements.txt) and
[`requirements-dev.txt`](../requirements-dev.txt).

See also: [FEATURES.md](FEATURES.md) (per-feature rationale) and
[the README tech table](../README.md#tech-stack) (one-line summary).

---

## At a glance

| Layer | Technology | Version | Where it lives |
|---|---|---|---|
| Language / runtime | Python | 3.11 | whole repo |
| Numerics | NumPy | 1.26.4 | `pipeline`, `monitoring/drift.py` |
| Dataframes | pandas | 2.2.3 | `pipeline`, `monitoring`, `scripts` |
| ML models | scikit-learn | 1.5.2 | `pipeline/`, `app/predictor.py` |
| Statistics | SciPy | 1.13.1 | `monitoring/drift.py` (KS test) |
| Serialization | joblib | 1.4.2 | `app/model_store.py` |
| Web framework | FastAPI | 0.115.6 | `app/` |
| ASGI server | Uvicorn | 0.32.1 | `app/main.py`, entrypoint |
| Validation | Pydantic v2 | 2.10.4 | `app/schemas.py`, `app/config.py` |
| Settings | pydantic-settings | 2.7.0 | `app/config.py` |
| ORM | SQLAlchemy | 2.0.36 | `app/db.py` |
| DB driver | psycopg2-binary | 2.9.10 | Postgres connections |
| Database | PostgreSQL | 16 | `scripts/init_db.sql`, compose |
| Cloud SDK | boto3 | 1.35.76 | `app/model_store.py` (S3 backend) |
| HTTP client | httpx | 0.28.1 | `scripts/`, tests |
| Dashboard | Streamlit | 1.41.1 | `monitoring/dashboard.py` |
| Charts | Plotly | 5.24.1 | `monitoring/dashboard.py` |
| Testing | pytest / pytest-cov | 8.3.4 / 6.0.0 | `tests/` |
| S3 mocking | moto | 5.0.24 | `tests/test_model_store.py` |
| Lint / format | Ruff | 0.8.4 | CI, `pyproject.toml` |
| Containers | Docker / Compose | 24+/2.x | `Dockerfile*`, `docker-compose.yml` |
| CI/CD | GitHub Actions | — | `.github/workflows/` |

---

## Core ML stack

### Python 3.11
The lingua franca of machine learning. We target **3.11** specifically for the
notable CPython speed boost, better error messages, and mature `X | Y` typing
syntax used throughout (`from __future__ import annotations`).
*Why not 3.12/3.13?* 3.11 is the sweet spot of broad library wheel support and
stability on AWS images — the goal is reproducibility, not the newest feature.

### NumPy + pandas
The universal tabular/numerical layer. pandas handles the Adult/breast-cancer
loading and the drift job's windowing; NumPy backs the PSI/KS math.
*Why not Polars?* pandas is what every screening interviewer expects and every
scikit-learn API speaks natively; interop mattered more than raw speed here.

### scikit-learn
Provides the dataset loaders (`fetch_openml`, `load_breast_cancer`), the
`Pipeline` + `ColumnTransformer` that bundle preprocessing with the model, the
`RandomForest`/`GradientBoosting` classifiers, and the metrics.
*Why:* a single `Pipeline` that carries preprocessing **into** the saved
`model.joblib` is the cleanest defence against train/serve skew — the exact same
transform runs at inference. *Why RandomForest first:* strong on tabular data
with minimal tuning, and gives calibrated `predict_proba` for confidence output.

### SciPy (`scipy.stats.ks_2samp`)
The two-sample Kolmogorov-Smirnov test. *Why use a library:* a hand-rolled KS
test is easy to get subtly wrong; SciPy is battle-tested reference statistical
infrastructure.

### joblib
Model serialization. *Why not plain pickle:* joblib is optimized for the large
NumPy arrays inside sklearn estimators (better memory mapping and compression).
Its code-execution-on-load risk is documented and mitigated by loading only from
trusted provenance (see [`app/model_store.py`](../app/model_store.py)).

---

## API layer

### FastAPI
The serving framework. Async, self-documenting (auto OpenAPI/Swagger at `/docs`),
and Pydantic-native. *Why not Flask:* FastAPI gives request/response validation,
typed dependency injection (`BackgroundTasks`, `Request`), and a lifespan hook
almost for free — which we use to load the model and initialize the DB. *Why not
a dedicated ML server like BentoML/SageMaker:* keeping it to a plain, understood
framework makes the ML-engineering patterns (validation, health/readiness, hot
reload) explicit rather than hidden behind a framework's magic.

### Uvicorn
The ASGI server that actually runs FastAPI. Chosen because FastAPI is ASGI-only;
`uvicorn[standard]` adds the performance loop and websockets extras.

### Pydantic v2
The validation contract shared by the routes and the predictor. v2's Rust core is
significantly faster than v1 — relevant because every request is validated. It
gives us the structured `422` error shape that keeps bad input from becoming a
`500`.

### pydantic-settings
Loads all configuration from environment variables / `.env` with typing and
validation. *Why:* 12-factor config — no secrets or hard-coded values in code,
and the same image can target local/S3/Postgres purely through env vars.

---

## Data layer

### PostgreSQL 16
The system of record for predictions, drift results and drift runs. A relational
DB (not a flat log file or NoSQL store) because the dashboard runs time-windowed
queries and aggregates (p50/p95, volume over time) that SQL and indexes
(`ts`) handle cleanly. *Why Postgres specifically:* ubiquitous on AWS (RDS), and
`JSONB` lets us store the model-specific feature payload without a migration per
dataset.

### SQLAlchemy 2.0 + psycopg2
The ORM and the Postgres driver. *Why an ORM:* the models (`Prediction`,
`DriftResult`, `DriftRun`) stay DB-portable — production points at Postgres,
while the **test suite swaps in SQLite with zero code changes**. That portability
is exactly what makes the tests offline and fast.

---

## Cloud / storage layer

### AWS S3 + boto3
Versioned model artifacts live in a private, versioned, encrypted bucket and are
downloaded by the API at startup. *Why:* demonstrates the real-world pattern of
keeping large model files **out** of the container/image and in shared storage,
enabling decoupled deployments and rollback. boto3 is the canonical SDK; the S3
code path is verified offline with `moto`. See
[`infra/deploy_ec2.md`](../infra/deploy_ec2.md).

### AWS EC2
The compute target for the containerised stack. *Why EC2 + Docker Compose (not
Lambda/ECS/K8s):* it matches the brief ("deployable to EC2 with model in S3")
directly and keeps the deployment story simple enough to fully document, without
over-engineering into orchestrators a reviewer can't reproduce quickly.

### httpx
The async-capable HTTP client used by the traffic simulator and the API tests.
*Why not `requests`:* httpx works both against the live endpoint and in-process
with Starlette's `TestClient`, giving one consistent client for real and test
calls.

---

## Monitoring / dashboard layer

### Streamlit
The drift dashboard. *Why:* one Python file produces an interactive, shareable
monitoring UI — the goal is to show **monitoring thinking**, not front-end
chops. *Why not a React app:* disproportionate effort for zero extra signal on
an ML-engineering review.

### Plotly
Interactive charts inside Streamlit — notably the per-feature PSI bar chart with
the **0.1 / 0.2 threshold lines** and latency-over-time lines. *Why:* declarative
reference lines make the drift decision readable at a glance.

---

## Quality & testing layer

### pytest + pytest-cov
The test runner and coverage reporter. Fixtures (`conftest.py`) train one
breast-cancer model per session and reuse it across API tests — fast and
deterministic. *Why:* the de-facto Python standard; parametrisation and fixtures
keep 44 tests readable.

### moto
Mocks AWS (S3) so the real `S3Store` code runs in tests **without an AWS
account**. *Why:* backend parity — the S3 save/load/versioning logic is genuinely
exercised, not faked, which is why we can honestly claim it works offline.

### Ruff
A single fast Rust tool for **lint + format** (replacing the flake8/isort/black
trio). Config in [`pyproject.toml`](../pyproject.toml) selects the `E, F, I, UP,
B, C4, SIM` rule sets. *Why:* one tool that CI runs, consistent style, near-zero
runtime cost.

---

## Delivery layer

### Docker (multi-stage) + Docker Compose
Multi-stage image (build venv → slim runtime, non-root, `HEALTHCHECK`); Compose
brings up api + postgres + dashboard (drift-job behind a profile). *Why:* the
project's headline promise — `docker compose up --build` runs the whole system
with one command. Multi-stage keeps compilers out of the shipped image.

### GitHub Actions
`.github/workflows/ci.yml` runs ruff → pytest → image builds on every push/PR;
`deploy.yml` is a manual EC2 redeploy. *Why:* free, native to the repo host, and
the Docker job depends on the test job so a failed build never ships.

---

## Cross-cutting choices

- **Pinned versions everywhere** → reproducible a year from now, not "works
  today on my machine".
- **Everything is env-configurable** (`MODEL_STORE`, thresholds, DB toggle) →
  the same artifact serves dev, demo and AWS.
- **Offline-first** (breast-cancer fixture, SQLite, moto) → CI and a fresh
  `docker compose up` need no network, AWS, or manual steps.
- **Honest metrics only** — anything not actually run is labelled **"not yet
  measured"** rather than invented (see [results.md](results.md)).
