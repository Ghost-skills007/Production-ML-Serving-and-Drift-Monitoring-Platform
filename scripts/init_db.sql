-- PostgreSQL schema for the ML serving & drift monitoring platform.
-- Mounted into the postgres container at /docker-entrypoint-initdb.d/init_db.sql
-- so it runs automatically on first start. Keep in sync with app/db.py.

CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT        NOT NULL,
    features      JSONB       NOT NULL,
    prediction    TEXT        NOT NULL,
    probability   DOUBLE PRECISION NOT NULL,
    latency_ms    DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_results (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT        NOT NULL,
    feature       TEXT        NOT NULL,
    ks_stat       DOUBLE PRECISION,
    ks_pvalue     DOUBLE PRECISION,
    psi           DOUBLE PRECISION,
    drifted       BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS drift_runs (
    id               BIGSERIAL PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version    TEXT        NOT NULL,
    window_rows      INTEGER     NOT NULL,
    features_drifted INTEGER     NOT NULL,
    features_total   INTEGER     NOT NULL,
    overall_drift    BOOLEAN     NOT NULL DEFAULT FALSE
);

-- Time-series queries (dashboard window filter, drift job window) rely on ts.
CREATE INDEX IF NOT EXISTS ix_predictions_ts    ON predictions (ts);
CREATE INDEX IF NOT EXISTS ix_drift_results_ts  ON drift_results (ts);
CREATE INDEX IF NOT EXISTS ix_drift_runs_ts     ON drift_runs (ts);
