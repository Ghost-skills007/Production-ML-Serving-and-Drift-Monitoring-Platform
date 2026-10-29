"""End-to-end drift job test on SQLite + the local model store."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import func, select

import app.db as app_db
from app.model_store import LocalStore
from monitoring.run_drift_job import run
from tests.conftest import make_settings


def _seed_predictions(db, rows: pd.DataFrame):
    with db.session() as session:
        for _, record in rows.iterrows():
            session.add(
                app_db.Prediction(
                    model_version="test",
                    features={k: float(v) for k, v in record.items()},
                    prediction="benign",
                    probability=0.9,
                    latency_ms=1.0,
                )
            )
        session.commit()


def test_run_drift_job_detects_shift(tmp_path, model_root, db_url):
    root, version = model_root
    store = LocalStore(root)
    baseline = store.load(version).baseline
    numeric = list(baseline.columns)

    # Build a current window that shifts every numeric feature strongly.
    rng = np.random.default_rng(1)
    shifted = baseline.copy()
    for col in numeric:
        offset = 100 * (abs(shifted[col].mean()) + 1)
        shifted[col] = shifted[col] + offset + rng.normal(0, 1, len(shifted))

    app_db._db = app_db.init_database(db_url)
    try:
        _seed_predictions(app_db._db, shifted.head(200))
        settings = make_settings(root, db_url, version="latest")
        summary = run(window=500, store=store, settings=settings)

        assert summary["features_total"] == len(numeric)
        assert summary["overall_drift"] is True
        # Results and a run summary were persisted.
        with app_db._db.session() as session:
            n_results = session.execute(select(func.count(app_db.DriftResult.id))).scalar_one()
            n_runs = session.execute(select(func.count(app_db.DriftRun.id))).scalar_one()
        assert n_results >= len(numeric)
        assert n_runs == 1
    finally:
        app_db._db = None


def test_run_drift_job_no_predictions(tmp_path, model_root, db_url):
    root, version = model_root
    store = LocalStore(root)
    app_db._db = app_db.init_database(db_url)
    try:
        settings = make_settings(root, db_url, version="latest")
        summary = run(window=500, store=store, settings=settings)
        assert summary["features_total"] == 0
        assert summary["overall_drift"] is False
    finally:
        app_db._db = None
