"""Model store tests (local + S3 via moto) and DB-logging failure tolerance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app.db as app_db
from app.model_store import LocalStore, S3Store


def _toy_model():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 0, 1])
    return Pipeline([("s", StandardScaler())]).fit(X, y), X


def _metadata():
    return {"feature_names": ["a"], "numeric_features": ["a"], "categorical_features": []}


def _baseline():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})


# ---- Local store ------------------------------------------------------
def test_local_save_load_roundtrip(tmp_path):
    store = LocalStore(tmp_path)
    model, _ = _toy_model()
    store.save("20260101T000000Z", model, _metadata(), _baseline())
    bundle = store.load("20260101T000000Z")
    assert bundle.version == "20260101T000000Z"
    assert bundle.metadata["feature_names"] == ["a"]
    assert list(bundle.baseline["a"]) == [1.0, 2.0, 3.0, 4.0]


def test_local_latest_resolution(tmp_path):
    store = LocalStore(tmp_path)
    model, _ = _toy_model()
    store.save("20260101T000000Z", model, _metadata(), _baseline())
    store.save("20260606T120000Z", model, _metadata(), _baseline())
    assert store.resolve("latest") == "20260606T120000Z"


def test_local_rollback_to_pinned_version(tmp_path):
    store = LocalStore(tmp_path)
    model, _ = _toy_model()
    store.save("20260101T000000Z", model, _metadata(), _baseline())
    store.save("20260606T120000Z", model, _metadata(), _baseline())
    # A pinned older version is loadable even when 'latest' is newer.
    bundle = store.load("20260101T000000Z")
    assert bundle.version == "20260101T000000Z"


def test_local_missing_version_raises(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("20990101T000000Z")


# ---- S3 store via moto ------------------------------------------------
@pytest.fixture
def s3_client():
    from moto import mock_aws

    with mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


def test_s3_save_load_roundtrip(s3_client):
    store = S3Store("test-bucket", "models", "us-east-1", client=s3_client)
    model, _ = _toy_model()
    store.save("20260202T000000Z", model, _metadata(), _baseline())
    bundle = store.load("20260202T000000Z")
    assert bundle.version == "20260202T000000Z"
    assert bundle.metadata["numeric_features"] == ["a"]
    assert len(bundle.baseline) == 4


def test_s3_latest_and_list_versions(s3_client):
    store = S3Store("test-bucket", "models", "us-east-1", client=s3_client)
    model, _ = _toy_model()
    store.save("20260202T000000Z", model, _metadata(), _baseline())
    store.save("20260909T000000Z", model, _metadata(), _baseline())
    assert store.list_versions() == ["20260202T000000Z", "20260909T000000Z"]
    assert store.resolve("latest") == "20260909T000000Z"


def test_s3_rollback_pinned(s3_client):
    store = S3Store("test-bucket", "models", "us-east-1", client=s3_client)
    model, _ = _toy_model()
    store.save("20260202T000000Z", model, _metadata(), _baseline())
    store.save("20260909T000000Z", model, _metadata(), _baseline())
    bundle = store.load("20260202T000000Z")
    assert bundle.version == "20260202T000000Z"


# ---- Database logging failure tolerance -------------------------------
def test_log_prediction_tolerates_db_failure():
    # Point the module singleton at an unreachable database.
    original = app_db._db
    app_db._db = app_db.Database("postgresql+psycopg2://nouser:nopass@127.0.0.1:1/nodb")
    try:
        ok = app_db.log_prediction("v1", {"a": 1.0}, "yes", 0.9, 1.2)
        assert ok is False  # failure is swallowed, serving continues
    finally:
        app_db._db = original


def test_log_prediction_without_db_returns_false():
    original = app_db._db
    app_db._db = None
    try:
        assert app_db.log_prediction("v1", {"a": 1.0}, "yes", 0.9, 1.2) is False
    finally:
        app_db._db = original
