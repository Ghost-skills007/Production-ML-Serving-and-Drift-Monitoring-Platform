"""API contract tests: predict, batch, health, readiness, model, reload, logging."""

from __future__ import annotations

from sqlalchemy import func, select

from app.db import Prediction, get_database


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "ml-serving-drift-platform"


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert body["dataset"] == "breast_cancer"


def test_ready_ok(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_model_metadata(client):
    body = client.get("/model").json()
    assert len(body["feature_names"]) == 30
    assert body["metrics"]["accuracy"] > 0.8
    assert body["training_row_count"] > 0
    assert set(body["library_versions"]) >= {"python", "scikit-learn"}


def test_predict_valid(client, sample_record):
    resp = client.post("/predict", json={"features": sample_record})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"malignant", "benign"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["latency_ms"] >= 0
    assert body["model_version"]


def test_predict_rejects_non_numeric(client, sample_record, numeric_features):
    sample_record[numeric_features[0]] = "not-a-number"
    resp = client.post("/predict", json={"features": sample_record})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any(e["type"] == "numeric_type" for e in detail)


def test_predict_rejects_missing_field(client, sample_record, numeric_features):
    del sample_record[numeric_features[1]]
    resp = client.post("/predict", json={"features": sample_record})
    assert resp.status_code == 422
    assert any(e["type"] == "missing_field" for e in resp.json()["detail"])


def test_predict_rejects_empty_features(client):
    resp = client.post("/predict", json={"features": {}})
    assert resp.status_code == 422


def test_predict_rejects_extra_field(client, sample_record):
    sample_record["totally_unknown"] = 1.0
    resp = client.post("/predict", json={"features": sample_record})
    assert resp.status_code == 422


def test_predict_batch_valid(client, sample_record):
    records = [sample_record, sample_record, sample_record]
    resp = client.post("/predict/batch", json={"records": records})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["results"]) == 3


def test_predict_batch_empty(client):
    resp = client.post("/predict/batch", json={"records": []})
    assert resp.status_code == 422


def test_predict_batch_too_large(client, sample_record):
    records = [sample_record] * 11  # batch_max_records is 10 in tests
    resp = client.post("/predict/batch", json={"records": records})
    assert resp.status_code == 413


def test_predict_batch_invalid_record(client, sample_record, numeric_features):
    bad = dict(sample_record)
    bad[numeric_features[2]] = {"nested": "invalid"}
    resp = client.post("/predict/batch", json={"records": [sample_record, bad]})
    assert resp.status_code == 422


def test_predictions_are_logged(client, sample_record):
    client.post("/predict", json={"features": sample_record})
    client.post("/predict/batch", json={"records": [sample_record, sample_record]})
    db = get_database()
    with db.session() as session:
        count = session.execute(select(func.count(Prediction.id))).scalar_one()
    assert count == 3


def test_reload(client):
    resp = client.post("/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "reloaded"
    assert body["new_version"] == body["previous_version"]


def test_health_reflects_loaded_version(client):
    version = client.get("/health").json()["model_version"]
    assert version == client.get("/model").json()["model_version"]
