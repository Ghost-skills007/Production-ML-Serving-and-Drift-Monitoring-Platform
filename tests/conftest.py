"""Shared pytest fixtures.

A single breast-cancer model is trained once (session scope) into a temporary
local model store; every API test builds a ``TestClient`` against it using an
isolated SQLite database so the suite runs offline and fast.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_store import LocalStore
from pipeline.train import train_and_save

BREAST_CANCER_ROWS = 400


@pytest.fixture(scope="session")
def model_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    store = LocalStore(root)
    meta = train_and_save(
        dataset="breast_cancer", model="random_forest", n_rows=BREAST_CANCER_ROWS, store=store
    )
    return root, meta["model_version"]


def make_settings(
    model_dir,
    db_url: str | None = None,
    version: str = "latest",
    **overrides,
) -> Settings:
    kwargs = {
        "model_store": "local",
        "model_dir": str(model_dir),
        "model_version": version,
        "db_enabled": db_url is not None,
        "database_url": db_url,
        "batch_max_records": 10,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def settings(model_root, db_url):
    root, _ = model_root
    return make_settings(root, db_url)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def numeric_features(client) -> list[str]:
    return client.get("/model").json()["numeric_features"]


@pytest.fixture
def sample_record(client, numeric_features) -> dict:
    """A valid single record: every numeric feature set to a plausible value."""
    return dict.fromkeys(numeric_features, 12.0)
