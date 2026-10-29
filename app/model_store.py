"""Versioned model artifact storage with two backends: local disk and S3.

Layout for a version ``V`` (a UTC timestamp such as ``20261101T093000Z``)::

    <root>/V/model.joblib     # the fitted scikit-learn Pipeline
    <root>/V/metadata.json    # feature order, class labels, metrics, provenance
    <root>/V/baseline.csv     # training reference sample used for drift detection

A version of ``"latest"`` resolves to the lexicographically greatest timestamp
(timestamps sort chronologically). Pinning an explicit version - or changing
``MODEL_VERSION`` - is how rollback is performed.

Security note: ``joblib``/pickle deserialization executes Python objects, so
artifacts are loaded only from trusted provenance - the local ``MODEL_DIR`` or
the private, versioned, encrypted S3 bucket configured via ``S3_BUCKET``. Never
point these settings at an untrusted location.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Protocol

import joblib
import pandas as pd

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
BASELINE_FILE = "baseline.csv"


class ModelBundle:
    """In-memory representation of a loaded model version."""

    def __init__(self, model, metadata: dict, baseline: pd.DataFrame, version: str):
        self.model = model
        self.metadata = metadata
        self.baseline = baseline
        self.version = version


class BaseStore(Protocol):
    def save(self, version, model, metadata, baseline) -> str: ...
    def load(self, version: str) -> ModelBundle: ...
    def list_versions(self) -> list[str]: ...
    def resolve(self, version: str) -> str: ...


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, version: str) -> Path:
        return self.root / version

    def list_versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if (p / MODEL_FILE).exists()
        )

    def resolve(self, version: str) -> str:
        if version != "latest":
            return version
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError(f"No model versions under {self.root}")
        return versions[-1]

    def save(self, version: str, model, metadata: dict, baseline: pd.DataFrame) -> str:
        target = self._dir(version)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target / MODEL_FILE)
        (target / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        baseline.to_csv(target / BASELINE_FILE, index=False)
        return version

    def load(self, version: str) -> ModelBundle:
        resolved = self.resolve(version)
        source = self._dir(resolved)
        if not (source / MODEL_FILE).exists():
            raise FileNotFoundError(f"Model version '{resolved}' not found at {source}")
        model = joblib.load(source / MODEL_FILE)
        metadata = json.loads((source / METADATA_FILE).read_text(encoding="utf-8"))
        baseline = pd.read_csv(source / BASELINE_FILE)
        return ModelBundle(model, metadata, baseline, resolved)


class S3Store:
    def __init__(self, bucket: str, prefix: str, region: str, client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        if client is not None:
            self.client = client
        else:  # pragma: no cover - real AWS path, exercised with moto in tests
            import boto3

            self.client = boto3.client("s3", region_name=region)

    def _key(self, version: str, name: str) -> str:
        return f"{self.prefix}/{version}/{name}"

    def list_versions(self) -> list[str]:
        versions: set[str] = set()
        base = f"{self.prefix}/"
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=base):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(base) :]
                parts = rel.split("/")
                if len(parts) >= 2 and parts[1] == MODEL_FILE:
                    versions.add(parts[0])
        return sorted(versions)

    def resolve(self, version: str) -> str:
        if version != "latest":
            return version
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError(f"No model versions under s3://{self.bucket}/{self.prefix}")
        return versions[-1]

    def save(self, version: str, model, metadata: dict, baseline: pd.DataFrame) -> str:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        self.client.upload_fileobj(buffer, self.bucket, self._key(version, MODEL_FILE))
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(version, METADATA_FILE),
            Body=json.dumps(metadata, indent=2).encode("utf-8"),
        )
        csv_text = io.StringIO()
        baseline.to_csv(csv_text, index=False)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(version, BASELINE_FILE),
            Body=csv_text.getvalue().encode("utf-8"),
        )
        return version

    def load(self, version: str) -> ModelBundle:
        resolved = self.resolve(version)
        model_bytes = io.BytesIO(
            self.client.get_object(Bucket=self.bucket, Key=self._key(resolved, MODEL_FILE))[
                "Body"
            ].read()
        )
        model = joblib.load(model_bytes)
        metadata = json.loads(
            self.client.get_object(
                Bucket=self.bucket, Key=self._key(resolved, METADATA_FILE)
            )["Body"]
            .read()
            .decode("utf-8")
        )
        baseline = pd.read_csv(
            io.BytesIO(
                self.client.get_object(
                    Bucket=self.bucket, Key=self._key(resolved, BASELINE_FILE)
                )["Body"].read()
            )
        )
        return ModelBundle(model, metadata, baseline, resolved)


def build_store(settings) -> LocalStore | S3Store:
    if settings.model_store == "s3":
        return S3Store(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.aws_region,
        )
    return LocalStore(settings.model_dir)
