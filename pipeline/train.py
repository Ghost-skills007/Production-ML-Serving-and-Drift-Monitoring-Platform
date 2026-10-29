"""Training entrypoint.

Loads a dataset, fits a scikit-learn ``Pipeline`` (preprocessor + classifier),
evaluates on a held-out stratified split, and saves a versioned artifact
(``model.joblib`` + ``metadata.json`` + ``baseline.csv``) to the configured
model store.

Run ``python -m pipeline.train --dataset breast_cancer`` for a fast, offline
model; the default ``--dataset adult`` uses the UCI income dataset.
"""

from __future__ import annotations

import argparse
import platform
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import get_settings
from app.model_store import build_store
from pipeline.datasets import load_dataset
from pipeline.evaluate import evaluate
from pipeline.preprocess import build_preprocessor

# Fraction of training rows kept as the drift-detection reference baseline.
BASELINE_MAX_ROWS = 5000


def utc_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def make_estimator(name: str):
    name = name.lower()
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError("model must be 'random_forest' or 'gradient_boosting'")


def train_and_save(
    dataset: str = "adult",
    model: str = "random_forest",
    n_rows: int | None = None,
    store=None,
) -> dict:
    df, spec = load_dataset(dataset, n_rows=n_rows)
    features = df[spec.feature_names].copy()
    target = df[spec.target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor(spec)),
            ("clf", make_estimator(model)),
        ]
    )
    pipeline.fit(X_train, y_train)

    positive = spec.class_labels[-1] if spec.class_labels else None
    metrics = evaluate(pipeline, X_test, y_test, positive_class=positive)

    baseline = features.sample(
        n=min(BASELINE_MAX_ROWS, len(features)), random_state=42
    ).reset_index(drop=True)

    metadata = {
        "dataset": spec.name,
        "estimator": model,
        "feature_names": spec.feature_names,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "class_labels": list(pipeline.classes_),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc": metrics["roc_auc"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "library_versions": _library_versions(),
        "training_row_count": int(len(X_train)),
        "created_at": datetime.now(UTC).isoformat(),
    }

    store = store or build_store(get_settings())
    version = utc_version()
    store.save(version, pipeline, metadata, baseline)
    metadata["model_version"] = version
    return metadata


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train and save a versioned model.")
    parser.add_argument("--dataset", default="adult", help="adult | breast_cancer")
    parser.add_argument(
        "--model", default="random_forest", help="random_forest | gradient_boosting"
    )
    parser.add_argument("--rows", type=int, default=None, help="Optional row cap (debugging)")
    args = parser.parse_args(argv)

    metadata = train_and_save(args.dataset, args.model, args.rows)
    print(f"Saved model version {metadata['model_version']}")
    print(f"Metrics: {metadata['metrics']}")
    return metadata


if __name__ == "__main__":
    main()
