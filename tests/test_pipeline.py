"""Pipeline tests: preprocessing shape, dataset loading, training, evaluation."""

from __future__ import annotations

import numpy as np

from app.model_store import LocalStore
from pipeline.datasets import load_dataset
from pipeline.evaluate import evaluate
from pipeline.preprocess import build_preprocessor
from pipeline.train import make_estimator, train_and_save


def test_load_breast_cancer_spec():
    df, spec = load_dataset("breast_cancer", n_rows=50)
    assert spec.name == "breast_cancer"
    assert len(spec.numeric_features) == 30
    assert spec.categorical_features == []
    assert df[spec.feature_names].shape == (50, 30)


def test_load_unknown_dataset_raises():
    import pytest

    with pytest.raises(ValueError):
        load_dataset("does-not-exist")


def test_preprocessor_output_shape():
    df, spec = load_dataset("breast_cancer", n_rows=50)
    prep = build_preprocessor(spec)
    Xt = prep.fit_transform(df[spec.feature_names])
    assert Xt.shape[0] == 50
    # StandardScaler output is finite for all numeric columns.
    assert np.isfinite(Xt).all()


def test_make_estimator_types():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    assert isinstance(make_estimator("random_forest"), RandomForestClassifier)
    assert isinstance(make_estimator("gradient_boosting"), GradientBoostingClassifier)


def test_evaluate_metrics_keys():
    df, spec = load_dataset("breast_cancer", n_rows=120)
    X = df[spec.feature_names]
    y = df[spec.target].astype(str)
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("prep", build_preprocessor(spec)), ("clf", make_estimator("random_forest"))])
    pipe.fit(X, y)
    metrics = evaluate(pipe, X, y)
    assert set(metrics) >= {"accuracy", "f1_macro", "roc_auc", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_and_save_writes_artifacts(tmp_path):
    store = LocalStore(tmp_path)
    meta = train_and_save("breast_cancer", "random_forest", n_rows=200, store=store)
    version = meta["model_version"]
    bundle = store.load(version)
    assert bundle.version == version
    assert bundle.metadata["dataset"] == "breast_cancer"
    assert len(bundle.baseline) > 0
    assert meta["metrics"]["accuracy"] > 0.7
