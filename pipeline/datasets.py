"""Dataset loading helpers.

Two sources are supported:

* ``adult`` - UCI Adult Income via ``sklearn.datasets.fetch_openml`` (default,
  needs network the first time).
* ``breast_cancer`` - the bundled sklearn dataset, used by unit tests and the
  fast/offline training path so CI never touches the network.

Both return a ``(dataframe, spec)`` tuple with consistent string categoricals
and a binarised target, ready for :func:`pipeline.preprocess.build_preprocessor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from pipeline.preprocess import ADULT_SPEC, DatasetSpec

# Categorical columns in Adult arrive as pandas ``category``/``object``; force
# them to plain strings so the one-hot encoder and drift job see stable values.
_ADULT_STR_COLS = ADULT_SPEC.categorical_features


def _spec_for_breast_cancer() -> DatasetSpec:
    bunch = load_breast_cancer()
    numeric = [str(c) for c in bunch.feature_names]
    return DatasetSpec(
        name="breast_cancer",
        numeric_features=numeric,
        categorical_features=[],
        target="target",
        class_labels=list(bunch.target_names),  # ['malignant', 'benign']
    )


def load_adult(n_rows: int | None = None) -> tuple[pd.DataFrame, DatasetSpec]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="adult", version=2, parser="pandas", as_frame=True)
    df = bunch.frame.copy()
    # Target column is 'class' (e.g. '<=50K' / '>50K'), sometimes with stray spaces.
    df["class"] = df["class"].astype(str).str.strip()
    for col in _ADULT_STR_COLS:
        df[col] = df[col].astype(str)
    # Replace the '?' sentinel used for missing categoricals.
    df = df.replace("?", np.nan)
    if n_rows is not None:
        df = df.head(n_rows)
    return df, ADULT_SPEC


def load_breast_cancer_df(n_rows: int | None = None) -> tuple[pd.DataFrame, DatasetSpec]:
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    # sklearn target: 0 -> malignant, 1 -> benign. Map to string labels.
    labels = np.array(bunch.target_names)
    df["target"] = pd.Categorical.from_codes(bunch.target, categories=labels).astype(str)
    if n_rows is not None:
        df = df.head(n_rows)
    return df, _spec_for_breast_cancer()


def load_dataset(name: str, n_rows: int | None = None) -> tuple[pd.DataFrame, DatasetSpec]:
    name = name.lower()
    if name == "adult":
        return load_adult(n_rows=n_rows)
    if name in {"breast_cancer", "breast-cancer", "bc"}:
        return load_breast_cancer_df(n_rows=n_rows)
    raise ValueError(f"Unknown dataset '{name}'. Use 'adult' or 'breast_cancer'.")
