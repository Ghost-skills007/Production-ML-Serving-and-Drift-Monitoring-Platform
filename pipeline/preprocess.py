"""Preprocessing: feature specification and the scikit-learn ``Pipeline``.

A single :class:`~sklearn.compose.ColumnTransformer` handles missing values,
categorical encoding and numeric scaling so that the exact same transformations
are applied at training and inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetSpec:
    """Describes the input columns of a dataset."""

    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    target: str
    class_labels: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_features + self.categorical_features


# UCI Adult Income (fetch_openml "adult", version=2). Target is binarised to
# "<=50K" / ">50K" at load time.
ADULT_SPEC = DatasetSpec(
    name="adult",
    numeric_features=[
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    categorical_features=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    target="class",
    class_labels=["<=50K", ">50K"],
)


def build_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    """ColumnTransformer: median+scale for numerics, mode+one-hot for categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ],
        remainder="drop",
    )
