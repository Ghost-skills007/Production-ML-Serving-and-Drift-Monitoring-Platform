"""Inference wrapper: input validation, prediction and timing.

:class:`Predictor` holds a loaded :class:`~app.model_store.ModelBundle` and turns
raw feature mappings into predictions. Feature-level validation runs against the
model's ``metadata`` (expected names, numeric vs. categorical) so bad input
raises a structured :class:`InputValidationError` that the route layer converts
to a ``422`` response - never a ``500``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    """Raised when a request record does not match the model's feature contract."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__(f"{len(errors)} input error(s)")


class Predictor:
    def __init__(self, bundle):
        self.bundle = bundle
        md = bundle.metadata
        self.feature_names: list[str] = md["feature_names"]
        self.numeric_features: list[str] = md.get("numeric_features", [])
        self.categorical_features: list[str] = md.get("categorical_features", [])
        self.class_labels: list[str] = md.get("class_labels", [])

    @property
    def model_version(self) -> str:
        return self.bundle.version

    # ---- validation ---------------------------------------------------
    def _validate_record(self, record: dict[str, Any], index: int) -> None:
        errors: list[dict[str, Any]] = []
        expected = set(self.feature_names)
        for name in record:
            if name not in expected:
                errors.append(
                    {
                        "loc": ["records", index, name],
                        "msg": "unexpected feature",
                        "type": "unexpected_field",
                    }
                )
        for name in self.feature_names:
            if name not in record:
                errors.append(
                    {
                        "loc": ["records", index, name],
                        "msg": "field required",
                        "type": "missing_field",
                    }
                )
                continue
            value = record[name]
            if value is None:
                continue  # missing values are handled by the imputer
            if name in self.numeric_features and not _is_number(value):
                errors.append(
                    {
                        "loc": ["records", index, name],
                        "msg": "value must be numeric",
                        "type": "numeric_type",
                    }
                )
        if errors:
            raise InputValidationError(errors)

    def _to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [{name: rec.get(name) for name in self.feature_names} for rec in records]
        df = pd.DataFrame(rows, columns=self.feature_names)
        for col in self.numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.categorical_features:
            # Keep non-missing values as strings; NaN stays for the imputer.
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
        return df

    # ---- prediction ---------------------------------------------------
    def predict(self, record: dict[str, Any]) -> tuple[str, float, float]:
        return self.predict_batch([record])[0]

    def predict_batch(
        self, records: list[dict[str, Any]]
    ) -> list[tuple[str, float, float]]:
        for i, rec in enumerate(records):
            self._validate_record(rec, i)

        frame = self._to_frame(records)
        start = time.perf_counter()
        model = self.bundle.model
        proba = model.predict_proba(frame)
        predictions = model.predict(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results: list[tuple[str, float, float]] = []
        for label, row_proba in zip(predictions, proba, strict=True):
            confidence = float(np.max(row_proba))
            results.append((str(label), confidence, round(elapsed_ms, 3)))
        return results


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))
