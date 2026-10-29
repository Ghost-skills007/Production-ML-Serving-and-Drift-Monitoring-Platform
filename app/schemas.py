"""Pydantic v2 request/response models for the serving API.

The feature set is model-dependent (UCI Adult vs. breast-cancer), so request
bodies carry a free-form ``features`` mapping. Feature-level validation
(required names, types, missing/extra keys) happens in
:mod:`app.predictor` against the loaded model's metadata, which lets us return
structured ``422`` errors instead of an unhandled ``500``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = float | int | str | bool | None


class PredictRequest(BaseModel):
    """A single record: mapping of feature name to raw value."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        ...,
        description="Feature name -> value mapping for one record.",
        min_length=1,
    )


class PredictionResult(BaseModel):
    prediction: str
    probability: float


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(
        ...,
        description="List of feature mappings (up to the configured batch limit).",
    )


class BatchResponse(BaseModel):
    results: list[PredictionResult]
    count: int
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    dataset: str | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    reason: str | None = None


class ModelMetadataResponse(BaseModel):
    model_version: str
    dataset: str
    feature_names: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    class_labels: list[str]
    metrics: dict[str, float]
    library_versions: dict[str, str]
    training_row_count: int
    created_at: str


class ReloadResponse(BaseModel):
    status: str
    previous_version: str | None = None
    new_version: str | None = None
