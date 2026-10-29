"""Health, readiness and model-metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas import HealthResponse, ModelMetadataResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    bundle = getattr(request.app.state, "bundle", None)
    loaded = predictor is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=bundle.version if bundle else None,
        dataset=bundle.metadata.get("dataset") if bundle else None,
    )


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        # Signal to orchestrators that the service cannot serve traffic yet.
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ReadinessResponse(ready=True)


@router.get("/model", response_model=ModelMetadataResponse)
def model(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    md = bundle.metadata
    return ModelMetadataResponse(
        model_version=bundle.version,
        dataset=md["dataset"],
        feature_names=md["feature_names"],
        categorical_features=md.get("categorical_features", []),
        numeric_features=md.get("numeric_features", []),
        class_labels=md.get("class_labels", []),
        metrics=md.get("metrics", {}),
        library_versions=md.get("library_versions", {}),
        training_row_count=md.get("training_row_count", 0),
        created_at=md.get("created_at", ""),
    )
