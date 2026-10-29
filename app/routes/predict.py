"""Prediction endpoints: ``POST /predict`` and ``POST /predict/batch``."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.db import log_prediction
from app.predictor import InputValidationError, Predictor
from app.schemas import (
    BatchRequest,
    BatchResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


def _get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def _log_task(version: str, features, prediction, probability, latency_ms):
    log_prediction(version, features, prediction, probability, latency_ms)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, background: BackgroundTasks):
    predictor = _get_predictor(request)
    try:
        label, probability, latency_ms = predictor.predict(req.features)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc

    background.add_task(
        _log_task, predictor.model_version, req.features, label, probability, latency_ms
    )
    return PredictResponse(
        prediction=label,
        probability=probability,
        model_version=predictor.model_version,
        latency_ms=latency_ms,
    )


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, request: Request, background: BackgroundTasks):
    predictor = _get_predictor(request)
    settings = request.app.state.settings
    if not req.records:
        raise HTTPException(status_code=422, detail="records must not be empty")
    if len(req.records) > settings.batch_max_records:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(req.records)} > {settings.batch_max_records}",
        )
    try:
        results = predictor.predict_batch(req.records)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc

    # Log the whole batch in a single background task (list of rows).
    background.add_task(_log_batch_task, predictor.model_version, req.records, results)

    payload = [
        PredictionResult(prediction=label, probability=prob)
        for (label, prob, _ms) in results
    ]
    total_ms = round(sum(r[2] for r in results), 3)
    return BatchResponse(
        results=payload,
        count=len(payload),
        model_version=predictor.model_version,
        latency_ms=total_ms,
    )


def _log_batch_task(version: str, records, results):
    for features, (label, prob, ms) in zip(records, results, strict=True):
        log_prediction(version, features, label, prob, ms)
