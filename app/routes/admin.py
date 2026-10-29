"""Admin endpoints. ``POST /reload`` hot-swaps the model from the store."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.predictor import Predictor
from app.schemas import ReloadResponse

logger = logging.getLogger("app.admin")
router = APIRouter()


@router.post("/reload", response_model=ReloadResponse)
def reload_model(request: Request):
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="No model store configured")

    version = request.app.state.settings.model_version
    previous = getattr(request.app.state, "bundle", None)
    previous_version = previous.version if previous else None

    try:
        bundle = store.load(version)
    except Exception as exc:
        logger.exception("Reload failed")
        # Keep serving the previous model; report the failure clearly.
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc

    request.app.state.bundle = bundle
    request.app.state.predictor = Predictor(bundle)
    logger.info("Reloaded model %s (was %s)", bundle.version, previous_version)
    return ReloadResponse(
        status="reloaded",
        previous_version=previous_version,
        new_version=bundle.version,
    )
