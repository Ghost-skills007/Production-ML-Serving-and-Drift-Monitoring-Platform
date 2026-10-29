"""FastAPI application factory.

The lifespan hook:

1. builds the configured model store (local or S3),
2. loads the requested model version - a broken artifact fails startup loudly,
3. initialises PostgreSQL logging - an unreachable DB is a non-fatal warning so
   the API still serves predictions without logging.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db import init_database
from app.model_store import build_store
from app.predictor import Predictor
from app.routes import admin, health, predict

logger = logging.getLogger("app.main")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    _setup_logging(settings.log_level)

    # --- Model load (fatal on failure) ---
    store = build_store(settings)
    app.state.store = store
    logger.info(
        "Loading model version '%s' from %s store",
        settings.model_version,
        settings.model_store,
    )
    bundle = store.load(settings.model_version)
    app.state.bundle = bundle
    app.state.predictor = Predictor(bundle)
    logger.info(
        "Model %s loaded (%d features)",
        bundle.version,
        len(bundle.metadata["feature_names"]),
    )

    # --- Database logging (non-fatal) ---
    if settings.db_enabled:
        try:
            init_database(settings.sqlalchemy_url)
            logger.info("PostgreSQL logging initialised")
        except Exception as exc:
            logger.warning("Database unavailable, prediction logging disabled: %s", exc)
    else:
        logger.info("Prediction logging disabled via DB_ENABLED=0")

    yield

    logger.info("Shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="ML Serving and Drift Monitoring Platform",
        version="1.0.0",
        description="Production-style model serving API with drift monitoring.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = None
    app.state.bundle = None
    app.state.store = None

    app.include_router(predict.router)
    app.include_router(health.router)
    app.include_router(admin.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "ml-serving-drift-platform", "docs": "/docs"}

    return app


app = create_app()
