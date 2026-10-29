"""SQLAlchemy engine, session factory and ORM models for prediction logging.

Three tables back the monitoring platform:

* ``predictions``   - one row per served inference (features stored as JSON).
* ``drift_results`` - per-feature KS/PSI results written by the drift job.
* ``drift_runs``    - one summary row per drift-job execution.

Logging is best-effort: :func:`log_prediction` swallows DB errors so a missing
or unhealthy PostgreSQL never breaks the serving path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

logger = logging.getLogger("app.db")


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[str] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)


class DriftResult(Base):
    __tablename__ = "drift_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    feature: Mapped[str] = mapped_column(String(128))
    ks_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    ks_pvalue: Mapped[float | None] = mapped_column(Float, nullable=True)
    psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    drifted: Mapped[bool] = mapped_column(Boolean, default=False)


class DriftRun(Base):
    __tablename__ = "drift_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    window_rows: Mapped[int] = mapped_column(Integer)
    features_drifted: Mapped[int] = mapped_column(Integer)
    features_total: Mapped[int] = mapped_column(Integer)
    overall_drift: Mapped[bool] = mapped_column(Boolean, default=False)


class Database:
    """Thin wrapper owning the engine and session factory."""

    def __init__(self, url: str):
        self.url = url
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, future=True)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self):
        return self.SessionLocal()


# Module-level singleton populated by the app lifespan / scripts.
_db: Database | None = None


def init_database(url: str, create: bool = True) -> Database:
    global _db
    _db = Database(url)
    if create:
        try:
            _db.create_all()
        except Exception as exc:  # pragma: no cover - depends on DB availability
            logger.warning("Could not create tables (will rely on init_db.sql): %s", exc)
    return _db


def get_database() -> Database | None:
    return _db


def log_prediction(
    model_version: str,
    features: dict[str, Any],
    prediction: str,
    probability: float,
    latency_ms: float,
) -> bool:
    """Persist one prediction row. Returns True on success, False (with a warning) otherwise."""
    db = get_database()
    if db is None:
        logger.warning("Prediction logging skipped: database not initialised")
        return False
    try:
        with db.session() as session:
            session.add(
                Prediction(
                    model_version=model_version,
                    features=features,
                    prediction=prediction,
                    probability=probability,
                    latency_ms=latency_ms,
                )
            )
            session.commit()
        return True
    except Exception as exc:
        logger.warning("Failed to log prediction (serving continues anyway): %s", exc)
        return False
