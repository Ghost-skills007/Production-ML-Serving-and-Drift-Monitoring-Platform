"""Drift job: compare recent logged predictions against the training baseline.

Reads the most recent ``--window`` prediction rows from PostgreSQL, rebuilds the
feature frame, loads the model's stored baseline, runs PSI + KS drift detection
and persists the per-feature results and a run summary back to PostgreSQL.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import DriftResult, DriftRun, Prediction, get_database, init_database
from app.model_store import build_store
from monitoring.drift import DriftConfig, detect_drift

logger = logging.getLogger("monitoring.drift_job")


def _current_window(db, window: int) -> pd.DataFrame:
    with db.session() as session:
        rows = (
            session.execute(
                select(Prediction.features)
                .order_by(Prediction.ts.desc())
                .limit(window)
            )
            .scalars()
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(list(rows))


def run(window: int = 1000, store=None, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    db = get_database()
    if db is None:
        db = init_database(settings.sqlalchemy_url)

    current = _current_window(db, window)
    if current.empty:
        logger.warning("No predictions logged yet; nothing to analyse")
        return {"features_total": 0, "features_drifted": 0, "overall_drift": False, "rows": 0}

    store = store or build_store(settings)
    bundle = store.load(settings.model_version)
    md = bundle.metadata
    numeric = md.get("numeric_features", [])
    categorical = md.get("categorical_features", [])
    baseline = bundle.baseline

    # Only keep columns present in both baseline and current traffic.
    cols = [c for c in current.columns if c in baseline.columns]
    config = DriftConfig(
        psi_threshold=settings.psi_threshold,
        ks_alpha=settings.ks_alpha,
        bins=settings.drift_bin_count,
        epsilon=settings.drift_epsilon,
        drift_feature_fraction=settings.drift_feature_fraction,
    )
    results, summary = detect_drift(
        baseline[cols], current[cols], numeric, categorical, config
    )

    _persist(db, bundle.version, len(current), results, summary)
    _print_summary(bundle.version, len(current), results, summary)
    summary["rows"] = len(current)
    return summary


def _persist(db, model_version: str, rows: int, results: list[dict], summary: dict) -> None:
    with db.session() as session:
        for r in results:
            session.add(
                DriftResult(
                    model_version=model_version,
                    feature=r["feature"],
                    ks_stat=_clean(r["ks_stat"]),
                    ks_pvalue=_clean(r["ks_pvalue"]),
                    psi=_clean(r["psi"]),
                    drifted=r["drifted"],
                )
            )
        session.add(
            DriftRun(
                model_version=model_version,
                window_rows=rows,
                features_drifted=summary["features_drifted"],
                features_total=summary["features_total"],
                overall_drift=summary["overall_drift"],
            )
        )
        session.commit()


def _clean(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None


def _print_summary(version: str, rows: int, results: list[dict], summary: dict) -> None:
    print(f"Drift job @ model {version}, {rows} recent predictions analysed")
    print(f"{'feature':22} {'PSI':>8} {'KS_stat':>9} {'KS_p':>10}  drifted")
    for r in results:
        ks_stat = "n/a" if r["ks_stat"] != r["ks_stat"] else f"{r['ks_stat']:.4f}"
        ks_p = "n/a" if r["ks_pvalue"] != r["ks_pvalue"] else f"{r['ks_pvalue']:.3g}"
        print(f"{r['feature']:22} {r['psi']:8.4f} {ks_stat:>9} {ks_p:>10}  {r['drifted']}")
    print(
        f"Overall drift: {summary['overall_drift']} "
        f"({summary['features_drifted']}/{summary['features_total']} features, "
        f"{summary['drifted_fraction']:.1%})"
    )


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run drift detection on logged predictions.")
    parser.add_argument("--window", type=int, default=1000, help="Number of recent rows to analyse")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return run(window=args.window)


if __name__ == "__main__":
    main()
