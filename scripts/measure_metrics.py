"""Measure real metrics and write ``docs/results.md``.

Produces every number quoted in the README, or the string ``not yet measured``
when a dependency (a live API endpoint, a reachable database) is unavailable:

* held-out accuracy / macro F1 / ROC-AUC - read from the trained model metadata;
* p50 / p95 latency - measured against the live endpoint when reachable;
* logged-request count - read from PostgreSQL when reachable;
* drift detection results - computed offline against the stored training baseline
  for a no-shift sample (false-alarm check) and a shifted sample (detection check).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from app.config import get_settings
from app.model_store import build_store
from monitoring.drift import DriftConfig, detect_drift

NOT_MEASURED = "not yet measured"


def _sample(baseline: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    idx = rng.integers(0, len(baseline), size=n)
    return baseline.iloc[idx].reset_index(drop=True)


def _shift(df: pd.DataFrame, features: list[str], factor: float, std_add: float) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            continue
        std = out[col].std()
        std = std if std and std == std else 1.0
        out[col] = out[col] * factor + std_add * std
    return out


def measure_latency(url: str, n: int, feature_record: dict) -> dict:
    latencies = []
    with httpx.Client(base_url=url, timeout=30.0) as client:
        for _ in range(n):
            start = time.perf_counter()
            resp = client.post("/predict", json={"features": feature_record})
            elapsed = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 200:
                latencies.append(elapsed)
    if not latencies:
        return {"p50": None, "p95": None, "count": 0}
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "count": len(latencies),
    }


def count_logged(url: str) -> int | None:
    settings = get_settings()
    if not settings.db_enabled:
        return None
    try:
        from sqlalchemy import func, select

        from app.db import Prediction, init_database

        db = init_database(settings.sqlalchemy_url, create=False)
        with db.session() as session:
            return int(session.execute(select(func.count(Prediction.id))).scalar_one())
    except Exception:
        return None


def run_metrics(url: str | None, latency_requests: int, window: int, seed: int) -> dict:
    settings = get_settings()
    store = build_store(settings)
    bundle = store.load(settings.model_version)
    md = bundle.metadata
    baseline = bundle.baseline
    numeric = md.get("numeric_features", [])
    categorical = md.get("categorical_features", [])
    config = DriftConfig(
        psi_threshold=settings.psi_threshold,
        ks_alpha=settings.ks_alpha,
        bins=settings.drift_bin_count,
        epsilon=settings.drift_epsilon,
        drift_feature_fraction=settings.drift_feature_fraction,
    )
    rng = np.random.default_rng(seed)

    # No-shift: false alarms should be ~0.
    nodrift_current = _sample(baseline, window, rng)
    _, nodrift_summary = detect_drift(
        baseline, nodrift_current, numeric, categorical, config
    )

    # Shift all numeric features: should be detected.
    shift_current = _shift(_sample(baseline, window, rng), numeric, 1.4, 0.5)
    results, shift_summary = detect_drift(baseline, shift_current, numeric, categorical, config)
    flagged = [r["feature"] for r in results if r["drifted"]]
    shifted_psi = {r["feature"]: r["psi"] for r in results if r["feature"] in numeric}

    metrics = {
        "model_version": bundle.version,
        "dataset": md["dataset"],
        "accuracy": md["metrics"]["accuracy"],
        "f1_macro": md["metrics"]["f1_macro"],
        "roc_auc": md["metrics"]["roc_auc"],
        "training_row_count": md["training_row_count"],
        "no_drift_false_alarms": nodrift_summary["features_drifted"],
        "no_drift_overall": nodrift_summary["overall_drift"],
        "shift_features_total": shift_summary["features_total"],
        "shift_features_drifted": shift_summary["features_drifted"],
        "shift_overall": shift_summary["overall_drift"],
        "shift_flagged_features": flagged,
        "shift_psi": shifted_psi,
    }

    # Live-only measurements.
    metrics["latency"] = {"p50": None, "p95": None, "count": 0}
    metrics["logged_requests"] = None
    if url:
        try:
            sample = _sample(baseline, 1, rng).iloc[0]
            record = {}
            for col in numeric:
                record[col] = float(sample[col]) if pd.notna(sample[col]) else None
            for col in categorical:
                record[col] = str(sample[col]) if pd.notna(sample[col]) else None
            metrics["latency"] = measure_latency(url, latency_requests, record)
            metrics["logged_requests"] = count_logged(url)
        except Exception as exc:
            print(f"Live measurement skipped ({url} unreachable): {exc}")
    return metrics


def _fmt(value, kind="num"):
    if value is None:
        return NOT_MEASURED
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "ms":
        return f"{value:.1f} ms"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_results(metrics: dict, path: Path) -> None:
    lat = metrics["latency"]
    flagged_text = ", ".join(metrics["shift_flagged_features"]) or "none"
    lines = [
        "# Measured Results",
        "",
        f"_Generated by `scripts/measure_metrics.py`. Model version `{metrics['model_version']}` "
        f"({metrics['dataset']})._",
        "",
        "## Model quality (held-out)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {_fmt(metrics['accuracy'])} |",
        f"| Macro F1 | {_fmt(metrics['f1_macro'])} |",
        f"| ROC-AUC | {_fmt(metrics['roc_auc'])} |",
        f"| Training rows | {_fmt(metrics['training_row_count'])} |",
        "",
        "## Serving latency (live endpoint)",
        "",
        "| Percentile | Value |",
        "|---|---|",
        f"| p50 | {_fmt(lat['p50'], 'ms')} |",
        f"| p95 | {_fmt(lat['p95'], 'ms')} |",
        f"| Requests measured | {_fmt(lat['count'])} |",
        f"| Logged requests in DB | {_fmt(metrics['logged_requests'])} |",
        "",
        "## Drift detection (against stored baseline)",
        "",
        "| Scenario | Features drifted | Features total | Overall drift |",
        "|---|---|---|---|",
        f"| No shift (false alarms) | {metrics['no_drift_false_alarms']} | "
        f"{metrics['shift_features_total']} | {metrics['no_drift_overall']} |",
        f"| All numeric shifted | {metrics['shift_features_drifted']} | "
        f"{metrics['shift_features_total']} | {metrics['shift_overall']} |",
        "",
        f"Shifted features flagged as drifted: {flagged_text}",
        "",
        "Per-feature PSI after shifting:",
        "",
        "| Feature | PSI |",
        "|---|---|",
    ]
    for feature, psi in metrics["shift_psi"].items():
        lines.append(f"| {feature} | {psi:.4f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Measure metrics, write docs/results.md")
    parser.add_argument("--url", default=None, help="Live API base URL for latency/DB checks")
    parser.add_argument("--latency-requests", type=int, default=100)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="docs/results.md")
    args = parser.parse_args(argv)

    metrics = run_metrics(args.url, args.latency_requests, args.window, args.seed)
    write_results(metrics, Path(args.out))
    import json

    print(json.dumps(metrics, indent=2, default=str))
    return metrics


if __name__ == "__main__":
    main()
