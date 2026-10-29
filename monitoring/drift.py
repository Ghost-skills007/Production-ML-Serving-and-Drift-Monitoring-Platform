"""Data-drift detection: Population Stability Index (PSI) and two-sample KS test.

Numeric features use PSI over 10 quantile bins taken from the training baseline
(with a small epsilon to avoid ``log(0)``) together with a two-sample
Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``). Categorical features use PSI
computed on category frequencies.

A feature is flagged *drifted* when ``PSI > psi_threshold`` and - for numerics -
the KS p-value is below ``ks_alpha``. Overall drift is reported when more than
``drift_feature_fraction`` of the monitored features drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftConfig:
    psi_threshold: float = 0.2
    ks_alpha: float = 0.05
    bins: int = 10
    epsilon: float = 1e-6
    drift_feature_fraction: float = 0.2


def psi_numeric(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI for a numeric feature using quantile bins derived from the baseline."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if baseline.size == 0 or current.size == 0:
        return 0.0

    # Quantile edges from the baseline distribution.
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        # Baseline is constant -> fall back to fixed +/- range around the value.
        val = float(edges[0])
        edges = np.array([val - 0.5, val + 0.5])
    # Widen the outermost bins to +/-inf so drift beyond the training range is
    # counted (np.histogram otherwise silently drops out-of-range values).
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    base_pct = (base_counts + epsilon) / (base_counts.sum() + epsilon * base_counts.size)
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * cur_counts.size)
    psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
    return float(psi)


def psi_categorical(
    baseline: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """PSI on category frequency distributions."""
    base = baseline.dropna().astype(str)
    cur = current.dropna().astype(str)
    if base.empty or cur.empty:
        return 0.0
    categories = set(base.unique()) | set(cur.unique())
    base_counts = base.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_counts = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    base_pct = base_counts.to_numpy() + epsilon
    cur_pct = cur_counts.to_numpy() + epsilon
    psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
    return float(psi)


def ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and p-value (None-robust)."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if baseline.size < 2 or current.size < 2:
        return float("nan"), float("nan")
    result = ks_2samp(baseline, current)
    return float(result.statistic), float(result.pvalue)


def detect_feature(
    name: str,
    baseline: pd.Series,
    current: pd.Series,
    is_categorical: bool,
    config: DriftConfig,
) -> dict:
    """Compute PSI (and, for numerics, KS) and decide whether the feature drifted."""
    if is_categorical:
        psi = psi_categorical(baseline, current, epsilon=config.epsilon)
        ks_stat, ks_p = float("nan"), float("nan")
        drifted = psi > config.psi_threshold
    else:
        psi = psi_numeric(
            baseline.to_numpy(), current.to_numpy(), bins=config.bins, epsilon=config.epsilon
        )
        ks_stat, ks_p = ks_test(baseline.to_numpy(), current.to_numpy())
        significant = bool(ks_p == ks_p and ks_p < config.ks_alpha)  # NaN-safe
        drifted = psi > config.psi_threshold and significant
    return {
        "feature": name,
        "psi": psi,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "drifted": bool(drifted),
    }


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: DriftConfig | None = None,
) -> tuple[list[dict], dict]:
    """Run per-feature drift detection and summarise the overall result."""
    config = config or DriftConfig()
    monitored = [f for f in numeric_features + categorical_features if f in baseline.columns]
    results = []
    for feature in monitored:
        if feature not in current.columns:
            continue
        is_cat = feature in categorical_features
        results.append(
            detect_feature(feature, baseline[feature], current[feature], is_cat, config)
        )

    features_total = len(results)
    features_drifted = sum(1 for r in results if r["drifted"])
    fraction = features_drifted / features_total if features_total else 0.0
    overall = fraction > config.drift_feature_fraction
    summary = {
        "features_total": features_total,
        "features_drifted": features_drifted,
        "drifted_fraction": fraction,
        "overall_drift": bool(overall),
    }
    return results, summary
