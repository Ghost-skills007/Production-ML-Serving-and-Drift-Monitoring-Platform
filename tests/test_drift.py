"""Drift-detection math tests on known distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift import (
    DriftConfig,
    detect_drift,
    detect_feature,
    ks_test,
    psi_categorical,
    psi_numeric,
)

RNG = np.random.default_rng(0)


def _series(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_psi_numeric_identical_is_near_zero():
    base = RNG.normal(0, 1, 2000)
    current = RNG.normal(0, 1, 2000)
    assert psi_numeric(base, current) < 0.05


def test_psi_numeric_shifted_is_high():
    base = RNG.normal(0, 1, 2000)
    current = RNG.normal(3, 1, 2000)
    assert psi_numeric(base, current) > 0.2


def test_psi_numeric_self_is_zero():
    base = RNG.normal(5, 2, 1000)
    assert psi_numeric(base, base) == _approx(0.0)


def _approx(value):  # tiny helper so self-PSI test reads clearly
    import pytest

    return pytest.approx(value, abs=1e-6)


def test_ks_identical_high_pvalue():
    base = RNG.normal(0, 1, 500)
    current = RNG.normal(0, 1, 500)
    stat, p = ks_test(base, current)
    assert p > 0.05
    assert stat < 0.1


def test_ks_shifted_low_pvalue():
    base = RNG.normal(0, 1, 500)
    current = RNG.normal(2, 1, 500)
    stat, p = ks_test(base, current)
    assert p < 0.05
    assert stat > 0.3


def test_psi_categorical_identical_low():
    base = pd.Series(RNG.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2]))
    current = pd.Series(RNG.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2]))
    assert psi_categorical(base, current) < 0.05


def test_psi_categorical_shifted_high():
    base = pd.Series(RNG.choice(["a", "b", "c"], size=1000, p=[0.8, 0.1, 0.1]))
    current = pd.Series(RNG.choice(["a", "b", "c"], size=1000, p=[0.1, 0.1, 0.8]))
    assert psi_categorical(base, current) > 0.2


def test_detect_feature_numeric_rule():
    config = DriftConfig()
    base = _series(RNG.normal(0, 1, 1000))
    shifted = _series(RNG.normal(4, 1, 1000))
    result = detect_feature("f", base, shifted, is_categorical=False, config=config)
    assert result["drifted"] is True
    assert result["psi"] > 0.2 and result["ks_pvalue"] < 0.05


def test_detect_feature_numeric_psi_only_not_enough():
    # If KS is not significant, a numeric feature is not flagged even at high PSI.
    config = DriftConfig(ks_alpha=1e-300)  # practically impossible to be significant
    base = _series(RNG.normal(0, 1, 500))
    shifted = _series(RNG.normal(5, 1, 500))
    result = detect_feature("f", base, shifted, is_categorical=False, config=config)
    assert result["drifted"] is False


def test_detect_drift_overall_flag():
    numeric = ["a", "b", "c", "d", "e"]
    baseline = pd.DataFrame({n: RNG.normal(0, 1, 1000) for n in numeric})
    current = pd.DataFrame({n: RNG.normal(0, 1, 1000) for n in numeric})
    for n in ["a", "b", "c", "d"]:  # drift 4 of 5 features
        current[n] = RNG.normal(6, 1, 1000)
    results, summary = detect_drift(baseline, current, numeric, [], DriftConfig())
    assert summary["features_total"] == 5
    assert summary["features_drifted"] >= 3
    assert summary["overall_drift"] is True


def test_detect_drift_no_false_overall():
    numeric = ["a", "b", "c", "d", "e"]
    baseline = pd.DataFrame({n: RNG.normal(0, 1, 1000) for n in numeric})
    current = pd.DataFrame({n: RNG.normal(0, 1, 1000) for n in numeric})
    _, summary = detect_drift(baseline, current, numeric, [], DriftConfig())
    assert summary["overall_drift"] is False
