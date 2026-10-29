"""Traffic simulator: sends normal and deliberately shifted requests to the API.

Records are drawn from the model's stored training baseline so the "normal"
stream matches what the model was trained on. The shifted stream multiplies
selected numeric features by a factor and adds a fraction of their standard
deviation, which should push their PSI above the drift threshold and make the
drift alert fire on the dashboard.
"""

from __future__ import annotations

import argparse
import logging

import httpx
import numpy as np
import pandas as pd

from app.config import get_settings
from app.model_store import build_store

logger = logging.getLogger("scripts.simulate")


def _load_baseline():
    settings = get_settings()
    store = build_store(settings)
    bundle = store.load(settings.model_version)
    md = bundle.metadata
    return bundle.baseline, md["numeric_features"], md["categorical_features"]


def _row_to_record(row: pd.Series, numeric: list[str], categorical: list[str]) -> dict:
    record = {}
    for col in numeric:
        if col in row:
            value = row[col]
            record[col] = None if pd.isna(value) else float(value)
    for col in categorical:
        if col in row:
            value = row[col]
            record[col] = None if pd.isna(value) else str(value)
    return record


def _shift(baseline: pd.DataFrame, features: list[str], factor: float, std_add: float):
    shifted = baseline.copy()
    stats = {}
    for col in features:
        if col not in shifted.columns:
            continue
        mean = shifted[col].mean()
        std = shifted[col].std()
        std = std if std and std == std else 1.0
        shifted[col] = shifted[col] * factor + std_add * std
        # Nudge the mean too so small-variance features clearly drift.
        if abs(shifted[col].mean() - mean) < 1e-9:
            shifted[col] = shifted[col] + std_add * std
        stats[col] = std
    return shifted, stats


def _send(url: str, records: list[dict]) -> int:
    sent = 0
    with httpx.Client(base_url=url, timeout=30.0) as client:
        # Batch in chunks to keep requests fast.
        for i in range(0, len(records), 100):
            chunk = records[i : i + 100]
            try:
                resp = client.post("/predict/batch", json={"records": chunk})
                if resp.status_code == 200:
                    sent += len(chunk)
                else:
                    logger.warning("Batch %d -> %s: %s", i, resp.status_code, resp.text[:200])
            except httpx.HTTPError as exc:
                logger.error("Request failed at %d: %s", i, exc)
    return sent


def simulate(
    url: str, normal: int, shifted: int, shift_features: list[str] | None,
    factor: float, std_add: float, seed: int,
) -> dict:
    baseline, numeric, categorical = _load_baseline()

    rng = np.random.default_rng(seed)
    def sample(n: int) -> pd.DataFrame:
        idx = rng.integers(0, len(baseline), size=n)
        return baseline.iloc[idx].reset_index(drop=True)

    normal_df = sample(normal)
    normal_records = [_row_to_record(r, numeric, categorical) for _, r in normal_df.iterrows()]
    normal_sent = _send(url, normal_records)

    features_to_shift = shift_features or numeric
    shifted_df, _ = _shift(sample(shifted), features_to_shift, factor, std_add)
    shifted_records = [_row_to_record(r, numeric, categorical) for _, r in shifted_df.iterrows()]
    shifted_sent = _send(url, shifted_records)

    result = {
        "normal_requested": normal,
        "normal_sent": normal_sent,
        "shifted_requested": shifted,
        "shifted_sent": shifted_sent,
        "shifted_features": features_to_shift,
        "factor": factor,
        "std_add": std_add,
    }
    print(f"Simulated traffic: {result}")
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Send normal + drifted traffic to the API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=int, default=500)
    parser.add_argument("--shifted", type=int, default=500)
    parser.add_argument("--shift-features", nargs="*", default=None,
                        help="Numeric features to shift (default: all numeric features)")
    parser.add_argument("--factor", type=float, default=1.4)
    parser.add_argument("--std-add", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return simulate(
        args.url, args.normal, args.shifted, args.shift_features,
        args.factor, args.std_add, args.seed,
    )


if __name__ == "__main__":
    main()
