"""Streamlit monitoring dashboard.

Reads prediction, drift-result and drift-run data directly from PostgreSQL and
renders five tabs - Overview, Latency, Drift, Recent predictions and Alerts -
with a sidebar time-window filter. Run via ``streamlit run monitoring/dashboard.py``
(or the ``dashboard`` service in docker-compose).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from app.config import get_settings


@st.cache_resource
def _engine():
    from sqlalchemy import create_engine

    settings = get_settings()
    return create_engine(settings.sqlalchemy_url, future=True, pool_pre_ping=True)


def _query(sql: str, params: dict) -> pd.DataFrame:
    try:
        with _engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:  # pragma: no cover - dashboard runtime concern
        st.error(f"Could not read from PostgreSQL: {exc}")
        return pd.DataFrame()


def _cutoff(hours: int) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


st.set_page_config(page_title="ML Serving & Drift Monitor", layout="wide")
st.title("ML Serving & Drift Monitoring Dashboard")

with st.sidebar:
    st.header("Filters")
    window_hours = st.slider("Time window (hours)", 1, 720, 72, step=1)
    st.caption(f"Showing data from the last {window_hours}h (UTC).")
    if st.button("Refresh"):
        st.cache_data.clear()

cutoff = _cutoff(window_hours)
params = {"cutoff": cutoff}

predictions = _query(
    "SELECT ts, model_version, prediction, probability, latency_ms "
    "FROM predictions WHERE ts >= :cutoff ORDER BY ts",
    params,
)
drift_results = _query(
    "SELECT ts, model_version, feature, ks_stat, ks_pvalue, psi, drifted "
    "FROM drift_results WHERE ts >= :cutoff ORDER BY ts",
    params,
)
drift_runs = _query(
    "SELECT ts, model_version, window_rows, features_drifted, features_total, overall_drift "
    "FROM drift_runs WHERE ts >= :cutoff ORDER BY ts",
    params,
)

# ---- Alerts banner (top-level, always visible) ----
if not drift_runs.empty and bool(drift_runs.iloc[-1]["overall_drift"]):
    last = drift_runs.iloc[-1]
    st.error(
        f"🚨 Overall drift detected on model `{last['model_version']}`: "
        f"{int(last['features_drifted'])}/{int(last['features_total'])} features drifted."
    )
else:
    st.success("✅ No overall drift in the latest run.")

tab_overview, tab_latency, tab_drift, tab_recent = st.tabs(
    ["Overview", "Latency", "Drift", "Recent predictions"]
)

with tab_overview:
    st.subheader("Request volume over time")
    if predictions.empty:
        st.info("No predictions logged in this window.")
    else:
        pred = predictions.copy()
        pred["ts"] = pd.to_datetime(pred["ts"], utc=True)
        buckets = pred.set_index("ts").resample("5min").size().reset_index(name="requests")
        st.plotly_chart(
            px.line(buckets, x="ts", y="requests", title="Predictions per 5 minutes"),
            use_container_width=True,
        )
    col1, col2, col3 = st.columns(3)
    col1.metric("Total requests", len(predictions))
    versions = predictions["model_version"].nunique() if not predictions.empty else 0
    col2.metric("Model versions", versions)
    latest = drift_runs.iloc[-1] if not drift_runs.empty else None
    col3.metric("Last drift run", "yes" if latest is not None else "no")

with tab_latency:
    st.subheader("Latency over time")
    if predictions.empty:
        st.info("No latency data in this window.")
    else:
        pred = predictions.copy()
        pred["ts"] = pd.to_datetime(pred["ts"], utc=True)
        st.metric("p50 (window)", f"{pred['latency_ms'].quantile(0.5):.1f} ms")
        st.metric("p95 (window)", f"{pred['latency_ms'].quantile(0.95):.1f} ms")
        roll = pred.set_index("ts")["latency_ms"].rolling("5min").median().reset_index()
        st.plotly_chart(
            px.line(roll, x="ts", y="latency_ms", title="Median latency per 5 minutes"),
            use_container_width=True,
        )

with tab_drift:
    st.subheader("Per-feature PSI")
    if drift_results.empty:
        st.info("No drift results in this window. Run the drift job to populate them.")
    else:
        latest_ts = drift_results["ts"].max()
        current = drift_results[drift_results["ts"] == latest_ts].copy()
        fig = px.bar(current, x="feature", y="psi", color="drifted",
                     title="PSI by feature (latest run)")
        fig.add_hline(y=0.1, line_dash="dot", annotation_text="0.1")
        fig.add_hline(y=0.2, line_dash="dot", annotation_text="0.2 threshold")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            current[["feature", "psi", "ks_stat", "ks_pvalue", "drifted"]],
            use_container_width=True,
        )
        st.subheader("Drift history")
        if not drift_runs.empty:
            runs = drift_runs.copy()
            runs["ts"] = pd.to_datetime(runs["ts"], utc=True)
            st.plotly_chart(
                px.line(runs, x="ts", y="features_drifted", title="Features drifted over time"),
                use_container_width=True,
            )

with tab_recent:
    st.subheader("Recent predictions")
    if predictions.empty:
        st.info("No predictions in this window.")
    else:
        st.dataframe(predictions.tail(200).iloc[::-1], use_container_width=True)
