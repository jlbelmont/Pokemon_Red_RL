"""Evaluation helpers: lead-time, alert quality."""
from __future__ import annotations

import pandas as pd


def compute_lead_time(alert_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Align alert episodes with stagnation labels to compute lead time."""
    if alert_df.empty or labels_df.empty:
        return pd.DataFrame()
    merged = alert_df.merge(labels_df, on=["run_id", "milestone"], how="inner", suffixes=("", "_label"))
    merged["lead_episodes"] = merged["stagnation_episode"] - merged["alert_episode"]
    return merged


def summary_metrics(lead_df: pd.DataFrame) -> dict:
    if lead_df.empty:
        return {"mean_lead": None, "median_lead": None, "fp_rate": None}
    return {
        "mean_lead": float(lead_df["lead_episodes"].mean()),
        "median_lead": float(lead_df["lead_episodes"].median()),
        "fp_rate": float((lead_df.get("alert", 0).astype(bool) & ~lead_df.get("stagnation", True).astype(bool)).mean()),
    }
