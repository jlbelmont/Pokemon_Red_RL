"""Posterior utilities for milestone success monitoring."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta

from . import config


def beta_update(successes: float, trials: float, alpha0: float = 1.0, beta0: float = 1.0) -> dict:
    """Return mean + 95% CI for a Beta posterior."""
    alpha = alpha0 + successes
    beta_param = beta0 + max(0.0, trials - successes)
    mean = alpha / (alpha + beta_param) if (alpha + beta_param) else float("nan")
    ci_low, ci_high = beta.ppf([0.025, 0.975], alpha, beta_param)
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "alpha": alpha, "beta": beta_param}


def compute_posteriors(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw progress rows into Beta summaries per milestone/run."""
    if frame.empty:
        return pd.DataFrame(columns=["run_id", "milestone", "successes", "trials", "posterior_mean", "ci_low", "ci_high"])
    if "milestone" not in frame.columns and "name" in frame.columns:
        frame = frame.rename(columns={"name": "milestone"})
    if "success" not in frame.columns:
        cols = [
            "run_id",
            "milestone",
            "successes",
            "trials",
            "posterior_mean",
            "ci_lower",
            "ci_upper",
            "decision_threshold",
            "decision",
        ]
        available = [c for c in cols if c in frame.columns]
        simplified = frame[available].copy()
        simplified.rename(columns={"ci_lower": "ci_low", "ci_upper": "ci_high"}, inplace=True)
        return simplified

    grouped = frame.groupby(["run_id", "milestone"], dropna=False)
    rows = []
    for (run_id, milestone), group in grouped:
        successes = group["success"].sum()
        trials = group["success"].count()
        stats = beta_update(successes, trials)
        rows.append(
            {
                "run_id": run_id,
                "milestone": milestone,
                "successes": successes,
                "trials": trials,
                "posterior_mean": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def attach_alerts(posterior_df: pd.DataFrame, milestones: Iterable[config.MilestoneConfig]) -> pd.DataFrame:
    """Add alert booleans by comparing posterior means to thresholds."""
    if posterior_df.empty:
        return posterior_df.copy()
    cfg_df = pd.DataFrame([m.__dict__ for m in milestones])
    merged = posterior_df.merge(cfg_df, left_on="milestone", right_on="name", how="left")
    merged["alert"] = merged["posterior_mean"] < merged["alert_threshold"]
    merged.rename(columns={"step_budget": "tm_step_budget"}, inplace=True)
    return merged
