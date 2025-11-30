"""Hierarchical reward calibration scaffolding."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import config


def fit_gamma_hyperparams(posterior_df: pd.DataFrame, intrinsic_df: pd.DataFrame) -> Dict[str, dict]:
    """Placeholder fit mapping milestone posterior -> Gamma params."""
    hyperparams: Dict[str, dict] = {}
    if posterior_df.empty or intrinsic_df.empty:
        return hyperparams
    grouped = posterior_df.groupby("milestone")
    for milestone, group in grouped:
        mean = group["posterior_mean"].mean()
        # Use simple method-of-moments style placeholder for now.
        kappa = max(1.0, 2.0 * (1.0 - mean))
        theta = max(0.1, 0.5 * (1.0 - mean))
        hyperparams[milestone] = {"kappa": kappa, "theta": theta}
    return hyperparams


def schedule_scale(posterior_mean: float, bounds: tuple[float, float]) -> float:
    """Map posterior mean → exploration scale (lower belief => higher scale)."""
    lo, hi = bounds
    posterior_mean = np.clip(posterior_mean, 0.0, 1.0)
    return float(lo + (hi - lo) * (1.0 - posterior_mean))
