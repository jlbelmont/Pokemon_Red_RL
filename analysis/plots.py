"""Visualization stubs."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_posterior_traces(df: pd.DataFrame, milestone: str) -> None:
    subset = df[df["milestone"] == milestone]
    if subset.empty:
        print(f"No data for milestone {milestone}")
        return
    plt.figure(figsize=(8, 4))
    for run_id, run_df in subset.groupby("run_id"):
        plt.plot(run_df["episode"], run_df["posterior_mean"], label=run_id)
    plt.title(f"Posterior mean over episodes ({milestone})")
    plt.xlabel("Episode")
    plt.ylabel("Posterior mean")
    plt.legend()
    plt.tight_layout()
    plt.show()
