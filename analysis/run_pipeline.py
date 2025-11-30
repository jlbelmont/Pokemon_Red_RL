"""End-to-end offline analysis skeleton."""
from __future__ import annotations

from pathlib import Path

from analysis import config, io, posterior, rewards, evaluation


def main() -> None:
    progress_df = io.load_all_progress(config.DATA_ROOT)
    summary_df = io.load_summary_csv(config.SUMMARY_CSV)
    print(f"Loaded {len(progress_df)} progress rows and {len(summary_df)} summary rows.")

    posterior_df = posterior.compute_posteriors(progress_df)
    alert_df = posterior.attach_alerts(posterior_df, config.MILESTONES)
    print(f"Computed {len(posterior_df)} posterior summaries.")

    # Placeholder labels: user should populate from annotated CSV.
    labels_path = Path("analysis/labels.csv")
    labels_df = io.load_summary_csv(labels_path) if labels_path.exists() else summary_df.head(0)
    lead_df = evaluation.compute_lead_time(alert_df, labels_df)
    print("Lead-time metrics:", evaluation.summary_metrics(lead_df))

    intrinsic_df = summary_df[config.INTRINSIC_COLUMNS] if not summary_df.empty else summary_df
    hyperparams = rewards.fit_gamma_hyperparams(posterior_df, intrinsic_df)
    print("Reward hyperparams:", hyperparams)


if __name__ == "__main__":
    main()
