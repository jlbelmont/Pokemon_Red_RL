# Posterior Snapshot (2025-11-30)

Artifacts captured after halting the 8-env parcel training run.

## Contents
- `local_train_20251130_020154.log`: raw stdout with reward-success blocks.
- `progress_metrics.json`: Bayesian milestone dump from `artifacts/runs/local_20251130/`.
- `reward_success_timeseries.csv`: parsed thresholds produced by `analysis/reward_success.py`.

## How to Reproduce Plots Later
```bash
cd /Users/jbelmont/Downloads/College/MS/DRL/Final
PYTHONPATH=. .venv/bin/python analysis/run_pipeline.py \
  --input archives/posterior_snapshot_20251130/progress_metrics.json
PYTHONPATH=. .venv/bin/python analysis/reward_success.py \
  --log archives/posterior_snapshot_20251130/local_train_20251130_020154.log \
  --csv-output artifacts/reward_success_timeseries.csv \
  --figure-output figures/reward_success_timeseries.pdf
```

You can also re-run LaTeX to embed the refreshed plots:
```bash
cd docs
latexmk -pdf rl_progress_report.tex
```

Keep this directory synced (e.g., commit or upload) before starting the next long run.
