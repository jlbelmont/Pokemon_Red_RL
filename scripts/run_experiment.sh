#!/usr/bin/env bash

# Helper for launching a tagged training run and archiving the outputs.
# Usage: scripts/run_experiment.sh RUN_ID

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 1
fi

RUN_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="$ROOT_DIR/checkpoints/$RUN_ID"
LOG_DIR="$ROOT_DIR/logs"
CURRICULUM_DIR="$ROOT_DIR/checkpoints/curriculum_states"
BEST_DIR="$ROOT_DIR/checkpoints/best_models"

mkdir -p "$SAVE_DIR" "$LOG_DIR" "$CURRICULUM_DIR" "$BEST_DIR"

TRAIN_SUMMARY="$LOG_DIR/train_summary_${RUN_ID}.csv"
CURRIC_LOG="$LOG_DIR/curriculum_events_${RUN_ID}.csv"

echo "[run_experiment] Launching training run $RUN_ID..."
caffeinate -dimsu bash -lc "
  cd \"$ROOT_DIR\" && \
  PYTHONUNBUFFERED=1 .venv/bin/python -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
    --config epsilon/pokemon_rl/configs/training_config.json \
    --save-dir \"$SAVE_DIR\" \
    --summary-log-path \"$TRAIN_SUMMARY\" \
    --curriculum-events-log-path \"$CURRIC_LOG\" \
    --replay-buffer-path \"$SAVE_DIR/replay_buffer.pt\" \
    --headless --no-pyboy-window --render-map --no-show-env-maps --no-gameplay-grid --display-envs 0 \
    --device mps \
    --log-interval 1000 \
    --progress-interval 3 \
    --perf-logging-enabled
"

echo "[run_experiment] Archiving artifacts..."
ARCHIVE_DIR="$ROOT_DIR/archives/$RUN_ID"
mkdir -p "$ARCHIVE_DIR"
RUN_ARTIFACT_DIR="$ROOT_DIR/artifacts/runs/$RUN_ID"
mkdir -p "$RUN_ARTIFACT_DIR"

PROGRESS_JSON="$SAVE_DIR/progress_metrics.json"
if [[ -f "$PROGRESS_JSON" ]]; then
  cp "$PROGRESS_JSON" "$ARCHIVE_DIR/progress_metrics.json"
  cp "$PROGRESS_JSON" "$RUN_ARTIFACT_DIR/progress_metrics.json"
fi

REWARD_METRICS="$SAVE_DIR/reward_metrics.json"
if [[ -f "$REWARD_METRICS" ]]; then
  cp "$REWARD_METRICS" "$ARCHIVE_DIR/reward_metrics.json"
fi

if [[ -f "$TRAIN_SUMMARY" ]]; then
  cp "$TRAIN_SUMMARY" "$ARCHIVE_DIR/"
fi
if [[ -f "$CURRIC_LOG" ]]; then
  cp "$CURRIC_LOG" "$ARCHIVE_DIR/"
fi

if [[ -f "$SAVE_DIR/dqn_route1_best.pt" ]]; then
  cp "$SAVE_DIR/dqn_route1_best.pt" "$BEST_DIR/${RUN_ID}.pt"
fi

echo "[run_experiment] Regenerating analysis figures..."
cd "$ROOT_DIR"
PYTHONPATH=. .venv/bin/python analysis/run_pipeline.py
if [[ -f "$ARCHIVE_DIR/progress_metrics.json" ]]; then
  PYTHONPATH=. .venv/bin/python analysis/reward_success.py \
    --log "$CURRIC_LOG" \
    --csv-output artifacts/reward_success_timeseries.csv \
    --figure-output figures/reward_success_timeseries.pdf || true
fi

echo "[run_experiment] Run $RUN_ID complete."
