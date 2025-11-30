#!/usr/bin/env bash

# Cluster-friendly training harness for epsilon/pokemon RL agent.
# - Runs headless with no GUI dependencies.
# - Safely resumes from the newest checkpoint in checkpoints/.
# - Designed for 4-hour chunks; just resubmit this script when a job ends.

set -euo pipefail

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${PROJECT_DIR}/epsilon/pokemon_rl/training_config.json"
CHECKPOINT_DIR="${PROJECT_DIR}/epsilon/pokemon_rl/checkpoints"

cd "$PROJECT_DIR"

mkdir -p "$CHECKPOINT_DIR"

# Pick the newest checkpoint (if any) so we can resume training seamlessly.
LATEST_CKPT="$(ls -t "$CHECKPOINT_DIR"/dqn_route1_*.pt 2>/dev/null | head -n 1 || true)"

RESUME_FLAGS=()
if [[ -n "$LATEST_CKPT" ]]; then
  echo "[cluster_train] Resuming from $LATEST_CKPT"
  RESUME_FLAGS+=("--replay-checkpoint" "$LATEST_CKPT")
else
  echo "[cluster_train] No checkpoint found; starting fresh."
fi

# Run headless training with two parallel environments.
python3 epsilon/pokemon_rl/minimal_epsilon_setup.py \
  --config "$CONFIG_PATH" \
  --num-envs 2 \
  --headless \
  --no-render-map \
  --no-gameplay-grid \
  --display-envs 0 \
  --no-pyboy-window \
  --save-every 1 \
  --auto-save-minutes 5 \
  --log-interval 200 \
  --progress-interval 50 \
  "${RESUME_FLAGS[@]}"

echo "[cluster_train] Training chunk completed."
