#!/usr/bin/env bash

# Cluster-friendly training harness for epsilon/pokemon RL agent.
# - Runs headless with no GUI dependencies.
# - Safely resumes from the newest checkpoint in checkpoints/.
# - Designed for 4-hour chunks; just resubmit this script when a job ends.

set -euo pipefail

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${PROJECT_DIR}/epsilon/pokemon_rl/configs/training_config.json"
CHECKPOINT_DIR="${PROJECT_DIR}/epsilon/pokemon_rl/checkpoints"

cd "$PROJECT_DIR"

mkdir -p "$CHECKPOINT_DIR"

# Run headless training with two parallel environments.
python3 -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
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
  --replay-buffer-path "$CHECKPOINT_DIR/replay_buffer.pt"

echo "[cluster_train] Training chunk completed."
