#!/usr/bin/env bash

# Convenience launcher for watching the latest checkpoint locally with
# Matplotlib aggregate map + gameplay grid (PyBoy hidden by default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_DIR}/.venv"
CONFIG_PATH="${PROJECT_DIR}/epsilon/pokemon_rl/configs/training_config.json"
CHECKPOINT_DIR="${PROJECT_DIR}/epsilon/pokemon_rl/checkpoints"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[watch_local] Missing virtual environment at $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$PROJECT_DIR"

BEST_CKPT="$(ls -t "$CHECKPOINT_DIR"/dqn_route1_best.pt 2>/dev/null | head -n 1 || true)"
if [[ -n "$BEST_CKPT" ]]; then
  TARGET_CKPT="$BEST_CKPT"
else
  TARGET_CKPT="$(ls -t "$CHECKPOINT_DIR"/dqn_route1_*.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$TARGET_CKPT" ]]; then
  echo "[watch_local] No checkpoint available in $CHECKPOINT_DIR" >&2
  exit 1
fi

python3 -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
  --config "$CONFIG_PATH" \
  --watch-only \
  --replay-checkpoint "$TARGET_CKPT" \
  --num-envs 1 \
  --render-map \
  --gameplay-grid \
  --display-envs 0 \
  --no-show-env-maps
