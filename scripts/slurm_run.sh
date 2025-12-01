#!/usr/bin/env bash
# Slurm launcher for headless Pokemon Red DQN training (5 x 100k steps).

#SBATCH --job-name=poke-dqn
#SBATCH --partition=gpu        # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -A engr-class-any      # set account (adjust if you have a lab account)
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

cd ~/Final
source .venv/bin/activate

# Unique save dir per job (fallback to "manual" if run without sbatch).
JOB_ID=${SLURM_JOB_ID:-manual}
SAVE_DIR="checkpoints/slurm_run_${JOB_ID}"
mkdir -p "$SAVE_DIR" logs

PYTHONUNBUFFERED=1 python -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
  --config epsilon/pokemon_rl/configs/training_config.json \
  --episodes 5 \
  --max-steps 100000 \
  --learning-starts 200 \
  --train-frequency 4 \
  --batch-size 32 \
  --save-dir "$SAVE_DIR" \
  --num-envs 1 \
  --headless --no-render-map --no-gameplay-grid --display-envs 0 --no-pyboy-window \
  --device cuda \
  --log-interval 10000 \
  --progress-interval 1 \
  --perf-logging-enabled
