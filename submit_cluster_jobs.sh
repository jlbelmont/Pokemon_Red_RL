#!/usr/bin/env bash
# Usage: ./submit_cluster_jobs.sh [JOBS]
set -euo pipefail

LOCAL_DIR="${HOME}/Downloads/College/MS/DRL/Final"
REMOTE_USER="belmont"
REMOTE_HOST="shell.engr.wustl.edu"
REMOTE_DIR="~/projects/Final"
JOB_COUNT="${1:-4}"

clean_path() {
  # Expand ~ and ensure trailing slash
  local path="$1"
  eval echo "${path%/}/"
}

LOCAL_CLEAN=$(clean_path "$LOCAL_DIR")

if [[ ! -d "$LOCAL_CLEAN" ]]; then
  echo "[error] Local directory $LOCAL_CLEAN not found" >&2
  exit 1
fi

echo "[sync] Mirroring local repo to cluster..."
rsync -av --delete "${LOCAL_CLEAN}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash <<'INNER'
set -euo pipefail
cd ~/projects/Final
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate poke
chmod +x cluster_train.sh slurm_train.sh
mkdir -p logs

JOB_COUNT=${JOB_COUNT}

echo "[sbatch] Submitting jobs..."
for i in $(seq 1 ${JOB_COUNT}); do
  sbatch slurm_train.sh
  sleep 1
fi

echo "[squeue] Current queue:"
squeue -u $USER

echo "[logs] Tail of newest job output (if any):"
LOG=$(ls -t poke-*.out 2>/dev/null | head -n 1 || true)
if [[ -n "$LOG" ]]; then
  tail -n 40 "$LOG"
else
  echo "No logs yet."
fi
INNER
