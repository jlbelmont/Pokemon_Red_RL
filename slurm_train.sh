#!/usr/bin/env bash
#SBATCH -p gpu-linuxlab
#SBATCH -A engr-class-any
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -J poke-train
#SBATCH -o poke-%j.out

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate poke
cd "$HOME/projects/Final"

./cluster_train.sh
