#!/bin/bash --login
# DimeNet++ doubled-layer (num_blocks=2, clipnorm=1.0) full-parameter-space run (eform/bandgap).
# DIMENET_TASK / DIMENET_SEEDS set by submitter.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=dim_2layer_mp
#SBATCH --output=logs/dim_2layer_mp_%x_%j.out
#SBATCH --error=logs/dim_2layer_mp_%x_%j.err

set -uo pipefail
mkdir -p logs
module purge
source scripts/dimenet_env.sh

echo "[INFO] host=$(hostname) task=${DIMENET_TASK:-<none>} seeds=${DIMENET_SEEDS:-<none>} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/run_dimenetpp_2layer_mp.sh
echo "[INFO] done=$(date -Iseconds)"
