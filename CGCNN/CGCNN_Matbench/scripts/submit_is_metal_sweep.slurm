#!/bin/bash --login
# CGCNN matbench_mp_is_metal classification sweep: 8 dims x 4 seeds = 32 runs.
# One array task = one (dim,seed) run. Submit from CGCNN_Matbench/:
#   sbatch --array=0-31 scripts/submit_is_metal_sweep.slurm
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=cgcnn_is_metal
#SBATCH --output=logs/cgcnn_is_metal_%A_%a.out
#SBATCH --error=logs/cgcnn_is_metal_%A_%a.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312
echo "[INFO] host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID:-none} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CGCNN_EPOCHS="${CGCNN_EPOCHS:-350}"
export CGCNN_BS="${CGCNN_BS:-256}"
export CGCNN_WORKERS="${CGCNN_WORKERS:-8}"
bash scripts/run_cgcnn_is_metal_sweep.sh
echo "[INFO] done=$(date -Iseconds)"
