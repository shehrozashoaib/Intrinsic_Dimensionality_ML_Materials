#!/bin/bash --login
# CGCNN doubled-layer (n_conv=2) full-parameter-space run. CGCNN_TASK / CGCNN_SEEDS set by submitter.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=cgcnn_2layer
#SBATCH --output=logs/cgcnn_2layer_%x_%j.out
#SBATCH --error=logs/cgcnn_2layer_%x_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312
echo "[INFO] host=$(hostname) task=${CGCNN_TASK:-<none>} seeds=${CGCNN_SEEDS:-<none>} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/run_cgcnn_2layer_full.sh
echo "[INFO] done=$(date -Iseconds)"
