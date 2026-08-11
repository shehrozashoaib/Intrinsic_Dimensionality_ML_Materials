#!/bin/bash --login
# CGCNN MP band-gap CUSTOM run: 1 conv layer (85,679 params), fastfood, 350 epochs,
# dims 100/60/40% x seeds {253,768,1653} (data_seed=model_seed). Runs all 9 in one job (no array).
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=cgcnn_bandgap_custom
#SBATCH --output=logs/cgcnn_bandgap_custom_%j.out
#SBATCH --error=logs/cgcnn_bandgap_custom_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312
echo "[INFO] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CGCNN_EPOCHS="${CGCNN_EPOCHS:-350}"
export CGCNN_BS="${CGCNN_BS:-256}"
export CGCNN_WORKERS="${CGCNN_WORKERS:-8}"
bash scripts/run_cgcnn_bandgap_custom_sweep.sh
echo "[INFO] done=$(date -Iseconds)"
