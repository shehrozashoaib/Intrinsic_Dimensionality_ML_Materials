#!/bin/bash --login
# One-GPU CGCNN fastfood job; one array task = one (dim, seed) run for 350 epochs.
# Resources / array / walltime are set by submit_cgcnn_mp_all.sh on the sbatch line.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=cgcnn_mp_sweep
#SBATCH --output=logs/cgcnn_mp_sweep_%A_%a.out
#SBATCH --error=logs/cgcnn_mp_sweep_%A_%a.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312

echo "[INFO] host=$(hostname) task=${CGCNN_TASK} array_id=${SLURM_ARRAY_TASK_ID} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export CGCNN_EPOCHS="${CGCNN_EPOCHS:-350}"
export CGCNN_BS="${CGCNN_BS:-256}"
export CGCNN_WORKERS="${CGCNN_WORKERS:-8}"

bash scripts/run_cgcnn_fastfood_mp_sweep.sh
echo "[INFO] done=$(date -Iseconds)"
