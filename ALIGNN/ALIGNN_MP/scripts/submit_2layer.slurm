#!/bin/bash --login
# ALIGNN doubled-layer (alignn_layers=2, gcn_layers=2) full-parameter-space run.
# ALIGNN_TASK / ALIGNN_SEEDS set by submitter.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name=alignn_2layer
#SBATCH --output=logs/alignn_2layer_%x_%j.out
#SBATCH --error=logs/alignn_2layer_%x_%j.err

set -uo pipefail
mkdir -p logs
module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "[INFO] host=$(hostname) task=${ALIGNN_TASK:-<none>} seeds=${ALIGNN_SEEDS:-<none>} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/run_alignn_2layer_full.sh
echo "[INFO] done=$(date -Iseconds)"
