#!/bin/bash --login
# One-GPU ALIGNN matbench log_kvrh FULL-dataset job (Exp 1).
# KVRH_RUNS (and optional KVRH_EPOCHS) are passed via --export by the submitter.
# #SBATCH values are fallback defaults; the submitter overrides --time/--job-name/logs.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=alignn_kvrh_job
#SBATCH --output=logs/alignn_kvrh_job_%j.out
#SBATCH --error=logs/alignn_kvrh_job_%j.err

set -uo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)"
echo "KVRH_RUNS: ${KVRH_RUNS:-<none>}  KVRH_EPOCHS: ${KVRH_EPOCHS:-120}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_alignn_fastfood_log_kvrh_sweep.sh

echo "End: $(date -Iseconds)"
