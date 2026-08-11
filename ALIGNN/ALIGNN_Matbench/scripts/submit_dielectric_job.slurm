#!/bin/bash --login
# One-GPU ALIGNN matbench dielectric FULL-dataset job.
# DIEL_RUNS (and optional DIEL_EPOCHS) are passed via --export by the submitter.
# #SBATCH values are fallback defaults; the submitter overrides --time/--job-name/logs.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=alignn_diel_job
#SBATCH --output=logs/alignn_diel_job_%j.out
#SBATCH --error=logs/alignn_diel_job_%j.err

set -uo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)"
echo "DIEL_RUNS: ${DIEL_RUNS:-<none>}  DIEL_EPOCHS: ${DIEL_EPOCHS:-150}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_alignn_fastfood_dielectric_sweep.sh

echo "End: $(date -Iseconds)"
