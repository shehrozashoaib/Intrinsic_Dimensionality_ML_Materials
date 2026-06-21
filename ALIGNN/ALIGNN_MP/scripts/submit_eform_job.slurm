#!/bin/bash --login
# Parametrized one-GPU job. The dims/seeds to run are passed in via the
# EFORM_RUNS environment variable (set by scripts/submit_eform_all.sh).
# The #SBATCH values below are fallback defaults; submit_eform_all.sh overrides
# --time, --job-name, and the log paths per job on the sbatch command line.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=09:00:00
#SBATCH --job-name=eform_job
#SBATCH --output=logs/eform_job_%j.out
#SBATCH --error=logs/eform_job_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)"
echo "Runs: ${EFORM_RUNS:-<none>}"
echo "Dataset: ${EFORM_DATASET:-MP_json_eform}"
echo "Epochs: ${EFORM_EPOCHS:-350}"
echo "ALIGNN_DETERMINISTIC: ${ALIGNN_DETERMINISTIC}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_alignn_fastfood_eform_sweep.sh

echo "End: $(date -Iseconds)"
