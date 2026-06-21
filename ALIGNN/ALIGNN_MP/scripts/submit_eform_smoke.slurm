#!/bin/bash --login
# Smoke test: ONE run, 3 epochs, on a tiny dataset, on a real A100.
# Validates the whole pipeline (train -> prediction CSV -> metadata.json) and
# gives you a real per-run time before committing the 9/16/24h wall numbers.
# Build the tiny dataset first (see runbook), then:
#   sbatch scripts/submit_eform_smoke.slurm
#SBATCH --job-name=eform_smoke
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eform_smoke_%j.out
#SBATCH --error=logs/eform_smoke_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)  Start: $(date -Iseconds)"
echo "Dataset: ${EFORM_DATASET:-MP_json_eform_smoke}"
echo "Epochs: ${EFORM_EPOCHS:-3}"
echo "ALIGNN_DETERMINISTIC: ${ALIGNN_DETERMINISTIC}"
nvidia-smi

# tiny dataset + 3 epochs + one run, through the real sweep machinery
export EFORM_DATASET=MP_json_eform_smoke
export EFORM_EPOCHS=3
export EFORM_RUNS="1.0:100:123"
bash scripts/run_alignn_fastfood_eform_sweep.sh

echo "End: $(date -Iseconds)"