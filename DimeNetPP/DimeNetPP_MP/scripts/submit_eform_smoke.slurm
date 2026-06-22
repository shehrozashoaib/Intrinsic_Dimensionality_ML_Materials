#!/bin/bash --login
# Full-data timing smoke for DimeNet++ eform. Uses the real tensor cache and
# runs enough epochs to expose model-build/startup cost separately from epoch cost.
#SBATCH --job-name=dim_eform_smoke
#SBATCH --gpus=1
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dim_eform_smoke_%j.out
#SBATCH --error=logs/dim_eform_smoke_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

export DIMENET_EPOCHS="${DIMENET_SMOKE_EPOCHS:-20}"
export DIMENET_RUNS="${DIMENET_SMOKE_RUN:-1.0:100:123:1123}"

echo "Node: $(hostname)"
echo "Smoke run: ${DIMENET_RUNS}"
echo "Epochs: ${DIMENET_EPOCHS}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_fastfood_eform_sweep.sh

echo "End: $(date -Iseconds)"
