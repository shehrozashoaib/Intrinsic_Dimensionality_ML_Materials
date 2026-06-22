#!/bin/bash --login
# Parametrized one-GPU DimeNet++ eform job. submit_eform_all.sh overrides these via the
# sbatch command line (array index -> run, V100 constraint, 64G). Defaults here match so the
# job also works when launched directly with DIMENET_RUNS set.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=dim_eform_job
#SBATCH --output=logs/dim_eform_job_%j.out
#SBATCH --error=logs/dim_eform_job_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "Runs: ${DIMENET_RUNS:-<none>}"
echo "Epochs: ${DIMENET_EPOCHS:-350}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Cache root: ${DIMENET_CACHE_ROOT:-cached_tensors_dimenetpp_eform}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_fastfood_eform_sweep.sh

echo "End: $(date -Iseconds)"
