#!/bin/bash --login
# Parametrized one-GPU DimeNet++ is_metal (binary classification) job. Launch one run per array
# index by passing DIMENET_RUNS (or via SLURM_ARRAY_TASK_ID indexing into the sweep RUNS grid).
# V100 constraint: DimeNet++ is TensorFlow (pydimnet env, num_blocks=1 peaks ~4.7 GiB GPU),
# matching all other DimeNet++ submits (eform/bandgap/kvrh/dielectric/phonons). Only CGCNN/ALIGNN
# (torch cu128, no sm_70) require A100. V100 is far less contended -> jobs start sooner.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=dim_is_metal_job
#SBATCH --output=logs/dim_is_metal_job_%j.out
#SBATCH --error=logs/dim_is_metal_job_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "Runs: ${DIMENET_RUNS:-<none>}"
echo "Epochs: ${DIMENET_EPOCHS:-350}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Cache root: ${DIMENET_CACHE_ROOT:-cached_tensors_dimenetpp_is_metal}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_fastfood_is_metal_sweep.sh

echo "End: $(date -Iseconds)"
