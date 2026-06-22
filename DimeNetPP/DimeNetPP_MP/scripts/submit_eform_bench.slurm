#!/bin/bash --login
# One-epoch full-model (100%) timing job. GPU type is chosen by the submitter via
# --constraint on the sbatch command line (see submit_eform_bench.sh).
# This doubles as the "full model, one epoch, 32GB card" sanity test.
#SBATCH --job-name=dim_eform_bench
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dim_eform_bench_%j.out
#SBATCH --error=logs/dim_eform_bench_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

export DIMENET_EPOCHS="${DIMENET_BENCH_EPOCHS:-1}"
export DIMENET_RUNS="${DIMENET_BENCH_RUN:-1.0:100:123:1123}"
export DIMENET_RESULT_ROOT="${DIMENET_RESULT_ROOT:-results_dimenetpp_eform_bench}"

echo "Node: $(hostname)"
echo "Bench run: ${DIMENET_RUNS}  epochs=${DIMENET_EPOCHS}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

bash scripts/run_dimenetpp_fastfood_eform_sweep.sh

echo "End: $(date -Iseconds)"
