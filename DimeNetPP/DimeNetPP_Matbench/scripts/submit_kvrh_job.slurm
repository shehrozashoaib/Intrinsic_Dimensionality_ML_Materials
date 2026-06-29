#!/bin/bash --login
# One-GPU DimeNet++ matbench log_kvrh job: runs the FULL 12-dim x 3-seed Fastfood sweep
# for a single layer count (NUM_BLOCKS), bundled into one job (fewer requests, more GPU each).
# Submit e.g.:
#   sbatch --export=ALL,NUM_BLOCKS=2,DIMENET_EPOCHS=120 scripts/submit_kvrh_job.slurm
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=dim_kvrh_job
#SBATCH --output=logs/dim_kvrh_job_%j.out
#SBATCH --error=logs/dim_kvrh_job_%j.err

set -uo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "NUM_BLOCKS: ${NUM_BLOCKS:-1}"
echo "Epochs: ${DIMENET_EPOCHS:-120}"
echo "DIMENET_RUNS: ${DIMENET_RUNS:-<full sweep>}"
echo "Cache root: ${DIMENET_CACHE_ROOT:-cached_tensors_dimenetpp_kvrh}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_dimenetpp_fastfood_kvrh_sweep.sh

echo "End: $(date -Iseconds)"
