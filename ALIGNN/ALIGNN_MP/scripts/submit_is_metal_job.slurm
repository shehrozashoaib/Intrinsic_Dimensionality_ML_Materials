#!/bin/bash --login
# One-GPU ALIGNN matbench_mp_is_metal FULL-dataset job (binary classification).
# IS_METAL_RUNS (and optional IS_METAL_EPOCHS) are passed via --export by the
# submitter. #SBATCH values are fallback defaults; the submitter overrides
# --time/--job-name/logs.
#
# Job-array usage (one run = one dim+seed of the 32-run grid):
#   sbatch --array=0-31 scripts/submit_is_metal_job.slurm
# With --array set, the runner selects RUNS[$SLURM_ARRAY_TASK_ID] automatically.
# Alternatively pass an explicit run via --export=ALL,IS_METAL_RUNS="1.0:100:101:1101".
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=alignn_is_metal_job
#SBATCH --output=logs/alignn_is_metal_job_%j.out
#SBATCH --error=logs/alignn_is_metal_job_%j.err

set -uo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)"
echo "IS_METAL_RUNS: ${IS_METAL_RUNS:-<none>}  IS_METAL_EPOCHS: ${IS_METAL_EPOCHS:-350}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_alignn_fastfood_is_metal_sweep.sh

echo "End: $(date -Iseconds)"
