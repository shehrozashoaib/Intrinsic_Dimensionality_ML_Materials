#!/bin/bash --login
# Build FIXED-TEST 2-PARTITION caches (quarter_1/2, tenth_1/2) for the one-cycle sweep.
# One array task = one task. eform/bandgap load a ~26 GB pickle per seed, hence the big --mem.
#
#   sbatch --array=0   build_fixedtest2.slurm     # log_kvrh  (fast, run this first)
#   sbatch --array=1-2 build_fixedtest2.slurm     # eform, bandgap
#SBATCH --job-name=dim_fixedtest2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dim_fixedtest2_%A_%a.out
#SBATCH --error=logs/dim_fixedtest2_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TASKS=(log_kvrh eform bandgap)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# The DimeNet++ TensorFlow env (py312 has no tensorflow).
PY="${DIMENET_PYTHON:-$SLURM_SUBMIT_DIR/DimeNetPP_MP/envs/pydimnet/bin/python}"
export CUDA_VISIBLE_DEVICES=""          # CPU only: this just slices cached tensors

echo "[INFO] task=$TASK  start=$(date -Iseconds)  host=$(hostname)  mem=${SLURM_MEM_PER_NODE:-?}M"
"$PY" build_fixedtest_partition_caches.py "$TASK"
echo "[INFO] done=$(date -Iseconds)"
