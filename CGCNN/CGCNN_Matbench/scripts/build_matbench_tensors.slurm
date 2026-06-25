#!/bin/bash --login
# Build CGCNN Matbench tensor caches: download+split (prepare) then materialize tensors.
# Usage: sbatch --export=ALL,MB_TASKS="matbench_dielectric",MB_SEED=123 scripts/build_matbench_tensors.slurm
#SBATCH --job-name=mb_build
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mb_build_%j.out
#SBATCH --error=logs/mb_build_%j.err

set -euo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASKS="${MB_TASKS:-matbench_dielectric}"
SEED="${MB_SEED:-123}"
echo "Node: $(hostname)  tasks=[$TASKS] seed=$SEED  start=$(date -Iseconds)"

echo "[1/2] prepare_matbench_splits (download + 80/10/10 split)"
python prepare_matbench_splits.py --output-dir cached_matbench/raw --tasks $TASKS --seed "$SEED"

echo "[2/2] materialize_cgcnn_tensors (graph tensors)"
python materialize_cgcnn_tensors.py \
  --raw-dir cached_matbench/raw \
  --output-dir cached_matbench/tensors \
  --root-dir root_dir \
  --tasks $TASKS --seed "$SEED"

echo "=== built tensor caches ==="
for t in $TASKS; do ls -la "cached_matbench/tensors/$t/seed${SEED}/" 2>/dev/null; done
echo "done=$(date -Iseconds)"
