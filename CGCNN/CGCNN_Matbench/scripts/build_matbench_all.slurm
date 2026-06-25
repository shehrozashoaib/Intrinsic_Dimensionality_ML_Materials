#!/bin/bash --login
# Build CGCNN Matbench tensor caches for all 3 tasks x 3 data-seeds, skipping any already built.
#SBATCH --job-name=mb_build_all
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mb_build_all_%j.out
#SBATCH --error=logs/mb_build_all_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASKS=(matbench_dielectric matbench_phonons matbench_log_kvrh)
SEEDS=(123 456 789)
echo "Node: $(hostname)  start=$(date -Iseconds)"

for SEED in "${SEEDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    out="cached_matbench/tensors/${TASK}/seed${SEED}/train.pt"
    if [[ -f "$out" ]]; then
      echo "[skip] $TASK seed$SEED already built"
      continue
    fi
    echo "[build] $TASK seed$SEED -- prepare"
    python prepare_matbench_splits.py --output-dir cached_matbench/raw --tasks "$TASK" --seed "$SEED" || { echo "PREPARE FAILED $TASK seed$SEED"; continue; }
    echo "[build] $TASK seed$SEED -- materialize"
    python materialize_cgcnn_tensors.py --raw-dir cached_matbench/raw --output-dir cached_matbench/tensors \
      --root-dir root_dir --tasks "$TASK" --seed "$SEED" || { echo "MATERIALIZE FAILED $TASK seed$SEED"; continue; }
  done
done

echo "=== final cache inventory ==="
for TASK in "${TASKS[@]}"; do for SEED in "${SEEDS[@]}"; do
  f="cached_matbench/tensors/${TASK}/seed${SEED}/train.pt"
  [[ -f "$f" ]] && echo "  OK   $TASK seed$SEED ($(du -h "$f" | cut -f1))" || echo "  MISS $TASK seed$SEED"
done; done
echo "done=$(date -Iseconds)"
