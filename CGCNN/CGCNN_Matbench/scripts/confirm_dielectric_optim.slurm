#!/bin/bash --login
# Confirmation test: with Adam + Fastfood, does dielectric test error CHANGE between
# id_dim 100% (1.0) and 1% (0.01) at 10 epochs? (Validates intrinsic-dim mechanism vs the
# frozen-SGD bug.) One job, two short runs.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=cgcnn_diel_confirm
#SBATCH --output=logs/cgcnn_diel_confirm_%j.out
#SBATCH --error=logs/cgcnn_diel_confirm_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASK=matbench_dielectric
ROOT=results_${TASK}_fastfood/_confirm_adam_ep10
mkdir -p "$ROOT"
echo "Node: $(hostname)  start=$(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader

for spec in "1.0:100" "0.01:1"; do
  FRAC=${spec%%:*}; PCT=${spec##*:}
  OUT="$ROOT/cgcnn_${TASK}_fastfood_dim${PCT}_dataseed123_modelseed123_ep10"
  mkdir -p "$OUT"
  echo "===== id_dim ${FRAC} (${PCT}%) ====="
  python main.py \
    --cached-data-dir cached_matbench/tensors \
    --matbench-task "$TASK" \
    --output-dir "$OUT" \
    --epochs 10 \
    --batch-size 512 \
    --optim Adam \
    --n-conv 4 --atom-fea-len 56 --h-fea-len 64 --n-h 3 \
    --subspace-method fastfood \
    --id-dim "$FRAC" \
    --data-seed 123 --random-seed 123 \
    --print-freq 1000 2>&1 | tee "$OUT/train.log"
  rm -f "$OUT"/*.pth.tar
done

echo; echo "==== CONFIRMATION SUMMARY (Adam, 10 epochs, dielectric) ===="
for d in "$ROOT"/*dim100* "$ROOT"/*dim1_*; do
  [ -d "$d" ] || continue
  val=$(grep -oE '\* MAE [-0-9.eE]+' "$d/train.log" | tail -1)
  test=$(grep -oE 'test metric is:\s*[-0-9.eE]+' "$d/train.log" | tail -1)
  echo "  $(basename $d): best_val=[$val]  $test"
done
echo "done=$(date -Iseconds)"
