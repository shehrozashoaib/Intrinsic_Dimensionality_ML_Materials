#!/bin/bash --login
# Feasibility smoke: can DimeNet++ log_kvrh compile + train at num_blocks=2 and =4 on a V100,
# with Fastfood? Runs 2 epochs each and records peak GPU memory (memory_growth is on, so
# nvidia-smi memory.used is the real footprint).
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=dim_kvrh_smoke
#SBATCH --output=logs/dim_kvrh_smoke_%j.out
#SBATCH --error=logs/dim_kvrh_smoke_%j.err

set -uo pipefail
mkdir -p logs
module purge
source scripts/dimenet_env.sh

CODE_DIR="dimenetpp_code_only"
CACHE_DIR="cached_tensors_dimenetpp_kvrh/splitseed1123"
SMOKE_ROOT="results_dimenetpp_log_kvrh_smoke"
mkdir -p "$SMOKE_ROOT"

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date -Iseconds)"

for NB in 2 4; do
  echo "==================== num_blocks=$NB ===================="
  memlog="${SMOKE_ROOT}/mem_nb${NB}.log"; : > "$memlog"
  # background GPU memory poller
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$memlog"; sleep 2; done ) &
  POLL=$!
  t0=$(date +%s)
  "$DIMENET_PYTHON" "${CODE_DIR}/dimenet_run_kvrh_v3.py" \
    --method fastfood --id_dim 0.5 --num_blocks "$NB" \
    --epochs 2 --batch_size 64 --seed 123 --split_seed 1123 --gpu 0 \
    --cache_dir "$CACHE_DIR" \
    --out_dir "${SMOKE_ROOT}/nb${NB}" > "${SMOKE_ROOT}/smoke_nb${NB}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  kill "$POLL" 2>/dev/null
  peak=$(sort -n "$memlog" | tail -1)
  echo "RESULT nb=$NB exit_code=$rc elapsed_2ep=$((t1-t0))s peak_gpu_mem=${peak} MiB"
  echo "--- key lines ---"
  grep -E 'D:|d:|Trainable|Test MAE|Validation MAE|num_blocks|Total params|wall time|Error|Traceback|ResourceExhausted|OOM' "${SMOKE_ROOT}/smoke_nb${NB}.log" | head -25
  echo
done

echo "End: $(date -Iseconds)"
