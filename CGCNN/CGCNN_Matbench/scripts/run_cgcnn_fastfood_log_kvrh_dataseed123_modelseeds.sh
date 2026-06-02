#!/usr/bin/env bash

# Run CGCNN Fastfood intrinsic-dimension sweep for matbench_log_kvrh.
# Run from this scripts/ directory or from the project root.
# Requires main.py patch adding --data-seed so cached split seed and model seed are separate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

CONDA_ENV="py312"
PYTHON="python"
CACHED_DATA_DIR="cached_matbench/tensors"
TASK="matbench_log_kvrh"
EPOCHS=120
BATCH_SIZE=512
N_CONV=4
ATOM_FEA_LEN=56
H_FEA_LEN=64
N_H=3
WRAPPER="fastfood"
DATA_SEED=123
MODEL_SEEDS=(123 456 789)
DIMS=(100 80 70 65 50 45 20 10 8 5 2 1)
PRINT_FREQ=1000

RESULT_ROOT="results_${TASK}_${WRAPPER}"
SUMMARY_CSV="${RESULT_ROOT}/${TASK}_${WRAPPER}_summary.csv"
mkdir -p "$RESULT_ROOT"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,wrapper,id_dim,data_seed,model_seed,epochs,batch_size,n_conv,atom_fea_len,h_fea_len,n_h,output_dir,predictions_csv,best_val_mae,test_mae,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

for DIM in "${DIMS[@]}"; do
  for MODEL_SEED in "${MODEL_SEEDS[@]}"; do
    EXP_NAME="cgcnn_${TASK}_${WRAPPER}_dim${DIM}_nconv${N_CONV}_atom${ATOM_FEA_LEN}_h${H_FEA_LEN}_nh${N_H}_dataseed${DATA_SEED}_modelseed${MODEL_SEED}"
    OUT_DIR="${RESULT_ROOT}/${EXP_NAME}"
    LOG_FILE="${OUT_DIR}/train.log"
    META_JSON="${OUT_DIR}/metadata.json"
    PRED_SRC="${OUT_DIR}/${TASK}_test_results.csv"
    PRED_DST="${OUT_DIR}/${EXP_NAME}_test_results.csv"
    mkdir -p "$OUT_DIR"

    START_TIME="$(date -Iseconds)"
    echo "[START] $EXP_NAME at $START_TIME"

    conda run -n "$CONDA_ENV" "$PYTHON" main.py \
      --cached-data-dir "$CACHED_DATA_DIR" \
      --matbench-task "$TASK" \
      --output-dir "$OUT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --n-conv "$N_CONV" \
      --atom-fea-len "$ATOM_FEA_LEN" \
      --h-fea-len "$H_FEA_LEN" \
      --n-h "$N_H" \
      --subspace-method "$WRAPPER" \
      --id-dim "$DIM" \
      --data-seed "$DATA_SEED" \
      --random-seed "$MODEL_SEED" \
      --print-freq "$PRINT_FREQ" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME="$(date -Iseconds)"

    STATUS="success"
    if [[ "$EXIT_CODE" -ne 0 ]]; then
      STATUS="failed"
    fi

    if [[ -f "$PRED_SRC" ]]; then
      mv -f "$PRED_SRC" "$PRED_DST"
    fi

    python - "$SUMMARY_CSV" "$META_JSON" "$LOG_FILE" "$PRED_DST" "$TASK" "$WRAPPER" "$DIM" "$DATA_SEED" "$MODEL_SEED" "$EPOCHS" "$BATCH_SIZE" "$N_CONV" "$ATOM_FEA_LEN" "$H_FEA_LEN" "$N_H" "$OUT_DIR" "$STATUS" "$EXIT_CODE" "$START_TIME" "$END_TIME" <<'INNERPY'
import csv
import json
import math
import re
import sys
from pathlib import Path

(summary_csv, meta_json, log_file, pred_csv, task, wrapper, dim, data_seed, model_seed,
 epochs, batch_size, n_conv, atom_fea_len, h_fea_len, n_h, out_dir, status, exit_code,
 start_time, end_time) = sys.argv[1:]

log_text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
val_matches = re.findall(r"^\s*\*\s+MAE\s+([-+0-9.eE]+)\s*$", log_text, flags=re.MULTILINE)
best_val_mae = min((float(x) for x in val_matches), default=math.nan)

test_mae = math.nan
m = re.search(r"test metric is:\s*([-+0-9.eE]+)", log_text)
if m:
    test_mae = float(m.group(1))
else:
    test_matches = re.findall(r"^\s*\*\*\s+MAE\s+([-+0-9.eE]+)\s*$", log_text, flags=re.MULTILINE)
    if test_matches:
        test_mae = float(test_matches[-1])

row = {
    "task": task,
    "wrapper": wrapper,
    "id_dim": int(dim),
    "data_seed": int(data_seed),
    "model_seed": int(model_seed),
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "n_conv": int(n_conv),
    "atom_fea_len": int(atom_fea_len),
    "h_fea_len": int(h_fea_len),
    "n_h": int(n_h),
    "output_dir": out_dir,
    "predictions_csv": pred_csv if Path(pred_csv).exists() else "",
    "best_val_mae": best_val_mae,
    "test_mae": test_mae,
    "status": status,
    "exit_code": int(exit_code),
    "start_time": start_time,
    "end_time": end_time,
}

Path(meta_json).write_text(json.dumps(row, indent=2))
fieldnames = list(row.keys())
with Path(summary_csv).open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(row)

print(f"[DONE] {Path(out_dir).name} status={status} exit_code={exit_code} best_val_mae={best_val_mae} test_mae={test_mae}")
INNERPY

    rm -f "$OUT_DIR"/*.pth.tar
  done
done

echo "[ALL DONE] Summary written to $SUMMARY_CSV"
