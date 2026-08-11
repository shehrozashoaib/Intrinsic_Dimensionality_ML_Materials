#!/usr/bin/env bash
# DimeNet++ "doubled" model (num_blocks=2, clipnorm=1.0) at FULL parameter space only (id_dim=1.0).
# Required: DIMENET_TASK=phonons|dielectric   (log_kvrh nb=2 dim100 already exists — not handled here)
# Required: DIMENET_SEEDS="s1 s2 ..." (split_seed = model_seed + 1000)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh" || exit 1

TASK="${DIMENET_TASK:?set DIMENET_TASK=phonons|dielectric}"
SEEDS="${DIMENET_SEEDS:?set DIMENET_SEEDS=\"s1 s2 ...\"}"
NUM_BLOCKS=2
CLIPNORM="${DIMENET_CLIPNORM:-1.0}"
BATCH_SIZE="${DIMENET_BATCH_SIZE:-64}"
GPU="${DIMENET_GPU:-0}"
WRAPPER="fastfood"

case "$TASK" in
  phonons)    EPOCHS=80;  CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_phonons";    CACHE_FILE_PFX="dimenetpp_phonons" ;;
  dielectric) EPOCHS=150; CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_dielectric"; CACHE_FILE_PFX="dimenetpp_dielectric" ;;
  *) echo "Unknown DIMENET_TASK=$TASK" >&2; exit 1 ;;
esac
EPOCHS="${DIMENET_EPOCHS_OVERRIDE:-$EPOCHS}"

RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_2layer_fastfood/${TASK}}"
SUMMARY_CSV="${RESULT_ROOT}/dimenetpp_2layer_${TASK}_fastfood_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,target_key,wrapper,num_blocks,clipnorm,id_dim,id_dim_percent,model_seed,split_seed,epochs,batch_size,cache_dir,output_dir,predictions_csv,history_json,val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

echo "[INFO] TASK=$TASK nb=$NUM_BLOCKS clipnorm=$CLIPNORM epochs=$EPOCHS seeds=$SEEDS"

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local pred_csv="$4"; local history_json="$5"; local model_seed="$6"; local split_seed="$7"
  local out_dir="$8"; local cache_dir="$9"; local duration_sec="${10}"; local status="${11}"
  local exit_code="${12}"; local start_time="${13}"; local end_time="${14}"

  "$DIMENET_PYTHON" - "$summary_csv" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$model_seed" "$split_seed" "$out_dir" "$cache_dir" "$duration_sec" "$status" "$exit_code" \
    "$start_time" "$end_time" "$TASK" "$NUM_BLOCKS" "$CLIPNORM" "$EPOCHS" "$BATCH_SIZE" <<'PY'
import csv, json, math, re, sys
from pathlib import Path
(summary_csv, metadata_json, log_file, pred_csv, history_json, model_seed, split_seed,
 output_dir, cache_dir, duration_sec, status, exit_code, start_time, end_time, task,
 num_blocks, clipnorm, epochs, batch_size) = sys.argv[1:]

test_mae = math.nan
pred_path = Path(pred_csv)
if pred_path.exists():
    targets, preds = [], []
    with pred_path.open() as f:
        for row in csv.DictReader(f):
            targets.append(float(row["target"])); preds.append(float(row["prediction"]))
    if targets:
        test_mae = sum(abs(t - p) for t, p in zip(targets, preds)) / len(targets)

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
val_mae = math.nan
m = re.findall(r"Validation MAE \(original units\):\s*([-+0-9.eE]+)", text)
if m: val_mae = float(m[-1])
if math.isnan(test_mae):
    m = re.findall(r"Test MAE:\s*([-+0-9.eE]+)", text)
    if m: test_mae = float(m[-1])

row = {
    "task": f"matbench_{task}", "target_key": task, "wrapper": "fastfood",
    "num_blocks": int(num_blocks), "clipnorm": float(clipnorm), "id_dim": "1.0",
    "id_dim_percent": 100, "model_seed": int(model_seed), "split_seed": int(split_seed),
    "epochs": int(epochs), "batch_size": int(batch_size), "cache_dir": cache_dir,
    "output_dir": output_dir, "predictions_csv": pred_csv if pred_path.exists() else "",
    "history_json": history_json if Path(history_json).exists() else "",
    "val_mae": val_mae, "test_mae": test_mae, "duration_sec": float(duration_sec),
    "status": status, "exit_code": int(exit_code), "start_time": start_time, "end_time": end_time,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
with Path(summary_csv).open("a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
print(f"[metadata] {Path(output_dir).name} status={status} val_mae={val_mae} test_mae={test_mae}")
PY
}

run_one() {
  local model_seed="$1"
  local split_seed=$((model_seed + 1000))
  local cache_dir="${CACHE_ROOT}/splitseed${split_seed}"
  local cache_file="${cache_dir}/${CACHE_FILE_PFX}_cached_tensors.pkl"
  if [[ ! -f "$cache_file" ]]; then
    echo "ERROR: missing tensor cache ${cache_file}" >&2
    return 1
  fi

  local run_name="dimenetpp_2layer_${TASK}_${WRAPPER}_nb${NUM_BLOCKS}_dim100_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "[skip] ${run_name} (already success)"; return 0
  fi

  local start_time end_time start_epoch_sec end_epoch_sec duration_sec status exit_code
  start_time="$(date -Iseconds)"; start_epoch_sec="$(date +%s)"
  echo "[START] ${run_name} at ${start_time}"

  "$DIMENET_PYTHON" "${CODE_DIR}/dimenet_run_${TASK}_v3.py" \
    --method "$WRAPPER" \
    --id_dim 1.0 \
    --num_blocks "$NUM_BLOCKS" \
    --clipnorm "$CLIPNORM" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$model_seed" \
    --split_seed "$split_seed" \
    --gpu "$GPU" \
    --cache_dir "$cache_dir" \
    --out_dir "$out_dir" \
    > "$log_file" 2>&1
  exit_code=$?
  end_time="$(date -Iseconds)"; end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  local pred_csv history_json
  pred_csv="$(find "$out_dir" -maxdepth 1 -name 'dimenetpp_test_predictions_*.csv' -print -quit)"
  history_json="$(find "$out_dir" -maxdepth 1 -name 'history_*.json' -print -quit)"

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$model_seed" "$split_seed" "$out_dir" "$cache_dir" "$duration_sec" "$status" "$exit_code" \
    "$start_time" "$end_time"

  echo "[DONE] ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

for seed in $SEEDS; do
  run_one "$seed"
done
echo "[ALL DONE] $TASK 2-layer -> $SUMMARY_CSV"
