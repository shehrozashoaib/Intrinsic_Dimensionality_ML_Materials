#!/usr/bin/env bash
set -uo pipefail

# ALIGNN Fastfood intrinsic-dimension sweep for MP band gap.
# Run from /workspace/alignn:
#   bash run_alignn_fastfood_bandgap_sweep.sh
#
# Schedule implemented from the request:
#   1) Existing MP_json, data/model/split seed 123: all dims
#   2) Reshuffled MP_json_seed456, data/model/split seed 456: all dims
#   3) Reshuffled MP_json_seed789, data/model/split seed 789: only 60% and 45%

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}/alignn" || exit 1

CONDA_ENV="py312"
PYTHON="python"
JSONL_PATH="mp21.jsonl"
CONFIG_NAME="../configs/config_bandgap.json"
TARGET_KEY="band_gap"
ID_KEY="material_id"
WRAPPER="fastfood"
EPOCHS=350
BATCH_SIZE=""  # leave empty to use config_bandgap.json batch_size
PRINT_PREFIX="[ALIGNN-FASTFOOD]"

RESULT_ROOT="../results_alignn_bandgap_fastfood"
SUMMARY_CSV="${RESULT_ROOT}/alignn_bandgap_fastfood_summary.csv"
mkdir -p "$RESULT_ROOT"

# Full sweep dims are fractions of full parameter count.
FULL_DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.05:5")
FOCUSED_DIMS=("0.6:60" "0.45:45")

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,target_key,wrapper,id_dim,id_dim_percent,data_seed,model_seed,split_seed,epochs,batch_size,root_dir,output_dir,predictions_csv,history_json,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

ensure_dataset() {
  local data_seed="$1"
  local root_dir="$2"

  if [[ -f "${root_dir}/id_prop.json" ]]; then
    if validate_dataset "$root_dir"; then
      echo "$PRINT_PREFIX using existing dataset ${root_dir}/id_prop.json"
      return 0
    fi
    echo "$PRINT_PREFIX existing dataset ${root_dir}/id_prop.json does not match target=${TARGET_KEY}; regenerating from ${JSONL_PATH}"
  fi

  if [[ ! -f "$JSONL_PATH" ]]; then
    echo "$PRINT_PREFIX ERROR: missing ${JSONL_PATH}; cannot create ${root_dir}" >&2
    return 1
  fi

  echo "$PRINT_PREFIX creating reshuffled dataset ${root_dir} with seed=${data_seed}"
  conda run -n "$CONDA_ENV" "$PYTHON" convert_jsonl_to_idprop.py \
    --jsonl "$JSONL_PATH" \
    --out_dir "$root_dir" \
    --id_key "$ID_KEY" \
    --target_key "$TARGET_KEY" \
    --seed "$data_seed"

  validate_dataset "$root_dir"
}

validate_dataset() {
  local root_dir="$1"
  "$PYTHON" - "$root_dir/id_prop.json" "$TARGET_KEY" "$ID_KEY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_key = sys.argv[2]
id_key = sys.argv[3]
try:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("id_prop.json is empty or is not a JSON list")
    first = data[0]
    missing = [key for key in (id_key, target_key, "atoms") if key not in first]
    if missing:
        raise KeyError(f"missing keys {missing}; found keys {list(first.keys())[:8]}")
except Exception as exc:
    print(f"[dataset-check] invalid {path}: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"[dataset-check] valid {path}: n={len(data)}, target_key={target_key}, id_key={id_key}")
PY
}

append_metadata() {
  local summary_csv="$1"
  local metadata_json="$2"
  local log_file="$3"
  local pred_csv="$4"
  local history_json="$5"
  local task="$6"
  local target_key="$7"
  local wrapper="$8"
  local id_dim="$9"
  local id_dim_percent="${10}"
  local data_seed="${11}"
  local model_seed="${12}"
  local split_seed="${13}"
  local epochs="${14}"
  local batch_size="${15}"
  local root_dir="${16}"
  local output_dir="${17}"
  local duration_sec="${18}"
  local status="${19}"
  local exit_code="${20}"
  local start_time="${21}"
  local end_time="${22}"

  "$PYTHON" - "$summary_csv" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$task" "$target_key" "$wrapper" "$id_dim" "$id_dim_percent" "$data_seed" "$model_seed" "$split_seed" \
    "$epochs" "$batch_size" "$root_dir" "$output_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path

(summary_csv, metadata_json, log_file, pred_csv, history_json, task, target_key, wrapper,
 id_dim, id_dim_percent, data_seed, model_seed, split_seed, epochs, batch_size, root_dir,
 output_dir, duration_sec, status, exit_code, start_time, end_time) = sys.argv[1:]

best_val_mae = math.nan
hist_path = Path(history_json)
if hist_path.exists():
    try:
        hist = json.loads(hist_path.read_text())
        vals = []
        for rec in hist:
            if isinstance(rec, dict):
                val = rec.get("mae_out")
                if val is not None:
                    vals.append(float(val))
        if vals:
            best_val_mae = min(vals)
    except Exception:
        pass

test_mae = math.nan
pred_path = Path(pred_csv)
if pred_path.exists():
    targets = []
    preds = []
    try:
        with pred_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    targets.append(float(str(row["target"]).strip()))
                    preds.append(float(str(row["prediction"]).strip()))
                except Exception:
                    continue
        if targets and len(targets) == len(preds):
            test_mae = sum(abs(t - p) for t, p in zip(targets, preds)) / len(targets)
    except Exception:
        pass

if math.isnan(test_mae):
    text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    matches = re.findall(r"Test MAE:\s*([-+0-9.eE]+)", text)
    if matches:
        test_mae = float(matches[-1])

row = {
    "task": task,
    "target_key": target_key,
    "wrapper": wrapper,
    "id_dim": id_dim,
    "id_dim_percent": int(id_dim_percent),
    "data_seed": int(data_seed),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "epochs": int(epochs),
    "batch_size": batch_size,
    "root_dir": root_dir,
    "output_dir": output_dir,
    "predictions_csv": pred_csv if pred_path.exists() else "",
    "history_json": history_json if hist_path.exists() else "",
    "best_val_mae": best_val_mae,
    "test_mae": test_mae,
    "duration_sec": float(duration_sec),
    "status": status,
    "exit_code": int(exit_code),
    "start_time": start_time,
    "end_time": end_time,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
with Path(summary_csv).open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(f"[metadata] {Path(output_dir).name} status={status} best_val_mae={best_val_mae} test_mae={test_mae}")
PY
}

recover_duration_sec() {
  local log_file="$1"
  "$PYTHON" - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="replace") if path.exists() else ""
matches = re.findall(r"Time taken \(s\)\s*([-+0-9.eE]+)", text)
print(int(float(matches[-1])) if matches else 0)
PY
}

run_one() {
  local root_dir="$1"
  local data_seed="$2"
  local model_seed="$3"
  local split_seed="$4"
  local id_dim="$5"
  local id_dim_percent="$6"

  local run_name="alignn_bandgap_${WRAPPER}_dim${id_dim_percent}_dataseed${data_seed}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  local pred_csv="${out_dir}/prediction_results_test_set.csv"
  local history_json="${out_dir}/history_val_mae.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "$PRINT_PREFIX skipping completed ${run_name}"
    return 0
  fi

  if [[ -s "$pred_csv" && -s "$history_json" ]]; then
    local recovered_duration_sec
    recovered_duration_sec="$(recover_duration_sec "$log_file")"
    append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
      "mp_bandgap" "$TARGET_KEY" "$WRAPPER" "$id_dim" "$id_dim_percent" "$data_seed" "$model_seed" "$split_seed" \
      "$EPOCHS" "${BATCH_SIZE:-config}" "$root_dir" "$out_dir" "$recovered_duration_sec" "success" 0 "recovered" "recovered"
    echo "$PRINT_PREFIX skipping completed ${run_name} (found test predictions/history; recovered metadata)"
    return 0
  fi

  local start_time end_time start_epoch_sec end_epoch_sec duration_sec status exit_code
  start_time="$(date -Iseconds)"
  start_epoch_sec="$(date +%s)"
  echo "$PRINT_PREFIX START ${run_name} at ${start_time}"

  local cmd=(conda run -n "$CONDA_ENV" "$PYTHON" train_alignn.py
    --root_dir "$root_dir"
    --config_name "$CONFIG_NAME"
    --target_key "$TARGET_KEY"
    --id_key "$ID_KEY"
    --subspace_method "$WRAPPER"
    --id_dim "$id_dim"
    --id_enable True
    --epochs "$EPOCHS"
    --output_dir "$out_dir"
    --random_seed "$model_seed"
    --split_seed "$split_seed")

  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  end_time="$(date -Iseconds)"
  end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))

  status="success"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
  fi

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "mp_bandgap" "$TARGET_KEY" "$WRAPPER" "$id_dim" "$id_dim_percent" "$data_seed" "$model_seed" "$split_seed" \
    "$EPOCHS" "${BATCH_SIZE:-config}" "$root_dir" "$out_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time"

  # Keep logs, metadata, predictions, and history. Remove checkpoints/weights to avoid filling SSD.
  rm -f "$out_dir"/*.pt "$out_dir"/*.pth "$out_dir"/*.pth.tar

  echo "$PRINT_PREFIX DONE ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

run_schedule() {
  local data_seed="$1"
  local root_dir="$2"
  local model_seed="$3"
  shift 3
  local dims=("$@")

  ensure_dataset "$data_seed" "$root_dir" || return 1
  for item in "${dims[@]}"; do
    IFS=':' read -r id_dim id_dim_percent <<< "$item"
    run_one "$root_dir" "$data_seed" "$model_seed" "$data_seed" "$id_dim" "$id_dim_percent"
  done
}

run_schedule 123 "MP_json" 123 "${FULL_DIMS[@]}"
run_schedule 456 "MP_json_seed456" 456 "${FULL_DIMS[@]}"
run_schedule 789 "MP_json_seed789" 789 "${FOCUSED_DIMS[@]}"

echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
