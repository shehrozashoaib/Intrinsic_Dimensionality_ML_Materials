#!/usr/bin/env bash
# DimeNet++ Fastfood intrinsic-dimension sweep for matbench phonons (full dataset).
# Layer count (num_blocks) is fixed per job via NUM_BLOCKS env (so one job = one layer count,
# full 12-dim x 3-seed sweep). Per-num_blocks summary CSV avoids races between bundled jobs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh" || exit 1

EPOCHS="${DIMENET_EPOCHS:-80}"
# One-cycle + best-model arm. Defaults keep the original behaviour (constant lr, final-epoch
# scoring) so the existing 100-run phonons sweep stays reproducible.
LR="${DIMENET_LR:-0.001}"
LR_SCHEDULE="${DIMENET_LR_SCHEDULE:-none}"
PATIENCE="${DIMENET_PATIENCE:-0}"
RESTORE_BEST="${DIMENET_RESTORE_BEST:-0}"
BATCH_SIZE="${DIMENET_BATCH_SIZE:-64}"
GPU="${DIMENET_GPU:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-1}"
WRAPPER="fastfood"
CACHE_ROOT="${DIMENET_CACHE_ROOT:-${PROJECT_DIR}/cached_tensors_dimenetpp_phonons}"
RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_phonons_fastfood}"
SUMMARY_CSV="${RESULT_ROOT}/dimenetpp_phonons_fastfood_nb${NUM_BLOCKS}_summary.csv"
PRINT_PREFIX="[DIMENET-KVRH-nb${NUM_BLOCKS}]"

# phonons sweep: dims 40/60/80/100% x 10 seeds (model:split = seed:seed+1000). 40 runs.
# Format: id_dim:id_dim_percent:model_seed:split_seed
DIMS=("0.2:20" "0.15:15" "0.1:10" "0.05:5" "0.02:2" "0.01:1")
SEEDS=("123:1123" "456:1456" "789:1789" "234:1234" "567:1567" "891:1891" "345:1345" "678:1678" "912:1912" "135:1135")
RUNS=()
for d in "${DIMS[@]}"; do
  IFS=':' read -r frac pct <<< "$d"
  for s in "${SEEDS[@]}"; do
    IFS=':' read -r ms ss <<< "$s"
    RUNS+=("${frac}:${pct}:${ms}:${ss}")
  done
done

mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,target_key,wrapper,num_blocks,id_dim,id_dim_percent,model_seed,split_seed,epochs,batch_size,cache_dir,output_dir,predictions_csv,history_json,val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local pred_csv="$4"; local history_json="$5"; local id_dim="$6"
  local id_dim_percent="$7"; local model_seed="$8"; local split_seed="$9"
  local epochs="${10}"; local batch_size="${11}"; local cache_dir="${12}"
  local output_dir="${13}"; local duration_sec="${14}"; local status="${15}"
  local exit_code="${16}"; local start_time="${17}"; local end_time="${18}"
  local write_summary="${19}"; local num_blocks="${20}"

  "$DIMENET_PYTHON" - "$summary_csv" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$epochs" "$batch_size" "$cache_dir" \
    "$output_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" "$write_summary" "$num_blocks" "$LR" "$LR_SCHEDULE" "$RESTORE_BEST" "$PATIENCE" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, pred_csv, history_json, id_dim, id_dim_percent,
 model_seed, split_seed, epochs, batch_size, cache_dir, output_dir, duration_sec, status,
 exit_code, start_time, end_time, write_summary, num_blocks, lr, lr_schedule,
 restore_best, patience) = sys.argv[1:]

test_mae = math.nan
pred_path = Path(pred_csv)
if pred_path.exists():
    targets, preds = [], []
    with pred_path.open() as f:
        for row in csv.DictReader(f):
            targets.append(float(row["target"]))
            preds.append(float(row["prediction"]))
    if targets:
        test_mae = sum(abs(t - p) for t, p in zip(targets, preds)) / len(targets)

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
val_mae = math.nan
matches = re.findall(r"Validation MAE \(original units\):\s*([-+0-9.eE]+)", text)
if matches:
    val_mae = float(matches[-1])
if math.isnan(test_mae):
    matches = re.findall(r"Test MAE:\s*([-+0-9.eE]+)", text)
    if matches:
        test_mae = float(matches[-1])

row = {
    "task": "matbench_phonons",
    "target_key": "phonons",
    "wrapper": "fastfood",
    "num_blocks": int(num_blocks),
    "lr": float(lr),
    "lr_schedule": lr_schedule,
    "restore_best": restore_best == "1",
    "patience": int(patience),
    "id_dim": id_dim,
    "id_dim_percent": int(id_dim_percent),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "cache_dir": cache_dir,
    "output_dir": output_dir,
    "predictions_csv": pred_csv if pred_path.exists() else "",
    "history_json": history_json if Path(history_json).exists() else "",
    "val_mae": val_mae,
    "test_mae": test_mae,
    "duration_sec": float(duration_sec),
    "status": status,
    "exit_code": int(exit_code),
    "start_time": start_time,
    "end_time": end_time,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
if write_summary == "1":
    with Path(summary_csv).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
print(f"[metadata] {Path(output_dir).name} status={status} val_mae={val_mae} test_mae={test_mae}")
PY
}

run_one() {
  local id_dim="$1"; local id_dim_percent="$2"; local model_seed="$3"; local split_seed="$4"; local write_summary="$5"
  local cache_dir="${CACHE_ROOT}/splitseed${split_seed}"
  local cache_file="${cache_dir}/dimenetpp_phonons_cached_tensors.pkl"
  if [[ ! -f "$cache_file" ]]; then
    echo "$PRINT_PREFIX ERROR: missing tensor cache ${cache_file}" >&2
    return 1
  fi

  local run_name="dimenetpp_phonons_${WRAPPER}_nb${NUM_BLOCKS}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "$PRINT_PREFIX skipping completed ${run_name}"
    return 0
  fi

  local start_time end_time start_epoch_sec end_epoch_sec duration_sec status exit_code
  start_time="$(date -Iseconds)"; start_epoch_sec="$(date +%s)"
  echo "$PRINT_PREFIX START ${run_name} at ${start_time}"

  "$DIMENET_PYTHON" "${CODE_DIR}/dimenet_run_phonons_v3.py" \
    --method "$WRAPPER" \
    --id_dim "$id_dim" \
    --num_blocks "$NUM_BLOCKS" \
    --clipnorm 0 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$model_seed" \
    --split_seed "$split_seed" \
    --gpu "$GPU" \
    --cache_dir "$cache_dir" \
    --out_dir "$out_dir" \
    --lr "$LR" \
    --lr_schedule "$LR_SCHEDULE" \
    --patience "$PATIENCE" \
    ${RESTORE_BEST:+$([[ "$RESTORE_BEST" == "1" ]] && echo --restore_best)} \
    > "$log_file" 2>&1
  exit_code=$?
  end_time="$(date -Iseconds)"; end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  local pred_csv history_json
  pred_csv="$(find "$out_dir" -maxdepth 1 -name 'dimenetpp_test_predictions_*.csv' -print -quit)"
  history_json="$(find "$out_dir" -maxdepth 1 -name 'history_*.json' -print -quit)"

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$EPOCHS" "$BATCH_SIZE" "$cache_dir" \
    "$out_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" "$write_summary" "$NUM_BLOCKS"

  echo "$PRINT_PREFIX DONE ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

SELECTED_RUNS="${DIMENET_RUNS:-}"
if [[ -n "$SELECTED_RUNS" ]]; then
  echo "$PRINT_PREFIX explicit selection: ${SELECTED_RUNS}"
  for item in $SELECTED_RUNS; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX SELECTED RUNS DONE. Summary: ${SUMMARY_CSV}"
else
  for item in "${RUNS[@]}"; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
fi
