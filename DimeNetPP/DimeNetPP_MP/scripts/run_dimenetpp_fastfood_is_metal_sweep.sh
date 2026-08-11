#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh" || exit 1

EPOCHS="${DIMENET_EPOCHS:-350}"
BATCH_SIZE="${DIMENET_BATCH_SIZE:-64}"
GPU="${DIMENET_GPU:-0}"
WRAPPER="fastfood"
NUM_BLOCKS="${DIMENET_NUM_BLOCKS:-1}"   # ~85k-param 1-layer convention used across this study
# One-cycle + best-model arm. Defaults keep the original behaviour (constant lr, final-epoch
# scoring) so the existing 44-run is_metal sweep stays reproducible.
LR="${DIMENET_LR:-0.001}"
LR_SCHEDULE="${DIMENET_LR_SCHEDULE:-none}"
PATIENCE="${DIMENET_PATIENCE:-0}"
RESTORE_BEST="${DIMENET_RESTORE_BEST:-0}"
CACHE_ROOT="${DIMENET_CACHE_ROOT:-${PROJECT_DIR}/cached_tensors_dimenetpp_is_metal}"
RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_is_metal_fastfood}"
SUMMARY_CSV="${RESULT_ROOT}/dimenetpp_is_metal_fastfood_summary.csv"
PRINT_PREFIX="[DIMENET-IS_METAL]"

# 8 intrinsic dims x 4 seeds = 32 runs. Entry format: frac:pct:model_seed:split_seed
# split_seed = model_seed + 1000.
RUNS=(
  "1.0:100:101:1101"   "1.0:100:202:1202"   "1.0:100:303:1303"   "1.0:100:404:1404"
  "0.8:80:101:1101"    "0.8:80:202:1202"    "0.8:80:303:1303"    "0.8:80:404:1404"
  "0.6:60:101:1101"    "0.6:60:202:1202"    "0.6:60:303:1303"    "0.6:60:404:1404"
  "0.4:40:101:1101"    "0.4:40:202:1202"    "0.4:40:303:1303"    "0.4:40:404:1404"
  "0.2:20:101:1101"    "0.2:20:202:1202"    "0.2:20:303:1303"    "0.2:20:404:1404"
  "0.15:15:101:1101"   "0.15:15:202:1202"   "0.15:15:303:1303"   "0.15:15:404:1404"
  "0.1:10:101:1101"    "0.1:10:202:1202"    "0.1:10:303:1303"    "0.1:10:404:1404"
  "0.05:5:101:1101"    "0.05:5:202:1202"    "0.05:5:303:1303"    "0.05:5:404:1404"
)

mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,id_dim_percent,id_dim_frac,model_seed,split_seed,test_auc,best_val_auc,test_acc,epochs,duration_sec,status,output_dir" > "$SUMMARY_CSV"
fi

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local id_dim="$4"; local id_dim_percent="$5"; local model_seed="$6"; local split_seed="$7"
  local epochs="$8"; local duration_sec="$9"; local exit_code="${10}"
  local output_dir="${11}"; local write_summary="${12}"

  "$DIMENET_PYTHON" - "$summary_csv" "$metadata_json" "$log_file" \
    "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$epochs" "$duration_sec" \
    "$exit_code" "$output_dir" "$write_summary" "$LR" "$LR_SCHEDULE" "$RESTORE_BEST" "$PATIENCE" <<'PY'
import csv, json, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, id_dim, id_dim_percent, model_seed, split_seed,
 epochs, duration_sec, exit_code, output_dir, write_summary, lr, lr_schedule,
 restore_best, patience) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

def _last(pattern):
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None

test_auc = _last(r"Test AUC:\s*([-+0-9.eE]+)")
test_acc = _last(r"Test ACC:\s*([-+0-9.eE]+)")
val_matches = re.findall(r"Validation AUC:\s*([-+0-9.eE]+)", text)
best_val_auc = max(float(v) for v in val_matches) if val_matches else None

status = "success" if (int(exit_code) == 0 and test_auc is not None) else "failed"

row = {
    "task": "is_metal",
    "lr": float(lr),
    "lr_schedule": lr_schedule,
    "restore_best": restore_best == "1",
    "patience": int(patience),
    "id_dim_percent": (int(float(id_dim_percent)) if float(id_dim_percent)==int(float(id_dim_percent)) else float(id_dim_percent)),
    "id_dim_frac": float(id_dim),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "test_auc": test_auc,
    "best_val_auc": best_val_auc,
    "epochs": int(epochs),
    "duration_sec": int(float(duration_sec)),
    "status": status,
    "output_dir": output_dir,
}
if test_acc is not None:
    row["test_acc"] = test_acc
Path(metadata_json).write_text(json.dumps(row, indent=2))

if write_summary == "1":
    csv_row = {
        "task": "is_metal",
        "id_dim_percent": row["id_dim_percent"],
        "id_dim_frac": row["id_dim_frac"],
        "model_seed": row["model_seed"],
        "split_seed": row["split_seed"],
        "test_auc": "" if test_auc is None else test_auc,
        "best_val_auc": "" if best_val_auc is None else best_val_auc,
        "test_acc": "" if test_acc is None else test_acc,
        "epochs": row["epochs"],
        "duration_sec": row["duration_sec"],
        "status": status,
        "output_dir": output_dir,
    }
    with Path(summary_csv).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writerow(csv_row)

print(f"[metadata] {Path(output_dir).name} status={status} test_auc={test_auc} best_val_auc={best_val_auc} test_acc={test_acc}")
PY
}

run_one() {
  local id_dim="$1"; local id_dim_percent="$2"; local model_seed="$3"; local split_seed="$4"; local write_summary="$5"
  local cache_dir="${CACHE_ROOT}/splitseed${split_seed}"
  local cache_file="${cache_dir}/dimenetpp_is_metal_cached_tensors.pkl"
  if [[ ! -f "$cache_file" ]]; then
    echo "$PRINT_PREFIX ERROR: missing tensor cache ${cache_file}" >&2
    return 1
  fi

  local run_name="dimenetpp_is_metal_${WRAPPER}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
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

  "$DIMENET_PYTHON" "${CODE_DIR}/dimenet_run_is_metal_v3.py" \
    --method "$WRAPPER" \
    --id_dim "$id_dim" \
    --num_blocks "$NUM_BLOCKS" \
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
    $([[ "$RESTORE_BEST" == "1" ]] && echo --restore_best) \
    > "$log_file" 2>&1
  exit_code=$?
  end_time="$(date -Iseconds)"; end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" \
    "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$EPOCHS" "$duration_sec" \
    "$exit_code" "$out_dir" "$write_summary"

  echo "$PRINT_PREFIX DONE ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

SELECTED_RUNS="${DIMENET_RUNS:-${IS_METAL_RUNS:-}}"
if [[ -n "$SELECTED_RUNS" ]]; then
  echo "$PRINT_PREFIX explicit selection: ${SELECTED_RUNS}"
  for item in $SELECTED_RUNS; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "0" || exit $?
  done
  echo "$PRINT_PREFIX SELECTED RUNS DONE."
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#RUNS[@]} )); then
    echo "$PRINT_PREFIX ERROR: array index ${idx} out of range 0..$(( ${#RUNS[@]} - 1 ))" >&2
    exit 1
  fi
  IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "${RUNS[$idx]}"
  run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "0"
else
  for item in "${RUNS[@]}"; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1" || exit $?
  done
  echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
fi
