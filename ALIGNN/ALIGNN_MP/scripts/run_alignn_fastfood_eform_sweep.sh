#!/usr/bin/env bash
set -uo pipefail

# ALIGNN Fastfood intrinsic-dimension sweep for MP FORMATION ENERGY.
# Concurrency-safe: each run executes inside its own output directory using
# absolute paths, so parallel jobs never clobber each other's CWD scratch
# files (e.g. sc.pkl). Seed variance comes from --random_seed / --split_seed
# on a single shared dataset (the eform converter cannot reshuffle per seed).
#
# Selection modes (checked in this order):
#   1. EFORM_RUNS env var set  -> run exactly those entries; write per-run
#      metadata.json only (NO shared-CSV append -> safe for parallel jobs).
#      Entries are space-separated "id_dim:percent:model_seed:split_seed".
#      This is what scripts/submit_eform_all.sh uses.
#   2. SLURM_ARRAY_TASK_ID set -> run the single RUNS[index] entry (metadata only).
#   3. neither                 -> run ALL runs sequentially and append to the
#      shared summary CSV (single-job use).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALIGNN_DIR="${PROJECT_DIR}/alignn"
cd "$ALIGNN_DIR" || exit 1

CONDA_ENV="py312"
PYTHON="python"
TRAIN_SCRIPT="${ALIGNN_DIR}/train_alignn.py"
CONFIG_ABS="${PROJECT_DIR}/configs/config_eform.json"
TARGET_KEY="formation_energy_per_atom"
ID_KEY="material_id"
WRAPPER="fastfood"
EPOCHS="${EFORM_EPOCHS:-350}"           # full sweep default; smoke overrides this
BATCH_SIZE="${EFORM_BATCH_SIZE:-}"      # empty -> use config_eform.json batch_size (64)
DATASET_DIR="${EFORM_DATASET:-MP_json_eform}"
if [[ "$DATASET_DIR" = /* ]]; then
  ROOT_ABS="$DATASET_DIR"
else
  ROOT_ABS="${ALIGNN_DIR}/${DATASET_DIR}"
fi
PRINT_PREFIX="[ALIGNN-EFORM]"

CONDA_BASE="/ibex/user/${USER}/miniconda3"
PYTHON_BIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/${PYTHON}"
if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC="${ALIGNN_DETERMINISTIC:-0}"

RESULT_ROOT="${PROJECT_DIR}/results_alignn_eform_fastfood"
SUMMARY_CSV="${RESULT_ROOT}/alignn_eform_fastfood_summary.csv"
mkdir -p "$RESULT_ROOT"

# Flat run list (used by array mode). "id_dim:percent:model_seed:split_seed"
RUNS=(
  "1.0:100:123:1123"  "1.0:100:456:1456"
  "0.8:80:123:1123"   "0.8:80:456:1456"
  "0.6:60:123:1123"   "0.6:60:456:1456"   "0.6:60:789:1789"
  "0.5:50:123:1123"   "0.5:50:456:1456"
  "0.45:45:123:1123"  "0.45:45:456:1456"  "0.45:45:789:1789"
  "0.2:20:123:1123"   "0.2:20:456:1456"
  "0.1:10:123:1123"   "0.1:10:456:1456"
)

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,target_key,wrapper,id_dim,id_dim_percent,data_seed,model_seed,split_seed,epochs,batch_size,root_dir,output_dir,predictions_csv,history_json,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

check_dataset() {
  if [[ ! -f "${ROOT_ABS}/id_prop.json" ]]; then
    echo "$PRINT_PREFIX ERROR: ${ROOT_ABS}/id_prop.json not found." >&2
    echo "$PRINT_PREFIX Create it once first:" >&2
    echo "  export MP_API_KEY=...; cd ${ALIGNN_DIR}; conda run -n ${CONDA_ENV} python convert_jsonl_to_idprop_formation_energy.py --out_dir MP_json_eform --id_key ${ID_KEY} --target_key ${TARGET_KEY}" >&2
    return 1
  fi
  echo "$PRINT_PREFIX using dataset ${ROOT_ABS}/id_prop.json"
}

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local pred_csv="$4"; local history_json="$5"; local task="$6"
  local target_key="$7"; local wrapper="$8"; local id_dim="$9"
  local id_dim_percent="${10}"; local data_seed="${11}"; local model_seed="${12}"
  local split_seed="${13}"; local epochs="${14}"; local batch_size="${15}"
  local root_dir="${16}"; local output_dir="${17}"; local duration_sec="${18}"
  local status="${19}"; local exit_code="${20}"; local start_time="${21}"
  local end_time="${22}"; local write_summary="${23}"

  "$PYTHON_BIN" - "$summary_csv" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$task" "$target_key" "$wrapper" "$id_dim" "$id_dim_percent" "$data_seed" "$model_seed" "$split_seed" \
    "$epochs" "$batch_size" "$root_dir" "$output_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" "$write_summary" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, pred_csv, history_json, task, target_key, wrapper,
 id_dim, id_dim_percent, data_seed, model_seed, split_seed, epochs, batch_size, root_dir,
 output_dir, duration_sec, status, exit_code, start_time, end_time, write_summary) = sys.argv[1:]

best_val_mae = math.nan
hist_path = Path(history_json)
if hist_path.exists():
    try:
        hist = json.loads(hist_path.read_text())
        vals = [float(rec["mae_out"]) for rec in hist
                if isinstance(rec, dict) and rec.get("mae_out") is not None]
        if vals:
            best_val_mae = min(vals)
    except Exception:
        pass

test_mae = math.nan
pred_path = Path(pred_csv)
if pred_path.exists():
    targets, preds = [], []
    try:
        with pred_path.open() as f:
            for row in csv.DictReader(f):
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
    "task": task, "target_key": target_key, "wrapper": wrapper, "id_dim": id_dim,
    "id_dim_percent": int(id_dim_percent), "data_seed": int(data_seed),
    "model_seed": int(model_seed), "split_seed": int(split_seed), "epochs": int(epochs),
    "batch_size": batch_size, "root_dir": root_dir, "output_dir": output_dir,
    "predictions_csv": pred_csv if pred_path.exists() else "",
    "history_json": history_json if hist_path.exists() else "",
    "best_val_mae": best_val_mae, "test_mae": test_mae, "duration_sec": float(duration_sec),
    "status": status, "exit_code": int(exit_code), "start_time": start_time, "end_time": end_time,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
if write_summary == "1":
    with Path(summary_csv).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
print(f"[metadata] {Path(output_dir).name} status={status} best_val_mae={best_val_mae} test_mae={test_mae}")
PY
}

run_one() {
  local id_dim="$1"; local id_dim_percent="$2"; local model_seed="$3"; local split_seed="$4"; local write_summary="$5"
  [[ -z "$split_seed" ]] && split_seed="$model_seed"

  local run_name="alignn_eform_${WRAPPER}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"      # absolute
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  local pred_csv="${out_dir}/prediction_results_test_set.csv"
  local history_json="${out_dir}/history_val_mae.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "$PRINT_PREFIX skipping completed ${run_name}"
    return 0
  fi

  local pred_match history_match
  local start_time end_time start_epoch_sec end_epoch_sec duration_sec status exit_code
  start_time="$(date -Iseconds)"; start_epoch_sec="$(date +%s)"
  echo "$PRINT_PREFIX START ${run_name} at ${start_time}"

  local cmd=(conda run -n "$CONDA_ENV" "$PYTHON" "$TRAIN_SCRIPT"
    --root_dir "$ROOT_ABS"
    --config_name "$CONFIG_ABS"
    --target_key "$TARGET_KEY"
    --id_key "$ID_KEY"
    --subspace_method "$WRAPPER"
    --id_dim "$id_dim"
    --id_enable True
    --epochs "$EPOCHS"
    --output_dir "$out_dir"
    --random_seed "$model_seed"
    --split_seed "$split_seed")
  if [[ -n "$BATCH_SIZE" ]]; then cmd+=(--batch_size "$BATCH_SIZE"); fi

  # Run INSIDE the per-run output dir so any CWD-relative scratch (sc.pkl, etc.)
  # is isolated from other concurrent jobs.
  ( cd "$out_dir" && "${cmd[@]}" ) 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  end_time="$(date -Iseconds)"; end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  if [[ ! -f "$pred_csv" ]]; then
    pred_match="$(find "$out_dir" -maxdepth 1 -name 'prediction_results_test_set*.csv' -print -quit)"
    [[ -n "$pred_match" ]] && pred_csv="$pred_match"
  fi
  if [[ ! -f "$history_json" ]]; then
    history_match="$(find "$out_dir" -maxdepth 1 -name 'history_val_mae*.json' -print -quit)"
    [[ -n "$history_match" ]] && history_json="$history_match"
  fi

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "mp_eform" "$TARGET_KEY" "$WRAPPER" "$id_dim" "$id_dim_percent" "$split_seed" "$model_seed" "$split_seed" \
    "$EPOCHS" "${BATCH_SIZE:-config}" "$ROOT_ABS" "$out_dir" "$duration_sec" "$status" "$exit_code" \
    "$start_time" "$end_time" "$write_summary"

  rm -f "$out_dir"/*.pt "$out_dir"/*.pth "$out_dir"/*.pth.tar
  echo "$PRINT_PREFIX DONE ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

check_dataset || exit 1

if [[ -n "${EFORM_RUNS:-}" ]]; then
  echo "$PRINT_PREFIX explicit selection: ${EFORM_RUNS}"
  for item in $EFORM_RUNS; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "0"
  done
  echo "$PRINT_PREFIX SELECTED RUNS DONE."
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#RUNS[@]} )); then
    echo "$PRINT_PREFIX ERROR: array index ${idx} out of range 0..$(( ${#RUNS[@]} - 1 ))" >&2
    exit 1
  fi
  IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "${RUNS[$idx]}"
  split_seed="${split_seed:-$model_seed}"
  run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "0"
else
  for item in "${RUNS[@]}"; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
fi
