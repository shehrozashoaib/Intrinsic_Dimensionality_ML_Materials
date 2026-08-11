#!/usr/bin/env bash
set -uo pipefail

# ALIGNN Fastfood intrinsic-dimension sweep for MATBENCH matbench_mp_is_metal
# (binary classification, full dataset, 106,113 structures).
#
# Concurrency-safe: each run executes inside its own output directory using
# absolute paths, so parallel jobs never clobber each other's CWD scratch
# files (e.g. sc.pkl). Seed variance comes from --random_seed / --split_seed
# on a single shared dataset.
#
# Selection modes (checked in this order):
#   1. IS_METAL_RUNS env var set -> run exactly those entries; write per-run
#      metadata.json only (NO shared-CSV append -> safe for parallel jobs).
#      Entries are space-separated "id_dim_frac:percent:model_seed:split_seed".
#   2. SLURM_ARRAY_TASK_ID set   -> run the single RUNS[index] entry (metadata only).
#   3. neither                   -> run ALL runs sequentially and append to the
#      shared summary CSV (single-job use).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALIGNN_DIR="${PROJECT_DIR}/alignn"
cd "$ALIGNN_DIR" || exit 1

CONDA_ENV="py312"
PYTHON="python"
TRAIN_SCRIPT="${ALIGNN_DIR}/train_alignn.py"
CONFIG_ABS="${PROJECT_DIR}/configs/config_is_metal.json"
TARGET_KEY="is_metal"
ID_KEY="material_id"
WRAPPER="fastfood"
CLASSIFICATION_THRESHOLD="0.5"
EPOCHS="${IS_METAL_EPOCHS:-350}"       # full sweep default; smoke overrides this
BATCH_SIZE="${IS_METAL_BATCH_SIZE:-}"  # empty -> use config_is_metal.json batch_size (64)
DATASET_DIR="${IS_METAL_DATASET:-MP_json_is_metal}"
if [[ "$DATASET_DIR" = /* ]]; then
  ROOT_ABS="$DATASET_DIR"
else
  ROOT_ABS="${ALIGNN_DIR}/${DATASET_DIR}"
fi
PRINT_PREFIX="[ALIGNN-IS_METAL]"

CONDA_BASE="/ibex/user/${USER}/miniconda3"
PYTHON_BIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/${PYTHON}"
if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC="${ALIGNN_DETERMINISTIC:-0}"

RESULT_ROOT="${IS_METAL_RESULT_ROOT:-${PROJECT_DIR}/results_alignn_is_metal_fastfood}"
SUMMARY_CSV="${RESULT_ROOT}/alignn_is_metal_fastfood_summary.csv"
mkdir -p "$RESULT_ROOT"

# Flat run list. "id_dim_frac:percent:model_seed:split_seed"
# is_metal: dims 5/10/15/20/40/60/80/100% x 4 seeds (model s, split s+1000). 32 runs.
RUNS=()
for spec in "1.0:100" "0.8:80" "0.6:60" "0.4:40" "0.2:20" "0.15:15" "0.1:10" "0.05:5"; do
  frac="${spec%%:*}"; pct="${spec##*:}"
  for s in 101 202 303 404; do
    RUNS+=("${frac}:${pct}:${s}:$((s+1000))")
  done
done

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "id_dim_percent,id_dim_frac,model_seed,split_seed,test_auc,best_val_auc,test_acc,epochs,duration_sec,status,output_dir" > "$SUMMARY_CSV"
fi

check_dataset() {
  if [[ ! -f "${ROOT_ABS}/id_prop.json" ]]; then
    echo "$PRINT_PREFIX ERROR: ${ROOT_ABS}/id_prop.json not found." >&2
    return 1
  fi
  echo "$PRINT_PREFIX using dataset ${ROOT_ABS}/id_prop.json"
}

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local id_dim_frac="$4"; local id_dim_percent="$5"; local model_seed="$6"
  local split_seed="$7"; local epochs="$8"; local duration_sec="$9"
  local exit_code="${10}"; local output_dir="${11}"; local write_summary="${12}"

  "$PYTHON_BIN" - "$summary_csv" "$metadata_json" "$log_file" "$id_dim_frac" \
    "$id_dim_percent" "$model_seed" "$split_seed" "$epochs" "$duration_sec" \
    "$exit_code" "$output_dir" "$write_summary" <<'PY'
import csv, json, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, id_dim_frac, id_dim_percent, model_seed,
 split_seed, epochs, duration_sec, exit_code, output_dir, write_summary) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

# test_auc: parse the true probabilistic ROC-AUC printed by train.py.
test_auc = None
m = re.findall(r"Test ROCAUC:\s*([-+0-9.eE]+)", text)
if m:
    try:
        test_auc = float(m[-1])
    except ValueError:
        test_auc = None

# test_acc: optional argmax accuracy printed by train.py.
test_acc = None
m = re.findall(r"Test ACC:\s*([-+0-9.eE]+)", text)
if m:
    try:
        test_acc = float(m[-1])
    except ValueError:
        test_acc = None

# best_val_auc: no per-epoch validation-AUC tracker exists, so report null.
best_val_auc = None

status = "success" if (int(exit_code) == 0 and test_auc is not None) else "failed"

row = {
    "task": "is_metal",
    "id_dim_percent": (int(float(id_dim_percent)) if float(id_dim_percent)==int(float(id_dim_percent)) else float(id_dim_percent)),
    "id_dim_frac": float(id_dim_frac),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "test_auc": test_auc,
    "best_val_auc": best_val_auc,
    "test_acc": test_acc,
    "epochs": int(epochs),
    "duration_sec": int(float(duration_sec)),
    "status": status,
    "output_dir": output_dir,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))

if write_summary == "1":
    header = ["id_dim_percent", "id_dim_frac", "model_seed", "split_seed",
              "test_auc", "best_val_auc", "test_acc", "epochs", "duration_sec",
              "status", "output_dir"]
    with Path(summary_csv).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({k: ("" if row[k] is None else row[k]) for k in header})

print(f"[metadata] {Path(output_dir).name} status={status} "
      f"test_auc={test_auc} best_val_auc={best_val_auc} test_acc={test_acc}")
PY
}

run_one() {
  local id_dim_frac="$1"; local id_dim_percent="$2"; local model_seed="$3"; local split_seed="$4"; local write_summary="$5"
  [[ -z "$split_seed" ]] && split_seed="$model_seed"

  local run_name="alignn_is_metal_${WRAPPER}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"      # absolute
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

  local cmd=(conda run -n "$CONDA_ENV" "$PYTHON" "$TRAIN_SCRIPT"
    --root_dir "$ROOT_ABS"
    --config_name "$CONFIG_ABS"
    --target_key "$TARGET_KEY"
    --id_key "$ID_KEY"
    --classification_threshold "$CLASSIFICATION_THRESHOLD"
    --subspace_method "$WRAPPER"
    --id_dim "$id_dim_frac"
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

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$id_dim_frac" \
    "$id_dim_percent" "$model_seed" "$split_seed" "$EPOCHS" "$duration_sec" \
    "$exit_code" "$out_dir" "$write_summary"

  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"
  rm -f "$out_dir"/*.pt "$out_dir"/*.pth "$out_dir"/*.pth.tar
  echo "$PRINT_PREFIX DONE ${run_name} exit_code=${exit_code} duration=${duration_sec}s"
}

check_dataset || exit 1

if [[ -n "${IS_METAL_RUNS:-}" ]]; then
  echo "$PRINT_PREFIX explicit selection: ${IS_METAL_RUNS}"
  for item in $IS_METAL_RUNS; do
    IFS=':' read -r id_dim_frac id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim_frac" "$id_dim_percent" "$model_seed" "$split_seed" "0"
  done
  echo "$PRINT_PREFIX SELECTED RUNS DONE."
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#RUNS[@]} )); then
    echo "$PRINT_PREFIX ERROR: array index ${idx} out of range 0..$(( ${#RUNS[@]} - 1 ))" >&2
    exit 1
  fi
  IFS=':' read -r id_dim_frac id_dim_percent model_seed split_seed <<< "${RUNS[$idx]}"
  split_seed="${split_seed:-$model_seed}"
  run_one "$id_dim_frac" "$id_dim_percent" "$model_seed" "$split_seed" "0"
else
  for item in "${RUNS[@]}"; do
    IFS=':' read -r id_dim_frac id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim_frac" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
fi
