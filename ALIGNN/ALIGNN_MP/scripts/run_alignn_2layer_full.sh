#!/usr/bin/env bash
# ALIGNN "doubled" model (alignn_layers=2, gcn_layers=2) at FULL parameter space only (id_dim=1.0).
# Required: ALIGNN_TASK=dielectric|phonons|log_kvrh|eform|bandgap
# Required: ALIGNN_SEEDS="s1 s2 ..." (model_seed=each seed; split_seed = model_seed+1000 for
#   Matbench tasks (dielectric/phonons/log_kvrh), = model_seed for MP tasks (eform/bandgap, matched)).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALIGNN_DIR="${PROJECT_DIR}/alignn"
cd "$ALIGNN_DIR" || exit 1

CONDA_ENV="py312"
PYTHON="python"
TRAIN_SCRIPT="${ALIGNN_DIR}/train_alignn.py"
TASK="${ALIGNN_TASK:?set ALIGNN_TASK=dielectric|phonons|log_kvrh|eform|bandgap}"
SEEDS="${ALIGNN_SEEDS:?set ALIGNN_SEEDS=\"s1 s2 ...\"}"
ID_KEY="material_id"
WRAPPER="fastfood"

case "$TASK" in
  dielectric) TARGET_KEY="dielectric";                 DATASET_DIR="MP_json_dielectric"; EPOCHS=150; OFFSET=1000; MB_TASK_NAME="matbench_dielectric" ;;
  phonons)    TARGET_KEY="phonons";                     DATASET_DIR="MP_json_phonons";    EPOCHS=80;  OFFSET=1000; MB_TASK_NAME="matbench_phonons" ;;
  log_kvrh)   TARGET_KEY="log_kvrh";                    DATASET_DIR="MP_json_log_kvrh";   EPOCHS=120; OFFSET=1000; MB_TASK_NAME="matbench_log_kvrh" ;;
  eform)      TARGET_KEY="formation_energy_per_atom";   DATASET_DIR="MP_json_eform";      EPOCHS=350; OFFSET=0;    MB_TASK_NAME="mp_eform" ;;
  bandgap)    TARGET_KEY="band_gap";                    DATASET_DIR="MP_json_bandgap";    EPOCHS=350; OFFSET=0;    MB_TASK_NAME="mp_bandgap" ;;
  *) echo "Unknown ALIGNN_TASK=$TASK" >&2; exit 1 ;;
esac
EPOCHS="${ALIGNN_EPOCHS_OVERRIDE:-$EPOCHS}"
CONFIG_ABS="${PROJECT_DIR}/configs/config_${TASK}_2layer.json"
ROOT_ABS="${ALIGNN_DIR}/${DATASET_DIR}"

CONDA_BASE="/ibex/user/${USER}/miniconda3"
PYTHON_BIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/${PYTHON}"
if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC="${ALIGNN_DETERMINISTIC:-0}"

RESULT_ROOT="${ALIGNN_RESULT_ROOT:-${PROJECT_DIR}/results_2layer_fastfood/${TASK}}"
SUMMARY_CSV="${RESULT_ROOT}/alignn_2layer_${TASK}_fastfood_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,target_key,wrapper,id_dim,id_dim_percent,data_seed,model_seed,split_seed,epochs,batch_size,root_dir,output_dir,predictions_csv,history_json,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

echo "[INFO] TASK=$TASK epochs=$EPOCHS wrapper=$WRAPPER seeds=$SEEDS config=$CONFIG_ABS root=$ROOT_ABS"

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local pred_csv="$4"; local history_json="$5"; local task="$6"
  local target_key="$7"; local wrapper="$8"; local id_dim="$9"
  local id_dim_percent="${10}"; local data_seed="${11}"; local model_seed="${12}"
  local split_seed="${13}"; local epochs="${14}"; local batch_size="${15}"
  local root_dir="${16}"; local output_dir="${17}"; local duration_sec="${18}"
  local status="${19}"; local exit_code="${20}"; local start_time="${21}"
  local end_time="${22}"

  "$PYTHON_BIN" - "$summary_csv" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$task" "$target_key" "$wrapper" "$id_dim" "$id_dim_percent" "$data_seed" "$model_seed" "$split_seed" \
    "$epochs" "$batch_size" "$root_dir" "$output_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, pred_csv, history_json, task, target_key, wrapper,
 id_dim, id_dim_percent, data_seed, model_seed, split_seed, epochs, batch_size, root_dir,
 output_dir, duration_sec, status, exit_code, start_time, end_time) = sys.argv[1:]

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
with Path(summary_csv).open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(f"[metadata] {Path(output_dir).name} status={status} best_val_mae={best_val_mae} test_mae={test_mae}")
PY
}

run_one() {
  local model_seed="$1"
  local split_seed=$((model_seed + OFFSET))
  local run_name="alignn_2layer_${TASK}_${WRAPPER}_dim100_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  local pred_csv="${out_dir}/prediction_results_test_set.csv"
  local history_json="${out_dir}/history_val_mae.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "[skip] ${run_name} (already success)"; return 0
  fi

  local start_time end_time start_epoch_sec end_epoch_sec duration_sec status exit_code
  start_time="$(date -Iseconds)"; start_epoch_sec="$(date +%s)"
  echo "[START] ${run_name} at ${start_time}"

  ( cd "$out_dir" && conda run -n "$CONDA_ENV" "$PYTHON" "$TRAIN_SCRIPT" \
      --root_dir "$ROOT_ABS" \
      --config_name "$CONFIG_ABS" \
      --target_key "$TARGET_KEY" \
      --id_key "$ID_KEY" \
      --subspace_method "$WRAPPER" \
      --id_dim 1.0 \
      --id_enable True \
      --epochs "$EPOCHS" \
      --output_dir "$out_dir" \
      --random_seed "$model_seed" \
      --split_seed "$split_seed" ) 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  end_time="$(date -Iseconds)"; end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  if [[ ! -f "$pred_csv" ]]; then
    pred_match="$(find "$out_dir" -maxdepth 1 -name 'prediction_results_test_set*.csv' -print -quit)"
    [[ -n "${pred_match:-}" ]] && pred_csv="$pred_match"
  fi
  if [[ ! -f "$history_json" ]]; then
    history_match="$(find "$out_dir" -maxdepth 1 -name 'history_val_mae*.json' -print -quit)"
    [[ -n "${history_match:-}" ]] && history_json="$history_match"
  fi

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" "$pred_csv" "$history_json" \
    "$MB_TASK_NAME" "$TARGET_KEY" "$WRAPPER" "1.0" "100" "$split_seed" "$model_seed" "$split_seed" \
    "$EPOCHS" "config" "$ROOT_ABS" "$out_dir" "$duration_sec" "$status" "$exit_code" \
    "$start_time" "$end_time"

  rm -f "$out_dir"/*.pt "$out_dir"/*.pth "$out_dir"/*.pth.tar
  echo "[DONE] ${run_name} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
}

for seed in $SEEDS; do
  run_one "$seed"
done
echo "[ALL DONE] $TASK 2-layer -> $SUMMARY_CSV"
