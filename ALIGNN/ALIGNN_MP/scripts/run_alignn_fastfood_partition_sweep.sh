#!/usr/bin/env bash
# ALIGNN Fastfood intrinsic-dimension sweep over DATA-SUBSET PARTITIONS (10% / 1%).
# Datasets are pre-built by scripts/partition_alignn_datasets.py under alignn/partitions/.
# Each partition keeps ALIGNN's internal 80:10:10 split (via --split_seed).
#
# Required env:
#   PART_TASK = eform | bandgap
#   PART_RUNS = space-separated entries "partition_name:seed:id_dim:id_dim_percent"
#               e.g. "eform_p10_1:123:0.45:45 eform_p10_1:456:0.45:45"
# Optional:
#   PART_EPOCHS (default 350)
#
# One Slurm job sets PART_RUNS to whatever (partition,dim,seed) combos it should run.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALIGNN_DIR="${PROJECT_DIR}/alignn"
cd "$ALIGNN_DIR" || exit 1

CONDA_ENV="py312"
PYTHON="python"
WRAPPER="fastfood"
EPOCHS="${PART_EPOCHS:-350}"
ID_KEY="material_id"
PART_TASK="${PART_TASK:?set PART_TASK=eform|bandgap|log_kvrh}"
PRINT_PREFIX="[ALIGNN-PART-${PART_TASK}]"

if [[ "$PART_TASK" == "eform" ]]; then
  TARGET_KEY="formation_energy_per_atom"
  CONFIG_NAME="../configs/config_eform.json"
  RESULT_ROOT="../results_alignn_eform_partitions_fastfood"
elif [[ "$PART_TASK" == "bandgap" ]]; then
  TARGET_KEY="band_gap"
  CONFIG_NAME="../configs/config_bandgap.json"
  RESULT_ROOT="../results_alignn_bandgap_partitions_fastfood"
elif [[ "$PART_TASK" == "log_kvrh" ]]; then
  TARGET_KEY="log_kvrh"
  CONFIG_NAME="../configs/config_log_kvrh.json"
  RESULT_ROOT="../results_alignn_log_kvrh_partitions_fastfood"
else
  echo "Unknown PART_TASK=$PART_TASK" >&2; exit 1
fi

: "${PART_RUNS:?set PART_RUNS to space-separated partition:seed:id_dim:id_dim_percent entries}"
mkdir -p "$RESULT_ROOT"
SUMMARY_CSV="${RESULT_ROOT}/alignn_${PART_TASK}_partitions_fastfood_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,partition,wrapper,id_dim,id_dim_percent,seed,epochs,root_dir,output_dir,predictions_csv,history_json,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

append_metadata() {
  local out_dir="$1" log_file="$2" pred_csv="$3" history_json="$4" partition="$5" \
        id_dim="$6" id_dim_percent="$7" seed="$8" root_dir="$9" duration_sec="${10}" \
        status="${11}" exit_code="${12}" start_time="${13}" end_time="${14}"
  "$PYTHON" - "$SUMMARY_CSV" "${out_dir}/metadata.json" "$log_file" "$pred_csv" "$history_json" \
    "$PART_TASK" "$partition" "$WRAPPER" "$id_dim" "$id_dim_percent" "$seed" "$EPOCHS" "$root_dir" \
    "$out_dir" "$duration_sec" "$status" "$exit_code" "$start_time" "$end_time" <<'PY'
import csv, json, math, re, sys
from pathlib import Path
(summary_csv, meta_json, log_file, pred_csv, history_json, task, partition, wrapper, id_dim,
 id_dim_percent, seed, epochs, root_dir, out_dir, duration_sec, status, exit_code,
 start_time, end_time) = sys.argv[1:]
best_val = math.nan
hp = Path(history_json)
if hp.exists():
    try:
        hist = json.loads(hp.read_text())
        vals = [float(r["mae_out"]) for r in hist if isinstance(r, dict) and r.get("mae_out") is not None]
        if vals: best_val = min(vals)
    except Exception: pass
test_mae = math.nan
pp = Path(pred_csv)
if pp.exists():
    try:
        t, p = [], []
        with pp.open() as f:
            for row in csv.DictReader(f):
                t.append(float(str(row["target"]).strip())); p.append(float(str(row["prediction"]).strip()))
        if t and len(t) == len(p): test_mae = sum(abs(a-b) for a,b in zip(t,p))/len(t)
    except Exception: pass
if math.isnan(test_mae):
    txt = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    m = re.findall(r"Test MAE:\s*([-+0-9.eE]+)", txt)
    if m: test_mae = float(m[-1])
row = dict(task=task, partition=partition, wrapper=wrapper, id_dim=id_dim,
           id_dim_percent=int(id_dim_percent), seed=int(seed), epochs=int(epochs), root_dir=root_dir,
           output_dir=out_dir, predictions_csv=(pred_csv if pp.exists() else ""),
           history_json=(history_json if hp.exists() else ""), best_val_mae=best_val, test_mae=test_mae,
           duration_sec=float(duration_sec), status=status, exit_code=int(exit_code),
           start_time=start_time, end_time=end_time)
Path(meta_json).write_text(json.dumps(row, indent=2))
lock = summary_csv + ".lock"
with open(lock, "w") as lf:
    try:
        import fcntl; fcntl.flock(lf, fcntl.LOCK_EX)
    except Exception: pass
    with open(summary_csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
print(f"[metadata] {Path(out_dir).name} status={status} best_val_mae={best_val} test_mae={test_mae}")
PY
}

run_one() {
  local partition="$1" seed="$2" id_dim="$3" id_dim_percent="$4"
  local root_dir="partitions/${partition}"
  if [[ ! -f "${root_dir}/id_prop.json" ]]; then
    echo "$PRINT_PREFIX ERROR: missing ${ALIGNN_DIR}/${root_dir}/id_prop.json" >&2
    return 1
  fi
  local run_name="alignn_${PART_TASK}_${WRAPPER}_${partition}_dim${id_dim_percent}_seed${seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${partition}/${run_name}"
  local log_file="${out_dir}/train.log"
  local pred_csv="${out_dir}/prediction_results_test_set.csv"
  local history_json="${out_dir}/history_val_mae.json"
  mkdir -p "$out_dir"

  if [[ -f "${out_dir}/metadata.json" ]] && grep -q '"status": "success"' "${out_dir}/metadata.json"; then
    echo "$PRINT_PREFIX skip completed ${run_name}"; return 0
  fi

  local start_time end_time t0 duration status exit_code
  start_time="$(date -Iseconds)"; t0="$(date +%s)"
  echo "$PRINT_PREFIX START ${run_name} at ${start_time}"

  conda run -n "$CONDA_ENV" "$PYTHON" train_alignn.py \
    --root_dir "$root_dir" \
    --config_name "$CONFIG_NAME" \
    --target_key "$TARGET_KEY" \
    --id_key "$ID_KEY" \
    --subspace_method "$WRAPPER" \
    --id_dim "$id_dim" \
    --id_enable True \
    --epochs "$EPOCHS" \
    --output_dir "$out_dir" \
    --random_seed "$seed" \
    --split_seed "$seed" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  end_time="$(date -Iseconds)"; duration=$(( $(date +%s) - t0 ))
  status="success"; [[ "$exit_code" -ne 0 ]] && status="failed"

  append_metadata "$out_dir" "$log_file" "$pred_csv" "$history_json" "$partition" \
    "$id_dim" "$id_dim_percent" "$seed" "$root_dir" "$duration" "$status" "$exit_code" "$start_time" "$end_time"
  rm -f "$out_dir"/*.pt "$out_dir"/*.pth "$out_dir"/*.pth.tar
  echo "$PRINT_PREFIX DONE ${run_name} status=${status} duration=${duration}s"
}

echo "$PRINT_PREFIX selection: ${PART_RUNS}"
for item in $PART_RUNS; do
  IFS=':' read -r partition seed id_dim id_dim_percent <<< "$item"
  run_one "$partition" "$seed" "$id_dim" "$id_dim_percent"
done
echo "$PRINT_PREFIX ALL SELECTED DONE. Summary: ${SUMMARY_CSV}"
