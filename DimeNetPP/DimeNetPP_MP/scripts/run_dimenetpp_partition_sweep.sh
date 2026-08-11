#!/usr/bin/env bash
# DimeNet++ Fastfood intrinsic-dimension sweep on DATASET PARTITIONS (quarter=25%, tenth=10%)
# for the MP tasks: eform and bandgap.
#
# One job = one (task, partition): full DIMS x SEEDS grid. Per-partition summary CSV avoids
# races between concurrently running partition jobs.
#
# Env vars:
#   DIMENET_TASK       eform | bandgap                       (REQUIRED)
#   DIMENET_PART       partition name, e.g. eform_quarter_1  (REQUIRED)
#   DIMENET_SEEDS      "model_seed:split_seed ..."           (default "123:1123 456:1456")
#   DIMENET_EPOCHS     default 350
#   DIMENET_CLIPNORM   default 0   (NO gradient clipping -- required for this study)
#   DIMENET_BATCH_SIZE default 64
#   DIMENET_GPU        default 0
#
# Model config is FIXED: --num_blocks 1 (the ~85k-param 1-layer baseline, D=83,685).
# The MP runners have no --int_emb_size arg, so it is not passed.
#
# Usage:
#   DIMENET_TASK=eform DIMENET_PART=eform_quarter_1 bash scripts/run_dimenetpp_partition_sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh" || exit 1

TASK="${DIMENET_TASK:-}"
PART="${DIMENET_PART:-}"

if [[ -z "$TASK" || -z "$PART" ]]; then
  echo "ERROR: DIMENET_TASK (eform|bandgap) and DIMENET_PART (e.g. eform_quarter_1) are required." >&2
  exit 1
fi
if [[ "$TASK" != "eform" && "$TASK" != "bandgap" ]]; then
  echo "ERROR: DIMENET_TASK must be 'eform' or 'bandgap' (got '${TASK}')." >&2
  exit 1
fi

EPOCHS="${DIMENET_EPOCHS:-350}"
BATCH_SIZE="${DIMENET_BATCH_SIZE:-64}"
GPU="${DIMENET_GPU:-0}"
CLIPNORM="${DIMENET_CLIPNORM:-0}"
NUM_BLOCKS=1
WRAPPER="fastfood"

CACHE_ROOT="${DIMENET_CACHE_ROOT:-${PROJECT_DIR}/cached_tensors_dimenetpp_part_${PART}}"
RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_${TASK}_partitions_fastfood}"
SUMMARY_CSV="${RESULT_ROOT}/dimenetpp_${TASK}_${PART}_summary.csv"
PRINT_PREFIX="[DIMENET-${TASK}-${PART}]"

# Per-task ID grids (id_dim_fraction:id_dim_percent).
if [[ "$TASK" == "eform" ]]; then
  DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.45:45" "0.2:20" "0.15:15" "0.1:10")
else
  DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.05:5")
fi

SEEDS_SPEC="${DIMENET_SEEDS:-123:1123 456:1456}"
read -r -a SEEDS <<< "$SEEDS_SPEC"

RUNS=()
for d in "${DIMS[@]}"; do
  IFS=':' read -r frac pct <<< "$d"
  for s in "${SEEDS[@]}"; do
    IFS=':' read -r ms ss <<< "$s"
    ss="${ss:-$ms}"
    RUNS+=("${frac}:${pct}:${ms}:${ss}")
  done
done

mkdir -p "${RESULT_ROOT}/${PART}"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,partition,num_blocks,id_dim,id_dim_percent,model_seed,split_seed,epochs,clipnorm,val_mae,test_mae,duration_sec,status,output_dir" > "$SUMMARY_CSV"
fi

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local task="$4"; local partition="$5"; local num_blocks="$6"
  local id_dim="$7"; local id_dim_percent="$8"; local model_seed="$9"
  local split_seed="${10}"; local epochs="${11}"; local clipnorm="${12}"
  local output_dir="${13}"; local duration_sec="${14}"; local exit_code="${15}"
  local write_summary="${16}"

  "$DIMENET_PYTHON" - "$summary_csv" "$metadata_json" "$log_file" "$task" "$partition" \
    "$num_blocks" "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$epochs" \
    "$clipnorm" "$output_dir" "$duration_sec" "$exit_code" "$write_summary" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, task, partition, num_blocks, id_dim, id_dim_percent,
 model_seed, split_seed, epochs, clipnorm, output_dir, duration_sec, exit_code,
 write_summary) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

val_mae = math.nan
matches = re.findall(r"Validation MAE \(original units\):\s*([-+0-9.eE]+)", text)
if matches:
    val_mae = float(matches[-1])

test_mae = math.nan
matches = re.findall(r"Test MAE:\s*([-+0-9.eE]+)", text)
if matches:
    test_mae = float(matches[-1])

status = "success" if (int(exit_code) == 0 and not math.isnan(test_mae)) else "failed"

row = {
    "task": task,
    "partition": partition,
    "num_blocks": int(num_blocks),
    "id_dim": float(id_dim),
    "id_dim_percent": int(id_dim_percent),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "epochs": int(epochs),
    "clipnorm": float(clipnorm),
    "val_mae": val_mae,
    "test_mae": test_mae,
    "duration_sec": int(float(duration_sec)),
    "status": status,
    "output_dir": output_dir,
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
  local cache_file="dimenetpp_${TASK}_cached_tensors.pkl"
  local scaler_file="scaler_${TASK}.pkl"
  if [[ ! -f "${cache_dir}/${cache_file}" ]]; then
    echo "$PRINT_PREFIX ERROR: missing tensor cache ${cache_dir}/${cache_file}" >&2
    return 1
  fi

  local run_name="dimenetpp_${TASK}_${WRAPPER}_${PART}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${PART}/${run_name}"
  local log_file="${out_dir}/train.log"
  local metadata_json="${out_dir}/metadata.json"
  mkdir -p "$out_dir"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "$PRINT_PREFIX skipping completed ${run_name}"
    return 0
  fi

  local start_time start_epoch_sec end_epoch_sec duration_sec exit_code
  start_time="$(date -Iseconds)"; start_epoch_sec="$(date +%s)"
  echo "$PRINT_PREFIX START ${run_name} at ${start_time}"

  "$DIMENET_PYTHON" "${CODE_DIR}/dimenet_run_${TASK}_v3.py" \
    --method "$WRAPPER" \
    --id_dim "$id_dim" \
    --num_blocks "$NUM_BLOCKS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$model_seed" \
    --split_seed "$split_seed" \
    --gpu "$GPU" \
    --cache_dir "$cache_dir" \
    --cache_file "$cache_file" \
    --scaler_file "$scaler_file" \
    --out_dir "$out_dir" \
    --clipnorm "$CLIPNORM" \
    > "$log_file" 2>&1
  exit_code=$?
  end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" \
    "$TASK" "$PART" "$NUM_BLOCKS" "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" \
    "$EPOCHS" "$CLIPNORM" "$out_dir" "$duration_sec" "$exit_code" "$write_summary"

  echo "$PRINT_PREFIX DONE ${run_name} exit_code=${exit_code} duration=${duration_sec}s"
}

echo "$PRINT_PREFIX task=${TASK} partition=${PART} num_blocks=${NUM_BLOCKS} epochs=${EPOCHS} clipnorm=${CLIPNORM} batch_size=${BATCH_SIZE}"
echo "$PRINT_PREFIX seeds=${SEEDS_SPEC} dims=${DIMS[*]}"
echo "$PRINT_PREFIX cache_root=${CACHE_ROOT}"
echo "$PRINT_PREFIX result_root=${RESULT_ROOT}"

SELECTED_RUNS="${DIMENET_RUNS:-}"
if [[ -n "$SELECTED_RUNS" ]]; then
  echo "$PRINT_PREFIX explicit selection: ${SELECTED_RUNS}"
  for item in $SELECTED_RUNS; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    split_seed="${split_seed:-$model_seed}"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX SELECTED RUNS DONE. Summary: ${SUMMARY_CSV}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#RUNS[@]} )); then
    echo "$PRINT_PREFIX ERROR: array index ${idx} out of range 0..$(( ${#RUNS[@]} - 1 ))" >&2
    exit 1
  fi
  IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "${RUNS[$idx]}"
  run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
else
  for item in "${RUNS[@]}"; do
    IFS=':' read -r id_dim id_dim_percent model_seed split_seed <<< "$item"
    run_one "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "1"
  done
  echo "$PRINT_PREFIX ALL DONE. Summary: ${SUMMARY_CSV}"
fi
