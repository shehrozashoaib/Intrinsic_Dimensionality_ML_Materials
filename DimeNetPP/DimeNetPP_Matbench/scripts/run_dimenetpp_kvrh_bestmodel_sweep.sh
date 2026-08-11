#!/usr/bin/env bash
# DimeNet++ Fastfood intrinsic-dimension sweep with BEST-MODEL evaluation, matbench log_kvrh.
#
# Why this exists: the earlier sweeps evaluated the FINAL-epoch model, which was overfit and biased
# the ID curves. Every run here trains with ModelCheckpoint(save_best_only) + EarlyStopping and
# restores the best-val_loss weights before predicting (--restore_best --patience N).
#
# One job = one dataset size: full 12-dim x SEEDS grid. Per-dataset summary CSV avoids races
# between concurrently running jobs.
#
# Env vars:
#   DIMENET_DATASET    full | quarter_1 | quarter_2 | tenth_1 | tenth_2   (REQUIRED)
#   DIMENET_SEEDS      "model_seed:split_seed ..."
#                      default: full       -> "123:1123 456:1456 789:1789"  (3 seeds)
#                               partitions -> "123:1123 456:1456"            (2 seeds)
#   DIMENET_EPOCHS     default 180
#   DIMENET_PATIENCE   default 50   -- EarlyStopping on val_loss, 0=off
#   DIMENET_CLIPNORM   default 0    (NO gradient clipping -- required for this study)
#   DIMENET_BATCH_SIZE default 64
#   DIMENET_GPU        default 0
#
# Model config is FIXED: --num_blocks 1 --int_emb_size 64 (the ~85k-param 1-layer baseline).
# --restore_best is ALWAYS passed.
#
# Usage:
#   DIMENET_DATASET=full     bash scripts/run_dimenetpp_kvrh_bestmodel_sweep.sh
#   DIMENET_DATASET=tenth_1  bash scripts/run_dimenetpp_kvrh_bestmodel_sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh" || exit 1

DATASET="${DIMENET_DATASET:-}"
if [[ -z "$DATASET" ]]; then
  echo "ERROR: DIMENET_DATASET (full|quarter_1|quarter_2|tenth_1|tenth_2) is required." >&2
  exit 1
fi
case "$DATASET" in
  full|quarter_1|quarter_2|tenth_1|tenth_2) ;;
  *)
    echo "ERROR: DIMENET_DATASET must be one of full|quarter_1|quarter_2|tenth_1|tenth_2 (got '${DATASET}')." >&2
    exit 1
    ;;
esac

TASK="kvrh"              # short name used in file/dir names and the runner filename
META_TASK="log_kvrh"     # value written to metadata.json "task"

EPOCHS="${DIMENET_EPOCHS:-180}"
PATIENCE="${DIMENET_PATIENCE:-50}"
BATCH_SIZE="${DIMENET_BATCH_SIZE:-64}"
GPU="${DIMENET_GPU:-0}"
CLIPNORM="${DIMENET_CLIPNORM:-0}"
# One-cycle arm: DIMENET_LR_SCHEDULE=onecycle + DIMENET_PATIENCE=0 (ES off so the anneal
# completes). Defaults keep the historical constant-lr behaviour untouched.
LR="${DIMENET_LR:-0.001}"
LR_SCHEDULE="${DIMENET_LR_SCHEDULE:-none}"
# DIMENET_RESTORE_BEST=0 -> reproduce the OLD protocol exactly (final-epoch model, no early
# stopping). Everything else identical, so it is a clean A/B against the best-model runs.
RESTORE_BEST="${DIMENET_RESTORE_BEST:-1}"
if [[ "$RESTORE_BEST" == "0" ]]; then RB_FLAGS=(); PATIENCE=0; else RB_FLAGS=(--restore_best); fi
NUM_BLOCKS=1
INT_EMB_SIZE=64
WRAPPER="${DIMENET_METHOD:-fastfood}"          # fastfood | dense
TRAIN_MODE="${DIMENET_TRAIN_MODE:-wrapped}"    # wrapped | base (base = NO subspace wrapper)
ORTHO="${DIMENET_ORTHO:-0}"                    # 1 -> --orthonormal
ROTATE="${DIMENET_ROTATE:-0}"                  # 1 -> --full_rotation (requires d == D)
# Orthonormal Q is built by an external torch process. The runner's default --torch_python is a
# container path (/venv/main/bin/python) that does not exist here, so point it at py312's torch.
# CPU QR is 3-6.5 h per build at high dims vs minutes on GPU -> pytorch_gpu (needs an A100).
ORTHO_BACKEND="${DIMENET_ORTHO_BACKEND:-pytorch_gpu}"
TORCH_PYTHON="${DIMENET_TORCH_PYTHON:-/ibex/user/shoaibsa/miniconda3/envs/py312/bin/python}"
TORCH_Q_CACHE="${DIMENET_TORCH_Q_CACHE:-${TMPDIR:-/tmp}/qcache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TORCH_Q_CACHE"
DENSE_BLOCK_COLS="${DIMENET_DENSE_BLOCK_COLS:-512}"

# Cache layout: FULL lives in cached_tensors_dimenetpp_kvrh/, partitions in
# cached_tensors_dimenetpp_part_kvrh_<dataset>/ -- both with the same pkl/scaler names.
if [[ "$DATASET" == "full" ]]; then
  DEFAULT_CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_${TASK}"
else
  # DIMENET_FIXEDTEST=1 -> fixed-test 2-partition caches (every partition of a seed is
  # scored on that seed's FULL test set); built by build_fixedtest_partition_caches.py.
  if [[ "${DIMENET_FIXEDTEST:-0}" == "1" ]]; then
    DEFAULT_CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_${TASK}_fixedtest2/${DATASET}"
  else
    DEFAULT_CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_part_${TASK}_${DATASET}"
  fi
fi
CACHE_ROOT="${DIMENET_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
CACHE_FILE="dimenetpp_${TASK}_cached_tensors.pkl"
SCALER_FILE="scaler_${TASK}.pkl"

RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_bestmodel/${META_TASK}/${DATASET}}"
SUMMARY_CSV="${RESULT_ROOT}/dimenetpp_${TASK}_${DATASET}_bestmodel_summary.csv"
PRINT_PREFIX="[DIMENET-BEST-KVRH-${DATASET}]"

DIMS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")

# Optional override: DIMENET_DIMS="1.0:100 0.8:80" -> run only those dims (lets us split
# long jobs across the <=20h walltime without changing the sweep definition).
if [[ -n "${DIMENET_DIMS:-}" ]]; then
  read -ra DIMS <<< "$DIMENET_DIMS"
fi

# FULL datasets get 3 seeds; partitions get 2 (2 partitions x 2 seeds = 4 samples per (dim, size)).
if [[ "$DATASET" == "full" ]]; then
  DEFAULT_SEEDS="123:1123 456:1456 789:1789"
else
  DEFAULT_SEEDS="123:1123 456:1456"
fi
SEEDS_SPEC="${DIMENET_SEEDS:-$DEFAULT_SEEDS}"
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

# Hard-error up front if any required tensor cache is missing -- never silently train on nothing.
for s in "${SEEDS[@]}"; do
  IFS=':' read -r ms ss <<< "$s"
  ss="${ss:-$ms}"
  if [[ ! -f "${CACHE_ROOT}/splitseed${ss}/${CACHE_FILE}" ]]; then
    echo "$PRINT_PREFIX ERROR: missing tensor cache ${CACHE_ROOT}/splitseed${ss}/${CACHE_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${CACHE_ROOT}/splitseed${ss}/${SCALER_FILE}" ]]; then
    echo "$PRINT_PREFIX ERROR: missing scaler ${CACHE_ROOT}/splitseed${ss}/${SCALER_FILE}" >&2
    exit 1
  fi
done

mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,dataset,num_blocks,id_dim,id_dim_percent,model_seed,split_seed,epochs,patience,clipnorm,restore_best,epochs_run,best_val_loss,final_val_loss,val_mae,test_mae,duration_sec,status,output_dir" > "$SUMMARY_CSV"
fi

append_metadata() {
  local summary_csv="$1"; local metadata_json="$2"; local log_file="$3"
  local task="$4"; local dataset="$5"; local num_blocks="$6"
  local id_dim="$7"; local id_dim_percent="$8"; local model_seed="$9"
  local split_seed="${10}"; local epochs="${11}"; local patience="${12}"; local clipnorm="${13}"
  local output_dir="${14}"; local duration_sec="${15}"; local exit_code="${16}"
  local write_summary="${17}"

  "$DIMENET_PYTHON" - "$summary_csv" "$metadata_json" "$log_file" "$task" "$dataset" \
    "$num_blocks" "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" "$epochs" \
    "$patience" "$clipnorm" "$output_dir" "$duration_sec" "$exit_code" "$write_summary" \
    "$LR" "$LR_SCHEDULE" "$WRAPPER" "$TRAIN_MODE" "$ORTHO" "$ROTATE" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

(summary_csv, metadata_json, log_file, task, dataset, num_blocks, id_dim, id_dim_percent,
 model_seed, split_seed, epochs, patience, clipnorm, output_dir, duration_sec, exit_code,
 write_summary, lr, lr_schedule, method, train_mode, ortho, rotate) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

def last_float(pattern):
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None

val_mae = last_float(r"Validation MAE \(original units\):\s*([0-9.eE+-]+)")
test_mae = last_float(r"Test MAE:\s*([0-9.eE+-]+)")

# The runner prints e.g.
#   [restore_best] RESTORED best weights (best val_loss=0.288650, final val_loss=0.364580, epochs_run=137)
# Some runners omit epochs_run; fall back to the last Keras "Epoch N/M" line.
best_val_loss = final_val_loss = epochs_run = None
rb = re.findall(r"\[restore_best\][^\n]*", text)
if rb:
    line = rb[-1]
    m = re.search(r"best val_loss=([0-9.eE+-]+)", line)
    if m:
        best_val_loss = float(m.group(1))
    m = re.search(r"final val_loss=([0-9.eE+-]+)", line)
    if m:
        final_val_loss = float(m.group(1))
    m = re.search(r"epochs_run=([0-9]+)", line)
    if m:
        epochs_run = int(m.group(1))
if epochs_run is None:
    m = re.findall(r"^Epoch (\d+)/\d+", text, flags=re.M)
    if m:
        epochs_run = int(m[-1])

status = "success" if (int(exit_code) == 0 and test_mae is not None and not math.isnan(test_mae)) else "failed"

row = {
    "task": task,
    "dataset": dataset,
    "num_blocks": int(num_blocks),
    "id_dim": float(id_dim),
    "id_dim_percent": int(id_dim_percent),
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "epochs": int(epochs),
    "patience": int(patience),
    "clipnorm": float(clipnorm),
    "method": method,
    "train_mode": train_mode,
    "orthonormal": ortho == "1",
    "full_rotation": rotate == "1",
    "lr": float(lr),
    "lr_schedule": lr_schedule,
    "restore_best": True,
    "epochs_run": epochs_run,
    "best_val_loss": best_val_loss,
    "final_val_loss": final_val_loss,
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
print(f"[metadata] {Path(output_dir).name} status={status} val_mae={val_mae} test_mae={test_mae} "
      f"epochs_run={epochs_run} best_val_loss={best_val_loss}")
PY
}

run_one() {
  local id_dim="$1"; local id_dim_percent="$2"; local model_seed="$3"; local split_seed="$4"; local write_summary="$5"
  local cache_dir="${CACHE_ROOT}/splitseed${split_seed}"
  if [[ ! -f "${cache_dir}/${CACHE_FILE}" ]]; then
    echo "$PRINT_PREFIX ERROR: missing tensor cache ${cache_dir}/${CACHE_FILE}" >&2
    exit 1
  fi

  local run_name="dimenetpp_${TASK}_bestmodel_${DATASET}_dim${id_dim_percent}_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  local out_dir="${RESULT_ROOT}/${run_name}"
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
    --train_mode "$TRAIN_MODE" \
    --dense_block_cols "$DENSE_BLOCK_COLS" \
    --orthonormal_backend "$ORTHO_BACKEND" \
    --torch_python "$TORCH_PYTHON" \
    --torch_q_cache_dir "$TORCH_Q_CACHE" \
    $([[ "$ORTHO" == "1" ]] && echo --orthonormal) \
    $([[ "$ROTATE" == "1" ]] && echo --full_rotation) \
    --id_dim "$id_dim" \
    --num_blocks "$NUM_BLOCKS" \
    --int_emb_size "$INT_EMB_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$model_seed" \
    --split_seed "$split_seed" \
    --gpu "$GPU" \
    --cache_dir "$cache_dir" \
    --cache_file "$CACHE_FILE" \
    --scaler_file "$SCALER_FILE" \
    --out_dir "$out_dir" \
    --clipnorm "$CLIPNORM" \
    "${RB_FLAGS[@]}" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    --lr_schedule "$LR_SCHEDULE" \
    > "$log_file" 2>&1
  exit_code=$?
  end_epoch_sec="$(date +%s)"
  duration_sec=$((end_epoch_sec - start_epoch_sec))

  append_metadata "$SUMMARY_CSV" "$metadata_json" "$log_file" \
    "$META_TASK" "$DATASET" "$NUM_BLOCKS" "$id_dim" "$id_dim_percent" "$model_seed" "$split_seed" \
    "$EPOCHS" "$PATIENCE" "$CLIPNORM" "$out_dir" "$duration_sec" "$exit_code" "$write_summary"

  echo "$PRINT_PREFIX DONE ${run_name} exit_code=${exit_code} duration=${duration_sec}s"
}

echo "$PRINT_PREFIX task=${META_TASK} dataset=${DATASET} num_blocks=${NUM_BLOCKS} int_emb_size=${INT_EMB_SIZE} epochs=${EPOCHS} patience=${PATIENCE} clipnorm=${CLIPNORM} restore_best=1 batch_size=${BATCH_SIZE} lr=${LR} lr_schedule=${LR_SCHEDULE} method=${WRAPPER} train_mode=${TRAIN_MODE} ortho=${ORTHO} rotate=${ROTATE} cache_root=${CACHE_ROOT}"
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
