#!/usr/bin/env bash
# CGCNN Fastfood intrinsic-dimension sweep on the MP eform / band-gap caches.
# Method = fastfood, optimizer = Adam, default 350 epochs.
#
# --id-dim is passed as a FRACTION (e.g. 0.45) so RandomSubspaceWrapper maps it to
# round(fraction * D) -- a true percentage of the model's D params. (A bare int would
# mean an absolute dim count, which is NOT what we want.)
#
# Run selection:
#   CGCNN_SMOKE=1              -> run ONE config (dim 1.0, seed 123/1123) for timing
#   SLURM_ARRAY_TASK_ID=<idx>  -> run only RUNS[idx]
#   neither                    -> run every RUN sequentially
#
# Required: CGCNN_TASK=eform|bandgap
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

TASK="${CGCNN_TASK:?set CGCNN_TASK=eform|bandgap}"
EPOCHS="${CGCNN_EPOCHS:-350}"
BATCH_SIZE="${CGCNN_BS:-256}"
WORKERS="${CGCNN_WORKERS:-4}"
PRINT_FREQ="${CGCNN_PRINT_FREQ:-200}"
N_CONV=4
ATOM_FEA_LEN=56
H_FEA_LEN=64
N_H=3
WRAPPER="fastfood"
OPTIM="Adam"
PYTHON="python"

if [[ "$TASK" == "eform" ]]; then
  CACHED_DATA_DIR="cached_mp/tensors/mp_eform"
  MB_TASK="mp_eform"
  DIM_ORDER=(100 80 60 50 45 20 10)
  RUNS=(
    "1.0:100:123:1123"  "1.0:100:456:1456"
    "0.8:80:123:1123"   "0.8:80:456:1456"
    "0.6:60:123:1123"   "0.6:60:456:1456"   "0.6:60:789:1789"
    "0.5:50:123:1123"   "0.5:50:456:1456"
    "0.45:45:123:1123"  "0.45:45:456:1456"  "0.45:45:789:1789"
    "0.2:20:123:1123"   "0.2:20:456:1456"
    "0.1:10:123:1123"   "0.1:10:456:1456"
  )
elif [[ "$TASK" == "bandgap" ]]; then
  CACHED_DATA_DIR="cached_mp/tensors/mp_bandgap"
  MB_TASK="mp_bandgap"
  DIM_ORDER=(100 80 60 50 45 20 10 5)
  RUNS=(
    "1.0:100:123:1123"  "1.0:100:456:1456"
    "0.8:80:123:1123"   "0.8:80:456:1456"
    "0.6:60:123:1123"   "0.6:60:456:1456"   "0.6:60:789:1789"
    "0.5:50:123:1123"   "0.5:50:456:1456"
    "0.45:45:123:1123"  "0.45:45:456:1456"  "0.45:45:789:1789"
    "0.2:20:123:1123"   "0.2:20:456:1456"
    "0.1:10:123:1123"   "0.1:10:456:1456"
    "0.05:5:123:1123"   "0.05:5:456:1456"
  )
else
  echo "Unknown CGCNN_TASK=$TASK (expected eform|bandgap)" >&2; exit 1
fi

RESULT_ROOT="${CGCNN_RESULT_ROOT:-results_mp_${TASK}_${WRAPPER}}"
SUMMARY_CSV="${RESULT_ROOT}/cgcnn_mp_${TASK}_${WRAPPER}_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,wrapper,optim,id_dim_frac,id_dim_percent,model_seed,split_seed,epochs,batch_size,n_conv,atom_fea_len,h_fea_len,n_h,output_dir,predictions_csv,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

# --- group runs by dim percent so one job can run all seeds for a dim ---
specs_for_dim() { local want="$1" s f p; for s in "${RUNS[@]}"; do IFS=':' read -r f p _ _ <<< "$s"; [[ "$p" == "$want" ]] && printf '%s\n' "$s"; done; }
nseeds_for_dim() { specs_for_dim "$1" | grep -c .; }
dims_in_set() { # $1 = double|triple|all : echo dims (in DIM_ORDER) with 2 / >=3 / any seeds
  local d c
  for d in "${DIM_ORDER[@]}"; do
    c=$(nseeds_for_dim "$d")
    case "$1" in
      double) [[ "$c" -eq 2 ]] && printf '%s\n' "$d" ;;
      triple) [[ "$c" -ge 3 ]] && printf '%s\n' "$d" ;;
      *)      printf '%s\n' "$d" ;;
    esac
  done
}

# Decide which runs to execute
SELECTED=()
if [[ -n "${CGCNN_RUNS:-}" ]]; then
  # explicit space-separated "frac:pct:model_seed:split_seed" entries (extends the sweep)
  read -ra SELECTED <<< "$CGCNN_RUNS"
elif [[ "${CGCNN_SMOKE:-0}" == "1" ]]; then
  SELECTED=("1.0:100:123:1123")
  RESULT_ROOT="${RESULT_ROOT}/_smoke"; mkdir -p "$RESULT_ROOT"
elif [[ -n "${CGCNN_DIM_GROUP:-}" ]]; then
  # run all seeds of one explicit dim percent (e.g. CGCNN_DIM_GROUP=60)
  mapfile -t SELECTED < <(specs_for_dim "$CGCNN_DIM_GROUP")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # array task -> one dim group (all its seeds), within the chosen set (double|triple|all)
  GROUP_SET="${CGCNN_GROUP_SET:-all}"
  mapfile -t SEL_DIMS < <(dims_in_set "$GROUP_SET")
  DIM="${SEL_DIMS[$SLURM_ARRAY_TASK_ID]:-}"
  [[ -z "$DIM" ]] && { echo "[ERR] no dim for array idx $SLURM_ARRAY_TASK_ID in set '$GROUP_SET' (have ${#SEL_DIMS[@]} dims)" >&2; exit 1; }
  mapfile -t SELECTED < <(specs_for_dim "$DIM")
  echo "[INFO] group_set=$GROUP_SET array_idx=$SLURM_ARRAY_TASK_ID dim=${DIM}% seeds=${#SELECTED[@]}"
else
  SELECTED=("${RUNS[@]}")
fi

echo "[INFO] TASK=$TASK epochs=$EPOCHS bs=$BATCH_SIZE optim=$OPTIM wrapper=$WRAPPER nruns=${#SELECTED[@]}"

run_one() {
  local spec="$1"
  IFS=':' read -r FRAC PCT MODEL_SEED SPLIT_SEED <<< "$spec"
  local EXP_NAME="cgcnn_mp_${TASK}_${WRAPPER}_dim${PCT}_dataseed${SPLIT_SEED}_modelseed${MODEL_SEED}"
  local OUT_DIR="${RESULT_ROOT}/${EXP_NAME}"
  local LOG_FILE="${OUT_DIR}/train.log"
  local META_JSON="${OUT_DIR}/metadata.json"
  mkdir -p "$OUT_DIR"

  local START_TIME; START_TIME="$(date -Iseconds)"
  local T0=$SECONDS
  echo "[START] $EXP_NAME (frac=$FRAC pct=$PCT model_seed=$MODEL_SEED split_seed=$SPLIT_SEED) at $START_TIME"

  "$PYTHON" main.py \
    --cached-data-dir "$CACHED_DATA_DIR" \
    --matbench-task "$MB_TASK" \
    --output-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --optim "$OPTIM" \
    --n-conv "$N_CONV" \
    --atom-fea-len "$ATOM_FEA_LEN" \
    --h-fea-len "$H_FEA_LEN" \
    --n-h "$N_H" \
    --subspace-method "$WRAPPER" \
    --id-dim "$FRAC" \
    --data-seed "$SPLIT_SEED" \
    --random-seed "$MODEL_SEED" \
    --print-freq "$PRINT_FREQ" 2>&1 | tee "$LOG_FILE"
  local EXIT_CODE=${PIPESTATUS[0]}
  local END_TIME; END_TIME="$(date -Iseconds)"
  local DURATION=$((SECONDS - T0))
  local STATUS="success"; [[ "$EXIT_CODE" -ne 0 ]] && STATUS="failed"

  local PRED_CSV="${OUT_DIR}/${MB_TASK}_test_results.csv"

  "$PYTHON" - "$SUMMARY_CSV" "$META_JSON" "$LOG_FILE" "$PRED_CSV" "$TASK" "$WRAPPER" "$OPTIM" \
    "$FRAC" "$PCT" "$MODEL_SEED" "$SPLIT_SEED" "$EPOCHS" "$BATCH_SIZE" "$N_CONV" "$ATOM_FEA_LEN" \
    "$H_FEA_LEN" "$N_H" "$OUT_DIR" "$DURATION" "$STATUS" "$EXIT_CODE" "$START_TIME" "$END_TIME" <<'PY'
import csv, json, math, re, sys
from pathlib import Path
(summary_csv, meta_json, log_file, pred_csv, task, wrapper, optim, frac, pct, model_seed,
 split_seed, epochs, batch_size, n_conv, atom_fea_len, h_fea_len, n_h, out_dir, duration,
 status, exit_code, start_time, end_time) = sys.argv[1:]
log = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
val = re.findall(r"^\s*\*\s+MAE\s+([-+0-9.eE]+)\s*$", log, flags=re.M)
best_val = min((float(x) for x in val), default=math.nan)
test_mae = math.nan
m = re.search(r"test metric is:\s*([-+0-9.eE]+)", log)
if m: test_mae = float(m.group(1))
row = dict(task=task, wrapper=wrapper, optim=optim, id_dim_frac=float(frac),
           id_dim_percent=int(pct), model_seed=int(model_seed), split_seed=int(split_seed),
           epochs=int(epochs), batch_size=int(batch_size), n_conv=int(n_conv),
           atom_fea_len=int(atom_fea_len), h_fea_len=int(h_fea_len), n_h=int(n_h),
           output_dir=out_dir, predictions_csv=(pred_csv if Path(pred_csv).exists() else ""),
           best_val_mae=best_val, test_mae=test_mae, duration_sec=int(duration),
           status=status, exit_code=int(exit_code), start_time=start_time, end_time=end_time)
Path(meta_json).write_text(json.dumps(row, indent=2))
import os
# append to shared summary under an flock to avoid interleaving across array tasks
lock = summary_csv + ".lock"
with open(lock, "w") as lf:
    try:
        import fcntl; fcntl.flock(lf, fcntl.LOCK_EX)
    except Exception:
        pass
    with open(summary_csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
print(f"[DONE] {Path(out_dir).name} status={status} best_val_mae={best_val} test_mae={test_mae} dur={duration}s")
PY

  rm -f "$OUT_DIR"/*.pth.tar
}

for spec in "${SELECTED[@]}"; do
  run_one "$spec"
done
echo "[ALL DONE] $TASK -> $SUMMARY_CSV"
