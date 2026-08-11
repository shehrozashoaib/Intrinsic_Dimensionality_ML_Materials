#!/usr/bin/env bash
# CGCNN Fastfood intrinsic-dimension sweep on matbench_mp_is_metal (BINARY CLASSIFICATION).
# 1 conv layer (~85k params), fastfood, NLLLoss (via --task classification), metric = ROC-AUC.
#
# --id-dim is passed as a FRACTION (e.g. 0.60) so RandomSubspaceWrapper maps it to
# round(fraction * D) -- a true percentage of the model's D params.
#
# Run selection:
#   CGCNN_RUNS="frac:pct:model_seed:split_seed ..."  -> run exactly those (space separated)
#   CGCNN_SMOKE=1                                     -> one short config into _smoke/ (honors CGCNN_EPOCHS/CGCNN_DATA_LIMIT-less)
#   SLURM_ARRAY_TASK_ID=<idx>                         -> run only RUNS[idx] (one dim+seed)
#   neither                                          -> run every RUN sequentially
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

EPOCHS="${CGCNN_EPOCHS:-350}"
BATCH_SIZE="${CGCNN_BS:-256}"
WORKERS="${CGCNN_WORKERS:-8}"
PRINT_FREQ="${CGCNN_PRINT_FREQ:-200}"
N_CONV=1
ATOM_FEA_LEN=82
H_FEA_LEN=128
N_H=3
WRAPPER="fastfood"
OPTIM="Adam"
PYTHON="python"

CACHED_DATA_DIR="cached_mp/tensors/mp_is_metal"
# One-cycle arm: CGCNN_LR_SCHEDULE=onecycle (+ CGCNN_LR / CGCNN_PCT_START).
# Defaults keep the historical MultiStepLR so the existing 44 runs stay reproducible.
LR_SCHEDULE="${CGCNN_LR_SCHEDULE:-multistep}"
LR="${CGCNN_LR:-0.001}"
PCT_START="${CGCNN_PCT_START:-0.3}"
MB_TASK="mp_is_metal"

# 8 dims x 4 seeds (data_seed = model_seed = split_seed, matching the CGCNN MP convention)
DIM_ORDER=(100 80 60 40 20 15 10 5)
SEEDS=(101 202 303 404)
RUNS=()
declare -A _FRAC=( [100]=1.0 [80]=0.8 [60]=0.6 [40]=0.4 [20]=0.2 [15]=0.15 [10]=0.1 [5]=0.05 )
for pct in "${DIM_ORDER[@]}"; do
  for s in "${SEEDS[@]}"; do
    RUNS+=("${_FRAC[$pct]}:${pct}:${s}:${s}")
  done
done

RESULT_ROOT="${CGCNN_RESULT_ROOT:-results_mp_is_metal_fastfood}"
SUMMARY_CSV="${RESULT_ROOT}/cgcnn_mp_is_metal_${WRAPPER}_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,wrapper,optim,id_dim_frac,id_dim_percent,model_seed,split_seed,epochs,batch_size,n_conv,atom_fea_len,h_fea_len,n_h,output_dir,predictions_csv,best_val_auc,test_auc,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

# Decide which runs to execute
SELECTED=()
if [[ -n "${CGCNN_RUNS:-}" ]]; then
  read -ra SELECTED <<< "$CGCNN_RUNS"
elif [[ "${CGCNN_SMOKE:-0}" == "1" ]]; then
  SELECTED=("1.0:100:101:101")
  RESULT_ROOT="${RESULT_ROOT}/_smoke"; mkdir -p "$RESULT_ROOT"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SPEC="${RUNS[$SLURM_ARRAY_TASK_ID]:-}"
  [[ -z "$SPEC" ]] && { echo "[ERR] no run for array idx $SLURM_ARRAY_TASK_ID (have ${#RUNS[@]})" >&2; exit 1; }
  SELECTED=("$SPEC")
  echo "[INFO] array_idx=$SLURM_ARRAY_TASK_ID -> $SPEC"
else
  SELECTED=("${RUNS[@]}")
fi

echo "[INFO] TASK=is_metal epochs=$EPOCHS bs=$BATCH_SIZE optim=$OPTIM wrapper=$WRAPPER nruns=${#SELECTED[@]}"

run_one() {
  local spec="$1"
  IFS=':' read -r FRAC PCT MODEL_SEED SPLIT_SEED <<< "$spec"
  local EXP_NAME="cgcnn_mp_is_metal_${WRAPPER}_dim${PCT}_dataseed${SPLIT_SEED}_modelseed${MODEL_SEED}"
  local OUT_DIR="${RESULT_ROOT}/${EXP_NAME}"
  local LOG_FILE="${OUT_DIR}/train.log"
  local META_JSON="${OUT_DIR}/metadata.json"
  mkdir -p "$OUT_DIR"

  # idempotent skip
  if [[ -f "$META_JSON" ]] && grep -q '"status": "success"' "$META_JSON" 2>/dev/null; then
    echo "[SKIP] $EXP_NAME already success"; return 0
  fi

  local START_TIME; START_TIME="$(date -Iseconds)"
  local T0=$SECONDS
  echo "[START] $EXP_NAME (frac=$FRAC pct=$PCT model_seed=$MODEL_SEED split_seed=$SPLIT_SEED) at $START_TIME"

  "$PYTHON" main.py \
    --task classification \
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
    --lr-schedule "$LR_SCHEDULE" \
    --lr "$LR" \
    --pct-start "$PCT_START" \
    --print-freq "$PRINT_FREQ" 2>&1 | tee "$LOG_FILE"
  local EXIT_CODE=${PIPESTATUS[0]}
  local END_TIME; END_TIME="$(date -Iseconds)"
  local DURATION=$((SECONDS - T0))
  local STATUS="success"; [[ "$EXIT_CODE" -ne 0 ]] && STATUS="failed"

  local PRED_CSV="${OUT_DIR}/${MB_TASK}_test_results.csv"

  "$PYTHON" - "$SUMMARY_CSV" "$META_JSON" "$LOG_FILE" "$PRED_CSV" "$WRAPPER" "$OPTIM" \
    "$FRAC" "$PCT" "$MODEL_SEED" "$SPLIT_SEED" "$EPOCHS" "$BATCH_SIZE" "$N_CONV" "$ATOM_FEA_LEN" \
    "$H_FEA_LEN" "$N_H" "$OUT_DIR" "$DURATION" "$STATUS" "$EXIT_CODE" "$START_TIME" "$END_TIME" \
    "$LR_SCHEDULE" "$LR" "$PCT_START" <<'PY'
import csv, json, math, re, sys
from pathlib import Path
(summary_csv, meta_json, log_file, pred_csv, wrapper, optim, frac, pct, model_seed,
 split_seed, epochs, batch_size, n_conv, atom_fea_len, h_fea_len, n_h, out_dir, duration,
 status, exit_code, start_time, end_time, lr_schedule, lr, pct_start) = sys.argv[1:]
log = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
# classification: main.py prints " * AUC <x>" / " ** AUC <x>" per validate(); best = MAX
auc = [float(x) for x in re.findall(r"\*\s+AUC\s+([-+0-9.eE]+)", log)]
best_val_auc = max(auc) if auc else math.nan
test_auc = math.nan
m = re.search(r"test metric is:\s*([-+0-9.eE]+)", log)   # task-agnostic; for classification = AUC
if m: test_auc = float(m.group(1))
row = dict(task="is_metal", wrapper=wrapper, optim=optim, id_dim_frac=float(frac),
           id_dim_percent=(int(float(pct)) if float(pct)==int(float(pct)) else float(pct)), model_seed=int(model_seed), split_seed=int(split_seed),
           epochs=int(epochs), batch_size=int(batch_size), n_conv=int(n_conv),
           atom_fea_len=int(atom_fea_len), h_fea_len=int(h_fea_len), n_h=int(n_h),
           output_dir=out_dir, predictions_csv=(pred_csv if Path(pred_csv).exists() else ""),
           best_val_auc=best_val_auc, test_auc=test_auc, duration_sec=int(duration),
           status=status, exit_code=int(exit_code), start_time=start_time, end_time=end_time,
           lr_schedule=lr_schedule, lr=float(lr), pct_start=float(pct_start))
# Collapse guard. CLASSIFICATION: the last column is the class-1 PROBABILITY, so a healthy model
# spreads its scores across [0,1] while a collapsed one emits one constant probability. The 0.05
# threshold is the same one used on the regression sweeps; a real is_metal model sits far above it
# (the task is 43.5/56.5 balanced, so even a mediocre model spreads widely).
try:
    _pred = []
    with open(pred_csv) as _f:
        for _r in csv.reader(_f):
            if len(_r) < 2:
                continue
            try:
                _pred.append(float(_r[-1]))
            except ValueError:
                pass
    if len(_pred) > 1:
        _mu = sum(_pred) / len(_pred)
        row["pred_spread"] = (sum((x - _mu) ** 2 for x in _pred) / len(_pred)) ** 0.5
        row["degenerate"] = row["pred_spread"] < 0.05
except Exception:
    pass
Path(meta_json).write_text(json.dumps(row, indent=2))
lock = summary_csv + ".lock"
with open(lock, "w") as lf:
    try:
        import fcntl; fcntl.flock(lf, fcntl.LOCK_EX)
    except Exception:
        pass
    with open(summary_csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
print(f"[DONE] {Path(out_dir).name} status={status} best_val_auc={best_val_auc} test_auc={test_auc} dur={duration}s")
PY

  rm -f "$OUT_DIR"/*.pth.tar
}

for spec in "${SELECTED[@]}"; do
  run_one "$spec"
done
echo "[ALL DONE] is_metal -> $SUMMARY_CSV"
