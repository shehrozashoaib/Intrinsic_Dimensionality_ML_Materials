#!/usr/bin/env bash
# CGCNN Fastfood intrinsic-dim sweep for a Matbench task, FIXED for percentages + Adam.
#   - id-dim passed as FRACTION (0.01..1.0) so it's a true % of the model's D params
#     (the old scripts passed bare ints => absolute dims, which all collapsed ~ the same).
#   - optimizer = Adam.
#   - matched 3 seeds: data_seed = model_seed in {123,456,789} (3 distinct splits).
# One job runs all 12 dims x 3 seeds for the given task (they're only minutes each).
#
# Required env: CGCNN_MB_TASK = dielectric | phonons | log_kvrh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

CONDA_ENV="py312"; PYTHON="python"
SHORT="${CGCNN_MB_TASK:?set CGCNN_MB_TASK=dielectric|phonons|log_kvrh}"
TASK="matbench_${SHORT}"
case "$SHORT" in
  dielectric) EPOCHS=80  ;;
  phonons)    EPOCHS=80  ;;
  log_kvrh)   EPOCHS=120 ;;
  *) echo "Unknown CGCNN_MB_TASK=$SHORT" >&2; exit 1 ;;
esac
EPOCHS="${CGCNN_EPOCHS_OVERRIDE:-$EPOCHS}"

CACHED_DATA_DIR="cached_matbench/tensors"
BATCH_SIZE=512; N_CONV=4; ATOM_FEA_LEN=56; H_FEA_LEN=64; N_H=3
WRAPPER="fastfood"; OPTIM="Adam"; PRINT_FREQ=1000
SEEDS=(123 456 789)
# id-dim percentages -> fractions
DIM_SPECS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")

RESULT_ROOT="results_matbench_${SHORT}_${WRAPPER}"
SUMMARY_CSV="${RESULT_ROOT}/cgcnn_matbench_${SHORT}_${WRAPPER}_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,wrapper,optim,id_dim_frac,id_dim_percent,data_seed,model_seed,epochs,batch_size,n_conv,atom_fea_len,h_fea_len,n_h,output_dir,predictions_csv,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

echo "[INFO] TASK=$TASK epochs=$EPOCHS optim=$OPTIM wrapper=$WRAPPER seeds=${SEEDS[*]} dims=${#DIM_SPECS[@]}"

run_one() {
  local frac="$1" pct="$2" seed="$3"
  local exp="cgcnn_matbench_${SHORT}_${WRAPPER}_dim${pct}_dataseed${seed}_modelseed${seed}"
  local out="${RESULT_ROOT}/${exp}"
  local log="${out}/train.log" meta="${out}/metadata.json"
  local pred_src="${out}/${TASK}_test_results.csv" pred_dst="${out}/${exp}_test_results.csv"
  mkdir -p "$out"
  if [[ -f "$meta" ]] && grep -q '"status": "success"' "$meta"; then
    echo "[skip] $exp (already success)"; return 0
  fi
  local t0=$SECONDS start; start="$(date -Iseconds)"
  echo "[START] $exp (frac=$frac pct=$pct seed=$seed) $start"
  conda run -n "$CONDA_ENV" "$PYTHON" main.py \
    --cached-data-dir "$CACHED_DATA_DIR" --matbench-task "$TASK" --output-dir "$out" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --optim "$OPTIM" \
    --n-conv "$N_CONV" --atom-fea-len "$ATOM_FEA_LEN" --h-fea-len "$H_FEA_LEN" --n-h "$N_H" \
    --subspace-method "$WRAPPER" --id-dim "$frac" \
    --data-seed "$seed" --random-seed "$seed" --print-freq "$PRINT_FREQ" 2>&1 | tee "$log"
  local ec=${PIPESTATUS[0]} end; end="$(date -Iseconds)"
  local dur=$((SECONDS - t0)) status="success"; [[ "$ec" -ne 0 ]] && status="failed"
  [[ -f "$pred_src" ]] && mv -f "$pred_src" "$pred_dst"
  "$PYTHON" - "$SUMMARY_CSV" "$meta" "$log" "$pred_dst" "$TASK" "$WRAPPER" "$OPTIM" "$frac" "$pct" \
    "$seed" "$EPOCHS" "$BATCH_SIZE" "$N_CONV" "$ATOM_FEA_LEN" "$H_FEA_LEN" "$N_H" "$out" "$dur" \
    "$status" "$ec" "$start" "$end" <<'PY'
import csv, json, math, re, sys
from pathlib import Path
(summary, meta, log, pred, task, wrapper, optim, frac, pct, seed, epochs, bs, nconv, atom,
 h, nh, out, dur, status, ec, start, end) = sys.argv[1:]
txt = Path(log).read_text(errors="replace") if Path(log).exists() else ""
val = re.findall(r"^\s*\*\s+MAE\s+([-+0-9.eE]+)\s*$", txt, flags=re.M)
best_val = min((float(x) for x in val), default=math.nan)
test = math.nan
m = re.search(r"test metric is:\s*([-+0-9.eE]+)", txt)
if m: test = float(m.group(1))
row = dict(task=task, wrapper=wrapper, optim=optim, id_dim_frac=float(frac), id_dim_percent=int(pct),
           data_seed=int(seed), model_seed=int(seed), epochs=int(epochs), batch_size=int(bs),
           n_conv=int(nconv), atom_fea_len=int(atom), h_fea_len=int(h), n_h=int(nh), output_dir=out,
           predictions_csv=(pred if Path(pred).exists() else ""), best_val_mae=best_val, test_mae=test,
           duration_sec=int(dur), status=status, exit_code=int(ec), start_time=start, end_time=end)
Path(meta).write_text(json.dumps(row, indent=2))
lock = summary + ".lock"
with open(lock, "w") as lf:
    try:
        import fcntl; fcntl.flock(lf, fcntl.LOCK_EX)
    except Exception: pass
    with open(summary, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
print(f"[DONE] {Path(out).name} status={status} best_val_mae={best_val} test_mae={test} dur={dur}s")
PY
  rm -f "$out"/*.pth.tar
}

for spec in "${DIM_SPECS[@]}"; do
  IFS=':' read -r frac pct <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    run_one "$frac" "$pct" "$seed"
  done
done
echo "[ALL DONE] $TASK -> $SUMMARY_CSV"
