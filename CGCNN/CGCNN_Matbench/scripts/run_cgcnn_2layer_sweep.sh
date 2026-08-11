#!/usr/bin/env bash
# CGCNN "doubled" model (n_conv=2) intrinsic-dim SWEEP (not just 100%).
# id-dim is passed as a FRACTION (0.01..1.0) so it is a true % of the model's D params.
# Writes into the SAME results_2layer_fastfood/<task>/ tree as run_cgcnn_2layer_full.sh, with the
# identical dim100 dir names, so already-finished 100% runs are skipped (idempotent resume).
# Required: CGCNN_TASK=dielectric|phonons|log_kvrh|eform|bandgap
# Required: CGCNN_SEEDS="s1 s2 ..." (data_seed=model_seed=each seed, matched convention)
# Optional: CGCNN_DIM_SPECS="frac:pct frac:pct ..."  (default = canonical 1..100% set)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

TASK="${CGCNN_TASK:?set CGCNN_TASK=dielectric|phonons|log_kvrh|eform|bandgap}"
SEEDS="${CGCNN_SEEDS:?set CGCNN_SEEDS=\"s1 s2 ...\"}"
N_CONV=2
ATOM_FEA_LEN=82
H_FEA_LEN=128
N_H=3
WRAPPER="fastfood"
OPTIM="Adam"
PYTHON="python"

# Canonical sweep (matches the 1-layer dielectric 5-seed sweep): 1,2,5,10,15,20,40,60,80,100 %.
DEFAULT_DIM_SPECS="0.01:1 0.02:2 0.05:5 0.10:10 0.15:15 0.20:20 0.40:40 0.60:60 0.80:80 1.0:100"
DIM_SPECS="${CGCNN_DIM_SPECS:-$DEFAULT_DIM_SPECS}"

case "$TASK" in
  dielectric) CACHED_DATA_DIR="cached_matbench/tensors"; MB_TASK="matbench_dielectric"; EPOCHS=150; BATCH_SIZE=512 ;;
  phonons)    CACHED_DATA_DIR="cached_matbench/tensors"; MB_TASK="matbench_phonons";    EPOCHS=80;  BATCH_SIZE=512 ;;
  log_kvrh)   CACHED_DATA_DIR="cached_matbench/tensors"; MB_TASK="matbench_log_kvrh";   EPOCHS=120; BATCH_SIZE=512 ;;
  eform)      CACHED_DATA_DIR="cached_mp/tensors/mp_eform";   MB_TASK="mp_eform";       EPOCHS=350; BATCH_SIZE=256 ;;
  bandgap)    CACHED_DATA_DIR="cached_mp/tensors/mp_bandgap"; MB_TASK="mp_bandgap";     EPOCHS=350; BATCH_SIZE=256 ;;
  *) echo "Unknown CGCNN_TASK=$TASK" >&2; exit 1 ;;
esac
EPOCHS="${CGCNN_EPOCHS_OVERRIDE:-$EPOCHS}"

RESULT_ROOT="${CGCNN_RESULT_ROOT:-results_2layer_fastfood/${TASK}}"
SUMMARY_CSV="${RESULT_ROOT}/cgcnn_2layer_${TASK}_fastfood_sweep_summary.csv"
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "task,wrapper,optim,id_dim_frac,id_dim_percent,data_seed,model_seed,epochs,batch_size,n_conv,atom_fea_len,h_fea_len,n_h,output_dir,predictions_csv,best_val_mae,test_mae,duration_sec,status,exit_code,start_time,end_time" > "$SUMMARY_CSV"
fi

echo "[INFO] TASK=$TASK n_conv=$N_CONV epochs=$EPOCHS optim=$OPTIM wrapper=$WRAPPER seeds=$SEEDS"
echo "[INFO] DIM_SPECS=$DIM_SPECS"

run_one() {
  local frac="$1" pct="$2" seed="$3"
  local exp="cgcnn_2layer_${TASK}_${WRAPPER}_dim${pct}_dataseed${seed}_modelseed${seed}"
  local out="${RESULT_ROOT}/${exp}"
  local log="${out}/train.log" meta="${out}/metadata.json"
  local pred_src="${out}/${MB_TASK}_test_results.csv" pred_dst="${out}/${exp}_test_results.csv"
  mkdir -p "$out"
  if [[ -f "$meta" ]] && grep -q '"status": "success"' "$meta"; then
    echo "[skip] $exp (already success)"; return 0
  fi
  local t0=$SECONDS start; start="$(date -Iseconds)"
  echo "[START] $exp (frac=$frac pct=$pct seed=$seed) $start"
  "$PYTHON" main.py \
    --cached-data-dir "$CACHED_DATA_DIR" --matbench-task "$MB_TASK" --output-dir "$out" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --optim "$OPTIM" \
    --n-conv "$N_CONV" --atom-fea-len "$ATOM_FEA_LEN" --h-fea-len "$H_FEA_LEN" --n-h "$N_H" \
    --subspace-method "$WRAPPER" --id-dim "$frac" \
    --data-seed "$seed" --random-seed "$seed" --print-freq 1000 2>&1 | tee "$log"
  local ec=${PIPESTATUS[0]} end; end="$(date -Iseconds)"
  local dur=$((SECONDS - t0)) status="success"; [[ "$ec" -ne 0 ]] && status="failed"
  [[ -f "$pred_src" ]] && mv -f "$pred_src" "$pred_dst"
  "$PYTHON" - "$SUMMARY_CSV" "$meta" "$log" "$pred_dst" "$MB_TASK" "$WRAPPER" "$OPTIM" "$frac" "$pct" \
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

for spec in $DIM_SPECS; do
  IFS=':' read -r frac pct <<< "$spec"
  for seed in $SEEDS; do
    run_one "$frac" "$pct" "$seed"
  done
done
echo "[ALL DONE] $TASK 2-layer sweep -> $SUMMARY_CSV"
