#!/usr/bin/env bash
# CGCNN matbench_dielectric — ONE-CYCLE LR replicate of the existing 10-seed Fastfood sweep.
#
# WHY: on DimeNet++ the wrapped dim-100% seed-variance spike turned out to be an artifact of the
# CONSTANT learning rate, not of the subspace wrapper — annealing the lr to ~0 (one-cycle) cut the
# dim-100% std ~6.7x on eform and ~2.2x on bandgap and flattened the std-vs-dim profile.
# CGCNN sits between the two models we have already compared: upstream it uses MultiStepLR
# (x0.1 at epoch 100, then FLAT for the rest of training), whereas ALIGNN uses one-cycle and
# DimeNet++ used a constant lr. This sweep swaps CGCNN's schedule for one-cycle, changing NOTHING
# else, so the two dielectric sweeps differ in exactly one variable.
#
# BASELINE BEING REPLICATED: results_matbench_dielectric_5seed_fastfood
#   10 dims x 10 seeds = 100 runs, 150 epochs, n_conv=1 atom_fea_len=82 h_fea_len=128 n_h=3,
#   batch 512, Adam lr 1e-3, and data_seed == model_seed (each seed varies BOTH the split and
#   the model init — so the spread here includes split variance, not just init variance).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

CONDA_ENV="${CGCNN_CONDA_ENV:-py312}"
# conda is not on PATH inside a batch shell -> source it explicitly
if ! command -v conda >/dev/null 2>&1; then
  source "/ibex/user/${USER}/miniconda3/etc/profile.d/conda.sh"
fi
PYTHON="python"
CACHED_DATA_DIR="cached_matbench/tensors"
TASK="matbench_dielectric"
EPOCHS="${CGCNN_EPOCHS:-150}"
BATCH_SIZE=512
N_CONV=1
ATOM_FEA_LEN=82
H_FEA_LEN=128
N_H=3
WRAPPER="fastfood"
LR_SCHEDULE="${CGCNN_LR_SCHEDULE:-onecycle}"
# OneCycleLR spends most of training BELOW max_lr (starts at max_lr/25, peaks at 30%, then
# anneals to max_lr/25/1e4). At max_lr=1e-3 -- the value the constant-lr baseline uses -- CGCNN
# collapsed to a constant predictor on this task, so one-cycle needs its own, larger max_lr.
LR="${CGCNN_LR:-0.001}"
# Fraction of training spent rising to max_lr. Longer warm-up keeps the lr high for longer, which
# small subspaces need to escape the constant-prediction plateau; the cosine tail still anneals to
# ~0 so seeds still converge (which is the point of using one-cycle at all).
PCT_START="${CGCNN_PCT_START:-0.3}"
PRINT_FREQ=1000

read -ra SEEDS <<< "${CGCNN_SEEDS:-123 234 456 567 629 789 1964 9809 14236 82738}"
read -ra DIMS  <<< "${CGCNN_DIMS:-100 80 60 40 20 15 10 5 2 1}"

RESULT_ROOT="${CGCNN_RESULT_ROOT:-results_${TASK}_${LR_SCHEDULE}_${WRAPPER}}"
mkdir -p "$RESULT_ROOT"

echo "[onecycle] task=$TASK sched=$LR_SCHEDULE epochs=$EPOCHS"
echo "[onecycle] dims=${DIMS[*]}"
echo "[onecycle] seeds=${SEEDS[*]}   (data_seed == model_seed, matching the baseline)"

FAILED_ANY=0
for DIM in "${DIMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # data_seed == model_seed, exactly as in the sweep being replicated
    EXP_NAME="cgcnn_${TASK}_${LR_SCHEDULE}_lr${LR}_pct${PCT_START}_${WRAPPER}_dim${DIM}_dataseed${SEED}_modelseed${SEED}"
    OUT_DIR="${RESULT_ROOT}/${EXP_NAME}"
    LOG_FILE="${OUT_DIR}/train.log"
    META_JSON="${OUT_DIR}/metadata.json"

    if [[ -f "$META_JSON" ]] && grep -q '"status": "success"' "$META_JSON"; then
      echo "[skip] $EXP_NAME"; continue
    fi
    mkdir -p "$OUT_DIR"
    echo "[run] $EXP_NAME"

    START_TIME="$(date -Iseconds)"
    START_S=$(date +%s)
    conda run -n "$CONDA_ENV" "$PYTHON" main.py \
      --cached-data-dir "$CACHED_DATA_DIR" \
      --matbench-task "$TASK" \
      --output-dir "$OUT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --n-conv "$N_CONV" \
      --atom-fea-len "$ATOM_FEA_LEN" \
      --h-fea-len "$H_FEA_LEN" \
      --n-h "$N_H" \
      --subspace-method "$WRAPPER" \
      --id-dim "$DIM" \
      --data-seed "$SEED" \
      --random-seed "$SEED" \
      --lr-schedule "$LR_SCHEDULE" \
      --lr "$LR" \
      --pct-start "$PCT_START" \
      --print-freq "$PRINT_FREQ" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    DURATION=$(( $(date +%s) - START_S ))
    END_TIME="$(date -Iseconds)"
    [[ $EXIT_CODE -ne 0 ]] && FAILED_ANY=1

    python - "$META_JSON" "$LOG_FILE" "$TASK" "$WRAPPER" "$LR_SCHEDULE" "$DIM" "$SEED" \
      "$EPOCHS" "$BATCH_SIZE" "$N_CONV" "$ATOM_FEA_LEN" "$H_FEA_LEN" "$N_H" "$OUT_DIR" \
      "$DURATION" "$EXIT_CODE" "$START_TIME" "$END_TIME" "$LR" "$PCT_START" <<'INNERPY'
import json, math, re, sys
from pathlib import Path
(meta_json, log_file, task, wrapper, sched, dim, seed, epochs, batch_size, n_conv,
 atom_fea_len, h_fea_len, n_h, out_dir, duration, exit_code, start_time, end_time, lr, pct_start) = sys.argv[1:]
text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

# Parsing copied verbatim from run_cgcnn_fastfood_dielectric_dataseed123_modelseeds.sh so the
# two sweeps are extracted identically:
#   "  * MAE x"  -> per-epoch VALIDATION  (best_val = min over epochs)
#   "test metric is: x" / "  ** MAE x" -> final TEST
val_matches = re.findall(r"^\s*\*\s+MAE\s+([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
best_val = min((float(x) for x in val_matches), default=None)

test_mae = None
m = re.search(r"test metric is:\s*([-+0-9.eE]+)", text)
if m:
    test_mae = float(m.group(1))
else:
    test_matches = re.findall(r"^\s*\*\*\s+MAE\s+([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
    if test_matches:
        test_mae = float(test_matches[-1])

ok = test_mae is not None and not (isinstance(test_mae, float) and math.isnan(test_mae))
status = "success" if (int(exit_code) == 0 and ok) else "failed"
row = {
    "task": task, "wrapper": wrapper, "lr_schedule": sched, "optim": "Adam", "lr": float(lr), "pct_start": float(pct_start),
    "id_dim_percent": int(dim), "id_dim_frac": int(dim) / 100.0,
    "data_seed": int(seed), "model_seed": int(seed),
    "epochs": int(epochs), "batch_size": int(batch_size), "n_conv": int(n_conv),
    "atom_fea_len": int(atom_fea_len), "h_fea_len": int(h_fea_len), "n_h": int(n_h),
    "best_val_mae": best_val, "test_mae": test_mae,
    "duration_sec": int(duration), "status": status, "exit_code": int(exit_code),
    "output_dir": out_dir, "start_time": start_time, "end_time": end_time,
}
# degeneracy check: std of the model's predictions on the test set. A collapsed
# (constant-output) model has spread ~0; a real model on this task has 0.3-0.9.
import csv as _csv
_pred = []
for _f in Path(out_dir).glob("*test_results.csv"):
    for _r in _csv.reader(open(_f)):
        if len(_r) >= 2:
            try:
                _pred.append(float(_r[-1]))
            except ValueError:
                pass
    break
if len(_pred) > 1:
    _mu = sum(_pred) / len(_pred)
    row["pred_spread"] = (sum((x - _mu) ** 2 for x in _pred) / len(_pred)) ** 0.5
    row["degenerate"] = row["pred_spread"] < 0.05
Path(meta_json).write_text(json.dumps(row, indent=2))
print(f"[metadata] {Path(out_dir).name} status={status} test_mae={test_mae} best_val={best_val}")
INNERPY
    rm -f "$OUT_DIR"/*.pth.tar
  done
done

if [[ $FAILED_ANY -ne 0 ]]; then
  echo "[onecycle] FAILED: at least one run errored" >&2
  exit 1
fi
echo "[onecycle] done"
