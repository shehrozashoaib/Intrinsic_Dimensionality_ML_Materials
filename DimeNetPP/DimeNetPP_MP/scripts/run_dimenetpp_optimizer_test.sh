#!/usr/bin/env bash
# OPTIMIZER TEST — does the dim-100% seed variance survive under SGD?
#
# BACKGROUND: on eform full, wrapped dim-100% has absolute test-MAE std 0.00924 (n=3) while the
# no-wrapper arm has 0.00067 (n=4) -- 13.9x, F=192 on df(2,3), p<0.01. Both arms train the SAME
# 83,685 free parameters, so it is not a degrees-of-freedom effect. The leading explanation is
# that Adam is not invariant under linear reparameterization: it preconditions per-coordinate in
# z-space, and that update reaches theta through P. Since P is redrawn per seed, each seed is
# effectively running a different optimizer geometry in theta-space.
#
# PREDICTION: plain SGD has no per-coordinate preconditioner, so the P-induced geometry
# distortion is far weaker. If the Adam hypothesis is right, the wrapped-vs-no-wrapper variance
# gap should shrink substantially under SGD. If the gap persists, the variance comes from
# something else (e.g. P changing the reachable solution set, not just the optimizer metric).
#
# DESIGN: 2x2 -- {base, wrapped@dim100} x {sgd, adamw}, 4 seeds each = 16 runs.
# The adamw/wrapped cell is re-run here at 200 epochs (not the sweep's 350) so that every cell
# shares an identical protocol and the comparison is free of the epoch confound.
# Everything else is held fixed: same splits, same seeds, same caches, same model, same lr
# WITHIN an optimizer (tuning per-arm would reintroduce the confound we are trying to remove).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/dimenet_env.sh"
cd "$PROJECT_DIR"

ARM="${OPT_ARM:?set OPT_ARM=base|wrapped}"
OPTIM="${OPT_OPTIM:?set OPT_OPTIM=sgd|adamw}"
LR="${OPT_LR:?set OPT_LR}"
MOMENTUM="${OPT_MOMENTUM:-0.9}"
SCHED="${OPT_SCHED:-none}"
IDDIM="${OPT_IDDIM:-1.0}"
DIMPCT="${OPT_DIMPCT:-100}"
EPOCHS="${OPT_EPOCHS:-200}"
PATIENCE="${OPT_PATIENCE:-50}"
CLIPNORM="${OPT_CLIPNORM:-0.0}"
NUM_BLOCKS=1
BATCH_SIZE=64
TASK=eform
RESULT_ROOT="${OPT_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_optimizer_test/${TASK}}"
CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_eform"
mkdir -p "$RESULT_ROOT"

PAIRS_DEFAULT="123:1123 456:1456 789:1789 234:1234"
read -ra PAIRS <<< "${OPT_SEEDS:-$PAIRS_DEFAULT}"

FAILED_ANY=0
echo "[opt-test] arm=$ARM optimizer=$OPTIM lr=$LR epochs=$EPOCHS patience=$PATIENCE"

for pair in "${PAIRS[@]}"; do
  model_seed="${pair%%:*}"; split_seed="${pair##*:}"
  name="eform_${ARM}_${OPTIM}_${SCHED}_lr${LR}_dim${DIMPCT}_modelseed${model_seed}_splitseed${split_seed}_ep${EPOCHS}"
  out_dir="${RESULT_ROOT}/${name}"
  metadata_json="${out_dir}/metadata.json"
  log_file="${out_dir}/train.log"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "[skip] $name"; continue
  fi
  mkdir -p "$out_dir"
  echo "[run] $name"

  start=$(date +%s)
  set +e
  "$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_eform_v3.py \
    --train_mode "$ARM" \
    --method fastfood \
    --id_dim "$IDDIM" \
    --optimizer "$OPTIM" \
    --lr "$LR" \
    --momentum "$MOMENTUM" \
    --lr_schedule "$SCHED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_blocks "$NUM_BLOCKS" \
    --seed "$model_seed" \
    --split_seed "$split_seed" \
    --clipnorm "$CLIPNORM" \
    --restore_best \
    --patience "$PATIENCE" \
    --cache_dir "${CACHE_ROOT}/splitseed${split_seed}" \
    --out_dir "$out_dir" > "$log_file" 2>&1
  exit_code=$?
  set -e
  duration=$(( $(date +%s) - start ))
  [[ $exit_code -ne 0 ]] && FAILED_ANY=1

  "$DIMENET_PYTHON" - "$metadata_json" "$log_file" "$ARM" "$OPTIM" "$LR" "$model_seed" \
    "$split_seed" "$EPOCHS" "$PATIENCE" "$CLIPNORM" "$out_dir" "$duration" "$exit_code" "$SCHED" "$DIMPCT" <<'PY'
import json, math, re, sys
from pathlib import Path
(metadata_json, log_file, arm, optim, lr, model_seed, split_seed, epochs,
 patience, clipnorm, output_dir, duration_sec, exit_code, sched, dimpct) = sys.argv[1:]
text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

def last_float(p):
    m = re.findall(p, text); return float(m[-1]) if m else None

val_mae = last_float(r"Validation MAE \(original units\):\s*([0-9.eE+-]+)")
test_mae = last_float(r"Test MAE:\s*([0-9.eE+-]+)")
best_val_loss = final_val_loss = epochs_run = None
rb = re.findall(r"\[restore_best\][^\n]*", text)
if rb:
    line = rb[-1]
    m = re.search(r"best val_loss=([0-9.eE+-]+)", line);  best_val_loss = float(m.group(1)) if m else None
    m = re.search(r"final val_loss=([0-9.eE+-]+)", line); final_val_loss = float(m.group(1)) if m else None
    m = re.search(r"epochs_run=([0-9]+)", line);          epochs_run = int(m.group(1)) if m else None
if epochs_run is None:
    m = re.findall(r"^Epoch (\d+)/\d+", text, flags=re.M)
    if m: epochs_run = int(m[-1])
m = re.search(r"Trainable params:\s*([\d,]+)", text)
total_params = int(m.group(1).replace(",", "")) if m else None
# d reported only in wrapped mode
m = re.search(r"^D: (\d+) d: (\d+)", text, flags=re.M)
D_val, d_val = (int(m.group(1)), int(m.group(2))) if m else (None, None)

status = "success" if (int(exit_code) == 0 and test_mae is not None and not math.isnan(test_mae)) else "failed"
row = {
    "task": "eform", "dataset": "full", "arm": arm, "wrapper": arm == "wrapped",
    "optimizer": optim, "lr": float(lr), "lr_schedule": sched, "num_blocks": 1,
    "id_dim": float(dimpct)/100.0, "id_dim_percent": int(dimpct),
    "D": D_val, "d": d_val, "trainable_params": total_params,
    "model_seed": int(model_seed), "split_seed": int(split_seed),
    "epochs": int(epochs), "patience": int(patience), "clipnorm": float(clipnorm),
    "restore_best": bool(rb),
    "early_stopped": epochs_run is not None and epochs_run < int(epochs),
    "epochs_run": epochs_run, "best_val_loss": best_val_loss, "final_val_loss": final_val_loss,
    "val_mae": val_mae, "test_mae": test_mae,
    "duration_sec": int(float(duration_sec)), "status": status, "output_dir": output_dir,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
print(f"[metadata] {Path(output_dir).name} status={status} test_mae={test_mae} "
      f"epochs_run={epochs_run} params={total_params} d={d_val}")
PY
done
if [[ $FAILED_ANY -ne 0 ]]; then
  echo "[opt-test] FAILED: at least one run errored" >&2
  exit 1
fi
echo "[opt-test] done"
