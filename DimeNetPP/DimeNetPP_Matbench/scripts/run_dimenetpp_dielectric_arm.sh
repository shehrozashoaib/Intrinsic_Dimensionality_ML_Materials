#!/usr/bin/env bash
# DimeNet++ matbench_dielectric — best-model + one-cycle LR arm.
#
# REPLICATES: results_dimenetpp_dielectric_fastfood (10 dims x 5 seeds, 150 epochs) but with
#   (a) --restore_best : the old sweep has restore_best=None, i.e. it scored the FINAL-epoch
#       (typically overfit) model. This arm scores the best-validation checkpoint.
#   (b) --lr_schedule onecycle : anneals the lr to ~0 so seeds converge instead of ending inside
#       a constant-lr noise ball.
#
# Records pred_spread + a `degenerate` flag: on CGCNN/dielectric, one-cycle at too low a max_lr
# collapsed to a CONSTANT predictor (spread ~0.01 vs ~0.8 healthy) which looks like a beautiful
# low-variance result but is worthless. Always check that flag before trusting a std.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/dimenet_env.sh"
cd "$PROJECT_DIR"

EPOCHS="${ARM_EPOCHS:-150}"
PATIENCE="${ARM_PATIENCE:-0}"
CLIPNORM="${ARM_CLIPNORM:-0.0}"
LR="${ARM_LR:-0.001}"
SCHED="${ARM_SCHED:-onecycle}"
PCT="${ARM_PCT:-0.3}"
OPTIM="${ARM_OPTIM:-adamw}"
NUM_BLOCKS=1
BATCH_SIZE=64
CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_dielectric"
RESULT_ROOT="${ARM_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_dielectric_onecycle}"
mkdir -p "$RESULT_ROOT"

read -ra DIMS  <<< "${ARM_DIMS:-1.0:100}"
read -ra PAIRS <<< "${ARM_SEEDS:-123:1123 234:1234 456:1456 567:1567 789:1789}"

echo "[arm] dielectric sched=$SCHED lr=$LR pct=$PCT ep=$EPOCHS patience=$PATIENCE restore_best=1"
FAILED_ANY=0

for spec in "${DIMS[@]}"; do
  iddim="${spec%%:*}"; pct="${spec##*:}"
  for pair in "${PAIRS[@]}"; do
    ms="${pair%%:*}"; ss="${pair##*:}"
    name="dimenetpp_dielectric_${SCHED}_lr${LR}_dim${pct}_modelseed${ms}_splitseed${ss}_ep${EPOCHS}"
    out_dir="${RESULT_ROOT}/${name}"
    meta="${out_dir}/metadata.json"
    log="${out_dir}/train.log"

    if [[ -f "$meta" ]] && grep -q '"status": "success"' "$meta"; then
      echo "[skip] $name"; continue
    fi
    mkdir -p "$out_dir"
    echo "[run] $name"

    start=$(date +%s)
    set +e
    "$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_dielectric_v3.py \
      --train_mode wrapped \
      --method fastfood \
      --id_dim "$iddim" \
      --optimizer "$OPTIM" \
      --lr "$LR" \
      --lr_schedule "$SCHED" \
      --pct_start "$PCT" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_blocks "$NUM_BLOCKS" \
      --seed "$ms" \
      --split_seed "$ss" \
      --clipnorm "$CLIPNORM" \
      --restore_best \
      --patience "$PATIENCE" \
      --cache_dir "${CACHE_ROOT}/splitseed${ss}" \
      --out_dir "$out_dir" > "$log" 2>&1
    rc=$?
    set -e
    dur=$(( $(date +%s) - start ))
    [[ $rc -ne 0 ]] && FAILED_ANY=1

    "$DIMENET_PYTHON" - "$meta" "$log" "$SCHED" "$LR" "$PCT" "$pct" "$ms" "$ss" "$EPOCHS" \
      "$PATIENCE" "$CLIPNORM" "$OPTIM" "$out_dir" "$dur" "$rc" <<'PY'
import json, math, re, sys
from pathlib import Path
(meta, log, sched, lr, pct_start, dimpct, ms, ss, epochs, patience, clipnorm,
 optim, out_dir, dur, rc) = sys.argv[1:]
t = Path(log).read_text(errors="replace") if Path(log).exists() else ""

def last(p):
    m = re.findall(p, t); return float(m[-1]) if m else None

test_mae = last(r"Test MAE:\s*([0-9.eE+-]+)")
val_mae = last(r"Validation MAE \(original units\):\s*([0-9.eE+-]+)")
spread = last(r"\[pred_spread\]\s*([0-9.eE+-]+)")
best_val_loss = final_val_loss = epochs_run = None
rb = re.findall(r"\[restore_best\][^\n]*", t)
if rb:
    line = rb[-1]
    m = re.search(r"best val_loss=([0-9.eE+-]+)", line);  best_val_loss = float(m.group(1)) if m else None
    m = re.search(r"final val_loss=([0-9.eE+-]+)", line); final_val_loss = float(m.group(1)) if m else None
    m = re.search(r"epochs_run=([0-9]+)", line);          epochs_run = int(m.group(1)) if m else None
if epochs_run is None:
    m = re.findall(r"^Epoch (\d+)/\d+", t, flags=re.M)
    if m: epochs_run = int(m[-1])
vl = [float(x) for x in re.findall(r"val_loss: ([0-9.eE+-]+)", t)]
best_epoch = (vl.index(min(vl)) + 1) if vl else None
m = re.search(r"^D: (\d+) d: (\d+)", t, flags=re.M)
D_val, d_val = (int(m.group(1)), int(m.group(2))) if m else (None, None)

ok = test_mae is not None and not math.isnan(test_mae)
row = {
    "task": "matbench_dielectric", "arm": "wrapped", "optimizer": optim,
    "lr_schedule": sched, "lr": float(lr), "pct_start": float(pct_start),
    "num_blocks": 1, "id_dim": float(dimpct)/100.0, "id_dim_percent": int(dimpct),
    "D": D_val, "d": d_val, "model_seed": int(ms), "split_seed": int(ss),
    "epochs": int(epochs), "patience": int(patience), "clipnorm": float(clipnorm),
    "restore_best": bool(rb), "epochs_run": epochs_run, "best_epoch": best_epoch,
    "best_val_loss": best_val_loss, "final_val_loss": final_val_loss,
    "val_mae": val_mae, "test_mae": test_mae,
    "pred_spread": spread,
    "degenerate": (spread is not None and spread < 0.05),
    "duration_sec": int(dur), "status": "success" if (int(rc) == 0 and ok) else "failed",
    "output_dir": out_dir,
}
Path(meta).write_text(json.dumps(row, indent=2))
print(f"[metadata] {Path(out_dir).name} status={row['status']} test_mae={test_mae} "
      f"spread={spread} degenerate={row['degenerate']} best_ep={best_epoch}/{epochs_run}")
PY
    rm -f "$out_dir"/best_weights.weights.h5
  done
done

if [[ $FAILED_ANY -ne 0 ]]; then
  echo "[arm] FAILED: at least one run errored" >&2
  exit 1
fi
echo "[arm] done"
