#!/usr/bin/env bash
# Generalised arm sweep for DimeNet++ (eform | bandgap): loops over DIMS x SEEDS inside ONE job,
# so several runs share a single Slurm allocation and fit a long batch window.
#
# Arms it can express:
#   ARM=base    OPTIM=adamw SCHED=none        -> no-wrapper baseline (Adam)
#   ARM=base    OPTIM=sgd   SCHED=none        -> no-wrapper baseline (SGD)
#   ARM=wrapped OPTIM=sgd   SCHED=none        -> wrapped, plain SGD
#   ARM=wrapped OPTIM=adamw SCHED=onecycle    -> wrapped, annealed LR (matches ALIGNN)
#
# WHY: on eform, wrapped dim-100 std was 0.00924 under a CONSTANT lr but 0.00120 under onecycle
# (7.7x lower, near the 0.00067 no-wrapper floor). This replicates that on bandgap, where the
# dim-100 std (0.02244 at n=5) is the highest of any bandgap dim.
#
# Env:
#   ARM_TASK      eform|bandgap        ARM_DIMS   "1.0:100 0.5:50 ..."   (id_dim:percent pairs)
#   ARM_SEEDS     "123:1123 456:1456"  ARM_ARM    base|wrapped
#   ARM_OPTIM     adamw|sgd            ARM_SCHED  none|onecycle|cosine
#   ARM_LR        default 0.001        ARM_EPOCHS default 350
#   ARM_PATIENCE  default 75           ARM_CLIPNORM default 0.0
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/dimenet_env.sh"
cd "$PROJECT_DIR"

TASK="${ARM_TASK:?set ARM_TASK=eform|bandgap}"
ARM="${ARM_ARM:?set ARM_ARM=base|wrapped}"
OPTIM="${ARM_OPTIM:-adamw}"
SCHED="${ARM_SCHED:-none}"
LR="${ARM_LR:-0.001}"
MOMENTUM="${ARM_MOMENTUM:-0.9}"
EPOCHS="${ARM_EPOCHS:-350}"
PATIENCE="${ARM_PATIENCE:-75}"
CLIPNORM="${ARM_CLIPNORM:-0.0}"
NUM_BLOCKS=1
BATCH_SIZE=64

case "$TASK" in
  eform)   RUNNER=dimenetpp_code_only/dimenet_run_eform_v3.py;   CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_eform" ;;
  bandgap) RUNNER=dimenetpp_code_only/dimenet_run_bandgap_v3.py; CACHE_ROOT="${PROJECT_DIR}/cached_tensors_dimenetpp_bandgap" ;;
  *) echo "unknown ARM_TASK=$TASK" >&2; exit 1 ;;
esac
RESULT_ROOT="${ARM_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_optimizer_test/${TASK}}"
mkdir -p "$RESULT_ROOT"

read -ra DIMS  <<< "${ARM_DIMS:-1.0:100}"
read -ra PAIRS <<< "${ARM_SEEDS:?set ARM_SEEDS}"

FAILED_ANY=0
echo "[arm] task=$TASK arm=$ARM opt=$OPTIM sched=$SCHED lr=$LR ep=$EPOCHS pat=$PATIENCE"
echo "[arm] dims=${DIMS[*]}  seeds=${PAIRS[*]}"

for spec in "${DIMS[@]}"; do
  iddim="${spec%%:*}"; pct="${spec##*:}"
  for pair in "${PAIRS[@]}"; do
    model_seed="${pair%%:*}"; split_seed="${pair##*:}"
    name="${TASK}_${ARM}_${OPTIM}_${SCHED}_lr${LR}_dim${pct}_modelseed${model_seed}_splitseed${split_seed}_ep${EPOCHS}"
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
    "$DIMENET_PYTHON" "$RUNNER" \
      --train_mode "$ARM" \
      --method fastfood \
      --id_dim "$iddim" \
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

    "$DIMENET_PYTHON" - "$metadata_json" "$log_file" "$TASK" "$ARM" "$OPTIM" "$SCHED" "$LR" \
      "$model_seed" "$split_seed" "$EPOCHS" "$PATIENCE" "$CLIPNORM" "$pct" "$out_dir" \
      "$duration" "$exit_code" <<'PY'
import json, math, re, sys
from pathlib import Path
(metadata_json, log_file, task, arm, optim, sched, lr, model_seed, split_seed,
 epochs, patience, clipnorm, dimpct, output_dir, duration_sec, exit_code) = sys.argv[1:]
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
# best epoch from the val_loss curve (NaN-safe)
vl = [float(x) for x in re.findall(r"val_loss: ([0-9.eE+-]+)", text)]
best_epoch = (vl.index(min(vl)) + 1) if vl else None
diverged = bool(re.search(r"loss: nan", text))
m = re.search(r"Trainable params:\s*([\d,]+)", text)
total_params = int(m.group(1).replace(",", "")) if m else None
m = re.search(r"^D: (\d+) d: (\d+)", text, flags=re.M)
D_val, d_val = (int(m.group(1)), int(m.group(2))) if m else (None, None)

status = "success" if (int(exit_code) == 0 and test_mae is not None and not math.isnan(test_mae)) else "failed"
row = {
    "task": task, "dataset": "full", "arm": arm, "wrapper": arm == "wrapped",
    "optimizer": optim, "lr_schedule": sched, "lr": float(lr), "num_blocks": 1,
    "id_dim": float(dimpct)/100.0, "id_dim_percent": int(dimpct),
    "D": D_val, "d": d_val, "trainable_params": total_params,
    "model_seed": int(model_seed), "split_seed": int(split_seed),
    "epochs": int(epochs), "patience": int(patience), "clipnorm": float(clipnorm),
    "restore_best": bool(rb), "diverged": diverged,
    "early_stopped": epochs_run is not None and epochs_run < int(epochs),
    "epochs_run": epochs_run, "best_epoch": best_epoch,
    "best_val_loss": best_val_loss, "final_val_loss": final_val_loss,
    "val_mae": val_mae, "test_mae": test_mae,
    "duration_sec": int(float(duration_sec)), "status": status, "output_dir": output_dir,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
print(f"[metadata] {Path(output_dir).name} status={status} test_mae={test_mae} "
      f"best_ep={best_epoch}/{epochs_run} diverged={diverged}")
PY
  done
done

if [[ $FAILED_ANY -ne 0 ]]; then
  echo "[arm] FAILED: at least one run errored" >&2
  exit 1
fi
echo "[arm] done"
