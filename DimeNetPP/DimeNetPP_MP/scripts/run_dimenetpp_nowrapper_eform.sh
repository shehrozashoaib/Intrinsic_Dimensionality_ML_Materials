#!/usr/bin/env bash
# NO-WRAPPER control for eform (full dataset).
#
# WHY: at dim=100% the wrapped runs show much larger seed-to-seed test-MAE spread
# (CV 13.8%) than at low dim (CV 8.8% at dim=15). This arm removes the Fastfood
# wrapper entirely (--train_mode base) so all D=83,685 parameters train directly.
# If the spread persists here, it is intrinsic to the task/split, NOT a wrapper artifact.
#
# Everything else matches the wrapped best-model protocol: num_blocks=1, int_emb_size=64
# (D = 83,685), AdamW lr=1e-3, --restore_best, EarlyStopping, same caches/scaler/test path.
#
# Seed/split pairs mirror the wrapped runs exactly (123/1123, 456/1456, 789/1789) so the
# comparison is head-to-head, plus a 4th pair (234/1653) for extra power.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/dimenet_env.sh"
cd "$PROJECT_DIR"

EPOCHS="${DIMENET_EPOCHS:-200}"
PATIENCE="${DIMENET_PATIENCE:-50}"
CLIPNORM="${DIMENET_CLIPNORM:-0.0}"
NUM_BLOCKS=1
BATCH_SIZE=64
TASK=eform
DATASET=full
RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PROJECT_DIR}/results_dimenetpp_nowrapper/${TASK}/${DATASET}}"
CACHE_ROOT="${DIMENET_CACHE_ROOT:-${PROJECT_DIR}/cached_tensors_dimenetpp_eform}"
mkdir -p "$RESULT_ROOT"

# "model_seed:split_seed" — mirrors the wrapped protocol's pairing.
PAIRS_DEFAULT="123:1123 456:1456 789:1789 234:1653"
read -ra PAIRS <<< "${DIMENET_SEEDS:-$PAIRS_DEFAULT}"

echo "[nowrapper] task=$TASK dataset=$DATASET epochs=$EPOCHS patience=$PATIENCE clipnorm=$CLIPNORM"
echo "[nowrapper] pairs: ${PAIRS[*]}"

for pair in "${PAIRS[@]}"; do
  model_seed="${pair%%:*}"; split_seed="${pair##*:}"
  name="dimenetpp_eform_nowrapper_full_modelseed${model_seed}_splitseed${split_seed}_epochs${EPOCHS}"
  out_dir="${RESULT_ROOT}/${name}"
  metadata_json="${out_dir}/metadata.json"
  log_file="${out_dir}/train.log"

  if [[ -f "$metadata_json" ]] && grep -q '"status": "success"' "$metadata_json"; then
    echo "[skip] $name already successful"; continue
  fi
  mkdir -p "$out_dir"
  echo "[run] $name"

  start=$(date +%s)
  set +e
  "$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_eform_v3.py \
    --train_mode base \
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

  # Metadata mirrors run_dimenetpp_bestmodel_sweep.sh so existing tooling can read it.
  # id_dim/id_dim_percent are recorded as 100 purely as a join key: with no wrapper there
  # is no subspace, so `train_mode: base` is what actually distinguishes these runs.
  "$DIMENET_PYTHON" - "$metadata_json" "$log_file" "$TASK" "$DATASET" "$NUM_BLOCKS" \
    "$model_seed" "$split_seed" "$EPOCHS" "$PATIENCE" "$CLIPNORM" "$out_dir" \
    "$duration" "$exit_code" <<'PY'
import json, math, re, sys
from pathlib import Path

(metadata_json, log_file, task, dataset, num_blocks, model_seed, split_seed,
 epochs, patience, clipnorm, output_dir, duration_sec, exit_code) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

def last_float(p):
    m = re.findall(p, text)
    return float(m[-1]) if m else None

val_mae = last_float(r"Validation MAE \(original units\):\s*([0-9.eE+-]+)")
test_mae = last_float(r"Test MAE:\s*([0-9.eE+-]+)")

best_val_loss = final_val_loss = epochs_run = None
rb = re.findall(r"\[restore_best\][^\n]*", text)
if rb:
    line = rb[-1]
    for key, cast in (("best val_loss", float), ("final val_loss", float), ("epochs_run", int)):
        m = re.search(re.escape(key) + r"=([0-9.eE+-]+)", line)
        if m:
            v = cast(m.group(1))
            if key == "best val_loss": best_val_loss = v
            elif key == "final val_loss": final_val_loss = v
            else: epochs_run = v
if epochs_run is None:
    m = re.findall(r"^Epoch (\d+)/\d+", text, flags=re.M)
    if m:
        epochs_run = int(m[-1])

# Did the run actually restore a best checkpoint, and did early stopping fire?
restored = bool(rb)
early_stopped = epochs_run is not None and epochs_run < int(epochs)
total_params = last_float(r"Trainable params:\s*([0-9,]+)".replace(",", ""))
m = re.search(r"Trainable params:\s*([\d,]+)", text)
total_params = int(m.group(1).replace(",", "")) if m else None

status = "success" if (int(exit_code) == 0 and test_mae is not None and not math.isnan(test_mae)) else "failed"

row = {
    "task": task, "dataset": dataset, "num_blocks": int(num_blocks),
    "train_mode": "base",          # <-- the distinguishing field: NO wrapper
    "wrapper": False,
    "id_dim": 1.0, "id_dim_percent": 100,
    "trainable_params": total_params,
    "model_seed": int(model_seed), "split_seed": int(split_seed),
    "epochs": int(epochs), "patience": int(patience), "clipnorm": float(clipnorm),
    "restore_best": restored, "early_stopped": early_stopped,
    "epochs_run": epochs_run,
    "best_val_loss": best_val_loss, "final_val_loss": final_val_loss,
    "val_mae": val_mae, "test_mae": test_mae,
    "duration_sec": int(float(duration_sec)), "status": status, "output_dir": output_dir,
}
Path(metadata_json).write_text(json.dumps(row, indent=2))
print(f"[metadata] {Path(output_dir).name} status={status} test_mae={test_mae} "
      f"epochs_run={epochs_run} early_stopped={early_stopped} params={total_params}")
PY
done

echo "[nowrapper] done"
