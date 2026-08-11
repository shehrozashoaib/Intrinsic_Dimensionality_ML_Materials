#!/usr/bin/env bash
# CGCNN MP eform (FULL dataset) intrinsic-dimension sweep under a ONE-CYCLE LR schedule.
#
# Grid: 8 dims (100/80/60/50/40/20/15/10) x 4 seeds (253/768/1653/5789) = 32 runs.
# CGCNN uses data_seed == model_seed, so each seed is a different split AND a different init.
# Tensor caches already exist for all four (cached_mp/tensors/mp_eform/seed<S>).
#
# Packing: measured 3.81 h median per run (350 epochs, 154,373 structures). One job = one DIM
# with all 4 seeds run sequentially = ~15.2 h, inside the 20 h rule. 8 dims -> 8 parallel jobs,
# which fits the ~12 free GPUs.
#
# CGCNN_LR must be set from the one-cycle LR probe: on matbench_dielectric CGCNN COLLAPSED to a
# constant predictor under one-cycle at max_lr=1e-3 and needed 0.1. Every run records
# pred_spread + a `degenerate` flag -- ALWAYS check those before quoting a std.
#
#   LR=0.01 bash launch_eform_onecycle.sh            # dry run
#   LR=0.01 SUBMIT=1 bash launch_eform_onecycle.sh   # submit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"
LR="${LR:?set LR=<max_lr chosen by the probe>}"
EXCLUDE="${EXCLUDE:-gpu212-14}"
RESULT_ROOT="${RESULT_ROOT:-${HERE}/results_mp_eform_onecycle_fastfood}"
SEEDS=(253 768 1653 5789)
DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.4:40" "0.2:20" "0.15:15" "0.1:10")
N=0

for d in "${DIMS[@]}"; do
  frac="${d%%:*}"; pct="${d##*:}"
  runs=""
  for s in "${SEEDS[@]}"; do runs+="${frac}:${pct}:${s}:${s} "; done
  runs="${runs% }"
  N=$((N+1))
  echo "[$N] dim ${pct}%  runs='${runs}'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$HERE" && sbatch --time="20:00:00" --job-name="cgoc_ef_${pct}" \
      --constraint="a100|rtx2080ti" --mem=96G --exclude="$EXCLUDE" \
      --export=ALL,CGCNN_TASK=eform,CGCNN_RUNS="${runs}",CGCNN_EPOCHS=350,CGCNN_LR_SCHEDULE=onecycle,CGCNN_LR="${LR}",CGCNN_RESULT_ROOT="${RESULT_ROOT}" \
      scripts/submit_eform_custom.slurm)
  fi
done

echo
echo "total jobs: $N   runs: $((N * ${#SEEDS[@]}))   max_lr=${LR}   (SUBMIT=${SUBMIT})"
