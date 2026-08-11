#!/usr/bin/env bash
# CGCNN matbench_mp_is_metal (106,113 structures) intrinsic-dimension sweep, ONE-CYCLE LR.
#
# Grid: 11 dims (100/80/60/40/20/15/10/5/2/1/0.1) x 4 seeds (101/202/303/404) = 44 runs.
# CGCNN uses data_seed == model_seed; tensor caches exist for all four seeds.
# Metric is ROC-AUC (higher better), NOT MAE -- `--task classification`, NLLLoss.
#
# Packing: measured 2.36 h median per run (300 epochs). One job = one DIM with all 4 seeds
# sequentially = ~9.4 h, well inside the 20 h rule. 11 dims -> 11 jobs.
#
# max_lr = 1e-3: the value the MultiStepLR baseline uses AND the winner of the eform one-cycle
# probe (0.001 -> healthy 0.0775; 0.01 -> worse; 0.03 -> COLLAPSED, pred_spread 0.009).
# Note CGCNN's collapse behaviour is task-specific: dielectric needed max_lr RAISED to 0.1,
# eform collapses when raised. Every run records pred_spread + `degenerate` so a constant
# predictor cannot masquerade as a low-variance result.
#
#   bash launch_is_metal_onecycle.sh            # dry run
#   SUBMIT=1 bash launch_is_metal_onecycle.sh   # submit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"
LR="${LR:-0.001}"
EXCLUDE="${EXCLUDE:-gpu212-14}"
RESULT_ROOT="${RESULT_ROOT:-${HERE}/results_mp_is_metal_onecycle_fastfood}"
SEEDS=(101 202 303 404)
# frac:pct -- 0.1% is a FLOAT percent; the sweep scripts were made float-tolerant for it.
DIMS=("1.0:100" "0.8:80" "0.6:60" "0.4:40" "0.2:20" "0.15:15" "0.1:10" "0.05:5" "0.02:2" "0.01:1" "0.001:0.1")
N=0

for d in "${DIMS[@]}"; do
  frac="${d%%:*}"; pct="${d##*:}"
  runs=""
  for s in "${SEEDS[@]}"; do runs+="${frac}:${pct}:${s}:${s} "; done
  runs="${runs% }"
  N=$((N+1))
  echo "[$N] dim ${pct}%  runs='${runs}'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$HERE" && sbatch --time="16:00:00" --job-name="cgoc_im_${pct}" \
      --constraint="a100|rtx2080ti" --mem=96G --exclude="$EXCLUDE" \
      --export=ALL,CGCNN_RUNS="${runs}",CGCNN_EPOCHS=300,CGCNN_LR_SCHEDULE=onecycle,CGCNN_LR="${LR}",CGCNN_RESULT_ROOT="${RESULT_ROOT}" \
      scripts/submit_is_metal_sweep.slurm)
  fi
done

echo
echo "total jobs: $N   runs: $((N * ${#SEEDS[@]}))   max_lr=${LR}   (SUBMIT=${SUBMIT})"
