#!/usr/bin/env bash
# DimeNet++ matbench_mp_is_metal — ONE-CYCLE LR + BEST-MODEL arm.
# Brings DimeNet++ in line with the CGCNN one-cycle arm: the old is_metal runs used a CONSTANT lr
# and scored the FINAL-epoch model (the runner had no --lr_schedule/--restore_best/--patience at
# all; all three were ported from the log_kvrh runner for this sweep).
#   11 dims x 4 seeds (101/202/303/404) = 44 runs, 300 epochs, num_blocks=1 (D=83,685), fastfood.
#   --lr_schedule onecycle (lr 1e-3), --patience 0 (ES off so the anneal completes), --restore_best
# Measured ~3 h/run -> one job per dim (4 seeds) ~12 h, inside the 20 h rule.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"; EXCLUDE="${EXCLUDE:-gpu212-14}"
ROOT="${HERE}/results_dimenetpp_is_metal_onecycle_fastfood"
SEEDS=(101 202 303 404)
DIMS=("1.0:100" "0.8:80" "0.6:60" "0.4:40" "0.2:20" "0.15:15" "0.1:10" "0.05:5" "0.02:2" "0.01:1" "0.001:0.1")
N=0
for d in "${DIMS[@]}"; do
  frac="${d%%:*}"; pct="${d##*:}"; runs=""
  for s in "${SEEDS[@]}"; do runs+="${frac}:${pct}:${s}:$((s+1000)) "; done
  N=$((N+1)); echo "[$N] dim ${pct}%  runs='${runs% }'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$HERE" && sbatch --time="16:00:00" --job-name="imoc_${pct}" --constraint=v100 \
      --mem=96G --exclude="$EXCLUDE" \
      --export=ALL,DIMENET_RUNS="${runs% }",DIMENET_EPOCHS=300,DIMENET_LR_SCHEDULE=onecycle,DIMENET_LR=0.001,DIMENET_RESTORE_BEST=1,DIMENET_PATIENCE=0,DIMENET_RESULT_ROOT="${ROOT}" \
      scripts/submit_is_metal_job.slurm)
  fi
done
echo; echo "total jobs: $N   runs: $((N*4))   (SUBMIT=${SUBMIT})"
