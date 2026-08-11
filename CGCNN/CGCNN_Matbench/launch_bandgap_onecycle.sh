#!/usr/bin/env bash
# CGCNN MP band gap (FULL dataset) intrinsic-dimension sweep, ONE-CYCLE LR + best-model eval.
# Best-model is NATIVE to CGCNN (main.py reloads model_best before test). This adds the one-cycle
# schedule, matching the eform/is_metal/phonons one-cycle arms.
#   8 dims (100/80/60/50/40/20/10/5) x 4 seeds (314/628/942/33221) = 32 runs, 350 epochs.
#   data_seed == model_seed == split_seed (CGCNN convention). All 4 seeds have tensor caches.
# max_lr must be set from the probe. CGCNN collapsed under one-cycle on dielectric (small data);
# bandgap is large (154,879) like eform, where 1e-3 trained cleanly. Every run records pred_spread.
#   LR=0.001 bash launch_bandgap_onecycle.sh            # dry run
#   LR=0.001 SUBMIT=1 bash launch_bandgap_onecycle.sh   # submit
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"; LR="${LR:?set LR=<max_lr from probe>}"
EXCLUDE="${EXCLUDE:-gpu212-14,gpu510-07,gpu510-12}"
ROOT="${RESULT_ROOT:-${HERE}/results_mp_bandgap_onecycle_fastfood}"
SEEDS=(314 628 942 33221)
DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.4:40" "0.2:20" "0.1:10" "0.05:5")
N=0
for d in "${DIMS[@]}"; do
  frac="${d%%:*}"; pct="${d##*:}"; runs=""
  for s in "${SEEDS[@]}"; do runs+="${frac}:${pct}:${s}:${s} "; done
  N=$((N+1)); echo "[$N] dim ${pct}%  runs='${runs% }'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$HERE" && sbatch --time="20:00:00" --job-name="cgoc_bg_${pct}" \
      --constraint="a100|rtx2080ti" --mem=96G --exclude="$EXCLUDE" \
      --export=ALL,CGCNN_TASK=bandgap,CGCNN_RUNS="${runs% }",CGCNN_EPOCHS=350,CGCNN_LR_SCHEDULE=onecycle,CGCNN_LR="${LR}",CGCNN_RESULT_ROOT="${ROOT}" \
      scripts/submit_eform_custom.slurm)
  fi
done
echo; echo "total jobs: $N   runs: $((N*4))   max_lr=${LR}   (SUBMIT=${SUBMIT})"
