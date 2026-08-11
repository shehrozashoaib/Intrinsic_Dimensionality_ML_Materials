#!/usr/bin/env bash
# Add eform dims 55% and 70% to the ONE-CYCLE fixed-test sweep, all 5 dataset sizes, same seeds.
#
# Protocol identical to the existing one-cycle runs (so these pool straight into the main tables):
#   --lr_schedule onecycle, lr 1e-3, pct_start 0.3, --patience 0, --restore_best,
#   clipnorm 0, num_blocks 1, 350 epochs.
#   full = 4 seeds (123/456/789/234); quarter/tenth = 2 partitions x 2 seeds (123/456).
#
# 24 runs total. Maximum parallelism, one GPU per job:
#   full      8 jobs x 1 run  (~5.3 h each)  -> 8 h walltime
#   quarter   4 jobs x 2 runs (~2.6 h each)  -> 6 h
#   tenth     4 jobs x 2 runs (~1.1 h each)  -> 4 h
#   = 16 concurrent jobs, inside the 24-GPU cap, every one well under 20 h.
#
# Results land in the SAME root as the July-22 sweep, so the generator picks them up with no
# extra wiring; the sweep scripts skip any run whose metadata.json already exists.
#
#   bash launch_eform_5570.sh            # dry run
#   SUBMIT=1 bash launch_eform_5570.sh   # submit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP="${ROOT}/DimeNetPP_MP"
SUBMIT="${SUBMIT:-0}"
COMMON="DIMENET_LR_SCHEDULE=onecycle,DIMENET_LR=0.001,DIMENET_PATIENCE=0,DIMENET_CLIPNORM=0"
N=0

go() {  # go <dataset> <dims> <seeds> <time> <mem> <name>
  local ds="$1" dims="$2" seeds="$3" tm="$4" mem="$5" name="$6"
  local ft=0; [[ "$ds" != "full" ]] && ft=1
  local rroot="${MP}/results_dimenetpp_onecycle_fixedtest/eform/${ds}"
  N=$((N+1))
  echo "[$N] eform/${ds}  dims='${dims}'  seeds='${seeds}'  t=${tm} mem=${mem}"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$MP" && sbatch --time="$tm" --mem="$mem" --job-name="$name" \
      --export=ALL,${COMMON},DIMENET_TASK=eform,DIMENET_DATASET=${ds},DIMENET_FIXEDTEST=${ft},DIMENET_DIMS="${dims}",DIMENET_SEEDS="${seeds}",DIMENET_RESULT_ROOT="${rroot}" \
      scripts/submit_bestmodel_job.slurm)
  fi
}

# ---- FULL: one job per (dim, seed) = 8 jobs, fully parallel ----
for d in "0.7:70" "0.55:55"; do
  for s in "123:1123" "456:1456" "789:1789" "234:1234"; do
    go full "$d" "$s" "08:00:00" 128G "e57_f_${d##*:}_${s%%:*}"
  done
done

# ---- QUARTER / TENTH: one job per (partition, seed), both dims inside ----
for ds in quarter_1 quarter_2; do
  for s in "123:1123" "456:1456"; do
    go "$ds" "0.7:70 0.55:55" "$s" "06:00:00" 64G "e57_${ds}_${s%%:*}"
  done
done
for ds in tenth_1 tenth_2; do
  for s in "123:1123" "456:1456"; do
    go "$ds" "0.7:70 0.55:55" "$s" "04:00:00" 64G "e57_${ds}_${s%%:*}"
  done
done

echo
echo "total jobs: $N   runs: 24   (SUBMIT=${SUBMIT})"
