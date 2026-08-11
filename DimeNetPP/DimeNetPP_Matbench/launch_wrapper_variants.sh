#!/usr/bin/env bash
# DimeNet++ log_kvrh, FULL dataset — WRAPPER-VARIATION study.
#
# The ONLY thing that changes between arms is the random-projection construction. Everything
# else is held fixed and matches the existing fastfood reference arm:
#   num_blocks=1, int_emb_size=64 -> D = 83,685 ; clipnorm 0 ; batch 64
#   --lr_schedule onecycle, lr 1e-3, pct_start 0.3
#   --patience 0 (early stopping OFF so the anneal completes)
#   --restore_best (scored on the BEST-VALIDATION checkpoint)
#   4 seeds: 123/456/789/234 (split seed = model seed + 1000), full log_kvrh
#
# ARMS
#   fastfood        M = S*H*G*Pi*H*B, G ~ i.i.d. Gaussian    ALREADY RUN (reference arm)
#   base            no wrapper at all, all 83,685 params trained directly
#   dense           exact D x d Gaussian projection matrix
#   dense_ortho     D x d orthonormal Q (QR of a Gaussian)
#   rotate          permutation+sign rotation; REQUIRES d == D, so dim-100% ONLY
#   fastfood_ortho  fastfood with |G_i| = 1 -> the product is orthogonal up to `scale`
#
# DISK: two traps, both fixed --
#   1. --restore_best used to write a full checkpoint INCLUDING the frozen P: 28 GB per save at
#      dim-100%. That filled the disk and killed the first attempt at this sweep. The runner now
#      keeps only the trainable weights (z) in memory, so nothing large is written.
#   2. The orthonormal Q cache is keyed by (rows, cols, seed), so every Q is used by exactly ONE
#      run -- caching to shared storage buys nothing and would accumulate ~511 GB. It is pointed
#      at node-local /tmp so it dies with the job.
#
# MEMORY: dense stores a real D x d matrix -- 28 GB at dim-100%, 14 GB at 50%, 2.8 GB at 10%.
# That does not fit a 32 GB V100, so the dense arms run on A100 (80 GB). DIM_CEILING caps the
# highest dim each dense arm attempts; set it from the smoke results.
#
#   bash launch_wrapper_variants.sh            # dry run
#   SUBMIT=1 bash launch_wrapper_variants.sh   # submit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"
EXCLUDE="${EXCLUDE:-gpu212-14}"
S="scripts/submit_kvrh_bestmodel_job.slurm"
ROOT="${HERE}/results_dimenetpp_kvrh_wrapper_variants"
SEEDS="123:1123 456:1456 789:1789 234:1234"

# Full 12-dim grid (matches the fastfood reference arm).
ALL_DIMS="1.0:100 0.8:80 0.7:70 0.65:65 0.5:50 0.45:45 0.2:20 0.1:10 0.08:8 0.05:5 0.02:2 0.01:1"
# Dense arms: highest dim that fits. Override once the smokes report, e.g. DENSE_DIMS="0.5:50 ...".
DENSE_DIMS="${DENSE_DIMS:-$ALL_DIMS}"

N=0
go() {  # go <name> <dims> <epochs> <constraint> <mem> <time> <extra env>
  local name="$1" dims="$2" ep="$3" con="$4" mem="$5" tm="$6" extra="$7"
  N=$((N+1))
  echo "[$N] ${name}  epochs=${ep} con=${con} mem=${mem} t=${tm}"
  echo "      dims: ${dims}"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$HERE" && sbatch --time="$tm" --job-name="wv_${name}" --constraint="$con" \
      --mem="$mem" --exclude="$EXCLUDE" \
      --export=ALL,DIMENET_DATASET=full,DIMENET_DIMS="${dims}",DIMENET_SEEDS="${SEEDS}",DIMENET_EPOCHS=${ep},DIMENET_LR_SCHEDULE=onecycle,DIMENET_LR=0.001,DIMENET_PATIENCE=0,DIMENET_RESULT_ROOT="${ROOT}/${name}",${extra} \
      "$S")
  fi
}

# ---- base: no wrapper. There is no subspace, so the dim grid collapses to a single point. ----
go base "1.0:100" 150 v100 64G "06:00:00" "DIMENET_TRAIN_MODE=base"

# ---- dense + dense_ortho: A100 for the memory. One job per dim keeps each well inside 20 h. ----
# Route by the actual size of P = D x d x 4 bytes: only the top dims need an A100.
#   dim 100% = 28.0 GB (A100 only)   80% = 22.4   70% = 19.6   65% = 18.2  -> A100
#   dim  50% = 14.0 GB               45% = 12.6   20% = 5.6    <=10% < 3   -> V100 is fine
gpu_for() {  # gpu_for <pct> -> "constraint mem time"
  local pct="$1"
  if (( $(echo "$pct > 60" | bc -l) )); then echo "a100 200G 12:00:00"; else echo "v100 96G 10:00:00"; fi
}
for d in $DENSE_DIMS; do
  pct="${d##*:}"; read -r con mem tm <<< "$(gpu_for "$pct")"
  go "dense_d${pct}" "$d" 180 "$con" "$mem" "$tm" "DIMENET_METHOD=dense"
done
for d in $DENSE_DIMS; do
  pct="${d##*:}"; read -r con mem tm <<< "$(gpu_for "$pct")"
  if [[ "$con" == "a100" ]]; then obk=pytorch_gpu; else obk=pytorch_cpu; fi
  go "denseortho_d${pct}" "$d" 180 "$con" "$mem" "$tm" \
     "DIMENET_METHOD=dense,DIMENET_ORTHO=1,DIMENET_ORTHO_BACKEND=${obk}"
done

# ---- rotate: dim-100% only (the wrapper asserts d == D for this mode) ----
go rotate "1.0:100" 180 a100 200G "08:00:00" "DIMENET_METHOD=dense,DIMENET_ROTATE=1"

# ---- fastfood + orthonormal: |G_i| = 1. Cheap (no dense matrix), so one job for all dims. ----
go fastfood_ortho "$ALL_DIMS" 180 v100 64G "10:00:00" "DIMENET_METHOD=fastfood,DIMENET_ORTHO=1"

echo
echo "total jobs: $N   (SUBMIT=${SUBMIT})"
echo "reference arm 'fastfood' already exists: results_dimenetpp_onecycle_fixedtest/log_kvrh/full"
