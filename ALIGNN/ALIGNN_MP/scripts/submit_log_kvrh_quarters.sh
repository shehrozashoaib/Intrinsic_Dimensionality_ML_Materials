#!/usr/bin/env bash
# ALIGNN Fastfood sweep on matbench log_kvrh QUARTER-dataset subsets.
# Two disjoint 25% quarters (kvrh_quarter_1, kvrh_quarter_2; built by alignn/build_kvrh_quarters.py,
# nested inside the halves via the same seed-0 shuffle). Each quarter x 2 seeds (123,456) x 12 dims.
#
# Packing (per the "cram runs, few jobs, <=20h" rule): ONE array task per QUARTER, each running both
# seeds x all 12 dims = 24 runs sequentially. => only 2 A100 jobs. Quarter per-dim ~5-6 min (half the
# halves' ~11 min), so a 24-run task is ~3 h -- far under 20 h.
#
# Run from ALIGNN/ALIGNN_MP:  bash scripts/submit_log_kvrh_quarters.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

QUARTERS=(kvrh_quarter_1 kvrh_quarter_2)
SEEDS=(123 456)
# Same 12-dim sweep as the halves: 100,80,70,65,50,45,20,10,8,5,2,1
DIMS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")
WALLTIME="${KVRH_QUARTERS_WALLTIME:-06:00:00}"   # generous; ~3 h expected per 24-run task
EPOCHS="${KVRH_QUARTERS_EPOCHS:-180}"

# One spec line per QUARTER = both seeds x 12 dims (24 runs), so the array has 2 tasks.
SPEC="logs/kvrh_quarters_spec.txt"; : > "$SPEC"
for q in "${QUARTERS[@]}"; do
  runs=""
  for seed in "${SEEDS[@]}"; do
    for d in "${DIMS[@]}"; do
      IFS=':' read -r frac pct <<< "$d"
      runs+="${q}:${seed}:${frac}:${pct} "
    done
  done
  echo "log_kvrh|${runs% }" >> "$SPEC"
done

N=$(wc -l < "$SPEC")
echo "spec: $N array tasks (= quarters), each runs ${#SEEDS[@]} seeds x ${#DIMS[@]} dims = $(( ${#SEEDS[@]} * ${#DIMS[@]} )) runs -> $SPEC"
sed 's/|/  |  /' "$SPEC" | cut -c1-120

# torch 2.11+cu128 needs cc>=sm_75, so V100(sm_70) is OUT; A100(sm_80) and RTX-2080-Ti(sm_75) both work.
# Feature names (NOT gres names): a100, rtx2080ti. Allow either so the tiny 1-layer log_kvrh quarter
# runs start on whichever pool is free first. mem=64G (quarters are only 2746 structures) so the older
# 2080-Ti nodes actually qualify.
COMMON=(--gpus=1 --constraint="a100|rtx2080ti" --cpus-per-task=8 --mem=64G)
JOB="scripts/submit_partition_job.slurm"

echo ">> submit log_kvrh quarters array (0-$((N-1))%2, ${WALLTIME}, ${EPOCHS} epochs)"
sbatch "${COMMON[@]}" --time="$WALLTIME" --array="0-$((N-1))%2" \
  --job-name=alignn_kvrh_quarter \
  --output="logs/alignn_kvrh_quarter_%A_%a.out" --error="logs/alignn_kvrh_quarter_%A_%a.err" \
  --export=ALL,PART_SPECFILE="$SPEC",PART_EPOCHS="$EPOCHS" "$JOB"

echo "Submitted. Track: squeue -u \$USER | grep alignn_kvrh_quarter"
