#!/usr/bin/env bash
# ALIGNN Fastfood sweep on matbench log_kvrh TENTH-dataset subsets (1/10 of the data).
# Two disjoint 10% partitions (kvrh_tenth_1, kvrh_tenth_2; built by alignn/build_kvrh_tenths.py,
# nested inside the quarters/halves via the same seed-0 shuffle). 1,098 structures each.
#
# Each tenth x 2 seeds (123,456) x 12 dims = 24 runs. Packing (per the "cram runs, few jobs, <=20h"
# rule): ONE array task per TENTH => only 2 GPU jobs. Tenth per-dim ~4 min (quarters were ~9 min on
# 2.5x the data), so a 24-run task is ~1.6 h -- far under 20 h.
#
# 180 epochs, matching the quarters, so quarter-vs-tenth is a controlled comparison.
#
# Run from ALIGNN/ALIGNN_MP:  bash scripts/submit_log_kvrh_tenths.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

TENTHS=(kvrh_tenth_1 kvrh_tenth_2)
SEEDS=(123 456)
# Same 12-dim sweep as the halves/quarters: 100,80,70,65,50,45,20,10,8,5,2,1
DIMS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")
WALLTIME="${KVRH_TENTHS_WALLTIME:-04:00:00}"   # generous; ~1.6 h expected per 24-run task
EPOCHS="${KVRH_TENTHS_EPOCHS:-180}"

# One spec line per TENTH = both seeds x 12 dims (24 runs), so the array has 2 tasks.
SPEC="logs/kvrh_tenths_spec.txt"; : > "$SPEC"
for t in "${TENTHS[@]}"; do
  runs=""
  for seed in "${SEEDS[@]}"; do
    for d in "${DIMS[@]}"; do
      IFS=':' read -r frac pct <<< "$d"
      runs+="${t}:${seed}:${frac}:${pct} "
    done
  done
  echo "log_kvrh|${runs% }" >> "$SPEC"
done

N=$(wc -l < "$SPEC")
echo "spec: $N array tasks (= tenths), each runs ${#SEEDS[@]} seeds x ${#DIMS[@]} dims = $(( ${#SEEDS[@]} * ${#DIMS[@]} )) runs -> $SPEC"
sed 's/|/  |  /' "$SPEC" | cut -c1-120

# torch 2.11+cu128 needs cc>=sm_75: V100(sm_70) is OUT; A100(sm_80) and RTX-2080-Ti(sm_75) both work.
# Feature names (NOT gres names): a100, rtx2080ti. Allow either so these tiny runs start wherever is
# free -- keeps them off the contended A100 pool when possible. mem=64G so 2080-Ti nodes qualify.
COMMON=(--gpus=1 --constraint="a100|rtx2080ti" --cpus-per-task=8 --mem=64G)
JOB="scripts/submit_partition_job.slurm"

echo ">> submit log_kvrh tenths array (0-$((N-1))%2, ${WALLTIME}, ${EPOCHS} epochs)"
sbatch "${COMMON[@]}" --time="$WALLTIME" --array="0-$((N-1))%2" \
  --job-name=alignn_kvrh_tenth \
  --output="logs/alignn_kvrh_tenth_%A_%a.out" --error="logs/alignn_kvrh_tenth_%A_%a.err" \
  --export=ALL,PART_SPECFILE="$SPEC",PART_EPOCHS="$EPOCHS" "$JOB"

echo "Submitted. Track: squeue -u \$USER | grep alignn_kvrh_tenth"
