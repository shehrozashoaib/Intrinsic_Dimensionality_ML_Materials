#!/usr/bin/env bash
# Exp 2: ALIGNN Fastfood sweep on matbench log_kvrh HALF-dataset subsets.
# Two disjoint 50% halves (kvrh_half_1, kvrh_half_2; built by alignn/build_log_kvrh_idprop.py).
# Each half x 2 model seeds (123,456) x 12 dims = 48 runs, fastfood, 120 epochs.
#
# Submitted as a job array: one task per (half, seed) = 4 tasks, each running all 12 dims
# sequentially. Reuses scripts/submit_partition_job.slurm (PART_SPECFILE mode, PART_TASK=log_kvrh).
#
# Run from ALIGNN/ALIGNN_MP:  bash scripts/submit_log_kvrh_halves.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

HALVES=(kvrh_half_1 kvrh_half_2)
SEEDS=(123 456)
# "the dims used for CGCNN smaller tasks": 100,80,70,65,50,45,20,10,8,5,2,1
DIMS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")
WALLTIME="${KVRH_HALVES_WALLTIME:-12:00:00}"
EPOCHS="${KVRH_HALVES_EPOCHS:-120}"

SPEC="logs/kvrh_halves_spec.txt"; : > "$SPEC"
for half in "${HALVES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    runs=""
    for d in "${DIMS[@]}"; do
      IFS=':' read -r frac pct <<< "$d"
      runs+="${half}:${seed}:${frac}:${pct} "
    done
    echo "log_kvrh|${runs% }" >> "$SPEC"
  done
done

N=$(wc -l < "$SPEC")
echo "spec: $N array tasks (= halves x seeds), each runs ${#DIMS[@]} dims -> $SPEC"
cat "$SPEC" | sed 's/|/  |  /'

COMMON=(--gpus=1 --constraint=a100 --cpus-per-task=16 --mem=128G)
JOB="scripts/submit_partition_job.slurm"

echo ">> submit log_kvrh halves array (0-$((N-1))%4, ${WALLTIME}, ${EPOCHS} epochs)"
sbatch "${COMMON[@]}" --time="$WALLTIME" --array="0-$((N-1))%4" \
  --job-name=alignn_kvrh_half \
  --output="logs/alignn_kvrh_half_%A_%a.out" --error="logs/alignn_kvrh_half_%A_%a.err" \
  --export=ALL,PART_SPECFILE="$SPEC",PART_EPOCHS="$EPOCHS" "$JOB"

echo "Submitted. Track: squeue -u \$USER | grep alignn_kvrh_half"
