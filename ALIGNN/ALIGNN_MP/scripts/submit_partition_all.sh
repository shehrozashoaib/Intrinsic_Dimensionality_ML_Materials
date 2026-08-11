#!/usr/bin/env bash
# Submit the full ALIGNN partition (data-subset) sweep as two capped job arrays.
#   10% array: one task per (task, partition, seed) = all dims          (12 tasks, %4, 23h)
#   1%  array: one task per (task, partition)       = both seeds x dims (20 tasks, %2,  7h)
# %4 + %2 => at most 6 concurrent A100 jobs.
#
# Seeds 123 & 456. Dims: eform {100,80,60,50,45,20,10}, bandgap {..,5}.
# Run from ALIGNN/ALIGNN_MP:  bash scripts/submit_partition_all.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

SEEDS=(123 456)
EFORM_DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.45:45" "0.2:20" "0.1:10")
BANDGAP_DIMS=("1.0:100" "0.8:80" "0.6:60" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.05:5")

dims_for() { if [[ "$1" == eform ]]; then printf '%s\n' "${EFORM_DIMS[@]}"; else printf '%s\n' "${BANDGAP_DIMS[@]}"; fi; }

SPEC10="logs/partspec_10pct.txt"; : > "$SPEC10"
SPEC01="logs/partspec_01pct.txt"; : > "$SPEC01"

# 10%: per (task, partition, seed) -> all dims for that seed
for task in eform bandgap; do
  for p in 1 2 3; do
    part="${task}_p10_${p}"
    for seed in "${SEEDS[@]}"; do
      runs=""
      while read -r d; do IFS=':' read -r frac pct <<< "$d"; runs+="${part}:${seed}:${frac}:${pct} "; done < <(dims_for "$task")
      echo "${task}|${runs% }" >> "$SPEC10"
    done
  done
done

# 1%: per (task, partition) -> both seeds x all dims
for task in eform bandgap; do
  for p in $(seq 1 10); do
    part="${task}_p01_${p}"
    runs=""
    for seed in "${SEEDS[@]}"; do
      while read -r d; do IFS=':' read -r frac pct <<< "$d"; runs+="${part}:${seed}:${frac}:${pct} "; done < <(dims_for "$task")
    done
    echo "${task}|${runs% }" >> "$SPEC01"
  done
done

N10=$(wc -l < "$SPEC10"); N01=$(wc -l < "$SPEC01")
echo "10% spec: $N10 jobs -> $SPEC10"
echo "1%  spec: $N01 jobs -> $SPEC01"

COMMON=(--gpus=1 --constraint=a100 --cpus-per-task=16 --mem=128G)
JOB="scripts/submit_partition_job.slurm"

echo ">> submit 10% array (0-$((N10-1))%4, 23h)"
sbatch "${COMMON[@]}" --time=23:00:00 --array="0-$((N10-1))%4" \
  --job-name=alignn_part10 \
  --output="logs/alignn_part10_%A_%a.out" --error="logs/alignn_part10_%A_%a.err" \
  --export=ALL,PART_SPECFILE="$SPEC10",PART_EPOCHS=350 "$JOB"

echo ">> submit 1% array (0-$((N01-1))%2, 7h)"
sbatch "${COMMON[@]}" --time=07:00:00 --array="0-$((N01-1))%2" \
  --job-name=alignn_part01 \
  --output="logs/alignn_part01_%A_%a.out" --error="logs/alignn_part01_%A_%a.err" \
  --export=ALL,PART_SPECFILE="$SPEC01",PART_EPOCHS=350 "$JOB"

echo "Submitted. Track: squeue -u \$USER | grep alignn_part"
