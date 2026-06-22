#!/usr/bin/env bash
# Submit identical 1-epoch full-model (100%) timing jobs to V100 (32GB) and A100,
# so we can compare per-epoch wall time and point the full sweep at the faster card.
# Run from DimeNetPP/DimeNetPP_MP:  bash scripts/submit_eform_bench.sh
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs

JOB="scripts/submit_eform_bench.slurm"
COMMON=(--gpus=1 --cpus-per-task=16 --mem=128G --time=02:00:00)

echo "Submitting 1-epoch 100% bench jobs (V100 vs A100)..."
sbatch "${COMMON[@]}" --constraint=v100 --job-name=dim_eform_bench_v100 \
  --output="logs/dim_eform_bench_v100_%j.out" --error="logs/dim_eform_bench_v100_%j.err" \
  --export=ALL,DIMENET_RESULT_ROOT=results_dimenetpp_eform_bench_v100 "$JOB"
sbatch "${COMMON[@]}" --constraint=a100 --job-name=dim_eform_bench_a100 \
  --output="logs/dim_eform_bench_a100_%j.out" --error="logs/dim_eform_bench_a100_%j.err" \
  --export=ALL,DIMENET_RESULT_ROOT=results_dimenetpp_eform_bench_a100 "$JOB"

echo "Submitted 2 bench jobs. Compare 'Training wall time' / per-epoch in the two logs."
