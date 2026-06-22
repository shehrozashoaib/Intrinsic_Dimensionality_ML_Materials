#!/usr/bin/env bash
# Submit all 16 DimeNet++ formation-energy Fastfood runs as ONE Slurm job array.
# Each array task = one (dim, seed) run on a single GPU (batch/sbatch format, never an
# interactive whole-GPU allocation). The sweep script maps SLURM_ARRAY_TASK_ID -> RUNS[idx].
# Run from DimeNetPP/DimeNetPP_MP: bash scripts/submit_eform_all.sh
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs

JOB="scripts/submit_eform_job.slurm"
# GPU type for the full sweep. Defaults to V100 (32GB); set DIMENET_SWEEP_CONSTRAINT=a100
# after the bench if the A100 turns out meaningfully faster.
SWEEP_CONSTRAINT="${DIMENET_SWEEP_CONSTRAINT:-v100}"
WALLTIME="${DIMENET_WALLTIME:-24:00:00}"

# 16 runs -> array indices 0-15. Leave the concurrency cap off by default (all at once is
# fine). Set DIMENET_ARRAY_CAP=N to throttle to N concurrent tasks, e.g. --array=0-15%N.
ARRAY_SPEC="0-15"
if [[ -n "${DIMENET_ARRAY_CAP:-}" ]]; then
  ARRAY_SPEC="0-15%${DIMENET_ARRAY_CAP}"
fi

echo "Submitting DimeNet++ eform sweep as array ${ARRAY_SPEC} on ${SWEEP_CONSTRAINT} (walltime ${WALLTIME})..."
sbatch \
  --gpus=1 --constraint="$SWEEP_CONSTRAINT" --cpus-per-task=16 --mem=128G \
  --time="$WALLTIME" --array="$ARRAY_SPEC" --job-name=dim_eform_sweep \
  --output="logs/dim_eform_sweep_%A_%a.out" --error="logs/dim_eform_sweep_%A_%a.err" \
  "$JOB"

echo
echo "Submitted 16-run array. Track with: squeue -u \$USER"
echo "Per-run metadata: results_dimenetpp_eform_fastfood/<run>/metadata.json"
