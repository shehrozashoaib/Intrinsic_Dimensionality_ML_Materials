#!/usr/bin/env bash
# Submit all 18 DimeNet++ band-gap Fastfood runs as ONE Slurm job array (sbatch, never an
# interactive whole-GPU allocation). Same dims/seeds as the eform sweep PLUS 5% at 2 seeds.
# Each array task = one (dim, seed) run on a single GPU; the sweep maps SLURM_ARRAY_TASK_ID.
# Run from DimeNetPP/DimeNetPP_MP: bash scripts/submit_bandgap_all.sh
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs

JOB="scripts/submit_bandgap_job.slurm"
SWEEP_CONSTRAINT="${DIMENET_SWEEP_CONSTRAINT:-v100}"
WALLTIME="${DIMENET_WALLTIME:-24:00:00}"

# 18 runs -> array indices 0-17 (16 shared with eform + 2 new 5% rows). All at once by default;
# set DIMENET_ARRAY_CAP=N to throttle to N concurrent tasks (--array=0-17%N).
ARRAY_SPEC="0-17"
if [[ -n "${DIMENET_ARRAY_CAP:-}" ]]; then
  ARRAY_SPEC="0-17%${DIMENET_ARRAY_CAP}"
fi

echo "Submitting DimeNet++ band-gap sweep as array ${ARRAY_SPEC} on ${SWEEP_CONSTRAINT} (walltime ${WALLTIME})..."
sbatch \
  --gpus=1 --constraint="$SWEEP_CONSTRAINT" --cpus-per-task=16 --mem=128G \
  --time="$WALLTIME" --array="$ARRAY_SPEC" --job-name=dim_bandgap_sweep \
  --output="logs/dim_bandgap_sweep_%A_%a.out" --error="logs/dim_bandgap_sweep_%A_%a.err" \
  "$JOB"

echo
echo "Submitted 18-run array. Track with: squeue -u \$USER"
echo "Per-run metadata: results_dimenetpp_bandgap_fastfood/<run>/metadata.json"
