#!/usr/bin/env bash
# Submit ONLY the band-gap runs not completed on the other device.
# Completed (12): seed123 all dims (100,80,60,50,45,20,10,5); seed456 dims 100,80,60,50.
# Remaining (6): seed456 dims 45,20,10,5 ; seed789 dims 60,45.
# One Slurm job per run (each ~14-15h for 350 epochs). Datasets must be pre-created
# (run scripts/precreate_bandgap_datasets.sh first) so concurrent jobs don't race.
#
# Run from ALIGNN/ALIGNN_MP:  bash scripts/submit_bandgap_remaining.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

JOB="scripts/submit_bandgap_job.slurm"
COMMON=(--gpus=1 --constraint=a100 --cpus-per-task=16 --mem=128G)
WALLTIME="${BANDGAP_WALLTIME:-20:00:00}"   # ~15h/run + buffer

submit() {  # name  runs("root_dir:seed:id_dim:id_dim_percent")
  local name="$1" runs="$2"
  sbatch "${COMMON[@]}" --time="$WALLTIME" --job-name="$name" \
    --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
    --export=ALL,BANDGAP_RUNS="$runs" "$JOB"
}

echo "Submitting 6 remaining band-gap runs (walltime ${WALLTIME})..."
# seed 456 (root MP_json_seed456) -- remaining dims 45,20,10,5
submit bg_d45_s456  "MP_json_seed456:456:0.45:45"
submit bg_d20_s456  "MP_json_seed456:456:0.2:20"
submit bg_d10_s456  "MP_json_seed456:456:0.1:10"
submit bg_d05_s456  "MP_json_seed456:456:0.05:5"
# seed 789 (root MP_json_seed789) -- only dims 60,45
submit bg_d60_s789  "MP_json_seed789:789:0.6:60"
submit bg_d45_s789  "MP_json_seed789:789:0.45:45"

echo
echo "Submitted 6 jobs. Track: squeue -u \$USER | grep bg_d"
echo "Results -> results_alignn_bandgap_fastfood/<run>/ ; aggregate when done."
