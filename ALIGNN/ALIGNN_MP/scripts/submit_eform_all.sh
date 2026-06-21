#!/usr/bin/env bash
# Submit all formation-energy runs with one Slurm job per (dim, seed).
# Run from ALIGNN/ALIGNN_MP:   bash scripts/submit_eform_all.sh
#
# Each job writes per-run metadata.json only; run aggregate_eform_summary.py
# afterwards to build the combined CSV (safe with parallel jobs). The runner
# skips an output directory that already has metadata.json with status=success.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> ALIGNN/ALIGNN_MP
mkdir -p logs

JOB="scripts/submit_eform_job.slurm"
COMMON=(--gpus=1 --constraint=a100 --cpus-per-task=16 --mem=128G)

submit() {  # name  time  runs
  local name="$1" time="$2" runs="$3"
  sbatch "${COMMON[@]}" --time="$time" --job-name="$name" \
    --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
    --export=ALL,EFORM_RUNS="$runs" "$JOB"
}

WALLTIME="${EFORM_WALLTIME:-24:00:00}"

echo "Submitting one job per dim/seed with walltime ${WALLTIME}..."
submit eform_d10_s123 "$WALLTIME" "0.1:10:123:1123"
submit eform_d10_s456 "$WALLTIME" "0.1:10:456:1456"
submit eform_d20_s123 "$WALLTIME" "0.2:20:123:1123"
submit eform_d20_s456 "$WALLTIME" "0.2:20:456:1456"
submit eform_d45_s123 "$WALLTIME" "0.45:45:123:1123"
submit eform_d45_s456 "$WALLTIME" "0.45:45:456:1456"
submit eform_d45_s789 "$WALLTIME" "0.45:45:789:1789"
submit eform_d50_s123 "$WALLTIME" "0.5:50:123:1123"
submit eform_d50_s456 "$WALLTIME" "0.5:50:456:1456"
submit eform_d60_s123 "$WALLTIME" "0.6:60:123:1123"
submit eform_d60_s456 "$WALLTIME" "0.6:60:456:1456"
submit eform_d60_s789 "$WALLTIME" "0.6:60:789:1789"
submit eform_d80_s123 "$WALLTIME" "0.8:80:123:1123"
submit eform_d80_s456 "$WALLTIME" "0.8:80:456:1456"
submit eform_d100_s123 "$WALLTIME" "1.0:100:123:1123"
submit eform_d100_s456 "$WALLTIME" "1.0:100:456:1456"

echo
echo "Submitted 16 jobs (one run per job). Track with: squeue -u \$USER"
echo "When all finish: conda run -n py312 python scripts/aggregate_eform_summary.py"
