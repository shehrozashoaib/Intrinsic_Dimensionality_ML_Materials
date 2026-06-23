#!/usr/bin/env bash
# Submit a full CGCNN fastfood MP sweep, GROUPED BY DIM (one job runs all seeds of a dim).
# Two arrays per task:
#   - "double" dims (2 seeds each): 100,80,50,20,10 (+5 for bandgap)
#   - "triple" dims (3 seeds each): 60,45
# Walltimes default to the smoke-based estimate (~4-4.5h/exp) x seeds + buffer, and are
# overridable via env. Concurrency is capped so we don't flood the A100 partition.
#
# Usage (from CGCNN_Matbench/):
#   bash scripts/submit_cgcnn_mp_all.sh eform
#   bash scripts/submit_cgcnn_mp_all.sh bandgap
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs

TASK="${1:?usage: submit_cgcnn_mp_all.sh eform|bandgap}"
CONSTRAINT="${CGCNN_CONSTRAINT:-a100}"
BS="${CGCNN_BS:-256}"
EPOCHS="${CGCNN_EPOCHS:-350}"
CAP="${CGCNN_ARRAY_CAP:-2}"   # max concurrent array tasks per array

case "$TASK" in
  eform)
    N_DOUBLE=5; N_TRIPLE=2
    WALL_DOUBLE="${CGCNN_WALL_DOUBLE:-10:00:00}"   # 2 x ~3.7h + buffer
    WALL_TRIPLE="${CGCNN_WALL_TRIPLE:-14:00:00}"   # 3 x ~3.7h + buffer
    ;;
  bandgap)
    N_DOUBLE=6; N_TRIPLE=2
    WALL_DOUBLE="${CGCNN_WALL_DOUBLE:-11:00:00}"   # 2 x ~4.4h + buffer
    WALL_TRIPLE="${CGCNN_WALL_TRIPLE:-16:00:00}"   # 3 x ~4.4h + buffer
    ;;
  *) echo "Unknown task $TASK (eform|bandgap)" >&2; exit 1 ;;
esac

submit_set() {
  local set_name="$1" ntasks="$2" walltime="$3"
  local last=$((ntasks - 1))
  local spec="0-${last}%${CAP}"
  echo ">> $TASK / $set_name : array=$spec walltime=$walltime constraint=$CONSTRAINT epochs=$EPOCHS bs=$BS"
  sbatch \
    --constraint="$CONSTRAINT" --time="$walltime" --array="$spec" \
    --job-name="cgcnn_mp_${TASK}_${set_name}" \
    --export=ALL,CGCNN_TASK="$TASK",CGCNN_GROUP_SET="$set_name",CGCNN_EPOCHS="$EPOCHS",CGCNN_BS="$BS" \
    scripts/submit_cgcnn_mp_job.slurm
}

submit_set double "$N_DOUBLE" "$WALL_DOUBLE"
submit_set triple "$N_TRIPLE" "$WALL_TRIPLE"
echo "Submitted $TASK sweep (grouped by dim). Track: squeue -u \$USER | grep cgcnn_mp_${TASK}"
