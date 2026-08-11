#!/bin/bash --login
# One-GPU ALIGNN partition job. PART_TASK + PART_RUNS (and optional PART_EPOCHS) are
# passed via --export by submit_partition_all.sh. #SBATCH values are fallback defaults;
# the submit script overrides --time/--job-name/log paths per job.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=alignn_part_job
#SBATCH --output=logs/alignn_part_job_%j.out
#SBATCH --error=logs/alignn_part_job_%j.err

set -uo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

# Array mode: if PART_SPECFILE is set, read the (SLURM_ARRAY_TASK_ID)-th line
# "PART_TASK|PART_RUNS" and export both. Otherwise use the env-passed PART_TASK/PART_RUNS.
if [[ -n "${PART_SPECFILE:-}" ]]; then
  line_no=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
  spec_line="$(sed -n "${line_no}p" "$PART_SPECFILE")"
  [[ -z "$spec_line" ]] && { echo "ERROR: no spec at line ${line_no} of ${PART_SPECFILE}" >&2; exit 1; }
  export PART_TASK="${spec_line%%|*}"
  export PART_RUNS="${spec_line#*|}"
fi

echo "Node: $(hostname)"
echo "PART_TASK: ${PART_TASK:-<none>}  PART_EPOCHS: ${PART_EPOCHS:-350}"
echo "PART_RUNS: ${PART_RUNS:-<none>}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_alignn_fastfood_partition_sweep.sh

echo "End: $(date -Iseconds)"
