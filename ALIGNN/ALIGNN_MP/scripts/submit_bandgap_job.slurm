#!/bin/bash --login
# Parametrized one-GPU ALIGNN band-gap job. The run(s) to execute are passed via the
# BANDGAP_RUNS env var (set by submit_bandgap_remaining.sh). #SBATCH values are fallback
# defaults; the submit script overrides --time/--job-name/log paths per job.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=bandgap_job
#SBATCH --output=logs/bandgap_job_%j.out
#SBATCH --error=logs/bandgap_job_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

echo "Node: $(hostname)"
echo "Runs: ${BANDGAP_RUNS:-<none>}"
echo "Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/run_alignn_fastfood_bandgap_sweep.sh

echo "End: $(date -Iseconds)"
