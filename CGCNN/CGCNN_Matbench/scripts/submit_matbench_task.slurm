#!/bin/bash --login
# One A100 job = all 12 dims x 3 seeds for one Matbench task (CGCNN_MB_TASK).
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cgcnn_mb_task
#SBATCH --output=logs/cgcnn_mb_%x_%j.out
#SBATCH --error=logs/cgcnn_mb_%x_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Node: $(hostname)  task=${CGCNN_MB_TASK:-<none>}  start=$(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader
bash scripts/run_cgcnn_matbench_fastfood_sweep.sh
echo "End: $(date -Iseconds)"
