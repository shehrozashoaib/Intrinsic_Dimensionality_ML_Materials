#!/bin/bash --login
# Full-dataset timing smoke: 3 epochs at dim=1.0 (the most expensive dim) to measure
# per-epoch wall time, so we can size --time for the real 350-epoch sweep.
# Submit from CGCNN_Matbench/ :
#   sbatch --export=ALL,CGCNN_TASK=eform   scripts/smoke_cgcnn_mp.slurm
#   sbatch --export=ALL,CGCNN_TASK=bandgap scripts/smoke_cgcnn_mp.slurm
#SBATCH --job-name=cgcnn_smoke_mp
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cgcnn_smoke_mp_%x_%j.out
#SBATCH --error=logs/cgcnn_smoke_mp_%x_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312

echo "[INFO] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export CGCNN_SMOKE=1
export CGCNN_EPOCHS="${CGCNN_EPOCHS:-3}"
export CGCNN_BS="${CGCNN_BS:-256}"
export CGCNN_WORKERS="${CGCNN_WORKERS:-8}"

bash scripts/run_cgcnn_fastfood_mp_sweep.sh
echo "[INFO] done=$(date -Iseconds)"
