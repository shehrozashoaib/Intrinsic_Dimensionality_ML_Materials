#!/bin/bash --login
# Quick smoke test: CGCNN n_conv=2 ("doubled" model), dim=100%, dielectric (smallest task), 2 epochs.
# Just checks the model constructs, prints param count, and fits on an A100 without OOM.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=cgcnn_smoke_2layer
#SBATCH --output=logs/cgcnn_smoke_2layer_%j.out
#SBATCH --error=logs/cgcnn_smoke_2layer_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312
echo "[INFO] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python main.py \
  --cached-data-dir cached_matbench/tensors \
  --matbench-task matbench_dielectric \
  --output-dir /tmp/cgcnn_smoke_2layer \
  --epochs 2 --batch-size 512 --optim Adam \
  --n-conv 2 --atom-fea-len 82 --h-fea-len 128 --n-h 3 \
  --subspace-method fastfood --id-dim 1.0 \
  --data-seed 123 --random-seed 123 --print-freq 1

echo "[INFO] peak GPU mem (MiB):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[INFO] done=$(date -Iseconds)"
