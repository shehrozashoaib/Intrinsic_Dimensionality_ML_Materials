#!/bin/bash --login
# Quick smoke test: ALIGNN alignn_layers=2, gcn_layers=2 ("doubled" model), dim=100%, dielectric
# (smallest task), 2 epochs. Just checks the model constructs and fits on an A100 without OOM.
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=alignn_smoke_2layer
#SBATCH --output=logs/alignn_smoke_2layer_%j.out
#SBATCH --error=logs/alignn_smoke_2layer_%j.err

set -uo pipefail
mkdir -p logs
source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

echo "[INFO] host=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p /tmp/alignn_smoke_2layer
cd alignn
python train_alignn.py \
  --root_dir MP_json_dielectric \
  --config_name ../configs/config_dielectric_2layer.json \
  --target_key dielectric \
  --id_key material_id \
  --subspace_method fastfood \
  --id_dim 1.0 \
  --id_enable True \
  --epochs 2 \
  --output_dir /tmp/alignn_smoke_2layer \
  --random_seed 123 \
  --split_seed 1123

echo "[INFO] peak GPU mem (MiB):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[INFO] done=$(date -Iseconds)"
