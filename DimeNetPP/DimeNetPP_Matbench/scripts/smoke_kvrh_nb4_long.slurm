#!/bin/bash --login
# Convergence check: does num_blocks=4 recover from its init loss spike within a modest epoch
# budget? Run 25 epochs at dim=0.5 and watch val_loss / MAE.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=dim_kvrh_smoke4
#SBATCH --output=logs/dim_kvrh_smoke4_%j.out
#SBATCH --error=logs/dim_kvrh_smoke4_%j.err

set -uo pipefail
mkdir -p logs
module purge
source scripts/dimenet_env.sh

"$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_kvrh_v3.py \
  --method fastfood --id_dim 0.5 --num_blocks 4 \
  --epochs 25 --batch_size 64 --seed 123 --split_seed 1123 --gpu 0 \
  --cache_dir cached_tensors_dimenetpp_kvrh/splitseed1123 \
  --out_dir results_dimenetpp_log_kvrh_smoke/nb4_ep25_clip
