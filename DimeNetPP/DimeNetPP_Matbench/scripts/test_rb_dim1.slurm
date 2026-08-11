#!/bin/bash --login
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rb_dim1
#SBATCH --output=logs/rb_dim1_%A_%a.out
set -uo pipefail
source scripts/dimenet_env.sh
SEEDS=(123 456 789); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; SS=$((S+1000))
O="restore_best_test/kvrh_full_dim1_seed${S}"; mkdir -p "$O"
# identical to the conv-width 1x baseline (ies=64, clipnorm=1.0, 180ep, full log_kvrh),
# ONLY difference: --restore_best (test the best-val checkpoint instead of the final epoch)
"$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_kvrh_v3.py \
  --method fastfood --id_dim 0.01 --num_blocks 1 --int_emb_size 64 --clipnorm 1.0 \
  --epochs 180 --batch_size 64 --seed $S --split_seed $SS \
  --cache_dir cached_tensors_dimenetpp_kvrh/splitseed${SS} \
  --cache_file dimenetpp_kvrh_cached_tensors.pkl --scaler_file scaler_kvrh.pkl \
  --out_dir "$O" --restore_best 2>&1 | tee "$O/train.log"
