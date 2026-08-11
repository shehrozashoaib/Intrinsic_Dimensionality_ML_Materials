#!/bin/bash --login
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rb_bg
#SBATCH --output=logs/rb_bg_%A_%a.out
set -uo pipefail
source scripts/dimenet_env.sh
SEEDS=(123 456); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; SS=$((S+1000))
O="restore_best_test/bandgap_q1_dim100_seed${S}"; mkdir -p "$O"
"$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_bandgap_v3.py \
  --method fastfood --id_dim 1.0 --num_blocks 1 --clipnorm 0 \
  --epochs 350 --batch_size 64 --seed $S --split_seed $SS \
  --cache_dir cached_tensors_dimenetpp_part_bandgap_quarter_1/splitseed${SS} \
  --cache_file dimenetpp_bandgap_cached_tensors.pkl --scaler_file scaler_bandgap.pkl \
  --out_dir "$O" --restore_best 2>&1 | tee "$O/train.log"
