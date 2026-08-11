#!/bin/bash --login
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=dim_convsmoke
#SBATCH --output=logs/dim_convsmoke_%A_%a.out
#SBATCH --error=logs/dim_convsmoke_%A_%a.err
set -uo pipefail
source scripts/dimenet_env.sh
# array idx -> (int_emb_size, clipnorm)
CFG=(64:0 128:0 256:0 64:1.0 128:1.0 256:1.0)
IFS=':' read -r IES CLIP <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
OUT="smoke_convwidth/ies${IES}_clip${CLIP}"
mkdir -p "$OUT"
echo "[SMOKE] int_emb_size=$IES clipnorm=$CLIP  (num_blocks=1, dim=100%, 120 epochs)"
"$DIMENET_PYTHON" dimenetpp_code_only/dimenet_run_kvrh_v3.py \
  --method fastfood --id_dim 1.0 \
  --num_blocks 1 --int_emb_size "$IES" --clipnorm "$CLIP" \
  --epochs 120 --batch_size 64 --seed 123 --split_seed 1123 \
  --cache_dir cached_tensors_dimenetpp_kvrh/splitseed1123 \
  --out_dir "$OUT" 2>&1 | tee "$OUT/train.log"
echo "[SMOKE DONE] ies=$IES clip=$CLIP"
