#!/bin/bash --login
# DimeNet++ log_kvrh CONV-WIDTH sweep: num_blocks=1 fixed, widen ONLY the interaction (conv) block.
#   array idx 0 -> int_emb_size=64  (1x, baseline)
#   array idx 1 -> int_emb_size=128 (2x)
#   array idx 2 -> int_emb_size=256 (4x)
# Each job runs the full 12-dim x 3-seed grid (36 runs) at 180 epochs (~4-6 h) -> 3 jobs total,
# well under the 20 h window (cram-runs/few-jobs rule).
#
# Clipping is set by DIMENET_CLIPNORM (0 = off). Decided by the conv-width smoke test.
#
# Submit from DimeNetPP_Matbench/:
#   sbatch --array=0-2 --export=ALL,DIMENET_CLIPNORM=0 scripts/submit_convwidth_job.slurm
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=dim_convwidth
#SBATCH --output=logs/dim_convwidth_%A_%a.out
#SBATCH --error=logs/dim_convwidth_%A_%a.err

set -uo pipefail
mkdir -p logs
source scripts/dimenet_env.sh

WIDTHS=(64 128 256)
export INT_EMB_SIZE="${WIDTHS[$SLURM_ARRAY_TASK_ID]}"
export DIMENET_EPOCHS="${DIMENET_EPOCHS:-180}"
export DIMENET_CLIPNORM="${DIMENET_CLIPNORM:-0}"

echo "[INFO] host=$(hostname) int_emb_size=$INT_EMB_SIZE epochs=$DIMENET_EPOCHS clipnorm=$DIMENET_CLIPNORM start=$(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader
bash scripts/run_dimenetpp_fastfood_kvrh_convwidth_sweep.sh
echo "[INFO] done=$(date -Iseconds)"
