#!/bin/bash --login
# One-GPU DimeNet++ partition sweep job for matbench log_kvrh.
# Runs the full 12-dim x SEEDS grid for ONE partition at num_blocks=1, int_emb_size=64, clipnorm=0.
#
# DIMENET_PART / DIMENET_SEEDS are read from the environment; pass them with --export:
#
#   sbatch --export=ALL,DIMENET_PART=kvrh_quarter_1 scripts/submit_kvrh_partition_job.slurm
#
#   sbatch --export=ALL,DIMENET_PART=kvrh_tenth_2,DIMENET_SEEDS="123:1123 456:1456" \
#          scripts/submit_kvrh_partition_job.slurm
#
# Optional overrides: DIMENET_EPOCHS (default 180), DIMENET_CLIPNORM (default 0),
# DIMENET_BATCH_SIZE (default 64).
# V100 is correct here -- DimeNet++ is TensorFlow and runs on V100.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=dim_kvrh_part
#SBATCH --output=logs/dim_kvrh_part_%j.out
#SBATCH --error=logs/dim_kvrh_part_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "Partition: ${DIMENET_PART:-<unset>}"
echo "Seeds: ${DIMENET_SEEDS:-123:1123 456:1456}"
echo "Epochs: ${DIMENET_EPOCHS:-180}"
echo "Clipnorm: ${DIMENET_CLIPNORM:-0}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_kvrh_partition_sweep.sh

echo "End: $(date -Iseconds)"
