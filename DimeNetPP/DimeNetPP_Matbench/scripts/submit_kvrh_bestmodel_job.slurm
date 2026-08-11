#!/bin/bash --login
# One-GPU DimeNet++ BEST-MODEL sweep job (matbench log_kvrh).
# Runs the full 12-dim x SEEDS grid for ONE dataset size at num_blocks=1, int_emb_size=64,
# clipnorm=0, with --restore_best + EarlyStopping patience=50.
#
# DIMENET_DATASET / DIMENET_SEEDS are read from the environment; pass them with --export:
#
#   sbatch --export=ALL,DIMENET_DATASET=full \
#          scripts/submit_kvrh_bestmodel_job.slurm
#
#   sbatch --export=ALL,DIMENET_DATASET=quarter_1 \
#          scripts/submit_kvrh_bestmodel_job.slurm
#
#   # override the seed grid explicitly (defaults: full = 3 seeds, partitions = 2 seeds)
#   sbatch --export=ALL,DIMENET_DATASET=tenth_1,DIMENET_SEEDS="123:1123 456:1456" \
#          scripts/submit_kvrh_bestmodel_job.slurm
#
# DIMENET_DATASET: full | quarter_1 | quarter_2 | tenth_1 | tenth_2
#   (partition tensor caches live under cached_tensors_dimenetpp_part_kvrh_<dataset>/)
# Optional overrides: DIMENET_EPOCHS (default 180), DIMENET_PATIENCE (default 50),
# DIMENET_CLIPNORM (default 0), DIMENET_BATCH_SIZE (default 64).
# V100 is correct here -- DimeNet++ is TensorFlow and runs on V100.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=dim_kvrh_best
#SBATCH --output=logs/dim_kvrh_best_%j.out
#SBATCH --error=logs/dim_kvrh_best_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "Task: log_kvrh"
echo "Dataset: ${DIMENET_DATASET:-<unset>}"
echo "Seeds: ${DIMENET_SEEDS:-<default: 3 seeds for full, 2 for partitions>}"
echo "Epochs: ${DIMENET_EPOCHS:-180}"
echo "Patience: ${DIMENET_PATIENCE:-50}"
echo "Clipnorm: ${DIMENET_CLIPNORM:-0}"
echo "Restore best: always on"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_kvrh_bestmodel_sweep.sh

echo "End: $(date -Iseconds)"
