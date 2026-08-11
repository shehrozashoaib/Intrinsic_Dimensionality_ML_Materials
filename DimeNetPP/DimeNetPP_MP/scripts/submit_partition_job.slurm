#!/bin/bash --login
# One-GPU DimeNet++ partition sweep job (MP tasks: eform | bandgap).
# Runs the full DIMS x SEEDS grid for ONE (task, partition) at num_blocks=1, clipnorm=0.
#
# DIMENET_TASK / DIMENET_PART / DIMENET_SEEDS are read from the environment; pass them with
# --export so they reach the job:
#
#   sbatch --export=ALL,DIMENET_TASK=eform,DIMENET_PART=eform_quarter_1 \
#          scripts/submit_partition_job.slurm
#
#   sbatch --export=ALL,DIMENET_TASK=bandgap,DIMENET_PART=bandgap_tenth_2,DIMENET_SEEDS="123:1123 456:1456" \
#          scripts/submit_partition_job.slurm
#
# Optional overrides: DIMENET_EPOCHS (default 350), DIMENET_CLIPNORM (default 0),
# DIMENET_BATCH_SIZE (default 64).
# V100 is correct here -- DimeNet++ is TensorFlow and runs on V100.
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=dim_part
#SBATCH --output=logs/dim_part_%j.out
#SBATCH --error=logs/dim_part_%j.err

set -euo pipefail
mkdir -p logs

module purge
source scripts/dimenet_env.sh

echo "Node: $(hostname)"
echo "Task: ${DIMENET_TASK:-<unset>}"
echo "Partition: ${DIMENET_PART:-<unset>}"
echo "Seeds: ${DIMENET_SEEDS:-123:1123 456:1456}"
echo "Epochs: ${DIMENET_EPOCHS:-350}"
echo "Clipnorm: ${DIMENET_CLIPNORM:-0}"
echo "Env: ${DIMENET_ENV_PREFIX}"
echo "Start: $(date -Iseconds)"
nvidia-smi

bash scripts/run_dimenetpp_partition_sweep.sh

echo "End: $(date -Iseconds)"
