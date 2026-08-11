#!/bin/bash --login
# Build disjoint subset partitions (3x10%, 10x1%) for eform and band gap ALIGNN datasets.
# CPU-only (loads ~250-280MB JSON). Submit from ALIGNN/ALIGNN_MP:
#   sbatch scripts/partition_alignn_datasets.slurm
#SBATCH --job-name=alignn_partition
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/alignn_partition_%j.out
#SBATCH --error=logs/alignn_partition_%j.err

set -euo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "[partition] eform"
python scripts/partition_alignn_datasets.py \
  --source alignn/MP_json_eform/id_prop.json \
  --out_root alignn/partitions \
  --prefix eform --shuffle_seed 0 --n_10pct 3 --n_1pct 10

echo "[partition] bandgap"
python scripts/partition_alignn_datasets.py \
  --source alignn/MP_json_seed456/id_prop.json \
  --out_root alignn/partitions \
  --prefix bandgap --shuffle_seed 0 --n_10pct 3 --n_1pct 10

echo "[partition] done $(date -Iseconds)"
ls -1 alignn/partitions | sort
