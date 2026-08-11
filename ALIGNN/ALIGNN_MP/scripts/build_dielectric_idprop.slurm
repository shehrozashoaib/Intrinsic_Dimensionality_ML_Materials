#!/bin/bash --login
#SBATCH --job-name=build_diel_idprop
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=../logs/build_diel_idprop_%j.out
#SBATCH --error=../logs/build_diel_idprop_%j.err

set -euo pipefail
mkdir -p ../logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Node: $(hostname)  start=$(date -Iseconds)"
python build_dielectric_idprop.py
echo "done=$(date -Iseconds)"
