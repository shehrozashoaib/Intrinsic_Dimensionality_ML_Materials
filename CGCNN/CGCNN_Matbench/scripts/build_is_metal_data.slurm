#!/bin/bash --login
# Stage the matbench_mp_is_metal dataset for all three models:
#   1) prepare_matbench_splits.py -> cached_matbench/raw/matbench_mp_is_metal_seed123_raw.pkl
#      (official 106,113-entry task; matminer json.gz must already be cached in the env)
#   2) build_is_metal_idprop.py   -> ALIGNN/.../MP_json_is_metal/id_prop.json
#      (unified {material_id, is_metal 0/1, atoms} records that feed CGCNN/ALIGNN/DimeNet++)
#
# Submit from CGCNN_Matbench/:
#   sbatch scripts/build_is_metal_data.slurm
#SBATCH --job-name=build_is_metal
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/build_is_metal_%j.out
#SBATCH --error=logs/build_is_metal_%j.err

set -euo pipefail
mkdir -p logs

source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312

echo "[INFO] host=$(hostname) start=$(date -Iseconds)"

echo "[STEP 1] prepare_matbench_splits.py (matbench_mp_is_metal)"
python prepare_matbench_splits.py \
  --tasks matbench_mp_is_metal \
  --output-dir cached_matbench/raw \
  --seed 123

echo "[STEP 2] build_is_metal_idprop.py -> MP_json_is_metal/id_prop.json"
python ../../ALIGNN/ALIGNN_MP/alignn/build_is_metal_idprop.py

echo "[INFO] done=$(date -Iseconds)"
