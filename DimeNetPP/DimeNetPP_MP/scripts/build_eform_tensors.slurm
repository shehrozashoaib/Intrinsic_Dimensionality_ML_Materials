#!/bin/bash --login
# CPU array job: build DimeNet++ eform tensor caches, one split seed per array task.
# Submit from DimeNetPP/DimeNetPP_MP:  sbatch scripts/build_eform_tensors.slurm
#SBATCH --job-name=dim_eform_tensors
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/dim_eform_tensors_%A_%a.out
#SBATCH --error=logs/dim_eform_tensors_%A_%a.err

set -euo pipefail
mkdir -p logs

source scripts/dimenet_env.sh

CODE_DIR="dimenetpp_code_only"
ID_PROP="${DIMENET_EFORM_ID_PROP:-../../ALIGNN/ALIGNN_MP/alignn/MP_json_eform/id_prop.json}"
CACHE_ROOT="${DIMENET_CACHE_ROOT:-cached_tensors_dimenetpp_eform}"
LIMIT="${DIMENET_DATA_LIMIT:-0}"

SEEDS=(1123 1456 1789)
split_seed="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
out_dir="${CACHE_ROOT}/splitseed${split_seed}"
tensor_file="${out_dir}/dimenetpp_eform_cached_tensors.npz"

if [[ ! -f "$ID_PROP" ]]; then
  echo "[DimeNet++] ERROR: id_prop.json not found: ${ID_PROP}" >&2
  exit 1
fi

if [[ -f "$tensor_file" && "${DIMENET_FORCE_REBUILD:-0}" != "1" ]]; then
  echo "[DimeNet++] tensor cache already exists, skipping: ${tensor_file}"
  exit 0
fi

mkdir -p "$out_dir"
echo "Node: $(hostname)"
echo "Building tensors split_seed=${split_seed} -> ${out_dir}"
echo "Start: $(date -Iseconds)"

"$DIMENET_PYTHON" "${CODE_DIR}/data_saving_from_alignn_eform.py" \
  --id_prop "$ID_PROP" \
  --seed "$split_seed" \
  --limit "$LIMIT" \
  --gpu -1 \
  --save_dir "$out_dir"

echo "End: $(date -Iseconds)"
