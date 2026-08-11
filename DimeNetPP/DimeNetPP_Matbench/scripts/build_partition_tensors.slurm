#!/bin/bash --login
# Build DimeNet++ graph caches for the log_kvrh QUARTER + TENTH partitions.
# One array task = one partition; each builds both split seeds (1123, 1456).
#
#   sbatch --array=0-3 scripts/build_partition_tensors.slurm
#SBATCH --job-name=dim_build_part_kvrh
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dim_build_part_kvrh_%A_%a.out
#SBATCH --error=logs/dim_build_part_kvrh_%A_%a.err

set -euo pipefail
mkdir -p logs
source scripts/dimenet_env.sh

PARTS=(kvrh_quarter_1 kvrh_quarter_2 kvrh_tenth_1 kvrh_tenth_2)
PART="${PARTS[$SLURM_ARRAY_TASK_ID]}"

INPUT="../../ALIGNN/ALIGNN_MP/alignn/partitions/${PART}/id_prop.json"
SAVE_ROOT="cached_tensors_dimenetpp_part_${PART}"
SEEDS="${DIMENET_BUILD_SEEDS:-1123,1456}"
TMPDIR_BUILD="${TMPDIR:-/tmp}/dimenet_part_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR_BUILD"

[[ -f "$INPUT" ]] || { echo "ERROR: missing $INPUT" >&2; exit 1; }
echo "[INFO] partition=$PART target_key=log_kvrh seeds=$SEEDS"
echo "[INFO] input=$INPUT -> $SAVE_ROOT  start=$(date -Iseconds)"

"$DIMENET_PYTHON" dimenetpp_code_only/data_saving_parallel.py \
  --source eform_idprop --input "$INPUT" \
  --target_key log_kvrh --id_key material_id \
  --seeds "$SEEDS" --workers 16 --limit 0 \
  --save_root "$SAVE_ROOT" \
  --tensor_filename "dimenetpp_kvrh_cached_tensors.pkl" \
  --scaler_filename "scaler_kvrh.pkl" \
  --tmpdir "$TMPDIR_BUILD"

rm -rf "$TMPDIR_BUILD" 2>/dev/null || true
echo "[INFO] done=$(date -Iseconds)"
