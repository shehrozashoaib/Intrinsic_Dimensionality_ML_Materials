#!/bin/bash --login
# Build DimeNet++ graph caches for the MP (eform / bandgap) QUARTER + TENTH partitions.
# One array task = one partition; each builds both split seeds (1123, 1456).
#
#   sbatch --array=0-7 scripts/build_partition_tensors.slurm
#
# All partitions are ALIGNN-style id_prop.json (JARVIS atoms dicts + material_id), so the
# generic `eform_idprop` reader works for both tasks -- only --target_key differs.
#SBATCH --job-name=dim_build_part
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dim_build_part_%A_%a.out
#SBATCH --error=logs/dim_build_part_%A_%a.err

set -euo pipefail
mkdir -p logs
source scripts/dimenet_env.sh

PARTS=(
  "eform_quarter_1:formation_energy_per_atom:eform"
  "eform_quarter_2:formation_energy_per_atom:eform"
  "eform_tenth_1:formation_energy_per_atom:eform"
  "eform_tenth_2:formation_energy_per_atom:eform"
  "bandgap_quarter_1:band_gap:bandgap"
  "bandgap_quarter_2:band_gap:bandgap"
  "bandgap_tenth_1:band_gap:bandgap"
  "bandgap_tenth_2:band_gap:bandgap"
)
IFS=':' read -r PART TARGET_KEY TASK <<< "${PARTS[$SLURM_ARRAY_TASK_ID]}"

INPUT="../../ALIGNN/ALIGNN_MP/alignn/partitions/${PART}/id_prop.json"
SAVE_ROOT="cached_tensors_dimenetpp_part_${PART}"
TENSOR_FN="dimenetpp_${TASK}_cached_tensors.pkl"
SCALER_FN="scaler_${TASK}.pkl"
SEEDS="${DIMENET_BUILD_SEEDS:-1123,1456}"
TMPDIR_BUILD="${TMPDIR:-/tmp}/dimenet_part_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR_BUILD"

[[ -f "$INPUT" ]] || { echo "ERROR: missing $INPUT" >&2; exit 1; }
echo "[INFO] partition=$PART task=$TASK target_key=$TARGET_KEY seeds=$SEEDS"
echo "[INFO] input=$INPUT -> $SAVE_ROOT  start=$(date -Iseconds)"

"$DIMENET_PYTHON" dimenetpp_code_only/data_saving_parallel.py \
  --source eform_idprop --input "$INPUT" \
  --target_key "$TARGET_KEY" --id_key material_id \
  --seeds "$SEEDS" --workers 16 --limit 0 \
  --save_root "$SAVE_ROOT" \
  --tensor_filename "$TENSOR_FN" --scaler_filename "$SCALER_FN" \
  --tmpdir "$TMPDIR_BUILD"

rm -rf "$TMPDIR_BUILD" 2>/dev/null || true
echo "[INFO] done=$(date -Iseconds)"
