#!/bin/bash --login
# Parallel build-once + reshuffle into 3 seed splits, for one dataset (16-worker spawn pool).
# Pick dataset with DIMENET_DATASET=eform|bandgap.
# Submit from DimeNetPP_MP:  sbatch --export=ALL,DIMENET_DATASET=eform  scripts/build_tensors_parallel.slurm
#SBATCH --job-name=dim_build_par
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dim_build_par_%x_%j.out
#SBATCH --error=logs/dim_build_par_%x_%j.err

set -euo pipefail
mkdir -p logs
source scripts/dimenet_env.sh

CODE_DIR="dimenetpp_code_only"
DATASET="${DIMENET_DATASET:?set DIMENET_DATASET=eform|bandgap}"
WORKERS="${DIMENET_BUILD_WORKERS:-16}"
SEEDS="${DIMENET_BUILD_SEEDS:-1123,1456,1789}"
LIMIT="${DIMENET_DATA_LIMIT:-0}"
# build chunk temp files on fast node-local scratch if available
TMPDIR_BUILD="${DIMENET_BUILD_TMPDIR:-${TMPDIR:-/tmp}/dimenet_build_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR_BUILD"

if [[ "$DATASET" == "eform" ]]; then
  SOURCE="eform_idprop"
  INPUT="${DIMENET_EFORM_ID_PROP:-../../ALIGNN/ALIGNN_MP/alignn/MP_json_eform/id_prop.json}"
  TARGET_KEY="formation_energy_per_atom"
  SAVE_ROOT="${DIMENET_CACHE_ROOT:-cached_tensors_dimenetpp_eform}"
  TENSOR_FN="dimenetpp_eform_cached_tensors.pkl"
  SCALER_FN="scaler_eform.pkl"
elif [[ "$DATASET" == "bandgap" ]]; then
  SOURCE="mp21_jsonl"
  INPUT="${DIMENET_BANDGAP_JSONL:-../../ALIGNN/ALIGNN_MP/alignn/mp21.jsonl}"
  TARGET_KEY="band_gap"
  SAVE_ROOT="${DIMENET_BANDGAP_CACHE_ROOT:-cached_tensors_dimenetpp_bandgap}"
  TENSOR_FN="dimenetpp_bandgap_cached_tensors.pkl"
  SCALER_FN="scaler_bandgap.pkl"
elif [[ "$DATASET" == "log_kvrh" ]]; then
  # Reuse the generic ALIGNN id_prop.json reader (JARVIS atoms -> pymatgen). The log_kvrh
  # id_prop.json was built by ALIGNN/ALIGNN_MP/alignn/build_log_kvrh_idprop.py.
  SOURCE="eform_idprop"
  INPUT="${DIMENET_KVRH_ID_PROP:-../../ALIGNN/ALIGNN_MP/alignn/MP_json_log_kvrh/id_prop.json}"
  TARGET_KEY="log_kvrh"
  SAVE_ROOT="${DIMENET_KVRH_CACHE_ROOT:-cached_tensors_dimenetpp_kvrh}"
  TENSOR_FN="dimenetpp_kvrh_cached_tensors.pkl"
  SCALER_FN="scaler_kvrh.pkl"
else
  echo "Unknown DIMENET_DATASET=$DATASET" >&2; exit 1
fi

if [[ ! -f "$INPUT" ]]; then echo "ERROR: input not found: $INPUT" >&2; exit 1; fi

echo "Node: $(hostname)  cores=$(nproc)  dataset=$DATASET workers=$WORKERS seeds=$SEEDS"
echo "Input: $INPUT  -> $SAVE_ROOT"
echo "Start: $(date -Iseconds)"

"$DIMENET_PYTHON" "${CODE_DIR}/data_saving_parallel.py" \
  --source "$SOURCE" --input "$INPUT" \
  --target_key "$TARGET_KEY" --id_key material_id \
  --seeds "$SEEDS" --workers "$WORKERS" --limit "$LIMIT" \
  --save_root "$SAVE_ROOT" \
  --tensor_filename "$TENSOR_FN" --scaler_filename "$SCALER_FN" \
  --tmpdir "$TMPDIR_BUILD"

rm -rf "$TMPDIR_BUILD" 2>/dev/null || true
echo "End: $(date -Iseconds)"
