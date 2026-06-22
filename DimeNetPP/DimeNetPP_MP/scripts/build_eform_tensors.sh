#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/dimenetpp_code_only"

source "${SCRIPT_DIR}/dimenet_env.sh"

ID_PROP="${DIMENET_EFORM_ID_PROP:-${PROJECT_DIR}/../../ALIGNN/ALIGNN_MP/alignn/MP_json_eform/id_prop.json}"
CACHE_ROOT="${DIMENET_CACHE_ROOT:-${PROJECT_DIR}/cached_tensors_dimenetpp_eform}"
SPLIT_SEEDS="${DIMENET_SPLIT_SEEDS:-1123 1456 1789}"
LIMIT="${DIMENET_DATA_LIMIT:-0}"
GPU="${DIMENET_DATA_GPU:--1}"

if [[ ! -f "$ID_PROP" ]]; then
  echo "[DimeNet++] ERROR: id_prop.json not found: ${ID_PROP}" >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT"

for split_seed in $SPLIT_SEEDS; do
  out_dir="${CACHE_ROOT}/splitseed${split_seed}"
  tensor_file="${out_dir}/dimenetpp_eform_cached_tensors.npz"
  if [[ -f "$tensor_file" && "${DIMENET_FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[DimeNet++] skipping existing tensor cache ${tensor_file}"
    continue
  fi
  mkdir -p "$out_dir"
  echo "[DimeNet++] building tensors for split_seed=${split_seed} -> ${out_dir}"
  "$DIMENET_PYTHON" "${CODE_DIR}/data_saving_from_alignn_eform.py" \
    --id_prop "$ID_PROP" \
    --seed "$split_seed" \
    --limit "$LIMIT" \
    --gpu "$GPU" \
    --save_dir "$out_dir"
done
