#!/usr/bin/env bash
# ============================================================================
# Download the datasets used in this project.
#
#   * Formation energy + band gap  ->  from our Zenodo record
#     DOI: 10.5281/zenodo.21871852
#     Each file is a gzipped JSON list of {material_id, structure, target}.
#     structure = pymatgen Structure.as_dict()  (Structure.from_dict to rebuild)
#
#   * Everything else (dielectric, log-K_VRH, phonons, is_metal) -> Matbench
#     via the `matbench` python package (installed in the CGCNN/ALIGNN env).
#
# Usage:  bash data/download_datasets.sh
# Output: data/MP_eform_dataset.json.gz, data/MP_bandgap_dataset.json.gz,
#         data/matbench/<task>.json.gz
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZENODO_RECORD=21871852
BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"

echo "[1/2] Zenodo (formation energy + band gap) ..."
for f in MP_eform_dataset.json.gz MP_bandgap_dataset.json.gz; do
  if [[ -f "$HERE/$f" ]]; then echo "  have $f"; continue; fi
  echo "  downloading $f"
  curl -fL --retry 3 -o "$HERE/$f" "${BASE}/${f}?download=1"
done
echo "  -> $HERE/MP_{eform,bandgap}_dataset.json.gz"

echo "[2/2] Matbench (dielectric, log_kvrh, phonons, is_metal) ..."
mkdir -p "$HERE/matbench"
# Uses the CGCNN/ALIGNN env (matbench==0.6). Adjust PYTHON if needed.
PYTHON="${MATBENCH_PYTHON:-python}"
"$PYTHON" "$HERE/fetch_matbench.py" --out_dir "$HERE/matbench"
echo "  -> $HERE/matbench/*.json.gz"

echo "[done] datasets in $HERE"
