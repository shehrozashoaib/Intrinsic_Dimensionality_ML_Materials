#!/usr/bin/env bash
# Refresh the eform + band-gap sweep summary CSVs from per-run metadata.json.
# Re-run any time: bash scripts/make_summaries.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${DIMENET_PYTHON:-envs/pydimnet/bin/python}"

"$PY" scripts/summarize_sweep.py \
  --results_root results_dimenetpp_eform_fastfood \
  --task mp_eform \
  --out results_dimenetpp_eform_fastfood/dimenetpp_eform_fastfood_summary.csv
echo
"$PY" scripts/summarize_sweep.py \
  --results_root results_dimenetpp_bandgap_fastfood \
  --task mp_bandgap \
  --out results_dimenetpp_bandgap_fastfood/dimenetpp_bandgap_fastfood_summary.csv
