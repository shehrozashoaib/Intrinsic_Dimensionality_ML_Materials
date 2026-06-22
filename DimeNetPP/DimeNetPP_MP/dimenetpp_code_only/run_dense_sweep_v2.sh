#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=dimenet_run_v2.py
OUTDIR=runs_dimenetpp_v2_dense
CACHEDIR=./cached_tensors_dimenetpp

EPOCHS=350
BATCH=64
SEED=123
GPU=0
BLOCKCOLS=512

DIMS=(0.05 0.1 0.2 0.5 0.8 1.0)

mkdir -p "${OUTDIR}" logs

for dim in "${DIMS[@]}"; do
  tag="wrapperv2_dense_dim$(python - <<PY
print(int(round(float("${dim}")*100)))
PY
)pct_epochs${EPOCHS}_seed${SEED}_blockcols${BLOCKCOLS}"
  logfile="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

  echo "===== Running ${tag} ====="
  ${PY} ${SCRIPT} \
    --method dense \
    --id_dim "${dim}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --dense_block_cols "${BLOCKCOLS}" \
    --cache_dir "${CACHEDIR}" \
    --out_dir "${OUTDIR}" \
    > "${logfile}" 2>&1

  echo "===== Finished ${tag}. Sleeping 10 minutes... ====="
  sleep 600
done

echo "All wrapper-v2 dense runs completed."
