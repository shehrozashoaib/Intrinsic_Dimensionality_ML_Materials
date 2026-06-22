#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=dimenet_run.py
OUTDIR=runs_dimenetpp
CACHEDIR=./cached_tensors_dimenetpp

EPOCHS=350
BATCH=64
SEED=123
GPU=0

DIMS=(0.05 0.1 0.2 0.5 0.8 1.0)

mkdir -p "${OUTDIR}" logs

for dim in "${DIMS[@]}"; do
  tag="fastfood_dim$(python - <<PY
d=float("${dim}")
print(int(round(d*100)))
PY
)pct_epochs${EPOCHS}_seed${SEED}"
  logfile="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

  echo "===== Running ${tag} =====" | tee -a "${logfile}"
  ${PY} ${SCRIPT} \
    --method fastfood \
    --id_dim "${dim}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --cache_dir "${CACHEDIR}" \
    --out_dir "${OUTDIR}" \
    > "${logfile}" 2>&1

  echo "===== Finished ${tag}. Sleeping 10 minutes... =====" | tee -a "${logfile}"
  sleep 600
done

echo "All runs completed."