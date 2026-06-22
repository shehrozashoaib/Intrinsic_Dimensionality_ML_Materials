#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/venv/pydimnet/lib/python3.12/site-packages/nvidia/cusolver/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/cufft/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/cusparse/lib:/venv/pydimnet/lib/python3.12/site-packages/nvidia/nccl/lib:/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

cd /workspace/dimenet++
PYTHON_BIN=/venv/pydimnet/bin/python
EPOCHS=350
BATCH=64
SEED=123
GPU=0
BLOCKCOLS=512
OUTDIR=./runs_dimenetpp_v3_dense
CACHE_DIR=./cached_tensors_dimenetpp
LOGDIR=./logs_v3
mkdir -p "$LOGDIR"

for DIM in 0.05 0.1 0.2 0.5 0.8 1.0; do
  DIMPCT=$(python - <<EOF
print(int(round(float("$DIM") * 100)))
EOF
)
  LOGFILE="$LOGDIR/dense_dim${DIMPCT}pct_epochs${EPOCHS}_seed${SEED}.log"
  "$PYTHON_BIN" ./dimenet_run_v3.py     --method dense     --id_dim "$DIM"     --dense_block_cols "$BLOCKCOLS"     --epochs "$EPOCHS"     --batch_size "$BATCH"     --seed "$SEED"     --gpu "$GPU"     --cache_dir "$CACHE_DIR"     --out_dir "$OUTDIR"     > "$LOGFILE" 2>&1
done
