#!/bin/bash --login
# Quick CPU benchmark: serial vs multiprocessing graph build. Submit from DimeNetPP_MP.
#SBATCH --job-name=dim_bench_par
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/dim_bench_par_%j.out
#SBATCH --error=logs/dim_bench_par_%j.err

set -euo pipefail
mkdir -p logs
source scripts/dimenet_env.sh

JSONL="${DIMENET_BANDGAP_JSONL:-../../ALIGNN/ALIGNN_MP/alignn/mp21.jsonl}"
N="${BENCH_N:-4000}"
WORKERS="${BENCH_WORKERS:-8,16}"

echo "Node: $(hostname)  cores=$(nproc)  N=$N workers=$WORKERS"
echo "Start: $(date -Iseconds)"
"$DIMENET_PYTHON" dimenetpp_code_only/bench_parallel_build.py --jsonl "$JSONL" --n "$N" --workers "$WORKERS"
echo "End: $(date -Iseconds)"
