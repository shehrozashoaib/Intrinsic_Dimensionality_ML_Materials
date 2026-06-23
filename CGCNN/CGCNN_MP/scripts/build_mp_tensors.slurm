#!/bin/bash --login
# Build CGCNN graph-tensor caches for the full MP eform / band-gap datasets.
# Build-once with a spawn process pool, then reshuffle into 3 seed splits (1123,1456,1789).
# Uses default graph settings from cgcnn/data.py (max_num_nbr=12, radius=8.0, dmin=0.0, step=0.2).
#
# Submit from CGCNN_Matbench/ :
#   sbatch --export=ALL,CGCNN_DATASET=eform   scripts/build_mp_tensors.slurm
#   sbatch --export=ALL,CGCNN_DATASET=bandgap scripts/build_mp_tensors.slurm
#SBATCH --job-name=cgcnn_build_mp
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cgcnn_build_mp_%x_%j.out
#SBATCH --error=logs/cgcnn_build_mp_%x_%j.err

set -euo pipefail
mkdir -p logs

source /ibex/user/shoaibsa/miniconda3/etc/profile.d/conda.sh
conda activate py312

DATASET="${CGCNN_DATASET:?set CGCNN_DATASET=eform|bandgap}"
WORKERS="${CGCNN_BUILD_WORKERS:-32}"
SEEDS="${CGCNN_BUILD_SEEDS:-1123,1456,1789}"
LIMIT="${CGCNN_DATA_LIMIT:-0}"
OUT_DIR="${CGCNN_OUTPUT_DIR:-cached_mp/tensors}"

if [[ "$DATASET" == "eform" ]]; then
  SOURCE="eform_idprop"
  INPUT="${CGCNN_EFORM_INPUT:-../../ALIGNN/ALIGNN_MP/alignn/MP_json_eform/id_prop.json}"
  TASK_NAME="mp_eform"
elif [[ "$DATASET" == "bandgap" ]]; then
  SOURCE="mp21_jsonl"
  INPUT="${CGCNN_BANDGAP_INPUT:-../../ALIGNN/ALIGNN_MP/alignn/mp21.jsonl}"
  TASK_NAME="mp_bandgap"
else
  echo "Unknown CGCNN_DATASET=$DATASET (expected eform|bandgap)" >&2
  exit 1
fi

echo "[INFO] dataset=$DATASET source=$SOURCE task=$TASK_NAME workers=$WORKERS seeds=$SEEDS"
echo "[INFO] input=$INPUT  out=$OUT_DIR"
echo "[INFO] host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  start=$(date -Iseconds)"

python materialize_mp_tensors.py \
  --source "$SOURCE" \
  --input "$INPUT" \
  --task-name "$TASK_NAME" \
  --seeds "$SEEDS" \
  --output-dir "$OUT_DIR" \
  --workers "$WORKERS" \
  --limit "$LIMIT"

echo "[INFO] done=$(date -Iseconds)"
