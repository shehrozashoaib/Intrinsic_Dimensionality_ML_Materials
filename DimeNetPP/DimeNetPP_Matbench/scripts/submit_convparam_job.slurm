#!/bin/bash --login
# DimeNet++ log_kvrh corrected interaction-block parameter sweep.
#
# Earlier "conv-width" jobs used int_emb_size=64/128/256, which only doubled a
# bottleneck subset of the interaction block. This array uses int_emb_size values
# chosen so the full DimNetInteractionPPBlock parameter count is ~1x/2x/4x:
#   baseline int_emb_size=64:  C = 36,096
#   2x       int_emb_size=370: C = 72,204
#   4x       int_emb_size=982: C = 144,420
#
# Array layout: 2 corrected widths x 12 dims x 4 seeds = 96 one-run jobs.
# Submit from DimeNetPP_Matbench/:
#   sbatch --array=0-95%24 scripts/submit_convparam_job.slurm
# Optional:
#   DIMENET_SEEDS="123:1123 456:1456 789:1789" sbatch --array=0-71%24 ...
#   DIMENET_DIMS="1.0:100 0.8:80" sbatch --array=0-15%24 ...
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=dim_convparam
#SBATCH --output=logs/dim_convparam_%A_%a.out
#SBATCH --error=logs/dim_convparam_%A_%a.err

set -uo pipefail

mkdir -p logs
source scripts/dimenet_env.sh || exit 1

WIDTHS=(370 982)
DEFAULT_DIMS=("1.0:100" "0.8:80" "0.7:70" "0.65:65" "0.5:50" "0.45:45" "0.2:20" "0.1:10" "0.08:8" "0.05:5" "0.02:2" "0.01:1")
DEFAULT_SEEDS=("123:1123" "456:1456" "789:1789" "234:1234")

if [[ -n "${DIMENET_DIMS:-}" ]]; then
  read -r -a DIMS <<< "$DIMENET_DIMS"
else
  DIMS=("${DEFAULT_DIMS[@]}")
fi

if [[ -n "${DIMENET_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "$DIMENET_SEEDS"
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

N_WIDTH=${#WIDTHS[@]}
N_DIMS=${#DIMS[@]}
N_SEEDS=${#SEEDS[@]}
N_TOTAL=$((N_WIDTH * N_DIMS * N_SEEDS))
IDX=${SLURM_ARRAY_TASK_ID:-0}

if (( IDX < 0 || IDX >= N_TOTAL )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${IDX} outside valid range 0..$((N_TOTAL - 1))" >&2
  exit 1
fi

seed_idx=$((IDX % N_SEEDS))
dim_idx=$(((IDX / N_SEEDS) % N_DIMS))
width_idx=$((IDX / (N_SEEDS * N_DIMS)))

export INT_EMB_SIZE="${WIDTHS[$width_idx]}"
IFS=':' read -r ID_FRAC ID_PCT <<< "${DIMS[$dim_idx]}"
IFS=':' read -r MODEL_SEED SPLIT_SEED <<< "${SEEDS[$seed_idx]}"
SPLIT_SEED="${SPLIT_SEED:-$MODEL_SEED}"

export DIMENET_RUNS="${ID_FRAC}:${ID_PCT}:${MODEL_SEED}:${SPLIT_SEED}"
export DIMENET_EPOCHS="${DIMENET_EPOCHS:-180}"
export DIMENET_LR="${DIMENET_LR:-0.001}"
export DIMENET_LR_SCHEDULE="${DIMENET_LR_SCHEDULE:-onecycle}"
export DIMENET_RESTORE_BEST="${DIMENET_RESTORE_BEST:-1}"
export DIMENET_PATIENCE="${DIMENET_PATIENCE:-0}"
export DIMENET_CLIPNORM="${DIMENET_CLIPNORM:-0}"
export DIMENET_RESULT_ROOT="${DIMENET_RESULT_ROOT:-${PWD}/results_dimenetpp_log_kvrh_convparam_onecycle_fastfood}"

case "$INT_EMB_SIZE" in
  370)
    CONV_PARAMS=72204
    TOTAL_D=119793
    CONV_MULT="2x"
    ;;
  982)
    CONV_PARAMS=144420
    TOTAL_D=192009
    CONV_MULT="4x"
    ;;
  *)
    CONV_PARAMS="unknown"
    TOTAL_D="unknown"
    CONV_MULT="custom"
    ;;
esac

echo "[INFO] host=$(hostname) array=${IDX}/${N_TOTAL} width=${INT_EMB_SIZE} conv=${CONV_MULT} conv_params=${CONV_PARAMS} total_D=${TOTAL_D} dim=${ID_PCT} seed=${MODEL_SEED}:${SPLIT_SEED} epochs=${DIMENET_EPOCHS} lr=${DIMENET_LR} lr_schedule=${DIMENET_LR_SCHEDULE} restore_best=${DIMENET_RESTORE_BEST} patience=${DIMENET_PATIENCE} clipnorm=${DIMENET_CLIPNORM} start=$(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader

bash scripts/run_dimenetpp_fastfood_kvrh_convwidth_sweep.sh

echo "[INFO] done=$(date -Iseconds)"
