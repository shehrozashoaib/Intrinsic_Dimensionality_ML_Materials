#!/usr/bin/env bash
# Matbench PHONONS intrinsic-dimension sweep, all 3 models, 10 dims x 10 seeds = 300 runs.
#
# Arms (per user spec):
#   CGCNN     : --lr-schedule onecycle (max_lr 1e-3).  Best-model eval is NATIVE (main.py reloads
#               the best checkpoint before test), so it gets it whether or not it is requested.
#   DimeNet++ : --lr_schedule onecycle + --restore_best --patience 0 (ES off so the anneal
#               completes). The phonons runner had NONE of these; ported from the kvrh runner.
#   ALIGNN    : config_phonons.json already uses `scheduler: onecycle`; ALIGNN_RESTORE_BEST=1
#               makes train.py reload best_model.pt before testing (the `best_model = net`
#               alias bug meant it previously scored the FINAL epoch).
#
# 80 epochs, 1 layer (~85k params), fastfood -- identical to the existing 10-seed phonons sweep
# so the two arms are directly comparable.
#
# Runtimes are tiny (CGCNN 0.4 / DimeNet++ 0.8 / ALIGNN 2.0 min per run), so 2 jobs per model
# (5 seeds each) is plenty of parallelism; more jobs would just queue behind the CGCNN MP work.
#
#   bash launch_phonons_onecycle.sh            # dry run
#   SUBMIT=1 bash launch_phonons_onecycle.sh   # submit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C="${ROOT}/CGCNN/CGCNN_Matbench"
D="${ROOT}/DimeNetPP/DimeNetPP_Matbench"
A="${ROOT}/ALIGNN/ALIGNN_MP"
SUBMIT="${SUBMIT:-0}"
EXCLUDE="${EXCLUDE:-gpu212-14}"

DIMS=("1.0:100" "0.8:80" "0.6:60" "0.4:40" "0.2:20" "0.15:15" "0.1:10" "0.05:5" "0.02:2" "0.01:1")
SEED_GROUPS=("123 456 789 234 567" "891 345 678 912 135")
N=0

for gi in 0 1; do
  seeds="${SEED_GROUPS[$gi]}"

  # ---- CGCNN: seeds + dim specs via env ----
  N=$((N+1)); echo "[$N] CGCNN     seeds='${seeds}'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$C" && sbatch --time="04:00:00" --job-name="phoc_cg_${gi}" \
      --constraint="a100|rtx2080ti" --exclude="$EXCLUDE" \
      --export=ALL,CGCNN_MB_TASK=phonons,CGCNN_SEEDS="${seeds}",CGCNN_DIM_SPECS="${DIMS[*]}",CGCNN_LR_SCHEDULE=onecycle,CGCNN_LR=0.001,CGCNN_RESULT_ROOT="${C}/results_matbench_phonons_onecycle_fastfood" \
      scripts/submit_phonons_10seed.slurm)
  fi

  # ---- DimeNet++ / ALIGNN: explicit "frac:pct:model_seed:split_seed" run lists ----
  runs=""
  for d in "${DIMS[@]}"; do
    frac="${d%%:*}"; pct="${d##*:}"
    for s in $seeds; do runs+="${frac}:${pct}:${s}:$((s+1000)) "; done
  done
  runs="${runs% }"

  N=$((N+1)); echo "[$N] DimeNet++ seeds='${seeds}'  (${#DIMS[@]} dims)"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$D" && sbatch --time="04:00:00" --job-name="phoc_dn_${gi}" \
      --constraint=v100 --exclude="$EXCLUDE" \
      --export=ALL,DIMENET_RUNS="${runs}",DIMENET_LR_SCHEDULE=onecycle,DIMENET_LR=0.001,DIMENET_RESTORE_BEST=1,DIMENET_PATIENCE=0,DIMENET_RESULT_ROOT="${D}/results_dimenetpp_phonons_onecycle_fastfood" \
      scripts/submit_phonons_job.slurm)
  fi

  N=$((N+1)); echo "[$N] ALIGNN    seeds='${seeds}'  (${#DIMS[@]} dims)"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$A" && sbatch --time="06:00:00" --job-name="phoc_al_${gi}" \
      --constraint="a100|rtx2080ti" --cpus-per-task=8 --mem=64G --exclude="$EXCLUDE" \
      --export=ALL,PHON_RUNS="${runs}",ALIGNN_RESTORE_BEST=1,PHON_RESULT_ROOT="${A}/results_alignn_phonons_bestmodel_fastfood" \
      scripts/submit_phonons_job.slurm)
  fi
done

echo
echo "total jobs: $N   runs: 300 (3 models x 10 dims x 10 seeds)   (SUBMIT=${SUBMIT})"
