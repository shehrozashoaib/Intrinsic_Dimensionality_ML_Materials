#!/usr/bin/env bash
# Launch the DimeNet++ ONE-CYCLE sweep across every task x dataset size that does not have it yet.
#
# Already done elsewhere, NOT relaunched here:
#   bandgap full  (8 dims x 4 seeds)  -> results_dimenetpp_optimizer_test/bandgap
#   eform   full  dims 100 and 50     -> results_dimenetpp_optimizer_test/eform
#
# Protocol (identical everywhere):
#   --lr_schedule onecycle, lr 1e-3, pct_start 0.3, --patience 0 (ES OFF so the anneal completes),
#   --restore_best, clipnorm 0, num_blocks 1, 350 epochs (MP) / 180 (log_kvrh).
#   full     = 4 seeds 123/456/789/234
#   quarter  = 2 partitions x 2 seeds (123/456);  tenth = same
#   Partitions use the FIXED-TEST caches: within a seed, every partition is scored on that
#   seed's full test set (DIMENET_FIXEDTEST=1).
#
# Job sizing comes from measured per-run times (median sec/epoch x epochs):
#   eform/bandgap full 5.3 h | quarter 1.3 h | tenth 0.55 h ; log_kvrh 0.13 / 0.05 / 0.03 h
# Every job is kept under the 20 h walltime rule.
#
#   bash launch_onecycle_sweep.sh            # dry run, prints what it would submit
#   SUBMIT=1 bash launch_onecycle_sweep.sh   # actually submit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP="${ROOT}/DimeNetPP_MP"
MB="${ROOT}/DimeNetPP_Matbench"
SUBMIT="${SUBMIT:-0}"

COMMON="DIMENET_LR_SCHEDULE=onecycle,DIMENET_LR=0.001,DIMENET_PATIENCE=0,DIMENET_CLIPNORM=0"
N=0

# submit_mp <task> <dataset> <time> <mem> <jobname> <dims> <seeds>
submit_mp() {
  local task="$1" ds="$2" tm="$3" mem="$4" name="$5" dims="$6" seeds="$7"
  local rroot="${MP}/results_dimenetpp_onecycle_fixedtest/${task}/${ds}"
  local ft=0; [[ "$ds" != "full" ]] && ft=1
  N=$((N+1))
  # Partition jobs must not start before their fixed-test caches exist: gate them on the
  # build array task (EFORM_DEP / BANDGAP_DEP = "<jobid>_<arraytask>").
  local dep=""
  if [[ "$ds" != "full" ]]; then
    local depjob="EFORM_DEP"; [[ "$task" == "bandgap" ]] && depjob="BANDGAP_DEP"
    local depval="${!depjob:-}"
    [[ -n "$depval" ]] && dep="--dependency=afterok:${depval}"
  fi
  echo "[$N] MP  ${task}/${ds}  t=${tm} mem=${mem}  dims='${dims}'  seeds='${seeds}'${dep:+  ${dep}}"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$MP" && sbatch $dep --time="$tm" --mem="$mem" --job-name="$name" \
      --export=ALL,${COMMON},DIMENET_TASK=${task},DIMENET_DATASET=${ds},DIMENET_FIXEDTEST=${ft},DIMENET_DIMS="${dims}",DIMENET_SEEDS="${seeds}",DIMENET_RESULT_ROOT="${rroot}" \
      scripts/submit_bestmodel_job.slurm)
  fi
}

# submit_kvrh <dataset> <time> <jobname> <dims> <seeds>
submit_kvrh() {
  local ds="$1" tm="$2" name="$3" dims="$4" seeds="$5"
  local rroot="${MB}/results_dimenetpp_onecycle_fixedtest/log_kvrh/${ds}"
  local ft=0; [[ "$ds" != "full" ]] && ft=1
  N=$((N+1))
  echo "[$N] KVRH ${ds}  t=${tm}  dims='${dims}'  seeds='${seeds}'"
  if [[ "$SUBMIT" == "1" ]]; then
    (cd "$MB" && sbatch --time="$tm" --job-name="$name" \
      --export=ALL,${COMMON},DIMENET_DATASET=${ds},DIMENET_FIXEDTEST=${ft},DIMENET_DIMS="${dims}",DIMENET_SEEDS="${seeds}",DIMENET_RESULT_ROOT="${rroot}" \
      scripts/submit_kvrh_bestmodel_job.slurm)
  fi
}

S3="123:1123 456:1456 789:1789"
S4="234:1234"
S2="123:1123 456:1456"
S_ALL4="123:1123 456:1456 789:1789 234:1234"

# ---------------------------------------------------------------- eform FULL (dims 100 & 50 exist)
# 5.3 h/run: 3 runs/job = 15.9 h. 7 dims x 4 seeds = 28 runs -> 10 jobs.
for d in "0.8:80" "0.6:60" "0.45:45" "0.3:30" "0.2:20" "0.15:15" "0.1:10"; do
  submit_mp eform full "20:00:00" 128G "oc_ef_f_${d##*:}" "$d" "$S3"
done
submit_mp eform full "20:00:00" 128G "oc_ef_f_s234a" "0.8:80 0.6:60 0.45:45" "$S4"
submit_mp eform full "15:00:00" 128G "oc_ef_f_s234b" "0.3:30 0.2:20"          "$S4"
submit_mp eform full "15:00:00" 128G "oc_ef_f_s234c" "0.15:15 0.1:10"         "$S4"

# ---------------------------------------------------------------- eform QUARTER / TENTH
EF_A="1.0:100 0.8:80 0.6:60 0.5:50 0.45:45"     # 5 dims x 2 seeds = 10 runs
EF_B="0.3:30 0.2:20 0.15:15 0.1:10"             # 4 dims x 2 seeds =  8 runs
for p in quarter_1 quarter_2; do
  submit_mp eform "$p" "18:00:00" 64G "oc_ef_${p}a" "$EF_A" "$S2"   # ~12.9 h
  submit_mp eform "$p" "15:00:00" 64G "oc_ef_${p}b" "$EF_B" "$S2"   # ~10.3 h
done
for p in tenth_1 tenth_2; do
  submit_mp eform "$p" "15:00:00" 64G "oc_ef_${p}" "$EF_A $EF_B" "$S2"   # 18 runs ~9.9 h
done

# ---------------------------------------------------------------- bandgap QUARTER / TENTH
BG_A="1.0:100 0.8:80 0.6:60 0.5:50"             # 4 dims x 2 seeds = 8 runs
BG_B="0.45:45 0.2:20 0.1:10 0.05:5"
for p in quarter_1 quarter_2; do
  submit_mp bandgap "$p" "16:00:00" 64G "oc_bg_${p}a" "$BG_A" "$S2"   # ~10.8 h
  submit_mp bandgap "$p" "16:00:00" 64G "oc_bg_${p}b" "$BG_B" "$S2"
done
for p in tenth_1 tenth_2; do
  submit_mp bandgap "$p" "14:00:00" 64G "oc_bg_${p}" "$BG_A $BG_B" "$S2"   # 16 runs ~8.8 h
done

# ---------------------------------------------------------------- log_kvrh (all sizes, cheap)
KV="1.0:100 0.8:80 0.7:70 0.65:65 0.5:50 0.45:45 0.2:20 0.1:10 0.08:8 0.05:5 0.02:2 0.01:1"
submit_kvrh full "10:00:00" "oc_kv_full" "$KV" "$S_ALL4"          # 48 runs ~6.2 h
for p in quarter_1 quarter_2 tenth_1 tenth_2; do
  submit_kvrh "$p" "06:00:00" "oc_kv_${p}" "$KV" "$S2"            # 24 runs, <2 h
done

echo
echo "total jobs: $N   (SUBMIT=${SUBMIT})"
