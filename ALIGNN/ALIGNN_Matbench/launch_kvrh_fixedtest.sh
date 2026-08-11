#!/usr/bin/env bash
# ALIGNN log_kvrh FIXED-TEST sweep: full / quarter / tenth, best-model scoring, 180 epochs.
#
# Protocol
#   * 12 dims (100/80/70/65/50/45/20/10/8/5/2/1), fastfood wrapper, h=64 (~85,953 params)
#   * 180 epochs, ALIGNN's own `scheduler: onecycle`
#   * ALIGNN_RESTORE_BEST=1 -> train.py reloads best_model.pt before testing.
#     WITHOUT this ALIGNN reports the FINAL-epoch model (`best_model = net` is an alias).
#   * FIXED TEST SET: within a seed, full/quarter_1/quarter_2/tenth_1/tenth_2 all score on the
#     SAME test set. Achieved with prebuilt datasets ordered [TRAIN,VAL,TEST] + keep_data_order=true
#     + explicit n_train/n_val/n_test, so ALIGNN never reshuffles.
#   * full = 4 seeds (123/456/789/234); quarter/tenth = 2 partitions x 2 seeds (123/456).
#     Each seed's dataset carries its own test set, so across seeds the test set differs.
#
# 144 runs. Measured: full ~10.2 s/epoch (0.51 h/run); quarter 530 s/run (smoke, 180 ep).
# a100-only at 16cpu/128G would not schedule under load; rtx2080ti (sm_75) runs ALIGNN fine and
# log_kvrh is tiny (10,987 structures, ~26 MB activations), so 8 cpu / 64 G is ample.
#   full     4 jobs x 12 runs (~6.2 h)   -> 10 h walltime
#   quarter  2 jobs x 24 runs (~3.9 h)   ->  8 h
#   tenth    2 jobs x 24 runs (~1.9 h)   ->  6 h
#   = 8 concurrent jobs, matching the GPUs left free by the eform 55/70 sweep.
#
#   bash launch_kvrh_fixedtest.sh            # dry run
#   SUBMIT=1 bash launch_kvrh_fixedtest.sh   # submit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-0}"
EXCLUDE="${EXCLUDE:-gpu212-14}"     # node with the recurring cudaSetDevice failure
N=0

go() {  # go <datasets(space-sep)> <seeds> <time> <name>
  local datasets="$1" seeds="$2" tm="$3" name="$4"
  for ds in $datasets; do
    N=$((N+1))
    echo "[$N] ${ds}  seeds='${seeds}'  t=${tm}"
    if [[ "$SUBMIT" == "1" ]]; then
      (cd "$HERE" && sbatch --time="$tm" --job-name="$name" --exclude="$EXCLUDE" \
        --constraint="a100|rtx2080ti" --cpus-per-task=8 --mem=64G \
        --export=ALL,KVRH_DATASET=${ds},KVRH_SEEDS="${seeds}",KVRH_EPOCHS=180,ALIGNN_RESTORE_BEST=1 \
        scripts/submit_kvrh_fixedtest_job.slurm)
    fi
  done
}

# FULL: one job per seed (each seed = its own dataset + its own test set)
for s in 123 456 789 234; do
  go "kvrh_full_s${s}" "$s" "10:00:00" "aft_full_${s}"
done

# QUARTER / TENTH: one job per (partition, seed) = 8 jobs... but that would exceed the free GPUs,
# so pair the two partitions per seed into one job instead (still one GPU each).
for s in 123 456; do
  go "kvrh_quarter_1_s${s} kvrh_quarter_2_s${s}" "$s" "08:00:00" "aft_q_${s}"
  go "kvrh_tenth_1_s${s} kvrh_tenth_2_s${s}"     "$s" "06:00:00" "aft_t_${s}"
done

echo
echo "total jobs: $N   runs: 144   (SUBMIT=${SUBMIT})"
