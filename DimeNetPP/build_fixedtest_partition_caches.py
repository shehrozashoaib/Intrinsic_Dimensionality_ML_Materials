"""Build FIXED-TEST, TWO-PARTITION caches for the DimeNet++ one-cycle dataset-size sweep.

Generalises DimeNetPP_Matbench/build_fixedtest_caches.py in two ways:
  1. task-parameterised (log_kvrh / eform / bandgap), not kvrh-only
  2. TWO DISJOINT partitions per size per seed, instead of one subsample

Design:
  For each base split seed S:
    * that split's TRAIN / VAL / TEST come from the existing full-dataset cache splitseed<S>.
    * quarter_1 / quarter_2 : two DISJOINT 25% chunks of that split's train (and of its val)
    * tenth_1   / tenth_2   : two DISJOINT 10% chunks of that split's train (and of its val)
    * every one of them is scored on splitseed<S>'s FULL TEST SET, unchanged
    * the SAME scaler (fit on that split's full train) is reused everywhere

  => within a seed, full / quarter_{1,2} / tenth_{1,2} all share ONE test set, so the
     dataset-size effect AND the partition effect are both clean.
  => across seeds the test set differs, preserving genuine split variance.

Disjointness is exact: a single seeded permutation of the train indices is cut into
consecutive blocks, so partition 1 and partition 2 share no structure. Same for val.

No graphs are rebuilt -- we slice the already-cached full-dataset tensors.

Usage:  python build_fixedtest_partition_caches.py <task>        # log_kvrh | eform | bandgap
Outputs: <out_root>/<size>_<part>/splitseed<S>/dimenetpp_<key>_cached_tensors.pkl
"""
import gc, os, pickle, shutil, sys

import numpy as np
import tensorflow as tf

# task -> (cache key used in filenames, full-cache root, output root, project dir)
TASKS = {
    "log_kvrh": ("kvrh", "cached_tensors_dimenetpp_kvrh",
                 "cached_tensors_dimenetpp_kvrh_fixedtest2", "DimeNetPP_Matbench"),
    "eform":    ("eform", "cached_tensors_dimenetpp_eform",
                 "cached_tensors_dimenetpp_eform_fixedtest2", "DimeNetPP_MP"),
    "bandgap":  ("bandgap", "cached_tensors_dimenetpp_bandgap",
                 "cached_tensors_dimenetpp_bandgap_fixedtest2", "DimeNetPP_MP"),
}

# 4 seeds for full; quarter/tenth take 2 seeds x 2 partitions. Building all 4 seeds for the
# partitions too costs little and leaves the option open.
BASE_SPLITS = [1123, 1456, 1789, 1234]
SIZES = {"quarter": 0.25, "tenth": 0.10}
N_PARTS = 2


def take(x_list, idx):
    return [tf.gather(t, idx) for t in x_list]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(TASKS)}>")
    task = sys.argv[1]
    key, full_root, out_root, proj = TASKS[task]
    here = os.path.dirname(os.path.abspath(__file__))
    full_root = os.path.join(here, proj, full_root)
    out_root = os.path.join(here, proj, out_root)
    pkl_name = f"dimenetpp_{key}_cached_tensors.pkl"
    scaler_name = f"scaler_{key}.pkl"

    print(f"[build] task={task}  full={full_root}\n[build] out={out_root}")
    written = []

    for S in BASE_SPLITS:
        base_dir = os.path.join(full_root, f"splitseed{S}")
        src_pkl = os.path.join(base_dir, pkl_name)
        src_scaler = os.path.join(base_dir, scaler_name)
        if not os.path.exists(src_pkl):
            sys.exit(f"MISSING full cache: {src_pkl}")
        with tf.device("/CPU:0"):
            cache = pickle.load(open(src_pkl, "rb"))
        n_tr = cache["y_train"].shape[0]
        n_va = cache["y_val"].shape[0]
        n_te = cache["y_test"].shape[0]
        print(f"\n[base split {S}] train={n_tr} val={n_va} test={n_te}"
              f"   <- this seed's TEST is reused by every partition below")

        # ONE permutation per seed, cut into consecutive disjoint blocks => partitions
        # never overlap, and quarter_1 is a superset-free sibling of quarter_2.
        rng = np.random.default_rng(S)
        perm_tr = rng.permutation(n_tr)
        perm_va = rng.permutation(n_va)

        for size, frac in SIZES.items():
            k_tr = int(round(frac * n_tr))
            k_va = max(1, int(round(frac * n_va)))
            if k_tr * N_PARTS > n_tr or k_va * N_PARTS > n_va:
                sys.exit(f"cannot cut {N_PARTS} disjoint {size} partitions from split {S}")
            for p in range(N_PARTS):
                idx_tr = np.sort(perm_tr[p * k_tr:(p + 1) * k_tr])
                idx_va = np.sort(perm_va[p * k_va:(p + 1) * k_va])
                with tf.device("/CPU:0"):
                    out = {
                        "x_train": take(cache["x_train"], idx_tr),
                        "y_train": cache["y_train"][idx_tr],
                        "y_train_scaled": cache["y_train_scaled"][idx_tr],
                        "train_ids": [cache["train_ids"][i] for i in idx_tr],
                        "train_targets": np.asarray(cache["train_targets"])[idx_tr],
                        "x_val": take(cache["x_val"], idx_va),
                        "y_val": cache["y_val"][idx_va],
                        "y_val_scaled": cache["y_val_scaled"][idx_va],
                        "val_ids": [cache["val_ids"][i] for i in idx_va],
                        "val_targets": np.asarray(cache["val_targets"])[idx_va],
                        # TEST: identical to THIS split's full test set, for every partition
                        "x_test": cache["x_test"],
                        "y_test": cache["y_test"],
                        "y_test_scaled": cache["y_test_scaled"],
                        "test_ids": cache["test_ids"],
                        "test_targets": cache["test_targets"],
                    }
                d = os.path.join(out_root, f"{size}_{p + 1}", f"splitseed{S}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, pkl_name), "wb") as f:
                    pickle.dump(out, f, protocol=4)
                if os.path.exists(src_scaler):
                    shutil.copy(src_scaler, os.path.join(d, scaler_name))
                else:
                    print(f"   WARNING: no {scaler_name} in {base_dir}")
                print(f"   {size}_{p+1} train={k_tr:6d} val={k_va:5d} "
                      f"test={n_te} (= full test) -> {d}")
                written.append((S, f"{size}_{p+1}", d))
                del out
                gc.collect()

        del cache
        gc.collect()

    # ---- verification ----
    print("\n[verify] WITHIN each seed: every partition's test == the full split's test?")
    all_ok = True
    per_seed_test = {}
    for S in BASE_SPLITS:
        with tf.device("/CPU:0"):
            full = pickle.load(open(os.path.join(full_root, f"splitseed{S}", pkl_name), "rb"))
        ref_y = np.asarray(full["y_test"]).ravel()
        ref_ids = list(full["test_ids"])
        per_seed_test[S] = ref_ids
        del full
        gc.collect()
        for size in SIZES:
            for p in range(N_PARTS):
                with tf.device("/CPU:0"):
                    c = pickle.load(open(os.path.join(out_root, f"{size}_{p+1}",
                                                      f"splitseed{S}", pkl_name), "rb"))
                same = (np.array_equal(np.asarray(c["y_test"]).ravel(), ref_y)
                        and list(c["test_ids"]) == ref_ids)
                all_ok &= same
                print(f"   seed {S}: {size}_{p+1} test == full test -> {same}")
                del c
                gc.collect()
    print(f"\n[verify] within-seed test identity: {all_ok}")

    print("\n[verify] partitions are DISJOINT in train (and val) within each seed/size")
    dis_ok = True
    for S in BASE_SPLITS:
        for size in SIZES:
            ids = []
            for p in range(N_PARTS):
                with tf.device("/CPU:0"):
                    c = pickle.load(open(os.path.join(out_root, f"{size}_{p+1}",
                                                      f"splitseed{S}", pkl_name), "rb"))
                ids.append((set(c["train_ids"]), set(c["val_ids"])))
                del c
                gc.collect()
            tr_ov = len(ids[0][0] & ids[1][0])
            va_ov = len(ids[0][1] & ids[1][1])
            dis_ok &= (tr_ov == 0 and va_ov == 0)
            print(f"   seed {S}: {size:8s} train overlap={tr_ov}  val overlap={va_ov}")
    print(f"\n[verify] partition disjointness: {dis_ok}")

    print("\n[verify] ACROSS seeds: test sets should DIFFER")
    import itertools
    pairs = list(itertools.combinations(BASE_SPLITS, 2))
    n_diff = sum(1 for x, y in pairs if per_seed_test[x] != per_seed_test[y])
    print(f"   {n_diff}/{len(pairs)} seed pairs differ "
          f"({'all differ - good' if n_diff == len(pairs) else 'WARNING: some identical'})")

    print(f"\n[build] wrote {len(written)} caches -> {out_root}")
    print(f"[build] OVERALL: {'OK' if (all_ok and dis_ok and n_diff == len(pairs)) else 'PROBLEM - see above'}")


if __name__ == "__main__":
    main()
