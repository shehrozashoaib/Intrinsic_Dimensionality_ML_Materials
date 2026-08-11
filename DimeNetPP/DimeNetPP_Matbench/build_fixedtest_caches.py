"""Build FIXED-TEST-SET caches for the log_kvrh dataset-size experiment.

Design (per user spec): the test set is held constant WITHIN a seed, across dataset sizes --
but VARIES across seeds, so we retain genuine split-to-split variance.

For each base split seed S in {1123, 1456, 1789}:
    * that split's TRAIN / VAL / TEST come from the existing full-dataset cache splitseed<S>.
    * full    : train = full train,            val = full val,            test = split S's TEST
    * quarter : train = 25% subsample of that train, val = 25% of that val, test = SAME (split S's TEST)
    * tenth   : train = 10% subsample of that train, val = 10% of that val, test = SAME (split S's TEST)
    * the SAME scaler (fit on that split's full train) is reused for every size, so target
      normalisation is identical and MAEs are directly comparable WITHIN a seed.

=> Within a seed, full/quarter/tenth are scored on the IDENTICAL test set (clean size effect).
=> Across seeds, the test set differs (real variance, not a single lucky split).

No graphs are rebuilt: we slice the already-cached full-dataset tensors.

Outputs: cached_tensors_dimenetpp_kvrh_fixedtest/<size>/splitseed<S>/dimenetpp_kvrh_cached_tensors.pkl
"""
import os, pickle, shutil
import numpy as np
import tensorflow as tf

FULL_ROOT = "cached_tensors_dimenetpp_kvrh"
OUT_ROOT = "cached_tensors_dimenetpp_kvrh_fixedtest"
SIZES = {"quarter": 0.25, "tenth": 0.10}     # 'full' reuses the base cache as-is
BASE_SPLITS = [1123, 1456, 1789, 1234, 1567, 1890, 1111, 1222, 1333, 1444]   # one per seed -> test varies ACROSS seeds


def take(x_list, idx):
    return [tf.gather(t, idx) for t in x_list]


def main():
    summary = []
    for S in BASE_SPLITS:
        base_dir = os.path.join(FULL_ROOT, f"splitseed{S}")
        src_pkl = os.path.join(base_dir, "dimenetpp_kvrh_cached_tensors.pkl")
        src_scaler = os.path.join(base_dir, "scaler_kvrh.pkl")
        with tf.device("/CPU:0"):
            cache = pickle.load(open(src_pkl, "rb"))
        n_tr = cache["y_train"].shape[0]
        n_va = cache["y_val"].shape[0]
        n_te = cache["y_test"].shape[0]
        print(f"\n[base split {S}] train={n_tr} val={n_va} test={n_te}  <- this seed's TEST is fixed across sizes")

        # subsample rng is seeded by the base split so it's reproducible per seed
        for size, frac in SIZES.items():
            rng = np.random.default_rng(S)
            k_tr = int(round(frac * n_tr))
            k_va = max(1, int(round(frac * n_va)))
            idx_tr = np.sort(rng.choice(n_tr, size=k_tr, replace=False))
            idx_va = np.sort(rng.choice(n_va, size=k_va, replace=False))
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
                    # TEST: identical to THIS split's full test set
                    "x_test": cache["x_test"],
                    "y_test": cache["y_test"],
                    "y_test_scaled": cache["y_test_scaled"],
                    "test_ids": cache["test_ids"],
                    "test_targets": cache["test_targets"],
                }
            d = os.path.join(OUT_ROOT, size, f"splitseed{S}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "dimenetpp_kvrh_cached_tensors.pkl"), "wb") as f:
                pickle.dump(out, f, protocol=4)
            shutil.copy(src_scaler, os.path.join(d, "scaler_kvrh.pkl"))
            print(f"   {size:8s} train={k_tr:5d} val={k_va:4d} test={n_te} (same as full) -> {d}")
            summary.append((S, size, d))

    # ---- verification ----
    print("\n[verify] WITHIN each seed: is test identical across full/quarter/tenth?")
    all_ok = True
    per_seed_test = {}
    for S in BASE_SPLITS:
        with tf.device("/CPU:0"):
            full = pickle.load(open(os.path.join(FULL_ROOT, f"splitseed{S}",
                                                 "dimenetpp_kvrh_cached_tensors.pkl"), "rb"))
        ref_y = np.asarray(full["y_test"]).ravel(); ref_ids = list(full["test_ids"])
        per_seed_test[S] = ref_ids
        for size in SIZES:
            with tf.device("/CPU:0"):
                c = pickle.load(open(os.path.join(OUT_ROOT, size, f"splitseed{S}",
                                                  "dimenetpp_kvrh_cached_tensors.pkl"), "rb"))
            same = np.array_equal(np.asarray(c["y_test"]).ravel(), ref_y) and list(c["test_ids"]) == ref_ids
            all_ok &= same
            print(f"   seed {S}: {size:8s} test == full test  -> {same}")
    print(f"\n[verify] within-seed test identity: {all_ok}")

    print("\n[verify] ACROSS seeds: test sets should DIFFER (we want real split variance)")
    import itertools
    pairs = list(itertools.combinations(BASE_SPLITS, 2))
    n_diff = sum(1 for x, y in pairs if per_seed_test[x] != per_seed_test[y])
    print(f"   {n_diff}/{len(pairs)} seed pairs have DIFFERENT test sets "
          f"({'all differ - good' if n_diff == len(pairs) else 'WARNING: some identical'})")


if __name__ == "__main__":
    main()
