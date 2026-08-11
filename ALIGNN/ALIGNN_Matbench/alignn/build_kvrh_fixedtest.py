#!/usr/bin/env python3
"""Build FIXED-TEST log_kvrh datasets for ALIGNN: full / quarter / tenth, per split seed.

WHY THIS EXISTS
---------------
ALIGNN normally does its own internal 80/10/10 split (`get_id_train_val_test`, shuffled by
`split_seed`). Run that on a quarter-sized id_prop.json and you get a quarter-sized TEST set that
is *different* from the full dataset's test set -- so a full-vs-quarter comparison mixes the
dataset-size effect with test-set difficulty.

THE TRICK
---------
`data.get_id_train_val_test` supports `keep_data_order=True`, in which case it does NOT shuffle:
    id_train = ids[:n_train]
    id_val   = ids[-(n_val+n_test):-n_test]
    id_test  = ids[-n_test:]
So if we write records already ordered as [TRAIN..., VAL..., TEST...] and pass explicit
n_train/n_val/n_test, ALIGNN uses exactly the split we intend.

For each split seed S we reproduce the split ALIGNN itself would have made on the FULL dataset
(same `random.seed(S); random.shuffle(ids)` as data.py), then:
    full    : train = full train,          val = full val,          test = S's TEST
    quarter : train = 25% of that train,   val = 25% of that val,   test = SAME (S's TEST)
    tenth   : train = 10% of that train,   val = 10% of that val,   test = SAME (S's TEST)

=> within a seed, full/quarter/tenth are scored on the IDENTICAL test set.
=> across seeds the test set differs, so real split variance is preserved.

Two disjoint partitions per reduced size (quarter_1/quarter_2, tenth_1/tenth_2) are cut from one
permutation of the train pool, so they never share a training structure.

Outputs:
  alignn/partitions_fixedtest/kvrh_<size>_s<seed>/id_prop.json
  alignn/partitions_fixedtest/kvrh_<size>_s<seed>/split.json   <- n_train/n_val/n_test to pass on
"""
import json, os, random

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(ALIGNN_DIR, "MP_json_log_kvrh", "id_prop.json")
OUT_ROOT = os.path.join(ALIGNN_DIR, "partitions_fixedtest")
SPLIT_SEEDS = [123, 456, 789, 234]
VAL_RATIO, TEST_RATIO = 0.1, 0.1
SIZES = {"quarter": 0.25, "tenth": 0.10}
N_PARTS = 2
ID_KEY = "material_id"


def alignn_split(total, split_seed):
    """Reproduce data.get_id_train_val_test EXACTLY (keep_data_order=False path)."""
    train_ratio = 1 - VAL_RATIO - TEST_RATIO
    n_train = int(train_ratio * total)
    n_test = int(TEST_RATIO * total)
    n_val = int(VAL_RATIO * total)
    ids = list(range(total))
    random.seed(split_seed)          # data.py uses stdlib random.seed + random.shuffle
    random.shuffle(ids)
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test):-n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def write(name, recs, n_train, n_val, n_test):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    json.dump(recs, open(os.path.join(d, "id_prop.json"), "w"))
    json.dump({"n_train": n_train, "n_val": n_val, "n_test": n_test,
               "total": len(recs), "keep_data_order": True},
              open(os.path.join(d, "split.json"), "w"), indent=2)
    assert n_train + n_val + n_test == len(recs), (name, n_train, n_val, n_test, len(recs))
    mids = [r[ID_KEY] for r in recs]
    assert len(set(mids)) == len(mids), f"{name} has duplicate ids"
    print(f"   {name:22s} total={len(recs):6d}  train={n_train:6d} val={n_val:5d} test={n_test:5d}")
    return d


def main():
    records = json.load(open(FULL))
    N = len(records)
    print(f"[build] {N} records from {FULL}\n[build] out -> {OUT_ROOT}")
    test_ids_by_seed = {}

    for S in SPLIT_SEEDS:
        id_train, id_val, id_test = alignn_split(N, S)
        print(f"\n[split seed {S}] full train={len(id_train)} val={len(id_val)} test={len(id_test)}")
        test_recs = [records[i] for i in id_test]
        test_ids_by_seed[S] = [r[ID_KEY] for r in test_recs]

        # ---- full: ordered [train, val, test] ----
        recs = [records[i] for i in id_train] + [records[i] for i in id_val] + test_recs
        write(f"kvrh_full_s{S}", recs, len(id_train), len(id_val), len(id_test))

        # ---- reduced sizes: disjoint partitions of the SAME train/val pools ----
        rng = random.Random(S)
        perm_tr = id_train[:]; rng.shuffle(perm_tr)
        perm_va = id_val[:];   rng.shuffle(perm_va)
        for size, frac in SIZES.items():
            k_tr = int(round(frac * len(id_train)))
            k_va = max(1, int(round(frac * len(id_val))))
            assert k_tr * N_PARTS <= len(id_train), (size, S)
            for p in range(N_PARTS):
                tr = perm_tr[p * k_tr:(p + 1) * k_tr]
                va = perm_va[p * k_va:(p + 1) * k_va]
                recs = ([records[i] for i in tr] + [records[i] for i in va] + test_recs)
                write(f"kvrh_{size}_{p+1}_s{S}", recs, len(tr), len(va), len(id_test))

    # ---- verification ----
    print("\n[verify] WITHIN each seed: identical test set across every size?")
    ok = True
    for S in SPLIT_SEEDS:
        ref = test_ids_by_seed[S]
        names = [f"kvrh_full_s{S}"] + [f"kvrh_{sz}_{p+1}_s{S}"
                                       for sz in SIZES for p in range(N_PARTS)]
        for name in names:
            d = os.path.join(OUT_ROOT, name)
            recs = json.load(open(os.path.join(d, "id_prop.json")))
            sp = json.load(open(os.path.join(d, "split.json")))
            got = [r[ID_KEY] for r in recs[-sp["n_test"]:]]
            same = got == ref
            ok &= same
            if not same:
                print(f"   *** {name}: test MISMATCH")
        print(f"   seed {S}: all {len(names)} datasets share one test set -> "
              f"{all(json.load(open(os.path.join(OUT_ROOT, n, 'id_prop.json')))[-len(ref):] and True for n in names)}")
    print(f"[verify] within-seed test identity: {ok}")

    print("\n[verify] partitions disjoint in train?")
    dis = True
    for S in SPLIT_SEEDS:
        for size in SIZES:
            sets = []
            for p in range(N_PARTS):
                d = os.path.join(OUT_ROOT, f"kvrh_{size}_{p+1}_s{S}")
                recs = json.load(open(os.path.join(d, "id_prop.json")))
                sp = json.load(open(os.path.join(d, "split.json")))
                sets.append({r[ID_KEY] for r in recs[:sp["n_train"]]})
            ov = len(sets[0] & sets[1])
            dis &= ov == 0
            print(f"   seed {S} {size:8s} train overlap = {ov}")
    print(f"[verify] disjointness: {dis}")

    print("\n[verify] ACROSS seeds: test sets differ?")
    import itertools
    pairs = list(itertools.combinations(SPLIT_SEEDS, 2))
    ndiff = sum(1 for a, b in pairs if test_ids_by_seed[a] != test_ids_by_seed[b])
    print(f"   {ndiff}/{len(pairs)} seed pairs differ")
    print(f"\n[build] OVERALL: {'OK' if (ok and dis and ndiff == len(pairs)) else 'PROBLEM'}")


if __name__ == "__main__":
    main()
