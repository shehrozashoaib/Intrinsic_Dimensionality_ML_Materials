#!/usr/bin/env python3
"""Carve two disjoint 10% TENTH partitions of matbench_log_kvrh for ALIGNN.

Pure reshuffle -- NO featurization, NO jarvis conversion, NO GPU. Same seed-0 shuffle as the
halves (build_log_kvrh_idprop.py) and quarters (build_kvrh_quarters.py), cut into 10% blocks.

Convention (matches partition_alignn_datasets.py): partitions are DISJOINT WITHIN each size
group; across size groups they nest (tenth_1 subset of quarter_1 subset of half_1).

Outputs:
  alignn/partitions/kvrh_tenth_1/id_prop.json   disjoint 10% (block 0)
  alignn/partitions/kvrh_tenth_2/id_prop.json   disjoint 10% (block 1)
"""
import json, os, random

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(ALIGNN_DIR, "MP_json_log_kvrh", "id_prop.json")
PART_ROOT = os.path.join(ALIGNN_DIR, "partitions")
SHUFFLE_SEED = 0   # same shuffle as halves/quarters
ID_KEY = "material_id"


def main():
    records = json.load(open(FULL))
    N = len(records)
    idx = list(range(N))
    random.Random(SHUFFLE_SEED).shuffle(idx)
    t = N // 10
    tenths = {
        "kvrh_tenth_1": [records[i] for i in idx[0:t]],
        "kvrh_tenth_2": [records[i] for i in idx[t:2 * t]],
    }
    for name, recs in tenths.items():
        d = os.path.join(PART_ROOT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(recs, open(os.path.join(d, "id_prop.json"), "w"))
        mids = [r[ID_KEY] for r in recs]
        assert len(set(mids)) == len(mids), f"{name} has duplicate ids"
        print(f"  wrote {len(recs):5d} records -> {os.path.join(d, 'id_prop.json')}")

    m1 = {r[ID_KEY] for r in tenths["kvrh_tenth_1"]}
    m2 = {r[ID_KEY] for r in tenths["kvrh_tenth_2"]}
    assert not (m1 & m2), "tenths overlap!"

    nesting = {}
    for parent in ("kvrh_quarter_1", "kvrh_half_1"):
        pp = os.path.join(PART_ROOT, parent, "id_prop.json")
        if os.path.exists(pp):
            pm = {r[ID_KEY] for r in json.load(open(pp))}
            nesting[f"tenth_1_subset_of_{parent}"] = m1.issubset(pm)
            nesting[f"tenth_2_subset_of_{parent}"] = m2.issubset(pm)

    manifest = {
        "task": "matbench_log_kvrh",
        "source_full": os.path.relpath(FULL, ALIGNN_DIR),
        "n_total": N,
        "tenth_shuffle_seed": SHUFFLE_SEED,
        "n_tenth_1": len(tenths["kvrh_tenth_1"]),
        "n_tenth_2": len(tenths["kvrh_tenth_2"]),
        "tenths_disjoint": True,
        "nesting": nesting,
    }
    with open(os.path.join(PART_ROOT, "kvrh_tenths_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[tenths] {manifest['n_tenth_1']} + {manifest['n_tenth_2']} "
          f"(disjoint, seed={SHUFFLE_SEED}); nesting={nesting}")


if __name__ == "__main__":
    main()
