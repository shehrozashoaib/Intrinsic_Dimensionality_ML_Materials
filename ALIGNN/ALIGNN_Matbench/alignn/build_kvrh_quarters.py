#!/usr/bin/env python3
"""Carve two disjoint 25% QUARTER partitions of matbench_log_kvrh for ALIGNN.

Pure reshuffle -- NO graph featurization, NO jarvis conversion, NO GPU. The full
`MP_json_log_kvrh/id_prop.json` is the exact pre-shuffle `records` list that
`build_log_kvrh_idprop.py` also used to carve the halves, so we reproduce the SAME
seed-0 shuffle and cut it into 25% blocks. Result: the quarters nest cleanly inside
the existing halves (kvrh_quarter_1 subset of kvrh_half_1, etc.) and are mutually disjoint.

Outputs (ALIGNN does its own internal 80/10/10 split via --split_seed):
  alignn/partitions/kvrh_quarter_1/id_prop.json   disjoint 25% quarter (block 0)
  alignn/partitions/kvrh_quarter_2/id_prop.json   disjoint 25% quarter (block 1)
"""
import json, os, random

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(ALIGNN_DIR, "MP_json_log_kvrh", "id_prop.json")
PART_ROOT = os.path.join(ALIGNN_DIR, "partitions")
SHUFFLE_SEED = 0   # same as the halves, so quarters nest inside halves
ID_KEY = "material_id"


def main():
    records = json.load(open(FULL))
    N = len(records)
    idx = list(range(N))
    random.Random(SHUFFLE_SEED).shuffle(idx)
    q = N // 4  # 25% block size, mirrors halves' len(idx)//2
    quarters = {
        "kvrh_quarter_1": [records[i] for i in idx[0:q]],
        "kvrh_quarter_2": [records[i] for i in idx[q:2 * q]],
    }
    for name, recs in quarters.items():
        d = os.path.join(PART_ROOT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(recs, open(os.path.join(d, "id_prop.json"), "w"))
        mids = [r[ID_KEY] for r in recs]
        assert len(set(mids)) == len(mids), f"{name} has duplicate ids"
        print(f"  wrote {len(recs):5d} records -> {os.path.join(d, 'id_prop.json')}")

    # disjointness between the two quarters
    m1 = {r[ID_KEY] for r in quarters["kvrh_quarter_1"]}
    m2 = {r[ID_KEY] for r in quarters["kvrh_quarter_2"]}
    assert not (m1 & m2), "quarters overlap!"

    # nesting check vs halves (if present)
    nesting = {}
    for qi, half in (("kvrh_quarter_1", "kvrh_half_1"), ("kvrh_quarter_2", "kvrh_half_1")):
        hp = os.path.join(PART_ROOT, half, "id_prop.json")
        if os.path.exists(hp):
            hm = {r[ID_KEY] for r in json.load(open(hp))}
            qm = {r[ID_KEY] for r in json.load(open(os.path.join(PART_ROOT, qi, "id_prop.json")))}
            nesting[f"{qi}_subset_of_{half}"] = qm.issubset(hm)

    manifest = {
        "task": "matbench_log_kvrh",
        "source_full": os.path.relpath(FULL, ALIGNN_DIR),
        "n_total": N,
        "quarter_shuffle_seed": SHUFFLE_SEED,
        "n_quarter_1": len(quarters["kvrh_quarter_1"]),
        "n_quarter_2": len(quarters["kvrh_quarter_2"]),
        "quarters_disjoint": True,
        "nesting_vs_halves": nesting,
    }
    with open(os.path.join(PART_ROOT, "kvrh_quarters_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[quarters] {manifest['n_quarter_1']} + {manifest['n_quarter_2']} "
          f"(disjoint, seed={SHUFFLE_SEED}); nesting={nesting}")


if __name__ == "__main__":
    main()
