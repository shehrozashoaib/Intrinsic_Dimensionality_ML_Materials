#!/usr/bin/env python3
"""Carve two disjoint 25% QUARTER partitions of the MP eform / bandgap datasets.

Pure reshuffle -- NO featurization, NO GPU. Uses the SAME seed-0 shuffle that
partition_alignn_datasets.py used to carve the existing 10% (p10_*) and 1% (p01_*) blocks,
so the quarters are consistent with (and nest around) those partitions.

Convention (matches partition_alignn_datasets.py): disjoint WITHIN each size group; across
size groups they nest (p10_1 subset of quarter_1).

Usage: python build_mp_quarters.py eform|bandgap
Outputs: alignn/partitions/<task>_quarter_1/id_prop.json , <task>_quarter_2/id_prop.json
"""
import json, os, random, sys

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
PART_ROOT = os.path.join(ALIGNN_DIR, "partitions")
SHUFFLE_SEED = 0          # same as partition_alignn_datasets.py default
ID_KEY = "material_id"

SOURCES = {
    "eform":   os.path.join(ALIGNN_DIR, "MP_json_eform", "id_prop.json"),
    "bandgap": os.path.join(ALIGNN_DIR, "MP_json_bandgap", "id_prop.json"),
}


def main():
    task = sys.argv[1]
    src = SOURCES[task]
    records = json.load(open(src))
    N = len(records)
    idx = list(range(N))
    random.Random(SHUFFLE_SEED).shuffle(idx)
    h = N // 2     # 50%
    q = N // 4     # 25%
    t = N // 10    # 10%
    parts = {
        # halves: same seed-0 shuffle, disjoint, and the quarter/tenth blocks nest inside half_1
        f"{task}_half_1": [records[i] for i in idx[0:h]],
        f"{task}_half_2": [records[i] for i in idx[h:2 * h]],
        f"{task}_quarter_1": [records[i] for i in idx[0:q]],
        f"{task}_quarter_2": [records[i] for i in idx[q:2 * q]],
        # tenths carved from the SAME shuffle -> nest inside quarter_1, mutually disjoint.
        # (We do NOT reuse the old <task>_p10_* blocks: bandgap's were carved from a different
        #  record ordering and do NOT nest, which would make the ladder inconsistent across tasks.)
        f"{task}_tenth_1": [records[i] for i in idx[0:t]],
        f"{task}_tenth_2": [records[i] for i in idx[t:2 * t]],
    }
    for name, recs in parts.items():
        d = os.path.join(PART_ROOT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(recs, open(os.path.join(d, "id_prop.json"), "w"))
        mids = [r[ID_KEY] for r in recs]
        assert len(set(mids)) == len(mids), f"{name} duplicate ids"
        print(f"  wrote {len(recs):6d} records -> {name}/id_prop.json")

    m1 = {r[ID_KEY] for r in parts[f"{task}_quarter_1"]}
    m2 = {r[ID_KEY] for r in parts[f"{task}_quarter_2"]}
    assert not (m1 & m2), "quarters overlap!"

    # nesting check vs the existing 10% blocks (p10_1 should sit inside quarter_1)
    nesting = {}
    p10 = os.path.join(PART_ROOT, f"{task}_p10_1", "id_prop.json")
    if os.path.exists(p10):
        pm = {r[ID_KEY] for r in json.load(open(p10))}
        nesting[f"{task}_p10_1_subset_of_quarter_1"] = pm.issubset(m1)

    manifest = {"task": task, "source_full": os.path.relpath(src, ALIGNN_DIR), "n_total": N,
                "quarter_shuffle_seed": SHUFFLE_SEED,
                "n_quarter_1": len(m1), "n_quarter_2": len(m2),
                "quarters_disjoint": True, "nesting": nesting}
    with open(os.path.join(PART_ROOT, f"{task}_quarters_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{task} quarters] {len(m1)} + {len(m2)} (disjoint, seed={SHUFFLE_SEED}); nesting={nesting}")


if __name__ == "__main__":
    main()
