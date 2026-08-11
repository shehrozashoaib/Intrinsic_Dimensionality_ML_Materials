"""Carve disjoint subset partitions of an ALIGNN id_prop.json dataset.

Creates ALIGNN root_dirs (each holding an id_prop.json subset):
  - n_10pct partitions of 10% each, mutually DISJOINT (different materials)
  - n_1pct  partitions of 1%  each, mutually DISJOINT among themselves
    (the 1% partitions are carved from the front of the same shuffle, so they may
     overlap the first 10% partition -- 'disjoint within each size only')

Each partition keeps the full record dict, so ALIGNN's internal 80:10:10 split
(via --split_seed) applies unchanged within every partition.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="full id_prop.json")
    p.add_argument("--out_root", required=True, help="dir to write <prefix>_pXX_k/id_prop.json")
    p.add_argument("--prefix", required=True, help="eform | bandgap")
    p.add_argument("--id_key", default="material_id")
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--n_10pct", type=int, default=3)
    p.add_argument("--n_1pct", type=int, default=10)
    args = p.parse_args()

    data = json.load(open(args.source))
    N = len(data)
    perm = list(range(N))
    random.Random(args.shuffle_seed).shuffle(perm)

    n10 = round(0.10 * N)
    n01 = round(0.01 * N)
    assert args.n_10pct * n10 <= N, "not enough data for disjoint 10% partitions"
    assert args.n_1pct * n01 <= N, "not enough data for disjoint 1% partitions"

    out_root = Path(args.out_root)
    manifest = {}

    def write_partition(name, idxs):
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        sub = [data[i] for i in idxs]
        with (d / "id_prop.json").open("w") as f:
            json.dump(sub, f)
        mids = [data[i][args.id_key] for i in idxs]
        manifest[name] = mids
        # quick within-partition uniqueness check
        assert len(set(mids)) == len(mids), f"{name} has duplicate material ids"
        print(f"{name}: n={len(sub)}  (idx {idxs[0]}..{idxs[-1]})", flush=True)

    # 10% partitions: consecutive disjoint blocks
    for k in range(args.n_10pct):
        write_partition(f"{args.prefix}_p10_{k+1}", perm[k * n10:(k + 1) * n10])
    # 1% partitions: consecutive disjoint blocks from the front
    for k in range(args.n_1pct):
        write_partition(f"{args.prefix}_p01_{k+1}", perm[k * n01:(k + 1) * n01])

    # disjointness verification within each size group
    def check_group(names):
        seen = {}
        for nm in names:
            for m in manifest[nm]:
                if m in seen:
                    raise SystemExit(f"OVERLAP: material {m} in both {seen[m]} and {nm}")
                seen[m] = nm
        return len(seen)

    n10_total = check_group([f"{args.prefix}_p10_{k+1}" for k in range(args.n_10pct)])
    n01_total = check_group([f"{args.prefix}_p01_{k+1}" for k in range(args.n_1pct)])
    print(f"[OK] {args.prefix}: total N={N}, n10={n10}/partition x{args.n_10pct} (disjoint, {n10_total} unique), "
          f"n01={n01}/partition x{args.n_1pct} (disjoint, {n01_total} unique)", flush=True)

    with (out_root / f"{args.prefix}_partition_manifest.json").open("w") as f:
        json.dump({k: v for k, v in manifest.items()}, f)
    print(f"[manifest] wrote {out_root}/{args.prefix}_partition_manifest.json", flush=True)


if __name__ == "__main__":
    main()
