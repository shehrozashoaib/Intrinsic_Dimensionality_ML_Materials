"""Benchmark: does multiprocessing speed up the kgcnn graph build (set_range_periodic + set_angle)?

Builds the SAME N structures (a) serially in one process and (b) with worker pools of W
processes (spawn, because TF + fork deadlocks), and reports wall time + speedup.
Uses mp21.jsonl pymatgen structure dicts (pickle-friendly across spawn).
"""
import argparse
import json
import random
import time
import multiprocessing as mp

import numpy as np
import pandas as pd


MODEL_INPUTS_SPEC = [
    {"shape": [None], "name": "node_number", "dtype": "int32", "ragged": True},
    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
    {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
    {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
    {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
]
DATASET_METHODS = [
    {"map_list": {"method": "set_range_periodic", "max_distance": 8.0, "max_neighbours": 17}},
    {"map_list": {"method": "set_angle", "allow_multi_edges": True}},
]


def _callbacks():
    return {
        "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
        "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
        "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
        "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
        "abc": lambda st, ds: np.array(st.lattice.abc),
        "charge": lambda st, ds: np.array([st.charge], dtype="float"),
        "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
        "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int32"),
    }


def build_chunk(payload):
    """Build crystal graphs for a chunk of pymatgen structure-dicts. Returns (n_in, n_removed)."""
    struct_dicts, ys = payload
    from pymatgen.core import Structure
    from kgcnn.data.crystal import CrystalDataset
    structs = [Structure.from_dict(d) for d in struct_dicts]
    ds = CrystalDataset()
    ds._map_callbacks(structs, pd.Series(ys), _callbacks())
    ds.set_methods(DATASET_METHODS)
    removed = ds.clean(MODEL_INPUTS_SPEC)
    _ = ds.tensor(MODEL_INPUTS_SPEC)
    return len(structs), (0 if removed is None else len(removed))


def chunkify(items, k):
    n = len(items)
    size = (n + k - 1) // k
    return [items[i:i + size] for i in range(0, n, size)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--workers", type=str, default="8,16", help="comma list of worker counts to test")
    p.add_argument("--seed", type=int, default=1123)
    args = p.parse_args()

    t = time.time()
    records = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = [r for r in records if r.get("band_gap") is not None and r.get("structure") is not None]
    random.seed(args.seed)
    random.shuffle(records)
    records = records[:args.n]
    sdicts = [r["structure"] for r in records]
    ys = [float(r["band_gap"]) for r in records]
    print(f"Loaded {len(sdicts)} structures in {time.time()-t:.1f}s", flush=True)

    ctx = mp.get_context("spawn")

    # Serial baseline (one process).
    t0 = time.time()
    n_in, n_rm = build_chunk((sdicts, ys))
    t_serial = time.time() - t0
    rate_serial = len(sdicts) / t_serial
    print(f"\n[SERIAL] {n_in} structs in {t_serial:.1f}s -> {rate_serial:.1f} struct/s (removed {n_rm})", flush=True)

    for w in [int(x) for x in args.workers.split(",")]:
        sd_chunks = chunkify(sdicts, w)
        y_chunks = chunkify(ys, w)
        payloads = list(zip(sd_chunks, y_chunks))
        t0 = time.time()
        with ctx.Pool(processes=w) as pool:
            res = pool.map(build_chunk, payloads)
        t_par = time.time() - t0
        rate_par = len(sdicts) / t_par
        speedup = t_serial / t_par
        print(f"[PAR w={w:>2}] {sum(r[0] for r in res)} structs in {t_par:.1f}s "
              f"-> {rate_par:.1f} struct/s | speedup x{speedup:.2f} "
              f"(incl. ~one-time worker TF import)", flush=True)

    print(f"\nFull-dataset (154k) projection from serial rate: {154000/rate_serial/60:.1f} min serial", flush=True)


if __name__ == "__main__":
    main()
