"""Parallel DimeNet++ tensor cache builder (build-once, reshuffle into N seed splits).

The expensive kgcnn graph build (set_range_periodic + set_angle) is run ONCE over all
structures using a spawn-based process pool (TF + fork deadlocks). Because graph cleaning is
per-structure deterministic, the per-seed train/val/test splits derived afterward are identical
to building each seed separately -- at ~1/N the cost.

Sources:
  --source eform_idprop  : ALIGNN id_prop.json (JARVIS atoms dicts), target formation_energy_per_atom
  --source mp21_jsonl    : mp21.jsonl (pymatgen Structure dicts),    target band_gap
"""
import argparse
import json
import os
import pickle
import random
import tempfile
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
    """Worker: build graphs for one contiguous chunk, save x to a temp npz. Returns metadata."""
    chunk_id, start, struct_dicts, ys, tmpdir = payload
    from pymatgen.core import Structure
    from kgcnn.data.crystal import CrystalDataset
    structs = [Structure.from_dict(d) for d in struct_dicts]
    ds = CrystalDataset()
    ds._map_callbacks(structs, pd.Series(ys), _callbacks())
    ds.set_methods(DATASET_METHODS)
    removed = ds.clean(MODEL_INPUTS_SPEC)
    x = ds.tensor(MODEL_INPUTS_SPEC)
    removed_local = sorted(int(r) for r in (removed if removed is not None else []))
    path = os.path.join(tmpdir, f"chunk_{chunk_id:05d}.npz")
    np.savez(path, x=np.array(x, dtype=object))
    return {"chunk_id": chunk_id, "start": start, "n": len(structs),
            "removed_local": removed_local, "path": path}


def load_records(source, input_path, target_key, id_key):
    """Return struct_dicts (pymatgen as_dict), targets (np float), ids -- in FILE order, no shuffle."""
    struct_dicts, targets, ids = [], [], []
    if source == "mp21_jsonl":
        with open(input_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows = [r for r in rows if r.get(target_key) is not None and r.get("structure") is not None]
        for r in rows:
            struct_dicts.append(r["structure"])
            targets.append(float(r[target_key]))
            ids.append(r[id_key])
    elif source == "eform_idprop":
        from jarvis.core.atoms import Atoms
        with open(input_path) as f:
            rows = json.load(f)
        rows = [r for r in rows if r.get(target_key) is not None and r.get("atoms") is not None]
        for r in rows:
            struct_dicts.append(Atoms.from_dict(r["atoms"]).pymatgen_converter().as_dict())
            targets.append(float(r[target_key]))
            ids.append(r[id_key])
    else:
        raise ValueError(f"Unknown source {source}")
    return struct_dicts, np.asarray(targets, dtype=float), ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["eform_idprop", "mp21_jsonl"])
    p.add_argument("--input", required=True)
    p.add_argument("--target_key", required=True)
    p.add_argument("--id_key", default="material_id")
    p.add_argument("--seeds", default="1123,1456,1789")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save_root", required=True)
    p.add_argument("--tensor_filename", required=True)
    p.add_argument("--scaler_filename", required=True)
    p.add_argument("--manifest_filename", default="manifest.pkl")
    p.add_argument("--tmpdir", default="")
    args = p.parse_args()

    t0 = time.time()
    sdicts, targets, ids = load_records(args.source, args.input, args.target_key, args.id_key)
    if args.limit:
        sdicts, targets, ids = sdicts[:args.limit], targets[:args.limit], ids[:args.limit]
    N = len(sdicts)
    print(f"Loaded {N} records in {time.time()-t0:.1f}s", flush=True)
    if N == 0:
        raise ValueError("No records loaded.")

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="dimenet_build_")
    os.makedirs(tmpdir, exist_ok=True)

    # ---- parallel build over file order ----
    payloads, start = [], 0
    cid = 0
    while start < N:
        end = min(start + args.chunk_size, N)
        payloads.append((cid, start, sdicts[start:end], targets[start:end].tolist(), tmpdir))
        start = end
        cid += 1
    print(f"Building {N} graphs in {len(payloads)} chunks x {args.workers} workers (spawn)...", flush=True)

    tb = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(build_chunk, payloads)
    print(f"Parallel graph build done in {time.time()-tb:.1f}s ({N/(time.time()-tb):.1f} struct/s)", flush=True)

    # ---- assemble full survivor set in file order ----
    import tensorflow as tf
    results.sort(key=lambda r: r["chunk_id"])
    raw_idx = []                      # original index of each surviving graph, in order
    per_input = [[] for _ in MODEL_INPUTS_SPEC]
    total_removed = 0
    for r in results:
        removed_set = set(r["removed_local"])
        total_removed += len(removed_set)
        kept = [l for l in range(r["n"]) if l not in removed_set]
        raw_idx.extend(r["start"] + l for l in kept)
        xc = np.load(r["path"], allow_pickle=True)["x"].tolist()
        for j in range(len(MODEL_INPUTS_SPEC)):
            per_input[j].append(xc[j])
    full_x = [tf.concat(parts, axis=0) for parts in per_input]
    raw_idx = np.asarray(raw_idx, dtype=int)
    n_surv = len(raw_idx)
    print(f"Survivors: {n_surv}/{N} (removed {total_removed})", flush=True)

    # position in full_x for each original index (-1 if removed)
    pos_of_raw = np.full(N, -1, dtype=int)
    pos_of_raw[raw_idx] = np.arange(n_surv)

    n_train = int(0.8 * N)
    n_val = int(0.1 * N)

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        perm = list(range(N))
        random.seed(seed)
        random.shuffle(perm)
        split_raw = {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }

        out = {}
        scaler = None
        for split in ["train", "val", "test"]:
            raws = [r for r in split_raw[split] if pos_of_raw[r] >= 0]   # survivors only, in perm order
            positions = [int(pos_of_raw[r]) for r in raws]
            x_split = [tf.gather(full_x[j], positions, axis=0) for j in range(len(MODEL_INPUTS_SPEC))]
            y_split = targets[raws].reshape(-1, 1)
            ids_split = [ids[r] for r in raws]
            if split == "train":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
                y_scaled = scaler.fit_transform(y_split)
            else:
                y_scaled = scaler.transform(y_split)
            out[split] = (x_split, y_split, y_scaled, ids_split, np.asarray(targets[raws], dtype=float))

        seed_dir = os.path.join(args.save_root, f"splitseed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        tensor_path = os.path.join(seed_dir, args.tensor_filename)
        # Save with pickle protocol 5: np.savez pickles object arrays with protocol 3, which
        # caps a single object at 4 GiB and overflows on the full x_train ragged set. Same dict
        # keys as before; x_* are plain lists of RaggedTensors (load with pickle, no .tolist()).
        cache_obj = {
            "y_train": out["train"][1], "y_train_scaled": out["train"][2],
            "y_val": out["val"][1], "y_val_scaled": out["val"][2],
            "y_test": out["test"][1], "y_test_scaled": out["test"][2],
            "x_train": out["train"][0], "x_val": out["val"][0], "x_test": out["test"][0],
            "train_ids": np.array(out["train"][3], dtype=object),
            "val_ids": np.array(out["val"][3], dtype=object),
            "test_ids": np.array(out["test"][3], dtype=object),
            "train_targets": out["train"][4], "val_targets": out["val"][4], "test_targets": out["test"][4],
        }
        with open(tensor_path, "wb") as fcache:
            pickle.dump(cache_obj, fcache, protocol=5)
        with open(os.path.join(seed_dir, args.scaler_filename), "wb") as f:
            pickle.dump(scaler, f)
        manifest = {
            "source": args.source, "input": os.path.abspath(args.input),
            "target_key": args.target_key, "id_key": args.id_key, "seed": seed,
            "n_records": int(N), "n_survivors": int(n_surv),
            "split_sizes": {k: len(out[k][3]) for k in ["train", "val", "test"]},
            "inputs_spec": MODEL_INPUTS_SPEC, "methods": DATASET_METHODS,
        }
        with open(os.path.join(seed_dir, args.manifest_filename), "wb") as f:
            pickle.dump(manifest, f)
        print(f"[seed {seed}] wrote {tensor_path}  sizes="
              f"{ {k: len(out[k][3]) for k in ['train','val','test']} }", flush=True)

    # cleanup temp chunks
    for r in results:
        try:
            os.remove(r["path"])
        except OSError:
            pass

    print(f"ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
