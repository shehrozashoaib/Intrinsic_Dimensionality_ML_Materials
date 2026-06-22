import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from kgcnn.data.crystal import CrystalDataset
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


t0 = time.time()

DEFAULT_RANDOM_SEED = 423

parser = argparse.ArgumentParser("Build DimeNet++ cached tensors from mp21.jsonl (pymatgen structures, band gap)")
parser.add_argument("--jsonl", required=True, help="Path to mp21.jsonl (one JSON record per line).")
parser.add_argument("--id_key", type=str, default="material_id")
parser.add_argument("--target_key", type=str, default="band_gap")
parser.add_argument("--structure_key", type=str, default="structure")
parser.add_argument("--limit", type=int, default=0, help="0 = all records.")
parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Shuffle/split seed.")
parser.add_argument("--gpu", type=int, default=-1, help="GPU index. Set to -1 for CPU-only tensor generation.")
parser.add_argument("--save_dir", type=str, default="./cached_tensors_dimenetpp_bandgap")
parser.add_argument("--tensor_filename", type=str, default="dimenetpp_bandgap_cached_tensors.npz")
parser.add_argument("--scaler_filename", type=str, default="scaler_bandgap.pkl")
parser.add_argument("--manifest_filename", type=str, default="manifest_bandgap.pkl")
args = parser.parse_args()


gpu_to_use = None if args.gpu is None or args.gpu < 0 else [args.gpu]
if args.gpu is not None and args.gpu < 0:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        if gpu_to_use is not None:
            visible = [gpus[i] for i in gpu_to_use]
            tf.config.set_visible_devices(visible, "GPU")
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU selection failed:", e)

print("Visible GPUs:", tf.config.list_physical_devices("GPU"))


callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int32"),
}


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


def load_records(jsonl_path, id_key, target_key, structure_key, limit=0, seed=DEFAULT_RANDOM_SEED):
    print(f"Reading {jsonl_path}")
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    filtered = [
        r for r in records
        if r.get(target_key) is not None and r.get(structure_key) is not None
    ]
    random.seed(seed)
    random.shuffle(filtered)
    if limit:
        filtered = filtered[:limit]

    structures = []
    targets = []
    ids = []
    for row in tqdm(filtered, desc="Loading pymatgen structures"):
        structures.append(Structure.from_dict(row[structure_key]))
        targets.append(float(row[target_key]))
        ids.append(row[id_key])

    return structures, np.asarray(targets, dtype=float), ids


def build_split_tensors(struct_list, y_list, id_list, methods, inputs_spec, callbacks, scaler=None, fit_scaler=False):
    ds = CrystalDataset()
    ds._map_callbacks(struct_list, pd.Series(y_list), callbacks)
    ds.set_methods(methods)

    removed = ds.clean(inputs_spec)
    y = np.array(ds.get("graph_labels"))
    x = ds.tensor(inputs_spec)

    keep = np.ones(len(struct_list), dtype=bool)
    if removed is not None and len(removed) > 0:
        keep[np.array(removed, dtype=int)] = False

    ids_clean = [id_list[i] for i in range(len(id_list)) if keep[i]]
    targets_clean = np.array(y_list)[keep]

    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True, copy=True)

    if fit_scaler:
        y_scaled = scaler.fit_transform(y)
    else:
        y_scaled = scaler.transform(y)

    return x, y, y_scaled, ids_clean, targets_clean, removed, scaler


structures, targets, ids = load_records(
    args.jsonl,
    id_key=args.id_key,
    target_key=args.target_key,
    structure_key=args.structure_key,
    limit=args.limit,
    seed=args.seed,
)

print(f"Loaded {len(structures)} band-gap samples.")
if len(structures) == 0:
    raise ValueError("No band-gap structures were loaded. Nothing to save.")
print("Example:", ids[0], targets[0], structures[0].composition.reduced_formula)

N = len(structures)
n_train = int(0.8 * N)
n_val = int(0.1 * N)
n_test = N - n_train - n_val

train_struct = structures[:n_train]
train_targets = targets[:n_train]
train_ids = ids[:n_train]

val_struct = structures[n_train:n_train + n_val]
val_targets = targets[n_train:n_train + n_val]
val_ids = ids[n_train:n_train + n_val]

test_struct = structures[n_train + n_val:]
test_targets = targets[n_train + n_val:]
test_ids = ids[n_train + n_val:]

x_train, y_train, y_train_scaled, train_ids_clean, train_targets_clean, removed_train, scaler = build_split_tensors(
    train_struct, train_targets, train_ids,
    methods=DATASET_METHODS, inputs_spec=MODEL_INPUTS_SPEC, callbacks=callbacks,
    scaler=None, fit_scaler=True,
)
x_val, y_val, y_val_scaled, val_ids_clean, val_targets_clean, removed_val, _ = build_split_tensors(
    val_struct, val_targets, val_ids,
    methods=DATASET_METHODS, inputs_spec=MODEL_INPUTS_SPEC, callbacks=callbacks,
    scaler=scaler, fit_scaler=False,
)
x_test, y_test, y_test_scaled, test_ids_clean, test_targets_clean, removed_test, _ = build_split_tensors(
    test_struct, test_targets, test_ids,
    methods=DATASET_METHODS, inputs_spec=MODEL_INPUTS_SPEC, callbacks=callbacks,
    scaler=scaler, fit_scaler=False,
)

print("Removed counts:", {
    "train": len(removed_train) if removed_train is not None else 0,
    "val": len(removed_val) if removed_val is not None else 0,
    "test": len(removed_test) if removed_test is not None else 0,
})
print("After clean sizes:", {
    "train": (len(train_ids_clean), y_train.shape),
    "val": (len(val_ids_clean), y_val.shape),
    "test": (len(test_ids_clean), y_test.shape),
})

os.makedirs(args.save_dir, exist_ok=True)

tensor_path = os.path.join(args.save_dir, args.tensor_filename)
scaler_path = os.path.join(args.save_dir, args.scaler_filename)
manifest_path = os.path.join(args.save_dir, args.manifest_filename)

np.savez_compressed(
    tensor_path,
    y_train=y_train,
    y_train_scaled=y_train_scaled,
    y_val=y_val,
    y_val_scaled=y_val_scaled,
    y_test=y_test,
    y_test_scaled=y_test_scaled,
    x_train=np.array(x_train, dtype=object),
    x_val=np.array(x_val, dtype=object),
    x_test=np.array(x_test, dtype=object),
    train_ids=np.array(train_ids_clean, dtype=object),
    val_ids=np.array(val_ids_clean, dtype=object),
    test_ids=np.array(test_ids_clean, dtype=object),
    train_targets=np.array(train_targets_clean, dtype=float),
    val_targets=np.array(val_targets_clean, dtype=float),
    test_targets=np.array(test_targets_clean, dtype=float),
    removed_train=np.array(removed_train if removed_train is not None else [], dtype=int),
    removed_val=np.array(removed_val if removed_val is not None else [], dtype=int),
    removed_test=np.array(removed_test if removed_test is not None else [], dtype=int),
)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

manifest = {
    "dataset_name": "band_gap",
    "source_jsonl": os.path.abspath(args.jsonl),
    "split_sizes_before_clean": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    "split_sizes_after_clean": {
        "train": int(len(train_ids_clean)),
        "val": int(len(val_ids_clean)),
        "test": int(len(test_ids_clean)),
    },
    "inputs_spec": MODEL_INPUTS_SPEC,
    "methods": DATASET_METHODS,
    "id_key": args.id_key,
    "target_key": args.target_key,
    "structure_key": args.structure_key,
    "seed": int(args.seed),
    "limit": int(args.limit),
    "tensor_filename": args.tensor_filename,
    "scaler_filename": args.scaler_filename,
}
with open(manifest_path, "wb") as f:
    pickle.dump(manifest, f)

print("Saved band-gap cached tensors to:", tensor_path)
print("Saved band-gap scaler to:", scaler_path)
print("Saved band-gap manifest to:", manifest_path)

t1 = time.time()
print(f"Dataset generation wall time: {t1 - t0:.2f} s ({(t1 - t0) / 60:.2f} min)")
