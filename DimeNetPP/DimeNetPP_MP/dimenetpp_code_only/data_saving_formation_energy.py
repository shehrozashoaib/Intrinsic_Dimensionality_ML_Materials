import os
import time
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from mp_api.client import MPRester
from kgcnn.data.crystal import CrystalDataset


t0 = time.time()

DEFAULT_API_KEY = os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY") or "HUYRyQqHnQMtHihJ6PETXJuI1oLax1HQ"
DEFAULT_RANDOM_SEED = 423


parser = argparse.ArgumentParser("Build DimeNet++ cached tensors for formation energy")
parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="Materials Project API key.")
parser.add_argument("--id_key", type=str, default="material_id", help="Material ID field to store.")
parser.add_argument(
    "--target_key",
    type=str,
    default="formation_energy_per_atom",
    help="Target property fetched from Materials Project.",
)
parser.add_argument("--limit", type=int, default=0, help="0 = all records.")
parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Shuffle seed.")
parser.add_argument("--gpu", type=int, default=0, help="GPU index. Set to -1 for CPU-only.")
parser.add_argument(
    "--save_dir",
    type=str,
    default="./cached_tensors_dimenetpp",
    help="Directory to save the cached tensors.",
)
parser.add_argument(
    "--tensor_filename",
    type=str,
    default="dimenetpp_eform_cached_tensors.npz",
    help="Output tensor archive name. Kept distinct from band-gap cache.",
)
parser.add_argument(
    "--scaler_filename",
    type=str,
    default="scaler_eform.pkl",
    help="Output scaler filename. Kept distinct from band-gap scaler.",
)
parser.add_argument(
    "--manifest_filename",
    type=str,
    default="manifest_eform.pkl",
    help="Output manifest filename. Kept distinct from band-gap manifest.",
)
args = parser.parse_args()


gpu_to_use = None if args.gpu is None or args.gpu < 0 else [args.gpu]
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


def fetch_mp_formation_energy_records(api_key, id_key, target_key, limit=0, seed=DEFAULT_RANDOM_SEED):
    print("Connecting to Materials Project API...")
    with MPRester(api_key) as mpr:
        results = mpr.materials.summary.search(fields=[id_key, "structure", target_key])

    filtered = [r for r in results if getattr(r, target_key, None) is not None]
    print(f"Fetched {len(filtered)} records with non-null {target_key}.")

    random.seed(seed)
    random.shuffle(filtered)
    print(f"Shuffled records with seed={seed}.")

    structures = []
    targets = []
    ids = []
    for idx, result in enumerate(tqdm(filtered, desc="Processing MP formation-energy data")):
        if limit and idx >= limit:
            break
        ids.append(getattr(result, id_key))
        targets.append(float(getattr(result, target_key)))
        structures.append(result.structure)

    return structures, np.array(targets, dtype=float), ids


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


hyper_1_alignnish = {
    "model": {
        "class_name": "make_crystal_model",
        "module_name": "kgcnn.literature.DimeNetPP",
        "config": {
            "name": "DimeNetPP",
            "inputs": [
                {"shape": [None], "name": "node_number", "dtype": "int32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
                {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
            ],
            "input_embedding": {
                "node": {
                    "input_dim": 95,
                    "output_dim": 64,
                    "embeddings_initializer": {
                        "class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772},
                    },
                }
            },
            "emb_size": 64,
            "out_emb_size": 128,
            "int_emb_size": 64,
            "basis_emb_size": 8,
            "num_blocks": 2,
            "num_spherical": 7,
            "num_radial": 6,
            "cutoff": 8.0,
            "envelope_exponent": 5,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "num_dense_output": 3,
            "num_targets": 1,
            "extensive": False,
            "output_init": "zeros",
            "activation": "swish",
            "verbose": 10,
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": {},
        },
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 8.0, "max_neighbours": 17}},
                {"map_list": {"method": "set_angle", "allow_multi_edges": True}},
            ],
        }
    },
}


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


structures, targets, ids = fetch_mp_formation_energy_records(
    api_key=args.api_key,
    id_key=args.id_key,
    target_key=args.target_key,
    limit=args.limit,
    seed=args.seed,
)

print(f"Loaded {len(structures)} formation-energy samples.")
if len(structures) == 0:
    raise ValueError("No formation-energy structures were fetched. Nothing to save.")

print("Example:")
print(ids[0], targets[0], structures[0])

methods = hyper_1_alignnish["data"]["dataset"]["methods"]
inputs_spec = hyper_1_alignnish["model"]["config"]["inputs"]

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
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=None, fit_scaler=True,
)

x_val, y_val, y_val_scaled, val_ids_clean, val_targets_clean, removed_val, _ = build_split_tensors(
    val_struct, val_targets, val_ids,
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=scaler, fit_scaler=False,
)

x_test, y_test, y_test_scaled, test_ids_clean, test_targets_clean, removed_test, _ = build_split_tensors(
    test_struct, test_targets, test_ids,
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=scaler, fit_scaler=False,
)

print("Removed counts:", {
    "train": len(removed_train) if removed_train is not None else 0,
    "val": len(removed_val) if removed_val is not None else 0,
    "test": len(removed_test) if removed_test is not None else 0,
})

print("After clean sizes:", {
    "train": (len(train_ids_clean), y_train.shape, getattr(x_train[0], "shape", None)),
    "val": (len(val_ids_clean), y_val.shape, getattr(x_val[0], "shape", None)),
    "test": (len(test_ids_clean), y_test.shape, getattr(x_test[0], "shape", None)),
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
    "dataset_name": "formation_energy_per_atom",
    "split_sizes_before_clean": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    "split_sizes_after_clean": {"train": int(len(train_ids_clean)), "val": int(len(val_ids_clean)), "test": int(len(test_ids_clean))},
    "inputs_spec": inputs_spec,
    "methods": methods,
    "id_key": args.id_key,
    "target_key": args.target_key,
    "seed": int(args.seed),
    "limit": int(args.limit),
    "tensor_filename": args.tensor_filename,
    "scaler_filename": args.scaler_filename,
}
with open(manifest_path, "wb") as f:
    pickle.dump(manifest, f)

print("Saved formation-energy cached tensors to:", tensor_path)
print("Saved formation-energy scaler to:", scaler_path)
print("Saved formation-energy manifest to:", manifest_path)

t1 = time.time()
data_seconds = t1 - t0
print(f"Dataset generation wall time: {data_seconds:.2f} s ({data_seconds/60:.2f} min)")
