import os, json, zipfile
import numpy as np
import time
t0 = time.time()
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms

# pymatgen Structure type
from pymatgen.core import Structure


import json, random
import numpy as np
from tqdm.auto import tqdm
from pymatgen.core import Structure

# ---- defaults from your script ----
jsonl_path = "mp21.jsonl"      # <-- set this to your file
id_key = "material_id"
target_key = "band_gap"
limit = 0  # 0 = all

structures = []
targets = []
ids = []

with open(jsonl_path, "r") as f:
    for idx, line in enumerate(tqdm(f, desc="Reading JSONL")):
        if limit and idx >= limit:
            break
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)

        mid = row[id_key]
        y = float(row[target_key])

        # structure stored as dict -> pymatgen Structure
        pmg_struct = Structure.from_dict(row["structure"])

        ids.append(mid)
        targets.append(y)
        structures.append(pmg_struct)

print(f"Loaded {len(structures)} samples.")

# shuffle like your script
print("Shuffling...")
perm = list(range(len(structures)))
random.shuffle(perm)

structures = [structures[i] for i in perm]
targets = np.array([targets[i] for i in perm], dtype=float)
ids = [ids[i] for i in perm]

print("Done. Example:")
print(ids[0], targets[0], structures[0])


import pandas as pd
from kgcnn.data.crystal import CrystalDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
}

from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.DimeNetPP import make_crystal_model
from sklearn.preprocessing import StandardScaler
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
import os
import tensorflow as tf

gpu_to_use = [0]  # e.g. [0] or None

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        if gpu_to_use is None:
            # use all visible GPUs (default)
            pass
        else:
            # pick specific GPU indices
            visible = [gpus[i] for i in gpu_to_use]
            tf.config.set_visible_devices(visible, "GPU")
        # optional but recommended
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU selection failed:", e)

print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
import numpy as np
from copy import deepcopy

import os.path
import argparse
import pandas as pd
import tensorflow as tf


hyper_1_alignnish = {
    "model": {
        "class_name": "make_crystal_model",
        "module_name": "kgcnn.literature.DimeNetPP",
        "config": {
            "name": "DimeNetPP",
            "inputs": [
                {"shape": [None],    "name": "node_number",      "dtype": "float32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices",    "dtype": "int64",   "ragged": True},
                {"shape": [None, 2], "name": "angle_indices",    "dtype": "int64",   "ragged": True},
                {"shape": (None, 3), "name": "range_image",      "dtype": "int64",   "ragged": True},
                {"shape": (3, 3),    "name": "graph_lattice",    "dtype": "float32", "ragged": False},
            ],

            # ALIGNN atom_input_features=92; DimeNet input_dim=95 is periodic-table sized; keep as-is.
            # Match ALIGNN embedding_features/hidden_features ~64 by reducing widths.
            "input_embedding": {
                "node": {
                    "input_dim": 95,
                    "output_dim": 64,   # was 128
                    "embeddings_initializer": {
                        "class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}
                    }
                }
            },

            # Capacity/depth match
            "emb_size": 64,        # was 128
            "out_emb_size": 128,   # was 256 (keep a bit larger for readout, but closer)
            "int_emb_size": 64,    # already matches
            "basis_emb_size": 8,

            # Depth: ALIGNN has 2 MP stages (alignn+gcn). Closest is 2 blocks.
            "num_blocks": 2,       # was 4

            # Keep angular/radial basis sizes (not directly comparable to ALIGNN triplet feats)
            "num_spherical": 7,
            "num_radial": 6,

            # Match neighbor cutoff (ALIGNN cutoff=8)
            "cutoff": 8.0,         # was 5.0
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
        }
    },

    "training": {
        "cross_validation": None,
        "execute_folds": None,

        "fit": {
            # Match ALIGNN batch_size=64 if memory allows
            "batch_size": 64,          # was 16
            "epochs": 350,             # was 780
            "validation_freq": 10,
            "verbose": 2,
            "callbacks": [],
            "validation_batch_size": 64
        },

        "compile": {
            # Remove Addons>MovingAverage (since you can't install TF-addons)
            # Match LR=1e-3 and "AdamW" flavor. wd=0 so Adam vs AdamW is equivalent.
            "optimizer": {
                "class_name": "AdamW",
                "config": {
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "amsgrad": True
                }
            },

            # Match ALIGNN criterion="mse"
            "loss": "mean_squared_error"
        },

        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
        "multi_target_indices": None
    },

    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                # Match cutoff here too (range construction uses max_distance)
                {"map_list": {"method": "set_range_periodic", "max_distance": 8.0, "max_neighbours": 17}},
                {"map_list": {"method": "set_angle", "allow_multi_edges": True}}
            ]
        }
    },

    "info": {"postfix": "", "postfix_file": "", "kgcnn_version": "2.1.0"}
}

# =========================
# FIXED: VAL + TEST clean() with removed-index handling + consistent splits
# Also: print shapes using x_*[0].shape (where applicable)
# =========================

import pandas as pd
import numpy as np
from kgcnn.data.crystal import CrystalDataset
from sklearn.preprocessing import StandardScaler

# ---- use the SAME split indices you originally intended (before any cleaning) ----
N = len(structures)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val

train_struct  = structures[:n_train]
train_targets = targets[:n_train]
train_ids     = ids[:n_train]

val_struct  = structures[n_train:n_train+n_val]
val_targets = targets[n_train:n_train+n_val]
val_ids     = ids[n_train:n_train+n_val]

test_struct  = structures[n_train+n_val:]
test_targets = targets[n_train+n_val:]
test_ids     = ids[n_train+n_val:]


def build_split_tensors(struct_list, y_list, id_list, methods, inputs_spec, callbacks, scaler=None, fit_scaler=False):
    """
    Build x/y tensors via CrystalDataset, run clean(), and return:
      x, y, y_scaled, ids_clean, targets_clean, removed_indices
    """
    ds = CrystalDataset()
    ds._map_callbacks(struct_list, pd.Series(y_list), callbacks)
    ds.set_methods(methods)

    removed = ds.clean(inputs_spec)  # list of indices removed

    y = np.array(ds.get("graph_labels"))
    x = ds.tensor(inputs_spec)

    # align ids/targets with removals
    keep = np.ones(len(struct_list), dtype=bool)
    if removed is not None and len(removed) > 0:
        keep[np.array(removed, dtype=int)] = False

    ids_clean = [id_list[i] for i in range(len(id_list)) if keep[i]]
    targets_clean = np.array(y_list)[keep]

    # scaling
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True, copy=True)

    if fit_scaler:
        y_scaled = scaler.fit_transform(y)
    else:
        y_scaled = scaler.transform(y)

    return x, y, y_scaled, ids_clean, targets_clean, removed, scaler


methods = hyper_1_alignnish["data"]["dataset"]["methods"]
inputs_spec = hyper_1_alignnish["model"]["config"]["inputs"]

# ---- TRAIN (fit scaler here) ----
x_train, y_train, y_train_scaled, train_ids_clean, train_targets_clean, removed_train, scaler = build_split_tensors(
    train_struct, train_targets, train_ids,
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=None, fit_scaler=True
)

# ---- VAL (use same scaler) ----
x_val, y_val, y_val_scaled, val_ids_clean, val_targets_clean, removed_val, _ = build_split_tensors(
    val_struct, val_targets, val_ids,
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=scaler, fit_scaler=False
)

# ---- TEST (use same scaler) ----
x_test, y_test, y_test_scaled, test_ids_clean, test_targets_clean, removed_test, _ = build_split_tensors(
    test_struct, test_targets, test_ids,
    methods=methods, inputs_spec=inputs_spec, callbacks=callbacks,
    scaler=scaler, fit_scaler=False
)

print("Removed counts:", {
    "train": len(removed_train) if removed_train is not None else 0,
    "val":   len(removed_val)   if removed_val is not None else 0,
    "test":  len(removed_test)  if removed_test is not None else 0,
})

# Use x_*[0].shape for a concrete tensor "shape" sanity check.
# (x_* is typically a list of inputs; x_*[0] corresponds to node_number ragged array container)
print("After clean sizes:", {
    "train": (len(train_ids_clean), y_train.shape, getattr(x_train[0], "shape", None)),
    "val":   (len(val_ids_clean),   y_val.shape,   getattr(x_val[0], "shape", None)),
    "test":  (len(test_ids_clean),  y_test.shape,  getattr(x_test[0], "shape", None)),
})


# =========================
# SAVE (now also saves cleaned ids/targets + removed indices)
# =========================
import os, pickle

save_dir = "./cached_tensors_dimenetpp"
os.makedirs(save_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(save_dir, "dimenetpp_cached_tensors.npz"),

    # y
    y_train=y_train, y_train_scaled=y_train_scaled,
    y_val=y_val,     y_val_scaled=y_val_scaled,
    y_test=y_test,   y_test_scaled=y_test_scaled,

    # x (ragged/list inputs)
    x_train=np.array(x_train, dtype=object),
    x_val=np.array(x_val, dtype=object),
    x_test=np.array(x_test, dtype=object),

    # ids + original targets aligned to cleaned sets
    train_ids=np.array(train_ids_clean, dtype=object),
    val_ids=np.array(val_ids_clean, dtype=object),
    test_ids=np.array(test_ids_clean, dtype=object),

    train_targets=np.array(train_targets_clean, dtype=float),
    val_targets=np.array(val_targets_clean, dtype=float),
    test_targets=np.array(test_targets_clean, dtype=float),

    # removed indices (relative to each split before cleaning)
    removed_train=np.array(removed_train if removed_train is not None else [], dtype=int),
    removed_val=np.array(removed_val if removed_val is not None else [], dtype=int),
    removed_test=np.array(removed_test if removed_test is not None else [], dtype=int),
)

with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

manifest = {
    "split_sizes_before_clean": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    "split_sizes_after_clean":  {"train": int(len(train_ids_clean)), "val": int(len(val_ids_clean)), "test": int(len(test_ids_clean))},
    "inputs_spec": inputs_spec,
    "methods": methods,
    "id_key": id_key,
    "target_key": target_key,
}
with open(os.path.join(save_dir, "manifest.pkl"), "wb") as f:
    pickle.dump(manifest, f)

print("Saved cached tensors (+cleaned ids/targets) to:", save_dir)




t1 = time.time()
data_seconds = t1 - t0
print(f"Dataset generation wall time: {data_seconds:.2f} s ({data_seconds/60:.2f} min)")