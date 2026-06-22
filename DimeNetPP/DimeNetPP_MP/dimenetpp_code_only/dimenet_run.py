# dimenet_run.py
import os, json, zipfile
import numpy as np
import time
import argparse
import pickle

from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms

t0 = time.time()

import pandas as pd
from kgcnn.data.crystal import CrystalDataset
from sklearn.preprocessing import StandardScaler

# -------------------------
# CLI ARGS (ONLY wrapper + epochs/batch/gpu/output)
# -------------------------
parser = argparse.ArgumentParser("DimeNet++ SubspaceProjectedGradTF runner")
parser.add_argument("--epochs", type=int, default=350, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--method", type=str, default="fastfood", choices=["dense", "fastfood"], help="Wrapper method")
parser.add_argument("--id_dim", type=float, default=1.0, help="Intrinsic dimension fraction in (0,1]")
parser.add_argument("--orthonormal", action="store_true", help="Orthonormalize dense P via QR")
parser.add_argument("--full_rotation", action="store_true", help="Use full rotation mode when d==D (wrapper-specific)")
parser.add_argument("--seed", type=int, default=123, help="Random seed for wrapper")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
parser.add_argument("--cache_dir", type=str, default="./cached_tensors_dimenetpp", help="Cache dir for tensors/scaler")
parser.add_argument("--out_dir", type=str, default="runs_dimenetpp", help="Output directory for CSV/history")
parser.add_argument(
    "--train_mode",
    type=str,
    default="wrapped",
    choices=["base", "wrapped"],
    help="Train the original base model or the intrinsic-dimension wrapped model."
)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

exp_tag = (
    f"method={args.method}"
    f"_dim={int(round(args.id_dim*100))}pct"
    f"_epochs={args.epochs}"
    f"_seed={args.seed}"
    f"_ortho={int(args.orthonormal)}"
    f"_rot={int(args.full_rotation)}"
    f"_train_mode={args.train_mode}"
    f"_task=band_gap"
)

# -------------------------
# Callbacks for CrystalDataset
# -------------------------
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

from kgcnn.literature.DimeNetPP import make_crystal_model
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
import tensorflow as tf

# -------------------------
# GPU selection
# -------------------------
gpu_to_use = None if args.gpu is None else [args.gpu]
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

# -------------------------
# Model hyperparams (UNCHANGED except epochs/batch below)
# -------------------------
hyper_1_alignnish = {
    "model": {
        "class_name": "make_crystal_model",
        "module_name": "kgcnn.literature.DimeNetPP",
        "config": {
            "name": "DimeNetPP",
            "inputs": [
                {"shape": [None],    "name": "node_number",      "dtype": "int32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices",    "dtype": "int64",   "ragged": True},
                {"shape": [None, 2], "name": "angle_indices",    "dtype": "int64",   "ragged": True},
                {"shape": (None, 3), "name": "range_image",      "dtype": "int64",   "ragged": True},
                {"shape": (3, 3),    "name": "graph_lattice",    "dtype": "float32", "ragged": False},
            ],
            "input_embedding": {
                "node": {
                    "input_dim": 95,
                    "output_dim": 55,
                    "embeddings_initializer": {
                        "class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}
                    }
                }
            },
            "emb_size": 55,
            "out_emb_size": 64,
            "int_emb_size": 64,
            "basis_emb_size": 8,
            "num_blocks": 1,
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
        }
    },
    "training": {
        "cross_validation": None,
        "execute_folds": None,
        "fit": {
            "batch_size": 64,
            "epochs": 1,
            "validation_freq": 10,
            "verbose": 2,
            "callbacks": [],
            "validation_batch_size": 64
        },
        "compile": {
            "optimizer": {
                "class_name": "AdamW",
                "config": {
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "amsgrad": True
                }
            },
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
                {"map_list": {"method": "set_range_periodic", "max_distance": 8.0, "max_neighbours": 17}},
                {"map_list": {"method": "set_angle", "allow_multi_edges": True}}
            ]
        }
    },
    "info": {"postfix": "", "postfix_file": "", "kgcnn_version": "2.1.0"}
}
# ONLY change epochs + batch_size from CLI
hyper_1_alignnish["training"]["fit"]["epochs"] = args.epochs
hyper_1_alignnish["training"]["fit"]["batch_size"] = args.batch_size

# -------------------------
# Make kgcnn 4 config explicit (keep behavior the same)
# -------------------------
cfg = hyper_1_alignnish["model"]["config"]
cfg["input_tensor_type"] = "ragged"
cfg["output_tensor_type"] = "padded"

if "input_embedding" in cfg:
    node_emb = cfg["input_embedding"].get("node", {})
    cfg.pop("input_embedding", None)
    cfg["input_node_embedding"] = {
        "input_dim": node_emb.get("input_dim", 95),
        "output_dim": node_emb.get("output_dim", 64),
        "embeddings_initializer": node_emb.get("embeddings_initializer", None),
    }

hyper_1_alignnish["model"]["config"] = cfg
callbacks["node_number"] = lambda st, ds: np.array(st.atomic_numbers, dtype="int32")

# -------------------------
# Load cached tensors
# -------------------------
save_dir = args.cache_dir
cache = np.load(os.path.join(save_dir, "dimenetpp_cached_tensors.npz"), allow_pickle=True)

x_train = cache["x_train"].tolist()
x_val   = cache["x_val"].tolist()
x_test  = cache["x_test"].tolist()

y_train        = cache["y_train"]
y_train_scaled = cache["y_train_scaled"]
y_val          = cache["y_val"]
y_val_scaled   = cache["y_val_scaled"]
y_test         = cache["y_test"]
y_test_scaled  = cache["y_test_scaled"]
test_ids_clean = cache["test_ids"]

with open(os.path.join(save_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

print("Loaded cached tensors from:", save_dir)
print("Shapes:", "y_train", y_train.shape, "y_val", y_val.shape, "y_test", y_test.shape)

# -------------------------
# Build base or projected model
# -------------------------
base_model = make_crystal_model(**hyper_1_alignnish["model"]["config"])

if args.train_mode == "base":
    run_model = base_model
    print("\nRunning BASE DimeNet++ model\n")

elif args.train_mode == "wrapped":
    from dimenet_uber import build_projected_dimenetpp_from_config

    run_model = build_projected_dimenetpp_from_config(
        base_model=base_model,
        model_config=hyper_1_alignnish["model"]["config"],
        method=args.method,        # "dense" or "fastfood"
        d=args.id_dim,             # fraction in (0,1] or integer
        seed=args.seed,
        orthonormal=args.orthonormal,
    )

    print("\nRunning UBER-STYLE PROJECTED DimeNet++ model\n")
    print("Projection method:", args.method)
    print("Intrinsic dimension spec:", args.id_dim)
    print("Trainable vars:", [v.name for v in run_model.trainable_variables])

else:
    raise ValueError(f"Unknown train_mode: {args.train_mode}")

# -------------------------
# Compile
# -------------------------
# Use the same optimizer recipe for both modes for fair comparison
run_model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.0,
        amsgrad=True,
    ),
    loss=tf.keras.losses.MeanSquaredError(),
    jit_compile=False,
)

run_model.summary()

# Optional sanity check for wrapped mode: z=0 should reproduce base init
if args.train_mode == "wrapped":
    try:
        base_pred = base_model.predict(x_val[:1], verbose=0)
        proj_pred = run_model.predict(x_val[:1], verbose=0)
        init_diff = float(np.max(np.abs(np.asarray(base_pred) - np.asarray(proj_pred))))
        print(f"Initialization max abs diff (base vs projected, z=0): {init_diff:.6e}")
    except Exception as e:
        print("Initialization check skipped due to error:", e)

# -------------------------
# Train
# -------------------------
hist = run_model.fit(
    x_train, y_train_scaled,
    validation_data=(x_val, y_val_scaled),
    batch_size=hyper_1_alignnish["training"]["fit"]["batch_size"],
    epochs=hyper_1_alignnish["training"]["fit"]["epochs"],
    verbose=hyper_1_alignnish["training"]["fit"]["verbose"],
    validation_freq=1,
)

# -------------------------
# Test inference + MAE + CSV
# -------------------------
pred_test_scaled = run_model.predict(x_test, verbose=0)
pred_test_scaled_2d = np.asarray(pred_test_scaled).reshape(-1, 1)
pred_test = scaler.inverse_transform(pred_test_scaled_2d).reshape(-1)

y_test_true = np.asarray(y_test).reshape(-1)
test_mae = float(np.mean(np.abs(pred_test - y_test_true)))
print("Test MAE:", test_mae)

# Also report validation MAE in original units for easier comparison
pred_val_scaled = run_model.predict(x_val, verbose=0)
pred_val = scaler.inverse_transform(np.asarray(pred_val_scaled).reshape(-1, 1)).reshape(-1)
y_val_true = np.asarray(y_val).reshape(-1)
val_mae = float(np.mean(np.abs(pred_val - y_val_true)))
print("Validation MAE (original units):", val_mae)

df = pd.DataFrame({
    "id": list(test_ids_clean),
    "target": y_test_true,
    "prediction": pred_test,
    "abs_error": np.abs(pred_test - y_test_true),
})

out_csv = os.path.join(args.out_dir, f"dimenetpp_test_predictions_{exp_tag}.csv")
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

# Save history
hist_path = os.path.join(args.out_dir, f"history_{exp_tag}.json")
with open(hist_path, "w") as f:
    json.dump(hist.history, f)
print("Wrote:", hist_path)

# Save weights
weights_path = os.path.join(args.out_dir, f"weights_{exp_tag}.weights.h5")
run_model.save_weights(weights_path)
print("Wrote:", weights_path)

# For wrapped mode, save z explicitly too
if args.train_mode == "wrapped":
    z_vars = [v for v in run_model.trainable_variables if v.name.endswith("/z:0") or v.name == "z" or v.name.endswith("z")]
    if len(z_vars) == 1:
        z_path = os.path.join(args.out_dir, f"z_{exp_tag}.npy")
        np.save(z_path, z_vars[0].numpy())
        print("Wrote:", z_path)

t1 = time.time()
train_seconds = t1 - t0
print(f"Training wall time: {train_seconds:.2f} s ({train_seconds/60:.2f} min)")