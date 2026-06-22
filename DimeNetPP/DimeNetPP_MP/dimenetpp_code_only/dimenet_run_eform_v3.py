import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd


t0 = time.time()

parser = argparse.ArgumentParser("DimeNet++ eform wrapper-v3 runner")
parser.add_argument("--epochs", type=int, default=350)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--method", type=str, default="fastfood", choices=["dense", "fastfood"])
parser.add_argument("--id_dim", type=float, default=1.0)
parser.add_argument("--orthonormal", action="store_true")
parser.add_argument("--full_rotation", action="store_true")
parser.add_argument("--dense_block_cols", type=int, default=512)
parser.add_argument("--orthonormal_backend", type=str, default="pytorch_cpu", choices=["tensorflow_cpu", "pytorch_cpu", "pytorch_gpu"])
parser.add_argument("--torch_python", type=str, default="/venv/main/bin/python")
parser.add_argument("--torch_q_cache_dir", type=str, default="./orthonormal_q_cache_v3")
parser.add_argument("--torch_q_gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--split_seed", type=int, default=None)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--cache_dir", type=str, default="./cached_tensors_dimenetpp_eform/splitseed1123")
parser.add_argument("--cache_file", type=str, default="dimenetpp_eform_cached_tensors.pkl")
parser.add_argument("--scaler_file", type=str, default="scaler_eform.pkl")
parser.add_argument("--out_dir", type=str, default="runs_dimenetpp_eform_v3_fastfood")
parser.add_argument("--train_mode", type=str, default="wrapped", choices=["base", "wrapped"])
parser.add_argument("--write_weights", action="store_true", help="Write Keras weights at the end. Disabled by default to save disk.")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

split_seed_label = args.split_seed if args.split_seed is not None else "cache"
exp_tag = (
    f"impl=wrapper_v3"
    f"_method={args.method}"
    f"_dim={int(round(args.id_dim * 100))}pct"
    f"_epochs={args.epochs}"
    f"_seed={args.seed}"
    f"_splitseed={split_seed_label}"
    f"_ortho={int(args.orthonormal)}"
    f"_rot={int(args.full_rotation)}"
    f"_blockcols={args.dense_block_cols if args.method == 'dense' else 0}"
    f"_orthobackend={args.orthonormal_backend if args.orthonormal else 'none'}"
    f"_train_mode={args.train_mode}"
    f"_task=eform"
)


import tensorflow as tf
from kgcnn.literature.DimeNetPP import make_crystal_model


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


hyper_eform_alignnish = {
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
                    "output_dim": 55,
                    "embeddings_initializer": {
                        "class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772},
                    },
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
        },
    },
    "training": {"fit": {"batch_size": args.batch_size, "epochs": args.epochs, "verbose": 2}},
}

cfg = hyper_eform_alignnish["model"]["config"]
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
hyper_eform_alignnish["model"]["config"] = cfg

cache_path = os.path.join(args.cache_dir, args.cache_file)
scaler_path = os.path.join(args.cache_dir, args.scaler_file)
print("Loading cached tensors:", cache_path)
# Pin the cached tensors to host memory. The pickle holds TF tensors, which would otherwise
# deserialize onto the default device (GPU) and OOM the card with the whole dataset. They must
# live in CPU RAM; Keras streams per-batch to the GPU during fit/predict.
with tf.device("/CPU:0"):
    with open(cache_path, "rb") as fcache:
        cache = pickle.load(fcache)
    x_train = cache["x_train"]
    x_val = cache["x_val"]
    x_test = cache["x_test"]
    y_train = cache["y_train"]
    y_train_scaled = cache["y_train_scaled"]
    y_val = cache["y_val"]
    y_val_scaled = cache["y_val_scaled"]
    y_test = cache["y_test"]
    test_ids_clean = cache["test_ids"]

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

print("Loaded cached tensors from:", args.cache_dir)
print("Shapes:", "y_train", y_train.shape, "y_val", y_val.shape, "y_test", y_test.shape)

base_model = make_crystal_model(**hyper_eform_alignnish["model"]["config"])

if args.train_mode == "base":
    run_model = base_model
    print("\nRunning BASE DimeNet++ eform model\n")
else:
    from wrapper_tensorflow_v3 import SubspaceProjectedGradTFV3

    run_model = SubspaceProjectedGradTFV3(
        base_model=base_model,
        d=args.id_dim,
        method=args.method,
        seed=args.seed,
        orthonormal=args.orthonormal,
        full_rotation=args.full_rotation,
        dense_block_cols=args.dense_block_cols,
        orthonormal_backend=args.orthonormal_backend,
        torch_python=args.torch_python,
        torch_q_cache_dir=args.torch_q_cache_dir,
        torch_q_gpu=args.torch_q_gpu,
        name=f"SubspaceProjectedGradTFV3_{args.method}",
    )
    print("\nRunning WRAPPER-V3 DimeNet++ eform model\n")
    print("Projection method:", args.method)
    print("Intrinsic dimension spec:", args.id_dim)
    print("D:", run_model.D, "d:", run_model.d)
    print("Trainable vars:", [v.name for v in run_model.trainable_variables])

run_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0, amsgrad=True),
    loss=tf.keras.losses.MeanSquaredError(),
    jit_compile=False,
)

run_model.summary()

hist = run_model.fit(
    x_train,
    y_train_scaled,
    validation_data=(x_val, y_val_scaled),
    batch_size=args.batch_size,
    epochs=args.epochs,
    verbose=2,
    validation_freq=1,
)

pred_test_scaled = run_model.predict(x_test, verbose=0)
pred_test = scaler.inverse_transform(np.asarray(pred_test_scaled).reshape(-1, 1)).reshape(-1)
y_test_true = np.asarray(y_test).reshape(-1)
test_mae = float(np.mean(np.abs(pred_test - y_test_true)))
print("Test MAE:", test_mae)

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

hist_path = os.path.join(args.out_dir, f"history_{exp_tag}.json")
with open(hist_path, "w") as f:
    json.dump(hist.history, f)
print("Wrote:", hist_path)

if args.write_weights:
    weights_path = os.path.join(args.out_dir, f"weights_{exp_tag}.weights.h5")
    run_model.save_weights(weights_path)
    print("Wrote:", weights_path)

if args.train_mode == "wrapped":
    z_vars = [v for v in run_model.trainable_variables if v.name.endswith("/z:0") or v.name == "z" or v.name.endswith("z")]
    if len(z_vars) == 1:
        z_path = os.path.join(args.out_dir, f"z_{exp_tag}.npy")
        np.save(z_path, z_vars[0].numpy())
        print("Wrote:", z_path)

t1 = time.time()
train_seconds = t1 - t0
print(f"Training wall time: {train_seconds:.2f} s ({train_seconds / 60:.2f} min)")
