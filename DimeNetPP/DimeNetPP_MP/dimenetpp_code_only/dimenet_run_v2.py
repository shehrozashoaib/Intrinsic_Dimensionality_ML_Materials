import os
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from kgcnn.literature.DimeNetPP import make_crystal_model


t0 = time.time()

parser = argparse.ArgumentParser("DimeNet++ wrapper-v2 runner")
parser.add_argument("--epochs", type=int, default=350, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--method", type=str, default="fastfood", choices=["dense", "fastfood"], help="Wrapper method")
parser.add_argument("--id_dim", type=float, default=1.0, help="Intrinsic dimension fraction in (0,1] or integer-like value")
parser.add_argument("--orthonormal", action="store_true", help="Orthonormalize dense P via QR")
parser.add_argument("--full_rotation", action="store_true", help="Use full rotation mode when d==D")
parser.add_argument("--dense_block_cols", type=int, default=512, help="Column block size for exact dense Gaussian projection storage")
parser.add_argument("--orthonormal_backend", type=str, default="tensorflow", choices=["tensorflow", "pytorch"], help="Backend for exact dense orthonormal projection")
parser.add_argument("--torch_python", type=str, default="/venv/main/bin/python", help="Python executable used for PyTorch orthonormal QR generation")
parser.add_argument("--torch_q_cache_dir", type=str, default="./orthonormal_q_cache", help="Cache directory for PyTorch-generated orthonormal Q matrices")
parser.add_argument("--torch_q_gpu", type=int, default=0, help="GPU index used by the external PyTorch QR generator")
parser.add_argument("--seed", type=int, default=123, help="Random seed for wrapper")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
parser.add_argument("--cache_dir", type=str, default="./cached_tensors_dimenetpp", help="Cache dir for tensors/scaler")
parser.add_argument("--cache_file", type=str, default="dimenetpp_cached_tensors.npz", help="Tensor cache filename")
parser.add_argument("--scaler_file", type=str, default="scaler.pkl", help="Scaler filename")
parser.add_argument("--out_dir", type=str, default="runs_dimenetpp_v2", help="Output directory for CSV/history")
parser.add_argument("--train_mode", type=str, default="wrapped", choices=["base", "wrapped"], help="Train base model or wrapper-v2 model")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

exp_tag = (
    f"impl=wrapper_v2"
    f"_method={args.method}"
    f"_dim={int(round(args.id_dim * 100))}pct"
    f"_epochs={args.epochs}"
    f"_seed={args.seed}"
    f"_ortho={int(args.orthonormal)}"
    f"_rot={int(args.full_rotation)}"
    f"_blockcols={args.dense_block_cols if args.method == 'dense' else 0}"
    f"_orthobackend={args.orthonormal_backend if args.orthonormal else 'none'}"
    f"_train_mode={args.train_mode}"
    f"_task=band_gap"
)


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
    "training": {
        "fit": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "verbose": 2,
        }
    },
}

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

cache = np.load(os.path.join(args.cache_dir, args.cache_file), allow_pickle=True)
x_train = cache["x_train"].tolist()
x_val = cache["x_val"].tolist()
x_test = cache["x_test"].tolist()
y_train = cache["y_train"]
y_train_scaled = cache["y_train_scaled"]
y_val = cache["y_val"]
y_val_scaled = cache["y_val_scaled"]
y_test = cache["y_test"]
test_ids_clean = cache["test_ids"]

with open(os.path.join(args.cache_dir, args.scaler_file), "rb") as f:
    scaler = pickle.load(f)

print("Loaded cached tensors from:", args.cache_dir)
print("Shapes:", "y_train", y_train.shape, "y_val", y_val.shape, "y_test", y_test.shape)

base_model = make_crystal_model(**hyper_1_alignnish["model"]["config"])

if args.train_mode == "base":
    run_model = base_model
    print("\nRunning BASE DimeNet++ model\n")
else:
    from wrapper_tensorflow_v2 import SubspaceProjectedGradTFV2

    run_model = SubspaceProjectedGradTFV2(
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
        name=f"SubspaceProjectedGradTFV2_{args.method}",
    )
    print("\nRunning WRAPPER-V2 DimeNet++ model\n")
    print("Projection method:", args.method)
    print("Intrinsic dimension spec:", args.id_dim)
    if args.method == "dense" and not args.orthonormal and not args.full_rotation:
        print("Dense block columns:", args.dense_block_cols)
    if args.method == "dense" and args.orthonormal:
        print("Orthonormal backend:", args.orthonormal_backend)
        if args.orthonormal_backend == "pytorch":
            print("PyTorch QR python:", args.torch_python)
            print("PyTorch QR cache dir:", args.torch_q_cache_dir)
            print("PyTorch QR GPU:", args.torch_q_gpu)
    print("Trainable vars:", [v.name for v in run_model.trainable_variables])

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

if args.train_mode == "wrapped":
    print("Initialization check skipped in v2 runner to avoid Keras ragged/XLA predict issues on this stack.")

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
pred_test_scaled_2d = np.asarray(pred_test_scaled).reshape(-1, 1)
pred_test = scaler.inverse_transform(pred_test_scaled_2d).reshape(-1)

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
