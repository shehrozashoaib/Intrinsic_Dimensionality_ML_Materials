"""VERIFY: does raising int_emb_size expand ONLY the convolutional (interaction) block?

Replicates the kvrh runner's EXACT config construction (including the
input_embedding -> input_node_embedding rename + tensor types), builds the model at
int_emb_size = 64 / 128 / 256, and diffs EVERY trainable weight by name.

A weight is "GREW" if its param count changed, "same" if identical.
"""
import numpy as np
import tensorflow as tf
from kgcnn.literature.DimeNetPP import make_crystal_model

INPUTS = [
    {"shape": [None], "name": "node_number", "dtype": "int32", "ragged": True},
    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
    {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
    {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
    {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
]


def runner_cfg(int_emb_size):
    """Exactly what dimenet_run_kvrh_v3.py builds (post-processing included)."""
    cfg = {
        "name": "DimeNetPP",
        "inputs": INPUTS,
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 55,
                     "embeddings_initializer": {"class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}}
        },
        "emb_size": 55, "out_emb_size": 64,
        "int_emb_size": int_emb_size,     # <-- the only thing we vary
        "basis_emb_size": 8,
        "num_blocks": 1,
        "num_spherical": 7, "num_radial": 6, "cutoff": 8.0, "envelope_exponent": 5,
        "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
        "num_targets": 1, "extensive": False, "output_init": "zeros",
        "activation": "swish", "verbose": 0, "output_embedding": "graph",
        "use_output_mlp": False, "output_mlp": {},
    }
    # --- the runner's post-processing (this is what my first script MISSED) ---
    cfg["input_tensor_type"] = "ragged"
    cfg["output_tensor_type"] = "padded"
    node_emb = cfg["input_embedding"].get("node", {})
    cfg.pop("input_embedding", None)
    cfg["input_node_embedding"] = {
        "input_dim": node_emb.get("input_dim", 95),
        "output_dim": node_emb.get("output_dim", 64),
        "embeddings_initializer": node_emb.get("embeddings_initializer", None),
    }
    return cfg


def weights_of(ies):
    m = make_crystal_model(**runner_cfg(ies))
    w = {}
    for v in m.trainable_weights:
        name = v.path if hasattr(v, "path") else v.name
        w[name] = (tuple(v.shape), int(np.prod(v.shape)))
    total = sum(c for _, c in w.values())
    tf.keras.backend.clear_session()
    return w, total


W = {}
T = {}
for ies in (64, 128, 256):
    W[ies], T[ies] = weights_of(ies)
    print(f"int_emb_size={ies:3d}: TOTAL trainable D = {T[ies]:,}  ({len(W[ies])} weight tensors)")

print("\n(sanity: the real runs print 'Trainable params: 83,685' for ies=64)\n")

names = sorted(set(W[64]) | set(W[128]) | set(W[256]))
grew, same, appeared = [], [], []
for n in names:
    c64 = W[64].get(n, (None, 0))[1]
    c128 = W[128].get(n, (None, 0))[1]
    c256 = W[256].get(n, (None, 0))[1]
    if n not in W[64]:
        appeared.append(n)
    elif c64 == c128 == c256:
        same.append((n, c64))
    else:
        grew.append((n, W[64][n][0], c64, W[128][n][0], c128, W[256][n][0], c256))

print("=" * 100)
print("WEIGHTS THAT GREW with int_emb_size (these should ALL be interaction/conv weights):")
print("=" * 100)
for n, s64, c64, s128, c128, s256, c256 in grew:
    print(f"  {n}")
    print(f"      64: {str(s64):18s} {c64:>7,}   128: {str(s128):18s} {c128:>7,}   256: {str(s256):18s} {c256:>7,}")
gs = sum(c for *_, c in [(g[0], g[2]) for g in grew])
print(f"\n  -> {len(grew)} weight tensors grew.")

print("\n" + "=" * 100)
print("WEIGHTS THAT ARE UNCHANGED (embedding / output / basis / etc.):")
print("=" * 100)
for n, c in same:
    print(f"  {n:70s} {c:>8,}")
print(f"\n  -> {len(same)} weight tensors unchanged, totalling {sum(c for _, c in same):,} params (identical at all 3 widths).")

if appeared:
    print("\n[!] weights that only exist at some widths:", appeared)

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
grew_64 = sum(g[2] for g in grew)
grew_256 = sum(g[6] for g in grew)
fixed = sum(c for _, c in same)
print(f"  fixed (non-conv) params : {fixed:,}  (identical across 64/128/256)")
print(f"  growing params          : {grew_64:,} (1x) -> {sum(g[4] for g in grew):,} (2x) -> {grew_256:,} (4x)")
print(f"  TOTAL D                 : {T[64]:,} -> {T[128]:,} -> {T[256]:,}")
conv_only = all(("interaction" in n.lower()) or ("_int" in n.lower()) for n, *_ in [(g[0],) for g in grew])
print(f"\n  ALL growing weights are inside the interaction/conv block? -> {conv_only}")
