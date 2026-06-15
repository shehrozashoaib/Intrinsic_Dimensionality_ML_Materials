"""Module to train for a folder with formatted dataset."""
import os
import torch.distributed as dist
import csv
import sys
import json
import zipfile
from data import get_train_val_loaders
from train import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
import numpy as np
import time
from jarvis.core.atoms import Atoms
import random
from ase.stress import voigt_6_to_full_3x3_stress
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.func import functional_call as _functional_call
import os
from pathlib import Path


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def set_global_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    # "12356"
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        # os.environ["MASTER_PORT"] = "12355"
        # Initialize the distributed environment.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

# parser.add_argument(
#    "--keep_data_order",
#    default=True,
#    help="Whether to randomly shuffle samples",
# )

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--random_seed",
    default=None,
    type=int,
    help="Seed for model initialization, torch/numpy/python RNG, and subspace projection.",
)

parser.add_argument(
    "--split_seed",
    default=None,
    type=int,
    help="Seed for train/val/test split. Defaults to --random_seed, then config random_seed.",
)

# --- Intrinsic-dimension / subspace wrapper args ---
parser.add_argument(
    "--subspace_method",
    default="none",
    choices=["none", "dense", "subspace_rotation", "fastfood"],
    help="Random-subspace projection: none | dense (Gaussian)."
)

parser.add_argument(
    "--id_dim",
    default=None,
    help="Subspace size: integer d or a float fraction in (0,1] meaning d = frac * D."
)

parser.add_argument(
    "--id_ortho",
    action="store_true",
    default = False,
    help="If set, make columns of P exactly orthonormal via QR (costly)."
)

parser.add_argument(
    "--subspace_full_rotation",
    action="store_true",
    help="If d == D: use a random orthonormal rotation (P=Q) instead of identity."
)



parser.add_argument(
    "--target_key",
    default="total_energy",
    help="Name of the key for graph level data such as total_energy",
)

parser.add_argument(
    "--id_key",
    default="jid",
    help="Name of the key for graph level id such as id",
)

parser.add_argument(
    "--force_key",
    default="forces",
    help="Name of key for gradient level data such as forces, (Natoms x p)",
)

parser.add_argument(
    "--atomwise_key",
    default="forces",
    help="Name of key for atomwise level data: forces, charges (Natoms x p)",
)


parser.add_argument(
    "--stresswise_key",
    default="stresses",
    help="Name of the key for stress (3x3) level data such as forces",
)

parser.add_argument(
    "--additional_output_key",
    default="additional_output",
    help="Name of the key for extra global output eg DOS",
)


parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)


parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)

parser.add_argument(
    "--id_enable",
    default=False,
    help="select if you want to use the ID wrapper",
)


import torch
import torch.nn as nn


import torch
import torch.nn as nn

try:
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call




import math
import torch
import torch.nn as nn

# ----- helpers -----

DTYPE_BYTES = {
    torch.float64: 8, torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    torch.int64: 8,   torch.int32: 4,   torch.int16: 2,   torch.int8: 1,
    torch.bool: 1,
}
"""
RandomSubspaceWrapper with both 'dense' and 'fastfood' projection methods.

Implements Li et al. (ICLR 2018), "Measuring the Intrinsic Dimension of
Objective Landscapes":   theta^(D) = theta_0^(D) + P * theta^(d)

- "dense":    P is an explicit D x d Gaussian matrix (with optional QR
              orthonormalization, plus a memory-safe permutation+sign rotation
              when d == D and full_rotation=True). Memory O(D*d).
- "fastfood": P is implicit; we apply M = S * H * G * Pi * H * B via two
              Fast Walsh-Hadamard Transforms (Le et al. 2013).
              Memory O(D), compute O(D log D) per forward.
              No orthonormal / no full_rotation knobs — supports d from
              ~1% to 100% of D.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call as _functional_call


# ============================================================================
# Memory accounting helpers (unchanged)
# ============================================================================

DTYPE_BYTES = {
    torch.float32: 4, torch.float: 4,
    torch.float64: 8, torch.double: 8,
    torch.float16: 2, torch.half: 2,
    torch.bfloat16: 2,
    torch.int64: 8, torch.long: 8,
    torch.int32: 4, torch.int: 4,
    torch.int16: 2, torch.short: 2,
    torch.int8: 1, torch.uint8: 1,
    torch.bool: 1,
}


def _num_bytes(t: torch.dtype) -> int:
    return DTYPE_BYTES.get(t, 4)  # default 4 bytes if unknown


def _pretty(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n_bytes)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def count_params_and_buffers(module: nn.Module):
    param_elems = 0
    param_bytes = 0
    buf_elems = 0
    buf_bytes = 0
    for p in module.parameters():
        param_elems += p.numel()
        param_bytes += p.numel() * _num_bytes(p.dtype)
    for b in module.buffers():
        buf_elems += b.numel()
        buf_bytes += b.numel() * _num_bytes(b.dtype)
    return (param_elems, param_bytes, buf_elems, buf_bytes)


def estimate_optimizer_bytes(module: nn.Module, optim_name: str = "AdamW"):
    """
    Very rough optimizer-state estimate for common optimizers.
    Adam/AdamW: 2 extra tensors per param (m and v) -> ~2x params
    SGD (no momentum): ~0x; with momentum: ~1x
    """
    nelem = sum(p.numel() for p in module.parameters() if p.requires_grad)
    state_bytes_per_elem = 0
    if optim_name.lower() in ("adam", "adamw"):
        state_bytes_per_elem = 2 * 4  # m and v in FP32
    elif optim_name.lower() in ("sgd",):
        state_bytes_per_elem = 0
    return nelem * state_bytes_per_elem


def estimate_activation_bytes(example_forward_bytes: int, grad: bool = True):
    return example_forward_bytes * (2 if grad else 1)


def report_memory(model: nn.Module,
                  optimizer: str = "AdamW",
                  note: str = "",
                  also_check_cuda: bool = True):
    pe, pb, be, bb = count_params_and_buffers(model)
    optb = estimate_optimizer_bytes(model, optim_name=optimizer)

    total_bytes = pb + bb + optb
    print("=" * 64)
    if note:
        print(f"[MEM REPORT] {note}")
    print(f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"All params       : {pe:,}  ({_pretty(pb)})")
    print(f"Buffers          : {be:,}  ({_pretty(bb)})")
    print(f"Optimizer({optimizer}) ~ {_pretty(optb)}")
    print("-" * 64)
    print(f"Estimated total (params + buffers + opt): {_pretty(total_bytes)}")

    if also_check_cuda and torch.cuda.is_available():
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda:0")
        torch.cuda.synchronize(device=dev)
        allocated = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        print("-" * 64)
        print(f"CUDA live: allocated={_pretty(allocated)}, reserved={_pretty(reserved)}, device={dev}")
    print("=" * 64)


# ============================================================================
# Fastfood helpers
# ============================================================================

def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized Fast Walsh-Hadamard Transform along the last dimension.
    Length must be a power of 2. Differentiable.

    Runs in O(n log n). Note: returns the *unnormalized* WHT, so applying it
    twice scales by n (which the Fastfood scaling absorbs).
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"FWHT length must be a power of 2, got {n}"
    h = 1
    while h < n:
        new_shape = x.shape[:-1] + (n // (2 * h), 2, h)
        x = x.reshape(new_shape)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(x.shape[:-3] + (n,))
        h *= 2
    return x


def _fastfood_apply(z_padded: torch.Tensor,
                    B: torch.Tensor,
                    Pi: torch.Tensor,
                    G: torch.Tensor,
                    S: torch.Tensor) -> torch.Tensor:
    """
    Apply M = S * H * G * Pi * H * B to a length-n padded vector.

    Per Le, Sarlos & Smola (2013), this yields an n x n matrix whose
    second-moment statistics approximate a dense Gaussian random matrix,
    using only O(n) parameters and O(n log n) compute.
    """
    x = z_padded * B          # diagonal +/-1
    x = _fwht(x)              # H
    x = x[Pi]                 # Pi (random permutation)
    x = x * G                 # diagonal Gaussian
    x = _fwht(x)              # H
    x = x * S                 # diagonal chi-scaling
    return x


def _sample_chi(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Sample n iid chi_n random variables (i.e. sqrt of chi-squared with n d.o.f.).
    Uses Chi2 distribution so memory stays O(n), not O(n^2).
    """
    df = torch.tensor(float(n), device=device)
    chi2 = torch.distributions.Chi2(df).sample((n,))
    return chi2.sqrt().to(dtype)


# ============================================================================
# RandomSubspaceWrapper
# ============================================================================

class RandomSubspaceWrapper(nn.Module):
    """
    Wraps a base nn.Module so only a d-dim parameter vector z is optimized and
    full parameters are constructed as theta = theta0 + P @ z on each forward.

    Args:
      base_model:    the nn.Module whose parameters define theta0 (frozen).
      d:             intrinsic dimension. int (absolute) or float in (0,1] for fraction of D.
      method:        "dense" | "fastfood".
      orthonormal:   (dense only) QR-orthonormalize columns of P.
      full_rotation: (dense, d==D only) use memory-safe permutation+sign rotation.
      device:        device on which to place buffers.

    For fastfood, `orthonormal` and `full_rotation` are ignored with a warning.
    Fastfood supports any d in [1, D] (the paper sweeps from ~1% to ~100%).
    """

    def __init__(
        self,
        base_model: nn.Module,
        d,
        method: str = "dense",
        orthonormal: bool = False,
        full_rotation: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert method in ("dense", "fastfood"), \
            f"Unknown method: {method!r}. Use 'dense' or 'fastfood'."
        self.method = method

        # DO NOT force eval here; let .train()/.eval() from the trainer propagate.
        self.model = base_model
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Flatten parameters, remember shapes and names
        self._param_names, self._param_shapes, flats = [], [], []
        for name, p in self.model.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(p.shape)
            flats.append(p.detach().reshape(-1))

        theta0 = torch.cat(flats, dim=0).to(device=device, dtype=flats[0].dtype)
        self.register_buffer("theta0", theta0)

        # Keep a snapshot of buffers so functional_call can use them statelessly.
        self._buffer_names = []
        for bname, buf in self.model.named_buffers():
            self._buffer_names.append(bname)
            self.register_buffer(
                f"_buf__{bname.replace('.', '__')}",
                buf.detach(),
                persistent=False,
            )

        D = theta0.numel()
        if isinstance(d, float):
            d = int(round(d * D))
        assert 1 <= d <= D, f"d must be in [1, D], got {d} for D={D}"
        self.D, self.d = D, d

        # default; dense path may override to "permute_sign"
        self._rotation_mode = "none"

        # --------------------------------------------------------------------
        # Build the projection
        # --------------------------------------------------------------------
        if method == "dense":
            self._build_dense_projection(theta0, D, d, orthonormal, full_rotation, device)
        else:  # method == "fastfood"
            if orthonormal:
                print("[Subspace][Fastfood] 'orthonormal=True' is ignored for fastfood (not applicable).")
            if full_rotation:
                print("[Subspace][Fastfood] 'full_rotation=True' is ignored for fastfood.")
            self._build_fastfood_projection(theta0, D, d, device)

        # Trainable intrinsic vector (initialized to zero -> start at theta0)
        self.z = nn.Parameter(torch.zeros(d, device=device, dtype=theta0.dtype))

        # Slices to unflatten the full theta back into per-parameter tensors
        self._slices = []
        off = 0
        for shp in self._param_shapes:
            n_elem = int(torch.tensor(shp).prod().item())
            self._slices.append(slice(off, off + n_elem))
            off += n_elem

    # ------------------------------------------------------------------------
    # Projection constructors
    # ------------------------------------------------------------------------

    def _build_dense_projection(self, theta0, D, d, orthonormal, full_rotation, device):
        """Identical to the original dense logic."""
        if d == D:
            if full_rotation:
                elem_bytes = _num_bytes(theta0.dtype)
                est_A_bytes = D * D * elem_bytes
                est_qr_bytes = est_A_bytes * 6
                print(
                    f"[Subspace][Preflight] Full rotation requested with D={D:,}.\n"
                    f"  Dense QR would allocate about {_pretty(est_qr_bytes)} (A ~ {_pretty(est_A_bytes)}).\n"
                    f"  Using memory-safe orthogonal rotation via random permutation with sign flips instead."
                )
                perm = torch.randperm(D, device=device)
                sign = torch.randint(0, 2, (D,), device=device, dtype=torch.int8)
                sign = (sign * 2 - 1).to(theta0.dtype)
                self.register_buffer("_rot_perm", perm)
                self.register_buffer("_rot_sign", sign)
                self._rotation_mode = "permute_sign"
                P = torch.eye(D, device=device, dtype=theta0.dtype)
            else:
                # Consistent with d < D case: use random dense Gaussian projection
                A = torch.randn(D, d, device=device, dtype=theta0.dtype)
                A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
                P = A
        else:
            A = torch.randn(D, d, device=device, dtype=theta0.dtype)
            if orthonormal:
                elem_bytes = _num_bytes(theta0.dtype)
                est_A_bytes = D * d * elem_bytes
                est_qr_bytes = est_A_bytes * 6
                print(
                    f"[Subspace][Preflight] Orthonormalizing subspace with D={D:,}, d={d:,}.\n"
                    f"  Estimated QR memory ~ {_pretty(est_qr_bytes)}."
                )
                Q, _ = torch.linalg.qr(A, mode="reduced")
                P = Q
            else:
                A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
                P = A

        self.register_buffer("P", P)

    def _build_fastfood_projection(self, theta0, D, d, device):
        """
        Build the four Fastfood components (B, Pi, G, S) implicitly defining
        a D x d projection. Pad to n = next_pow2(max(D, d)) so FWHT works.
        """
        n_pad = _next_pow2(max(D, d))
        self._ff_n = n_pad

        elem_bytes = _num_bytes(theta0.dtype)
        # B + G + S are float diagonals (n_pad each); Pi is int64 (8 bytes)
        est_buf_bytes = 3 * n_pad * elem_bytes + n_pad * 8
        log2_n = int(math.log2(max(n_pad, 2)))
        print(
            f"[Subspace][Fastfood] D={D:,}, d={d:,}, padded n={n_pad:,}.\n"
            f"  Fastfood diagonals + permutation: ~{_pretty(est_buf_bytes)} "
            f"(vs. dense P would need ~{_pretty(D * d * elem_bytes)}).\n"
            f"  Compute per forward: O(n log n) ~ {n_pad * log2_n:,} ops."
        )

        # B: diagonal +/- 1
        B = (torch.randint(0, 2, (n_pad,), device=device) * 2 - 1).to(theta0.dtype)
        # Pi: random permutation
        Pi = torch.randperm(n_pad, device=device)
        # G: diagonal N(0, 1)
        G = torch.randn(n_pad, device=device, dtype=theta0.dtype)
        # S: chi_n / ||G||  (Le et al. 2013 scaling -> Gaussian-equivalent stats)
        chi_samples = _sample_chi(n_pad, device, theta0.dtype)
        S = chi_samples / (G.norm() + 1e-12)

        self.register_buffer("_ff_B", B)
        self.register_buffer("_ff_Pi", Pi)
        self.register_buffer("_ff_G", G)
        self.register_buffer("_ff_S", S)

        # Empirically estimate one column norm of the implicit D x d projection
        # so we can rescale to ~unit-norm columns (matching the dense default).
        with torch.no_grad():
            e0 = torch.zeros(n_pad, device=device, dtype=theta0.dtype)
            e0[0] = 1.0
            col0 = _fastfood_apply(e0, B, Pi, G, S)[:D]
            col_norm = float(col0.norm().item())
        self._ff_scale = 1.0 / (col_norm + 1e-12)

    # ------------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------------

    def _theta_full(self) -> torch.Tensor:
        if self.method == "dense":
            if self._rotation_mode == "permute_sign":
                rotated = self.z[self._rot_perm] * self._rot_sign
                return self.theta0 + rotated
            return self.theta0 + self.P.matmul(self.z)

        # fastfood
        n = self._ff_n
        # Zero-pad z to length n (autograd-safe via F.pad)
        if self.d < n:
            z_padded = F.pad(self.z, (0, n - self.d))
        else:
            z_padded = self.z
        projected = _fastfood_apply(
            z_padded, self._ff_B, self._ff_Pi, self._ff_G, self._ff_S
        )
        return self.theta0 + projected[:self.D] * self._ff_scale

    def _unflatten_to_paramdict(self, theta: torch.Tensor):
        out = {}
        for name, shp, sl in zip(self._param_names, self._param_shapes, self._slices):
            out[name] = theta[sl].view(shp)
        return out

    def _buffers_dict(self):
        bdict = {}
        for bname in self._buffer_names:
            key = f"_buf__{bname.replace('.', '__')}"
            bdict[bname] = getattr(self, key)
        return bdict

    def forward(self, *args, **kwargs):
        theta = self._theta_full()
        param_map = self._unflatten_to_paramdict(theta)
        buffers_map = self._buffers_dict()
        param_and_buffers = {**param_map, **buffers_map}
        return _functional_call(
            self.model,
            param_and_buffers,
            tuple(args),
            dict(kwargs or {}),
        )





def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    additional_output_key="additional_output",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
    # Intrinsic-dimension arguments
    subspace_method="none",
    id_dim=None,
    id_ortho=False,
    subspace_full_rotation=False,
    id_enable=False,
    random_seed=None,
    split_seed=None,
):
    """Train for a folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    id_prop_csv_file = False
    multioutput = False
    # lists_length_equal = True
    if os.path.exists(id_prop_json_zip):
        dat = json.loads(
            zipfile.ZipFile(id_prop_json_zip).read("id_prop.json")
        )
    elif os.path.exists(id_prop_json):
        dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")
    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    # config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    if random_seed is not None:
        config.random_seed = int(random_seed)
    if split_seed is None:
        split_seed = config.random_seed
    else:
        split_seed = int(split_seed)
    id_enable = _str_to_bool(id_enable)
    print(f"[Seed] random_seed={config.random_seed}, split_seed={split_seed}")
    set_global_seed(config.random_seed)

    train_grad = False
    train_stress = False
    train_additional_output = False
    train_atom = False
    try:
        if (
            config.model.calculate_gradient
            and config.model.gradwise_weight != 0
        ):
            train_grad = True
        else:
            train_grad = False
        if (
            config.model.calculate_gradient
            and config.model.stresswise_weight != 0
        ):
            train_stress = True
        else:
            train_stress = False
        if config.model.atomwise_weight != 0:
            train_atom = True
        else:
            train_atom = False
        if (
            config.model.additional_output_features > 0
            and config.model.additional_output_weight != 0
        ):
            train_additional_output = True
        else:
            train_additional_output = False
    except Exception as exp:
        print("exp", exp)
        pass
    # if config.model.atomwise_weight == 0:
    #    train_atom = False
    # if config.model.gradwise_weight == 0:
    #    train_grad = False
    # if config.model.stresswise_weight == 0:
    #    train_stress = False
    target_atomwise = None  # "atomwise_target"
    target_grad = None  # "atomwise_grad"
    target_stress = None  # "stresses"
    target_additional_output = None  # "stresses"

    # mem = []
    # enp = []
    n_outputs = []
    dataset = []
    for i in dat:
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]  # float(i[1])
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = tmp
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                # Note using 500 angstrom as box size
                # Recommended install pytraj
                # conda install -c ambermd pytraj
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            info["atoms"] = atoms.to_dict()
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]
        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]  # such as charges
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]  # - mean_force
        if train_stress:
            if len(i[stresswise_key]) == 6:

                stress = voigt_6_to_full_3x3_stress(i[stresswise_key])
            else:
                stress = i[stresswise_key]
            info["stresses"] = stress  # - mean_force
            target_stress = "stresses"

        if train_additional_output:
            target_additional_output = "additional"
            info["additional"] = i[additional_output_key]  # - mean_force
        if "extra_features" in i:
            info["extra_features"] = i["extra_features"]
        dataset.append(info)
    print("len dataset", len(dataset))
    print("train_stress", train_stress)
    del dat
    # multioutput = False
    lists_length_equal = True
    line_graph = False
    # alignn_models = {
    #    # "alignn",
    #    # "alignn_layernorm",
    #    "alignn_atomwise",
    # }

    if config.compute_line_graph > 0:
        # if config.model.alignn_layers > 0:
        line_graph = True

    if multioutput:
        print("multioutput", multioutput)
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]
        print("lists_length_equal", lists_length_equal, len(n_outputs[0]))
        if lists_length_equal:
            config.model.output_features = len(n_outputs[0])

        else:
            raise ValueError("Make sure the outputs are of same size.")
    model = None
    def build_base_model_from_cfg(cfg):
        # cfg is a pydantic object (TrainingConfig.model)
        if cfg.name == "alignn_atomwise":
            return ALIGNNAtomWise(ALIGNNAtomWiseConfig(**cfg.model_dump()))
        raise NotImplementedError(f"Unsupported model: {cfg.name}")

    # Helper to compute d from CLI flags you added
    def resolve_d_for_wrapper(d_value):
        if d_value is None:
            return 1.0  # full space by default
        # Accept int or float fraction
        try:
            if isinstance(d_value, str):
                if "." in d_value or "e" in d_value.lower():
                    return float(d_value)
                return int(d_value)
            return int(d_value)
        except Exception:
            return float(d_value)

    if restart_model_path is not None:
        print("Restarting the model training:", restart_model_path)

        print("restarting form this file")

        # 1) Rebuild the SAME base model as the checkpoint
        #    (read the saved config that sits next to the checkpoint)
        rest_cfg_json = restart_model_path.replace("best_model.pt", "config.json")
        # If you save as best_model.pt, use .replace("best_model.pt","config.json") instead.
        rest_config = loadjson(rest_cfg_json)
        base_cfg = ALIGNNAtomWiseConfig(**rest_config["model"])
        base_model = ALIGNNAtomWise(base_cfg)

        # 2) Load the checkpoint weights into the BASE model first
        state = torch.load(restart_model_path, map_location="cpu")
        missing, unexpected = base_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[checkpoint->base] missing:", missing)
            print("[checkpoint->base] unexpected:", unexpected)

        # 3) Optionally wrap WITH intrinsic-dimension after the base is populated,
        #    so wrapper.theta0 is built from the loaded weights.
        if id_enable:
            d_val = resolve_d_for_wrapper(id_dim)
            model = RandomSubspaceWrapper(
                base_model,
                d=d_val,
                method=subspace_method,
                orthonormal=bool(id_ortho),
                full_rotation=bool(subspace_full_rotation),
                device=device,
            )
            # If the checkpoint was from a PREVIOUS wrapped run, try loading z/P/theta0 too:
            miss2, unexp2 = model.load_state_dict(state, strict=False)
            if miss2 or unexp2:
                print("[checkpoint->wrapper] missing:", miss2)
                print("[checkpoint->wrapper] unexpected:", unexp2)
        else:
            model = base_model

        model = model.to(device)

    else:
        # Fresh init

        print("starting from this file")
        base_model = build_base_model_from_cfg(config.model)

        if id_enable and (subspace_method != "none"):
            d_val = resolve_d_for_wrapper(id_dim)
            model = RandomSubspaceWrapper(
                base_model,
                d=d_val,
                method=subspace_method,
                orthonormal=bool(id_ortho),
                full_rotation=bool(subspace_full_rotation),
                device=device,
            )
        else:
            model = base_model
            print("base model used")

        model = model.to(device)
        if isinstance(model, RandomSubspaceWrapper):
            print(f"[Subspace] D={model.D:,}, d={model.d:,}")
            report_memory(model, optimizer="AdamW", note="Wrapped model (runtime footprint)")


        # print ('n_outputs',n_outputs[0])
        # if multioutput and classification_threshold is not None:
        #    raise ValueError("Classification for multi-output not implemented.")
        # if multioutput and lists_length_equal:
        #    config.model.output_features = len(n_outputs[0])
        # else:
        #    # TODO: Pad with NaN
    #    if not lists_length_equal:
    #        raise ValueError("Make sure the outputs are of same size.")
    #    else:
    #        config.model.output_features = 1
    # print('config.neighbor_strategy',config.neighbor_strategy)
    # import sys
    # sys.exit()
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        #target_additional_output=target_additional_output,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        split_seed=split_seed,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
        dtype=config.dtype,
    )
    # print("dataset", dataset[0])
    t1 = time.time()
    # world_size = torch.cuda.device_count()
    print("rank", rank)
    print("world_size", world_size)
    train_dgl(
        config,
        model=model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        rank=rank,
        world_size=world_size,
    )
    t2 = time.time()
    print("Time taken (s)", t2 - t1)


    # Assuming args is already defined in your script
    new_name = f"prediction_results_test_set_fulldataset_{args.target_key}_method:{args.subspace_method}_dim:{str(int(float(args.id_dim)*100))}_epochs:{args.epochs}.csv"

    # Define the file path
    old_file = Path("prediction_results_test_set.csv")

    # Check if the file exists before renaming to avoid errors
    if old_file.exists():
        old_file.rename(new_name)
        print(f"Successfully renamed to: {new_name}")
    else:
        print("Error: The source file 'prediction_results_test_set.csv' was not found.")

    new_name = f"history_val_mae_fulldataset_{args.target_key}_method:{args.subspace_method}_dim:{str(int(float(args.id_dim)*100))}_epochs:{args.epochs}.json"

    # Define the file path
    old_file = Path("history_val_mae.json")

    # Check if the file exists before renaming to avoid errors
    if old_file.exists():
        old_file.rename(new_name)
        print(f"Successfully renamed to: {new_name}")
    else:
        print("Error: The source file 'history_val_mae.json' was not found.")

    # train_data = get_torch_dataset(


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    print("world_size", world_size)
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.additional_output_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
                args.subspace_method,
                args.id_dim,
                args.id_ortho,
                args.subspace_full_rotation,
                args.id_enable,
                args.random_seed,
                args.split_seed,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.additional_output_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
            args.subspace_method,
            args.id_dim,
            args.id_ortho,
            args.subspace_full_rotation,
            args.id_enable,
            args.random_seed,
            args.split_seed,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass




#python train_alignn.py   --root_dir MP_json   --config_name config_eform.json   --target_key formation_energy_per_atom --id_key material_id   --subspace_method dense --id_dim 1.0   --epochs 1 --id_enable True