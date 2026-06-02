import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover - older PyTorch fallback
    from torch.nn.utils.stateless import functional_call


DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def _num_bytes(dtype):
    return DTYPE_BYTES.get(dtype, 4)


def _pretty(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n_bytes)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _fwht(x):
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be a power of two, got {n}")

    h = 1
    while h < n:
        x = x.reshape(x.shape[:-1] + (n // (2 * h), 2, h))
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2)
        x = x.reshape(x.shape[:-3] + (n,))
        h *= 2
    return x


def _sample_chi(n, device, dtype):
    df = torch.tensor(float(n), device=device)
    return torch.distributions.Chi2(df).sample((n,)).sqrt().to(dtype)


def _fastfood_apply(z_padded, B, Pi, G, S):
    x = z_padded * B
    x = _fwht(x)
    x = x[Pi]
    x = x * G
    x = _fwht(x)
    return x * S


def resolve_subspace_dim(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    return value


class RandomSubspaceWrapper(nn.Module):
    """
    Optimize a low-dimensional vector z while evaluating the wrapped model at
    theta = theta0 + P z. Projection P can be dense Gaussian or implicit
    Fastfood.
    """

    def __init__(
        self,
        base_model,
        d,
        method="dense",
        orthonormal=False,
        full_rotation=False,
        seed=None,
        z_init_std=0.0,
        device=None,
    ):
        super().__init__()
        if method not in ("dense", "fastfood"):
            raise ValueError(f"Unknown subspace method {method!r}")

        self.method = method
        self.model = base_model
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._param_names = []
        self._param_shapes = []
        flats = []
        for name, param in self.model.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(param.shape)
            flats.append(param.detach().reshape(-1))
        if not flats:
            raise ValueError("Wrapped model has no parameters.")

        if device is None:
            device = flats[0].device
        theta0 = torch.cat(flats, dim=0).to(device=device, dtype=flats[0].dtype)
        self.register_buffer("theta0", theta0)

        self._buffer_names = []
        for name, buf in self.model.named_buffers():
            self._buffer_names.append(name)
            self.register_buffer(
                f"_buf__{name.replace('.', '__')}",
                buf.detach().clone().to(device=device),
                persistent=False,
            )

        D = theta0.numel()
        if isinstance(d, float):
            d = int(round(d * D))
        if d is None:
            d = D
        if not 1 <= d <= D:
            raise ValueError(f"d must be in [1, D], got d={d} for D={D}")

        self.D = D
        self.d = int(d)
        self._rotation_mode = "none"

        rng_state = None
        cuda_rng_state = None
        if seed is not None:
            rng_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state_all()
            torch.manual_seed(int(seed))

        try:
            if method == "dense":
                self._build_dense_projection(
                    theta0, self.D, self.d, orthonormal, full_rotation, device
                )
            else:
                if orthonormal:
                    print("[Subspace][Fastfood] --id-ortho is ignored for fastfood.")
                if full_rotation:
                    print("[Subspace][Fastfood] --subspace-full-rotation is ignored for fastfood.")
                self._build_fastfood_projection(theta0, self.D, self.d, device)

            if z_init_std and float(z_init_std) > 0:
                z = torch.randn(self.d, device=device, dtype=theta0.dtype) * float(z_init_std)
            else:
                z = torch.zeros(self.d, device=device, dtype=theta0.dtype)
        finally:
            if rng_state is not None:
                torch.random.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        self.z = nn.Parameter(z)

        self._slices = []
        offset = 0
        for shape in self._param_shapes:
            n_elem = int(torch.tensor(shape).prod().item())
            self._slices.append(slice(offset, offset + n_elem))
            offset += n_elem

    def _build_dense_projection(self, theta0, D, d, orthonormal, full_rotation, device):
        if d == D:
            if full_rotation:
                perm = torch.randperm(D, device=device)
                sign = torch.randint(0, 2, (D,), device=device, dtype=torch.int8)
                sign = (sign * 2 - 1).to(theta0.dtype)
                self.register_buffer("_rot_perm", perm)
                self.register_buffer("_rot_sign", sign)
                self._rotation_mode = "permute_sign"
                print("[Subspace][Dense] full dimension with permutation/sign rotation.")
            else:
                self._rotation_mode = "identity"
                print("[Subspace][Dense] full dimension with identity projection.")
            return

        elem_bytes = _num_bytes(theta0.dtype)
        print(
            f"[Subspace][Dense] D={D:,}, d={d:,}, "
            f"P memory ~{_pretty(D * d * elem_bytes)}."
        )
        A = torch.randn(D, d, device=device, dtype=theta0.dtype)
        if orthonormal:
            print("[Subspace][Dense] QR orthonormalizing projection columns.")
            P, _ = torch.linalg.qr(A, mode="reduced")
        else:
            P = A / (A.norm(dim=0, keepdim=True) + 1e-12)
        self.register_buffer("P", P)

    def _build_fastfood_projection(self, theta0, D, d, device):
        n_pad = _next_pow2(max(D, d))
        self._ff_n = n_pad
        elem_bytes = _num_bytes(theta0.dtype)
        print(
            f"[Subspace][Fastfood] D={D:,}, d={d:,}, padded n={n_pad:,}, "
            f"buffers ~{_pretty(3 * n_pad * elem_bytes + n_pad * 8)}."
        )

        B = (torch.randint(0, 2, (n_pad,), device=device) * 2 - 1).to(theta0.dtype)
        Pi = torch.randperm(n_pad, device=device)
        G = torch.randn(n_pad, device=device, dtype=theta0.dtype)
        S = _sample_chi(n_pad, device, theta0.dtype) / (G.norm() + 1e-12)

        self.register_buffer("_ff_B", B)
        self.register_buffer("_ff_Pi", Pi)
        self.register_buffer("_ff_G", G)
        self.register_buffer("_ff_S", S)

        with torch.no_grad():
            e0 = torch.zeros(n_pad, device=device, dtype=theta0.dtype)
            e0[0] = 1.0
            col0 = _fastfood_apply(e0, B, Pi, G, S)[:D]
            scale = 1.0 / (col0.norm() + 1e-12)
        self.register_buffer("_ff_scale", scale)

    def _theta_full(self):
        if self.method == "dense":
            if self._rotation_mode == "identity":
                return self.theta0 + self.z
            if self._rotation_mode == "permute_sign":
                return self.theta0 + self.z[self._rot_perm] * self._rot_sign
            return self.theta0 + self.P.matmul(self.z)

        n = self._ff_n
        z_padded = F.pad(self.z, (0, n - self.d)) if self.d < n else self.z
        projected = _fastfood_apply(
            z_padded, self._ff_B, self._ff_Pi, self._ff_G, self._ff_S
        )
        return self.theta0 + projected[:self.D] * self._ff_scale

    def _unflatten_to_paramdict(self, theta):
        out = {}
        for name, shape, slc in zip(self._param_names, self._param_shapes, self._slices):
            out[name] = theta[slc].view(shape)
        return out

    def _buffers_dict(self):
        out = {}
        for name in self._buffer_names:
            out[name] = getattr(self, f"_buf__{name.replace('.', '__')}")
        return out

    def forward(self, *args, **kwargs):
        theta = self._theta_full()
        params_and_buffers = {
            **self._unflatten_to_paramdict(theta),
            **self._buffers_dict(),
        }
        return functional_call(self.model, params_and_buffers, args, kwargs)
