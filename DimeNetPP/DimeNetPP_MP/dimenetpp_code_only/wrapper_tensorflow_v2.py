import gc
import os
import subprocess
import numpy as np
import tensorflow as tf


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def hadamard_1d_static(x: tf.Tensor, LL: int) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, [LL])
    x.set_shape([LL])

    h = 1
    while h < LL:
        x2 = tf.reshape(x, [-1, 2 * h])
        a = x2[:, :h]
        b = x2[:, h:2 * h]
        x2 = tf.concat([a + b, a - b], axis=1)
        x = tf.reshape(x2, [LL])
        x.set_shape([LL])
        h *= 2
    return x


def _make_fastfood_params(D: int, dtype=tf.float32, seed: int = 123):
    LL = _next_power_of_two(D)

    b01 = tf.random.stateless_uniform([LL], seed=[seed, 1], minval=0, maxval=2, dtype=tf.int32)
    B = tf.cast(b01 * 2 - 1, dtype)

    keys = tf.random.stateless_uniform([LL], seed=[seed, 2], dtype=tf.float32)
    Pi = tf.cast(tf.argsort(keys, axis=0, stable=True), tf.int32)
    Pi_inv = tf.cast(tf.argsort(Pi, axis=0, stable=True), tf.int32)

    G = tf.random.stateless_normal([LL], seed=[seed, 3], dtype=dtype)

    divisor = tf.sqrt(tf.cast(LL, dtype) * tf.reduce_sum(G * G))
    scale = divisor * tf.sqrt(tf.cast(D, dtype) / tf.cast(LL, dtype))
    return B, Pi, Pi_inv, G, scale, LL


def fastfood_forward(z, D: int, d: int, B, Pi, G, scale, LL: int):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z_pad = tf.pad(z, [[0, LL - d]])
    z_pad = tf.reshape(z_pad, [LL])
    z_pad.set_shape([LL])

    y = B * z_pad
    y = hadamard_1d_static(y, LL)
    y = tf.gather(y, Pi)
    y = G * y
    y = hadamard_1d_static(y, LL)
    return y[:D] / scale


def fastfood_transpose(g_theta, D: int, d: int, B, Pi_inv, G, scale, LL: int):
    g_theta = tf.convert_to_tensor(g_theta, dtype=tf.float32)
    g_full = tf.pad(g_theta, [[0, LL - D]])
    g_full = tf.reshape(g_full, [LL])
    g_full.set_shape([LL])

    g = g_full / scale
    g = hadamard_1d_static(g, LL)
    g = G * g
    g = tf.gather(g, Pi_inv)
    g = hadamard_1d_static(g, LL)
    g = B * g
    return g[:d]


def _build_orthonormal_q_with_torch(rows: int, cols: int, seed: int, torch_python: str, cache_dir: str, gpu: int):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"Q_rows{rows}_cols{cols}_seed{seed}_gpu{gpu}.npy")
    if not os.path.exists(out_path):
        script_path = os.path.join(os.path.dirname(__file__), "torch_orthonormal_q.py")
        cmd = [
            torch_python,
            script_path,
            "--rows", str(rows),
            "--cols", str(cols),
            "--seed", str(seed),
            "--gpu", str(gpu),
            "--out", out_path,
        ]
        print(f"[DenseV2] Building orthonormal Q with PyTorch: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print(f"[DenseV2] Reusing cached PyTorch orthonormal Q: {out_path}")
    return np.load(out_path)


class SubspaceProjectedGradTFV2(tf.keras.Model):
    """
    Wrapper-style intrinsic-dimension training.

    Dense mode remains an exact dense Gaussian projection, but can be stored blockwise to
    avoid the one-shot peak-memory blow-up from constructing a full [D, d] matrix twice.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        d,
        method: str = "dense",
        seed: int = 123,
        orthonormal: bool = False,
        full_rotation: bool = False,
        dense_block_cols: int = 512,
        orthonormal_backend: str = "tensorflow",
        torch_python: str = "/venv/main/bin/python",
        torch_q_cache_dir: str = "/workspace/dimenet++/orthonormal_q_cache",
        torch_q_gpu: int = 0,
        name: str = "SubspaceProjectedGradTFV2",
    ):
        super().__init__(name=name)
        assert method in ("dense", "fastfood")

        self.base_model = base_model
        self.base_model.trainable = True
        self.theta_vars = list(self.base_model.trainable_variables)
        self.shapes = [v.shape for v in self.theta_vars]
        self.sizes = [int(tf.size(v)) for v in self.theta_vars]
        self.D = int(sum(self.sizes))

        if isinstance(d, float):
            d = int(round(d * self.D))
        self.d = int(d)
        if not (1 <= self.d <= self.D):
            raise ValueError(f"d must be in [1, D]; got d={self.d}, D={self.D}")

        self.method = method
        self.seed = int(seed)
        self.orthonormal = bool(orthonormal)
        self.full_rotation = bool(full_rotation)
        self.dense_block_cols = max(1, int(dense_block_cols))
        self.orthonormal_backend = str(orthonormal_backend)
        self.torch_python = str(torch_python)
        self.torch_q_cache_dir = str(torch_q_cache_dir)
        self.torch_q_gpu = int(torch_q_gpu)
        if self.orthonormal_backend not in ("tensorflow", "pytorch"):
            raise ValueError(f"Unsupported orthonormal_backend={self.orthonormal_backend}")

        theta0_np = tf.concat([tf.reshape(tf.cast(v, tf.float32), [-1]) for v in self.theta_vars], axis=0).numpy()
        self.theta0 = self.add_weight(
            name="theta0",
            shape=(self.D,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(theta0_np),
            trainable=False,
        )
        del theta0_np
        gc.collect()
        self.z = self.add_weight(
            name="z",
            shape=(self.d,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self._rotation_mode = "none"
        self._rot_perm = None
        self._rot_sign = None
        self.P = None
        self.P_blocks = []
        self.P_block_ranges = []
        self._dense_use_blocks = False

        if self.method == "dense":
            if (self.d == self.D) and self.full_rotation:
                eye_np = tf.eye(self.D, dtype=tf.float32).numpy()
                self.P = self.add_weight(
                    name="P",
                    shape=(self.D, self.d),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(eye_np),
                    trainable=False,
                )
                del eye_np

                keys = tf.random.stateless_uniform([self.D], seed=[self.seed, 777], dtype=tf.float32)
                perm_np = tf.argsort(keys, axis=0, stable=True).numpy().astype(np.int32)
                signs = tf.random.stateless_uniform([self.D], seed=[self.seed, 778], minval=0, maxval=2, dtype=tf.int32)
                sign_np = tf.cast(signs * 2 - 1, tf.float32).numpy()
                del keys, signs
                gc.collect()

                self._rotation_mode = "permute_sign"
                self._rot_perm = self.add_weight(
                    name="rot_perm",
                    shape=(self.D,),
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(perm_np),
                    trainable=False,
                )
                self._rot_sign = self.add_weight(
                    name="rot_sign",
                    shape=(self.D,),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(sign_np),
                    trainable=False,
                )
            elif self.orthonormal:
                if self.orthonormal_backend == "pytorch":
                    Q_np = _build_orthonormal_q_with_torch(
                        rows=self.D,
                        cols=self.d,
                        seed=self.seed,
                        torch_python=self.torch_python,
                        cache_dir=self.torch_q_cache_dir,
                        gpu=self.torch_q_gpu,
                    )
                else:
                    # Exact global QR still needs the full dense matrix, so build the large temporaries on CPU
                    # and only materialize the final stored projection variable afterward.
                    with tf.device("/CPU:0"):
                        A = tf.random.stateless_normal([self.D, self.d], seed=[self.seed, 42], dtype=tf.float32)
                        Q, _ = tf.linalg.qr(A, full_matrices=False)
                        Q_np = Q.numpy()
                    del A, Q
                    gc.collect()
                self.P = self.add_weight(
                    name="P",
                    shape=(self.D, self.d),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(Q_np),
                    trainable=False,
                )
                del Q_np
                gc.collect()
            else:
                # Exact dense Gaussian projection, stored blockwise to control initialization peak memory.
                self._dense_use_blocks = True
                start = 0
                block_index = 0
                while start < self.d:
                    end = min(start + self.dense_block_cols, self.d)
                    width = end - start
                    block_seed = [self.seed, 42 + block_index]
                    A_block = tf.random.stateless_normal([self.D, width], seed=block_seed, dtype=tf.float32)
                    P_block = A_block / (tf.norm(A_block, axis=0, keepdims=True) + 1e-12)
                    P_block_np = P_block.numpy()
                    del A_block, P_block
                    gc.collect()
                    block_var = self.add_weight(
                        name=f"P_block_{block_index}",
                        shape=(self.D, width),
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(P_block_np),
                        trainable=False,
                    )
                    del P_block_np
                    gc.collect()
                    self.P_blocks.append(block_var)
                    self.P_block_ranges.append((start, end))
                    start = end
                    block_index += 1
                print(f"[DenseV2] Stored dense Gaussian projection in {len(self.P_blocks)} block(s) of up to {self.dense_block_cols} columns.")
        else:
            B, Pi, Pi_inv, G, scale, LL = _make_fastfood_params(self.D, dtype=tf.float32, seed=self.seed)
            self._ff_B = self.add_weight(
                name="ff_B",
                shape=(LL,),
                dtype=tf.float32,
                initializer=tf.constant_initializer(B.numpy()),
                trainable=False,
            )
            self._ff_Pi = self.add_weight(
                name="ff_Pi",
                shape=(LL,),
                dtype=tf.int32,
                initializer=tf.constant_initializer(Pi.numpy()),
                trainable=False,
            )
            self._ff_Pi_inv = self.add_weight(
                name="ff_Pi_inv",
                shape=(LL,),
                dtype=tf.int32,
                initializer=tf.constant_initializer(Pi_inv.numpy()),
                trainable=False,
            )
            self._ff_G = self.add_weight(
                name="ff_G",
                shape=(LL,),
                dtype=tf.float32,
                initializer=tf.constant_initializer(G.numpy()),
                trainable=False,
            )
            self._ff_scale = self.add_weight(
                name="ff_scale",
                shape=(),
                dtype=tf.float32,
                initializer=tf.constant_initializer(scale.numpy()),
                trainable=False,
            )
            self._ff_LL = int(LL)

        self._assign_theta(self.theta0)

    @property
    def trainable_variables(self):
        return [self.z]

    @property
    def trainable_weights(self):
        return [self.z]

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _assign_theta(self, theta_vec: tf.Tensor):
        off = 0
        for var, shape, size in zip(self.theta_vars, self.shapes, self.sizes):
            sl = theta_vec[off:off + size]
            var.assign(tf.cast(tf.reshape(sl, shape), var.dtype))
            off += size

    def _flatten_grads(self, grads):
        flats = []
        for grad, size in zip(grads, self.sizes):
            if grad is None:
                flats.append(tf.zeros([size], dtype=tf.float32))
            else:
                flats.append(tf.reshape(tf.cast(grad, tf.float32), [-1]))
        return tf.concat(flats, axis=0)

    def _dense_delta(self):
        if self._rotation_mode == "permute_sign":
            return tf.gather(self.z, self._rot_perm) * self._rot_sign
        if self._dense_use_blocks:
            delta = tf.zeros((self.D,), dtype=tf.float32)
            for block, (start, end) in zip(self.P_blocks, self.P_block_ranges):
                delta = delta + tf.linalg.matvec(block, self.z[start:end])
            return delta
        return tf.linalg.matvec(self.P, self.z)

    def _theta_from_z(self):
        if self.method == "dense":
            return self.theta0 + self._dense_delta()

        ray = fastfood_forward(
            self.z, self.D, self.d,
            self._ff_B, self._ff_Pi, self._ff_G, self._ff_scale, self._ff_LL,
        )
        return self.theta0 + ray

    def _gz_from_gtheta(self, g_theta):
        if self.method == "dense":
            if self._rotation_mode == "permute_sign":
                inv = tf.argsort(self._rot_perm)
                tmp = tf.gather(g_theta, self._rot_perm) * self._rot_sign
                return tf.gather(tmp, inv)
            if self._dense_use_blocks:
                parts = []
                for block, _ in zip(self.P_blocks, self.P_block_ranges):
                    parts.append(tf.linalg.matvec(tf.transpose(block), g_theta))
                return tf.concat(parts, axis=0)
            return tf.linalg.matvec(tf.transpose(self.P), g_theta)

        return fastfood_transpose(
            g_theta, self.D, self.d,
            self._ff_B, self._ff_Pi_inv, self._ff_G, self._ff_scale, self._ff_LL,
        )

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        theta = self._theta_from_z()
        self._assign_theta(theta)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss(y, y_pred)

        grads_theta = tape.gradient(loss, self.theta_vars)
        g_theta = self._flatten_grads(grads_theta)
        g_z = self._gz_from_gtheta(g_theta)
        self.optimizer.apply_gradients([(g_z, self.z)])

        theta_new = self._theta_from_z()
        self._assign_theta(theta_new)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        theta = self._theta_from_z()
        self._assign_theta(theta)

        y_pred = self(x, training=False)
        loss = self.loss(y, y_pred)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        theta = self._theta_from_z()
        self._assign_theta(theta)
        return self(x, training=False)
