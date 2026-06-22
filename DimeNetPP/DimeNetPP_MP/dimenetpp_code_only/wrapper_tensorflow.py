import math
import tensorflow as tf


import tensorflow as tf
import numpy as np




import math
import tensorflow as tf
import numpy as np



import tensorflow as tf

def hadamard_1d_static(x: tf.Tensor, LL: int) -> tf.Tensor:
    """
    Unnormalized Walsh–Hadamard transform for fixed length LL (power of two).
    Uses Python loop => static shapes => no tf.while_loop shape invariants needed.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    # Force a static shape known to the tracer
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

    


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1)) == 0 and n != 0
import tensorflow as tf
import math

def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def _make_fastfood_params(D: int, dtype=tf.float32, seed: int = 123):
    """Deterministic Fastfood params for LL=nextpow2(D)."""
    LL = _next_power_of_two(D)

    # B: ±1
    B01 = tf.random.stateless_uniform([LL], seed=[seed, 1], minval=0, maxval=2, dtype=tf.int32)
    B = tf.cast(B01 * 2 - 1, dtype)

    # Permutation Pi via random keys -> argsort
    keys = tf.random.stateless_uniform([LL], seed=[seed, 2], dtype=tf.float32)
    Pi = tf.cast(tf.argsort(keys, axis=0, stable=True), tf.int32)
    Pi_inv = tf.cast(tf.argsort(Pi, axis=0, stable=True), tf.int32)

    # G: Gaussian
    G = tf.random.stateless_normal([LL], seed=[seed, 3], dtype=dtype)

    divisor = tf.sqrt(tf.cast(LL, dtype) * tf.reduce_sum(G * G))
    scale = divisor * tf.sqrt(tf.cast(D, dtype) / tf.cast(LL, dtype))
    return B, Pi, Pi_inv, G, scale, LL

def _fast_walsh_hadamard_1d(x: tf.Tensor) -> tf.Tensor:
    """Unnormalized Hadamard; assumes len(x)=2^k."""
    x = tf.convert_to_tensor(x)
    n = tf.shape(x)[0]
    y = x
    h = tf.constant(1, dtype=tf.int32)

    def cond(h, y):
        return h < n

    def body(h, y):
        y2 = tf.reshape(y, [-1, 2 * h])
        a = y2[:, :h]
        b = y2[:, h:2*h]
        y2 = tf.concat([a + b, a - b], axis=1)
        y2 = tf.reshape(y2, [n])
        return h * 2, y2

    _, y = tf.while_loop(cond, body, [h, y], parallel_iterations=1)
    return y

def fastfood_forward(z, D: int, d: int, B, Pi, G, scale, LL: int):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z_pad = tf.pad(z, [[0, LL - d]])          # [LL]
    z_pad = tf.reshape(z_pad, [LL]); z_pad.set_shape([LL])

    y = B * z_pad                              # [LL]
    y = hadamard_1d_static(y, LL)
    y = tf.gather(y, Pi)
    y = G * y
    y = hadamard_1d_static(y, LL)

    return y[:D] / scale


def fastfood_transpose(g_theta, D: int, d: int, B, Pi_inv, G, scale, LL: int):
    g_theta = tf.convert_to_tensor(g_theta, dtype=tf.float32)
    g_full = tf.pad(g_theta, [[0, LL - D]])    # [LL]
    g_full = tf.reshape(g_full, [LL]); g_full.set_shape([LL])

    g = g_full / scale
    g = hadamard_1d_static(g, LL)
    g = G * g
    g = tf.gather(g, Pi_inv)
    g = hadamard_1d_static(g, LL)
    g = B * g

    return g[:d]
import math
import tensorflow as tf


import tensorflow as tf
import numpy as np



import math
import tensorflow as tf
import numpy as np



import tensorflow as tf

def hadamard_1d_static(x: tf.Tensor, LL: int) -> tf.Tensor:
    """
    Unnormalized Walsh–Hadamard transform for fixed length LL (power of two).
    Uses Python loop => static shapes => no tf.while_loop shape invariants needed.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    # Force a static shape known to the tracer
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

    


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1)) == 0 and n != 0
import tensorflow as tf
import math

def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def _make_fastfood_params(D: int, dtype=tf.float32, seed: int = 123):
    """Deterministic Fastfood params for LL=nextpow2(D)."""
    LL = _next_power_of_two(D)

    # B: ±1
    B01 = tf.random.stateless_uniform([LL], seed=[seed, 1], minval=0, maxval=2, dtype=tf.int32)
    B = tf.cast(B01 * 2 - 1, dtype)

    # Permutation Pi via random keys -> argsort
    keys = tf.random.stateless_uniform([LL], seed=[seed, 2], dtype=tf.float32)
    Pi = tf.cast(tf.argsort(keys, axis=0, stable=True), tf.int32)
    Pi_inv = tf.cast(tf.argsort(Pi, axis=0, stable=True), tf.int32)

    # G: Gaussian
    G = tf.random.stateless_normal([LL], seed=[seed, 3], dtype=dtype)

    divisor = tf.sqrt(tf.cast(LL, dtype) * tf.reduce_sum(G * G))
    scale = divisor * tf.sqrt(tf.cast(D, dtype) / tf.cast(LL, dtype))
    return B, Pi, Pi_inv, G, scale, LL

def _fast_walsh_hadamard_1d(x: tf.Tensor) -> tf.Tensor:
    """Unnormalized Hadamard; assumes len(x)=2^k."""
    x = tf.convert_to_tensor(x)
    n = tf.shape(x)[0]
    y = x
    h = tf.constant(1, dtype=tf.int32)

    def cond(h, y):
        return h < n

    def body(h, y):
        y2 = tf.reshape(y, [-1, 2 * h])
        a = y2[:, :h]
        b = y2[:, h:2*h]
        y2 = tf.concat([a + b, a - b], axis=1)
        y2 = tf.reshape(y2, [n])
        return h * 2, y2

    _, y = tf.while_loop(cond, body, [h, y], parallel_iterations=1)
    return y

def fastfood_forward(z, D: int, d: int, B, Pi, G, scale, LL: int):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z_pad = tf.pad(z, [[0, LL - d]])          # [LL]
    z_pad = tf.reshape(z_pad, [LL]); z_pad.set_shape([LL])

    y = B * z_pad                              # [LL]
    y = hadamard_1d_static(y, LL)
    y = tf.gather(y, Pi)
    y = G * y
    y = hadamard_1d_static(y, LL)

    return y[:D] / scale


def fastfood_transpose(g_theta, D: int, d: int, B, Pi_inv, G, scale, LL: int):
    g_theta = tf.convert_to_tensor(g_theta, dtype=tf.float32)
    g_full = tf.pad(g_theta, [[0, LL - D]])    # [LL]
    g_full = tf.reshape(g_full, [LL]); g_full.set_shape([LL])

    g = g_full / scale
    g = hadamard_1d_static(g, LL)
    g = G * g
    g = tf.gather(g, Pi_inv)
    g = hadamard_1d_static(g, LL)
    g = B * g

    return g[:d]
import tensorflow as tf
import numpy as np

class SubspaceProjectedGradTF(tf.keras.Model):
    """
    Intrinsic-dimension training via projected gradients.
    Supports method="dense" or "fastfood".
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        d,
        method: str = "dense",           # "dense" | "fastfood"
        seed: int = 123,
        orthonormal: bool = False,       # dense only
        full_rotation: bool = False,     # dense only when d==D
        name="SubspaceProjectedGradTF",
    ):
        super().__init__(name=name)
        assert method in ("dense", "fastfood")
        self.base_model = base_model
        self.base_model.trainable = True  # need grads wrt these
        
        self.theta_vars = list(self.base_model.trainable_variables)
        self.shapes = [v.shape for v in self.theta_vars]
        self.sizes = [int(tf.size(v)) for v in self.theta_vars]
        self.D = int(sum(self.sizes))

        # Resolve d: int or fraction
        if isinstance(d, float):
            d = int(round(d * self.D))
        d = int(d)
        if not (1 <= d <= self.D):
            raise ValueError(f"d must be in [1, D]; got d={d} D={self.D}")
        self.d = d
        self.method = method

        print("Summary:", "Total parameters =", self.D, "Trainable parameters =", self.d, "Method:", self.method)

        # Snapshot theta0 (non-trainable)
        theta0_np = tf.concat([tf.reshape(tf.cast(v, tf.float32), [-1]) for v in self.theta_vars], axis=0).numpy()
        self.theta0 = self.add_weight(
            name="theta0",
            shape=(self.D,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(theta0_np),
            trainable=False,
        )

        # Trainable intrinsic vector z
        self.z = self.add_weight(
            name="z",
            shape=(self.d,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )

        # ----------------------------
        # Build projection mechanism
        # ----------------------------
        if self.method == "dense":
            # NOTE: tf.random.Generator has no .shuffle() in TF 2.21, so for any permutation
            # we use stateless keys->argsort.
            self._rotation_mode = "none"
            self._rot_perm = None
            self._rot_sign = None

            if (self.d == self.D) and full_rotation:
                # Per your instruction: "unless it is the rotation, then we use I."
                P_np = tf.eye(self.D, dtype=tf.float32).numpy()
                self.P = self.add_weight(
                    name="P",
                    shape=(self.D, self.d),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(P_np),
                    trainable=False,
                )

                # Keep the permute+sign rotation mode (this is the actual rotation effect).
                # Deterministic permutation via random keys:
                keys = tf.random.stateless_uniform([self.D], seed=[seed, 777], dtype=tf.float32)
                perm_np = tf.argsort(keys, axis=0, stable=True).numpy().astype(np.int32)

                s = tf.random.stateless_uniform([self.D], seed=[seed, 778], minval=0, maxval=2, dtype=tf.int32)
                sign_np = tf.cast(s * 2 - 1, tf.float32).numpy()

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

            else:
                # Per your instruction: random P for ALL other d (including d==D without rotation),
                # and normalize columns like PyTorch: A / ||A||_col.
                A = tf.random.stateless_normal([self.D, self.d], seed=[seed, 42], dtype=tf.float32)

                if orthonormal:
                    # QR reduced: Q is [D, d]
                    Q, _ = tf.linalg.qr(A, full_matrices=False)
                    P = Q
                else:
                    P = A / (tf.norm(A, axis=0, keepdims=True) + 1e-12)

                self.P = self.add_weight(
                    name="P",
                    shape=(self.D, self.d),
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(P.numpy()),
                    trainable=False,
                )

        else:
            # Fastfood params (implicit projection)
            B, Pi, Pi_inv, G, scale, LL = _make_fastfood_params(self.D, dtype=tf.float32, seed=seed)
            print("D =", self.D, "d =", self.d, "LL =", LL)

            self._ff_B = tf.Variable(B, trainable=False, dtype=tf.float32, name="ff_B")
            self._ff_Pi = tf.Variable(Pi, trainable=False, dtype=tf.int32, name="ff_Pi")
            self._ff_Pi_inv = tf.Variable(Pi_inv, trainable=False, dtype=tf.int32, name="ff_Pi_inv")
            self._ff_G = tf.Variable(G, trainable=False, dtype=tf.float32, name="ff_G")
            self._ff_scale = tf.Variable(scale, trainable=False, dtype=tf.float32, name="ff_scale")
            self._ff_LL = int(LL)

        # Initialize base weights to theta0 + Pz (z=0 => theta0)
        self._assign_theta(self.theta0)

    def _assign_theta(self, theta_vec: tf.Tensor):
        off = 0
        for v, shp, n in zip(self.theta_vars, self.shapes, self.sizes):
            sl = theta_vec[off:off+n]
            v.assign(tf.cast(tf.reshape(sl, shp), v.dtype))
            off += n

    def _flatten_grads(self, grads):
        flats = []
        for g, n in zip(grads, self.sizes):
            if g is None:
                flats.append(tf.zeros([n], dtype=tf.float32))
            else:
                flats.append(tf.reshape(tf.cast(g, tf.float32), [-1]))
        return tf.concat(flats, axis=0)  # [D]

    def _theta_from_z(self):
        if self.method == "dense":
            if getattr(self, "_rotation_mode", "none") == "permute_sign":
                rotated = tf.gather(self.z, self._rot_perm) * self._rot_sign
                return self.theta0 + rotated
            return self.theta0 + tf.linalg.matvec(self.P, self.z)

        ray = fastfood_forward(
            self.z, self.D, self.d,
            self._ff_B, self._ff_Pi, self._ff_G, self._ff_scale, self._ff_LL
        )
        return self.theta0 + ray

    def _gz_from_gtheta(self, g_theta):
        if self.method == "dense":
            if getattr(self, "_rotation_mode", "none") == "permute_sign":
                inv = tf.argsort(self._rot_perm)
                tmp = tf.gather(g_theta, self._rot_perm) * self._rot_sign
                return tf.gather(tmp, inv)
            return tf.linalg.matvec(tf.transpose(self.P), g_theta)

        return fastfood_transpose(
            g_theta, self.D, self.d,
            self._ff_B, self._ff_Pi_inv, self._ff_G, self._ff_scale, self._ff_LL
        )

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
    
        # Ensure base weights reflect current z
        theta = self._theta_from_z()
        self._assign_theta(theta)
    
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
    
        grads_theta = tape.gradient(loss, self.theta_vars)
        g_theta = self._flatten_grads(grads_theta)
        g_z = self._gz_from_gtheta(g_theta)
    
        self.optimizer.apply_gradients([(g_z, self.z)])
    
        # Re-assign after z update
        theta_new = self._theta_from_z()
        self._assign_theta(theta_new)
    
        # Let Keras update the right metrics the right way
        logs = self.compute_metrics(x=x, y=y, y_pred=y_pred)
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        x, y = data
    
        theta = self._theta_from_z()
        self._assign_theta(theta)
    
        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
    
        logs = self.compute_metrics(x=x, y=y, y_pred=y_pred)
        logs["loss"] = loss
        return logs

    def predict_step(self, data):
        # Keras may pass x directly, or (x,), or (x, y, sample_weight)-like tuples
        if isinstance(data, (tuple, list)):
            x = data[0]
        else:
            x = data

        # Make sure base-model weights match current z before prediction
        theta = self._theta_from_z()
        self._assign_theta(theta)

        return self(x, training=False)