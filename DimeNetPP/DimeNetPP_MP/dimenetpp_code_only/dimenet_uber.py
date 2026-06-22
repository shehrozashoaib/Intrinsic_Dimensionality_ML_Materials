import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import keras as ks
from keras import ops
from keras.layers import Layer, Add, Subtract, Concatenate, Multiply
import tensorflow as tf

from kgcnn.layers.geom import (
    NodePosition,
    NodeDistanceEuclidean,
    BesselBasisLayer,          # only used for reference, not for projected version
    EdgeAngle,
    ShiftPeriodicLattice,
    SphericalBasisLayer
)
from kgcnn.layers.gather import GatherNodes, GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.aggr import AggregateLocalEdges


# ============================================================
# Parameter specs / helpers
# ============================================================

@dataclass
class ParamSpec:
    name: str
    shape: Tuple[int, ...]
    size: int
    start: int


def _prod(shape) -> int:
    out = 1
    for x in shape:
        out *= int(x)
    return int(out)


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def collect_theta0_and_specs(base_model: ks.Model):
    """
    Collect theta0 and slice specs from an already-built base model.
    Order follows base_model.trainable_variables exactly.
    """
    tvars = list(base_model.trainable_variables)
    if len(tvars) == 0:
        raise ValueError("Base model has no trainable variables. Build it first.")

    specs: List[ParamSpec] = []
    flats = []
    start = 0
    for v in tvars:
        shape = tuple(int(x) for x in v.shape)
        size = _prod(shape)
        specs.append(ParamSpec(
            name=v.name,
            shape=shape,
            size=size,
            start=start
        ))
        flats.append(tf.reshape(tf.cast(v, tf.float32), [-1]))
        start += size

    theta0 = tf.concat(flats, axis=0).numpy().astype(np.float32)
    return theta0, specs


class SpecCursor:
    """
    Consumes ParamSpecs sequentially in the exact same order as the original model's
    trainable_variables. This is the simplest way to stay aligned with the original graph.
    """
    def __init__(self, specs: List[ParamSpec]):
        self.specs = specs
        self.i = 0

    def take(self, shape: Tuple[int, ...], what: str = "") -> ParamSpec:
        if self.i >= len(self.specs):
            raise IndexError(f"Ran out of specs while requesting {what} shape={shape}.")
        spec = self.specs[self.i]
        self.i += 1
        expected = tuple(int(x) for x in shape)
        got = tuple(int(x) for x in spec.shape)
        if expected != got:
            raise ValueError(
                f"Spec mismatch for {what}: expected shape={expected}, got spec={got} name={spec.name}"
            )
        return spec

    def done(self):
        return self.i == len(self.specs)


# ============================================================
# Fastfood projection
# ============================================================

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

    # B in {+1, -1}
    B01 = tf.random.stateless_uniform([LL], seed=[seed, 1], minval=0, maxval=2, dtype=tf.int32)
    B = tf.cast(B01 * 2 - 1, dtype)

    # permutation Pi
    keys = tf.random.stateless_uniform([LL], seed=[seed, 2], dtype=tf.float32)
    Pi = tf.cast(tf.argsort(keys, axis=0, stable=True), tf.int32)

    # Gaussian diagonal
    G = tf.random.stateless_normal([LL], seed=[seed, 3], dtype=dtype)

    divisor = tf.sqrt(tf.cast(LL, dtype) * tf.reduce_sum(G * G))
    scale = divisor * tf.sqrt(tf.cast(D, dtype) / tf.cast(LL, dtype))
    return B, Pi, G, scale, LL


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


# ============================================================
# Global projector
# ============================================================

class GlobalProjectorTF(Layer):
    """
    One global projector for the whole model.
    Uber-style idea:
      theta = theta0 + delta(z)
    and each tensor consumes a fixed slice.
    """
    def __init__(
        self,
        theta0_flat: np.ndarray,
        d,
        method: str = "dense",
        seed: int = 123,
        orthonormal: bool = False,
        name="global_projector",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        assert method in ("dense", "fastfood")

        theta0_flat = np.asarray(theta0_flat, dtype=np.float32).reshape(-1)
        self.D = int(theta0_flat.shape[0])

        if isinstance(d, float):
            d = int(round(d * self.D))
        self.d = int(d)
        if not (1 <= self.d <= self.D):
            raise ValueError(f"d must be in [1,D], got d={self.d}, D={self.D}")

        self.method = method
        self.seed = int(seed)
        self.orthonormal = bool(orthonormal)

        self.theta0 = self.add_weight(
            name="theta0",
            shape=(self.D,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(theta0_flat),
            trainable=False,
        )

        self.z = self.add_weight(
            name="z",
            shape=(self.d,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )

        if self.method == "dense":
            A = tf.random.stateless_normal([self.D, self.d], seed=[self.seed, 42], dtype=tf.float32)
            if self.orthonormal:
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
            B, Pi, G, scale, LL = _make_fastfood_params(self.D, dtype=tf.float32, seed=self.seed)
            self.ff_B = self.add_weight(
                name="ff_B", shape=(LL,), dtype=tf.float32,
                initializer=tf.constant_initializer(B.numpy()), trainable=False
            )
            self.ff_Pi = self.add_weight(
                name="ff_Pi", shape=(LL,), dtype=tf.int32,
                initializer=tf.constant_initializer(Pi.numpy()), trainable=False
            )
            self.ff_G = self.add_weight(
                name="ff_G", shape=(LL,), dtype=tf.float32,
                initializer=tf.constant_initializer(G.numpy()), trainable=False
            )
            self.ff_scale = self.add_weight(
                name="ff_scale", shape=(), dtype=tf.float32,
                initializer=tf.constant_initializer(scale.numpy()), trainable=False
            )
            self.ff_LL = int(LL)

    def flat_delta(self):
        if self.method == "dense":
            return tf.linalg.matvec(self.P, self.z)
        return fastfood_forward(
            self.z, self.D, self.d,
            self.ff_B, self.ff_Pi, self.ff_G, self.ff_scale, self.ff_LL
        )

    def flat_theta(self):
        return self.theta0 + self.flat_delta()

    def get_tensor(self, spec: ParamSpec):
        flat = self.flat_theta()[spec.start:spec.start + spec.size]
        return tf.reshape(flat, spec.shape)


# ============================================================
# Projected primitives
# ============================================================

class ProjectedEmbeddingDimeBlock(Layer):
    def __init__(self, projector: GlobalProjectorTF, spec: ParamSpec, **kwargs):
        super().__init__(**kwargs)
        self.projector = projector
        self.spec = spec

    def call(self, inputs):
        emb = self.projector.get_tensor(self.spec)
        return ops.take(emb, inputs, axis=0)


class ProjectedDense(Layer):
    def __init__(
        self,
        projector: GlobalProjectorTF,
        kernel_spec: ParamSpec,
        bias_spec: Optional[ParamSpec] = None,
        activation=None,
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projector = projector
        self.kernel_spec = kernel_spec
        self.bias_spec = bias_spec
        self.use_bias = bool(use_bias)
        self.activation = ks.activations.get(activation)

    def call(self, inputs):
        kernel = self.projector.get_tensor(self.kernel_spec)
        x = ops.matmul(inputs, kernel)
        if self.use_bias and self.bias_spec is not None:
            bias = self.projector.get_tensor(self.bias_spec)
            x = x + bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class ProjectedBesselBasisLayer(Layer):
    def __init__(
        self,
        projector: GlobalProjectorTF,
        freq_spec: ParamSpec,
        cutoff: float,
        envelope_exponent: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projector = projector
        self.freq_spec = freq_spec
        self.cutoff = float(cutoff)
        self.inv_cutoff = ops.convert_to_tensor(1.0 / cutoff, dtype="float32")
        self.envelope_exponent = int(envelope_exponent)

    def envelope(self, inputs):
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return ops.where(inputs < 1, env_val, ops.zeros_like(inputs))

    def call(self, inputs):
        freqs = self.projector.get_tensor(self.freq_spec)
        d_scaled = inputs * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * ops.sin(freqs * d_scaled)


class ProjectedResidualLayer(Layer):
    """
    Matches kgcnn.layers.update.ResidualLayer:
      dense_1 -> dense_2 -> add(input)
    """
    def __init__(
        self,
        projector: GlobalProjectorTF,
        cursor: SpecCursor,
        units: int,
        activation="swish",
        use_bias=True,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        k1 = cursor.take((units, units), f"{name}.dense_1.kernel")
        b1 = cursor.take((units,), f"{name}.dense_1.bias") if use_bias else None
        k2 = cursor.take((units, units), f"{name}.dense_2.kernel")
        b2 = cursor.take((units,), f"{name}.dense_2.bias") if use_bias else None

        self.dense_1 = ProjectedDense(
            projector, k1, b1, activation=activation, use_bias=use_bias, name=f"{name}_dense1"
        )
        self.dense_2 = ProjectedDense(
            projector, k2, b2, activation=activation, use_bias=use_bias, name=f"{name}_dense2"
        )
        self.add_end = Add()

    def call(self, inputs):
        x = self.dense_1(inputs, )
        x = self.dense_2(x, )
        return self.add_end([inputs, x], )


class ProjectedSimpleMLP(Layer):
    """
    DimeNet output MLP in your instantiated model is just 3 Dense(64)->swish layers,
    no normalization/dropout in the current trainable inventory.
    """
    def __init__(
        self,
        projector: GlobalProjectorTF,
        cursor: SpecCursor,
        units_list: List[int],
        activation="swish",
        use_bias=True,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.layers_dense = []

        in_dim = units_list[0]
        for i, out_dim in enumerate(units_list):
            k = cursor.take((in_dim, out_dim), f"{name}.dense{i}.kernel")
            b = cursor.take((out_dim,), f"{name}.dense{i}.bias") if use_bias else None
            self.layers_dense.append(
                ProjectedDense(
                    projector, k, b, activation=activation,
                    use_bias=use_bias, name=f"{name}_dense_{i}"
                )
            )
            in_dim = out_dim

    def call(self, inputs):
        x = inputs
        for layer in self.layers_dense:
            x = layer(x, )
        return x


# ============================================================
# Projected DimeNet blocks
# ============================================================

class ProjectedDimNetInteractionPPBlock(Layer):
    def __init__(
        self,
        projector: GlobalProjectorTF,
        cursor: SpecCursor,
        emb_size,
        int_emb_size,
        basis_emb_size,
        num_before_skip,
        num_after_skip,
        activation="swish",
        pooling_method="sum",
        name="dim_net_interaction_pp_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Basis transforms
        self.dense_rbf1 = ProjectedDense(
            projector,
            cursor.take((6, basis_emb_size), f"{name}.dense_rbf1.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_rbf1"
        )
        self.dense_rbf2 = ProjectedDense(
            projector,
            cursor.take((basis_emb_size, emb_size), f"{name}.dense_rbf2.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_rbf2"
        )
        self.dense_sbf1 = ProjectedDense(
            projector,
            cursor.take((42, basis_emb_size), f"{name}.dense_sbf1.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_sbf1"
        )
        self.dense_sbf2 = ProjectedDense(
            projector,
            cursor.take((basis_emb_size, int_emb_size), f"{name}.dense_sbf2.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_sbf2"
        )

        # Message transforms
        self.dense_ji = ProjectedDense(
            projector,
            cursor.take((emb_size, emb_size), f"{name}.dense_ji.kernel"),
            cursor.take((emb_size,), f"{name}.dense_ji.bias"),
            activation=activation,
            use_bias=True,
            name=f"{name}_dense_ji"
        )
        self.dense_kj = ProjectedDense(
            projector,
            cursor.take((emb_size, emb_size), f"{name}.dense_kj.kernel"),
            cursor.take((emb_size,), f"{name}.dense_kj.bias"),
            activation=activation,
            use_bias=True,
            name=f"{name}_dense_kj"
        )

        # Projections
        self.down_projection = ProjectedDense(
            projector,
            cursor.take((emb_size, int_emb_size), f"{name}.down_projection.kernel"),
            None,
            activation=activation,
            use_bias=False,
            name=f"{name}_down_projection"
        )
        self.up_projection = ProjectedDense(
            projector,
            cursor.take((int_emb_size, emb_size), f"{name}.up_projection.kernel"),
            None,
            activation=activation,
            use_bias=False,
            name=f"{name}_up_projection"
        )

        # Residual stack before skip
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ProjectedResidualLayer(
                    projector, cursor, units=emb_size, activation=activation, use_bias=True,
                    name=f"{name}_before_skip_{i}"
                )
            )

        self.final_before_skip = ProjectedDense(
            projector,
            cursor.take((emb_size, emb_size), f"{name}.final_before_skip.kernel"),
            cursor.take((emb_size,), f"{name}.final_before_skip.bias"),
            activation=activation,
            use_bias=True,
            name=f"{name}_final_before_skip"
        )

        # Residual stack after skip
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ProjectedResidualLayer(
                    projector, cursor, units=emb_size, activation=activation, use_bias=True,
                    name=f"{name}_after_skip_{i}"
                )
            )

        self.lay_add1 = Add()
        self.lay_add2 = Add()
        self.lay_mult1 = Multiply()
        self.lay_mult2 = Multiply()
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = AggregateLocalEdges(pooling_method=pooling_method)

    def call(self, inputs):
        x, rbf, sbf, id_expand = inputs

        x_ji = self.dense_ji(x,)
        x_kj = self.dense_kj(x, )

        rbf2 = self.dense_rbf1(rbf, )
        rbf2 = self.dense_rbf2(rbf2)
        x_kj = self.lay_mult1([x_kj, rbf2],)

        x_kj = self.down_projection(x_kj, )
        x_kj = self.lay_gather([x_kj, id_expand], )

        sbf2 = self.dense_sbf1(sbf, )
        sbf2 = self.dense_sbf2(sbf2, )
        x_kj = self.lay_mult2([x_kj, sbf2], )

        x_kj = self.lay_pool([rbf2, x_kj, id_expand], )
        x_kj = self.up_projection(x_kj, )

        x2 = self.lay_add1([x_ji, x_kj], )
        for layer in self.layers_before_skip:
            x2 = layer(x2, )
        x2 = self.final_before_skip(x2, )

        x = self.lay_add2([x, x2], )

        for layer in self.layers_after_skip:
            x = layer(x, )

        return x


class ProjectedDimNetOutputBlock(Layer):
    def __init__(
        self,
        projector: GlobalProjectorTF,
        cursor: SpecCursor,
        emb_size,
        out_emb_size,
        num_dense,
        num_targets=1,
        activation="swish",
        pooling_method="sum",
        name="dim_net_output_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dense_rbf = ProjectedDense(
            projector,
            cursor.take((6, emb_size), f"{name}.dense_rbf.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_rbf"
        )
        self.up_projection = ProjectedDense(
            projector,
            cursor.take((emb_size, out_emb_size), f"{name}.up_projection.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_up_projection"
        )

        # In your instantiated model, GraphMLP resolves to 3 Dense(64,64)+bias layers
        # because num_dense_output=3 and out_emb_size=64.
        mlp_units = [out_emb_size] * num_dense
        self.dense_mlp = ProjectedSimpleMLP(
            projector, cursor,
            units_list=mlp_units,
            activation=activation,
            use_bias=True,
            name=f"{name}_mlp"
        )

        self.dense_final = ProjectedDense(
            projector,
            cursor.take((out_emb_size, num_targets), f"{name}.dense_final.kernel"),
            None,
            activation=None,
            use_bias=False,
            name=f"{name}_dense_final"
        )

        self.dimnet_mult = Multiply()
        self.pool = AggregateLocalEdges(pooling_method=pooling_method)

    def call(self, inputs):
        n_atoms, x, rbf, idnb_i = inputs
        g = self.dense_rbf(rbf)
        x = self.dimnet_mult([g, x])
        x = self.pool([n_atoms, x, idnb_i])
        x = self.up_projection(x)
        x = self.dense_mlp(x)
        x = self.dense_final(x)
        return x


# ============================================================
# Projected model builder
# ============================================================

def make_projected_crystal_model(
    base_model: ks.Model,
    inputs: list,
    input_tensor_type: str,
    cast_disjoint_kwargs: dict,
    input_node_embedding: dict,
    emb_size: int,
    out_emb_size: int,
    int_emb_size: int,
    basis_emb_size: int,
    num_blocks: int,
    num_spherical: int,
    num_radial: int,
    cutoff: float,
    envelope_exponent: int,
    num_before_skip: int,
    num_after_skip: int,
    num_dense_output: int,
    num_targets: int,
    activation: str,
    extensive: bool,
    output_embedding: str,
    output_tensor_type: str,
    use_output_mlp: bool,
    output_mlp: dict,
    method: str = "fastfood",
    d=1.0,
    seed: int = 123,
    orthonormal: bool = False,
    name: str = "ProjectedDimeNetPP"
):
    """
    Build Uber-style projected DimeNet++ from an already-built base model.
    Assumes the same architecture/config as the base model.
    """

    from kgcnn.layers.modules import Input
    from kgcnn.models.casting import template_cast_list_input, template_cast_output
    from kgcnn.layers.mlp import MLP

    if use_output_mlp:
        raise NotImplementedError(
            "This projected builder currently targets the pre-final-output-MLP graph "
            "used in your current setup (use_output_mlp=False)."
        )

    theta0, specs = collect_theta0_and_specs(base_model)
    cursor = SpecCursor(specs)
    projector = GlobalProjectorTF(
        theta0_flat=theta0,
        d=d,
        method=method,
        seed=seed,
        orthonormal=orthonormal,
        name="global_projector"
    )

    # Inputs and disjoint casting
    model_inputs = [Input(**x) for x in inputs]
    disjoint_inputs = template_cast_list_input(
        model_inputs,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
        index_assignment=[None, None, 0, 2, None, None],
        mask_assignment=[0, 0, 1, 2, 1, None]
    )
    n, x, edi, angi, img, lattice, batch_id_node, batch_id_edge, batch_id_angles, node_id, edge_id, angle_id, count_nodes, count_edges, count_angles = disjoint_inputs

    # Atom embedding
    if input_node_embedding is not None:
        emb_spec = cursor.take((input_node_embedding["input_dim"] + 1, input_node_embedding["output_dim"]),
                               "embedding_dime_block.embeddings")
        n = ProjectedEmbeddingDimeBlock(projector, emb_spec, name="embedding_dime_block")(n)

    # Distances
    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, img, lattice, batch_id_edge])
    d_ij = NodeDistanceEuclidean()([pos1, pos2])

    # Bessel basis
    freq_spec = cursor.take((num_radial,), "bessel_basis_layer.frequencies")
    rbf = ProjectedBesselBasisLayer(
        projector, freq_spec,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        name="bessel_basis_layer"
    )(d_ij)

    # Angles / spherical basis
    v12 = Subtract()([pos1, pos2])
    a = EdgeAngle()([v12, angi])
    sbf = SphericalBasisLayer(
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent
    )([d_ij, a, angi])

    # Top embedding block: Dense(rbf)->GatherNodes->Concat->Dense
    rbf_emb = ProjectedDense(
        projector,
        cursor.take((num_radial, emb_size), "dense.kernel"),
        cursor.take((emb_size,), "dense.bias"),
        activation=activation,
        use_bias=True,
        name="dense"
    )(rbf)

    n_pairs = GatherNodes()([n, edi])
    x_msg = Concatenate(axis=-1)([n_pairs, rbf_emb])

    x_msg = ProjectedDense(
        projector,
        cursor.take((emb_size * 3, emb_size), "dense_1.kernel"),  # 2*emb_size from n_pairs + emb_size from rbf_emb
        cursor.take((emb_size,), "dense_1.bias"),
        activation=activation,
        use_bias=True,
        name="dense_1"
    )(x_msg)

    # First output block
    # ------------------------------------------------------------
    # IMPORTANT:
    # Consume specs in the same order as base_model.trainable_variables,
    # which for this DimeNet++ build is:
    #   interaction blocks first, then output blocks.
    # But CALL the blocks in the original graph order.
    # ------------------------------------------------------------
    
    interaction_blocks = []
    for i in range(num_blocks):
        block_name = "dim_net_interaction_pp_block" if i == 0 else f"dim_net_interaction_pp_block_{i}"
        interaction_blocks.append(
            ProjectedDimNetInteractionPPBlock(
                projector, cursor,
                emb_size=emb_size,
                int_emb_size=int_emb_size,
                basis_emb_size=basis_emb_size,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                activation=activation,
                name=block_name
            )
        )
    
    output_blocks = []
    for i in range(num_blocks + 1):
        if i == 0:
            out_name = "dim_net_output_block"
        elif i == 1:
            out_name = "dim_net_output_block_1"
        else:
            out_name = f"dim_net_output_block_{i}"
        output_blocks.append(
            ProjectedDimNetOutputBlock(
                projector, cursor,
                emb_size=emb_size,
                out_emb_size=out_emb_size,
                num_dense=num_dense_output,
                num_targets=num_targets,
                activation=activation,
                name=out_name
            )
        )
    
    # Graph order: first output block before interaction loop
    ps = output_blocks[0]([n, x_msg, rbf, edi])
    
    add_xp = Add()
    x_cur = x_msg
    
    for i in range(num_blocks):
        x_cur = interaction_blocks[i]([x_cur, rbf, sbf, angi])
        p_update = output_blocks[i + 1]([n, x_cur, rbf, edi])
        ps = add_xp([ps, p_update])
    # Pool
    if extensive:
        out = PoolingNodes(pooling_method="sum")([count_nodes, ps, batch_id_node])
    else:
        out = PoolingNodes(pooling_method="mean")([count_nodes, ps, batch_id_node])

    if output_embedding != "graph":
        raise ValueError("Only output_embedding='graph' is supported.")

    out = template_cast_output(
        [out, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges],
        output_embedding=output_embedding,
        output_tensor_type=output_tensor_type,
        input_tensor_type=input_tensor_type,
        cast_disjoint_kwargs=cast_disjoint_kwargs,
    )

    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)

    if not cursor.done():
        remaining = len(specs) - cursor.i
        raise ValueError(
            f"Projected build did not consume all parameter specs. Remaining: {remaining}"
        )

    return model


# ============================================================
# Convenience wrapper
# ============================================================

def build_projected_dimenetpp_from_config(
    base_model: ks.Model,
    model_config: dict,
    method: str = "fastfood",
    d=1.0,
    seed: int = 123,
    orthonormal: bool = False,
):
    """
    Convenience entry point matching your hyper config style.
    """
    return make_projected_crystal_model(
        base_model=base_model,
        inputs=model_config["inputs"],
        input_tensor_type=model_config["input_tensor_type"],
        cast_disjoint_kwargs=model_config.get("cast_disjoint_kwargs", {}),
        input_node_embedding=model_config.get("input_node_embedding", None),
        emb_size=model_config["emb_size"],
        out_emb_size=model_config["out_emb_size"],
        int_emb_size=model_config["int_emb_size"],
        basis_emb_size=model_config["basis_emb_size"],
        num_blocks=model_config["num_blocks"],
        num_spherical=model_config["num_spherical"],
        num_radial=model_config["num_radial"],
        cutoff=model_config["cutoff"],
        envelope_exponent=model_config["envelope_exponent"],
        num_before_skip=model_config["num_before_skip"],
        num_after_skip=model_config["num_after_skip"],
        num_dense_output=model_config["num_dense_output"],
        num_targets=model_config["num_targets"],
        activation=model_config["activation"],
        extensive=model_config["extensive"],
        output_embedding=model_config["output_embedding"],
        output_tensor_type=model_config["output_tensor_type"],
        use_output_mlp=model_config.get("use_output_mlp", False),
        output_mlp=model_config.get("output_mlp", None),
        method=method,
        d=d,
        seed=seed,
        orthonormal=orthonormal,
        name=f"ProjectedDimeNetPP_{method}"
    )