import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp

import keras
from keras import ops
from keras import layers

from functools import partial


def precompute_frequencies(dim, max_pos, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    t = jnp.arange(0, max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


@partial(jax.jit, static_argnums=(3,))
def calculate_rope(x, cos_freq, sin_freq, offset=0):
    # x shape  is [batch_size, seqlen, num_heads, heads_dim]

    # Get the sequence length
    seqlen = x.shape[1]

    # Get the corresponding positional embeddings
    sin = sin_freq[offset : offset + seqlen, :]
    cos = cos_freq[offset : offset + seqlen, :]

    # Positional embeddings are 2D while our input is 3D
    # if `num_heads` dimension is present in the inputs.
    # We need to add another dimension to our positional embeddings
    sin = sin[jnp.newaxis, :, jnp.newaxis, :]
    cos = cos[jnp.newaxis, :, jnp.newaxis, :]

    # Get the even-odd positions from the inputs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Matmul with the rotation matrix
    # [cos_nθ, -sin_nθ] [x1]
    # [sin_nθ,  cos_nθ] [x2]
    # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
    pos_embed = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    pos_embed = jax.lax.collapse(pos_embed, -2)
    return pos_embed.astype(x.dtype)


@jax.jit
@jax.vmap
def update_tensor(source, target, indices):
    return target.at[indices].set(source[: len(indices)])


class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-6, name=None, dtype="float16"):
        super().__init__()
        self.norm_epsilon = eps
        self.layer_dtype = dtype
        self.kernel_initializer = keras.initializers.get("ones")
        self.name = name if name is not None else self.__class__.__name__

    def build(self, inputs_shape):
        self.weight = self.add_weight(
            shape=(inputs_shape[-1],),
            initializer=self.kernel_initializer,
            trainable=True,
            name=self.name + ".weight",
            # dtype=self.layer_dtype,
        )
        self.built = True

    def _norm(self, x):
        return x * ops.rsqrt(ops.mean(x**2, axis=-1, keepdims=True) + self.norm_epsilon)

    def call(self, x):
        output = ops.cast(self._norm(ops.cast(x, "float32")), x.dtype)
        return output * self.weight


class FeedForward(layers.Layer):
    def __init__(self, args, name=None, dtype="float16"):
        super().__init__()
        self.dim = args.dim
        self.layer_dtype = dtype
        self.hidden_dim = args.hidden_dim
        self.act = keras.activations.silu
        self.name = name if name is not None else self.__class__.__name__

    def build(self, inputs_shape):
        self.w1 = layers.Dense(
            self.hidden_dim,
            use_bias=False,
            name=f"{self.name}.w1",
            dtype=self.layer_dtype,
        )
        w1_input_shape = list(inputs_shape)
        w1_input_shape[-1] = self.dim
        self.w1.build(w1_input_shape)

        self.w2 = layers.Dense(
            self.dim, use_bias=False, name=f"{self.name}.w2", dtype=self.layer_dtype
        )
        w2_input_shape = list(inputs_shape)
        w2_input_shape[-1] = self.hidden_dim
        self.w2.build(w2_input_shape)

        self.w3 = layers.Dense(
            self.hidden_dim,
            use_bias=False,
            name=f"{self.name}.w3",
            dtype=self.layer_dtype,
        )
        w3_input_shape = list(inputs_shape)
        w3_input_shape[-1] = self.dim
        self.w3.build(w3_input_shape)

        self.built = True

    def call(self, x):
        return self.w2(
            ops.cast(self.act(ops.cast(self.w1(x), "float32")), x.dtype) * self.w3(x)
        )
        # return self.w2(self.w1(x) * self.w3(x))


class Attention(layers.Layer):
    def __init__(self, args, name=None, dtype="float16"):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = args.head_dim**-0.5
        self.head_dim = args.head_dim
        self.dim = args.dim
        self.layer_dtype = dtype
        self.name = name if name is not None else self.__class__.__name__

    def build(self, inputs_shape):
        wqkv_input_shape = list(inputs_shape)
        wqkv_input_shape[-1] = self.dim

        self.wq = layers.Dense(
            self.n_heads * self.head_dim,
            use_bias=False,
            name=f"{self.name}.wq",
            dtype=self.layer_dtype,
        )
        self.wq.build(wqkv_input_shape)

        self.wk = layers.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            name=f"{self.name}.wk",
            dtype=self.layer_dtype,
        )
        self.wk.build(wqkv_input_shape)

        self.wv = layers.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            name=f"{self.name}.wv",
            dtype=self.layer_dtype,
        )
        self.wv.build(wqkv_input_shape)

        self.wo = layers.Dense(
            self.dim, use_bias=False, name=f"{self.name}.wo", dtype=self.layer_dtype
        )
        wo_input_shape = list(inputs_shape)
        wo_input_shape[-1] = self.n_heads * self.head_dim
        self.wo.build(wo_input_shape)

        self.built = True

    def call(
        self,
        x,
        cos_freq,
        sin_freq,
        positions,
        mask=None,
        cache_k=None,
        cache_v=None,
        training=True,
    ):
        # x shape: [batch_size, seqlen, num_heads, head_dim]
        bsz, seqlen, _ = ops.shape(x)

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, (bsz, seqlen, self.n_heads, self.head_dim))
        xk = ops.reshape(xk, (bsz, seqlen, self.n_kv_heads, self.head_dim))
        xv = ops.reshape(xv, (bsz, seqlen, self.n_kv_heads, self.head_dim))

        xq = calculate_rope(xq, cos_freq, sin_freq, 0)
        xk = calculate_rope(xk, cos_freq, sin_freq, 0)

        updated_cache_k = cache_k.at[:, positions, :, :].set(xk[:, positions])
        cache_k = jnp.where(updated_cache_k != 0, updated_cache_k, cache_k)

        updated_cache_v = cache_v.at[:, positions, :, :].set(xv[:, positions])
        cache_v = jnp.where(updated_cache_v != 0, updated_cache_v, cache_v)

        if positions.shape[0] > 1:
            # prefill
            key = jnp.repeat(xk, self.kv_repeats, axis=2)
            value = jnp.repeat(xv, self.kv_repeats, axis=2)
        else:
            cur_pos = positions[-1] + 1
            key = jnp.repeat(cache_k[:bsz, :cur_pos, :], self.kv_repeats, axis=2)
            value = jnp.repeat(cache_v[:bsz, :cur_pos, :], self.kv_repeats, axis=2)

        # [bsz, seqlen, num_heads, head_dim] -> [bsz, num_heads, seqlen, head_dim]
        query = jnp.transpose(xq, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # # scores : [n_heads, seqlen | 1, seqlen]
        scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) * self.scale

        if mask is not None:
            # Mask will of shape [seqlen, seqlen] but our scores
            # have shape [batch_size, num_heads, seqlen, seqlen], hence we need
            # to introduce another dimension in the mask
            mask = mask[jnp.newaxis, jnp.newaxis, ...]
            scores = scores + mask

        scores = ops.cast(
            ops.softmax(ops.cast(scores, "float32"), axis=-1), query.dtype
        )

        output = jnp.matmul(scores, value)
        output = jnp.reshape(jnp.transpose(output, (0, 2, 1, 3)), (bsz, seqlen, -1))
        output = self.wo(output)
        return output, cache_k, cache_v


class TransformerBlock(layers.Layer):
    def __init__(self, args, name=None, dtype="float16"):
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.attention = Attention(args, name=f"{self.name}.attention", dtype=dtype)
        self.ffn = FeedForward(args, name=f"{self.name}.feed_forward", dtype=dtype)
        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, name=f"{self.name}.attention_norm"
        )
        self.ffn_norm = RMSNorm(
            args.dim, eps=args.norm_eps, name=f"{self.name}.ffn_norm", dtype=dtype
        )

    def build(self, inputs_shape):
        self.attention.build(inputs_shape)
        self.ffn.build(inputs_shape)
        self.attention_norm.build(inputs_shape)
        self.ffn_norm.build(inputs_shape)
        self.built = True

    def call(
        self,
        x,
        cos_freq,
        sin_freq,
        positions,
        mask=None,
        cache_k=None,
        cache_v=None,
        training=True,
    ):
        r, cache_k, cache_v = self.attention(
            self.attention_norm(x),
            cos_freq,
            sin_freq,
            positions,
            mask=mask,
            cache_k=cache_k,
            cache_v=cache_v,
            training=False,
        )
        h = x + r
        r = self.ffn(self.ffn_norm(h))
        out = h + r
        return out, cache_k, cache_v


class Transformer(keras.Model):
    def __init__(self, args, dtype="float16"):
        super().__init__()
        self.sliding_window = args.sliding_window
        self.tok_embeddings = layers.Embedding(
            args.vocab_size, args.dim, name="tok_embeddings", dtype=dtype
        )
        self.tf_layers = [
            TransformerBlock(args, name=f"transformer_layer.{i}", dtype=dtype)
            for i in range(args.n_layers)
        ]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps, name="norm")
        self.out = layers.Dense(args.vocab_size, use_bias=False, name="out")
        self.cos_freq, self.sin_freq = precompute_frequencies(args.head_dim, 128_000)

    def call(self, inputs, training=False):
        x, positions, cache_k, cache_v = inputs
        # caches are of the shape (max_batch_size, num_layers, max_seq_length, num_kv_heads, head_im)
        # if x.dtype != "float16":
        #     x = ops.cast(x, "float16")

        # positions = positions[0]
        # if positions.ndim == 0:
        #     positions = positions.reshape(-1, 1)

        # x is of shape (batch_size, seqlen)
        h = self.tok_embeddings(x)
        sin_freq = self.sin_freq[positions]
        cos_freq = self.cos_freq[positions]

        if ops.shape(x)[1] > 1:
            seqlen = x.shape[1]
            t = jnp.full((seqlen, seqlen), dtype=h.dtype, fill_value=1)
            mask = jnp.tril(t, k=0)
            mask = jnp.log(jnp.triu(mask, k=-self.sliding_window))
        else:
            mask = None

        for i, layer in enumerate(self.tf_layers):
            h, cache_k_i, cache_v_i = layer(
                h,
                cos_freq,
                sin_freq,
                positions,
                mask=mask,
                cache_k=cache_k[:, i],
                cache_v=cache_v[:, i],
                training=training,
            )
            cache_k = ops.slice_update(cache_k, [0, i, 0, 0, 0], cache_k_i[:, None, :])
            cache_v = ops.slice_update(cache_v, [0, i, 0, 0, 0], cache_v_i[:, None, :])

        h = self.out(self.norm(h))
        return h, cache_k, cache_v
