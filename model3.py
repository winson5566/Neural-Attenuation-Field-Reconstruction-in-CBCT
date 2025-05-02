import tensorflow as tf
import numpy as np

# ---------- Fourier Encoder ----------
class FourierFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, num_frequencies=10, max_freq_log2=4, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** tf.linspace(0., max_freq_log2, num_frequencies)

    def call(self, inputs):
        # inputs: (N, 3)
        inputs = tf.cast(inputs, tf.float32)
        fb   = tf.reshape(self.freq_bands, (1, 1, -1))          # (1,1,F)
        inpE = tf.expand_dims(inputs, -1)                       # (N,3,1)
        scaled = 2. * np.pi * inpE * fb                         # (N,3,F)
        sin, cos = tf.sin(scaled), tf.cos(scaled)
        sin = tf.reshape(sin, (tf.shape(inputs)[0], -1))
        cos = tf.reshape(cos, (tf.shape(inputs)[0], -1))
        out = tf.concat([sin, cos], axis=-1)
        if self.include_input:
            out = tf.concat([inputs, out], axis=-1)
        return out

    def get_output_dim(self):
        return (1 + 2 * self.num_frequencies) * 3 if self.include_input else 2 * self.num_frequencies * 3

# ---------- Transformer Block ----------
def transformer_layer(embed_dim, num_heads, ff_dim):
    def _layer(x):
        attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_dim)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
        return x
    return _layer

# ---------- Chunk-aware Transformer Model ----------
class ChunkTransformer(tf.keras.layers.Layer):
    """
    Drop-in replacement for your original Model:
    * 输入  (N,3) 平铺所有射线的点
    * 内部按 chunk_size 切分序列跑 Transformer，避免 OOM
    """
    def __init__(self,
                 encoder,
                 chunk_size=2048,            # 每块 ≤2048 token，显存安全
                 embed_dim=64,
                 num_heads=4,
                 ff_dim=128,
                 num_layers=4,
                 out_dim=1):
        super().__init__()
        self.encoder    = encoder
        self.chunk_size = chunk_size
        self.embed_dim  = embed_dim

        self.input_proj = tf.keras.layers.Dense(embed_dim)
        self.transformers = [transformer_layer(embed_dim, num_heads, ff_dim)
                             for _ in range(num_layers)]
        self.out_dense = tf.keras.layers.Dense(out_dim)

    def _forward_chunk(self, pts, training):
        """pts: (M,3) where M<=chunk_size"""
        x = self.encoder(pts)                   # (M, D_enc)
        x = self.input_proj(x)                  # (M, embed_dim)
        x = tf.expand_dims(x, 0)                # (1,M,D)
        for block in self.transformers:
            x = block(x)
        x = tf.squeeze(x, 0)                    # (M,D)
        return self.out_dense(x)                # (M,1)

    def call(self, inputs, training=False):
        # inputs: (N,3)    →  输出 (N,1)
        chunks = tf.split(inputs, self.chunk_size, axis=0)
        outs   = [self._forward_chunk(c, training) for c in chunks]
        return tf.concat(outs, axis=0)
