import tensorflow as tf
import numpy as np

# -----------------------------
# Fourier Positional Encoding
# -----------------------------
class FourierFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, num_frequencies=10, max_freq_log2=4, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** tf.linspace(0.0, max_freq_log2, num_frequencies)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        freq_bands = tf.reshape(self.freq_bands, (1, 1, -1))
        inputs_exp = tf.expand_dims(inputs, -1)
        scaled = 2.0 * np.pi * inputs_exp * freq_bands

        sin_feat = tf.sin(scaled)
        cos_feat = tf.cos(scaled)

        sin_feat = tf.reshape(sin_feat, (tf.shape(inputs)[0], -1))
        cos_feat = tf.reshape(cos_feat, (tf.shape(inputs)[0], -1))

        out = tf.concat([sin_feat, cos_feat], axis=-1)
        if self.include_input:
            out = tf.concat([inputs, out], axis=-1)
        return out

    def get_output_dim(self):
        return (1 + 2 * self.num_frequencies) * 3 if self.include_input else 2 * self.num_frequencies * 3

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.get_output_dim())

# -----------------------------
# Transformer Encoder Block (functional)
# -----------------------------
def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, dropout=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    x = tf.keras.layers.Add()([inputs, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim),
    ])
    ffn_output = ffn(x)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# -----------------------------
# Build Transformer Model
# -----------------------------
def build_transformer_model(encoder,
                             embed_dim=64,
                             num_heads=4,
                             ff_dim=128,
                             num_layers=4,
                             out_dim=1):
    input_xyz = tf.keras.Input(shape=(3,), name="xyz")

    # 1) 编码 + 投影
    x = encoder(input_xyz)                          # (N, D_enc)
    x = tf.keras.layers.Lambda(lambda t: t)(x)      # 固定 shape
    x = tf.keras.layers.Dense(embed_dim)(x)         # (N, embed_dim)

    # 2) 添加 batch 维（seq_len=N, batch=1）
    x = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=0))(x)  # (1, N, embed_dim)

    # 3) 多层 Transformer Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, embed_dim, num_heads, ff_dim)        # (1, N, embed_dim)

    # 4) 去掉 batch 维
    x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=0))(x)      # (N, embed_dim)

    # 5) 回归头
    output = tf.keras.layers.Dense(out_dim)(x)       # (N, 1)

    return tf.keras.Model(inputs=input_xyz,
                          outputs=output,
                          name="FourierTransformerModel")

