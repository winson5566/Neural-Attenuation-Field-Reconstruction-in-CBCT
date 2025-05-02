import tensorflow as tf
import numpy as np

# -----------------------------
# Fourier Feature Positional Encoder
# -----------------------------
class FourierFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, num_frequencies=10, max_freq_log2=4, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** tf.linspace(0.0, max_freq_log2, num_frequencies)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)  # shape: (N, 3)
        embed = []
        for freq in self.freq_bands:
            embed.append(tf.sin(freq * inputs))
            embed.append(tf.cos(freq * inputs))

        if self.include_input:
            embed = [inputs] + embed

        return tf.concat(embed, axis=-1)  # shape: (N, D)

    def get_output_dim(self):
        return (1 + 2 * self.num_frequencies) * 3 if self.include_input else (2 * self.num_frequencies) * 3

# -----------------------------
# Transformer Encoder Block
# -----------------------------
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# -----------------------------
# Full Transformer Model
# -----------------------------
class TransformerModel(tf.keras.layers.Layer):
    def __init__(self, encoder, embed_dim=64, num_heads=4, ff_dim=128, num_layers=4, out_dim=1):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        self.input_proj = tf.keras.layers.Dense(embed_dim)
        self.transformer_blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=False):
        x = self.encoder(x)                     # shape: (N, D_enc)
        x = self.input_proj(x)                  # shape: (N, embed_dim)
        x = tf.expand_dims(x, axis=0)           # shape: (1, N, embed_dim)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = tf.squeeze(x, axis=0)               # shape: (N, embed_dim)
        return self.output_layer(x)             # shape: (N, 1)