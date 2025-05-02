import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class TransformerModel(tf.keras.layers.Layer):
    def __init__(self, encoder, embed_dim=64, num_heads=4, ff_dim=128, num_layers=4, out_dim=1):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        # Project positional encoding to embedding dimension
        self.input_proj = Dense(embed_dim)

        # Stack of Transformer Encoder blocks
        self.transformer_blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

        # Regression head
        self.output_layer = Dense(out_dim)

    def call(self, x, training=False):
        x = self.encoder(x)                      # Positional encoding
        x = self.input_proj(x)                   # Project to embed_dim
        x = tf.expand_dims(x, axis=0)            # Transformer expects shape (batch, seq_len, embed_dim)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = tf.squeeze(x, axis=0)                # Remove batch dim for regression
        return self.output_layer(x)