import tensorflow as tf
import numpy as np

tf.keras.mixed_precision.set_global_policy('float32')


# ---------- 1. Window Partition and Reverse Functions ----------
def window_partition(x, window_size):
    """
    Partition the input tensor into non-overlapping windows.
    Args:
        x: Input tensor of shape (B, D, H, W, C)
        window_size: Window size
    Returns:
        Windows tensor of shape (num_windows*B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = tf.reshape(x, [
        B,
        D // window_size, window_size,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    ])
    # Permute and reshape to get windows
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = tf.reshape(x, [-1, window_size, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Reverse window partitioning.
    Args:
        windows: Windows tensor of shape (num_windows*B, window_size, window_size, window_size, C)
        window_size: Window size
        B, D, H, W: Original tensor dimensions
    Returns:
        Reversed tensor of shape (B, D, H, W, C)
    """
    x = tf.reshape(windows, [
        B,
        D // window_size, H // window_size, W // window_size,
        window_size, window_size, window_size,
        -1
    ])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, [B, D, H, W, -1])
    return x


# ---------- 2. Window Multi-Head Self Attention ----------
class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, window_size=4, shift_size=0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.shift_size = shift_size

        # QKV linear projection
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)

        # Final projection
        self.proj = tf.keras.layers.Dense(dim)

    def build(self, input_shape):
        # Build relative position embedding
        ws = self.window_size
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * ws - 1) * (2 * ws - 1) * (2 * ws - 1), self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name='relative_position_bias_table'
        )

        # Get pair-wise relative position index for each token in the window
        coords = tf.meshgrid(
            tf.range(ws), tf.range(ws), tf.range(ws), indexing='ij'
        )
        coords = tf.stack(coords, axis=0)
        coords = tf.reshape(coords, [3, -1])

        # Compute relative coordinates
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])

        # Shift relative coordinates to start from 0
        relative_coords = relative_coords + (self.window_size - 1)

        # Flatten relative coordinates
        relative_position_index = (
                relative_coords[:, :, 0] * (2 * ws - 1) * (2 * ws - 1) +
                relative_coords[:, :, 1] * (2 * ws - 1) +
                relative_coords[:, :, 2]
        )

        self.relative_position_index = tf.Variable(
            initial_value=tf.reshape(relative_position_index, [-1]),
            trainable=False,
            name='relative_position_index'
        )

        super().build(input_shape)

    def call(self, x, mask=None, training=True):
        B, D, H, W, C = x.shape
        # Apply cyclic shift if specified
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x,
                shift=[-self.shift_size, -self.shift_size, -self.shift_size],
                axis=[1, 2, 3]
            )
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (B*num_windows, window_size, window_size, window_size, C)
        x_windows = tf.reshape(x_windows, [-1, self.window_size ** 3,
                                           C])  # (B*num_windows, window_size*window_size*window_size, C)

        # W-MSA
        qkv = self.qkv(x_windows)  # (B*num_windows, window_size*window_size*window_size, 3*C)
        qkv = tf.reshape(qkv, [-1, self.window_size ** 3, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1,
                                 4])  # (3, B*num_windows, num_heads, window_size*window_size*window_size, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = tf.matmul(q, k,
                         transpose_b=True) * self.scale  # (B*num_windows, num_heads, window_size*window_size*window_size, window_size*window_size*window_size)

        # Add relative position bias
        relative_position_index = tf.reshape(
            self.relative_position_index,
            [self.window_size ** 3, self.window_size ** 3]
        )

        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(relative_position_index, [-1])
        )

        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_size ** 3, self.window_size ** 3, -1]
        )

        relative_position_bias = tf.transpose(relative_position_bias, [2, 0,
                                                                       1])  # (num_heads, window_size*window_size*window_size, window_size*window_size*window_size)
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(attn, [B // nW, nW, self.num_heads, self.window_size ** 3, self.window_size ** 3])
            attn = attn + mask
            attn = tf.reshape(attn, [-1, self.num_heads, self.window_size ** 3, self.window_size ** 3])

        # Normalize the attention weights
        attn = tf.nn.softmax(attn, axis=-1)

        # Weighted sum
        x = tf.matmul(attn, v)  # (B*num_windows, num_heads, window_size*window_size*window_size, C//num_heads)
        x = tf.transpose(x,
                         [0, 2, 1, 3])  # (B*num_windows, window_size*window_size*window_size, num_heads, C//num_heads)
        x = tf.reshape(x, [-1, self.window_size ** 3, C])  # (B*num_windows, window_size*window_size*window_size, C)

        # Linear projection
        x = self.proj(x)

        # Reshape back to window size
        x = tf.reshape(x, [-1, self.window_size, self.window_size, self.window_size, C])

        # Reverse window partitioning
        if self.shift_size > 0:
            # Reverse windows
            x = window_reverse(x, self.window_size, B, D, H, W)
            # Reverse cyclic shift
            x = tf.roll(
                x,
                shift=[self.shift_size, self.shift_size, self.shift_size],
                axis=[1, 2, 3]
            )
        else:
            x = window_reverse(x, self.window_size, B, D, H, W)

        return x


# ---------- 3. Feed-Forward Network ----------
class FFN(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(dim)
        ])

    def call(self, x):
        return self.layers(x)


# ---------- 4. Swin Transformer Block ----------
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # Window attention
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size if shift_size > 0 else 0,
            qkv_bias=qkv_bias
        )

        # Layer normalization
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # MLP
        self.mlp = FFN(dim, int(dim * mlp_ratio))

    def call(self, x, mask=None, training=True):
        shortcut = x

        # LayerNorm
        x = self.norm1(x)

        # Window Attention
        x = self.attn(x, mask=mask, training=training)

        # First residual connection
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


# ---------- 5. Residual Swin Transformer Block (RSTB) ----------
class RSTB(tf.keras.layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.):
        super().__init__()
        self.blocks = [
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            ) for i in range(depth)
        ]
        self.conv = tf.keras.layers.Conv3D(
            filters=dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

    def call(self, x, training=True):
        residual = x
        for block in self.blocks:
            x = block(x, training=training)
        x = self.conv(x)
        return x + residual


# ---------- 6. SwinIR for CBCT Reconstruction ----------
class SwinIR_CBCT(tf.keras.Model):
    """
    SwinIR model for CBCT reconstruction.

    Args:
        in_chans (int): Number of input channels, default 1 for CT
        embed_dim (int): Embedding dimension
        depths (tuple): Depth of each stage
        num_heads (tuple): Number of attention heads in different layers
        window_size (int): Window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim, default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value, default: True

    Input shape:
        (B, D, H, W, 1) tensor in [0, 1] range (float32)

    Output shape:
        Same as input, values in [0, 1] range (float32)
    """

    def __init__(
            self,
            in_chans=1,
            dim=32,
            depths=(2, 2, 2, 2),
            num_heads=4,
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            **kwargs
    ):
        super().__init__()

        # Shallow feature extraction
        self.shallow_conv = tf.keras.layers.Conv3D(
            filters=dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        # Deep feature extraction (RSTB blocks)
        self.rstbs = [
            RSTB(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio
            ) for depth in depths
        ]

        # Feature aggregation
        self.body_conv = tf.keras.layers.Conv3D(
            filters=dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        # Reconstruction
        self.conv_before_upsample = tf.keras.layers.Conv3D(
            filters=dim,
            kernel_size=3,
            padding='same',
            activation=tf.nn.gelu,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )

        self.final_conv = tf.keras.layers.Conv3D(
            filters=in_chans,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            dtype='float32'
        )

        self.sigmoid = tf.keras.layers.Activation('sigmoid', dtype='float32')

    def call(self, x, training=True):
        # Input normalization
        B, D, H, W, C = x.shape
        x_orig = x

        # Shallow feature extraction
        shallow_features = self.shallow_conv(x)

        # Deep feature extraction
        x = shallow_features
        for rstb in self.rstbs:
            x = rstb(x, training=training)

        # Feature aggregation
        x = self.body_conv(x)
        x = x + shallow_features

        # Reconstruction
        x = self.conv_before_upsample(x)
        x = self.final_conv(x)

        # Refinement output (residual learning)
        x = self.sigmoid(x)

        return x