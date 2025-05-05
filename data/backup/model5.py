import tensorflow as tf

# ========== CBAM ==========
class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

        # 空间注意力
        self.conv_spatial = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')

    def call(self, x):
        # ----- Channel Attention -----
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        avg = self.fc2(self.fc1(avg))
        max_ = self.fc2(self.fc1(max_))
        scale = tf.reshape(avg + max_, (-1, 1, 1, tf.shape(x)[-1]))
        x = x * scale

        # ----- Spatial Attention -----
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg, max_], axis=-1)
        spatial = self.conv_spatial(concat)
        return x * spatial

# ========== ConvBlock with CBAM ==========
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.cbam = CBAM(filters)
        self.activation = tf.keras.layers.ReLU()
        self.project = tf.keras.layers.Conv2D(filters, 1, padding='same')

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = self.project(residual)
        return self.activation(x + residual)

# ========== U-Net ==========

class DownSample(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2)
        self.block = ConvBlock(filters)

    def call(self, x):
        x = self.pool(x)
        return self.block(x)

class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.up = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')
        self.block = ConvBlock(filters)

    def call(self, x, skip):
        x = self.up(x)
        x = tf.concat([x, skip], axis=-1)
        return self.block(x)

class RADUNet(tf.keras.Model):
    def __init__(self, input_channels=1, output_channels=1, base_filters=32):
        super().__init__()
        self.encoder1 = ConvBlock(base_filters)
        self.down1 = DownSample(base_filters * 2)
        self.down2 = DownSample(base_filters * 4)

        self.bottleneck = ConvBlock(base_filters * 8)

        self.up2 = UpSample(base_filters * 4)
        self.up1 = UpSample(base_filters * 2)
        self.decoder_final = tf.keras.layers.Conv2D(output_channels, kernel_size=1)

    def call(self, x):
        e1 = self.encoder1(x)
        d1 = self.down1(e1)
        d2 = self.down2(d1)

        bn = self.bottleneck(d2)

        u2 = self.up2(bn, d1)
        u1 = self.up1(u2, e1)

        x = self.decoder_final(u1)
        return tf.clip_by_value(x, 0.0, 1.0)

# ========== Model Wrapper ==========

class Model5(tf.keras.layers.Layer):
    def __init__(self, encoder, n_points=192, n_rays=2048, base_filters=32):
        super().__init__()
        self.encoder = encoder
        self.n_points = n_points
        self.n_rays = n_rays
        self.radunet = RADUNet(input_channels=1, output_channels=1, base_filters=base_filters)

    def call(self, points):
        features = self.encoder(points)  # [N, D]
        N = tf.shape(features)[0]

        expected = self.n_rays * self.n_points
        if tf.not_equal(N, expected):
            sqrt_N = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)
            x = tf.reshape(features, (1, sqrt_N, sqrt_N, -1))
            x = self.radunet(x)
            x = tf.squeeze(x, axis=0)
            return tf.squeeze(x, axis=-1)

        x = tf.reshape(features, (1, self.n_rays, self.n_points, -1))
        x = self.radunet(x)
        x = tf.squeeze(x, axis=0)
        return tf.reshape(x, (self.n_rays, self.n_points))
