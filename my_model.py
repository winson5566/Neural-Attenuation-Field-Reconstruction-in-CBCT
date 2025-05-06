import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        w = self.pool(x)  # shape [B, C]
        w = self.fc1(w)
        w = self.fc2(w)
        w = tf.reshape(w, (-1, 1, 1, tf.shape(x)[-1]))  # shape [B,1,1,C]
        return x * w

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        self.se = SEBlock(filters)
        self.activation = tf.keras.layers.ReLU()
        self.project = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = self.project(residual)
        return self.activation(x + residual)

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
        # x shape: [B, H, W, C]
        e1 = self.encoder1(x)
        d1 = self.down1(e1)
        d2 = self.down2(d1)

        bn = self.bottleneck(d2)

        u2 = self.up2(bn, d1)
        u1 = self.up1(u2, e1)

        x = self.decoder_final(u1)
        return tf.clip_by_value(x, 0.0, 1.0)  # 或者 return x 后在 loss 中 clip

class MyModel(tf.keras.layers.Layer):
    def __init__(self, encoder, n_points=192, n_rays=2048, base_filters=32):
        super().__init__()
        self.encoder = encoder
        self.n_points = n_points
        self.n_rays = n_rays
        self.radunet = RADUNet(input_channels=1, output_channels=1, base_filters=base_filters)

    def call(self, points):
        features = self.encoder(points)  # [N, D]
        N = tf.shape(features)[0]

        # 如果是 get_sample_slices 的情况（即不是 n_rays * n_points），直接回归推理模式
        expected = self.n_rays * self.n_points
        if tf.not_equal(N, expected):
            # 推理模式（e.g. 16384 点 → reshape 成 [H, W, C]）
            sqrt_N = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)
            x = tf.reshape(features, (1, sqrt_N, sqrt_N, -1))  # e.g. [1,128,128,C]
            x = self.radunet(x)  # [1,H,W,1]
            x = tf.squeeze(x, axis=0)  # [H,W,1]
            return tf.squeeze(x, axis=-1)  # [H,W]

        # 否则是训练模式，reshape 成射线样本格式
        x = tf.reshape(features, (1, self.n_rays, self.n_points, -1))  # [1,H,W,C]
        x = self.radunet(x)  # [1,H,W,1]
        x = tf.squeeze(x, axis=0)  # [H,W,1]
        return tf.reshape(x, (self.n_rays, self.n_points))  # [n_rays, n_points]
