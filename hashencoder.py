import tensorflow as tf
class HashEmbeddingEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.hashmap_size = 2 ** log2_hashmap_size

        self.embeddings = []
        self.resolutions = []

        for i in range(num_levels):
            res = base_resolution * (2 ** i)
            self.resolutions.append(res)

            table = self.add_weight(
                shape=(self.hashmap_size, level_dim),
                initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4),
                trainable=True,
                name=f"embedding_l{i}"
            )
            self.embeddings.append(table)

    def hash_fn(self, coords, level):
        primes = tf.constant([1546061, 1005013, 1673733], dtype=tf.int64)
        coords = tf.cast(coords, tf.int64)
        hashed = tf.reduce_sum(coords * primes, axis=-1)
        hashed = (hashed ^ tf.cast(level, tf.int64)) % self.hashmap_size
        return tf.cast(hashed, tf.int32)

    def trilinear_interp(self, x, level, res, emb_table):
        scaled = x * tf.cast(res, x.dtype)
        floor = tf.floor(scaled)
        frac = scaled - floor
        floor = tf.cast(floor, tf.int32)

        offsets = tf.constant([
            [0, 0, 0], [0, 0, 1],
            [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1],
            [1, 1, 0], [1, 1, 1]
        ], dtype=tf.int32)

        interpolated = tf.zeros([tf.shape(x)[0], self.level_dim], dtype=tf.float32)

        for i in range(8):
            neighbor = floor + offsets[i]
            idx = self.hash_fn(neighbor, level)
            emb = tf.gather(emb_table, idx)

            w = tf.reduce_prod(
                tf.where(tf.equal(offsets[i], 1), frac, 1.0 - frac),
                axis=-1, keepdims=True
            )
            interpolated += emb * w

        return interpolated

    def call(self, x, size=1.0):
        x = (x + size) / (2 * size)
        x = tf.clip_by_value(x, 0.0, 1.0 - 1e-6)

        outputs = []
        for i in range(self.num_levels):
            res = self.resolutions[i]
            emb = self.embeddings[i]
            out = self.trilinear_interp(x, i, res, emb)
            outputs.append(out)

        return tf.concat(outputs, axis=-1)

    def get_output_dim(self):
        return self.num_levels * self.level_dim
