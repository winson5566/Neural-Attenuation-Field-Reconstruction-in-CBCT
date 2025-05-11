import tensorflow as tf

class HashEmbeddingEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, dtype=tf.float32):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.hashmap_size = 2 ** log2_hashmap_size
        self.embedding_dtype = dtype  # ⚠️ 注意：不要用 self.dtype，这会冲突

        self.embeddings = []
        self.resolutions = []

        for i in range(num_levels):
            res = base_resolution * (2 ** i)
            self.resolutions.append(res)

            table = self.add_weight(
                shape=(self.hashmap_size, level_dim),
                initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4),
                trainable=True,
                name=f"embedding_l{i}",
                dtype=self.embedding_dtype
            )
            self.embeddings.append(table)

        # Use prime numbers for XOR-based spatial hashing (as per Müller et al.)
        self.primes = tf.constant([1, 2654435761, 805459861], dtype=tf.int64)

    def hash_fn(self, coords, level):
        coords = tf.cast(coords, tf.int64)
        hashed = coords[:, 0] * self.primes[0]
        hashed ^= coords[:, 1] * self.primes[1]
        hashed ^= coords[:, 2] * self.primes[2]
        hashed ^= tf.cast(level, tf.int64)
        hashed = tf.math.floormod(hashed, self.hashmap_size)
        return tf.cast(hashed, tf.int32)

    def trilinear_interp(self, x, level, res, emb_table):
        scaled = x * tf.cast(res, x.dtype)
        floor = tf.floor(scaled)
        frac = scaled - floor
        floor = tf.cast(floor, tf.int32)

        # 8 corner offsets
        offsets = tf.constant([
            [0, 0, 0], [0, 0, 1],
            [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1],
            [1, 1, 0], [1, 1, 1]
        ], dtype=tf.int32)

        batch_size = tf.shape(x)[0]
        output = tf.zeros((batch_size, self.level_dim), dtype=self.embedding_dtype)

        for i in range(8):
            neighbor = floor + offsets[i]
            idx = self.hash_fn(neighbor, level)
            emb = tf.gather(emb_table, idx)

            weight = tf.reduce_prod(
                tf.where(tf.equal(offsets[i], 1), frac, 1.0 - frac),
                axis=-1, keepdims=True
            )
            output += emb * tf.cast(weight, self.embedding_dtype)  # ✅ 确保一致类型

        return output

    def call(self, x, size=1.0):
        x = tf.cast(x, self.embedding_dtype)  # ✅ 保证输入类型一致
        x = (x + size) / (2 * size)
        x = tf.clip_by_value(x, 0.0, 1.0 - 1e-6)

        outputs = []
        for i in range(self.num_levels):
            out = self.trilinear_interp(x, i, self.resolutions[i], self.embeddings[i])
            outputs.append(out)

        return tf.concat(outputs, axis=-1)

    def get_output_dim(self):
        return self.num_levels * self.level_dim
