import tensorflow as tf

# Residual Dense Block
class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(dim)
        self.project = tf.keras.layers.Dense(dim)  # 用于调整 residual 的维度

    def call(self, x):
        residual = x
        out = self.fc1(x)
        out = self.fc2(out)
        # 保证 residual 和 out 的形状相同
        if residual.shape[-1] != out.shape[-1]:
            residual = self.project(residual)

        return tf.nn.relu(out + residual)

# Model 2 with Residual Connections
class Model2(tf.keras.layers.Layer):
    def __init__(self, encoder, bound=0.2, num_layers=4, hidden_dim=32, skips=[1], out_dim=1,
                 last_activation="sigmoid"):
        super(Model2, self).__init__()

        self.encoder = encoder
        self.bound = bound
        self.in_dim = self.encoder.get_output_dim()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim

        self.layers = []

        # First layer (standard dense)
        self.layers.append(tf.keras.layers.Dense(hidden_dim))

        # Intermediate layers with residual blocks
        for i in range(1, num_layers - 1):
            if i in skips:
                self.layers.append(ResidualDenseBlock(hidden_dim + self.in_dim))
            else:
                self.layers.append(ResidualDenseBlock(hidden_dim))

        # Final output layer
        self.layers.append(tf.keras.layers.Dense(out_dim))

        # Last activation
        if last_activation == "sigmoid":
            self.final_activation = tf.keras.layers.Activation("sigmoid")
        elif last_activation == "relu":
            self.final_activation = tf.keras.layers.ReLU()
        else:
            raise NotImplementedError("Unknown last activation")

    def call(self, x):
        x = self.encoder(x)
        input_pts = x[..., :self.in_dim]

        for i in range(self.num_layers):
            layer = self.layers[i]

            if i in self.skips:
                x = tf.concat([input_pts, x], axis=-1)

            x = layer(x)

        x = self.final_activation(x)
        return x
