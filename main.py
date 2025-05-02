import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from geometry import TIGREDataset
from todo import *
from datetime import datetime
import skimage.io

# NOTE: The hyperparameter values in this file are set to similar numbers to the NAF paper.
# You are encouraged to experiment and change them to find something that works better
# for your architectural change. These should work fine for Step 1.
class Model(tf.keras.layers.Layer):
    """
    A model class for Attenuation coefficient prediction from https://arxiv.org/abs/2209.14540
    This implementation uses an argument encoder to encode points in 3-dimensional space and
    then passes the encoding to several dense layers to produce the predicted attenuation
    at that point in 3-dimensional space.
    """

    def __init__(self, encoder, bound=0.2, num_layers=4, hidden_dim=32, skips=[1], out_dim=1,
                 last_activation="sigmoid"):
        super(Model, self).__init__()

        self.encoder = encoder
        self.bound = bound
        self.in_dim = self.encoder.get_output_dim()  # Get the input dimension from the encoder
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim

        # Define the layers
        self.layers = []
        # First layer
        self.layers.append(tf.keras.layers.Dense(hidden_dim))

        # Intermediate layers
        for i in range(1, num_layers - 1):
            if i in skips:
                self.layers.append(tf.keras.layers.Dense(hidden_dim + self.in_dim))
            else:
                self.layers.append(tf.keras.layers.Dense(hidden_dim))

        # Output layer
        self.layers.append(tf.keras.layers.Dense(out_dim))

        # Activation functions
        self.activations = []
        for i in range(num_layers - 1):
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))  # Equivalent to nn.LeakyReLU() in PyTorch

        # Handle last activation
        if last_activation == "sigmoid":
            self.activations.append(tf.keras.layers.Activation("sigmoid"))
        elif last_activation == "relu":
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        else:
            raise NotImplementedError("Unknown last activation")

    def call(self, x):
        # First, encode the input using the encoder
        x = self.encoder(x)

        # Extract input points (if needed for skip connections)
        input_pts = x[..., :self.in_dim]

        # Apply the layers
        for i in range(self.num_layers):
            layer = self.layers[i]
            activation = self.activations[i] if i < len(self.activations) else None

            # If this layer is a skip layer, concatenate the input points
            if i in self.skips:
                x = tf.concat([input_pts, x], axis=-1)

            # Apply the linear transformation
            x = layer(x)

            # Apply the activation function
            if activation:
                x = activation(x)

        return x

def train(model, dataset, optimizer, n_points):
    """
    Simple training loop that iterates through each projection image, samples rays from that image,
    sends the points of those rays through the network, computes the predicted attenuation per ray,
    computes the loss between the predicted value and the true value, and then updates the network.
    """
    num_projections = dataset.rays.shape[-1]
    total_loss = 0
    for i in range(num_projections):
        projection, rays = dataset[i]
        points, distances = rays_to_points(rays, n_points, dataset.near, dataset.far)
        magnitudes = tf.norm(rays[..., 3:6], axis=-1)
        n_rays = points.shape[0]
        points = tf.reshape(points, (-1, 3))

        with tf.GradientTape() as tape:
            attenuation = model(points)
            attenuation = tf.reshape(attenuation, (n_rays, -1))
            predicted_attenuation = ray_attenuation(attenuation, distances, magnitudes, dataset.near, dataset.far)
            loss = tf.keras.losses.MSE(projection, predicted_attenuation)
            total_loss += loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss / num_projections


def get_sample_slices(model, dataset):
    """
    Queries the network at many 3-dimensional points to produce a voxelized grid of attenuation values.
    Note that the returned values are scaled by the max value.
    """
    slices = dataset.voxels.shape[2]
    slice_list = []
    for i in range(slices):
        voxels = tf.convert_to_tensor(dataset.voxels[:, :, i])
        shape = voxels.shape[0:2]
        voxels = tf.reshape(voxels, (-1, 3))
        image_pred = model(voxels)
        image_pred = tf.reshape(image_pred, shape)
        slice_list.append(image_pred)

    imarr = np.array(slice_list)
    imarr = ((imarr / np.max(imarr)) * 255).astype(np.uint8)
    return imarr

def main(dataset_path, epochs, n_points, n_rays):
    """
    Loads the data, saves a ground truth image, and then creates the model.
    Runs for a given number of epochs and number of sample points/sample rays for each projection image training loop.
    Saves a TIFF image of the sample slice output every 10 epochs.
    """
    # dataset = TIGREDataset(dataset_path, device="cuda", n_rays=n_rays)
    dataset = TIGREDataset(dataset_path, device="cuda", n_rays=n_rays)

    # need to transpose to get top down view
    ground_truth_volume = (dataset.ground_truth.transpose((2,0,1))*255).astype(np.uint8)

    skimage.io.imsave(f'data/out/gt.tiff', ground_truth_volume)

    size = dataset.far - dataset.near
    encoder = PositionEmbeddingEncoder(size, 8, 3, 3)
    model = Model(encoder)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    print(f'Starting training...')
    for epoch in range(epochs):
        epoch_loss = train(model, dataset, optimizer, n_points)

        pred_vol = get_sample_slices(model, dataset)  # shape=(Z,H,W), uint8

        ssim_val = tf.image.ssim(pred_vol, ground_truth_volume, max_val=255)
        psnr_val = tf.image.psnr(pred_vol, ground_truth_volume, max_val=255)
        m = tf.keras.metrics.MeanSquaredError()
        m.update_state(ground_truth_volume, pred_vol)
        mse_vol = m.result().numpy()

        # print(f'Epoch {epoch} loss: {epoch_loss}')
        print(f"Epoch {epoch}  loss={epoch_loss}  SSIM={ssim_val:.2f}  PSNR={psnr_val:.2f}  MSE={mse_vol:.2f}")

        import csv
        with open('data/out/foot_50/metrics_foot.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['dataset', 'epoch', 'loss', 'ssim', 'psnr','mse', 'timestamp'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([dataset_path, epoch, epoch_loss.numpy(), ssim_val.numpy().item(), psnr_val.numpy().item(),mse_vol.item(), timestamp])

        if epoch % 10 == 0 and epoch != 0:
            predicted_volume = get_sample_slices(model, dataset)
            print(f'Epoch {epoch} SSIM: {tf.image.ssim(predicted_volume, ground_truth_volume, max_val=255)}'+ \
                f' PSNR {tf.image.psnr(predicted_volume, ground_truth_volume, max_val=255)}')

            if not os.path.exists('data/out/'):
                os.mkdir('data/out/')

            skimage.io.imsave(f'data/out/{epoch}.tiff', predicted_volume)

if __name__ == '__main__':
    # dataset_path = 'data/ct_data/chest_50.pickle'
    # dataset_path = 'data/ct_data/abdomen_50.pickle'
    # dataset_path = 'data/ct_data/foot_50.pickle'
    dataset_path = 'data/ct_data/jaw_50.pickle'

    # 250 epochs is not enough to produce a high quality reconstruction but you should see
    # a clear shape after 10 epochs
    main(dataset_path, epochs=250, n_points=192, n_rays=2048)
