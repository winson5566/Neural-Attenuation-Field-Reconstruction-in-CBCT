import os
import numpy as np
import tensorflow as tf
from my_model import MyModel
from tensorflow import keras
from geometry import TIGREDataset
from todo import *
from HashGridEncoder import *
from datetime import datetime
import skimage.io
import csv
import time
import yaml
import argparse
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

    def __init__(self, encoder, bound=0.3, num_layers=4, hidden_dim=32, skips=[2], out_dim=1,
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

# def train(model, dataset, optimizer, n_points):
#     """
#     Simple training loop that iterates through each projection image, samples rays from that image,
#     sends the points of those rays through the network, computes the predicted attenuation per ray,
#     computes the loss between the predicted value and the true value, and then updates the network.
#     """
#     num_projections = dataset.rays.shape[-1]
#     total_loss = 0
#     for i in range(num_projections):
#         projection, rays = dataset[i]
#         points, distances = rays_to_points(rays, n_points, dataset.near, dataset.far)
#         magnitudes = tf.norm(rays[..., 3:6], axis=-1)
#         n_rays = points.shape[0]
#         points = tf.reshape(points, (-1, 3))
#
#         with tf.GradientTape() as tape:
#             attenuation = model(points)
#             attenuation = tf.reshape(attenuation, (n_rays, -1))
#             predicted_attenuation = ray_attenuation(attenuation, distances, magnitudes, dataset.near, dataset.far)
#             loss = tf.keras.losses.MSE(projection, predicted_attenuation)
#             total_loss += loss
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return total_loss / num_projections

@tf.function(reduce_retracing=True)
def train_step(model, points, distances, magnitudes, projection, optimizer, n_rays, near, far):
    with tf.GradientTape() as tape:
        attenuation = model(points)
        attenuation = tf.reshape(attenuation, (n_rays, -1))
        predicted_attenuation = ray_attenuation(attenuation, distances, magnitudes, near, far)
        loss = tf.keras.losses.MSE(projection, predicted_attenuation)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, dataset, optimizer, n_points):
    num_projections = dataset.rays.shape[-1]
    total_loss = 0
    for i in range(num_projections):
        projection, rays = dataset[i]  # eager 运行，绕过 autograph 的 reshape bug
        points, distances = rays_to_points(rays, n_points, dataset.near, dataset.far)
        magnitudes = tf.norm(rays[..., 3:6], axis=-1)
        n_rays = points.shape[0]
        points = tf.reshape(points, (-1, 3))

        loss = train_step(model, points, distances, magnitudes, projection, optimizer, n_rays, dataset.near, dataset.far)
        total_loss += loss
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
    config = load_config()
    net_type = config['network']['net_type']
    encoding_type = config['encoder']['encoding']
    i_eval = config['log']['i_eval']
    i_save = config['log']['i_save']
    device  = config['exp']['device']

    # 1. 准备输出目录
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_dir = os.path.join('data', 'out', f'{dataset_name}_train')
    os.makedirs(output_dir, exist_ok=True)

    """
    Loads the data, saves a ground truth image, and then creates the model.
    Runs for a given number of epochs and number of sample points/sample rays for each projection image training loop.
    Saves a TIFF image of the sample slice output every 10 epochs.
    """
    dataset = TIGREDataset(dataset_path, device=device, n_rays=n_rays)

    # need to transpose to get top down view
    ground_truth_volume = (dataset.ground_truth.transpose((2,0,1))*255).astype(np.uint8)

    # skimage.io.imsave(f'data/out/gt.tiff', ground_truth_volume)
    skimage.io.imsave(os.path.join(output_dir, 'gt.tiff'), ground_truth_volume)

    size = dataset.far - dataset.near

    # === Choose encoder ===
    if encoding_type == 'HASH':
        encoder = HashEmbeddingEncoder(
            input_dim=3,
            num_levels=16,
            level_dim=2,
            base_resolution=16,
            log2_hashmap_size=19,
            dtype=tf.float32  # 可切换为 float32 以获得更高精度
        )
    elif encoding_type == 'PSNR':
        encoder = PositionEmbeddingEncoder(size, 8, 3, 3)
    else:
        raise NotImplementedError(f"Unknown encoding type: {encoding_type}")

    # === Choose model ===
    if net_type == 'MLP':
        model = Model(encoder)
    elif net_type == 'RAD-UNet':
        model = MyModel(encoder, n_points=n_points, n_rays=n_rays)
    else:
        raise NotImplementedError(f"Unknown network type: {net_type}")

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # 固定 0.001
        beta_1=0.9,  # 对应论文里的 β₁=0.9
        beta_2=0.999,  # 对应论文里的 β₂=0.999
        epsilon=1e-7,  # TensorFlow 默认 epsilon
        amsgrad=False  # 与原论文不使用 AMSGrad 保持一致
    )

    # 4. 初始化 CSV
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'epoch', 'loss', 'ssim', 'psnr', 'mse', 'timestamp'])

    print(f'Starting training...')
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = train(model, dataset, optimizer, n_points)
        print(f"Epoch {epoch:04d}, time={time.time() - epoch_start:.2f}s")
        if epoch % i_eval == 0 and epoch != 0:
            predicted_volume = get_sample_slices(model, dataset)

            ssim_val = tf.image.ssim(predicted_volume, ground_truth_volume, max_val=255)
            psnr_val = tf.image.psnr(predicted_volume, ground_truth_volume, max_val=255)
            m = tf.keras.metrics.MeanSquaredError()
            m.update_state(ground_truth_volume, predicted_volume)
            mse_vol = m.result().numpy()

            # print(f'Epoch {epoch} loss: {epoch_loss}')
            print(f"Epoch {epoch}  loss={epoch_loss}  SSIM={ssim_val:.4f}  PSNR={psnr_val:.3f}  MSE={mse_vol:.3f}")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([dataset_path, epoch, epoch_loss.numpy(), round(ssim_val.numpy().item(), 4),
                                 round(psnr_val.numpy().item(), 4), round(mse_vol.item(), 4), timestamp])

            # print(f'Epoch {epoch} SSIM: {tf.image.ssim(predicted_volume, ground_truth_volume, max_val=255)}'+ \
            #     f' PSNR {tf.image.psnr(predicted_volume, ground_truth_volume, max_val=255)}')

            # if not os.path.exists('data/out/'):
            #     os.mkdir('data/out/')
        if epoch % i_save == 0 and epoch != 0:
            skimage.io.imsave(os.path.join(output_dir, f'{epoch:04d}.tiff'), predicted_volume)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == '__main__':
    # dataset_path = 'data/ct_data/chest_50.pickle'
    # dataset_path = 'data/ct_data/abdomen_50.pickle'
    # dataset_path = 'data/ct_data/foot_50.pickle'
    # dataset_path = 'data/ct_data/jaw_50.pickle'

    # # 250 epochs is not enough to produce a high quality reconstruction but you should see
    # # a clear shape after 10 epochs
    # main('data/ct_data/chest_50.pickle', epochs=1010, n_points=192, n_rays=2048)
    # main('data/ct_data/abdomen_50.pickle', epochs=1010, n_points=192, n_rays=2048)
    # main('data/ct_data/foot_50.pickle', epochs=1010, n_points=192, n_rays=2048)
    # main('data/ct_data/jaw_50.pickle', epochs=1010, n_points=192, n_rays=2048)

    config = load_config()
    dataset_path = config['exp']['datadir']
    epochs = config['train']['epoch']
    n_points = config['train']['n_points']
    n_rays = config['train']['n_rays']

    main(dataset_path, epochs=epochs, n_points=n_points, n_rays=n_rays)

