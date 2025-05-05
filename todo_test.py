import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# --------------------------------------------
# Fourier Feature Positional Encoding
# --------------------------------------------
def fourier_encode(x, num_bands=10, include_input=True, bound=1.0):
    """
    Apply Fourier feature mapping to input coordinates.
    x: [N,3] tensor, clipped to [-bound, bound]
    num_bands: number of freq bands L
    include_input: include raw coords
    bound: clipping bound
    returns: [N, feat_dim]
    """
    # Normalize coords to [-1,1]
    x = tf.clip_by_value(x, -bound, bound) / bound
    # Frequency bands: [2^0*pi, 2^1*pi, ..., 2^{L-1}*pi]
    bands = 2.0 ** tf.range(num_bands, dtype=x.dtype) * np.pi  # [L]
    x_expanded = tf.expand_dims(x, -1)                         # [N,3,1]
    freqs = tf.reshape(bands, [1, 1, -1])                      # [1,1,L]
    x_proj = x_expanded * freqs                                # [N,3,L]
    sin = tf.sin(x_proj)
    cos = tf.cos(x_proj)
    fourier = tf.reshape(tf.concat([sin, cos], axis=-1), [tf.shape(x)[0], -1])  # [N,6L]
    if include_input:
        return tf.concat([x, fourier], axis=-1)
    return fourier

# --------------------------------------------
# PositionEmbeddingEncoder wrapper
# --------------------------------------------
class PositionEmbeddingEncoder(tf.keras.layers.Layer):
    def __init__(self, size, n_depth, embedding_dim, input_dim):
        """
        Wrapper to match original signature: (size, n_depth, embedding_dim, input_dim)
        size: full span; bound = size/2
        n_depth: number of Fourier bands
        embedding_dim, input_dim: ignored
        """
        super().__init__()
        self.bound = size / 2.0
        self.num_bands = n_depth
        self.include_input = True

    def call(self, x):
        return fourier_encode(x,
                              num_bands=self.num_bands,
                              include_input=self.include_input,
                              bound=self.bound)

    def get_output_dim(self):
        # raw coords (3) + 6*num_bands
        return 3 + 6 * self.num_bands

    def get_flattened_position(self, x, depth):
        """
        将 N×3 的点 (x,y,z)，假设已经在 [0,1] 范围内，
        映射到分辨率 (2^depth)^3 网格上的扁平化索引：
          idx = floor( x * 2^depth ), clamp 到 [0,2^depth-1]
          flat = ix + iy * 2^depth + iz * (2^depth)^2
        返回形状 [N] 的 int32 张量。
        """
        # 1. 限制到 [0,1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        # 2. 网格分辨率
        res = 2 ** depth
        # 3. 计算每个坐标的格点索引
        idx = tf.cast(tf.floor(x * tf.cast(res, x.dtype)), tf.int32)
        idx = tf.clip_by_value(idx, 0, res - 1)
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        # 4. 扁平化
        return ix + iy * res + iz * (res ** 2)

# --------------------------------------------
# Stratified sampling
# --------------------------------------------
def stratified_sampling(near, far, n_coarse, n_rays):
    """
    Uniform stratified sampling per ray.
    returns z: [n_rays, n_coarse]
    """
    t_vals = tf.linspace(near, far, n_coarse + 1)  # [n_coarse+1]
    lower = t_vals[:-1]
    upper = t_vals[1:]
    u = tf.random.uniform([n_rays, n_coarse], dtype=near.dtype)
    z = lower + u * (upper - lower)
    return z

# --------------------------------------------
# Importance sampling (inverse CDF)
# --------------------------------------------
def sample_pdf(z_vals, weights, n_fine):
    """
    Sample n_fine points based on weights.
    z_vals: [n_rays, n_coarse]
    weights: same shape
    returns z_fine: [n_rays, n_fine]
    """
    eps = 1e-5
    pdf = weights + eps
    pdf = pdf / tf.reduce_sum(pdf, axis=-1, keepdims=True)
    cdf = tf.cumsum(pdf, axis=-1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)  # [n_rays, n_coarse+1]
    u = tf.random.uniform([tf.shape(z_vals)[0], n_fine], dtype=z_vals.dtype)
    inds = tf.searchsorted(cdf, u, side='right') - 1
    inds = tf.clip_by_value(inds, 0, tf.shape(z_vals)[1] - 1)
    z_mid = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    bins = tf.concat([z_vals[..., :1], z_mid, z_vals[..., -1:]], axis=-1)
    z_fine = tf.gather(bins, inds, batch_dims=1)
    return z_fine

# --------------------------------------------
# Rays to points (overloaded)
# --------------------------------------------
def rays_to_points(rays, *args):
    """
    Overloaded:
      - rays_to_points(rays, n_points, near, far)
         -> returns (points: [n_rays,n_points,3], distances: [n_points])
      - rays_to_points(rays, n_coarse, n_fine, near, far)
         -> returns (pts_coarse, pts_all, z_all)
    """
    if len(args) == 3:
        n_points, near, far = args
        origins = rays[..., :3]
        dirs    = rays[..., 3:6]
        distances = tf.sort(tf.random.uniform([n_points], minval=near, maxval=far, dtype=rays.dtype))
        points = origins[:, None, :] + dirs[:, None, :] * distances[None, :, None]
        return points, distances
    elif len(args) == 4:
        n_coarse, n_fine, near, far = args
        origins = rays[..., :3]
        dirs    = rays[..., 3:6]
        n_rays  = tf.shape(origins)[0]
        z_coarse = stratified_sampling(near, far, n_coarse, n_rays)
        pts_coarse = origins[:, None, :] + dirs[:, None, :] * z_coarse[..., None]
        weights = tf.ones_like(z_coarse)
        z_fine = sample_pdf(z_coarse, weights, n_fine)
        z_all = tf.sort(tf.concat([z_coarse, z_fine], axis=-1), axis=-1)
        pts_all = origins[:, None, :] + dirs[:, None, :] * z_all[..., None]
        return pts_coarse, pts_all, z_all
    else:
        raise ValueError(f"rays_to_points expects 3 or 4 args, got {len(args)}")

# --------------------------------------------
# Trapezoidal ray attenuation with magnitudes
# --------------------------------------------
def ray_attenuation(attenuations, distances, magnitudes, near, far):
    distances = tf.cast(distances, attenuations.dtype)
    near = tf.cast(near, attenuations.dtype)
    far  = tf.cast(far,  attenuations.dtype)
    mags = tf.cast(tf.reshape(magnitudes, [-1]), attenuations.dtype)
    x = tf.concat([[near], distances, [far]], axis=0)
    y = tf.concat([attenuations[:, :1], attenuations, attenuations[:, -1:]], axis=1)
    dx = tf.reshape(x[1:] - x[:-1], [1, -1])
    trap = (y[:, :-1] + y[:, 1:]) * (dx * 0.5)
    integral = tf.reduce_sum(trap, axis=1, keepdims=True)
    return integral * mags[:, None]

# --------------------------------------------
# Example test
# --------------------------------------------
if __name__ == "__main__":
    rays = tf.convert_to_tensor(np.array([[1.,0.,0.,-1.,0.1,0.1]]), dtype=tf.float64) # not realistic, just for the test
    near = np.float64(0.9)
    far = np.float64(1.1)
    # n_points = np.int32(10)
    # small test
    points, scalars = rays_to_points(rays, 10, near, far)
    #NOTE: there is randomness in the ray generation so you won't get the exact values shown
    print("rays_to_points output:")
    print(points)
    print(scalars)

    attenuations = tf.convert_to_tensor(np.array([[0.5, 0.3, 0.1]]), dtype=tf.float32)
    distances = tf.convert_to_tensor(np.array([0.9, 1.0, 1.1]), dtype=tf.float32)
    magnitudes = tf.convert_to_tensor(np.array([[2.0]]), dtype=tf.float32) # not realistic, just for the test
    result = ray_attenuation(attenuations, distances, magnitudes, near, far)
    # You should get close to exact values here
    print("ray_attenuation output:")
    print(result)

    encoder = PositionEmbeddingEncoder(size=2.0, n_depth=8, embedding_dim=3, input_dim=3)
    test_values = tf.convert_to_tensor(np.array([[0.5, 0.5, 0.5],[0.52,0.500001,0.5]]), dtype=tf.float32)
    # You should get exact values here
    print("get_flattened_position output:")
    for depth in range(1,9):
        print(encoder.get_flattened_position(test_values, depth))


""" Expected output:
rays_to_points output:
tf.Tensor(
[[[ 0.09659314  0.09034069  0.09034069]
  [ 0.08707216  0.09129278  0.09129278]
  [ 0.0536455   0.09463545  0.09463545]
  [ 0.04227487  0.09577251  0.09577251]
  [ 0.01069508  0.09893049  0.09893049]
  [-0.00671153  0.10067115  0.10067115]
  [-0.03321439  0.10332144  0.10332144]
  [-0.06313688  0.10631369  0.10631369]
  [-0.07580807  0.10758081  0.10758081]
  [-0.0987646   0.10987646  0.10987646]]], shape=(1, 10, 3), dtype=float64)
tf.Tensor(
[0.90340686 0.91292784 0.9463545  0.95772513 0.98930492 1.00671153
 1.03321439 1.06313688 1.07580807 1.0987646 ], shape=(10,), dtype=float64)
tf.Tensor([1.00995049], shape=(1,), dtype=float64)
ray_attenuation output:
tf.Tensor([[0.14000005]], shape=(1, 1), dtype=float32)
get_flattened_position output:
tf.Tensor([7 7], shape=(2,), dtype=int32)
tf.Tensor([42 42], shape=(2,), dtype=int32)
tf.Tensor([292 292], shape=(2,), dtype=int32)
tf.Tensor([2184 2184], shape=(2,), dtype=int32)
tf.Tensor([16912 16912], shape=(2,), dtype=int32)
tf.Tensor([133152 133153], shape=(2,), dtype=int32)
tf.Tensor([1056832 1056834], shape=(2,), dtype=int32)
tf.Tensor([8421504 8421509], shape=(2,), dtype=int32)
"""