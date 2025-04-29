import tensorflow as tf
from tensorflow import keras
import numpy as np

# This encoder currently does nothing but allows the pipeline to run
class PositionEmbeddingEncoder(tf.keras.layers.Layer):
    def __init__(self, size, n_depth, embedding_dim, input_dim):
        """
        The positional encoder needs to store the size for use in the call function and create
        keras EmbeddingLayer instances for the embeddings. There should be embedding_dim number of layers,
        each which has an output dimension of embedding_dim and an input_dimension according to the power of 2
        hierarchical decomposition, i.e.

        layer 0 = an embedding layer for a 2x2x2 grid which is 8 for input_dimension
        layer 1 = an embedding layer for a 4x4x4 grid which is 64 for input_dimension
        layer 2 = an embedding layer for a 8x8x8 grid which is 512 for input_dimension
        ... for n_depth layers total (0 based so with n_depth=8 the last layer would be layer 7 with 256x256x256 = 16777216 input_dimension)

        The input dimension of an embedding layer is the same range as the get_flattened_position function returns,
        so calling get_flattened_position(value, 8) for the above network would produce a flattened position in the range
        [0, 16777216)
        """
        super(PositionEmbeddingEncoder, self).__init__()
        self.size = size
        self.n_depth = n_depth
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        #todo fill this in

    def get_flattened_position(self, scaled_values, depth):
        """
        Take in a 3-dimensional position and map it onto multiple 3-D grids, where each grid is the original grid divided
         by a factor of 2

        For example, in a 4x4x4 grid, there are 3 multi-resolutions -- 1x1x1, 2x2x2, 4x4x4
        We will ignore the 1x1x1 dimension because all points would have position 0.
        If the query point came in as 0.5, 0.25, 0.125 for this grid, the positional encoder should produce two values:
        1 (position of the point in a 2x2x2 grid which has 8 positions total)
        6 (position of the point in a 4x4x4 grid which has 64 positions total)

        Input: tensor of float32, shape [N, 3] of N points in 3-D space normalized such that all points are [0,1)
        Output: tensor of int32, shape [N, 1] of N flattened positions
        """
        return None
    def call(self, input):
        """
        This positional encoder should do a few things when called:
        1. Scale the input values by size so that all points are between [0,1) in 3 dimensions
        2. Call get_flattened_position for each depth
        3. Get the embedding for each flattened depth position
        4. Concatenate all the embeddings

        Input: tensor of float32, shape [N, 3] of N points in 3-D space
        Output: tensor of float32, shape [N, N_POS*EMBEDDING_DIM] of N embeddings in positional encoded space
        """
        # todo fill this in

        return input

    def get_output_dim(self):
        return self.n_depth * self.embedding_dim

def rays_to_points(rays, n_points, near, far):
    """
    Computes the sample points for the given rays. First, samples a uniform distribution to produce a set of scalars for n_points.
    Second, multiplies those scalars by the distance between far-near to produce a vector multiplier that is within the region of interest.
    Third, multiplies each ray's directional vector by the vector multiplier and adds it to the ray origin to produce a point.

    :param rays: tensor of float32 [N, 6] for N rays defined by 6 values: A 3D point for the origin of the ray and a 3D vector of the direction of the ray
    :param n_points: int, the number of points to sample along each ray
    :param near: float, the closest distance to scale the ray by
    :param far: float, the farthest distance to scale the ray by
    :return: a two element tuple: (points, scalars) where
      points is [N, n_points, 3] for N rays with n_points each of 3D points
      scalars is [n_points] for the scalar values used to multiply the direction ray vector
        (as these are randomly generated they need to be returned for later use)
    """
    # todo fill this in

    return None

def ray_attenuation(attenuations, distances, magnitudes, near, far):
    """
    Computes the sum of attenuation for each sampled set of attenuations given the distances
     of the points along the source to detector axis and magnitudes of the vectors.

    A basic algorithm for this is to find the difference of distances and multiply the attenuations each by their distance
    to get a weighted sum. Slightly more correct (and what my reference implementation does) is to use the magnitude
    of the ray, compute the total distance from near to far along that ray using the magnitude, interpolate the attenuations
    between points, and then create a weighted sum including the first and last points attenuation as an assumed value
    for the region not covered by the distance between the first and last point sampled.

    You can implement the simpler algorithm for this as with the given geometry it doesn't make much difference.

    :param attenuations: tensor of floats [n_rays, n_points] where n_rays is the number of rays per image used and n_points is the number of points per ray used
    :param distances: tensor of floats [n_points] which is the distance along each ray in the source to detector axis
    :param magnitudes: tensor of floats [n_rays] which is the magnitude of each directional ray
    :param near: float, the closest distance to region of interest
    :param far: float, the farthest distance to region of interest
    :return: tensor of floats [n_rays] which is the attenuation value for each ray
    """
    # todo fill this in

    return None

if __name__ == "__main__":
    rays = tf.convert_to_tensor(np.array([[1.,0.,0.,-1.,0.1,0.1]]), dtype=tf.float64) # not realistic, just for the test
    near = np.float64(0.9)
    far = np.float64(1.1)
    # n_points = np.int32(10)
    # small test
    points, scalars, norms = rays_to_points(rays, 10, near, far)
    #NOTE: there is randomness in the ray generation so you won't get the exact values shown
    print("rays_to_points output:")
    print(points)
    print(scalars)
    print(norms)

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