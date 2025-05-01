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
        # 为每一个深度创建一个 Embedding 层
        # 第 i 层网格大小为 (2**(i+1))^3，总位置数为 (2**(i+1))**3
        self.embeddings = []
        for d in range(1, n_depth + 1):
            grid_size = 2 ** d
            vocab_size = grid_size ** 3  # 扁平化后的位置总数
            emb = tf.keras.layers.Embedding(
                input_dim=vocab_size,  # 词表大小
                output_dim=embedding_dim  # 嵌入向量维度
            )
            self.embeddings.append(emb)

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

        位置编码器需要存储尺寸信息，以便在 call 函数中使用，并为嵌入创建 Keras 的 Embedding 层实例。应该创建 embedding_dim 个嵌入层，
        每个嵌入层的输出维度为 embedding_dim，输入维度则根据 2 的幂次进行分层划分，即所谓的分层分解（hierarchical decomposition）：
        第 0 层：用于一个 2×2×2 网格的嵌入层，输入维度为 8
        第 1 层：用于一个 4×4×4 网格的嵌入层，输入维度为 64
        第 2 层：用于一个 8×8×8 网格的嵌入层，输入维度为 512
        …… 以此类推，总共构建 n_depth 层（以 0 为起始索引，因此当 n_depth=8 时，最后一层是第 7 层，其输入维度为 256×256×256 = 16,777,216）
        每个嵌入层的输入维度必须与 get_flattened_position 函数所返回的位置索引范围一致。
        因此，当你调用 get_flattened_position(value, 8) 时，输出的扁平化位置索引范围应为 [0, 16,777,216)。

        """
        # todo
        """
        将归一化到 [0,1) 的坐标映射到指定深度的扁平化索引：
        1. 计算每个轴上的网格索引：floor(value * grid_size)
        2. 扁平化：按 z, y, x 顺序展开：
           index = z_idx * (grid_size^2) + y_idx * grid_size + x_idx
        scaled_values: Tensor, shape [N,3]，值在 [0,1)
        depth: int，当前深度（1 到 n_depth）
        返回：Tensor, shape [N]，dtype int32
        """
        grid_size = 2 ** depth
        # 轴向索引
        idx = tf.floor(scaled_values * tf.cast(grid_size, scaled_values.dtype))
        idx = tf.cast(idx, tf.int32)  # 转为整数索引
        x_idx, y_idx, z_idx = idx[:, 0], idx[:, 1], idx[:, 2]

        # 按 z, y, x 顺序扁平化
        flattened = z_idx * (grid_size ** 2) + y_idx * grid_size + x_idx
        return flattened  # shape [N]

    def call(self, input):
        """
        This positional encoder should do a few things when called:
        1. Scale the input values by size so that all points are between [0,1) in 3 dimensions
        2. Call get_flattened_position for each depth
        3. Get the embedding for each flattened depth position
        4. Concatenate all the embeddings

        Input: tensor of float32, shape [N, 3] of N points in 3-D space
        Output: tensor of float32, shape [N, N_POS*EMBEDDING_DIM] of N embeddings in positional encoded space

        这个位置编码器在被调用时应完成以下几项操作：
        1.将输入值按尺寸进行缩放，使所有点在三维空间中的坐标都归一化到 [0, 1) 范围内
        2.针对每一层深度，调用 get_flattened_position 函数
        3.获取每个扁平化位置对应的嵌入向量
        4.将所有嵌入向量拼接在一起
        输入： 一个形状为 [N, 3] 的 float32 类型张量，表示 N 个三维空间中的点
        输出： 一个形状为 [N, N_POS × EMBEDDING_DIM] 的 float32 类型张量，表示 N 个点在位置编码空间中的嵌入向量
        """
        # todo fill this in
        """
        前向计算：
        1. 将坐标从实际空间[-size/2, size/2)归一化到[0,1)
        2. 针对每一层深度，计算扁平化位置并通过对应的 Embedding
        3. 将所有层的嵌入向量在最后一维上拼接
        inputs: Tensor, shape [N,3]
        返回：Tensor, shape [N, n_depth*embedding_dim]
        """
        # 1. 先把坐标缩放到 [0,1)
        scaled = (input + self.size / 2.0) / self.size
        # 2. 截断到 [0, 1)，防止正好等于 1 导致 floor(grid_size*1)==grid_size
        scaled = tf.clip_by_value(scaled, 0.0, 1.0 - 1e-6)

        embeddings = []
        # 遍历每一层深度
        for i, emb_layer in enumerate(self.embeddings, start=1):
            # 计算扁平化位置
            pos = self.get_flattened_position(scaled, depth=i)  # [N]
            # 取对应的 embedding 向量
            e = emb_layer(pos)  # [N, embedding_dim]
            embeddings.append(e)

        # 拼接所有深度的 embedding
        output = tf.concat(embeddings, axis=-1)  # [N, n_depth*embedding_dim]
        return output

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

    计算给定射线的采样点。该过程包括三个步骤：
    首先，从均匀分布中采样，生成 n_points 个标量值；
    然后，将这些标量乘以 (far - near)，得到一个位于感兴趣区域内的距离向量；
    最后，将每条射线的方向向量与距离向量逐元素相乘，并加上射线的起点，从而得到采样点。

    参数说明：
        rays：形状为 [N, 6] 的 float32 类型张量，表示 N 条射线；每条射线由 6 个值构成，分别是一个三维起点和一个三维方向向量
        n_points：整数，表示每条射线上要采样的点数
        near：浮点数，表示采样时与起点的最近距离
        far：浮点数，表示采样时与起点的最远距离

    返回值：返回一个包含两个元素的元组 (points, scalars)：
        points：形状为 [N, n_points, 3] 的张量，表示每条射线上采样得到的 3D 空间点
        scalars：形状为 [n_points] 的张量，表示用于缩放方向向量的随机标量值（因其是随机生成的，需返回以供后续使用）
    """
    # todo fill this in
    """
        采样射线上的点并返回对应的标量和方向向量模长：
        - rays: Tensor [N,6]，前 3 维是起点，后 3 维是方向向量
        - n_points: 每条射线上采样的点数
        - near, far: 浮点，采样区间的起止距离
        返回: (points, scalars, magnitudes)
          points: Tensor [N, n_points, 3]
          scalars: Tensor [n_points]
          magnitudes: Tensor [N]
        """
    """
        返回：
          points: Tensor [N, n_points, 3]
          scalars: Tensor [n_points]
        """
    origins = rays[..., :3]  # [N,3]
    dirs = rays[..., 3:6]  # [N,3]

    # 在 [near, far) 区间采样 n_points 个标量，并升序
    scalars = tf.random.uniform(
        shape=(n_points,),
        minval=near,
        maxval=far,
        dtype=rays.dtype
    )
    scalars = tf.sort(scalars)

    # 广播
    origins_exp = tf.expand_dims(origins, axis=1)  # [N,1,3]
    dirs_exp = tf.expand_dims(dirs, axis=1)  # [N,1,3]
    scalars_exp = tf.reshape(scalars, (1, n_points, 1))  # [1,n_points,1]

    # 计算采样点：origin + direction * scalar
    points = origins_exp + dirs_exp * scalars_exp  # [N,n_points,3]

    return points, scalars

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

    根据采样点在源到探测器轴线上的距离，以及方向向量的模长，计算每条射线采样点的衰减总和。
    一种基本算法是：计算相邻采样点之间的距离差值，并将每个衰减值乘以对应的距离，以获得加权和。
    更精确的做法（也是参考实现中采用的方法）是：使用射线方向向量的模长，计算该射线从 near 到 far 的总距离，然后对衰减值进行插值，并将首尾两个采样点的衰减值也计入加权和，用以近似覆盖未被采样的区域。
    考虑到给定的几何结构，上述两种方法在结果上差别不大，因此可以选择实现较为简单的版本。

    参数说明：
        attenuations：形状为 [n_rays, n_points] 的浮点数张量，其中 n_rays 是每张图像中的射线数量，n_points 是每条射线上的采样点数
        distances：形状为 [n_points] 的浮点数张量，表示每条射线沿源-探测器轴线方向上各个采样点的距离
        magnitudes：形状为 [n_rays] 的浮点数张量，表示每条射线方向向量的模长
        near：浮点数，表示感兴趣区域的最近距离
        far：浮点数，表示感兴趣区域的最远距离

    返回值：
        形状为 [n_rays] 的浮点数张量，表示每条射线的衰减总值

    """
    # todo fill this in
    """
        简单计算射线衰减：
        对相邻采样点之间的距离差做加权和。
        attenuations: Tensor [n_rays, n_points]
        distances:    Tensor [n_points]
        magnitudes:   Tensor [n_rays]（可以用来放缩结果）
        near, far:    浮点

        返回: Tensor [n_rays]，每条射线的衰减值
        """
    # 将 distances、near、far、magnitudes 全都转换到和 attenuations 一样的 dtype
    distances = tf.cast(distances, attenuations.dtype)
    near = tf.cast(near, attenuations.dtype)
    far = tf.cast(far, attenuations.dtype)
    magnitudes = tf.cast(magnitudes, attenuations.dtype)

    # 构造积分网格 x = [near, distances..., far]
    x = tf.concat([[near], distances, [far]], axis=0)  # [n_points+2]
    # y 两端填充相同的 attenuation
    y = tf.concat([attenuations[:, :1], attenuations, attenuations[:, -1:]], axis=1)  # [N, n_points+2]

    dx = x[1:] - x[:-1]  # [n_points+1]
    dx = tf.reshape(dx, (1, -1))  # [1, n_points+1]

    trap = (y[:, :-1] + y[:, 1:]) * (dx * 0.5)
    integral = tf.reduce_sum(trap, axis=1, keepdims=True)  # [N,1]

    return integral

if __name__ == "__main__":
    rays = tf.convert_to_tensor(np.array([[1.,0.,0.,-1.,0.1,0.1]]), dtype=tf.float64) # not realistic, just for the test
    near = np.float64(0.9)
    far = np.float64(1.1)
    # n_points = np.int32(10)
    # small test
    points, scalars, norms = rays_to_points(rays, 10, near, far)
    #NOTE: there is randomness in the ray generation so you won't get the exact values shown
    # 射线采样有随机性，不必追求输出值逐一相同
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