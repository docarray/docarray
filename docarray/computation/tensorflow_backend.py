import typing
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore

from docarray.computation import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend
from docarray.typing import TensorFlowTensor


def _unsqueeze_if_single_axis(*matrices: tf.Tensor) -> List[tf.Tensor]:
    """Unsqueezes tensors that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    unsqueezed = []
    for m in matrices:
        if len(m.shape) == 1:
            unsqueezed.append(tf.expand_dims(m, axis=0))
        else:
            unsqueezed.append(m)
    return unsqueezed


def _unsqueeze_if_scalar(t: tf.Tensor):
    if len(t.shape) == 0:  # avoid scalar output
        t = tf.expand_dims(t, 0)
    return t


def norm_left(t: tf.Tensor) -> TensorFlowTensor:
    return TensorFlowTensor(tensor=t)


def norm_right(t: TensorFlowTensor) -> tf.Tensor:
    return t.tensor


class TensorFlowCompBackend(AbstractNumpyBasedBackend[TensorFlowTensor]):
    """
    Computational backend for TensorFlow.
    """

    _module = tnp
    _norm_left: Callable = norm_left
    _norm_right: Callable = norm_right

    @classmethod
    def to_numpy(cls, array: 'TensorFlowTensor') -> 'np.ndarray':
        return cls._norm_right(array).numpy()

    @classmethod
    def none_value(cls) -> typing.Any:
        """Provide a compatible value that represents None in numpy."""
        return tf.constant(float('nan'))

    @classmethod
    def to_device(cls, tensor: 'TensorFlowTensor', device: str) -> 'TensorFlowTensor':
        """Move the tensor to the specified device."""
        if cls.device(tensor) == device:
            return tensor
        else:
            with tf.device(device):
                return cls._norm_left(tf.identity(cls._norm_right(tensor)))

    @classmethod
    def device(cls, tensor: 'TensorFlowTensor') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        return cls._norm_right(tensor).device

    @classmethod
    def detach(cls, tensor: 'TensorFlowTensor') -> 'TensorFlowTensor':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return cls._norm_left(tf.stop_gradient(cls._norm_right(tensor)))

    @classmethod
    def dtype(cls, tensor: 'TensorFlowTensor') -> tf.dtypes:
        """Get the data type of the tensor."""
        return cls._norm_right(tensor).dtype

    @classmethod
    def minmax_normalize(
        cls,
        tensor: 'TensorFlowTensor',
        t_range: Tuple = (0.0, 1.0),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'TensorFlowTensor':
        a, b = t_range

        t = tf.cast(cls._norm_right(tensor), tf.float32)
        min_d = x_range[0] if x_range else tnp.min(t, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else tnp.max(t, axis=-1, keepdims=True)

        i = (b - a) * (t - min_d) / (max_d - min_d + tf.constant(eps) + a)

        normalized = tnp.clip(i, *((a, b) if a < b else (b, a)))
        return cls._norm_left(tf.cast(normalized, tensor.tensor.dtype))

    class Retrieval(AbstractComputationalBackend.Retrieval[TensorFlowTensor]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'TensorFlowTensor',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['TensorFlowTensor', 'TensorFlowTensor']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: TensorFlowTensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: the computational device to use.
            :return: Tuple of TensorFlowTensors containing the retrieved values, and
                their indices. Both are of shape (n_queries, k)
            """
            comp_be = TensorFlowCompBackend
            if device is not None:
                values = comp_be.to_device(values, device)

            tf_values: tf.Tensor = comp_be._norm_right(values)
            if len(tf_values.shape) <= 1:
                tf_values = tf.expand_dims(tf_values, axis=0)

            len_tf_values = (
                tf_values.shape[-1] if len(tf_values.shape) > 1 else len(tf_values)
            )
            k = min(k, len_tf_values)

            if not descending:
                tf_values = -tf_values

            result = tf.math.top_k(input=tf_values, k=k, sorted=True)
            res_values = result.values
            res_indices = result.indices

            if not descending:
                res_values = -result.values

            return comp_be._norm_left(res_values), comp_be._norm_left(res_indices)

    class Metrics(AbstractComputationalBackend.Metrics[TensorFlowTensor]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        def cosine_sim(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor  of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            x_mat_tf: tf.Tensor = TensorFlowCompBackend._norm_right(x_mat)
            y_mat_tf: tf.Tensor = TensorFlowCompBackend._norm_right(y_mat)

            with tf.device(device):
                x_mat_tf, y_mat_tf = _unsqueeze_if_single_axis(x_mat_tf, y_mat_tf)

                a_n = tf.linalg.normalize(x_mat_tf, axis=1)[1]
                b_n = tf.linalg.normalize(y_mat_tf, axis=1)[1]
                a_norm = x_mat_tf / tf.clip_by_value(
                    a_n, clip_value_min=eps, clip_value_max=tf.float32.max
                )
                b_norm = y_mat_tf / tf.clip_by_value(
                    b_n, clip_value_min=eps, clip_value_max=tf.float32.max
                )
                sims = tf.squeeze(tf.linalg.matmul(a_norm, tf.transpose(b_norm)))
                sims = _unsqueeze_if_scalar(sims)

            return TensorFlowCompBackend._norm_left(sims)

        @staticmethod
        def euclidean_dist(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            x_mat_tf: tf.Tensor = TensorFlowCompBackend._norm_right(x_mat)
            y_mat_tf: tf.Tensor = TensorFlowCompBackend._norm_right(y_mat)

            with tf.device(device):
                x_mat_tf, y_mat_tf = _unsqueeze_if_single_axis(x_mat_tf, y_mat_tf)

                dists = tf.squeeze(
                    tf.norm(tf.subtract(x_mat_tf, y_mat_tf), axis=-1, ord='euclidean')
                )
                dists = _unsqueeze_if_scalar(dists)

            return TensorFlowCompBackend._norm_left(dists)

        @staticmethod
        def sqeuclidean_dist(
            x_mat: 'TensorFlowTensor',
            y_mat: 'TensorFlowTensor',
            device: Optional[str] = None,
        ) -> 'TensorFlowTensor':
            """Pairwise Squared Euclidian distances between all vectors
                in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            dists = TensorFlowCompBackend.Metrics.euclidean_dist(x_mat, y_mat)
            squared: tf.Tensor = tf.math.square(
                TensorFlowCompBackend._norm_right(dists)
            )

            return TensorFlowCompBackend._norm_left(squared)
