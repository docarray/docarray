from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from docarray.computation.abstract_comp_backend import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend
from docarray.typing import JaxArray

if TYPE_CHECKING:
    pass

jax.config.update("jax_enable_x64", True)


def _unsqueeze_if_single_axis(*matrices) -> List[jnp.ndarray]:
    """Unsqueezes tensors that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    pass


def _unsqueeze_if_scalar(t):
    pass


def norm_left(t: jnp.ndarray) -> JaxArray:
    return JaxArray(tensor=t)


def norm_right(t: JaxArray) -> jnp.ndarray:
    return t.tensor


class JaxCompBackend(AbstractNumpyBasedBackend):
    """
    Computational backend for Numpy.
    """

    _module = jnp
    _cast_output: Callable = norm_left
    _get_tensor: Callable = norm_right

    @classmethod
    def to_device(cls, tensor: 'JaxArray', device: str) -> 'JaxArray':
        """Move the tensor to the specified device."""
        if cls.device(tensor) == device:
            return tensor
        else:
            jax_devices = jax.devices(device)
            return cls._cast_output(
                jax.device_put(cls._get_tensor(tensor), jax_devices)
            )

    @classmethod
    def device(cls, tensor: 'JaxArray') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        return cls._get_tensor(tensor).device().platform

    @classmethod
    def to_numpy(cls, array: 'jax.numpy.array') -> 'np.ndarray':
        return np.array(cls._get_tensor(array))

    @classmethod
    def none_value(cls) -> Any:
        """Provide a compatible value that represents None in numpy."""
        return jnp.nan

    @classmethod
    def detach(cls, tensor: 'jax.numpy.array') -> 'jax.numpy.array':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return cls._cast_output(jax.lax.stop_gradient(cls._get_tensor(tensor)))

    @classmethod
    def dtype(cls, tensor: 'JaxArray') -> np.dtype:
        """Get the data type of the tensor."""
        d_type = cls._get_tensor(tensor).dtype
        return d_type.name

    @classmethod
    def minmax_normalize(
        cls,
        tensor: 'JaxArray',
        t_range: Tuple = (0, 1),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'jax.numpy.array':
        """
        Normalize values in `tensor` into `t_range`.

        `tensor` can be a 1D array or a 2D array. When `tensor` is a 2D array, then
        normalization is row-based.

        !!! note

            - with `t_range=(0, 1)` will normalize the min-value of data to 0, max to 1;
            - with `t_range=(1, 0)` will normalize the min-value of data to 1, max value
              of the data to 0.

        :param tensor: the data to be normalized
        :param t_range: a tuple represents the target range.
        :param x_range: a tuple represents tensors range.
        :param eps: a small jitter to avoid divide by zero
        :return: normalized data in `t_range`
        """
        a, b = t_range

        t = jnp.asarray(cls._get_tensor(tensor), jnp.float32)

        min_d = x_range[0] if x_range else jnp.min(t, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else jnp.max(t, axis=-1, keepdims=True)
        r = (b - a) * (t - min_d) / (max_d - min_d + eps) + a

        normalized = jnp.clip(r, *((a, b) if a < b else (b, a)))
        return cls._cast_output(jnp.asarray(normalized, cls._get_tensor(tensor).dtype))

    class Retrieval(AbstractComputationalBackend.Retrieval[jax.numpy.array]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'JaxArray',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['JaxArray', 'JaxArray']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: Torch tensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: Not supported for this backend
            :return: Tuple containing the retrieved values, and their indices.
                Both ar of shape (n_queries, k)
            """
            comp_be = JaxCompBackend
            if device is not None:
                values = comp_be.to_device(values, device)

            values: jnp.ndarray = comp_be._get_tensor(values)

            if len(values.shape) == 1:
                values = jnp.expand_dims(values, axis=0)

            if descending:
                values = -values

            if k >= values.shape[1]:
                idx = values.argsort(axis=1)[:, :k]
                values = jnp.take_along_axis(values, idx, axis=1)
            else:
                idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
                values = jnp.take_along_axis(values, idx_ps, axis=1)
                idx_fs = values.argsort(axis=1)
                idx = jnp.take_along_axis(idx_ps, idx_fs, axis=1)
                values = jnp.take_along_axis(values, idx_fs, axis=1)

            if descending:
                values = -values

            return comp_be._cast_output(values), comp_be._cast_output(idx)

    class Metrics(AbstractComputationalBackend.Metrics[jax.numpy.array]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        def cosine_sim(
            x_mat: 'JaxArray',
            y_mat: 'JaxArray',
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> 'JaxArray':
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid divde by zero
            :param device: Not supported for this backend
            :return: jax.numpy.array  of shape (n_vectors, n_vectors) containing all
                pairwise cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @classmethod
        def euclidean_dist(
            cls,
            x_mat: 'JaxArray',
            y_mat: 'JaxArray',
            device: Optional[str] = None,
        ) -> 'JaxArray':
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid divde by zero
            :param device: Not supported for this backend
            :return: jax.numpy.array  of shape (n_vectors, n_vectors) containing all
                pairwise euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            pass

        @staticmethod
        def sqeuclidean_dist(
            x_mat: 'JaxArray',
            y_mat: 'JaxArray',
            device: Optional[str] = None,
        ) -> 'JaxArray':
            """Pairwise Squared Euclidian distances between all vectors in
            x_mat and y_mat.

            :param x_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: jax.numpy.array of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: Not supported for this backend
            :return: jax.numpy.array  of shape (n_vectors, n_vectors) containing all
                pairwise Squared Euclidian distances.
                The index [i_x, i_y] contains the cosine Squared Euclidian between
                x_mat[i_x] and y_mat[i_y].
            """
