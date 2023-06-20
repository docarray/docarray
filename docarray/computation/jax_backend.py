from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from docarray.computation.abstract_comp_backend import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend
from docarray.typing import JaxArray

if TYPE_CHECKING:
    pass


def _unsqueeze_if_single_axis(*matrices) -> List[jnp.ndarray]:
    """Unsqueezes tensors that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    unsqueezed = []
    for m in matrices:
        if len(m.shape) == 1:
            unsqueezed.append(jnp.expand_dims(m, axis=0))
        else:
            unsqueezed.append(m)
    return unsqueezed


def _unsqueeze_if_scalar(t):
    """
    Unsqueezes tensor of a scalar, from shape () to shape (1,).

    :param t: tensor to unsqueeze.
    :return: unsqueezed tf.Tensor
    """
    if len(t.shape) == 0:  # avoid scalar output
        t = jnp.expand_dims(t, 0)
    return t


def _expand_if_single_axis(*matrices: jnp.ndarray) -> List[jnp.ndarray]:
    """Expands arrays that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be expanded
    :return: List of the input matrices,
        where single axis matrices are expanded at dim 0.
    """
    expanded = []
    for m in matrices:
        if len(m.shape) == 1:
            expanded.append(jnp.expand_dims(m, axis=0))
        else:
            expanded.append(m)
    return expanded


def _expand_if_scalar(arr: jnp.ndarray) -> jnp.ndarray:
    if len(arr.shape) == 0:  # avoid scalar output
        arr = jnp.expand_dims(arr, axis=0)
    return arr


def identity(array: jnp.ndarray) -> jnp.ndarray:
    return array


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

    class Metrics(AbstractComputationalBackend.Metrics[jnp.ndarray]):
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

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divide by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            comp_be = JaxCompBackend
            x_mat_jax: jnp.ndarray = comp_be._get_tensor(x_mat)
            y_mat_jax: jnp.ndarray = comp_be._get_tensor(y_mat)

            x_mat_jax, y_mat_jax = _unsqueeze_if_single_axis(x_mat_jax, y_mat_jax)

            sims = jnp.clip(
                (jnp.dot(x_mat_jax, y_mat_jax.T) + eps)
                / (
                    jnp.outer(
                        jnp.linalg.norm(x_mat_jax, axis=1),
                        jnp.linalg.norm(y_mat_jax, axis=1),
                    )
                    + eps
                ),
                -1,
                1,
            ).squeeze()
            sims = _unsqueeze_if_scalar(sims)

            return comp_be._cast_output(sims)

        @classmethod
        def euclidean_dist(
            cls, x_mat: jnp.ndarray, y_mat: jnp.ndarray, device: Optional[str] = None
        ) -> JaxArray:
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid divde by zero
            :param device: Not supported for this backend
            :return: np.ndarray  of shape (n_vectors, n_vectors) containing all
                pairwise euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            comp_be = JaxCompBackend
            x_mat: jnp.ndarray = comp_be._get_tensor(x_mat)
            y_mat: jnp.ndarray = comp_be._get_tensor(y_mat)
            if device is not None:
                # warnings.warn('`device` is not supported for numpy operations')
                pass

            x_mat, y_mat = _expand_if_single_axis(x_mat, y_mat)

            x_mat = comp_be._cast_output(x_mat)
            y_mat = comp_be._cast_output(y_mat)

            dists = _expand_if_scalar(
                jnp.sqrt(
                    comp_be._get_tensor(cls.sqeuclidean_dist(x_mat, y_mat))
                ).squeeze()
            )

            return comp_be._cast_output(dists)

        @staticmethod
        def sqeuclidean_dist(
            x_mat: jnp.ndarray,
            y_mat: jnp.ndarray,
            device: Optional[str] = None,
        ) -> JaxArray:
            """Pairwise Squared Euclidian distances between all vectors in
            x_mat and y_mat.

            :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: Not supported for this backend
            :return: np.ndarray  of shape (n_vectors, n_vectors) containing all
                pairwise Squared Euclidian distances.
                The index [i_x, i_y] contains the cosine Squared Euclidian between
                x_mat[i_x] and y_mat[i_y].
            """
            comp_be = JaxCompBackend
            x_mat_jax: jnp.ndarray = comp_be._get_tensor(x_mat)
            y_mat_jax: jnp.ndarray = comp_be._get_tensor(y_mat)
            eps: float = 1e-7  # avoid problems with numerical inaccuracies

            if device is not None:
                pass
                # warnings.warn('`device` is not supported for numpy operations')

            x_mat_jax, y_mat_jax = _expand_if_single_axis(x_mat_jax, y_mat_jax)

            dists = (
                jnp.sum(y_mat_jax**2, axis=1)
                + jnp.sum(x_mat_jax**2, axis=1)[:, jnp.newaxis]
                - 2 * jnp.dot(x_mat_jax, y_mat_jax.T)
            ).squeeze()

            # remove numerical artifacts
            dists = jnp.where(np.logical_and(dists < 0, dists > -eps), 0, dists)
            dists = _expand_if_scalar(dists)
            return comp_be._cast_output(dists)
