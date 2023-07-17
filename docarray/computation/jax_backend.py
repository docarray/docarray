from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np

from docarray.computation.abstract_comp_backend import AbstractComputationalBackend
from docarray.computation.abstract_numpy_based_backend import AbstractNumpyBasedBackend
from docarray.typing import JaxArray
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
else:
    jax = import_library('jax', raise_error=True)
    jnp = jax.numpy


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


def norm_left(t: jnp.ndarray) -> JaxArray:
    return JaxArray(tensor=t)


def norm_right(t: JaxArray) -> jnp.ndarray:
    return t.tensor


class JaxCompBackend(AbstractNumpyBasedBackend):
    """
    Computational backend for Jax.
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
    def to_numpy(cls, array: 'JaxArray') -> 'np.ndarray':
        return cls._get_tensor(array).__array__()

    @classmethod
    def none_value(cls) -> Any:
        """Provide a compatible value that represents None in JAX."""
        return jnp.nan

    @classmethod
    def detach(cls, tensor: 'JaxArray') -> 'JaxArray':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return cls._cast_output(jax.lax.stop_gradient(cls._get_tensor(tensor)))

    @classmethod
    def dtype(cls, tensor: 'JaxArray') -> jnp.dtype:
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
    ) -> 'JaxArray':
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
        :param eps: a small jitter to avoid dividing by zero
        :return: normalized data in `t_range`
        """
        a, b = t_range

        t = jnp.asarray(cls._get_tensor(tensor), jnp.float32)

        min_d = x_range[0] if x_range else jnp.min(t, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else jnp.max(t, axis=-1, keepdims=True)
        r = (b - a) * (t - min_d) / (max_d - min_d + eps) + a

        normalized = jnp.clip(r, *((a, b) if a < b else (b, a)))
        return cls._cast_output(jnp.asarray(normalized, cls._get_tensor(tensor).dtype))

    @classmethod
    def equal(cls, tensor1: 'JaxArray', tensor2: 'JaxArray') -> bool:
        """
        Check if two tensors are equal.

        :param tensor1: the first tensor
        :param tensor2: the second tensor
        :return: True if two tensors are equal, False otherwise.
            If one or more of the inputs is not a TensorFlowTensor, return False.
        """
        t1, t2 = getattr(tensor1, 'tensor', None), getattr(tensor2, 'tensor', None)
        if isinstance(t1, jnp.ndarray) and isinstance(t2, jnp.ndarray):
            # mypy doesn't know that tf.is_tensor implies that t1, t2 are not None
            return t1.shape == t2.shape and jnp.all(jnp.equal(t1, t1))  # type: ignore
        return False

    class Retrieval(AbstractComputationalBackend.Retrieval[JaxArray]):
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
            Returns the k smallest values in `values` along with their indices.
            Can also be used to retrieve the k largest values,
            by setting the `descending` flag.

            :param values: Jax tensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: Not supported for this backend
            :return: Tuple containing the retrieved values, and their indices.
                Both are of shape (n_queries, k)
            """
            comp_be = JaxCompBackend
            if device is not None:
                values = comp_be.to_device(values, device)

            jax_values: jnp.ndarray = comp_be._get_tensor(values)

            if len(jax_values.shape) == 1:
                jax_values = jnp.expand_dims(jax_values, axis=0)

            if descending:
                jax_values = -jax_values

            if k >= jax_values.shape[1]:
                idx = jax_values.argsort(axis=1)[:, :k]
                jax_values = jnp.take_along_axis(jax_values, idx, axis=1)
            else:
                idx_ps = jax_values.argpartition(kth=k, axis=1)[:, :k]
                jax_values = jnp.take_along_axis(jax_values, idx_ps, axis=1)
                idx_fs = jax_values.argsort(axis=1)
                idx = jnp.take_along_axis(idx_ps, idx_fs, axis=1)
                jax_values = jnp.take_along_axis(jax_values, idx_fs, axis=1)

            if descending:
                jax_values = -jax_values

            return comp_be._cast_output(jax_values), comp_be._cast_output(idx)

    class Metrics(AbstractComputationalBackend.Metrics[JaxArray]):
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
            :param eps: a small jitter to avoid dividing by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: JaxArray of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            comp_be = JaxCompBackend
            x_mat_jax: jnp.ndarray = comp_be._get_tensor(x_mat)
            y_mat_jax: jnp.ndarray = comp_be._get_tensor(y_mat)

            x_mat_jax, y_mat_jax = _expand_if_single_axis(x_mat_jax, y_mat_jax)

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
            sims = _expand_if_scalar(sims)

            return comp_be._cast_output(sims)

        @classmethod
        def euclidean_dist(
            cls, x_mat: JaxArray, y_mat: JaxArray, device: Optional[str] = None
        ) -> JaxArray:
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: jnp.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: jnp.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param eps: a small jitter to avoid dividing by zero
            :param device: Not supported for this backend
            :return: JaxArray  of shape (n_vectors, n_vectors) containing all
                pairwise euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            comp_be = JaxCompBackend
            x_mat_jax: jnp.ndarray = comp_be._get_tensor(x_mat)
            y_mat_jax: jnp.ndarray = comp_be._get_tensor(y_mat)
            if device is not None:
                # warnings.warn('`device` is not supported for numpy operations')
                pass

            x_mat_jax, y_mat_jax = _expand_if_single_axis(x_mat_jax, y_mat_jax)

            x_mat_jax_arr: JaxArray = comp_be._cast_output(x_mat_jax)
            y_mat_jax_arr: JaxArray = comp_be._cast_output(y_mat_jax)

            dists = _expand_if_scalar(
                jnp.sqrt(
                    comp_be._get_tensor(
                        cls.sqeuclidean_dist(x_mat_jax_arr, y_mat_jax_arr)
                    )
                ).squeeze()
            )

            return comp_be._cast_output(dists)

        @staticmethod
        def sqeuclidean_dist(
            x_mat: JaxArray,
            y_mat: JaxArray,
            device: Optional[str] = None,
        ) -> JaxArray:
            """Pairwise Squared Euclidian distances between all vectors in
            x_mat and y_mat.

            :param x_mat: jnp.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: jnp.ndarray of shape (n_vectors, n_dim), where n_vectors is
                the number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: Not supported for this backend
            :return: JaxArray  of shape (n_vectors, n_vectors) containing all
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
