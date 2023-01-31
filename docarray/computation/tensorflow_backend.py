import typing
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tnp

from docarray.computation import AbstractComputationalBackend
from docarray.typing import TensorFlowTensor


class TensorFlowCompBackend(AbstractComputationalBackend[tf.Tensor]):
    """
    Computational backend for TensorFlow.
    """

    @staticmethod
    def stack(
        tensors: Union[List['TensorFlowTensor'], Tuple['TensorFlowTensor']],
        dim: int = 0,
    ) -> 'TensorFlowTensor':
        return TensorFlowTensor(tnp.stack([t.tensor for t in tensors], axis=dim))

    @staticmethod
    def n_dim(array: 'TensorFlowTensor') -> int:
        return tnp.ndim(array.tensor)

    @staticmethod
    def squeeze(tensor: 'TensorFlowTensor') -> 'TensorFlowTensor':
        return TensorFlowTensor(tnp.squeeze(tensor.tensor))

    @staticmethod
    def to_numpy(array: 'TensorFlowTensor') -> 'np.ndarray':
        return array.tensor.numpy()

    @staticmethod
    def empty(
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> 'TensorFlowTensor':
        return TensorFlowTensor(tnp.empty(shape=shape, dtype=dtype))

    @staticmethod
    def none_value() -> typing.Any:
        return tf.constant(float('nan'))

    @staticmethod
    def to_device(tensor: 'TensorFlowTensor', device: str) -> 'TensorFlowTensor':
        pass

    @staticmethod
    def device(tensor: 'TensorFlowTensor') -> Optional[str]:
        return tensor.device

    @staticmethod
    def shape(tensor: 'TensorFlowTensor') -> Tuple[int, ...]:
        return tuple(tnp.shape(tensor.tensor))

    @staticmethod
    def reshape(
        tensor: 'TensorFlowTensor', shape: Tuple[int, ...]
    ) -> 'TensorFlowTensor':
        return tf.reshape(tensor.tensor, shape)

    @staticmethod
    def detach(tensor: 'TensorFlowTensor') -> 'TensorFlowTensor':
        return TensorFlowTensor(tf.stop_gradient(tensor))

    @staticmethod
    def dtype(tensor: 'TensorFlowTensor') -> tf.dtypes:
        return tensor.tensor.dtype

    @staticmethod
    def isnan(tensor: 'TensorFlowTensor') -> TensorFlowTensor:
        return TensorFlowTensor(tnp.isnan(tensor.tensor))

    @staticmethod
    def minmax_normalize(
        tensor: 'TensorFlowTensor',
        t_range: Tuple = (0.0, 1.0),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'TensorFlowTensor':
        a, b = t_range

        t = tf.cast(tensor.tensor, tf.float32)
        min_d = x_range[0] if x_range else tnp.min(t, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else tnp.max(t, axis=-1, keepdims=True)

        i = (b - a) * (t - min_d) / (max_d - min_d + tf.constant(eps) + a)
        print(f"i = {i}")

        normalized = tnp.clip(i, *((a, b) if a < b else (b, a)))
        return tf.cast(normalized, tensor.tensor.dtype)

    class Retrieval(AbstractComputationalBackend.Retrieval[tf.Tensor]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'TensorFlowTensor',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['tf.Tensor', 'tf.Tensor']:
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
            :param device: the computational device to use,
                can be either `cpu` or a `cuda` device.
            :return: Tuple containing the retrieved values, and their indices.
                Both ar of shape (n_queries, k)
            """
            values = values.tensor
            if device is not None:
                values = values.to(device)
            if len(values.shape) <= 1:
                values = tf.expand_dims(values, axis=0)

            len_values = values.shape[-1] if len(values.shape) > 1 else len(values)
            k = min(k, len_values)

            if not descending:
                values = -values

            result = tf.math.top_k(input=values, k=k, sorted=True)
            res_values = result.values
            res_indices = result.indices

            if not descending:
                res_values = -result.values

            return TensorFlowTensor(res_values), TensorFlowTensor(res_indices)
