import typing
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tnp

from docarray.computation import AbstractComputationalBackend
from docarray.computation.numpy_backend import NumpyCompBackend
from docarray.typing import TensorFlowTensor


class TensorFlowCompBackend(NumpyCompBackend, AbstractComputationalBackend[tf.Tensor]):
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
