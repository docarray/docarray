import typing
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tnp

from docarray.computation import AbstractComputationalBackend
from docarray.computation.numpy_backend import NumpyCompBackend


class TensorFlowCompBackend(NumpyCompBackend, AbstractComputationalBackend[tf.Tensor]):
    """
    Computational backend for TensorFlow.
    """

    @staticmethod
    def stack(
        tensors: Union[List['tf.Tensor'], Tuple['tf.Tensor']], dim: int = 0
    ) -> 'tf.Tensor':
        return tnp.stack(tensors, axis=dim)

    @staticmethod
    def n_dim(array: 'tf.Tensor') -> int:
        return tnp.ndim(array)

    @staticmethod
    def squeeze(tensor: 'tf.Tensor') -> 'tf.Tensor':
        return tnp.squeeze(tensor)

    @staticmethod
    def to_numpy(array: 'tf.Tensor') -> 'np.ndarray':
        return array.numpy()

    @staticmethod
    def empty(
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> 'tf.Tensor':
        return tnp.empty(shape=shape, dtype=dtype)

    @staticmethod
    def none_value() -> typing.Any:
        return tf.constant(float('nan'))

    @staticmethod
    def to_device(tensor: 'tf.Tensor', device: str) -> 'tf.Tensor':
        pass

    @staticmethod
    def device(tensor: 'tf.Tensor') -> Optional[str]:
        return tensor.device

    @staticmethod
    def shape(tensor: 'tf.Tensor') -> Tuple[int, ...]:
        return tuple(tnp.shape(tensor))

    @staticmethod
    def reshape(tensor: 'tf.Tensor', shape: Tuple[int, ...]) -> 'tf.Tensor':
        return tf.reshape(tensor, shape)

    @staticmethod
    def detach(tensor: 'tf.Tensor') -> 'tf.Tensor':
        return tf.stop_gradient(tensor)

    @staticmethod
    def dtype(tensor: 'tf.Tensor') -> tf.dtypes:
        return tensor.dtype

    @staticmethod
    def isnan(tensor: 'tf.Tensor') -> 'tf.Tensor':
        return tnp.isnan(tensor)

    @staticmethod
    def minmax_normalize(
        tensor: 'tf.Tensor',
        t_range: Tuple = (0.0, 1.0),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'tf.Tensor':
        a, b = t_range

        t = tf.cast(tensor, tf.float32)
        min_d = x_range[0] if x_range else tnp.min(t, axis=-1, keepdims=True)
        max_d = x_range[1] if x_range else tnp.max(t, axis=-1, keepdims=True)

        i = (b - a) * (t - min_d) / (max_d - min_d + tf.constant(eps) + a)
        print(f"i = {i}")

        normalized = tnp.clip(i, *((a, b) if a < b else (b, a)))
        return tf.cast(normalized, tensor.dtype)
