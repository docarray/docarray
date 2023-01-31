import types
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf  # type: ignore

from docarray.computation import AbstractComputationalBackend


class AbstractNumpyBasedBackend(
    AbstractComputationalBackend[Union[np.ndarray, tf.Tensor]]
):
    _module: types.ModuleType

    @classmethod
    def stack(
        cls, tensors: Union[List['np.ndarray'], Tuple['np.ndarray']], dim: int = 0
    ) -> 'np.ndarray':
        return cls._module.stack(tensors, axis=dim)

    @classmethod
    def n_dim(cls, array: 'np.ndarray') -> int:
        return cls._module.ndim(array)

    @classmethod
    def squeeze(cls, tensor: 'np.ndarray') -> 'np.ndarray':
        """
        Returns a tensor with all the dimensions of tensor of size 1 removed.
        """
        return cls._module.squeeze(tensor)

    @classmethod
    def empty(
        cls,
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> 'np.ndarray':
        if cls._module is np and device is not None:
            raise NotImplementedError('Numpy does not support devices (GPU).')
        return cls._module.empty(shape, dtype=dtype)

    @classmethod
    def shape(cls, array: 'np.ndarray') -> Tuple[int, ...]:
        """Get shape of array"""
        return tuple(cls._module.shape(array))

    @classmethod
    def reshape(cls, array: 'np.ndarray', shape: Tuple[int, ...]) -> 'np.ndarray':
        """
        Gives a new shape to array without changing its data.

        :param array: array to be reshaped
        :param shape: the new shape
        :return: a array with the same data and number of elements as array
            but with the specified shape.
        """
        return cls._module.reshape(array, shape)

    @classmethod
    def isnan(cls, tensor: 'np.ndarray') -> 'np.ndarray':
        """Check element-wise for nan and return result as a boolean array"""
        return cls._module.isnan(tensor)
