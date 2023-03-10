import types
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

from docarray.computation import AbstractComputationalBackend

T = TypeVar('T')


class AbstractNumpyBasedBackend(AbstractComputationalBackend[T], ABC):
    """
    Abstract base class for computational backends that are based on numpy.
    This includes numpy (np) itself and tensorflow.experimental.numpy (tnp).
    The overlap of those two is gathered in this abstract backend. Other functions
    should be defined in corresponding subclasses.
    """

    _module: types.ModuleType
    # The method _get_tensor() transforms the input of the backends methods to a
    # handleable type that the backends _module can work with, whereas _cast_output()
    # casts the output of a methods back to the original input type. This is especially
    # relevant w.r.t. the TensorFlowTensor class:
    # If a TensorFlowTensor instance is input to a function, we first want to transform
    # it to a tf.Tensor, since the tf.Tensor is what the TensorFlowBackend's _module
    # (tnp) works on. If the function returns a tf.Tensor, we want to cast it back to a
    # TensorFlowTensor.
    _cast_output: Callable
    _get_tensor: Callable

    @classmethod
    def stack(cls, tensors: Union[List[T], Tuple[T]], dim: int = 0) -> T:
        """Stack a list of tensors along a new axis."""
        t = [cls._get_tensor(t) for t in tensors]
        return cls._cast_output(cls._module.stack(t, axis=dim))

    @classmethod
    def n_dim(cls, array: T) -> int:
        """Get the number of the array dimensions."""
        return cls._module.ndim(cls._get_tensor(array))

    @classmethod
    def squeeze(cls, tensor: T) -> T:
        """
        Returns a tensor with all the dimensions of tensor of size 1 removed.
        """
        return cls._cast_output(cls._module.squeeze(cls._get_tensor(tensor)))

    @classmethod
    def empty(
        cls,
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> T:
        if cls._module is np and device is not None:
            raise NotImplementedError('Numpy does not support devices (GPU).')
        return cls._cast_output(cls._module.empty(shape, dtype=dtype))

    @classmethod
    def shape(cls, array: T) -> Tuple[int, ...]:
        """Get shape of array"""
        return tuple(cls._module.shape(cls._get_tensor(array)))

    @classmethod
    def reshape(cls, array: T, shape: Tuple[int, ...]) -> T:
        """
        Gives a new shape to array without changing its data.

        :param array: array to be reshaped
        :param shape: the new shape
        :return: a array with the same data and number of elements as array
            but with the specified shape.
        """
        return cls._cast_output(cls._module.reshape(cls._get_tensor(array), shape))

    @classmethod
    def isnan(cls, tensor: T) -> T:
        """Check element-wise for nan and return result as a boolean array"""
        return cls._cast_output(cls._module.isnan(cls._get_tensor(tensor)))

    @classmethod
    def copy(cls, tensor: 'T') -> 'T':
        """return a copy/clone of the tensor"""
        return cls._cast_output(cls._module.array(cls._get_tensor(tensor), copy=True))
