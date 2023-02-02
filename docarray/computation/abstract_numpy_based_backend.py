import types
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

from docarray.computation import AbstractComputationalBackend

T = TypeVar('T')


class AbstractNumpyBasedBackend(AbstractComputationalBackend[T], ABC):
    _module: types.ModuleType

    # _norm_left() and _norm_right() are functions to transform the input/output
    # from cls_A -> cls_B and back depending on the subclass. This is especially
    # important for the TensorFlowTensor class:
    # If a TensorFlowTensor instance is input to a function, we first want to
    # transform it to a tf.Tensor, since the tf.Tensor is what the _module of
    # AbstractNumpyBasedBackend works on. Vice versa for the output.
    _norm_left: Callable
    _norm_right: Callable

    @classmethod
    def stack(cls, tensors: Union[List[T], Tuple[T]], dim: int = 0) -> T:
        """Stack a list of tensors along a new axis."""
        norm_right = [cls._norm_right(t) for t in tensors]
        return cls._norm_left(cls._module.stack(norm_right, axis=dim))

    @classmethod
    def n_dim(cls, array: T) -> int:
        """Get the number of the array dimensions."""
        return cls._module.ndim(cls._norm_right(array))

    @classmethod
    def squeeze(cls, tensor: T) -> T:
        """
        Returns a tensor with all the dimensions of tensor of size 1 removed.
        """
        return cls._norm_left(cls._module.squeeze(cls._norm_right(tensor)))

    @classmethod
    def empty(
        cls,
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> T:
        if cls._module is np and device is not None:
            raise NotImplementedError('Numpy does not support devices (GPU).')
        return cls._norm_left(cls._module.empty(shape, dtype=dtype))

    @classmethod
    def shape(cls, array: T) -> Tuple[int, ...]:
        """Get shape of array"""
        return tuple(cls._module.shape(cls._norm_right(array)))

    @classmethod
    def reshape(cls, array: T, shape: Tuple[int, ...]) -> T:
        """
        Gives a new shape to array without changing its data.

        :param array: array to be reshaped
        :param shape: the new shape
        :return: a array with the same data and number of elements as array
            but with the specified shape.
        """
        return cls._norm_left(cls._module.reshape(cls._norm_right(array), shape))

    @classmethod
    def isnan(cls, tensor: T) -> T:
        """Check element-wise for nan and return result as a boolean array"""
        return cls._norm_left(cls._module.isnan(cls._norm_right(tensor)))
