from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available  # noqa

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401


if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar("T", bound="AnyTensor")
ShapeT = TypeVar('ShapeT')


class AnyTensor(AbstractTensor, Generic[ShapeT]):
    """
    Represents a tensor object that can be used with TensorFlow, PyTorch, and NumPy type.

    ---

    '''python
    from docarray import BaseDoc
    from docarray.typing import AnyTensor


    class MyDoc(BaseDoc):
        tensor: AnyTensor


    # Example usage with TensorFlow:
    import tensorflow as tf

    doc = MyDoc(tensor=tf.zeros(1000, 2))

    # Example usage with PyTorch:
    import torch

    doc = MyDoc(tensor=torch.zeros(1000, 2))

    # Example usage with NumPy:
    import numpy as np

    doc = MyDoc(tensor=np.zeros((1000, 2)))
    '''

    Returns:
        Union[TorchTensor, TensorFlowTensor, NdArray]: The validated and converted tensor.

    Raises:
        TypeError: If the input value is not a compatible type (torch.Tensor, tensorflow.Tensor, numpy.ndarray).

    """

    def __getitem__(self: T, item):
        pass

    def __setitem__(self, index, value):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    @classmethod
    def _docarray_from_native(cls: Type[T], value: Any):
        raise AttributeError('This method should not be called on AnyTensor.')

    @staticmethod
    def get_comp_backend():
        raise AttributeError('This method should not be called on AnyTensor.')

    def to_protobuf(self):
        raise AttributeError('This method should not be called on AnyTensor.')

    def _docarray_to_json_compatible(self):
        raise AttributeError('This method should not be called on AnyTensor.')

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T):
        raise AttributeError('This method should not be called on AnyTensor.')

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: "ModelField",
        config: "BaseConfig",
    ):
        # Check for TorchTensor first, then TensorFlowTensor, then NdArray
        if torch_available:
            if isinstance(value, TorchTensor):
                return value
            elif isinstance(value, torch.Tensor):
                return TorchTensor._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return value
            elif isinstance(value, tf.Tensor):
                return TensorFlowTensor._docarray_from_native(value)  # noqa
        try:
            return NdArray.validate(value, field, config)
        except Exception as e:  # noqa
            print(e)
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
