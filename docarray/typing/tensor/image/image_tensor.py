from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor
from docarray.typing.tensor.torch_tensor import TorchTensor
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.image.image_tensorflow_tensor import (
        ImageTensorFlowTensor,
    )


if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


T = TypeVar("T", bound="ImageTensor")


class ImageTensor(AbstractImageTensor):
    """
    Represents an image tensor object that can be used with TensorFlow, PyTorch, and NumPy type.

    ---

    '''python
    from docarray import BaseDoc
    from docarray.typing import ImageTensor


    class MyImageDoc(BaseDoc):
        image: ImageTensor


    # Example usage with TensorFlow:
    import tensorflow as tf

    doc = MyImageDoc(image=tf.zeros((1000, 2)))

    # Example usage with PyTorch:
    import torch

    doc = MyImageDoc(image=torch.zeros((1000, 2)))

    # Example usage with NumPy:
    import numpy as np

    doc = MyImageDoc(image=np.zeros((1000, 2)))
    '''

    Returns:
        Union[ImageTorchTensor, ImageTensorFlowTensor, ImageNdArray]: The validated and converted image tensor.

    Raises:
        TypeError: If the input type is not one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray].
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
        raise AttributeError('This method should not be called on ImageTensor.')

    @staticmethod
    def get_comp_backend():
        raise AttributeError('This method should not be called on ImageTensor.')

    def to_protobuf(self):
        raise AttributeError('This method should not be called on ImageTensor.')

    def _docarray_to_json_compatible(self):
        raise AttributeError('This method should not be called on ImageTensor.')

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T):
        raise AttributeError('This method should not be called on ImageTensor.')

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
        if torch_available:
            if isinstance(value, TorchTensor):
                return cast(ImageTorchTensor, value)
            elif isinstance(value, torch.Tensor):
                return ImageTorchTensor._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return cast(ImageTensorFlowTensor, value)
            elif isinstance(value, tf.Tensor):
                return ImageTensorFlowTensor._docarray_from_native(value)  # noqa
        try:
            return ImageNdArray.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
