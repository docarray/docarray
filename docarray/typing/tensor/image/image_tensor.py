from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.utils._internal.misc import (
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp  # type: ignore

    from docarray.typing.tensor.image.image_jax_array import ImageJaxArray
    from docarray.typing.tensor.jaxarray import JaxArray

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor
    from docarray.typing.tensor.torch_tensor import TorchTensor

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.image.image_tensorflow_tensor import (
        ImageTensorFlowTensor,
    )
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor


if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


T = TypeVar("T", bound="ImageTensor")


class ImageTensor(AnyTensor, AbstractImageTensor):
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
    type(doc.image) # ImageTensorFlowTensor

    # Example usage with PyTorch:
    import torch

    doc = MyImageDoc(image=torch.zeros((1000, 2)))
    type(doc.image) # ImageTorchTensor

    # Example usage with NumPy:
    import numpy as np

    doc = MyImageDoc(image=np.zeros((1000, 2)))
    type(doc.image) # ImageNdArray
    '''
    ---

    Returns:
        Union[ImageTorchTensor, ImageTensorFlowTensor, ImageNdArray]: The validated and converted image tensor.

    Raises:
        TypeError: If the input type is not one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray].
    """

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
        if jax_available:
            if isinstance(value, JaxArray):
                return cast(ImageJaxArray, value)
            elif isinstance(value, jnp.ndarray):
                return ImageJaxArray._docarray_from_native(value)  # noqa
        try:
            return ImageNdArray.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
