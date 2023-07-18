from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.utils._internal.misc import (
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor
    from docarray.typing.tensor.torch_tensor import TorchTensor

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (
        AudioTensorFlowTensor,
    )
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp  # type: ignore

    from docarray.typing.tensor.audio.audio_jax_array import AudioJaxArray
    from docarray.typing.tensor.jaxarray import JaxArray

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar("T", bound="AudioTensor")


class AudioTensor(AnyTensor, AbstractAudioTensor):
    """
    Represents an audio tensor object that can be used with TensorFlow, PyTorch, and NumPy type.

    ---
    '''python
    from docarray import BaseDoc
    from docarray.typing import AudioTensor


    class MyAudioDoc(BaseDoc):
        tensor: AudioTensor


    # Example usage with TensorFlow:
    import tensorflow as tf

    doc = MyAudioDoc(tensor=tf.zeros(1000, 2))
    type(doc.tensor) # AudioTensorFlowTensor

    # Example usage with PyTorch:
    import torch

    doc = MyAudioDoc(tensor=torch.zeros(1000, 2))
    type(doc.tensor) # AudioTorchTensor

    # Example usage with NumPy:
    import numpy as np

    doc = MyAudioDoc(tensor=np.zeros((1000, 2)))
    type(doc.tensor) # AudioNdArray
    '''
    ---

    Raises:
        TypeError: If the input value is not a compatible type (torch.Tensor, tensorflow.Tensor, numpy.ndarray).

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
                return cast(AudioTorchTensor, value)
            elif isinstance(value, torch.Tensor):
                return AudioTorchTensor._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return cast(AudioTensorFlowTensor, value)
            elif isinstance(value, tf.Tensor):
                return AudioTensorFlowTensor._docarray_from_native(value)  # noqa
        if jax_available:
            if isinstance(value, JaxArray):
                return cast(AudioJaxArray, value)
            elif isinstance(value, jnp.ndarray):
                return AudioJaxArray._docarray_from_native(value)  # noqa
        try:
            return AudioNdArray.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
