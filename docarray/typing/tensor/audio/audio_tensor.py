from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available

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


if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar("T", bound="AudioTensor")


class AudioTensor:
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
                return cast(AudioTorchTensor, value)
            elif isinstance(value, torch.Tensor):
                return AudioTorchTensor._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return cast(AudioTensorFlowTensor, value)
            elif isinstance(value, tf.Tensor):
                return AudioTFTensor._docarray_from_native(value)  # noqa
        try:
            return AudioNdArray.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
