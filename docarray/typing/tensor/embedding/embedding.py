from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.utils._internal.misc import is_tf_available, is_torch_available  # noqa

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.embedding.torch import TorchEmbedding
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401


tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.embedding.tensorflow import TensorFlowEmbedding
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401


if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar("T", bound="AnyEmbedding")


class AnyEmbedding(EmbeddingMixin):
    """
    Represents an embedding tensor object that can be used with TensorFlow, PyTorch, and NumPy type.

    ---

    '''python
    from docarray import BaseDoc
    from docarray.typing import AnyEmbedding


    class MyEmbeddingDoc(BaseDoc):
        embedding: AnyEmbedding


    # Example usage with TensorFlow:
    import tensorflow as tf

    doc = MyEmbeddingDoc(embedding=tf.zeros(1000, 2))

    # Example usage with PyTorch:
    import torch

    doc = MyEmbeddingDoc(embedding=torch.zeros(1000, 2))

    # Example usage with NumPy:
    import numpy as np

    doc = MyEmbeddingDoc(embedding=np.zeros((1000, 2)))
    '''

    Raises:
        TypeError: If the type of the value is not one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray]
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
        if torch_available:
            if isinstance(value, TorchTensor):
                return cast(TorchEmbedding, value)
            elif isinstance(value, torch.Tensor):
                return TorchEmbedding._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return cast(TensorFlowEmbedding, value)
            elif isinstance(value, tf.Tensor):
                return TensorFlowEmbedding._docarray_from_native(value)  # noqa
        try:
            return NdArrayEmbedding.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
