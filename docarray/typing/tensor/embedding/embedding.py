from typing import Any, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.typing.tensor.tensor import AnyTensor
from docarray.utils._internal.misc import (  # noqa
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp  # type: ignore

    from docarray.typing.tensor.embedding.jax_array import JaxArrayEmbedding
    from docarray.typing.tensor.jaxarray import JaxArray  # noqa: F401

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


T = TypeVar("T", bound="AnyEmbedding")


class AnyEmbedding(AnyTensor, EmbeddingMixin):
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
    type(doc.embedding)  # TensorFlowEmbedding

    # Example usage with PyTorch:
    import torch

    doc = MyEmbeddingDoc(embedding=torch.zeros(1000, 2))
    type(doc.embedding)  # TorchEmbedding

    # Example usage with NumPy:
    import numpy as np

    doc = MyEmbeddingDoc(embedding=np.zeros((1000, 2)))
    type(doc.embedding)  # NdArrayEmbedding
    '''
    ---

    Raises:
        TypeError: If the type of the value is not one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray]
    """

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
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
        if jax_available:
            if isinstance(value, JaxArray):
                return cast(JaxArrayEmbedding, value)
            elif isinstance(value, jnp.ndarray):
                return JaxArrayEmbedding._docarray_from_native(value)  # noqa
        try:
            return NdArrayEmbedding._docarray_validate(value)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
