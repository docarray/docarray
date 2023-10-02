from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal.misc import (  # noqa
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

    from docarray.typing.tensor.jaxarray import JaxArray  # noqa: F401

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401


if TYPE_CHECKING:

    # Below is the hack to make the type checker happy. But `AnyTensor` is defined as a class and with same underlying
    # behavior as `Union[TorchTensor, TensorFlowTensor, NdArray]` so it should be fine to use `AnyTensor` as
    # the type for `tensor` field in `BaseDoc` class.
    AnyTensor = Union[NdArray]
    if torch_available and tf_available and jax_available:
        AnyTensor = Union[NdArray, TorchTensor, TensorFlowTensor, JaxArray]  # type: ignore
    elif torch_available and tf_available:
        AnyTensor = Union[NdArray, TorchTensor, TensorFlowTensor]  # type: ignore
    elif tf_available and jax_available:
        AnyTensor = Union[NdArray, TensorFlowTensor, JaxArray]  # type: ignore
    elif torch_available and jax_available:
        AnyTensor = Union[NdArray, TorchTensor, JaxArray]  # type: ignore
    elif tf_available:
        AnyTensor = Union[NdArray, TensorFlowTensor]  # type: ignore
    elif torch_available:
        AnyTensor = Union[NdArray, TorchTensor]  # type: ignore
    elif jax_available:
        AnyTensor = Union[NdArray, JaxArray]  # type: ignore

else:

    T = TypeVar("T", bound="AnyTensor")
    ShapeT = TypeVar('ShapeT')

    class AnyTensor(AbstractTensor, Generic[ShapeT]):
        """
        Represents a tensor object that can be used with TensorFlow, PyTorch, and NumPy type.
        !!! note:
            when doing type checking (mypy or pycharm type checker), this class will actually be replace by a Union of the three
            tensor types. You can reason about this class as if it was a Union.

        ```python
        from docarray import BaseDoc
        from docarray.typing import AnyTensor


        class MyTensorDoc(BaseDoc):
            tensor: AnyTensor


        # Example usage with TensorFlow:
        # import tensorflow as tf

        # doc = MyTensorDoc(tensor=tf.zeros(1000, 2))

        # Example usage with PyTorch:
        import torch

        doc = MyTensorDoc(tensor=torch.zeros(1000, 2))

        # Example usage with NumPy:
        import numpy as np

        doc = MyTensorDoc(tensor=np.zeros((1000, 2)))
        ```
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
            raise RuntimeError(f'This method should not be called on {cls}.')

        @staticmethod
        def get_comp_backend():
            raise RuntimeError('This method should not be called on AnyTensor.')

        def to_protobuf(self):
            raise RuntimeError(f'This method should not be called on {self.__class__}.')

        def _docarray_to_json_compatible(self):
            raise RuntimeError(f'This method should not be called on {self.__class__}.')

        @classmethod
        def from_protobuf(cls: Type[T], pb_msg: T):
            raise RuntimeError(f'This method should not be called on {cls}.')

        @classmethod
        def _docarray_validate(
            cls: Type[T],
            value: Union[T, np.ndarray, Any],
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
            if jax_available:
                if isinstance(value, JaxArray):
                    return value
                elif isinstance(value, jnp.ndarray):
                    return JaxArray._docarray_from_native(value)  # noqa
            try:
                return NdArray._docarray_validate(value)
            except Exception as e:  # noqa
                print(e)
                pass
            raise TypeError(
                f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
                f"compatible type, got {type(value)}"
            )
