from typing import Union

from docarray.typing.tensor.ndarray import NdArray

try:
    import torch  # noqa: F401

    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    is_torch_available = True
except ImportError:
    is_torch_available = False

try:
    import tensorflow as tf  # type: ignore # noqa: F401

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401

    is_tf_available = True
except (ImportError, TypeError):
    is_tf_available = False


if is_torch_available and is_tf_available:
    AnyTensor = Union[NdArray, TorchTensor, TensorFlowTensor]
elif is_torch_available:
    AnyTensor = Union[NdArray, TorchTensor]  # type: ignore
elif is_tf_available:
    AnyTensor = Union[NdArray, TensorFlowTensor]  # type: ignore
else:
    AnyTensor = Union[NdArray]  # type: ignore
