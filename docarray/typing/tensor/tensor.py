from typing import Union

from docarray.typing.tensor.ndarray import NdArray

try:
    import torch  # noqa: F401
except ImportError:
    AnyTensor = Union[NdArray]  # type: ignore

else:
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    AnyTensor = Union[NdArray, TorchTensor]  # type: ignore
