from typing import Union

from docarray.typing.tensor.ndarray import NdArray

try:
    import torch  # noqa: F401
except ImportError:
    Tensor = Union[NdArray]  # type: ignore

else:
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    Tensor = Union[NdArray, TorchTensor]  # type: ignore
