from typing import Union

from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.torch_tensor import TorchTensor

Tensor = Union[NdArray, TorchTensor]
