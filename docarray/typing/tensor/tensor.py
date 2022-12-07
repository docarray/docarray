from typing import TYPE_CHECKING, Union

from docarray.typing.tensor.ndarray import NdArray

if TYPE_CHECKING:
    from docarray.typing.tensor.torch_tensor import TorchTensor

Tensor = Union[NdArray, 'TorchTensor']
