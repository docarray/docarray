from typing import Union

from docarray.typing.tensor.ndarray import NdArray
from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401


if torch_available and tf_available:
    AnyTensor = Union[NdArray, TorchTensor, TensorFlowTensor]
elif torch_available:
    AnyTensor = Union[NdArray, TorchTensor]  # type: ignore
elif tf_available:
    AnyTensor = Union[NdArray, TensorFlowTensor]  # type: ignore
else:
    AnyTensor = Union[NdArray]  # type: ignore
