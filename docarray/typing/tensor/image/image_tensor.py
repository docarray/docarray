from typing import Union

from docarray.typing.tensor.image.image_ndarray import ImageNdArray

try:
    torch_available = True
    import torch  # noqa: F401

    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor
except ImportError:
    torch_available = False

try:
    tf_available = True
    import tensorflow as tf  # noqa: F401

    from docarray.typing.tensor.image.image_tensorflow_tensor import (
        ImageTensorFlowTensor as ImageTFTensor,
    )
except ImportError:
    tf_available = False

if tf_available and torch_available:
    ImageTensor = Union[ImageNdArray, ImageTorchTensor, ImageTFTensor]  # type: ignore
elif tf_available:
    ImageTensor = Union[ImageNdArray, ImageTFTensor]  # type: ignore
elif tf_available:
    ImageTensor = Union[ImageNdArray, ImageTorchTensor]  # type: ignore
else:
    ImageTensor = Union[ImageNdArray]  # type: ignore
