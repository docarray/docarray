from typing import Union

from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (
        ImageTensorFlowTensor as ImageTFTensor,
    )

ImageTensor = Union[ImageNdArray]  # type: ignore
if tf_available and torch_available:
    ImageTensor = Union[ImageNdArray, ImageTorchTensor, ImageTFTensor]  # type: ignore
elif tf_available:
    ImageTensor = Union[ImageNdArray, ImageTFTensor]  # type: ignore
elif torch_available:
    ImageTensor = Union[ImageNdArray, ImageTorchTensor]  # type: ignore
