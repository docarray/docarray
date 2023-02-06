from typing import Union

from docarray.typing.tensor.image.image_ndarray import ImageNdArray

try:
    import torch  # noqa: F401
except ImportError:
    ImageTensor = ImageNdArray

else:
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor

    ImageTensor = Union[ImageNdArray, ImageTorchTensor]  # type: ignore
