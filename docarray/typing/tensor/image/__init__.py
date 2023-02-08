from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor

__all__ = ['ImageNdArray', 'ImageTensor']

from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor  # noqa

    __all__.extend(['ImageTorchTensor'])


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
        ImageTensorFlowTensor,
    )

    __all__.extend(['ImageTensorFlowTensor'])
