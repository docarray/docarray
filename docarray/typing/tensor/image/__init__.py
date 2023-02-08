from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor

__all__ = ['ImageNdArray', 'ImageTensor']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor  # noqa

    __all__.extend(['ImageTorchTensor'])

try:
    import tensorflow as tf  # noqa: F401
except (ImportError, TypeError):
    pass
else:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
        ImageTensorFlowTensor,
    )

    __all__.extend(['ImageTensorFlowTensor'])
