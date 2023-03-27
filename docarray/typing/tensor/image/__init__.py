from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor

__all__ = ['ImageNdArray', 'ImageTensor']

from docarray.utils._internal.misc import import_library

torch_tensors = ['ImageTorchTensor']
tf_tensors = ['ImageTensorFlowTensor']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.image.image_torch_tensor import (  # noqa
            ImageTorchTensor,
        )

        __all__.extend(torch_tensors)

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
            ImageTensorFlowTensor,
        )

        __all__.extend(tf_tensors)
