from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor

__all__ = ['ImageNdArray', 'ImageTensor']

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
        ImageTensorFlowTensor,
    )
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor  # noqa


def __getattr__(name: str):
    if name == 'ImageTorchTensor':
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.image.image_torch_tensor import (  # noqa
            ImageTorchTensor,
        )

        __all__.append('ImageTorchTensor')
        return ImageTorchTensor

    elif name == 'ImageTensorFlowTensor':
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
            ImageTensorFlowTensor,
        )

        __all__.append('ImageTensorFlowTensor')
        return ImageTensorFlowTensor
