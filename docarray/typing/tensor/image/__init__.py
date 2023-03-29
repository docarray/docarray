from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
        ImageTensorFlowTensor,
    )
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor  # noqa

__all__ = ['ImageNdArray', 'ImageTensor']


def __getattr__(name: str):
    if name not in __all__:
        __all__.append(name)

    if name == 'ImageTorchTensor':
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.image.image_torch_tensor import (  # noqa
            ImageTorchTensor,
        )

        return ImageTorchTensor

    elif name == 'ImageTensorFlowTensor':
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
            ImageTensorFlowTensor,
        )

        return ImageTensorFlowTensor
