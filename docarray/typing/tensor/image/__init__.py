import types

from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.typing.tensor.image.image_tensor import ImageTensor
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.image.image_jax_array import ImageJaxArray  # noqa
    from docarray.typing.tensor.image.image_tensorflow_tensor import (  # noqa
        ImageTensorFlowTensor,
    )
    from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor  # noqa

__all__ = ['ImageNdArray', 'ImageTensor']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'ImageTorchTensor':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.image.image_torch_tensor as lib
    elif name == 'ImageTensorFlowTensor':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.image.image_tensorflow_tensor as lib
    elif name == 'ImageJaxArray':
        import_library('jax', raise_error=True)
        import docarray.typing.tensor.image.image_jax_array as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
