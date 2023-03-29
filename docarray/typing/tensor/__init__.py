from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.image import ImageNdArray, ImageTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.tensor.video import VideoNdArray

__all__ = [
    'NdArray',
    'AnyTensor',
    'AnyEmbedding',
    'NdArrayEmbedding',
    'ImageNdArray',
    'ImageTensor',
    'AudioNdArray',
    'VideoNdArray',
]

from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F401
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F401

    __all__.extend(
        [
            'TorchEmbedding',
            'TorchTensor',
            'ImageTorchTensor',
            'AudioTorchTensor',
            'VideoTorchTensor',
        ]
    )

    torch_available = is_torch_available()


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa: F401

    __all__.extend(
        [
            'TensorFlowEmbedding',
            'TensorFlowTensor',
            'ImageTensorFlowTensor',
            'AudioTensorFlowTensor',
            'VideoTensorFlowTensor',
        ]
    )
