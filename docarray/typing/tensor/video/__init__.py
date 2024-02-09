// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import types

from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.video.video_jax_array import VideoJaxArray  # noqa
    from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa
        VideoTensorFlowTensor,
    )
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

__all__ = ['VideoNdArray', 'VideoTensor']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'VideoTorchTensor':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.video.video_torch_tensor as lib
    elif name == 'VideoTensorFlowTensor':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.video.video_tensorflow_tensor as lib
    elif name == 'VideoJaxArray':
        import_library('jax', raise_error=True)
        import docarray.typing.tensor.video.video_jax_array as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
