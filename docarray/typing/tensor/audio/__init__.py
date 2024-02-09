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
from typing import TYPE_CHECKING

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.audio.audio_jax_array import AudioJaxArray  # noqa
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
        AudioTensorFlowTensor,
    )
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

__all__ = ['AudioNdArray', 'AudioTensor', 'AudioJaxArray']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'AudioTorchTensor':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.audio.audio_torch_tensor as lib
    elif name == 'AudioTensorFlowTensor':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.audio.audio_tensorflow_tensor as lib
    elif name == 'AudioJaxArray':
        import_library('jax', raise_error=True)
        import docarray.typing.tensor.audio.audio_jax_array as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
