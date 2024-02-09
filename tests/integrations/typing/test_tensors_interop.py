# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import torch

from docarray.typing import (
    AudioNdArray,
    AudioTorchTensor,
    NdArrayEmbedding,
    TorchEmbedding,
)
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing import AudioTensorFlowTensor, TensorFlowEmbedding


def test_torch_tensors_interop():
    t1 = AudioTorchTensor(torch.rand(128))
    t2 = TorchEmbedding(torch.rand(128))

    t_result = t1 + t2
    assert isinstance(t_result, AudioTorchTensor)
    assert isinstance(t_result, torch.Tensor)
    assert t_result.shape == (128,)


@pytest.mark.tensorflow
def test_tensorflow_tensors_interop():
    t1 = AudioTensorFlowTensor(tf.random.normal((128,)))
    t2 = TensorFlowEmbedding(tf.random.normal((128,)))

    t_result = t1.tensor + t2.tensor
    assert isinstance(t_result, tf.Tensor)
    assert t_result.shape == (128,)


def test_np_arrays_interop():
    t1 = AudioNdArray((128,))
    t2 = NdArrayEmbedding((128,))

    t_result = t1 + t2
    assert isinstance(t_result, AudioNdArray)
    assert isinstance(t_result, np.ndarray)
    assert t_result.shape == (128,)
