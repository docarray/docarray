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
import numpy as np
import pytest
import torch

from docarray import BaseDoc
from docarray.typing import AnyTensor, NdArray, TorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.computation.tensorflow_backend import tnp
    from docarray.typing import TensorFlowTensor


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(1000, 2), TorchTensor, torch.Tensor),
        (np.zeros((1000, 2)), NdArray, np.ndarray),
    ],
)
def test_torch_ndarray_to_any_tensor(tensor, cls_audio_tensor, cls_tensor):
    class MyTensorDoc(BaseDoc):
        tensor: AnyTensor

    doc = MyTensorDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert doc.tensor.shape == (1000, 2)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_to_any_tensor():
    class MyTensorDoc(BaseDoc):
        tensor: AnyTensor

    doc = MyTensorDoc(tensor=tf.zeros((1000, 2)))
    assert isinstance(doc.tensor, TensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1000, 2)))


def test_equals_type():
    # see https://github.com/docarray/docarray/pull/1739
    assert not (TorchTensor == type)
