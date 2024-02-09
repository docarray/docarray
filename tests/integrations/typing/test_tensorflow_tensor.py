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
import pytest

from docarray import BaseDoc
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore

    from docarray.typing import TensorFlowEmbedding, TensorFlowTensor


@pytest.mark.tensorflow
def test_set_tensorflow_tensor():
    class MyDocument(BaseDoc):
        t: TensorFlowTensor

    doc = MyDocument(t=tf.zeros((3, 224, 224)))

    assert isinstance(doc.t, TensorFlowTensor)
    assert isinstance(doc.t.tensor, tf.Tensor)
    assert tnp.allclose(doc.t.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_set_tf_embedding():
    class MyDocument(BaseDoc):
        embedding: TensorFlowEmbedding

    doc = MyDocument(embedding=tf.zeros((128,)))

    assert isinstance(doc.embedding, TensorFlowTensor)
    assert isinstance(doc.embedding, TensorFlowEmbedding)
    assert isinstance(doc.embedding.tensor, tf.Tensor)
    assert tnp.allclose(doc.embedding.tensor, tf.zeros((128,)))
