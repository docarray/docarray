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

from docarray import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import NdArray, TorchTensor


class NpDoc(BaseDoc):
    embedding: NdArray[3, 4]
    embedding_no_shape: NdArray


class TorchDoc(BaseDoc):
    embedding: TorchTensor[3, 4]
    embedding_no_shape: TorchTensor


def test_np_schema():
    schema = NpDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert (
        schema['properties']['embedding']['example']
        == orjson_dumps(np.zeros([3, 4])).decode()
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'


def test_torch_schema():
    schema = TorchDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert (
        schema['properties']['embedding']['example']
        == orjson_dumps(np.zeros([3, 4])).decode()
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'


@pytest.mark.tensorflow
def test_tensorflow_schema():
    from docarray.typing import TensorFlowTensor

    class TensorflowDoc(BaseDoc):
        embedding: TensorFlowTensor[3, 4]
        embedding_no_shape: TensorFlowTensor

    schema = TensorflowDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert (
        schema['properties']['embedding']['example']
        == orjson_dumps(np.zeros([3, 4])).decode()
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
