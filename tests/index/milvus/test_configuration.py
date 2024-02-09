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
from pydantic import Field

from docarray import BaseDoc
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray
from tests.index.milvus.fixtures import start_storage, tmp_index_name  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema1(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = MilvusDocumentIndex[Schema1]()

    docs = [Schema1(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10

    class Schema2(BaseDoc):
        tens: NdArray = Field(is_embedding=True, dim=10)

    index = MilvusDocumentIndex[Schema2]()

    docs = [Schema2(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10

    class Schema3(BaseDoc):
        tens: NdArray = Field(is_embedding=True)

    with pytest.raises(ValueError, match='The dimension information is missing'):
        MilvusDocumentIndex[Schema3]()


def test_incorrect_vector_field():
    class Schema1(BaseDoc):
        tens: NdArray[10]

    with pytest.raises(ValueError, match='Unable to find any vector columns'):
        MilvusDocumentIndex[Schema1]()

    class Schema2(BaseDoc):
        tens1: NdArray[10] = Field(is_embedding=True)
        tens2: NdArray[20] = Field(is_embedding=True)

    with pytest.raises(
        ValueError, match='Specifying multiple vector fields is not supported'
    ):
        MilvusDocumentIndex[Schema2]()


def test_runtime_config():
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10, is_embedding=True)

    index = MilvusDocumentIndex[Schema]()
    assert index._runtime_config.batch_size == 100

    index.configure(batch_size=10)
    assert index._runtime_config.batch_size == 10
