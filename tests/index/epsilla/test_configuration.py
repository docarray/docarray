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
from docarray.index import EpsillaDocumentIndex
from docarray.typing import NdArray
from tests.index.epsilla.common import epsilla_config
from tests.index.epsilla.fixtures import start_storage  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema1(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = EpsillaDocumentIndex[Schema1](**epsilla_config)

    docs = [Schema1(tens=np.random.random((10,))) for _ in range(10)]

    assert len(index.find(docs[0], limit=30, search_field="tens")[0]) == 0

    index.index(docs)

    doc_found = index.find(docs[0], limit=1, search_field="tens")[0][0]
    assert doc_found.id == docs[0].id

    assert len(index.find(docs[0], limit=30, search_field="tens")[0]) == 10

    class Schema2(BaseDoc):
        tens: NdArray = Field(is_embedding=True, dim=10)

    index = EpsillaDocumentIndex[Schema2](**epsilla_config)

    docs = [Schema2(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert len(index.find(docs[0], limit=30, search_field="tens")[0]) == 10

    class Schema3(BaseDoc):
        tens: NdArray = Field(is_embedding=True)

    with pytest.raises(ValueError, match='The dimension information is missing'):
        EpsillaDocumentIndex[Schema3](**epsilla_config)


def test_incorrect_vector_field():
    class Schema1(BaseDoc):
        tens: NdArray[10]

    with pytest.raises(ValueError, match='Unable to find any vector columns'):
        EpsillaDocumentIndex[Schema1](**epsilla_config)

    class Schema2(BaseDoc):
        tens1: NdArray[10] = Field(is_embedding=True)
        tens2: NdArray[20] = Field(is_embedding=True)

    with pytest.raises(
        ValueError, match='Specifying multiple vector fields is not supported'
    ):
        EpsillaDocumentIndex[Schema2](**epsilla_config)
