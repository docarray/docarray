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


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(is_embedding=True)


def test_persist(tmp_index_name):
    query = SimpleDoc(tens=np.random.random((10,)))

    # create index
    index = MilvusDocumentIndex[SimpleDoc](index_name=tmp_index_name)

    index_name = index.index_name

    assert index.num_docs() == 0

    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert index.num_docs() == 10
    find_results_before = index.find(query, limit=5)

    # load existing index
    index = MilvusDocumentIndex[SimpleDoc](index_name=index_name)
    assert index.num_docs() == 10
    find_results_after = index.find(query, limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # add new data
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert index.num_docs() == 15
