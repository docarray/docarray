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
from pydantic import Field
from qdrant_client.http import models as rest

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import qdrant, qdrant_config  # noqa: F401


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    number: int


def test_filter_range(qdrant_config):  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        number: int

    index = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            number=i,
        )
        for i in range(10)
    ]
    index.index(index_docs)

    filter_query = rest.Filter(
        must=[rest.FieldCondition(key='number', range=rest.Range(gte=5, lte=7))]
    )
    docs = index.filter(filter_query, limit=5)

    assert len(docs) == 3
