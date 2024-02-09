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
from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import qdrant, qdrant_config  # noqa: F401

from qdrant_client.http import models


def test_external_collection_without_generated_vectors(qdrant_config):
    class Restaurant(BaseDoc):
        city: str
        price: float
        cuisine_vector: NdArray[4]

    qdrant_config.collection_name = 'test'
    doc_index = QdrantDocumentIndex[Restaurant](qdrant_config)
    qdrant_client = doc_index._client

    qdrant_client.recreate_collection(
        collection_name='test',
        vectors_config={
            'cuisine_vector': models.VectorParams(
                size=4, distance=models.Distance.COSINE
            )
        },
    )

    qdrant_client.upsert(
        collection_name='test',
        points=[
            models.PointStruct(
                id=1,
                vector={'cuisine_vector': [0.05, 0.61, 0.76, 0.74]},
                payload={
                    'city': 'Berlin',
                    'price': 1.99,
                },
            ),
            models.PointStruct(
                id=2,
                vector={'cuisine_vector': [0.19, 0.81, 0.75, 0.11]},
                payload={
                    'city': 'Berlin',
                    'price': 1.99,
                },
            ),
            models.PointStruct(
                id=3,
                vector={'cuisine_vector': [0.36, 0.55, 0.47, 0.94]},
                payload={
                    'city': 'Moscow',
                    'price': 1.99,
                },
            ),
        ],
    )

    results = doc_index.find(
        query=[0.36, 0.55, 0.47, 0.94],
        search_field='cuisine_vector',
        limit=3,
    )

    assert results is not None
