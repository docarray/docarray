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
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis, tmp_index_name  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10)

    index = RedisDocumentIndex[Schema](host='localhost')

    docs = [Schema(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10


def test_configure_index(tmp_index_name):
    class Schema(BaseDoc):
        tens: NdArray[100] = Field(space='cosine')
        title: str
        year: int

    types = {'id': 'TAG', 'tens': 'VECTOR', 'title': 'TEXT', 'year': 'NUMERIC'}
    index = RedisDocumentIndex[Schema](host='localhost', index_name=tmp_index_name)

    attr_bytes = index._client.ft(index.index_name).info()['attributes']
    attr = [[byte.decode() for byte in sublist] for sublist in attr_bytes]

    assert len(Schema.__fields__) == len(attr)
    for field, attr in zip(Schema.__fields__, attr):
        assert field in attr and types[field] in attr


def test_runtime_config():
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10)

    index = RedisDocumentIndex[Schema](host='localhost')
    assert index._runtime_config.batch_size == 100

    index.configure(batch_size=10)
    assert index._runtime_config.batch_size == 10
