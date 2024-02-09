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
# TODO: enable ruff qa on this file when we figure out why it thinks weaviate_client is
#       redefined at each test that fixture
# ruff: noqa
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex
from tests.index.weaviate.fixture_weaviate import (  # noqa: F401
    start_storage,
    weaviate_client,
)

pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_column_config(weaviate_client):
    def get_text_field_data_type(index, index_name):
        props = index._client.schema.get(index_name)["properties"]
        text_field = [p for p in props if p["name"] == "text"][0]

        return text_field["dataType"][0]

    class TextDoc(BaseDoc):
        text: str = Field()

    class StringDoc(BaseDoc):
        text: str = Field(col_type="string")

    dbconfig = WeaviateDocumentIndex.DBConfig(index_name="TextDoc")
    index = WeaviateDocumentIndex[TextDoc](db_config=dbconfig)
    assert get_text_field_data_type(index, "TextDoc") == "text"

    dbconfig = WeaviateDocumentIndex.DBConfig(index_name="StringDoc")
    index = WeaviateDocumentIndex[StringDoc](db_config=dbconfig)
    assert get_text_field_data_type(index, "StringDoc") == "string"


def test_index_name():
    class TextDoc(BaseDoc):
        text: str = Field()

    class StringDoc(BaseDoc):
        text: str = Field(col_type="string")

    index = WeaviateDocumentIndex[TextDoc]()
    assert index.index_name == TextDoc.__name__

    index = WeaviateDocumentIndex[StringDoc]()
    assert index.index_name == StringDoc.__name__

    index = WeaviateDocumentIndex[StringDoc](index_name='BaseDoc')
    assert index.index_name == 'BaseDoc'

    index = WeaviateDocumentIndex[StringDoc](index_name='index_name')
    assert index.index_name == 'Index_name'
