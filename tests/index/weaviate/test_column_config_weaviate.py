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
