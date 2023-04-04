import pytest
import weaviate
from pydantic import Field

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex


@pytest.fixture
def weaviate_client():
    client = weaviate.Client("http://weaviate:8080")
    client.schema.delete_all()
    yield client
    client.schema.delete_all()


def test_column_config(weaviate_client):
    def get_text_field_data_type(store, index_name):
        props = store._client.schema.get(index_name)["properties"]
        text_field = [p for p in props if p["name"] == "text"][0]

        return text_field["dataType"][0]

    class TextDoc(BaseDoc):
        text: str = Field()

    class StringDoc(BaseDoc):
        text: str = Field(col_type="string")

    dbconfig = WeaviateDocumentIndex.DBConfig(index_name="TextDoc")
    store = WeaviateDocumentIndex[TextDoc](db_config=dbconfig)
    assert get_text_field_data_type(store, "TextDoc") == "text"

    dbconfig = WeaviateDocumentIndex.DBConfig(index_name="StringDoc")
    store = WeaviateDocumentIndex[StringDoc](db_config=dbconfig)
    assert get_text_field_data_type(store, "StringDoc") == "string"
