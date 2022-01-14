from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.testclient import TestClient

from docarray import DocumentArray, Document
from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray


def test_pydantic_doc_da(pytestconfig):
    da = DocumentArray.from_files(
        [
            f'{pytestconfig.rootdir}/**/*.png',
            f'{pytestconfig.rootdir}/**/*.jpg',
            f'{pytestconfig.rootdir}/**/*.jpeg',
        ]
    )

    assert da
    assert da.get_json_schema(2)
    assert da.from_pydantic_model(da.to_pydantic_model())
    da.embeddings = np.random.random([len(da), 10])
    da_r = da.from_pydantic_model(da.to_pydantic_model())
    assert da_r.embeddings.shape == (len(da), 10)


app = FastAPI()


class IdOnly(BaseModel):
    id: str


class TextOnly(BaseModel):
    text: str


@app.post('/single', response_model=IdOnly)
async def create_item(item: PydanticDocument):
    return Document.from_pydantic_model(item).to_pydantic_model()


@app.post('/multi', response_model=List[TextOnly])
async def create_item(items: PydanticDocumentArray):
    da = DocumentArray.from_pydantic_model(items)
    da.texts = [f'hello_{j}' for j in range(len(da))]
    return da.to_pydantic_model()


client = TestClient(app)


def test_read_main():
    response = client.post('/single', Document(text='hello').to_json())
    r = response.json()
    assert r['id']
    assert 'text' not in r
    assert len(r) == 1

    response = client.post('/multi', DocumentArray.empty(2).to_json())

    r = response.json()
    assert isinstance(r, list)
    assert len(r[0]) == 1
    assert len(r[1]) == 1
    assert r[0]['text'] == 'hello_0'
    assert r[1]['text'] == 'hello_1'
