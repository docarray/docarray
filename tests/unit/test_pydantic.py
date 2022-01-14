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
    return item


@app.post('/multi', response_model=List[TextOnly])
async def create_item(items: PydanticDocumentArray):
    da = DocumentArray.from_pydantic_model(items)
    da.texts = [f'hello_{j}' for j in range(len(da))]
    return da.to_pydantic_model()


client = TestClient(app)


def test_read_main():
    response = client.post('/single', Document(text='hello').to_json())

    assert response.json()['id']

    response = client.post('/multi', DocumentArray.empty(2).to_json())

    assert isinstance(response.json(), list)
    assert response.json()[0]['text'] == 'hello_0'
    assert response.json()[1]['text'] == 'hello_1'
