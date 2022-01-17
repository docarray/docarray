from collections import defaultdict
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.testclient import TestClient

from docarray import DocumentArray, Document
from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray
from docarray.score import NamedScore


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


class IdMatch(BaseModel):
    id: str
    matches: Optional[List['IdMatch']]


@app.post('/single', response_model=IdOnly)
async def create_item(item: PydanticDocument):
    return Document.from_pydantic_model(item).to_pydantic_model()


@app.post('/multi', response_model=List[TextOnly])
async def create_item(items: PydanticDocumentArray):
    da = DocumentArray.from_pydantic_model(items)
    da.texts = [f'hello_{j}' for j in range(len(da))]
    return da.to_pydantic_model()


@app.get('/get_match', response_model=List[IdMatch], response_model_exclude_none=True)
async def get_match_id_only():
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([len(da), 3])
    da.match(da)
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

    response = client.get('/get_match')
    r = response.json()
    assert isinstance(r, list)
    assert isinstance(r[0], dict)
    assert len(r[0].keys()) == 2
    assert len(r[-1].keys()) == 2
    assert list(r[0].keys()) == ['id', 'matches']
    assert list(r[0]['matches'][0].keys()) == ['id']
    assert list(r[-1]['matches'][-1].keys()) == ['id']


def test_nested_doc_pydantic():
    d = Document(chunks=[Document()], matches=[Document()])
    pd = d.to_pydantic_model()
    assert isinstance(pd, BaseModel)
    assert isinstance(pd.chunks[0], BaseModel)
    assert isinstance(pd.matches[0], BaseModel)


def test_match_to_from_pydantic():
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([len(da), 3])
    da.match(da, exclude_self=True)
    dap = da.to_pydantic_model()
    da_r = DocumentArray.from_pydantic_model(dap)
    assert da_r[0].matches[0].scores['cosine']
    assert isinstance(da_r[0].matches[0].scores, defaultdict)
    assert isinstance(da_r[0].matches[0].scores['random_score'], NamedScore)


def test_pydantic_from_dict_ndarray():
    from docarray import Document
    import numpy as np
    doc = Document()
    doc.embedding = np.random.rand(2, 2)
    pydantic_model1 = doc.to_pydantic_model()
    assert len(pydantic_model1.embedding) == 2
    for emb_dim in pydantic_model1.embedding:
        assert len(emb_dim) == 2
    pydantic_model2 = PydanticDocument(**doc.to_dict())
    print(pydantic_model2.embedding)

