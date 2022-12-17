import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.testclient import TestClient

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite
from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray
from docarray.score import NamedScore


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArraySqlite])
def test_pydantic_doc_da(pytestconfig, da_cls):
    da = da_cls.from_files(
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


IdMatch.update_forward_refs()


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
    response = client.post('/single', content=Document(text='hello').to_json())
    r = response.json()
    assert r['id']
    assert 'text' not in r
    assert len(r) == 1

    response = client.post('/multi', content=DocumentArray.empty(2).to_json())

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
    assert isinstance(da_r[0].matches[0].scores['cosine'], NamedScore)
    assert isinstance(da_r[0].matches[0].scores, defaultdict)
    assert isinstance(da_r[0].matches[0].scores['random_score'], NamedScore)


def test_with_embedding_no_tensor():
    d = Document(embedding=np.random.rand(2, 2))
    PydanticDocument.parse_obj(d.to_pydantic_model().dict())


@pytest.mark.parametrize(
    'tag_value, tag_type',
    [
        (3.0, float),
        ('hello', str),
        ('1', str),
        (True, bool),
        (False, bool),
        ([1, 2, 3], list),
        ({'x': 1}, dict),
    ],
)
@pytest.mark.parametrize('protocol', ['protobuf', 'jsonschema'])
def test_tags_int_float_str_bool(tag_type, tag_value, protocol):
    d = Document(tags={'hello': tag_value})
    dd = d.to_dict(protocol=protocol)['tags']['hello']
    assert dd == tag_value
    assert isinstance(dd, tag_type)

    # now nested tags in dict

    d = Document(tags={'hello': {'world': tag_value}})
    dd = d.to_dict(protocol=protocol)['tags']['hello']['world']
    assert dd == tag_value
    assert isinstance(dd, tag_type)

    # now nested in list
    d = Document(tags={'hello': [tag_value] * 10})
    dd = d.to_dict(protocol=protocol)['tags']['hello'][-1]
    assert dd == tag_value
    assert isinstance(dd, tag_type)


@pytest.mark.parametrize('protocol', ['protobuf', 'jsonschema'])
def test_infinity_no_coercion(protocol):
    # Test for issue #948: https://github.com/docarray/docarray/issues/948
    d = Document()
    d.tags['title'] = 'Infinity'

    d_pydantic = d.to_pydantic_model()
    d_pydantic.tags['title'] = 'Infinity'

    d_json = d.to_json(protocol=protocol)
    assert '"title": "Infinity"' in d_json

    d_dict = d.to_dict(protocol=protocol)
    assert d_dict['tags']['title'] == 'Infinity'


@pytest.mark.parametrize(
    'blob', [None, b'123', bytes(Document()), bytes(bytearray(os.urandom(512 * 4)))]
)
@pytest.mark.parametrize('protocol', ['jsonschema', 'protobuf'])
@pytest.mark.parametrize('to_fn', ['dict', 'json'])
def test_to_from_with_blob(protocol, to_fn, blob):
    d = Document(blob=blob)
    r_d = getattr(Document, f'from_{to_fn}')(
        getattr(d, f'to_{to_fn}')(protocol=protocol), protocol=protocol
    )

    assert d.blob == r_d.blob
    if d.blob:
        assert isinstance(r_d.blob, bytes)


def test_pydantic_not_id():
    _ = PydanticDocument()
