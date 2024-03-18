from typing import List

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDoc, DocList
from docarray.base_doc import DocArrayResponse
from docarray.documents import ImageDoc, TextDoc
from docarray.typing import NdArray
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.mark.asyncio
async def test_fast_api():
    class Mmdoc(BaseDoc):
        img: ImageDoc
        text: TextDoc
        title: str

    input_doc = Mmdoc(
        img=ImageDoc(tensor=np.zeros((3, 224, 224))), text=TextDoc(), title='hello'
    )

    app = FastAPI()

    @app.post("/doc/", response_model=Mmdoc, response_class=DocArrayResponse)
    async def create_item(doc: Mmdoc) -> Mmdoc:
        return doc

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=input_doc.json())
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200


@pytest.mark.asyncio
async def test_image():
    class InputDoc(BaseDoc):
        img: ImageDoc

    class OutputDoc(BaseDoc):
        embedding_clip: NdArray
        embedding_bert: NdArray

    input_doc = InputDoc(img=ImageDoc(tensor=np.zeros((3, 224, 224))))

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc, response_class=DocArrayResponse)
    async def create_item(doc: InputDoc) -> OutputDoc:
        ## call my fancy model to generate the embeddings
        doc = OutputDoc(
            embedding_clip=np.zeros((100, 1)), embedding_bert=np.zeros((100, 1))
        )
        return doc

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=input_doc.json())
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    doc = OutputDoc.parse_raw(response.content.decode())

    assert isinstance(doc, OutputDoc)
    assert doc.embedding_clip.shape == (100, 1)
    assert doc.embedding_bert.shape == (100, 1)


@pytest.mark.asyncio
async def test_sentence_to_embeddings():
    class InputDoc(BaseDoc):
        text: str

    class OutputDoc(BaseDoc):
        embedding_clip: NdArray
        embedding_bert: NdArray

    input_doc = InputDoc(text='hello')

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc, response_class=DocArrayResponse)
    async def create_item(doc: InputDoc) -> OutputDoc:
        ## call my fancy model to generate the embeddings
        return OutputDoc(
            embedding_clip=np.zeros((100, 1)), embedding_bert=np.zeros((100, 1))
        )

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=input_doc.json())
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    doc = OutputDoc.parse_raw(response.content.decode())

    assert isinstance(doc, OutputDoc)
    assert doc.embedding_clip.shape == (100, 1)
    assert doc.embedding_bert.shape == (100, 1)


@pytest.mark.asyncio
async def test_docarray():
    doc = ImageDoc(tensor=np.zeros((3, 224, 224)))
    docs = DocList[ImageDoc]([doc, doc])

    app = FastAPI()

    @app.post("/doc/", response_class=DocArrayResponse)
    async def func(fastapi_docs: List[ImageDoc]) -> List[ImageDoc]:
        docarray_docs = DocList[ImageDoc].construct(fastapi_docs)
        return list(docarray_docs)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=docs.to_json())
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    docs = DocList[ImageDoc].from_json(response.content.decode())
    assert len(docs) == 2
    assert docs[0].tensor.shape == (3, 224, 224)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_pydantic_v2, reason='Behavior is only available for Pydantic V2'
)
async def test_doclist_directly():
    from fastapi import Body

    doc = ImageDoc(tensor=np.zeros((3, 224, 224)))
    docs = DocList[ImageDoc]([doc, doc])

    app = FastAPI()

    @app.post("/doc/", response_class=DocArrayResponse)
    async def func_embed_false(
        fastapi_docs: DocList[ImageDoc] = Body(embed=False),
    ) -> DocList[ImageDoc]:
        return fastapi_docs

    @app.post("/doc_default/", response_class=DocArrayResponse)
    async def func_default(fastapi_docs: DocList[ImageDoc]) -> DocList[ImageDoc]:
        return fastapi_docs

    @app.post("/doc_embed/", response_class=DocArrayResponse)
    async def func_embed_true(
        fastapi_docs: DocList[ImageDoc] = Body(embed=True),
    ) -> DocList[ImageDoc]:
        return fastapi_docs

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=docs.to_json())
        response_default = await ac.post("/doc_default/", data=docs.to_json())
        response_embed = await ac.post(
            "/doc_embed/",
            json={
                'fastapi_docs': [
                    {'tensor': doc.tensor.tolist()},
                    {'tensor': doc.tensor.tolist()},
                ]
            },
        )
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert response_default.status_code == 200
    assert response_embed.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    docs = DocList[ImageDoc].from_json(response.content.decode())
    assert len(docs) == 2
    assert docs[0].tensor.shape == (3, 224, 224)

    docs_default = DocList[ImageDoc].from_json(response_default.content.decode())
    assert len(docs_default) == 2
    assert docs_default[0].tensor.shape == (3, 224, 224)

    docs_embed = DocList[ImageDoc].from_json(response_embed.content.decode())
    assert len(docs_embed) == 2
    assert docs_embed[0].tensor.shape == (3, 224, 224)
