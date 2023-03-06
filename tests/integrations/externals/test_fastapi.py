import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDocument
from docarray.base_document import DocumentResponse
from docarray.documents import ImageDoc, TextDoc
from docarray.typing import NdArray


@pytest.mark.asyncio
async def test_fast_api():
    class Mmdoc(BaseDocument):
        img: ImageDoc
        text: TextDoc
        title: str

    input_doc = Mmdoc(
        img=ImageDoc(tensor=np.zeros((3, 224, 224))), text=TextDoc(), title='hello'
    )

    app = FastAPI()

    @app.post("/doc/", response_model=Mmdoc, response_class=DocumentResponse)
    async def create_item(doc: Mmdoc):
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
    class InputDoc(BaseDocument):
        img: ImageDoc

    class OutputDoc(BaseDocument):
        embedding_clip: NdArray
        embedding_bert: NdArray

    input_doc = InputDoc(img=ImageDoc(tensor=np.zeros((3, 224, 224))))

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc, response_class=DocumentResponse)
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
    class InputDoc(BaseDocument):
        text: str

    class OutputDoc(BaseDocument):
        embedding_clip: NdArray
        embedding_bert: NdArray

    input_doc = InputDoc(text='hello')

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc, response_class=DocumentResponse)
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
