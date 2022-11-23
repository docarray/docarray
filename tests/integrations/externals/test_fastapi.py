import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import Document, Image, Text
from docarray.typing import Tensor


@pytest.mark.asyncio
async def test_fast_api():
    class Mmdoc(Document):
        img: Image
        text: Text
        title: str

    input_doc = Mmdoc(
        img=Image(tensor=np.zeros((3, 224, 224))), text=Text(), title='hello'
    )

    app = FastAPI()

    @app.post("/doc/")
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
    class InputDoc(Document):
        img: Image

    class OutputDoc(Document):
        embedding_clip: Tensor
        embedding_bert: Tensor

    input_doc = InputDoc(img=Image(tensor=np.zeros((3, 224, 224))))

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc)
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


@pytest.mark.asyncio
async def test_sentence_to_embeddings():
    class InputDoc(Document):
        text: str

    class OutputDoc(Document):
        embedding_clip: Tensor
        embedding_bert: Tensor

    input_doc = InputDoc(text='hello')

    app = FastAPI()

    @app.post("/doc/", response_model=OutputDoc)
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
