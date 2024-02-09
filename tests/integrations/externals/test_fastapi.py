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
from typing import List

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDoc, DocList
from docarray.base_doc import DocArrayResponse
from docarray.documents import ImageDoc, TextDoc
from docarray.typing import NdArray


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
