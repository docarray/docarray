from typing import Any, Dict, List, Optional, Union, ClassVar
import json
import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDoc, DocList
from docarray.base_doc import DocArrayResponse
from docarray.documents import ImageDoc, TextDoc
from docarray.typing import NdArray, AnyTensor, ImageUrl

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

    doc = ImageDoc(tensor=np.zeros((3, 224, 224)), url='url')
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
        embed_content_json = {'fastapi_docs': json.loads(docs.to_json())}
        response_embed = await ac.post(
            "/doc_embed/",
            json=embed_content_json,
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


@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_pydantic_v2, reason='Behavior is only available for Pydantic V2'
)
async def test_doclist_complex_schema():
    from fastapi import Body

    class Nested2Doc(BaseDoc):
        value: str
        classvar: ClassVar[str] = 'classvar2'

    class Nested1Doc(BaseDoc):
        nested: Nested2Doc
        classvar: ClassVar[str] = 'classvar1'

    class CustomDoc(BaseDoc):
        tensor: Optional[AnyTensor] = None
        url: ImageUrl
        num: float = 0.5
        num_num: List[float] = [1.5, 2.5]
        lll: List[List[List[int]]] = [[[5]]]
        fff: List[List[List[float]]] = [[[5.2]]]
        single_text: TextDoc
        texts: DocList[TextDoc]
        d: Dict[str, str] = {'a': 'b'}
        di: Optional[Dict[str, int]] = None
        u: Union[str, int]
        lu: List[Union[str, int]] = [0, 1, 2]
        tags: Optional[Dict[str, Any]] = None
        nested: Nested1Doc
        classvar: ClassVar[str] = 'classvar'

    docs = DocList[CustomDoc](
        [
            CustomDoc(
                num=3.5,
                num_num=[4.5, 5.5],
                url='photo.jpg',
                lll=[[[40]]],
                fff=[[[40.2]]],
                d={'b': 'a'},
                texts=DocList[TextDoc]([TextDoc(text='hey ha', embedding=np.zeros(3))]),
                single_text=TextDoc(text='single hey ha', embedding=np.zeros(2)),
                u='a',
                lu=[3, 4],
                nested=Nested1Doc(nested=Nested2Doc(value='hello world')),
            )
        ]
    )

    app = FastAPI()

    @app.post("/doc/", response_class=DocArrayResponse)
    async def func_embed_false(
        fastapi_docs: DocList[CustomDoc] = Body(embed=False),
    ) -> DocList[CustomDoc]:
        for doc in fastapi_docs:
            doc.tensor = np.zeros((10, 10, 10))
            doc.di = {'a': 2}

        return fastapi_docs

    @app.post("/doc_default/", response_class=DocArrayResponse)
    async def func_default(fastapi_docs: DocList[CustomDoc]) -> DocList[CustomDoc]:
        for doc in fastapi_docs:
            doc.tensor = np.zeros((10, 10, 10))
            doc.di = {'a': 2}
        return fastapi_docs

    @app.post("/doc_embed/", response_class=DocArrayResponse)
    async def func_embed_true(
        fastapi_docs: DocList[CustomDoc] = Body(embed=True),
    ) -> DocList[CustomDoc]:
        for doc in fastapi_docs:
            doc.tensor = np.zeros((10, 10, 10))
            doc.di = {'a': 2}
        return fastapi_docs

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=docs.to_json())
        response_default = await ac.post("/doc_default/", data=docs.to_json())
        embed_content_json = {'fastapi_docs': json.loads(docs.to_json())}
        response_embed = await ac.post(
            "/doc_embed/",
            json=embed_content_json,
        )
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response.status_code == 200
    assert response_default.status_code == 200
    assert response_embed.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    docs_response = DocList[CustomDoc].from_json(response.content.decode())
    assert len(docs_response) == 1
    assert docs_response[0].url == 'photo.jpg'
    assert docs_response[0].num == 3.5
    assert docs_response[0].num_num == [4.5, 5.5]
    assert docs_response[0].lll == [[[40]]]
    assert docs_response[0].lu == [3, 4]
    assert docs_response[0].fff == [[[40.2]]]
    assert docs_response[0].di == {'a': 2}
    assert docs_response[0].d == {'b': 'a'}
    assert len(docs_response[0].texts) == 1
    assert docs_response[0].texts[0].text == 'hey ha'
    assert docs_response[0].texts[0].embedding.shape == (3,)
    assert docs_response[0].tensor.shape == (10, 10, 10)
    assert docs_response[0].u == 'a'
    assert docs_response[0].single_text.text == 'single hey ha'
    assert docs_response[0].single_text.embedding.shape == (2,)

    docs_default = DocList[CustomDoc].from_json(response_default.content.decode())
    assert len(docs_default) == 1
    assert docs_default[0].url == 'photo.jpg'
    assert docs_default[0].num == 3.5
    assert docs_default[0].num_num == [4.5, 5.5]
    assert docs_default[0].lll == [[[40]]]
    assert docs_default[0].lu == [3, 4]
    assert docs_default[0].fff == [[[40.2]]]
    assert docs_default[0].di == {'a': 2}
    assert docs_default[0].d == {'b': 'a'}
    assert len(docs_default[0].texts) == 1
    assert docs_default[0].texts[0].text == 'hey ha'
    assert docs_default[0].texts[0].embedding.shape == (3,)
    assert docs_default[0].tensor.shape == (10, 10, 10)
    assert docs_default[0].u == 'a'
    assert docs_default[0].single_text.text == 'single hey ha'
    assert docs_default[0].single_text.embedding.shape == (2,)

    docs_embed = DocList[CustomDoc].from_json(response_embed.content.decode())
    assert len(docs_embed) == 1
    assert docs_embed[0].url == 'photo.jpg'
    assert docs_embed[0].num == 3.5
    assert docs_embed[0].num_num == [4.5, 5.5]
    assert docs_embed[0].lll == [[[40]]]
    assert docs_embed[0].lu == [3, 4]
    assert docs_embed[0].fff == [[[40.2]]]
    assert docs_embed[0].di == {'a': 2}
    assert docs_embed[0].d == {'b': 'a'}
    assert len(docs_embed[0].texts) == 1
    assert docs_embed[0].texts[0].text == 'hey ha'
    assert docs_embed[0].texts[0].embedding.shape == (3,)
    assert docs_embed[0].tensor.shape == (10, 10, 10)
    assert docs_embed[0].u == 'a'
    assert docs_embed[0].single_text.text == 'single hey ha'
    assert docs_embed[0].single_text.embedding.shape == (2,)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_pydantic_v2, reason='Behavior is only available for Pydantic V2'
)
async def test_simple_directly():
    app = FastAPI()

    @app.post("/doc_list/", response_class=DocArrayResponse)
    async def func_doc_list(fastapi_docs: DocList[TextDoc]) -> DocList[TextDoc]:
        return fastapi_docs

    @app.post("/doc_single/", response_class=DocArrayResponse)
    async def func_doc_single(fastapi_doc: TextDoc) -> TextDoc:
        return fastapi_doc

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response_doc_list = await ac.post(
            "/doc_list/", data=json.dumps([{"text": "text"}])
        )
        response_single = await ac.post(
            "/doc_single/", data=json.dumps({"text": "text"})
        )
        resp_doc = await ac.get("/docs")
        resp_redoc = await ac.get("/redoc")

    assert response_doc_list.status_code == 200
    assert response_single.status_code == 200
    assert resp_doc.status_code == 200
    assert resp_redoc.status_code == 200

    docs = DocList[TextDoc].from_json(response_doc_list.content.decode())
    assert len(docs) == 1
    assert docs[0].text == 'text'

    doc = TextDoc.from_json(response_single.content.decode())
    assert doc == 'text'
