import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import DocArray

from docarray.documents import TextDoc


@pytest.mark.asyncio
async def test_fast_api():
    doc = TextDoc(text='some txt')
    docs = DocArray[TextDoc](docs=[doc])
    app = FastAPI()

    @app.post("/doc/")
    async def func(fastapi_docs: DocArray[TextDoc]) -> DocArray[TextDoc]:
        return fastapi_docs

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/doc/", data=docs.json())

    assert response.status_code == 200

    returned_docs = DocArray[TextDoc].from_json(response.content.decode())
    returned_docs.summary()
