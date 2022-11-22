import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import Document, Image, Text


@pytest.mark.asyncio
async def test_fast_api():
    class Mmdoc(Document):
        img: Image
        text: Text
        title: str

    input_doc = Mmdoc(img=Image(), text=Text(), title='hello')

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
