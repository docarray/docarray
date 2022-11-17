import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import Document, Image, Text


class Mmdoc(Document):
    img: Image
    text: Text
    title: str


@pytest.mark.asyncio
async def test_fast_api():

    app = FastAPI()

    @app.post("/doc/")
    async def create_item(doc: Mmdoc):
        return doc

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # response = await ac.get("/doc")
        response2 = await ac.get("/docs")

    # assert response.status_code == 200
    assert response2.status_code == 200

    # assert response.json() == {"message": "Tomato"}
