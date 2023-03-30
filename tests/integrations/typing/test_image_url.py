from docarray import BaseDoc
from docarray.typing import ImageUrl


def test_set_image_url():
    class MyDocument(BaseDoc):
        image_url: ImageUrl

    d = MyDocument(image_url="https://jina.ai/img.png")

    assert isinstance(d.image_url, ImageUrl)
    assert d.image_url == "https://jina.ai/img.png"
