from docarray import Document
from docarray.typing import AnyUrl


def test_set_any_url():
    class MyDocument(Document):
        any_url: AnyUrl

    d = MyDocument(any_url="https://jina.ai")

    assert isinstance(d.any_url, AnyUrl)
    assert d.any_url == "https://jina.ai"
