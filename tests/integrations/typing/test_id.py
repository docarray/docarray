from docarray import Document
from docarray.typing import ID


def test_set_id():
    class MyDocument(Document):
        id: ID

    d = MyDocument(id="123")

    assert isinstance(d.id, ID)
    assert d.id == "123"
