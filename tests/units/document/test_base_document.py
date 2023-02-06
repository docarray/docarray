from typing import Optional, List
from docarray.base_document.document import BaseDocument


def test_base_document_init():
    doc = BaseDocument()

    assert doc.id is not None


def test_update():
    class MyDocument(BaseDocument):
        content: str
        title: Optional[str] = None
        tags_: List

    doc1 = MyDocument(
        content='Core content of the document', title='Title', tags_=['python', 'AI']
    )
    doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

    doc1.update(doc2)
    assert doc1.content == 'Core content updated'
    assert doc1.title == 'Title'
    assert doc1.tags_ == ['python', 'AI', 'docarray']
