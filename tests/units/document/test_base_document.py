from typing import List, Optional

from docarray.base_doc.doc import BaseDoc


def test_base_document_init():
    doc = BaseDoc()

    assert doc.id is not None


def test_update():
    class MyDocument(BaseDoc):
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
